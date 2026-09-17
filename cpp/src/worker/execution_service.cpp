#include "hllm/worker/execution_service.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hllm/runtime/error.hpp"
#include "hllm/runtime/tensor_envelope.hpp"

namespace hllm::worker {

std::shared_ptr<grpc::Channel> LoadedDeployment::downstream_channel() {
  std::call_once(downstream_once_, [&] {
    std::string endpoint;
    for (const auto& stage : spec.stage_endpoints()) {
      if (stage.stage_index() == 1U) endpoint = stage.endpoint();
    }
    if (endpoint.empty()) throw std::logic_error("missing downstream endpoint");
    grpc::ChannelArguments arguments;
    arguments.SetMaxReceiveMessageSize(kMaximumRpcBytes);
    arguments.SetMaxSendMessageSize(kMaximumRpcBytes);
    downstream_channel_ = grpc::CreateCustomChannel(
        endpoint, grpc::InsecureChannelCredentials(), arguments);
  });
  return downstream_channel_;
}

namespace {

class PeerFailure final : public std::runtime_error {
 public:
  explicit PeerFailure(grpc::Status status)
      : std::runtime_error(status.error_message()), status_(std::move(status)) {}
  const grpc::Status& status() const { return status_; }

 private:
  grpc::Status status_;
};

[[noreturn]] void peer_failure(grpc::ClientReaderWriter<v1::StageMessage, v1::StageMessage>& peer,
                               const std::string& detail) {
  peer.WritesDone();
  auto status = peer.Finish();
  if (status.ok()) {
    status = {grpc::StatusCode::DATA_LOSS, detail};
  }
  throw PeerFailure(std::move(status));
}

struct CancelPeerOnExit {
  grpc::ClientContext& context;
  ~CancelPeerOnExit() { context.TryCancel(); }
};

class RequestGuard final {
 public:
  RequestGuard(ControlService& control, ExecutionLease lease)
      : control_(control), lease_(std::move(lease)) {}
  ~RequestGuard() { control_.release(lease_.request); }
  ExecutionLease& lease() { return lease_; }

 private:
  ControlService& control_;
  ExecutionLease lease_;
};

class ActiveIo final {
 public:
  explicit ActiveIo(std::atomic_bool& active) : active_(active) { active_.store(true); }
  ~ActiveIo() { active_.store(false); }
 private:
  std::atomic_bool& active_;
};

// Synchronous gRPC reads/writes must also wake when cancellation arrives over
// the independent control RPC. Join before any referenced context is destroyed.
class Watchdog final {
 public:
  Watchdog(grpc::ServerContext& server, const std::shared_ptr<ActiveRequest>& request,
           grpc::ClientContext* peer = nullptr, const std::atomic_bool* client_write = nullptr)
      : thread_([&server, request, peer, client_write](std::stop_token stop) {
          while (!stop.stop_requested()) {
            if (request->cancelled.load() || server.IsCancelled() ||
                std::chrono::system_clock::now() >= request->deadline) {
              request->cancelled.store(true);
              if (peer != nullptr) {
                peer->TryCancel();
              }
              // TryCancel forces a CANCELLED transport status, even if the
              // handler returns DEADLINE_EXCEEDED. For an application deadline,
              // cancel the peer and let compute unwind with its precise status.
              // Only blocked stream I/O needs the transport fallback.
              // The I/O flag is set before checking cancellation, so no new
              // operation can start after this watchdog observes it as false.
              if (client_write != nullptr &&
                  std::chrono::system_clock::now() >= request->deadline) {
                for (int attempt = 0; attempt < 20 && !stop.stop_requested(); ++attempt) {
                  std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (stop.stop_requested() || !client_write->load()) {
                  return;
                }
              }
              server.TryCancel();
              return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
          }
        }) {}

  // Call once the request has reached a terminal outcome; afterwards a late
  // cancellation must not disturb the acknowledgment still being written.
  void stop() {
    thread_.request_stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  std::jthread thread_;
};

std::uint64_t milliseconds(std::chrono::system_clock::time_point time) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
}

std::uint64_t effective_deadline(grpc::ServerContext& context, std::uint64_t requested) {
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  if (requested != 0U) {
    const auto now = milliseconds(std::chrono::system_clock::now());
    if (requested <= now) {
      throw runtime::Error::deadline_exceeded("request deadline already expired");
    }
    if (requested - now > 3'600'000U) {
      throw runtime::Error::invalid_request("deadline exceeds one hour");
    }
    deadline = std::chrono::system_clock::time_point(std::chrono::milliseconds(requested));
  }
  return milliseconds(std::min(deadline, context.deadline()));
}

void check_running(const ActiveRequest& request, grpc::ServerContext& context) {
  if (request.cancelled.load() || context.IsCancelled() ||
      std::chrono::system_clock::now() >= request.deadline) {
    throw std::runtime_error("request cancelled or deadline exceeded");
  }
}

void validate_stop_ids(const google::protobuf::RepeatedField<std::uint64_t>& ids,
                       const runtime::StageBackend& backend) {
  for (const auto id : ids) {
    if (id >= backend.vocabulary_size()) {
      throw runtime::Error::invalid_request("stop token exceeds vocabulary");
    }
  }
}

void validate_identity(const std::string& deployment, std::uint64_t version,
                       const std::string& request, std::uint64_t microbatch,
                       const ExecutionLease& lease) {
  if (deployment != lease.deployment->spec.plan().plan_id() ||
      version != lease.deployment->spec.plan().deployment_version() ||
      request != lease.request->id || microbatch != 0U) {
    throw runtime::Error::invalid_request("message identity does not match the active sequence");
  }
}

BoundaryIdentity boundary_identity(const ExecutionLease& lease) {
  return {lease.deployment->spec.plan().plan_id(),
          lease.deployment->spec.plan().deployment_version(), lease.request->id,
          lease.request->maximum_tokens, lease.deployment->backend->hidden_size()};
}
runtime::BoundaryActivation decode_tensor(const v1::TensorEnvelope& tensor,
                                          const ExecutionLease& lease, std::uint64_t sequence,
                                          std::size_t position, bool prefill) {
  return decode_tensor(tensor, boundary_identity(lease), sequence, position, prefill);
}
v1::StageMessage encode_tensor(const runtime::BoundaryActivation& hidden,
                               const ExecutionLease& lease, std::uint64_t sequence,
                               std::size_t position, bool prefill) {
  return encode_tensor(hidden, boundary_identity(lease), sequence, position, prefill);
}

grpc::Status failure(const std::exception& error,
                     const std::shared_ptr<ActiveRequest>& request = {}) {
  if (request && std::chrono::system_clock::now() >= request->deadline) {
    return {grpc::StatusCode::DEADLINE_EXCEEDED, error.what()};
  }
  if (request && request->cancelled.load()) {
    return {grpc::StatusCode::CANCELLED, error.what()};
  }
  if (const auto* peer = dynamic_cast<const PeerFailure*>(&error)) {
    return peer->status();
  }
  if (const auto* typed = dynamic_cast<const runtime::Error*>(&error)) {
    switch (typed->code()) {
      case runtime::ErrorCode::kInvalidRequest:
        return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
      case runtime::ErrorCode::kIncompatibleWorker:
        return {grpc::StatusCode::FAILED_PRECONDITION, error.what()};
      case runtime::ErrorCode::kResourceExhausted:
        return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
      case runtime::ErrorCode::kDeadlineExceeded:
        return {grpc::StatusCode::DEADLINE_EXCEEDED, error.what()};
      case runtime::ErrorCode::kCancelled:
        return {grpc::StatusCode::CANCELLED, error.what()};
      case runtime::ErrorCode::kInternal:
        break;
    }
  }
  if (dynamic_cast<const std::bad_alloc*>(&error)) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
  }
  return {grpc::StatusCode::INTERNAL, error.what()};
}

}  // namespace

namespace {
runtime::SamplingOptions sampling_options(const v1::SamplingOptions* wire) {
  runtime::SamplingOptions result;
  if (!wire) return result;
  result.temperature = wire->temperature();
  result.top_p = wire->top_p();
  result.top_k = wire->top_k();
  result.return_logprobs = wire->return_logprobs();
  result.top_logprobs = wire->top_logprobs();
  runtime::validate_sampling(result);
  if (wire->has_seed())
    result.seed = wire->seed();
  else {
    std::random_device random;
    result.seed = (static_cast<std::uint64_t>(random()) << 32U) ^ random();
  }
  return result;
}
template<class WireToken>
void copy_logprobs(const runtime::SampledToken& source, WireToken* target) {
  if (source.logprob) target->set_logprob(*source.logprob);
  for (const auto& candidate : source.top_logprobs) {
    auto* entry = target->add_top_logprobs();
    entry->set_token_id(candidate.token_id);
    entry->set_logprob(candidate.logprob);
  }
}
void set_sampling(runtime::SequenceState& state, const runtime::SamplingOptions& options,
                  std::size_t vocabulary) {
  if (options.top_k > vocabulary) throw runtime::Error::invalid_request("top_k exceeds vocabulary");
  state.sampling = options;
  state.generated_index = 0;
}
}  // namespace

grpc::Status ExecutionService::Execute(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<v1::StageMessage, v1::StageMessage>* stream) {
  std::shared_ptr<ActiveRequest> active;
  try {
    v1::StageMessage message;
    if (!stream->Read(&message) || !message.has_open_sequence()) {
      throw runtime::Error::invalid_request("stage stream must start with SequenceOpen");
    }
    const auto open = message.open_sequence();
    const auto sampling = sampling_options(open.has_sampling() ? &open.sampling() : nullptr);
    if (open.protocol_version() != 1U || open.microbatch_id() != 0U ||
        open.maximum_new_tokens() == 0U ||
        open.maximum_new_tokens() >= open.maximum_total_tokens()) {
      throw runtime::Error::invalid_request("invalid sequence opening");
    }
    RequestGuard guard(control_,
                       control_.acquire(open.deployment_id(), open.deployment_version(),
                                        open.request_id(), open.maximum_total_tokens(),
                                        effective_deadline(*context, open.deadline_unix_ms()), 1U));
    auto& lease = guard.lease();
    active = lease.request;
    set_sampling(*active->sequence, sampling, lease.deployment->backend->vocabulary_size());
    std::atomic_bool stream_io{false};
    Watchdog watchdog(*context, active, nullptr, &stream_io);
    validate_stop_ids(open.stop_token_ids(), *lease.deployment->backend);
    std::size_t position = 0U;
    std::uint64_t sequence = 0U;
    std::size_t prompt = open.prompt_tokens();
    if (prompt > active->maximum_tokens - open.maximum_new_tokens()) {
      throw runtime::Error::invalid_request("prompt exceeds reservation");
    }
    std::uint64_t generated = 0;
    bool stopped = false;
    auto read = [&] {
      ActiveIo reading(stream_io);
      check_running(*active, *context);
      return stream->Read(&message);
    };
    while (read()) {
      check_running(*active, *context);
      if (message.has_terminate()) {
        const auto& terminal = message.terminate();
        validate_identity(terminal.deployment_id(), terminal.deployment_version(),
                          terminal.request_id(), terminal.microbatch_id(), lease);
        if (terminal.state() != v1::TERMINAL_STATE_COMPLETED || !stopped ||
            terminal.prompt_tokens() != prompt || terminal.generated_tokens() != generated) {
          throw runtime::Error::invalid_request("invalid sequence termination");
        }
        watchdog.stop();
        control_.release(active);
        if (!stream->Write(message)) {
          throw std::runtime_error("termination acknowledgment failed");
        }
        return grpc::Status::OK;
      }
      if (!message.has_tensor() || stopped) {
        throw runtime::Error::invalid_request("unexpected stage message");
      }
      const bool prefilling = sequence == 0U || position < prompt;
      auto input = decode_tensor(message.tensor(), lease, sequence, position, prefilling);
      if (sequence == 0U && prompt == 0U) {
        prompt = input.tokens;
        if (prompt > active->maximum_tokens - open.maximum_new_tokens()) {
          throw runtime::Error::invalid_request("prompt and output exceed reservation");
        }
      }
      const auto count = input.tokens;
      if ((prefilling && count > prompt - position) || (!prefilling && count != 1U)) {
        throw runtime::Error::invalid_request("invalid prefill/decode chunk length");
      }
      active->sequence->prefilling = prefilling;
      active->sequence->prefill_only = prefilling && position + count < prompt;
      auto output = lease.deployment->scheduler.execute(
          *lease.deployment->backend, std::move(input), position, *active->sequence,
          active->cancelled, active->deadline);
      if (active->sequence->prefill_only) {
        if (!std::holds_alternative<runtime::PrefillProgress>(output)) {
          throw runtime::Error::internal("backend sampled an incomplete prompt");
        }
        v1::StageMessage response;
        auto* progress = response.mutable_prefill_progress();
        progress->set_deployment_id(open.deployment_id());
        progress->set_deployment_version(open.deployment_version());
        progress->set_request_id(open.request_id());
        progress->set_sequence_number(sequence++);
        position += count;
        progress->set_next_position(position);
        ActiveIo writing(stream_io);
        check_running(*active, *context);
        if (!stream->Write(response)) throw std::runtime_error("prefill acknowledgment failed");
        continue;
      }
      const auto* sampled_output = std::get_if<runtime::SampledToken>(&output);
      if (sampled_output == nullptr ||
          sampled_output->id >= lease.deployment->backend->vocabulary_size()) {
        throw runtime::Error::internal("final backend did not return a valid sampled token");
      }
      const auto token = sampled_output->id;
      check_running(*active, *context);
      v1::StageMessage response;
      auto* sampled = response.mutable_sampled_token();
      sampled->set_deployment_id(open.deployment_id());
      sampled->set_deployment_version(open.deployment_version());
      sampled->set_request_id(open.request_id());
      sampled->set_sequence_number(sequence);
      sampled->set_token_position(position + count);
      sampled->set_token_id(token);
      copy_logprobs(*sampled_output, sampled);
      ++sequence;
      position += count;
      stopped = ++generated == open.maximum_new_tokens() ||
                std::find(open.stop_token_ids().begin(), open.stop_token_ids().end(), token) !=
                    open.stop_token_ids().end();
      {
        ActiveIo writing(stream_io);
        check_running(*active, *context);
        if (!stream->Write(response)) {
          throw std::runtime_error("sampled token send failed");
        }
      }
    }
    throw std::runtime_error("stage stream disconnected before termination");
  } catch (const std::exception& error) {
    return failure(error, active);
  }
}

grpc::Status GenerationService::Generate(grpc::ServerContext* context,
                                         const v1::GenerationRequest* request,
                                         grpc::ServerWriter<v1::GenerationEvent>* writer) {
  const auto timing_start = request->capture_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  std::shared_ptr<ActiveRequest> active;
  try {
    if (request->token_ids_size() == 0 || request->maximum_new_tokens() == 0U ||
        request->maximum_new_tokens() > std::numeric_limits<std::size_t>::max() -
                                            static_cast<std::size_t>(request->token_ids_size())) {
      throw runtime::Error::invalid_request(
          "generation requires prompt tokens and a bounded positive output length");
    }
    const auto sampling =
        sampling_options(request->has_sampling() ? &request->sampling() : nullptr);
    const auto total =
        static_cast<std::size_t>(request->token_ids_size()) + request->maximum_new_tokens();
    const auto acquire_start = request->capture_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    RequestGuard guard(
        control_, control_.acquire(request->deployment_id(), request->deployment_version(),
                                   request->request_id(), total,
                                   effective_deadline(*context, request->deadline_unix_ms()), 0U));
    const auto acquire_end = request->capture_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& lease = guard.lease();
    active = lease.request;
    auto& backend = *lease.deployment->backend;
    set_sampling(*active->sequence, sampling, backend.vocabulary_size());
    validate_stop_ids(request->stop_token_ids(), backend);
    const bool split = lease.deployment->spec.plan().stages_size() == 2;
    // Only a split deployment ships the prefill activation over the wire; the
    // planner applies the same ceiling before it emits a two-stage plan.
    const auto chunk_size =
        request->prefill_chunk_tokens() == 0U
            ? static_cast<std::size_t>(request->token_ids_size())
            : std::min(static_cast<std::size_t>(request->prefill_chunk_tokens()),
                       static_cast<std::size_t>(request->token_ids_size()));
    if (split &&
        chunk_size > static_cast<std::size_t>(kMaximumRpcBytes / 4) / backend.hidden_size()) {
      throw runtime::Error::resource_exhausted("prefill activation exceeds transport limit");
    }
    std::vector<std::uint64_t> tokens(request->token_ids().begin(), request->token_ids().end());
    for (const auto token : tokens) {
      if (token >= backend.vocabulary_size()) {
        throw runtime::Error::invalid_request("token ID exceeds vocabulary");
      }
    }
    grpc::ClientContext peer_context;
    peer_context.set_deadline(active->deadline);
    std::atomic_bool client_write{false};
    Watchdog watchdog(*context, active, &peer_context, &client_write);
    std::unique_ptr<v1::StageExecution::Stub> stub;
    std::unique_ptr<grpc::ClientReaderWriter<v1::StageMessage, v1::StageMessage>> peer;
    CancelPeerOnExit cancel_peer{peer_context};
    if (split) {
      stub = v1::StageExecution::NewStub(lease.deployment->downstream_channel());
      peer = stub->Execute(&peer_context);
      v1::StageMessage opening;
      auto* open = opening.mutable_open_sequence();
      open->set_protocol_version(1U);
      open->set_deployment_id(request->deployment_id());
      open->set_deployment_version(request->deployment_version());
      open->set_request_id(request->request_id());
      open->set_maximum_total_tokens(total);
      open->set_prompt_tokens(static_cast<std::uint64_t>(request->token_ids_size()));
      open->set_maximum_new_tokens(request->maximum_new_tokens());
      open->set_deadline_unix_ms(milliseconds(active->deadline));
      *open->mutable_stop_token_ids() = request->stop_token_ids();
      auto* wire_sampling = open->mutable_sampling();
      wire_sampling->set_temperature(sampling.temperature);
      wire_sampling->set_top_p(sampling.top_p);
      wire_sampling->set_top_k(sampling.top_k);
      wire_sampling->set_seed(sampling.seed);
      wire_sampling->set_return_logprobs(sampling.return_logprobs);
      wire_sampling->set_top_logprobs(sampling.top_logprobs);
      if (!peer->Write(opening)) {
        peer_failure(*peer, "cannot open downstream sequence");
      }
    }
    auto write_event = [&](const v1::GenerationEvent& event) {
      if (!writer->Write(event)) {
        throw std::runtime_error("generation client disconnected");
      }
    };
    auto emit = [&](const v1::GenerationEvent& event) {
      ActiveIo writing(client_write);
      check_running(*active, *context);
      write_event(event);
    };
    const auto setup_ms = request->capture_timing() ? std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - timing_start - (acquire_end - acquire_start)).count() : 0.0;
    std::size_t position = 0U;
    std::uint64_t generated = 0U;
    const auto prompt = tokens;
    for (std::uint64_t step = 0U; generated < request->maximum_new_tokens(); ++step) {
      check_running(*active, *context);
      const bool prefilling = position < prompt.size();
      if (prefilling) {
        const auto count = std::min(chunk_size, prompt.size() - position);
        tokens.assign(prompt.begin() + static_cast<std::ptrdiff_t>(position),
                      prompt.begin() + static_cast<std::ptrdiff_t>(position + count));
      }
      active->sequence->prefilling = prefilling;
      active->sequence->prefill_only = prefilling && position + tokens.size() < prompt.size();
      auto output = lease.deployment->scheduler.execute(backend, runtime::TokenInput{tokens},
                                                        position, *active->sequence,
                                                        active->cancelled, active->deadline);
      std::uint64_t token;
      v1::TokenEvent token_metadata;
      if (peer) {
        const auto* boundary = std::get_if<runtime::BoundaryActivation>(&output);
        if (boundary == nullptr || boundary->tokens != tokens.size()) {
          throw runtime::Error::internal(
              "intermediate backend did not return boundary activations");
        }
        if (!peer->Write(encode_tensor(*boundary, lease, step, position, prefilling))) {
          peer_failure(*peer, "downstream activation send failed");
        }
        v1::StageMessage response;
        if (!peer->Read(&response)) {
          peer_failure(*peer, "downstream failed to return a sampled token");
        }
        if (active->sequence->prefill_only) {
          if (!response.has_prefill_progress()) {
            throw runtime::Error::internal("expected prefill acknowledgment");
          }
          const auto& progress = response.prefill_progress();
          validate_identity(progress.deployment_id(), progress.deployment_version(),
                            progress.request_id(), progress.microbatch_id(), lease);
          if (progress.sequence_number() != step ||
              progress.next_position() != position + tokens.size()) {
            throw runtime::Error::internal("invalid prefill acknowledgment");
          }
          position += tokens.size();
          continue;
        }
        if (!response.has_sampled_token()) {
          throw runtime::Error::internal("unexpected downstream response");
        }
        const auto& sampled = response.sampled_token();
        validate_identity(sampled.deployment_id(), sampled.deployment_version(),
                          sampled.request_id(), sampled.microbatch_id(), lease);
        if (sampled.sequence_number() != step ||
            sampled.token_position() != position + tokens.size() ||
            sampled.token_id() >= backend.vocabulary_size()) {
          throw runtime::Error::internal("invalid sampled token feedback");
        }
        token = sampled.token_id();
        if (sampled.has_logprob()) token_metadata.set_logprob(sampled.logprob());
        *token_metadata.mutable_top_logprobs() = sampled.top_logprobs();
      } else {
        if (active->sequence->prefill_only) {
          if (!std::holds_alternative<runtime::PrefillProgress>(output)) {
            throw runtime::Error::internal("backend sampled an incomplete prompt");
          }
          position += tokens.size();
          continue;
        }
        const auto* sampled_output = std::get_if<runtime::SampledToken>(&output);
        if (sampled_output == nullptr || sampled_output->id >= backend.vocabulary_size()) {
          throw runtime::Error::internal("final backend did not return a valid sampled token");
        }
        token = sampled_output->id;
        copy_logprobs(*sampled_output, &token_metadata);
      }
      check_running(*active, *context);
      if (generated == 0U) {
        v1::GenerationEvent event;
        event.set_request_id(request->request_id());
        event.mutable_prefill_complete()->set_prompt_tokens(prompt.size());
        emit(event);
      }
      position += tokens.size();
      v1::GenerationEvent event;
      event.set_request_id(request->request_id());
      *event.mutable_token() = token_metadata;
      event.mutable_token()->set_token_id(token);
      event.mutable_token()->set_token_position(position);
      if (request->capture_timing()) event.mutable_token()->set_native_elapsed_ms(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timing_start).count());
      if (request->capture_timing() && generated == 0U)
        event.mutable_token()->set_native_request_setup_ms(setup_ms);
      emit(event);
      ++generated;
      if (std::find(request->stop_token_ids().begin(), request->stop_token_ids().end(), token) !=
          request->stop_token_ids().end()) {
        break;
      }
      tokens = {token};
    }
    if (peer) {
      v1::StageMessage terminal;
      auto* finish = terminal.mutable_terminate();
      finish->set_deployment_id(request->deployment_id());
      finish->set_deployment_version(request->deployment_version());
      finish->set_request_id(request->request_id());
      finish->set_state(v1::TERMINAL_STATE_COMPLETED);
      finish->set_prompt_tokens(static_cast<std::uint64_t>(request->token_ids_size()));
      finish->set_generated_tokens(generated);
      if (!peer->Write(terminal)) {
        throw std::runtime_error("downstream termination failed");
      }
      peer->WritesDone();
      v1::StageMessage ack;
      if (!peer->Read(&ack) || ack.SerializeAsString() != terminal.SerializeAsString() ||
          peer->Read(&ack) || !peer->Finish().ok()) {
        throw std::runtime_error("downstream cleanup was not acknowledged");
      }
    }
    watchdog.stop();
    check_running(*active, *context);
    // Generation and downstream acknowledgment have completed. Commit the
    // outcome before releasing sequence state; slow cleanup or a late control
    // cancellation must not change it while usage/terminal events are written.
    control_.release(active);
    v1::GenerationEvent event;
    event.set_request_id(request->request_id());
    event.mutable_usage()->set_prompt_tokens(static_cast<std::uint64_t>(request->token_ids_size()));
    event.mutable_usage()->set_generated_tokens(generated);
    write_event(event);
    event.clear_usage();
    event.mutable_terminal()->set_state(v1::TERMINAL_STATE_COMPLETED);
    write_event(event);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return failure(error, active);
  }
}

}  // namespace hllm::worker
