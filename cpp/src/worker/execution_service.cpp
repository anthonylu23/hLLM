#include "hllm/worker/execution_service.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hllm/runtime/tensor_envelope.hpp"

namespace hllm::worker {
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

// Synchronous gRPC reads/writes must also wake when cancellation arrives over
// the independent control RPC. Join before any referenced context is destroyed.
class Watchdog final {
 public:
  Watchdog(grpc::ServerContext& server, const std::shared_ptr<ActiveRequest>& request,
           grpc::ClientContext* peer = nullptr)
      : thread_([&server, request, peer](std::stop_token stop) {
          while (!stop.stop_requested()) {
            if (request->cancelled.load() || server.IsCancelled() ||
                std::chrono::system_clock::now() >= request->deadline) {
              request->cancelled.store(true);
              if (peer != nullptr) {
                peer->TryCancel();
              }
              server.TryCancel();
              return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
          }
        }) {}

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
      throw std::invalid_argument("request deadline already expired");
    }
    if (requested - now > 3'600'000U) {
      throw std::invalid_argument("deadline exceeds one hour");
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
      throw std::invalid_argument("stop token exceeds vocabulary");
    }
  }
}

void validate_identity(const std::string& deployment, std::uint64_t version,
                       const std::string& request, std::uint64_t microbatch,
                       const ExecutionLease& lease) {
  if (deployment != lease.deployment->spec.plan().plan_id() ||
      version != lease.deployment->spec.plan().deployment_version() ||
      request != lease.request->id || microbatch != 0U) {
    throw std::invalid_argument("message identity does not match the active sequence");
  }
}

runtime::BoundaryActivation decode_tensor(const v1::TensorEnvelope& tensor,
                                          const ExecutionLease& lease, std::uint64_t sequence,
                                          std::size_t position) {
  validate_identity(tensor.deployment_id(), tensor.deployment_version(), tensor.request_id(),
                    tensor.microbatch_id(), lease);
  if (tensor.sequence_number() != sequence || tensor.first_position() != position ||
      tensor.phase() !=
          (sequence == 0U ? v1::EXECUTION_PHASE_PREFILL : v1::EXECUTION_PHASE_DECODE) ||
      tensor.dtype() != v1::DATA_TYPE_F16 || tensor.layout() != "dense_row_major_le" ||
      !tensor.checksum().empty()) {
    throw std::invalid_argument("invalid tensor order, phase, encoding or unsupported checksum");
  }
  runtime::TensorEnvelopeMetadata metadata{
      tensor.protocol_version(),
      tensor.deployment_id(),
      tensor.deployment_version(),
      tensor.request_id(),
      tensor.microbatch_id(),
      tensor.sequence_number(),
      sequence == 0U ? runtime::ExecutionPhase::kPrefill : runtime::ExecutionPhase::kDecode,
      tensor.first_position(),
      {tensor.sequence_lengths().begin(), tensor.sequence_lengths().end()},
      {tensor.cache_slot_ids().begin(), tensor.cache_slot_ids().end()},
      {tensor.shape().begin(), tensor.shape().end()},
      runtime::DataType::kF16,
      runtime::TensorLayout::kDenseRowMajorLittleEndian,
      tensor.payload_length()};
  runtime::validate_tensor_envelope(
      metadata, tensor.payload().size(),
      {.maximum_sequence_length = lease.request->maximum_tokens,
       .hidden_size = lease.deployment->backend->hidden_size(),
       .maximum_payload_bytes = static_cast<std::size_t>(kMaximumRpcBytes / 2)});
  if (tensor.cache_slot_ids_size() != 1 || tensor.cache_slot_ids(0) != 0U) {
    throw std::invalid_argument("single-sequence execution requires cache slot zero");
  }
  runtime::BoundaryActivation output{static_cast<std::size_t>(tensor.shape(1)),
                                     lease.deployment->backend->hidden_size(),
                                     std::vector<std::byte>(tensor.payload().size())};
  std::memcpy(output.payload.data(), tensor.payload().data(), tensor.payload().size());
  for (std::size_t i = 0U; i < output.payload.size(); i += 2U) {
    const auto high = std::to_integer<unsigned char>(output.payload[i + 1U]);
    if ((high & 0x7cU) == 0x7cU) {
      throw std::invalid_argument("non-finite boundary activation");
    }
  }
  return output;
}

v1::StageMessage encode_tensor(const runtime::BoundaryActivation& hidden,
                               const ExecutionLease& lease, std::uint64_t sequence,
                               std::size_t position) {
  if (hidden.payload.size() > static_cast<std::size_t>(kMaximumRpcBytes / 2)) {
    throw std::length_error("prefill activation exceeds transport limit");
  }
  v1::StageMessage message;
  auto* tensor = message.mutable_tensor();
  tensor->set_protocol_version(1U);
  tensor->set_deployment_id(lease.deployment->spec.plan().plan_id());
  tensor->set_deployment_version(lease.deployment->spec.plan().deployment_version());
  tensor->set_request_id(lease.request->id);
  tensor->set_sequence_number(sequence);
  tensor->set_phase(sequence == 0U ? v1::EXECUTION_PHASE_PREFILL : v1::EXECUTION_PHASE_DECODE);
  tensor->set_first_position(position);
  tensor->add_sequence_lengths(hidden.tokens);
  tensor->add_cache_slot_ids(0U);
  tensor->add_shape(1U);
  tensor->add_shape(hidden.tokens);
  tensor->add_shape(hidden.width);
  tensor->set_dtype(v1::DATA_TYPE_F16);
  tensor->set_layout("dense_row_major_le");
  if (hidden.width == 0U || hidden.width != lease.deployment->backend->hidden_size() ||
      hidden.tokens == 0U ||
      hidden.tokens > static_cast<std::size_t>(kMaximumRpcBytes / 4) / hidden.width ||
      hidden.payload.size() != hidden.tokens * hidden.width * 2U) {
    throw std::runtime_error("backend returned an invalid boundary shape");
  }
  for (std::size_t i = 1U; i < hidden.payload.size(); i += 2U) {
    if ((std::to_integer<unsigned char>(hidden.payload[i]) & 0x7cU) == 0x7cU) {
      throw std::runtime_error("backend returned a non-finite boundary activation");
    }
  }
  tensor->set_payload(hidden.payload.data(), hidden.payload.size());
  tensor->set_payload_length(hidden.payload.size());
  return message;
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
  if (dynamic_cast<const std::bad_alloc*>(&error) ||
      dynamic_cast<const std::length_error*>(&error)) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
  }
  if (dynamic_cast<const std::invalid_argument*>(&error) ||
      dynamic_cast<const runtime::EnvelopeError*>(&error)) {
    return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
  }
  return {grpc::StatusCode::INTERNAL, error.what()};
}

}  // namespace

grpc::Status ExecutionService::Execute(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<v1::StageMessage, v1::StageMessage>* stream) {
  std::shared_ptr<ActiveRequest> active;
  try {
    v1::StageMessage message;
    if (!stream->Read(&message) || !message.has_open_sequence()) {
      throw std::invalid_argument("stage stream must start with SequenceOpen");
    }
    const auto open = message.open_sequence();
    if (open.protocol_version() != 1U || open.microbatch_id() != 0U ||
        open.maximum_new_tokens() == 0U ||
        open.maximum_new_tokens() >= open.maximum_total_tokens()) {
      throw std::invalid_argument("invalid sequence opening");
    }
    RequestGuard guard(control_,
                       control_.acquire(open.deployment_id(), open.deployment_version(),
                                        open.request_id(), open.maximum_total_tokens(),
                                        effective_deadline(*context, open.deadline_unix_ms()), 1U));
    auto& lease = guard.lease();
    active = lease.request;
    Watchdog watchdog(*context, active);
    validate_stop_ids(open.stop_token_ids(), *lease.deployment->backend);
    std::size_t position = 0U;
    std::uint64_t sequence = 0U;
    std::size_t prompt = 0U;
    bool stopped = false;
    while (stream->Read(&message)) {
      check_running(*active, *context);
      if (message.has_terminate()) {
        const auto& terminal = message.terminate();
        validate_identity(terminal.deployment_id(), terminal.deployment_version(),
                          terminal.request_id(), terminal.microbatch_id(), lease);
        if (terminal.state() != v1::TERMINAL_STATE_COMPLETED || !stopped ||
            terminal.prompt_tokens() != prompt || terminal.generated_tokens() != sequence) {
          throw std::invalid_argument("invalid sequence termination");
        }
        control_.release(active);
        if (!stream->Write(message)) {
          throw std::runtime_error("termination acknowledgment failed");
        }
        return grpc::Status::OK;
      }
      if (!message.has_tensor() || stopped) {
        throw std::invalid_argument("unexpected stage message");
      }
      auto input = decode_tensor(message.tensor(), lease, sequence, position);
      if (sequence == 0U) {
        prompt = input.tokens;
        if (prompt > active->maximum_tokens - open.maximum_new_tokens()) {
          throw std::invalid_argument("prompt and output exceed reservation");
        }
      }
      const auto count = input.tokens;
      auto output = lease.deployment->backend->execute(std::move(input), position,
                                                       *active->sequence, active->cancelled);
      const auto* sampled_output = std::get_if<runtime::SampledToken>(&output);
      if (sampled_output == nullptr ||
          sampled_output->id >= lease.deployment->backend->vocabulary_size()) {
        throw std::runtime_error("final backend did not return a valid sampled token");
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
      ++sequence;
      position += count;
      stopped = sequence == open.maximum_new_tokens() ||
                std::find(open.stop_token_ids().begin(), open.stop_token_ids().end(), token) !=
                    open.stop_token_ids().end();
      if (!stream->Write(response)) {
        throw std::runtime_error("sampled token send failed");
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
  std::shared_ptr<ActiveRequest> active;
  try {
    if (request->token_ids_size() == 0 || request->maximum_new_tokens() == 0U ||
        request->maximum_new_tokens() > std::numeric_limits<std::size_t>::max() -
                                            static_cast<std::size_t>(request->token_ids_size())) {
      throw std::invalid_argument(
          "generation requires prompt tokens and a "
          "bounded positive output length");
    }
    const auto total =
        static_cast<std::size_t>(request->token_ids_size()) + request->maximum_new_tokens();
    RequestGuard guard(
        control_, control_.acquire(request->deployment_id(), request->deployment_version(),
                                   request->request_id(), total,
                                   effective_deadline(*context, request->deadline_unix_ms()), 0U));
    auto& lease = guard.lease();
    active = lease.request;
    auto& backend = *lease.deployment->backend;
    validate_stop_ids(request->stop_token_ids(), backend);
    if (static_cast<std::size_t>(request->token_ids_size()) >
        static_cast<std::size_t>(kMaximumRpcBytes / 4) / backend.hidden_size()) {
      throw std::length_error("prefill activation exceeds transport limit");
    }
    std::vector<std::uint64_t> tokens(request->token_ids().begin(), request->token_ids().end());
    for (const auto token : tokens) {
      if (token >= backend.vocabulary_size()) {
        throw std::invalid_argument("token ID exceeds vocabulary");
      }
    }
    grpc::ClientContext peer_context;
    peer_context.set_deadline(active->deadline);
    Watchdog watchdog(*context, active, &peer_context);
    std::unique_ptr<v1::StageExecution::Stub> stub;
    std::unique_ptr<grpc::ClientReaderWriter<v1::StageMessage, v1::StageMessage>> peer;
    CancelPeerOnExit cancel_peer{peer_context};
    if (lease.deployment->spec.plan().stages_size() == 2) {
      std::string endpoint;
      for (const auto& stage : lease.deployment->spec.stage_endpoints()) {
        if (stage.stage_index() == 1U) {
          endpoint = stage.endpoint();
        }
      }
      grpc::ChannelArguments arguments;
      arguments.SetMaxReceiveMessageSize(kMaximumRpcBytes);
      arguments.SetMaxSendMessageSize(kMaximumRpcBytes);
      stub = v1::StageExecution::NewStub(
          grpc::CreateCustomChannel(endpoint, grpc::InsecureChannelCredentials(), arguments));
      peer = stub->Execute(&peer_context);
      v1::StageMessage opening;
      auto* open = opening.mutable_open_sequence();
      open->set_protocol_version(1U);
      open->set_deployment_id(request->deployment_id());
      open->set_deployment_version(request->deployment_version());
      open->set_request_id(request->request_id());
      open->set_maximum_total_tokens(total);
      open->set_maximum_new_tokens(request->maximum_new_tokens());
      open->set_deadline_unix_ms(milliseconds(active->deadline));
      *open->mutable_stop_token_ids() = request->stop_token_ids();
      if (!peer->Write(opening)) {
        peer_failure(*peer, "cannot open downstream sequence");
      }
    }
    auto emit = [&](const v1::GenerationEvent& event) {
      if (!writer->Write(event)) {
        throw std::runtime_error("generation client disconnected");
      }
    };
    std::size_t position = 0U;
    std::uint64_t generated = 0U;
    for (std::uint64_t step = 0U; step < request->maximum_new_tokens(); ++step) {
      check_running(*active, *context);
      auto output = backend.execute(runtime::TokenInput{tokens}, position, *active->sequence,
                                    active->cancelled);
      std::uint64_t token;
      if (peer) {
        const auto* boundary = std::get_if<runtime::BoundaryActivation>(&output);
        if (boundary == nullptr || boundary->tokens != tokens.size()) {
          throw std::runtime_error("intermediate backend did not return boundary activations");
        }
        if (!peer->Write(encode_tensor(*boundary, lease, step, position))) {
          peer_failure(*peer, "downstream activation send failed");
        }
        v1::StageMessage response;
        if (!peer->Read(&response)) {
          peer_failure(*peer, "downstream failed to return a sampled token");
        }
        if (!response.has_sampled_token()) {
          throw std::invalid_argument("unexpected downstream response");
        }
        const auto& sampled = response.sampled_token();
        validate_identity(sampled.deployment_id(), sampled.deployment_version(),
                          sampled.request_id(), sampled.microbatch_id(), lease);
        if (sampled.sequence_number() != step ||
            sampled.token_position() != position + tokens.size() ||
            sampled.token_id() >= backend.vocabulary_size()) {
          throw std::invalid_argument("invalid sampled token feedback");
        }
        token = sampled.token_id();
      } else {
        const auto* sampled_output = std::get_if<runtime::SampledToken>(&output);
        if (sampled_output == nullptr || sampled_output->id >= backend.vocabulary_size()) {
          throw std::runtime_error("final backend did not return a valid sampled token");
        }
        token = sampled_output->id;
      }
      check_running(*active, *context);
      if (step == 0U) {
        v1::GenerationEvent event;
        event.set_request_id(request->request_id());
        event.mutable_prefill_complete()->set_prompt_tokens(tokens.size());
        emit(event);
      }
      position += tokens.size();
      v1::GenerationEvent event;
      event.set_request_id(request->request_id());
      event.mutable_token()->set_token_id(token);
      event.mutable_token()->set_token_position(position);
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
    control_.release(active);
    v1::GenerationEvent event;
    event.set_request_id(request->request_id());
    event.mutable_usage()->set_prompt_tokens(static_cast<std::uint64_t>(request->token_ids_size()));
    event.mutable_usage()->set_generated_tokens(generated);
    emit(event);
    event.clear_usage();
    event.mutable_terminal()->set_state(v1::TERMINAL_STATE_COMPLETED);
    emit(event);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return failure(error, active);
  }
}

}  // namespace hllm::worker
