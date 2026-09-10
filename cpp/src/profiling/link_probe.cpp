// Isolated, bounded transport qualification. This executable never loads a model.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <sys/utsname.h>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include <google/protobuf/stubs/common.h>
#include <nlohmann/json.hpp>
#include "control.grpc.pb.h"
#include "execution.grpc.pb.h"
#include "hllm/runtime/timing.hpp"
#include "hllm/worker/boundary_codec.hpp"

namespace {
namespace v1 = hllm::v1;
namespace rt = hllm::runtime;
namespace worker = hllm::worker;
using Clock = std::chrono::system_clock;
using namespace std::chrono_literals;
constexpr std::uint64_t kMaxTraffic = 256U * 1024U * 1024U;
constexpr std::uint32_t kMaxSteps = 1024U;
constexpr auto kMaxTime = 120s;
struct Busy {
  std::atomic_bool& value;
  explicit Busy(std::atomic_bool& v) : value(v) {
    if (value.exchange(true)) throw std::runtime_error("probe already active");
  }
  ~Busy() { value.store(false); }
};
struct Cancel {
  grpc::ClientContext& context;
  ~Cancel() { context.TryCancel(); }
};
// Bound blocking reads as well as computation, and forward control cancellation.
class Deadline {
 public:
  Deadline(grpc::ServerContext& server, Clock::time_point deadline,
           grpc::ClientContext* peer = nullptr)
      : thread_([&server, deadline, peer](std::stop_token stop) {
          while (!stop.stop_requested()) {
            if (server.IsCancelled() || Clock::now() >= deadline) {
              if (peer) peer->TryCancel();
              server.TryCancel();
              return;
            }
            std::this_thread::sleep_for(5ms);
          }
        }) {}
 private:
  std::jthread thread_;
};
void require(bool condition, const char* detail) {
  if (!condition) throw std::invalid_argument(detail);
}
std::shared_ptr<grpc::Channel> channel(const std::string& endpoint) {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(worker::kMaximumRpcBytes);
  args.SetMaxSendMessageSize(worker::kMaximumRpcBytes);
  return grpc::CreateCustomChannel(endpoint, grpc::InsecureChannelCredentials(), args);
}
void checked(const grpc::Status& status) {
  if (!status.ok()) throw std::runtime_error(status.error_message());
}
bool identity(const v1::SampledToken& token, const worker::BoundaryIdentity& id,
              std::uint64_t sequence, std::size_t position) {
  return token.deployment_id() == id.deployment_id &&
         token.deployment_version() == id.deployment_version && token.request_id() == id.request_id &&
         token.microbatch_id() == 0 && token.sequence_number() == sequence &&
         token.token_position() == position && token.token_id() == 0;
}
class Probe final : public v1::WorkerControl::Service, public v1::StageExecution::Service {
 public:
  explicit Probe(const nlohmann::json& config) {
    info_.set_worker_id(config.at("worker_id"));
    info_.set_endpoint(config.at("listen"));
    info_.set_binary_digest(config.at("binary_digest"));
    info_.set_source_revision(config.at("source_revision"));
    info_.set_compiler(__VERSION__);
    info_.set_grpc_version(grpc_version_string());
    info_.set_protobuf_version(std::to_string(GOOGLE_PROTOBUF_VERSION));
    utsname host{};
    if (uname(&host) != 0) throw std::runtime_error("uname failed");
    info_.set_host(host.nodename);
    info_.set_os(std::string(host.sysname) + " " + host.release + " " + host.machine);
    peers_ = config.at("peers").get<std::map<std::string, std::string>>();
    require(!info_.worker_id().empty() && info_.binary_digest().size() == 64 &&
                !info_.source_revision().empty(), "missing probe build identity");
    require(!peers_.contains(info_.worker_id()), "self peer is forbidden");
  }
  grpc::Status GetLinkProbeInfo(grpc::ServerContext*, const v1::Empty*,
                               v1::LinkProbeIdentity* out) override {
    *out = info_;
    return grpc::Status::OK;
  }
  grpc::Status QualifyLink(grpc::ServerContext* context,
                           const v1::LinkQualificationRequest* request,
                           v1::LinkProfile* response) override {
    try {
      Busy busy(active_);
      require(peers_.contains(request->target_worker_id()), "target not in explicit peer allowlist");
      const auto prompt = request->prompt_tokens();
      const auto width = request->hidden_size();
      const auto steps = request->output_tokens();
      const auto cycles = request->warmup_cycles() + std::uint64_t(request->measured_cycles());
      require(prompt > 0 && prompt <= 4096 && width > 0 && width <= 8192 && steps > 0 &&
              steps <= kMaxSteps && request->measured_cycles() > 0 && cycles <= 32 &&
              request->timeout_ms() > 0 && request->timeout_ms() <= 120000, "invalid probe limits");
      require(std::uint64_t(prompt) * width * 2 <= worker::kMaximumRpcBytes / 2 &&
              (std::uint64_t(prompt) + steps - 1) * width * 2 * cycles <= kMaxTraffic,
              "probe payload or aggregate traffic exceeds limit");
      const auto deadline = std::min(context->deadline(),
          Clock::now() + std::chrono::milliseconds(request->timeout_ms()));
      auto peer = channel(peers_.at(request->target_worker_id()));
      auto start = rt::ProfileClock::now();
      while (!peer->WaitForConnected(std::min(deadline, Clock::now() + 100ms)))
        require(!context->IsCancelled() && Clock::now() < deadline,
                "peer connection cancelled or timed out");
      auto* result = response->mutable_qualification();
      result->set_channel_ready_ms(rt::elapsed_ms(start));
      *result->mutable_source() = info_;
      auto control = v1::WorkerControl::NewStub(peer);
      grpc::ClientContext info_context;
      info_context.set_deadline(deadline);
      {
        Deadline watch(*context, deadline, &info_context);
        checked(control->GetLinkProbeInfo(&info_context, {}, result->mutable_target()));
      }
      require(result->target().worker_id() == request->target_worker_id(), "peer identity mismatch");
      result->set_prompt_tokens(prompt);
      result->set_hidden_size(width);
      result->set_output_tokens(steps);
      result->set_warmup_cycles(request->warmup_cycles());
      result->set_measured_cycles(request->measured_cycles());
      auto execution = v1::StageExecution::NewStub(peer);
      for (std::uint32_t cycle = 0; cycle < cycles; ++cycle) {
        require(!context->IsCancelled() && Clock::now() < deadline, "probe cancelled or expired");
        grpc::ClientContext stream_context;
        stream_context.set_deadline(deadline);
        Cancel cancel{stream_context};
        Deadline watch(*context, deadline, &stream_context);
        worker::BoundaryIdentity id{"hllm-link-probe-v1", 1, "qualification-" + std::to_string(cycle),
                                    prompt + steps, width};
        v1::StageMessage open;
        auto* sequence = open.mutable_open_sequence();
        sequence->set_protocol_version(1);
        sequence->set_deployment_id(id.deployment_id);
        sequence->set_deployment_version(id.deployment_version);
        sequence->set_request_id(id.request_id);
        sequence->set_maximum_total_tokens(id.maximum_tokens);
        sequence->set_maximum_new_tokens(steps);
        sequence->set_deadline_unix_ms(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count()));
        auto* stream_timing = result->add_streams();
        stream_timing->set_cycle(cycle);
        start = rt::ProfileClock::now();
        auto stream = execution->Execute(&stream_context);
        require(stream->Write(open), "sequence open failed");
        stream->WaitForInitialMetadata();
        stream_timing->set_setup_ms(rt::elapsed_ms(start));
        for (std::uint32_t step = 0; step < steps; ++step) {
          const auto count = step == 0 ? prompt : 1U;
          const auto position = step == 0 ? 0U : prompt + step - 1;
          // F16 0.5 is finite, nonzero and byte-stable. Allocation is outside the timer.
          rt::BoundaryActivation hidden{count, width,
              std::vector<std::byte>(std::size_t(count) * width * 2, std::byte{0})};
          for (std::size_t i = 1; i < hidden.payload.size(); i += 2) hidden.payload[i] = std::byte{0x38};
          auto* sample = result->add_samples();
          sample->set_cycle(cycle);
          sample->set_step(step);
          sample->set_payload_bytes(hidden.payload.size());
          start = rt::ProfileClock::now();
          auto message = worker::encode_tensor(hidden, id, step, position);
          sample->set_sender_encode_ms(rt::elapsed_ms(start));
          sample->set_message_bytes(message.ByteSizeLong());
          v1::StageMessage reply;
          start = rt::ProfileClock::now();
          require(stream->Write(message) && stream->Read(&reply), "activation exchange failed");
          require(reply.has_sampled_token() && identity(reply.sampled_token(), id, step,
                      position + count), "invalid sampled-token feedback");
          sample->set_round_trip_ms(rt::elapsed_ms(start));
          sample->set_feedback_bytes(reply.ByteSizeLong());
        }
        v1::StageMessage termination;
        auto* end = termination.mutable_terminate();
        end->set_deployment_id(id.deployment_id);
        end->set_deployment_version(id.deployment_version);
        end->set_request_id(id.request_id);
        end->set_state(v1::TERMINAL_STATE_COMPLETED);
        start = rt::ProfileClock::now();
        require(stream->Write(termination), "termination write failed");
        stream->WritesDone();
        v1::StageMessage ack;
        require(stream->Read(&ack) && ack.SerializeAsString() == termination.SerializeAsString(),
                "invalid termination acknowledgment");
        require(!stream->Read(&ack), "unexpected trailing response");
        checked(stream->Finish());
        stream_timing->set_teardown_ms(rt::elapsed_ms(start));
      }
      response->set_source_worker_id(info_.worker_id());
      response->set_target_worker_id(request->target_worker_id());
      response->set_connection_type(v1::CONNECTION_TYPE_UNKNOWN); // Path is observed by orchestrator.
      response->mutable_schema_version()->set_major(1);
      response->mutable_schema_version()->set_minor(1);
      return grpc::Status::OK;
    } catch (const std::invalid_argument& error) {
      return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
    } catch (const std::exception& error) {
      return {grpc::StatusCode::FAILED_PRECONDITION, error.what()};
    }
  }
  grpc::Status Execute(grpc::ServerContext* context,
                       grpc::ServerReaderWriter<v1::StageMessage, v1::StageMessage>* stream) override {
    try {
      Busy busy(active_);
      Deadline watch(*context, std::min(context->deadline(), Clock::now() + kMaxTime));
      v1::StageMessage message;
      require(stream->Read(&message) && message.has_open_sequence(), "expected sequence open");
      const auto open = message.open_sequence();
      require(open.protocol_version() == 1 && open.deployment_id() == "hllm-link-probe-v1" &&
                  open.deployment_version() == 1 && !open.request_id().empty() &&
                  open.request_id().size() <= 128 && open.microbatch_id() == 0 &&
                  open.maximum_total_tokens() > 0 && open.maximum_total_tokens() <= 5120 &&
                  open.maximum_new_tokens() > 0 && open.maximum_new_tokens() <= kMaxSteps &&
                  open.stop_token_ids().empty(), "invalid probe sequence open");
      stream->SendInitialMetadata();
      std::size_t position = 0;
      std::uint64_t bytes = 0;
      std::uint64_t sequence = 0;
      std::size_t width = 0;
      while (stream->Read(&message)) {
        if (message.has_terminate()) {
          const auto& end = message.terminate();
          require(sequence == open.maximum_new_tokens() &&
                      end.deployment_id() == open.deployment_id() &&
                      end.deployment_version() == open.deployment_version() &&
                      end.request_id() == open.request_id() && end.microbatch_id() == 0 &&
                      end.state() == v1::TERMINAL_STATE_COMPLETED && !end.has_error(),
                      "invalid probe termination");
          require(stream->Write(message), "termination acknowledgment failed");
          return grpc::Status::OK;
        }
        require(sequence < open.maximum_new_tokens() && message.has_tensor(), "unexpected probe message");
        const auto& tensor = message.tensor();
        require(tensor.shape_size() == 3 && tensor.shape(2) > 0 && tensor.shape(2) <= 8192,
                "invalid probe width");
        if (sequence == 0) width = static_cast<std::size_t>(tensor.shape(2));
        bytes += tensor.payload().size();
        require(bytes <= kMaxTraffic, "probe traffic limit exceeded");
        worker::BoundaryIdentity id{open.deployment_id(), open.deployment_version(),
            open.request_id(), static_cast<std::size_t>(open.maximum_total_tokens()), width};
        const auto hidden = worker::decode_tensor(tensor, id, sequence, position);
        position += hidden.tokens;
        v1::StageMessage reply;
        auto* token = reply.mutable_sampled_token();
        token->set_deployment_id(id.deployment_id);
        token->set_deployment_version(id.deployment_version);
        token->set_request_id(id.request_id);
        token->set_sequence_number(sequence++);
        token->set_token_position(position);
        token->set_token_id(0);
        require(stream->Write(reply), "feedback write failed");
      }
      throw std::invalid_argument("probe stream ended without termination");
    } catch (const std::exception& error) {
      return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
    }
  }
 private:
  v1::LinkProbeIdentity info_;
  std::map<std::string, std::string> peers_;
  std::atomic_bool active_{false};
};
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: hllm-profile-link server-config.json");
    std::ifstream input(argv[1]);
    const auto config = nlohmann::json::parse(input);
    Probe service(config);
    grpc::ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(worker::kMaximumRpcBytes);
    builder.SetMaxSendMessageSize(worker::kMaximumRpcBytes);
    builder.AddListeningPort(config.at("listen").get<std::string>(), grpc::InsecureServerCredentials());
    builder.RegisterService(static_cast<v1::WorkerControl::Service*>(&service));
    builder.RegisterService(static_cast<v1::StageExecution::Service*>(&service));
    const auto server = builder.BuildAndStart();
    if (!server) throw std::runtime_error("failed to start probe server");
    std::cout << "link probe ready" << std::endl;
    server->Wait();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
