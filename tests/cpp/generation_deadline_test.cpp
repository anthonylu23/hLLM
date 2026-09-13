#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "hllm/worker/execution_service.hpp"
#include "model_fixture.hpp"

namespace hllm::worker {
namespace {
using namespace std::chrono_literals;

struct ExecutionState {
  std::atomic_bool delay_decode{true};
  std::atomic_size_t steps{0U};
  std::chrono::milliseconds unwind{250};
};

// A private backend simulates non-preemptible work without a GPU or timing hooks
// in production worker arguments/RPCs.
class DeadlineStage final : public runtime::StageBackend {
 public:
  explicit DeadlineStage(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}
  runtime::MemoryAmounts weight_memory() const override { return {20U, 0U, 0U}; }
  std::size_t hidden_size() const override { return 6U; }
  std::size_t vocabulary_size() const override { return 11U; }
  std::size_t maximum_tokens() const override { return 1'000'001U; }
  runtime::SequenceMemory sequence_memory(std::size_t) const override {
    return {{10U, 0U, 0U}, {10U, 0U, 0U}};
  }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t) const override {
    return std::make_unique<runtime::SequenceState>();
  }
  runtime::StageOutput execute(runtime::StageInput, std::size_t position,
                               runtime::SequenceState&, const std::atomic_bool& cancelled) const override {
    ++state_->steps;
    if (position > 0U && state_->delay_decode) {
      const auto until = std::chrono::steady_clock::now() + 5s;
      while (!cancelled && std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(1ms);
      }
      std::this_thread::sleep_for(state_->unwind);
    }
    return runtime::SampledToken{1U};
  }
 private:
  std::shared_ptr<ExecutionState> state_;
};

class DeadlineFactory final : public runtime::BackendFactory {
 public:
  explicit DeadlineFactory(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}
  runtime::BackendCapabilities capabilities() const override {
    return {v1::BACKEND_CPU, v1::MEMORY_DOMAIN_HOST, {"qwen3.v1"},
            {v1::DATA_TYPE_F32}, "test deadline backend"};
  }
  std::unique_ptr<runtime::StageBackend> load(const v1::LoadStageRequest&,
      const std::filesystem::path&, const runtime::MemoryAmounts&) const override {
    return std::make_unique<DeadlineStage>(state_);
  }
 private:
  std::shared_ptr<ExecutionState> state_;
};

class GenerationDeadlineTest : public ::testing::Test {
 protected:
  test::ModelFixture model;
  std::shared_ptr<ExecutionState> state = std::make_shared<ExecutionState>();
  ControlService control{{"cpu-a", "127.0.0.1:0", model.root, 1'000'000U},
                         std::make_unique<DeadlineFactory>(state)};
  GenerationService generation{control};
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<v1::Generation::Stub> stub;

  void SetUp() override {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&generation);
    server = builder.BuildAndStart();
    ASSERT_TRUE(server);
    auto load = model.load(0U, false);
    v1::LoadStageResponse loaded;
    ASSERT_TRUE(control.LoadStage(nullptr, &load, &loaded).ok());
    ASSERT_TRUE(loaded.accepted()) << loaded.detail();
    grpc::ChannelArguments arguments;
    arguments.SetInt(GRPC_ARG_HTTP2_BDP_PROBE, 0);
    arguments.SetInt(GRPC_ARG_HTTP2_STREAM_LOOKAHEAD_BYTES, 0);
    stub = v1::Generation::NewStub(grpc::CreateCustomChannel(
        "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials(), arguments));
  }
  void TearDown() override { if (server) server->Shutdown(); }
  v1::GenerationRequest request(std::uint64_t count, std::chrono::milliseconds duration) {
    v1::GenerationRequest result;
    result.set_deployment_id("plan-1");
    result.set_deployment_version(1U);
    result.set_request_id("deadline-test");
    result.add_token_ids(1U);
    result.set_maximum_new_tokens(count);
    result.set_deadline_unix_ms(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() + duration).time_since_epoch()).count()));
    return result;
  }
  void clean() {
    const auto until = std::chrono::steady_clock::now() + 5s;
    v1::MemoryReport report;
    do {
      ASSERT_TRUE(control.GetMemoryReport(nullptr, nullptr, &report).ok());
      if (!report.active_requests()) break;
      std::this_thread::sleep_for(5ms);
    } while (std::chrono::steady_clock::now() < until);
    ASSERT_EQ(report.active_requests(), 0U);
    EXPECT_EQ(report.reserved_cache_bytes(), 0U);
    EXPECT_EQ(report.reserved_workspace_bytes(), 0U);
  }
  void recover_and_unload() {
    state->delay_decode = false;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 5s);
    auto call = stub->Generate(&context, request(1U, 2s));
    v1::GenerationEvent event;
    int tokens = 0;
    bool completed = false;
    while (call->Read(&event)) {
      if (event.has_token()) { ++tokens; EXPECT_EQ(event.token().token_id(), 1U); }
      if (event.has_terminal()) completed = event.terminal().state() == v1::TERMINAL_STATE_COMPLETED;
    }
    EXPECT_TRUE(call->Finish().ok());
    EXPECT_EQ(tokens, 1);
    EXPECT_TRUE(completed);
    clean();
    v1::UnloadStageRequest unload;
    unload.set_plan_id("plan-1");
    unload.set_deployment_version(1U);
    v1::Empty empty;
    EXPECT_TRUE(control.UnloadStage(nullptr, &unload, &empty).ok());
  }
};

TEST(ExecutionDeadlineTest, SlowDecodeUnwindPreservesDeadlineStatus) {
  test::ModelFixture model;
  auto state = std::make_shared<ExecutionState>();
  ControlService control{{"cpu-b", "127.0.0.1:0", model.root, 1'000'000U},
                         std::make_unique<DeadlineFactory>(state)};
  ExecutionService execution{control};
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&execution);
  auto server = builder.BuildAndStart();
  ASSERT_TRUE(server);
  auto load = model.load(1U, true);
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(control.LoadStage(nullptr, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted());
  auto stub = v1::StageExecution::NewStub(grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 10s);
  auto call = stub->Execute(&context);
  v1::StageMessage message;
  auto* open = message.mutable_open_sequence();
  open->set_protocol_version(1U);
  open->set_deployment_id("plan-1");
  open->set_deployment_version(1U);
  open->set_request_id("downstream-deadline");
  open->set_maximum_total_tokens(3U);
  open->set_maximum_new_tokens(2U);
  open->set_deadline_unix_ms(static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (std::chrono::system_clock::now() + 500ms).time_since_epoch()).count()));
  ASSERT_TRUE(call->Write(message));
  for (std::uint64_t step = 0; step < 2; ++step) {
    message.Clear();
    auto* tensor = message.mutable_tensor();
    tensor->set_protocol_version(1U);
    tensor->set_deployment_id("plan-1");
    tensor->set_deployment_version(1U);
    tensor->set_request_id("downstream-deadline");
    tensor->set_sequence_number(step);
    tensor->set_phase(step == 0 ? v1::EXECUTION_PHASE_PREFILL : v1::EXECUTION_PHASE_DECODE);
    tensor->set_first_position(step);
    tensor->add_sequence_lengths(1U);
    tensor->add_cache_slot_ids(0U);
    tensor->add_shape(1U); tensor->add_shape(1U); tensor->add_shape(6U);
    tensor->set_dtype(v1::DATA_TYPE_F16);
    tensor->set_layout("dense_row_major_le");
    tensor->set_payload_length(12U);
    tensor->set_payload(std::string(12, '\0'));
    ASSERT_TRUE(call->Write(message));
    if (step == 0) {
      ASSERT_TRUE(call->Read(&message));
      ASSERT_TRUE(message.has_sampled_token());
    }
  }
  call->WritesDone();
  while (call->Read(&message)) {}
  EXPECT_EQ(call->Finish().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
  v1::MemoryReport memory;
  ASSERT_TRUE(control.GetMemoryReport(nullptr, nullptr, &memory).ok());
  EXPECT_EQ(memory.active_requests(), 0U);
  EXPECT_EQ(memory.reserved_cache_bytes(), 0U);
  EXPECT_EQ(memory.reserved_workspace_bytes(), 0U);
  server->Shutdown();
}

TEST_F(GenerationDeadlineTest, SlowDecodeUnwindPreservesDeadlineStatus) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 10s);
  auto call = stub->Generate(&context, request(2U, 300ms));
  v1::GenerationEvent event;
  int tokens = 0;
  while (call->Read(&event)) {
    if (event.has_token()) ++tokens;
    EXPECT_FALSE(event.has_terminal());
  }
  EXPECT_EQ(call->Finish().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
  EXPECT_EQ(tokens, 1);
  clean();
  recover_and_unload();
}

TEST_F(GenerationDeadlineTest, ControlCancellationStillInterruptsSlowDecode) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 10s);
  auto call = stub->Generate(&context, request(2U, 5s));
  v1::GenerationEvent event;
  while (call->Read(&event) && !event.has_token()) {}
  ASSERT_TRUE(event.has_token());
  v1::CancelRequestMessage cancel;
  cancel.set_plan_id("plan-1");
  cancel.set_deployment_version(1U);
  cancel.set_request_id("deadline-test");
  v1::Empty empty;
  ASSERT_TRUE(control.CancelRequest(nullptr, &cancel, &empty).ok());
  while (call->Read(&event)) EXPECT_FALSE(event.has_terminal());
  EXPECT_EQ(call->Finish().error_code(), grpc::StatusCode::CANCELLED);
  clean();
  recover_and_unload();
}

TEST_F(GenerationDeadlineTest, DeadlineUnblocksBackpressuredClientWrite) {
  state->delay_decode = false;
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 20s);
  auto call = stub->Generate(&context, request(1'000'000U, 3s));
  // Do not read: wait for HTTP/2 flow control to stall native writes before the
  // application deadline. The long client deadline cannot cause early cleanup.
  const auto until = std::chrono::steady_clock::now() + 2s;
  bool stalled = false;
  while (std::chrono::steady_clock::now() < until) {
    const auto before = state->steps.load();
    std::this_thread::sleep_for(100ms);
    if (before > 0U && before == state->steps.load()) { stalled = true; break; }
  }
  EXPECT_TRUE(stalled);
  clean();
  v1::GenerationEvent event;
  while (call->Read(&event)) EXPECT_FALSE(event.has_terminal());
  EXPECT_EQ(call->Finish().error_code(), grpc::StatusCode::CANCELLED);
  recover_and_unload();
}
}  // namespace
}  // namespace hllm::worker
