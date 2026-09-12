// Diagnostic only: link unchanged frozen runtime; delay a test backend at cancellation.
#include <gtest/gtest.h>
#include <thread>
#include <cstdlib>
#include <iostream>
#include "hllm/worker/execution_service.hpp"
#include "model_fixture.hpp"
using namespace hllm;
struct State { std::atomic_bool slow{true}; int delay_ms=std::getenv("HLLM_DIAG_DELAY_MS")?std::stoi(std::getenv("HLLM_DIAG_DELAY_MS")):250; };
class Stage final : public runtime::StageBackend {
 public:
  explicit Stage(std::shared_ptr<State> s): s_(std::move(s)) {}
  runtime::MemoryAmounts weight_memory() const override { return {20U,0U,0U}; }
  std::size_t hidden_size() const override { return 6U; }
  std::size_t vocabulary_size() const override { return 11U; }
  std::size_t maximum_tokens() const override { return 16U; }
  runtime::SequenceMemory sequence_memory(std::size_t) const override { return {{10U,0U,0U},{10U,0U,0U}}; }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t) const override { return std::make_unique<runtime::SequenceState>(); }
  runtime::StageOutput execute(runtime::StageInput, std::size_t, runtime::SequenceState&, const std::atomic_bool& cancelled) const override {
    if(s_->slow) {
      const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
      while(!cancelled && std::chrono::steady_clock::now()<until) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::this_thread::sleep_for(std::chrono::milliseconds(s_->delay_ms));
    }
    return runtime::SampledToken{1U};
  }
 private: std::shared_ptr<State> s_;
};
class Factory final : public runtime::BackendFactory {
 public:
  explicit Factory(std::shared_ptr<State> s):s_(std::move(s)){}
  runtime::BackendCapabilities capabilities() const override { return {v1::BACKEND_CPU,v1::MEMORY_DOMAIN_HOST,{"qwen3.v1"},{v1::DATA_TYPE_F32},"diagnostic delayed backend"}; }
  std::unique_ptr<runtime::StageBackend> load(const v1::LoadStageRequest&,const std::filesystem::path&,const runtime::MemoryAmounts&) const override { return std::make_unique<Stage>(s_); }
 private: std::shared_ptr<State> s_;
};
TEST(WatchdogDiagnostic, SlowComputeDeadlineStatusAndRecovery) {
 test::ModelFixture model; auto state=std::make_shared<State>();
 worker::ControlService control({"cpu-a","127.0.0.1:0",model.root,1000000U},std::make_unique<Factory>(state));
 worker::GenerationService generation(control);
 grpc::ServerBuilder builder;int port=0;builder.AddListeningPort("127.0.0.1:0",grpc::InsecureServerCredentials(),&port);builder.RegisterService(&generation);auto server=builder.BuildAndStart();ASSERT_TRUE(server);
 auto load=model.load(0U,false);v1::LoadStageResponse loaded;ASSERT_TRUE(control.LoadStage(nullptr,&load,&loaded).ok());ASSERT_TRUE(loaded.accepted())<<loaded.detail();
 auto stub=v1::Generation::NewStub(grpc::CreateChannel("127.0.0.1:"+std::to_string(port),grpc::InsecureChannelCredentials()));
 for(int i=0;i<2;++i) {
  v1::GenerationRequest request;request.set_deployment_id("plan-1");request.set_deployment_version(1U);request.set_request_id("diag-"+std::to_string(i));request.add_token_ids(1U);request.set_maximum_new_tokens(1U);
  const auto deadline=std::chrono::system_clock::now()+std::chrono::milliseconds(i==0?200:2000);request.set_deadline_unix_ms(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count()));
  grpc::ClientContext context;context.set_deadline(deadline+std::chrono::seconds(5));
  const auto started=std::chrono::steady_clock::now();auto call=stub->Generate(&context,request);v1::GenerationEvent event;int tokens=0;while(call->Read(&event)) if(event.has_token()) ++tokens;auto status=call->Finish();
  std::cout<<"iteration="<<i<<" code="<<status.error_code()<<" detail="<<status.error_message()<<" elapsed_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<" tokens="<<tokens<<std::endl;
  // Documents the current fallback, NOT the stricter acceptance contract.
  EXPECT_EQ(status.error_code(),i==0?(state->delay_ms>100?grpc::StatusCode::CANCELLED:grpc::StatusCode::DEADLINE_EXCEEDED):grpc::StatusCode::OK);
  const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);v1::MemoryReport report;
  do { ASSERT_TRUE(control.GetMemoryReport(nullptr,nullptr,&report).ok());if(!report.active_requests())break;std::this_thread::sleep_for(std::chrono::milliseconds(5)); } while(std::chrono::steady_clock::now()<until);
  EXPECT_EQ(report.active_requests(),0U);EXPECT_EQ(report.reserved_cache_bytes(),0U);EXPECT_EQ(report.reserved_workspace_bytes(),0U);state->slow=false;
 }
 v1::UnloadStageRequest unload;unload.set_plan_id("plan-1");unload.set_deployment_version(1U);v1::Empty empty;EXPECT_TRUE(control.UnloadStage(nullptr,&unload,&empty).ok());server->Shutdown();
}
