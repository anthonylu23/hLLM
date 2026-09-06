#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "hllm/cpu/stage.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"

namespace hllm::worker {
namespace {

TEST(WorkerControlTest, LoadsReservesCancelsAndUnloadsOneCpuStage) {
  const test::ModelFixture model;
  ControlService service(
      {
          .worker_id = "cpu-a",
          .endpoint = "127.0.0.1:50051",
          .model_root = model.root,
          .host_memory_capacity_bytes = 1'000'000U,
      },
      cpu::make_backend_factory());
  grpc::ServerContext context;
  auto load = model.load();
  v1::LoadStageResponse loaded;
  EXPECT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted()) << loaded.detail();

  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("request-1");
  reserve.set_maximum_total_tokens(16U);
  v1::ReserveResponse reserved;
  EXPECT_TRUE(service.ReserveRequest(&context, &reserve, &reserved).ok());
  EXPECT_TRUE(reserved.accepted());

  // Retrying a successful load must preserve the reservation.
  v1::LoadStageResponse retried;
  ASSERT_TRUE(service.LoadStage(&context, &load, &retried).ok());
  ASSERT_TRUE(retried.accepted());
  reserve.set_request_id("request-2");
  v1::ReserveResponse second;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &second).ok());
  EXPECT_FALSE(second.accepted());
  EXPECT_EQ(second.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);

  v1::UnloadStageRequest busy_unload;
  busy_unload.set_plan_id("plan-1");
  busy_unload.set_deployment_version(1U);
  v1::Empty busy_response;
  EXPECT_EQ(service.UnloadStage(&context, &busy_unload, &busy_response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);

  load.mutable_plan()->set_plan_digest("changed-digest");
  v1::LoadStageResponse changed;
  ASSERT_TRUE(service.LoadStage(&context, &load, &changed).ok());
  EXPECT_FALSE(changed.accepted());
  EXPECT_EQ(changed.error().code(), v1::ERROR_CODE_STALE_DEPLOYMENT);

  v1::CancelRequestMessage cancel;
  cancel.set_plan_id("plan-1");
  cancel.set_deployment_version(1U);
  cancel.set_request_id("request-1");
  v1::Empty empty;
  EXPECT_TRUE(service.CancelRequest(&context, &cancel, &empty).ok());

  v1::UnloadStageRequest unload;
  unload.set_plan_id("plan-1");
  unload.set_deployment_version(1U);
  EXPECT_TRUE(service.UnloadStage(&context, &unload, &empty).ok());
}

TEST(WorkerControlTest, AdvertisesAndLoadsExecutableQwen3) {
  const test::ModelFixture model;
  ControlService service(
      {
          .worker_id = "cpu-a",
          .endpoint = "127.0.0.1:50051",
          .model_root = model.root,
          .host_memory_capacity_bytes = 1'000'000U,
      },
      cpu::make_backend_factory());
  grpc::ServerContext context;
  v1::Empty empty;
  v1::Capabilities capabilities;
  ASSERT_TRUE(service.GetCapabilities(&context, &empty, &capabilities).ok());
  ASSERT_EQ(capabilities.worker().supported_architectures_size(), 2);
  EXPECT_EQ(capabilities.worker().supported_architectures(0), "llama.v1");
  EXPECT_EQ(capabilities.worker().supported_architectures(1), "qwen3.v1");

  auto load = model.load();
  load.mutable_manifest()->mutable_architecture()->set_architecture_id("qwen3.v1");
  load.mutable_manifest()->mutable_config()->mutable_rope_scaling()->set_factor(2.0);
  v1::LoadStageResponse rejected;
  ASSERT_TRUE(service.LoadStage(&context, &load, &rejected).ok());
  EXPECT_FALSE(rejected.accepted());

  load.mutable_manifest()->mutable_config()->clear_rope_scaling();
  v1::LoadStageResponse accepted;
  ASSERT_TRUE(service.LoadStage(&context, &load, &accepted).ok());
  EXPECT_TRUE(accepted.accepted()) << accepted.detail();
}

TEST(WorkerControlTest, RejectsTraversalAndStaleDeployment) {
  const test::ModelFixture model;
  ControlService service(
      {
          .worker_id = "cpu-a",
          .endpoint = "127.0.0.1:50051",
          .model_root = model.root,
          .host_memory_capacity_bytes = 1'000'000U,
      },
      cpu::make_backend_factory());
  grpc::ServerContext context;
  auto load = model.load();
  load.mutable_manifest()->mutable_tensors(0)->set_file("../model.safetensors");
  v1::LoadStageResponse loaded;
  EXPECT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  EXPECT_FALSE(loaded.accepted());
  EXPECT_EQ(loaded.error().code(), v1::ERROR_CODE_INCOMPATIBLE_WORKER);

  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("other");
  reserve.set_deployment_version(2U);
  reserve.set_request_id("request-1");
  reserve.set_maximum_total_tokens(16U);
  v1::ReserveResponse reserved;
  EXPECT_TRUE(service.ReserveRequest(&context, &reserve, &reserved).ok());
  EXPECT_FALSE(reserved.accepted());
  EXPECT_EQ(reserved.error().code(), v1::ERROR_CODE_STALE_DEPLOYMENT);
}

}  // namespace
}  // namespace hllm::worker

namespace hllm::worker {
namespace {
TEST(WorkerControlTest, RejectsOversizedAndConflictingReservationsAndReportsAllocations) {
  const test::ModelFixture model;
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 100'000U},
                         cpu::make_backend_factory());
  grpc::ServerContext context;
  auto load = model.load();
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted());
  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("one");
  reserve.set_maximum_total_tokens(32U);
  v1::ReserveResponse rejected;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &rejected).ok());
  EXPECT_FALSE(rejected.accepted());
  EXPECT_EQ(rejected.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
  reserve.set_maximum_total_tokens(4U);
  v1::ReserveResponse accepted;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &accepted).ok());
  ASSERT_TRUE(accepted.accepted());
  reserve.set_maximum_total_tokens(5U);
  v1::ReserveResponse conflict;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &conflict).ok());
  EXPECT_FALSE(conflict.accepted());
  v1::Empty empty;
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(&context, &empty, &report).ok());
  EXPECT_EQ(report.active_requests(), 1U);
  EXPECT_GT(report.loaded_weight_bytes(), 0U);
  EXPECT_EQ(report.reserved_cache_bytes(), 4U * 2U * 2U * 4U * 4U);
  EXPECT_GT(report.reserved_workspace_bytes(), 0U);
}
}  // namespace
}  // namespace hllm::worker
