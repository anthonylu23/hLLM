#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "hllm/cpu/stage.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"

namespace hllm::worker {
namespace {

TEST(WorkerControlTest, ProcessMetricsAreAvailableWithoutALoadedStage) {
  const test::ModelFixture model;
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 1'000'000U},
                         cpu::make_backend_factory());
  v1::WorkerMetrics metrics;
  // Reused protobuf responses must not retain counters absent on this backend/OS.
  metrics.mutable_allocator()->set_active_bytes(123U);
  metrics.mutable_process_memory()->set_physical_footprint_bytes(123U);
  ASSERT_TRUE(service.GetMetrics(nullptr, nullptr, &metrics).ok());
  EXPECT_FALSE(metrics.has_allocator());
  ASSERT_TRUE(metrics.has_process_memory());
  const auto& process = metrics.process_memory();
  EXPECT_GT(process.process_id(), 0U);
  EXPECT_GT(process.observed_at_unix_ns(), 0U);
  EXPECT_TRUE(process.has_rss_bytes());
  EXPECT_GT(process.rss_bytes(), 0U);
  EXPECT_TRUE(process.has_rss_lifetime_peak_bytes());
#ifdef __APPLE__
  EXPECT_TRUE(process.has_physical_footprint_bytes());
  EXPECT_GT(process.physical_footprint_bytes(), 0U);
  EXPECT_TRUE(process.has_physical_footprint_lifetime_peak_bytes());
#else
  EXPECT_FALSE(process.has_physical_footprint_bytes());
  EXPECT_FALSE(process.has_physical_footprint_lifetime_peak_bytes());
#endif
}

TEST(WorkerControlTest, CpuRejectsMixedPrecisionBeforeLoading) {
  const test::ModelFixture model;
  auto factory = cpu::make_backend_factory();
  EXPECT_FALSE(factory->capabilities().supports_mixed_precision);
  auto request = model.load();
  request.mutable_plan()->mutable_schema_version()->set_minor(2U);
  request.mutable_plan()->set_weight_dtype(v1::DATA_TYPE_F16);
  EXPECT_THROW(static_cast<void>(factory->load(request, model.root, {1'000'000U, 0U, 0U})), std::exception);
  ControlService service({.worker_id = "cpu-a", .endpoint = "127.0.0.1:50051",
                          .model_root = model.root, .host_memory_capacity_bytes = 1'000'000U},
                         std::move(factory));
  grpc::ServerContext context;
  v1::LoadStageResponse rejected;
  ASSERT_TRUE(service.LoadStage(&context, &request, &rejected).ok());
  EXPECT_FALSE(rejected.accepted());
  EXPECT_EQ(rejected.error().code(), v1::ERROR_CODE_INCOMPATIBLE_WORKER);
}

TEST(WorkerControlTest, VersionedPlanErrorsAreIncompatibleAndLeaveWorkerUnloaded) {
  const test::ModelFixture model;
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 1'000'000U},
                         cpu::make_backend_factory());
  for (int scenario = 0; scenario < 6; ++scenario) {
    SCOPED_TRACE(scenario);
    auto request = model.load();
    auto* plan = request.mutable_plan();
    if (scenario < 4) {
      plan->mutable_schema_version()->set_minor(1U);
      plan->set_planning_mode("measured");
      plan->set_workload_digest(std::string(64, 'a'));
      plan->set_profile_bundle_digest(std::string(64, 'b'));
    }
    switch (scenario) {
      case 0: plan->clear_workload_digest(); break;
      case 1: plan->set_profile_bundle_digest("invalid"); break;
      case 2: plan->mutable_schema_version()->set_minor(0U); break;
      case 3: break;  // Valid measured identities, but the old plan hash is invalid.
      case 4: plan->set_workload_digest(std::string(64, 'a')); break;
      case 5: plan->mutable_schema_version()->set_minor(1U); break;
    }
    v1::LoadStageResponse rejected;
    ASSERT_TRUE(service.LoadStage(nullptr, &request, &rejected).ok());
    EXPECT_FALSE(rejected.accepted());
    EXPECT_EQ(rejected.error().code(), v1::ERROR_CODE_INCOMPATIBLE_WORKER);
    if (scenario == 5) {
      EXPECT_NE(rejected.detail().find("schema version"), std::string::npos);
    }
    v1::MemoryReport memory;
    ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &memory).ok());
    EXPECT_EQ(memory.loaded_weight_bytes(), 0U);
    EXPECT_EQ(memory.active_requests(), 0U);
  }
  auto valid = model.load();
  v1::LoadStageResponse accepted;
  ASSERT_TRUE(service.LoadStage(nullptr, &valid, &accepted).ok());
  EXPECT_TRUE(accepted.accepted()) << accepted.detail();
}

TEST(WorkerControlTest, UnknownPlanDtypesAreIncompatibleNotBackendErrors) {
  const test::ModelFixture model;
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 1'000'000U},
                         cpu::make_backend_factory());
  for (const bool activation : {false, true}) {
    SCOPED_TRACE(activation);
    auto request = model.load();
    auto* plan = request.mutable_plan();
    // A measured plan reaches the digest check, which names both dtypes.
    plan->mutable_schema_version()->set_minor(1U);
    plan->set_planning_mode("measured");
    plan->set_workload_digest(std::string(64, 'a'));
    plan->set_profile_bundle_digest(std::string(64, 'b'));
    const auto unknown = static_cast<v1::DataType>(99);
    if (activation) {
      plan->set_activation_dtype(unknown);
    } else {
      plan->set_execution_dtype(unknown);
    }
    v1::LoadStageResponse rejected;
    ASSERT_TRUE(service.LoadStage(nullptr, &request, &rejected).ok());
    EXPECT_FALSE(rejected.accepted());
    EXPECT_EQ(rejected.error().code(), v1::ERROR_CODE_INCOMPATIBLE_WORKER) << rejected.detail();
  }
}

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

TEST(WorkerControlTest, DistinguishesExpiredDeadlinesFromInvalidOnes) {
  const test::ModelFixture model;
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 1'000'000U},
                         cpu::make_backend_factory());
  grpc::ServerContext context;
  auto load = model.load();
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted()) << loaded.detail();
  const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("late");
  reserve.set_maximum_total_tokens(16U);
  reserve.set_deadline_unix_ms(static_cast<std::uint64_t>(now) - 1U);
  v1::ReserveResponse expired;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &expired).ok());
  EXPECT_FALSE(expired.accepted());
  EXPECT_EQ(expired.error().code(), v1::ERROR_CODE_DEADLINE_EXCEEDED);
  reserve.set_deadline_unix_ms(static_cast<std::uint64_t>(now) + 2U * 3'600'000U);
  v1::ReserveResponse distant;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &distant).ok());
  EXPECT_FALSE(distant.accepted());
  EXPECT_EQ(distant.error().code(), v1::ERROR_CODE_INVALID_REQUEST);
  v1::Empty empty;
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(&context, &empty, &report).ok());
  EXPECT_EQ(report.active_requests(), 0U);
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
  // Four tokens of KV cache plus the handlers' transport buffers.
  const auto hidden =
      cpu::make_backend_factory()->load(load, model.root, {1'000'000U, 0U, 0U})->hidden_size();
  EXPECT_EQ(report.reserved_cache_bytes(),
            4U * 2U * 2U * 4U * 4U + runtime::request_transport_bytes(4U, hidden));
  EXPECT_GT(report.reserved_workspace_bytes(), 0U);
}
}  // namespace
}  // namespace hllm::worker

namespace hllm::worker {
TEST(WorkerConcurrencyTest, AccountsCombinedReservationsAndCancelsOnlyOneRequest) {
  const test::ModelFixture model;
  ControlService service({.worker_id = "cpu-a",
                          .endpoint = "127.0.0.1:50051",
                          .model_root = model.root,
                          .host_memory_capacity_bytes = 1'000'000U,
                          .maximum_active_requests = 3U,
                          .maximum_cached_tokens = 32U},
                         cpu::make_backend_factory());
  grpc::ServerContext context;
  auto load = model.load();
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted());
  v1::ReserveRequestMessage request;
  request.set_plan_id("plan-1");
  request.set_deployment_version(1U);
  request.set_maximum_total_tokens(16U);
  request.set_request_id("a");
  v1::ReserveResponse accepted;
  service.ReserveRequest(&context, &request, &accepted);
  ASSERT_TRUE(accepted.accepted());
  v1::Empty empty;
  v1::MemoryReport one;
  service.GetMemoryReport(&context, &empty, &one);
  request.set_request_id("b");
  service.ReserveRequest(&context, &request, &accepted);
  ASSERT_TRUE(accepted.accepted());
  v1::MemoryReport two;
  service.GetMemoryReport(&context, &empty, &two);
  EXPECT_EQ(two.active_requests(), 2U);
  EXPECT_EQ(two.loaded_weight_bytes(), one.loaded_weight_bytes());
  EXPECT_EQ(two.reserved_cache_bytes(), 2U * one.reserved_cache_bytes());
  // Execution is serialized, so both requests share one workspace reservation.
  EXPECT_EQ(two.reserved_workspace_bytes(), one.reserved_workspace_bytes());
  request.set_request_id("c");
  service.ReserveRequest(&context, &request, &accepted);
  EXPECT_FALSE(accepted.accepted());
  EXPECT_EQ(accepted.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
  v1::CancelRequestMessage cancel;
  cancel.set_plan_id("plan-1");
  cancel.set_deployment_version(1U);
  cancel.set_request_id("a");
  service.CancelRequest(&context, &cancel, &empty);
  v1::MemoryReport remaining;
  service.GetMemoryReport(&context, &empty, &remaining);
  EXPECT_EQ(remaining.active_requests(), 1U);
  EXPECT_EQ(remaining.reserved_cache_bytes(), one.reserved_cache_bytes());
  auto lease = service.acquire("plan-1", 1U, "b", 16U, 0U, 0U);
  EXPECT_FALSE(lease.request->cancelled.load());
  service.release(lease.request, RequestOutcome::kCompleted);
  v1::MemoryReport clean;
  service.GetMemoryReport(&context, &empty, &clean);
  EXPECT_EQ(clean.active_requests(), 0U);
  EXPECT_EQ(clean.reserved_cache_bytes(), 0U);
}

TEST(WorkerConcurrencyTest, RequestObservationsRecordOverlapAndRetirementOnlyWhenEnabled) {
  const test::ModelFixture model;
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    ControlService service({.worker_id = "cpu-a",
                            .endpoint = "127.0.0.1:50051",
                            .model_root = model.root,
                            .host_memory_capacity_bytes = 1'000'000U,
                            .maximum_active_requests = 2U,
                            .request_observations = enabled},
                           cpu::make_backend_factory());
    grpc::ServerContext context;
    v1::Capabilities capabilities;
    ASSERT_TRUE(service.GetCapabilities(&context, nullptr, &capabilities).ok());
    EXPECT_EQ(capabilities.worker().supports_request_observations(), enabled);
    auto load = model.load();
    v1::LoadStageResponse loaded;
    ASSERT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
    ASSERT_TRUE(loaded.accepted());
    v1::ReserveRequestMessage request;
    request.set_plan_id("plan-1");
    request.set_deployment_version(1U);
    request.set_maximum_total_tokens(16U);
    v1::ReserveResponse accepted;
    v1::Empty empty;
    v1::MemoryReport report;
    // Reused responses must not retain observations from an enabled worker.
    report.mutable_request_observations()->set_events_total(99U);
    for (const auto* id : {"a", "b"}) {
      request.set_request_id(id);
      service.ReserveRequest(&context, &request, &accepted);
      ASSERT_TRUE(accepted.accepted());
    }
    ASSERT_TRUE(service.GetMemoryReport(&context, &empty, &report).ok());
    EXPECT_EQ(report.active_requests(), 2U);
    if (!enabled) {
      EXPECT_FALSE(report.has_request_observations());
      continue;
    }
    ASSERT_TRUE(report.has_request_observations());
    const auto& live = report.request_observations();
    EXPECT_GT(live.process_id(), 0U);
    EXPECT_GT(live.observed_at_monotonic_ns(), 0U);
    EXPECT_GT(live.observed_at_unix_ns(), 0U);
    ASSERT_EQ(live.requests_size(), 2);
    for (const auto& row : live.requests()) {
      EXPECT_EQ(row.maximum_total_tokens(), 16U);
      EXPECT_FALSE(row.allocating());
      EXPECT_FALSE(row.running());
      EXPECT_FALSE(row.cancelled());
      EXPECT_LE(row.admitted_at_monotonic_ns(), live.observed_at_monotonic_ns());
      ASSERT_EQ(row.memory_size(), 1);
      EXPECT_EQ(row.memory(0).domain(), v1::MEMORY_DOMAIN_HOST);
      EXPECT_GT(row.memory(0).cache_bytes(), 0U);
    }
    // Per-request cache bytes sum to the aggregate; weights are not per request.
    std::uint64_t cache = 0U;
    for (const auto& row : live.requests()) cache += row.memory(0).cache_bytes();
    EXPECT_EQ(cache, report.reserved_cache_bytes());
    ASSERT_EQ(live.events_size(), 2);
    EXPECT_EQ(live.events_total(), 2U);
    EXPECT_EQ(live.peak_concurrent_requests(), 2U);
    EXPECT_EQ(live.events(0).kind(), v1::RequestLifecycleEvent::REQUEST_LIFECYCLE_KIND_ADMITTED);
    EXPECT_EQ(live.events(0).request_id(), "a");
    EXPECT_EQ(live.events(0).concurrent_request_ids_size(), 0);
    EXPECT_EQ(live.events(1).request_id(), "b");
    ASSERT_EQ(live.events(1).concurrent_request_ids_size(), 1);
    EXPECT_EQ(live.events(1).concurrent_request_ids(0), "a");
    EXPECT_LE(live.events(0).at_monotonic_ns(), live.events(1).at_monotonic_ns());
    EXPECT_LT(live.events(0).sequence(), live.events(1).sequence());
    // Cancel one; the other keeps its reservation and later completes.
    v1::CancelRequestMessage cancel;
    cancel.set_plan_id("plan-1");
    cancel.set_deployment_version(1U);
    cancel.set_request_id("a");
    ASSERT_TRUE(service.CancelRequest(&context, &cancel, &empty).ok());
    ASSERT_TRUE(service.GetMemoryReport(&context, &empty, &report).ok());
    ASSERT_EQ(report.request_observations().requests_size(), 1);
    EXPECT_EQ(report.request_observations().requests(0).request_id(), "b");
    const auto& retired = report.request_observations().events(2);
    EXPECT_EQ(retired.kind(), v1::RequestLifecycleEvent::REQUEST_LIFECYCLE_KIND_RETIRED);
    EXPECT_EQ(retired.request_id(), "a");
    EXPECT_EQ(retired.reason(), "cancelled");
    ASSERT_EQ(retired.concurrent_request_ids_size(), 1);
    EXPECT_EQ(retired.concurrent_request_ids(0), "b");
    auto lease = service.acquire("plan-1", 1U, "b", 16U, 0U, 0U);
    service.release(lease.request, RequestOutcome::kCompleted);
    ASSERT_TRUE(service.GetMemoryReport(&context, &empty, &report).ok());
    EXPECT_EQ(report.request_observations().requests_size(), 0);
    ASSERT_EQ(report.request_observations().events_size(), 4);
    EXPECT_EQ(report.request_observations().events(3).reason(), "completed");
    EXPECT_EQ(report.request_observations().events_total(), 4U);
    EXPECT_EQ(report.request_observations().peak_concurrent_requests(), 2U);
  }
}

TEST(WorkerConcurrencyTest, SharedWorkspaceFollowsTheLargestActiveRequest) {
  const test::ModelFixture model;
  auto factory = cpu::make_backend_factory();
  const auto load = model.load();
  auto stage = factory->load(load, model.root, {1'000'000U, 0U, 0U});
  const auto small = stage->sequence_memory(16U);
  const auto large = stage->sequence_memory(32U);
  ASSERT_LT(small.workspace.host_bytes, large.workspace.host_bytes);
  // The handlers hold transport buffers between passes, so each request's own cache,
  // not the shared workspace, covers them.
  EXPECT_GE(small.cache.host_bytes, runtime::request_transport_bytes(16U, stage->hidden_size()));
  EXPECT_GE(large.cache.host_bytes, runtime::request_transport_bytes(32U, stage->hidden_size()));
  // Weights, both caches and only the larger workspace.
  const auto exact = stage->weight_memory().host_bytes + small.cache.host_bytes +
                     large.cache.host_bytes + large.workspace.host_bytes;
  for (const bool short_budget : {false, true}) {
    SCOPED_TRACE(short_budget);
    ControlService service({.worker_id = "cpu-a",
                            .endpoint = "127.0.0.1:50051",
                            .model_root = model.root,
                            .host_memory_capacity_bytes = exact - (short_budget ? 1U : 0U),
                            .maximum_active_requests = 2U},
                           cpu::make_backend_factory());
    grpc::ServerContext context;
    v1::LoadStageResponse loaded;
    service.LoadStage(&context, &load, &loaded);
    ASSERT_TRUE(loaded.accepted());
    auto first = service.acquire("plan-1", 1U, "large", 32U, 0U, 0U);
    if (short_budget) {
      EXPECT_THROW(static_cast<void>(service.acquire("plan-1", 1U, "small", 16U, 0U, 0U)),
                   runtime::Error);
      service.release(first.request, RequestOutcome::kCompleted);
      continue;
    }
    auto second = service.acquire("plan-1", 1U, "small", 16U, 0U, 0U);
    v1::Empty empty;
    v1::MemoryReport report;
    service.GetMemoryReport(&context, &empty, &report);
    EXPECT_EQ(report.reserved_cache_bytes(), small.cache.host_bytes + large.cache.host_bytes);
    EXPECT_EQ(report.reserved_workspace_bytes(), large.workspace.host_bytes);
    // Retiring the larger request shrinks the shared workspace to the smaller one.
    service.release(first.request, RequestOutcome::kCompleted);
    service.GetMemoryReport(&context, &empty, &report);
    EXPECT_EQ(report.reserved_workspace_bytes(), small.workspace.host_bytes);
    service.release(second.request, RequestOutcome::kCompleted);
  }
}

TEST(WorkerConcurrencyTest, ChunkedReservationAdmitsLessWorkspaceAndBindsTheChunk) {
  const test::ModelFixture model;
  const auto load = model.load();
  auto stage = cpu::make_backend_factory()->load(load, model.root, {1'000'000U, 0U, 0U});
  constexpr std::size_t tokens = 32U;
  const auto whole = stage->sequence_memory(tokens);
  const auto chunked = stage->sequence_memory(tokens, 4U);
  ASSERT_LT(chunked.workspace.host_bytes, whole.workspace.host_bytes);
  // Fits the chunked reservation exactly, but not the whole-prompt one.
  ControlService service({.worker_id = "cpu-a",
                          .endpoint = "127.0.0.1:50051",
                          .model_root = model.root,
                          .host_memory_capacity_bytes = stage->weight_memory().host_bytes +
                                                        chunked.cache.host_bytes +
                                                        chunked.workspace.host_bytes,
                          .maximum_active_requests = 2U},
                         cpu::make_backend_factory());
  grpc::ServerContext context;
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted()) << loaded.detail();
  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("one");
  reserve.set_maximum_total_tokens(tokens);
  for (const std::uint32_t whole_prompt : {0U, 32U, 100U}) {
    reserve.set_prefill_chunk_tokens(whole_prompt);
    v1::ReserveResponse rejected;
    ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &rejected).ok());
    EXPECT_FALSE(rejected.accepted());
    EXPECT_EQ(rejected.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
  }
  reserve.set_prefill_chunk_tokens(4U);
  v1::ReserveResponse accepted;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &accepted).ok());
  ASSERT_TRUE(accepted.accepted()) << accepted.detail();
  v1::Empty empty;
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(&context, &empty, &report).ok());
  EXPECT_EQ(report.reserved_cache_bytes(), chunked.cache.host_bytes);
  EXPECT_EQ(report.reserved_workspace_bytes(), chunked.workspace.host_bytes);
  // A retry must name the same chunk; another chunk is a conflicting reservation.
  for (const std::uint32_t other : {0U, 3U, 8U}) {
    reserve.set_prefill_chunk_tokens(other);
    v1::ReserveResponse conflict;
    ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &conflict).ok());
    EXPECT_FALSE(conflict.accepted());
    EXPECT_EQ(conflict.error().code(), v1::ERROR_CODE_INVALID_REQUEST);
  }
  EXPECT_THROW(static_cast<void>(service.acquire("plan-1", 1U, "one", tokens, 0U, 0U)),
               runtime::Error);
  reserve.set_prefill_chunk_tokens(4U);
  v1::ReserveResponse retried;
  ASSERT_TRUE(service.ReserveRequest(&context, &reserve, &retried).ok());
  EXPECT_TRUE(retried.accepted());
  auto lease = service.acquire("plan-1", 1U, "one", tokens, 0U, 0U, 4U);
  EXPECT_EQ(lease.request->maximum_query_tokens, 4U);
  service.release(lease.request, RequestOutcome::kCompleted);
  // Whole-prompt requests record the full sequence as their query bound.
  ControlService roomy({"cpu-a", "127.0.0.1:50051", model.root, 1'000'000U},
                       cpu::make_backend_factory());
  ASSERT_TRUE(roomy.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted());
  auto full = roomy.acquire("plan-1", 1U, "two", tokens, 0U, 0U);
  EXPECT_EQ(full.request->maximum_query_tokens, tokens);
  roomy.release(full.request, RequestOutcome::kCompleted);
}

TEST(WorkerConcurrencyTest, CombinedMemoryAdmissionRejectsSecondRequest) {
  const test::ModelFixture model;
  auto factory = cpu::make_backend_factory();
  const auto load = model.load();
  auto stage = factory->load(load, model.root, {1'000'000U, 0U, 0U});
  const auto memory = stage->sequence_memory(16U);
  const auto limit = runtime::add_memory(stage->weight_memory(),
                                         runtime::add_memory(memory.cache, memory.workspace))
                         .host_bytes;
  ControlService service({.worker_id = "cpu-a",
                          .endpoint = "127.0.0.1:50051",
                          .model_root = model.root,
                          .host_memory_capacity_bytes = limit,
                          .maximum_active_requests = 2U},
                         std::move(factory));
  grpc::ServerContext context;
  v1::LoadStageResponse loaded;
  service.LoadStage(&context, &load, &loaded);
  ASSERT_TRUE(loaded.accepted());
  auto first = service.acquire("plan-1", 1U, "a", 16U, 0U, 0U);
  EXPECT_THROW(static_cast<void>(service.acquire("plan-1", 1U, "b", 16U, 0U, 0U)), runtime::Error);
  service.release(first.request, RequestOutcome::kCompleted);
  auto second = service.acquire("plan-1", 1U, "b", 16U, 0U, 0U);
  service.release(second.request, RequestOutcome::kCompleted);
}
}  // namespace hllm::worker
