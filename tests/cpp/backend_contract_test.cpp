#include <gtest/gtest.h>

#include <limits>

#include "hllm/runtime/error.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"

namespace hllm::worker {
namespace {

struct AllocationPlan {
  runtime::MemoryAmounts weights{20U, 100U, 0U};
  runtime::SequenceMemory sequence{{0U, 200U, 0U}, {50U, 100U, 40U}};
  int allocations{0};
};

class TestStage final : public runtime::StageBackend {
 public:
  explicit TestStage(std::shared_ptr<AllocationPlan> plan) : plan_(std::move(plan)) {}
  runtime::MemoryAmounts weight_memory() const override { return plan_->weights; }
  std::size_t hidden_size() const override { return 8U; }
  std::size_t vocabulary_size() const override { return 16U; }
  std::size_t maximum_tokens() const override { return 16U; }
  runtime::SequenceMemory sequence_memory(std::size_t) const override { return plan_->sequence; }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t) const override {
    ++plan_->allocations;
    return std::make_unique<runtime::SequenceState>();
  }
  runtime::StageOutput execute(runtime::StageInput, std::size_t, runtime::SequenceState&,
                               const std::atomic_bool&) const override {
    return runtime::SampledToken{1U};
  }

 private:
  std::shared_ptr<AllocationPlan> plan_;
};
class TestFactory final : public runtime::BackendFactory {
 public:
  explicit TestFactory(std::shared_ptr<AllocationPlan> plan) : plan_(std::move(plan)) {}
  runtime::BackendCapabilities capabilities() const override {
    return {v1::BACKEND_CUDA,
            v1::MEMORY_DOMAIN_DEVICE,
            {"qwen3.v1"},
            {v1::DATA_TYPE_F32},
            "test backend"};
  }
  std::unique_ptr<runtime::StageBackend> load(const v1::LoadStageRequest&,
                                              const std::filesystem::path&,
                                              const runtime::MemoryAmounts&) const override {
    return std::make_unique<TestStage>(plan_);
  }

 private:
  std::shared_ptr<AllocationPlan> plan_;
};

TEST(BackendContractTest, AdmissionEnforcesEveryDomainBeforeAllocation) {
  const test::ModelFixture model;
  // Spare device memory cannot fund host/pinned allocations, nor vice versa.
  for (const runtime::MemoryAmounts budget :
       {runtime::MemoryAmounts{69U, 1000U, 40U}, {1000U, 399U, 40U}, {1000U, 1000U, 39U}}) {
    auto plan = std::make_shared<AllocationPlan>();
    ControlService service({"cpu-a", "127.0.0.1:50051", model.root, budget.host_bytes,
                            budget.device_bytes, budget.pinned_host_bytes},
                           std::make_unique<TestFactory>(plan));
    auto load = model.load();
    v1::LoadStageResponse loaded;
    ASSERT_TRUE(service.LoadStage(nullptr, &load, &loaded).ok());
    ASSERT_TRUE(loaded.accepted()) << loaded.detail();
    v1::ReserveRequestMessage request;
    request.set_plan_id("plan-1");
    request.set_deployment_version(1U);
    request.set_request_id("one");
    request.set_maximum_total_tokens(4U);
    v1::ReserveResponse reserved;
    ASSERT_TRUE(service.ReserveRequest(nullptr, &request, &reserved).ok());
    EXPECT_FALSE(reserved.accepted());
    EXPECT_EQ(reserved.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(plan->allocations, 0);
    v1::MemoryReport report;
    ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
    EXPECT_EQ(report.active_requests(), 0U);
  }
}

TEST(BackendContractTest, ReportsIndependentDomainsAndReleasesReservations) {
  const test::ModelFixture model;
  auto plan = std::make_shared<AllocationPlan>();
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 70U, 400U, 40U},
                         std::make_unique<TestFactory>(plan));
  v1::Capabilities caps;
  ASSERT_TRUE(service.GetCapabilities(nullptr, nullptr, &caps).ok());
  EXPECT_EQ(caps.worker().backend(), v1::BACKEND_CUDA);
  EXPECT_EQ(caps.worker().primary_memory_domain(), v1::MEMORY_DOMAIN_DEVICE);
  ASSERT_EQ(caps.worker().memory_budgets_size(), 3);
  auto load = model.load();
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(nullptr, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted()) << loaded.detail();
  auto lease = service.acquire("plan-1", 1U, "one", 4U, 0U, 0U);
  EXPECT_EQ(plan->allocations, 1);
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
  EXPECT_EQ(report.loaded_weight_bytes(), 120U);
  EXPECT_EQ(report.reserved_cache_bytes(), 200U);
  EXPECT_EQ(report.reserved_workspace_bytes(), 150U);  // Includes pinned once.
  ASSERT_EQ(report.domain_usage_size(), 3);
  EXPECT_EQ(report.domain_usage(0).reserved_workspace_bytes(), 50U);
  EXPECT_EQ(report.domain_usage(1).reserved_cache_bytes(), 200U);
  EXPECT_EQ(report.domain_usage(2).reserved_workspace_bytes(), 40U);
  service.release(lease.request);
  v1::MemoryReport released;
  ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &released).ok());
  EXPECT_EQ(released.active_requests(), 0U);
  EXPECT_EQ(released.reserved_cache_bytes(), 0U);
  EXPECT_EQ(released.reserved_workspace_bytes(), 0U);
  EXPECT_EQ(released.loaded_weight_bytes(), 120U);
}

TEST(BackendContractTest, RejectsInvalidWeightAccountingWithoutPublishingStage) {
  const test::ModelFixture model;
  auto plan = std::make_shared<AllocationPlan>();
  plan->weights.device_bytes = 401U;
  ControlService service({"cpu-a", "127.0.0.1:50051", model.root, 70U, 400U, 40U},
                         std::make_unique<TestFactory>(plan));
  auto load = model.load();
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(nullptr, &load, &loaded).ok());
  EXPECT_FALSE(loaded.accepted());
  EXPECT_EQ(loaded.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
  EXPECT_EQ(report.loaded_weight_bytes(), 0U);
}

TEST(BackendContractTest, RejectsOverflowAndUnaccountedPinnedMemory) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(static_cast<void>(runtime::add_memory({maximum, 0U, 0U}, {1U, 0U, 0U})),
               runtime::Error);
  EXPECT_THROW(static_cast<void>(runtime::add_memory({0U, maximum, 0U}, {0U, 1U, 0U})),
               runtime::Error);
  try {
    runtime::require_memory({1U, 0U, 2U}, {10U, 10U, 10U});
    FAIL() << "unaccounted pinned memory must be rejected";
  } catch (const runtime::Error& error) {
    EXPECT_EQ(error.code(), runtime::ErrorCode::kInternal);
  }
  try {
    runtime::require_memory({4U, 0U, 0U}, {2U, 0U, 0U});
    FAIL() << "budget overrun must be rejected";
  } catch (const runtime::Error& error) {
    EXPECT_EQ(error.code(), runtime::ErrorCode::kResourceExhausted);
  }
  const test::ModelFixture model;
  EXPECT_THROW((ControlService({"cpu-a", "localhost", model.root, maximum, 1U, 0U},
                               std::make_unique<TestFactory>(std::make_shared<AllocationPlan>()))),
               std::invalid_argument);
}

}  // namespace
}  // namespace hllm::worker
