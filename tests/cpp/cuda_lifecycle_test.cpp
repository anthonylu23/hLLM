#include <gtest/gtest.h>
#include <algorithm>
#include <future>
#include <iostream>

#include "cuda_diagnostics.hpp"
#include "hllm/cpu/stage.hpp"
#include "hllm/cuda/backend.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"

namespace hllm::cuda {
namespace {
constexpr runtime::MemoryAmounts kBudget{64U * 1024U * 1024U, 128U * 1024U * 1024U,
                                         8U * 1024U * 1024U};
class FailOnceStage final : public runtime::StageBackend {
 public:
  explicit FailOnceStage(std::unique_ptr<runtime::StageBackend> inner) : inner_(std::move(inner)) {}
  runtime::MemoryAmounts weight_memory() const override { return inner_->weight_memory(); }
  std::size_t hidden_size() const override { return inner_->hidden_size(); }
  std::size_t vocabulary_size() const override { return inner_->vocabulary_size(); }
  std::size_t maximum_tokens() const override { return inner_->maximum_tokens(); }
  runtime::SequenceMemory sequence_memory(std::size_t n) const override { return inner_->sequence_memory(n); }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t n) const override {
    auto allocated = inner_->allocate_sequence(n);
    if (fail_.exchange(false)) throw std::bad_alloc();
    return allocated;
  }
  runtime::StageOutput execute(runtime::StageInput input, std::size_t position,
                               runtime::SequenceState& state, const std::atomic_bool& cancelled) const override {
    return inner_->execute(std::move(input), position, state, cancelled);
  }
 private:
  std::unique_ptr<runtime::StageBackend> inner_;
  mutable std::atomic_bool fail_{true};
};
class FailOnceFactory final : public runtime::BackendFactory {
 public:
  runtime::BackendCapabilities capabilities() const override { return inner_->capabilities(); }
  std::unique_ptr<runtime::StageBackend> load(const v1::LoadStageRequest& request,
      const std::filesystem::path& root, const runtime::MemoryAmounts& budget) const override {
    return std::make_unique<FailOnceStage>(inner_->load(request, root, budget));
  }
 private:
  std::unique_ptr<runtime::BackendFactory> inner_ = make_backend_factory(0, true);
};

TEST(CudaLifecycleTest, AllocationFailureAfterRealSequenceAllocationRollsBackAndRecovers) {
  const hllm::test::ModelFixture fixture;
  worker::ControlService service({"cpu-a", "localhost", fixture.root, kBudget.host_bytes,
      kBudget.device_bytes, kBudget.pinned_host_bytes}, std::make_unique<FailOnceFactory>());
  auto load = fixture.load(0U);
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(nullptr, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted());
  const auto baseline = test::memory_snapshot();
  for (const bool retry : {false, true}) {
    v1::ReserveRequestMessage reserve;
    reserve.set_plan_id("plan-1");
    reserve.set_deployment_version(1U);
    reserve.set_request_id("fault");
    reserve.set_maximum_total_tokens(16U);
    v1::ReserveResponse response;
    ASSERT_TRUE(service.ReserveRequest(nullptr, &reserve, &response).ok());
    EXPECT_EQ(response.accepted(), retry);
    if (!retry) {
      EXPECT_EQ(response.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
      EXPECT_EQ(test::memory_snapshot().allocated, baseline.allocated);
      EXPECT_EQ(test::memory_snapshot().pinned, 0U);
    } else {
      auto lease = service.acquire("plan-1", 1U, "fault", 16U, 0U, 0U);
      EXPECT_NO_THROW(static_cast<void>(lease.deployment->backend->execute(
          runtime::TokenInput{{1U, 4U, 2U}}, 0U, *lease.request->sequence, lease.request->cancelled)));
      service.release(lease.request);
    }
    v1::MemoryReport report;
    ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
    EXPECT_EQ(report.active_requests(), 0U);
    EXPECT_EQ(report.reserved_cache_bytes(), 0U);
    EXPECT_EQ(report.reserved_workspace_bytes(), 0U);
    EXPECT_EQ(test::memory_snapshot().pinned, 0U);
  }
}

TEST(CudaLifecycleTest, RepeatedMixedExecutionAndUnloadHasBoundedMemoryAfterWarmup) {
  const hllm::test::ModelFixture fixture;
  for (const bool pinned : {false, true}) {
    auto factory = make_backend_factory(0, pinned);
    for (const bool cuda_first : {false, true}) {
      std::vector<test::MemorySnapshot> samples;
      test::reset_peak();
      std::size_t payload{}, pinned_active{};
      // Identical assignments should reuse cached buffers after two loads,
      // without warming every stream in LibTorch's 32-entry pool.
      for (std::size_t cycle = 0; cycle < 12U; ++cycle) {
        // gRPC dispatches assignments on host threads that can change over time.
        std::async(std::launch::async, [&] {
          auto gpu = factory->load(fixture.load(cuda_first ? 0U : 1U), fixture.root, kBudget);
          auto cpu = cpu::load_stage(fixture.load(cuda_first ? 1U : 0U), fixture.root, kBudget.host_bytes);
          auto a = gpu->allocate_sequence(16U);
          auto b = cpu->allocate_sequence(16U);
          payload = gpu->weight_memory().device_bytes + gpu->sequence_memory(16U).cache.device_bytes;
          pinned_active = test::memory_snapshot().pinned;
          EXPECT_EQ(pinned_active, gpu->sequence_memory(16U).workspace.pinned_host_bytes);
          std::atomic_bool cancelled{false};
          runtime::TokenInput input{{1U, 4U, 2U}};
          for (std::size_t step = 0; step < 8U; ++step) {
            const auto position = step == 0U ? 0U : step + 2U;
            auto* first = cuda_first ? gpu.get() : cpu.get();
            auto* final = cuda_first ? cpu.get() : gpu.get();
            auto& first_state = cuda_first ? *a : *b;
            auto& final_state = cuda_first ? *b : *a;
            auto boundary = std::get<runtime::BoundaryActivation>(first->execute(input, position, first_state, cancelled));
            const auto result = final->execute(std::move(boundary), position, final_state, cancelled);
            input.ids = {std::get<runtime::SampledToken>(result).id};
          }
        }).get();
        const auto snapshot = test::memory_snapshot();
        EXPECT_EQ(snapshot.pinned, 0U);
        if (cycle >= 2U) samples.push_back(snapshot);
      }
      const auto range = [&](auto field) {
        const auto [low, high] = std::minmax_element(samples.begin(), samples.end(),
            [&](const auto& a, const auto& b) { return a.*field < b.*field; });
        return (*high).*field - (*low).*field;
      };
      EXPECT_EQ(range(&test::MemorySnapshot::allocated), 0);
      EXPECT_LE(range(&test::MemorySnapshot::reserved), 2 * 1024 * 1024);
      EXPECT_LE(range(&test::MemorySnapshot::rss), 8U * 1024U * 1024U);
      const auto& last = samples.back();
      std::cout << "lifecycle pinned=" << pinned << " cuda_first=" << cuda_first
                << " payload=" << payload << " pinned_active=" << pinned_active
                << " allocated_after_unload=" << last.allocated << " reserved=" << last.reserved
                << " peak_allocated=" << last.peak_allocated << " rss=" << last.rss
                << " rss_range=" << range(&test::MemorySnapshot::rss)
                << " reserved_range=" << range(&test::MemorySnapshot::reserved) << '\n';
    }
  }
}

TEST(CudaLifecycleTest, ReportsAllocatorResidencySeparatelyFromModelReservations) {
  const hllm::test::ModelFixture fixture;
  worker::ControlService service({"cpu-a", "localhost", fixture.root, kBudget.host_bytes,
      kBudget.device_bytes, kBudget.pinned_host_bytes}, make_backend_factory(0, true));
  auto sample = [&] {
    v1::WorkerMetrics metrics;
    EXPECT_TRUE(service.GetMetrics(nullptr, nullptr, &metrics).ok());
    EXPECT_TRUE(metrics.has_allocator());
    EXPECT_EQ(metrics.allocator().domain(), v1::MEMORY_DOMAIN_DEVICE);
    const auto native = test::memory_snapshot();
    EXPECT_EQ(metrics.allocator().active_bytes(), native.allocated);
    EXPECT_EQ(metrics.allocator().cached_bytes() + metrics.allocator().active_bytes(),
              native.reserved);
    EXPECT_GE(metrics.allocator().peak_bytes(), metrics.allocator().active_bytes());
    return metrics;
  };
  const auto before = sample();
  auto load = fixture.load(0U);
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(nullptr, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted());
  const auto resident = sample();
  EXPECT_GT(resident.allocator().active_bytes(), before.allocator().active_bytes());
  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("memory");
  reserve.set_maximum_total_tokens(16U);
  v1::ReserveResponse reserved;
  ASSERT_TRUE(service.ReserveRequest(nullptr, &reserve, &reserved).ok());
  ASSERT_TRUE(reserved.accepted());
  EXPECT_GT(sample().allocator().active_bytes(), resident.allocator().active_bytes());
  v1::CancelRequestMessage cancel;
  cancel.set_plan_id("plan-1");
  cancel.set_deployment_version(1U);
  cancel.set_request_id("memory");
  v1::Empty empty;
  ASSERT_TRUE(service.CancelRequest(nullptr, &cancel, &empty).ok());
  EXPECT_EQ(sample().allocator().active_bytes(), resident.allocator().active_bytes());
  v1::UnloadStageRequest unload;
  unload.set_plan_id("plan-1");
  unload.set_deployment_version(1U);
  ASSERT_TRUE(service.UnloadStage(nullptr, &unload, &empty).ok());
  const auto after = sample();
  EXPECT_EQ(after.allocator().active_bytes(), before.allocator().active_bytes());
  EXPECT_GT(after.allocator().cached_bytes(), 0U);
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
  EXPECT_EQ(report.loaded_weight_bytes(), 0U);
  EXPECT_EQ(report.reserved_cache_bytes(), 0U);
  EXPECT_EQ(report.active_requests(), 0U);
}
}  // namespace
}  // namespace hllm::cuda
