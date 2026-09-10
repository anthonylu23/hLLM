#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "hllm/cpu/stage.hpp"
#include "hllm/cuda/backend.hpp"
#include "hllm/cuda/stage.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/half.hpp"
#include "hllm/runtime/safetensors.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"
#include "profiling_contract.hpp"

namespace hllm::cuda {
namespace {
constexpr runtime::MemoryAmounts kBudget{64U * 1024U * 1024U, 128U * 1024U * 1024U, 0U};

void near(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0U; i < actual.size(); ++i) {
    ASSERT_NEAR(actual[i], expected[i], tolerance) << "element " << i;
  }
}
std::vector<float> oracle_rows(const nlohmann::json& tensor, std::size_t position,
                               std::size_t count) {
  const auto width = tensor.at("shape").at(2).get<std::size_t>();
  const auto values = tensor.at("values").get<std::vector<float>>();
  return {values.begin() + static_cast<std::ptrdiff_t>(position * width),
          values.begin() + static_cast<std::ptrdiff_t>((position + count) * width)};
}

TEST(CudaStageTest, ProfilingPreservesNonzeroBoundaryAndDecodeOutputs) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
    auto a = fixture.load(0U); auto b = fixture.load(1U);
    a.mutable_plan()->set_execution_dtype(dtype);
    b.mutable_plan()->set_execution_dtype(dtype);
    auto first = factory->load(a, fixture.root, kBudget);
    auto last = factory->load(b, fixture.root, kBudget);
    test::paired_timing_contract(*first, *last);
  }
}

TEST(CudaStageTest, QwenPrefillDecodeLayersCachesAndLogitsMatchIndependentOracle) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
    auto request = fixture.load(0U, false);
    request.mutable_plan()->set_execution_dtype(dtype);
    auto stage = factory->load(request, fixture.root, kBudget);
    auto& reference = dynamic_cast<ReferenceStage&>(*stage);
    auto state = stage->allocate_sequence(16U);
    const std::array<std::uint64_t, 5U> ids{1U, 4U, 2U, 8U, 3U};
    const float tolerance = dtype == v1::DATA_TYPE_F32 ? 3e-5F : 5e-3F;
    std::atomic_bool cancelled{false};
    for (const std::size_t position : {0U, 3U, 4U}) {
      const std::size_t count = position == 0U ? 3U : 1U;
      runtime::TokenInput input{{ids.begin() + static_cast<std::ptrdiff_t>(position),
                                 ids.begin() + static_cast<std::ptrdiff_t>(position + count)}};
      ExecutionTrace trace;
      const auto result =
          reference.execute_traced(std::move(input), position, *state, cancelled, trace);
      ASSERT_TRUE(std::holds_alternative<runtime::SampledToken>(result));
      ASSERT_EQ(trace.layers.size(), 2U);
      for (std::size_t layer = 0U; layer < 2U; ++layer) {
        near(trace.layers[layer],
             oracle_rows(fixture.oracle.at("layer_outputs").at(layer), position, count), tolerance);
        for (const auto key : {true, false}) {
          const auto source = fixture.oracle.at(key ? "keys" : "values")
                                  .at(layer)
                                  .at("values")
                                  .get<std::vector<float>>();
          const auto& actual = key ? trace.keys[layer] : trace.values[layer];
          const auto heads = fixture.manifest.config().num_kv_heads();
          const auto width = fixture.manifest.config().head_dim();
          std::vector<float> expected;
          for (std::size_t head = 0U; head < heads; ++head) {
            for (std::size_t row = 0U; row < position + count; ++row) {
              for (std::size_t dim = 0U; dim < width; ++dim) {
                expected.push_back(source[(head * 5U + row) * width + dim]);
              }
            }
          }
          near(actual, expected, tolerance);
        }
      }
      near(trace.last_logits, oracle_rows(fixture.oracle.at("logits"), position + count - 1U, 1U),
           tolerance);
    }
    EXPECT_EQ(stage->weight_memory().host_bytes, 0U);
    EXPECT_GT(stage->weight_memory().device_bytes, 0U);
    const auto memory = stage->sequence_memory(16U);
    EXPECT_EQ(memory.cache.host_bytes, 0U);
    EXPECT_EQ(memory.workspace.pinned_host_bytes, 0U);
    EXPECT_GT(memory.cache.device_bytes, 0U);
    EXPECT_GT(memory.workspace.host_bytes, 0U);
  }
}

TEST(CudaStageTest, LlamaAndQwenStorageFormatsAndTiedHeadsMatchCpuTokens) {
  auto factory = make_backend_factory();
  for (const bool qwen : {false, true}) {
    for (const bool tied : {false, true}) {
      for (const auto* storage : {"F32", "F16", "BF16"}) {
        const test::ModelFixture fixture(qwen, tied, storage);
        auto request = fixture.load(0U, false);
        auto cpu = cpu::load_stage(request, fixture.root, kBudget.host_bytes);
        for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
          request.mutable_plan()->set_execution_dtype(dtype);
          auto gpu = factory->load(request, fixture.root, kBudget);
          auto a = cpu->allocate_sequence(16U);
          auto b = gpu->allocate_sequence(16U);
          std::atomic_bool cancelled{false};
          runtime::TokenInput input{{1U, 4U, 2U}};
          for (std::size_t step = 0U; step < 8U; ++step) {
            const std::size_t position = step == 0U ? 0U : step + 2U;
            const auto expected = cpu->execute(input, position, *a, cancelled);
            const auto actual = gpu->execute(input, position, *b, cancelled);
            const auto token = std::get<runtime::SampledToken>(expected).id;
            ASSERT_EQ(std::get<runtime::SampledToken>(actual).id, token)
                << qwen << ' ' << tied << ' ' << storage << ' ' << dtype << ' ' << step;
            input.ids = {token};
          }
        }
      }
    }
  }
}

TEST(CudaStageTest, SplitStagesMatchCpuBoundaryAndTokens) {
  auto factory = make_backend_factory();
  for (const bool qwen : {false, true}) {
    const test::ModelFixture fixture(qwen);
    for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
      auto first_request = fixture.load(0U);
      auto last_request = fixture.load(1U);
      auto cpu_first = cpu::load_stage(first_request, fixture.root, kBudget.host_bytes);
      auto cpu_last = cpu::load_stage(last_request, fixture.root, kBudget.host_bytes);
      first_request.mutable_plan()->set_execution_dtype(dtype);
      last_request.mutable_plan()->set_execution_dtype(dtype);
      auto first = factory->load(first_request, fixture.root, kBudget);
      auto last = factory->load(last_request, fixture.root, kBudget);
      auto a = cpu_first->allocate_sequence(16U);
      auto b = cpu_last->allocate_sequence(16U);
      auto c = first->allocate_sequence(16U);
      auto d = last->allocate_sequence(16U);
      std::atomic_bool cancelled{false};
      for (const std::size_t position : {0U, 3U, 4U}) {
        runtime::TokenInput input{position == 0U ? std::vector<std::uint64_t>{1U, 4U, 2U}
                                                 : std::vector<std::uint64_t>{8U}};
        auto cpu_output = cpu_first->execute(input, position, *a, cancelled);
        auto gpu_output = first->execute(input, position, *c, cancelled);
        auto expected = std::get<runtime::BoundaryActivation>(std::move(cpu_output));
        auto actual = std::get<runtime::BoundaryActivation>(std::move(gpu_output));
        ASSERT_EQ(actual.payload.size(), expected.payload.size());
        for (std::size_t i = 0U; i < actual.payload.size(); i += 2U) {
          const auto value = [&](const auto& data) {
            return runtime::float16_to_float(
                static_cast<std::uint16_t>(std::to_integer<unsigned>(data[i]) |
                                           (std::to_integer<unsigned>(data[i + 1U]) << 8U)));
          };
          ASSERT_NEAR(value(actual.payload), value(expected.payload), 5e-3F);
        }
        const auto cpu_token = cpu_last->execute(std::move(expected), position, *b, cancelled);
        const auto gpu_token = last->execute(std::move(actual), position, *d, cancelled);
        EXPECT_EQ(std::get<runtime::SampledToken>(gpu_token).id,
                  std::get<runtime::SampledToken>(cpu_token).id);
      }
    }
  }
}

TEST(CudaStageTest, RejectsInvalidMetadataBudgetsContextAndCancelledState) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  auto request = fixture.load(0U, false);
  EXPECT_THROW(
      static_cast<void>(factory->load(request, fixture.root, {1U, kBudget.device_bytes, 0U})),
      runtime::Error);
  EXPECT_THROW(
      static_cast<void>(factory->load(request, fixture.root, {kBudget.host_bytes, 1U, 0U})),
      runtime::Error);
  request.mutable_manifest()->mutable_architecture()->set_architecture_revision("2");
  EXPECT_THROW(static_cast<void>(factory->load(request, fixture.root, kBudget)),
               runtime::Error);
  request = fixture.load(0U, false);
  request.set_stage_index(10U);
  EXPECT_THROW(static_cast<void>(factory->load(request, fixture.root, kBudget)),
               runtime::Error);
  request = fixture.load(0U, false);
  auto stage = factory->load(request, fixture.root, kBudget);
  EXPECT_THROW(static_cast<void>(stage->allocate_sequence(0U)), runtime::Error);
  EXPECT_THROW(static_cast<void>(stage->allocate_sequence(stage->maximum_tokens() + 1U)),
               runtime::Error);
  auto state = stage->allocate_sequence(4U);
  std::atomic_bool cancelled{false};
  EXPECT_THROW(static_cast<void>(stage->execute(runtime::TokenInput{{1U}}, 1U, *state, cancelled)),
               runtime::Error);
  cancelled.store(true);
  EXPECT_THROW(static_cast<void>(stage->execute(runtime::TokenInput{{1U}}, 0U, *state, cancelled)),
               std::runtime_error);
  cancelled.store(false);
  EXPECT_THROW(
      static_cast<void>(stage->execute(runtime::TokenInput{{99999U}}, 0U, *state, cancelled)),
      runtime::Error);
  EXPECT_THROW(static_cast<void>(stage->execute(runtime::TokenInput{{1U}}, 0U, *state, cancelled)),
               runtime::Error);
  // A fresh reservation remains usable after failed/cancelled execution.
  auto fresh = stage->allocate_sequence(4U);
  EXPECT_NO_THROW(
      static_cast<void>(stage->execute(runtime::TokenInput{{1U}}, 0U, *fresh, cancelled)));
}

TEST(CudaStageTest, LlamaLayerOutputsMatchCpuReference) {
  const test::ModelFixture fixture(false, false);
  auto first = cpu::load_stage(fixture.load(0U), fixture.root, kBudget.host_bytes);
  auto last = cpu::load_stage(fixture.load(1U), fixture.root, kBudget.host_bytes);
  auto factory = make_backend_factory();
  for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
    auto request = fixture.load(0U, false);
    request.mutable_plan()->set_execution_dtype(dtype);
    auto gpu = factory->load(request, fixture.root, kBudget);
    auto a = first->allocate_sequence(16U);
    auto b = last->allocate_sequence(16U);
    auto c = gpu->allocate_sequence(16U);
    std::atomic_bool cancelled{false};
    for (const std::size_t position : {0U, 3U, 4U}) {
      runtime::TokenInput input{position == 0U ? std::vector<std::uint64_t>{1U, 4U, 2U}
                                               : std::vector<std::uint64_t>{8U}};
      auto hidden = first->forward(first->embed(input.ids), position, *a, cancelled);
      ExecutionTrace trace;
      static_cast<void>(dynamic_cast<ReferenceStage&>(*gpu).execute_traced(input, position, *c,
                                                                           cancelled, trace));
      const float tolerance = dtype == v1::DATA_TYPE_F32 ? 3e-5F : 5e-3F;
      near(trace.layers.at(0U), hidden.values, tolerance);
      hidden = last->forward(std::move(hidden), position, *b, cancelled);
      near(trace.layers.at(1U), hidden.values, tolerance);
    }
  }
}

TEST(CudaStageTest, DeviceWorkspaceAdmissionRejectsBeforeReservation) {
  const test::ModelFixture fixture;
  worker::ControlService service(
      {"cpu-a", "localhost", fixture.root, kBudget.host_bytes, 1024U * 1024U, 0U},
      make_backend_factory());
  auto request = fixture.load(0U, false);
  v1::LoadStageResponse loaded;
  ASSERT_TRUE(service.LoadStage(nullptr, &request, &loaded).ok());
  ASSERT_TRUE(loaded.accepted()) << loaded.detail();
  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("one");
  reserve.set_maximum_total_tokens(1U);
  v1::ReserveResponse reserved;
  ASSERT_TRUE(service.ReserveRequest(nullptr, &reserve, &reserved).ok());
  EXPECT_FALSE(reserved.accepted());
  EXPECT_EQ(reserved.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
  v1::MemoryReport report;
  ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
  EXPECT_EQ(report.active_requests(), 0U);
  EXPECT_EQ(report.reserved_cache_bytes(), 0U);
}

TEST(CudaStageTest, PinnedAdmissionCountsHostAndPinnedAndReleasesReservation) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory(0, true);
  auto budget = kBudget;
  budget.pinned_host_bytes = 1024U * 1024U;
  auto stage = factory->load(fixture.load(0U), fixture.root, budget);
  const auto memory = stage->sequence_memory(32U);
  EXPECT_EQ(memory.workspace.pinned_host_bytes, 32U * stage->hidden_size() * 2U);
  auto single = factory->load(fixture.load(0U, false), fixture.root, budget);
  EXPECT_EQ(single->sequence_memory(32U).workspace.pinned_host_bytes, 0U);
  const auto needed = runtime::add_memory(stage->weight_memory(),
                                        runtime::add_memory(memory.cache, memory.workspace));
  for (const int short_domain : {0, 1, 2}) {
    const bool short_budget = short_domain != 0;
    worker::ControlService service(
        {"cpu-a", "localhost", fixture.root,
         needed.host_bytes - (short_domain == 2 ? 1U : 0U), budget.device_bytes,
         needed.pinned_host_bytes - (short_domain == 1 ? 1U : 0U)}, make_backend_factory(0, true));
    auto request = fixture.load(0U);
    v1::LoadStageResponse loaded;
    ASSERT_TRUE(service.LoadStage(nullptr, &request, &loaded).ok());
    ASSERT_TRUE(loaded.accepted());
    v1::ReserveRequestMessage reserve;
    reserve.set_plan_id("plan-1");
    reserve.set_deployment_version(1U);
    reserve.set_request_id("pinned");
    reserve.set_maximum_total_tokens(32U);
    v1::ReserveResponse reserved;
    ASSERT_TRUE(service.ReserveRequest(nullptr, &reserve, &reserved).ok());
    EXPECT_EQ(reserved.accepted(), !short_budget);
    if (short_budget) {
      EXPECT_EQ(reserved.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
    }
    v1::MemoryReport report;
    ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
    EXPECT_EQ(report.active_requests(), short_budget ? 0U : 1U);
    if (!short_budget) {
      std::size_t host = 0U, device = 0U, pinned = 0U;
      for (const auto& domain : report.domain_usage()) {
        if (domain.domain() == v1::MEMORY_DOMAIN_HOST) host = domain.reserved_workspace_bytes();
        if (domain.domain() == v1::MEMORY_DOMAIN_DEVICE) device = domain.reserved_workspace_bytes();
        if (domain.domain() == v1::MEMORY_DOMAIN_HOST_PINNED) pinned = domain.reserved_workspace_bytes();
      }
      EXPECT_EQ(pinned, needed.pinned_host_bytes);
      EXPECT_EQ(host, memory.workspace.host_bytes);
      EXPECT_EQ(report.reserved_workspace_bytes(), host + device);
      v1::CancelRequestMessage cancel;
      cancel.set_plan_id("plan-1");
      cancel.set_deployment_version(1U);
      cancel.set_request_id("pinned");
      v1::Empty empty;
      ASSERT_TRUE(service.CancelRequest(nullptr, &cancel, &empty).ok());
      ASSERT_TRUE(service.GetMemoryReport(nullptr, nullptr, &report).ok());
      EXPECT_EQ(report.active_requests(), 0U);
      EXPECT_EQ(report.reserved_workspace_bytes(), 0U);
    }
    // The common admission rule must enforce the host subset independently.
    auto host_short = needed;
    --host_short.host_bytes;
    EXPECT_THROW(runtime::require_memory(needed, host_short), runtime::Error);
    EXPECT_NO_THROW(runtime::require_memory(needed, needed));
  }
}

TEST(CudaStageTest, RejectsNonfiniteWeightsAndHalfOverflowWithoutPublishingStage) {
  const test::ModelFixture fixture;
  const runtime::SafetensorsFile file(fixture.root / "model.safetensors");
  const auto& tensor = file.tensor("model.embed_tokens.weight");
  auto factory = make_backend_factory();
  for (const auto bits : {0x7fc00000U, 0x7f800000U, 0x47800000U}) {
    {
      std::fstream output(file.path(), std::ios::binary | std::ios::in | std::ios::out);
      output.seekp(static_cast<std::streamoff>(file.header_size() + tensor.data_offset));
      for (std::size_t byte = 0U; byte < 4U; ++byte) {
        output.put(static_cast<char>((bits >> (8U * byte)) & 0xffU));
      }
    }
    auto request = fixture.load(0U, false);
    request.mutable_plan()->set_execution_dtype(v1::DATA_TYPE_F16);
    EXPECT_THROW(static_cast<void>(factory->load(request, fixture.root, kBudget)),
                 runtime::Error);
  }
}

}  // namespace
}  // namespace hllm::cuda
