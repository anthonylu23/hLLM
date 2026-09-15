#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

#include "hllm/cpu/stage.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/mlx/backend.hpp"
#include "hllm/mlx/stage.hpp"
#include "hllm/runtime/half.hpp"
#include "hllm/runtime/safetensors.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"
#include "profiling_contract.hpp"

namespace hllm::mlx {
namespace {
constexpr runtime::MemoryAmounts kBudget{0U, 0U, 0U, 128U * 1024U * 1024U};

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

TEST(MlxStageTest, MixedPrecisionAccountsForResidentWeightsCachesAndCastWorkspace) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  EXPECT_TRUE(factory->capabilities().supports_mixed_precision);
  auto request = fixture.load(0U, false);
  auto baseline = factory->load(request, fixture.root, kBudget);
  const auto weights = baseline->weight_memory().unified_bytes;
  const auto memory = baseline->sequence_memory(16U);
  baseline.reset();
  request.mutable_plan()->set_weight_dtype(v1::DATA_TYPE_F16);
  EXPECT_THROW(static_cast<void>(factory->load(request, fixture.root, kBudget)), runtime::Error);
  request.mutable_plan()->mutable_schema_version()->set_minor(2U);
  auto mixed = factory->load(request, fixture.root, kBudget);
  EXPECT_EQ(mixed->weight_memory().unified_bytes * 2U, weights);
  EXPECT_EQ(mixed->sequence_memory(16U).cache.unified_bytes, memory.cache.unified_bytes);
  EXPECT_GT(mixed->sequence_memory(16U).workspace.unified_bytes, memory.workspace.unified_bytes);
  mixed.reset();
  request.mutable_plan()->set_execution_dtype(v1::DATA_TYPE_F16);
  EXPECT_THROW(static_cast<void>(factory->load(request, fixture.root, kBudget)), runtime::Error);
}

TEST(MlxStageTest, ProfilingPreservesNonzeroBoundaryAndDecodeOutputs) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  for (const auto mode : {0, 1, 2}) {
    const auto dtype = mode == 1 ? v1::DATA_TYPE_F16 : v1::DATA_TYPE_F32;
    auto a = fixture.load(0U); auto b = fixture.load(1U);
    a.mutable_plan()->set_execution_dtype(dtype);
    b.mutable_plan()->set_execution_dtype(dtype);
    if (mode == 2) {
      for (auto* request : {&a, &b}) {
        request->mutable_plan()->set_weight_dtype(v1::DATA_TYPE_F16);
        request->mutable_plan()->mutable_schema_version()->set_minor(2U);
      }
    }
    auto first = factory->load(a, fixture.root, kBudget);
    auto last = factory->load(b, fixture.root, kBudget);
    test::paired_timing_contract(*first, *last);
  }
}

TEST(MlxStageTest, QwenPrefillDecodeLayersCachesAndLogitsMatchIndependentOracle) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  for (const auto mode : {0, 1, 2}) {
    const auto dtype = mode == 1 ? v1::DATA_TYPE_F16 : v1::DATA_TYPE_F32;
    auto request = fixture.load(0U, false);
    request.mutable_plan()->set_execution_dtype(dtype);
    if (mode == 2) {
      request.mutable_plan()->set_weight_dtype(v1::DATA_TYPE_F16);
      request.mutable_plan()->mutable_schema_version()->set_minor(2U);
    }
    auto stage = factory->load(request, fixture.root, kBudget);
    auto& reference = dynamic_cast<ReferenceStage&>(*stage);
    auto state = stage->allocate_sequence(16U);
    const std::array<std::uint64_t, 5U> ids{1U, 4U, 2U, 8U, 3U};
    const float tolerance = mode == 0 ? 3e-5F : 5e-3F;
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
    EXPECT_GT(stage->weight_memory().unified_bytes, 0U);
    const auto memory = stage->sequence_memory(16U);
    EXPECT_EQ(memory.cache.host_bytes, 0U);
    EXPECT_EQ(memory.workspace.pinned_host_bytes, 0U);
    EXPECT_GT(memory.cache.unified_bytes, 0U);
    EXPECT_GT(memory.workspace.unified_bytes, 0U);
  }
}

TEST(MlxStageTest, LlamaAndQwenStorageFormatsAndTiedHeadsMatchCpuTokens) {
  auto factory = make_backend_factory();
  for (const bool qwen : {false, true}) {
    for (const bool tied : {false, true}) {
      for (const auto* storage : {"F32", "F16", "BF16"}) {
        const test::ModelFixture fixture(qwen, tied, storage);
        auto request = fixture.load(0U, false);
        auto cpu = cpu::load_stage(request, fixture.root, kBudget.unified_bytes);
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

TEST(MlxStageTest, SplitStagesMatchCpuBoundaryAndTokens) {
  auto factory = make_backend_factory();
  for (const bool qwen : {false, true}) {
    const test::ModelFixture fixture(qwen);
    for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
      auto first_request = fixture.load(0U);
      auto last_request = fixture.load(1U);
      auto cpu_first = cpu::load_stage(first_request, fixture.root, kBudget.unified_bytes);
      auto cpu_last = cpu::load_stage(last_request, fixture.root, kBudget.unified_bytes);
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

TEST(MlxStageTest, RejectsInvalidMetadataBudgetsContextAndCancelledState) {
  const auto expect_error = [](auto&& action, runtime::ErrorCode expected) {
    try {
      action();
      FAIL() << "expected a categorized runtime error";
    } catch (const runtime::Error& error) {
      EXPECT_EQ(error.code(), expected);
    }
  };
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  auto request = fixture.load(0U, false);
  expect_error([&] { static_cast<void>(factory->load(request, fixture.root, {0U, 0U, 0U, 1U})); },
               runtime::ErrorCode::kResourceExhausted);
  request.mutable_manifest()->mutable_architecture()->set_architecture_revision("2");
  expect_error([&] { static_cast<void>(factory->load(request, fixture.root, kBudget)); },
               runtime::ErrorCode::kIncompatibleWorker);
  request = fixture.load(0U, false);
  request.set_stage_index(10U);
  expect_error([&] { static_cast<void>(factory->load(request, fixture.root, kBudget)); },
               runtime::ErrorCode::kIncompatibleWorker);
  request = fixture.load(0U, false);
  auto stage = factory->load(request, fixture.root, kBudget);
  expect_error([&] { static_cast<void>(stage->allocate_sequence(0U)); }, runtime::ErrorCode::kInvalidRequest);
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

TEST(MlxStageTest, LlamaLayerOutputsMatchCpuReference) {
  const test::ModelFixture fixture(false, false);
  auto first = cpu::load_stage(fixture.load(0U), fixture.root, kBudget.unified_bytes);
  auto last = cpu::load_stage(fixture.load(1U), fixture.root, kBudget.unified_bytes);
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

TEST(MlxStageTest, UnifiedWorkspaceAdmissionRejectsBeforeReservation) {
  const test::ModelFixture fixture;
  worker::ControlService service({"cpu-a", "localhost", fixture.root, 0U, 0U, 0U, 1024U * 1024U},
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

TEST(MlxStageTest, RejectsNonfiniteWeightsAndHalfOverflowWithoutPublishingStage) {
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

TEST(MlxStageTest, RejectsForeignStateAndMalformedBoundaries) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory();
  auto first = factory->load(fixture.load(0U), fixture.root, kBudget);
  auto last = factory->load(fixture.load(1U), fixture.root, kBudget);
  auto foreign = first->allocate_sequence(4U);
  std::atomic_bool cancelled{false};
  runtime::BoundaryActivation valid{1U, last->hidden_size(),
                                    std::vector<std::byte>(last->hidden_size() * 2U)};
  EXPECT_THROW(static_cast<void>(last->execute(valid, 0U, *foreign, cancelled)),
               runtime::Error);
  for (int mutation = 0; mutation < 3; ++mutation) {
    auto state = last->allocate_sequence(4U);
    auto input = valid;
    if (mutation == 0) ++input.width;
    if (mutation == 1) input.payload.pop_back();
    if (mutation == 2) {
      input.payload[0] = std::byte{0};
      input.payload[1] = std::byte{0x7c};
    }
    EXPECT_THROW(static_cast<void>(last->execute(std::move(input), 0U, *state, cancelled)),
                 runtime::Error);
    EXPECT_THROW(static_cast<void>(last->execute(valid, 0U, *state, cancelled)),
                 runtime::Error);
  }
  auto state = last->allocate_sequence(4U);
  EXPECT_NO_THROW(static_cast<void>(last->execute(valid, 0U, *state, cancelled)));
}

TEST(MlxStageTest, GreedySamplingChoosesFirstMaximumOnTies) {
  const test::ModelFixture fixture(true, false);
  const runtime::SafetensorsFile file(fixture.root / "model.safetensors");
  const auto& tensor = file.tensor("lm_head.weight");
  {
    std::fstream output(file.path(), std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(static_cast<std::streamoff>(file.header_size() + tensor.data_offset));
    for (std::size_t i = 0U; i < fixture.manifest.config().vocabulary_size() *
                                     fixture.manifest.config().hidden_size() * sizeof(float);
         ++i) {
      output.put(0);
    }
  }
  auto factory = make_backend_factory();
  for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
    auto request = fixture.load(0U, false);
    request.mutable_plan()->set_execution_dtype(dtype);
    auto stage = factory->load(request, fixture.root, kBudget);
    auto state = stage->allocate_sequence(4U);
    std::atomic_bool cancelled{false};
    auto result = stage->execute(runtime::TokenInput{{1U, 4U, 2U}}, 0U, *state, cancelled);
    EXPECT_EQ(std::get<runtime::SampledToken>(result).id, 0U);
  }
}

TEST(MlxStageTest, AllocatorPlateausAfterRepeatedLoadGenerateUnload) {
  const test::ModelFixture fixture;
  auto factory = make_backend_factory(kBudget.unified_bytes);
  for (const auto dtype : {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16}) {
    std::vector<std::size_t> active, cached;
    for (int cycle = 0; cycle < 12; ++cycle) {
      {
        auto request = fixture.load(0U, false);
        request.mutable_plan()->set_execution_dtype(dtype);
        auto stage = factory->load(request, fixture.root, kBudget);
        auto state = stage->allocate_sequence(16U);
        std::atomic_bool cancelled{false};
        runtime::TokenInput input{{1U, 4U, 2U}};
        for (std::size_t step = 0U; step < 8U; ++step) {
          auto output = stage->execute(input, step == 0U ? 0U : step + 2U, *state, cancelled);
          input.ids = {std::get<runtime::SampledToken>(output).id};
        }
      }
      auto metrics = factory->allocator_metrics();
      ASSERT_TRUE(metrics.has_value());
      EXPECT_GE(metrics->peak_bytes, metrics->active_bytes);
      if (cycle >= 3) {
        active.push_back(metrics->active_bytes);
        cached.push_back(metrics->cached_bytes);
      }
    }
    const auto range = [](const auto& samples) {
      return *std::max_element(samples.begin(), samples.end()) -
             *std::min_element(samples.begin(), samples.end());
    };
    EXPECT_EQ(range(active), 0U);
    EXPECT_LE(range(cached), 1024U * 1024U);
    std::cout << "MLX allocator dtype=" << dtype << " active=" << active.back()
              << " cached=" << cached.back() << " cached_range=" << range(cached) << '\n';
  }
}

}  // namespace
}  // namespace hllm::mlx
