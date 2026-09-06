#include "hllm/cpu/stage.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>

#include "hllm/runtime/error.hpp"
#include "hllm/runtime/half.hpp"
#include "model_fixture.hpp"

namespace hllm::cpu {
namespace {

TEST(CpuStageTest, LoadedPayloadsMatchIndependentTransformersAtBothLayerBoundaries) {
  const test::ModelFixture fixture;
  auto first = load_stage(fixture.load(0U), fixture.root, 1'000'000U);
  auto last = load_stage(fixture.load(1U), fixture.root, 1'000'000U);
  auto full = load_stage(fixture.load(0U, false), fixture.root, 1'000'000U);
  auto a = first->allocate_sequence(16U);
  auto b = last->allocate_sequence(16U);
  auto c = full->allocate_sequence(16U);
  std::atomic_bool cancelled{false};
  const std::array<std::uint64_t, 5> tokens{1U, 4U, 2U, 8U, 3U};
  for (std::size_t position : {0U, 3U, 4U}) {
    const auto count = position == 0U ? 3U : 1U;
    auto input = first->embed(std::span(tokens).subspan(position, count));
    auto hidden = first->forward(input, position, *a, cancelled);
    auto expected = fixture.oracle.at("layer_outputs").at(0).at("values").get<std::vector<float>>();
    for (std::size_t i = 0U; i < hidden.values.size(); ++i) {
      EXPECT_NEAR(hidden.values[i], expected[position * hidden.width + i], 2e-5F);
    }
    hidden = last->forward(hidden, position, *b, cancelled);
    expected = fixture.oracle.at("layer_outputs").at(1).at("values").get<std::vector<float>>();
    for (std::size_t i = 0U; i < hidden.values.size(); ++i) {
      EXPECT_NEAR(hidden.values[i], expected[position * hidden.width + i], 2e-5F);
    }
    const auto unsplit = full->forward(input, position, *c, cancelled);
    EXPECT_EQ(hidden.values, unsplit.values);
    EXPECT_EQ(last->sample(hidden), full->sample(unsplit));
  }
  EXPECT_GT(first->weight_memory().host_bytes, 0U);
  EXPECT_GT(first->sequence_memory(16U).cache.host_bytes, 0U);
}

TEST(CpuStageTest, RejectsInvalidArchitecturesWeightsAndBudgets) {
  const test::ModelFixture fixture;
  auto request = fixture.load();
  request.mutable_manifest()->mutable_architecture()->set_architecture_revision("2");
  const auto code = [&](const v1::LoadStageRequest& attempt, std::size_t limit) {
    try {
      static_cast<void>(load_stage(attempt, fixture.root, limit));
    } catch (const runtime::Error& error) {
      return error.code();
    }
    return runtime::ErrorCode::kInternal;
  };
  EXPECT_EQ(code(request, 1'000'000U), runtime::ErrorCode::kIncompatibleWorker);
  request = fixture.load();
  request.mutable_manifest()->mutable_architecture()->add_feature_flags("unknown");
  EXPECT_EQ(code(request, 1'000'000U), runtime::ErrorCode::kIncompatibleWorker);
  request = fixture.load();
  request.mutable_manifest()->mutable_tensors()->DeleteSubrange(0, 1);
  EXPECT_EQ(code(request, 1'000'000U), runtime::ErrorCode::kIncompatibleWorker);
  EXPECT_EQ(code(fixture.load(), 1U), runtime::ErrorCode::kResourceExhausted);
}

TEST(CpuStageTest, DoesNotReadUnassignedWeights) {
  const test::ModelFixture fixture;
  auto request = fixture.load();
  for (auto& tensor : *request.mutable_manifest()->mutable_tensors()) {
    if (tensor.has_layer_index() && tensor.layer_index() == 1U) {
      tensor.set_file("not-present.safetensors");
    }
  }
  EXPECT_NO_THROW(static_cast<void>(load_stage(request, fixture.root, 1'000'000U)));
}

}  // namespace
}  // namespace hllm::cpu

namespace hllm::cpu {
namespace {
TEST(CpuStageTest, Float16BoundaryHasBoundedErrorAgainstUnsplitOracle) {
  const test::ModelFixture fixture;
  auto first = load_stage(fixture.load(0U), fixture.root, 1'000'000U);
  auto last = load_stage(fixture.load(1U), fixture.root, 1'000'000U);
  auto a = first->allocate_sequence(16U);
  auto b = last->allocate_sequence(16U);
  const std::array<std::uint64_t, 5> tokens{1U, 4U, 2U, 8U, 3U};
  std::atomic_bool cancelled{false};
  auto hidden = first->forward(first->embed(tokens), 0U, *a, cancelled);
  for (auto& value : hidden.values) {
    value = runtime::float16_to_float(runtime::float_to_float16(value));
  }
  hidden = last->forward(hidden, 0U, *b, cancelled);
  const auto expected =
      fixture.oracle.at("layer_outputs").at(1).at("values").get<std::vector<float>>();
  ASSERT_EQ(hidden.values.size(), expected.size());
  for (std::size_t i = 0U; i < expected.size(); ++i) {
    EXPECT_NEAR(hidden.values[i], expected[i], 1e-3F);
  }
}
}  // namespace
}  // namespace hllm::cpu

namespace hllm::cpu {
namespace {
TEST(CpuStageTest, ExecutesThroughOpaqueBoundaryContractForPrefillAndDecode) {
  const test::ModelFixture fixture;
  auto first = load_stage(fixture.load(0U), fixture.root, 1'000'000U);
  auto last = load_stage(fixture.load(1U), fixture.root, 1'000'000U);
  auto oracle = load_stage(fixture.load(0U, false), fixture.root, 1'000'000U);
  auto a = first->allocate_sequence(16U);
  auto b = last->allocate_sequence(16U);
  auto c = oracle->allocate_sequence(16U);
  std::atomic_bool cancelled{false};
  for (const std::size_t position : {0U, 3U, 4U}) {
    runtime::TokenInput input{position == 0U ? std::vector<std::uint64_t>{1U, 4U, 2U}
                                             : std::vector<std::uint64_t>{8U}};
    const auto expected =
        oracle->sample(oracle->forward(oracle->embed(input.ids), position, *c, cancelled));
    auto output = first->execute(input, position, *a, cancelled);
    ASSERT_TRUE(std::holds_alternative<runtime::BoundaryActivation>(output));
    auto boundary = std::get<runtime::BoundaryActivation>(std::move(output));
    EXPECT_EQ(boundary.tokens, input.ids.size());
    EXPECT_EQ(boundary.payload.size(), boundary.tokens * boundary.width * 2U);
    const auto token = last->execute(std::move(boundary), position, *b, cancelled);
    ASSERT_TRUE(std::holds_alternative<runtime::SampledToken>(token));
    EXPECT_EQ(std::get<runtime::SampledToken>(token).id, expected);
  }
  EXPECT_THROW(static_cast<void>(last->execute(runtime::TokenInput{{1U}}, 5U, *b, cancelled)),
               runtime::Error);
  EXPECT_THROW(
      static_cast<void>(first->execute(runtime::BoundaryActivation{1U, 8U, {}}, 5U, *a, cancelled)),
      runtime::Error);
}
}  // namespace
}  // namespace hllm::cpu
