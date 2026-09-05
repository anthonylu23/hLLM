#include "hllm/cpu/stage.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>

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
  EXPECT_GT(first->weight_bytes(), 0U);
  EXPECT_GT(first->sequence_memory(16U).cache_bytes, 0U);
}

TEST(CpuStageTest, RejectsInvalidArchitecturesWeightsAndBudgets) {
  const test::ModelFixture fixture;
  auto request = fixture.load();
  request.mutable_manifest()->mutable_architecture()->set_architecture_revision("2");
  EXPECT_THROW(static_cast<void>(load_stage(request, fixture.root, 1'000'000U)),
               std::invalid_argument);
  request = fixture.load();
  request.mutable_manifest()->mutable_architecture()->add_feature_flags("unknown");
  EXPECT_THROW(static_cast<void>(load_stage(request, fixture.root, 1'000'000U)),
               std::invalid_argument);
  request = fixture.load();
  request.mutable_manifest()->mutable_tensors()->DeleteSubrange(0, 1);
  EXPECT_THROW(static_cast<void>(load_stage(request, fixture.root, 1'000'000U)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(load_stage(fixture.load(), fixture.root, 1U)), std::length_error);
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
