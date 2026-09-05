#include "hllm/cpu/llama.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace hllm::cpu {
namespace {

[[nodiscard]] Matrix deterministic_matrix(const std::size_t rows,
                                          const std::size_t columns,
                                          const float phase) {
  std::vector<float> values(rows * columns);
  for (std::size_t index = 0U; index < values.size(); ++index) {
    values[index] = 0.12F * std::sin(static_cast<float>(index + 1U) * 0.37F + phase);
  }
  return Matrix(rows, columns, std::move(values));
}

[[nodiscard]] LayerWeights deterministic_layer(const LlamaConfig& config,
                                               const float phase) {
  const auto key_value_width = config.key_value_heads * config.head_dimension;
  std::vector<float> input_norm(config.hidden_size);
  std::vector<float> post_norm(config.hidden_size);
  for (std::size_t index = 0U; index < config.hidden_size; ++index) {
    input_norm[index] = 0.9F + static_cast<float>(index) * 0.03F;
    post_norm[index] = 1.1F - static_cast<float>(index) * 0.02F;
  }
  return LayerWeights{
      .input_norm = std::move(input_norm),
      .query = deterministic_matrix(config.hidden_size, config.hidden_size, phase),
      .key = deterministic_matrix(key_value_width, config.hidden_size, phase + 0.1F),
      .value = deterministic_matrix(key_value_width, config.hidden_size, phase + 0.2F),
      .attention_output =
          deterministic_matrix(config.hidden_size, config.hidden_size, phase + 0.3F),
      .post_attention_norm = std::move(post_norm),
      .gate = deterministic_matrix(config.intermediate_size, config.hidden_size,
                                   phase + 0.4F),
      .up = deterministic_matrix(config.intermediate_size, config.hidden_size,
                                 phase + 0.5F),
      .down = deterministic_matrix(config.hidden_size, config.intermediate_size,
                                   phase + 0.6F),
  };
}

constexpr LlamaConfig kConfig{
    .hidden_size = 4U,
    .intermediate_size = 6U,
    .attention_heads = 2U,
    .key_value_heads = 1U,
    .head_dimension = 2U,
    .maximum_sequence_length = 16U,
    .rms_norm_epsilon = 1e-5F,
    .rope_theta = 10'000.0F,
};

void expect_near(const Matrix& actual, const Matrix& expected, const float tolerance) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.columns(), expected.columns());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    EXPECT_NEAR(actual.values()[index], expected.values()[index], tolerance) << index;
  }
}

TEST(CpuLlamaTest, RmsNormMatchesHandCalculation) {
  const Matrix input(1U, 2U, {3.0F, 4.0F});
  const auto output = rms_norm(input, std::vector<float>{1.0F, 2.0F}, 1e-6F);
  const auto inverse_rms = 1.0F / std::sqrt(12.5F + 1e-6F);
  EXPECT_NEAR(output(0U, 0U), 3.0F * inverse_rms, 1e-6F);
  EXPECT_NEAR(output(0U, 1U), 8.0F * inverse_rms, 1e-6F);
}

TEST(CpuLlamaTest, RopeUsesLlamaHalfRotationLayout) {
  Matrix values(1U, 4U, {1.0F, 2.0F, 3.0F, 4.0F});
  apply_rope(values, 1U, 4U, 1U, 10'000.0F);
  EXPECT_NEAR(values(0U, 0U), std::cos(1.0F) - 3.0F * std::sin(1.0F), 1e-6F);
  EXPECT_NEAR(values(0U, 2U), 3.0F * std::cos(1.0F) + std::sin(1.0F), 1e-6F);
  EXPECT_NEAR(values(0U, 1U), 2.0F * std::cos(0.01F) - 4.0F * std::sin(0.01F),
              1e-6F);
}

TEST(CpuLlamaTest, PrefillThenDecodeMatchesOneShotCausalExecution) {
  const auto weights = deterministic_layer(kConfig, 0.2F);
  const Matrix prompt(2U, 4U,
                      {0.2F, -0.1F, 0.5F, 0.7F, -0.4F, 0.3F, 0.1F, 0.6F});
  const Matrix decode(1U, 4U, {0.8F, -0.2F, 0.4F, -0.5F});
  const Matrix combined(3U, 4U,
                        {0.2F, -0.1F, 0.5F, 0.7F, -0.4F, 0.3F, 0.1F, 0.6F, 0.8F,
                         -0.2F, 0.4F, -0.5F});

  LayerKvCache one_shot_cache(16U, 1U, 2U);
  const auto one_shot = transformer_layer(combined, weights, kConfig, 0U, one_shot_cache);

  LayerKvCache incremental_cache(16U, 1U, 2U);
  static_cast<void>(transformer_layer(prompt, weights, kConfig, 0U, incremental_cache));
  const auto incremental =
      transformer_layer(decode, weights, kConfig, 2U, incremental_cache);

  const Matrix expected(1U, 4U,
                        {one_shot(2U, 0U), one_shot(2U, 1U), one_shot(2U, 2U),
                         one_shot(2U, 3U)});
  expect_near(incremental, expected, 1e-6F);
  EXPECT_EQ(incremental_cache.length(), 3U);
}

TEST(CpuLlamaTest, SplitPipelineMatchesReferenceWithTheSameFp16Boundary) {
  const auto first = deterministic_layer(kConfig, 0.1F);
  const auto second = deterministic_layer(kConfig, 0.9F);
  const Matrix input(2U, 4U,
                     {0.1F, 0.2F, -0.3F, 0.4F, 0.6F, -0.2F, 0.5F, 0.3F});

  LayerKvCache reference_first_cache(16U, 1U, 2U);
  LayerKvCache reference_second_cache(16U, 1U, 2U);
  const auto reference_boundary = quantize_float16(
      transformer_layer(input, first, kConfig, 0U, reference_first_cache));
  const auto reference = transformer_layer(reference_boundary, second, kConfig, 0U,
                                           reference_second_cache);

  LayerKvCache stage_zero_cache(16U, 1U, 2U);
  const auto transmitted =
      quantize_float16(transformer_layer(input, first, kConfig, 0U, stage_zero_cache));
  LayerKvCache final_stage_cache(16U, 1U, 2U);
  const auto split =
      transformer_layer(transmitted, second, kConfig, 0U, final_stage_cache);

  expect_near(split, reference, 0.0F);
}

TEST(CpuLlamaTest, GreedySamplingSelectsFirstMaximumOnTheLastRow) {
  const Matrix logits(2U, 4U, {9.0F, 1.0F, 2.0F, 3.0F, -1.0F, 4.0F, 4.0F, 0.0F});
  EXPECT_EQ(greedy_sample_last(logits), 1U);
}

}  // namespace
}  // namespace hllm::cpu
