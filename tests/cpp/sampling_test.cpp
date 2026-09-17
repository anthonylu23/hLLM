#include "hllm/runtime/sampling.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

#include "hllm/runtime/error.hpp"

TEST(Sampling, GreedyFilteringAndInvalidInputs) {
  using namespace hllm::runtime;
  const std::array<float, 3> logits{0.0F, 2.0F, 2.0F};
  EXPECT_EQ(sample_logits(logits, {}, 0), 1U);
  for (std::uint64_t seed = 0; seed < 100; ++seed) {
    EXPECT_EQ(sample_logits(logits, {1.0, 1.0, 1, seed}, 0), 1U);
    EXPECT_EQ(sample_logits(logits, {1.0, 0.1, 0, seed}, 0), 1U);
  }
  EXPECT_THROW(sample_logits(logits, {-1.0, 1.0, 0, 0}, 0), Error);
  EXPECT_THROW(sample_logits(logits, {1.0, 0.0, 0, 0}, 0), Error);
  const std::array<float, 1> invalid{std::numeric_limits<float>::quiet_NaN()};
  EXPECT_THROW(sample_logits(invalid, {}, 0), Error);
}

TEST(Sampling, RequestRandomnessAndDistribution) {
  using namespace hllm::runtime;
  const std::array<float, 3> logits{0.0F, 0.0F, 0.0F};
  std::array<unsigned, 3> counts{};
  for (std::uint64_t step = 0; step < 6000; ++step) {
    const auto selected = sample_logits(logits, {1.0, 1.0, 0, 42}, step);
    ++counts[selected];
    static_cast<void>(sample_logits(logits, {1.0, 1.0, 0, 999}, step));
    EXPECT_EQ(selected, sample_logits(logits, {1.0, 1.0, 0, 42}, step));
  }
  for (auto count : counts) {
    EXPECT_GT(count, 1800U);
    EXPECT_LT(count, 2200U);
  }
}

TEST(Sampling, BoundedModelLogProbabilitiesPreserveGreedyAndSeededTokens) {
  using namespace hllm::runtime;
  const std::array<float, 3> logits{0.0F, 0.0F, 0.0F};
  SamplingOptions options{0.0, 1.0, 0, 42, true, 2};
  const auto result = sample_token(logits, options, 0);
  EXPECT_EQ(result.id, 0U);
  ASSERT_TRUE(result.logprob.has_value());
  EXPECT_NEAR(*result.logprob, -std::log(3.0), 1e-12);
  ASSERT_EQ(result.top_logprobs.size(), 2U);
  EXPECT_EQ(result.top_logprobs[0].token_id, 0U);
  EXPECT_EQ(result.top_logprobs[1].token_id, 1U);
  options.temperature = 0.8;
  EXPECT_EQ(sample_token(logits, options, 8).id, sample_logits(logits, options, 8));
  options.top_logprobs = 6;
  EXPECT_THROW(sample_token(logits, options, 0), Error);
}
