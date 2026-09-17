#pragma once

#include <gtest/gtest.h>

#include "hllm/runtime/stage_backend.hpp"

namespace hllm::test {
inline void decode_batch_case(runtime::StageBackend& stage, double temperature) {
  ASSERT_TRUE(stage.supports_decode_batch());
  std::atomic_bool cancelled{false};
  std::vector<std::unique_ptr<runtime::SequenceState>> reference, batched;
  const std::vector<std::size_t> lengths{7, 2, 4, 1};
  std::vector<runtime::SampledToken> expected;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    reference.push_back(stage.allocate_sequence(lengths[i] + 1U));
    batched.push_back(stage.allocate_sequence(lengths[i] + 1U));
    for (auto* state : {reference.back().get(), batched.back().get()}) {
      state->sampling = {temperature, 0.95, 8, i + 42U, true, 3};
      static_cast<void>(stage.execute(
          runtime::TokenInput{std::vector<std::uint64_t>(lengths[i], 1U)}, 0U, *state, cancelled));
    }
    expected.push_back(
        std::get<runtime::SampledToken>(
            stage.execute(runtime::TokenInput{{4U}}, lengths[i], *reference.back(), cancelled)));
  }
  std::vector<runtime::DecodeBatchItem> items;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    items.push_back({runtime::TokenInput{{4U}}, lengths[i], batched[i].get(), &cancelled});
  }
  const auto actual = stage.execute_decode_batch(std::move(items));
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& token = std::get<runtime::SampledToken>(actual[i]);
    // Exact token equality is a regression for this fixed tiny fixture only.
    // Full checkpoints can change seeded selections with batch-dependent rounding.
    EXPECT_EQ(token.id, expected[i].id);
    ASSERT_TRUE(token.logprob.has_value());
    ASSERT_TRUE(expected[i].logprob.has_value());
    EXPECT_NEAR(*token.logprob, *expected[i].logprob, 0.02);
    EXPECT_EQ(token.top_logprobs.size(), 3U);
    EXPECT_EQ(batched[i]->generated_index, reference[i]->generated_index);
  }
}
inline void decode_batch_contract(runtime::StageBackend& stage) {
  decode_batch_case(stage, 0.0);
  decode_batch_case(stage, 0.8);
}
}  // namespace hllm::test
