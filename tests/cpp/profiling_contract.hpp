#pragma once
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include "hllm/runtime/stage_backend.hpp"
namespace hllm::test {
inline void paired_timing_contract(runtime::StageBackend& first, runtime::StageBackend& last) {
  auto first_plain = first.allocate_sequence(16);
  auto first_timed = first.allocate_sequence(16);
  auto last_plain = last.allocate_sequence(16);
  auto last_timed = last.allocate_sequence(16);
  const std::array<std::uint64_t, 5> ids{1, 4, 2, 8, 3};
  std::atomic_bool cancelled{false};
  for (std::size_t position : {0U, 3U, 4U}) {
    const std::size_t count = position == 0 ? 3 : 1;
    const runtime::TokenInput input{{ids.begin() + static_cast<std::ptrdiff_t>(position),
                                    ids.begin() + static_cast<std::ptrdiff_t>(position + count)}};
    runtime::ExecutionTiming timing;
    const auto a = first.execute(input, position, *first_plain, cancelled);
    const auto b = first.execute_profiled(input, position, *first_timed, cancelled, timing);
    ASSERT_TRUE(std::holds_alternative<runtime::BoundaryActivation>(a));
    ASSERT_TRUE(std::holds_alternative<runtime::BoundaryActivation>(b));
    EXPECT_EQ(std::get<runtime::BoundaryActivation>(a).payload,
              std::get<runtime::BoundaryActivation>(b).payload);
    ASSERT_FALSE(timing.components.empty());
    EXPECT_EQ(timing.components.front().component, "embedding");
    EXPECT_EQ(timing.components.back().component, "to-wire");
    for (const auto& t : timing.components) {
      EXPECT_TRUE(std::isfinite(t.milliseconds));
      EXPECT_GE(t.milliseconds, 0);
      if (t.component == "layer") { EXPECT_EQ(t.layer_index, 0U); }
    }
    const auto c = last.execute(std::get<runtime::BoundaryActivation>(a), position,
                                *last_plain, cancelled);
    const auto d = last.execute_profiled(std::get<runtime::BoundaryActivation>(b), position,
                                         *last_timed, cancelled, timing);
    EXPECT_EQ(std::get<runtime::SampledToken>(c).id, std::get<runtime::SampledToken>(d).id);
    EXPECT_EQ(timing.components.front().component, "from-wire");
    EXPECT_EQ(timing.components.back().component, "sampling");
    for (const auto& t : timing.components) {
      if (t.component == "layer") { EXPECT_EQ(t.layer_index, 1U); }
    }
  }
}
}
