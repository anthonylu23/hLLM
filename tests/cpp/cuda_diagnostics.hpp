#pragma once
#include <cstddef>
#include <cstdint>
namespace hllm::cuda::test {
struct MemorySnapshot {
  std::int64_t allocated, reserved, peak_allocated;
  std::size_t pinned, rss;
};
MemorySnapshot memory_snapshot();
void reset_peak();
}  // namespace hllm::cuda::test
