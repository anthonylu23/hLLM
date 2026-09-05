#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hllm::runtime {

// Pinned bytes are a subset of host bytes, with an additional independent cap.
struct MemoryAmounts {
  std::size_t host_bytes{0U};
  std::size_t device_bytes{0U};
  std::size_t pinned_host_bytes{0U};
};

inline MemoryAmounts add_memory(const MemoryAmounts& a, const MemoryAmounts& b) {
  const auto add = [](std::size_t x, std::size_t y) {
    if (y > std::numeric_limits<std::size_t>::max() - x) {
      throw std::length_error("memory accounting overflow");
    }
    return x + y;
  };
  return {add(a.host_bytes, b.host_bytes), add(a.device_bytes, b.device_bytes),
          add(a.pinned_host_bytes, b.pinned_host_bytes)};
}

inline void require_memory(const MemoryAmounts& used, const MemoryAmounts& capacity) {
  if (used.pinned_host_bytes > used.host_bytes) {
    throw std::invalid_argument("pinned memory must also be counted in host memory");
  }
  if (used.host_bytes > capacity.host_bytes || used.device_bytes > capacity.device_bytes ||
      used.pinned_host_bytes > capacity.pinned_host_bytes) {
    throw std::length_error("allocation exceeds host, device, or pinned-host memory budget");
  }
}

struct SequenceMemory {
  MemoryAmounts cache;
  MemoryAmounts workspace;
};

}  // namespace hllm::runtime
