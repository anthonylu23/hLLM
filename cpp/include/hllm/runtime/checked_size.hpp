#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hllm::runtime {
inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error("stage memory size overflow");
  }
  return a + b;
}
inline std::size_t checked_multiply(std::size_t a, std::size_t b) {
  if (b != 0U && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::length_error("stage memory size overflow");
  }
  return a * b;
}
}  // namespace hllm::runtime
