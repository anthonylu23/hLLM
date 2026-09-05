#pragma once

#include <cstdint>

namespace hllm::runtime {

[[nodiscard]] float float16_to_float(std::uint16_t value) noexcept;
[[nodiscard]] std::uint16_t float_to_float16(float value) noexcept;
[[nodiscard]] float bfloat16_to_float(std::uint16_t value) noexcept;
[[nodiscard]] std::uint16_t float_to_bfloat16(float value) noexcept;

}  // namespace hllm::runtime
