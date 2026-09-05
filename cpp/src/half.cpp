#include "hllm/runtime/half.hpp"

#include <bit>
#include <cstdint>

namespace hllm::runtime {

float float16_to_float(const std::uint16_t value) noexcept {
  const auto sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  auto exponent = static_cast<std::uint32_t>((value >> 10U) & 0x1fU);
  auto mantissa = static_cast<std::uint32_t>(value & 0x03ffU);

  std::uint32_t bits;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      auto unbiased_exponent = -14;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --unbiased_exponent;
      }
      mantissa &= 0x03ffU;
      exponent = static_cast<std::uint32_t>(unbiased_exponent + 127);
      bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    exponent += 127U - 15U;
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::uint16_t float_to_float16(const float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const auto exponent = static_cast<std::uint32_t>((bits >> 23U) & 0xffU);
  const auto mantissa = bits & 0x007fffffU;

  if (exponent == 0xffU) {
    if (mantissa == 0U) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    const auto payload = static_cast<std::uint16_t>(mantissa >> 13U);
    return static_cast<std::uint16_t>(sign | 0x7c00U | payload | 1U);
  }

  const auto half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    const auto normalized = mantissa | 0x00800000U;
    const auto shift = static_cast<std::uint32_t>(14 - half_exponent);
    auto rounded = normalized >> shift;
    const auto remainder_mask = (1U << shift) - 1U;
    const auto remainder = normalized & remainder_mask;
    const auto halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  auto rounded_mantissa = mantissa >> 13U;
  const auto remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_mantissa & 1U) != 0U)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x0400U) {
      rounded_mantissa = 0U;
      const auto incremented_exponent = half_exponent + 1;
      if (incremented_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
      }
      return static_cast<std::uint16_t>(
          sign | (static_cast<std::uint16_t>(incremented_exponent) << 10U));
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint16_t>(half_exponent) << 10U) |
      static_cast<std::uint16_t>(rounded_mantissa));
}

float bfloat16_to_float(const std::uint16_t value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bfloat16(const float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0U) {
    // Keep NaNs as NaNs even when their payload is entirely in the discarded bits.
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const auto least_significant_bit = (bits >> 16U) & 1U;
  const auto rounded = bits + 0x7fffU + least_significant_bit;
  return static_cast<std::uint16_t>(rounded >> 16U);
}

}  // namespace hllm::runtime
