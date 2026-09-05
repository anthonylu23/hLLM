#include "hllm/runtime/half.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace hllm::runtime {
namespace {

TEST(Float16Test, ConvertsKnownValues) {
  EXPECT_FLOAT_EQ(float16_to_float(0x0000U), 0.0F);
  EXPECT_FLOAT_EQ(float16_to_float(0x3c00U), 1.0F);
  EXPECT_FLOAT_EQ(float16_to_float(0xc000U), -2.0F);
  EXPECT_EQ(float_to_float16(1.0F), 0x3c00U);
  EXPECT_EQ(float_to_float16(-2.0F), 0xc000U);
}

TEST(Float16Test, PreservesSpecialValues) {
  EXPECT_TRUE(std::isinf(float16_to_float(0x7c00U)));
  EXPECT_TRUE(std::isnan(float16_to_float(0x7e00U)));
  EXPECT_EQ(float_to_float16(std::numeric_limits<float>::infinity()), 0x7c00U);
  EXPECT_EQ(float_to_float16(-std::numeric_limits<float>::infinity()), 0xfc00U);
  EXPECT_TRUE((float_to_float16(std::numeric_limits<float>::quiet_NaN()) & 0x7c00U) ==
              0x7c00U);
}

TEST(Float16Test, UsesRoundToNearestEven) {
  EXPECT_EQ(float_to_float16(1.00048828125F), 0x3c00U);
  EXPECT_EQ(float_to_float16(1.00146484375F), 0x3c02U);
}

TEST(BFloat16Test, ConvertsAndRounds) {
  EXPECT_FLOAT_EQ(bfloat16_to_float(0x3f80U), 1.0F);
  EXPECT_EQ(float_to_bfloat16(1.0F), 0x3f80U);
  EXPECT_EQ(float_to_bfloat16(bfloat16_to_float(0x3f81U)), 0x3f81U);
}

}  // namespace
}  // namespace hllm::runtime
