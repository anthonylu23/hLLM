#include "hllm/runtime/buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

namespace hllm::runtime {
namespace {

TEST(BufferPoolTest, HoldsLeaseUntilDestructionAndReusesStorage) {
  BufferPool pool;
  std::byte* original = nullptr;
  {
    auto buffer = pool.acquire(1024U, 64U);
    original = buffer.data();
    buffer.resize(512U);
    buffer.bytes()[0] = std::byte{0x2a};

    const auto active = pool.stats();
    EXPECT_EQ(active.buffers_in_use, 1U);
    EXPECT_EQ(active.bytes_in_use, 1024U);
    EXPECT_EQ(active.buffers_cached, 0U);
  }

  const auto released = pool.stats();
  EXPECT_EQ(released.buffers_in_use, 0U);
  EXPECT_EQ(released.buffers_cached, 1U);
  EXPECT_EQ(released.allocations, 1U);

  auto reused = pool.acquire(512U, 64U);
  EXPECT_EQ(reused.data(), original);
  EXPECT_EQ(reused.size(), 0U);
  EXPECT_GE(reused.capacity(), 512U);
  EXPECT_EQ(pool.stats().allocations, 1U);
}

TEST(BufferPoolTest, MoveTransfersTheOnlyLease) {
  BufferPool pool;
  auto first = pool.acquire(128U);
  auto second = std::move(first);

  EXPECT_FALSE(first);
  EXPECT_TRUE(second);
  EXPECT_EQ(pool.stats().buffers_in_use, 1U);
}

TEST(BufferPoolTest, RejectsInvalidSizesAndBounds) {
  BufferPool pool;
  EXPECT_THROW(static_cast<void>(pool.acquire(0U)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(pool.acquire(64U, 3U)), std::invalid_argument);

  auto buffer = pool.acquire(64U);
  EXPECT_THROW(buffer.resize(65U), std::length_error);
}

TEST(BufferPoolTest, RespectsBoundedCache) {
  BufferPool pool(64U);
  {
    auto too_large = pool.acquire(128U);
  }
  EXPECT_EQ(pool.stats().buffers_cached, 0U);

  {
    auto fits = pool.acquire(64U);
  }
  EXPECT_EQ(pool.stats().buffers_cached, 1U);
  pool.clear_cached();
  EXPECT_EQ(pool.stats().bytes_cached, 0U);
}

}  // namespace
}  // namespace hllm::runtime
