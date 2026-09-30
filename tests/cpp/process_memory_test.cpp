#include <gtest/gtest.h>

#include <unistd.h>
#include <cstddef>
#include <memory>

#include "hllm/runtime/process_memory.hpp"

namespace hllm::runtime {
namespace {
TEST(ProcessMemoryTest, ObservesResidentPagesAndPreservesLifetimePeaks) {
  const auto before = observe_process_memory();
  EXPECT_EQ(before.process_id, static_cast<std::uint64_t>(getpid()));
  EXPECT_GT(before.observed_at_unix_ns, 0U);
  ASSERT_TRUE(before.rss_bytes);
  ASSERT_TRUE(before.rss_lifetime_peak_bytes);
  EXPECT_GT(*before.rss_bytes, 0U);
  // Touch pages through a volatile pointer so the allocation cannot be elided.
  constexpr std::size_t size = 16U * 1024U * 1024U;
  auto allocation = std::make_unique<std::byte[]>(size);
  volatile std::byte* pages = allocation.get();
  for (std::size_t i = 0; i < size; i += 4096U) pages[i] = std::byte{1};
  const auto loaded = observe_process_memory();
  allocation.reset();
  const auto released = observe_process_memory();
  ASSERT_TRUE(loaded.rss_bytes);
  ASSERT_TRUE(loaded.rss_lifetime_peak_bytes);
  ASSERT_TRUE(released.rss_lifetime_peak_bytes);
  EXPECT_GT(*loaded.rss_bytes, *before.rss_bytes);
  // The lifetime peak must come from the same unit/source as the current value: the
  // Python footprint policy rejects any sample whose peak is below its current RSS.
  EXPECT_GE(*before.rss_lifetime_peak_bytes, *before.rss_bytes);
  EXPECT_GE(*loaded.rss_lifetime_peak_bytes, *loaded.rss_bytes);
  EXPECT_GE(*released.rss_lifetime_peak_bytes, *loaded.rss_lifetime_peak_bytes);
  EXPECT_GE(*loaded.rss_lifetime_peak_bytes, *before.rss_lifetime_peak_bytes);
#ifdef __APPLE__
  ASSERT_TRUE(before.physical_footprint_bytes);
  ASSERT_TRUE(before.physical_footprint_lifetime_peak_bytes);
  ASSERT_TRUE(loaded.physical_footprint_bytes);
  ASSERT_TRUE(loaded.physical_footprint_lifetime_peak_bytes);
  ASSERT_TRUE(released.physical_footprint_lifetime_peak_bytes);
  EXPECT_GT(*loaded.physical_footprint_bytes, *before.physical_footprint_bytes);
  EXPECT_GE(*loaded.physical_footprint_lifetime_peak_bytes, *loaded.physical_footprint_bytes);
  EXPECT_GE(*released.physical_footprint_lifetime_peak_bytes,
            *loaded.physical_footprint_lifetime_peak_bytes);
#else
  EXPECT_FALSE(loaded.physical_footprint_bytes);
  EXPECT_FALSE(loaded.physical_footprint_lifetime_peak_bytes);
#endif
}
}  // namespace
}  // namespace hllm::runtime
