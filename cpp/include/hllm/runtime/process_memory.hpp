#pragma once

#include <cstdint>
#include <optional>

namespace hllm::runtime {

// OS accounting, separate from logical reservations and framework allocations.
// Missing counters are unavailable, never zero. Peaks cover the process lifetime.
struct ProcessMemoryObservation {
  std::uint64_t process_id{};
  std::uint64_t observed_at_unix_ns{};
  std::optional<std::uint64_t> rss_bytes;
  std::optional<std::uint64_t> rss_lifetime_peak_bytes;
  std::optional<std::uint64_t> physical_footprint_bytes;
  std::optional<std::uint64_t> physical_footprint_lifetime_peak_bytes;
};

// Read-only: does not reset peaks, clear allocator caches, or synchronize devices.
ProcessMemoryObservation observe_process_memory();

}  // namespace hllm::runtime
