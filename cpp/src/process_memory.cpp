#include "hllm/runtime/process_memory.hpp"

#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#include <mach/mach.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>

namespace hllm::runtime {
namespace {
std::optional<std::uint64_t> bytes(std::uint64_t count, std::uint64_t unit) {
  if (unit == 0U || count > std::numeric_limits<std::uint64_t>::max() / unit)
    return std::nullopt;
  return count * unit;
}
}  // namespace

ProcessMemoryObservation observe_process_memory() {
  ProcessMemoryObservation result;
  result.process_id = static_cast<std::uint64_t>(getpid());
  result.observed_at_unix_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
#ifdef __APPLE__
  rusage_info_v4 info{};
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                     reinterpret_cast<rusage_info_t*>(&info)) == 0) {
    result.rss_bytes = info.ri_resident_size;
    result.physical_footprint_bytes = info.ri_phys_footprint;
    result.physical_footprint_lifetime_peak_bytes = info.ri_lifetime_max_phys_footprint;
  } else {
    // Preserve RSS even if footprint accounting is unavailable.
    mach_task_basic_info_data_t task{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&task), &count) == KERN_SUCCESS)
      result.rss_bytes = task.resident_size;
  }
#elif defined(__linux__)
  std::ifstream stat("/proc/self/statm");
  std::uint64_t total = 0U, resident = 0U;
  const auto page = sysconf(_SC_PAGESIZE);
  if (stat >> total >> resident && page > 0)
    result.rss_bytes = bytes(resident, static_cast<std::uint64_t>(page));
#endif
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0) {
#ifdef __APPLE__
    constexpr auto unit = 1U;
#else
    constexpr auto unit = 1024U;
#endif
    result.rss_lifetime_peak_bytes = bytes(static_cast<std::uint64_t>(usage.ru_maxrss), unit);
  }
  // Consumers require every lifetime peak to cover its adjacent current value and to
  // never decrease. Linux reads both through approximate per-CPU RSS counters, so
  // ru_maxrss can trail /proc/self/statm by a few pages; fold the current value and
  // this process's previously reported peak into the reported peak.
  static std::atomic<std::uint64_t> reported_peak{0U};
  if (result.rss_lifetime_peak_bytes) {
    const auto peak = std::max(*result.rss_lifetime_peak_bytes, result.rss_bytes.value_or(0U));
    auto previous = reported_peak.load();
    while (previous < peak && !reported_peak.compare_exchange_weak(previous, peak)) {
    }
    result.rss_lifetime_peak_bytes = std::max(peak, previous);
  }
  return result;
}
}  // namespace hllm::runtime
