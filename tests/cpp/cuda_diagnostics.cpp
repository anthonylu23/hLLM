// Test-only Torch translation unit, isolated from protobuf headers.
#include "cuda_diagnostics.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <unistd.h>
#include <fstream>
#include "pinned_buffer.hpp"
namespace hllm::cuda::test {
MemorySnapshot memory_snapshot() {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  std::size_t virtual_pages{}, resident_pages{};
  std::ifstream("/proc/self/statm") >> virtual_pages >> resident_pages;
  return {stats.allocated_bytes[0].current, stats.reserved_bytes[0].current,
          stats.allocated_bytes[0].peak, allocated_pinned_bytes.load(),
          resident_pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE))};
}
void reset_peak() { c10::cuda::CUDACachingAllocator::resetPeakStats(0); }
}  // namespace hllm::cuda::test
