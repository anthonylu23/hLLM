#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "control.pb.h"
#include "hllm/runtime/profiling.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::runtime {

struct BackendCapabilities {
  v1::Backend kind;
  v1::MemoryDomain primary_memory_domain;
  std::vector<std::string> architectures;
  std::vector<v1::DataType> execution_dtypes;
  std::string detail;
};

struct AllocatorMetrics {
  std::size_t active_bytes, cached_bytes, peak_bytes;
};

class BackendFactory {
 public:
  virtual ~BackendFactory() = default;
  [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;
  [[nodiscard]] virtual std::optional<AllocatorMetrics> allocator_metrics() const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::string boundary_transport_mode() const { return "pageable"; }
  // Dedicated profiling processes only. Never exposed through WorkerControl.
  // Synchronize completed work before resetting; do not evict allocator caches.
  virtual bool profiling_reset_peak() const { return false; }
  [[nodiscard]] virtual ProfilingDeviceInfo profiling_device_info() const {
    return {"host", "CPU", "hllm-native", "not-applicable", "system", std::nullopt};
  }
  // Validate peak load/conversion memory in every domain before allocating.
  // Return only a complete stage; partial allocations must be owned by RAII.
  [[nodiscard]] virtual std::unique_ptr<StageBackend> load(const v1::LoadStageRequest& request,
                                                           const std::filesystem::path& root,
                                                           const MemoryAmounts& capacity) const = 0;
};

}  // namespace hllm::runtime
