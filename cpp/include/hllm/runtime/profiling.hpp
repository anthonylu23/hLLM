#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace hllm::runtime {
struct ProfilingDeviceInfo {
  std::string identity;
  std::string name;
  std::string backend_version;
  std::string driver_version;
  std::string allocator;
  std::optional<std::size_t> available_bytes;
};
}  // namespace hllm::runtime
