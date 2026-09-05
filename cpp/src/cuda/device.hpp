#pragma once

#include <string>

namespace hllm::cuda {
// Keep Torch and its bundled dependencies out of protobuf-facing translation units.
[[nodiscard]] std::string probe_device(int device_id);
}  // namespace hllm::cuda
