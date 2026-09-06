#pragma once

#include "hllm/runtime/backend_factory.hpp"

namespace hllm::cuda {

// Probes the selected CUDA device and LibTorch execution. Throws if unavailable.
// This integration target does not yet advertise executable model architectures.
[[nodiscard]] std::unique_ptr<runtime::BackendFactory> make_backend_factory(int device_id = 0);

}  // namespace hllm::cuda
