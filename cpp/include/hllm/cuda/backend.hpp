#pragma once

#include "hllm/runtime/backend_factory.hpp"

namespace hllm::cuda {

// Probes the selected CUDA device and LibTorch execution. Throws if unavailable.
// Dense Llama/Qwen3 execution supports resident F32 or F16 weights and caches.
[[nodiscard]] std::unique_ptr<runtime::BackendFactory> make_backend_factory(int device_id = 0);

}  // namespace hllm::cuda
