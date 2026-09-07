#pragma once

#include "hllm/runtime/backend_factory.hpp"

namespace hllm::mlx {
// A nonzero limit configures the process-wide MLX allocator guideline. Runtime
// admission enforces the separate model/workspace reservation budget.
[[nodiscard]] std::unique_ptr<runtime::BackendFactory> make_backend_factory(
    std::size_t memory_limit = 0U);
}  // namespace hllm::mlx
