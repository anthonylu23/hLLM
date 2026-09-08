#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace hllm::cuda {
// Keep Torch and its bundled dependencies out of protobuf-facing translation units.
[[nodiscard]] std::string probe_device(int device_id);
// Active (including pending frees), unallocated reserved, peak active bytes.
[[nodiscard]] std::optional<std::array<std::size_t, 3>> device_allocator_metrics(int device_id);
}  // namespace hllm::cuda

#include "hllm/model/dense_source.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::cuda {
[[nodiscard]] std::unique_ptr<runtime::StageBackend> load_device_stage(
    const model::DenseSource& source, int device_id, const runtime::MemoryAmounts& capacity,
    bool pinned = false);
}  // namespace hllm::cuda
