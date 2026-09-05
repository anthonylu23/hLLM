#pragma once

#include <filesystem>
#include <memory>

#include "control.pb.h"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::cpu {

// Validates all selected weights and peak loading memory before reading
// payloads.
[[nodiscard]] std::unique_ptr<runtime::StageBackend> load_stage(const v1::LoadStageRequest& request,
                                                                const std::filesystem::path& root,
                                                                std::size_t memory_limit);

}  // namespace hllm::cpu
