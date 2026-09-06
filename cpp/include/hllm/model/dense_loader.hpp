#pragma once

#include "control.pb.h"
#include "hllm/model/dense_source.hpp"

namespace hllm::model {
// Reject unsupported semantics/ownership and validate all assigned file metadata
// before reading weight payloads. Source files must remain immutable while loading.
[[nodiscard]] DenseSource inspect_dense_stage(const v1::LoadStageRequest& request,
                                              const std::filesystem::path& root);
}  // namespace hllm::model
