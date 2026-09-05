#pragma once

#include <filesystem>
#include <memory>
#include <span>

#include "control.pb.h"
#include "hllm/runtime/backend_factory.hpp"

namespace hllm::cpu {

// CPU-only inspection surface used by the numerical oracle tests. Device
// backends and common orchestration do not implement or depend on this API.
struct HostActivation {
  std::size_t tokens;
  std::size_t width;
  std::vector<float> values;
};
class ReferenceStage : public runtime::StageBackend {
 public:
  [[nodiscard]] virtual HostActivation embed(std::span<const std::uint64_t>) const = 0;
  [[nodiscard]] virtual HostActivation forward(HostActivation, std::size_t, runtime::SequenceState&,
                                               const std::atomic_bool&) const = 0;
  [[nodiscard]] virtual std::uint64_t sample(const HostActivation&) const = 0;
};

[[nodiscard]] std::unique_ptr<ReferenceStage> load_stage(const v1::LoadStageRequest& request,
                                                         const std::filesystem::path& root,
                                                         std::size_t memory_limit);
[[nodiscard]] std::unique_ptr<runtime::BackendFactory> make_backend_factory();

}  // namespace hllm::cpu
