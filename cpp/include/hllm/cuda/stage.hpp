#pragma once

#include "hllm/runtime/stage_backend.hpp"

namespace hllm::cuda {
// Numerical-test snapshots only. Production execute() never materializes these
// intermediates on the host. Keys/values use [kv_head, sequence, head_dimension].
struct ExecutionTrace {
  std::vector<std::vector<float>> layers;
  std::vector<std::vector<float>> keys;
  std::vector<std::vector<float>> values;
  std::vector<float> last_logits;
};
class ReferenceStage : public runtime::StageBackend {
 public:
  [[nodiscard]] virtual runtime::StageOutput execute_traced(runtime::StageInput input,
                                                            std::size_t position,
                                                            runtime::SequenceState& state,
                                                            const std::atomic_bool& cancelled,
                                                            ExecutionTrace& trace) const = 0;
};
}  // namespace hllm::cuda
