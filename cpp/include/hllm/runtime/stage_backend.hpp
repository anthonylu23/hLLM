#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "hllm/runtime/memory.hpp"

namespace hllm::runtime {

// Only stage-boundary data is materialized on the host. Payload is contiguous,
// row-major little-endian FP16, with exactly tokens * width * 2 bytes.
struct BoundaryActivation {
  std::size_t tokens;
  std::size_t width;
  std::vector<std::byte> payload;
};
struct TokenInput {
  std::vector<std::uint64_t> ids;
};
struct SampledToken {
  std::uint64_t id;
};
using StageInput = std::variant<TokenInput, BoundaryActivation>;
using StageOutput = std::variant<BoundaryActivation, SampledToken>;

struct SequenceState {
  virtual ~SequenceState() = default;
};

class StageBackend {
 public:
  virtual ~StageBackend() = default;
  [[nodiscard]] virtual MemoryAmounts weight_memory() const = 0;
  [[nodiscard]] virtual std::size_t hidden_size() const = 0;
  [[nodiscard]] virtual std::size_t vocabulary_size() const = 0;
  [[nodiscard]] virtual std::size_t maximum_tokens() const = 0;
  [[nodiscard]] virtual SequenceMemory sequence_memory(std::size_t tokens) const = 0;
  [[nodiscard]] virtual std::unique_ptr<SequenceState> allocate_sequence(
      std::size_t tokens) const = 0;
  // Execute embedding (when first), owned layers, and sampling (when last) as
  // one backend operation. Internal activations/cache never cross this API.
  // Returned host bytes must be ready for transport; GPU events must complete
  // before returning or releasing any buffers used by that operation.
  [[nodiscard]] virtual StageOutput execute(StageInput input, std::size_t first_position,
                                            SequenceState& state,
                                            const std::atomic_bool& cancelled) const = 0;
};

}  // namespace hllm::runtime
