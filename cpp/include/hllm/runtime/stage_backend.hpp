#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/memory.hpp"
#include "hllm/runtime/sampling.hpp"
#include "hllm/runtime/timing.hpp"

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
using StageInput = std::variant<TokenInput, BoundaryActivation>;

// Host bytes the serving handlers hold for one request outside the stage's serialized
// passes: the whole prompt's token IDs up to three times (request message, prompt and
// input vectors, whatever the chunk size) and one input's F16 boundary activations, of
// at most `rows` rows, up to three times (the activation, its message payload and
// gRPC's serialized copy). Queued and streaming requests hold them at the same time, so
// backends reserve them per request, never in the shared workspace.
[[nodiscard]] inline std::size_t request_transport_bytes(std::size_t tokens, std::size_t rows,
                                                         std::size_t hidden) {
  return checked_add(checked_multiply(tokens, 3U * sizeof(std::uint64_t)),
                     checked_multiply(checked_multiply(rows, hidden), 3U * 2U));
}
struct PrefillProgress {};
using StageOutput = std::variant<BoundaryActivation, SampledToken, PrefillProgress>;

struct SequenceState {
  SamplingOptions sampling;
  bool prefilling{false};
  bool prefill_only{false};
  std::uint64_t generated_index{0};
  virtual ~SequenceState() = default;
};

struct DecodeBatchItem {
  StageInput input;
  std::size_t position;
  SequenceState* sequence;
  const std::atomic_bool* cancelled;
};
StageInput combine_decode_inputs(const std::vector<DecodeBatchItem>& items);
std::vector<StageOutput> split_decode_boundary(BoundaryActivation output);

// Effective query rows for a reservation: zero or anything at least `tokens` is the
// whole sequence.
[[nodiscard]] constexpr std::size_t query_rows(std::size_t tokens,
                                               std::size_t maximum_query_tokens) noexcept {
  return maximum_query_tokens == 0U || maximum_query_tokens > tokens ? tokens
                                                                     : maximum_query_tokens;
}

class StageBackend {
 public:
  virtual ~StageBackend() = default;
  [[nodiscard]] virtual MemoryAmounts weight_memory() const = 0;
  [[nodiscard]] virtual std::size_t hidden_size() const = 0;
  [[nodiscard]] virtual std::size_t vocabulary_size() const = 0;
  [[nodiscard]] virtual std::size_t maximum_tokens() const = 0;
  // Reservation for a sequence of `tokens` total positions whose prefill inputs carry
  // at most `maximum_query_tokens` rows each (chunked prefill). Zero, or a value of at
  // least `tokens`, means whole-prompt prefill: the reservation for `tokens` rows.
  [[nodiscard]] virtual SequenceMemory sequence_memory(std::size_t tokens,
                                                       std::size_t maximum_query_tokens) const = 0;
  [[nodiscard]] SequenceMemory sequence_memory(std::size_t tokens) const {
    return sequence_memory(tokens, tokens);
  }
  [[nodiscard]] virtual std::unique_ptr<SequenceState> allocate_sequence(
      std::size_t tokens) const = 0;
  // Execute embedding (when first), owned layers, and sampling (when last) as
  // one backend operation. Internal activations/cache never cross this API.
  // Returned host bytes must be ready for transport; GPU events must complete
  // before returning or releasing any buffers used by that operation.
  [[nodiscard]] virtual StageOutput execute(StageInput input, std::size_t first_position,
                                            SequenceState& state,
                                            const std::atomic_bool& cancelled) const = 0;
  virtual bool supports_decode_batch() const { return false; }
  virtual std::vector<StageOutput> execute_decode_batch(std::vector<DecodeBatchItem> items) const {
    std::vector<StageOutput> outputs;
    for (auto& item : items) {
      outputs.push_back(
          execute(std::move(item.input), item.position, *item.sequence, *item.cancelled));
    }
    return outputs;
  }
  // Opt-in diagnostics for dedicated profiling processes. Serving uses execute().
  [[nodiscard]] virtual StageOutput execute_profiled(StageInput, std::size_t,
      SequenceState&, const std::atomic_bool&, ExecutionTiming&) const {
    throw std::logic_error("backend does not implement compute profiling");
  }
};

}  // namespace hllm::runtime
