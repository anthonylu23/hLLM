#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace hllm::runtime {

// Owned host staging data, not a universal device tensor. Backends retain all
// internal tensor/cache types; transport explicitly converts to FP16 bytes.
struct HostActivation {
  std::size_t tokens;
  std::size_t width;
  std::vector<float> values;
};

struct SequenceState {
  virtual ~SequenceState() = default;
};

struct SequenceMemory {
  std::size_t cache_bytes;
  std::size_t workspace_bytes;
};

class StageBackend {
 public:
  virtual ~StageBackend() = default;
  [[nodiscard]] virtual std::size_t weight_bytes() const = 0;
  [[nodiscard]] virtual std::size_t hidden_size() const = 0;
  [[nodiscard]] virtual std::size_t vocabulary_size() const = 0;
  [[nodiscard]] virtual std::size_t maximum_tokens() const = 0;
  [[nodiscard]] virtual SequenceMemory sequence_memory(std::size_t tokens) const = 0;
  [[nodiscard]] virtual std::unique_ptr<SequenceState> allocate_sequence(
      std::size_t tokens) const = 0;
  [[nodiscard]] virtual HostActivation embed(std::span<const std::uint64_t> tokens) const = 0;
  [[nodiscard]] virtual HostActivation forward(HostActivation input, std::size_t first_position,
                                               SequenceState& state,
                                               const std::atomic_bool& cancelled) const = 0;
  [[nodiscard]] virtual std::uint64_t sample(const HostActivation& hidden) const = 0;
};

}  // namespace hllm::runtime
