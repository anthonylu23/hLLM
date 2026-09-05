#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hllm/runtime/safetensors.hpp"

namespace hllm::runtime {

enum class ExecutionPhase : std::uint8_t { kPrefill, kDecode };
enum class TensorLayout : std::uint8_t { kDenseRowMajorLittleEndian };

struct TensorEnvelopeMetadata {
  std::uint32_t protocol_version;
  std::string deployment_id;
  std::uint64_t deployment_version;
  std::string request_id;
  std::uint64_t microbatch_id;
  std::uint64_t sequence_number;
  ExecutionPhase phase;
  std::uint64_t first_position;
  std::vector<std::uint64_t> sequence_lengths;
  std::vector<std::uint64_t> cache_slot_ids;
  std::vector<std::uint64_t> shape;
  DataType dtype;
  TensorLayout layout;
  std::uint64_t payload_length;
};

struct TensorEnvelopeLimits {
  std::uint32_t protocol_version{1U};
  std::size_t maximum_batch{1U};
  std::size_t maximum_sequence_length;
  std::size_t hidden_size;
  std::size_t maximum_payload_bytes;
  DataType boundary_dtype{DataType::kF16};
};

class EnvelopeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void validate_tensor_envelope(const TensorEnvelopeMetadata& envelope,
                              std::size_t received_payload_bytes,
                              const TensorEnvelopeLimits& limits);

class SequenceTracker final {
 public:
  void accept(std::uint64_t sequence_number);
  [[nodiscard]] std::uint64_t expected() const noexcept { return expected_; }

 private:
  std::uint64_t expected_{0U};
};

}  // namespace hllm::runtime
