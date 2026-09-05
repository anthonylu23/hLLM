#include "hllm/runtime/tensor_envelope.hpp"

#include <limits>

namespace hllm::runtime {
namespace {

[[nodiscard]] std::size_t checked_size(const std::uint64_t value,
                                       const char* const description) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw EnvelopeError(std::string(description) + " exceeds native size limits");
  }
  return static_cast<std::size_t>(value);
}

[[nodiscard]] std::size_t checked_multiply(const std::size_t lhs, const std::size_t rhs) {
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw EnvelopeError("tensor payload size overflows native size limits");
  }
  return lhs * rhs;
}

}  // namespace

void validate_tensor_envelope(const TensorEnvelopeMetadata& envelope,
                              const std::size_t received_payload_bytes,
                              const TensorEnvelopeLimits& limits) {
  if (envelope.protocol_version != limits.protocol_version) {
    throw EnvelopeError("unsupported tensor protocol version");
  }
  if (envelope.deployment_id.empty() || envelope.deployment_version == 0U ||
      envelope.request_id.empty()) {
    throw EnvelopeError("tensor envelope is missing deployment or request identity");
  }
  if (limits.maximum_batch == 0U || limits.maximum_sequence_length == 0U ||
      limits.hidden_size == 0U || limits.maximum_payload_bytes == 0U) {
    throw EnvelopeError("tensor envelope limits are invalid");
  }
  if (envelope.layout != TensorLayout::kDenseRowMajorLittleEndian ||
      envelope.dtype != limits.boundary_dtype) {
    throw EnvelopeError("tensor boundary representation is incompatible");
  }
  if (envelope.shape.size() != 3U) {
    throw EnvelopeError("tensor boundary shape must be [batch, sequence, hidden]");
  }

  const auto batch = checked_size(envelope.shape[0], "batch dimension");
  const auto sequence = checked_size(envelope.shape[1], "sequence dimension");
  const auto hidden = checked_size(envelope.shape[2], "hidden dimension");
  if (batch == 0U || batch > limits.maximum_batch || sequence == 0U ||
      hidden != limits.hidden_size || envelope.sequence_lengths.size() != batch ||
      envelope.cache_slot_ids.size() != batch) {
    throw EnvelopeError("tensor boundary dimensions or batch metadata are invalid");
  }
  if (envelope.phase == ExecutionPhase::kDecode && sequence != 1U) {
    throw EnvelopeError("decode tensor must have a singleton sequence dimension");
  }

  for (const auto raw_length : envelope.sequence_lengths) {
    const auto length = checked_size(raw_length, "sequence length");
    const auto first_position = checked_size(envelope.first_position, "first position");
    if (length == 0U || length != sequence || first_position > limits.maximum_sequence_length ||
        length > limits.maximum_sequence_length - first_position) {
      throw EnvelopeError("tensor sequence metadata exceeds configured capacity");
    }
  }

  auto expected_payload = checked_multiply(batch, sequence);
  expected_payload = checked_multiply(expected_payload, hidden);
  expected_payload = checked_multiply(expected_payload, data_type_bytes(envelope.dtype));
  const auto declared_payload = checked_size(envelope.payload_length, "payload length");
  if (expected_payload != declared_payload || declared_payload != received_payload_bytes ||
      declared_payload > limits.maximum_payload_bytes) {
    throw EnvelopeError("tensor payload length does not match its shape or limits");
  }
}

void SequenceTracker::accept(const std::uint64_t sequence_number) {
  if (sequence_number != expected_) {
    throw EnvelopeError("stage message arrived out of order");
  }
  if (expected_ == std::numeric_limits<std::uint64_t>::max()) {
    throw EnvelopeError("stage sequence number exhausted");
  }
  ++expected_;
}

}  // namespace hllm::runtime
