#include "hllm/worker/boundary_codec.hpp"
#include <cstring>
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/tensor_envelope.hpp"
namespace hllm::worker {
namespace {
void validate_identity(const std::string& deployment, std::uint64_t version,
                       const std::string& request, std::uint64_t microbatch,
                       const BoundaryIdentity& identity) {
  if (deployment != identity.deployment_id || version != identity.deployment_version ||
      request != identity.request_id || microbatch != 0U)
    throw runtime::Error::invalid_request("message identity does not match the active sequence");
}
}
runtime::BoundaryActivation decode_tensor(const v1::TensorEnvelope& tensor,
                                          const BoundaryIdentity& identity, std::uint64_t sequence,
                                          std::size_t position) {
  validate_identity(tensor.deployment_id(), tensor.deployment_version(), tensor.request_id(),
                    tensor.microbatch_id(), identity);
  if (tensor.sequence_number() != sequence || tensor.first_position() != position ||
      tensor.phase() !=
          (sequence == 0U ? v1::EXECUTION_PHASE_PREFILL : v1::EXECUTION_PHASE_DECODE) ||
      tensor.dtype() != v1::DATA_TYPE_F16 || tensor.layout() != "dense_row_major_le" ||
      !tensor.checksum().empty()) {
    throw runtime::Error::invalid_request(
        "invalid tensor order, phase, encoding or unsupported checksum");
  }
  runtime::TensorEnvelopeMetadata metadata{
      tensor.protocol_version(),
      tensor.deployment_id(),
      tensor.deployment_version(),
      tensor.request_id(),
      tensor.microbatch_id(),
      tensor.sequence_number(),
      sequence == 0U ? runtime::ExecutionPhase::kPrefill : runtime::ExecutionPhase::kDecode,
      tensor.first_position(),
      {tensor.sequence_lengths().begin(), tensor.sequence_lengths().end()},
      {tensor.cache_slot_ids().begin(), tensor.cache_slot_ids().end()},
      {tensor.shape().begin(), tensor.shape().end()},
      runtime::DataType::kF16,
      runtime::TensorLayout::kDenseRowMajorLittleEndian,
      tensor.payload_length()};
  runtime::validate_tensor_envelope(
      metadata, tensor.payload().size(),
      {.maximum_sequence_length = identity.maximum_tokens,
       .hidden_size = identity.hidden_size,
       .maximum_payload_bytes = static_cast<std::size_t>(kMaximumRpcBytes / 2)});
  if (tensor.cache_slot_ids_size() != 1 || tensor.cache_slot_ids(0) != 0U) {
    throw runtime::Error::invalid_request("single-sequence execution requires cache slot zero");
  }
  runtime::BoundaryActivation output{static_cast<std::size_t>(tensor.shape(1)),
                                     identity.hidden_size,
                                     std::vector<std::byte>(tensor.payload().size())};
  std::memcpy(output.payload.data(), tensor.payload().data(), tensor.payload().size());
  for (std::size_t i = 0U; i < output.payload.size(); i += 2U) {
    const auto high = std::to_integer<unsigned char>(output.payload[i + 1U]);
    if ((high & 0x7cU) == 0x7cU) {
      throw runtime::Error::invalid_request("non-finite boundary activation");
    }
  }
  return output;
}

v1::StageMessage encode_tensor(const runtime::BoundaryActivation& hidden,
                               const BoundaryIdentity& identity, std::uint64_t sequence,
                               std::size_t position) {
  if (hidden.payload.size() > static_cast<std::size_t>(kMaximumRpcBytes / 2)) {
    throw runtime::Error::resource_exhausted("prefill activation exceeds transport limit");
  }
  v1::StageMessage message;
  auto* tensor = message.mutable_tensor();
  tensor->set_protocol_version(1U);
  tensor->set_deployment_id(identity.deployment_id);
  tensor->set_deployment_version(identity.deployment_version);
  tensor->set_request_id(identity.request_id);
  tensor->set_sequence_number(sequence);
  tensor->set_phase(sequence == 0U ? v1::EXECUTION_PHASE_PREFILL : v1::EXECUTION_PHASE_DECODE);
  tensor->set_first_position(position);
  tensor->add_sequence_lengths(hidden.tokens);
  tensor->add_cache_slot_ids(0U);
  tensor->add_shape(1U);
  tensor->add_shape(hidden.tokens);
  tensor->add_shape(hidden.width);
  tensor->set_dtype(v1::DATA_TYPE_F16);
  tensor->set_layout("dense_row_major_le");
  if (hidden.width == 0U || hidden.width != identity.hidden_size ||
      hidden.tokens == 0U ||
      hidden.tokens > static_cast<std::size_t>(kMaximumRpcBytes / 4) / hidden.width ||
      hidden.payload.size() != hidden.tokens * hidden.width * 2U) {
    throw runtime::Error::internal("backend returned an invalid boundary shape");
  }
  for (std::size_t i = 1U; i < hidden.payload.size(); i += 2U) {
    if ((std::to_integer<unsigned char>(hidden.payload[i]) & 0x7cU) == 0x7cU) {
      throw runtime::Error::internal("backend returned a non-finite boundary activation");
    }
  }
  tensor->set_payload(hidden.payload.data(), hidden.payload.size());
  tensor->set_payload_length(hidden.payload.size());
  return message;
}

}  // namespace hllm::worker
