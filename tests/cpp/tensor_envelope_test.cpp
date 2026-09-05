#include "hllm/runtime/tensor_envelope.hpp"

#include <cstddef>

#include <gtest/gtest.h>

namespace hllm::runtime {
namespace {

[[nodiscard]] TensorEnvelopeMetadata valid_envelope() {
  return TensorEnvelopeMetadata{
      .protocol_version = 1U,
      .deployment_id = "plan-1",
      .deployment_version = 1U,
      .request_id = "request-1",
      .microbatch_id = 0U,
      .sequence_number = 0U,
      .phase = ExecutionPhase::kPrefill,
      .first_position = 0U,
      .sequence_lengths = {4U},
      .cache_slot_ids = {0U},
      .shape = {1U, 4U, 8U},
      .dtype = DataType::kF16,
      .layout = TensorLayout::kDenseRowMajorLittleEndian,
      .payload_length = 64U,
  };
}

constexpr TensorEnvelopeLimits kLimits{
    .protocol_version = 1U,
    .maximum_batch = 1U,
    .maximum_sequence_length = 16U,
    .hidden_size = 8U,
    .maximum_payload_bytes = 256U,
    .boundary_dtype = DataType::kF16,
};

TEST(TensorEnvelopeTest, AcceptsTheMvpBoundaryRepresentation) {
  EXPECT_NO_THROW(validate_tensor_envelope(valid_envelope(), 64U, kLimits));
}

TEST(TensorEnvelopeTest, RejectsPayloadShapeMismatchBeforeAllocation) {
  auto envelope = valid_envelope();
  envelope.payload_length = 63U;
  EXPECT_THROW(validate_tensor_envelope(envelope, 63U, kLimits), EnvelopeError);
}

TEST(TensorEnvelopeTest, RejectsDecodeWithMultiplePositions) {
  auto envelope = valid_envelope();
  envelope.phase = ExecutionPhase::kDecode;
  EXPECT_THROW(validate_tensor_envelope(envelope, 64U, kLimits), EnvelopeError);
}

TEST(TensorEnvelopeTest, RejectsSequenceBeyondCapacityWithoutUnsignedUnderflow) {
  auto envelope = valid_envelope();
  envelope.first_position = 20U;
  EXPECT_THROW(validate_tensor_envelope(envelope, 64U, kLimits), EnvelopeError);
}

TEST(SequenceTrackerTest, EnforcesStrictMonotonicOrdering) {
  SequenceTracker tracker;
  tracker.accept(0U);
  tracker.accept(1U);
  EXPECT_EQ(tracker.expected(), 2U);
  EXPECT_THROW(tracker.accept(3U), EnvelopeError);
  EXPECT_EQ(tracker.expected(), 2U);
}

}  // namespace
}  // namespace hllm::runtime
