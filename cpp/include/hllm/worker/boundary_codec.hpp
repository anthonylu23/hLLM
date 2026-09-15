#pragma once
#include "execution.pb.h"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::worker {
inline constexpr int kMaximumRpcBytes = 16 * 1024 * 1024;
struct BoundaryIdentity {
  std::string deployment_id;
  std::uint64_t deployment_version;
  std::string request_id;
  std::size_t maximum_tokens;
  std::size_t hidden_size;
};
runtime::BoundaryActivation decode_tensor(const v1::TensorEnvelope&, const BoundaryIdentity&,
                                          std::uint64_t sequence, std::size_t position);
v1::StageMessage encode_tensor(const runtime::BoundaryActivation&, const BoundaryIdentity&,
                               std::uint64_t sequence, std::size_t position);
}  // namespace hllm::worker
