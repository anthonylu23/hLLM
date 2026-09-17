#include <set>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::runtime {
StageInput combine_decode_inputs(const std::vector<DecodeBatchItem>& items) {
  if (items.empty() || items.size() > 64U)
    throw Error::invalid_request("invalid decode batch size");
  const bool tokens = std::holds_alternative<TokenInput>(items.front().input);
  std::set<SequenceState*> sequences;
  TokenInput ids;
  BoundaryActivation boundary{items.size(), 0U, {}};
  for (const auto& item : items) {
    if (!item.sequence || !item.cancelled || item.position == 0 || item.sequence->prefilling ||
        item.sequence->prefill_only || !sequences.insert(item.sequence).second ||
        tokens != std::holds_alternative<TokenInput>(item.input)) {
      throw Error::invalid_request("decode batch requires independent, ready decode sequences");
    }
    if (tokens) {
      const auto& input = std::get<TokenInput>(item.input);
      if (input.ids.size() != 1)
        throw Error::invalid_request("decode batch requires one token per sequence");
      ids.ids.push_back(input.ids.front());
    } else {
      const auto& input = std::get<BoundaryActivation>(item.input);
      if (input.tokens != 1 || input.width == 0 ||
          input.payload.size() != checked_multiply(input.width, 2U) ||
          (boundary.width && boundary.width != input.width)) {
        throw Error::invalid_request("invalid decode batch boundary");
      }
      boundary.width = input.width;
      boundary.payload.insert(boundary.payload.end(), input.payload.begin(), input.payload.end());
    }
  }
  if (tokens) return ids;
  return boundary;
}
std::vector<StageOutput> split_decode_boundary(BoundaryActivation output) {
  const auto bytes = checked_multiply(output.width, 2U);
  if (output.payload.size() != checked_multiply(output.tokens, bytes)) {
    throw Error::internal("invalid batched output boundary");
  }
  std::vector<StageOutput> results;
  for (std::size_t i = 0; i < output.tokens; ++i) {
    auto start = output.payload.begin() + static_cast<std::ptrdiff_t>(i * bytes);
    results.push_back(
        BoundaryActivation{1U, output.width, {start, start + static_cast<std::ptrdiff_t>(bytes)}});
  }
  return results;
}
}  // namespace hllm::runtime
