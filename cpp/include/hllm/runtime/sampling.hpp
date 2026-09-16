#pragma once

#include <cstdint>
#include <span>
#include <optional>
#include <vector>

namespace hllm::runtime {

struct SamplingOptions {
  double temperature{0.0};
  double top_p{1.0};
  std::uint64_t top_k{0};  // Zero disables top-k filtering.
  std::uint64_t seed{0};
  bool return_logprobs{false};
  std::uint32_t top_logprobs{0};
};

struct TokenLogProbability {
  std::uint64_t token_id;
  double logprob;
};
struct SampledToken {
  std::uint64_t id{};
  std::optional<double> logprob{};
  std::vector<TokenLogProbability> top_logprobs{};
};
SampledToken sample_token(std::span<const float> logits, const SamplingOptions& options,
                         std::uint64_t generated_index);

void validate_sampling(const SamplingOptions& options);
// Counter-based randomness belongs to the request, independent of scheduling.
std::uint64_t sample_logits(std::span<const float> logits, const SamplingOptions& options,
                            std::uint64_t generated_index);

}  // namespace hllm::runtime
