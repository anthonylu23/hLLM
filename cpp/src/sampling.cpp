#include "hllm/runtime/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "hllm/runtime/error.hpp"

namespace hllm::runtime {
void validate_sampling(const SamplingOptions& options) {
  if (!std::isfinite(options.temperature) || options.temperature < 0.0 ||
      options.temperature > 100.0 || !std::isfinite(options.top_p) || options.top_p <= 0.0 ||
      options.top_p > 1.0 || options.top_logprobs > 5U ||
      (!options.return_logprobs && options.top_logprobs != 0U)) {
    throw Error::invalid_request("invalid sampling parameters");
  }
}

std::uint64_t sample_logits(std::span<const float> logits, const SamplingOptions& options,
                            std::uint64_t generated_index) {
  validate_sampling(options);
  if (logits.empty()) throw Error::internal("cannot sample empty logits");
  for (const auto value : logits) {
    if (!std::isfinite(value)) throw Error::internal("cannot sample nonfinite logits");
  }
  const auto best = std::max_element(logits.begin(), logits.end());
  if (options.temperature == 0.0) return static_cast<std::uint64_t>(best - logits.begin());
  struct Candidate {
    double probability;
    std::uint64_t token;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    candidates.push_back(
        {std::exp((static_cast<double>(logits[i]) - *best) / options.temperature), i});
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.probability == b.probability ? a.token < b.token : a.probability > b.probability;
  });
  if (options.top_k != 0 && options.top_k < candidates.size()) {
    candidates.resize(static_cast<std::size_t>(options.top_k));
  }
  double total = 0.0;
  for (const auto& candidate : candidates) total += candidate.probability;
  const auto threshold = total * options.top_p;
  double retained = 0.0;
  std::size_t count = 0;
  do {
    retained += candidates[count++].probability;
  } while (retained < threshold && count < candidates.size());
  // SplitMix64: fixed integer operations and a 53-bit [0,1) double. No standard
  // library distribution or shared engine whose state depends on interleaving.
  auto bits = options.seed + 0x9e3779b97f4a7c15ULL * (generated_index + 1U);
  bits = (bits ^ (bits >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  bits = (bits ^ (bits >> 27U)) * 0x94d049bb133111ebULL;
  bits ^= bits >> 31U;
  auto target = static_cast<double>(bits >> 11U) * 0x1.0p-53 * retained;
  for (std::size_t i = 0; i + 1U < count; ++i) {
    target -= candidates[i].probability;
    if (target < 0.0) return candidates[i].token;
  }
  return candidates[count - 1U].token;
}
SampledToken sample_token(std::span<const float> logits, const SamplingOptions& options,
                         std::uint64_t generated_index) {
  SampledToken result{sample_logits(logits, options, generated_index)};
  if (!options.return_logprobs) return result;
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (const auto value : logits) sum += std::exp(static_cast<double>(value) - maximum);
  const auto log_sum = std::log(sum);
  result.logprob = static_cast<double>(logits[result.id]) - maximum - log_sum;
  std::vector<TokenLogProbability> candidates;
  if (options.top_logprobs) {
    candidates.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
      candidates.push_back({i, static_cast<double>(logits[i]) - maximum - log_sum});
    }
    const auto count = std::min(candidates.size(), static_cast<std::size_t>(options.top_logprobs));
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(count),
                      candidates.end(), [](const auto& a, const auto& b) {
      return a.logprob == b.logprob ? a.token_id < b.token_id : a.logprob > b.logprob;
    });
    candidates.resize(count);
    result.top_logprobs = std::move(candidates);
  }
  return result;
}
}  // namespace hllm::runtime
