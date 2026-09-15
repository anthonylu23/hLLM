#include "hllm/cpu/stage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "hllm/cpu/llama.hpp"
#include "hllm/model/dense_loader.hpp"
#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/half.hpp"

namespace hllm::cpu {
namespace {

constexpr auto multiply = runtime::checked_multiply;
constexpr auto add = runtime::checked_add;

struct CpuSequence final : runtime::SequenceState {
  std::vector<LayerKvCache> caches;
};

class DenseStage final : public ReferenceStage {
 public:
  LlamaConfig config{};
  std::size_t vocab{};
  std::size_t bytes{};
  std::size_t layer_start{};
  std::vector<LayerWeights> layers;
  Matrix embedding{0U, 0U};
  Matrix head{0U, 0U};
  bool tied_head{false};
  bool first_stage{false};
  bool final_stage{false};
  std::vector<float> norm;

  runtime::MemoryAmounts weight_memory() const override { return {.host_bytes = bytes}; }
  std::size_t hidden_size() const override { return config.hidden_size; }
  std::size_t vocabulary_size() const override { return vocab; }
  std::size_t maximum_tokens() const override { return config.maximum_sequence_length; }
  runtime::SequenceMemory sequence_memory(std::size_t tokens) const override {
    if (tokens == 0U || tokens > maximum_tokens()) {
      throw runtime::Error::invalid_request("request exceeds model context capacity");
    }
    const auto cache =
        multiply(multiply(multiply(multiply(tokens, layers.size()), config.key_value_heads),
                          config.head_dimension),
                 2U * sizeof(float));
    // Conservative upper bound for simultaneous dense temporaries, wire copies,
    // attention scores and final-row logits. No quadratic attention matrix is
    // stored.
    const auto width =
        add(add(multiply(config.hidden_size, 24U),
                multiply(multiply(config.attention_heads, config.head_dimension), 12U)),
            add(multiply(config.intermediate_size, 6U), 4U));
    const auto workspace =
        add(add(multiply(multiply(tokens, width), sizeof(float)), multiply(vocab, sizeof(float))),
            65536U);
    return {{.host_bytes = cache}, {.host_bytes = workspace}};
  }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t tokens) const override {
    static_cast<void>(sequence_memory(tokens));
    auto state = std::make_unique<CpuSequence>();
    state->caches.reserve(layers.size());
    for (std::size_t i = 0U; i < layers.size(); ++i) {
      state->caches.emplace_back(tokens, config.key_value_heads, config.head_dimension);
    }
    return state;
  }
  HostActivation embed(std::span<const std::uint64_t> tokens) const override {
    if (embedding.rows() == 0U || tokens.empty() || tokens.size() > maximum_tokens()) {
      throw runtime::Error::invalid_request("stage cannot embed this input");
    }
    HostActivation output{tokens.size(), hidden_size(),
                          std::vector<float>(multiply(tokens.size(), hidden_size()))};
    for (std::size_t row = 0U; row < tokens.size(); ++row) {
      if (tokens[row] >= vocab) {
        throw runtime::Error::invalid_request("token ID exceeds vocabulary");
      }
      for (std::size_t col = 0U; col < hidden_size(); ++col) {
        output.values[row * hidden_size() + col] = embedding(tokens[row], col);
      }
    }
    return output;
  }
  HostActivation forward(HostActivation input, std::size_t position, runtime::SequenceState& opaque,
                         const std::atomic_bool& cancelled) const override {
    return forward_profiled(std::move(input), position, opaque, cancelled, nullptr);
  }
  HostActivation forward_profiled(HostActivation input, std::size_t position,
      runtime::SequenceState& opaque, const std::atomic_bool& cancelled,
      runtime::ExecutionTiming* timing) const {
    auto& state = dynamic_cast<CpuSequence&>(opaque);
    Matrix hidden(input.tokens, input.width, std::move(input.values));
    runtime::PhaseTimer timer(timing, [] {});
    for (std::size_t i = 0U; i < layers.size(); ++i) {
      if (cancelled.load()) {
        throw std::runtime_error("request cancelled");
      }
      hidden = transformer_layer(hidden, layers[i], config, position, state.caches.at(i));
      timer.mark("layer", layer_start + i);
    }
    return {hidden.rows(), hidden.columns(),
            std::vector<float>(hidden.values().begin(), hidden.values().end())};
  }
  runtime::StageOutput execute(runtime::StageInput input, std::size_t position,
                               runtime::SequenceState& state,
                               const std::atomic_bool& cancelled) const override {
    return run(std::move(input), position, state, cancelled, nullptr);
  }
  runtime::StageOutput execute_profiled(runtime::StageInput input, std::size_t position,
      runtime::SequenceState& state, const std::atomic_bool& cancelled,
      runtime::ExecutionTiming& timing) const override {
    timing = {};
    return run(std::move(input), position, state, cancelled, &timing);
  }
  runtime::StageOutput run(runtime::StageInput input, std::size_t position,
      runtime::SequenceState& state, const std::atomic_bool& cancelled,
      runtime::ExecutionTiming* timing) const {
    runtime::PhaseTimer timer(timing, [] {});
    HostActivation hidden;
    if (const auto* tokens = std::get_if<runtime::TokenInput>(&input)) {
      if (!first_stage) {
        throw runtime::Error::invalid_request("only stage zero accepts tokens");
      }
      hidden = embed(tokens->ids);
    } else {
      if (first_stage) {
        throw runtime::Error::invalid_request("stage zero requires tokens");
      }
      const auto& boundary = std::get<runtime::BoundaryActivation>(input);
      const auto elements = multiply(boundary.tokens, boundary.width);
      if (boundary.width != hidden_size() || boundary.tokens == 0U ||
          boundary.payload.size() != multiply(elements, 2U)) {
        throw runtime::Error::invalid_request("invalid boundary activation");
      }
      hidden = {boundary.tokens, boundary.width, std::vector<float>(elements)};
      for (std::size_t i = 0U; i < elements; ++i) {
        const auto low = std::to_integer<unsigned char>(boundary.payload[2U * i]);
        const auto high = std::to_integer<unsigned char>(boundary.payload[2U * i + 1U]);
        hidden.values[i] =
            runtime::float16_to_float(static_cast<std::uint16_t>(low | (high << 8U)));
        if (!std::isfinite(hidden.values[i])) {
          throw runtime::Error::invalid_request("non-finite activation");
        }
      }
    }
    timer.mark(first_stage ? "embedding" : "from-wire");
    hidden = forward_profiled(std::move(hidden), position, state, cancelled, timing);
    timer.restart();
    if (final_stage) {
      if (!timing) return runtime::SampledToken{sample(hidden)};
      Matrix last(1U, hidden.width);
      std::copy_n(hidden.values.end() - static_cast<std::ptrdiff_t>(hidden.width), hidden.width,
                  last.values().begin());
      auto normalized = rms_norm(last, norm, config.rms_norm_epsilon);
      timer.mark("final_norm");
      auto logits = linear(normalized, tied_head ? embedding : head);
      timer.mark("lm_head");
      const auto token = greedy_sample_last(logits);
      timer.mark("sampling");
      return runtime::SampledToken{token};
    }
    runtime::BoundaryActivation output{hidden.tokens, hidden.width,
                                       std::vector<std::byte>(multiply(hidden.values.size(), 2U))};
    for (std::size_t i = 0U; i < hidden.values.size(); ++i) {
      const auto bits = runtime::float_to_float16(hidden.values[i]);
      if (!std::isfinite(runtime::float16_to_float(bits))) {
        throw runtime::Error::internal("activation is not representable as finite FP16");
      }
      output.payload[2U * i] = static_cast<std::byte>(bits & 0xffU);
      output.payload[2U * i + 1U] = static_cast<std::byte>(bits >> 8U);
    }
    timer.mark("to-wire");
    return output;
  }
  std::uint64_t sample(const HostActivation& hidden) const override {
    if (norm.empty() || hidden.tokens == 0U || hidden.width != hidden_size() ||
        hidden.values.size() != multiply(hidden.tokens, hidden.width)) {
      throw runtime::Error::invalid_request("stage cannot sample this activation");
    }
    Matrix last(1U, hidden.width);
    std::copy_n(hidden.values.end() - static_cast<std::ptrdiff_t>(hidden.width), hidden.width,
                last.values().begin());
    return greedy_sample_last(
        final_logits(last, norm, tied_head ? embedding : head, config.rms_norm_epsilon));
  }
};

}  // namespace

std::unique_ptr<ReferenceStage> load_stage(const v1::LoadStageRequest& request,
                                           const std::filesystem::path& root,
                                           std::size_t memory_limit) {
  auto source = model::inspect_dense_stage(request, root);
  if (source.execution_dtype != runtime::DataType::kF32 ||
      source.weight_dtype != runtime::DataType::kF32) {
    throw runtime::Error::incompatible_worker("CPU execution requires F32");
  }
  if (add(add(source.float32_weight_bytes, add(source.largest_payload_bytes, source.verification_workspace_bytes)), 65536U) > memory_limit) {
    throw runtime::Error::resource_exhausted("stage loading exceeds host memory budget");
  }
  auto stage = std::make_unique<DenseStage>();
  stage->config = source.config;
  stage->layer_start = source.layer_start;
  stage->vocab = source.vocabulary_size;
  stage->first_stage = source.first;
  stage->final_stage = source.final;
  stage->bytes = source.float32_weight_bytes;
  const auto vector = [&](const std::string& name) { return source.read_float32(name); };
  auto matrix = [&](const std::string& name) {
    const auto& shape = source.tensors.at(name).shape;
    return Matrix(shape[0], shape[1], vector(name));
  };
  if (source.tensors.contains("model.embed_tokens.weight")) {
    stage->embedding = matrix("model.embed_tokens.weight");
  }
  if (source.final) {
    stage->norm = vector("model.norm.weight");
  }
  stage->tied_head = source.tied_head;
  if (source.tensors.contains("lm_head.weight")) {
    stage->head = matrix("lm_head.weight");
  }
  for (auto i = source.layer_start; i < source.layer_end; ++i) {
    const auto p = "model.layers." + std::to_string(i) + ".";
    stage->layers.push_back(
        {vector(p + "input_layernorm.weight"), matrix(p + "self_attn.q_proj.weight"),
         matrix(p + "self_attn.k_proj.weight"), matrix(p + "self_attn.v_proj.weight"),
         matrix(p + "self_attn.o_proj.weight"), vector(p + "post_attention_layernorm.weight"),
         matrix(p + "mlp.gate_proj.weight"), matrix(p + "mlp.up_proj.weight"),
         matrix(p + "mlp.down_proj.weight"),
         source.config.query_key_norm ? vector(p + "self_attn.q_norm.weight")
                                      : std::vector<float>{},
         source.config.query_key_norm ? vector(p + "self_attn.k_norm.weight")
                                      : std::vector<float>{}});
  }
  return stage;
}

namespace {
class CpuFactory final : public runtime::BackendFactory {
 public:
  runtime::BackendCapabilities capabilities() const override {
    return {v1::BACKEND_CPU,
            v1::MEMORY_DOMAIN_HOST,
            {"llama.v1", "qwen3.v1"},
            {v1::DATA_TYPE_F32},
            "CPU reference execution ready"};
  }
  std::unique_ptr<runtime::StageBackend> load(
      const v1::LoadStageRequest& request, const std::filesystem::path& root,
      const runtime::MemoryAmounts& capacity) const override {
    return load_stage(request, root, capacity.host_bytes);
  }
};
}  // namespace

std::unique_ptr<runtime::BackendFactory> make_backend_factory() {
  return std::make_unique<CpuFactory>();
}

}  // namespace hllm::cpu
