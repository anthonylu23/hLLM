#include "hllm/cpu/stage.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "hllm/cpu/llama.hpp"
#include "hllm/runtime/half.hpp"
#include "hllm/runtime/safetensors.hpp"

namespace hllm::cpu {
namespace {

std::size_t multiply(std::size_t a, std::size_t b) {
  if (b != 0U && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::length_error("stage memory size overflow");
  }
  return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    throw std::length_error("stage memory size overflow");
  }
  return a + b;
}

struct CpuSequence final : runtime::SequenceState {
  std::vector<LayerKvCache> caches;
};

class DenseStage final : public runtime::StageBackend {
 public:
  LlamaConfig config{};
  std::size_t vocab{};
  std::size_t bytes{};
  std::vector<LayerWeights> layers;
  Matrix embedding{0U, 0U};
  Matrix head{0U, 0U};
  bool tied_head{false};
  std::vector<float> norm;

  std::size_t weight_bytes() const override { return bytes; }
  std::size_t hidden_size() const override { return config.hidden_size; }
  std::size_t vocabulary_size() const override { return vocab; }
  std::size_t maximum_tokens() const override { return config.maximum_sequence_length; }
  runtime::SequenceMemory sequence_memory(std::size_t tokens) const override {
    if (tokens == 0U || tokens > maximum_tokens()) {
      throw std::invalid_argument("request exceeds model context capacity");
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
    return {cache, workspace};
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
  runtime::HostActivation embed(std::span<const std::uint64_t> tokens) const override {
    if (embedding.rows() == 0U || tokens.empty() || tokens.size() > maximum_tokens()) {
      throw std::invalid_argument("stage cannot embed this input");
    }
    runtime::HostActivation output{tokens.size(), hidden_size(),
                                   std::vector<float>(multiply(tokens.size(), hidden_size()))};
    for (std::size_t row = 0U; row < tokens.size(); ++row) {
      if (tokens[row] >= vocab) {
        throw std::invalid_argument("token ID exceeds vocabulary");
      }
      for (std::size_t col = 0U; col < hidden_size(); ++col) {
        output.values[row * hidden_size() + col] = embedding(tokens[row], col);
      }
    }
    return output;
  }
  runtime::HostActivation forward(runtime::HostActivation input, std::size_t position,
                                  runtime::SequenceState& opaque,
                                  const std::atomic_bool& cancelled) const override {
    auto& state = dynamic_cast<CpuSequence&>(opaque);
    Matrix hidden(input.tokens, input.width, std::move(input.values));
    for (std::size_t i = 0U; i < layers.size(); ++i) {
      if (cancelled.load()) {
        throw std::runtime_error("request cancelled");
      }
      hidden = transformer_layer(hidden, layers[i], config, position, state.caches.at(i));
    }
    return {hidden.rows(), hidden.columns(),
            std::vector<float>(hidden.values().begin(), hidden.values().end())};
  }
  std::uint64_t sample(const runtime::HostActivation& hidden) const override {
    if (norm.empty() || hidden.tokens == 0U || hidden.width != hidden_size() ||
        hidden.values.size() != multiply(hidden.tokens, hidden.width)) {
      throw std::invalid_argument("stage cannot sample this activation");
    }
    Matrix last(1U, hidden.width);
    std::copy_n(hidden.values.end() - static_cast<std::ptrdiff_t>(hidden.width), hidden.width,
                last.values().begin());
    return greedy_sample_last(
        final_logits(last, norm, tied_head ? embedding : head, config.rms_norm_epsilon));
  }
};

struct ExpectedTensor {
  std::vector<std::size_t> shape;
  v1::TensorRole role;
  int layer{-1};
};

}  // namespace

std::unique_ptr<runtime::StageBackend> load_stage(const v1::LoadStageRequest& request,
                                                  const std::filesystem::path& root,
                                                  std::size_t memory_limit) {
  const auto& manifest = request.manifest();
  const auto& descriptor = manifest.architecture();
  // Explicit architecture registry; unrelated families need their own
  // implementation.
  const std::map<std::string, bool> architectures{{"llama.v1", false}, {"qwen3.v1", true}};
  const auto architecture = architectures.find(descriptor.architecture_id());
  if (architecture == architectures.end() || descriptor.architecture_revision() != "1") {
    throw std::invalid_argument("unsupported architecture ID or revision");
  }
  const bool qwen = architecture->second;
  const std::set<std::string> supported{"gqa", "mha", "tied_embeddings", "untied_embeddings",
                                        "explicit_head_dim"};
  bool qk_flag = false;
  for (const auto& feature : descriptor.feature_flags()) {
    if (qwen && feature == "qk_norm") {
      qk_flag = true;
    } else if (!supported.contains(feature)) {
      throw std::invalid_argument("unsupported architecture feature: " + feature);
    }
  }
  if (qwen && !qk_flag) {
    throw std::invalid_argument("Qwen3 requires qk_norm feature");
  }
  const auto& cfg = manifest.config();
  if (cfg.hidden_size() == 0U || cfg.intermediate_size() == 0U || cfg.num_layers() == 0U ||
      cfg.num_attention_heads() == 0U || cfg.num_kv_heads() == 0U || cfg.head_dim() == 0U ||
      cfg.head_dim() % 2U != 0U || cfg.num_attention_heads() % cfg.num_kv_heads() != 0U ||
      cfg.maximum_sequence_length() == 0U || cfg.vocabulary_size() == 0U ||
      cfg.hidden_activation() != "silu" || cfg.attention_bias() || cfg.mlp_bias() ||
      cfg.has_rope_scaling() || !std::isfinite(cfg.rms_norm_eps()) || cfg.rms_norm_eps() <= 0 ||
      !std::isfinite(cfg.rope_theta()) || cfg.rope_theta() <= 0 ||
      !std::isfinite(static_cast<float>(cfg.rope_theta())) ||
      !std::isfinite(static_cast<float>(cfg.rms_norm_eps())) ||
      static_cast<float>(cfg.rms_norm_eps()) <= 0) {
    throw std::invalid_argument("unsupported dense CPU model configuration");
  }
  const auto& assignment = request.plan().stages(static_cast<int>(request.stage_index()));
  const auto layer_count = assignment.layer_end() - assignment.layer_start();
  const auto tensors_per_layer = qwen ? 11U : 9U;
  if (assignment.layer_start() >= assignment.layer_end() ||
      assignment.layer_end() > cfg.num_layers() ||
      layer_count > static_cast<std::uint32_t>(manifest.tensors_size()) / tensors_per_layer) {
    throw std::invalid_argument("layer range exceeds available tensor records");
  }
  auto stage = std::make_unique<DenseStage>();
  stage->config = {cfg.hidden_size(),
                   cfg.intermediate_size(),
                   cfg.num_attention_heads(),
                   cfg.num_kv_heads(),
                   cfg.head_dim(),
                   cfg.maximum_sequence_length(),
                   static_cast<float>(cfg.rms_norm_eps()),
                   static_cast<float>(cfg.rope_theta()),
                   qwen};
  stage->vocab = cfg.vocabulary_size();
  const auto h = stage->hidden_size();
  const auto attention = multiply(cfg.num_attention_heads(), cfg.head_dim());
  const auto kv = multiply(cfg.num_kv_heads(), cfg.head_dim());
  std::map<std::string, ExpectedTensor> expected;
  if (assignment.owns_token_embedding() || (assignment.owns_lm_head() && cfg.tied_embeddings())) {
    expected.emplace("model.embed_tokens.weight",
                     ExpectedTensor{{stage->vocab, h}, v1::TENSOR_ROLE_TOKEN_EMBEDDING});
  }
  if (assignment.owns_final_norm()) {
    expected.emplace("model.norm.weight", ExpectedTensor{{h}, v1::TENSOR_ROLE_FINAL_NORM});
  }
  if (assignment.owns_lm_head() && !cfg.tied_embeddings()) {
    expected.emplace("lm_head.weight", ExpectedTensor{{stage->vocab, h}, v1::TENSOR_ROLE_LM_HEAD});
  }
  for (auto i = assignment.layer_start(); i < assignment.layer_end(); ++i) {
    const auto prefix = "model.layers." + std::to_string(i) + ".";
    const std::map<std::string, std::vector<std::size_t>> shapes{
        {"input_layernorm.weight", {h}},
        {"post_attention_layernorm.weight", {h}},
        {"self_attn.q_proj.weight", {attention, h}},
        {"self_attn.k_proj.weight", {kv, h}},
        {"self_attn.v_proj.weight", {kv, h}},
        {"self_attn.o_proj.weight", {h, attention}},
        {"mlp.gate_proj.weight", {cfg.intermediate_size(), h}},
        {"mlp.up_proj.weight", {cfg.intermediate_size(), h}},
        {"mlp.down_proj.weight", {h, cfg.intermediate_size()}}};
    for (const auto& [name, shape] : shapes) {
      expected.emplace(prefix + name, ExpectedTensor{shape, v1::TENSOR_ROLE_TRANSFORMER_LAYER,
                                                     static_cast<int>(i)});
    }
    if (qwen) {
      for (const auto* suffix : {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}) {
        expected.emplace(prefix + suffix, ExpectedTensor{{cfg.head_dim()},
                                                         v1::TENSOR_ROLE_TRANSFORMER_LAYER,
                                                         static_cast<int>(i)});
      }
    }
  }
  std::map<std::string, const v1::TensorRecord*> records;
  for (const auto& tensor : manifest.tensors()) {
    if (!records.emplace(tensor.name(), &tensor).second) {
      throw std::invalid_argument("duplicate tensor in manifest");
    }
  }
  for (const auto& [name, record] : records) {
    const bool selected =
        (record->role() == v1::TENSOR_ROLE_TRANSFORMER_LAYER && record->has_layer_index() &&
         record->layer_index() >= assignment.layer_start() &&
         record->layer_index() < assignment.layer_end()) ||
        (record->role() == v1::TENSOR_ROLE_TOKEN_EMBEDDING && assignment.owns_token_embedding()) ||
        (record->role() == v1::TENSOR_ROLE_FINAL_NORM && assignment.owns_final_norm()) ||
        (record->role() == v1::TENSOR_ROLE_LM_HEAD && assignment.owns_lm_head());
    if (selected && !expected.contains(name)) {
      throw std::invalid_argument("unsupported selected tensor: " + name);
    }
  }
  std::map<std::string, runtime::SafetensorsFile> files;
  std::size_t largest_payload = 0U;
  for (const auto& [name, spec] : expected) {
    const auto found = records.find(name);
    if (found == records.end()) {
      throw std::invalid_argument("missing required tensor: " + name);
    }
    const auto& record = *found->second;
    if (record.role() != spec.role ||
        (spec.layer >= 0 && (!record.has_layer_index() ||
                             record.layer_index() != static_cast<std::uint32_t>(spec.layer))) ||
        (spec.layer < 0 && record.has_layer_index()) ||
        std::vector<std::size_t>(record.shape().begin(), record.shape().end()) != spec.shape) {
      throw std::invalid_argument("incorrect tensor shape or ownership: " + name);
    }
    const std::filesystem::path relative(record.file());
    if (relative.empty() || relative.is_absolute() || relative.has_parent_path() ||
        relative == "." || relative == ".." ||
        std::filesystem::canonical(root / relative).parent_path() !=
            std::filesystem::canonical(root)) {
      throw std::invalid_argument("tensor path escapes model root");
    }
    auto file = files.find(record.file());
    if (file == files.end()) {
      file = files.emplace(record.file(), runtime::SafetensorsFile(root / relative)).first;
    }
    const auto& tensor = file->second.tensor(name);
    const auto dtype = tensor.dtype == runtime::DataType::kF32    ? v1::DATA_TYPE_F32
                       : tensor.dtype == runtime::DataType::kF16  ? v1::DATA_TYPE_F16
                       : tensor.dtype == runtime::DataType::kBF16 ? v1::DATA_TYPE_BF16
                                                                  : v1::DATA_TYPE_UNSPECIFIED;
    if (dtype == v1::DATA_TYPE_UNSPECIFIED || record.dtype() != dtype ||
        tensor.shape != spec.shape || tensor.data_offset != record.data_offset() ||
        tensor.byte_length != record.byte_length()) {
      throw std::invalid_argument("Safetensors metadata mismatch or unsupported dtype: " + name);
    }
    std::size_t elements = 1U;
    for (const auto dim : spec.shape) {
      elements = multiply(elements, dim);
    }
    stage->bytes = add(stage->bytes, multiply(elements, sizeof(float)));
    largest_payload = std::max(largest_payload, tensor.byte_length);
  }
  // Payload conversion holds one source buffer alongside the resident float32
  // weights.
  if (add(add(stage->bytes, largest_payload), 65536U) > memory_limit) {
    throw std::length_error("stage loading exceeds host memory budget");
  }
  runtime::BufferPool pool(0U);
  auto vector = [&](const std::string& name) {
    const auto& record = *records.at(name);
    auto buffer = files.at(record.file()).read_tensor(name, pool);
    const auto& shape = expected.at(name).shape;
    const auto elements = shape.size() == 1U ? shape[0] : multiply(shape[0], shape[1]);
    std::vector<float> values(elements);
    const auto data = buffer.bytes();
    for (std::size_t i = 0U; i < elements; ++i) {
      const auto offset = i * (record.dtype() == v1::DATA_TYPE_F32 ? 4U : 2U);
      std::uint32_t bits = 0U;
      for (std::size_t b = 0U; b < (record.dtype() == v1::DATA_TYPE_F32 ? 4U : 2U); ++b) {
        bits |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[offset + b]))
                << (8U * b);
      }
      values[i] = record.dtype() == v1::DATA_TYPE_F32 ? std::bit_cast<float>(bits)
                  : record.dtype() == v1::DATA_TYPE_F16
                      ? runtime::float16_to_float(static_cast<std::uint16_t>(bits))
                      : runtime::bfloat16_to_float(static_cast<std::uint16_t>(bits));
      if (!std::isfinite(values[i])) {
        throw std::invalid_argument("non-finite weight: " + name);
      }
    }
    return values;
  };
  auto matrix = [&](const std::string& name) {
    const auto& shape = expected.at(name).shape;
    return Matrix(shape[0], shape[1], vector(name));
  };
  if (expected.contains("model.embed_tokens.weight")) {
    stage->embedding = matrix("model.embed_tokens.weight");
  }
  if (assignment.owns_final_norm()) {
    stage->norm = vector("model.norm.weight");
  }
  stage->tied_head = assignment.owns_lm_head() && cfg.tied_embeddings();
  if (expected.contains("lm_head.weight")) {
    stage->head = matrix("lm_head.weight");
  }
  for (auto i = assignment.layer_start(); i < assignment.layer_end(); ++i) {
    const auto p = "model.layers." + std::to_string(i) + ".";
    stage->layers.push_back(
        {vector(p + "input_layernorm.weight"), matrix(p + "self_attn.q_proj.weight"),
         matrix(p + "self_attn.k_proj.weight"), matrix(p + "self_attn.v_proj.weight"),
         matrix(p + "self_attn.o_proj.weight"), vector(p + "post_attention_layernorm.weight"),
         matrix(p + "mlp.gate_proj.weight"), matrix(p + "mlp.up_proj.weight"),
         matrix(p + "mlp.down_proj.weight"),
         qwen ? vector(p + "self_attn.q_norm.weight") : std::vector<float>{},
         qwen ? vector(p + "self_attn.k_norm.weight") : std::vector<float>{}});
  }
  return stage;
}

}  // namespace hllm::cpu
