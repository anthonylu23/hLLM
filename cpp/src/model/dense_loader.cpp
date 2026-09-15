#include "hllm/model/dense_loader.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"

namespace hllm::model {
namespace {
constexpr auto multiply = runtime::checked_multiply;
constexpr auto add = runtime::checked_add;
struct ExpectedTensor {
  std::vector<std::size_t> shape;
  v1::TensorRole role;
  int layer{-1};
};

}  // namespace

DenseSource inspect_dense_stage(const v1::LoadStageRequest& request,
                                const std::filesystem::path& root) {
  if (request.stage_index() >= static_cast<std::uint32_t>(request.plan().stages_size()) ||
      (request.plan().execution_dtype() != v1::DATA_TYPE_F32 &&
       request.plan().execution_dtype() != v1::DATA_TYPE_F16)) {
    throw runtime::Error::incompatible_worker("invalid stage index or unsupported execution dtype");
  }
  const auto& plan = request.plan();
  const bool mixed = plan.has_weight_dtype();
  if (mixed != (plan.schema_version().major() == 1U && plan.schema_version().minor() == 2U) ||
      (mixed && (plan.weight_dtype() != v1::DATA_TYPE_F16 ||
                 plan.execution_dtype() != v1::DATA_TYPE_F32))) {
    throw runtime::Error::incompatible_worker("unsupported resident/execution precision contract");
  }
  const auto& manifest = request.manifest();
  const auto& descriptor = manifest.architecture();
  // Explicit architecture registry; unrelated families need their own
  // implementation.
  const std::map<std::string, bool> architectures{{"llama.v1", false}, {"qwen3.v1", true}};
  const auto architecture = architectures.find(descriptor.architecture_id());
  if (architecture == architectures.end() || descriptor.architecture_revision() != "1") {
    throw runtime::Error::incompatible_worker("unsupported architecture ID or revision");
  }
  const bool qwen = architecture->second;
  const std::set<std::string> supported{"gqa", "mha", "tied_embeddings", "untied_embeddings",
                                        "explicit_head_dim"};
  bool qk_flag = false;
  for (const auto& feature : descriptor.feature_flags()) {
    if (qwen && feature == "qk_norm") {
      qk_flag = true;
    } else if (!supported.contains(feature)) {
      throw runtime::Error::incompatible_worker("unsupported architecture feature: " + feature);
    }
  }
  if (qwen && !qk_flag) {
    throw runtime::Error::incompatible_worker("Qwen3 requires qk_norm feature");
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
    throw runtime::Error::incompatible_worker("unsupported dense model configuration");
  }
  const auto& assignment = request.plan().stages(static_cast<int>(request.stage_index()));
  const auto layer_count = assignment.layer_end() - assignment.layer_start();
  const auto tensors_per_layer = qwen ? 11U : 9U;
  if (assignment.layer_start() >= assignment.layer_end() ||
      assignment.layer_end() > cfg.num_layers() ||
      layer_count > static_cast<std::uint32_t>(manifest.tensors_size()) / tensors_per_layer) {
    throw runtime::Error::incompatible_worker("layer range exceeds available tensor records");
  }
  DenseSource source;
  source.config = {cfg.hidden_size(),
                   cfg.intermediate_size(),
                   cfg.num_attention_heads(),
                   cfg.num_kv_heads(),
                   cfg.head_dim(),
                   cfg.maximum_sequence_length(),
                   static_cast<float>(cfg.rms_norm_eps()),
                   static_cast<float>(cfg.rope_theta()),
                   qwen};
  source.vocabulary_size = cfg.vocabulary_size();
  source.first = assignment.owns_token_embedding();
  source.final = assignment.owns_sampling();
  source.layer_start = assignment.layer_start();
  source.layer_end = assignment.layer_end();
  source.tied_head = assignment.owns_lm_head() && cfg.tied_embeddings();
  source.execution_dtype = request.plan().execution_dtype() == v1::DATA_TYPE_F16
                               ? runtime::DataType::kF16
                               : runtime::DataType::kF32;
  source.weight_dtype = mixed ? runtime::DataType::kF16 : source.execution_dtype;
  const bool first = request.stage_index() == 0U;
  const bool final =
      request.stage_index() + 1U == static_cast<std::uint32_t>(request.plan().stages_size());
  if (assignment.owns_token_embedding() != first || assignment.owns_sampling() != final ||
      assignment.owns_final_norm() != final || assignment.owns_lm_head() != final) {
    throw runtime::Error::incompatible_worker("inconsistent stage tensor ownership");
  }
  const auto h = source.config.hidden_size;
  const auto attention = multiply(cfg.num_attention_heads(), cfg.head_dim());
  const auto kv = multiply(cfg.num_kv_heads(), cfg.head_dim());
  std::map<std::string, ExpectedTensor> expected;
  if (assignment.owns_token_embedding() || (assignment.owns_lm_head() && cfg.tied_embeddings())) {
    expected.emplace("model.embed_tokens.weight",
                     ExpectedTensor{{source.vocabulary_size, h}, v1::TENSOR_ROLE_TOKEN_EMBEDDING});
  }
  if (assignment.owns_final_norm()) {
    expected.emplace("model.norm.weight", ExpectedTensor{{h}, v1::TENSOR_ROLE_FINAL_NORM});
  }
  if (assignment.owns_lm_head() && !cfg.tied_embeddings()) {
    expected.emplace("lm_head.weight",
                     ExpectedTensor{{source.vocabulary_size, h}, v1::TENSOR_ROLE_LM_HEAD});
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
      throw runtime::Error::incompatible_worker("duplicate tensor in manifest");
    }
  }
  const bool redundant_tied_head = source.tied_head && records.contains("lm_head.weight");
  if (redundant_tied_head) {
    expected.emplace("lm_head.weight",
                     ExpectedTensor{{source.vocabulary_size, h}, v1::TENSOR_ROLE_LM_HEAD});
  }
  for (const auto& [name, record] : records) {
    // The embedding is authoritative for a tied head. An optional serialized
    // head is validated separately below, without duplicate resident storage.
    const bool selected =
        (record->role() == v1::TENSOR_ROLE_TRANSFORMER_LAYER && record->has_layer_index() &&
         record->layer_index() >= assignment.layer_start() &&
         record->layer_index() < assignment.layer_end()) ||
        (record->role() == v1::TENSOR_ROLE_TOKEN_EMBEDDING && assignment.owns_token_embedding()) ||
        (record->role() == v1::TENSOR_ROLE_FINAL_NORM && assignment.owns_final_norm()) ||
        (record->role() == v1::TENSOR_ROLE_LM_HEAD && assignment.owns_lm_head() &&
         !cfg.tied_embeddings());
    if (selected && !expected.contains(name)) {
      throw runtime::Error::incompatible_worker("unsupported selected tensor: " + name);
    }
  }
  auto& files = source.files;

  std::map<int, std::size_t> cast_groups;
  for (const auto& [name, spec] : expected) {
    const auto found = records.find(name);
    if (found == records.end()) {
      throw runtime::Error::incompatible_worker("missing required tensor: " + name);
    }
    const auto& record = *found->second;
    if (record.role() != spec.role ||
        (spec.layer >= 0 && (!record.has_layer_index() ||
                             record.layer_index() != static_cast<std::uint32_t>(spec.layer))) ||
        (spec.layer < 0 && record.has_layer_index()) ||
        std::vector<std::size_t>(record.shape().begin(), record.shape().end()) != spec.shape) {
      throw runtime::Error::incompatible_worker("incorrect tensor shape or ownership: " + name);
    }
    const std::filesystem::path relative(record.file());
    if (relative.empty() || relative.is_absolute() || relative.has_parent_path() ||
        relative == "." || relative == ".." ||
        std::filesystem::canonical(root / relative).parent_path() !=
            std::filesystem::canonical(root)) {
      throw runtime::Error::incompatible_worker("tensor path escapes model root");
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
      throw runtime::Error::incompatible_worker(
          "Safetensors metadata mismatch or unsupported dtype: " + name);
    }
    if (redundant_tied_head && name == "lm_head.weight") {
      continue;  // Payload is verified when the admitted loader reads the embedding.
    }
    std::size_t elements = 1U;
    for (const auto dim : spec.shape) {
      elements = multiply(elements, dim);
    }
    source.float32_weight_bytes =
        add(source.float32_weight_bytes, multiply(elements, sizeof(float)));
    source.largest_payload_bytes = std::max(source.largest_payload_bytes, tensor.byte_length);
    source.largest_float32_tensor_bytes =
        std::max(source.largest_float32_tensor_bytes, multiply(elements, sizeof(float)));
    source.tensors.emplace(name, TensorSource{record.file(), spec.shape, tensor.dtype});
    if (mixed && (spec.layer >= 0 || source.final)) {
      // Non-layer tensors form the head group; embedding lookup itself is F16.
      cast_groups[spec.layer] = add(cast_groups[spec.layer], multiply(elements, sizeof(float)));
      source.weight_cast_workspace_bytes =
          std::max(source.weight_cast_workspace_bytes, cast_groups[spec.layer]);
    }
  }
  if (redundant_tied_head) {
    const auto& embedding = files.at(records.at("model.embed_tokens.weight")->file())
                                .tensor("model.embed_tokens.weight");
    const auto& head_record = *records.at("lm_head.weight");
    const auto& head = files.at(head_record.file()).tensor("lm_head.weight");
    if (embedding.dtype != head.dtype || embedding.shape != head.shape ||
        embedding.byte_length != head.byte_length) {
      throw runtime::Error::incompatible_worker(
          "redundant tied head must match embedding storage metadata");
    }
    source.redundant_head_file = head_record.file();
    source.verification_workspace_bytes = std::min(head.byte_length, std::size_t{1024U * 1024U});
  }
  return source;
}
}  // namespace hllm::model
