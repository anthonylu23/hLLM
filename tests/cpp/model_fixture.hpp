#pragma once

#include <atomic>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "control.pb.h"
#include "hllm/runtime/half.hpp"

namespace hllm::test {

class ModelFixture {
 public:
  std::filesystem::path root;
  v1::ModelManifest manifest;
  nlohmann::json oracle;

  explicit ModelFixture(bool qwen = true, bool tied = true, const std::string& dtype = "F32",
                        bool redundant_tied_head = false, const std::string& head_dtype = "") {
    static std::atomic<unsigned> counter{0U};
    root = std::filesystem::temp_directory_path() /
           ("hllm-stage-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(counter++));
    std::filesystem::create_directory(root);
    std::ifstream input(std::string(HLLM_TEST_DATA_DIR) + "/qwen3/tiny-reference.json");
    oracle = nlohmann::json::parse(input);
    manifest.mutable_schema_version()->set_major(1U);
    manifest.mutable_schema_version()->set_minor(1U);
    manifest.set_manifest_digest("fixture-manifest");
    auto* architecture = manifest.mutable_architecture();
    architecture->set_architecture_id(qwen ? "qwen3.v1" : "llama.v1");
    architecture->set_architecture_revision("1");
    if (qwen) {
      architecture->add_feature_flags("qk_norm");
    }
    const auto& cfg = oracle.at("config");
    auto* config = manifest.mutable_config();
    config->set_hidden_size(cfg.at("hidden_size").get<std::uint64_t>());
    config->set_intermediate_size(cfg.at("intermediate_size").get<std::uint64_t>());
    config->set_num_layers(cfg.at("num_hidden_layers").get<std::uint32_t>());
    config->set_num_attention_heads(cfg.at("num_attention_heads").get<std::uint32_t>());
    config->set_num_kv_heads(cfg.at("num_key_value_heads").get<std::uint32_t>());
    config->set_head_dim(cfg.at("head_dim").get<std::uint32_t>());
    config->set_vocabulary_size(cfg.at("vocab_size").get<std::uint64_t>());
    config->set_maximum_sequence_length(cfg.at("max_position_embeddings").get<std::uint64_t>());
    config->set_rms_norm_eps(cfg.at("rms_norm_eps").get<double>());
    config->set_rope_theta(cfg.at("rope_theta").get<double>());
    config->set_hidden_activation("silu");
    config->set_tied_embeddings(tied);
    nlohmann::json header = nlohmann::json::object();
    std::string payload;
    if (redundant_tied_head) {
      oracle["weights"]["lm_head.weight"] = oracle["weights"]["model.embed_tokens.weight"];
    }
    for (const auto& [name, tensor] : oracle.at("weights").items()) {
      if ((tied && name == "lm_head.weight" && !redundant_tied_head) ||
          (!qwen && (name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight")))) {
        continue;
      }
      const auto& storage = name == "lm_head.weight" && !head_dtype.empty() ? head_dtype : dtype;
      const auto start = payload.size();
      for (const float value : tensor.at("values").get<std::vector<float>>()) {
        const auto bits =
            storage == "F16"    ? static_cast<std::uint32_t>(runtime::float_to_float16(value))
            : storage == "BF16" ? static_cast<std::uint32_t>(runtime::float_to_bfloat16(value))
                              : std::bit_cast<std::uint32_t>(value);
        for (std::size_t b = 0U; b < (storage == "F32" ? 4U : 2U); ++b) {
          payload.push_back(static_cast<char>((bits >> (8U * b)) & 0xffU));
        }
      }
      header[name] = {{"dtype", storage},
                      {"shape", tensor.at("shape")},
                      {"data_offsets", {start, payload.size()}}};
      auto* record = manifest.add_tensors();
      record->set_name(name);
      record->set_file("model.safetensors");
      record->set_dtype(storage == "F16"    ? v1::DATA_TYPE_F16
                        : storage == "BF16" ? v1::DATA_TYPE_BF16
                                          : v1::DATA_TYPE_F32);
      for (auto dim : tensor.at("shape")) {
        record->add_shape(dim.get<std::uint64_t>());
      }
      record->set_data_offset(start);
      record->set_byte_length(payload.size() - start);
      if (name == "model.embed_tokens.weight") {
        record->set_role(v1::TENSOR_ROLE_TOKEN_EMBEDDING);
      } else if (name == "model.norm.weight") {
        record->set_role(v1::TENSOR_ROLE_FINAL_NORM);
      } else if (name == "lm_head.weight") {
        record->set_role(v1::TENSOR_ROLE_LM_HEAD);
      } else {
        record->set_role(v1::TENSOR_ROLE_TRANSFORMER_LAYER);
        record->set_layer_index(static_cast<std::uint32_t>(std::stoul(name.substr(13U))));
      }
    }
    auto encoded = header.dump();
    encoded.append((8U - encoded.size() % 8U) % 8U, ' ');
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    const auto size = encoded.size();
    for (std::size_t b = 0U; b < 8U; ++b) {
      output.put(static_cast<char>((size >> (8U * b)) & 0xffU));
    }
    output << encoded << payload;
  }
  ~ModelFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }

  v1::LoadStageRequest load(std::uint32_t stage = 0U, bool split = true) const {
    v1::LoadStageRequest request;
    *request.mutable_manifest() = manifest;
    request.set_stage_index(stage);
    auto* plan = request.mutable_plan();
    plan->mutable_schema_version()->set_major(1U);
    plan->set_plan_id("plan-1");
    plan->set_plan_digest("plan-digest");
    plan->set_manifest_digest(manifest.manifest_digest());
    plan->set_deployment_version(1U);
    plan->set_execution_dtype(v1::DATA_TYPE_F32);
    plan->set_activation_dtype(v1::DATA_TYPE_F16);
    plan->set_split_layer(split ? 1U : 0U);
    if (split) {
      plan->add_duplicated_tensor_groups("token_embeddings");
    }
    for (std::uint32_t i = 0U; i < (split ? 2U : 1U); ++i) {
      auto* assignment = plan->add_stages();
      assignment->set_stage_index(i);
      assignment->set_worker_id(i == 0U ? "cpu-a" : "cpu-b");
      assignment->set_layer_start(i);
      assignment->set_layer_end(split ? i + 1U : 2U);
      assignment->set_owns_token_embedding(i == 0U);
      const bool last = !split || i == 1U;
      assignment->set_owns_final_norm(last);
      assignment->set_owns_lm_head(last);
      assignment->set_owns_sampling(last);
      auto* endpoint = request.add_stage_endpoints();
      endpoint->set_stage_index(i);
      endpoint->set_worker_id(assignment->worker_id());
      endpoint->set_endpoint(i == 0U ? "127.0.0.1:50051" : "127.0.0.1:50052");
    }
    return request;
  }
};

}  // namespace hllm::test
