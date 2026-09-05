#include "hllm/cpu/llama.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace hllm::cpu {
namespace {

const nlohmann::json& reference() {
  static const auto fixture = [] {
    std::ifstream stream(std::string(HLLM_TEST_DATA_DIR) + "/qwen3/tiny-reference.json");
    return nlohmann::json::parse(stream);
  }();
  return fixture;
}

LlamaConfig config() {
  const auto& source = reference().at("config");
  return {
      .hidden_size = source.at("hidden_size").get<std::size_t>(),
      .intermediate_size = source.at("intermediate_size").get<std::size_t>(),
      .attention_heads = source.at("num_attention_heads").get<std::size_t>(),
      .key_value_heads = source.at("num_key_value_heads").get<std::size_t>(),
      .head_dimension = source.at("head_dim").get<std::size_t>(),
      .maximum_sequence_length = source.at("max_position_embeddings").get<std::size_t>(),
      .rms_norm_epsilon = source.at("rms_norm_eps").get<float>(),
      .rope_theta = source.at("rope_theta").get<float>(),
      .query_key_norm = true,
  };
}

std::vector<float> vector_weight(const std::string& name) {
  return reference().at("weights").at(name).at("values").get<std::vector<float>>();
}

Matrix matrix_weight(const std::string& name) {
  const auto& tensor = reference().at("weights").at(name);
  return Matrix(tensor.at("shape").at(0).get<std::size_t>(),
                tensor.at("shape").at(1).get<std::size_t>(), vector_weight(name));
}

LayerWeights weights(const std::size_t layer) {
  const auto prefix = "model.layers." + std::to_string(layer) + ".";
  return {
      .input_norm = vector_weight(prefix + "input_layernorm.weight"),
      .query = matrix_weight(prefix + "self_attn.q_proj.weight"),
      .key = matrix_weight(prefix + "self_attn.k_proj.weight"),
      .value = matrix_weight(prefix + "self_attn.v_proj.weight"),
      .attention_output = matrix_weight(prefix + "self_attn.o_proj.weight"),
      .post_attention_norm = vector_weight(prefix + "post_attention_layernorm.weight"),
      .gate = matrix_weight(prefix + "mlp.gate_proj.weight"),
      .up = matrix_weight(prefix + "mlp.up_proj.weight"),
      .down = matrix_weight(prefix + "mlp.down_proj.weight"),
      .query_norm = vector_weight(prefix + "self_attn.q_norm.weight"),
      .key_norm = vector_weight(prefix + "self_attn.k_norm.weight"),
  };
}

// The oracle tensors have shape [1, sequence, width].
Matrix rows(const nlohmann::json& tensor, const std::size_t start, const std::size_t count) {
  const auto width = tensor.at("shape").at(2).get<std::size_t>();
  const auto values = tensor.at("values").get<std::vector<float>>();
  std::vector<float> selected(count * width);
  for (std::size_t index = 0U; index < selected.size(); ++index) {
    selected[index] = values.at(start * width + index);
  }
  return Matrix(count, width, std::move(selected));
}

void expect_near(const Matrix& actual, const Matrix& expected) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.columns(), expected.columns());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    EXPECT_NEAR(actual.values()[index], expected.values()[index], 2e-5F) << index;
  }
}

void expect_cache(const LayerKvCache& cache, const std::size_t layer) {
  const auto cfg = config();
  const auto& source = reference();
  const auto keys = source.at("keys").at(layer).at("values").get<std::vector<float>>();
  const auto values = source.at("values").at(layer).at("values").get<std::vector<float>>();
  // Transformers stores [batch, kv_head, sequence, head_dim].
  for (std::size_t head = 0U; head < cfg.key_value_heads; ++head) {
    for (std::size_t token = 0U; token < cache.length(); ++token) {
      for (std::size_t dimension = 0U; dimension < cfg.head_dimension; ++dimension) {
        const auto index = (head * 5U + token) * cfg.head_dimension + dimension;
        EXPECT_NEAR(cache.key(token, head, dimension), keys.at(index), 2e-5F);
        EXPECT_NEAR(cache.value(token, head, dimension), values.at(index), 2e-5F);
      }
    }
  }
}

TEST(Qwen3Test, FullPrefillMatchesTransformersLayersCachesAndLogits) {
  const auto cfg = config();
  ASSERT_NE(cfg.hidden_size, cfg.attention_heads * cfg.head_dimension);
  auto hidden = rows(reference().at("embeddings"), 0U, 5U);
  for (std::size_t layer = 0U; layer < 2U; ++layer) {
    LayerKvCache cache(cfg.maximum_sequence_length, cfg.key_value_heads, cfg.head_dimension);
    hidden = transformer_layer(hidden, weights(layer), cfg, 0U, cache);
    expect_near(hidden, rows(reference().at("layer_outputs").at(layer), 0U, 5U));
    expect_cache(cache, layer);
  }
  expect_near(final_logits(hidden, vector_weight("model.norm.weight"),
                           matrix_weight("model.embed_tokens.weight"), cfg.rms_norm_epsilon),
              rows(reference().at("logits"), 0U, 5U));
}

TEST(Qwen3Test, PrefillAndTwoDecodeStepsMatchTransformers) {
  const auto cfg = config();
  std::array<LayerKvCache, 2> caches{
      LayerKvCache(cfg.maximum_sequence_length, cfg.key_value_heads, cfg.head_dimension),
      LayerKvCache(cfg.maximum_sequence_length, cfg.key_value_heads, cfg.head_dimension)};
  for (const std::size_t position : {0U, 3U, 4U}) {
    const auto count = position == 0U ? 3U : 1U;
    auto hidden = rows(reference().at("embeddings"), position, count);
    for (std::size_t layer = 0U; layer < caches.size(); ++layer) {
      hidden = transformer_layer(hidden, weights(layer), cfg, position, caches[layer]);
      expect_near(hidden, rows(reference().at("layer_outputs").at(layer), position, count));
      expect_cache(caches[layer], layer);
    }
    expect_near(final_logits(hidden, vector_weight("model.norm.weight"),
                             matrix_weight("model.embed_tokens.weight"), cfg.rms_norm_epsilon),
                rows(reference().at("logits"), position, count));
  }
}

TEST(Qwen3Test, RejectsMissingOrWrongNormsBeforeMutatingCache) {
  const auto cfg = config();
  const auto input = rows(reference().at("embeddings"), 0U, 1U);
  for (const bool missing : {true, false}) {
    auto layer = weights(0U);
    if (missing) {
      layer.query_norm.clear();
    } else {
      layer.key_norm.resize(cfg.hidden_size);
    }
    LayerKvCache cache(cfg.maximum_sequence_length, cfg.key_value_heads, cfg.head_dimension);
    EXPECT_THROW(static_cast<void>(transformer_layer(input, layer, cfg, 0U, cache)),
                 std::invalid_argument);
    EXPECT_EQ(cache.length(), 0U);
  }
}

TEST(Qwen3Test, RejectsCacheWithSameWidthButDifferentHeadGeometry) {
  const auto cfg = config();
  LayerKvCache cache(cfg.maximum_sequence_length, 1U, 8U);
  EXPECT_THROW(static_cast<void>(transformer_layer(
                   rows(reference().at("embeddings"), 0U, 1U), weights(0U), cfg, 0U, cache)),
               std::invalid_argument);
  EXPECT_EQ(cache.length(), 0U);
}

TEST(Qwen3Test, RejectsNormWeightsWhenQwenSemanticsAreDisabled) {
  auto cfg = config();
  cfg.query_key_norm = false;
  LayerKvCache cache(cfg.maximum_sequence_length, cfg.key_value_heads, cfg.head_dimension);
  EXPECT_THROW(static_cast<void>(transformer_layer(
                   rows(reference().at("embeddings"), 0U, 1U), weights(0U), cfg, 0U, cache)),
               std::invalid_argument);
}

}  // namespace
}  // namespace hllm::cpu
