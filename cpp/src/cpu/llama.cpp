#include "hllm/cpu/llama.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hllm::cpu {
namespace {

void validate_config(const LlamaConfig& config) {
  if (config.hidden_size == 0U || config.intermediate_size == 0U ||
      config.attention_heads == 0U || config.key_value_heads == 0U ||
      config.head_dimension == 0U || config.maximum_sequence_length == 0U ||
      config.rms_norm_epsilon <= 0.0F || config.rope_theta <= 0.0F) {
    throw std::invalid_argument("Llama configuration values must be positive");
  }
  if (config.attention_heads % config.key_value_heads != 0U ||
      config.attention_heads * config.head_dimension != config.hidden_size) {
    throw std::invalid_argument("Llama attention dimensions are inconsistent");
  }
}

void validate_weights(const LayerWeights& weights, const LlamaConfig& config) {
  const auto key_value_width = config.key_value_heads * config.head_dimension;
  const auto valid_matrix = [](const Matrix& matrix, const std::size_t rows,
                               const std::size_t columns) {
    return matrix.rows() == rows && matrix.columns() == columns;
  };
  if (weights.input_norm.size() != config.hidden_size ||
      weights.post_attention_norm.size() != config.hidden_size ||
      !valid_matrix(weights.query, config.hidden_size, config.hidden_size) ||
      !valid_matrix(weights.key, key_value_width, config.hidden_size) ||
      !valid_matrix(weights.value, key_value_width, config.hidden_size) ||
      !valid_matrix(weights.attention_output, config.hidden_size, config.hidden_size) ||
      !valid_matrix(weights.gate, config.intermediate_size, config.hidden_size) ||
      !valid_matrix(weights.up, config.intermediate_size, config.hidden_size) ||
      !valid_matrix(weights.down, config.hidden_size, config.intermediate_size)) {
    throw std::invalid_argument("transformer layer weights do not match configuration");
  }
}

[[nodiscard]] Matrix attention(const Matrix& queries, const LlamaConfig& config,
                               const std::size_t first_position,
                               const LayerKvCache& cache) {
  Matrix output(queries.rows(), config.hidden_size);
  const auto query_heads_per_key_value_head =
      config.attention_heads / config.key_value_heads;
  const auto scale = 1.0F / std::sqrt(static_cast<float>(config.head_dimension));
  std::vector<float> scores;

  for (std::size_t token = 0U; token < queries.rows(); ++token) {
    const auto maximum_key_position = first_position + token;
    scores.resize(maximum_key_position + 1U);
    for (std::size_t query_head = 0U; query_head < config.attention_heads;
         ++query_head) {
      const auto key_value_head = query_head / query_heads_per_key_value_head;
      float maximum_score = -std::numeric_limits<float>::infinity();
      for (std::size_t position = 0U; position <= maximum_key_position; ++position) {
        float score = 0.0F;
        for (std::size_t dimension = 0U; dimension < config.head_dimension;
             ++dimension) {
          score += queries(token, query_head * config.head_dimension + dimension) *
                   cache.key(position, key_value_head, dimension);
        }
        score *= scale;
        scores[position] = score;
        maximum_score = std::max(maximum_score, score);
      }

      float denominator = 0.0F;
      for (auto& score : scores) {
        score = std::exp(score - maximum_score);
        denominator += score;
      }
      for (std::size_t dimension = 0U; dimension < config.head_dimension;
           ++dimension) {
        float result = 0.0F;
        for (std::size_t position = 0U; position <= maximum_key_position; ++position) {
          result += (scores[position] / denominator) *
                    cache.value(position, key_value_head, dimension);
        }
        output(token, query_head * config.head_dimension + dimension) = result;
      }
    }
  }
  return output;
}

void add_in_place(Matrix& destination, const Matrix& source) {
  if (destination.rows() != source.rows() || destination.columns() != source.columns()) {
    throw std::invalid_argument("residual matrices must have matching shapes");
  }
  for (std::size_t index = 0U; index < destination.size(); ++index) {
    destination.values()[index] += source.values()[index];
  }
}

}  // namespace

LayerKvCache::LayerKvCache(const std::size_t maximum_sequence_length,
                           const std::size_t key_value_heads,
                           const std::size_t head_dimension)
    : maximum_sequence_length_(maximum_sequence_length),
      key_value_heads_(key_value_heads),
      head_dimension_(head_dimension),
      keys_(maximum_sequence_length * key_value_heads * head_dimension),
      values_(maximum_sequence_length * key_value_heads * head_dimension) {
  if (maximum_sequence_length == 0U || key_value_heads == 0U || head_dimension == 0U) {
    throw std::invalid_argument("KV cache dimensions must be positive");
  }
}

std::size_t LayerKvCache::offset(const std::size_t position, const std::size_t head,
                                 const std::size_t dimension) const {
  if (position >= length_ || head >= key_value_heads_ || dimension >= head_dimension_) {
    throw std::out_of_range("KV cache index is out of range");
  }
  return (position * key_value_heads_ + head) * head_dimension_ + dimension;
}

float LayerKvCache::key(const std::size_t position, const std::size_t head,
                        const std::size_t dimension) const {
  return keys_[offset(position, head, dimension)];
}

float LayerKvCache::value(const std::size_t position, const std::size_t head,
                          const std::size_t dimension) const {
  return values_[offset(position, head, dimension)];
}

void LayerKvCache::append(const std::size_t first_position, const Matrix& keys,
                          const Matrix& values) {
  const auto expected_width = key_value_heads_ * head_dimension_;
  if (first_position != length_) {
    throw std::invalid_argument("KV append position does not match cache length");
  }
  if (keys.rows() != values.rows() || keys.columns() != expected_width ||
      values.columns() != expected_width) {
    throw std::invalid_argument("KV append matrices have invalid shapes");
  }
  if (keys.rows() > maximum_sequence_length_ - length_) {
    throw std::length_error("KV append exceeds cache capacity");
  }
  for (std::size_t token = 0U; token < keys.rows(); ++token) {
    const auto destination = (length_ + token) * expected_width;
    for (std::size_t column = 0U; column < expected_width; ++column) {
      keys_[destination + column] = keys(token, column);
      values_[destination + column] = values(token, column);
    }
  }
  length_ += keys.rows();
}

Matrix rms_norm(const Matrix& input, const std::span<const float> weights,
                const float epsilon) {
  if (input.columns() != weights.size() || epsilon <= 0.0F) {
    throw std::invalid_argument("RMSNorm parameters do not match input");
  }
  Matrix output(input.rows(), input.columns());
  for (std::size_t row = 0U; row < input.rows(); ++row) {
    float square_sum = 0.0F;
    for (std::size_t column = 0U; column < input.columns(); ++column) {
      const auto value = input(row, column);
      square_sum += value * value;
    }
    const auto inverse_rms =
        1.0F / std::sqrt(square_sum / static_cast<float>(input.columns()) + epsilon);
    for (std::size_t column = 0U; column < input.columns(); ++column) {
      output(row, column) = input(row, column) * inverse_rms * weights[column];
    }
  }
  return output;
}

void apply_rope(Matrix& values, const std::size_t heads,
                const std::size_t head_dimension, const std::size_t first_position,
                const float theta) {
  if (heads == 0U || head_dimension == 0U || head_dimension % 2U != 0U ||
      heads * head_dimension != values.columns() || theta <= 0.0F) {
    throw std::invalid_argument("RoPE dimensions are invalid");
  }
  const auto half_dimension = head_dimension / 2U;
  for (std::size_t token = 0U; token < values.rows(); ++token) {
    const auto position = static_cast<float>(first_position + token);
    for (std::size_t head = 0U; head < heads; ++head) {
      const auto offset = head * head_dimension;
      for (std::size_t dimension = 0U; dimension < half_dimension; ++dimension) {
        const auto exponent =
            static_cast<float>(2U * dimension) / static_cast<float>(head_dimension);
        const auto angle = position / std::pow(theta, exponent);
        const auto cosine = std::cos(angle);
        const auto sine = std::sin(angle);
        const auto first = values(token, offset + dimension);
        const auto second = values(token, offset + half_dimension + dimension);
        values(token, offset + dimension) = first * cosine - second * sine;
        values(token, offset + half_dimension + dimension) =
            second * cosine + first * sine;
      }
    }
  }
}

Matrix transformer_layer(const Matrix& input, const LayerWeights& weights,
                         const LlamaConfig& config, const std::size_t first_position,
                         LayerKvCache& cache) {
  validate_config(config);
  validate_weights(weights, config);
  if (input.columns() != config.hidden_size || input.rows() == 0U ||
      first_position != cache.length() || first_position > config.maximum_sequence_length ||
      input.rows() > config.maximum_sequence_length - first_position) {
    throw std::invalid_argument("transformer input or cache position is invalid");
  }

  const auto normalized = rms_norm(input, weights.input_norm, config.rms_norm_epsilon);
  auto queries = linear(normalized, weights.query);
  auto keys = linear(normalized, weights.key);
  const auto values = linear(normalized, weights.value);
  apply_rope(queries, config.attention_heads, config.head_dimension, first_position,
             config.rope_theta);
  apply_rope(keys, config.key_value_heads, config.head_dimension, first_position,
             config.rope_theta);
  cache.append(first_position, keys, values);

  auto hidden = linear(attention(queries, config, first_position, cache),
                       weights.attention_output);
  add_in_place(hidden, input);

  const auto post_attention =
      rms_norm(hidden, weights.post_attention_norm, config.rms_norm_epsilon);
  auto gated = linear(post_attention, weights.gate);
  const auto up = linear(post_attention, weights.up);
  for (std::size_t index = 0U; index < gated.size(); ++index) {
    const auto value = gated.values()[index];
    gated.values()[index] = (value / (1.0F + std::exp(-value))) * up.values()[index];
  }
  add_in_place(hidden, linear(gated, weights.down));
  return hidden;
}

Matrix final_logits(const Matrix& hidden,
                    const std::span<const float> final_norm_weights,
                    const Matrix& language_model_head, const float epsilon) {
  return linear(rms_norm(hidden, final_norm_weights, epsilon), language_model_head);
}

std::size_t greedy_sample_last(const Matrix& logits) {
  if (logits.rows() == 0U || logits.columns() == 0U) {
    throw std::invalid_argument("cannot sample empty logits");
  }
  const auto last_row = logits.rows() - 1U;
  std::size_t selected = 0U;
  for (std::size_t token = 1U; token < logits.columns(); ++token) {
    if (logits(last_row, token) > logits(last_row, selected)) {
      selected = token;
    }
  }
  return selected;
}

}  // namespace hllm::cpu
