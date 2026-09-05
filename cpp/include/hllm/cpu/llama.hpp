#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "hllm/cpu/tensor.hpp"

namespace hllm::cpu {

struct LlamaConfig {
  std::size_t hidden_size;
  std::size_t intermediate_size;
  std::size_t attention_heads;
  std::size_t key_value_heads;
  std::size_t head_dimension;
  std::size_t maximum_sequence_length;
  float rms_norm_epsilon;
  float rope_theta;
  bool query_key_norm{false};
};

struct LayerWeights {
  std::vector<float> input_norm;
  Matrix query;
  Matrix key;
  Matrix value;
  Matrix attention_output;
  std::vector<float> post_attention_norm;
  Matrix gate;
  Matrix up;
  Matrix down;
  std::vector<float> query_norm{};
  std::vector<float> key_norm{};
};

class LayerKvCache final {
 public:
  LayerKvCache(std::size_t maximum_sequence_length, std::size_t key_value_heads,
               std::size_t head_dimension);

  [[nodiscard]] std::size_t length() const noexcept { return length_; }
  [[nodiscard]] std::size_t heads() const noexcept { return key_value_heads_; }
  [[nodiscard]] std::size_t head_dimension() const noexcept { return head_dimension_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return maximum_sequence_length_; }
  [[nodiscard]] float key(std::size_t position, std::size_t head,
                          std::size_t dimension) const;
  [[nodiscard]] float value(std::size_t position, std::size_t head,
                            std::size_t dimension) const;
  void append(std::size_t first_position, const Matrix& keys, const Matrix& values);
  void clear() noexcept { length_ = 0U; }

 private:
  [[nodiscard]] std::size_t offset(std::size_t position, std::size_t head,
                                   std::size_t dimension) const;

  std::size_t maximum_sequence_length_;
  std::size_t key_value_heads_;
  std::size_t head_dimension_;
  std::size_t length_{0U};
  std::vector<float> keys_;
  std::vector<float> values_;
};

[[nodiscard]] Matrix rms_norm(const Matrix& input, std::span<const float> weights,
                              float epsilon);
void apply_rope(Matrix& values, std::size_t heads, std::size_t head_dimension,
                std::size_t first_position, float theta);
[[nodiscard]] Matrix transformer_layer(const Matrix& input, const LayerWeights& weights,
                                       const LlamaConfig& config,
                                       std::size_t first_position, LayerKvCache& cache);
[[nodiscard]] Matrix final_logits(const Matrix& hidden,
                                  std::span<const float> final_norm_weights,
                                  const Matrix& language_model_head, float epsilon);
[[nodiscard]] std::size_t greedy_sample_last(const Matrix& logits);

}  // namespace hllm::cpu
