#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "hllm/runtime/safetensors.hpp"

namespace hllm::model {

struct DenseConfig {
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

struct TensorSource {
  std::string file;
  std::vector<std::size_t> shape;
  runtime::DataType dtype;
};

// Validated metadata only. Read and convert one tensor at a time; a GPU loader
// need not hold a complete CPU copy of the assigned weights.
struct DenseSource {
  DenseConfig config{};
  std::size_t vocabulary_size{};
  std::size_t layer_start{};
  std::size_t layer_end{};
  bool first{};
  bool final{};
  bool tied_head{};
  runtime::DataType execution_dtype{};
  runtime::DataType weight_dtype{runtime::DataType::kF32};
  // Maximum simultaneous F32 weight casts during one evaluated layer or head.
  std::size_t weight_cast_workspace_bytes{};
  std::size_t float32_weight_bytes{};
  std::size_t largest_payload_bytes{};
  std::size_t largest_float32_tensor_bytes{};
  // Bounded tied-head comparison scratch, included in peak load admission.
  std::size_t verification_workspace_bytes{};
  std::optional<std::string> redundant_head_file;
  std::map<std::string, TensorSource> tensors;
  std::map<std::string, runtime::SafetensorsFile> files;

  // Call only after admitting load memory. Reading an embedding also verifies
  // any redundant tied head against its raw buffer before conversion.
  // At most 1 MiB source + 2 MiB F32 conversion + 1 MiB tied-head comparison.
  static constexpr std::size_t conversion_chunk_bytes = 1024U * 1024U;
  [[nodiscard]] std::size_t conversion_workspace_bytes() const;
  void for_each_float32_chunk(const std::string& name,
                              const std::function<void(std::span<const float>)>& consume) const;
  [[nodiscard]] std::vector<float> read_float32(const std::string& name) const;
};

}  // namespace hllm::model
