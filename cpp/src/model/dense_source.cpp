#include "hllm/model/dense_source.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <cmath>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/half.hpp"

namespace hllm::model {
std::vector<float> DenseSource::read_float32(const std::string& name) const {
  std::size_t elements = 1U;
  for (auto dimension : tensors.at(name).shape) {
    elements = runtime::checked_multiply(elements, dimension);
  }
  std::vector<float> values;
  values.reserve(elements);
  for_each_float32_chunk(name, [&](std::span<const float> chunk) {
    values.insert(values.end(), chunk.begin(), chunk.end());
  });
  return values;
}
}  // namespace hllm::model

namespace hllm::model {
std::size_t DenseSource::conversion_workspace_bytes() const {
  const auto raw = std::min(largest_payload_bytes, conversion_chunk_bytes);
  return runtime::checked_add(runtime::checked_multiply(raw, 3U),
                              std::min(verification_workspace_bytes, raw));
}
void DenseSource::for_each_float32_chunk(
    const std::string& name, const std::function<void(std::span<const float>)>& consume) const {
  const auto& tensor = tensors.at(name);
  const auto width = runtime::data_type_bytes(tensor.dtype);
  if (tensor.dtype != runtime::DataType::kF32 && tensor.dtype != runtime::DataType::kF16 &&
      tensor.dtype != runtime::DataType::kBF16) {
    throw runtime::Error::incompatible_worker("unsupported weight dtype");
  }
  const auto& file = files.at(tensor.file);
  auto open_payload = [](const runtime::SafetensorsFile& source, const std::string& tensor_name) {
    std::ifstream input(source.path(), std::ios::binary);
    const auto offset =
        runtime::checked_add(source.header_size(), source.tensor(tensor_name).data_offset);
    if (!input || offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
      throw runtime::Error::incompatible_worker("cannot open tensor payload");
    }
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input) throw runtime::Error::incompatible_worker("cannot seek tensor payload");
    return input;
  };
  auto input = open_payload(file, name);
  std::optional<std::ifstream> head;
  if (name == "model.embed_tokens.weight" && redundant_head_file) {
    head.emplace(open_payload(files.at(*redundant_head_file), "lm_head.weight"));
  }
  std::size_t remaining = 1U;
  for (auto dimension : tensor.shape) remaining = runtime::checked_multiply(remaining, dimension);
  remaining = runtime::checked_multiply(remaining, width);
  std::vector<char> raw(std::min(remaining, conversion_chunk_bytes));
  std::vector<char> comparison(head ? raw.size() : 0U);
  std::vector<float> decoded(raw.size() / width);
  while (remaining) {
    const auto bytes = std::min(remaining, raw.size());
    if (!input.read(raw.data(), static_cast<std::streamsize>(bytes))) {
      throw runtime::Error::incompatible_worker("truncated weight payload: " + name);
    }
    if (head && (!head->read(comparison.data(), static_cast<std::streamsize>(bytes)) ||
                 std::memcmp(raw.data(), comparison.data(), bytes) != 0)) {
      throw runtime::Error::incompatible_worker("redundant tied head differs from token embedding");
    }
    for (std::size_t i = 0; i < bytes / width; ++i) {
      std::uint32_t bits = 0;
      for (std::size_t b = 0; b < width; ++b) {
        bits |= static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i * width + b]))
                << (8U * b);
      }
      decoded[i] = tensor.dtype == runtime::DataType::kF32 ? std::bit_cast<float>(bits)
                   : tensor.dtype == runtime::DataType::kF16
                       ? runtime::float16_to_float(static_cast<std::uint16_t>(bits))
                       : runtime::bfloat16_to_float(static_cast<std::uint16_t>(bits));
      if (!std::isfinite(decoded[i])) {
        throw runtime::Error::incompatible_worker("non-finite weight: " + name);
      }
    }
    consume(std::span<const float>(decoded.data(), bytes / width));
    remaining -= bytes;
  }
}
}  // namespace hllm::model
