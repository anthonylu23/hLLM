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
  const auto& tensor = tensors.at(name);
  if (tensor.dtype != runtime::DataType::kF32 && tensor.dtype != runtime::DataType::kF16 &&
      tensor.dtype != runtime::DataType::kBF16) {
    throw runtime::Error::incompatible_worker("unsupported weight dtype");
  }
  runtime::BufferPool pool(0U);
  auto buffer = files.at(tensor.file).read_tensor(name, pool);
  if (name == "model.embed_tokens.weight" && redundant_head_file) {
    const auto& head_file = files.at(*redundant_head_file);
    const auto& head = head_file.tensor("lm_head.weight");
    std::ifstream input(head_file.path(), std::ios::binary);
    if (!input) {
      throw runtime::Error::incompatible_worker("cannot reopen tied-head Safetensors file");
    }
    const auto offset = runtime::checked_add(head_file.header_size(), head.data_offset);
    if (offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
      throw runtime::Error::incompatible_worker("tied tensor offset exceeds stream capacity");
    }
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input) throw runtime::Error::incompatible_worker("cannot seek tied tensor payload");
    std::vector<char> chunk(verification_workspace_bytes);
    const auto embedding = buffer.bytes();
    for (std::size_t position = 0U; position < embedding.size();) {
      const auto size = std::min(chunk.size(), embedding.size() - position);
      if (!input.read(chunk.data(), static_cast<std::streamsize>(size))) {
        throw runtime::Error::incompatible_worker("truncated tied tensor payload");
      }
      if (std::memcmp(chunk.data(), embedding.data() + position, size) != 0) {
        throw runtime::Error::incompatible_worker("redundant tied head differs from token embedding");
      }
      position += size;
    }
  }
  std::size_t elements = 1U;
  for (const auto dimension : tensor.shape) {
    elements = runtime::checked_multiply(elements, dimension);
  }
  std::vector<float> values(elements);
  const auto data = buffer.bytes();
  const auto width = runtime::data_type_bytes(tensor.dtype);
  for (std::size_t i = 0U; i < elements; ++i) {
    std::uint32_t bits = 0U;
    for (std::size_t b = 0U; b < width; ++b) {
      bits |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[i * width + b]))
              << (8U * b);
    }
    values[i] = tensor.dtype == runtime::DataType::kF32 ? std::bit_cast<float>(bits)
                : tensor.dtype == runtime::DataType::kF16
                    ? runtime::float16_to_float(static_cast<std::uint16_t>(bits))
                    : runtime::bfloat16_to_float(static_cast<std::uint16_t>(bits));
    if (!std::isfinite(values[i])) {
      throw runtime::Error::incompatible_worker("non-finite weight: " + name);
    }
  }
  return values;
}
}  // namespace hllm::model
