#include "hllm/model/dense_source.hpp"

#include <bit>
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
