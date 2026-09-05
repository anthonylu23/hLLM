#include "hllm/runtime/safetensors.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace hllm::runtime {
namespace {

constexpr std::size_t kMaximumHeaderBytes = 128U * 1024U * 1024U;

[[nodiscard]] std::string quote_name(const std::string_view value) {
  return "'" + std::string(value) + "'";
}

[[nodiscard]] std::size_t checked_size(const nlohmann::json& value,
                                       const std::string_view field,
                                       const std::string_view tensor_name) {
  if (!value.is_number_unsigned()) {
    throw SafetensorsError("tensor " + quote_name(tensor_name) + " has invalid " +
                           std::string(field));
  }
  const auto raw = value.get<std::uint64_t>();
  if (raw > std::numeric_limits<std::size_t>::max()) {
    throw SafetensorsError("tensor " + quote_name(tensor_name) + " has oversized " +
                           std::string(field));
  }
  return static_cast<std::size_t>(raw);
}

[[nodiscard]] DataType parse_dtype(const std::string_view value,
                                   const std::string_view tensor_name) {
  if (value == "BOOL") return DataType::kBool;
  if (value == "U8") return DataType::kU8;
  if (value == "I8") return DataType::kI8;
  if (value == "I16") return DataType::kI16;
  if (value == "U16") return DataType::kU16;
  if (value == "I32") return DataType::kI32;
  if (value == "U32") return DataType::kU32;
  if (value == "I64") return DataType::kI64;
  if (value == "U64") return DataType::kU64;
  if (value == "F16") return DataType::kF16;
  if (value == "BF16") return DataType::kBF16;
  if (value == "F32") return DataType::kF32;
  if (value == "F64") return DataType::kF64;
  throw SafetensorsError("tensor " + quote_name(tensor_name) +
                         " has unsupported dtype " + quote_name(value));
}

[[nodiscard]] std::size_t checked_product(const std::vector<std::size_t>& shape,
                                          const std::size_t element_size,
                                          const std::string_view tensor_name) {
  std::size_t result = element_size;
  for (const auto dimension : shape) {
    if (dimension != 0U && result > std::numeric_limits<std::size_t>::max() / dimension) {
      throw SafetensorsError("tensor " + quote_name(tensor_name) +
                             " byte length overflows");
    }
    result *= dimension;
  }
  return result;
}

[[nodiscard]] std::uint64_t read_little_endian_u64(
    const std::array<unsigned char, 8>& bytes) noexcept {
  std::uint64_t result = 0U;
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    result |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  }
  return result;
}

[[nodiscard]] TensorMetadata parse_tensor(const std::string& name,
                                          const nlohmann::json& entry) {
  if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
      !entry.contains("data_offsets")) {
    throw SafetensorsError("tensor " + quote_name(name) + " has invalid metadata");
  }
  if (!entry.at("dtype").is_string() || !entry.at("shape").is_array() ||
      !entry.at("data_offsets").is_array() || entry.at("data_offsets").size() != 2U) {
    throw SafetensorsError("tensor " + quote_name(name) + " has invalid metadata types");
  }

  const auto dtype = parse_dtype(entry.at("dtype").get_ref<const std::string&>(), name);
  std::vector<std::size_t> shape;
  shape.reserve(entry.at("shape").size());
  for (const auto& dimension : entry.at("shape")) {
    shape.push_back(checked_size(dimension, "shape", name));
  }
  const auto start = checked_size(entry.at("data_offsets").at(0U), "data offset", name);
  const auto end = checked_size(entry.at("data_offsets").at(1U), "data offset", name);
  if (end < start) {
    throw SafetensorsError("tensor " + quote_name(name) + " has a reversed data range");
  }
  const auto expected = checked_product(shape, data_type_bytes(dtype), name);
  if (end - start != expected) {
    std::ostringstream message;
    message << "tensor " << quote_name(name) << " byte length is " << end - start
            << ", expected " << expected;
    throw SafetensorsError(message.str());
  }
  return TensorMetadata{
      .name = name,
      .dtype = dtype,
      .shape = std::move(shape),
      .data_offset = start,
      .byte_length = end - start,
  };
}

}  // namespace

std::size_t data_type_bytes(const DataType dtype) noexcept {
  switch (dtype) {
    case DataType::kBool:
    case DataType::kU8:
    case DataType::kI8:
      return 1U;
    case DataType::kI16:
    case DataType::kU16:
    case DataType::kF16:
    case DataType::kBF16:
      return 2U;
    case DataType::kI32:
    case DataType::kU32:
    case DataType::kF32:
      return 4U;
    case DataType::kI64:
    case DataType::kU64:
    case DataType::kF64:
      return 8U;
  }
  return 0U;
}

SafetensorsFile::SafetensorsFile(std::filesystem::path path) : path_(std::move(path)) {
  std::error_code size_error;
  const auto raw_size = std::filesystem::file_size(path_, size_error);
  if (size_error || raw_size > std::numeric_limits<std::size_t>::max()) {
    throw SafetensorsError("cannot determine Safetensors file size: " + path_.string());
  }
  file_size_ = static_cast<std::size_t>(raw_size);
  if (file_size_ < 8U) {
    throw SafetensorsError("Safetensors file is shorter than its header prefix");
  }

  std::ifstream stream(path_, std::ios::binary);
  if (!stream) {
    throw SafetensorsError("cannot open Safetensors file: " + path_.string());
  }
  std::array<unsigned char, 8> prefix{};
  stream.read(reinterpret_cast<char*>(prefix.data()),
              static_cast<std::streamsize>(prefix.size()));
  const auto raw_header_length = read_little_endian_u64(prefix);
  if (raw_header_length == 0U || raw_header_length > kMaximumHeaderBytes ||
      raw_header_length > std::numeric_limits<std::size_t>::max()) {
    throw SafetensorsError("Safetensors file has an invalid header length");
  }
  const auto header_length = static_cast<std::size_t>(raw_header_length);
  if (header_length > file_size_ - 8U) {
    throw SafetensorsError("Safetensors header extends beyond the file");
  }
  header_size_ = 8U + header_length;
  std::string header(header_length, '\0');
  stream.read(header.data(), static_cast<std::streamsize>(header_length));
  if (!stream) {
    throw SafetensorsError("cannot read complete Safetensors header");
  }

  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(header);
  } catch (const nlohmann::json::exception& error) {
    throw SafetensorsError("Safetensors file contains invalid JSON: " +
                           std::string(error.what()));
  }
  if (!parsed.is_object()) {
    throw SafetensorsError("Safetensors header must be a JSON object");
  }
  for (const auto& [name, entry] : parsed.items()) {
    if (name != "__metadata__") {
      tensors_.push_back(parse_tensor(name, entry));
    }
  }
  if (tensors_.empty()) {
    throw SafetensorsError("Safetensors file does not contain tensors");
  }
  std::sort(tensors_.begin(), tensors_.end(),
            [](const TensorMetadata& lhs, const TensorMetadata& rhs) {
              return lhs.name < rhs.name;
            });

  auto ranges = tensors_;
  std::sort(ranges.begin(), ranges.end(),
            [](const TensorMetadata& lhs, const TensorMetadata& rhs) {
              return lhs.data_offset < rhs.data_offset;
            });
  const auto data_size = file_size_ - header_size_;
  std::size_t previous_end = 0U;
  for (const auto& tensor_metadata : ranges) {
    if (tensor_metadata.data_offset < previous_end) {
      throw SafetensorsError("tensor " + quote_name(tensor_metadata.name) +
                             " overlaps an earlier tensor");
    }
    if (tensor_metadata.data_offset > data_size ||
        tensor_metadata.byte_length > data_size - tensor_metadata.data_offset) {
      throw SafetensorsError("tensor " + quote_name(tensor_metadata.name) +
                             " extends beyond the data section");
    }
    previous_end = tensor_metadata.data_offset + tensor_metadata.byte_length;
  }
}

const TensorMetadata& SafetensorsFile::tensor(const std::string_view name) const {
  const auto found = std::lower_bound(
      tensors_.begin(), tensors_.end(), name,
      [](const TensorMetadata& candidate, const std::string_view requested) {
        return candidate.name < requested;
      });
  if (found == tensors_.end() || found->name != name) {
    throw SafetensorsError("tensor " + quote_name(name) + " is not present in " +
                           path_.string());
  }
  return *found;
}

Buffer SafetensorsFile::read_tensor(const std::string_view name, BufferPool& pool) const {
  const auto& metadata = tensor(name);
  if (metadata.byte_length == 0U) {
    return {};
  }
  auto buffer = pool.acquire(metadata.byte_length);
  buffer.resize(metadata.byte_length);

  std::ifstream stream(path_, std::ios::binary);
  if (!stream) {
    throw SafetensorsError("cannot reopen Safetensors file: " + path_.string());
  }
  const auto absolute_offset = header_size_ + metadata.data_offset;
  if (absolute_offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
    throw SafetensorsError("tensor offset is not representable by the native file API");
  }
  stream.seekg(static_cast<std::streamoff>(absolute_offset));
  stream.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(metadata.byte_length));
  if (!stream) {
    throw SafetensorsError("cannot read tensor " + quote_name(name));
  }
  return buffer;
}

}  // namespace hllm::runtime
