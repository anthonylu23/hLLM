#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "hllm/runtime/buffer.hpp"
#include "hllm/runtime/error.hpp"

namespace hllm::runtime {

enum class DataType : std::uint8_t {
  kBool,
  kU8,
  kI8,
  kI16,
  kU16,
  kI32,
  kU32,
  kI64,
  kU64,
  kF16,
  kBF16,
  kF32,
  kF64,
};

struct TensorMetadata {
  std::string name;
  DataType dtype;
  std::vector<std::size_t> shape;
  std::size_t data_offset;
  std::size_t byte_length;
};

// A malformed or unsupported checkpoint cannot be executed by this worker.
class SafetensorsError : public Error {
 public:
  explicit SafetensorsError(const std::string& message)
      : Error(ErrorCode::kIncompatibleWorker, message) {}
};

class SafetensorsFile final {
 public:
  explicit SafetensorsFile(std::filesystem::path path);

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
  [[nodiscard]] std::size_t file_size() const noexcept { return file_size_; }
  [[nodiscard]] std::size_t header_size() const noexcept { return header_size_; }
  [[nodiscard]] const std::vector<TensorMetadata>& tensors() const noexcept {
    return tensors_;
  }

  [[nodiscard]] const TensorMetadata& tensor(std::string_view name) const;
  [[nodiscard]] Buffer read_tensor(std::string_view name, BufferPool& pool) const;

 private:
  std::filesystem::path path_;
  std::size_t file_size_{0U};
  std::size_t header_size_{0U};
  std::vector<TensorMetadata> tensors_;
};

[[nodiscard]] std::size_t data_type_bytes(DataType dtype) noexcept;

}  // namespace hllm::runtime
