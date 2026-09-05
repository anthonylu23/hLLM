#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace hllm::cpu {

class Matrix final {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t columns);
  Matrix(std::size_t rows, std::size_t columns, std::vector<float> values);

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
  [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
  [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

  [[nodiscard]] float& operator()(std::size_t row, std::size_t column);
  [[nodiscard]] float operator()(std::size_t row, std::size_t column) const;
  [[nodiscard]] std::span<float> values() noexcept { return values_; }
  [[nodiscard]] std::span<const float> values() const noexcept { return values_; }

 private:
  std::size_t rows_{0U};
  std::size_t columns_{0U};
  std::vector<float> values_;
};

[[nodiscard]] Matrix linear(const Matrix& input, const Matrix& weights);
[[nodiscard]] Matrix quantize_float16(const Matrix& input);

}  // namespace hllm::cpu
