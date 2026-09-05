#include "hllm/cpu/tensor.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

#include "hllm/runtime/half.hpp"

namespace hllm::cpu {
namespace {

[[nodiscard]] std::size_t checked_size(const std::size_t rows,
                                       const std::size_t columns) {
  if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::length_error("matrix size overflows size_t");
  }
  return rows * columns;
}

}  // namespace

Matrix::Matrix(const std::size_t rows, const std::size_t columns)
    : rows_(rows), columns_(columns), values_(checked_size(rows, columns)) {}

Matrix::Matrix(const std::size_t rows, const std::size_t columns,
               std::vector<float> values)
    : rows_(rows), columns_(columns), values_(std::move(values)) {
  if (values_.size() != checked_size(rows, columns)) {
    throw std::invalid_argument("matrix payload does not match its shape");
  }
}

float& Matrix::operator()(const std::size_t row, const std::size_t column) {
  if (row >= rows_ || column >= columns_) {
    throw std::out_of_range("matrix index is out of range");
  }
  return values_[row * columns_ + column];
}

float Matrix::operator()(const std::size_t row, const std::size_t column) const {
  if (row >= rows_ || column >= columns_) {
    throw std::out_of_range("matrix index is out of range");
  }
  return values_[row * columns_ + column];
}

Matrix linear(const Matrix& input, const Matrix& weights) {
  if (input.columns() != weights.columns()) {
    throw std::invalid_argument("linear input width does not match weight width");
  }
  Matrix output(input.rows(), weights.rows());
  for (std::size_t row = 0U; row < input.rows(); ++row) {
    for (std::size_t out = 0U; out < weights.rows(); ++out) {
      float sum = 0.0F;
      for (std::size_t inner = 0U; inner < input.columns(); ++inner) {
        sum += input(row, inner) * weights(out, inner);
      }
      output(row, out) = sum;
    }
  }
  return output;
}

Matrix quantize_float16(const Matrix& input) {
  Matrix output(input.rows(), input.columns());
  for (std::size_t index = 0U; index < input.size(); ++index) {
    output.values()[index] = runtime::float16_to_float(
        runtime::float_to_float16(input.values()[index]));
  }
  return output;
}

}  // namespace hllm::cpu
