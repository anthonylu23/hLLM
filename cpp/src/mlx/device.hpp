#pragma once

#include <mlx/mlx.h>

#include <mutex>
#include <stdexcept>
#include <string_view>

#include "hllm/model/dense_source.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::mlx {
namespace mx = ::mlx::core;
// Serialize this backend's framework entry points. A stage's stream is shared
// across gRPC handler threads, but never used concurrently.
inline std::mutex& device_mutex() {
  static std::mutex mutex;
  return mutex;
}
inline mx::Stream execution_stream() {
  static const auto stream = mx::new_thread_unsafe_stream(mx::Device::gpu);
  return stream;
}
class StreamScope {
 public:
  explicit StreamScope(mx::Stream stream)
      : previous_device_(mx::default_device()),
        previous_stream_(mx::default_stream(mx::Device::gpu)) {
    mx::set_default_device(mx::Device::gpu);
    mx::set_default_stream(stream);
  }
  ~StreamScope() {
    mx::set_default_stream(previous_stream_);
    mx::set_default_device(previous_device_);
  }

 private:
  mx::Device previous_device_;
  mx::Stream previous_stream_;
};
template <class Function>
auto completed(mx::Stream stream, Function&& function) {
  const StreamScope scope(stream);
  try {
    auto result = function();
    mx::synchronize(stream);
    return result;
  } catch (const std::runtime_error& error) {
    try {
      mx::synchronize(stream);
    } catch (...) {
    }
    // MLX 0.32.2 Metal allocator reports these allocation failures as
    // runtime_error. Preserve all other device/evaluation error classes.
    const std::string_view message(error.what());
    if (message.starts_with("[malloc] Unable to allocate ") ||
        message.starts_with("[metal::malloc] Resource limit (") ||
        message.starts_with("[metal::malloc] Attempting to allocate ")) {
      throw std::bad_alloc();
    }
    throw;
  } catch (...) {
    // Evaluation may already have submitted work before a later operation fails.
    try {
      mx::synchronize(stream);
    } catch (...) {
    }
    throw;
  }
}
[[nodiscard]] std::unique_ptr<runtime::StageBackend> load_device_stage(
    const model::DenseSource& source, const runtime::MemoryAmounts& capacity);
}  // namespace hllm::mlx
