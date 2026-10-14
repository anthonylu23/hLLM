#pragma once

#include <mlx/mlx.h>

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "hllm/model/dense_source.hpp"
#include "hllm/runtime/backend_factory.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::mlx {
namespace mx = ::mlx::core;
// Serialize this backend's framework entry points, including the release of
// device arrays, which mutates the allocator's counters: a stage's weights and a
// sequence's caches are freed under this mutex by their destructors. Recursive
// because such an owner may be destroyed while an entry point already holds the
// mutex (an allocation failure inside allocate_sequence unwinds a partial state).
// A stage's stream is shared across gRPC handler threads, but never used
// concurrently.
inline std::recursive_mutex& device_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}
inline mx::Stream execution_stream() {
  static const auto stream = mx::new_thread_unsafe_stream(mx::Device::gpu);
  return stream;
}
// MLX 0.32.2 reports its allocator counters from plain fields that malloc/free
// update under the allocator's own mutex on the scheduler thread; the getters take
// no lock. Reading them while a step is in flight is a data race, so they are read
// only while the device mutex is held and the stream is synchronized, and that
// reading is published here for scrapes that arrive mid-step. The published sample
// is at most one execution step old; the peak counter is monotonic between resets.
class AllocatorBoundary {
 public:
  // Caller holds the device mutex and has synchronized the execution stream.
  void record_quiescent() {
    const runtime::AllocatorMetrics live{mx::get_active_memory(), mx::get_cache_memory(),
                                         mx::get_peak_memory()};
    std::scoped_lock lock(mutex_);
    sample_ = live;
  }
  [[nodiscard]] std::optional<runtime::AllocatorMetrics> latest() const {
    std::scoped_lock lock(mutex_);
    return sample_;
  }

 private:
  mutable std::mutex mutex_;
  std::optional<runtime::AllocatorMetrics> sample_;
};
inline AllocatorBoundary& allocator_boundary() {
  static AllocatorBoundary boundary;
  return boundary;
}
class StreamScope {
 public:
  explicit StreamScope(mx::Stream stream)
      : previous_device_(mx::default_device()) {
    // Bind every calling thread to the worker's shared stream. Querying its
    // previous default would lazily allocate a permanent stream per thread.
    mx::set_default_device(mx::Device::gpu);
    mx::set_default_stream(stream);
  }
  ~StreamScope() {
    mx::set_default_device(previous_device_);
  }

 private:
  mx::Device previous_device_;
};
// Runs `function` on the worker's stream and waits for its completion. Every
// caller holds the device mutex, so the allocator is quiescent after the
// synchronize and its counters are published as this boundary's sample.
template <class Function>
auto completed(mx::Stream stream, Function&& function) {
  const StreamScope scope(stream);
  try {
    auto result = function();
    mx::synchronize(stream);
    allocator_boundary().record_quiescent();
    return result;
  } catch (const std::runtime_error& error) {
    try {
      mx::synchronize(stream);
      allocator_boundary().record_quiescent();
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
      allocator_boundary().record_quiescent();
    } catch (...) {
    }
    throw;
  }
}
[[nodiscard]] std::unique_ptr<runtime::StageBackend> load_device_stage(
    const model::DenseSource& source, const runtime::MemoryAmounts& capacity);
}  // namespace hllm::mlx
