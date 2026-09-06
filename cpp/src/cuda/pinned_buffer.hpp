#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace hllm::cuda {
// Injectable CUDA calls are private backend machinery used by native fault tests;
// workers expose no fault-injection flags or RPCs.
struct PinnedApi {
  cudaError_t (*allocate)(void**, std::size_t, unsigned int) = cudaHostAlloc;
  decltype(&cudaFreeHost) free = cudaFreeHost;
  decltype(&cudaEventCreateWithFlags) create_event = cudaEventCreateWithFlags;
  decltype(&cudaEventDestroy) destroy_event = cudaEventDestroy;
  cudaError_t (*copy)(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t) = cudaMemcpyAsync;
  decltype(&cudaEventRecord) record = cudaEventRecord;
  decltype(&cudaEventSynchronize) wait = cudaEventSynchronize;
  decltype(&cudaStreamSynchronize) drain = cudaStreamSynchronize;
};
inline std::atomic_size_t allocated_pinned_bytes{0U};

// Sequence-owned staging. Every reuse waits for the preceding copy. Destruction
// drains the stream even if recording its completion event failed.
class PinnedBuffer {
 public:
  PinnedBuffer(std::size_t bytes, cudaStream_t stream, PinnedApi api = {})
      : bytes_(bytes), stream_(stream), api_(api) {
    check(api_.allocate(&data_, bytes, cudaHostAllocDefault));
    const auto status = api_.create_event(&event_, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      static_cast<void>(api_.free(data_));
      check(status);
    }
    allocated_pinned_bytes.fetch_add(bytes_);
  }
  ~PinnedBuffer() {
    if (pending_) static_cast<void>(api_.drain(stream_));
    static_cast<void>(api_.destroy_event(event_));
    if (api_.free(data_) == cudaSuccess) allocated_pinned_bytes.fetch_sub(bytes_);
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  void upload(void* device, const void* host, std::size_t bytes) {
    prepare(bytes);
    std::memcpy(data_, host, bytes);
    pending_ = true;
    try {
      check(api_.copy(device, data_, bytes, cudaMemcpyHostToDevice, stream_));
      check(api_.record(event_, stream_));
      wait();
    } catch (...) {
      failed_ = true;
      throw;
    }
  }
  void download(void* host, const void* device, std::size_t bytes) {
    prepare(bytes);
    pending_ = true;
    try {
      check(api_.copy(data_, device, bytes, cudaMemcpyDeviceToHost, stream_));
      check(api_.record(event_, stream_));
      wait();
    } catch (...) {
      failed_ = true;
      throw;
    }
    std::memcpy(host, data_, bytes);
  }
 private:
  static void check(cudaError_t status) {
    if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  void wait() {
    if (pending_) {
      check(api_.wait(event_));
      pending_ = false;
    }
  }
  void prepare(std::size_t bytes) {
    if (failed_) throw std::runtime_error("failed pinned staging cannot be reused");
    if (bytes > bytes_) throw std::length_error("boundary exceeds pinned staging capacity");
    wait();
  }
  std::size_t bytes_;
  cudaStream_t stream_;
  void* data_{};
  cudaEvent_t event_{};
  bool pending_{false};
  bool failed_{false};
  PinnedApi api_;
};
}  // namespace hllm::cuda
