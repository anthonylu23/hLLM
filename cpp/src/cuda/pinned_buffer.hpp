#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace hllm::cuda {
// Sequence-owned staging. Every reuse waits for the preceding copy. Destruction
// drains the stream even if recording its completion event failed.
class PinnedBuffer {
 public:
  PinnedBuffer(std::size_t bytes, cudaStream_t stream) : bytes_(bytes), stream_(stream) {
    if (cudaHostAlloc(&data_, bytes, cudaHostAllocDefault) != cudaSuccess) {
      throw std::bad_alloc();
    }
    const auto status = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      static_cast<void>(cudaFreeHost(data_));
      check(status);
    }
  }
  ~PinnedBuffer() {
    if (pending_) static_cast<void>(cudaStreamSynchronize(stream_));
    static_cast<void>(cudaEventDestroy(event_));
    static_cast<void>(cudaFreeHost(data_));
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  void upload(void* device, const void* host, std::size_t bytes) {
    prepare(bytes);
    std::memcpy(data_, host, bytes);
    pending_ = true;
    check(cudaMemcpyAsync(device, data_, bytes, cudaMemcpyHostToDevice, stream_));
    check(cudaEventRecord(event_, stream_));
    wait();
  }
  void download(void* host, const void* device, std::size_t bytes) {
    prepare(bytes);
    pending_ = true;
    check(cudaMemcpyAsync(data_, device, bytes, cudaMemcpyDeviceToHost, stream_));
    check(cudaEventRecord(event_, stream_));
    wait();
    std::memcpy(host, data_, bytes);
  }
 private:
  static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  void wait() {
    if (pending_) {
      check(cudaEventSynchronize(event_));
      pending_ = false;
    }
  }
  void prepare(std::size_t bytes) {
    if (bytes > bytes_) throw std::length_error("boundary exceeds pinned staging capacity");
    wait();
  }
  std::size_t bytes_;
  cudaStream_t stream_;
  void* data_{};
  cudaEvent_t event_{};
  bool pending_{false};
};
}  // namespace hllm::cuda
