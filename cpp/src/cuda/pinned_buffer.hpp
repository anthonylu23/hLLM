#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
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
    if (bytes == 0U) {
      throw std::invalid_argument("pinned staging capacity must be positive");
    }
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
    wait();
    auto* destination = static_cast<std::byte*>(device);
    const auto* source = static_cast<const std::byte*>(host);
    for (std::size_t offset = 0U; offset < bytes;) {
      const auto count = std::min(bytes_, bytes - offset);
      std::memcpy(data_, source + offset, count);
      pending_ = true;
      check(cudaMemcpyAsync(destination + offset, data_, count, cudaMemcpyHostToDevice, stream_));
      check(cudaEventRecord(event_, stream_));
      wait();
      offset += count;
    }
  }
  void download(void* host, const void* device, std::size_t bytes) {
    wait();
    auto* destination = static_cast<std::byte*>(host);
    const auto* source = static_cast<const std::byte*>(device);
    for (std::size_t offset = 0U; offset < bytes;) {
      const auto count = std::min(bytes_, bytes - offset);
      pending_ = true;
      check(cudaMemcpyAsync(data_, source + offset, count, cudaMemcpyDeviceToHost, stream_));
      check(cudaEventRecord(event_, stream_));
      wait();
      std::memcpy(destination + offset, data_, count);
      offset += count;
    }
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
  std::size_t bytes_;
  cudaStream_t stream_;
  void* data_{};
  cudaEvent_t event_{};
  bool pending_{false};
};
}  // namespace hllm::cuda
