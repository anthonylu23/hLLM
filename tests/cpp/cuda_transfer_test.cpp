#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "pinned_buffer.hpp"

namespace hllm::cuda {
TEST(CudaTransferTest, TransfersPayloadsLargerThanStagingWithoutGrowingTheBuffer) {
  cudaStream_t stream{};
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  constexpr std::size_t capacity = 8U * 1024U * 1024U;
  constexpr std::size_t maximum_payload = 2U * capacity + 17U;
  void* device{};
  ASSERT_EQ(cudaMalloc(&device, maximum_payload), cudaSuccess);
  {
    PinnedBuffer staging(capacity, stream);
    for (const auto bytes : {capacity + 1U, maximum_payload, std::size_t{31U}}) {
      std::vector<unsigned char> input(bytes), output(bytes);
      for (std::size_t i = 0U; i < bytes; ++i) {
        input[i] = static_cast<unsigned char>((i + i / capacity) % 251U);
      }
      staging.upload(device, input.data(), bytes);
      staging.download(output.data(), device, bytes);
      EXPECT_EQ(input, output);
    }
  }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CudaTransferTest, ReusesBoundedStorageAndPreservesPayloadBytes) {
  cudaStream_t stream{};
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  void* device{};
  constexpr std::size_t capacity = 65536U;
  ASSERT_EQ(cudaMalloc(&device, capacity), cudaSuccess);
  {
    PinnedBuffer staging(capacity, stream);
    for (const std::size_t bytes : {12U, 6144U, 65536U}) {
      std::vector<unsigned char> input(bytes), output(bytes);
      for (std::size_t i = 0; i < bytes; ++i) input[i] = static_cast<unsigned char>(i % 251U);
      for (const bool pinned : {false, true}) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < 100U; ++repeat) {
          if (pinned) {
            staging.upload(device, input.data(), bytes);
            staging.download(output.data(), device, bytes);
          } else {
            ASSERT_EQ(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
          }
          ASSERT_EQ(input, output);
        }
        const auto micros = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / 100.0;
        std::cout << "transfer roundtrip bytes=" << bytes << " pinned=" << pinned
                  << " mean_us=" << micros << '\n';
      }
    }
  }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
}  // namespace hllm::cuda
