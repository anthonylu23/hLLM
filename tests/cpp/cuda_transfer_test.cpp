#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <future>
#include <thread>
#include <vector>

#include "pinned_buffer.hpp"

namespace hllm::cuda {
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
    EXPECT_THROW(staging.upload(device, nullptr, capacity + 1U), std::length_error);
  }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CudaTransferTest, AllocationAndEventCreationFailuresRollBack) {
  const auto before = allocated_pinned_bytes.load();
  PinnedApi api;
  api.allocate = [](void**, std::size_t, unsigned int) { return cudaErrorMemoryAllocation; };
  EXPECT_THROW(PinnedBuffer(4096U, nullptr, api), std::bad_alloc);
  api = {};
  static int frees = 0;
  frees = 0;
  api.free = [](void* pointer) { ++frees; return cudaFreeHost(pointer); };
  api.create_event = [](cudaEvent_t*, unsigned int) { return cudaErrorMemoryAllocation; };
  EXPECT_THROW(PinnedBuffer(4096U, nullptr, api), std::bad_alloc);
  EXPECT_EQ(frees, 1);
  EXPECT_EQ(allocated_pinned_bytes.load(), before);
  { PinnedBuffer fresh(4096U, nullptr); }
  EXPECT_EQ(allocated_pinned_bytes.load(), before);
}

TEST(CudaTransferTest, FailedEventRecordingDrainsPendingCopyBeforeFree) {
  cudaStream_t stream{};
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  void* device{};
  ASSERT_EQ(cudaMalloc(&device, 4096U), cudaSuccess);
  PinnedApi api;
  api.record = [](cudaEvent_t, cudaStream_t) { return cudaErrorUnknown; };
  const auto before = allocated_pinned_bytes.load();
  auto staging = std::make_unique<PinnedBuffer>(4096U, stream, api);
  std::atomic_bool release{false}, failed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream, [](void* opaque) {
    auto& ready = *static_cast<std::atomic_bool*>(opaque);
    while (!ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }, &release), cudaSuccess);
  std::vector<unsigned char> source(4096U, 42U);
  auto cleanup = std::async(std::launch::async, [&, owned = std::move(staging)]() mutable {
    try { owned->upload(device, source.data(), source.size()); }
    catch (const std::runtime_error&) { failed.store(true); }
    EXPECT_THROW(owned->upload(device, source.data(), source.size()), std::runtime_error);
    owned.reset();
  });
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!failed.load() && std::chrono::steady_clock::now() < until) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(failed.load());
  EXPECT_EQ(cleanup.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
  EXPECT_EQ(allocated_pinned_bytes.load(), before + 4096U);
  release.store(true);
  cleanup.get();
  EXPECT_EQ(allocated_pinned_bytes.load(), before);
  std::vector<unsigned char> output(4096U);
  EXPECT_EQ(cudaMemcpy(output.data(), device, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(source, output);
  EXPECT_EQ(cudaFree(device), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
}  // namespace hllm::cuda
