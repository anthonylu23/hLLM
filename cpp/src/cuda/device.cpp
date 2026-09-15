#include "device.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/version.h>

#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <iomanip>
#include <sstream>

#include "execution_stream.hpp"

namespace hllm::cuda {
namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA initialization failed: ") +
                             cudaGetErrorString(result));
  }
}

}  // namespace

c10::cuda::CUDAStream execution_stream(int device_id) {
  static std::mutex mutex;
  static std::map<int, c10::cuda::CUDAStream> streams;
  const std::lock_guard lock(mutex);
  const auto found = streams.find(device_id);
  if (found != streams.end()) return found->second;
  const auto stream =
      c10::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(device_id));
  return streams.emplace(device_id, stream).first->second;
}

std::optional<std::array<std::size_t, 3>> device_allocator_metrics(int device_id) {
  // These counters are defined by the native allocator. Other implementations
  // may return placeholder zeros; do not present those as measured free memory.
  if (c10::cuda::CUDACachingAllocator::name() != "native") return std::nullopt;
  // Allocator statistics take their own lock; no GPU synchronization or cache
  // eviction is needed to observe them during an in-flight request.
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
      static_cast<c10::DeviceIndex>(device_id));  // Validated by the factory's probe.
  const auto active = stats.active_bytes[0].current;
  const auto reserved = stats.reserved_bytes[0].current;
  return std::array<std::size_t, 3>{
      static_cast<std::size_t>(active), static_cast<std::size_t>(reserved - active),
      static_cast<std::size_t>(stats.active_bytes[0].peak)};
}

std::string probe_device(int device_id) {
  static std::once_flag precision;
  std::call_once(precision, [] {
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowFP16ReductionCuBLAS(false);
  });
  int count = 0;
  check(cudaGetDeviceCount(&count));
  if (device_id < 0 || device_id >= count ||
      device_id > std::numeric_limits<c10::DeviceIndex>::max()) {
    throw std::invalid_argument("CUDA device ID is out of range");
  }
  const auto index = static_cast<c10::DeviceIndex>(device_id);
  const c10::cuda::CUDAGuard guard(index);
  const c10::cuda::CUDAStreamGuard stream_guard(execution_stream(device_id));
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device_id));
  // Exercise the linked ATen CUDA dispatcher, allocation, kernel and completed
  // device-to-host result. A driver-only probe would miss a CPU-only Torch build.
  const auto value =
      at::ones({4}, at::TensorOptions().device(at::Device(at::kCUDA, index)).dtype(at::kFloat));
  if (value.sum().item<float>() != 4.0F) {
    throw std::runtime_error("LibTorch CUDA startup probe failed");
  }
  return "CUDA runtime ready on device " + std::to_string(device_id) + " (" + properties.name +
         "); dense Llama/Qwen3 F32/F16 execution ready";
}

bool reset_device_peak(int device_id) {
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_id));
  execution_stream(device_id).synchronize();
  if (c10::cuda::CUDACachingAllocator::name() != "native") return false;
  c10::cuda::CUDACachingAllocator::resetPeakStats(static_cast<c10::DeviceIndex>(device_id));
  return true;
}

runtime::ProfilingDeviceInfo profiling_device_info(int device_id) {
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_id));
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device_id));
  int driver = 0;
  check(cudaDriverGetVersion(&driver));
  std::size_t available = 0U, total = 0U;
  check(cudaMemGetInfo(&available, &total));
  std::ostringstream identity;
  identity << std::hex << std::setfill('0');
  for (const auto byte : properties.uuid.bytes) {
    identity << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(byte));
  }
  return {identity.str(), properties.name, TORCH_VERSION, std::to_string(driver),
          c10::cuda::CUDACachingAllocator::name(), available};
}
}  // namespace hllm::cuda
