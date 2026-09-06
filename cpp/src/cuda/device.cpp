#include "device.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>

namespace hllm::cuda {
namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA initialization failed: ") +
                             cudaGetErrorString(result));
  }
}

}  // namespace

std::string probe_device(int device_id) {
  int count = 0;
  check(cudaGetDeviceCount(&count));
  if (device_id < 0 || device_id >= count ||
      device_id > std::numeric_limits<c10::DeviceIndex>::max()) {
    throw std::invalid_argument("CUDA device ID is out of range");
  }
  const auto index = static_cast<c10::DeviceIndex>(device_id);
  const c10::cuda::CUDAGuard guard(index);
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
         "); model execution is not implemented";
}
}  // namespace hllm::cuda
