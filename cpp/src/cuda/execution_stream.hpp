#pragma once

#include <c10/cuda/CUDAStream.h>

namespace hllm::cuda {
// One process-lifetime stream per device. CUDA streams support submissions from
// multiple host threads; stage operations still synchronize before returning.
// Keep this Torch-only interface out of protobuf-facing translation units.
[[nodiscard]] c10::cuda::CUDAStream execution_stream(int device_id);
}  // namespace hllm::cuda
