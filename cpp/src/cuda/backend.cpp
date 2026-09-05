#include "hllm/cuda/backend.hpp"

#include <stdexcept>
#include <string>

#include "device.hpp"

namespace hllm::cuda {
namespace {
class CudaFactory final : public runtime::BackendFactory {
 public:
  explicit CudaFactory(int device_id) : detail_(probe_device(device_id)) {}
  runtime::BackendCapabilities capabilities() const override {
    return {v1::BACKEND_CUDA, v1::MEMORY_DOMAIN_DEVICE, {}, {}, detail_};
  }
  std::unique_ptr<runtime::StageBackend> load(const v1::LoadStageRequest&,
                                              const std::filesystem::path&,
                                              const runtime::MemoryAmounts&) const override {
    throw std::invalid_argument("CUDA model execution is not implemented");
  }

 private:
  std::string detail_;
};
}  // namespace

std::unique_ptr<runtime::BackendFactory> make_backend_factory(int device_id) {
  return std::make_unique<CudaFactory>(device_id);
}
}  // namespace hllm::cuda
