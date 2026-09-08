#include "hllm/cuda/backend.hpp"

#include <stdexcept>
#include <string>

#include "device.hpp"
#include "hllm/model/dense_loader.hpp"

namespace hllm::cuda {
namespace {
class CudaFactory final : public runtime::BackendFactory {
 public:
  CudaFactory(int device_id, bool pinned)
      : detail_(probe_device(device_id)), device_id_(device_id), pinned_(pinned) {}
  runtime::BackendCapabilities capabilities() const override {
    return {v1::BACKEND_CUDA,
            v1::MEMORY_DOMAIN_DEVICE,
            {"llama.v1", "qwen3.v1"},
            {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16},
            detail_};
  }
  std::optional<runtime::AllocatorMetrics> allocator_metrics() const override {
    const auto values = device_allocator_metrics(device_id_);
    if (!values) return std::nullopt;
    return runtime::AllocatorMetrics{(*values)[0], (*values)[1], (*values)[2]};
  }
  std::unique_ptr<runtime::StageBackend> load(
      const v1::LoadStageRequest& request, const std::filesystem::path& root,
      const runtime::MemoryAmounts& capacity) const override {
    return load_device_stage(model::inspect_dense_stage(request, root), device_id_, capacity, pinned_);
  }

 private:
  std::string detail_;
  int device_id_;
  bool pinned_;
};
}  // namespace

std::unique_ptr<runtime::BackendFactory> make_backend_factory(int device_id, bool pinned) {
  return std::make_unique<CudaFactory>(device_id, pinned);
}
}  // namespace hllm::cuda
