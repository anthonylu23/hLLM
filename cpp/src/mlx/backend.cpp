#include "hllm/mlx/backend.hpp"

#include <mlx/backend/metal/metal.h>
#include <mlx/memory.h>
#include <mlx/version.h>

#include "device.hpp"
#include "hllm/model/dense_loader.hpp"

namespace hllm::mlx {
namespace {
class MlxFactory final : public runtime::BackendFactory {
 public:
  explicit MlxFactory(std::size_t limit) {
    std::scoped_lock lock(device_mutex());
    if (!mx::metal::is_available()) {
      throw std::runtime_error("MLX worker requires an available Metal GPU");
    }
    if (limit != 0U) {
      mx::set_memory_limit(limit);
      // TODO(measured placement): profile cache reuse against full-checkpoint
      // workspace before increasing this conservative tiny-model guideline.
      mx::set_cache_limit(std::min(limit / 8U, std::size_t{64U * 1024U * 1024U}));
    }
    completed(execution_stream(), [] {
      auto probe = mx::sum(mx::ones({2, 2}, mx::float32));
      if (probe.item<float>() != 4.0F) throw std::runtime_error("MLX GPU probe failed");
      return true;
    });
  }
  runtime::BackendCapabilities capabilities() const override {
    return {v1::BACKEND_MLX,
            v1::MEMORY_DOMAIN_UNIFIED,
            {"llama.v1", "qwen3.v1"},
            {v1::DATA_TYPE_F32, v1::DATA_TYPE_F16},
            "MLX Metal runtime ready"};
  }
  std::optional<runtime::AllocatorMetrics> allocator_metrics() const override {
    // Scrapes must not wait behind checkpoint I/O or a long execution step.
    std::unique_lock lock(device_mutex(), std::try_to_lock);
    if (!lock.owns_lock()) return std::nullopt;
    return runtime::AllocatorMetrics{mx::get_active_memory(), mx::get_cache_memory(),
                                     mx::get_peak_memory()};
  }
  bool profiling_reset_peak() const override {
    std::scoped_lock lock(device_mutex());
    mx::synchronize(execution_stream());
    mx::reset_peak_memory();
    return true;
  }
  runtime::ProfilingDeviceInfo profiling_device_info() const override {
    return {"metal-default", "Apple Metal", mx::version(), "os-bundled", "mlx-metal",
            std::nullopt};
  }
  std::unique_ptr<runtime::StageBackend> load(
      const v1::LoadStageRequest& request, const std::filesystem::path& root,
      const runtime::MemoryAmounts& capacity) const override {
    return load_device_stage(model::inspect_dense_stage(request, root), capacity);
  }
};
}  // namespace
std::unique_ptr<runtime::BackendFactory> make_backend_factory(std::size_t memory_limit) {
  return std::make_unique<MlxFactory>(memory_limit);
}
}  // namespace hllm::mlx
