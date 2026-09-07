#include <grpcpp/grpcpp.h>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef HLLM_WORKER_CUDA
#include "hllm/cuda/backend.hpp"
#elif defined(HLLM_WORKER_MLX)
#include "hllm/mlx/backend.hpp"
#else
#include "hllm/cpu/stage.hpp"
#endif
#include "hllm/worker/control_service.hpp"
#include "hllm/worker/execution_service.hpp"

namespace {
#ifdef HLLM_WORKER_CUDA
constexpr auto kWorkerName = "hllm-worker-cuda";
#elif defined(HLLM_WORKER_MLX)
constexpr auto kWorkerName = "hllm-worker-mlx";
#else
constexpr auto kWorkerName = "hllm-worker-cpu";
#endif

struct Arguments {
  std::string listen;
  std::string worker_id;
  std::filesystem::path model_root;
  std::uint64_t memory_limit_bytes{0U};
  std::uint64_t device_memory_limit_bytes{0U};
  std::uint64_t pinned_memory_limit_bytes{0U};
  int device_id{0};
  bool pinned{false};
};

[[nodiscard]] std::uint64_t parse_u64(const std::string_view value) {
  std::uint64_t parsed = 0U;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0U) {
    throw std::invalid_argument("memory limit must be a positive integer");
  }
  return parsed;
}

[[nodiscard]] Arguments parse_arguments(const int argc, char** const argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string_view flag(argv[index]);
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(flag));
    }
    const std::string_view value(argv[++index]);
    if (flag == "--listen") {
      arguments.listen = value;
    } else if (flag == "--worker-id") {
      arguments.worker_id = value;
    } else if (flag == "--model-root") {
      arguments.model_root = value;
    } else if (flag == "--memory-limit-bytes") {
      arguments.memory_limit_bytes = parse_u64(value);
#ifdef HLLM_WORKER_CUDA
    } else if (flag == "--device-memory-limit-bytes") {
      arguments.device_memory_limit_bytes = parse_u64(value);
    } else if (flag == "--pinned-host-memory-limit-bytes") {
      arguments.pinned_memory_limit_bytes = parse_u64(value);
    } else if (flag == "--boundary-transfer-mode") {
      if (value != "pageable" && value != "pinned") {
        throw std::invalid_argument("boundary transfer mode must be pageable or pinned");
      }
      arguments.pinned = value == "pinned";
    } else if (flag == "--device-id") {
      const auto result =
          std::from_chars(value.data(), value.data() + value.size(), arguments.device_id);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
          arguments.device_id < 0) {
        throw std::invalid_argument("device ID must be a nonnegative integer");
      }
#endif
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(flag));
    }
  }
  if (arguments.listen.empty() || arguments.worker_id.empty() || arguments.model_root.empty() ||
      arguments.memory_limit_bytes == 0U) {
    throw std::invalid_argument(
        "required arguments: --listen, --worker-id, "
        "--model-root, --memory-limit-bytes");
  }
#ifdef HLLM_WORKER_CUDA
  if (arguments.pinned && arguments.pinned_memory_limit_bytes == 0U) {
    throw std::invalid_argument("pinned transfer mode requires --pinned-host-memory-limit-bytes");
  }
  if (arguments.device_memory_limit_bytes == 0U) {
    throw std::invalid_argument("CUDA worker requires --device-memory-limit-bytes");
  }
#endif
  return arguments;
}

}  // namespace

int main(const int argc, char** const argv) {
  try {
    const auto arguments = parse_arguments(argc, argv);
#ifdef HLLM_WORKER_CUDA
    auto factory = hllm::cuda::make_backend_factory(arguments.device_id, arguments.pinned);
#elif defined(HLLM_WORKER_MLX)
    auto factory = hllm::mlx::make_backend_factory(arguments.memory_limit_bytes);
#else
    auto factory = hllm::cpu::make_backend_factory();
#endif
    hllm::worker::ControlService control(
        {
            .worker_id = arguments.worker_id,
            .endpoint = arguments.listen,
            .model_root = arguments.model_root,
#ifdef HLLM_WORKER_MLX
            .host_memory_capacity_bytes = 0U,
#else
            .host_memory_capacity_bytes = arguments.memory_limit_bytes,
#endif
            .device_memory_capacity_bytes = arguments.device_memory_limit_bytes,
            .pinned_host_memory_capacity_bytes = arguments.pinned_memory_limit_bytes,
#ifdef HLLM_WORKER_MLX
            .unified_memory_capacity_bytes = arguments.memory_limit_bytes,
#endif
        },
        std::move(factory));

    hllm::worker::GenerationService generation(control);
    hllm::worker::ExecutionService execution(control);
    grpc::ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(hllm::worker::kMaximumRpcBytes);
    builder.SetMaxSendMessageSize(hllm::worker::kMaximumRpcBytes);
    builder.AddListeningPort(arguments.listen, grpc::InsecureServerCredentials());
    builder.RegisterService(&control);
    builder.RegisterService(&generation);
    builder.RegisterService(&execution);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (server == nullptr) {
      std::cerr << "failed to start worker on " << arguments.listen << '\n';
      return 2;
    }
    std::cout << kWorkerName << " listening " << arguments.worker_id << ' ' << arguments.listen
              << std::endl;
    server->Wait();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << kWorkerName << ": " << error.what() << '\n';
    return 2;
  }
}
