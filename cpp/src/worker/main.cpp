#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>

#include "hllm/worker/control_service.hpp"

namespace {

struct Arguments {
  std::string listen;
  std::string worker_id;
  std::filesystem::path model_root;
  std::uint64_t memory_limit_bytes{0U};
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
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(flag));
    }
  }
  if (arguments.listen.empty() || arguments.worker_id.empty() || arguments.model_root.empty() ||
      arguments.memory_limit_bytes == 0U) {
    throw std::invalid_argument(
        "required arguments: --listen, --worker-id, --model-root, --memory-limit-bytes");
  }
  return arguments;
}

}  // namespace

int main(const int argc, char** const argv) {
  try {
    const auto arguments = parse_arguments(argc, argv);
    hllm::worker::ControlService control({
        .worker_id = arguments.worker_id,
        .endpoint = arguments.listen,
        .model_root = arguments.model_root,
        .host_memory_capacity_bytes = arguments.memory_limit_bytes,
    });

    grpc::ServerBuilder builder;
    builder.AddListeningPort(arguments.listen, grpc::InsecureServerCredentials());
    builder.RegisterService(&control);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (server == nullptr) {
      std::cerr << "failed to start CPU worker on " << arguments.listen << '\n';
      return 2;
    }
    std::cout << "hllm-worker-cpu ready " << arguments.worker_id << ' ' << arguments.listen
              << std::endl;
    server->Wait();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hllm-worker-cpu: " << error.what() << '\n';
    return 2;
  }
}
