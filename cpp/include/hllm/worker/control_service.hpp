#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "control.grpc.pb.h"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::worker {

struct WorkerConfig {
  std::string worker_id;
  std::string endpoint;
  std::filesystem::path model_root;
  std::uint64_t host_memory_capacity_bytes;
};

struct LoadedDeployment {
  v1::LoadStageRequest spec;
  std::unique_ptr<runtime::StageBackend> backend;
};

struct ActiveRequest {
  std::string id;
  std::size_t maximum_tokens;
  std::chrono::system_clock::time_point deadline;
  runtime::SequenceMemory memory;
  std::unique_ptr<runtime::SequenceState> sequence;
  std::atomic_bool cancelled{false};
  bool running{false};  // Protected by the control mutex; sequence has one
                        // execution owner.
};

struct ExecutionLease {
  std::shared_ptr<LoadedDeployment> deployment;
  std::shared_ptr<ActiveRequest> request;
};

class ControlService final : public v1::WorkerControl::Service {
 public:
  explicit ControlService(WorkerConfig config);
  grpc::Status GetCapabilities(grpc::ServerContext*, const v1::Empty*, v1::Capabilities*) override;
  grpc::Status LoadStage(grpc::ServerContext*, const v1::LoadStageRequest*,
                         v1::LoadStageResponse*) override;
  grpc::Status UnloadStage(grpc::ServerContext*, const v1::UnloadStageRequest*,
                           v1::Empty*) override;
  grpc::Status ReserveRequest(grpc::ServerContext*, const v1::ReserveRequestMessage*,
                              v1::ReserveResponse*) override;
  grpc::Status CancelRequest(grpc::ServerContext*, const v1::CancelRequestMessage*,
                             v1::Empty*) override;
  grpc::Status GetMemoryReport(grpc::ServerContext*, const v1::Empty*, v1::MemoryReport*) override;
  grpc::Status GetMetrics(grpc::ServerContext*, const v1::Empty*, v1::WorkerMetrics*) override;
  grpc::Status Health(grpc::ServerContext*, const v1::Empty*, v1::HealthResponse*) override;

  // Atomically claims a prior reservation, or reserves and claims for native
  // execution.
  ExecutionLease acquire(const std::string& deployment, std::uint64_t version,
                         const std::string& request, std::size_t tokens, std::uint64_t deadline_ms,
                         std::uint32_t required_stage);
  void release(const std::shared_ptr<ActiveRequest>& request);

 private:
  void prune_expired();
  bool deployment_matches(const std::string&, std::uint64_t) const;
  std::shared_ptr<ActiveRequest> reserve(const std::string&, std::size_t, std::uint64_t);
  WorkerConfig config_;
  mutable std::mutex mutex_;
  std::shared_ptr<LoadedDeployment> deployment_;
  std::shared_ptr<ActiveRequest> active_;
  std::jthread reservation_reaper_;
};

}  // namespace hllm::worker
