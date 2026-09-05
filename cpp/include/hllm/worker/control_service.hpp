#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>

#include <grpcpp/grpcpp.h>

#include "control.grpc.pb.h"

namespace hllm::worker {

struct WorkerConfig {
  std::string worker_id;
  std::string endpoint;
  std::filesystem::path model_root;
  std::uint64_t host_memory_capacity_bytes;
};

class ControlService final : public v1::WorkerControl::Service {
 public:
  explicit ControlService(WorkerConfig config);

  grpc::Status GetCapabilities(grpc::ServerContext* context, const v1::Empty* request,
                               v1::Capabilities* response) override;
  grpc::Status LoadStage(grpc::ServerContext* context, const v1::LoadStageRequest* request,
                         v1::LoadStageResponse* response) override;
  grpc::Status UnloadStage(grpc::ServerContext* context,
                           const v1::UnloadStageRequest* request,
                           v1::Empty* response) override;
  grpc::Status ReserveRequest(grpc::ServerContext* context,
                              const v1::ReserveRequestMessage* request,
                              v1::ReserveResponse* response) override;
  grpc::Status CancelRequest(grpc::ServerContext* context,
                             const v1::CancelRequestMessage* request,
                             v1::Empty* response) override;
  grpc::Status GetMemoryReport(grpc::ServerContext* context, const v1::Empty* request,
                               v1::MemoryReport* response) override;
  grpc::Status GetMetrics(grpc::ServerContext* context, const v1::Empty* request,
                          v1::WorkerMetrics* response) override;
  grpc::Status Health(grpc::ServerContext* context, const v1::Empty* request,
                      v1::HealthResponse* response) override;

 private:
  struct DeploymentState {
    std::string plan_id;
    std::string plan_digest;
    std::uint64_t deployment_version;
    std::uint32_t stage_index;
    std::unordered_set<std::string> active_requests;
  };

  [[nodiscard]] std::optional<std::string> validate_stage(
      const v1::LoadStageRequest& request, std::size_t& selected_weight_bytes) const;
  [[nodiscard]] bool deployment_matches(std::string_view plan_id,
                                        std::uint64_t deployment_version) const;

  WorkerConfig config_;
  mutable std::mutex mutex_;
  std::optional<DeploymentState> deployment_;
};

}  // namespace hllm::worker
