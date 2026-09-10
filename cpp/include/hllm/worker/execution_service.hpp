#pragma once

#include "execution.grpc.pb.h"
#include "hllm/worker/boundary_codec.hpp"
#include "hllm/worker/control_service.hpp"

namespace hllm::worker {


class GenerationService final : public v1::Generation::Service {
 public:
  explicit GenerationService(ControlService& control) : control_(control) {}
  grpc::Status Generate(grpc::ServerContext*, const v1::GenerationRequest*,
                        grpc::ServerWriter<v1::GenerationEvent>*) override;

 private:
  ControlService& control_;
};

class ExecutionService final : public v1::StageExecution::Service {
 public:
  explicit ExecutionService(ControlService& control) : control_(control) {}
  grpc::Status Execute(grpc::ServerContext*,
                       grpc::ServerReaderWriter<v1::StageMessage, v1::StageMessage>*) override;

 private:
  ControlService& control_;
};

}  // namespace hllm::worker
