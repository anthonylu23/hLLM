#include "hllm/worker/control_service.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "hllm/runtime/safetensors.hpp"

namespace hllm::worker {
namespace {

void reject(v1::LoadStageResponse& response, const v1::ErrorCode code,
            const std::string_view detail) {
  response.set_accepted(false);
  response.set_detail(detail);
  auto* error = response.mutable_error();
  error->set_code(code);
  error->set_detail(detail);
}

void reject(v1::ReserveResponse& response, const v1::ErrorCode code,
            const std::string_view detail) {
  response.set_accepted(false);
  response.set_detail(detail);
  auto* error = response.mutable_error();
  error->set_code(code);
  error->set_detail(detail);
}

[[nodiscard]] bool safe_file_name(const std::string_view name) {
  const std::filesystem::path path(name);
  return !name.empty() && !path.is_absolute() && !path.has_parent_path() && name != "." &&
         name != "..";
}

[[nodiscard]] bool tensor_is_selected(const v1::TensorRecord& tensor,
                                      const v1::StageAssignment& stage,
                                      const bool tied_embeddings) {
  switch (tensor.role()) {
    case v1::TENSOR_ROLE_TOKEN_EMBEDDING:
      return stage.owns_token_embedding() || (stage.owns_lm_head() && tied_embeddings);
    case v1::TENSOR_ROLE_TRANSFORMER_LAYER:
      return tensor.has_layer_index() && tensor.layer_index() >= stage.layer_start() &&
             tensor.layer_index() < stage.layer_end();
    case v1::TENSOR_ROLE_FINAL_NORM:
      return stage.owns_final_norm();
    case v1::TENSOR_ROLE_LM_HEAD:
      return stage.owns_lm_head();
    case v1::TENSOR_ROLE_ARCHITECTURE_STATE:
      return true;
    case v1::TENSOR_ROLE_UNSPECIFIED:
      return false;
    default:
      return false;
  }
}

[[nodiscard]] std::optional<runtime::DataType> runtime_dtype(const v1::DataType dtype) {
  switch (dtype) {
    case v1::DATA_TYPE_BOOL:
      return runtime::DataType::kBool;
    case v1::DATA_TYPE_U8:
      return runtime::DataType::kU8;
    case v1::DATA_TYPE_I8:
      return runtime::DataType::kI8;
    case v1::DATA_TYPE_I16:
      return runtime::DataType::kI16;
    case v1::DATA_TYPE_U16:
      return runtime::DataType::kU16;
    case v1::DATA_TYPE_I32:
      return runtime::DataType::kI32;
    case v1::DATA_TYPE_U32:
      return runtime::DataType::kU32;
    case v1::DATA_TYPE_I64:
      return runtime::DataType::kI64;
    case v1::DATA_TYPE_U64:
      return runtime::DataType::kU64;
    case v1::DATA_TYPE_F16:
      return runtime::DataType::kF16;
    case v1::DATA_TYPE_BF16:
      return runtime::DataType::kBF16;
    case v1::DATA_TYPE_F32:
      return runtime::DataType::kF32;
    case v1::DATA_TYPE_F64:
      return runtime::DataType::kF64;
    case v1::DATA_TYPE_UNSPECIFIED:
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

}  // namespace

ControlService::ControlService(WorkerConfig config) : config_(std::move(config)) {
  if (config_.worker_id.empty() || config_.endpoint.empty() ||
      !std::filesystem::is_directory(config_.model_root) ||
      config_.host_memory_capacity_bytes == 0U) {
    throw std::invalid_argument("worker configuration is incomplete or invalid");
  }
  config_.model_root = std::filesystem::canonical(config_.model_root);
}

grpc::Status ControlService::GetCapabilities(grpc::ServerContext*, const v1::Empty*,
                                             v1::Capabilities* const response) {
  auto* profile = response->mutable_worker();
  profile->mutable_schema_version()->set_major(1U);
  profile->mutable_schema_version()->set_minor(0U);
  profile->set_worker_id(config_.worker_id);
  profile->set_endpoint(config_.endpoint);
  profile->set_backend(v1::BACKEND_CPU);
  profile->set_primary_memory_domain(v1::MEMORY_DOMAIN_HOST);
  profile->add_supported_architectures("llama.v1");
  profile->add_supported_execution_dtypes(v1::DATA_TYPE_F16);
  profile->add_supported_execution_dtypes(v1::DATA_TYPE_BF16);
  profile->add_supported_execution_dtypes(v1::DATA_TYPE_F32);
  auto* budget = profile->add_memory_budgets();
  budget->set_domain(v1::MEMORY_DOMAIN_HOST);
  budget->set_capacity_bytes(config_.host_memory_capacity_bytes);
  profile->set_activation_buffer_count(2U);
  profile->set_provenance(v1::PROVENANCE_CONFIGURED);
  return grpc::Status::OK;
}

std::optional<std::string> ControlService::validate_stage(
    const v1::LoadStageRequest& request, std::size_t& selected_weight_bytes) const {
  if (!request.has_plan() || !request.has_manifest()) {
    return "load request must include a plan and manifest";
  }
  const auto& plan = request.plan();
  const auto& manifest = request.manifest();
  if (plan.plan_id().empty() || plan.plan_digest().empty() || plan.deployment_version() == 0U ||
      manifest.manifest_digest().empty() || plan.manifest_digest() != manifest.manifest_digest()) {
    return "plan and manifest identities are incomplete or inconsistent";
  }
  if (request.stage_index() >= static_cast<std::uint32_t>(plan.stages_size())) {
    return "stage index is outside the deployment plan";
  }
  const auto& stage = plan.stages(static_cast<int>(request.stage_index()));
  if (stage.stage_index() != request.stage_index() || stage.worker_id() != config_.worker_id ||
      stage.layer_start() >= stage.layer_end() ||
      stage.layer_end() > manifest.config().num_layers()) {
    return "stage assignment is invalid or belongs to another worker";
  }
  if (manifest.architecture().architecture_id() != "llama.v1" ||
      manifest.config().hidden_activation() != "silu" || manifest.config().attention_bias() ||
      manifest.config().mlp_bias() || manifest.config().has_rope_scaling()) {
    return "manifest requests unsupported CPU reference semantics";
  }

  std::unordered_map<std::string, runtime::SafetensorsFile> inspected_files;
  try {
    for (const auto& tensor : manifest.tensors()) {
      if (!tensor_is_selected(tensor, stage, manifest.config().tied_embeddings())) {
        continue;
      }
      if (!safe_file_name(tensor.file())) {
        return "manifest tensor file name is not a safe relative file";
      }
      const auto dtype = runtime_dtype(tensor.dtype());
      if (!dtype.has_value()) {
        return "manifest contains an unsupported tensor dtype";
      }
      auto file = inspected_files.find(tensor.file());
      if (file == inspected_files.end()) {
        file = inspected_files
                   .emplace(tensor.file(),
                            runtime::SafetensorsFile(config_.model_root / tensor.file()))
                   .first;
      }
      const auto& native = file->second.tensor(tensor.name());
      if (native.dtype != *dtype || native.data_offset != tensor.data_offset() ||
          native.byte_length != tensor.byte_length() ||
          native.shape.size() != static_cast<std::size_t>(tensor.shape_size()) ||
          !std::equal(native.shape.begin(), native.shape.end(), tensor.shape().begin())) {
        return "native Safetensors metadata does not match the manifest";
      }
      if (native.byte_length > std::numeric_limits<std::size_t>::max() - selected_weight_bytes) {
        return "selected stage weight size overflows native accounting";
      }
      selected_weight_bytes += native.byte_length;
    }
  } catch (const runtime::SafetensorsError& error) {
    return std::string("cannot validate selected Safetensors weights: ") + error.what();
  }
  if (selected_weight_bytes == 0U) {
    return "stage assignment did not select any weights";
  }
  return std::nullopt;
}

grpc::Status ControlService::LoadStage(grpc::ServerContext*,
                                       const v1::LoadStageRequest* const request,
                                       v1::LoadStageResponse* const response) {
  std::size_t selected_weight_bytes = 0U;
  if (const auto problem = validate_stage(*request, selected_weight_bytes); problem.has_value()) {
    reject(*response, v1::ERROR_CODE_INCOMPATIBLE_WORKER, *problem);
    return grpc::Status::OK;
  }

  std::scoped_lock lock(mutex_);
  if (deployment_.has_value() &&
      (deployment_->plan_id != request->plan().plan_id() ||
       deployment_->deployment_version != request->plan().deployment_version() ||
       deployment_->plan_digest != request->plan().plan_digest() ||
       deployment_->stage_index != request->stage_index())) {
    reject(*response, v1::ERROR_CODE_STALE_DEPLOYMENT,
           "worker already has a different deployment loaded");
    return grpc::Status::OK;
  }
  if (deployment_.has_value()) {
    response->set_accepted(true);
    response->set_detail("stage already loaded");
    return grpc::Status::OK;
  }
  deployment_ = DeploymentState{
      .plan_id = request->plan().plan_id(),
      .plan_digest = request->plan().plan_digest(),
      .deployment_version = request->plan().deployment_version(),
      .stage_index = request->stage_index(),
      .active_requests = {},
  };
  response->set_accepted(true);
  response->set_detail("stage metadata and selected weights validated");
  return grpc::Status::OK;
}

bool ControlService::deployment_matches(const std::string_view plan_id,
                                        const std::uint64_t deployment_version) const {
  return deployment_.has_value() && deployment_->plan_id == plan_id &&
         deployment_->deployment_version == deployment_version;
}

grpc::Status ControlService::UnloadStage(grpc::ServerContext*,
                                         const v1::UnloadStageRequest* const request,
                                         v1::Empty*) {
  std::scoped_lock lock(mutex_);
  if (!deployment_.has_value()) {
    return grpc::Status::OK;
  }
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "stale deployment identity"};
  }
  if (!deployment_->active_requests.empty()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "deployment still has active requests"};
  }
  deployment_.reset();
  return grpc::Status::OK;
}

grpc::Status ControlService::ReserveRequest(grpc::ServerContext*,
                                            const v1::ReserveRequestMessage* const request,
                                            v1::ReserveResponse* const response) {
  std::scoped_lock lock(mutex_);
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    reject(*response, v1::ERROR_CODE_STALE_DEPLOYMENT, "request targets a stale deployment");
  } else if (request->request_id().empty() || request->maximum_total_tokens() == 0U) {
    reject(*response, v1::ERROR_CODE_INVALID_REQUEST, "request reservation is incomplete");
  } else if (!deployment_->active_requests.empty() &&
             !deployment_->active_requests.contains(request->request_id())) {
    reject(*response, v1::ERROR_CODE_RESOURCE_EXHAUSTED,
           "CPU reference worker supports one active request");
  } else {
    deployment_->active_requests.insert(request->request_id());
    response->set_accepted(true);
    response->set_detail("request capacity reserved");
  }
  return grpc::Status::OK;
}

grpc::Status ControlService::CancelRequest(grpc::ServerContext*,
                                           const v1::CancelRequestMessage* const request,
                                           v1::Empty*) {
  std::scoped_lock lock(mutex_);
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "stale deployment identity"};
  }
  deployment_->active_requests.erase(request->request_id());
  return grpc::Status::OK;
}

grpc::Status ControlService::GetMemoryReport(grpc::ServerContext*, const v1::Empty*,
                                             v1::MemoryReport* const response) {
  auto* budget = response->add_budgets();
  budget->set_domain(v1::MEMORY_DOMAIN_HOST);
  budget->set_capacity_bytes(config_.host_memory_capacity_bytes);
  return grpc::Status::OK;
}

grpc::Status ControlService::GetMetrics(grpc::ServerContext*, const v1::Empty*,
                                        v1::WorkerMetrics* const response) {
  response->set_worker_id(config_.worker_id);
  return grpc::Status::OK;
}

grpc::Status ControlService::Health(grpc::ServerContext*, const v1::Empty*,
                                    v1::HealthResponse* const response) {
  response->set_serving(true);
  std::scoped_lock lock(mutex_);
  response->set_detail(deployment_.has_value() ? "serving with a loaded CPU stage"
                                               : "serving without a loaded stage");
  return grpc::Status::OK;
}

}  // namespace hllm::worker
