#include "hllm/worker/control_service.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

#include "hllm/cpu/stage.hpp"

namespace hllm::worker {
namespace {

void validate_plan(const v1::LoadStageRequest& request, const std::string& worker) {
  const auto& plan = request.plan();
  const auto& manifest = request.manifest();
  if (!request.has_plan() || !request.has_manifest() || plan.schema_version().major() != 1U ||
      plan.schema_version().minor() != 0U || manifest.schema_version().major() != 1U ||
      manifest.schema_version().minor() > 1U || plan.plan_id().empty() ||
      plan.plan_digest().empty() || plan.deployment_version() == 0U ||
      manifest.manifest_digest().empty() || plan.manifest_digest() != manifest.manifest_digest()) {
    throw std::invalid_argument("unsupported schema or inconsistent deployment identity");
  }
  if (plan.execution_dtype() != v1::DATA_TYPE_F32 || plan.activation_dtype() != v1::DATA_TYPE_F16) {
    throw std::invalid_argument("CPU execution requires F32 compute and F16 boundaries");
  }
  if ((plan.stages_size() != 1 && plan.stages_size() != 2) ||
      request.stage_index() >= static_cast<std::uint32_t>(plan.stages_size())) {
    throw std::invalid_argument("CPU pipeline supports one or two stages");
  }
  std::uint32_t end = 0U;
  std::set<std::string> workers;
  for (int i = 0; i < plan.stages_size(); ++i) {
    const auto& stage = plan.stages(i);
    const bool first = i == 0;
    const bool last = i == plan.stages_size() - 1;
    if (stage.stage_index() != static_cast<std::uint32_t>(i) || stage.worker_id().empty() ||
        !workers.insert(stage.worker_id()).second || stage.layer_start() != end ||
        stage.layer_end() <= end || stage.layer_end() > manifest.config().num_layers() ||
        stage.owns_token_embedding() != first || stage.owns_lm_head() != last ||
        stage.owns_final_norm() != last || stage.owns_sampling() != last) {
      throw std::invalid_argument("invalid stage partition or tensor ownership");
    }
    end = stage.layer_end();
  }
  if (end != manifest.config().num_layers() ||
      plan.stages(static_cast<int>(request.stage_index())).worker_id() != worker) {
    throw std::invalid_argument("incomplete partition or wrong worker");
  }
  if (plan.stages_size() == 2 && plan.split_layer() != plan.stages(0).layer_end()) {
    throw std::invalid_argument("split layer does not match assignments");
  }
  if (request.stage_endpoints_size() != plan.stages_size()) {
    throw std::invalid_argument("each stage requires an endpoint");
  }
  std::set<std::uint32_t> indices;
  std::set<std::string> endpoints;
  for (const auto& endpoint : request.stage_endpoints()) {
    if (endpoint.stage_index() >= static_cast<std::uint32_t>(plan.stages_size()) ||
        !indices.insert(endpoint.stage_index()).second || endpoint.endpoint().empty() ||
        !endpoints.insert(endpoint.endpoint()).second ||
        endpoint.worker_id() != plan.stages(static_cast<int>(endpoint.stage_index())).worker_id()) {
      throw std::invalid_argument("invalid stage endpoint mapping");
    }
  }
  if (manifest.config().tied_embeddings() && plan.stages_size() == 2 &&
      std::find(plan.duplicated_tensor_groups().begin(), plan.duplicated_tensor_groups().end(),
                "token_embeddings") == plan.duplicated_tensor_groups().end()) {
    throw std::invalid_argument("split tied embedding must be explicitly duplicated");
  }
}

template <class Response>
void reject(Response* response, v1::ErrorCode code, const std::string& detail) {
  response->set_accepted(false);
  response->set_detail(detail);
  response->mutable_error()->set_code(code);
  response->mutable_error()->set_detail(detail);
}

}  // namespace

ControlService::ControlService(WorkerConfig config) : config_(std::move(config)) {
  if (config_.worker_id.empty() || config_.endpoint.empty() ||
      !std::filesystem::is_directory(config_.model_root) ||
      config_.host_memory_capacity_bytes == 0U) {
    throw std::invalid_argument("worker configuration is incomplete");
  }
  config_.model_root = std::filesystem::canonical(config_.model_root);
  reservation_reaper_ = std::jthread([this](std::stop_token stop) {
    while (!stop.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      std::scoped_lock lock(mutex_);
      prune_expired();
    }
  });
}

grpc::Status ControlService::GetCapabilities(grpc::ServerContext*, const v1::Empty*,
                                             v1::Capabilities* out) {
  auto* profile = out->mutable_worker();
  profile->mutable_schema_version()->set_major(1U);
  profile->set_worker_id(config_.worker_id);
  profile->set_endpoint(config_.endpoint);
  profile->set_backend(v1::BACKEND_CPU);
  profile->set_primary_memory_domain(v1::MEMORY_DOMAIN_HOST);
  profile->add_supported_architectures("llama.v1");
  profile->add_supported_architectures("qwen3.v1");
  profile->add_supported_execution_dtypes(v1::DATA_TYPE_F32);
  auto* budget = profile->add_memory_budgets();
  budget->set_domain(v1::MEMORY_DOMAIN_HOST);
  budget->set_capacity_bytes(config_.host_memory_capacity_bytes);
  profile->set_provenance(v1::PROVENANCE_CONFIGURED);
  return grpc::Status::OK;
}

void ControlService::prune_expired() {
  if (active_ && std::chrono::system_clock::now() >= active_->deadline) {
    active_->cancelled.store(true);
    if (!active_->running) {
      active_.reset();
    }
  }
}
bool ControlService::deployment_matches(const std::string& id, std::uint64_t version) const {
  return deployment_ && deployment_->spec.plan().plan_id() == id &&
         deployment_->spec.plan().deployment_version() == version;
}

grpc::Status ControlService::LoadStage(grpc::ServerContext*, const v1::LoadStageRequest* request,
                                       v1::LoadStageResponse* response) {
  std::scoped_lock lock(mutex_);
  prune_expired();
  if (deployment_) {
    // Exact retries preserve executable weights, reservation and all KV state.
    if (deployment_->spec.SerializeAsString() == request->SerializeAsString()) {
      response->set_accepted(true);
      response->set_detail("executable stage already loaded");
    } else {
      reject(response, v1::ERROR_CODE_STALE_DEPLOYMENT, "unload the existing deployment first");
    }
    return grpc::Status::OK;
  }
  try {
    validate_plan(*request, config_.worker_id);
    auto next = std::make_shared<LoadedDeployment>();
    next->spec = *request;
    next->backend =
        cpu::load_stage(*request, config_.model_root, config_.host_memory_capacity_bytes);
    deployment_ = std::move(next);
    response->set_accepted(true);
    response->set_detail("executable CPU stage loaded");
  } catch (const std::bad_alloc&) {
    reject(response, v1::ERROR_CODE_RESOURCE_EXHAUSTED, "weight allocation failed");
  } catch (const std::length_error& error) {
    reject(response, v1::ERROR_CODE_RESOURCE_EXHAUSTED, error.what());
  } catch (const std::exception& error) {
    reject(response, v1::ERROR_CODE_INCOMPATIBLE_WORKER, error.what());
  }
  return grpc::Status::OK;
}

std::shared_ptr<ActiveRequest> ControlService::reserve(const std::string& id, std::size_t tokens,
                                                       std::uint64_t deadline_ms) {
  prune_expired();
  if (id.empty() || id.size() > 256U) {
    throw std::invalid_argument("invalid request ID");
  }
  const auto now = std::chrono::system_clock::now();
  auto deadline = now + std::chrono::seconds(60);
  if (deadline_ms != 0U) {
    const auto now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    if (deadline_ms <= static_cast<std::uint64_t>(now_ms) ||
        deadline_ms - static_cast<std::uint64_t>(now_ms) > 3'600'000U) {
      throw std::invalid_argument("deadline must be in the next hour");
    }
    deadline = std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms));
  }
  if (active_) {
    if (active_->id != id) {
      throw std::length_error("worker already has an active request");
    }
    if (active_->maximum_tokens != tokens || active_->cancelled.load()) {
      throw std::invalid_argument("conflicting request reservation retry");
    }
    if (!active_->running) {
      active_->deadline = std::min(active_->deadline, deadline);
    }
    return active_;
  }
  const auto memory = deployment_->backend->sequence_memory(tokens);
  const auto available = config_.host_memory_capacity_bytes - deployment_->backend->weight_bytes();
  if (memory.cache_bytes > available || memory.workspace_bytes > available - memory.cache_bytes) {
    throw std::length_error("request cache and workspace exceed host memory budget");
  }
  auto next = std::make_shared<ActiveRequest>();
  next->id = id;
  next->maximum_tokens = tokens;
  next->deadline = deadline;
  next->memory = memory;
  next->sequence = deployment_->backend->allocate_sequence(tokens);
  active_ = next;
  return next;
}

ExecutionLease ControlService::acquire(const std::string& id, std::uint64_t version,
                                       const std::string& request, std::size_t tokens,
                                       std::uint64_t deadline, std::uint32_t required_stage) {
  std::scoped_lock lock(mutex_);
  if (!deployment_matches(id, version) || deployment_->spec.stage_index() != required_stage) {
    throw std::invalid_argument("stale deployment or incorrect execution stage");
  }
  auto state = reserve(request, tokens, deadline);
  if (state->running) {
    throw std::length_error("request already executing");
  }
  state->running = true;
  return {deployment_, state};
}
void ControlService::release(const std::shared_ptr<ActiveRequest>& state) {
  std::scoped_lock lock(mutex_);
  if (active_ == state) {
    // The caller has stopped using the backend state before admission is
    // reopened.
    active_->sequence.reset();
    active_.reset();
  }
}

grpc::Status ControlService::ReserveRequest(grpc::ServerContext*,
                                            const v1::ReserveRequestMessage* request,
                                            v1::ReserveResponse* response) {
  std::scoped_lock lock(mutex_);
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    reject(response, v1::ERROR_CODE_STALE_DEPLOYMENT, "stale deployment");
    return grpc::Status::OK;
  }
  try {
    static_cast<void>(reserve(request->request_id(), request->maximum_total_tokens(),
                              request->deadline_unix_ms()));
    response->set_accepted(true);
    response->set_detail("CPU cache allocated and workspace reserved");
  } catch (const std::bad_alloc&) {
    reject(response, v1::ERROR_CODE_RESOURCE_EXHAUSTED, "cache allocation failed");
  } catch (const std::length_error& error) {
    reject(response, v1::ERROR_CODE_RESOURCE_EXHAUSTED, error.what());
  } catch (const std::exception& error) {
    reject(response, v1::ERROR_CODE_INVALID_REQUEST, error.what());
  }
  return grpc::Status::OK;
}

grpc::Status ControlService::CancelRequest(grpc::ServerContext*,
                                           const v1::CancelRequestMessage* request, v1::Empty*) {
  std::scoped_lock lock(mutex_);
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "stale deployment"};
  }
  if (active_ && active_->id == request->request_id()) {
    active_->cancelled.store(true);
    if (!active_->running) {
      active_.reset();
    }
  }
  return grpc::Status::OK;
}
grpc::Status ControlService::UnloadStage(grpc::ServerContext*,
                                         const v1::UnloadStageRequest* request, v1::Empty*) {
  std::scoped_lock lock(mutex_);
  prune_expired();
  if (!deployment_) {
    return grpc::Status::OK;
  }
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "stale deployment"};
  }
  if (active_) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "deployment has an active request"};
  }
  deployment_.reset();
  return grpc::Status::OK;
}
grpc::Status ControlService::GetMemoryReport(grpc::ServerContext*, const v1::Empty*,
                                             v1::MemoryReport* response) {
  std::scoped_lock lock(mutex_);
  prune_expired();
  auto* budget = response->add_budgets();
  budget->set_domain(v1::MEMORY_DOMAIN_HOST);
  budget->set_capacity_bytes(config_.host_memory_capacity_bytes);
  if (deployment_) {
    response->set_loaded_weight_bytes(deployment_->backend->weight_bytes());
  }
  if (active_) {
    response->set_active_requests(1U);
    response->set_reserved_cache_bytes(active_->memory.cache_bytes);
    response->set_reserved_workspace_bytes(active_->memory.workspace_bytes);
  }
  return grpc::Status::OK;
}
grpc::Status ControlService::GetMetrics(grpc::ServerContext*, const v1::Empty*,
                                        v1::WorkerMetrics* response) {
  response->set_worker_id(config_.worker_id);
  return grpc::Status::OK;
}
grpc::Status ControlService::Health(grpc::ServerContext*, const v1::Empty*,
                                    v1::HealthResponse* response) {
  std::scoped_lock lock(mutex_);
  response->set_serving(true);
  response->set_detail(deployment_ ? "executable CPU stage ready"
                                   : "worker ready; no stage loaded");
  return grpc::Status::OK;
}

}  // namespace hllm::worker
