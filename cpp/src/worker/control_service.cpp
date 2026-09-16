#include "hllm/worker/control_service.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"
#include "qualification.hpp"

namespace hllm::worker {
namespace {

v1::ErrorCode wire_code(runtime::ErrorCode code) {
  switch (code) {
    case runtime::ErrorCode::kInvalidRequest:
      return v1::ERROR_CODE_INVALID_REQUEST;
    case runtime::ErrorCode::kIncompatibleWorker:
      return v1::ERROR_CODE_INCOMPATIBLE_WORKER;
    case runtime::ErrorCode::kResourceExhausted:
      return v1::ERROR_CODE_RESOURCE_EXHAUSTED;
    case runtime::ErrorCode::kDeadlineExceeded:
      return v1::ERROR_CODE_DEADLINE_EXCEEDED;
    case runtime::ErrorCode::kCancelled:
      return v1::ERROR_CODE_CANCELLED;
    case runtime::ErrorCode::kInternal:
      break;
  }
  return v1::ERROR_CODE_BACKEND_ERROR;
}

void validate_plan(const v1::LoadStageRequest& request, const std::string& worker,
                   const runtime::BackendCapabilities& capabilities) {
  const auto& plan = request.plan();
  const auto& manifest = request.manifest();
  if (!request.has_plan() || !request.has_manifest() || plan.schema_version().major() != 1U ||
      plan.schema_version().minor() > 2U || manifest.schema_version().major() != 1U ||
      manifest.schema_version().minor() > 1U || plan.plan_id().empty() ||
      plan.plan_digest().empty() || plan.deployment_version() == 0U ||
      manifest.manifest_digest().empty() || plan.manifest_digest() != manifest.manifest_digest()) {
    throw runtime::Error::incompatible_worker(
        "unsupported schema or inconsistent deployment identity");
  }
  const auto digest = [](const std::string& value) {
    return value.size() == 64U && value.find_first_not_of("0123456789abcdef") == std::string::npos;
  };
  const bool mixed = plan.has_weight_dtype();
  if (mixed != (plan.schema_version().minor() == 2U) ||
      (mixed && (!capabilities.supports_mixed_precision ||
                 plan.weight_dtype() != v1::DATA_TYPE_F16 ||
                 plan.execution_dtype() != v1::DATA_TYPE_F32))) {
    throw runtime::Error::incompatible_worker("unsupported resident/execution precision contract");
  }
  const bool measured = plan.planning_mode() == "measured";
  if (measured && (plan.schema_version().minor() != (mixed ? 2U : 1U) ||
                  !digest(plan.workload_digest()) || !digest(plan.profile_bundle_digest()))) {
    throw runtime::Error::incompatible_worker(
        "measured plan requires versioned workload/profile identities");
  }
  if (!measured && !mixed && plan.schema_version().minor() != 0U) {
    throw runtime::Error::incompatible_worker("unsupported feasibility plan schema version");
  }
  if (!measured && (!plan.workload_digest().empty() || !plan.profile_bundle_digest().empty())) {
    throw runtime::Error::incompatible_worker("feasibility plan cannot carry measured identities");
  }
  if (measured || mixed) {
    nlohmann::json stages = nlohmann::json::array();
    for (const auto& s : plan.stages()) stages.push_back({
      {"stage_index", s.stage_index()}, {"worker_id", s.worker_id()},
      {"layer_start", s.layer_start()}, {"layer_end", s.layer_end()},
      {"owns_token_embedding", s.owns_token_embedding()}, {"owns_final_norm", s.owns_final_norm()},
      {"owns_lm_head", s.owns_lm_head()}, {"owns_sampling", s.owns_sampling()}});
    nlohmann::json unsigned_plan = {
      {"schema_version", mixed ? "1.2" : "1.1"}, {"planner_version", plan.planner_version()},
      {"deployment_version", plan.deployment_version()}, {"manifest_digest", plan.manifest_digest()},
      {"workload_id", plan.workload_id()}, {"planning_mode", plan.planning_mode()},
      {"execution_dtype", v1::DataType_Name(plan.execution_dtype()).substr(10)},
      {"activation_dtype", v1::DataType_Name(plan.activation_dtype()).substr(10)},
      {"split_layer", plan.split_layer()}, {"stages", stages},
      {"duplicated_tensor_groups", std::vector<std::string>(plan.duplicated_tensor_groups().begin(), plan.duplicated_tensor_groups().end())},
      {"selected_candidate_id", plan.selected_candidate_id()},
      {"workload_digest", plan.workload_digest()}, {"profile_bundle_digest", plan.profile_bundle_digest()}};
    if (mixed) unsigned_plan["weight_dtype"] = "F16";
    if (!measured) {
      unsigned_plan["workload_digest"] = nullptr;
      unsigned_plan["profile_bundle_digest"] = nullptr;
    }
    if (text_digest(unsigned_plan.dump()) != plan.plan_digest() || plan.plan_id() != "plan-" + plan.plan_digest().substr(0, 16)) {
      throw runtime::Error::incompatible_worker("versioned plan hash mismatch");
    }
  }
  if (std::find(capabilities.architectures.begin(), capabilities.architectures.end(),
                manifest.architecture().architecture_id()) == capabilities.architectures.end() ||
      std::find(capabilities.execution_dtypes.begin(), capabilities.execution_dtypes.end(),
                plan.execution_dtype()) == capabilities.execution_dtypes.end() ||
      plan.activation_dtype() != v1::DATA_TYPE_F16) {
    throw runtime::Error::incompatible_worker("unsupported architecture or execution dtype: " +
                                capabilities.detail);
  }
  if ((plan.stages_size() != 1 && plan.stages_size() != 2) ||
      request.stage_index() >= static_cast<std::uint32_t>(plan.stages_size())) {
    throw runtime::Error::incompatible_worker("pipeline supports one or two stages");
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
      throw runtime::Error::incompatible_worker("invalid stage partition or tensor ownership");
    }
    end = stage.layer_end();
  }
  if (end != manifest.config().num_layers() ||
      plan.stages(static_cast<int>(request.stage_index())).worker_id() != worker) {
    throw runtime::Error::incompatible_worker("incomplete partition or wrong worker");
  }
  if (plan.stages_size() == 2 && plan.split_layer() != plan.stages(0).layer_end()) {
    throw runtime::Error::incompatible_worker("split layer does not match assignments");
  }
  if (request.stage_endpoints_size() != plan.stages_size()) {
    throw runtime::Error::incompatible_worker("each stage requires an endpoint");
  }
  std::set<std::uint32_t> indices;
  std::set<std::string> endpoints;
  for (const auto& endpoint : request.stage_endpoints()) {
    if (endpoint.stage_index() >= static_cast<std::uint32_t>(plan.stages_size()) ||
        !indices.insert(endpoint.stage_index()).second || endpoint.endpoint().empty() ||
        !endpoints.insert(endpoint.endpoint()).second ||
        endpoint.worker_id() != plan.stages(static_cast<int>(endpoint.stage_index())).worker_id()) {
      throw runtime::Error::incompatible_worker("invalid stage endpoint mapping");
    }
  }
  if (manifest.config().tied_embeddings() && plan.stages_size() == 2 &&
      std::find(plan.duplicated_tensor_groups().begin(), plan.duplicated_tensor_groups().end(),
                "token_embeddings") == plan.duplicated_tensor_groups().end()) {
    throw runtime::Error::incompatible_worker("split tied embedding must be explicitly duplicated");
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

ControlService::ControlService(WorkerConfig config,
                               std::unique_ptr<runtime::BackendFactory> factory)
    : config_(std::move(config)),
      factory_(std::move(factory)),
      capacity_{config_.host_memory_capacity_bytes, config_.device_memory_capacity_bytes,
                config_.pinned_host_memory_capacity_bytes, config_.unified_memory_capacity_bytes} {
  if (config_.worker_id.empty() || config_.endpoint.empty() ||
      !std::filesystem::is_directory(config_.model_root) ||
      (config_.host_memory_capacity_bytes == 0U && config_.unified_memory_capacity_bytes == 0U)) {
    throw std::invalid_argument("worker configuration is incomplete");
  }
  if (!factory_ || capacity_.pinned_host_bytes > capacity_.host_bytes) {
    throw std::invalid_argument("invalid backend factory or pinned host memory budget");
  }
  if (capacity_.device_bytes > std::numeric_limits<std::size_t>::max() - capacity_.host_bytes ||
      capacity_.unified_bytes >
          std::numeric_limits<std::size_t>::max() - capacity_.host_bytes - capacity_.device_bytes) {
    throw std::invalid_argument("combined memory capacity overflows accounting");
  }
  capabilities_ = factory_->capabilities();
  if (capabilities_.primary_memory_domain != v1::MEMORY_DOMAIN_UNIFIED &&
      (capacity_.host_bytes == 0U || capacity_.unified_bytes != 0U)) {
    throw std::invalid_argument(
        "host/device backends require a host memory budget and no unified budget");
  }
  if (capabilities_.primary_memory_domain == v1::MEMORY_DOMAIN_DEVICE &&
      capacity_.device_bytes == 0U) {
    throw std::invalid_argument("device backend requires a device memory budget");
  }
  if (capabilities_.primary_memory_domain == v1::MEMORY_DOMAIN_UNIFIED &&
      (capacity_.unified_bytes == 0U || capacity_.host_bytes != 0U || capacity_.device_bytes != 0U)) {
    throw std::invalid_argument("unified backend requires one unified memory budget");
  }
  if (config_.maximum_active_requests == 0U || config_.maximum_active_requests > 64U) {
    throw std::invalid_argument("maximum active requests must be between 1 and 64");
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
  profile->set_maximum_active_requests(config_.maximum_active_requests);
  profile->set_maximum_cached_tokens(config_.maximum_cached_tokens);
  profile->set_supports_chunked_prefill(true);
  profile->set_supports_sampling(true);
  profile->set_supports_logprobs(true);
  profile->set_maximum_decode_batch(config_.maximum_decode_batch);
  profile->set_endpoint(config_.endpoint);
  profile->set_backend(capabilities_.kind);
  profile->set_supports_mixed_precision(capabilities_.supports_mixed_precision);
  profile->set_primary_memory_domain(capabilities_.primary_memory_domain);
  for (const auto& architecture : capabilities_.architectures) {
    profile->add_supported_architectures(architecture);
  }
  for (const auto dtype : capabilities_.execution_dtypes) {
    profile->add_supported_execution_dtypes(dtype);
  }
  const auto budget = [&](v1::MemoryDomain domain, std::size_t bytes) {
    if (bytes == 0U) {
      return;
    }
    auto* value = profile->add_memory_budgets();
    value->set_domain(domain);
    value->set_capacity_bytes(bytes);
  };
  budget(v1::MEMORY_DOMAIN_UNIFIED, capacity_.unified_bytes);
  budget(v1::MEMORY_DOMAIN_HOST, capacity_.host_bytes);
  budget(v1::MEMORY_DOMAIN_DEVICE, capacity_.device_bytes);
  budget(v1::MEMORY_DOMAIN_HOST_PINNED, capacity_.pinned_host_bytes);
  profile->set_provenance(v1::PROVENANCE_CONFIGURED);
  return grpc::Status::OK;
}

grpc::Status ControlService::GetQualificationState(grpc::ServerContext*, const v1::Empty*,
                                                   v1::QualificationState* out) {
  try {
    const auto info = factory_->profiling_device_info();
    if (const auto host = available_host_memory()) out->set_available_host_bytes(*host);
    if (info.available_bytes) out->set_available_device_bytes(*info.available_bytes);
    out->set_backend_version(info.backend_version);
    out->set_driver_version(info.driver_version);
    out->set_allocator(info.allocator);
    out->set_device_identity(info.identity);
    std::array<char, 256> hostname{};
    if (gethostname(hostname.data(), hostname.size()-1) == 0) {
      const nlohmann::json device_key = {{"host", hostname.data()}, {"device", info.identity},
          {"hardware", qualified_device_name(info.name)}};
      out->set_device_fingerprint(text_digest(device_key.dump()));
    }
    static const auto binary = executable_digest();
    out->set_binary_digest(binary);
    out->set_process_id(static_cast<std::uint64_t>(getpid()));
    out->set_boundary_transport_mode(factory_->boundary_transport_mode());
    if (const auto* value = std::getenv("PYTORCH_ALLOC_CONF")) out->set_pytorch_alloc_conf(value);
    if (const auto* value = std::getenv("PYTORCH_CUDA_ALLOC_CONF")) out->set_pytorch_cuda_alloc_conf(value);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return {grpc::StatusCode::INTERNAL, error.what()};
  }
}

void ControlService::prune_expired() {
  for (auto it = active_.begin(); it != active_.end();) {
    auto& request = it->second;
    if (std::chrono::system_clock::now() >= request->deadline) {
      request->cancelled.store(true);
      if (!request->running && !request->allocating) {
        it = active_.erase(it);
        continue;
      }
    }
    ++it;
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
    validate_plan(*request, config_.worker_id, capabilities_);
    auto next = std::make_shared<LoadedDeployment>();
    next->spec = *request;
    next->scheduler.configure(config_.maximum_decode_batch);
    next->backend = factory_->load(*request, config_.model_root, capacity_);
    if (!next->backend) {
      throw runtime::Error::internal("backend factory returned no stage");
    }
    runtime::require_memory(next->backend->weight_memory(), capacity_);
    deployment_ = std::move(next);
    response->set_accepted(true);
    response->set_detail("executable stage loaded");
  } catch (const runtime::Error& error) {
    reject(response, wire_code(error.code()), error.what());
  } catch (const std::bad_alloc&) {
    reject(response, v1::ERROR_CODE_RESOURCE_EXHAUSTED, "weight allocation failed");
  } catch (const std::exception& error) {
    // Uncategorized failures (I/O, backend libraries) are the worker's problem,
    // not evidence that the request itself was wrong.
    reject(response, v1::ERROR_CODE_BACKEND_ERROR, error.what());
  }
  return grpc::Status::OK;
}

std::shared_ptr<ActiveRequest> ControlService::reserve(const std::string& id, std::size_t tokens,
                                                       std::uint64_t deadline_ms,
                                                       std::unique_lock<std::mutex>& lock) {
  prune_expired();
  if (id.empty() || id.size() > 256U) {
    throw runtime::Error::invalid_request("invalid request ID");
  }
  const auto now = std::chrono::system_clock::now();
  auto deadline = now + std::chrono::seconds(60);
  if (deadline_ms != 0U) {
    const auto now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    if (deadline_ms <= static_cast<std::uint64_t>(now_ms)) {
      throw runtime::Error::deadline_exceeded("request deadline already expired");
    }
    if (deadline_ms - static_cast<std::uint64_t>(now_ms) > 3'600'000U) {
      throw runtime::Error::invalid_request("deadline exceeds one hour");
    }
    deadline = std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms));
  }
  if (const auto found = active_.find(id); found != active_.end()) {
    auto& existing = found->second;
    if (existing->allocating) {
      throw runtime::Error::resource_exhausted("request allocation is in progress");
    }
    if (existing->maximum_tokens != tokens || existing->cancelled.load()) {
      throw runtime::Error::invalid_request("conflicting request reservation retry");
    }
    if (!existing->running) existing->deadline = std::min(existing->deadline, deadline);
    return existing;
  }
  if (active_.size() >= config_.maximum_active_requests) {
    throw runtime::Error::resource_exhausted("worker active request limit reached");
  }
  const auto memory = deployment_->backend->sequence_memory(tokens);
  auto required = runtime::add_memory(deployment_->backend->weight_memory(),
                                      runtime::add_memory(memory.cache, memory.workspace));
  auto cached_tokens = tokens;
  for (const auto& [identifier, request] : active_) {
    required = runtime::add_memory(
        required, runtime::add_memory(request->memory.cache, request->memory.workspace));
    cached_tokens = runtime::checked_add(cached_tokens, request->maximum_tokens);
  }
  if (config_.maximum_cached_tokens && cached_tokens > config_.maximum_cached_tokens) {
    throw runtime::Error::resource_exhausted("worker cached token limit reached");
  }
  runtime::require_memory(required, capacity_);
  auto next = std::make_shared<ActiveRequest>();
  next->id = id;
  next->maximum_tokens = tokens;
  next->deadline = deadline;
  next->memory = memory;
  next->allocating = true;
  active_.emplace(id, next);
  const auto deployment = deployment_;
  // Reserve capacity before allocating, but never block cancellation/telemetry
  // on a backend device lock or a slow allocation.
  lock.unlock();
  try {
    next->sequence = deployment->backend->allocate_sequence(tokens);
    if (!next->sequence) throw runtime::Error::internal("backend returned no sequence state");
  } catch (...) {
    lock.lock();
    active_.erase(id);
    throw;
  }
  lock.lock();
  next->allocating = false;
  if (next->cancelled.load() || std::chrono::system_clock::now() >= next->deadline) {
    active_.erase(id);
    if (std::chrono::system_clock::now() >= next->deadline) {
      throw runtime::Error::deadline_exceeded("reservation expired during allocation");
    }
    throw runtime::Error::cancelled("reservation cancelled during allocation");
  }
  return next;
}

ExecutionLease ControlService::acquire(const std::string& id, std::uint64_t version,
                                       const std::string& request, std::size_t tokens,
                                       std::uint64_t deadline, std::uint32_t required_stage) {
  std::unique_lock lock(mutex_);
  if (!deployment_matches(id, version) || deployment_->spec.stage_index() != required_stage) {
    throw runtime::Error::invalid_request("stale deployment or incorrect execution stage");
  }
  auto state = reserve(request, tokens, deadline, lock);
  if (state->running) {
    throw runtime::Error::resource_exhausted("request already executing");
  }
  state->running = true;
  return {deployment_, state};
}
void ControlService::release(const std::shared_ptr<ActiveRequest>& state) {
  std::scoped_lock lock(mutex_);
  const auto found = active_.find(state->id);
  if (found != active_.end() && found->second == state) {
    // All compute and transport using this sequence have finished.
    state->sequence.reset();
    active_.erase(found);
  }
}

grpc::Status ControlService::ReserveRequest(grpc::ServerContext*,
                                            const v1::ReserveRequestMessage* request,
                                            v1::ReserveResponse* response) {
  std::unique_lock lock(mutex_);
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    reject(response, v1::ERROR_CODE_STALE_DEPLOYMENT, "stale deployment");
    return grpc::Status::OK;
  }
  try {
    static_cast<void>(reserve(request->request_id(), request->maximum_total_tokens(),
                              request->deadline_unix_ms(), lock));
    response->set_accepted(true);
    response->set_detail("backend sequence allocated and workspace reserved");
  } catch (const runtime::Error& error) {
    reject(response, wire_code(error.code()), error.what());
  } catch (const std::bad_alloc&) {
    reject(response, v1::ERROR_CODE_RESOURCE_EXHAUSTED, "cache allocation failed");
  } catch (const std::exception& error) {
    reject(response, v1::ERROR_CODE_BACKEND_ERROR, error.what());
  }
  return grpc::Status::OK;
}

grpc::Status ControlService::CancelRequest(grpc::ServerContext*,
                                           const v1::CancelRequestMessage* request, v1::Empty*) {
  std::scoped_lock lock(mutex_);
  if (!deployment_matches(request->plan_id(), request->deployment_version())) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "stale deployment"};
  }
  const auto found = active_.find(request->request_id());
  if (found != active_.end()) {
    found->second->cancelled.store(true);
    if (!found->second->running && !found->second->allocating) active_.erase(found);
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
  if (!active_.empty()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "deployment has an active request"};
  }
  deployment_.reset();
  return grpc::Status::OK;
}
grpc::Status ControlService::GetMemoryReport(grpc::ServerContext*, const v1::Empty*,
                                             v1::MemoryReport* response) {
  std::scoped_lock lock(mutex_);
  prune_expired();
  const auto weights =
      deployment_ ? deployment_->backend->weight_memory() : runtime::MemoryAmounts{};
  runtime::MemoryAmounts cache{}, workspace{};
  for (const auto& [identifier, request] : active_) {
    response->add_active_request_ids(identifier);
    cache = runtime::add_memory(cache, request->memory.cache);
    workspace = runtime::add_memory(workspace, request->memory.workspace);
  }
  const auto report_domain = [&](v1::MemoryDomain domain, std::size_t capacity, std::size_t weight,
                                 std::size_t cached, std::size_t work) {
    if (capacity == 0U) {
      return;
    }
    auto* budget = response->add_budgets();
    budget->set_domain(domain);
    budget->set_capacity_bytes(capacity);
    auto* usage = response->add_domain_usage();
    usage->set_domain(domain);
    usage->set_loaded_weight_bytes(weight);
    usage->set_reserved_cache_bytes(cached);
    usage->set_reserved_workspace_bytes(work);
  };
  report_domain(v1::MEMORY_DOMAIN_HOST, capacity_.host_bytes, weights.host_bytes, cache.host_bytes,
                workspace.host_bytes);
  report_domain(v1::MEMORY_DOMAIN_DEVICE, capacity_.device_bytes, weights.device_bytes,
                cache.device_bytes, workspace.device_bytes);
  report_domain(v1::MEMORY_DOMAIN_HOST_PINNED, capacity_.pinned_host_bytes,
                weights.pinned_host_bytes, cache.pinned_host_bytes, workspace.pinned_host_bytes);
  report_domain(v1::MEMORY_DOMAIN_UNIFIED, capacity_.unified_bytes, weights.unified_bytes,
                cache.unified_bytes, workspace.unified_bytes);
  // Legacy totals count each byte once: pinned memory is already included in host.
  response->set_loaded_weight_bytes(weights.host_bytes + weights.device_bytes + weights.unified_bytes);
  response->set_reserved_cache_bytes(cache.host_bytes + cache.device_bytes + cache.unified_bytes);
  response->set_reserved_workspace_bytes(workspace.host_bytes + workspace.device_bytes +
                                         workspace.unified_bytes);
  response->set_active_requests(active_.size());
  return grpc::Status::OK;
}
grpc::Status ControlService::GetMetrics(grpc::ServerContext*, const v1::Empty*,
                                        v1::WorkerMetrics* response) {
  response->set_worker_id(config_.worker_id);
  response->clear_allocator();
  {
    std::scoped_lock lock(mutex_);
    const auto metrics = deployment_ ? deployment_->scheduler.metrics() : SchedulerMetrics{};
    response->set_queued_prefills(metrics.queued_prefills);
    response->set_queued_decodes(metrics.queued_decodes);
    response->set_executing_microbatches(metrics.executing);
    response->set_completed_steps(metrics.completed_steps);
    response->set_queue_wait_ms(metrics.queue_wait_ms);
    response->set_decode_batches(metrics.decode_batches);
    response->set_largest_decode_batch(metrics.largest_decode_batch);
  }
  if (const auto metrics = factory_->allocator_metrics()) {
    auto* allocator = response->mutable_allocator();
    allocator->set_domain(capabilities_.primary_memory_domain);
    allocator->set_active_bytes(metrics->active_bytes);
    allocator->set_cached_bytes(metrics->cached_bytes);
    allocator->set_peak_bytes(metrics->peak_bytes);
  }
  return grpc::Status::OK;
}
grpc::Status ControlService::Health(grpc::ServerContext*, const v1::Empty*,
                                    v1::HealthResponse* response) {
  std::scoped_lock lock(mutex_);
  response->set_serving(!capabilities_.architectures.empty());
  response->set_detail(deployment_ ? "executable stage ready" : capabilities_.detail);
  return grpc::Status::OK;
}

}  // namespace hllm::worker
