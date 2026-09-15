from . import common_pb2 as _common_pb2
from . import model_pb2 as _model_pb2
from . import placement_pb2 as _placement_pb2
from . import profile_pb2 as _profile_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Capabilities(_message.Message):
    __slots__ = ("worker",)
    WORKER_FIELD_NUMBER: _ClassVar[int]
    worker: _profile_pb2.WorkerProfile
    def __init__(self, worker: _Optional[_Union[_profile_pb2.WorkerProfile, _Mapping]] = ...) -> None: ...

class QualificationState(_message.Message):
    __slots__ = ("available_host_bytes", "available_device_bytes", "backend_version", "driver_version", "allocator", "binary_digest", "device_identity", "process_id", "boundary_transport_mode", "pytorch_alloc_conf", "pytorch_cuda_alloc_conf", "device_fingerprint")
    AVAILABLE_HOST_BYTES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_DEVICE_BYTES_FIELD_NUMBER: _ClassVar[int]
    BACKEND_VERSION_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATOR_FIELD_NUMBER: _ClassVar[int]
    BINARY_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DEVICE_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    PROCESS_ID_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_TRANSPORT_MODE_FIELD_NUMBER: _ClassVar[int]
    PYTORCH_ALLOC_CONF_FIELD_NUMBER: _ClassVar[int]
    PYTORCH_CUDA_ALLOC_CONF_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    available_host_bytes: int
    available_device_bytes: int
    backend_version: str
    driver_version: str
    allocator: str
    binary_digest: str
    device_identity: str
    process_id: int
    boundary_transport_mode: str
    pytorch_alloc_conf: str
    pytorch_cuda_alloc_conf: str
    device_fingerprint: str
    def __init__(self, available_host_bytes: _Optional[int] = ..., available_device_bytes: _Optional[int] = ..., backend_version: _Optional[str] = ..., driver_version: _Optional[str] = ..., allocator: _Optional[str] = ..., binary_digest: _Optional[str] = ..., device_identity: _Optional[str] = ..., process_id: _Optional[int] = ..., boundary_transport_mode: _Optional[str] = ..., pytorch_alloc_conf: _Optional[str] = ..., pytorch_cuda_alloc_conf: _Optional[str] = ..., device_fingerprint: _Optional[str] = ...) -> None: ...

class LinkQualificationRequest(_message.Message):
    __slots__ = ("target_worker_id", "prompt_tokens", "hidden_size", "output_tokens", "warmup_cycles", "measured_cycles", "timeout_ms")
    TARGET_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_SIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    WARMUP_CYCLES_FIELD_NUMBER: _ClassVar[int]
    MEASURED_CYCLES_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    target_worker_id: str
    prompt_tokens: int
    hidden_size: int
    output_tokens: int
    warmup_cycles: int
    measured_cycles: int
    timeout_ms: int
    def __init__(self, target_worker_id: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., hidden_size: _Optional[int] = ..., output_tokens: _Optional[int] = ..., warmup_cycles: _Optional[int] = ..., measured_cycles: _Optional[int] = ..., timeout_ms: _Optional[int] = ...) -> None: ...

class StageEndpoint(_message.Message):
    __slots__ = ("stage_index", "worker_id", "endpoint")
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    stage_index: int
    worker_id: str
    endpoint: str
    def __init__(self, stage_index: _Optional[int] = ..., worker_id: _Optional[str] = ..., endpoint: _Optional[str] = ...) -> None: ...

class LoadStageRequest(_message.Message):
    __slots__ = ("plan", "stage_index", "manifest", "stage_endpoints")
    PLAN_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    STAGE_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    plan: _placement_pb2.DeploymentPlan
    stage_index: int
    manifest: _model_pb2.ModelManifest
    stage_endpoints: _containers.RepeatedCompositeFieldContainer[StageEndpoint]
    def __init__(self, plan: _Optional[_Union[_placement_pb2.DeploymentPlan, _Mapping]] = ..., stage_index: _Optional[int] = ..., manifest: _Optional[_Union[_model_pb2.ModelManifest, _Mapping]] = ..., stage_endpoints: _Optional[_Iterable[_Union[StageEndpoint, _Mapping]]] = ...) -> None: ...

class LoadStageResponse(_message.Message):
    __slots__ = ("accepted", "detail", "error")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    error: _common_pb2.RuntimeError
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., error: _Optional[_Union[_common_pb2.RuntimeError, _Mapping]] = ...) -> None: ...

class UnloadStageRequest(_message.Message):
    __slots__ = ("plan_id", "deployment_version")
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    plan_id: str
    deployment_version: int
    def __init__(self, plan_id: _Optional[str] = ..., deployment_version: _Optional[int] = ...) -> None: ...

class ReserveRequestMessage(_message.Message):
    __slots__ = ("plan_id", "request_id", "deployment_version", "maximum_total_tokens", "deadline_unix_ms")
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    plan_id: str
    request_id: str
    deployment_version: int
    maximum_total_tokens: int
    deadline_unix_ms: int
    def __init__(self, plan_id: _Optional[str] = ..., request_id: _Optional[str] = ..., deployment_version: _Optional[int] = ..., maximum_total_tokens: _Optional[int] = ..., deadline_unix_ms: _Optional[int] = ...) -> None: ...

class ReserveResponse(_message.Message):
    __slots__ = ("accepted", "detail", "error")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    error: _common_pb2.RuntimeError
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., error: _Optional[_Union[_common_pb2.RuntimeError, _Mapping]] = ...) -> None: ...

class CancelRequestMessage(_message.Message):
    __slots__ = ("plan_id", "request_id", "deployment_version", "reason")
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    plan_id: str
    request_id: str
    deployment_version: int
    reason: str
    def __init__(self, plan_id: _Optional[str] = ..., request_id: _Optional[str] = ..., deployment_version: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class DomainMemoryUsage(_message.Message):
    __slots__ = ("domain", "loaded_weight_bytes", "reserved_cache_bytes", "reserved_workspace_bytes")
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    LOADED_WEIGHT_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESERVED_CACHE_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESERVED_WORKSPACE_BYTES_FIELD_NUMBER: _ClassVar[int]
    domain: _profile_pb2.MemoryDomain
    loaded_weight_bytes: int
    reserved_cache_bytes: int
    reserved_workspace_bytes: int
    def __init__(self, domain: _Optional[_Union[_profile_pb2.MemoryDomain, str]] = ..., loaded_weight_bytes: _Optional[int] = ..., reserved_cache_bytes: _Optional[int] = ..., reserved_workspace_bytes: _Optional[int] = ...) -> None: ...

class MemoryReport(_message.Message):
    __slots__ = ("budgets", "loaded_weight_bytes", "reserved_cache_bytes", "reserved_workspace_bytes", "active_requests", "domain_usage")
    BUDGETS_FIELD_NUMBER: _ClassVar[int]
    LOADED_WEIGHT_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESERVED_CACHE_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESERVED_WORKSPACE_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_USAGE_FIELD_NUMBER: _ClassVar[int]
    budgets: _containers.RepeatedCompositeFieldContainer[_profile_pb2.MemoryBudget]
    loaded_weight_bytes: int
    reserved_cache_bytes: int
    reserved_workspace_bytes: int
    active_requests: int
    domain_usage: _containers.RepeatedCompositeFieldContainer[DomainMemoryUsage]
    def __init__(self, budgets: _Optional[_Iterable[_Union[_profile_pb2.MemoryBudget, _Mapping]]] = ..., loaded_weight_bytes: _Optional[int] = ..., reserved_cache_bytes: _Optional[int] = ..., reserved_workspace_bytes: _Optional[int] = ..., active_requests: _Optional[int] = ..., domain_usage: _Optional[_Iterable[_Union[DomainMemoryUsage, _Mapping]]] = ...) -> None: ...

class AllocatorMetrics(_message.Message):
    __slots__ = ("domain", "active_bytes", "cached_bytes", "peak_bytes")
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CACHED_BYTES_FIELD_NUMBER: _ClassVar[int]
    PEAK_BYTES_FIELD_NUMBER: _ClassVar[int]
    domain: _profile_pb2.MemoryDomain
    active_bytes: int
    cached_bytes: int
    peak_bytes: int
    def __init__(self, domain: _Optional[_Union[_profile_pb2.MemoryDomain, str]] = ..., active_bytes: _Optional[int] = ..., cached_bytes: _Optional[int] = ..., peak_bytes: _Optional[int] = ...) -> None: ...

class WorkerMetrics(_message.Message):
    __slots__ = ("worker_id", "allocator")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOCATOR_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    allocator: AllocatorMetrics
    def __init__(self, worker_id: _Optional[str] = ..., allocator: _Optional[_Union[AllocatorMetrics, _Mapping]] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("serving", "detail")
    SERVING_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    serving: bool
    detail: str
    def __init__(self, serving: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...
