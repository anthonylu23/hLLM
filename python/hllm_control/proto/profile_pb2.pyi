from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Backend(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BACKEND_UNSPECIFIED: _ClassVar[Backend]
    BACKEND_CPU: _ClassVar[Backend]
    BACKEND_MLX: _ClassVar[Backend]
    BACKEND_CUDA: _ClassVar[Backend]

class MemoryDomain(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MEMORY_DOMAIN_UNSPECIFIED: _ClassVar[MemoryDomain]
    MEMORY_DOMAIN_UNIFIED: _ClassVar[MemoryDomain]
    MEMORY_DOMAIN_DEVICE: _ClassVar[MemoryDomain]
    MEMORY_DOMAIN_HOST: _ClassVar[MemoryDomain]
    MEMORY_DOMAIN_HOST_PINNED: _ClassVar[MemoryDomain]

class ConnectionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONNECTION_TYPE_UNSPECIFIED: _ClassVar[ConnectionType]
    CONNECTION_TYPE_DIRECT: _ClassVar[ConnectionType]
    CONNECTION_TYPE_PEER_RELAYED: _ClassVar[ConnectionType]
    CONNECTION_TYPE_DERP_RELAYED: _ClassVar[ConnectionType]
    CONNECTION_TYPE_UNKNOWN: _ClassVar[ConnectionType]
BACKEND_UNSPECIFIED: Backend
BACKEND_CPU: Backend
BACKEND_MLX: Backend
BACKEND_CUDA: Backend
MEMORY_DOMAIN_UNSPECIFIED: MemoryDomain
MEMORY_DOMAIN_UNIFIED: MemoryDomain
MEMORY_DOMAIN_DEVICE: MemoryDomain
MEMORY_DOMAIN_HOST: MemoryDomain
MEMORY_DOMAIN_HOST_PINNED: MemoryDomain
CONNECTION_TYPE_UNSPECIFIED: ConnectionType
CONNECTION_TYPE_DIRECT: ConnectionType
CONNECTION_TYPE_PEER_RELAYED: ConnectionType
CONNECTION_TYPE_DERP_RELAYED: ConnectionType
CONNECTION_TYPE_UNKNOWN: ConnectionType

class MemoryBudget(_message.Message):
    __slots__ = ("domain", "capacity_bytes", "runtime_reserve_bytes", "safety_fraction")
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_RESERVE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SAFETY_FRACTION_FIELD_NUMBER: _ClassVar[int]
    domain: MemoryDomain
    capacity_bytes: int
    runtime_reserve_bytes: int
    safety_fraction: float
    def __init__(self, domain: _Optional[_Union[MemoryDomain, str]] = ..., capacity_bytes: _Optional[int] = ..., runtime_reserve_bytes: _Optional[int] = ..., safety_fraction: _Optional[float] = ...) -> None: ...

class WorkerProfile(_message.Message):
    __slots__ = ("schema_version", "worker_id", "endpoint", "backend", "primary_memory_domain", "supported_architectures", "supported_execution_dtypes", "supports_mixed_precision", "memory_budgets", "fixed_workspace_bytes", "activation_buffer_count", "allocator_allowance_fraction", "host_transport_buffer_bytes", "provenance", "observed_available_host_bytes", "observed_at")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_MEMORY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ARCHITECTURES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_EXECUTION_DTYPES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MIXED_PRECISION_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BUDGETS_FIELD_NUMBER: _ClassVar[int]
    FIXED_WORKSPACE_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_BUFFER_COUNT_FIELD_NUMBER: _ClassVar[int]
    ALLOCATOR_ALLOWANCE_FRACTION_FIELD_NUMBER: _ClassVar[int]
    HOST_TRANSPORT_BUFFER_BYTES_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AVAILABLE_HOST_BYTES_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    schema_version: _common_pb2.ArtifactVersion
    worker_id: str
    endpoint: str
    backend: Backend
    primary_memory_domain: MemoryDomain
    supported_architectures: _containers.RepeatedScalarFieldContainer[str]
    supported_execution_dtypes: _containers.RepeatedScalarFieldContainer[_common_pb2.DataType]
    supports_mixed_precision: bool
    memory_budgets: _containers.RepeatedCompositeFieldContainer[MemoryBudget]
    fixed_workspace_bytes: int
    activation_buffer_count: int
    allocator_allowance_fraction: float
    host_transport_buffer_bytes: int
    provenance: _common_pb2.Provenance
    observed_available_host_bytes: int
    observed_at: str
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., worker_id: _Optional[str] = ..., endpoint: _Optional[str] = ..., backend: _Optional[_Union[Backend, str]] = ..., primary_memory_domain: _Optional[_Union[MemoryDomain, str]] = ..., supported_architectures: _Optional[_Iterable[str]] = ..., supported_execution_dtypes: _Optional[_Iterable[_Union[_common_pb2.DataType, str]]] = ..., supports_mixed_precision: _Optional[bool] = ..., memory_budgets: _Optional[_Iterable[_Union[MemoryBudget, _Mapping]]] = ..., fixed_workspace_bytes: _Optional[int] = ..., activation_buffer_count: _Optional[int] = ..., allocator_allowance_fraction: _Optional[float] = ..., host_transport_buffer_bytes: _Optional[int] = ..., provenance: _Optional[_Union[_common_pb2.Provenance, str]] = ..., observed_available_host_bytes: _Optional[int] = ..., observed_at: _Optional[str] = ...) -> None: ...

class LinkProfile(_message.Message):
    __slots__ = ("schema_version", "source_worker_id", "target_worker_id", "connection_type", "fixed_latency_ms", "bandwidth_bytes_per_second", "sender_conversion_ms", "receiver_conversion_ms", "provenance", "observed_at", "qualification")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    FIXED_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_BYTES_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    SENDER_CONVERSION_MS_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_CONVERSION_MS_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    QUALIFICATION_FIELD_NUMBER: _ClassVar[int]
    schema_version: _common_pb2.ArtifactVersion
    source_worker_id: str
    target_worker_id: str
    connection_type: ConnectionType
    fixed_latency_ms: float
    bandwidth_bytes_per_second: float
    sender_conversion_ms: float
    receiver_conversion_ms: float
    provenance: _common_pb2.Provenance
    observed_at: str
    qualification: LinkQualificationResult
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., source_worker_id: _Optional[str] = ..., target_worker_id: _Optional[str] = ..., connection_type: _Optional[_Union[ConnectionType, str]] = ..., fixed_latency_ms: _Optional[float] = ..., bandwidth_bytes_per_second: _Optional[float] = ..., sender_conversion_ms: _Optional[float] = ..., receiver_conversion_ms: _Optional[float] = ..., provenance: _Optional[_Union[_common_pb2.Provenance, str]] = ..., observed_at: _Optional[str] = ..., qualification: _Optional[_Union[LinkQualificationResult, _Mapping]] = ...) -> None: ...

class WorkloadProfile(_message.Message):
    __slots__ = ("schema_version", "workload_id", "prompt_tokens", "output_tokens", "concurrency", "total_cached_tokens", "kv_overhead_factor", "kv_dtype", "activation_dtype")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    KV_OVERHEAD_FACTOR_FIELD_NUMBER: _ClassVar[int]
    KV_DTYPE_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_DTYPE_FIELD_NUMBER: _ClassVar[int]
    schema_version: _common_pb2.ArtifactVersion
    workload_id: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    total_cached_tokens: int
    kv_overhead_factor: float
    kv_dtype: _common_pb2.DataType
    activation_dtype: _common_pb2.DataType
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., workload_id: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., concurrency: _Optional[int] = ..., total_cached_tokens: _Optional[int] = ..., kv_overhead_factor: _Optional[float] = ..., kv_dtype: _Optional[_Union[_common_pb2.DataType, str]] = ..., activation_dtype: _Optional[_Union[_common_pb2.DataType, str]] = ...) -> None: ...

class LinkProbeIdentity(_message.Message):
    __slots__ = ("worker_id", "binary_digest", "source_revision", "compiler", "grpc_version", "protobuf_version", "host", "os", "endpoint")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    BINARY_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REVISION_FIELD_NUMBER: _ClassVar[int]
    COMPILER_FIELD_NUMBER: _ClassVar[int]
    GRPC_VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOBUF_VERSION_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    binary_digest: str
    source_revision: str
    compiler: str
    grpc_version: str
    protobuf_version: str
    host: str
    os: str
    endpoint: str
    def __init__(self, worker_id: _Optional[str] = ..., binary_digest: _Optional[str] = ..., source_revision: _Optional[str] = ..., compiler: _Optional[str] = ..., grpc_version: _Optional[str] = ..., protobuf_version: _Optional[str] = ..., host: _Optional[str] = ..., os: _Optional[str] = ..., endpoint: _Optional[str] = ...) -> None: ...

class LinkTiming(_message.Message):
    __slots__ = ("cycle", "step", "payload_bytes", "message_bytes", "feedback_bytes", "sender_encode_ms", "round_trip_ms")
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_BYTES_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_BYTES_FIELD_NUMBER: _ClassVar[int]
    SENDER_ENCODE_MS_FIELD_NUMBER: _ClassVar[int]
    ROUND_TRIP_MS_FIELD_NUMBER: _ClassVar[int]
    cycle: int
    step: int
    payload_bytes: int
    message_bytes: int
    feedback_bytes: int
    sender_encode_ms: float
    round_trip_ms: float
    def __init__(self, cycle: _Optional[int] = ..., step: _Optional[int] = ..., payload_bytes: _Optional[int] = ..., message_bytes: _Optional[int] = ..., feedback_bytes: _Optional[int] = ..., sender_encode_ms: _Optional[float] = ..., round_trip_ms: _Optional[float] = ...) -> None: ...

class LinkStreamTiming(_message.Message):
    __slots__ = ("cycle", "setup_ms", "teardown_ms")
    CYCLE_FIELD_NUMBER: _ClassVar[int]
    SETUP_MS_FIELD_NUMBER: _ClassVar[int]
    TEARDOWN_MS_FIELD_NUMBER: _ClassVar[int]
    cycle: int
    setup_ms: float
    teardown_ms: float
    def __init__(self, cycle: _Optional[int] = ..., setup_ms: _Optional[float] = ..., teardown_ms: _Optional[float] = ...) -> None: ...

class LinkQualificationResult(_message.Message):
    __slots__ = ("source", "target", "channel_ready_ms", "streams", "samples", "prompt_tokens", "hidden_size", "output_tokens", "warmup_cycles", "measured_cycles")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_READY_MS_FIELD_NUMBER: _ClassVar[int]
    STREAMS_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_SIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    WARMUP_CYCLES_FIELD_NUMBER: _ClassVar[int]
    MEASURED_CYCLES_FIELD_NUMBER: _ClassVar[int]
    source: LinkProbeIdentity
    target: LinkProbeIdentity
    channel_ready_ms: float
    streams: _containers.RepeatedCompositeFieldContainer[LinkStreamTiming]
    samples: _containers.RepeatedCompositeFieldContainer[LinkTiming]
    prompt_tokens: int
    hidden_size: int
    output_tokens: int
    warmup_cycles: int
    measured_cycles: int
    def __init__(self, source: _Optional[_Union[LinkProbeIdentity, _Mapping]] = ..., target: _Optional[_Union[LinkProbeIdentity, _Mapping]] = ..., channel_ready_ms: _Optional[float] = ..., streams: _Optional[_Iterable[_Union[LinkStreamTiming, _Mapping]]] = ..., samples: _Optional[_Iterable[_Union[LinkTiming, _Mapping]]] = ..., prompt_tokens: _Optional[int] = ..., hidden_size: _Optional[int] = ..., output_tokens: _Optional[int] = ..., warmup_cycles: _Optional[int] = ..., measured_cycles: _Optional[int] = ...) -> None: ...
