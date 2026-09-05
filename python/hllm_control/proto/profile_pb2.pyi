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
    __slots__ = ("schema_version", "worker_id", "endpoint", "backend", "primary_memory_domain", "supported_architectures", "supported_execution_dtypes", "memory_budgets", "fixed_workspace_bytes", "activation_buffer_count", "allocator_allowance_fraction", "host_transport_buffer_bytes", "provenance", "observed_available_host_bytes", "observed_at")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_MEMORY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ARCHITECTURES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_EXECUTION_DTYPES_FIELD_NUMBER: _ClassVar[int]
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
    memory_budgets: _containers.RepeatedCompositeFieldContainer[MemoryBudget]
    fixed_workspace_bytes: int
    activation_buffer_count: int
    allocator_allowance_fraction: float
    host_transport_buffer_bytes: int
    provenance: _common_pb2.Provenance
    observed_available_host_bytes: int
    observed_at: str
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., worker_id: _Optional[str] = ..., endpoint: _Optional[str] = ..., backend: _Optional[_Union[Backend, str]] = ..., primary_memory_domain: _Optional[_Union[MemoryDomain, str]] = ..., supported_architectures: _Optional[_Iterable[str]] = ..., supported_execution_dtypes: _Optional[_Iterable[_Union[_common_pb2.DataType, str]]] = ..., memory_budgets: _Optional[_Iterable[_Union[MemoryBudget, _Mapping]]] = ..., fixed_workspace_bytes: _Optional[int] = ..., activation_buffer_count: _Optional[int] = ..., allocator_allowance_fraction: _Optional[float] = ..., host_transport_buffer_bytes: _Optional[int] = ..., provenance: _Optional[_Union[_common_pb2.Provenance, str]] = ..., observed_available_host_bytes: _Optional[int] = ..., observed_at: _Optional[str] = ...) -> None: ...

class LinkProfile(_message.Message):
    __slots__ = ("schema_version", "source_worker_id", "target_worker_id", "connection_type", "fixed_latency_ms", "bandwidth_bytes_per_second", "sender_conversion_ms", "receiver_conversion_ms", "provenance", "observed_at")
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
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., source_worker_id: _Optional[str] = ..., target_worker_id: _Optional[str] = ..., connection_type: _Optional[_Union[ConnectionType, str]] = ..., fixed_latency_ms: _Optional[float] = ..., bandwidth_bytes_per_second: _Optional[float] = ..., sender_conversion_ms: _Optional[float] = ..., receiver_conversion_ms: _Optional[float] = ..., provenance: _Optional[_Union[_common_pb2.Provenance, str]] = ..., observed_at: _Optional[str] = ...) -> None: ...

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
