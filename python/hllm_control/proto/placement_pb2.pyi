from . import common_pb2 as _common_pb2
from . import profile_pb2 as _profile_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StageMemory(_message.Message):
    __slots__ = ("worker_id", "domain", "layer_start", "layer_end", "weight_bytes", "kv_cache_bytes", "workspace_bytes", "activation_buffer_bytes", "allocator_allowance_bytes", "required_bytes", "usable_bytes", "pressure")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    LAYER_START_FIELD_NUMBER: _ClassVar[int]
    LAYER_END_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_BYTES_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKSPACE_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_BUFFER_BYTES_FIELD_NUMBER: _ClassVar[int]
    ALLOCATOR_ALLOWANCE_BYTES_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    USABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    PRESSURE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    domain: _profile_pb2.MemoryDomain
    layer_start: int
    layer_end: int
    weight_bytes: int
    kv_cache_bytes: int
    workspace_bytes: int
    activation_buffer_bytes: int
    allocator_allowance_bytes: int
    required_bytes: int
    usable_bytes: int
    pressure: float
    def __init__(self, worker_id: _Optional[str] = ..., domain: _Optional[_Union[_profile_pb2.MemoryDomain, str]] = ..., layer_start: _Optional[int] = ..., layer_end: _Optional[int] = ..., weight_bytes: _Optional[int] = ..., kv_cache_bytes: _Optional[int] = ..., workspace_bytes: _Optional[int] = ..., activation_buffer_bytes: _Optional[int] = ..., allocator_allowance_bytes: _Optional[int] = ..., required_bytes: _Optional[int] = ..., usable_bytes: _Optional[int] = ..., pressure: _Optional[float] = ...) -> None: ...

class StageAssignment(_message.Message):
    __slots__ = ("stage_index", "worker_id", "layer_start", "layer_end", "owns_token_embedding", "owns_final_norm", "owns_lm_head", "owns_sampling")
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_START_FIELD_NUMBER: _ClassVar[int]
    LAYER_END_FIELD_NUMBER: _ClassVar[int]
    OWNS_TOKEN_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    OWNS_FINAL_NORM_FIELD_NUMBER: _ClassVar[int]
    OWNS_LM_HEAD_FIELD_NUMBER: _ClassVar[int]
    OWNS_SAMPLING_FIELD_NUMBER: _ClassVar[int]
    stage_index: int
    worker_id: str
    layer_start: int
    layer_end: int
    owns_token_embedding: bool
    owns_final_norm: bool
    owns_lm_head: bool
    owns_sampling: bool
    def __init__(self, stage_index: _Optional[int] = ..., worker_id: _Optional[str] = ..., layer_start: _Optional[int] = ..., layer_end: _Optional[int] = ..., owns_token_embedding: _Optional[bool] = ..., owns_final_norm: _Optional[bool] = ..., owns_lm_head: _Optional[bool] = ..., owns_sampling: _Optional[bool] = ...) -> None: ...

class DeploymentPlan(_message.Message):
    __slots__ = ("schema_version", "planner_version", "plan_id", "plan_digest", "manifest_digest", "workload_id", "planning_mode", "execution_dtype", "activation_dtype", "split_layer", "stages", "selected_candidate_id", "duplicated_tensor_groups", "deployment_version", "workload_digest", "profile_bundle_digest")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    PLANNER_VERSION_FIELD_NUMBER: _ClassVar[int]
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    PLANNING_MODE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_DTYPE_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_DTYPE_FIELD_NUMBER: _ClassVar[int]
    SPLIT_LAYER_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    SELECTED_CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    DUPLICATED_TENSOR_GROUPS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PROFILE_BUNDLE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    schema_version: _common_pb2.ArtifactVersion
    planner_version: str
    plan_id: str
    plan_digest: str
    manifest_digest: str
    workload_id: str
    planning_mode: str
    execution_dtype: _common_pb2.DataType
    activation_dtype: _common_pb2.DataType
    split_layer: int
    stages: _containers.RepeatedCompositeFieldContainer[StageAssignment]
    selected_candidate_id: str
    duplicated_tensor_groups: _containers.RepeatedScalarFieldContainer[str]
    deployment_version: int
    workload_digest: str
    profile_bundle_digest: str
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., planner_version: _Optional[str] = ..., plan_id: _Optional[str] = ..., plan_digest: _Optional[str] = ..., manifest_digest: _Optional[str] = ..., workload_id: _Optional[str] = ..., planning_mode: _Optional[str] = ..., execution_dtype: _Optional[_Union[_common_pb2.DataType, str]] = ..., activation_dtype: _Optional[_Union[_common_pb2.DataType, str]] = ..., split_layer: _Optional[int] = ..., stages: _Optional[_Iterable[_Union[StageAssignment, _Mapping]]] = ..., selected_candidate_id: _Optional[str] = ..., duplicated_tensor_groups: _Optional[_Iterable[str]] = ..., deployment_version: _Optional[int] = ..., workload_digest: _Optional[str] = ..., profile_bundle_digest: _Optional[str] = ...) -> None: ...
