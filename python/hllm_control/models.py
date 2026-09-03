"""Versioned domain models shared by preparation and placement planning."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_SCHEMA_VERSION = "1.0"
PROFILE_SCHEMA_VERSION = "1.0"
PLAN_SCHEMA_VERSION = "1.0"
PLANNER_VERSION = "0.1.0"

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0.0, lt=1.0)]


class StrictModel(BaseModel):
    """Base model which rejects misspelled or unsupported input fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DType(StrEnum):
    BOOL = "BOOL"
    U8 = "U8"
    I8 = "I8"
    I16 = "I16"
    U16 = "U16"
    I32 = "I32"
    U32 = "U32"
    I64 = "I64"
    U64 = "U64"
    F16 = "F16"
    BF16 = "BF16"
    F32 = "F32"
    F64 = "F64"


DTYPE_BYTES: dict[DType, int] = {
    DType.BOOL: 1,
    DType.U8: 1,
    DType.I8: 1,
    DType.I16: 2,
    DType.U16: 2,
    DType.I32: 4,
    DType.U32: 4,
    DType.I64: 8,
    DType.U64: 8,
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.F64: 8,
}


class TensorRole(StrEnum):
    TOKEN_EMBEDDING = "TOKEN_EMBEDDING"
    TRANSFORMER_LAYER = "TRANSFORMER_LAYER"
    FINAL_NORM = "FINAL_NORM"
    LM_HEAD = "LM_HEAD"
    ARCHITECTURE_STATE = "ARCHITECTURE_STATE"


class Provenance(StrEnum):
    THEORETICAL = "THEORETICAL"
    CONFIGURED = "CONFIGURED"
    OBSERVED = "OBSERVED"
    MEASURED = "MEASURED"


class Backend(StrEnum):
    CPU = "cpu"
    MLX = "mlx"
    CUDA = "cuda"


class MemoryDomain(StrEnum):
    UNIFIED = "UNIFIED"
    DEVICE = "DEVICE"
    HOST = "HOST"
    HOST_PINNED = "HOST_PINNED"


class ConnectionType(StrEnum):
    DIRECT = "DIRECT"
    PEER_RELAYED = "PEER_RELAYED"
    DERP_RELAYED = "DERP_RELAYED"
    UNKNOWN = "UNKNOWN"


class PlanningMode(StrEnum):
    FEASIBILITY = "feasibility"
    ESTIMATED = "estimated"


class SourceDescriptor(StrictModel):
    model_id: str
    revision: str | None = None
    config_sha256: str
    index_sha256: str | None = None
    tensor_metadata_sha256: str | None = None


class ArchitectureDescriptor(StrictModel):
    architecture_id: str
    architecture_revision: str
    feature_flags: tuple[str, ...] = ()


class ModelConfig(StrictModel):
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_kv_heads: PositiveInt
    head_dim: PositiveInt
    vocabulary_size: PositiveInt
    maximum_sequence_length: PositiveInt
    tied_embeddings: bool

    @model_validator(mode="after")
    def validate_attention_shape(self) -> ModelConfig:
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        return self


class TensorFile(StrictModel):
    name: str
    size_bytes: NonNegativeInt
    header_size_bytes: NonNegativeInt
    sha256: str | None = None


class TensorRecord(StrictModel):
    name: str
    file: str
    dtype: DType
    shape: tuple[NonNegativeInt, ...]
    data_offset: NonNegativeInt
    byte_length: NonNegativeInt
    role: TensorRole
    layer_index: NonNegativeInt | None = None
    shared_weight_group: str | None = None

    @property
    def num_elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


class ComponentMemory(StrictModel):
    role: TensorRole
    layer_index: NonNegativeInt | None = None
    storage_bytes: NonNegativeInt


class ModelManifest(StrictModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    manifest_id: str
    manifest_digest: str
    source: SourceDescriptor
    architecture: ArchitectureDescriptor
    config: ModelConfig
    tensor_files: tuple[TensorFile, ...]
    tensors: tuple[TensorRecord, ...]
    components: tuple[ComponentMemory, ...]
    total_storage_bytes: NonNegativeInt


class MemoryBudget(StrictModel):
    domain: MemoryDomain
    capacity_bytes: PositiveInt
    runtime_reserve_bytes: NonNegativeInt = 0
    safety_fraction: Fraction = 0.1

    @property
    def usable_bytes(self) -> int:
        safety_bytes = int(self.capacity_bytes * self.safety_fraction)
        return max(0, self.capacity_bytes - self.runtime_reserve_bytes - safety_bytes)


class WorkerProfile(StrictModel):
    schema_version: str = PROFILE_SCHEMA_VERSION
    worker_id: str
    endpoint: str
    backend: Backend
    primary_memory_domain: MemoryDomain
    supported_architectures: tuple[str, ...]
    supported_execution_dtypes: tuple[DType, ...]
    memory_budgets: tuple[MemoryBudget, ...]
    fixed_workspace_bytes: NonNegativeInt = 0
    activation_buffer_count: PositiveInt = 2
    allocator_allowance_fraction: Fraction = 0.05
    host_transport_buffer_bytes: NonNegativeInt = 0
    provenance: Provenance = Provenance.CONFIGURED
    observed_available_host_bytes: NonNegativeInt | None = None
    observed_at: str | None = None

    @model_validator(mode="after")
    def validate_memory_domains(self) -> WorkerProfile:
        domains = [budget.domain for budget in self.memory_budgets]
        if len(domains) != len(set(domains)):
            raise ValueError("worker memory budget domains must be unique")
        if self.primary_memory_domain not in domains:
            raise ValueError("primary_memory_domain requires a matching memory budget")
        return self

    def budget_for(self, domain: MemoryDomain) -> MemoryBudget | None:
        return next((item for item in self.memory_budgets if item.domain == domain), None)


class LinkProfile(StrictModel):
    schema_version: str = PROFILE_SCHEMA_VERSION
    source_worker_id: str
    target_worker_id: str
    connection_type: ConnectionType = ConnectionType.UNKNOWN
    fixed_latency_ms: Annotated[float, Field(ge=0.0)]
    bandwidth_bytes_per_second: Annotated[float, Field(gt=0.0)]
    sender_conversion_ms: Annotated[float, Field(ge=0.0)] = 0.0
    receiver_conversion_ms: Annotated[float, Field(ge=0.0)] = 0.0
    provenance: Provenance = Provenance.CONFIGURED
    observed_at: str | None = None


class WorkloadProfile(StrictModel):
    schema_version: str = PROFILE_SCHEMA_VERSION
    workload_id: str
    prompt_tokens: PositiveInt
    output_tokens: PositiveInt
    concurrency: PositiveInt = 1
    total_cached_tokens: PositiveInt
    kv_overhead_factor: Annotated[float, Field(ge=1.0)] = 1.1
    kv_dtype: DType = DType.F16
    activation_dtype: DType = DType.F16


class ObjectiveWeights(StrictModel):
    ttft: Annotated[float, Field(ge=0.0)] = 0.25
    itl: Annotated[float, Field(ge=0.0)] = 0.50
    pipeline_period: Annotated[float, Field(ge=0.0)] = 0.20
    memory_pressure: Annotated[float, Field(ge=0.0)] = 0.05


class PlannerSettings(StrictModel):
    mode: PlanningMode = PlanningMode.FEASIBILITY
    execution_dtype: DType = DType.F16
    objective_weights: ObjectiveWeights = ObjectiveWeights()


class StageMemory(StrictModel):
    worker_id: str
    domain: MemoryDomain
    layer_start: NonNegativeInt
    layer_end: PositiveInt
    weight_bytes: NonNegativeInt
    kv_cache_bytes: NonNegativeInt
    workspace_bytes: NonNegativeInt
    activation_buffer_bytes: NonNegativeInt
    allocator_allowance_bytes: NonNegativeInt
    required_bytes: NonNegativeInt
    usable_bytes: NonNegativeInt
    pressure: Annotated[float, Field(ge=0.0)]


class PerformanceEstimate(StrictModel):
    boundary_prefill_ms: Annotated[float, Field(ge=0.0)]
    boundary_decode_ms: Annotated[float, Field(ge=0.0)]
    confidence: str


class PlanCandidate(StrictModel):
    candidate_id: str
    stage_zero_worker_id: str
    final_stage_worker_id: str
    split_layer: PositiveInt
    stages: tuple[StageMemory, StageMemory]
    feasible: bool
    rejection_reasons: tuple[str, ...] = ()
    performance: PerformanceEstimate | None = None
    score: float | None = None
    rank: PositiveInt | None = None


class StageAssignment(StrictModel):
    stage_index: NonNegativeInt
    worker_id: str
    layer_start: NonNegativeInt
    layer_end: PositiveInt
    owns_token_embedding: bool
    owns_final_norm: bool
    owns_lm_head: bool
    owns_sampling: bool


class DeploymentPlan(StrictModel):
    schema_version: str = PLAN_SCHEMA_VERSION
    planner_version: str = PLANNER_VERSION
    plan_id: str
    plan_digest: str
    manifest_digest: str
    workload_id: str
    planning_mode: PlanningMode
    execution_dtype: DType
    activation_dtype: DType
    split_layer: PositiveInt
    stages: tuple[StageAssignment, StageAssignment]
    duplicated_tensor_groups: tuple[str, ...] = ()
    selected_candidate_id: str


class PlanningReport(StrictModel):
    schema_version: str = PLAN_SCHEMA_VERSION
    planner_version: str = PLANNER_VERSION
    manifest_digest: str
    workload_id: str
    mode: PlanningMode
    selected_candidate_id: str | None
    candidates: tuple[PlanCandidate, ...]
    plan: DeploymentPlan | None
    notes: tuple[str, ...] = ()


JsonObject = dict[str, Any]
