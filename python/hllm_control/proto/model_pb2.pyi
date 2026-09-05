from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TENSOR_ROLE_UNSPECIFIED: _ClassVar[TensorRole]
    TENSOR_ROLE_TOKEN_EMBEDDING: _ClassVar[TensorRole]
    TENSOR_ROLE_TRANSFORMER_LAYER: _ClassVar[TensorRole]
    TENSOR_ROLE_FINAL_NORM: _ClassVar[TensorRole]
    TENSOR_ROLE_LM_HEAD: _ClassVar[TensorRole]
    TENSOR_ROLE_ARCHITECTURE_STATE: _ClassVar[TensorRole]
TENSOR_ROLE_UNSPECIFIED: TensorRole
TENSOR_ROLE_TOKEN_EMBEDDING: TensorRole
TENSOR_ROLE_TRANSFORMER_LAYER: TensorRole
TENSOR_ROLE_FINAL_NORM: TensorRole
TENSOR_ROLE_LM_HEAD: TensorRole
TENSOR_ROLE_ARCHITECTURE_STATE: TensorRole

class ModelSource(_message.Message):
    __slots__ = ("model_id", "revision", "config_sha256", "index_sha256", "tensor_metadata_sha256")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    CONFIG_SHA256_FIELD_NUMBER: _ClassVar[int]
    INDEX_SHA256_FIELD_NUMBER: _ClassVar[int]
    TENSOR_METADATA_SHA256_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    revision: str
    config_sha256: str
    index_sha256: str
    tensor_metadata_sha256: str
    def __init__(self, model_id: _Optional[str] = ..., revision: _Optional[str] = ..., config_sha256: _Optional[str] = ..., index_sha256: _Optional[str] = ..., tensor_metadata_sha256: _Optional[str] = ...) -> None: ...

class ArchitectureDescriptor(_message.Message):
    __slots__ = ("architecture_id", "architecture_revision", "feature_flags")
    ARCHITECTURE_ID_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_REVISION_FIELD_NUMBER: _ClassVar[int]
    FEATURE_FLAGS_FIELD_NUMBER: _ClassVar[int]
    architecture_id: str
    architecture_revision: str
    feature_flags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, architecture_id: _Optional[str] = ..., architecture_revision: _Optional[str] = ..., feature_flags: _Optional[_Iterable[str]] = ...) -> None: ...

class RopeScaling(_message.Message):
    __slots__ = ("scaling_type", "factor", "original_max_position_embeddings")
    SCALING_TYPE_FIELD_NUMBER: _ClassVar[int]
    FACTOR_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_MAX_POSITION_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    scaling_type: str
    factor: float
    original_max_position_embeddings: int
    def __init__(self, scaling_type: _Optional[str] = ..., factor: _Optional[float] = ..., original_max_position_embeddings: _Optional[int] = ...) -> None: ...

class ModelConfig(_message.Message):
    __slots__ = ("hidden_size", "intermediate_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "vocabulary_size", "maximum_sequence_length", "tied_embeddings", "rms_norm_eps", "rope_theta", "rope_scaling", "hidden_activation", "attention_bias", "mlp_bias", "eos_token_ids")
    HIDDEN_SIZE_FIELD_NUMBER: _ClassVar[int]
    INTERMEDIATE_SIZE_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NUM_ATTENTION_HEADS_FIELD_NUMBER: _ClassVar[int]
    NUM_KV_HEADS_FIELD_NUMBER: _ClassVar[int]
    HEAD_DIM_FIELD_NUMBER: _ClassVar[int]
    VOCABULARY_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_SEQUENCE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    TIED_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    RMS_NORM_EPS_FIELD_NUMBER: _ClassVar[int]
    ROPE_THETA_FIELD_NUMBER: _ClassVar[int]
    ROPE_SCALING_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_BIAS_FIELD_NUMBER: _ClassVar[int]
    MLP_BIAS_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocabulary_size: int
    maximum_sequence_length: int
    tied_embeddings: bool
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: RopeScaling
    hidden_activation: str
    attention_bias: bool
    mlp_bias: bool
    eos_token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, hidden_size: _Optional[int] = ..., intermediate_size: _Optional[int] = ..., num_layers: _Optional[int] = ..., num_attention_heads: _Optional[int] = ..., num_kv_heads: _Optional[int] = ..., head_dim: _Optional[int] = ..., vocabulary_size: _Optional[int] = ..., maximum_sequence_length: _Optional[int] = ..., tied_embeddings: _Optional[bool] = ..., rms_norm_eps: _Optional[float] = ..., rope_theta: _Optional[float] = ..., rope_scaling: _Optional[_Union[RopeScaling, _Mapping]] = ..., hidden_activation: _Optional[str] = ..., attention_bias: _Optional[bool] = ..., mlp_bias: _Optional[bool] = ..., eos_token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class TensorFile(_message.Message):
    __slots__ = ("name", "size_bytes", "header_size_bytes", "sha256")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    HEADER_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    name: str
    size_bytes: int
    header_size_bytes: int
    sha256: str
    def __init__(self, name: _Optional[str] = ..., size_bytes: _Optional[int] = ..., header_size_bytes: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class TensorRecord(_message.Message):
    __slots__ = ("name", "file", "dtype", "shape", "data_offset", "byte_length", "role", "layer_index", "shared_weight_group")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_OFFSET_FIELD_NUMBER: _ClassVar[int]
    BYTE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    SHARED_WEIGHT_GROUP_FIELD_NUMBER: _ClassVar[int]
    name: str
    file: str
    dtype: _common_pb2.DataType
    shape: _containers.RepeatedScalarFieldContainer[int]
    data_offset: int
    byte_length: int
    role: TensorRole
    layer_index: int
    shared_weight_group: str
    def __init__(self, name: _Optional[str] = ..., file: _Optional[str] = ..., dtype: _Optional[_Union[_common_pb2.DataType, str]] = ..., shape: _Optional[_Iterable[int]] = ..., data_offset: _Optional[int] = ..., byte_length: _Optional[int] = ..., role: _Optional[_Union[TensorRole, str]] = ..., layer_index: _Optional[int] = ..., shared_weight_group: _Optional[str] = ...) -> None: ...

class ComponentMemory(_message.Message):
    __slots__ = ("role", "layer_index", "storage_bytes")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    STORAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    role: TensorRole
    layer_index: int
    storage_bytes: int
    def __init__(self, role: _Optional[_Union[TensorRole, str]] = ..., layer_index: _Optional[int] = ..., storage_bytes: _Optional[int] = ...) -> None: ...

class ModelManifest(_message.Message):
    __slots__ = ("schema_version", "manifest_id", "manifest_digest", "source", "architecture", "config", "tensor_files", "tensors", "components", "total_storage_bytes")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_ID_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FILES_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STORAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    schema_version: _common_pb2.ArtifactVersion
    manifest_id: str
    manifest_digest: str
    source: ModelSource
    architecture: ArchitectureDescriptor
    config: ModelConfig
    tensor_files: _containers.RepeatedCompositeFieldContainer[TensorFile]
    tensors: _containers.RepeatedCompositeFieldContainer[TensorRecord]
    components: _containers.RepeatedCompositeFieldContainer[ComponentMemory]
    total_storage_bytes: int
    def __init__(self, schema_version: _Optional[_Union[_common_pb2.ArtifactVersion, _Mapping]] = ..., manifest_id: _Optional[str] = ..., manifest_digest: _Optional[str] = ..., source: _Optional[_Union[ModelSource, _Mapping]] = ..., architecture: _Optional[_Union[ArchitectureDescriptor, _Mapping]] = ..., config: _Optional[_Union[ModelConfig, _Mapping]] = ..., tensor_files: _Optional[_Iterable[_Union[TensorFile, _Mapping]]] = ..., tensors: _Optional[_Iterable[_Union[TensorRecord, _Mapping]]] = ..., components: _Optional[_Iterable[_Union[ComponentMemory, _Mapping]]] = ..., total_storage_bytes: _Optional[int] = ...) -> None: ...
