"""Explicit mappings between persisted domain models and hLLM v1 wire messages."""

from __future__ import annotations

from hllm_control import models
from hllm_control.proto import common_pb2, model_pb2, placement_pb2

_DTYPE_TO_PROTO: dict[models.DType, int] = {
    models.DType.BOOL: common_pb2.DATA_TYPE_BOOL,
    models.DType.U8: common_pb2.DATA_TYPE_U8,
    models.DType.I8: common_pb2.DATA_TYPE_I8,
    models.DType.I16: common_pb2.DATA_TYPE_I16,
    models.DType.U16: common_pb2.DATA_TYPE_U16,
    models.DType.I32: common_pb2.DATA_TYPE_I32,
    models.DType.U32: common_pb2.DATA_TYPE_U32,
    models.DType.I64: common_pb2.DATA_TYPE_I64,
    models.DType.U64: common_pb2.DATA_TYPE_U64,
    models.DType.F16: common_pb2.DATA_TYPE_F16,
    models.DType.BF16: common_pb2.DATA_TYPE_BF16,
    models.DType.F32: common_pb2.DATA_TYPE_F32,
    models.DType.F64: common_pb2.DATA_TYPE_F64,
}
_PROTO_TO_DTYPE = {value: key for key, value in _DTYPE_TO_PROTO.items()}

_ROLE_TO_PROTO: dict[models.TensorRole, int] = {
    models.TensorRole.TOKEN_EMBEDDING: model_pb2.TENSOR_ROLE_TOKEN_EMBEDDING,
    models.TensorRole.TRANSFORMER_LAYER: model_pb2.TENSOR_ROLE_TRANSFORMER_LAYER,
    models.TensorRole.FINAL_NORM: model_pb2.TENSOR_ROLE_FINAL_NORM,
    models.TensorRole.LM_HEAD: model_pb2.TENSOR_ROLE_LM_HEAD,
    models.TensorRole.ARCHITECTURE_STATE: model_pb2.TENSOR_ROLE_ARCHITECTURE_STATE,
}
_PROTO_TO_ROLE = {value: key for key, value in _ROLE_TO_PROTO.items()}


class WireMappingError(ValueError):
    """Raised when an artifact cannot be represented by the current wire contract."""


def _version_to_proto(version: str) -> common_pb2.ArtifactVersion:
    parts = version.split(".")
    if len(parts) != 2 or any(not item.isdecimal() for item in parts):
        raise WireMappingError(f"artifact version {version!r} is not major.minor")
    return common_pb2.ArtifactVersion(major=int(parts[0]), minor=int(parts[1]))


def _version_from_proto(version: common_pb2.ArtifactVersion) -> str:
    return f"{version.major}.{version.minor}"


def _required_dtype(value: int) -> models.DType:
    try:
        return _PROTO_TO_DTYPE[value]
    except KeyError as error:
        raise WireMappingError(f"unsupported wire dtype value {value}") from error


def _required_role(value: int) -> models.TensorRole:
    try:
        return _PROTO_TO_ROLE[value]
    except KeyError as error:
        raise WireMappingError(f"unsupported wire tensor role value {value}") from error


def model_manifest_to_proto(manifest: models.ModelManifest) -> model_pb2.ModelManifest:
    rope_scaling = None
    if manifest.config.rope_scaling is not None:
        rope_scaling = model_pb2.RopeScaling(
            scaling_type=manifest.config.rope_scaling.scaling_type,
            factor=manifest.config.rope_scaling.factor,
        )
        if manifest.config.rope_scaling.original_max_position_embeddings is not None:
            rope_scaling.original_max_position_embeddings = (
                manifest.config.rope_scaling.original_max_position_embeddings
            )

    config = model_pb2.ModelConfig(
        hidden_size=manifest.config.hidden_size,
        intermediate_size=manifest.config.intermediate_size,
        num_layers=manifest.config.num_layers,
        num_attention_heads=manifest.config.num_attention_heads,
        num_kv_heads=manifest.config.num_kv_heads,
        head_dim=manifest.config.head_dim,
        vocabulary_size=manifest.config.vocabulary_size,
        maximum_sequence_length=manifest.config.maximum_sequence_length,
        tied_embeddings=manifest.config.tied_embeddings,
        rms_norm_eps=manifest.config.rms_norm_eps,
        rope_theta=manifest.config.rope_theta,
        hidden_activation=manifest.config.hidden_activation,
        attention_bias=manifest.config.attention_bias,
        mlp_bias=manifest.config.mlp_bias,
        eos_token_ids=manifest.config.eos_token_ids,
    )
    if rope_scaling is not None:
        config.rope_scaling.CopyFrom(rope_scaling)

    tensor_records: list[model_pb2.TensorRecord] = []
    for tensor in manifest.tensors:
        message = model_pb2.TensorRecord(
            name=tensor.name,
            file=tensor.file,
            dtype=f"DATA_TYPE_{tensor.dtype.value}",
            shape=tensor.shape,
            data_offset=tensor.data_offset,
            byte_length=tensor.byte_length,
            role=f"TENSOR_ROLE_{tensor.role.value}",
            shared_weight_group=tensor.shared_weight_group or "",
        )
        if tensor.layer_index is not None:
            message.layer_index = tensor.layer_index
        tensor_records.append(message)

    components: list[model_pb2.ComponentMemory] = []
    for component in manifest.components:
        message = model_pb2.ComponentMemory(
            role=f"TENSOR_ROLE_{component.role.value}", storage_bytes=component.storage_bytes
        )
        if component.layer_index is not None:
            message.layer_index = component.layer_index
        components.append(message)

    return model_pb2.ModelManifest(
        schema_version=_version_to_proto(manifest.schema_version),
        manifest_id=manifest.manifest_id,
        manifest_digest=manifest.manifest_digest,
        source=model_pb2.ModelSource(
            model_id=manifest.source.model_id,
            revision=manifest.source.revision or "",
            config_sha256=manifest.source.config_sha256,
            index_sha256=manifest.source.index_sha256 or "",
            tensor_metadata_sha256=manifest.source.tensor_metadata_sha256 or "",
        ),
        architecture=model_pb2.ArchitectureDescriptor(
            architecture_id=manifest.architecture.architecture_id,
            architecture_revision=manifest.architecture.architecture_revision,
            feature_flags=manifest.architecture.feature_flags,
        ),
        config=config,
        tensor_files=(
            model_pb2.TensorFile(
                name=item.name,
                size_bytes=item.size_bytes,
                header_size_bytes=item.header_size_bytes,
                sha256=item.sha256 or "",
            )
            for item in manifest.tensor_files
        ),
        tensors=tensor_records,
        components=components,
        total_storage_bytes=manifest.total_storage_bytes,
    )


def model_manifest_from_proto(message: model_pb2.ModelManifest) -> models.ModelManifest:
    rope_scaling = None
    if message.config.HasField("rope_scaling"):
        rope_scaling = models.RopeScaling(
            scaling_type=message.config.rope_scaling.scaling_type,
            factor=message.config.rope_scaling.factor,
            original_max_position_embeddings=(
                message.config.rope_scaling.original_max_position_embeddings
                if message.config.rope_scaling.HasField("original_max_position_embeddings")
                else None
            ),
        )
    return models.ModelManifest(
        schema_version=_version_from_proto(message.schema_version),
        manifest_id=message.manifest_id,
        manifest_digest=message.manifest_digest,
        source=models.SourceDescriptor(
            model_id=message.source.model_id,
            revision=message.source.revision or None,
            config_sha256=message.source.config_sha256,
            index_sha256=message.source.index_sha256 or None,
            tensor_metadata_sha256=message.source.tensor_metadata_sha256 or None,
        ),
        architecture=models.ArchitectureDescriptor(
            architecture_id=message.architecture.architecture_id,
            architecture_revision=message.architecture.architecture_revision,
            feature_flags=tuple(message.architecture.feature_flags),
        ),
        config=models.ModelConfig(
            hidden_size=message.config.hidden_size,
            intermediate_size=message.config.intermediate_size,
            num_layers=message.config.num_layers,
            num_attention_heads=message.config.num_attention_heads,
            num_kv_heads=message.config.num_kv_heads,
            head_dim=message.config.head_dim,
            vocabulary_size=message.config.vocabulary_size,
            maximum_sequence_length=message.config.maximum_sequence_length,
            tied_embeddings=message.config.tied_embeddings,
            rms_norm_eps=message.config.rms_norm_eps,
            rope_theta=message.config.rope_theta,
            rope_scaling=rope_scaling,
            hidden_activation=message.config.hidden_activation,
            attention_bias=message.config.attention_bias,
            mlp_bias=message.config.mlp_bias,
            eos_token_ids=tuple(message.config.eos_token_ids),
        ),
        tensor_files=tuple(
            models.TensorFile(
                name=item.name,
                size_bytes=item.size_bytes,
                header_size_bytes=item.header_size_bytes,
                sha256=item.sha256 or None,
            )
            for item in message.tensor_files
        ),
        tensors=tuple(
            models.TensorRecord(
                name=item.name,
                file=item.file,
                dtype=_required_dtype(item.dtype),
                shape=tuple(item.shape),
                data_offset=item.data_offset,
                byte_length=item.byte_length,
                role=_required_role(item.role),
                layer_index=item.layer_index if item.HasField("layer_index") else None,
                shared_weight_group=item.shared_weight_group or None,
            )
            for item in message.tensors
        ),
        components=tuple(
            models.ComponentMemory(
                role=_required_role(item.role),
                layer_index=item.layer_index if item.HasField("layer_index") else None,
                storage_bytes=item.storage_bytes,
            )
            for item in message.components
        ),
        total_storage_bytes=message.total_storage_bytes,
    )


def deployment_plan_to_proto(plan: models.DeploymentPlan) -> placement_pb2.DeploymentPlan:
    return placement_pb2.DeploymentPlan(
        schema_version=_version_to_proto(plan.schema_version),
        planner_version=plan.planner_version,
        plan_id=plan.plan_id,
        plan_digest=plan.plan_digest,
        manifest_digest=plan.manifest_digest,
        workload_id=plan.workload_id,
        planning_mode=plan.planning_mode.value,
        execution_dtype=f"DATA_TYPE_{plan.execution_dtype.value}",
        activation_dtype=f"DATA_TYPE_{plan.activation_dtype.value}",
        split_layer=plan.split_layer,
        stages=(
            placement_pb2.StageAssignment(
                stage_index=item.stage_index,
                worker_id=item.worker_id,
                layer_start=item.layer_start,
                layer_end=item.layer_end,
                owns_token_embedding=item.owns_token_embedding,
                owns_final_norm=item.owns_final_norm,
                owns_lm_head=item.owns_lm_head,
                owns_sampling=item.owns_sampling,
            )
            for item in plan.stages
        ),
        selected_candidate_id=plan.selected_candidate_id,
        duplicated_tensor_groups=plan.duplicated_tensor_groups,
        deployment_version=plan.deployment_version,
    )


def deployment_plan_from_proto(message: placement_pb2.DeploymentPlan) -> models.DeploymentPlan:
    stages = tuple(
        models.StageAssignment(
            stage_index=item.stage_index,
            worker_id=item.worker_id,
            layer_start=item.layer_start,
            layer_end=item.layer_end,
            owns_token_embedding=item.owns_token_embedding,
            owns_final_norm=item.owns_final_norm,
            owns_lm_head=item.owns_lm_head,
            owns_sampling=item.owns_sampling,
        )
        for item in message.stages
    )
    if len(stages) not in (1, 2):
        raise WireMappingError(f"deployment plan has {len(stages)} stages; expected one or two")
    return models.DeploymentPlan(
        schema_version=_version_from_proto(message.schema_version),
        planner_version=message.planner_version,
        plan_id=message.plan_id,
        plan_digest=message.plan_digest,
        manifest_digest=message.manifest_digest,
        workload_id=message.workload_id,
        planning_mode=models.PlanningMode(message.planning_mode),
        execution_dtype=_required_dtype(message.execution_dtype),
        activation_dtype=_required_dtype(message.activation_dtype),
        split_layer=message.split_layer,
        stages=stages,
        duplicated_tensor_groups=tuple(message.duplicated_tensor_groups),
        selected_candidate_id=message.selected_candidate_id,
        deployment_version=message.deployment_version,
    )
