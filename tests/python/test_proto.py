from __future__ import annotations

from pathlib import Path

from google.protobuf.descriptor_pb2 import FileDescriptorSet
from grpc_tools import protoc


def test_protobuf_contracts_compile(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    proto = root / "proto"
    output = tmp_path / "generated"
    output.mkdir()
    descriptor_path = tmp_path / "hllm.pb"
    files = sorted(str(path) for path in proto.glob("*.proto"))

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            f"--descriptor_set_out={descriptor_path}",
            "--include_imports",
            *files,
        ]
    )

    assert result == 0

    descriptor_set = FileDescriptorSet.FromString(descriptor_path.read_bytes())
    messages = {
        f"{file.package}.{message.name}": message
        for file in descriptor_set.file
        for message in file.message_type
    }

    model_config_fields = {field.name for field in messages["hllm.v1.ModelConfig"].field}
    assert {
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "hidden_activation",
        "eos_token_ids",
    } <= model_config_fields

    tensor_fields = {field.name for field in messages["hllm.v1.TensorEnvelope"].field}
    assert "deployment_version" in tensor_fields

    stage_message = messages["hllm.v1.StageMessage"]
    assert {field.name for field in stage_message.field} == {
        "open_sequence",
        "tensor",
        "sampled_token",
        "terminate",
        "error",
    }
    assert [oneof.name for oneof in stage_message.oneof_decl] == ["body"]

    generation_event = messages["hllm.v1.GenerationEvent"]
    assert {field.name for field in generation_event.field} == {
        "request_id",
        "prefill_complete",
        "token",
        "usage",
        "terminal",
    }
    assert [oneof.name for oneof in generation_event.oneof_decl] == ["body"]
