from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExecutionPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_PHASE_UNSPECIFIED: _ClassVar[ExecutionPhase]
    EXECUTION_PHASE_PREFILL: _ClassVar[ExecutionPhase]
    EXECUTION_PHASE_DECODE: _ClassVar[ExecutionPhase]

class TerminalState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TERMINAL_STATE_UNSPECIFIED: _ClassVar[TerminalState]
    TERMINAL_STATE_COMPLETED: _ClassVar[TerminalState]
    TERMINAL_STATE_CANCELLED: _ClassVar[TerminalState]
    TERMINAL_STATE_DEADLINE_EXCEEDED: _ClassVar[TerminalState]
    TERMINAL_STATE_FAILED: _ClassVar[TerminalState]
EXECUTION_PHASE_UNSPECIFIED: ExecutionPhase
EXECUTION_PHASE_PREFILL: ExecutionPhase
EXECUTION_PHASE_DECODE: ExecutionPhase
TERMINAL_STATE_UNSPECIFIED: TerminalState
TERMINAL_STATE_COMPLETED: TerminalState
TERMINAL_STATE_CANCELLED: TerminalState
TERMINAL_STATE_DEADLINE_EXCEEDED: TerminalState
TERMINAL_STATE_FAILED: TerminalState

class SequenceOpen(_message.Message):
    __slots__ = ("protocol_version", "deployment_id", "deployment_version", "request_id", "microbatch_id", "maximum_total_tokens", "maximum_new_tokens", "stop_token_ids", "deadline_unix_ms")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    deployment_id: str
    deployment_version: int
    request_id: str
    microbatch_id: int
    maximum_total_tokens: int
    maximum_new_tokens: int
    stop_token_ids: _containers.RepeatedScalarFieldContainer[int]
    deadline_unix_ms: int
    def __init__(self, protocol_version: _Optional[int] = ..., deployment_id: _Optional[str] = ..., deployment_version: _Optional[int] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., maximum_total_tokens: _Optional[int] = ..., maximum_new_tokens: _Optional[int] = ..., stop_token_ids: _Optional[_Iterable[int]] = ..., deadline_unix_ms: _Optional[int] = ...) -> None: ...

class TensorEnvelope(_message.Message):
    __slots__ = ("protocol_version", "deployment_id", "request_id", "microbatch_id", "sequence_number", "phase", "first_position", "sequence_lengths", "cache_slot_ids", "shape", "dtype", "layout", "payload_length", "checksum", "payload", "deployment_version")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    FIRST_POSITION_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_LENGTHS_FIELD_NUMBER: _ClassVar[int]
    CACHE_SLOT_IDS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_LENGTH_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    deployment_id: str
    request_id: str
    microbatch_id: int
    sequence_number: int
    phase: ExecutionPhase
    first_position: int
    sequence_lengths: _containers.RepeatedScalarFieldContainer[int]
    cache_slot_ids: _containers.RepeatedScalarFieldContainer[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: _common_pb2.DataType
    layout: str
    payload_length: int
    checksum: bytes
    payload: bytes
    deployment_version: int
    def __init__(self, protocol_version: _Optional[int] = ..., deployment_id: _Optional[str] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., sequence_number: _Optional[int] = ..., phase: _Optional[_Union[ExecutionPhase, str]] = ..., first_position: _Optional[int] = ..., sequence_lengths: _Optional[_Iterable[int]] = ..., cache_slot_ids: _Optional[_Iterable[int]] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[_common_pb2.DataType, str]] = ..., layout: _Optional[str] = ..., payload_length: _Optional[int] = ..., checksum: _Optional[bytes] = ..., payload: _Optional[bytes] = ..., deployment_version: _Optional[int] = ...) -> None: ...

class SampledToken(_message.Message):
    __slots__ = ("deployment_id", "deployment_version", "request_id", "microbatch_id", "sequence_number", "token_position", "token_id")
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TOKEN_POSITION_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    deployment_id: str
    deployment_version: int
    request_id: str
    microbatch_id: int
    sequence_number: int
    token_position: int
    token_id: int
    def __init__(self, deployment_id: _Optional[str] = ..., deployment_version: _Optional[int] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., sequence_number: _Optional[int] = ..., token_position: _Optional[int] = ..., token_id: _Optional[int] = ...) -> None: ...

class SequenceTermination(_message.Message):
    __slots__ = ("deployment_id", "deployment_version", "request_id", "microbatch_id", "state", "error", "prompt_tokens", "generated_tokens")
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    deployment_id: str
    deployment_version: int
    request_id: str
    microbatch_id: int
    state: TerminalState
    error: _common_pb2.RuntimeError
    prompt_tokens: int
    generated_tokens: int
    def __init__(self, deployment_id: _Optional[str] = ..., deployment_version: _Optional[int] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., state: _Optional[_Union[TerminalState, str]] = ..., error: _Optional[_Union[_common_pb2.RuntimeError, _Mapping]] = ..., prompt_tokens: _Optional[int] = ..., generated_tokens: _Optional[int] = ...) -> None: ...

class StageError(_message.Message):
    __slots__ = ("deployment_id", "deployment_version", "request_id", "microbatch_id", "sequence_number", "error")
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    deployment_id: str
    deployment_version: int
    request_id: str
    microbatch_id: int
    sequence_number: int
    error: _common_pb2.RuntimeError
    def __init__(self, deployment_id: _Optional[str] = ..., deployment_version: _Optional[int] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., sequence_number: _Optional[int] = ..., error: _Optional[_Union[_common_pb2.RuntimeError, _Mapping]] = ...) -> None: ...

class StageMessage(_message.Message):
    __slots__ = ("open_sequence", "tensor", "sampled_token", "terminate", "error")
    OPEN_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    SAMPLED_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TERMINATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    open_sequence: SequenceOpen
    tensor: TensorEnvelope
    sampled_token: SampledToken
    terminate: SequenceTermination
    error: StageError
    def __init__(self, open_sequence: _Optional[_Union[SequenceOpen, _Mapping]] = ..., tensor: _Optional[_Union[TensorEnvelope, _Mapping]] = ..., sampled_token: _Optional[_Union[SampledToken, _Mapping]] = ..., terminate: _Optional[_Union[SequenceTermination, _Mapping]] = ..., error: _Optional[_Union[StageError, _Mapping]] = ...) -> None: ...

class GenerationRequest(_message.Message):
    __slots__ = ("deployment_id", "request_id", "token_ids", "maximum_new_tokens", "deployment_version", "stop_token_ids", "deadline_unix_ms")
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    deployment_id: str
    request_id: str
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    maximum_new_tokens: int
    deployment_version: int
    stop_token_ids: _containers.RepeatedScalarFieldContainer[int]
    deadline_unix_ms: int
    def __init__(self, deployment_id: _Optional[str] = ..., request_id: _Optional[str] = ..., token_ids: _Optional[_Iterable[int]] = ..., maximum_new_tokens: _Optional[int] = ..., deployment_version: _Optional[int] = ..., stop_token_ids: _Optional[_Iterable[int]] = ..., deadline_unix_ms: _Optional[int] = ...) -> None: ...

class PrefillComplete(_message.Message):
    __slots__ = ("prompt_tokens",)
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    def __init__(self, prompt_tokens: _Optional[int] = ...) -> None: ...

class TokenEvent(_message.Message):
    __slots__ = ("token_id", "token_position")
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_POSITION_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    token_position: int
    def __init__(self, token_id: _Optional[int] = ..., token_position: _Optional[int] = ...) -> None: ...

class UsageEvent(_message.Message):
    __slots__ = ("prompt_tokens", "generated_tokens")
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    generated_tokens: int
    def __init__(self, prompt_tokens: _Optional[int] = ..., generated_tokens: _Optional[int] = ...) -> None: ...

class TerminalEvent(_message.Message):
    __slots__ = ("state", "error")
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    state: TerminalState
    error: _common_pb2.RuntimeError
    def __init__(self, state: _Optional[_Union[TerminalState, str]] = ..., error: _Optional[_Union[_common_pb2.RuntimeError, _Mapping]] = ...) -> None: ...

class GenerationEvent(_message.Message):
    __slots__ = ("request_id", "prefill_complete", "token", "usage", "terminal")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PREFILL_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prefill_complete: PrefillComplete
    token: TokenEvent
    usage: UsageEvent
    terminal: TerminalEvent
    def __init__(self, request_id: _Optional[str] = ..., prefill_complete: _Optional[_Union[PrefillComplete, _Mapping]] = ..., token: _Optional[_Union[TokenEvent, _Mapping]] = ..., usage: _Optional[_Union[UsageEvent, _Mapping]] = ..., terminal: _Optional[_Union[TerminalEvent, _Mapping]] = ...) -> None: ...
