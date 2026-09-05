from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TraceContext(_message.Message):
    __slots__ = ("trace_id", "deployment_id", "request_id", "microbatch_id", "token_position", "stage_index", "worker_id", "sequence_number")
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MICROBATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_POSITION_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    deployment_id: str
    request_id: str
    microbatch_id: int
    token_position: int
    stage_index: int
    worker_id: str
    sequence_number: int
    def __init__(self, trace_id: _Optional[str] = ..., deployment_id: _Optional[str] = ..., request_id: _Optional[str] = ..., microbatch_id: _Optional[int] = ..., token_position: _Optional[int] = ..., stage_index: _Optional[int] = ..., worker_id: _Optional[str] = ..., sequence_number: _Optional[int] = ...) -> None: ...
