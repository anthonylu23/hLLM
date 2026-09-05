from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_TYPE_UNSPECIFIED: _ClassVar[DataType]
    DATA_TYPE_BOOL: _ClassVar[DataType]
    DATA_TYPE_U8: _ClassVar[DataType]
    DATA_TYPE_I8: _ClassVar[DataType]
    DATA_TYPE_I16: _ClassVar[DataType]
    DATA_TYPE_U16: _ClassVar[DataType]
    DATA_TYPE_I32: _ClassVar[DataType]
    DATA_TYPE_U32: _ClassVar[DataType]
    DATA_TYPE_I64: _ClassVar[DataType]
    DATA_TYPE_U64: _ClassVar[DataType]
    DATA_TYPE_F16: _ClassVar[DataType]
    DATA_TYPE_BF16: _ClassVar[DataType]
    DATA_TYPE_F32: _ClassVar[DataType]
    DATA_TYPE_F64: _ClassVar[DataType]

class Provenance(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROVENANCE_UNSPECIFIED: _ClassVar[Provenance]
    PROVENANCE_THEORETICAL: _ClassVar[Provenance]
    PROVENANCE_CONFIGURED: _ClassVar[Provenance]
    PROVENANCE_OBSERVED: _ClassVar[Provenance]
    PROVENANCE_MEASURED: _ClassVar[Provenance]

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CODE_UNSPECIFIED: _ClassVar[ErrorCode]
    ERROR_CODE_INVALID_REQUEST: _ClassVar[ErrorCode]
    ERROR_CODE_INCOMPATIBLE_WORKER: _ClassVar[ErrorCode]
    ERROR_CODE_STALE_DEPLOYMENT: _ClassVar[ErrorCode]
    ERROR_CODE_OUT_OF_ORDER: _ClassVar[ErrorCode]
    ERROR_CODE_RESOURCE_EXHAUSTED: _ClassVar[ErrorCode]
    ERROR_CODE_DEADLINE_EXCEEDED: _ClassVar[ErrorCode]
    ERROR_CODE_WORKER_UNAVAILABLE: _ClassVar[ErrorCode]
    ERROR_CODE_BACKEND_ERROR: _ClassVar[ErrorCode]
    ERROR_CODE_TRANSPORT_ERROR: _ClassVar[ErrorCode]
DATA_TYPE_UNSPECIFIED: DataType
DATA_TYPE_BOOL: DataType
DATA_TYPE_U8: DataType
DATA_TYPE_I8: DataType
DATA_TYPE_I16: DataType
DATA_TYPE_U16: DataType
DATA_TYPE_I32: DataType
DATA_TYPE_U32: DataType
DATA_TYPE_I64: DataType
DATA_TYPE_U64: DataType
DATA_TYPE_F16: DataType
DATA_TYPE_BF16: DataType
DATA_TYPE_F32: DataType
DATA_TYPE_F64: DataType
PROVENANCE_UNSPECIFIED: Provenance
PROVENANCE_THEORETICAL: Provenance
PROVENANCE_CONFIGURED: Provenance
PROVENANCE_OBSERVED: Provenance
PROVENANCE_MEASURED: Provenance
ERROR_CODE_UNSPECIFIED: ErrorCode
ERROR_CODE_INVALID_REQUEST: ErrorCode
ERROR_CODE_INCOMPATIBLE_WORKER: ErrorCode
ERROR_CODE_STALE_DEPLOYMENT: ErrorCode
ERROR_CODE_OUT_OF_ORDER: ErrorCode
ERROR_CODE_RESOURCE_EXHAUSTED: ErrorCode
ERROR_CODE_DEADLINE_EXCEEDED: ErrorCode
ERROR_CODE_WORKER_UNAVAILABLE: ErrorCode
ERROR_CODE_BACKEND_ERROR: ErrorCode
ERROR_CODE_TRANSPORT_ERROR: ErrorCode

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ArtifactVersion(_message.Message):
    __slots__ = ("major", "minor")
    MAJOR_FIELD_NUMBER: _ClassVar[int]
    MINOR_FIELD_NUMBER: _ClassVar[int]
    major: int
    minor: int
    def __init__(self, major: _Optional[int] = ..., minor: _Optional[int] = ...) -> None: ...

class RuntimeError(_message.Message):
    __slots__ = ("code", "detail", "retryable")
    CODE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    detail: str
    retryable: bool
    def __init__(self, code: _Optional[_Union[ErrorCode, str]] = ..., detail: _Optional[str] = ..., retryable: _Optional[bool] = ...) -> None: ...
