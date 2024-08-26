from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientGrpcMessage(_message.Message):
    __slots__ = ("send_parameters", "get_new_version")
    class SendParameters(_message.Message):
        __slots__ = ("client_id", "train_dataset_length", "new_model_weight")
        class NewModelWeightEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: bytes
            def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
        CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
        TRAIN_DATASET_LENGTH_FIELD_NUMBER: _ClassVar[int]
        NEW_MODEL_WEIGHT_FIELD_NUMBER: _ClassVar[int]
        client_id: str
        train_dataset_length: int
        new_model_weight: _containers.ScalarMap[str, bytes]
        def __init__(self, client_id: _Optional[str] = ..., train_dataset_length: _Optional[int] = ..., new_model_weight: _Optional[_Mapping[str, bytes]] = ...) -> None: ...
    class GetNewVersion(_message.Message):
        __slots__ = ("version_path",)
        VERSION_PATH_FIELD_NUMBER: _ClassVar[int]
        version_path: str
        def __init__(self, version_path: _Optional[str] = ...) -> None: ...
    SEND_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    GET_NEW_VERSION_FIELD_NUMBER: _ClassVar[int]
    send_parameters: ClientGrpcMessage.SendParameters
    get_new_version: ClientGrpcMessage.GetNewVersion
    def __init__(self, send_parameters: _Optional[_Union[ClientGrpcMessage.SendParameters, _Mapping]] = ..., get_new_version: _Optional[_Union[ClientGrpcMessage.GetNewVersion, _Mapping]] = ...) -> None: ...

class TransferStatus(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: bool
    message: str
    def __init__(self, code: bool = ..., message: _Optional[str] = ...) -> None: ...
