import hashlib
import socket
import struct
from typing import Iterable


KV_ATTENTION_K = 0
KV_ATTENTION_V = 1
DSA_INDEX = 2
DSA_INDEX_SCALE = 3

KEY_LENGTH = 32

CMD_PUT = 0x01
CMD_GET = 0x02
CMD_EXISTS = 0x03
CMD_BATCH_PUT = 0x04
CMD_BATCH_GET = 0x05
CMD_BATCH_EXISTS = 0x06
CMD_CLEAR = 0x11
CMD_SIZE = 0x12

STATUS_OK = 0
STATUS_NOT_FOUND = 1
STATUS_BAD_REQUEST = 2
STATUS_INTERNAL_ERROR = 3

REQUEST_HEADER = struct.Struct("<BBHHI")
RESPONSE_HEADER = struct.Struct("<BI")
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")


class MicroKVError(RuntimeError):
    pass


class ProtocolError(MicroKVError):
    pass


def make_key(req_index: int | str, layer_id: int, token_pos: int, cache_type: int = KV_ATTENTION_K) -> bytes:
    if isinstance(req_index, str):
        req_bytes = hashlib.sha256(req_index.encode()).digest()[:16]
    elif isinstance(req_index, int):
        if req_index < 0 or req_index > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("integer req_index must fit uint64")
        req_bytes = struct.pack("<Q", req_index) + b"\x00" * 8
    else:
        raise TypeError("req_index must be int or str")

    _validate_u32("layer_id", layer_id)
    _validate_u32("token_pos", token_pos)
    _validate_type(cache_type)

    return (
        req_bytes
        + struct.pack("<I", layer_id)
        + struct.pack("<I", token_pos)
        + struct.pack("<B", cache_type)
        + b"\x00" * 7
    )


class KVStoreClient:
    def __init__(self, socket_path: str = "/tmp/microkv.sock") -> None:
        self.socket_path = socket_path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "KVStoreClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def put(self, cache_type: int, key: bytes, value: bytes) -> bool:
        _validate_type(cache_type)
        _validate_key(key)
        value = bytes(value)
        status, _ = self._request(CMD_PUT, cache_type, key, value)
        return status == STATUS_OK

    def get(self, cache_type: int, key: bytes) -> bytes | None:
        _validate_type(cache_type)
        _validate_key(key)
        status, payload = self._request(CMD_GET, cache_type, key, b"")
        if status == STATUS_NOT_FOUND:
            return None
        self._raise_unless_ok(status)
        return payload

    def exists(self, cache_type: int, key: bytes) -> bool:
        _validate_type(cache_type)
        _validate_key(key)
        status, payload = self._request(CMD_EXISTS, cache_type, key, b"")
        self._raise_unless_ok(status)
        if len(payload) != 1:
            raise ProtocolError(f"exists response length must be 1, got {len(payload)}")
        return payload[0] != 0

    def batch_put(self, cache_type: int, keys: list[bytes], values: list[bytes]) -> bool:
        _validate_type(cache_type)
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")
        body = bytearray()
        body += U32.pack(len(keys))
        for key, value in zip(keys, values, strict=True):
            _validate_key(key)
            value = bytes(value)
            body += key
            body += U32.pack(len(value))
            body += value
        status, _ = self._request_raw(CMD_BATCH_PUT, cache_type, KEY_LENGTH, 0, bytes(body))
        return status == STATUS_OK

    def batch_get(self, cache_type: int, keys: list[bytes]) -> list[bytes | None]:
        _validate_type(cache_type)
        body = _pack_keys(keys)
        status, payload = self._request_raw(CMD_BATCH_GET, cache_type, KEY_LENGTH, 0, body)
        self._raise_unless_ok(status)
        return _parse_batch_get_payload(payload)

    def batch_exists(self, cache_type: int, keys: list[bytes]) -> list[bool]:
        _validate_type(cache_type)
        body = _pack_keys(keys)
        status, payload = self._request_raw(CMD_BATCH_EXISTS, cache_type, KEY_LENGTH, 0, body)
        self._raise_unless_ok(status)
        return _parse_batch_exists_payload(payload)

    def clear(self, cache_type: int) -> None:
        _validate_type(cache_type)
        status, _ = self._request_raw(CMD_CLEAR, cache_type, 0, 0, b"")
        self._raise_unless_ok(status)

    def size(self, cache_type: int) -> int:
        _validate_type(cache_type)
        status, payload = self._request_raw(CMD_SIZE, cache_type, 0, 0, b"")
        self._raise_unless_ok(status)
        if len(payload) != U64.size:
            raise ProtocolError(f"size response length must be 8, got {len(payload)}")
        return U64.unpack(payload)[0]

    def _request(self, command: int, cache_type: int, key: bytes, value: bytes) -> tuple[int, bytes]:
        return self._request_raw(command, cache_type, len(key), len(value), key + value)

    def _request_raw(
        self,
        command: int,
        cache_type: int,
        key_len: int,
        val_len: int,
        body: bytes,
    ) -> tuple[int, bytes]:
        if self._sock is None:
            raise MicroKVError("client is closed")
        header = REQUEST_HEADER.pack(command, cache_type, 0, key_len, val_len)
        self._sock.sendall(header + body)
        response_header = _recv_exact(self._sock, RESPONSE_HEADER.size)
        status, payload_len = RESPONSE_HEADER.unpack(response_header)
        payload = _recv_exact(self._sock, payload_len)
        return status, payload

    def _raise_unless_ok(self, status: int) -> None:
        if status == STATUS_OK:
            return
        if status == STATUS_BAD_REQUEST:
            raise ProtocolError("server rejected request as bad protocol")
        if status == STATUS_INTERNAL_ERROR:
            raise MicroKVError("server returned internal error")
        raise MicroKVError(f"server returned status {status}")


def _pack_keys(keys: Iterable[bytes]) -> bytes:
    keys = list(keys)
    body = bytearray()
    body += U32.pack(len(keys))
    for key in keys:
        _validate_key(key)
        body += key
    return bytes(body)


def _parse_batch_get_payload(payload: bytes) -> list[bytes | None]:
    if len(payload) < U32.size:
        raise ProtocolError("batch_get payload missing count")
    count = U32.unpack_from(payload, 0)[0]
    offset = U32.size
    values: list[bytes | None] = []
    for _ in range(count):
        if offset + 5 > len(payload):
            raise ProtocolError("truncated batch_get item header")
        exists = payload[offset] != 0
        offset += 1
        value_len = U32.unpack_from(payload, offset)[0]
        offset += U32.size
        if offset + value_len > len(payload):
            raise ProtocolError("truncated batch_get item value")
        value = payload[offset : offset + value_len]
        offset += value_len
        values.append(value if exists else None)
    if offset != len(payload):
        raise ProtocolError("batch_get payload has trailing bytes")
    return values


def _parse_batch_exists_payload(payload: bytes) -> list[bool]:
    if len(payload) < U32.size:
        raise ProtocolError("batch_exists payload missing count")
    count = U32.unpack_from(payload, 0)[0]
    expected = U32.size + count
    if len(payload) != expected:
        raise ProtocolError(f"batch_exists payload length must be {expected}, got {len(payload)}")
    return [payload[U32.size + i] != 0 for i in range(count)]


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError("socket closed while reading response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if len(key) != KEY_LENGTH:
        raise ValueError(f"key must be {KEY_LENGTH} bytes, got {len(key)}")


def _validate_type(cache_type: int) -> None:
    if not isinstance(cache_type, int):
        raise TypeError("cache_type must be int")
    if cache_type < 0 or cache_type > 0xFF:
        raise ValueError("cache_type must fit uint8")


def _validate_u32(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit uint32")
