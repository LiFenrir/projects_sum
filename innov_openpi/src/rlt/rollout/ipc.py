"""length-prefix TCP 帧 + msgpack(numpy) 编解码。

RoboDojo sim server 与 rlt client 共用;仅依赖 numpy/msgpack(两边环境均有)。
帧格式: 8 字节大端无符号长度前缀 + msgpack payload。
"""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack
import numpy as np

_HEADER = struct.Struct(">Q")
_MAX_MSG = 1 << 30  # 1GB 帧上限,防脏数据


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": obj.dtype.str,
            "shape": obj.shape,
            "data": obj.tobytes(),
        }
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def _object_hook(obj: dict) -> Any:
    if "__ndarray__" in obj:
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def dumps(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def loads(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_object_hook)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload = dumps(obj)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Any:
    """接收一条消息;对端关闭抛 ConnectionError。"""
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if length > _MAX_MSG:
        raise ConnectionError(f"frame too large: {length}")
    return loads(_recv_exact(sock, length))
