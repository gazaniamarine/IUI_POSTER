"""
Thin, swappable wrapper around InternVL inference (next.md sec 5.1). Callers
only ever see the InternVLClient Protocol; how a concrete client gets its
answer (canned text for tests, or a live model over a socket) is an
implementation detail selected by get_client(). This module deliberately
excludes prompt construction, output parsing, and mode/context knowledge --
those live in prompts/, parser.py, and directive_engine.py respectively
(mirrors the documentation style of sam_vla/policy/base_policy.py).

InternVL is not installed anywhere in this repo yet (verified: no conda env,
no requirements entry). InternVLSocketClient talks the exact same wire
protocol as sam_vla/vlm/qwen_client.py (4-byte big-endian length prefix +
JSON body) to a server this package also scaffolds (internvl_server.py) --
that server can't actually answer requests until internvl_model_runner.py's
NotImplementedError stubs are filled in with a real checkpoint later.
"""

import base64
import io
import itertools
import json
import socket
import struct
import typing

import numpy as np

from vl_direction import config

_HEADER_SIZE = 4


class InternVLClient(typing.Protocol):
    def generate(self, frames: list, prompt: str, max_new_tokens: int) -> str: ...


class MockInternVLClient:
    """Returns canned text with no I/O -- the default backend, and the only
    one exercised by this package's tests (next.md sec 9: tests never
    require a live InternVL checkpoint)."""

    def __init__(
        self,
        canned_response: str = "LEFT",
        canned_responses: typing.Optional[list] = None,
    ):
        self._responses = (
            canned_responses if canned_responses is not None else [canned_response]
        )
        self._cycle = itertools.cycle(self._responses)
        self.calls: list = []

    def generate(self, frames: list, prompt: str, max_new_tokens: int) -> str:
        self.calls.append(
            {
                "frame_count": len(frames),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
        )
        return next(self._cycle)


def _encode_image(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError(
                "connection closed before expected bytes were received"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_request(payload: dict, host: str, port: int, timeout: float) -> dict:
    message = json.dumps(payload).encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.sendall(struct.pack(">I", len(message)) + message)
            header = _recv_exact(conn, _HEADER_SIZE)
            (body_len,) = struct.unpack(">I", header)
            body = _recv_exact(conn, body_len)
    except OSError as e:
        raise ConnectionError(
            f"could not reach internvl_server at {host}:{port}: {e}. Is the server running?"
        ) from e
    return json.loads(body.decode("utf-8"))


class InternVLSocketClient:
    def __init__(
        self,
        host: typing.Optional[str] = None,
        port: typing.Optional[int] = None,
        timeout: float = 30.0,
    ):
        self.host = host or config.INTERNVL_SERVER_HOST
        self.port = port or config.INTERNVL_SERVER_PORT
        self.timeout = timeout

    def generate(self, frames: list, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "mode": "generate",
            "payload": {
                "images_b64": [_encode_image(f) for f in frames],
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
        }
        response = _send_request(payload, self.host, self.port, self.timeout)
        if "error" in response:
            raise ValueError(f"internvl_server generate failed: {response['error']}")
        return response["result"]["text"]


class QwenSocketClient:
    """Same generate() shape as InternVLSocketClient, talking to
    vl_direction/qwen_server.py (port distinct from InternVL's and from
    sam_vla/vlm/qwen_server's) instead of internvl_server.py."""

    def __init__(
        self,
        host: typing.Optional[str] = None,
        port: typing.Optional[int] = None,
        timeout: float = 30.0,
    ):
        self.host = host or config.QWEN_SERVER_HOST
        self.port = port or config.QWEN_SERVER_PORT
        self.timeout = timeout

    def generate(self, frames: list, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "mode": "generate",
            "payload": {
                "images_b64": [_encode_image(f) for f in frames],
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
        }
        response = _send_request(payload, self.host, self.port, self.timeout)
        if "error" in response:
            raise ValueError(f"qwen_server generate failed: {response['error']}")
        return response["result"]["text"]


def get_client(backend: typing.Optional[str] = None) -> InternVLClient:
    backend = backend if backend is not None else config.INTERNVL_BACKEND
    if backend == "mock":
        return MockInternVLClient()
    if backend in ("hf", "vllm", "api"):
        return InternVLSocketClient()
    if backend == "qwen":
        return QwenSocketClient()
    raise ValueError(f"unknown INTERNVL_BACKEND {backend!r}")


if __name__ == "__main__":
    client = get_client("mock")
    print(
        client.generate(
            [np.zeros((4, 4, 3), dtype=np.uint8)], "test prompt", max_new_tokens=8
        )
    )
