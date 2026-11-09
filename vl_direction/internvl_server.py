"""
Persistent TCP server that would keep InternVL resident in memory, mirroring
sam_vla/vlm/qwen_server.py's shape exactly. Only the "ping" mode actually
works today -- "generate" calls run_internvl_inference, which raises
NotImplementedError until a real checkpoint is wired into
internvl_model_runner.py. This file is scaffolding: it establishes the wire
protocol InternVLSocketClient already speaks, so swapping in a real model
later is a one-file change plus running this server in its own conda env.
"""

import base64
import io
import json
import os
import socket
import struct

import numpy as np
from PIL import Image

from vl_direction.config import (
    INTERNVL_MODEL_PATH,
    INTERNVL_SERVER_HOST,
    INTERNVL_SERVER_PORT,
)
from vl_direction.internvl_model_runner import (
    load_internvl_model,
    run_internvl_inference,
)

# Ablation harness sets these to point the same server code at a different
# checkpoint/port without editing config.py -- unset, these fall back to the
# InternVL3-8B / default-port behavior.
_MODEL_PATH = os.environ.get("INTERNVL_MODEL_PATH", INTERNVL_MODEL_PATH)
_SERVER_PORT = int(os.environ.get("INTERNVL_SERVER_PORT", INTERNVL_SERVER_PORT))

_HEADER_SIZE = 4


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


def _recv_message(conn: socket.socket) -> dict:
    header = _recv_exact(conn, _HEADER_SIZE)
    (payload_len,) = struct.unpack(">I", header)
    payload = _recv_exact(conn, payload_len)
    return json.loads(payload.decode("utf-8"))


def _send_message(conn: socket.socket, message: dict) -> None:
    payload = json.dumps(message).encode("utf-8")
    header = struct.pack(">I", len(payload))
    conn.sendall(header + payload)


def _decode_image(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def _handle_generate(model, tokenizer, payload: dict) -> dict:
    images = [_decode_image(b64) for b64 in payload["images_b64"]]
    prompt = payload["prompt"]
    max_new_tokens = payload["max_new_tokens"]
    text = run_internvl_inference(model, tokenizer, images, prompt, max_new_tokens)
    return {"result": {"text": text}}


def _dispatch(model, tokenizer, message: dict) -> dict:
    mode = message.get("mode")
    payload = message.get("payload", {})

    if mode == "ping":
        return {"status": "ok"}
    if mode == "generate":
        return _handle_generate(model, tokenizer, payload)
    raise ValueError(f"unknown mode: {mode!r}")


def serve_forever(
    model, tokenizer, host: str = INTERNVL_SERVER_HOST, port: int = INTERNVL_SERVER_PORT
) -> None:
    # Single-threaded blocking accept loop, same rationale as qwen_server.py:
    # inference isn't safe to run concurrently across connections anyway.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    print(f"[!] internvl_server listening on {host}:{port}")

    while True:
        conn, _addr = listener.accept()
        with conn:
            try:
                message = _recv_message(conn)
                response = _dispatch(model, tokenizer, message)
            except Exception as e:
                response = {"error": str(e)}
            _send_message(conn, response)


if __name__ == "__main__":
    print(f"[!] Loading InternVL model: {_MODEL_PATH}")
    _model, _tokenizer = load_internvl_model(_MODEL_PATH)
    print("Model loaded, starting server.")
    serve_forever(_model, _tokenizer, port=_SERVER_PORT)
