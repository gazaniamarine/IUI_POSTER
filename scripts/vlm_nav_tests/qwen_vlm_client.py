"""Stdlib-only TCP client for the persistent Qwen VLM server
(qwen_vlm_server.py) -- importable from torch-less envs (e.g. `habitat`) to
send low-latency inference requests to the resident model process. A
separate calling path from vlm_nav_interactive.py's one-shot subprocess flow.
Not run directly; imported by rollout_navdp*.py / qwen_search_*.py.
"""

import json
import socket
import struct

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _request(obj, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=10.0):
    payload = json.dumps(obj).encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall(struct.pack(">I", len(payload)) + payload)
        header = _recv_exact(conn, 4)
        (length,) = struct.unpack(">I", header)
        body = _recv_exact(conn, length)
        return json.loads(body.decode("utf-8"))


def ping(host=DEFAULT_HOST, port=DEFAULT_PORT):
    return _request({"cmd": "ping"}, host, port)


def query_vlm_persistent(
    rgb_path,
    prompt=None,
    max_new_tokens=None,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    timeout=10.0,
):
    req = {"cmd": "infer", "rgb": rgb_path}
    if prompt is not None:
        req["prompt"] = prompt
    if max_new_tokens is not None:
        req["max_new_tokens"] = max_new_tokens
    resp = _request(req, host, port, timeout=timeout)
    if not resp.get("ok"):
        raise RuntimeError(f"qwen_vlm_server error: {resp.get('error')}")
    return resp["text"]
