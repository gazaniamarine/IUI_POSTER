"""
Client for the persistent Qwen3.5-9B server (qwen35_server.py). Sibling of
qwen_client.py (Qwen2.5-VL-3B) -- same socket/wire-protocol plumbing,
different port/payload shape (a single select_point_goal mode carrying
instruction + rolling history instead of qwen_client.py's per-task modes).
"""

import base64
import io
import json
import socket
import struct

import numpy as np

from sam_vla.vlm.qwen35_config import QWEN35_SERVER_PORT

_HOST = "127.0.0.1"
_HEADER_SIZE = 4


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


def _send_request(mode: str, payload: dict, port: int = None, timeout: float = 60.0) -> dict:
    if port is None:
        port = QWEN35_SERVER_PORT
    message = json.dumps({"mode": mode, "payload": payload}).encode("utf-8")
    try:
        with socket.create_connection((_HOST, port), timeout=timeout) as conn:
            conn.sendall(struct.pack(">I", len(message)) + message)
            header = _recv_exact(conn, _HEADER_SIZE)
            (body_len,) = struct.unpack(">I", header)
            body = _recv_exact(conn, body_len)
    except OSError as e:
        raise ConnectionError(
            f"could not reach qwen35_server at {_HOST}:{port} (mode={mode!r}): {e}. "
            "Is the server running?"
        ) from e
    return json.loads(body.decode("utf-8"))


def select_point_goal(
    rgb: np.ndarray,
    instruction_text: str,
    frame_idx: int,
    history: list[tuple[np.ndarray, str, str]],
    port: int = None,
) -> tuple[dict, str, str]:
    """history is a list of (past_image, past_prompt, past_raw_response_text)
    triples, oldest first -- see nav.vlm_point_navigator.VlmPointNavigator,
    which owns and caps this buffer. Returns (parsed_result, prompt,
    raw_text): the navigator appends (image, prompt, raw_text) onto its own
    history using the last two, so the exact prompt text sent this call
    never needs reconstructing client-side."""
    history_json = [
        {
            "image_b64": _encode_image(img),
            "prompt": prompt,
            "response": response,
        }
        for img, prompt, response in history
    ]
    response = _send_request(
        "select_point_goal",
        {
            "image_b64": _encode_image(rgb),
            "instruction_text": instruction_text,
            "frame_idx": frame_idx,
            "history": history_json,
        },
        port=port,
    )
    if "error" in response:
        raise ValueError(f"select_point_goal failed: {response['error']}")
    return response["result"], response["prompt"], response["raw_text"]


def verify_target(
    crop_rgb: np.ndarray, instruction_text: str, port: int = None
) -> tuple[bool, str]:
    """Ask the VLM whether a zoomed crop shows the destination named in the
    instruction (type AND colour). Returns (match, what_it_saw). Used to
    reject false 'target visible' claims on tiny/edge objects."""
    response = _send_request(
        "verify_target",
        {"image_b64": _encode_image(crop_rgb), "instruction_text": instruction_text},
        port=port,
    )
    if "error" in response:
        raise ValueError(f"verify_target failed: {response['error']}")
    return bool(response["match"]), str(response.get("object", ""))


def ground(rgb: np.ndarray, phrase: str, port: int = None) -> list:
    """Open-vocabulary grounding: Qwen3-VL returns bounding boxes for the
    named thing. -> [(x1, y1, x2, y2, label)] normalized to [0, 1]."""
    response = _send_request(
        "ground", {"image_b64": _encode_image(rgb), "phrase": phrase}, port=port
    )
    if "error" in response:
        raise ValueError(f"ground failed: {response['error']}")
    return [(*b["box"], b.get("label", "")) for b in response["boxes"]]


if __name__ == "__main__":
    import sys

    from PIL import Image

    image_path = sys.argv[1] if len(sys.argv) > 1 else "marsyard2022_terrain_texture.png"
    rgb = np.array(Image.open(image_path).convert("RGB"))

    result, prompt, raw_text = select_point_goal(
        rgb, "Drive to the large rock cluster ahead.", frame_idx=0, history=[]
    )
    print("select_point_goal ->", result)
