"""
Persistent TCP server that keeps Qwen3.5-9B resident in memory. Sibling of
qwen_server.py (Qwen2.5-VL-3B) -- same length-prefixed JSON wire protocol,
different model/port (qwen35_config.py), started/stopped independently (see
qwen35_server_manager.QwenServerManager).
"""

import base64
import io
import json
import os
import re
import socket
import struct

import numpy as np
from PIL import Image

from sam_vla.vlm.qwen35_config import QWEN35_SERVER_HOST, QWEN35_SERVER_PORT
from sam_vla.vlm.qwen35_model_runner import (
    load_qwen35_model,
    resolve_model_id,
    run_qwen35_inference_with_history,
)
from sam_vla.vlm.qwen35_prompts import (
    build_point_goal_history_prompt,
    build_reasoning_goal_prompt,
)
from sam_vla.vlm.qwen35_response_parser import parse_point_goal_history_response

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


_NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "half",
)
_ACTION_KEYWORDS = {
    "stop": ("stop", "wait", "hold", "stay", "halt"),
    "home": ("home", "base", "start", "back", "return"),
}


def _guard_motion_action(result: dict, instruction_text: str) -> dict:
    """A small VLM sometimes hallucinates forward/stop/home for an instruction
    that never asked for it (live: "go to red flag" -> forward 3 m, because
    the model invented a distance). Accept these actions only if the
    instruction's own wording supports them; otherwise downgrade to a
    found=False point so the controller creeps forward and re-queries."""
    action = result.get("action")
    text = instruction_text.lower()
    if action == "forward":
        supported = any(c.isdigit() for c in text) or any(
            w in text.split() or f"{w} " in text for w in _NUMBER_WORDS
        )
    elif action in _ACTION_KEYWORDS:
        supported = any(k in text for k in _ACTION_KEYWORDS[action])
    else:
        return result
    if supported:
        return result
    return {
        "action": "point",
        "found": False,
        "u": None,
        "v": None,
        "target_visible": False,
        "target_u": None,
        "target_v": None,
        "advance": False,
        "reasoning": f"[ignored unsupported '{action}'] {result.get('reasoning', '')}",
    }


def _handle_select_point_goal(model, processor, payload: dict) -> dict:
    image = _decode_image(payload["image_b64"])
    instruction_text = payload["instruction_text"]
    frame_idx = payload["frame_idx"]
    history = [
        (_decode_image(turn["image_b64"]), turn["prompt"], turn["response"])
        for turn in payload.get("history", [])
    ]
    if os.environ.get("QWEN35_PROMPT_STYLE") == "reasoning":
        prompt = build_reasoning_goal_prompt(instruction_text, frame_idx)
    else:
        prompt = build_point_goal_history_prompt(instruction_text, frame_idx)
    raw_text = run_qwen35_inference_with_history(
        model, processor, image, prompt, history
    )
    result = parse_point_goal_history_response(raw_text)
    result = _guard_motion_action(result, instruction_text)
    # prompt/raw_text round-tripped so the caller (VlmPointNavigator) can
    # append this turn onto its own history buffer without reconstructing
    # the prompt text client-side.
    return {"result": result, "prompt": prompt, "raw_text": raw_text}


_NOUN_GROUPS = (
    {"rover", "robot", "vehicle", "perseverance", "car", "buggy"},
    {"antenna", "dish", "tower", "station", "base", "satellite"},
    {"flag", "banner", "pennant"},
    {"rock", "stone", "boulder"},
)


def _head_noun(text: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in ("the", "a", "an", "of", "to", "go")]
    return words[-1] if words else ""


def _same_kind(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return True
    return any(a in g and b in g for g in _NOUN_GROUPS)


def _handle_ground(model, processor, payload: dict) -> dict:
    image = _decode_image(payload["image_b64"])
    phrase = payload["phrase"]
    prompt = (
        f"Locate the {phrase} in the image. Output the bounding box of the "
        'single best match in JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": '
        '"<what it is>"}]. If the %s is not visible, output [].' % phrase
    )
    raw = run_qwen35_inference_with_history(
        model, processor, image, prompt, [], max_new_tokens=160, temperature=0.1
    )
    boxes = []
    for m in re.finditer(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", raw):
        x1, y1, x2, y2 = (float(g) for g in m.groups())
        # Qwen3-VL grounding is on a 0-1000 grid
        x1, x2 = sorted((min(max(x1 / 1000.0, 0.0), 1.0), min(max(x2 / 1000.0, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1 / 1000.0, 0.0), 1.0), min(max(y2 / 1000.0, 0.0), 1.0)))
        lab = re.search(r'"label"\s*:\s*"([^"]*)"', raw[m.end():m.end() + 80])
        boxes.append({"box": [x1, y1, x2, y2], "label": lab.group(1) if lab else ""})
    return {"boxes": boxes, "raw_text": raw}


_COLOR_WORDS = ("red", "green", "blue", "white", "black", "grey", "gray", "yellow", "orange", "purple", "pink")


def _handle_verify_target(model, processor, payload: dict) -> dict:
    image = _decode_image(payload["image_b64"])
    instruction = payload["instruction_text"]
    prompt = (
        f'A rover was told: "{instruction}". This is a zoomed-in crop of its '
        "camera image around the spot where the destination is believed to "
        "be. Look carefully at the colour and shape of the object in the "
        "centre of the crop. Does it clearly show the specific destination "
        "named in the instruction (the right kind of object AND the right "
        'colour)? Reply with ONLY JSON: {"object": "<what is at the centre, '
        'e.g. blue flag, white flag, rock, nothing>", "match": true|false}'
    )
    raw = run_qwen35_inference_with_history(
        model, processor, image, prompt, [], max_new_tokens=80, temperature=0.1
    )
    try:
        from sam_vla.vlm.qwen_response_parser import _load_json_object

        parsed = _load_json_object(raw)
        match = parsed.get("match") is True
        obj = str(parsed.get("object", ""))
        # The yes/no is unreliable on its own (live: said match=true for a
        # "red flag" instruction while naming the object "blue flag"). If the
        # instruction names a colour, the colour the model says it SEES must
        # agree; "nothing"/"rock" never matches.
        wanted = [c for c in _COLOR_WORDS if c in instruction.lower()]
        if any(c in wanted for c in ("grey", "gray", "white", "black")):
            # the sim's white flag is a dark grey/black checker -- the model
            # calls it white, grey or black interchangeably
            wanted = ["white", "grey", "gray", "black"]
        seen = obj.lower()
        # the thing it sees must be the same KIND as the instruction's target
        # (live: "go to the rover" was 'verified' on a red flag)
        if not _same_kind(_head_noun(seen), _head_noun(instruction)):
            match = False
        if "nothing" in seen or not seen.strip():
            match = False
        elif wanted:
            match = match and any(c in seen for c in wanted)
    except Exception:
        match, obj = False, f"unparsed: {raw[:60]}"
    return {"match": match, "object": obj, "raw_text": raw}


def _dispatch(model, processor, message: dict, lora_adapter: str = "") -> dict:
    mode = message.get("mode")
    payload = message.get("payload", {})
    if mode == "ground":
        return _handle_ground(model, processor, payload)
    if mode == "verify_target":
        return _handle_verify_target(model, processor, payload)

    if mode == "ping":
        return {"status": "ok", "lora_adapter": lora_adapter}
    if mode == "select_point_goal":
        return _handle_select_point_goal(model, processor, payload)
    raise ValueError(f"unknown mode: {mode!r}")


def serve_forever(
    model,
    processor,
    host: str = QWEN35_SERVER_HOST,
    port: int = QWEN35_SERVER_PORT,
    lora_adapter: str = "",
) -> None:
    # Single-threaded blocking accept loop, same reasoning as qwen_server.py:
    # inference isn't safe to run concurrently across connections anyway.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    print(f"[!] qwen35_server listening on {host}:{port}", flush=True)

    while True:
        conn, _addr = listener.accept()
        with conn:
            try:
                message = _recv_message(conn)
                response = _dispatch(model, processor, message, lora_adapter)
            except Exception as e:
                response = {"error": str(e)}
            _send_message(conn, response)


if __name__ == "__main__":
    print(f"[!] Loading VLM model {resolve_model_id()}", flush=True)
    _model, _processor = load_qwen35_model()
    # QWEN35_LORA_ADAPTER: opt-in override for A/B-testing a fine-tuned
    # checkpoint (see sam_vla/vlm/finetune_qwen35_lora.py) against the
    # eval harness without a permanent CLI flag -- Qwen35ServerManager
    # spawns this as a plain subprocess.Popen with no env= override, so it
    # inherits whatever the launching shell/script set. Unset (the
    # default) loads the frozen base model exactly as before. Also
    # returned in every "ping" response (see _dispatch) -- confirmed live
    # this subprocess's stdout can sit fully buffered (non-TTY, long-
    # running) until killed, so a print()-only confirmation isn't reliable;
    # a caller that needs to know for certain should ping and check.
    _lora_adapter = os.environ.get("QWEN35_LORA_ADAPTER", "")
    if _lora_adapter:
        from peft import PeftModel

        print(f"[!] loading LoRA adapter from {_lora_adapter}", flush=True)
        _model = PeftModel.from_pretrained(_model, _lora_adapter)
        _model.eval()
    print("Model loaded, starting server.", flush=True)
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--port", type=int, default=QWEN35_SERVER_PORT)
    _args = _ap.parse_args()
    serve_forever(_model, _processor, port=_args.port, lora_adapter=_lora_adapter)
