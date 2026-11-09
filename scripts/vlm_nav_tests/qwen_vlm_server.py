"""
Persistent Qwen2.5-VL-3B-Instruct inference server.

Runs in the `qwen_vlm` conda env. Loads the model once and then serves
repeated inference requests over a local TCP socket, so callers can poll
it at a few Hz without paying subprocess + model-load cost per call (as
the one-shot vlm_query.py path does). Mirrors the model-loading logic in
vlm_query.py / qwen_vlm_smoke_test.py.

Wire protocol: each message (both directions) is a 4-byte big-endian
length prefix followed by that many bytes of UTF-8 JSON.

Request:  {"cmd": "ping"}
       or {"cmd": "infer", "rgb": "<path>", "prompt": "<text>"}
Response: {"ok": true, "text": "...", "latency_s": ...}
       or {"ok": false, "error": "..."}

Usage:
    conda run -n qwen_vlm python scripts/vlm_nav_tests/qwen_vlm_server.py [--host 127.0.0.1] [--port <port>]
"""

import argparse
import json
import socket
import struct
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PROMPT = "Describe this image in one short sentence."


def load_model():
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(
        f"[qwen_vlm_server] {MODEL_ID} loaded in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return processor, model


def run_inference(processor, model, rgb_path, prompt, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    header = _recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = _recv_exact(conn, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def send_msg(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def serve(host, port):
    processor, model = load_model()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    print(f"[qwen_vlm_server] listening on {host}:{port}", file=sys.stderr)

    while True:
        conn, _ = sock.accept()
        with conn:
            try:
                req = recv_msg(conn)
                if req is None:
                    continue
                if req.get("cmd") == "ping":
                    send_msg(conn, {"ok": True, "text": "pong"})
                    continue
                t0 = time.time()
                text = run_inference(
                    processor,
                    model,
                    req["rgb"],
                    req.get("prompt", DEFAULT_PROMPT),
                    max_new_tokens=req.get("max_new_tokens", 256),
                )
                send_msg(
                    conn, {"ok": True, "text": text, "latency_s": time.time() - t0}
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    send_msg(conn, {"ok": False, "error": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.host, args.port)
