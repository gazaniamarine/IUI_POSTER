"""
Generic Qwen2.5-VL server for the ablation harness. Speaks the exact same
ping/generate wire protocol as vl_direction/internvl_server.py (4-byte
big-endian length prefix + JSON body, {"images_b64": [...], "prompt": str,
"max_new_tokens": int} -> {"result": {"text": str}}) so run_ablation.py can
point vl_direction.client.InternVLSocketClient at either backend
interchangeably -- the client doesn't know or care which model is behind it.

Qwen is the "robotics domain specific VL" arm of the ablation: this repo
already drives Qwen2.5-VL for the VLA rollout policy (sam_vla/vlm/), so it's
a real, already-integrated alternative to compare against InternVL, not a
speculative pick. Runs in the "qwen_vlm" conda env (transformers +
Qwen2.5-VL stack), spawned by QwenAblationServerManager.

Unlike sam_vla/vlm/qwen_model_runner.py (single image, task-specific JSON
protocols), this loader accepts a list of images per call so exploration
mode's multi-frame burst (see vl_direction/config.py FRAME_BURST_SIZE) gets
the same multi-image treatment InternVL's model.chat() gives it.
"""

import base64
import io
import json
import os
import socket
import struct

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

QWEN_ABLATION_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_HOST = "127.0.0.1"
_DEFAULT_PORT = 8768  # distinct from qwen_server's 8765 and internvl_server's 8766
_HEADER_SIZE = 4

_MODEL_PATH = os.environ.get("QWEN_ABLATION_MODEL_PATH", QWEN_ABLATION_MODEL_ID)
_PORT = int(os.environ.get("QWEN_ABLATION_SERVER_PORT", _DEFAULT_PORT))


def load_qwen_ablation_model(
    model_path: str = _MODEL_PATH, device: str = "cuda"
) -> tuple:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def run_qwen_ablation_inference(
    model, processor, images: list, prompt: str, max_new_tokens: int = 32
) -> str:
    if not images:
        raise ValueError("run_qwen_ablation_inference requires at least one image")

    pil_images = [Image.fromarray(img) for img in images]
    content = [{"type": "image", "image": img} for img in pil_images] + [
        {"type": "text", "text": prompt}
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=pil_images, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


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


def _handle_generate(model, processor, payload: dict) -> dict:
    images = [_decode_image(b64) for b64 in payload["images_b64"]]
    prompt = payload["prompt"]
    max_new_tokens = payload["max_new_tokens"]
    text = run_qwen_ablation_inference(model, processor, images, prompt, max_new_tokens)
    return {"result": {"text": text}}


def _dispatch(model, processor, message: dict) -> dict:
    mode = message.get("mode")
    payload = message.get("payload", {})

    if mode == "ping":
        return {"status": "ok"}
    if mode == "generate":
        return _handle_generate(model, processor, payload)
    raise ValueError(f"unknown mode: {mode!r}")


def serve_forever(model, processor, host: str = _HOST, port: int = _PORT) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    print(f"[!] qwen_ablation_server listening on {host}:{port}")

    while True:
        conn, _addr = listener.accept()
        with conn:
            try:
                message = _recv_message(conn)
                response = _dispatch(model, processor, message)
            except Exception as e:
                response = {"error": str(e)}
            _send_message(conn, response)


if __name__ == "__main__":
    print(f"[!] Loading Qwen ablation model: {_MODEL_PATH}")
    _model, _processor = load_qwen_ablation_model()
    print("Model loaded, starting server.")
    serve_forever(_model, _processor)
