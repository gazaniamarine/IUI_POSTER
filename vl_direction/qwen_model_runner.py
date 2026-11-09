"""
Loads Qwen2.5-VL-3B-Instruct and runs single-call image+text generation,
intended to run inside qwen_server.py's process in the "qwen_vlm" conda env.
Mirrors internvl_model_runner.py's role for InternVL3-8B, but Qwen2.5-VL
needs no dynamic-tiling preprocessing of its own -- its processor handles
resizing internally, so this module is a thin wrapper around
transformers' chat-template + generate flow (same approach as
sam_vla/vlm/qwen_model_runner.py, extended here to accept a list of images
so multi-frame bursts -- e.g. exploration mode's FRAME_BURST_SIZE -- work,
which sam_vla's single-image version doesn't need).
"""

import sys

import numpy as np
from PIL import Image

from vl_direction.config import QWEN_MODEL_PATH


def load_qwen_model(model_path: str = QWEN_MODEL_PATH, device: str = "cuda") -> tuple:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def run_qwen_inference(
    model,
    processor,
    images: list,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    if not images:
        raise ValueError("run_qwen_inference requires at least one image")

    pil_images = [Image.fromarray(img) for img in images]

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": pil_image} for pil_image in pil_images]
            + [{"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=pil_images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
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


if __name__ == "__main__":
    image_path = sys.argv[1]
    prompt = (
        sys.argv[2] if len(sys.argv) > 2 else "Describe this image in one sentence."
    )

    test_image = np.array(Image.open(image_path).convert("RGB"))

    print(f"[!] Loading {QWEN_MODEL_PATH}")
    model, processor = load_qwen_model()
    print("Model loaded, running inference.")

    result = run_qwen_inference(model, processor, [test_image], prompt)
    print("response:", result)
