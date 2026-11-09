"""Loads Qwen2.5-VL and runs single-call image+text generation.

Uses transformers directly (not Ollama) so prompts/outputs stay structured
JSON-friendly for the caller. This module only does text-in/text-out; JSON
parsing of the response is the caller's responsibility.
"""

import sys

import numpy as np
from PIL import Image
import os

from transformers import AutoModelForImageTextToText, AutoProcessor

# Same model as the point-goal navigator (qwen35 server): Qwen3-VL-8B-Instruct.
# Was Qwen/Qwen2.5-VL-3B-Instruct. Override with QWEN_MODEL_ID=... if needed.
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")


def load_qwen_model(device: str = "cuda") -> tuple:
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype="auto",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor


def run_qwen_inference(
    model,
    processor,
    image,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    # `image` is either a single frame (all of this module's own callers) or
    # a list of frames (vl_direction's generate mode, e.g. exploration's
    # multi-frame bursts -- see vl_direction/qwen_model_runner.py, which this
    # mirrors for that case).
    images = image if isinstance(image, list) else [image]
    pil_images = [Image.fromarray(img) for img in images]

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_images]
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
    test_image = np.array(Image.open(image_path).convert("RGB"))

    model, processor = load_qwen_model()
    result = run_qwen_inference(
        model, processor, test_image, "Describe this image in one sentence."
    )
    print(result)
