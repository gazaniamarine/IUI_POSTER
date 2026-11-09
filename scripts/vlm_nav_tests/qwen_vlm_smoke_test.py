"""One-shot smoke test for Qwen2.5-VL-3B-Instruct: loads the model, feeds one
image + a text prompt, and prints the generated text -- confirms env/model
load/inference work end-to-end. Not wired into the sim/capture/labelme
pipeline.

Usage:
    conda run -n qwen_vlm python scripts/vlm_nav_tests/qwen_vlm_smoke_test.py
"""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "vlm_nav_out/rgb_0000.png"
PROMPT = "Describe what you see in this image. Also count the number of stones."


def main():
    print(f"[info] Loading processor for {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[info] Loading model {MODEL_ID} (this may take a while on first run)")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": PROMPT},
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

    print("[info] Running inference")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n[result]")
    print(output_text[0])


if __name__ == "__main__":
    main()
