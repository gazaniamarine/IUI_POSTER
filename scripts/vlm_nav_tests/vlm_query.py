"""
Standalone Qwen2.5-VL query runner.

Runs in the `qwen_vlm` conda env, invoked as a subprocess from
vlm_nav_interactive.py (which runs in the `habitat` env and doesn't have
torch/transformers installed). Mirrors the model-loading logic in
qwen_vlm_smoke_test.py.

Feeds the raw RGB frame, its annotated overlay image, and a text summary of
the labelme annotation JSON (object ids, labels, boxes) to the model and
writes the generated response to --out.

Usage:
    conda run -n qwen_vlm python scripts/vlm_nav_tests/vlm_query.py \\
        --rgb rgb_0000.png --annotation rgb_0000.json --prompt "..." --out response.txt
"""

import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_prompt(prompt, annotation_path):
    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    objects_summary = json.dumps(
        [
            {"id": s["id"], "label": s["label"], "points": s["points"]}
            for s in annotation_data["shapes"]
        ],
        indent=2,
    )

    return (
        f"{prompt}\n\n"
        f"The first image is the raw camera frame. "
        f"The second image is the same frame with detected object bounding boxes overlaid, each tagged with its object id."
        f"Detected objects (id, label, bounding "
        f"box points in pixels):\n{objects_summary}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, help="path to raw RGB frame")
    parser.add_argument(
        "--overlay", required=False, help="path to annotated overlay image"
    )
    parser.add_argument(
        "--annotation", required=True, help="path to labelme annotation JSON"
    )
    parser.add_argument("--prompt", required=True, help="question to ask the VLM")
    parser.add_argument(
        "--out", required=True, help="path to write the generated response"
    )
    args = parser.parse_args()

    text_prompt = build_prompt(args.prompt, args.annotation)

    prompt_out = args.out.rsplit(".", 1)[0] + "_prompt.txt"
    with open(prompt_out, "w") as f:
        f.write(text_prompt)

    print(f"[vlm] loading {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.rgb},
                # {"type": "image", "image": args.overlay},
                {"type": "text", "text": text_prompt},
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

    print("[vlm] running inference")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    with open(args.out, "w") as f:
        f.write(output_text)

    print("\n[vlm result]")
    print(output_text)


if __name__ == "__main__":
    main()
