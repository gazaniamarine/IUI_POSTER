"""
Loads InternVL3-8B and runs single-call image+text generation, intended to
run inside internvl_server.py's process in the dedicated "vl" conda env
(mirrors sam_vla/vlm/qwen_model_runner.py's role for Qwen2.5-VL).

InternVL3-8B was picked as the balanced choice for this task (next.md's
LEFT/RIGHT/FRONT/BACK direction calls and short uncertainty-sweep
descriptions): strong spatial/grounding performance for its size, fits
comfortably in a single GPU's memory (~16GB bf16 weights) with room to
spare for KV cache and multi-frame bursts, and per-call latency is
dominated by ~16-32 new tokens of greedy decoding rather than model size.

Preprocessing follows InternVL's own published "dynamic tiling" convention
(build_transform / dynamic_preprocess / load_image_array below) exactly, so
this stays compatible with however InternVL's own model.chat() expects
pixel_values to be shaped -- this is not a simplification, it's required by
the checkpoint's own vision encoder.

flash-attn is intentionally NOT used (use_flash_attn=False): this machine
has no nvcc/CUDA toolkit for building it from source, and eager/sdpa
attention is fast enough for the short generations this module needs.
"""

import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from vl_direction.config import INTERNVL_MODEL_PATH as INTERNVL_MODEL_ID

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_image_array(
    rgb: np.ndarray, input_size: int = 448, max_num: int = 12
) -> torch.Tensor:
    image = Image.fromarray(rgb).convert("RGB")
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    return torch.stack([transform(tile) for tile in tiles])


def load_internvl_model(
    model_path: str = INTERNVL_MODEL_ID, device: str = "cuda"
) -> tuple:
    from transformers import AutoModel, AutoTokenizer

    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        # transformers' own _from_pretrained silently returns False here on
        # certain sentencepiece RuntimeErrors (tokenization_utils_base.py's
        # "loading from TikToken will be attempted instead" branch, which
        # doesn't actually happen for every repo) instead of raising -- seen
        # with InternVL2.5-8B's InternLM2 tokenizer.model under sentencepiece
        # >=0.1.99 ("piece must not include null character"). Left uncaught,
        # this surfaces three calls later as a baffling
        # "'bool' object has no attribute 'convert_tokens_to_ids'" inside
        # model.chat(). If this fires, try downgrading sentencepiece
        # (0.1.99 is confirmed to work for InternVL2.5-8B in the "vl" env).
        raise RuntimeError(
            f"AutoTokenizer.from_pretrained({model_path!r}) returned {tokenizer!r} instead of a "
            "tokenizer -- likely a sentencepiece incompatibility with this checkpoint's tokenizer.model"
        )
    return model, tokenizer


def run_internvl_inference(
    model,
    tokenizer,
    images: list,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    if not images:
        raise ValueError("run_internvl_inference requires at least one image")

    pixel_values_per_image = [_load_image_array(img) for img in images]
    num_patches_list = [pv.size(0) for pv in pixel_values_per_image]
    pixel_values = (
        torch.cat(pixel_values_per_image, dim=0).to(torch.bfloat16).to(model.device)
    )

    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}

    if len(images) == 1:
        question = f"<image>\n{prompt}"
        return model.chat(tokenizer, pixel_values, question, generation_config)

    image_tags = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(images)))
    question = f"{image_tags}{prompt}"
    return model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
    )


if __name__ == "__main__":
    image_path = sys.argv[1]
    prompt = (
        sys.argv[2] if len(sys.argv) > 2 else "Describe this image in one sentence."
    )

    test_image = np.array(Image.open(image_path).convert("RGB"))

    print(f"[!] Loading {INTERNVL_MODEL_ID}")
    model, tokenizer = load_internvl_model()
    print("Model loaded, running inference.")

    result = run_internvl_inference(model, tokenizer, [test_image], prompt)
    print("response:", result)
