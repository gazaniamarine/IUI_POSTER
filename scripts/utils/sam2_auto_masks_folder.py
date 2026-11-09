"""Runs SAM2's automatic (unprompted) mask generator over a folder of
kb_teleop.py-style rgb_*.png frames, writing per-frame mask/overlay images.
Standalone: uses the sibling `sam2` package install (SAM2_ROOT), not this
repo's `sam_vla`/`sam` LoRA pipeline.

Usage:
    python scripts/utils/sam2_auto_masks_folder.py --input mars_teleop_out --output sam2_out [--max-frames 20]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

DEFAULT_INPUT_DIR = "mars_teleop_out"
DEFAULT_OUTPUT_DIR = "sam2_out"

# SAM2_ROOT overrides where the separate sam2 package + checkpoints live,
# since that install is shared across projects and isn't part of this repo.
DEFAULT_SAM2_ROOT = os.environ.get(
    "SAM2_ROOT", "/home/nahar/Desktop/pineapple/packages/sam2"
)

DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"

POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.92

MIN_MASK_REGION_AREA = 0

OVERLAY_ALPHA = 0.45


def natural_sort_key(path: Path):
    import re

    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def make_overlay(image: np.ndarray, masks: list, alpha: float = 0.45):
    overlay = image.astype(np.float32).copy()
    h, w = image.shape[:2]

    id_map = np.zeros((h, w), dtype=np.uint16)

    rng = np.random.default_rng(12345)
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

    for idx, mask_data in enumerate(masks_sorted, start=1):
        mask = mask_data["segmentation"].astype(bool)

        color = rng.integers(30, 255, size=(3,), dtype=np.uint8).astype(np.float32)

        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
        id_map[mask] = idx

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay, id_map, masks_sorted


def serialize_mask_metadata(masks_sorted: list):
    meta = []

    for idx, m in enumerate(masks_sorted, start=1):
        item = {}

        for k, v in m.items():
            if k == "segmentation":
                continue

            if isinstance(v, np.ndarray):
                item[k] = v.tolist()
            elif isinstance(v, (np.integer,)):
                item[k] = int(v)
            elif isinstance(v, (np.floating,)):
                item[k] = float(v)
            else:
                item[k] = v

        item["mask_id"] = idx
        meta.append(item)

    return meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_DIR,
        help="Folder containing rgb_0000.png, rgb_0001.png, etc, etc, blah, blah",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for segmented masks/overlays.",
    )

    parser.add_argument(
        "--sam2-root",
        default=os.path.expanduser(DEFAULT_SAM2_ROOT),
        help="Root of the sam2 package + checkpoints install. Overridable via the SAM2_ROOT env var.",
    )

    parser.add_argument(
        "--model-cfg",
        default=DEFAULT_MODEL_CFG,
        help="SAM2 config path relative to sam2-root.",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="SAM2 checkpoint path relative to sam2-root.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames for testing.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    sam2_root = Path(args.sam2_root).resolve()

    checkpoint = str((sam2_root / args.checkpoint).resolve())

    # Resolve the model config directly against sam2_root's own config folder,
    # instead of going through sam2's hydra-module search path (pkg://sam2),
    # which chokes on absolute/foreign paths (MissingConfigException).
    config_dir = (sam2_root / "sam2" / "configs").resolve()
    config_name = args.model_cfg.replace("\\", "/")
    if "configs/" in config_name:
        # Strip everything up through the last "configs/" segment, so this
        # works whether --model-cfg was given relative (configs/sam2.1/x.yaml)
        # or as a full absolute path (.../sam2/configs/sam2.1/x.yaml).
        config_name = config_name.rsplit("configs/", 1)[1]
    if config_name.endswith(".yaml"):
        config_name = config_name[: -len(".yaml")]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if not config_dir.exists():
        raise FileNotFoundError(f"SAM2 config folder not found: {config_dir}")

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Check your SAM2 checkpoint name with: ls {sam2_root}/checkpoints"
        )

    overlay_dir = output_dir / "overlays"
    mask_dir = output_dir / "mask_ids"
    meta_dir = output_dir / "metadata"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(input_dir.glob("frame_*.png"), key=natural_sort_key)

    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    if not frames:
        raise RuntimeError(f"No rgb_*.png frames found in {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Frames:", len(frames))
    print("Input:", input_dir)
    print("Output:", output_dir)
    print("SAM2 root:", sam2_root)
    print("Config dir:", config_dir)
    print("Config name:", config_name)
    print("Checkpoint:", checkpoint)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Failed to load checkpoint cleanly.\n"
            f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
        )

    model = model.to(device)
    model.eval()

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )

    for frame_path in tqdm(frames):
        image = load_rgb(frame_path)

        masks = mask_generator.generate(image)

        overlay, id_map, masks_sorted = make_overlay(
            image,
            masks,
            alpha=OVERLAY_ALPHA,
        )

        stem = frame_path.stem

        Image.fromarray(overlay).save(overlay_dir / f"{stem}_overlay.png")
        Image.fromarray(id_map).save(mask_dir / f"{stem}_mask_ids.png")

        metadata = {
            "frame": frame_path.name,
            "num_masks": len(masks_sorted),
            "masks": serialize_mask_metadata(masks_sorted),
        }

        with open(meta_dir / f"{stem}.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print("Overlays:", overlay_dir)
    print("Mask IDs:", mask_dir)
    print("Metadata:", meta_dir)


if __name__ == "__main__":
    main()

"""
ausage: python sam2_auto_masks_folder.py --input mars_teleop_out1783002646 --output sam2_test_out --sam2-root ~/Desktop/pineapple/sam2 --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml --checkpoint checkpoints/sam2.1_hiera_tiny.pt --max-frames 5
"""
