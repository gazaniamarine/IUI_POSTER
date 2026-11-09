"""Runs a finetune_sam2_lora.py checkpoint (e.g. sam_lora_runs/exp1/best) on
one or more RGB images and writes bbox detections: an overlay PNG (bbox +
class label drawn on the original-resolution frame) and a matching
annotations .txt (one detection per line: class x_min y_min x_max y_max
confidence, in original-image pixel coords).

The model (SimpleSAM2Seg) is a dense per-pixel classifier, not a detector --
there's no box head. Boxes here come from a post-hoc step: predict the dense
class mask, then cv2.connectedComponentsWithStats per foreground class to
turn each connected blob of that class into one bbox. Fine for compact
objects like rocks; would over/under-count for masks with touching or
donut-shaped instances.

Usage:
    python -m sam_vla.perception.predict_lora \\
        --checkpoint-dir sam_lora_runs/exp1/best \\
        --image path/to/frame.png --out-dir sam_lora_runs/exp1/test_preds \\
        [--classes big_rock] [--min-area 64] [--device cuda]

    # or a whole folder of images:
    python -m sam_vla.perception.predict_lora \\
        --checkpoint-dir sam_lora_runs/exp1/best \\
        --images-dir some/folder --out-dir sam_lora_runs/exp1/test_preds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam_vla.perception.finetune_sam2_lora import _normalize_image, load_finetuned_model
from sam_vla.perception.spot_check_segmentation import build_palette

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def predict_mask(model, rgb: np.ndarray, image_size: int, device: str):
    """rgb: HxWx3 uint8, original resolution. Returns (pred, conf), both
    HxW at the *original* resolution -- pred is the argmax class index per
    pixel, conf is that class's softmax probability per pixel."""
    resized = cv2.resize(rgb, (image_size, image_size))
    x = _normalize_image(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        logits = F.interpolate(
            logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)[0]

    conf, pred = probs.max(dim=0)
    return pred.cpu().numpy(), conf.cpu().numpy()


def boxes_from_mask(
    pred: np.ndarray,
    conf: np.ndarray,
    class_names: Sequence[str],
    classes: Optional[Sequence[str]],
    min_area: int,
) -> List[dict]:
    """One bbox per connected blob of each requested class (default: every
    class but background). Confidence is that blob's mean per-pixel
    softmax probability."""
    wanted = classes if classes else class_names[1:]
    detections: List[dict] = []
    for name in wanted:
        class_idx = class_names.index(name)
        binary = (pred == class_idx).astype(np.uint8)
        if not binary.any():
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label in range(1, n_labels):
            x, y, w, h, area = stats[label]
            if area < min_area:
                continue
            blob_conf = float(conf[labels == label].mean())
            detections.append(
                {
                    "class": name,
                    "x_min": int(x),
                    "y_min": int(y),
                    "x_max": int(x + w),
                    "y_max": int(y + h),
                    "confidence": blob_conf,
                }
            )
    return detections


def draw_detections(
    rgb: np.ndarray, detections: List[dict], palette: dict
) -> np.ndarray:
    overlay = rgb.copy()
    for det in detections:
        color = palette.get(det["class"], (255, 255, 255))
        cv2.rectangle(
            overlay,
            (det["x_min"], det["y_min"]),
            (det["x_max"], det["y_max"]),
            color,
            2,
        )
        label = f"{det['class']} {det['confidence']:.2f}"
        text_y = det["y_min"] - 6 if det["y_min"] - 6 >= 10 else det["y_min"] + 14
        cv2.putText(
            overlay,
            label,
            (det["x_min"], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def write_annotations(path: Path, detections: List[dict]) -> None:
    lines = [
        f"{d['class']} {d['x_min']} {d['y_min']} {d['x_max']} {d['y_max']} {d['confidence']:.4f}"
        for d in detections
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def run_one(model, image_path: Path, out_dir: Path, class_names, args) -> List[dict]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"could not read image at '{image_path}'")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pred, conf = predict_mask(model, rgb, args.image_size, args.device)
    detections = boxes_from_mask(pred, conf, class_names, args.classes, args.min_area)

    palette = build_palette(class_names)
    overlay = draw_detections(rgb, detections, palette)

    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    cv2.imwrite(
        str(out_dir / "overlays" / f"{stem}.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    write_annotations(out_dir / "annotations" / f"{stem}.txt", detections)

    return detections


def main(args: argparse.Namespace) -> None:
    model = load_finetuned_model(Path(args.checkpoint_dir), device=args.device)
    class_names = json.loads((Path(args.checkpoint_dir) / "metadata.json").read_text())[
        "class_names"
    ]

    if args.classes:
        unknown = set(args.classes) - set(class_names)
        if unknown:
            raise SystemExit(
                f"unknown class(es) {unknown}, checkpoint has {class_names}"
            )

    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(
            p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            raise SystemExit(f"no images found in {args.images_dir}")

    out_dir = Path(args.out_dir)
    total_by_class: dict = {}
    for image_path in images:
        detections = run_one(model, image_path, out_dir, class_names, args)
        for d in detections:
            total_by_class[d["class"]] = total_by_class.get(d["class"], 0) + 1
        print(f"[{image_path.name}] {len(detections)} detection(s)")

    print(f"done -- {len(images)} image(s) -> {out_dir}")
    print(f"totals by class: {total_by_class}")


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint-dir", required=True, help="e.g. sam_lora_runs/exp1/best"
    )

    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default=None, help="single image to run on")
    source.add_argument("--images-dir", default=None, help="folder of images to run on")

    ap.add_argument(
        "--out-dir", required=True, help="where to write overlays/ and annotations/"
    )
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="class names to detect (default: all non-background classes in the checkpoint)",
    )
    ap.add_argument(
        "--min-area",
        type=int,
        default=64,
        help="drop connected blobs smaller than this many pixels (noise)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(_build_argparser().parse_args())
