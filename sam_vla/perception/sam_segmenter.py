"""Single-frame wrapper around the custom SimpleSAM2Seg segmentation head.

SimpleSAM2Seg (sam/train_sam2_simple_fast.py) currently only has a
process_video entry point. This module is the missing single-frame path:
one forward pass on one RGB frame, then per-pixel logits turned into
per-instance bounding boxes for the classes this pipeline surfaces.

Two checkpoints can back this (sam_weights_loader.BACKEND_LEGACY/_LORA,
selected by the `backend` kwarg on segment_frame, default `lora`) with
different class vocabularies (4-class soil/bedrock/sand/bigrock vs. 5-class
background/small_rock/big_rock/bedrock/hole_in_ground) -- _CLASS_ALIASES
below normalizes both onto the flat names sam_output_adapter._CLASS_MAP
keys on, so the downstream goal/obstacle pipeline doesn't need to know
which backend produced a detection.
"""

import pprint
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam_vla.perception import sam_weights_loader

IMAGE_SIZE = 1024
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Raw checkpoint class name -> normalized name. Only normalized names in
# SURFACED_CLASS_NAMES are turned into detections; everything else (soil,
# sand, background, small_rock, hole_in_ground) is dropped here, same as
# the original hardcoded {1: "bedrock", 3: "bigrock"} did for the legacy
# checkpoint -- bedrock is broad background segmentation, not a candidate
# rock (see sam_output_adapter's docstring for why it's still surfaced here
# but dropped one stage later).
_CLASS_ALIASES = {
    "bigrock": "bigrock",
    "big_rock": "bigrock",
    "bedrock": "bedrock",
}
SURFACED_CLASS_NAMES = {"bedrock", "bigrock"}

# Minimum connected-component area (px, at model resolution) to keep as a real instance instead of segmentation noise.
_MIN_INSTANCE_AREA = 4

_model_cache: dict = {}


def _get_model(backend: str, checkpoint_path):
    key = (backend, checkpoint_path)
    if key not in _model_cache:
        _model_cache[key] = sam_weights_loader.load_sam_model(
            backend=backend, checkpoint_path=checkpoint_path
        )
    return _model_cache[key]


def _preprocess(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    img = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def segment_frame(
    rgb: np.ndarray,
    model=None,
    class_names=None,
    backend: str = sam_weights_loader.DEFAULT_BACKEND,
    checkpoint_path=None,
) -> list[dict]:
    """`class_names` is required when passing an explicit `model` (it can't
    be recovered from a bare nn.Module); omit both to load/cache the model
    for (backend, checkpoint_path) -- default backend='lora' (sam_lora_runs/
    exp10/best, the mesh_tight_bound2-overlay-trained checkpoint)."""
    if model is None:
        model, class_names = _get_model(backend, checkpoint_path)
    elif class_names is None:
        raise ValueError("class_names is required when passing an explicit model")

    device = next(model.parameters()).device
    h0, w0 = rgb.shape[:2]
    scale_x = w0 / IMAGE_SIZE
    scale_y = h0 / IMAGE_SIZE

    with torch.no_grad():
        tensor = _preprocess(rgb, device)
        logits = model(tensor)  # (1, num_classes, IMAGE_SIZE, IMAGE_SIZE)
        probs = F.softmax(logits, dim=1)[0]  # (num_classes, IMAGE_SIZE, IMAGE_SIZE)
        class_map = probs.argmax(dim=0).cpu().numpy()
        probs_np = probs.cpu().numpy()

    detections = []
    for class_idx, raw_name in enumerate(class_names):
        class_name = _CLASS_ALIASES.get(raw_name)
        if class_name not in SURFACED_CLASS_NAMES:
            continue
        mask = (class_map == class_idx).astype(np.uint8)
        if not mask.any():
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for label_id in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[label_id]
            if area < _MIN_INSTANCE_AREA:
                continue

            score = float(probs_np[class_idx][labels == label_id].mean())
            detections.append(
                {
                    "class_name": class_name,
                    "x": float(x * scale_x),
                    "y": float(y * scale_y),
                    "width": float(w * scale_x),
                    "height": float(h * scale_y),
                    "score": score,
                }
            )

    return detections


if __name__ == "__main__":
    image_path = sys.argv[1]
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")
    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pprint.pprint(segment_frame(rgb_image))
