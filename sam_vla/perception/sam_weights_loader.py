"""Loads the segmentation model backing sam_segmenter.

Two backends, both SimpleSAM2Seg (sam2_custom_head.py):
  legacy -- the original single fully-merged checkpoint (best_model.pth),
            trained on plain camera frames (soil/bedrock/sand/bigrock).
  lora   -- a finetune_sam2_lora.py checkpoint dir (LoRA-adapted encoder +
            seg_head.pt + metadata.json), trained on frames with the
            mesh_tight_bound2 annotation hulls composited in (see
            sam_vla.env.habitat_env.MarsHabitatEnv.get_mesh_overlay_rgb) --
            this is the default, since it's the better-performing checkpoint
            (sam_lora_runs/exp10/best, val_mIoU ~0.80 vs. the legacy model).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from sam_vla.perception.sam2_custom_head import (
    CLASS_NAMES as LEGACY_CLASS_NAMES,
    NUM_CLASSES as LEGACY_NUM_CLASSES,
    SimpleSAM2Seg,
    build_sam2_backbone,
)

BACKEND_LEGACY = "legacy"
BACKEND_LORA = "lora"
# NOTE: this is the *library* default for callers that don't pick a backend
# explicitly (sam_segmenter.segment_frame's bare signature, and every
# first_frame_resolver caller that doesn't pass backend=...). It stays
# 'legacy' on purpose: BACKEND_LORA (sam_lora_runs/exp10) was trained on
# frames with the mesh_tight_bound2 annotation hulls composited in and
# predicts background-only on a plain camera frame (verified empirically --
# 0 detections on plain frames vs. real detections once the same frame has
# the mesh overlay applied). nav/gui.py's RoverController is the one caller
# that defaults to 'lora' -- it always pairs it with the mesh overlay (see
# MarsHabitatEnv.get_mesh_overlay_rgb / RoverController.seg_overlay). Any
# other caller wanting the lora backend must supply the same overlay itself,
# or it will silently see no detections.
DEFAULT_BACKEND = BACKEND_LEGACY

LEGACY_CHECKPOINT_PATH = (
    "sam/sam/sam_v2_dataset/training_output/checkpoints/best_model.pth"
)
LORA_CHECKPOINT_DIR = "sam_lora_runs/exp10/best"

DEFAULT_CHECKPOINT = {
    BACKEND_LEGACY: LEGACY_CHECKPOINT_PATH,
    BACKEND_LORA: LORA_CHECKPOINT_DIR,
}


def load_legacy_model(
    checkpoint_path: str = LEGACY_CHECKPOINT_PATH, device: str = "cuda"
) -> Tuple[nn.Module, List[str]]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Custom SimpleSAM2Seg weights required, none found at '{checkpoint_path}'. "
        )

    sam2_backbone = build_sam2_backbone(device)
    model = SimpleSAM2Seg(sam2_backbone, LEGACY_NUM_CLASSES).to(device)

    # best_model.pth is a training checkpoint dict, not a bare state dict.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, list(LEGACY_CLASS_NAMES)


def load_lora_model(
    checkpoint_dir: str = LORA_CHECKPOINT_DIR, device: str = "cuda"
) -> Tuple[nn.Module, List[str]]:
    from sam_vla.perception.finetune_sam2_lora import load_finetuned_model

    checkpoint_dir_path = Path(checkpoint_dir)
    metadata_path = checkpoint_dir_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"LoRA checkpoint dir required, no metadata.json found at '{checkpoint_dir_path}'. "
        )

    model = load_finetuned_model(checkpoint_dir_path, device=device)
    class_names = json.loads(metadata_path.read_text())["class_names"]
    return model, class_names


def load_sam_model(
    backend: str = DEFAULT_BACKEND,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[nn.Module, List[str]]:
    """Returns (model, class_names) -- class_names[i] is the label sam_segmenter
    should use for class index i's logits, in the order the model was trained with."""
    path = checkpoint_path or DEFAULT_CHECKPOINT.get(backend)
    if backend == BACKEND_LEGACY:
        return load_legacy_model(path, device)
    if backend == BACKEND_LORA:
        return load_lora_model(path, device)
    raise ValueError(f"unknown segmentation backend {backend!r} (expected 'legacy' or 'lora')")
