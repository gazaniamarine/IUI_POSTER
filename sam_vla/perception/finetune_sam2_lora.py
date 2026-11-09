"""LoRA finetuning for the SAM2 segmentation model used by this pipeline.

This model has no mask decoder to LoRA-adapt: sam2_custom_head.SimpleSAM2Seg
uses SAM2.1 Hiera-L's image encoder only as a feature backbone, feeding a
from-scratch conv decoder (always trained in full) -- so LoRA here targets
the Hiera trunk's attention projections (`attn.qkv`/`attn.proj`) instead.
Trains directly on run_segmentation_sweep's masks_category/<frame_id>.png
dense per-class PNGs (SegmentationRunDataset), not export_annotations' COCO/
YOLO output -- CocoPolygonDataset (rasterizing COCO polygons back to a dense
mask) is only for the case where no run_dir is available, only a portable
export. Needs torch/peft/cv2, no habitat_sim; runs in the `sam2` conda env.

Usage:
    python -m sam_vla.perception.finetune_sam2_lora \\
        --run-dir output/<run_id> --out-dir sam_lora_runs/exp1 \\
        [--coco-json <run>/annotations_export/coco/instances.json --images-root <run>]  # instead of --run-dir
        [--encoder-mode lora|frozen|full] [--lora-rank 8] [--lora-alpha 16] \\
        [--epochs 20] [--batch-size 8] [--lr 1e-4] [--val-frac 0.1] \\
        [--dry-run] [--resume <out_dir>/training_state.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sam_vla.perception.sam2_custom_head import (
    IMAGE_SIZE,
    SAM2_CONFIG_DIR,
    SAM2_MODEL_CONFIG,
    SAM2_ROOT,
    SimpleSAM2Seg,
    build_sam2_backbone,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_LORA_TARGET_MODULES = ("qkv", "proj")

# Matches Meta's own SAM2.1 MOSE finetune config
# (sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml:
# loss_mask=20, loss_dice=1) -- there's no loss_iou here since this head
# has no IoU prediction output.
DEFAULT_FOCAL_WEIGHT = 20.0
DEFAULT_DICE_WEIGHT = 1.0

prev = time.time()


def _amp_autocast(device: str):
    """bf16 autocast on CUDA -- no GradScaler needed (bf16 has fp32's
    exponent range, unlike fp16), and Blackwell's tensor cores run bf16
    matmuls natively at full rate."""
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _load_and_resize_rgb(path: Path, image_size: int) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"could not read image at '{path}'")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (image_size, image_size))


def _normalize_image(rgb: np.ndarray) -> torch.Tensor:
    img = rgb.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


class SegmentationRunDataset(Dataset):
    """Reads (rgb, dense category mask) pairs straight from a
    run_segmentation_sweep run directory: rgb/<frame_id>.png +
    masks_category/<frame_id>.png, indexed via segmentation_frames.jsonl,
    with class_names from summary.json (class_names[0] == background, see
    segmentation_capture.build_category_lut). This is the lossless path --
    masks_category/*.png already are exactly this architecture's training
    target, no polygon round-trip needed.
    """

    def __init__(
        self, run_dir: Path, frame_ids: Sequence[str], image_size: int = IMAGE_SIZE
    ):
        self.run_dir = Path(run_dir)
        self.frame_ids = list(frame_ids)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2

        frame_id = self.frame_ids[idx]
        rgb = _load_and_resize_rgb(
            self.run_dir / "rgb" / f"{frame_id}.png", self.image_size
        )
        mask = cv2.imread(
            str(self.run_dir / "masks_category" / f"{frame_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise FileNotFoundError(
                f"could not read category mask for frame '{frame_id}' in {self.run_dir}"
            )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        return _normalize_image(rgb), torch.from_numpy(mask.astype(np.int64))


def load_run_frame_ids(run_dir: Path) -> Tuple[List[str], List[str]]:
    """Returns (frame_ids, class_names) for a run directory, reading
    segmentation_frames.jsonl + summary.json the same way
    export_annotations.load_run does."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    class_names = summary["class_names"]
    lines = (run_dir / "segmentation_frames.jsonl").read_text().splitlines()
    frame_ids = [json.loads(line)["frame_id"] for line in lines]
    return frame_ids, class_names


class CocoPolygonDataset(Dataset):
    """Reads (rgb, dense category mask) pairs from an export_annotations
    COCO instances.json, rasterizing each annotation's polygon(s) back into
    a dense mask via cv2.fillPoly. Lossier than SegmentationRunDataset
    (contour -> polygon approximation -> rasterize) -- use this only when
    the run_dir's masks_category/ isn't available, e.g. a portable export
    handed off on its own. category_id in the COCO file is already the
    class index (see export_annotations.export_coco), so no remapping is
    needed; unannotated pixels default to 0 (background).
    """

    def __init__(
        self, coco_json: Path, images_root: Path, image_size: int = IMAGE_SIZE
    ):
        self.images_root = Path(images_root)
        self.image_size = image_size

        coco = json.loads(Path(coco_json).read_text())
        self.class_names = ["background"] + [
            c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])
        ]

        self.images_by_id = {img["id"]: img for img in coco["images"]}
        self.anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images_by_id.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2

        image_id = self.image_ids[idx]
        image_rec = self.images_by_id[image_id]
        orig_w, orig_h = image_rec["width"], image_rec["height"]

        rgb = _load_and_resize_rgb(
            self.images_root / image_rec["file_name"], self.image_size
        )

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in self.anns_by_image.get(image_id, []):
            for polygon in ann["segmentation"]:
                pts = (
                    np.array(polygon, dtype=np.float64)
                    .reshape(-1, 2)
                    .round()
                    .astype(np.int32)
                )
                cv2.fillPoly(mask, [pts], color=int(ann["category_id"]))
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )

        return _normalize_image(rgb), torch.from_numpy(mask.astype(np.int64))


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, List[str]]:
    """Builds (train_ds, val_ds, class_names) from either --run-dir or
    --coco-json, splitting deterministically by --val-frac/--seed."""
    if args.run_dir:
        run_dir = Path(args.run_dir)
        frame_ids, class_names = load_run_frame_ids(run_dir)
        rng = random.Random(args.seed)
        shuffled = frame_ids[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_frac)) if len(shuffled) > 1 else 0
        val_ids, train_ids = shuffled[:n_val], shuffled[n_val:]
        return (
            SegmentationRunDataset(run_dir, train_ids, args.image_size),
            SegmentationRunDataset(run_dir, val_ids, args.image_size),
            class_names,
        )

    images_root = (
        Path(args.images_root) if args.images_root else Path(args.coco_json).parents[2]
    )
    full = CocoPolygonDataset(Path(args.coco_json), images_root, args.image_size)
    rng = random.Random(args.seed)
    indices = list(range(len(full)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_frac)) if len(indices) > 1 else 0
    val_idx, train_idx = set(indices[:n_val]), set(indices[n_val:])

    class _Subset(Dataset):
        def __init__(self, base: CocoPolygonDataset, ids: List[int]):
            self.base, self.ids = base, ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return self.base[self.ids[i]]

    return (
        _Subset(full, [i for i in indices if i in train_idx]),
        _Subset(full, [i for i in indices if i in val_idx]),
        full.class_names,
    )


# ---------------------------------------------------------------------------
# LoRA wrapping
# ---------------------------------------------------------------------------


@dataclass
class LoraSettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES


def apply_lora_to_image_encoder(
    image_encoder: nn.Module, settings: LoraSettings
) -> nn.Module:
    """Wraps the Hiera trunk's attention Linear layers (attn.qkv, attn.proj
    -- see hieradet.MultiScaleAttention) with LoRA adapters via peft, and
    freezes everything else in the encoder. target_modules matches by
    module-name suffix, so it will also catch the occasional stage-transition
    MultiScaleBlock.proj (used only when a block changes channel dims) --
    harmless (it's still just a Linear), just worth knowing it's not
    exclusively attention."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
    )
    return get_peft_model(image_encoder, config)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def build_model(
    num_classes: int,
    encoder_mode: str,
    lora_settings: Optional[LoraSettings],
    device: str,
) -> SimpleSAM2Seg:
    """Reuses sam2_custom_head's backbone builder + SimpleSAM2Seg
    architecture (drop-in with the deployed model) rather than
    reimplementing it. num_classes/CLASS ordering come from the dataset's
    own class_names, NOT sam2_custom_head.NUM_CLASSES/CLASS_NAMES -- those
    are specific to the older soil/bedrock/sand/bigrock checkpoint and
    don't match next.md's category set (small_rock/big_rock/bedrock/
    hole_in_ground + background). SimpleSAM2Seg's constructor already takes
    num_classes as a parameter, so no architecture change is needed, only
    not hardcoding the legacy constant.
    """
    sam2_backbone = build_sam2_backbone(device)
    model = SimpleSAM2Seg(sam2_backbone, num_classes=num_classes).to(device)

    if encoder_mode == "frozen":
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    elif encoder_mode == "lora":
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        assert lora_settings is not None
        model.image_encoder = apply_lora_to_image_encoder(
            model.image_encoder, lora_settings
        )
    elif encoder_mode == "full":
        for p in model.image_encoder.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown encoder_mode {encoder_mode!r}")

    return model


# ---------------------------------------------------------------------------
# Loss: dice + focal, adapted from SAM2's own reference training loss
# (sam2/training/loss_fns.py) from its binary per-mask form (sigmoid over
# one predicted mask at a time) to this head's dense multi-class softmax
# form (one classifier over all pixels at once) -- same focal
# alpha/gamma defaults and the same 20:1 focal:dice weighting as Meta's
# sam2.1_hiera_b+_MOSE_finetune.yaml, just the multi-class generalization
# since there's no per-object binary mask here.
# ---------------------------------------------------------------------------


def multiclass_focal_loss(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return (alpha * (1 - pt).pow(gamma) * ce).mean()


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    exclude_background: bool = False,
    eps: float = 1.0,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    start = 1 if exclude_background else 0
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))[start:]
    cardinality = (probs + one_hot).sum(dim=(0, 2, 3))[start:]
    dice_per_class = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice_per_class.mean()


def seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    focal_weight: float = DEFAULT_FOCAL_WEIGHT,
    dice_weight: float = DEFAULT_DICE_WEIGHT,
    dice_exclude_background: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    focal = multiclass_focal_loss(logits, target)
    dice = multiclass_dice_loss(
        logits, target, num_classes, exclude_background=dice_exclude_background
    )
    total = focal_weight * focal + dice_weight * dice
    return total, {"focal": focal.item(), "dice": dice.item(), "total": total.item()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_miou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Tuple[float, List[float]]:
    ious = []
    for cls in range(num_classes):
        pred_cls, target_cls = pred == cls, target == cls
        union = (pred_cls | target_cls).sum().item()
        ious.append(
            (pred_cls & target_cls).sum().item() / union if union > 0 else float("nan")
        )
    valid = [v for v in ious if not np.isnan(v)]
    return (sum(valid) / len(valid) if valid else 0.0), ious


def compute_confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Pixel-level confusion matrix, rows = ground truth class, columns =
    predicted class. pred/target expected on CPU."""
    idx = (target.reshape(-1) * num_classes + pred.reshape(-1)).to(torch.int64)
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def print_confusion_matrix(cm: torch.Tensor, class_names: List[str]) -> None:
    cm = cm.cpu().numpy()
    col_w = 12
    name_w = max(len(n) for n in class_names) + 2
    header = " " * name_w + "".join(f"{n[:col_w - 2]:>{col_w}}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>{col_w},}" for j in range(len(class_names)))
        print(f"{name:<{name_w}}{row}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    out_dir: Path,
    model: SimpleSAM2Seg,
    encoder_mode: str,
    class_names: List[str],
    args: argparse.Namespace,
    epoch: int,
    val_metrics: dict,
) -> None:
    """Saves adapters/heads separately, never merged into the base
    checkpoint, so swapping the base SAM2 checkpoint later stays possible:
      lora_adapter/    -- peft's own save_pretrained output (encoder_mode=lora only)
      encoder_full.pt  -- full image_encoder state dict (encoder_mode=full only)
      seg_head.pt      -- always: the task head is never a LoRA target
      metadata.json    -- class list, image size, LoRA config, base checkpoint used
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if encoder_mode == "lora":
        model.image_encoder.save_pretrained(out_dir / "lora_adapter")
    elif encoder_mode == "full":
        torch.save(model.image_encoder.state_dict(), out_dir / "encoder_full.pt")

    torch.save(model.seg_head.state_dict(), out_dir / "seg_head.pt")

    metadata = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": args.image_size,
        "encoder_mode": encoder_mode,
        "lora": (
            {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": args.lora_target_modules,
            }
            if encoder_mode == "lora"
            else None
        ),
        "sam2_root": str(SAM2_ROOT),
        "sam2_config": SAM2_MODEL_CONFIG,
        "epoch": epoch,
        "val_metrics": val_metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_finetuned_model(checkpoint_dir: Path, device: str = "cuda") -> SimpleSAM2Seg:
    """Convenience loader mirroring save_checkpoint's layout -- rebuilds
    the base backbone, re-attaches the LoRA adapter (or full/frozen encoder
    weights) and the task head. This is NOT wired into
    sam_weights_loader.load_sam_model / sam_segmenter yet; swapping the
    pipeline's frozen-checkpoint model for this one is a deliberate manual
    follow-up (see module docstring "still manual/TODO" note), since
    sam_weights_loader.SAM_CHECKPOINT_PATH currently points at a single
    fully-merged checkpoint rather than an adapter directory."""
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())

    sam2_backbone = build_sam2_backbone(device)
    model = SimpleSAM2Seg(sam2_backbone, num_classes=metadata["num_classes"]).to(device)

    encoder_mode = metadata["encoder_mode"]
    if encoder_mode == "lora":
        from peft import PeftModel

        model.image_encoder = PeftModel.from_pretrained(
            model.image_encoder, checkpoint_dir / "lora_adapter"
        )
    elif encoder_mode == "full":
        state = torch.load(
            checkpoint_dir / "encoder_full.pt", map_location=device, weights_only=True
        )
        model.image_encoder.load_state_dict(state)

    seg_head_state = torch.load(
        checkpoint_dir / "seg_head.pt", map_location=device, weights_only=True
    )
    model.seg_head.load_state_dict(seg_head_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    device = args.device
    if device.startswith("cuda"):
        # TF32 for any matmul/conv that falls outside the bf16 autocast
        # region (e.g. LoRA's fp32 master weights) -- free precision/perf
        # tradeoff on Blackwell's tensor cores, same as A100/H100.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Fixed image_size + drop_last=True train batches means stable
        # input shapes, so cudnn autotuning pays for itself.
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, class_names = build_datasets(args)
    num_classes = len(class_names)
    print(f"[input] classes: {class_names} ({num_classes})")
    print(f"[input] train: {len(train_ds)} frames, val: {len(val_ds)} frames")

    if args.dry_run:
        print(
            "[input] dry run -- dataset/config validated, stopping before model build"
        )
        return Path(args.out_dir)

    lora_settings = LoraSettings(
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        tuple(args.lora_target_modules),
    )
    model = build_model(
        num_classes,
        args.encoder_mode,
        lora_settings if args.encoder_mode == "lora" else None,
        device,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[config] encoder_mode={args.encoder_mode}, trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > args.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        print(f"[bro] resumed from {args.resume} at epoch {start_epoch}")
        # load_state_dict copies these into model/optimizer already; without
        # freeing this, the raw checkpoint (a full duplicate set of weights +
        # AdamW momentum buffers on `device`) sits unused for the whole run.
        del state
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best_miou = -1.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        t0 = time.monotonic()
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(
                device, non_blocking=True
            )
            with _amp_autocast(device):
                logits = model(images)
                loss, parts = seg_loss(
                    logits,
                    masks,
                    num_classes,
                    args.focal_weight,
                    args.dice_weight,
                    args.dice_exclude_background,
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0
            )
            optimizer.step()
            epoch_losses.append(parts["total"])

        model.eval()
        val_losses: List[float] = []
        val_ious: List[List[float]] = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(
                    device, non_blocking=True
                )
                with _amp_autocast(device):
                    logits = model(images)
                    _, parts = seg_loss(
                        logits,
                        masks,
                        num_classes,
                        args.focal_weight,
                        args.dice_weight,
                        args.dice_exclude_background,
                    )
                val_losses.append(parts["total"])
                pred = logits.argmax(dim=1)
                _, per_class = compute_miou(pred, masks, num_classes)
                val_ious.append(per_class)

        per_class_arr = np.array(val_ious, dtype=np.float64)
        per_class_miou = (
            np.nanmean(per_class_arr, axis=0).tolist()
            if len(val_ious)
            else [float("nan")] * num_classes
        )
        mean_iou = float(np.nanmean(per_class_miou)) if per_class_miou else 0.0

        metrics = {
            "epoch": epoch,
            "train_loss": (
                float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            ),
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            "val_mean_iou": mean_iou,
            "val_class_iou": dict(zip(class_names, per_class_miou)),
            "epoch_seconds": time.monotonic() - t0,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(
            f"[{time.strftime('%H:%M:%S')}]][ninni tem] epoch {epoch}/{args.epochs} "
            f"train_loss={metrics['train_loss']:.4f} val_loss={metrics['val_loss']:.4f} val_mIoU={mean_iou:.4f}"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            out_dir / "training_state.pt",
        )

        if mean_iou > best_miou:
            best_miou = mean_iou
            save_checkpoint(
                out_dir / "best",
                model,
                args.encoder_mode,
                class_names,
                args,
                epoch,
                metrics,
            )
            print(
                f"[{time.strftime('%H:%M:%S')}]][ninni tem] saved new best (val_mIoU={mean_iou:.4f}) -> {out_dir / 'best'}"
            )

    save_checkpoint(
        out_dir / "final",
        model,
        args.encoder_mode,
        class_names,
        args,
        args.epochs,
        metrics,
    )
    print(
        f"[{time.strftime('%H:%M:%S')}]][wakeup tem] done. best val_mIoU={best_miou:.4f}, final checkpoint -> {out_dir / 'final'}"
    )
    print(f"training took {time.time() - prev:.2f} seconds")

    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            with _amp_autocast(device):
                logits = model(images)
            pred = logits.argmax(dim=1).cpu()
            cm += compute_confusion_matrix(pred, masks, num_classes)
    print(f"\n[eval] confusion matrix on validation set ({len(val_ds)} frames):")
    print_confusion_matrix(cm, class_names)
    total_pixels = cm.sum().item()
    if total_pixels:
        print(
            f"[eval] overall pixel accuracy (val set): {cm.diagonal().sum().item() / total_pixels:.4f}"
        )

    n_random = min(args.n_random_images, len(train_ds) + len(val_ds))
    if n_random > 0:
        from torch.utils.data import ConcatDataset

        full_ds = ConcatDataset([train_ds, val_ds])
        sample_idx = random.Random(args.seed + 1).sample(range(len(full_ds)), n_random)
        print(
            f"\n[eval] pixel accuracy on {n_random} random image(s) from the dataset:"
        )
        accs = []
        with torch.no_grad():
            for i in sample_idx:
                image, mask = full_ds[i]
                with _amp_autocast(device):
                    logits = model(image.unsqueeze(0).to(device))
                pred = logits.argmax(dim=1).squeeze(0).cpu()
                acc = (pred == mask).float().mean().item()
                accs.append(acc)
                print(f"    idx={i:>6}  accuracy={acc:.4f}")
        print(
            f"[eval] mean accuracy over {n_random} random image(s): {np.mean(accs):.4f}"
        )

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        default=None,
        help="a run directory produced by run_segmentation_sweep.py (preferred, lossless)",
    )
    source.add_argument(
        "--coco-json",
        default=None,
        help="an export_annotations coco/instances.json (fallback when run_dir isn't available)",
    )
    ap.add_argument(
        "--images-root",
        default=None,
        help="root that --coco-json's file_name paths are relative to (default: two levels up, i.e. the run_dir)",
    )

    ap.add_argument("--out-dir", required=True, help="where to write checkpoints/logs")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--encoder-mode",
        choices=["lora", "frozen", "full"],
        default="lora",
        help="lora: LoRA-adapt trunk attention (default); frozen: matches sam/train_sam2_simple_fast.py's original decoder-only default; full: full encoder finetune",
    )
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules", nargs="+", default=list(DEFAULT_LORA_TARGET_MODULES)
    )

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="cv2 decode/resize is CPU-bound; 4x active GPU count is a common floor, "
        "but with few GPUs and many cores it's worth going higher to keep the GPU fed "
        "(default: min(8, cpu_count))",
    )
    ap.add_argument("--focal-weight", type=float, default=DEFAULT_FOCAL_WEIGHT)
    ap.add_argument("--dice-weight", type=float, default=DEFAULT_DICE_WEIGHT)
    ap.add_argument(
        "--dice-exclude-background",
        action="store_true",
        help="drop background from the dice term's class average -- background/bedrock pixel counts dominate a scene (next.md's noted class imbalance), this keeps dice from being swamped by it",
    )

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--resume", default=None, help="path to a training_state.pt to resume from"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate dataset/class list and stop before building the model or training",
    )
    ap.add_argument(
        "--n-random-images",
        type=int,
        default=10,
        help="after training, report per-image pixel accuracy on this many random images "
        "sampled from the full (train+val) dataset",
    )

    return ap


if __name__ == "__main__":
    parsed = _build_argparser().parse_args()
    if (
        parsed.coco_json
        and not parsed.images_root
        and len(Path(parsed.coco_json).parents) < 3
    ):
        raise SystemExit(
            "--coco-json path too shallow to infer --images-root (need "
            "<run_dir>/annotations_export/coco/instances.json or deeper); pass --images-root explicitly"
        )
    train(parsed)
