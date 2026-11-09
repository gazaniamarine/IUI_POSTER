"""
Evaluation and inference utilities for Simple Fast SAM2 model.

Features:
- Load best trained model (`training_output/checkpoints/best_model.pth`)
- Evaluate on the full test set with all segmentation metrics
- Generate confusion matrix and metric plots
- Save sample predictions vs ground truth with colored overlays
- Run inference on a single image and save/show multiclass segmentation
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Reuse components from training script
from train_sam2_simple_fast import (
    ROOT_DIR,
    DATASET_ROOT,
    SAM2_CHECKPOINT,
    SAM2_CONFIG_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    DEVICE,
    MarsDataset,
    SimpleSAM2Seg,
    compute_segmentation_metrics,
    aggregate_metrics,
)


# =====================================================================
# PATHS / OUTPUT DIRS
# =====================================================================

OUTPUT_DIR = DATASET_ROOT / "training_output"
MODEL_SAVE_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
SAMPLES_DIR = OUTPUT_DIR / "samples"

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# COLOR MAP FOR VISUALIZATION
# =====================================================================

# Define distinct colors for each class (in RGB, 0-255)
CLASS_COLORS = {
    0: (150, 75, 0),    # soil - brown
    1: (128, 128, 128), # bedrock - gray
    2: (255, 215, 0),   # sand - yellow
    3: (220, 20, 60),   # big_rock - red
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2D mask of class indices to a color RGB image.

    mask: (H, W) with values in {0,1,2,3,255}
    returns: (H, W, 3) uint8 RGB image
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

    # Ignore 255 (NULL) -> leave as black
    return color_mask


def overlay_masks(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a color mask on top of an RGB image.

    image_rgb: (H, W, 3) uint8
    mask_rgb: (H, W, 3) uint8
    """
    image = image_rgb.astype(np.float32) / 255.0
    mask = mask_rgb.astype(np.float32) / 255.0
    overlay = (1 - alpha) * image + alpha * mask
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay


# =====================================================================
# SAM2 BACKBONE + MODEL LOADING
# =====================================================================

def load_sam2_backbone() -> nn.Module:
    """Load SAM2 backbone exactly as in training."""
    print("Loading SAM2 backbone for evaluation...")
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(SAM2_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="sam2.1/sam2.1_hiera_l")
        OmegaConf.resolve(cfg)
        sam2_model = instantiate(cfg.model, _recursive_=True)

        checkpoint = torch.load(str(SAM2_CHECKPOINT), map_location=DEVICE, weights_only=False)
        if "model" in checkpoint:
            sam2_model.load_state_dict(checkpoint["model"], strict=False)
        else:
            sam2_model.load_state_dict(checkpoint, strict=False)

        sam2_model = sam2_model.to(DEVICE)
        sam2_model.eval()

    print("  SAM2 backbone loaded.")
    return sam2_model


def load_best_model() -> nn.Module:
    """
    Load SimpleSAM2Seg model with weights from best checkpoint.

    Expects: `training_output/checkpoints/best_model.pth`
    """
    ckpt_path = MODEL_SAVE_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found at: {ckpt_path}")

    print(f"Loading best model from: {ckpt_path}")
    sam2_backbone = load_sam2_backbone()
    model = SimpleSAM2Seg(sam2_backbone, NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("  Best model loaded.")
    return model


# =====================================================================
# FULL TEST-SET EVALUATION
# =====================================================================

def evaluate_on_test_set(save_prefix: str = "eval_test") -> None:
    """
    Evaluate best model on the full test set.

    Saves:
      - metrics JSON: `metrics/{save_prefix}_metrics.json`
      - confusion matrix plot: `plots/{save_prefix}_confusion_matrix.png`
    """
    print("=" * 80)
    print("Evaluating best model on full test set")
    print("=" * 80)

    test_ds = MarsDataset("test", augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"Test samples: {len(test_ds)} | Test batches: {len(test_loader)}")

    model = load_best_model()
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

    test_loss = 0.0
    test_batches = 0
    test_metrics_list = []
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="[Eval Test]"):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if not torch.isnan(loss) and not torch.isinf(loss):
                test_loss += loss.item()
                test_batches += 1

            pred = torch.argmax(logits, dim=1)

            batch_metrics = compute_segmentation_metrics(pred, masks, NUM_CLASSES, ignore_index=255)
            test_metrics_list.append(batch_metrics)

            mask_valid = masks != 255
            if mask_valid.sum() > 0:
                all_test_preds.append(pred[mask_valid].cpu().numpy())
                all_test_targets.append(masks[mask_valid].cpu().numpy())

    avg_test_loss = test_loss / max(test_batches, 1)
    test_metrics = aggregate_metrics(test_metrics_list)

    print("\nTest results (best model):")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"  Test Mean IoU (mIoU): {test_metrics['mean_iou']:.4f}")
    print(f"  Test Mean Dice: {test_metrics['mean_dice']:.4f}")
    print("\nPer-Class IoU:")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {test_metrics['class_iou'][cls]:.4f}")
    print("\nPer-Class Dice:")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {test_metrics['class_dice'][cls]:.4f}")

    # Save metrics JSON
    metrics_path = METRICS_DIR / f"{save_prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "avg_test_loss": avg_test_loss,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved test metrics to: {metrics_path}")

    # Confusion matrix
    if all_test_preds:
        y_pred = np.concatenate(all_test_preds, axis=0)
        y_true = np.concatenate(all_test_targets, axis=0)

        mask = y_true != 255
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if y_true.size > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
            cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".3f",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cbar_kws={"label": "Normalized Count"},
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Normalized Confusion Matrix (Test Set, Best Model)")
            plt.tight_layout()

            cm_path = PLOTS_DIR / f"{save_prefix}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved confusion matrix to: {cm_path}")


# =====================================================================
# SAMPLE PREDICTION VISUALIZATIONS
# =====================================================================

def save_sample_predictions(num_samples: int = 8) -> None:
    """
    Save sample predictions vs ground truth overlays for the test set.

    Saves images into `training_output/samples/`.
    """
    print("=" * 80)
    print(f"Saving {num_samples} sample predictions from test set")
    print("=" * 80)

    test_ds = MarsDataset("test", augment=False)
    model = load_best_model()

    indices = np.linspace(0, len(test_ds) - 1, num=min(num_samples, len(test_ds)), dtype=int)

    for idx in indices:
        img_tensor, mask_tensor = test_ds[idx]

        # Prepare input
        image_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        mask_np = mask_tensor.numpy().astype(np.int64)

        # Run model
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(DEVICE)
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Colorize masks
        gt_color = colorize_mask(mask_np)
        pred_color = colorize_mask(pred)

        # Overlays
        overlay_gt = overlay_masks(image_np, gt_color, alpha=0.5)
        overlay_pred = overlay_masks(image_np, pred_color, alpha=0.5)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(overlay_gt)
        axes[1].set_title("Ground Truth Overlay")
        axes[1].axis("off")

        axes[2].imshow(overlay_pred)
        axes[2].set_title("Prediction Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = SAMPLES_DIR / f"sample_{idx}_prediction.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved sample prediction: {out_path}")


# =====================================================================
# SINGLE-IMAGE INFERENCE
# =====================================================================

def preprocess_image_for_model(image_bgr: np.ndarray) -> torch.Tensor:
    """Resize and normalize image as in `MarsDataset`."""
    image_rgb = image_bgr[..., ::-1]
    image_rgb = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img = image_rgb.astype(np.float32) / 255.0

    # Same normalization as dataset
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(img).float()
    return tensor


def run_inference_on_image(image_path: str, output_path: str | None = None, show: bool = False) -> str:
    """
    Run multiclass segmentation on a single image and save overlay.

    image_path: path to input image (any RGB/JPG/PNG)
    output_path: where to save the overlay (if None, save under `samples/`)
    show: if True, display using matplotlib

    Returns: path to saved overlay image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Running inference on image: {image_path}")

    # Read image (BGR)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Keep original for visualization (resize for overlay)
    image_rgb_orig = image_bgr[..., ::-1]
    image_rgb_resized = cv2.resize(image_rgb_orig, (IMAGE_SIZE, IMAGE_SIZE))

    # Preprocess for model
    img_tensor = preprocess_image_for_model(image_bgr)

    model = load_best_model()
    model.eval()

    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(DEVICE)
        logits = model(inp)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    pred_color = colorize_mask(pred)
    overlay_pred = overlay_masks(image_rgb_resized, pred_color, alpha=0.5)

    if output_path is None:
        output_path = SAMPLES_DIR / f"inference_{image_path.stem}.png"
    else:
        output_path = Path(output_path)

    # Save result
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_pred)
    plt.axis("off")
    plt.title("Predicted Multiclass Segmentation Overlay")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved inference overlay to: {output_path}")

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_pred)
        plt.axis("off")
        plt.title("Predicted Multiclass Segmentation Overlay")
        plt.show()

    return str(output_path)


# =====================================================================
# MAIN (for direct CLI usage)
# =====================================================================

if __name__ == "__main__":
    # Example usage:
    #   conda activate sam3
    #   python evaluate_sam2_simple_fast.py
    #
    # This will:
    #   1) Evaluate best model on full test set
    #   2) Save a few sample predictions
    #
    # For single-image inference, import and call `run_inference_on_image(...)`.

    evaluate_on_test_set(save_prefix="best_model_test")
    save_sample_predictions(num_samples=8)

