"""
Simple and Fast SAM2 Fine-tuning for 4-Class Semantic Segmentation
Based on reference code structure - optimized for speed
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================
# CONFIGURATION
# ==========================

ROOT_DIR = Path(__file__).parent
DATASET_ROOT = ROOT_DIR / "sam_v2_dataset"
SAM2_CHECKPOINT = ROOT_DIR / "sam2.1_hiera_large.pt"
SAM2_CONFIG_DIR = ROOT_DIR / "sam2" / "sam2" / "configs"

# Training settings
IMAGE_SIZE = 1024 # Smaller for faster training
NUM_CLASSES = 4
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]

BATCH_SIZE = 8
NUM_EPOCHS = 10
USE_FULL_DATASET = True  # Use all data, not random samples

# Learning rates
LEARNING_RATE = 1e-4  # Reduced from default to prevent NaN
WEIGHT_DECAY = 1e-4

# Performance
USE_MIXED_PRECISION = True
ACCUMULATION_STEPS = 4

# Output
OUTPUT_DIR = DATASET_ROOT / "training_output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_SAVE_DIR = OUTPUT_DIR / "checkpoints"
MODEL_SAVE_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training_log.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "training_summary.json"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================
# IMPORT SAM2
# ==========================

try:
    from sam2.build_sam import build_sam2
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    print("[OK] SAM2 imported")
except ImportError as e:
    print(f"[ERROR] {e}")
    raise

# ==========================
# DATASET CLASS
# ==========================

class MarsDataset(Dataset):
    """Dataset for Mars terrain segmentation."""
    def __init__(self, split="train", augment=False):
        self.split = split
        self.augment = augment and (split == "train")
        self.img_dir = DATASET_ROOT / split / "images"
        self.mask_dir = DATASET_ROOT / split / "labels"
        
        # Get all image-mask pairs
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            mask_path = self.mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                self.samples.append({
                    "image": str(img_path),
                    "mask": str(mask_path)
                })
        
        print(f"  Loaded {len(self.samples)} samples for {split}")
        
        # Transforms
        if self.augment:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ent = self.samples[idx]
        
        # Read image
        img = cv2.imread(ent["image"])[..., ::-1]  # BGR to RGB
        mask = cv2.imread(ent["mask"], cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            # Return dummy data if read fails
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Validate mask values (should be 0, 1, 2, 3, or 255)
        unique_vals = np.unique(mask)
        valid_vals = {0, 1, 2, 3, 255}
        if not all(v in valid_vals for v in unique_vals):
            # Clip invalid values
            mask = np.clip(mask, 0, 3)
        
        # Convert image
        img = img.astype(np.float32) / 255.0
        img = self.img_transform(img)
        
        # Check for NaN/Inf in image
        if torch.isnan(img).any() or torch.isinf(img).any():
            img = torch.zeros_like(img)
        
        # Convert mask
        mask = torch.from_numpy(mask.astype(np.int64))
        
        return img, mask

# ==========================
# SIMPLE MODEL
# ==========================

class SimpleSegHead(nn.Module):
    """Simple segmentation head."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            
            nn.Conv2d(64, num_classes, 1),
        )
    
    def forward(self, x):
        return self.decoder(x)

class SimpleSAM2Seg(nn.Module):
    """Simple SAM2 + segmentation head."""
    def __init__(self, sam2_model, num_classes):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
            enc_out = self.image_encoder(dummy)
            if isinstance(enc_out, dict):
                feat = enc_out.get('vision_features', enc_out.get('backbone_fpn', [None])[0])
            else:
                feat = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out
            embed_dim = feat.shape[1] if isinstance(feat, torch.Tensor) else 256
        
        self.seg_head = SimpleSegHead(embed_dim, num_classes)
    
    def forward(self, x):
        enc_out = self.image_encoder(x)
        if isinstance(enc_out, dict):
            features = enc_out.get('vision_features', enc_out.get('backbone_fpn', [None])[0])
        else:
            features = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out
        
        logits = self.seg_head(features)
        if logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            logits = F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), 
                                  mode="bilinear", align_corners=False)
        return logits

# ==========================
# METRICS COMPUTATION
# ==========================

def compute_segmentation_metrics(pred, target, num_classes=4, ignore_index=255):
    """Compute comprehensive segmentation metrics."""
    mask_valid = target != ignore_index
    
    if mask_valid.sum() == 0:
        return {
            "pixel_accuracy": 0.0,
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "class_iou": {CLASS_NAMES[i]: 0.0 for i in range(num_classes)},
            "class_dice": {CLASS_NAMES[i]: 0.0 for i in range(num_classes)},
        }
    
    # Pixel accuracy
    correct = (pred[mask_valid] == target[mask_valid]).sum().item()
    total = mask_valid.sum().item()
    pixel_acc = correct / total if total > 0 else 0.0
    
    # Per-class metrics
    ious = []
    dices = []
    class_iou_dict = {}
    class_dice_dict = {}
    
    for cls in range(num_classes):
        pred_cls = (pred == cls) & mask_valid
        target_cls = (target == cls) & mask_valid
        
        intersection = (pred_cls & target_cls).sum().float().item()
        union = (pred_cls | target_cls).sum().float().item()
        
        # IoU
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        ious.append(iou)
        class_iou_dict[CLASS_NAMES[cls]] = iou
        
        # Dice
        if intersection + union > 0:
            dice = (2 * intersection) / (pred_cls.sum().float().item() + target_cls.sum().float().item() + 1e-8)
        else:
            dice = 0.0
        dices.append(dice)
        class_dice_dict[CLASS_NAMES[cls]] = dice
    
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    
    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "class_iou": class_iou_dict,
        "class_dice": class_dice_dict,
    }

def aggregate_metrics(metrics_list):
    """Aggregate metrics across batches."""
    if not metrics_list:
        return None
    
    aggregated = {
        "pixel_accuracy": np.mean([m["pixel_accuracy"] for m in metrics_list]),
        "mean_iou": np.mean([m["mean_iou"] for m in metrics_list]),
        "mean_dice": np.mean([m["mean_dice"] for m in metrics_list]),
        "class_iou": {},
        "class_dice": {},
    }
    
    # Aggregate per-class metrics
    for cls in CLASS_NAMES:
        aggregated["class_iou"][cls] = np.mean([m["class_iou"][cls] for m in metrics_list])
        aggregated["class_dice"][cls] = np.mean([m["class_dice"][cls] for m in metrics_list])
    
    return aggregated

# ==========================
# PLOTTING FUNCTIONS
# ==========================

def plot_training_curves(epoch_metrics, output_path):
    """Plot training curves for all metrics."""
    epochs = [m["epoch"] for m in epoch_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    axes[0, 0].plot(epochs, [m["train_loss"] for m in epoch_metrics], label="Train", linewidth=2, marker='o')
    axes[0, 0].plot(epochs, [m["val_loss"] for m in epoch_metrics], label="Val", linewidth=2, marker='s')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU curves
    axes[0, 1].plot(epochs, [m["train_miou"] for m in epoch_metrics], label="Train mIoU", linewidth=2, marker='o')
    axes[0, 1].plot(epochs, [m["val_miou"] for m in epoch_metrics], label="Val mIoU", linewidth=2, marker='s')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mIoU")
    axes[0, 1].set_title("Mean IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pixel Accuracy curves
    axes[1, 0].plot(epochs, [m["train_pixel_acc"] for m in epoch_metrics], label="Train Pixel Acc", linewidth=2, marker='o')
    axes[1, 0].plot(epochs, [m["val_pixel_acc"] for m in epoch_metrics], label="Val Pixel Acc", linewidth=2, marker='s')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Pixel Accuracy")
    axes[1, 0].set_title("Pixel Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Per-class IoU (last epoch)
    if epoch_metrics:
        last_metrics = epoch_metrics[-1]
        classes = list(last_metrics["val_class_iou"].keys())
        ious = list(last_metrics["val_class_iou"].values())
        bars = axes[1, 1].bar(classes, ious, color=['brown', 'gray', 'yellow', 'red'])
        axes[1, 1].set_xlabel("Class")
        axes[1, 1].set_ylabel("IoU")
        axes[1, 1].set_title("Per-Class IoU (Validation, Last Epoch)")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        for bar, iou in zip(bars, ious):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{iou:.3f}', ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path, num_classes=4):
    """Generate confusion matrix heatmap."""
    mask = y_true != 255
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if y_true.size == 0:
        return
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Normalized Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Normalized Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(epoch_metrics, output_path):
    """Plot per-class metrics over epochs."""
    epochs = [m["epoch"] for m in epoch_metrics]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Per-class IoU over epochs
    for cls in CLASS_NAMES:
        ious = [m["val_class_iou"][cls] for m in epoch_metrics]
        axes[0].plot(epochs, ious, label=cls, linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Per-Class IoU (Validation) Over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Per-class Dice over epochs
    for cls in CLASS_NAMES:
        dices = [m["val_class_dice"][cls] for m in epoch_metrics]
        axes[1].plot(epochs, dices, label=cls, linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].set_title("Per-Class Dice Coefficient (Validation) Over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================
# TRAINING
# ==========================

def main():
    print("=" * 80)
    print("Simple Fast SAM2 Fine-tuning for 4-Class Segmentation")
    print("=" * 80)
    
    # Load datasets
    print("Loading datasets...")
    train_ds = MarsDataset("train", augment=True)
    val_ds = MarsDataset("val", augment=False)
    test_ds = MarsDataset("test", augment=False)
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE == "cuda"))
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Load SAM2
    print("Loading SAM2 model...")
    GlobalHydra.instance().clear()
    
    try:
        with initialize_config_dir(config_dir=str(SAM2_CONFIG_DIR), version_base=None):
            cfg = compose(config_name="sam2.1/sam2.1_hiera_l")
            OmegaConf.resolve(cfg)
            sam2_model = instantiate(cfg.model, _recursive_=True)
            
            checkpoint = torch.load(str(SAM2_CHECKPOINT), map_location=DEVICE)
            if 'model' in checkpoint:
                sam2_model.load_state_dict(checkpoint['model'], strict=False)
            else:
                sam2_model.load_state_dict(checkpoint, strict=False)
            
            sam2_model = sam2_model.to(DEVICE)
            sam2_model.eval()
            print("  SAM2 loaded")
    except Exception as e:
        print(f"  Error: {e}")
        raise
    
    # Create model
    print("Building segmentation model...")
    model = SimpleSAM2Seg(sam2_model, NUM_CLASSES)
    model.to(DEVICE)
    
    # Train only decoder initially (faster)
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    print("  Encoder frozen - training decoder only")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Mixed precision
    scaler = None
    if USE_MIXED_PRECISION and DEVICE == "cuda":
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
        print("  Using mixed precision (FP16)")
    
    print(f"  Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print()
    
    # Training loop
    print("Starting training...")
    print("-" * 80)
    
    best_val_iou = 0.0
    best_epoch = 0
    epoch_metrics_list = []  # Store metrics for each epoch
    log_file = open(LOG_FILE, 'w')
    log_file.write("# Training log - one entry per epoch\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0
        step = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            step += 1
            
            # Forward
            if scaler:
                with torch.amp.autocast('cuda') if hasattr(torch, 'amp') else torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, masks)
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                        continue
                    
                    loss = loss / ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if step % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                    continue
                
                loss = loss / ACCUMULATION_STEPS
                loss.backward()
                
                if step % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Compute metrics (store for aggregation)
            pred = torch.argmax(logits, dim=1)
            mask_valid = masks != 255
            batch_metrics = compute_segmentation_metrics(pred, masks, NUM_CLASSES, ignore_index=255)
            
            if mask_valid.sum() > 0:
                # Pixel accuracy
                correct = (pred[mask_valid] == masks[mask_valid]).sum().item()
                total = mask_valid.sum().item()
                epoch_correct += correct
                epoch_total += total
            
            loss_val = loss.item() * ACCUMULATION_STEPS
            if not np.isnan(loss_val) and not np.isinf(loss_val):
                epoch_loss += loss_val
                n_batches += 1
            
            # Update progress bar
            if epoch_total > 0:
                pixel_acc = epoch_correct / epoch_total
                avg_loss = epoch_loss / max(n_batches, 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "pixel_acc": f"{pixel_acc:.4f}"})
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_pixel_acc = epoch_correct / max(epoch_total, 1)
        
        # Validation - use ALL validation data
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_metrics_list = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:3d} [Val]", leave=False):
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda') if hasattr(torch, 'amp') else torch.cuda.amp.autocast():
                        logits = model(images)
                        loss = criterion(logits, masks)
                else:
                    logits = model(images)
                    loss = criterion(logits, masks)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1
                
                # Compute metrics
                pred = torch.argmax(logits, dim=1)
                batch_metrics = compute_segmentation_metrics(pred, masks, NUM_CLASSES, ignore_index=255)
                val_metrics_list.append(batch_metrics)
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_metrics = aggregate_metrics(val_metrics_list)
        
        # Store epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_pixel_acc": train_pixel_acc,
            "val_loss": avg_val_loss,
            "val_pixel_acc": val_metrics["pixel_accuracy"],
            "train_miou": 0.0,  # Can compute if needed
            "val_miou": val_metrics["mean_iou"],
            "val_mean_dice": val_metrics["mean_dice"],
            "val_class_iou": val_metrics["class_iou"],
            "val_class_dice": val_metrics["class_dice"],
            "timestamp": datetime.now().isoformat(),
        }
        epoch_metrics_list.append(epoch_metrics)
        
        # Save metrics to JSON file
        metrics_file = METRICS_DIR / f"epoch_{epoch}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(epoch_metrics, f, indent=2)
        
        # Log to JSONL
        log_file.write(json.dumps(epoch_metrics) + "\n")
        log_file.flush()
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Pixel Acc: {train_pixel_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f} | "
              f"Val mIoU: {val_metrics['mean_iou']:.4f} | Val mDice: {val_metrics['mean_dice']:.4f}")
        
        # Save best model based on mIoU
        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, MODEL_SAVE_DIR / "best_model.pth")
            print(f"  [Saved best model: mIoU = {val_metrics['mean_iou']:.4f}]")
        
        # Generate plots for this epoch
        if len(epoch_metrics_list) > 0:
            plot_training_curves(epoch_metrics_list, PLOTS_DIR / f"training_curves_epoch_{epoch}.png")
            plot_per_class_metrics(epoch_metrics_list, PLOTS_DIR / f"per_class_metrics_epoch_{epoch}.png")
        
        scheduler.step()
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_SAVE_DIR / f"checkpoint_epoch_{epoch}.pth")
    
    # Final test evaluation - use ALL test data
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_batches = 0
    test_metrics_list = []
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="[Test]"):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            if scaler:
                with torch.amp.autocast('cuda') if hasattr(torch, 'amp') else torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                test_loss += loss.item()
                test_batches += 1
            
            pred = torch.argmax(logits, dim=1)
            
            # Compute metrics
            batch_metrics = compute_segmentation_metrics(pred, masks, NUM_CLASSES, ignore_index=255)
            test_metrics_list.append(batch_metrics)
            
            # Store for confusion matrix
            mask_valid = masks != 255
            if mask_valid.sum() > 0:
                all_test_preds.append(pred[mask_valid].cpu().numpy())
                all_test_targets.append(masks[mask_valid].cpu().numpy())
    
    avg_test_loss = test_loss / max(test_batches, 1)
    test_metrics = aggregate_metrics(test_metrics_list)
    
    # Generate confusion matrix
    if all_test_preds:
        y_pred = np.concatenate(all_test_preds, axis=0)
        y_true = np.concatenate(all_test_targets, axis=0)
        plot_confusion_matrix(y_true, y_pred, PLOTS_DIR / "confusion_matrix_test.png", NUM_CLASSES)
    
    # Generate final plots
    plot_training_curves(epoch_metrics_list, PLOTS_DIR / "training_curves_final.png")
    plot_per_class_metrics(epoch_metrics_list, PLOTS_DIR / "per_class_metrics_final.png")
    
    # Save comprehensive summary
    summary = {
        "training_config": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "device": DEVICE,
            "mixed_precision": USE_MIXED_PRECISION,
        },
        "dataset_info": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        },
        "training_stats": {
            "best_epoch": best_epoch,
            "best_val_miou": best_val_iou,
        },
        "test_metrics": test_metrics,
        "per_epoch_metrics": epoch_metrics_list,
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_file.close()
    
    print("-" * 80)
    print(f"Training complete!")
    print(f"Best validation mIoU: {best_val_iou:.4f} at epoch {best_epoch}")
    print(f"\nTest Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"  Test Mean IoU (mIoU): {test_metrics['mean_iou']:.4f}")
    print(f"  Test Mean Dice: {test_metrics['mean_dice']:.4f}")
    print(f"\nPer-Class IoU:")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {test_metrics['class_iou'][cls]:.4f}")
    print(f"\nPer-Class Dice:")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {test_metrics['class_dice'][cls]:.4f}")
    print(f"\nOutputs saved:")
    print(f"  Metrics per epoch: {METRICS_DIR}")
    print(f"  Training plots: {PLOTS_DIR}")
    print(f"  Training log: {LOG_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")
    print(f"  Best model: {MODEL_SAVE_DIR / 'best_model.pth'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
