"""
SAM2.1 Hiera-L backbone + 4-class semantic segmentation head.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SAM2_ROOT = Path(
    os.environ.get("SAM2_ROOT", "/home/gpu/Desktop/pineapple/packages/sam2")
)
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

SAM2_CONFIG_DIR = SAM2_ROOT / "sam2" / "configs"
SAM2_MODEL_CONFIG = "sam2.1/sam2.1_hiera_l"
SAM2_BACKBONE_CHECKPOINT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"

IMAGE_SIZE = 1024
NUM_CLASSES = 4
CLASS_NAMES = ["soil", "bedrock", "sand", "bigrock"]


class SimpleSegHead(nn.Module):
    """4-class decoder on top of the SAM2 image encoder's features."""

    def __init__(self, in_channels: int, num_classes: int):
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
    """SAM2.1 Hiera-L image encoder + SimpleSegHead.

    Layer names/shapes must match sam/sam/train_sam2_simple_fast.py exactly
    -- best_model.pth's state dict was produced by that definition.
    """

    def __init__(self, sam2_model: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder

        with torch.no_grad():
            device = next(self.image_encoder.parameters()).device
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            enc_out = self.image_encoder(dummy)
            if isinstance(enc_out, dict):
                feat = enc_out.get(
                    "vision_features", enc_out.get("backbone_fpn", [None])[0]
                )
            else:
                feat = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out
            embed_dim = feat.shape[1] if isinstance(feat, torch.Tensor) else 256

        self.seg_head = SimpleSegHead(embed_dim, num_classes)

    def forward(self, x):
        enc_out = self.image_encoder(x)
        if isinstance(enc_out, dict):
            features = enc_out.get(
                "vision_features", enc_out.get("backbone_fpn", [None])[0]
            )
        else:
            features = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out

        logits = self.seg_head(features)
        if logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            logits = F.interpolate(
                logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_sam2_backbone(device: str = "cuda") -> nn.Module:
    """Build the SAM2.1 Hiera-L backbone and load its pretrained weights.

    This just gives SimpleSAM2Seg an image_encoder with the right shapes --
    its weights get overwritten right after by best_model.pth's fine-tuned
    state dict in sam_weights_loader.load_sam_model().
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(SAM2_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=SAM2_MODEL_CONFIG)
        OmegaConf.resolve(cfg)
        sam2_model = instantiate(cfg.model, _recursive_=True)

    checkpoint = torch.load(
        str(SAM2_BACKBONE_CHECKPOINT), map_location=device, weights_only=False
    )
    state_dict = checkpoint.get("model", checkpoint)
    sam2_model.load_state_dict(state_dict, strict=False)

    return sam2_model.to(device).eval()
