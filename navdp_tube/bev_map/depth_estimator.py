"""Depth Anything V2 (metric, indoor) wrapper. Returns metric depth at input resolution."""

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


class DepthAnything:
    def __init__(self, model_id=DEFAULT_MODEL, device=None, fp16=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        print(f"[Depth] loading {model_id} on {self.device} ({self.dtype})")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(
            self.device, self.dtype).eval()

    @torch.inference_mode()
    def __call__(self, rgb_frames):
        """List of HxWx3 uint8 RGB frames -> list of HxW float32 depth maps (metres)."""
        shapes = [f.shape[:2] for f in rgb_frames]
        inputs = self.processor(images=list(rgb_frames), return_tensors="pt").to(self.device, self.dtype)
        pred = self.model(**inputs).predicted_depth.float()          # (B, h', w')
        out = []
        for p, (h, w) in zip(pred, shapes):
            p = torch.nn.functional.interpolate(p[None, None], size=(h, w), mode="bilinear",
                                                align_corners=False)
            out.append(p[0, 0].cpu().numpy().astype(np.float32))
        return out
