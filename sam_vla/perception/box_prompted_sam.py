"""SAM 2.1 box-prompted segmentation: turns a Grounding-DINO bbox into an
instance mask, so a text-goal detection gets a clean object depth (median
over mask pixels, not a bbox interior that includes background) and a real
pixel footprint for BeliefGoalTracker.observe() / the obstacle guard's
exclude_mask.

Ported from a teammate's Nav_new/MARS DINO+NavDP navigation stack
(github.com/priyan212/Nav_new/tree/master/MARS, nav_pipeline/sam_segmenter.py)
against this repo's own perception/ conventions -- a new module rather than
importing that repo, so this stays self-contained. See dino_goal_detector.py's
docstring for why this open-vocabulary pairing doesn't already exist here:
neither perception/sam_segmenter.py (automatic, prompt-free mask generation
for the annotation-dataset pipeline) nor sam_weights_loader.py (this
project's own SimpleSAM2Seg LoRA checkpoint, closed-vocabulary) can turn an
arbitrary bounding box into an instance mask on demand.

Loads facebook/sam2.1-hiera-small (~39M params) from the local HF cache.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor


class Sam2BoxSegmenter:
    def __init__(
        self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda:0"
    ):
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def segment_box(self, rgb: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """rgb: HxWx3 uint8; box: [x0,y0,x1,y1] pixels -> bool mask HxW, or
        None if SAM returned an empty mask."""
        inputs = self.processor(
            images=rgb,
            input_boxes=[[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]  # (num_objects, num_masks, H, W)
        mask = masks[0, 0].numpy().astype(bool)
        if mask.sum() == 0:
            return None
        return mask


if __name__ == "__main__":
    import argparse

    from PIL import Image

    ap = argparse.ArgumentParser(
        description="Smoke-test Sam2BoxSegmenter against a single image + box."
    )
    ap.add_argument("image_path")
    ap.add_argument("box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.image_path).convert("RGB"))
    segmenter = Sam2BoxSegmenter(device=args.device)
    mask = segmenter.segment_box(rgb, np.asarray(args.box, dtype=np.float32))
    if mask is None:
        print("empty mask")
    else:
        print(f"mask: {mask.sum()} px of {mask.size} ({100.0 * mask.mean():.1f}%)")
