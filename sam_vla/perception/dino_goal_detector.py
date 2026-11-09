"""Open-vocabulary text -> bounding-box detector (Grounding DINO)

Gives sam_vla an open-vocabulary "find the object described by this text"
capability the rest of the repo doesn't have: goal_resolution/'s two
resolvers are either a fixed first-frame bbox (first_frame_resolver.py) or
SAM3+CLIP matching against a closed category list (goal_vocabulary_resolver.py)
-- neither takes a live, per-frame free-text query like "the big rock" the
way Grounding DINO does.

Loads IDEA-Research/grounding-dino-base from the local HF cache. Needs an env
whose transformers build has AutoModelForZeroShotObjectDetection -- this
repo's sam2/sam3 envs already carry a recent enough transformers for
Sam2Model (see box_prompted_sam.py), so Grounding DINO should load in the
same env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class DinoDetection:
    box: np.ndarray  # [x0, y0, x1, y1] pixels in the input image
    score: float
    label: str


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(device)
            .eval()
        )

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        # Grounding DINO expects lowercase phrases terminated by a period.
        text = text.strip().lower()
        if not text.endswith("."):
            text += "."
        return text

    @torch.no_grad()
    def detect(self, image: np.ndarray, text: str) -> List[DinoDetection]:
        """image: HxWx3 uint8 RGB. Returns detections sorted by score desc."""
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        prompt = self._normalize_prompt(text)
        inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        dets = [
            DinoDetection(
                box=np.asarray(b, dtype=np.float32), score=float(s), label=str(lbl)
            )
            for b, s, lbl in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results.get("text_labels", results.get("labels", [])),
            )
        ]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def detect_best(self, image: np.ndarray, text: str) -> Optional[DinoDetection]:
        dets = self.detect(image, text)
        return dets[0] if dets else None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Smoke-test GroundingDinoDetector against a single image file."
    )
    ap.add_argument("image_path")
    ap.add_argument("text", help='free-text target, e.g. "a big rock"')
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.image_path).convert("RGB"))
    detector = GroundingDinoDetector(device=args.device)
    best = detector.detect_best(rgb, args.text)
    if best is None:
        print(f"no detection for {args.text!r}")
    else:
        print(f"best: label={best.label!r} score={best.score:.3f} box={best.box}")
