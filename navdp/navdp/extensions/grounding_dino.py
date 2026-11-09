"""
GroundingDINODetector: open-vocabulary object detector used to CONFIRM the
cosmetic flag mesh (spawn_flag_mesh in rollout_navdp_policy.py) is actually
visually resolved in the RGB frame, rather than trusting the known world
point + geometric frustum/occlusion check alone.

Usage pattern (search rollout, per tick):
    if real_goal_info["visible"] and real_goal_info["range"] <= dino_max_range:
        det = dino.detect(rgb, text_prompt="a flag")
    else:
        det = None
    found = det is not None and det.confirms(real_goal_info["u"], real_goal_info["v"])

Kept as a CONFIRMATION step (not a replacement) for the frustum/occlusion/range
math already computed by project_goal_mask -- that geometry is free and exact;
DINO only needs to answer "is the flag actually resolved in pixels right now",
which the geometric check cannot know.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    u: float            # box centroid column (x), pixels
    v: float             # box centroid row (y), pixels
    score: float
    box: tuple           # (x0, y0, x1, y1) in pixels
    label: str

    def confirms(self, ref_u: float, ref_v: float, pixel_tol: float = 60.0) -> bool:
        """True if this detection's centroid is within pixel_tol of a reference
        pixel (e.g. the geometric projection of the known world point). Guards
        against a DINO false-positive elsewhere in frame (a rock, terrain
        texture, etc.) being accepted as "the flag" just because SOME box with
        the right label showed up somewhere in the image."""
        return float(np.hypot(self.u - ref_u, self.v - ref_v)) <= float(pixel_tol)


class GroundingDINODetector:
    """Frozen, zero-shot open-vocabulary detector (HF `transformers`
    AutoModelForZeroShotObjectDetection + grounding-dino-tiny by default).
    Inference only, lazy-loaded on first call -- same pattern as QwenExplorer.

    Install: `pip install transformers --break-system-packages` is sufficient;
    grounding-dino-tiny ships as a plain HF checkpoint (no separate weights/
    config repo, unlike the original IDEA-Research/GroundingDINO codebase).
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.model_id = model_id
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device).eval()

    def detect(self, rgb: np.ndarray, text_prompt: str = "a flag.") -> List[Detection]:
        """Returns all boxes above threshold, sorted by score descending.
        `text_prompt` should be lowercase and end with a period (GroundingDINO
        convention); multiple classes can be queried by chaining "cls1. cls2.".
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        prompt = text_prompt if text_prompt.strip().endswith(".") else text_prompt.strip() + "."
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        h, w = rgb.shape[:2]
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                target_sizes=[(h, w)],
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

        dets: List[Detection] = []
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("text_labels", results.get("labels", []))
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = [float(v) for v in box]
            dets.append(Detection(
                u=(x0 + x1) / 2.0, v=(y0 + y1) / 2.0,
                score=float(score), box=(x0, y0, x1, y1), label=str(label),
            ))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def detect_best(self, rgb: np.ndarray, text_prompt: str = "a flag.") -> Optional[Detection]:
        dets = self.detect(rgb, text_prompt=text_prompt)
        return dets[0] if dets else None
