"""
CLIP dual-role classifier for the multi-goal path (see next.md and the
approved plan at /home/gpu/.claude/plans/formulate-the-plan-next-md-stateless-thimble.md).

One `open_clip` model instance serves two jobs, per mask returned by
Sam3GoalTracker.resegment:
  - goal-worthiness: `classify(crop)` scores a mask crop against the goal
    vocabulary's text embeddings (best match above `--clip-goal-thresh`,
    left to the caller to threshold).
  - cross-frame re-identification: `match_or_mint(...)` compares a new mask
    crop's image embedding against every already-tracked goal's stored
    embedding, so the same physical object isn't re-minted as a new goal
    every re-segmentation pass.

Runs in the `sam3` conda env alongside the SAM3 predictor (open_clip_torch
is installed there already), not a separate process/env.
"""

import uuid

import numpy as np
import open_clip
import torch
from PIL import Image

from sam_vla.core.types import TrackedGoal


class ClipGoalClassifier:
    """Loads open_clip once; classify() scores a crop, match_or_mint() re-IDs it."""

    def __init__(
        self,
        goal_vocabulary: list[str],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ):
        if not goal_vocabulary:
            raise ValueError("goal_vocabulary must be non-empty")

        self.goal_vocabulary = list(goal_vocabulary)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            tokens = tokenizer(self.goal_vocabulary).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_bank = text_features

    def classify(self, crop: np.ndarray) -> tuple[str, float, np.ndarray]:
        """crop: (H, W, 3) uint8 RGB mask crop.

        Returns (category, score, embedding) — category is the best-matching
        goal_vocabulary term, score its cosine similarity, embedding the
        crop's raw CLIP image embedding (stored on TrackedGoal for later
        match_or_mint re-ID calls).
        """
        image_input = (
            self.preprocess(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ self.text_bank.T).squeeze(0)
        best_idx = int(torch.argmax(sims).item())
        category = self.goal_vocabulary[best_idx]
        score = float(sims[best_idx].item())
        embedding = image_features.squeeze(0).cpu().numpy()
        return category, score, embedding

    def match_or_mint(
        self,
        embedding: np.ndarray,
        tracked_goals: dict[str, TrackedGoal],
        reid_thresh: float,
    ) -> str:
        """Best cosine match above reid_thresh reuses that goal_id;
        otherwise mints and returns a new one (caller mints the TrackedGoal
        and appends it to the route only if the returned id isn't already
        in tracked_goals)."""
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_goal_id = None
        best_sim = -1.0
        for goal_id, goal in tracked_goals.items():
            other_norm = goal.clip_embedding / (
                np.linalg.norm(goal.clip_embedding) + 1e-8
            )
            sim = float(np.dot(emb_norm, other_norm))
            if sim > best_sim:
                best_sim = sim
                best_goal_id = goal_id

        if best_goal_id is not None and best_sim >= reid_thresh:
            return best_goal_id
        return f"goal_{uuid.uuid4().hex[:8]}"
