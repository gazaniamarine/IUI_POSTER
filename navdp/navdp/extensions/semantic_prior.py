"""semantic_prior.py - HeckNav-style semantic prior for the metric belief.

Idea (from HeckNav's Bayesian knowledge graph): before the goal is ever seen, the
objects you CAN see ("anchors") carry statistical evidence about where the goal is
(a stove implies a kitchen implies a fridge nearby). HeckNav does this over discrete
ROOM categories; here we spatialize it: anchors co-occurring with the goal pull a
broad Gaussian PRIOR toward their metric positions. That prior seeds the existing
SubgoalBeliefBank Gaussian so the belief is informative *before* the first sighting
(renderable as an exploration ghost), and collapses to the tight metric posterior
the instant the goal is actually seen.

This is complementary to the metric belief, not a replacement:
  - SemanticPrior  : where to LOOK for an unseen goal (semantic, broad, long-horizon)
  - SubgoalBeliefBank: where the goal WAS (metric, tight, after a sighting)

The affinity table is P(goal near anchor) weights. Mine it once, offline, from
scene statistics (see scripts/eval_semantic_prior.py --mine) or supply your own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SemanticPrediction:
    mu: np.ndarray        # (2,) predicted goal position, body frame (same convention as the belief)
    Sigma: np.ndarray     # (2, 2) covariance (bigger = less certain)
    confidence: float     # 0..1 total semantic evidence
    n_anchors: int        # how many anchors contributed

    @property
    def bearing(self) -> float:
        return float(np.arctan2(self.mu[1], self.mu[0]))

    @property
    def range(self) -> float:
        return float(np.hypot(self.mu[0], self.mu[1]))


class AffinityTable:
    """Weights ``w[(anchor_cat -> goal_cat)]`` = strength of "goal is near this anchor".

    Weight >= 0. Higher = the goal tends to sit near objects of that anchor category.
    default is used for unknown pairs (1.0 => a neutral "goal near any object" prior).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, default: float = 1.0):
        self.weights: Dict[str, float] = dict(weights or {})
        self.default = float(default)

    @staticmethod
    def _key(anchor_cat: str, goal_cat: str) -> str:
        return f"{anchor_cat}->{goal_cat}"

    def get(self, anchor_cat: str, goal_cat: str) -> float:
        return float(self.weights.get(self._key(anchor_cat, goal_cat), self.default))

    def set(self, anchor_cat: str, goal_cat: str, w: float) -> None:
        self.weights[self._key(anchor_cat, goal_cat)] = float(w)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"default": self.default, "weights": self.weights}, indent=2))

    @classmethod
    def load(cls, path) -> "AffinityTable":
        d = json.loads(Path(path).read_text())
        return cls(d.get("weights"), float(d.get("default", 1.0)))


class SemanticPrior:
    """Turn observed anchors into a Gaussian goal-position prior in the body frame."""

    def __init__(self, table: AffinityTable, base_sigma: float = 2.0, min_weight: float = 1e-3):
        self.table = table
        self.base_var = float(base_sigma) ** 2   # floor on prior uncertainty (m^2)
        self.min_weight = float(min_weight)

    def predict(
        self,
        goal_cat: str,
        anchors: Sequence[Tuple[str, Sequence[float]]],
    ) -> Optional[SemanticPrediction]:
        """anchors: [(category, [x, y] in body frame), ...] -> Gaussian prior, or None.

        The prior mean is the affinity-weighted centroid of the anchors (the goal is
        pulled toward objects it co-occurs with); the covariance grows when the anchors
        disagree (ambiguous evidence). Confidence saturates with total evidence."""
        pts, ws = [], []
        for cat, pos in anchors:
            p = np.asarray(pos, dtype=np.float32).reshape(-1)
            if p.shape[0] < 2 or not np.all(np.isfinite(p[:2])):
                continue
            w = self.table.get(str(cat), str(goal_cat))
            if w <= self.min_weight:
                continue
            pts.append(p[:2])
            ws.append(w)
        if not pts:
            return None
        pts = np.stack(pts).astype(np.float32)
        ws = np.asarray(ws, dtype=np.float32)
        wsum = float(ws.sum())
        mu = (ws[:, None] * pts).sum(axis=0) / wsum
        d = pts - mu[None]
        spread = (ws[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(axis=0) / wsum
        Sigma = spread.astype(np.float32) + np.eye(2, dtype=np.float32) * self.base_var
        conf = float(1.0 - np.exp(-wsum))   # 0 anchors->0, saturates toward 1
        return SemanticPrediction(mu.astype(np.float32), Sigma, conf, len(pts))


def seed_belief_bank(
    bank,
    goal_id: str,
    pred: Optional[SemanticPrediction],
    min_conf: float = 0.05,
    only_before_first_sighting: bool = True,
) -> bool:
    """Inject the semantic prior into the bank as a BROAD, non-visible belief.

    By default only seeds a goal that has NEVER been seen (so a real metric sighting
    is never clobbered by the coarser semantic guess). Returns True if it seeded."""
    if pred is None or pred.confidence < min_conf or goal_id not in bank:
        return False
    slot = bank.get(goal_id)
    if only_before_first_sighting and slot.initialized:
        return False
    mu = np.zeros(bank.dim, dtype=np.float32)
    mu[:2] = pred.mu[:2]
    Sigma = np.eye(bank.dim, dtype=np.float32) * bank.large_uncertainty
    Sigma[:2, :2] = pred.Sigma
    slot.mu = mu
    slot.Sigma = Sigma
    slot.initialized = True
    slot.visible = False
    slot.confidence = float(pred.confidence)
    return True
