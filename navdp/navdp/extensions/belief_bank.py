from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


@dataclass
class BeliefSlot:
    """Gaussian belief for one named route subgoal.

    Shapes:
        mu: [dim]
        Sigma: [dim, dim]
    """

    goal_id: str
    mu: np.ndarray
    Sigma: np.ndarray
    visible: bool
    initialized: bool
    last_seen_step: int
    time_since_seen: int
    confidence: float


class SubgoalBeliefBank:
    """Persistent Gaussian belief bank for all named route subgoals."""

    def __init__(
        self,
        goal_ids: Iterable[str],
        dim: int = 2,
        sigma_init: float = 1.0,
        sigma_visible: float = 0.05,
        odom_noise: float = 0.02,
        decay_factor: float = 0.95,
        large_uncertainty: float = 1_000.0,
    ):
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.goal_ids = list(dict.fromkeys(goal_ids))
        self.dim = int(dim)
        self.sigma_init = float(sigma_init)
        self.sigma_visible = float(sigma_visible)
        self.odom_noise = float(odom_noise)
        self.decay_factor = float(decay_factor)
        self.large_uncertainty = float(large_uncertainty)
        self.slots: MutableMapping[str, BeliefSlot] = {}
        self.reset()

    def reset(self) -> None:
        self.slots = {gid: self._new_slot(gid) for gid in self.goal_ids}

    def _new_slot(self, goal_id: str) -> BeliefSlot:
        return BeliefSlot(
            goal_id=goal_id,
            mu=np.zeros(self.dim, dtype=np.float32),
            Sigma=np.eye(self.dim, dtype=np.float32) * self.large_uncertainty,
            visible=False,
            initialized=False,
            last_seen_step=-1,
            time_since_seen=0,
            confidence=0.0,
        )

    def __contains__(self, goal_id: str) -> bool:
        return goal_id in self.slots

    def __getitem__(self, goal_id: str) -> BeliefSlot:
        return self.get(goal_id)

    def get(self, goal_id: str) -> BeliefSlot:
        if goal_id not in self.slots:
            raise KeyError(f"unknown goal id: {goal_id}")
        return self.slots[goal_id]

    def update(
        self,
        observations: Mapping[str, Mapping[str, object]],
        odom_delta: Sequence[float],
        step: int,
    ) -> Dict[str, BeliefSlot]:
        """Update all slots from current observations and ego-motion.

        Observation format per goal:
            {
                "visible": bool,
                "position": np.ndarray [dim] or [2],
                "confidence": float,
            }

        odom_delta is [dx, dy, dtheta], the robot motion from the previous
        local frame into the current one. Occluded target coordinates are
        transformed into the new local frame with the SE(2) inverse transform:
            p_new = R(-dtheta) @ (p_old - [dx, dy])
        """
        for goal_id, obs in observations.items():
            if goal_id not in self.slots:
                self.goal_ids.append(goal_id)
                self.slots[goal_id] = self._new_slot(goal_id)

        for goal_id in self.goal_ids:
            slot = self.slots[goal_id]
            obs = observations.get(goal_id, {})
            visible = bool(obs.get("visible", False))
            pos = obs.get("position", None)
            conf = float(obs.get("confidence", 0.0))

            if visible and pos is not None and _is_valid_position(pos, self.dim):
                measurement = np.asarray(pos, dtype=np.float32).reshape(-1)
                slot.mu = np.zeros(self.dim, dtype=np.float32)
                slot.mu[: min(self.dim, measurement.shape[0])] = measurement[: self.dim]
                slot.Sigma = np.eye(self.dim, dtype=np.float32) * self.sigma_visible
                slot.visible = True
                slot.initialized = True
                slot.last_seen_step = int(step)
                slot.time_since_seen = 0
                slot.confidence = float(np.clip(conf, 0.0, 1.0))
            elif slot.initialized:
                slot.mu = ego_motion_update(slot.mu, odom_delta)
                slot.Sigma = (
                    slot.Sigma + np.eye(self.dim, dtype=np.float32) * self.odom_noise
                )
                slot.visible = False
                slot.time_since_seen += 1
                slot.confidence = float(
                    np.clip(slot.confidence * self.decay_factor, 0.0, 1.0)
                )
            else:
                slot.mu = np.zeros(self.dim, dtype=np.float32)
                slot.Sigma = np.eye(self.dim, dtype=np.float32) * self.large_uncertainty
                slot.visible = False
                slot.initialized = False
                slot.time_since_seen = 0
                slot.confidence = 0.0
        return dict(self.slots)

    def as_tensor(
        self,
        goal_order: Sequence[str],
        active_goal_id: Optional[str] = None,
        route_index: int = 0,
        route_length: Optional[int] = None,
        device: Optional[object] = None,
        dtype: Optional[object] = None,
    ):
        """Return slot features as [N_goals, 11].

        Feature layout:
            [mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
             visible, initialized, time_since_seen, confidence,
             is_active, route_index_normalized]
        """
        denom = max(int(route_length or len(goal_order)) - 1, 1)
        route_norm = float(route_index) / float(denom)
        rows = []
        for goal_id in goal_order:
            slot = self.get(goal_id)
            Sigma = slot.Sigma
            rows.append(
                [
                    float(slot.mu[0]),
                    float(slot.mu[1]),
                    float(Sigma[0, 0]),
                    float(Sigma[0, 1]),
                    float(Sigma[1, 1]),
                    float(slot.visible),
                    float(slot.initialized),
                    float(slot.time_since_seen),
                    float(slot.confidence),
                    float(active_goal_id is not None and goal_id == active_goal_id),
                    route_norm,
                ]
            )
        import torch

        return torch.tensor(rows, dtype=dtype or torch.float32, device=device)

    def as_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            goal_id: {
                "goal_id": slot.goal_id,
                "mu": slot.mu.copy(),
                "Sigma": slot.Sigma.copy(),
                "visible": slot.visible,
                "initialized": slot.initialized,
                "last_seen_step": slot.last_seen_step,
                "time_since_seen": slot.time_since_seen,
                "confidence": slot.confidence,
            }
            for goal_id, slot in self.slots.items()
        }


def ego_motion_update(mu: np.ndarray, odom_delta: Sequence[float]) -> np.ndarray:
    """Transform a local-frame target coordinate into the new robot frame."""
    out = np.asarray(mu, dtype=np.float32).copy()
    dx, dy, dtheta = _parse_odom_delta(odom_delta)
    c = float(np.cos(-dtheta))
    s = float(np.sin(-dtheta))
    p = out[:2] - np.asarray([dx, dy], dtype=np.float32)
    out[:2] = np.asarray([c * p[0] - s * p[1], s * p[0] + c * p[1]], dtype=np.float32)
    if out.shape[0] >= 3:
        out[2] = out[2] - dtheta
    return out


def _parse_odom_delta(odom_delta: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(odom_delta, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError("odom_delta must contain at least dx and dy")
    dx = float(arr[0])
    dy = float(arr[1])
    dtheta = float(arr[2]) if arr.shape[0] >= 3 else 0.0
    return dx, dy, dtheta


def _is_valid_position(pos: object, dim: int) -> bool:
    try:
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    return (
        arr.shape[0] >= min(dim, 2) and np.isfinite(arr[: min(dim, arr.shape[0])]).all()
    )
