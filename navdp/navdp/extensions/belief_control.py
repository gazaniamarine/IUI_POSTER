from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass
class EpistemicGateDecision:
    should_scan: bool
    sigma_epi: float
    active_goal: Optional[str]
    reason: str
    u_occ: float = 0.0
    speed_scale: float = 1.0


class EpistemicGate:
    """Information-seeking gate for stale/recoverable uncertainty.

    If epistemic uncertainty is high, the controller should rotate/scan to
    disocclude instead of committing to a long waypoint chunk.

    Two uncertainty sources feed the gate, and either can trigger a scan:
      * ``sigma_epi`` from the refined belief (goal memory is stale/recoverable);
      * ``u_occ`` from the occupancy foresight head (the geometry forecast is
        uncertain -- e.g. about to drive into a blind corner). ``u_occ`` is the
        occupancy analogue of ``sigma_aleatoric``; high u_occ also scales the
        commanded speed down (slow-down) rather than committing to a chunk.
    ``u_occ=None`` preserves the original belief-only behaviour exactly.
    """

    def __init__(
        self,
        sigma_epi_threshold: float = 0.75,
        u_occ_threshold: float = 0.20,
        u_occ_low: float = 0.10,
        u_occ_high: float = 0.40,
        slow_factor_min: float = 0.3,
    ):
        self.sigma_epi_threshold = float(sigma_epi_threshold)
        self.u_occ_threshold = float(u_occ_threshold)
        self.u_occ_low = float(u_occ_low)
        self.u_occ_high = float(u_occ_high)
        self.slow_factor_min = float(slow_factor_min)

    def __call__(
        self,
        refined_belief: np.ndarray,
        active_goal_index: int,
        goal_order: Optional[Sequence[str]] = None,
        u_occ: Optional[float] = None,
    ) -> EpistemicGateDecision:
        arr = np.asarray(refined_belief, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("refined_belief must have shape [N,F]")
        idx = int(np.clip(active_goal_index, 0, arr.shape[0] - 1))
        sigma_epi = float(arr[idx, 12]) if arr.shape[1] >= 13 else 0.0
        active_goal = goal_order[idx] if goal_order is not None and idx < len(goal_order) else None

        scan_belief = sigma_epi > self.sigma_epi_threshold
        u = float(u_occ) if u_occ is not None else 0.0
        scan_occ = (u_occ is not None) and (u > self.u_occ_threshold)
        should_scan = bool(scan_belief or scan_occ)
        speed_scale = (
            speed_scale_from_u_occ(u, self.u_occ_low, self.u_occ_high, self.slow_factor_min)
            if u_occ is not None
            else 1.0
        )

        if scan_belief and scan_occ:
            reason = "belief stale and occupancy forecast uncertain; scan to disocclude"
        elif scan_belief:
            reason = "epistemic uncertainty high; scan to disocclude"
        elif scan_occ:
            reason = "occupancy forecast uncertain (blind corner); slow down / scan"
        else:
            reason = "memory reliable enough to act"
        return EpistemicGateDecision(
            should_scan=should_scan,
            sigma_epi=sigma_epi,
            active_goal=active_goal,
            reason=reason,
            u_occ=u,
            speed_scale=speed_scale,
        )


def strength_from_sigma_ale(
    sigma_ale: float,
    sigma_low: float = 0.05,
    sigma_high: float = 1.0,
    strength_min: float = 0.05,
    strength_max: float = 0.65,
) -> float:
    """Map aleatoric uncertainty to an SDEdit strength in [strength_min, strength_max]."""
    denom = max(float(sigma_high) - float(sigma_low), 1e-6)
    u = (float(sigma_ale) - float(sigma_low)) / denom
    u = float(np.clip(u, 0.0, 1.0))
    return float(strength_min + u * (strength_max - strength_min))


def speed_scale_from_u_occ(
    u_occ: float,
    u_low: float = 0.5,
    u_high: float = 2.5,
    scale_min: float = 0.3,
) -> float:
    """Map occupancy-forecast uncertainty to a speed multiplier in [scale_min, 1].

    Low u_occ -> full speed; high u_occ -> slow down (don't commit into a region
    the forecast can't vouch for).
    """
    denom = max(float(u_high) - float(u_low), 1e-6)
    t = (float(u_occ) - float(u_low)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return float(1.0 - t * (1.0 - float(scale_min)))


def active_sigmas_from_refined_belief(
    refined_belief: np.ndarray,
    active_goal_index: int,
) -> tuple[float, float]:
    arr = np.asarray(refined_belief, dtype=np.float32)
    idx = int(np.clip(active_goal_index, 0, arr.shape[0] - 1))
    sigma_ale = float(arr[idx, 11]) if arr.shape[1] >= 12 else float(np.sqrt(max(arr[idx, 2], arr[idx, 4], 0.0)))
    sigma_epi = float(arr[idx, 12]) if arr.shape[1] >= 13 else 0.0
    return sigma_ale, sigma_epi


def build_warm_start_path(
    active_goal_mu: Sequence[float],
    horizon: int,
    action_dim: int = 3,
    max_step: float = 0.25,
    final_theta: float = 0.0,
) -> np.ndarray:
    """Build a simple served path from robot origin to active belief mean.

    Output:
        warm_x: [horizon, action_dim]

    The first two dimensions are local-frame x-forward/y-left waypoints. If
    action_dim >= 3, dimension 2 is heading.
    """
    mu = np.asarray(active_goal_mu, dtype=np.float32).reshape(-1)
    if mu.shape[0] < 2:
        raise ValueError("active_goal_mu must contain at least [x, y]")
    target = mu[:2]
    dist = float(np.linalg.norm(target))
    if dist > max_step * horizon:
        target = target / max(dist, 1e-6) * (max_step * horizon)
    path = np.zeros((int(horizon), int(action_dim)), dtype=np.float32)
    for i in range(horizon):
        frac = float(i + 1) / float(horizon)
        xy = frac * target
        path[i, 0:2] = xy
        if action_dim >= 3:
            path[i, 2] = final_theta
    return path


def refine_bank_with_model(
    relational_model,
    bank_tensor,
    active_goal_index: Optional[int] = None,
):
    """Small convenience wrapper for act-time code that may use torch."""
    import torch

    x = bank_tensor if torch.is_tensor(bank_tensor) else torch.as_tensor(bank_tensor, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    active = None
    if active_goal_index is not None:
        active = torch.tensor([active_goal_index], device=x.device)
    with torch.no_grad():
        return relational_model(x, active_goal_index=active)

