"""Shared bootstrap + tiny control-law helpers for the belief_exp harness.

This file imports the REAL belief classes from navdp/ (never copies their logic).
The only things defined locally are generic robot-control math that has nothing to
do with belief tracking itself (a bearing-following P-controller and a noise-free
SE(2) ego-motion integrator for the ground-truth goal) -- matching the existing
navdp/scripts/*.py convention of keeping this tiny controller as a local copy
in every file that needs it, rather than inventing a shared import path for it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

NAVDP_ROOT = Path(__file__).resolve().parents[1] / "navdp"
if str(NAVDP_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVDP_ROOT))

from navdp.extensions import (  # noqa: E402
    RouteManager,
    SubgoalBeliefBank,
    strength_from_sigma_ale,
)

GOAL_ID = "target"


def p_controller(
    mu: np.ndarray,
    turn_kp: float = 1.4,
    base_forward: float = 0.5,
    max_yaw_rate: float = 1.0,
) -> Tuple[float, float, float]:
    """Bearing-proportional steering law.

    Same recipe as gen_belief_propagation_data.py / train_belief_only_policy.py /
    rollout_habitat_policy.py's local p_controller() -- yaw proportional to bearing,
    forward speed tapering as the goal swings abeam. Returns (v_fwd, v_lat, yaw_rate).
    """
    bearing = math.atan2(float(mu[1]), float(mu[0]))
    yaw = float(np.clip(turn_kp * bearing, -max_yaw_rate, max_yaw_rate))
    fwd = float(base_forward * max(0.0, math.cos(bearing)))
    return fwd, 0.0, yaw


def ego_motion_true(
    point: np.ndarray, v_fwd: float, v_lat: float, yaw_rate: float, dt: float
) -> np.ndarray:
    """Noise-free SE(2) transform of a body-frame point under one step of robot motion.

    Same math as SubgoalBeliefBank.ego_motion_update (belief_bank.py), applied here only
    to advance the GROUND-TRUTH goal position -- the belief bank itself is never touched
    by this function, it only ever sees the (possibly noisy) odom_delta passed to update().
    """
    dx, dy, dtheta = v_fwd * dt, v_lat * dt, yaw_rate * dt
    c, s = math.cos(-dtheta), math.sin(-dtheta)
    p = np.asarray(point, dtype=np.float32)[:2] - np.asarray([dx, dy], dtype=np.float32)
    return np.asarray([c * p[0] - s * p[1], s * p[0] + c * p[1]], dtype=np.float32)


def sigma_ale_from_bank(bank: SubgoalBeliefBank, goal_id: str = GOAL_ID) -> float:
    """sqrt(max(Sigma_xx, Sigma_yy)) for one slot -- the aleatoric-uncertainty fallback
    formula already documented in navdp/navdp/extensions/belief_control.py for when no
    RelationalBelief refinement is available (this harness is Kalman-bank-only)."""
    slot = bank.get(goal_id)
    return float(np.sqrt(max(float(slot.Sigma[0, 0]), float(slot.Sigma[1, 1]), 0.0)))


__all__ = [
    "GOAL_ID",
    "RouteManager",
    "SubgoalBeliefBank",
    "ego_motion_true",
    "p_controller",
    "sigma_ale_from_bank",
    "strength_from_sigma_ale",
]
