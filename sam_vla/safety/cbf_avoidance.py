"""Cone-mode CBF obstacle avoidance, ported from rollout_navdp_policy.py's
cone-mode CBF+orbit+hard-gate block.

Operates on the single per-tick action NavdpPolicy.act_verbose() already returns
(after action-smoothing), not the raw diffusion chunk before smoothing: the
original's additional soft chunk-shaping (project_chunk_cone, run at replan time
on the full predicted horizon) needs direct access to that chunk, which
NavdpPolicy deliberately doesn't expose (see its module docstring). The per-tick
orbit controller and hard-gate backstop below are what actually keep the rover off
the obstacle; the chunk-shaping was a smoother multi-step anticipation on top.

Must be constructed AFTER a NavdpPolicy, which adds navdp_root to sys.path so
`navdp.extensions` can be imported.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sam_vla.core.types import Action


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _blocked(
    obstacle_point: Optional[np.ndarray],
    goal_bearing: Optional[float],
    r_gate: float,
    deadzone: float,
    orbit_hyst: float,
    side_committed: bool,
) -> tuple[bool, float, float]:
    """Is the obstacle within the avoidance shell AND actually between us and the
    goal? Hysteresis (orbit_hyst, only once a side is committed) on the perpendicular
    clearance so the orbit<->goal decision cannot rapid-toggle at the boundary."""
    if obstacle_point is None:
        return False, 0.0, 0.0
    ox, oy = float(obstacle_point[0]), float(obstacle_point[1])
    L = math.hypot(ox, oy)
    phi = math.atan2(oy, ox)
    beta = float(goal_bearing) if goal_bearing is not None else 0.0
    proj = ox * math.cos(beta) + oy * math.sin(beta)
    perp = math.sqrt(max(L * L - proj * proj, 0.0))
    avoiding = L < r_gate + deadzone
    thresh = r_gate + (orbit_hyst if side_committed else 0.0)
    return (avoiding and proj > 0.0 and perp < thresh), L, phi


class CbfObstacleAvoidance:
    """Stateful per-rollout cone-mode CBF: near an obstacle that BLOCKS the path to
    the goal, orbit around it (tangent pursuit + radial pull-back onto the d_safe
    circle, side committed until the obstacle clears the goal ray -- a smooth
    line-arc-line detour at constant cruise, no stop/rotate/go judder); a hard
    per-tick backstop brakes forward motion if the executed action would still
    breach the physical collision radius."""

    def __init__(
        self,
        d_safe: float = 0.75,
        gamma: float = 0.3,
        deadzone: float = 0.6,
        orbit_kr: float = 0.8,
        orbit_hyst: float = 0.4,
        pursuit_kp: float = 1.8,
        goaround_forward: float = 0.5,
        escape_yaw: bool = True,
        hard_gate: bool = True,
        robot_radius: float = 0.25,
        safety_margin: float = 0.15,
        obstacle_radius: float = 0.25,
        max_yaw_rate: float = 1.0,
    ):
        from navdp.extensions import (
            nearest_obstacle_point,
            project_forward_velocity_cbf,
        )

        self._nearest_obstacle_point = nearest_obstacle_point
        self._project_forward_velocity_cbf = project_forward_velocity_cbf

        self.d_safe = float(d_safe)
        self.gamma = float(gamma)
        self.deadzone = float(deadzone)
        self.orbit_kr = float(orbit_kr)
        self.orbit_hyst = float(orbit_hyst)
        self.pursuit_kp = float(pursuit_kp)
        self.goaround_forward = float(goaround_forward)
        self.escape_yaw = bool(escape_yaw)
        self.hard_gate = bool(hard_gate)
        # Physical collision radius (obstacle + rover + margin); the orbit hugs the
        # larger d_safe circle, this smaller radius is only the hard-breach backstop.
        self.r_cone = (
            float(robot_radius) + float(safety_margin) + float(obstacle_radius)
        )
        self.max_yaw_rate = float(max_yaw_rate)
        self._around_side: Optional[float] = (
            None  # committed orbit side (+1 left, -1 right)
        )

    def nearest_obstacle(
        self, obstacle_mask: np.ndarray, depth: np.ndarray, intrinsics: dict
    ) -> Optional[np.ndarray]:
        """Nearest point (body-frame [forward, left]) on the segmented obstacle mask,
        or None if the mask is empty / has no valid depth."""
        if int((np.asarray(obstacle_mask) > 0).sum()) == 0:
            return None
        return self._nearest_obstacle_point(obstacle_mask, depth, intrinsics)

    def is_blocked(
        self, obstacle_point: Optional[np.ndarray], goal_bearing: Optional[float]
    ) -> bool:
        """Pure read of the blocked state (no side effects) -- lets a caller decide
        whether to run some OTHER steering behavior (e.g. lost-goal heading assist)
        before calling apply(), which recomputes the same thing and then acts on it."""
        blocked, _, _ = _blocked(
            obstacle_point,
            goal_bearing,
            self.d_safe,
            self.deadzone,
            self.orbit_hyst,
            self._around_side is not None,
        )
        return blocked

    def apply(
        self,
        action: Action,
        obstacle_point: Optional[np.ndarray],
        goal_bearing: Optional[float],
    ) -> tuple[Action, dict]:
        info = {"blocked": False, "orbiting": False, "hard_gate_fired": False}
        r_gate = self.d_safe
        blocked, L, phi = _blocked(
            obstacle_point,
            goal_bearing,
            r_gate,
            self.deadzone,
            self.orbit_hyst,
            self._around_side is not None,
        )
        info["blocked"] = blocked

        out = action
        if blocked and self.escape_yaw:
            beta = float(goal_bearing) if goal_bearing is not None else 0.0
            if self._around_side is None:
                # Commit which way around: the tangent heading closest to the goal
                # bearing (least detour, natural return). Latched until released below.
                a = math.asin(min(1.0, r_gate / max(L, 1e-6)))
                dl = abs(_wrap_angle(phi + a - beta))
                dr = abs(_wrap_angle(phi - a - beta))
                self._around_side = 1.0 if dl <= dr else -1.0
            corr = max(-1.2, min(1.2, self.orbit_kr * (L - r_gate)))
            psi = _wrap_angle(phi + self._around_side * (0.5 * math.pi - corr))
            yaw_cmd = float(
                np.clip(self.pursuit_kp * psi, -self.max_yaw_rate, self.max_yaw_rate)
            )
            out = Action(v_fwd=self.goaround_forward, v_lat=0.0, yaw_rate=yaw_cmd)
            info["orbiting"] = True
        elif self._around_side is not None:
            self._around_side = (
                None  # obstacle no longer blocks the goal ray -> release the side
            )

        if self.hard_gate and obstacle_point is not None:
            p_fwd, p_lat = float(obstacle_point[0]), float(obstacle_point[1])
            # Release on LATERAL clearance, not distance: driving straight forward
            # MISSES the obstacle once its lateral offset exceeds the collision
            # radius (heading has turned enough); releasing on distance alone would
            # brake forever once the policy drives up close.
            cone_clears = (p_fwd <= 0.0) or (abs(p_lat) >= self.r_cone)
            if not (cone_clears and out.v_fwd > 0.0):
                a_arr, gated = self._project_forward_velocity_cbf(
                    np.asarray([out.v_fwd, out.v_lat, out.yaw_rate], dtype=np.float32),
                    obstacle_point,
                    np.zeros(2, dtype=np.float32),
                    d_safe=r_gate,
                    gamma=self.gamma,
                    deadzone=self.deadzone,
                    trust=None,
                )
                if gated:
                    out = Action(
                        v_fwd=float(a_arr[0]), v_lat=out.v_lat, yaw_rate=out.yaw_rate
                    )
                    info["hard_gate_fired"] = True

        return out, info
