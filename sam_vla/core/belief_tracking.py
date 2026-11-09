"""Body-frame belief tracking for a goal whose live semantic mask (MESH_GOAL_ID)
may drop out of view, ported from rollout_navdp_policy.py's mesh_tracking_mode /
--belief-goal machinery: re-seed a body-frame [forward, left] estimate from the
rendered mask whenever it's visible enough, and dead-reckon it by the robot's own
executed motion between sightings so a brief occlusion doesn't lose the goal.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sam_vla.core.goal_geometry import intrinsics_from_hfov
from sam_vla.core.types import Action


def mask_to_body(
    mask: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
    hfov_deg: float,
    fallback_range: float,
    min_px: int = 1,
) -> Optional[np.ndarray]:
    """Body-frame goal point [forward, left] from a rendered mask: bearing from the
    mask's centroid column, range from the MEDIAN depth over all mask pixels -- robust
    to a single mask pixel landing on a depth discontinuity at the object's silhouette
    edge, which would otherwise seed a badly wrong range that dead-reckons uncorrected.
    """
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size < min_px:
        return None
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    patch = np.asarray(depth)[ys, xs]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    rng = float(np.median(valid)) if valid.size > 0 else float(fallback_range)
    u = float(xs.mean())
    right = (u - intr["cx"]) * rng / max(intr["fx"], 1e-6)
    return np.asarray([rng, -right], dtype=np.float32)  # [forward, left]


def propagate_body_point(
    bg: np.ndarray,
    v_fwd: float,
    v_left: float,
    yaw_rate: float,
    dt: float,
    odom_noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Move a body-frame point [forward, left] under the robot's own SE(2) motion
    (dead-reckoning): translate back by v*dt and rotate by -yaw_rate*dt -- the point
    stays fixed in the world while the robot moves out from under it. Optional
    Gaussian odom noise makes the belief drift, so it must be corrected by sightings
    to stay good."""
    if odom_noise > 0.0 and rng is not None:
        v_fwd = v_fwd + float(rng.normal(0.0, odom_noise))
        yaw_rate = yaw_rate + float(rng.normal(0.0, odom_noise))
    th = -float(yaw_rate) * float(dt)
    c, s = math.cos(th), math.sin(th)
    qx = float(bg[0]) - float(v_fwd) * float(dt)
    qy = float(bg[1]) - float(v_left) * float(dt)
    return np.asarray([c * qx - s * qy, s * qx + c * qy], dtype=np.float32)


DEFAULT_SIGMA_VISIBLE = 0.05
"""Uncertainty floor a fresh sighting resets to -- mirrors SubgoalBeliefBank's
sigma_visible semantics (navdp/navdp/extensions/belief_bank.py) conceptually,
without importing it."""


def uncertainty_growth_increment(
    odom_noise: float, odom_noise_growth_rate: float, time_since_seen: float
) -> float:
    """Accelerating-drift increment added to a scalar uncertainty state for one
    step while its target is unseen: a base per-step growth (`odom_noise`)
    scaled up the longer it's been since the last sighting
    (`odom_noise_growth_rate`). Shared standalone so callers with no full
    `BeliefGoalTracker` instance -- e.g. kb_teleop_vl.py's synthetic
    covariance proxy -- exercise the exact same formula rather than a
    divergent reimplementation."""
    return float(odom_noise) * (
        1.0 + float(odom_noise_growth_rate) * float(time_since_seen)
    )


class BeliefGoalTracker:
    """Tracks a body-frame [forward, left] estimate of the goal position. Re-seeds
    from the live rendered goal mask whenever it's visible (>= min_px pixels);
    dead-reckons by the robot's executed motion the rest of the time.

    Also tracks a scalar `uncertainty` state: it resets to `sigma_visible` on
    every fresh sighting and grows each step the goal stays unseen, at a rate
    of `odom_noise * (1 + odom_noise_growth_rate * time_since_seen)` -- an
    accelerating-drift term so a long occlusion is penalized more than a brief
    one. Numpy-only, no `navdp.extensions`/torch dependency, so this stays
    runnable in the `habitat` conda env."""

    def __init__(
        self,
        hfov_deg: float,
        goal_range: float = 8.0,
        min_px: int = 10,
        odom_noise: float = 0.0,
        odom_noise_growth_rate: float = 0.0,
        sigma_visible: float = DEFAULT_SIGMA_VISIBLE,
        seed: int = 0,
    ):
        self.hfov_deg = float(hfov_deg)
        self.goal_range = float(goal_range)
        self.min_px = int(min_px)
        self.odom_noise = float(odom_noise)
        self.odom_noise_growth_rate = float(odom_noise_growth_rate)
        self.sigma_visible = float(sigma_visible)
        self._rng = np.random.default_rng(seed)
        self.belief_g: Optional[np.ndarray] = None
        self.uncertainty = self.sigma_visible
        self._time_since_seen = 0.0

    def observe(self, goal_mask: np.ndarray, depth: np.ndarray) -> bool:
        """Re-seed from the live mask if it's visible enough; returns whether it was."""
        height, width = np.asarray(depth).shape[:2]
        if int((np.asarray(goal_mask) > 0).sum()) < self.min_px:
            return False
        seed = mask_to_body(
            goal_mask, depth, height, width, self.hfov_deg, self.goal_range, self.min_px
        )
        if seed is not None:
            self.belief_g = seed
            self.uncertainty = self.sigma_visible
            self._time_since_seen = 0.0
        return seed is not None

    def observe_body_point(self, forward: float, left: float) -> None:
        """Re-seed belief from a known body-frame point directly, bypassing
        mask/depth -- for callers with ground-truth goal position instead of
        a rendered mask (e.g. kb_teleop_vl.py's --goal-x/--goal-z, which has
        no semantic renderer to source a mask from). Same reset semantics as
        a successful observe(): floor uncertainty, zero time_since_seen."""
        self.belief_g = np.asarray([forward, left], dtype=np.float32)
        self.uncertainty = self.sigma_visible
        self._time_since_seen = 0.0

    def propagate(self, action: Action, dt: float) -> None:
        """Dead-reckon the belief by the just-executed action. Action.v_lat is
        RIGHTWARD-positive (sam_vla's pose_integrator convention, see its docstring)
        while propagate_body_point wants the LEFTWARD component -- hence the sign flip.
        Also grows `uncertainty` for the elapsed dt (called every step regardless
        of whether observe() just reset it, so a fresh-sighting step accrues one
        step's worth of growth on top of the floor -- negligible in practice).
        """
        if self.belief_g is None:
            return
        self.belief_g = propagate_body_point(
            self.belief_g,
            v_fwd=action.v_fwd,
            v_left=-action.v_lat,
            yaw_rate=action.yaw_rate,
            dt=dt,
            odom_noise=self.odom_noise,
            rng=self._rng,
        )
        self._time_since_seen += float(dt)
        self.uncertainty += uncertainty_growth_increment(
            self.odom_noise, self.odom_noise_growth_rate, self._time_since_seen
        )

    def uncertainty_value(self) -> float:
        return self.uncertainty

    def bearing(self) -> Optional[float]:
        if self.belief_g is None:
            return None
        return math.atan2(float(self.belief_g[1]), float(self.belief_g[0]))

    def distance(self) -> Optional[float]:
        if self.belief_g is None:
            return None
        return float(math.hypot(float(self.belief_g[0]), float(self.belief_g[1])))


def lost_goal_heading_assist(
    action: Action,
    bearing: float,
    goal_lost: bool,
    turn_kp: float,
    forward_floor: float,
    bearing_deg_thresh: float,
    max_yaw_rate: float,
) -> Action:
    """Proportional heading override toward the tracked goal bearing when the goal
    has drifted off-centre or fully out of view: the mask-conditioned policy's own
    yaw response saturates weakly once the goal nears the frame edge, so it can't
    tell "just off-screen" from "behind me" without this. Off to the side (still
    ahead): keep the policy's forward speed and only steer harder. Fully behind
    (goal_lost): pivot with the forward floor."""
    goal_offcentre = bearing_deg_thresh > 0.0 and abs(bearing) > math.radians(
        bearing_deg_thresh
    )
    if not (goal_lost or goal_offcentre):
        return action
    yaw_cmd = float(np.clip(turn_kp * bearing, -max_yaw_rate, max_yaw_rate))
    fwd = forward_floor if goal_lost else action.v_fwd
    fwd = max(fwd, forward_floor)
    return Action(v_fwd=fwd, v_lat=0.0, yaw_rate=yaw_cmd)
