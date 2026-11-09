"""Depth-based local obstacle guard: a hard, model-agnostic avoidance layer
that runs directly on the depth image, independent of whatever policy is
driving this tick -- ported from a teammate's Nav_new/MARS DINO+NavDP stack
(github.com/priyan212/Nav_new/tree/master/MARS, nav_pipeline/obstacle_guard.py)
against this repo's own conventions (core.goal_geometry.intrinsics_from_hfov,
core.types.Action) rather than importing that repo directly.

Distinct from safety/cbf_avoidance.py's CBF+orbit controller, which
continuously reshapes the outgoing action every tick based on a nearest-
obstacle vector: this is a *discrete* hard veto matching
run_navdp_dino_rollout.py's SEARCH/TRACK/AVOID/STOP state machine -- below
`hard_stop_dist` it takes over entirely (AVOID), above it `min_forward` is
just informational. Only upstream MARS's swept-hull trajectory clearance
(swept_clearance, vetoing individual sampled NavDP trajectories) is dropped
here: this repo's NavdpUpstreamPolicy only ever returns ONE already-server-
selected trajectory (see navdp_upstream_client.py's docstring), not the full
sampled candidate set a clearance veto would choose among.

depth_to_obstacle_points uses a LOCAL-SLOPE ground test: each depth column is
walked near -> far, and a point counts as ground if its rise since the last
confirmed ground point is within `max_climb_deg` over the distance traveled
since then -- not a flat plane through the rover's own footing, and not a
fixed-origin tolerance that widens with range. A flat plane reads a rising
slope/hill as a wall (the true ground climbs above it); a fixed-origin
tolerance has to be as large as a real rock's height by 1-2m range just to
admit a climbable slope, which makes it blind to obstacles at exactly the
range the thresholds below care about. Comparing against the local,
walked-forward ground profile means a slope of any total height stays
"ground" while a rock/step's abrupt LOCAL rise over a short run is still
flagged regardless of distance.

forward_guard requires several corridor points to agree before reporting an
obstacle (robust to a single noisy point, even though this repo's depth is
exact sim depth rather than the monocular-estimated depth upstream MARS was
built for). apply_avoid_cooldown biases the yaw-rate command toward the
escape side for a few ticks after AVOID releases, so the rover actually
clears the obstacle's lateral footprint before goal-seeking steering resumes
at full strength -- without this, a reactive-only guard is prone to a limit
cycle: escape left, clear hard_stop_dist, goal-bearing steering immediately
re-aims back at the same obstacle, AVOID re-triggers, repeat, no net
progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sam_vla.core.goal_geometry import intrinsics_from_hfov


@dataclass
class GuardConfig:
    cam_height: float = 0.4  # camera height above ground (m) -- MarsHabitatEnv's sensor mount
    ground_band: float = 0.08  # points below this height above the local ground ref are "ground" (m)
    max_climb_deg: float = 20.0  # terrain rising this steeply from the rover is still "ground", not an obstacle
    slope_ref_step: float = 0.15  # min forward distance before re-anchoring the local-slope ground reference (m)
    overhead: float = 1.2  # ignore points above this height (m)
    max_range: float = 4.0  # ignore points farther than this (m)
    hard_stop_dist: float = 0.60  # forward obstacle closer than this -> AVOID (m)
    reverse_dist: float = 0.35  # too close to rotate safely in place -> back up first (m)
    slow_dist: float = 2.5  # informational: min_forward below this counts as "urgent" for callers
    corridor_half_width: float = 0.35  # forward corridor half-width (m)
    stride: int = 4  # depth subsampling stride (speed)


def depth_to_obstacle_points(
    depth_m: np.ndarray,
    height: int,
    width: int,
    hfov_deg: float,
    cfg: GuardConfig,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Depth image -> (N, 2) obstacle points [x forward, y left] in the robot
    frame, via the same pinhole model (core.goal_geometry.intrinsics_from_hfov)
    every other module that unprojects a pixel through depth already agrees
    on (belief_tracking.mask_to_body, the CBF's nearest-obstacle lookup)."""
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    s = cfg.stride
    d = np.asarray(depth_m, dtype=np.float32)[::s, ::s]
    rows, cols = d.shape[:2]
    us = np.arange(0, width, s, dtype=np.float32)[:cols][None, :].repeat(rows, axis=0)
    vs = np.arange(0, height, s, dtype=np.float32)[:rows][:, None].repeat(cols, axis=1)

    valid = np.isfinite(d) & (d > 0.15) & (d < cfg.max_range)
    if exclude_mask is not None and exclude_mask.shape[:2] == (height, width):
        excl = np.asarray(exclude_mask)[::s, ::s][:rows, :cols]
        valid &= ~excl

    x_cam = (us - cx) / fx * d  # right
    y_cam = (vs - cy) / fy * d  # down
    z_up = cfg.cam_height - y_cam

    # Local-slope ground scan: walk each column near (bottom row) -> far (top
    # row), comparing each point to the last confirmed ground reference
    # rather than a flat plane through the rover's own footing (see module
    # docstring). ref_h/ref_d start at (0, 0), the rover's own footing.
    #
    # The reference is only re-anchored once slope_ref_step of forward
    # distance has actually accumulated since it was last set, not on every
    # accepted row -- re-anchoring every row would let ground_band's noise
    # allowance re-grant on each sub-cm depth step, so a near-vertical
    # obstacle sampled finely enough could "staircase" past the guard even
    # though its TOTAL rise over that short run is obstacle-sized.
    ref_h = np.zeros(cols, dtype=np.float32)
    ref_d = np.zeros(cols, dtype=np.float32)
    is_ground = np.zeros_like(d, dtype=bool)
    slope_tan = float(np.tan(np.radians(cfg.max_climb_deg)))
    for row in range(rows - 1, -1, -1):
        rise = z_up[row] - ref_h
        run = np.maximum(d[row] - ref_d, 0.0)
        tolerance = cfg.ground_band + slope_tan * run
        ground_here = valid[row] & (rise <= tolerance)
        is_ground[row] = ground_here
        advance = ground_here & (run >= cfg.slope_ref_step)
        ref_h = np.where(advance, z_up[row], ref_h)
        ref_d = np.where(advance, d[row], ref_d)

    obstacle = valid & ~is_ground & (z_up < cfg.overhead)
    return np.stack([d[obstacle], -x_cam[obstacle]], axis=1)  # [x fwd, y left]


def forward_guard(
    points: np.ndarray, cfg: GuardConfig, min_points: int = 8
) -> Tuple[float, float]:
    """Distance to the nearest obstacle CLUSTER in the forward corridor.

    At least `min_points` corridor points must agree before an obstacle is
    reported (a single stray point cannot trigger the guard); the distance
    reported is the 10th percentile of the cluster, not the bare minimum.
    Returns (forward_dist, escape_sign): escape +1 = turn left (positive
    Action.yaw_rate, this repo's CCW-positive convention -- see
    core.pose_integrator's docstring and belief_tracking.lost_goal_heading_assist,
    which steers the same sign toward a goal on the left). (inf, 0.0) means
    clear."""
    if points.shape[0] == 0:
        return float("inf"), 0.0
    in_corridor = (points[:, 0] > 0.0) & (np.abs(points[:, 1]) < cfg.corridor_half_width)
    if in_corridor.sum() < min_points:
        return float("inf"), 0.0
    corridor_pts = points[in_corridor]
    dist = float(np.percentile(corridor_pts[:, 0], 10))
    near = corridor_pts[corridor_pts[:, 0] < dist + 0.3]
    # obstacle cluster's median is on the right (negative "left") -> escape left (+1)
    escape = 1.0 if float(np.median(near[:, 1])) < 0 else -1.0
    return dist, escape


def apply_avoid_cooldown(
    yaw_rate: float,
    state: str,
    avoid_side: float,
    cooldown_left: int,
    bias_gain: float,
    max_yaw_rate: float,
) -> Tuple[float, int]:
    """Bias `yaw_rate` toward `avoid_side` for `cooldown_left` more ticks
    after an AVOID escape, instead of letting goal-seeking steering snap
    straight back toward the just-avoided obstacle the instant the corridor
    clears (see module docstring).

    Call site contract: when AVOID triggers, the caller latches
    ``avoid_side = escape`` and resets ``cooldown_left = avoid_cooldown_ticks``
    (every trigger re-arms it, so a persistent obstacle keeps the bias alive
    for the whole encounter). This is then called once per non-AVOID tick on
    whatever yaw_rate the TRACK branch computed. A no-op while
    ``state == "AVOID"`` itself (that branch already commands the full escape
    turn) or once the cooldown has run out.

    Returns (yaw_rate, cooldown_left) -- the caller stores cooldown_left back
    onto its own state for the next tick."""
    if state == "AVOID" or cooldown_left <= 0:
        return yaw_rate, cooldown_left
    biased = float(np.clip(yaw_rate + bias_gain * avoid_side, -max_yaw_rate, max_yaw_rate))
    return biased, cooldown_left - 1


if __name__ == "__main__":
    cfg = GuardConfig(stride=1, max_range=5.0)
    height, width = 16, 16

    # Flat, level ground: for a camera at cam_height with no tilt, a ground
    # point at forward depth X projects to row v = cy + fy*cam_height/X, i.e.
    # X = fy*cam_height/(v - cy) below the horizon row. Build depth from that
    # closed form so every pixel is a true ground-plane hit -- unlike a
    # uniform-depth image (which is a frontal WALL, not ground; see below).
    intr = intrinsics_from_hfov(height, width, 90.0)
    fy, cy = intr["fy"], intr["cy"]
    rows_below_horizon = np.maximum(np.arange(height, dtype=np.float32) - cy, 0.5)
    depth_per_row = np.clip(fy * cfg.cam_height / rows_below_horizon, 0.2, cfg.max_range)
    flat = np.repeat(depth_per_row[:, None], width, axis=1)
    pts_flat = depth_to_obstacle_points(flat, height, width, 90.0, cfg)
    print(f"flat ground: {pts_flat.shape[0]} obstacle points (expect 0)")
    assert pts_flat.shape[0] == 0

    # A close, near-vertical wall dead ahead: uniform near depth -> every
    # column has a sharp local rise once vertical extent is projected.
    wall = np.full((height, width), 0.5, dtype=np.float32)
    pts_wall = depth_to_obstacle_points(wall, height, width, 90.0, cfg)
    min_fwd, escape = forward_guard(pts_wall, cfg)
    print(f"close wall: {pts_wall.shape[0]} obstacle points, min_forward={min_fwd:.2f}")
    assert pts_wall.shape[0] > 0
    assert min_fwd < cfg.hard_stop_dist

    yaw_rate, cooldown = apply_avoid_cooldown(0.0, "TRACK", avoid_side=1.0, cooldown_left=3,
                                              bias_gain=0.3, max_yaw_rate=1.0)
    print(f"cooldown bias: yaw_rate={yaw_rate:.2f} cooldown_left={cooldown}")
    assert yaw_rate > 0.0
    assert cooldown == 2
    print("OK")
