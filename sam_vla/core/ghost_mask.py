"""Belief-uncertainty-driven "ghost mask": a translucent circle drawn on a
frame at the belief's projected pixel, radius proportional to uncertainty.
Pure numpy, mirrors navdp/navdp/extensions/ghost_geometry.py's shape
(body-frame bearing/range -> pixel -> filled-circle mask) but built against
sam_vla.core.goal_geometry.intrinsics_from_hfov and BeliefGoalTracker's
[forward, left] body-frame convention, so it composes directly with
.bearing()/.distance() instead of needing a separate world-frame projection
step. Ported as a pattern, not an import -- navdp/ stays read-only per
CLAUDE.md.

Non-destructive and policy-blind: callers draw this onto a copy of the frame
handed to a VLM/visualization, never onto anything a policy consumes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from sam_vla.core.goal_geometry import intrinsics_from_hfov

GHOST_COLOR = np.array([0, 255, 0], dtype=np.float32)


def uncertainty_to_radius_px(
    uncertainty: float, min_px: float, max_px: float, scale: float
) -> float:
    """Maps a scalar uncertainty to an on-screen circle radius in pixels,
    clamped to [min_px, max_px] -- same clamp pattern kb_teleop_vl.py already
    uses for obstacle overlays (OVERLAY_MIN_PIXEL_RADIUS/OVERLAY_MAX_PIXEL_RADIUS).
    """
    return float(
        np.clip(float(uncertainty) * float(scale), float(min_px), float(max_px))
    )


def project_body_point_to_pixel(
    forward: float, left: float, hfov_deg: float, h: int, w: int
) -> Optional[Tuple[float, float]]:
    """Body-frame [forward, left] point -> pixel (u, v), or None if behind the
    camera or outside the frame's horizontal extent.

    BeliefGoalTracker's belief state is 2D (no elevation component), so v is
    fixed at the frame's vertical center (cy) -- the belief only ever carries
    bearing/range, not height.
    """
    if forward <= 1e-3:
        return None
    intr = intrinsics_from_hfov(h, w, hfov_deg)
    u = intr["cx"] - float(left) * intr["fx"] / float(forward)
    if u < 0.0 or u > float(w - 1):
        return None
    return u, intr["cy"]


def project_or_clamp_body_point_to_pixel(
    forward: float,
    left: float,
    hfov_deg: float,
    h: int,
    w: int,
    back_margin_deg: float = 25.0,
) -> Tuple[float, float]:
    """Same body-frame [forward, left] -> pixel (u, v) projection as
    project_body_point_to_pixel, but never returns None: whenever the point
    isn't directly projectable, it's clamped to the frame edge it's headed
    toward instead, so the ghost stays visible and keeps pointing at the
    belief every frame instead of disappearing while out of view.

    Three cases:
    - In front (forward > 0) but beyond the horizontal FOV: clamp u to the
      near edge (0 or w-1) at v=cy -- "the side it'll swing into view from"
      as the rover turns toward it.
    - Behind (forward <= 0) but with a meaningful left/right component:
      same side-edge clamp, since the point is still closer to that side
      than the other.
    - Behind and (near-)centered -- bearing within back_margin_deg of
      straight back, or the point is essentially at the rover's own
      position (r ~ 0, direction undefined either way: not really left,
      right, or ahead) -- placed at the bottom edge (v=h-1, u=cx),
      i.e. "down", since neither side edge is meaningfully closer.
    """
    intr = intrinsics_from_hfov(h, w, hfov_deg)
    r = math.hypot(forward, left)
    if r < 1e-3:
        return intr["cx"], float(h - 1)
    if forward > 1e-3:
        u = intr["cx"] - float(left) * intr["fx"] / float(forward)
        u = float(np.clip(u, 0.0, float(w - 1)))
        return u, intr["cy"]
    bearing_deg = math.degrees(math.atan2(left, forward))
    if abs(bearing_deg) >= 180.0 - float(back_margin_deg):
        return intr["cx"], float(h - 1)
    return (0.0 if left > 0 else float(w - 1)), intr["cy"]


def draw_ghost_mask(
    rgb: np.ndarray,
    u: float,
    v: float,
    radius_px: float,
    color: np.ndarray = GHOST_COLOR,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blends a translucent filled circle onto rgb (uint8 HWC) at pixel
    (u, v) with the given radius. Non-destructive -- returns a copy, modeled
    on kb_teleop_vl.py's overlay_obstacles()."""
    rgb = np.asarray(rgb)
    annotated = rgb.astype(np.float32)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    annotated[mask] = annotated[mask] * (1.0 - alpha) + np.asarray(
        color, dtype=np.float32
    ) * alpha
    return annotated.astype(np.uint8)


def draw_ghost_ellipse(
    rgb: np.ndarray,
    u: float,
    v: float,
    radius_u_px: float,
    radius_v_px: float,
    color: np.ndarray = GHOST_COLOR,
    alpha: float = 0.45,
) -> np.ndarray:
    """Axis-aligned generalization of draw_ghost_mask: an anisotropic belief
    (radius_u_px != radius_v_px) draws a translucent ellipse instead of a
    circle, so a covariance that's more uncertain along one image axis than
    the other actually looks that way. radius_u_px == radius_v_px reduces to
    the same shape draw_ghost_mask produces. Non-destructive -- returns a
    copy."""
    rgb = np.asarray(rgb)
    annotated = rgb.astype(np.float32)
    ru = max(float(radius_u_px), 1e-6)
    rv = max(float(radius_v_px), 1e-6)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    mask = ((xx - u) / ru) ** 2 + ((yy - v) / rv) ** 2 <= 1.0
    annotated[mask] = annotated[mask] * (1.0 - alpha) + np.asarray(
        color, dtype=np.float32
    ) * alpha
    return annotated.astype(np.uint8)


def belief_to_bearing_range_uncertainty(
    mean_forward_left: np.ndarray, cov_forward_left: np.ndarray
) -> Tuple[float, float, float, float]:
    """Body-frame Gaussian belief [forward, left] mean + 2x2 covariance ->
    (bearing_deg, distance_m, bearing_uncertainty_deg, distance_uncertainty_m).

    The mean alone gives bearing/distance via atan2/hypot (same convention as
    BeliefGoalTracker.bearing()/.distance()). The covariance doesn't have a
    single natural image-space axis to project onto the way the mean does --
    there is no fixed camera pose once the goal may be entirely out of frame
    -- so instead of a Jacobian through the (undefined-when-behind-camera)
    pixel projection, this resolves the covariance into the belief's own
    radial/tangential frame: variance along the radial direction (toward/away
    from the believed point) becomes range uncertainty directly; variance
    along the tangential direction (perpendicular, at radius r) becomes an
    angular spread via the small-angle arc/r relation. That gives two
    physically meaningful scalars -- "how sure are we of the direction" and
    "how sure are we of the distance" -- a VLM prompt can reason about and
    use to size a ghost-mask ellipse, without needing the goal to actually be
    projectable into the current frame.

    At the origin (r ~ 0) radial/tangential are undefined; falls back to an
    isotropic spread from the covariance trace and reports full (180 deg)
    bearing uncertainty, since direction is meaningless with zero range.
    """
    fwd, left = float(mean_forward_left[0]), float(mean_forward_left[1])
    r = math.hypot(fwd, left)
    bearing_rad = math.atan2(left, fwd)
    cov = np.asarray(cov_forward_left, dtype=np.float64)

    if r < 1e-3:
        sigma = math.sqrt(max(float(np.trace(cov)) / 2.0, 0.0))
        return math.degrees(bearing_rad), r, 180.0, sigma

    e_r = np.array([fwd, left]) / r
    e_t = np.array([-left, fwd]) / r
    sigma_r = math.sqrt(max(float(e_r @ cov @ e_r), 0.0))
    sigma_t = math.sqrt(max(float(e_t @ cov @ e_t), 0.0))
    bearing_unc_deg = math.degrees(sigma_t / r)
    return math.degrees(bearing_rad), r, bearing_unc_deg, sigma_r
