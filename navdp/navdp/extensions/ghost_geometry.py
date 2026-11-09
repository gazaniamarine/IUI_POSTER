"""Pure geometry for ghost-subgoal rendering: belief bearing/range -> pixel mask.

Extracted from roundtrip_rollout.py so both roundtrip_rollout.py and
rollout_habitat_policy.py can use the SAME, already-validated math without one
script importing the other (that created a circular import).

No dependency on any project module beyond math/numpy -- keep it that way.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def gc_intrinsics(h: int, w: int, hfov_deg: float = 90.0) -> Dict[str, float]:
    """Habitat camera intrinsics: fx = (W/2)/tan(HFOV/2)."""
    fx = (w / 2) / math.tan(math.radians(hfov_deg / 2))
    return {"fx": fx, "fy": fx, "cx": (w - 1) / 2.0, "cy": (h - 1) / 2.0}


def gc_body_point(cur_pos, cur_yaw: float, bearing_rad: float, dist_m: float) -> Tuple[float, float, float]:
    """World (x, y, z) at body-frame bearing_rad (+left = positive), dist_m from robot.

    Convention matches motion_basis('habitat'):
      fwd  = (-sin(yaw), -, -cos(yaw))
      left = ( cos(yaw), -,  -sin(yaw))
    """
    c, s = math.cos(cur_yaw), math.sin(cur_yaw)
    fwd_x, fwd_z = -s, -c
    left_x, left_z = c, -s
    fwd_d = dist_m * math.cos(bearing_rad)
    left_d = dist_m * math.sin(bearing_rad)
    return (
        cur_pos[0] + fwd_d * fwd_x + left_d * left_x,
        cur_pos[1],
        cur_pos[2] + fwd_d * fwd_z + left_d * left_z,
    )


def gc_project(world_pt, robot_pos, robot_yaw: float, intr: Dict[str, float]) -> Tuple[Optional[float], Optional[float], float]:
    """Project world_pt to image pixel (u, v) and forward depth z_cam.

    Returns (None, None, z_cam) when point is behind the camera.
    """
    dx = world_pt[0] - robot_pos[0]
    dz = world_pt[2] - robot_pos[2]
    dy = world_pt[1] - robot_pos[1]
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    z_cam = -s * dx - c * dz   # forward depth
    x_cam = -c * dx + s * dz   # rightward in camera
    y_cam = -dy                 # downward in camera (+y down)
    if z_cam <= 1e-4:
        return None, None, float(z_cam)
    u = intr["fx"] * x_cam / z_cam + intr["cx"]
    v = intr["fy"] * y_cam / z_cam + intr["cy"]
    return float(u), float(v), float(z_cam)


def gc_make_mask(h: int, w: int, u: Optional[float], v: Optional[float], radius: float) -> np.ndarray:
    """Filled-circle ghost mask at pixel (u, v)."""
    mask = np.zeros((h, w), dtype=np.float32)
    if u is None or v is None:
        return mask
    yi, xi = np.ogrid[:h, :w]
    mask[(xi - round(u)) ** 2 + (yi - round(v)) ** 2 <= radius ** 2] = 1.0
    return mask
