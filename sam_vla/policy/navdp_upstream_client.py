"""HTTP client for the vendored upstream InternRobotics/NavDP server --
navdp_server.py (official) or navdp_s2diff_server.py (this project's
obstacle-guided fork of it, see navdp_upstream_server_manager.py's
docstring) -- next.md's "Integration project" Phase 2.

Pure module: no torch import, so it's importable from the habitat conda env
(unlike sam_vla.policy.navdp_policy, which needs this repo's own navdp/'s
torch stack). Request/response shapes below were read directly from
navdp_server.py's /pointgoal_step route and policy_agent.py's
NavDP_Agent.process_pointgoal / project_trajectory
(github.com/InternRobotics/NavDP@master, baselines/navdp/), not guessed:

- image: multipart file, any PIL-loadable format (server does
  Image.open(...).convert('RGB')) -- JPEG here.
- depth: multipart file, a single-channel PNG the server opens and
  .convert('I') (32-bit int) then divides by 10000.0 -- i.e. depth_meters *
  10000 clipped to uint16 range, matching this repo's belief_exp-adjacent
  noise-injection convention of documenting exact numeric formulas rather
  than re-deriving them.
- goal_data: a form field (not a file), a JSON string {"goal_x": [...],
  "goal_y": [...]} -- one point per batch element; this repo only ever runs
  batch_size=1.
- response 'trajectory': good_trajectory[:, 0] server-side, shape
  (batch, predict_size, 3) -- confirmed from policy_network.py's
  predict_pointgoal_action: `all_trajectory = torch.cumsum(naction / 4.0,
  dim=1)`, i.e. each of the predict_size waypoints is a CUMULATIVE
  (x=forward, y=left, z=unused) position in the body frame AT REQUEST TIME,
  not a per-step delta and not a velocity chunk (this repo's own NavdpPolicy
  outputs velocities -- see next.md's "What custom S2DiT + navdp means
  today" section for that distinction). We keep only the first two columns.
"""

from __future__ import annotations

import io
import json
import math
from typing import Optional

import numpy as np
import requests
from PIL import Image

from sam_vla.core.types import Action
from sam_vla.core.uncertainty_motion import yaw_rate_toward_heading

_DEPTH_UNITS_PER_METER = 10000.0
_DEPTH_MAX_UNITS = 65535


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
        buf, format="JPEG", quality=quality
    )
    return buf.getvalue()


def encode_depth_png16(depth_m: np.ndarray) -> bytes:
    """depth_meters * 10000 clipped to [0, 65535], saved as a 16-bit PNG --
    see this module's docstring for why this exact scale/dtype (the server's
    receiving side, not a convention we chose freely)."""
    depth_units = np.clip(
        np.asarray(depth_m, dtype=np.float64) * _DEPTH_UNITS_PER_METER,
        0,
        _DEPTH_MAX_UNITS,
    )
    depth_u16 = depth_units.astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(depth_u16).save(buf, format="PNG")  # dtype uint16 -> mode "I;16"
    return buf.getvalue()


def pointgoal_step(
    base_url: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    goal_forward: float,
    goal_left: float,
    timeout: float = 30.0,
    obstacle_pixels: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One /pointgoal_step call. goal_forward is clamped to [0, 10],
    goal_left to [-10, 10] (policy_agent.py's process_pointgoal -- the server
    re-clips regardless, this just keeps what we log honest). Returns
    (trajectory_xy [predict_size, 2] body-frame-at-request-time cumulative
    (forward, left) waypoints, all_values [n_candidates] critic scores --
    informational only, the server already used them internally to pick
    the returned candidate and to force a stop when all_values.max() is
    below its own stop_threshold).

    obstacle_pixels: a flat [[u, v], ...] list of pixel coordinates for this
    (sole, batch_size=1 -- see module docstring) batch item. Only
    navdp_s2diff_server.py's /pointgoal_step reads this (its S2Diff guidance
    steers sampled trajectories away from these pixels' projected obstacle
    points); navdp_server.py's route ignores unknown goal_data keys
    entirely. It's still sent unconditionally rather than only for the
    s2diff variant, because navdp_s2diff_server.py's request decoder raises
    ValueError if the key is missing at all -- even in its --planner-mode
    pure-navdp mode -- so omitting it would 400 against that server
    regardless of mode. Defaults to no obstacle points, which degrades
    s2diff guidance to unguided sampling; pass real projected obstacle
    pixels to get actual avoidance."""
    goal_x = float(np.clip(goal_forward, 0.0, 10.0))
    goal_y = float(np.clip(goal_left, -10.0, 10.0))
    obstacle_pixels_batch = [list(obstacle_pixels) if obstacle_pixels else []]
    files = {
        "image": ("rgb.jpg", encode_rgb_jpeg(rgb), "image/jpeg"),
        "depth": ("depth.png", encode_depth_png16(depth_m), "image/png"),
    }
    data = {
        "goal_data": json.dumps(
            {
                "goal_x": [goal_x],
                "goal_y": [goal_y],
                "obstacle_pixels": obstacle_pixels_batch,
            }
        )
    }
    resp = requests.post(
        f"{base_url}/pointgoal_step", files=files, data=data, timeout=timeout
    )
    resp.raise_for_status()
    payload = resp.json()

    trajectory = np.asarray(payload["trajectory"], dtype=np.float32).reshape(-1, 3)
    trajectory_xy = trajectory[:, :2]
    all_values = np.asarray(payload.get("all_values", []), dtype=np.float32).reshape(
        -1
    )
    return trajectory_xy, all_values


def select_action_from_trajectory(
    trajectory_xy: np.ndarray,
    waypoint_index: int,
    max_forward_speed: float = 1.0,
    turn_kp: float = 1.4,
    max_yaw_rate: float = 1.0,
) -> Action:
    """Turns one cumulative body-frame waypoint (trajectory_xy[waypoint_index]
    = (forward, left), already relative to the current body frame) into a
    single Action -- the same "steer toward a point" problem
    sam_vla.core.uncertainty_motion.yaw_rate_toward_heading already solves
    for Study 1's Phase 1 (next.md), reused here with current_yaw=0 since the
    target is given in body frame already, not world frame. v_fwd is the
    remaining distance to the waypoint clamped to max_forward_speed (not a
    fixed cruise speed), so the rover naturally slows near a waypoint close
    to the plan's tail. waypoint_index is clamped into range, so callers can
    pass an ever-growing "steps since replan + lookahead" index without
    bounds-checking themselves."""
    idx = int(np.clip(waypoint_index, 0, trajectory_xy.shape[0] - 1))
    forward, left = float(trajectory_xy[idx, 0]), float(trajectory_xy[idx, 1])
    target_yaw = math.atan2(left, forward)
    yaw_rate = yaw_rate_toward_heading(0.0, target_yaw, turn_kp, max_yaw_rate)
    distance = math.hypot(forward, left)
    v_fwd = float(np.clip(distance, 0.0, max_forward_speed))
    return Action(v_fwd=v_fwd, v_lat=0.0, yaw_rate=yaw_rate)
