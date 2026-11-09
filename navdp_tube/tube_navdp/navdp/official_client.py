"""Inference-only official NavDP HTTP boundary (no torch/Habitat dependency).

Protocol checked against InternRobotics/NavDP baselines/navdp/navdp_server.py.
This is NOT the differentiable interface for later Qwen/belief training.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


def pointgoal_body(goal_xz, position_xz, yaw: float) -> np.ndarray:
    """Habitat world X/Z -> metres forward/left; yaw=0 faces world -Z."""
    d = np.asarray(goal_xz, dtype=float) - np.asarray(position_xz, dtype=float)
    if d.shape != (2,) or not np.isfinite(d).all() or not math.isfinite(yaw):
        raise ValueError("finite X/Z pairs and yaw required")
    s, c = math.sin(yaw), math.cos(yaw)
    return np.asarray([-s * d[0] - c * d[1], -c * d[0] + s * d[1]], dtype=np.float32)


def validate_trajectory(value) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float32)
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if (trajectory.ndim != 2 or trajectory.shape[0] == 0
            or trajectory.shape[1] not in (2, 3) or not np.isfinite(trajectory).all()):
        raise ValueError("expected finite trajectory [H,2/3] or [1,H,2/3]")
    return trajectory


def track_trajectory(trajectory, *, lookahead=3, max_speed=.4, max_yaw_rate=.8,
                     yaw_gain=1.5, stop_radius=.05) -> np.ndarray:
    """Forward-only tracker; behind/lateral waypoints cause in-place rotation.

    Outputs [forward m/s, lateral m/s (=0), yaw rad/s]. This is not collision
    checking. The optional third trajectory component is not interpreted.
    """
    if (lookahead < 0 or not np.isfinite([max_speed, max_yaw_rate, yaw_gain, stop_radius]).all()
            or min(max_speed, max_yaw_rate, yaw_gain, stop_radius) <= 0):
        raise ValueError("nonnegative lookahead and positive finite controller limits required")
    path = validate_trajectory(trajectory)
    forward, left = path[min(lookahead, len(path) - 1), :2]
    distance = math.hypot(float(forward), float(left))
    if distance <= stop_radius:
        return np.zeros(3, dtype=np.float32)
    bearing = math.atan2(float(left), float(forward))
    # Never average a rear waypoint into a forward-driving command.
    speed = min(max_speed, distance) * max(0., math.cos(bearing))
    if abs(bearing) >= math.pi / 3:
        speed = 0.
    return np.asarray([speed, 0., np.clip(yaw_gain * bearing, -max_yaw_rate, max_yaw_rate)],
                      dtype=np.float32)


def encode_rgbd(rgb, depth):
    rgb, depth = np.asarray(rgb), np.asarray(depth, dtype=np.float32)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB must be uint8 [H,W,3], not BGR or normalized floats")
    if depth.shape != rgb.shape[:2] or depth.size == 0:
        raise ValueError("depth must be [H,W] metres matching RGB")
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("depth has no valid positive measurements")
    # Official wire format: uint16 PNG / 10000. Keep raw depth separately.
    wire = np.rint(np.clip(np.where(valid, depth, 0), 0, 6.5535) * 10000).astype(np.uint16)
    image_bytes, depth_bytes = io.BytesIO(), io.BytesIO()
    Image.fromarray(rgb).save(image_bytes, format="PNG")
    Image.fromarray(wire).save(depth_bytes, format="PNG")
    stats = {"valid_fraction": float(valid.mean()),
             "wire_clipped_fraction": float((valid & (depth > 6.5535)).mean()),
             "official_usable_fraction": float((valid & (depth >= .1) & (depth <= 5.)).mean())}
    return image_bytes.getvalue(), depth_bytes.getvalue(), stats


@dataclass
class NavDPResult:
    trajectory: np.ndarray
    depth_stats: dict
    candidates: np.ndarray = None  # [N, H, 2/3]: every sampled trajectory, when the server returns them
    values: np.ndarray = None      # [N]: the critic's value of each (the server executes the highest)


def validate_candidates(payload):
    """The server's ``all_trajectory`` [1, N, H, 2/3] and ``all_values`` [1, N] -> ([N, H, 2/3], [N]) or (None, None)
    when it sends none or they are malformed (older servers return only the executed trajectory)."""
    if "all_trajectory" not in payload or "all_values" not in payload:
        return None, None
    candidates = np.asarray(payload["all_trajectory"], dtype=np.float32)
    values = np.asarray(payload["all_values"], dtype=np.float32)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    values = values.reshape(-1)
    if (candidates.ndim != 3 or candidates.shape[2] not in (2, 3) or len(values) != len(candidates)
            or not np.isfinite(candidates).all() or not np.isfinite(values).all()):
        return None, None
    return candidates, values


class OfficialNavDPClient:
    def __init__(self, base_url="http://127.0.0.1:8888", timeout=120., session=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.ready = False

    def reset(self, intrinsic, stop_threshold=-4.):
        intrinsic = np.asarray(intrinsic, dtype=float)
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not math.isfinite(stop_threshold)):
            raise ValueError("valid 3x3 camera intrinsics and finite threshold required")
        self.ready = False
        response = self.session.post(self.base_url + "/navigator_reset", json={
            "intrinsic": intrinsic.tolist(), "stop_threshold": stop_threshold, "batch_size": 1,
        }, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("algo") != "navdp":
            raise ValueError("server did not identify itself as navdp")
        self.ready = True

    def step(self, rgb, depth, point_goal) -> NavDPResult:
        if not self.ready:
            raise RuntimeError("reset the server at the start of every episode")
        goal = np.asarray(point_goal, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("point goal must be finite [forward,left] metres")
        image, depth_png, stats = encode_rgbd(rgb, depth)
        response = self.session.post(self.base_url + "/pointgoal_step", files={
            "image": ("image.png", image, "image/png"),
            "depth": ("depth.png", depth_png, "image/png"),
        }, data={"goal_data": json.dumps({"goal_x": [float(goal[0])], "goal_y": [float(goal[1])]})},
            timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if "trajectory" not in payload:
            raise ValueError("server response is missing trajectory")
        return NavDPResult(validate_trajectory(payload["trajectory"]), stats, *validate_candidates(payload))

    def close(self):
        self.session.close()
