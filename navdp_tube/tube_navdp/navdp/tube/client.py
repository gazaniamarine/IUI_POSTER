"""Client for scripts/tube_navdp_server.py: the official point-goal request plus the BEV grid and the mode."""
from __future__ import annotations

import json

import numpy as np

from navdp.official_client import NavDPResult, OfficialNavDPClient, encode_rgbd, validate_candidates, validate_trajectory


def raise_with_reason(response):
    """raise_for_status, with the server's own error text (e.g. an option an outdated server does not know)."""
    try:
        response.raise_for_status()
    except Exception as e:  # noqa: BLE001  (requests.HTTPError, or a stand-in's own error)
        code = getattr(response, "status_code", None)
        try:
            reason = response.json().get("error") or response.text
        except Exception:  # noqa: BLE001  (not JSON)
            reason = getattr(response, "text", "") or e
        raise RuntimeError(f"server answered {code}: {str(reason).strip()[:200]}"
                           + (" (outdated server? restart it)" if code == 400 else "")) from e


class TubeNavDPClient(OfficialNavDPClient):
    acts_path = None    # set before a step to have the server save that step's activations there (acts.py)
    step_seed = None    # set before a step to seed the server's sampling noise for it
    def step(self, rgb, depth, point_goal, bev=None, mode="navdp", start_forward=0.0, split=None,
             record_x0=False, push=None, prev=None, consistency=None, smooth=None, rebuild=None, lyapunov=None,
             lyapunov_delta=None, guide_until=None, push_step=None, push_margin=None) -> NavDPResult:
        """bev: a navdp.tube.stt.Bev in the body frame with the rover's size included (None = no obstacles). prev:
        the previous plan [m, 2] in this body frame, from the waypoint reached on (driver.carry_over; None = none; the
        server uses it only with --consistency > 0). The result has an extra attribute ``tube`` (the server's tube
        report) when the server sends one."""
        if not self.ready:
            raise RuntimeError("reset the server at the start of every episode")
        goal = np.asarray(point_goal, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("point goal must be finite [forward,left] metres")
        image, depth_png, stats = encode_rgbd(rgb, depth)
        tube_data = {"mode": mode}
        if start_forward:
            tube_data["start_forward"] = float(start_forward)
        if split is not None:
            tube_data["split"] = int(split)
        if record_x0:
            tube_data["record_x0"] = True
        if rebuild is not None:                # overrides --rebuild-tubes (None = the server's default)
            tube_data["rebuild"] = bool(rebuild)
        if lyapunov is not None:               # off | min | ramp, overrides --lyapunov
            tube_data["lyapunov"] = str(lyapunov)
        if lyapunov_delta is not None:         # overrides --lyapunov-delta
            tube_data["lyapunov_delta"] = float(lyapunov_delta)
        if guide_until is not None:            # overrides --guide-until (0 = guide to the last step)
            tube_data["guide_until"] = int(guide_until)
        if push_step is not None:              # obstacle mode, overrides --push-step (m per waypoint and axis)
            tube_data["push_step"] = float(push_step)
        if push_margin is not None:            # obstacle mode, overrides --push-margin (m past the grown obstacles)
            tube_data["push_margin"] = float(push_margin)
        if consistency is not None:            # mu, overrides the server's --consistency (None = its default)
            tube_data["consistency"] = float(consistency)
        if smooth is not None:                 # beta, overrides the server's --smooth (None = its default)
            tube_data["smooth"] = float(smooth)
        if push is not None:                   # tube push mode (nearest | smooth); None = the server's --tube-push
            tube_data["push"] = str(push)
        if bev is not None:
            tube_data["bev"] = bev.encode()
        if prev is not None and len(prev):
            tube_data["prev"] = np.asarray(prev, dtype=float)[:, :2].round(4).tolist()
        if self.acts_path:
            tube_data["acts_path"] = str(self.acts_path)
        if self.step_seed is not None:
            tube_data["seed"] = int(self.step_seed)
        response = self.session.post(self.base_url + "/pointgoal_step", files={
            "image": ("image.png", image, "image/png"),
            "depth": ("depth.png", depth_png, "image/png"),
        }, data={"goal_data": json.dumps({"goal_x": [float(goal[0])], "goal_y": [float(goal[1])]}),
                 "tube_data": json.dumps(tube_data)}, timeout=self.timeout)
        raise_with_reason(response)
        payload = response.json()
        if "trajectory" not in payload:
            raise ValueError("server response is missing trajectory")
        result = NavDPResult(validate_trajectory(payload["trajectory"]), stats, *validate_candidates(payload))
        result.tube = payload.get("tube")
        result.acts_saved = bool(payload.get("acts_saved", False))
        if mode != "navdp" and result.tube is None:
            raise ValueError("the server sent no tube report: is it scripts/tube_navdp_server.py?")
        return result


class XNavDPClient(OfficialNavDPClient):
    """Client for scripts/xnavdp_server.py (upstream X-NavDP, port 8891). Same request as NavDP plus the robot's pose
    (``state_data``: X-NavDP uses it to keep to its previous path and to detect being stuck). No BEV grid, no tubes:
    the result is X-NavDP's best-valued path, with ``tube = None``."""

    wants_pose = True
    acts_path = None    # set before a step to have the server save that step's activations there (acts.py)
    step_seed = None    # set before a step to seed the server's sampling noise for it

    def reset(self, intrinsic, stop_threshold=-4.):
        intrinsic = np.asarray(intrinsic, dtype=float)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("valid 3x3 camera intrinsics required")
        self.ready = False
        response = self.session.post(self.base_url + "/navigator_reset", json={
            "intrinsic": intrinsic.tolist(), "batch_size": 1}, timeout=self.timeout)
        response.raise_for_status()
        if response.json().get("algo") != "navdp-rl":
            raise ValueError("server did not identify itself as X-NavDP (navdp-rl): is it scripts/xnavdp_server.py?")
        self.ready = True

    def step(self, rgb, depth, point_goal, bev=None, mode="navdp", start_forward=0.0, pose=None, **_) -> NavDPResult:
        """pose: (x, y, yaw) of the robot in the odom frame when the image was taken (None = no state guidance)."""
        if not self.ready:
            raise RuntimeError("reset the server at the start of every episode")
        goal = np.asarray(point_goal, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("point goal must be finite [forward,left] metres")
        image, depth_png, stats = encode_rgbd(rgb, depth)
        data = {"goal_data": json.dumps({"goal_x": [float(goal[0])], "goal_y": [float(goal[1])]})}
        if pose is not None:
            x, y, yaw = (float(v) for v in pose)
            data["state_data"] = json.dumps({"robot_pos": [[x, y, 0.0]],        # quaternion xyzw (SciPy order)
                                             "robot_quat": [[0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))]]})
        if self.acts_path:
            data["acts_path"] = str(self.acts_path)
        if self.step_seed is not None:
            data["seed"] = str(int(self.step_seed))
        response = self.session.post(self.base_url + "/pointgoal_step", files={
            "image": ("image.png", image, "image/png"),
            "depth": ("depth.png", depth_png, "image/png"),
        }, data=data, timeout=self.timeout)
        raise_with_reason(response)
        payload = response.json()
        if "trajectory" not in payload:
            raise ValueError("server response is missing trajectory")
        result = NavDPResult(validate_trajectory(payload["trajectory"]), stats, *validate_candidates(payload))
        result.tube = None
        result.acts_saved = bool(payload.get("acts_saved", False))
        result.stuck = payload.get("stuck")    # X-NavDP's own stuck flag (it then runs a random sample); None = old server
        return result


SAND_CAMERA = (640, 480, 389.551)   # SanD-Planner's training camera: width, height, focal length (px)


def to_virtual_camera(rgb, depth, intrinsic, size=SAND_CAMERA):
    """RGB [H, W, 3] uint8 + depth [H, W] m from a camera with ``intrinsic`` -> the same view through a pinhole camera
    of ``size`` = (width, height, f) at the same place and heading (centred principal point), and its 3x3 intrinsic.
    Each target pixel's ray is looked up in the source image (depth along the optical axis does not change). The sim
    camera's view is a little shorter than the target's (60 vs 63 deg): those few rows repeat the nearest edge row,
    not depth 0, which SanD-Planner could read as an obstacle at the camera."""
    import cv2
    w, h, f = size
    k = np.asarray(intrinsic, np.float64)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (k[0, 2] + k[0, 0] * (u - w / 2) / f).astype(np.float32)
    map_y = (k[1, 2] + k[1, 1] * (v - h / 2) / f).astype(np.float32)
    rgb_v = cv2.remap(np.ascontiguousarray(rgb), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    depth_v = cv2.remap(np.ascontiguousarray(depth, np.float32), map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REPLICATE)
    return rgb_v, depth_v, np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


class SandPlannerClient(OfficialNavDPClient):
    """Client for scripts/sandplanner_server.py (upstream SanD-Planner, port 8892). Same request as NavDP, but every
    frame is first re-projected into SanD-Planner's training camera (to_virtual_camera: 640x480, f 389.55; the sim
    camera is wider and 16:9), and that camera's intrinsic is what the server gets. No BEV grid, no tubes: the result
    is SanD-Planner's own best path; its costs (lower = better) come back negated, as values (higher = better)."""

    def reset(self, intrinsic, stop_threshold=-4.):
        intrinsic = np.asarray(intrinsic, dtype=float)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("valid 3x3 camera intrinsics required")
        self.ready, self.source_intrinsic = False, intrinsic
        response = self.session.post(self.base_url + "/navigator_reset", json={
            "intrinsic": to_virtual_camera(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2)), intrinsic)[2].tolist(),
            "stop_threshold": float(stop_threshold), "batch_size": 1}, timeout=self.timeout)
        response.raise_for_status()
        if response.json().get("algo") != "sand_planner":
            raise ValueError("server did not identify itself as SanD-Planner: is it scripts/sandplanner_server.py?")
        self.ready = True

    def step(self, rgb, depth, point_goal, bev=None, mode="navdp", start_forward=0.0, **_) -> NavDPResult:
        if not self.ready:
            raise RuntimeError("reset the server at the start of every episode")
        goal = np.asarray(point_goal, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("point goal must be finite [forward,left] metres")
        rgb_v, depth_v, _ = to_virtual_camera(rgb, depth, self.source_intrinsic)
        image, depth_png, stats = encode_rgbd(rgb_v, np.where(depth_v > 0, depth_v, np.nan))
        response = self.session.post(self.base_url + "/pointgoal_step", files={
            "image": ("image.png", image, "image/png"),
            "depth": ("depth.png", depth_png, "image/png"),
        }, data={"goal_data": json.dumps({"goal_x": [float(goal[0])], "goal_y": [float(goal[1])]})},
            timeout=self.timeout)
        raise_with_reason(response)
        payload = response.json()
        if "trajectory" not in payload:
            raise ValueError("server response is missing trajectory")
        if "all_values" in payload:
            payload["all_values"] = (-np.asarray(payload["all_values"], np.float64)).tolist()
        result = NavDPResult(validate_trajectory(payload["trajectory"]), stats, *validate_candidates(payload))
        result.tube = None
        return result
