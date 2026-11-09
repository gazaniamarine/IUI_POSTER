"""NavDP guided by a segmentation BEV obstacle map (navdp_tube, "ours").

Same surface as NavdpUpstreamPolicy (start / stop / set_goal_body /
act_verbose / _last_trajectory) so the rover controller can swap it in:

  camera RGB-D --> best.pth SAM2 terrain segmentation --> BEV occupancy grid
                                                            |
  body-frame goal --> tube NavDP server (:8890) score-guided away from the
                      grid's obstacles --> path --> PathFollower --> (v, w)

The segmentation map + TubeNavigator run in nav/tube_worker.py, a separate
process (navdp_tube's `navdp` package would shadow this project's `navdp/`).
The map carries terrain/obstacle information ONLY -- where objects (flags,
rover, antenna) are is remembered by the belief (nav/object_belief.py).

Frames: this project's world is Y-up with forward = (-sin yaw, -cos yaw) in
(X, Z); navdp_tube's planner frame is z-up with x = X, y = -Z and yaw0 = +x,
so planner pose = (X, -Z, yaw + pi/2) and body (forward, left) is the same.
"""

from __future__ import annotations

import math
import os
import secrets
import subprocess
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Optional

import numpy as np

from sam_vla.core.types import Action, GoalSpec, Observation

REPO = Path(__file__).resolve().parent.parent
RASTER_STEP_M = 0.25


def sample_terrain_raster(terrain, size_x: float, size_z: float, step: float = RASTER_STEP_M):
    """Sample the env's own terrain (any Terrain type) on a regular grid so the
    worker can do vectorised height lookups. -> (raster[nz, nx], x0, z0, step)."""
    x0, z0 = -size_x / 2.0, -size_z / 2.0
    nx, nz = int(round(size_x / step)) + 1, int(round(size_z / step)) + 1
    raster = np.empty((nz, nx), np.float32)
    for j in range(nz):
        z = z0 + j * step
        for i in range(nx):
            raster[j, i] = terrain(x0 + i * step, z)
    return raster, x0, z0, step


def _plan_frame(x: float, z: float, yaw: float) -> tuple[float, float, float]:
    return x, -z, yaw + math.pi / 2.0


class TubeGuidedPolicy:  # implements NavigationPolicy via act_verbose
    def __init__(
        self,
        terrain,
        size_x: float,
        size_z: float,
        start_xz: tuple[float, float],
        image_hw: tuple[int, int] = (480, 640),
        hfov_deg: float = 90.0,
        max_forward_speed: float = 0.6,
        max_yaw_rate: float = 0.6,
        robot_radius: float = 0.45,
        obstacle_range: float = 4.0,
        server: str = "http://127.0.0.1:8890",
        seg_checkpoint: Optional[str] = None,
        frame_every: int = 3,
        python: Optional[str] = None,
        worker_timeout: float = 240.0,
        navdp_root: Optional[str] = None,
        navdp_ckpt: Optional[str] = None,
        seg_classes: Optional[dict] = None,
        plan_timeout: float = 1.0,
    ):
        self._terrain = terrain
        self._size = (float(size_x), float(size_z))
        self._start_xz = (float(start_xz[0]), float(start_xz[1]))
        self.image_hw = image_hw
        self.hfov_deg = float(hfov_deg)
        self.max_forward_speed = float(max_forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.robot_radius = float(robot_radius)
        self.obstacle_range = float(obstacle_range)
        self.server = server
        self.seg_checkpoint = seg_checkpoint or str(REPO / "navdp_tube" / "best.pth")
        self.frame_every = max(int(frame_every), 1)
        self.python = python or sys.executable
        self.worker_timeout = float(worker_timeout)
        self.navdp_root = navdp_root
        self.navdp_ckpt = navdp_ckpt
        # map class roles for the worker (occupied_classes / free_classes /
        # rock_classes); None = the worker's sim defaults
        self.seg_classes = dict(seg_classes or {})
        self.plan_timeout = float(plan_timeout)
        self._server_proc: Optional[subprocess.Popen] = None
        self._goal_forward = 0.0
        self._goal_left = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._conn = None
        self._last_trajectory: Optional[np.ndarray] = None
        self._last_candidates: Optional[np.ndarray] = None
        self._last_values: Optional[np.ndarray] = None
        self._last_status: dict = {}
        self._frames_sent = 0
        self.last_error = ""

    # -- same surface as NavdpUpstreamPolicy ---------------------------------
    def set_goal_body(self, forward: float, left: float) -> None:
        self._goal_forward = float(forward)
        self._goal_left = float(left)

    def set_obstacle_pixels(self, obstacle_pixels) -> None:  # parity; unused
        pass

    def _server_up(self) -> bool:
        import socket
        from urllib.parse import urlparse

        u = urlparse(self.server)
        try:
            with socket.create_connection((u.hostname, u.port), timeout=0.5):
                return True
        except OSError:
            return False

    def _ensure_server(self) -> None:
        """Start the tube NavDP server (navdp_tube/tube_navdp, `navdp` conda env)
        unless something already listens on its port."""
        if self._server_up():
            return
        if not (self.navdp_root and self.navdp_ckpt):
            raise RuntimeError(
                f"tube NavDP server not running at {self.server} and no navdp_root/"
                "navdp_ckpt given to start it (see navdp_tube/bev_map/sim/start_tube_server.sh)"
            )
        from urllib.parse import urlparse

        py = os.environ.get("TUBE_SERVER_PYTHON") or str(
            Path(sys.executable).resolve().parents[3] / "envs" / "navdp" / "bin" / "python"
        )
        port = urlparse(self.server).port
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        self._server_proc = subprocess.Popen(
            [py, "scripts/tube_navdp_server.py", "--navdp-root", os.path.abspath(self.navdp_root),
             "--checkpoint", os.path.abspath(self.navdp_ckpt), "--port", str(port)],
            cwd=str(REPO / "navdp_tube" / "tube_navdp"), env=env,
        )
        t0 = time.time()
        while time.time() - t0 < 120 and not self._server_up():
            if self._server_proc.poll() is not None:
                raise RuntimeError("tube NavDP server exited during startup")
            time.sleep(0.5)

    def start(self) -> None:
        if self._conn is not None:
            return
        self._ensure_server()
        raster, x0, z0, step = sample_terrain_raster(self._terrain, *self._size)
        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        port = listener.address[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            [self.python, "-m", "nav.tube_worker", "--port", str(port), "--authkey", key.hex()],
            cwd=str(REPO), env=env,
        )
        listener._listener._socket.settimeout(self.worker_timeout)
        try:
            self._conn = listener.accept()
        finally:
            listener.close()
        h, w = self.image_hw
        sx, sy, _ = _plan_frame(self._start_xz[0], self._start_xz[1], 0.0)
        self._conn.send(("init", dict(
            width=w, height=h, hfov_deg=self.hfov_deg,
            raster=raster, x0=x0, z0=z0, step=step,
            start_xy=(sx, sy), seg_checkpoint=self.seg_checkpoint,
            max_speed=self.max_forward_speed, max_yaw_rate=self.max_yaw_rate,
            robot_radius=self.robot_radius, obstacle_range=self.obstacle_range,
            server=self.server, plan_timeout=self.plan_timeout, **self.seg_classes,
        )))
        self._expect("ready", self.worker_timeout)

    def stop(self) -> None:
        try:
            if self._conn is not None:
                self._conn.send(("quit", None))
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        if self._server_proc is not None:
            self._server_proc.terminate()
            self._server_proc = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    # -- pipe helpers ----------------------------------------------------------
    def _expect(self, kind: str, timeout: float):
        if not self._conn.poll(timeout):
            raise TimeoutError(f"tube worker gave no '{kind}' within {timeout:.0f}s")
        got, payload = self._conn.recv()
        if got == "error":
            raise RuntimeError(f"tube worker: {payload}")
        if got != kind:
            raise RuntimeError(f"tube worker sent {got!r}, expected {kind!r}")
        return payload

    def _call(self, msg, kind: str, timeout: float = 30.0):
        self._conn.send(msg)
        return self._expect(kind, timeout)

    def ego_map_view(self, obs: Observation):
        """Egocentric BEV map image (BGR) for the GUI, or None."""
        if self._conn is None:
            return None
        pose = self._pose_tuple(obs)
        return self._call(("map", pose), "map", 15.0)

    @staticmethod
    def _pose_tuple(obs: Observation):
        x, y, yaw = _plan_frame(obs.pose.x, obs.pose.z, obs.pose.yaw)
        return (x, y, yaw, float(obs.pose.y))

    def _goal_plan_frame(self, obs: Observation):
        yaw = obs.pose.yaw
        f, l = self._goal_forward, self._goal_left
        gx = obs.pose.x + f * -math.sin(yaw) + l * -math.cos(yaw)
        gz = obs.pose.z + f * -math.cos(yaw) + l * math.sin(yaw)
        return (gx, -gz)

    # -- act ---------------------------------------------------------------
    def act_verbose(
        self, obs: Observation, semantic, goal_spec: GoalSpec, step: int
    ) -> tuple[Action, dict]:
        if obs.depth is None:
            raise ValueError("TubeGuidedPolicy requires depth in the observation")
        self.start()
        pose = self._pose_tuple(obs)
        replanned = False
        if step % self.frame_every == 0 or self._frames_sent == 0:
            # map update on colour+depth+pose of the same moment (real rover:
            # depth lags colour, see RealRoverEnv obs.synced); steering below
            # still uses the current pose
            f_obs = getattr(obs, "synced", None) or obs
            depth = np.asarray(f_obs.depth, np.float32)
            ack = self._call(("frame", np.ascontiguousarray(f_obs.rgb[..., :3]), depth,
                              self._pose_tuple(f_obs), self._goal_plan_frame(obs)),
                             "ack", self.worker_timeout)
            self.last_error = ack.get("error", "")
            self._frames_sent += 1
            replanned = True
        res = self._call(("cmd", pose), "cmd", 10.0)
        self._last_status = res["status"]
        traj = res.get("trajectory")
        # current-body-frame path still ahead (tube_worker.cmd); None = no
        # path right now -- don't keep drawing an old one in a stale frame
        self._last_trajectory = traj if traj is not None and len(traj) else None
        # all sampled NavDP paths (+ critic values), current body frame
        self._last_candidates = res.get("candidates")
        self._last_values = res.get("values")
        v = float(np.clip(res["v"], 0.0, self.max_forward_speed))
        w = float(np.clip(res["w"], -self.max_yaw_rate, self.max_yaw_rate))
        action = Action(v_fwd=v, v_lat=0.0, yaw_rate=w)
        info = {
            "goal_forward": self._goal_forward,
            "goal_left": self._goal_left,
            "replanned": replanned,
            "trajectory_len": 0 if self._last_trajectory is None else int(len(self._last_trajectory)),
            "critic_value_max": None,
            "tube_state": self._last_status.get("state"),
            "tube_choice": self._last_status.get("choice"),
            "tube_error": self.last_error,
            "tube_diag": res.get("diag"),
        }
        return action, info
