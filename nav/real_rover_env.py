"""Real-rover stand-in for MarsHabitatEnv, so nav/gui1.py + RoverController
(belief memory, human hand-over, Qwen point-goal, NavDP) drive the physical
rover instead of the sim.

Same surface RoverController uses from MarsHabitatEnv (get_observation,
step, get_semantic_frame, get_height_at_xz, ...), backed by the rover's zenoh
bridge (camera + depth in, cmd_vel out) -- the same wiring the validated
nav/real_rover_qwen35_navdp.py driver uses (connect / heartbeat / depth
guard are imported from there, not re-implemented).

Pose: the real rover has no ground-truth pose, so it is dead-reckoned from
the commanded twist over real wall-clock time. Good enough for the belief
memory over a few metres; it drifts over long runs (wheel slip is not seen).

step(new_pose): the controller integrates its action into a target pose
(integrate_mars with its dt). That pose delta is turned back into the
(v, w) it came from and sent to the rover, clamped to the real-rover caps
and zeroed forward if real depth says an obstacle is closer than
DEPTH_STOP_M. A delta far beyond the caps is a teleport (the GUI's Reset):
the odometry frame is re-anchored there and the rover is not moved.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from sam_vla.core.lifecycle import ServiceRegistry
from sam_vla.core.types import Observation, Pose

# real_rover_qwen35_navdp puts Nav_new on sys.path for nav_pipeline.zenoh_node
from nav.real_rover_qwen35_navdp import (  # noqa: E402
    DEPTH_STOP_M,
    HEARTBEAT_PERIOD_S,
    connect,
    min_forward_depth,
    serialize_twist,
    wait_for_frames,
)

REAL_HFOV_DEG = 55.5  # measured, this D435i at 640x480 (real_rover_qwen35_navdp.HFOV_DEG)
FRAME_W, FRAME_H = 640, 480


class _FlatTerrain:
    """Flat floor at y=0 (TubeGuidedPolicy samples env._terrain(x, z))."""

    def __call__(self, x, z):
        return 0.0

    def local_height_max(self, x, z, r=0.0):
        return 0.0


class RealRoverEnv:
    def __init__(
        self,
        endpoint: str,
        services: list = None,
        dt: float = 0.1,
        max_linear: float = 0.15,
        max_angular: float = 0.3,
        camera_height: float = 0.45,
        frame_timeout_s: float = 30.0,
        yaw_rate_scale: float = 1.0,
        linear_scale: float = 1.0,
        use_visual_gyro: bool = True,
    ):
        self.endpoint = endpoint
        self.dt = float(dt)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.camera_height = float(camera_height)
        self.frame_timeout_s = float(frame_timeout_s)
        # Calibration: the rover's REAL rate = scale x commanded rate. The
        # pose is dead-reckoned from commands (the IMU heading on /rover/rpm
        # is NaN on this rover), so without this a rover that physically
        # turns 2x faster than commanded overshoots every turn by 2x
        # (live 2026-11-09: "turn right" -> ~180 deg, "turn around" -> ~360).
        self.yaw_rate_scale = float(yaw_rate_scale)
        self.linear_scale = float(linear_scale)
        self._registry = ServiceRegistry()
        for s in services or []:
            self._registry.register(s)
        self._terrain = _FlatTerrain()
        self._session = self._pub = self._st = None
        self._cmd = (0.0, 0.0)
        self._cmd_lock = threading.Lock()
        self._stop = threading.Event()
        # Visual gyro (nav/visual_gyro.py): measured yaw from camera frames,
        # used for the heading while turning in place (see _advance_pose).
        self.use_visual_gyro = use_visual_gyro
        self._vis_lock = threading.Lock()
        self._vis_yaw = 0.0      # measured yaw since the last _advance_pose
        self._vis_frames = 0
        self._vis_fail = 0
        self._last_turn_t = -1e9  # last time a non-zero turn was commanded
        self._last_turn_w = 0.0
        self.vis_yaw_total = 0.0  # diagnostics
        self._vis_last_frame_t = -1e9
        self.vis_accepted = 0  # diagnostics: visual-gyro frames used / rejected
        self.vis_rejected = 0
        self._hb: Optional[threading.Thread] = None
        self._pose = Pose(x=0.0, y=self.camera_height, z=0.0, yaw=0.0)
        self._pose_t = time.monotonic()
        # pose handed out by the last get_observation: the controller
        # integrates its action FROM this pose, so step() must measure the
        # commanded delta from it too (not from the pose advanced since).
        self._obs_pose = self._pose
        from collections import deque

        self._pose_hist = deque(maxlen=60)  # (t, pose), ~3-6 s
        self.last_depth_stop = False
        self.last_forward_depth: Optional[float] = None

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self):
        self._session, self._pub, self._st = connect(self.endpoint)
        if not wait_for_frames(self._st, self.frame_timeout_s):
            self.close()
            raise RuntimeError(
                f"no camera/depth frames from the rover at {self.endpoint} "
                f"within {self.frame_timeout_s:.0f}s"
            )
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()
        if self.use_visual_gyro:
            threading.Thread(target=self._gyro_loop, daemon=True, name="visual-gyro").start()
        self._registry.start_all()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._set_cmd(0.0, 0.0)
        self._stop.set()
        if self._pub is not None:
            for _ in range(5):  # make sure the zero twist lands
                try:
                    self._pub.put(serialize_twist(0.0, 0.0))
                except Exception:
                    break
                time.sleep(0.05)
        self._registry.stop_all()
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _heartbeat(self):
        # rover firmware zeroes velocity if no cmd_vel for ~500 ms
        # The depth guard is re-checked HERE on the newest depth frame, not
        # only in step(): while the controller is blocked in a Qwen/NavDP call
        # (seconds), step() isn't called but the last twist keeps being sent.
        while not self._stop.is_set():
            with self._cmd_lock:
                lin, ang = self._cmd
            if lin > 0.0:
                with self._st.lock:
                    d = self._st.depth
                fd = min_forward_depth(d)
                if fd is not None and fd < DEPTH_STOP_M:
                    lin = 0.0
                    self.last_depth_stop = True
                    with self._cmd_lock:
                        self._cmd = (0.0, self._cmd[1])
            try:
                self._pub.put(serialize_twist(lin, ang))
            except Exception:
                pass
            time.sleep(HEARTBEAT_PERIOD_S)

    def _set_cmd(self, lin, ang):
        with self._cmd_lock:
            self._cmd = (float(lin), float(ang))

    # -- odometry ------------------------------------------------------------
    def _gyro_loop(self):
        """Measure yaw between every pair of consecutive camera frames."""
        from nav.real_rover_env import REAL_HFOV_DEG
        from nav.visual_gyro import VisualGyro

        gyro, last = VisualGyro(REAL_HFOV_DEG), None
        while not self._stop.is_set():
            with self._st.lock:
                rgb = self._st.rgb
            if rgb is None or rgb is last:
                time.sleep(0.01)
                continue
            last = rgb
            try:
                y = gyro.update(np.ascontiguousarray(rgb[..., :3]))
            except Exception:
                y = None
            t_now = time.monotonic()
            with self._vis_lock:
                dt_f = min(max(t_now - self._vis_last_frame_t, 0.0), 0.5)
                self._vis_last_frame_t = t_now
                if y is not None:
                    # Plausibility gate: a frame's rotation can't exceed what the
                    # rover can physically do since the last frame (2.5x the
                    # commanded rate x calibration, + 1 deg). A bad fit (moving
                    # people, glass reflections, low texture) gave ~80-100 deg
                    # heading jumps while the rover stood still (live log
                    # 20261109-154300: door "78 deg right" 2 s after "left of
                    # centre") -> unnecessary hard turns. Rejected frames fall
                    # back to the command estimate.
                    with self._cmd_lock:
                        w_cmd = abs(self._cmd[1])
                    if t_now - self._last_turn_t < 0.6:  # coasting after a turn command
                        w_cmd = max(w_cmd, self._last_turn_w)
                    allowed = w_cmd * self.yaw_rate_scale * dt_f * 2.5 + math.radians(1.0)
                    if abs(y) > allowed:
                        self.vis_rejected += 1
                        y = None
                    else:
                        self.vis_accepted += 1
                if y is None:
                    self._vis_fail += 1
                else:
                    self._vis_yaw += y
                    self._vis_frames += 1

    def _advance_pose(self):
        """Advance the pose over the real time since the last update. Heading:
        while turning in place (no forward command), the camera-MEASURED yaw
        (visual gyro); otherwise, or when the camera can't measure it (too
        little texture), dead reckoning from the command x yaw_rate_scale."""
        now = time.monotonic()
        el = min(now - self._pose_t, 1.0)  # a stalled loop must not jump the pose
        self._pose_t = now
        with self._cmd_lock:
            v, w = self._cmd
        v, w = v * self.linear_scale, w * self.yaw_rate_scale  # real rates
        with self._vis_lock:
            vis_yaw, vis_n, vis_fail = self._vis_yaw, self._vis_frames, self._vis_fail
            self._vis_yaw, self._vis_frames, self._vis_fail = 0.0, 0, 0
            frames_live = now - self._vis_last_frame_t < 0.3
        if w != 0.0:
            self._last_turn_t = now
            self._last_turn_w = abs(w) / max(self.yaw_rate_scale, 1e-6)  # commanded units
        in_place = abs(v) < 0.02
        # turning, or still coasting <0.6 s after a turn command ended
        rotating = w != 0.0 or now - self._last_turn_t < 0.6
        # Measured yaw whenever the rover moves at all (turning in place AND
        # steering while driving: at <=0.15 m/s a frame moves the camera
        # <1 cm, whose parallax the homography absorbs as zoom, not yaw).
        moving = rotating or not in_place
        if self.use_visual_gyro and moving and frames_live and (vis_n + vis_fail) > 0:
            # measured; the share of frames that couldn't be measured (too
            # little texture) is filled in from the command
            dyaw = vis_yaw + w * el * vis_fail / float(vis_n + vis_fail)
            self.vis_yaw_total += vis_yaw
        elif self.use_visual_gyro and moving and frames_live:
            dyaw = 0.0  # no new frame this tick: the next frame carries this rotation
        elif in_place and not rotating:
            dyaw = 0.0  # standing still: don't integrate camera noise
        else:
            dyaw = w * el
        p = self._pose
        yaw = p.yaw + dyaw
        self._pose = Pose(
            x=p.x - v * math.sin(p.yaw) * el,
            y=self.camera_height,
            z=p.z - v * math.cos(p.yaw) * el,
            yaw=(yaw + math.pi) % (2.0 * math.pi) - math.pi,
        )
        self._pose_hist.append((now, self._pose))

    # -- MarsHabitatEnv surface ---------------------------------------------
    @staticmethod
    def _fit(rgb, depth):
        rgb = np.ascontiguousarray(rgb[..., :3])
        if rgb.shape[:2] != (FRAME_H, FRAME_W):
            rgb = cv2.resize(rgb, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        if depth is not None and depth.shape[:2] != (FRAME_H, FRAME_W):
            depth = cv2.resize(depth, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        return rgb, depth

    def _pose_at(self, t):
        """Pose closest in time to t (pose history, ~3 s)."""
        best = min(self._pose_hist, key=lambda e: abs(e[0] - t), default=None)
        return self._pose if best is None else best[1]

    def get_observation(self, frame_idx: int) -> Observation:
        self._advance_pose()
        with self._st.lock:
            rgb, depth, depth_t = self._st.rgb, self._st.depth, getattr(self._st, "depth_t", 0.0)
            hist = list(getattr(self._st, "rgb_hist", ()))
        rgb, depth = self._fit(rgb, depth)
        self.last_forward_depth = min_forward_depth(depth)
        self._obs_pose = self._pose
        obs = Observation(rgb=rgb, depth=depth, pose=self._pose, frame_idx=frame_idx)
        # Depth arrives ~2 Hz, colour ~15 Hz: the newest colour frame and the
        # newest depth frame are up to ~0.5 s apart, so after any turn the
        # target's pixels get the depth of something else (live: a door
        # measured at 3, 7, 11, 16 m and once BEHIND the start point).
        # obs.synced pairs the depth frame with the colour frame and pose of
        # the same moment -- used for everything that turns pixels + depth
        # into world positions (target finding, the tube map).
        if depth is not None and hist and depth_t > 0:
            t_rgb, rgb_d = min(hist, key=lambda e: abs(e[0] - depth_t))
            if abs(t_rgb - depth_t) < 0.1:
                rgb_d, _ = self._fit(rgb_d, None)
                obs.synced = Observation(rgb=rgb_d, depth=depth, pose=self._pose_at(depth_t),
                                         frame_idx=frame_idx)
        return obs

    def step(self, new_pose: Pose) -> None:
        # The controller's new_pose = obs pose + action * dt. Recover that
        # action from the delta against the SAME obs pose. (Measuring it from
        # the pose advanced during the tick's NavDP call made every slow tick
        # look like "target is behind me" -> reverse, next tick forward:
        # the live forward/backward jitter.)
        self._advance_pose()
        p = self._obs_pose
        dx, dz = new_pose.x - p.x, new_pose.z - p.z
        fwd = dx * -math.sin(p.yaw) + dz * -math.cos(p.yaw)
        dyaw = (new_pose.yaw - p.yaw + math.pi) % (2.0 * math.pi) - math.pi
        v, w = fwd / self.dt, dyaw / self.dt
        if abs(v) > 5.0 * max(self.max_linear, 0.5) or abs(w) > 5.0 * max(self.max_angular, 1.0):
            # teleport (Reset to spawn): re-anchor odometry, don't drive
            self._set_cmd(0.0, 0.0)
            self._pose = Pose(x=new_pose.x, y=self.camera_height, z=new_pose.z, yaw=new_pose.yaw)
            return
        v = max(-self.max_linear, min(self.max_linear, v))
        w = max(-self.max_angular, min(self.max_angular, w))
        fd = self.last_forward_depth
        self.last_depth_stop = v > 0.0 and fd is not None and fd < DEPTH_STOP_M
        if self.last_depth_stop:
            v = 0.0  # hard collision guard on REAL depth, whatever the planner says
        self._set_cmd(v, w)

    def stop_motion(self) -> None:
        self._set_cmd(0.0, 0.0)

    def get_semantic_frame(self) -> np.ndarray:
        return np.zeros((FRAME_H, FRAME_W), dtype=np.int32)  # no mask meshes in the real world

    def get_height_at_xz(self, x: float, z: float) -> float:
        return self.camera_height

    def get_mesh_overlay_rgb(self):
        return None

    def get_topdown_rgb(self):
        return None

    def look_down_rgb(self):
        with self._st.lock:
            return None if self._st.rgb is None else np.ascontiguousarray(self._st.rgb[..., :3])

    def register_object_mask(self, *args, **kwargs):
        return None

    def remove_object_mask(self, obj) -> None:
        return None
