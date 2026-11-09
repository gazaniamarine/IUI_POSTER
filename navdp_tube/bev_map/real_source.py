"""Real-rover frame source: camera stream (gstream.py) + telemetry -> the same Frame the simulator produces.

LiveSource    reads the rover camera (gstream.py, RTP/JPEG on UDP 5000) and telemetry.
ReplaySource  plays back a session recorded with --record (no rover needed; good for tuning).
Recorder      writes a session: rgb/*.jpg, frames.jsonl (arrival times), telemetry.jsonl (raw), camera.json.

Frames are undistorted and resized once (calibration from calibrate_camera.py), so the rest of the
pipeline sees an ideal pinhole camera with intrinsics K.
"""

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Frame:
    rgb: np.ndarray
    K: np.ndarray
    cam_pos: np.ndarray
    cam_rot: np.ndarray
    robot_xy: np.ndarray
    robot_yaw: float
    time: float
    depth: np.ndarray = None
    mask: np.ndarray = None


# ---------------------------------------------------------------- calibration
def calib_path(rcfg):
    p = rcfg.get("calibration", "calib/real_camera.yaml")
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def load_calibration(rcfg):
    p = calib_path(rcfg)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def save_calibration(rcfg, calib):
    p = calib_path(rcfg)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    return p


def hfov_K(width, height, hfov_deg):
    fx = width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


class Undistorter:
    """Native camera image -> undistorted, resized RGB with intrinsics K (built on the first frame)."""

    def __init__(self, rcfg, calib):
        self.out_w = int(rcfg.get("process_width", 960))
        self.hfov = rcfg.get("fallback_hfov_deg", 60.0)
        self.cam = calib.get("camera")
        self.size, self.maps, self.K = None, None, None

    @property
    def calibrated(self):
        return self.cam is not None

    def _build(self, w, h):
        s = self.out_w / w
        ow, oh = self.out_w, int(round(h * s))
        if self.cam is None:
            self.maps, self.K = None, hfov_K(ow, oh, self.hfov)
        else:
            K0 = np.asarray(self.cam["K"], float).copy()
            cw, ch = self.cam["width"], self.cam["height"]
            if (cw, ch) != (w, h):                       # stream resolution differs from calibration
                K0[0] *= w / cw
                K0[1] *= h / ch
            dist = np.asarray(self.cam.get("dist", [0, 0, 0, 0, 0]), float)
            newK, _ = cv2.getOptimalNewCameraMatrix(K0, dist, (w, h), 0.0)
            Ko = newK.copy()
            Ko[:2] *= s
            self.maps = cv2.initUndistortRectifyMap(K0, dist, None, Ko, (ow, oh), cv2.CV_16SC2)
            self.K = Ko
        self.size = (w, h)

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        if self.size != (w, h):
            self._build(w, h)
        if self.maps is None:
            return cv2.resize(rgb, (self.out_w, int(round(h * self.out_w / w))), interpolation=cv2.INTER_AREA)
        return cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR)


# ---------------------------------------------------------------- sources
class LiveSource:
    """Latest camera frame + its arrival time + telemetry (if enabled)."""

    def __init__(self, cfg, calib, telemetry=True):
        rc = cfg["real"]
        from gstream import CameraStream                  # gstream.py in this folder
        self.camera = CameraStream(port=rc.get("camera_port", 5000))
        self.undistort = Undistorter(rc, calib)
        self.latency = float(rc.get("camera_latency", 0.1))
        self.telemetry = None
        if telemetry:
            from telemetry import Telemetry
            self.telemetry = Telemetry(rc.get("telemetry_bind", "tcp://*:5557"))

    def next(self, timeout=5.0):
        """(rgb, K, t_arrival, telemetry sample or None); None if no frame arrives in time."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            bgr = self.camera.read()
            if bgr is not None:
                t = time.monotonic()
                rgb = self.undistort(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                s = self.telemetry.at(t - self.latency) if self.telemetry else None
                return rgb, self.undistort.K, t, s
            time.sleep(0.005)
        return None

    def close(self):
        self.camera.release()
        if self.telemetry:
            self.telemetry.close()


class ReplaySource:
    """Play back a recorded session. Pose is looked up with the *current* camera_latency, so the
    latency can be tuned offline (--set real.camera_latency=0.15)."""

    def __init__(self, cfg, run):
        from telemetry import lookup
        self.lookup, self.run = lookup, run
        self.latency = float(cfg["real"].get("camera_latency", 0.1))
        cam = json.load(open(f"{run}/camera.json"))
        self.K = np.asarray(cam["K"], float)
        self.frames = [json.loads(l) for l in open(f"{run}/frames.jsonl")]
        tp = f"{run}/telemetry.jsonl"
        self.tel = [(d["t"], d["msg"]) for d in map(json.loads, open(tp))] if os.path.exists(tp) else []
        self.i = 0

    def next(self, timeout=None):
        if self.i >= len(self.frames):
            return None
        m = self.frames[self.i]
        self.i += 1
        bgr = cv2.imread(f"{self.run}/rgb/{m['file']}")
        s = self.lookup(self.tel, m["t"] - self.latency) if self.tel else None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.K, m["t"], s

    def close(self):
        pass


class Recorder:
    def __init__(self, out, K, telemetry=None):
        self.out, self.n = out, 0
        os.makedirs(f"{out}/rgb", exist_ok=True)
        json.dump({"K": np.asarray(K).tolist()}, open(f"{out}/camera.json", "w"), indent=2)
        self.frames = open(f"{out}/frames.jsonl", "w")
        self.tel_f = open(f"{out}/telemetry.jsonl", "w") if telemetry else None
        self.telemetry, self.last_t = telemetry, -1.0

    def add(self, rgb, t):
        name = f"{self.n:06d}.jpg"
        cv2.imwrite(f"{self.out}/rgb/{name}", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.frames.write(json.dumps({"file": name, "t": t}) + "\n")
        if self.telemetry is not None:                  # raw telemetry received since the last frame
            with self.telemetry.lock:
                new = [(tt, m) for tt, m in self.telemetry.buf if tt > self.last_t]
            for tt, m in new:
                self.tel_f.write(json.dumps({"t": tt, "msg": m}) + "\n")
            if new:
                self.last_t = new[-1][0]
        self.n += 1

    def close(self):
        self.frames.close()
        if self.tel_f:
            self.tel_f.close()
