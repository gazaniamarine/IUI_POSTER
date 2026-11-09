"""SERVER SIDE: receive rover telemetry (rover_bridge/rover_state_bridge.py) and look up the pose
at a camera frame's time.

All timing uses this machine's clock (time of arrival), so the rover and server clocks do not need
to agree. A frame's pose is interpolated at (frame arrival - camera_latency).
"""

import json
import threading
import time
from collections import deque

import numpy as np
import zmq

LINEAR = ("x", "y", "roll", "pitch", "vx", "wz", "gx", "gy", "gz")


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interpolate(s0, s1, w):
    """Blend two telemetry samples (w = 0 -> s0, 1 -> s1). yaw is interpolated on the circle."""
    out = dict(s1)
    for k in LINEAR:
        if s0.get(k) is not None and s1.get(k) is not None:
            out[k] = (1 - w) * s0[k] + w * s1[k]
    if s0.get("yaw") is not None and s1.get("yaw") is not None:
        out["yaw"] = float(_wrap(s0["yaw"] + w * _wrap(s1["yaw"] - s0["yaw"])))
    return out


def lookup(buf, t):
    """buf: time-sorted list of (t_recv, sample). Sample at time t (clamped to the buffer ends)."""
    if not buf:
        return None
    if t <= buf[0][0]:
        return dict(buf[0][1], _age=buf[0][0] - t)
    if t >= buf[-1][0]:
        return dict(buf[-1][1], _age=t - buf[-1][0])
    lo, hi = 0, len(buf) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if buf[mid][0] <= t else (lo, mid)
    (t0, s0), (t1, s1) = buf[lo], buf[hi]
    w = (t - t0) / max(t1 - t0, 1e-6)
    return dict(interpolate(s0, s1, w), _age=0.0)


class Telemetry:
    def __init__(self, bind="tcp://*:5557", keep_s=5.0):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.setsockopt(zmq.RCVTIMEO, 200)
        self.sock.bind(bind)
        self.buf = deque()
        self.keep = keep_s
        self.lock = threading.Lock()
        self.running = True
        self.n = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                msg = json.loads(self.sock.recv_string())
            except zmq.Again:
                continue
            except (ValueError, zmq.ZMQError):
                continue
            now = time.monotonic()
            with self.lock:
                self.buf.append((now, msg))
                self.n += 1
                while self.buf and now - self.buf[0][0] > self.keep:
                    self.buf.popleft()

    def at(self, t):
        """Telemetry at monotonic time t; '_age' = how far t is outside the received data (s)."""
        with self.lock:
            buf = list(self.buf)
        return lookup(buf, t)

    def latest_age(self):
        with self.lock:
            return time.monotonic() - self.buf[-1][0] if self.buf else float("inf")

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.sock.close(0)
        self.ctx.term()
