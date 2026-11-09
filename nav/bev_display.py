"""GUI-side driver for LemNAV's monocular BEV map (nav/bev_worker.py).

A background thread (like LemNAV's BevDisplay): every period_s it takes the
controller's newest raw camera frame + pose, skips it while the rover turns
fast (pose and image disagree), sends it to the BEV worker and keeps the
returned PNG for the GUI. Display only -- nothing here reaches the planner,
the safety stop, or the belief memory.

Pose conversion (this project's world, Y-up, forward = (-sin yaw, -cos yaw)
in (X, Z)) -> LemNAV's z-up odometry (x, y, theta), forward = (cos, sin):
x = X, y = -Z, theta = yaw + pi/2 (same mapping as nav/tube_guided_policy.py).
"""

from __future__ import annotations

import math
import os
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
TURN_SKIP_RAD_S = 0.3  # LemNAV BevDisplay: skip frames while turning faster than this


def _planner_pose(pose):
    return (float(pose.x), -float(pose.z), float(pose.yaw) + math.pi / 2.0)


class BevMapDisplay(threading.Thread):
    def __init__(self, source: Callable, hfov_fn: Callable[[], float], cam_height: float,
                 cam_forward: float = 0.07, period_s: float = 0.5, range_m: float = 8.0,
                 python: Optional[str] = None, timeout_s: float = 180.0):
        """source() -> (rgb, pose) or None: the controller's newest raw frame."""
        super().__init__(daemon=True, name="bev-map-display")
        self.source, self.hfov_fn = source, hfov_fn
        self.cam_height, self.cam_forward = float(cam_height), float(cam_forward)
        self.period_s, self.range_m = float(period_s), float(range_m)
        self.python = python or os.environ.get("BEV_PYTHON") or str(
            Path(sys.executable).resolve().parents[3] / "envs" / "navdp" / "bin" / "python")
        self.timeout_s = float(timeout_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._png: Optional[bytes] = None
        self._png_t = 0.0
        self._theta = None  # LemNAV-frame heading of the newest map, for the arrow
        self.status = "starting BEV map (loading Depth Anything)..."
        self._clear = threading.Event()
        self._proc = self._conn = None

    # -- for the GUI ------------------------------------------------------------
    def latest(self):
        """(png bytes or None, age_s or None, theta_rad or None, status text)."""
        with self._lock:
            age = time.time() - self._png_t if self._png_t else None
            return self._png, age, self._theta, self.status

    def request_clear(self) -> None:
        self._clear.set()

    def stop(self) -> None:
        self._stop.set()

    # -- worker -------------------------------------------------------------------
    def _start_worker(self, center_xy):
        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            [self.python, "-m", "nav.bev_worker", "--port", str(listener.address[1]), "--authkey", key.hex()],
            cwd=str(REPO), env=env)
        listener._listener._socket.settimeout(self.timeout_s)
        try:
            self._conn = listener.accept()
        finally:
            listener.close()
        self._conn.send(("init", dict(cam_height=self.cam_height, cam_forward=self.cam_forward,
                                      range_m=self.range_m, center_xy=center_xy, extent_m=30.0)))
        self._expect("ready", self.timeout_s)

    def _expect(self, kind, timeout):
        if not self._conn.poll(timeout):
            raise TimeoutError(f"BEV worker gave no '{kind}' within {timeout:.0f}s")
        got, payload = self._conn.recv()
        if got == "error":
            raise RuntimeError(payload)
        if got != kind:
            raise RuntimeError(f"BEV worker sent {got!r}, expected {kind!r}")
        return payload

    def run(self) -> None:
        last_theta = last_t = None
        try:
            while not self._stop.is_set():  # wait for the first frame: the grid is centred on it
                src = self.source()
                if src is not None:
                    break
                time.sleep(0.2)
            else:
                return
            x, y, _ = _planner_pose(src[1])
            self._start_worker((x, y))
            self.status = "BEV map running"
            while not self._stop.wait(self.period_s):
                if self._clear.is_set():
                    self._clear.clear()
                    self._conn.send(("clear", None))
                    self._expect("ok", 10.0)
                src = self.source()
                if src is None:
                    continue
                rgb, pose = src
                x, y, th = _planner_pose(pose)
                now = time.time()
                if last_theta is not None and now > last_t:
                    dth = math.atan2(math.sin(th - last_theta), math.cos(th - last_theta))
                    turning = abs(dth) / (now - last_t) > TURN_SKIP_RAD_S
                else:
                    turning = False
                last_theta, last_t = th, now
                if turning:
                    continue
                h, w = rgb.shape[:2]
                f = w / (2.0 * math.tan(math.radians(self.hfov_fn()) / 2.0))
                self._conn.send(("frame", rgb, (x, y, th), (f, f, w / 2.0, h / 2.0)))
                out = self._expect("png", 30.0)
                with self._lock:
                    if out["png"]:
                        self._png, self._png_t, self._theta = out["png"], time.time(), th
                    info = out.get("info") or {}
                    self.status = ("BEV map running" if out["fused"]
                                   else "BEV map: frame skipped (floor fit failed)")
                    if info.get("error"):
                        self.status = "BEV map: frame skipped (" + str(info["error"])[:60] + ")"
                    if "floor_inliers" in info and info["floor_inliers"] is not None:
                        self.status += f" -- floor inliers {info['floor_inliers']:.0%}"
        except Exception as e:  # a display thread must never take the GUI down
            with self._lock:
                self.status = f"BEV map stopped: {type(e).__name__}: {str(e)[:120]}"
            print(f"[bev-map] {self.status}", flush=True)
        finally:
            try:
                if self._conn is not None:
                    self._conn.send(("quit", None))
                    self._conn.close()
            except Exception:
                pass
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()


class BevMapWindow:
    """Separate window showing the BEV map (LemNAV gui/view/render.py
    _draw_bev): grey unknown / white free / black occupied, rover-centred,
    north (+y) up, green dot + heading arrow for the rover."""

    SIZE = 360

    def __init__(self, root, display: BevMapDisplay, refresh_ms: int = 300):
        import tkinter as tk

        self.root, self.display, self.refresh_ms = root, display, int(refresh_ms)
        self.win = tk.Toplevel(root)
        self.win.title("BEV map (display only)")
        self.win.configure(bg="#111827")
        self.canvas = tk.Canvas(self.win, width=self.SIZE, height=self.SIZE, bg="#1f2937", highlightthickness=0)
        self.canvas.pack(padx=8, pady=(8, 4))
        row = tk.Frame(self.win, bg="#111827")
        row.pack(fill="x", padx=8, pady=(0, 8))
        self.label = tk.Label(row, text="", fg="#cbd5e1", bg="#111827", anchor="w", justify="left",
                              wraplength=self.SIZE - 70)
        self.label.pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Clear", command=display.request_clear).pack(side="right")
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)  # hide, keep mapping
        self._photo = None
        self._tick()

    def _tick(self):
        import io

        from PIL import Image, ImageTk

        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return
        png, age, theta, status = self.display.latest()
        c, size = self.canvas, self.SIZE
        c.delete("all")
        stale = age is None or age > 3.0
        if png:
            try:
                img = Image.open(io.BytesIO(png)).convert("L").transpose(Image.FLIP_TOP_BOTTOM)
                img = img.resize((size, size), Image.NEAREST)
                if stale:
                    img = Image.eval(img, lambda p: int(p * 0.5))
                self._photo = ImageTk.PhotoImage(img)
                c.create_image(0, 0, anchor="nw", image=self._photo)
            except Exception:
                pass
        cx = cy = size / 2
        c.create_line(cx, 0, cx, size, fill="#333")
        c.create_line(0, cy, size, cy, fill="#333")
        if png and theta is not None:
            c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#00e676", outline="")
            c.create_line(cx, cy, cx + 14 * math.cos(theta), cy - 14 * math.sin(theta), fill="#00e676", width=2)
        rng = self.display.range_m
        txt = status
        if age is not None:
            txt += f"\n{'[stale] ' if stale else ''}updated {age:.0f}s ago  ·  ±{rng:.0f} m"
        self.label.configure(text=txt)
        self.root.after(self.refresh_ms, self._tick)
