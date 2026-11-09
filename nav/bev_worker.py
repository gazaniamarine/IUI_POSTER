"""Worker process for LemNAV's monocular BEV map (display only).

Runs Nihil/LemNAV's own BevOccupancy (Depth Anything V2 metric depth,
corrected every frame against the known floor geometry, fused into a log-odds
grid) and render_png, imported from LemNAV -- nothing is copied. It is the
same map launch_rover1_graph.sh shows with BEV_DISPLAY=1: it never feeds the
planner or any stop decision, it is only drawn in the GUI.

Separate process (`navdp` conda env, torch) because LemNAV's package is
called `nav_pipeline`, the same name as the Nav_new package the real-rover
GUI already imports for its zenoh helpers.

Protocol (multiprocessing.connection, pickled tuples). Pose is z-up
(x, y, theta) with forward = (cos theta, sin theta), as LemNAV's odometry:
  ("init", cfg)                     -> ("ready", None)
  ("frame", rgb, pose, intrinsics)  -> ("png", {"png": bytes|None, "fused": bool, "info": dict})
      intrinsics = (fx, fy, cx, cy)
  ("clear", None)                   -> ("ok", None)
  ("quit", None)
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing.connection import Client

LEMNAV = os.environ.get("LEMNAV_ROOT", "/home/gpu/Desktop/Nihil/LemNAV")
if LEMNAV not in sys.path:
    sys.path.insert(0, LEMNAV)


class Worker:
    def __init__(self, cfg):
        from nav_pipeline.bev_display import render_png  # LemNAV
        from nav_pipeline.bev_occupancy import BevOccupancy  # LemNAV
        from nav_pipeline.depth_estimator import MetricDepthEstimator  # LemNAV

        self.render_png = render_png
        self.range_m = float(cfg.get("range_m", 8.0))
        depther = MetricDepthEstimator(encoder=cfg.get("encoder", "vits"), device=cfg.get("device", "cuda:0"))
        # same construction as LemNAV's BevDisplay (nav_pipeline/bev_display.py)
        self.bev = BevOccupancy(
            tuple(cfg.get("center_xy", (0.0, 0.0))), extent_m=float(cfg.get("extent_m", 30.0)), res=0.10,
            depth_fn=lambda frames: [depther.estimate(f) for f in frames],
            cam_height=float(cfg["cam_height"]), cam_forward=float(cfg.get("cam_forward", 0.07)),
            min_period_s=0.0,
        )

    def frame(self, rgb, pose, intrinsics):
        try:
            fused = self.bev.integrate(None, tuple(pose), tuple(intrinsics), rgb=rgb)
        except Exception as e:
            # LemNAV's floor fit can fail on a frame (live: scipy "`x0` is
            # infeasible" -- its start guess fell outside its own bounds on a
            # poor floor view). Skip this frame, reset the fit state so the
            # next frame starts fresh, keep the map.
            joint = getattr(getattr(self.bev, "_bev", None), "joint", None)
            if joint is not None:
                try:
                    joint.reset()
                except Exception:
                    pass
            png = self.render_png(self.bev, self.range_m, origin=(float(pose[0]), float(pose[1])))
            return {"png": png, "fused": False, "info": {"error": f"{type(e).__name__}: {e}"[:120]}}
        png = self.render_png(self.bev, self.range_m, origin=(float(pose[0]), float(pose[1])))
        info = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in (self.bev.last_info or {}).items()}
        return {"png": png, "fused": bool(fused), "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--authkey", required=True)
    args = ap.parse_args()
    conn = Client(("127.0.0.1", args.port), authkey=bytes.fromhex(args.authkey))
    worker = None
    try:
        while True:
            msg = conn.recv()
            kind = msg[0]
            try:
                if kind == "init":
                    worker = Worker(msg[1])
                    conn.send(("ready", None))
                elif kind == "frame":
                    conn.send(("png", worker.frame(*msg[1:])))
                elif kind == "clear":
                    worker.bev.clear()
                    conn.send(("ok", None))
                elif kind == "quit":
                    break
            except Exception as e:  # report, keep serving
                import traceback

                conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
