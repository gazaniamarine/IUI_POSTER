"""Worker process for the segmentation-BEV + tube-guided NavDP planner.

Runs navdp_tube's SegBEVMap (best.pth SAM2 terrain segmentation -> occupancy
grid) and TubeNavigator (NavDP score-guided away from the map's obstacles, via
the tube server on :8890) in their own interpreter. Why a separate process:
navdp_tube/tube_navdp ships a package literally named `navdp`, which would
shadow this project's own `navdp/` (the CBF layer imports navdp.extensions
lazily) if imported into the GUI process.

Protocol (multiprocessing.connection, pickled tuples), planner frame
(z up: x = world X, y = -world Z, yaw = world yaw + pi/2, see TubeGuidedPolicy):
  ("init", cfg)                       -> ("ready", None)
  ("frame", rgb, depth, pose, goal)   -> ("ack", {"plans": n, "error": str})
      pose=(x, y, yaw, cam_h): update the segmentation map, (re)set the goal,
      start an async plan.
  ("cmd", pose)                       -> ("cmd", {"v","w","status","trajectory"})
  ("map", pose)                       -> ("map", {"ego": HxWx3 uint8 BGR, "seg": ...})
  ("quit", None)

Launched by TubeGuidedPolicy.start() as `python -m nav.tube_worker`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from multiprocessing.connection import Client
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUBE_DIR = os.path.join(REPO, "navdp_tube")
BEV_DIR = os.path.join(TUBE_DIR, "bev_map")
for _p in (os.path.join(BEV_DIR, "habitat"), BEV_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# "ours" in navdp_tube/bev_map/habitat/run_mars_habitat.py: NavDP score-guided
# away from the map's obstacles, no tubes.
OURS_NAV = [
    "nav.planner=navdp", "nav.mode=obstacle", "nav.pure_navdp=false",
    "nav.controller=track", "nav.tube_split=1", "nav.tube_lyapunov=min",
    "nav.tube_smooth=2",
]


class RasterScene:
    """What SegBEVMap needs from the 'scene': height_at(x, y) (vectorised, planner
    frame) and a .grid with the map's extent. Heights come from a raster the
    GUI process sampled from the env's own terrain (any terrain type)."""

    def __init__(self, raster, x0, z0, step, map_res=0.05):
        self.raster = np.asarray(raster, np.float32)  # [nz, nx], world X along columns, world Z along rows
        self.x0, self.z0, self.step = float(x0), float(z0), float(step)
        nz, nx = self.raster.shape
        x_max, z_max = self.x0 + (nx - 1) * self.step, self.z0 + (nz - 1) * self.step
        # planner y = -Z
        y_min, y_max = -z_max, -self.z0
        self.grid = SimpleNamespace(
            x_min=self.x0, y_min=y_min, res=map_res,
            nx=int(round((x_max - self.x0) / map_res)),
            ny=int(round((y_max - y_min) / map_res)),
        )

    def height_at(self, x, y):
        X = np.asarray(x, np.float64)
        Z = -np.asarray(y, np.float64)
        fx = np.clip((X - self.x0) / self.step, 0, self.raster.shape[1] - 1.001)
        fz = np.clip((Z - self.z0) / self.step, 0, self.raster.shape[0] - 1.001)
        ix, iz = fx.astype(np.int64), fz.astype(np.int64)
        ax, az = fx - ix, fz - iz
        r = self.raster
        return ((r[iz, ix] * (1 - ax) + r[iz, ix + 1] * ax) * (1 - az)
                + (r[iz + 1, ix] * (1 - ax) + r[iz + 1, ix + 1] * ax) * az)


def _intrinsics(width, height, hfov_deg):
    f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


class Worker:
    def __init__(self, cfg):
        from bev_pipeline import load_config  # noqa: E402
        from seg_bev_map import SegBEVMap, add_args  # noqa: E402
        from tube_nav import TubeNavigator  # noqa: E402
        import argparse as ap_

        self.cfg = cfg
        self.W, self.H = cfg["width"], cfg["height"]
        self.K = _intrinsics(self.W, self.H, cfg["hfov_deg"])
        self.scene = RasterScene(cfg["raster"], cfg["x0"], cfg["z0"], cfg["step"])
        ap = ap_.ArgumentParser()
        add_args(ap)
        margs = ap.parse_args([
            "--range", str(cfg.get("range", 3.5)),
            "--min-conf", str(cfg.get("min_conf", 0.0)),
            # Bedrock is drivable here: only sand (and bigrock, via --rocks)
            # is an obstacle; soil and bedrock are free ground.
            "--occupied", cfg.get("occupied_classes", "sand"),
            "--free", cfg.get("free_classes", "soil,bedrock"),
            # The model labels big pale ground patches "bigrock" (live: whole
            # patches boxed the rover in), so that label is NOT an obstacle
            # here. Real standing rocks / flags / the rover are still caught
            # by their height above the terrain (SegBEVMap's depth objects).
            "--rocks", cfg.get("rock_classes", ""),
        ])
        sx, sy = cfg["start_xy"]
        self.bev = SegBEVMap(self.scene, self.K, self.W, self.H, (sx, sy), margs,
                             overrides=[f"seg.checkpoint={cfg['seg_checkpoint']}"])
        nav_cfg = load_config(overrides=OURS_NAV + [
            f"nav.max_speed={cfg['max_speed']}", f"nav.max_yaw_rate={cfg['max_yaw_rate']}",
            "nav.stop_distance=0.0", f"nav.robot_radius={cfg['robot_radius']}",
            "nav.follow_lookahead_m=0.5", "nav.max_accel=1.0", "nav.max_yaw_accel=3.0",
            f"nav.obstacle_range={cfg.get('obstacle_range', 4.0)}",
            f"nav.server={cfg['server']}",
            # how old a plan may get before the navigator stops the rover
            f"nav.plan_timeout={cfg.get('plan_timeout', 1.0)}",
        ])
        self.nc = nav_cfg["nav"]
        self.nav = TubeNavigator(self.nc)
        self.goal = None
        self.plans = 0
        self.last_world_map = None

    def _ensure_goal(self, goal):
        if self.goal is None or math.hypot(goal[0] - self.goal[0], goal[1] - self.goal[1]) > 2.5:
            self.nav.set_goal(goal, self.K)  # resets NavDP's frame memory
            self.goal = goal
        else:
            self.nav.goal = (float(goal[0]), float(goal[1]))
            self.goal = goal

    def frame(self, rgb, depth, pose, goal):
        x, y, yaw, cam_h = pose
        self._ensure_goal(goal)
        self.bev.update(rgb, depth, x, y, yaw, cam_h)
        world_map = self.bev.world_map(x, y, goal, self.nc)
        self.last_world_map = world_map
        dbg = os.environ.get("TUBE_DEBUG_DIR")
        if dbg:
            import json

            from occupancy_grid import OCCUPIED  # noqa: E402

            g = self.bev.grid
            st = g.state()
            ix, iy, ok = g.world_to_cell(np.array([x - self.bev.sx]), np.array([y - self.bev.sy]))
            under = int(st[iy[0], ix[0]]) if ok[0] else -1
            lab = self.bev.last[1]["labels"] if self.bev.last else None
            hist = (np.bincount(lab[self.bev.last[0].mask].ravel(), minlength=4).tolist()
                    if lab is not None else None)
            os.makedirs(dbg, exist_ok=True)
            with open(os.path.join(dbg, "log.jsonl"), "a") as f:
                f.write(json.dumps({
                    "n": self.plans, "xy": [round(float(x), 2), round(float(y), 2)],
                    "occupied_cells": int((st == OCCUPIED).sum()),
                    "cell_under_rover": under, "seg_px_soil_bed_sand_rock": hist,
                    "world_map_occ": int(world_map.occupied.sum()),
                    "status": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                               for k, v in self.nav.status.items()},
                }) + "\n")
        if dbg and self.plans % 4 == 0:
            import cv2

            v = self.view((x, y, yaw, cam_h))
            seg = cv2.resize(v["seg"], (480, 360))
            ego = cv2.resize(v["ego"], (360, 360))
            pad = np.full((360, 480 + 360, 3), 110, np.uint8)
            pad[:, :480] = seg
            pad[:, 480:] = ego
            os.makedirs(dbg, exist_ok=True)
            cv2.imwrite(os.path.join(dbg, f"map_{self.plans:04d}.png"), pad)
        started = self.nav.plan_async(rgb, depth, (x, y, yaw), time.monotonic(), world_map)
        if started:
            self.plans += 1
        return {"plans": self.plans, "error": self.nav.error}

    def cmd(self, pose):
        x, y, yaw = pose[:3]
        if self.nav.goal is None:
            return {"v": 0.0, "w": 0.0, "status": {"state": "idle"}, "trajectory": None}
        v, w, st = self.nav.command((x, y, yaw), time.monotonic())
        # For the GUI: the path the follower is actually driving (anchored in
        # the world at the planning pose), still ahead of the rover, in the
        # rover's CURRENT body frame. (res.trajectory is in the body frame of
        # the planning pose -- older, the depth frame's -- so drawing it as if
        # it were relative to the rover now showed a rotated/shifted path.)
        traj = None
        fol = getattr(self.nav.driver, "follower", None) if self.nav.driver is not None else None
        if fol is not None and fol.path is not None and len(fol.path) > fol.idx + 1:
            p = np.asarray(fol.path[fol.idx:], np.float64)
            c, s_ = math.cos(yaw), math.sin(yaw)
            dx, dy = p[:, 0] - x, p[:, 1] - y
            traj = np.stack([c * dx + s_ * dy, -s_ * dx + c * dy], 1).astype(np.float32)
        cands, values = self._candidates_now(x, y, yaw)
        return {"v": float(v), "w": float(w), "status": dict(st), "trajectory": traj,
                "error": self.nav.error, "diag": self._diag(st, traj),
                "candidates": cands, "values": values}

    def _candidates_now(self, x, y, yaw):
        """Every path NavDP sampled in the last plan ([N, H, 2], body frame of
        the PLANNING pose) -> the rover's current body frame, for drawing."""
        drv = self.nav.driver
        res = getattr(drv, "result", None) if drv is not None else None
        cands = None if res is None else getattr(res, "candidates", None)
        pp = None if drv is None else drv.plan_pose
        if cands is None or pp is None or len(cands) == 0:
            return None, None
        c = np.asarray(cands, np.float64)[..., :2]
        cp, sp = math.cos(pp[2]), math.sin(pp[2])
        wx = pp[0] + cp * c[..., 0] - sp * c[..., 1]          # plan body -> world
        wy = pp[1] + sp * c[..., 0] + cp * c[..., 1]
        cn, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = wx - x, wy - y                                # world -> current body
        body = np.stack([cn * dx + sn * dy, -sn * dx + cn * dy], -1).astype(np.float32)
        vals = getattr(res, "values", None)
        return body, (None if vals is None else np.asarray(vals, np.float32))

    def _diag(self, st, traj):
        """Why there is / isn't a path right now (logged by the controller)."""
        drv, now = self.nav.driver, time.monotonic()
        res = getattr(drv, "result", None) if drv is not None else None
        raw = None if res is None else getattr(res, "trajectory", None)
        raw_len = 0.0
        if raw is not None and len(raw):
            pts = np.asarray(raw)[:, :2]
            raw_len = float(np.linalg.norm(pts[-1]))
        th = getattr(self.nav, "thread", None)
        wm = self.last_world_map
        return {
            "state": str(st.get("state")),
            "plans": int(self.plans),
            "plan_age_s": (round(now - drv.plan_time, 2) if drv is not None and drv.plan_pose is not None else None),
            "plan_timeout_s": float(self.nc.get("plan_timeout", 1.0)),
            "plan_ms": round(float(getattr(self.nav, "plan_ms", 0.0)), 0),
            "planning_now": bool(th is not None and th.is_alive()),
            "error": self.nav.error or "",
            "choice": str(st.get("choice", "")),
            "n_valid": st.get("n_valid"),
            "navdp_path_m": round(raw_len, 2),   # length of NavDP's chosen path (0 = NavDP said stop)
            "path_pts_ahead": 0 if traj is None else int(len(traj)),
            "occ_cells": None if wm is None else int(np.asarray(wm.occupied).sum()),
            "goal_dist": st.get("goal_dist"),
        }

    def view(self, pose):
        x, y, yaw, _ = pose
        ego = self.bev.ego_view(x, y, yaw, 320, self.cfg["robot_radius"])
        seg = self.bev.seg_view() if self.bev.last is not None else None
        return {"ego": ego, "seg": seg}

    def close(self):
        try:
            self.nav.close()
        except Exception:
            pass


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
                    conn.send(("ack", worker.frame(*msg[1:])))
                elif kind == "cmd":
                    conn.send(("cmd", worker.cmd(msg[1])))
                elif kind == "map":
                    conn.send(("map", worker.view(msg[1])))
                elif kind == "quit":
                    break
            except Exception as e:  # report, keep serving
                import traceback

                conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"))
    finally:
        if worker is not None:
            worker.close()
        conn.close()


if __name__ == "__main__":
    main()
