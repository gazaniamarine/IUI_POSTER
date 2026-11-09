"""Baselines on the same BEV map as the tube method, for the benchmark: A*, A* + DWA, NavDP + DWA.

    nav = make_navigator(cfg["nav"])      # nav.planner: navdp (default) | xnavdp | astar | astar_dwa | navdp_dwa

Every navigator has TubeNavigator's interface (set_goal, plan / plan_async, command, cancel, close, status, map,
driver.result for drawing), so run_tube_sim.py, run_bev_tube.py and the metrics work unchanged. All methods get the
same inputs and limits: the BEV map's occupied cells (nav_map: within nav.obstacle_range, goal disk erased), the
inflation radius (tube_nav.inflation_radii: nav.robot_radius + nav.inflation_margin), nav.max_speed, nav.max_yaw_rate.

  astar      A* on the map grown by the inflation radius (falls back to the least one when the rover is already closer),
             costs rise near obstacles so the path keeps clear; the rover follows it with the same path tracker as
             NavDP (track_trajectory). Re-planned at every camera frame.
  astar_dwa  the A* path as the global path, followed by the Dynamic Window Approach (Fox, Burgard, Thrun 1997):
             (v, omega) sampled inside the window reachable under the acceleration limits, each rolled out for
             nav.dwa.horizon_s, rollouts closer than hard_radius to an occupied cell rejected, the rest scored on
             progress toward a look-ahead point of the global path, heading, clearance and speed.
  navdp_dwa  NavDP's own path (plain NavDP via the tube server, as nav.pure_navdp: no tubes) as the global path,
             followed by the same DWA on the same map. NavDP's own "stop" / turn-in-place (no forward path) is kept.
"""
from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from tube_nav import TubeNavigator, body_to_world, drop_stalls, inflation_radii, occupied_at, track_trajectory

NAMES = {"navdp": "PURE NavDP", "xnavdp": "X-NavDP", "sandplanner": "SanD-Planner", "astar": "A*", "astar_dwa": "A*+DWA", "navdp_dwa": "NavDP+DWA"}
CLASSIC = ("astar", "astar_dwa", "navdp_dwa")


def make_navigator(nc):
    planner = nc.get("planner", "navdp")
    if planner in ("astar", "astar_dwa"):
        return AStarNavigator(nc, dwa=planner == "astar_dwa")
    if planner == "navdp_dwa":
        return NavDPDWANavigator(nc)
    return TubeNavigator(nc)


# ---------------------------------------------------------------------------------------------------- map helpers
class Clearance:
    """Distance (m) from every cell to the nearest occupied cell, for one map (cached on the map object)."""

    def __init__(self, world_map):
        from scipy.ndimage import distance_transform_edt
        occ = np.asarray(world_map.occupied, bool)
        self.map = world_map
        self.dist = distance_transform_edt(~occ) * world_map.resolution if occ.any() else \
            np.full(occ.shape, np.inf)

    def at(self, xy):
        """[..., 2] world points -> [...] metres to the nearest occupied cell (inf outside the map)."""
        xy = np.asarray(xy, np.float64)
        m, res = self.map, self.map.resolution
        ix = np.floor((xy[..., 0] - m.origin[0]) / res).astype(np.int64)
        iy = np.floor((xy[..., 1] - m.origin[1]) / res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.dist.shape[1]) & (iy >= 0) & (iy < self.dist.shape[0])
        out = np.full(xy.shape[:-1], np.inf)
        out[ok] = self.dist[iy[ok], ix[ok]]
        return out


def clearance_of(world_map):
    cached = getattr(world_map, "_clearance", None)
    if cached is None:
        cached = Clearance(world_map)
        try:
            object.__setattr__(world_map, "_clearance", cached)
        except (AttributeError, TypeError):
            pass
    return cached


def astar(world_map, start, goal, radius, fallback_radius, prefer_m=0.35, weight=2.0):
    """A* on the map's cells (8-connected). Cells closer than radius to an occupied cell are blocked (fallback_radius
    when the start itself is that close; escaping from closer still is allowed). Step cost = length * (1 + weight *
    (prefer_m - d) / prefer_m) near obstacles (d < prefer_m), so the path keeps clear where it can.
    -> [K, 2] world points from start to goal, or None (no path)."""
    clr = clearance_of(world_map)
    d = clr.dist
    res, (ox, oy) = world_map.resolution, world_map.origin
    ny, nx = d.shape
    to_cell = lambda p: (int(np.clip(np.floor((p[1] - oy) / res), 0, ny - 1)),  # noqa: E731
                         int(np.clip(np.floor((p[0] - ox) / res), 0, nx - 1)))
    s, g = to_cell(start), to_cell(goal)
    limit = radius if d[s] > radius else fallback_radius
    if d[s] <= limit:
        limit = 0.9 * d[s]                       # already inside: let it leave
    free = d > limit
    free[g] = True
    if not free[s]:
        free[s] = True
    penalty = 1.0 + weight * np.clip((prefer_m - d) / prefer_m, 0.0, 1.0)
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
    best = {s: 0.0}
    came = {}
    heap = [(h(s), 0.0, s)]
    while heap:
        _, cost, c = heapq.heappop(heap)
        if c == g:
            break
        if cost > best.get(c, math.inf):
            continue
        for di, dj, step in moves:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < ny and 0 <= n[1] < nx) or not free[n]:
                continue
            if di and dj and not (free[c[0] + di, c[1]] and free[c[0], c[1] + dj]):
                continue                         # no corner cutting between two blocked cells
            nc_ = cost + step * penalty[n]
            if nc_ < best.get(n, math.inf):
                best[n], came[n] = nc_, c
                heapq.heappush(heap, (nc_ + h(n), nc_, n))
    if g not in best:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    cells.reverse()
    pts = np.array([(ox + (j + 0.5) * res, oy + (i + 0.5) * res) for i, j in cells], np.float64)
    pts[0], pts[-1] = start, goal
    return pts


def ahead_of(path, xy, spacing=0.1, n=24):
    """The part of a world path [K, 2] after the point nearest xy, resampled every spacing m (up to n points)."""
    if path is None or len(path) < 2:
        return None
    k = int(np.argmin(np.linalg.norm(path - xy, axis=1)))
    rest = np.vstack([xy, path[k + 1:]]) if k + 1 < len(path) else np.vstack([xy, path[-1:]])
    seg = np.linalg.norm(np.diff(rest, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return rest[-1:]
    at = np.arange(spacing, min(s[-1], spacing * n) + 1e-9, spacing)
    if len(at) == 0 or at[-1] < s[-1] - 1e-6 and len(at) < n:
        at = np.append(at, s[-1])
    return np.stack([np.interp(at, s, rest[:, 0]), np.interp(at, s, rest[:, 1])], axis=1)


def to_body_points(world_pts, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(world_pts) - (x, y)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


# ---------------------------------------------------------------------------------------------------- DWA
@dataclass
class DWAConfig:
    max_speed: float = 0.10
    max_yaw_rate: float = 0.35
    accel: float = 0.5                 # m/s^2
    yaw_accel: float = 2.0             # rad/s^2
    horizon_s: float = 2.5             # rollout length
    dt: float = 0.1                    # rollout step
    v_samples: int = 6
    w_samples: int = 15
    lookahead_m: float = 0.6           # target point this far along the global path
    hard_radius: float = 0.13          # rollouts closer than this to an occupied cell are rejected
    prefer_m: float = 0.35             # clearance above this earns nothing more
    w_target: float = 1.0              # distance from the rollout's end to the target point
    w_heading: float = 0.3             # final heading vs the direction to the target point (rad)
    w_clear: float = 0.6               # (prefer_m - clearance) / prefer_m
    w_speed: float = 0.4               # (max_speed - v) / max_speed

    @classmethod
    def from_nav(cls, nc):
        d = dict(nc.get("dwa") or {})
        return cls(max_speed=nc.get("max_speed", 0.10), max_yaw_rate=nc.get("max_yaw_rate", 0.35),
                   hard_radius=inflation_radii(nc)[1], **d)


class DWA:
    def __init__(self, cfg: DWAConfig):
        self.cfg = cfg
        self.v = self.w = 0.0
        self.t = None
        self.best = None                   # [T, 2] world points of the chosen rollout (for drawing)

    def reset(self):
        self.v = self.w = 0.0
        self.t = None
        self.best = None

    def step(self, pose, path, world_map, now):
        """pose (x, y, yaw), path [K, 2] world (the global path), -> (v, omega, info)."""
        c = self.cfg
        dt_cmd = 0.05 if self.t is None else float(np.clip(now - self.t, 0.0, 0.5))
        self.t = now
        ahead = ahead_of(path, np.asarray(pose[:2], np.float64), spacing=0.05, n=int(c.lookahead_m / 0.05) + 1)
        if ahead is None or not len(ahead):
            self.v, self.w = 0.0, 0.0
            return 0.0, 0.0, {"dwa": "no path"}
        target = ahead[-1]
        v_lo, v_hi = max(0.0, self.v - c.accel * dt_cmd), min(c.max_speed, self.v + c.accel * dt_cmd)
        w_lo = max(-c.max_yaw_rate, self.w - c.yaw_accel * dt_cmd)
        w_hi = min(c.max_yaw_rate, self.w + c.yaw_accel * dt_cmd)
        vs = np.unique(np.concatenate([np.linspace(v_lo, v_hi, c.v_samples), [0.0] if v_lo <= 1e-9 else []]))
        ws = np.unique(np.concatenate([np.linspace(w_lo, w_hi, c.w_samples), [0.0] if w_lo <= 0 <= w_hi else []]))
        V, W = np.meshgrid(vs, ws, indexing="ij")
        V, W = V.ravel(), W.ravel()
        steps = int(round(c.horizon_s / c.dt))
        x = np.full_like(V, pose[0])
        y = np.full_like(V, pose[1])
        th = np.full_like(V, pose[2])
        traj = np.empty((len(V), steps, 2))
        for k in range(steps):                                     # unicycle rollouts
            th = th + W * c.dt
            x = x + V * np.cos(th) * c.dt
            y = y + V * np.sin(th) * c.dt
            traj[:, k, 0], traj[:, k, 1] = x, y
        clr = clearance_of(world_map).at(traj).min(axis=1) if world_map is not None and \
            np.asarray(world_map.occupied).any() else np.full(len(V), np.inf)
        start_clear = clearance_of(world_map).at(np.asarray(pose[:2]))[()] if world_map is not None and \
            np.asarray(world_map.occupied).any() else np.inf
        # a rollout may not come closer than hard_radius; when the rover already is, it may not come closer still
        ok = clr >= min(c.hard_radius, start_clear - 1e-3)
        if not ok.any():
            bearing = math.atan2(*(to_body_points(target[None], pose)[0][::-1]))
            self.v, self.w = 0.0, float(np.clip(1.5 * bearing, -c.max_yaw_rate, c.max_yaw_rate))
            self.best = None
            return self.v, self.w, {"dwa": "all rollouts collide: turn"}
        end = traj[:, -1]
        dist = np.linalg.norm(end - target, axis=1)
        to_t = np.arctan2(target[1] - end[:, 1], target[0] - end[:, 0])
        head = np.abs((to_t - th + np.pi) % (2 * np.pi) - np.pi)
        clear_cost = np.clip((c.prefer_m - np.minimum(clr, c.prefer_m)) / c.prefer_m, 0.0, 1.0)
        cost = (c.w_target * dist + c.w_heading * head + c.w_clear * clear_cost
                + c.w_speed * (c.max_speed - V) / max(c.max_speed, 1e-6))
        cost[~ok] = np.inf
        i = int(np.argmin(cost))
        self.v, self.w = float(V[i]), float(W[i])
        self.best = traj[i]
        return self.v, self.w, {"dwa": f"{int(ok.sum())}/{len(ok)} ok", "clear": float(clr[i])}


# ---------------------------------------------------------------------------------------------------- navigators
@dataclass
class _Result:
    trajectory: np.ndarray             # [H, 3] body frame of plan_pose (drawing; the tracker's input)
    choice_kind: str = ""
    candidates: np.ndarray = None
    values: np.ndarray = None
    tube: dict = None
    sent_cells: int = 0
    world_path: np.ndarray = None


@dataclass
class _Driver:
    """What run_tube_sim.py / draw_nav read from a navigator's driver."""
    result: _Result = None
    plan_pose: tuple = None
    plan_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class AStarNavigator:
    """A* (+ optional DWA) on the BEV map. No images, no server."""
    pure = True
    push = consistency = smooth = None
    split = rebuild = lyapunov = lyapunov_delta = guide_until = push_step = push_margin = None

    def __init__(self, nc, dwa=False):
        self.nc = nc
        self.planner = "astar_dwa" if dwa else "astar"
        self.dwa = DWA(DWAConfig.from_nav(nc)) if dwa else None
        self.driver = _Driver()
        self.goal = None
        self.map = None
        self.path = None                   # [K, 2] world, the newest A* path
        self.thread = None
        self.last_plan_start = -1e9
        self.plan_ms = 0.0
        self.error = ""
        self.status = {"state": "idle"}

    def set_goal(self, goal_xy, K=None):
        self._wait()
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.path, self.driver = None, _Driver()
        if self.dwa is not None:
            self.dwa.reset()
        self.error = ""
        self.status = {"state": "planning"}

    def cancel(self, state="idle"):
        self.goal = None
        self.status = {"state": state}

    def want_plan(self, now):
        busy = self.thread is not None and self.thread.is_alive()
        return self.goal is not None and not busy and now - self.last_plan_start >= 1.0 / self.nc.get("plan_hz", 5.0)

    def plan(self, rgb, depth, pose, t, world_map):
        goal = self.goal
        if goal is None:
            return None
        self.map = world_map
        t0 = time.perf_counter()
        path = astar(world_map, np.asarray(pose[:2], np.float64), np.asarray(goal, np.float64),
                     *inflation_radii(self.nc))
        self.plan_ms = 1000 * (time.perf_counter() - t0)
        if path is None:
            self.error = "A*: no path to the goal on the map"
            return None
        self.error = ""
        self.path = path
        body = to_body_points(ahead_of(path, np.asarray(pose[:2], np.float64)), pose)
        traj = np.concatenate([body, np.zeros((len(body), 1))], axis=1)
        res = _Result(traj, choice_kind=self.planner, world_path=path,
                      sent_cells=int(np.asarray(world_map.occupied).sum()))
        with self.driver.lock:
            self.driver.result, self.driver.plan_pose, self.driver.plan_time = res, tuple(map(float, pose)), float(t)
        return res

    def plan_async(self, rgb, depth, pose, t, world_map, now=None):
        now = time.monotonic() if now is None else now
        if not self.want_plan(now):
            return False
        self.last_plan_start = now
        self.thread = threading.Thread(target=self.plan, args=(None, None, tuple(pose), t, world_map), daemon=True)
        self.thread.start()
        return True

    def _wait(self):
        if self.thread is not None:
            self.thread.join(timeout=10.0)

    def close(self):
        self._wait()

    def goal_distance(self, pose):
        return None if self.goal is None else math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])

    def command(self, pose, now):
        if self.goal is None:
            return 0.0, 0.0, self.status
        d = self.goal_distance(pose)
        if d <= self.nc.get("stop_distance", 0.4):
            self.cancel("arrived")
            self.status["goal_dist"] = d
            return 0.0, 0.0, self.status
        drv = self.driver
        if drv.plan_pose is None or now - drv.plan_time > self.nc.get("plan_timeout", 1.0) or self.path is None:
            self.status = {"state": "no plan" + (f" ({self.error})" if self.error else ""), "goal_dist": d}
            return 0.0, 0.0, self.status
        if self.dwa is not None:
            v, w, info = self.dwa.step(pose, self.path, self.map, now)
        else:
            body = to_body_points(ahead_of(self.path, np.asarray(pose[:2], np.float64)), pose)
            cmd = track_trajectory(drop_stalls(body), lookahead=3, max_speed=self.nc.get("max_speed", 0.10),
                                   max_yaw_rate=self.nc.get("max_yaw_rate", 0.35))
            v, w, info = float(cmd[0]), float(cmd[2]), {}
        self.status = {"state": "driving", "pure": True, "planner": self.planner, "goal_dist": d, "v": v, "w": w,
                       "plan_ms": self.plan_ms, "info": info.get("dwa", "")}
        return v, w, self.status


class NavDPDWANavigator(TubeNavigator):
    """Plain NavDP (via the tube server, no tubes, no map) for the global path; DWA on the BEV map follows it."""

    def __init__(self, nc):
        super().__init__({**nc, "planner": "navdp", "pure_navdp": True})
        self.planner = "navdp_dwa"
        self.dwa = DWA(DWAConfig.from_nav(nc))

    def set_goal(self, goal_xy, K):
        super().set_goal(goal_xy, K)
        self.dwa.reset()

    def command(self, pose, now):
        if self.goal is None:
            return 0.0, 0.0, self.status
        d = self.goal_distance(pose)
        if d <= self.nc.get("stop_distance", 0.4):
            self.cancel("arrived")
            self.status["goal_dist"] = d
            return 0.0, 0.0, self.status
        drv = self.driver
        if drv is None or drv.plan_pose is None or now - drv.plan_time > self.nc.get("plan_timeout", 1.0):
            self.status = {"state": "no plan" + (f" ({self.error})" if self.error else ""), "goal_dist": d}
            return 0.0, 0.0, self.status
        with drv.lock:
            traj, plan_pose, track = drv.result.trajectory[:, :2], drv.plan_pose, drv.track_command
        path = drop_stalls(traj)
        if len(path) == 0 or np.linalg.norm(path[-1]) < 0.05:
            # NavDP's own stop / turn in place (no forward path): keep it, as plain NavDP does
            v, w, info = float(track[0]), float(track[2]), {"dwa": "NavDP stop/turn"}
            self.dwa.v, self.dwa.w = v, w
        else:
            world = np.vstack([plan_pose[:2], body_to_world(path, plan_pose)])
            v, w, info = self.dwa.step(pose, world, self.map, now)
        self.status = {"state": "driving", "pure": True, "planner": self.planner, "goal_dist": d, "v": v, "w": w,
                       "plan_ms": self.plan_ms, "info": info.get("dwa", "")}
        return v, w, self.status
