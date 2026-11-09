"""Tube-guided NavDP on top of the BEV map: goal, planning, velocity commands, drawing.

Uses ../tube_navdp (copied from navdp_sam). Start its server first (see ../tube_navdp/README.md).

    nav = TubeNavigator(cfg["nav"])
    nav.set_goal((x, y), K)                         # odom frame; resets NavDP's image memory
    nav.plan_async(rgb, depth, pose, t, world_map)  # after each BEV frame (pose = robot pose at the image time);
                                                    # world_map = nav_map(grid, state, robot_xy, goal, nc)
    v, w, status = nav.command(pose, now)           # at the control rate, pose = latest robot pose

Planning (an HTTP call, ~0.15 s) runs in a background thread so the control loop keeps going; the new plan is
swapped in at once, so a command never mixes an old plan's pose with a new tube.

Obstacles: the BEV map's occupied cells themselves (no circles). Robot footprint: the tube treats the robot as a
point and the driver grows the occupied cells by nav.robot_radius (plus the controller's look-ahead in stt mode).
Goal: nav_obstacles.obstacle_cells erases the goal's surroundings from the map, and the rover stops
nav.stop_distance short of the goal point.
"""

import math
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUBE_ROOT = os.path.normpath(os.path.join(HERE, "..", "tube_navdp"))
if TUBE_ROOT not in sys.path:
    sys.path.insert(0, TUBE_ROOT)

from navdp.official_client import track_trajectory  # noqa: E402
from navdp.tube.client import SandPlannerClient, TubeNavDPClient, XNavDPClient  # noqa: E402
from navdp.tube.controller import FunnelConfig, TubeClock  # noqa: E402
from navdp.tube.driver import DriverConfig, TubeDriver, WorldGrid, body_bev, executed_tube, to_body  # noqa: E402
from navdp.tube.stt import TubeConfig, start_box  # noqa: E402

from bev_render import world_to_px  # noqa: E402
from nav_obstacles import obstacle_cells  # noqa: E402


def inflation_radii(nc) -> tuple:
    """-> (normal, least) inflation radius [m], both the fixed robot size nav.robot_radius plus nav.inflation_margin:
    there is no smaller fallback radius. The planners grow every obstacle cell by it (the robot centre keeps that far)."""
    r = float(nc.get("robot_radius", 0.16)) + float(nc.get("inflation_margin", 0.0))
    return r, r


GUIDE_KEYS = ("push", "consistency", "smooth", "split", "rebuild", "lyapunov", "lyapunov_delta", "guide_until",
              "push_step", "push_margin")  # sent, then checked


def drop_stalls(trajectory, eps=0.02, near=0.08):
    """The path without its stalled start: waypoints within ``near`` m of the rover before the path first gets
    farther than that, and waypoints that repeat the one before (within eps m).

    Guided paths can stall: next to an obstacle the first waypoints may all sit (or jitter, 2-3 cm apart) a few cm
    ahead before the path turns away. The tracker steers toward waypoint ``lookahead`` (3): on such a path that is a
    stalled point inside its 5 cm stop radius, so it commanded v = w = 0 and the rover never moved (stuck at sharp
    corners). Without the stalled start it steers toward where the path actually goes. A path that never leaves
    ``near`` (NavDP's stop / turn in place) is kept as it is."""
    pts = np.asarray(trajectory)
    if len(pts) == 0:
        return pts
    far = np.hypot(pts[:, 0], pts[:, 1]) > near
    if not far.any():
        return pts
    keep, last = [], np.zeros(2)
    for p in pts[int(np.argmax(far)):]:
        if np.hypot(*(p[:2] - last)) > eps:
            keep.append(p)
            last = p[:2]
    return np.asarray(keep, pts.dtype) if keep else pts


def body_to_world(points, pose):
    """[N, 2] (forward, left) of a robot at pose -> world (x, y)."""
    p = np.asarray(points, np.float64)[..., :2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.stack([pose[0] + c * p[..., 0] - s * p[..., 1], pose[1] + s * p[..., 0] + c * p[..., 1]], -1)


def nav_map(grid, state, robot_xy, goal_xy, nc) -> WorldGrid:
    """The BEV map as the planner sees it: occupied cells within nav.obstacle_range, the goal's surroundings erased,
    specks removed (nav_obstacles.obstacle_cells); without the robot's size (the driver adds it)."""
    return WorldGrid(obstacle_cells(grid, state, robot_xy, goal_xy, nc), grid.res, (grid.x_min, grid.y_min))


def grow(world_map, radius):
    """The map with every occupied cell grown by radius metres."""
    occ = np.asarray(world_map.occupied, bool)
    if radius > 0 and occ.any():
        from scipy.ndimage import distance_transform_edt
        occ = distance_transform_edt(~occ) * world_map.resolution <= radius
    return WorldGrid(occ, world_map.resolution, world_map.origin)


def occupied_at(world_map, xy):
    """[..., 2] world points -> [...] bool: the point is in an occupied cell (outside the map counts as free)."""
    xy = np.asarray(xy, np.float64)
    occ, res = world_map.occupied, world_map.resolution
    ix = np.floor((xy[..., 0] - world_map.origin[0]) / res).astype(np.int64)
    iy = np.floor((xy[..., 1] - world_map.origin[1]) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < occ.shape[1]) & (iy >= 0) & (iy < occ.shape[0])
    out = np.zeros(xy.shape[:-1], bool)
    out[ok] = occ[iy[ok], ix[ok]]
    return out


def clear_length(paths, pose, grown_map, step=0.05) -> np.ndarray:
    """[N, H, 2+] body-frame paths (the robot at the origin) vs a map already grown by the inflation radius -> [N] m:
    how far along each path the robot centre gets before it first enters an occupied cell (the whole length when it
    never does). A robot that starts inside the grown cells (map noise, or it got close) may still leave them: the
    cells it is in from the start until it first leaves them do not count."""
    paths = np.asarray(paths, np.float64)[..., :2]
    full = np.concatenate([np.zeros((len(paths), 1, 2)), paths], axis=1)
    seg = np.diff(full, axis=1)
    n = max(1, int(np.ceil(np.linalg.norm(seg, axis=-1).max() / step)))
    s = (np.arange(n) / n)[None, None, :, None]
    pts = (full[:, :-1, None] + s * seg[:, :, None]).reshape(len(paths), -1, 2)
    pts = np.concatenate([pts, full[:, -1:]], axis=1)
    along = np.concatenate([np.zeros((len(pts), 1)), np.cumsum(np.linalg.norm(np.diff(pts, axis=1), axis=-1), 1)], 1)
    if grown_map is None or not grown_map.occupied.any():
        return along[:, -1]
    hit = occupied_at(grown_map, body_to_world(pts, pose))
    hit &= ~np.logical_and.accumulate(hit, axis=1)             # not while still inside the cells it started in
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), pts.shape[1] - 1)
    return along[np.arange(len(pts)), first]


class PathFollower:
    """Pure pursuit on the chosen path, at the control rate (the old tracker computed one command per plan and chased
    waypoint 3). Every call: the rover's place on the path (the nearest waypoint, never going back), the point
    ``lookahead`` m ahead along it, and the arc to that point: w = v * kappa, kappa = 2 sin(alpha) / distance. Slower on
    tight arcs (v = max_speed / (1 + curve_slow |kappa|)); a target more than ``turn_in_place`` to the side: turn on
    the spot towards it. Speed and turn rate change at most max_accel / max_yaw_accel per second, so it stays smooth."""

    def __init__(self, lookahead=0.20, max_speed=0.10, max_yaw_rate=0.35, max_accel=0.4, max_yaw_accel=3.0,
                 curve_slow=0.5, stop_radius=0.05, turn_in_place=math.radians(60)):
        self.lookahead, self.max_speed, self.max_yaw_rate = lookahead, max_speed, max_yaw_rate
        self.max_accel, self.max_yaw_accel, self.curve_slow = max_accel, max_yaw_accel, curve_slow
        self.stop_radius, self.turn_in_place = stop_radius, turn_in_place
        self.path, self.idx, self.v, self.w, self.t = None, 0, 0.0, 0.0, None

    def set_path(self, world_xy):
        """The new plan's path in world (x, y), from the rover's position at planning time."""
        self.path, self.idx = np.asarray(world_xy, np.float64), 0

    def reset(self):
        self.path, self.idx, self.v, self.w, self.t = None, 0, 0.0, 0.0, None

    def _target(self, pose):
        p, xy = self.path, np.array(pose[:2])
        d = np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1])
        self.idx = self.idx + int(np.argmin(d[self.idx:]))      # where the rover is on the path (never back)
        for j in range(self.idx + 1, len(p)):                    # the first point lookahead m away, interpolated
            if d[j] >= self.lookahead:
                a, b = p[j - 1], p[j]
                da, db = d[j - 1], d[j]
                f = 1.0 if db <= da else float(np.clip((self.lookahead - da) / (db - da), 0.0, 1.0))
                return a + f * (b - a), False
        return p[-1], True

    def _pursue(self, pose):
        if self.path is None or len(self.path) < 2:
            return 0.0, 0.0
        target, at_end = self._target(pose)
        fwd, left = to_body(target, pose)
        dist = math.hypot(fwd, left)
        if at_end and dist <= self.stop_radius:
            return 0.0, 0.0
        alpha = math.atan2(left, fwd)
        if abs(alpha) > self.turn_in_place:                      # the path goes off to the side / behind: turn first
            return 0.0, math.copysign(self.max_yaw_rate, alpha)
        kappa = 2.0 * math.sin(alpha) / max(dist, 1e-3)
        v = self.max_speed / (1.0 + self.curve_slow * abs(kappa))
        w = v * kappa
        if abs(w) > self.max_yaw_rate:                           # keep the arc: slow down instead of cutting it
            w = math.copysign(self.max_yaw_rate, w)
            v = abs(w / kappa)
        return v, w

    def command(self, pose, now):
        """-> (v m/s, w rad/s), rate limited."""
        dt = 0.05 if self.t is None else float(np.clip(now - self.t, 0.0, 0.2))
        self.t = now
        v_ref, w_ref = self._pursue(pose)
        self.v += float(np.clip(v_ref - self.v, -self.max_accel * dt, self.max_accel * dt))
        self.w += float(np.clip(w_ref - self.w, -self.max_yaw_accel * dt, self.max_yaw_accel * dt))
        return self.v, self.w


class ThreadedTubeDriver(TubeDriver):
    """TubeDriver whose plan() may run in another thread than command(): the HTTP call runs outside the lock and
    the new plan's state is swapped in all at once.

    One change of choice: when no candidate is certified, the tube code drives NavDP's best-valued path even if
    it runs through a known obstacle. Here, with the waypoint tracker, the robot drives the farthest path instead:
    the candidate that goes farthest before it first enters the map's occupied cells grown by the inflation radius
    (clear_length); among those within 5 cm of the farthest, NavDP's best-valued one. No fallback, no turning in
    place. Paths NavDP shortened to "stop" (all zero) never count. result.choice_kind: certified | farthest."""

    def __init__(self, *a, hard_radius=0.13, follower=None, pure=False, push=None,
                 consistency=None, smooth=None, split=None, rebuild=None, lyapunov=None, lyapunov_delta=None,
                 guide_until=None, push_step=None, push_margin=None, **kw):
        self.lock = threading.Lock()
        self.push = push                  # tube push mode sent with each request (None = the server's default)
        self.consistency, self.smooth = consistency, smooth   # mu, beta sent with each request (None = server's)
        self.split, self.rebuild = split, rebuild             # guided-from step, rebuild tubes every step
        self.lyapunov, self.lyapunov_delta = lyapunov, lyapunov_delta   # Lyapunov-guided pull (None = server's)
        self.guide_until = guide_until    # pull only before this denoising step (None = server's --guide-until)
        self.push_step, self.push_margin = push_step, push_margin   # obstacle push reach / margin (None = server's)
        self.pure = pure                  # pure NavDP: no map to the server, no map-based override, track only
        self.hard_radius = hard_radius    # the map the server gets when the rover is already inside the grown cells
        self.follower = follower          # PathFollower for the track controller (None = the old per-plan tracker)
        super().__init__(*a, **kw)

    def plan(self, rgb, depth, goal_xy, pose, bev=None, now=0.0):
        """bev: the world-frame map (WorldGrid, nav_map), without the robot's size; None = no obstacles."""
        cfg = self.config
        grid, tube_radius = None, None
        if bev is not None and not self.pure:
            # The tube treats the rover as a point on the map grown by robot_radius. The fallback paths may take the
            # rover closer (down to hard_radius), and then its own start box lies in the grown cells:
            # no tube can exist until it is well clear again (most plans next to an obstacle). There the tube is
            # built on the map grown by hard_radius, the clearance the fallback already holds.
            tube_radius = cfg.robot_radius
            grid = body_bev(bev, pose, tube_radius + self.start_forward)
            if grid.box_blocked(*start_box(TubeConfig(start_forward=self.start_forward))):
                tube_radius = self.hard_radius
                grid = body_bev(bev, pose, tube_radius + self.start_forward)
        goal_b = to_body(goal_xy, pose)
        extra = {"pose": pose} if getattr(self.client, "wants_pose", False) else {}
        if not getattr(self.client, "wants_pose", False):   # the tube server
            for key in GUIDE_KEYS:
                if getattr(self, key) is not None:
                    extra[key] = getattr(self, key)
        if not getattr(self.client, "wants_pose", False) and not self.pure:   # the tube server: the previous plan
            with self.lock:
                prev = self.previous_plan(pose)
            if prev is not None:
                extra["prev"] = prev
        result = self.client.step(rgb, depth, goal_b, grid, cfg.mode, self.start_forward, **extra)
        for key in GUIDE_KEYS:                             # an old server ignores what it does not know
            used = (result.tube or {}).get(key)
            if key in extra and result.tube is not None and used != extra[key]:
                raise RuntimeError(f"the tube server used {key} {used if used is not None else 'unknown (old server code)'}"
                                   f", not {extra[key]}: restart it (start_tube_server.sh) so it runs the current code")
        tube = executed_tube(result.tube or {})
        result.choice_kind = "certified" if (result.tube or {}).get("chosen_certified") else "navdp"
        if self.pure:                                         # NavDP's own best-valued path, as it is
            result.choice_kind, tube, result.tube = "navdp", None, None   # drop the server's tube report too
        elif result.choice_kind != "certified" and result.candidates is not None:
            moving = np.linalg.norm(result.candidates[:, -1, :2], axis=-1) > 0.05
            length = clear_length(result.candidates, pose, None if bev is None else grow(bev, cfg.robot_radius))
            length = np.where(moving, length, -1.0)
            near = length >= length.max() - 0.05             # the farthest, and those within 5 cm of it
            pick = int(np.argmax(np.where(near, result.values, -np.inf)))
            result.trajectory, result.choice_kind = result.candidates[pick].copy(), "farthest"
            result.clear_m = float(length[pick])              # how far it goes before the grown cells
            tube = None                                       # the tube law only for a certified path
        clock = None if tube is None else TubeClock(tube["lo_coef"], tube["hi_coef"], len(result.trajectory),
                                                    cfg.funnel)
        track = track_trajectory(drop_stalls(result.trajectory), lookahead=cfg.lookahead, max_speed=cfg.funnel.max_speed,
                                 max_yaw_rate=cfg.funnel.max_yaw_rate)
        result.sent_cells = 0 if grid is None else int(grid.occupied.sum())
        result.tube_radius = tube_radius     # the clearance the tube was built with (robot_radius or hard_radius)
        with self.lock:
            self.clock, self.track_command, self.result = clock, track, result
            self.plan_pose, self.plan_time = tuple(float(v) for v in pose), float(now)
            if self.follower is not None:
                self.follower.set_path(body_to_world(np.vstack([[0.0, 0.0], drop_stalls(result.trajectory)[:, :2]]), pose))
        return result

    def command(self, pose, now):
        with self.lock:
            if self.follower is not None and self.config.controller == "track" and self.plan_pose is not None:
                v, w = self.follower.command(pose, now)
                return v, w, {"inside": None, "saturated": False, "fallback": False}
            return super().command(pose, now)

    def reset(self):
        with self.lock:
            super().reset()
            if self.follower is not None:
                self.follower.reset()


class TubeNavigator:
    def __init__(self, nc):
        self.nc = nc
        self.planner = nc.get("planner", "navdp")      # navdp: the tube NavDP server | xnavdp: the X-NavDP server |
        if self.planner not in ("navdp", "xnavdp", "sandplanner"):   # sandplanner: the SanD-Planner server
            raise ValueError("nav.planner must be navdp, xnavdp or sandplanner")
        self.pure = bool(nc.get("pure_navdp", False)) or self.planner in ("xnavdp", "sandplanner")   # no tubes
        self.push = nc.get("tube_push")                 # nearest | smooth | None (server's --tube-push)
        self.consistency = None if nc.get("tube_consistency") is None else float(nc["tube_consistency"])  # mu
        self.smooth = None if nc.get("tube_smooth") is None else float(nc["tube_smooth"])                 # beta
        opt = lambda key, kind: None if nc.get(key) is None else kind(nc[key])
        self.split = opt("tube_split", int)              # guided from this scheduler step (None = server's --split)
        self.rebuild = opt("tube_rebuild", bool)         # tubes rebuilt at every guided step
        self.lyapunov = opt("tube_lyapunov", str)        # off | min | ramp (LyaGuide-style pull)
        self.lyapunov_delta = opt("tube_lyapunov_delta", float)
        self.guide_until = opt("tube_guide_until", int)  # pull only before this denoising step (0 = to the end)
        self.push_step = opt("tube_push_step", float)    # obstacle push: m per waypoint and axis (0.25 = NavDP's)
        self.push_margin = opt("tube_push_margin", float)   # obstacle push: m past the grown obstacles
        if self.lyapunov not in (None, "off", "min", "ramp"):
            raise ValueError("nav.tube_lyapunov must be off, min or ramp")
        if self.pure:                     # NavDP alone: no pull toward the last plan, no smoothing (also not from
            self.consistency = self.smooth = 0.0   # the server's --consistency / --smooth)
            self.split = self.rebuild = self.lyapunov = self.lyapunov_delta = None   # no guidance at all
            self.guide_until = self.push_step = self.push_margin = None
        if self.push not in (None, "nearest", "smooth"):
            raise ValueError("nav.tube_push must be nearest or smooth")
        if self.pure:                     # NavDP by itself: plain sampling, path tracker (no tube / stt law)
            nc = {**nc, "mode": "navdp", "controller": "track"}
        self.funnel = FunnelConfig(max_speed=nc.get("max_speed", 0.10), max_yaw_rate=nc.get("max_yaw_rate", 0.35),
                                   lookahead_m=nc.get("lookahead_m", 0.2), gain=nc.get("gain", 0.035))
        self.dcfg = DriverConfig(mode=nc.get("mode", "tube"), controller=nc.get("controller", "stt"),
                                 robot_radius=inflation_radii(nc)[0],   # the map is grown by the inflation radius
                                 stop_threshold=nc.get("stop_threshold", -3.0), funnel=self.funnel)
        self.scale = float(nc.get("image_scale", 0.5))
        self.driver = None
        self.goal = None
        self.map = None                   # WorldGrid (nav_map): the newest map, for planning and drawing
        self.thread = None
        self.last_plan_start = -1e9
        self.plan_ms = 0.0
        self.error = ""
        self.status = {"state": "idle"}

    # ------------------------------------------------------------------ goal
    def set_goal(self, goal_xy, K):
        """New goal (odom x, y). Resets NavDP's memory of the last frames."""
        self._wait()
        Ks = np.array(K, np.float64)
        Ks[:2] *= self.scale
        server, client_cls, name, script = {
            "xnavdp": (self.nc.get("xnavdp_server", "http://127.0.0.1:8891"), XNavDPClient, "X-NavDP",
                       "start_xnavdp_server.sh"),
            "sandplanner": (self.nc.get("sandplanner_server", "http://127.0.0.1:8892"), SandPlannerClient,
                            "SanD-Planner", "start_sandplanner_server.sh"),
        }.get(self.planner, (self.nc.get("server", "http://127.0.0.1:8890"), TubeNavDPClient, "tube NavDP",
                             "start_tube_server.sh"))
        # the server loads the NavDP model on its first reset: allow longer than for a plan
        try:
            if self.driver is None:
                client = client_cls(server, timeout=self.nc.get("reset_timeout", 60.0))
                self.driver = ThreadedTubeDriver(intrinsic=Ks, config=self.dcfg, client=client,
                                                 hard_radius=inflation_radii(self.nc)[1],
                                                 follower=PathFollower(
                                                     lookahead=self.nc.get("follow_lookahead_m", 0.25),
                                                     max_speed=self.nc.get("max_speed", 0.10),
                                                     max_yaw_rate=self.nc.get("max_yaw_rate", 0.35),
                                                     max_accel=self.nc.get("max_accel", 0.4),
                                                     max_yaw_accel=self.nc.get("max_yaw_accel", 3.0),
                                                     curve_slow=self.nc.get("curve_slow", 0.5)),
                                                 pure=self.pure,
                                                 push=self.push, consistency=self.consistency,
                                                 smooth=self.smooth, split=self.split, rebuild=self.rebuild,
                                                 lyapunov=self.lyapunov, lyapunov_delta=self.lyapunov_delta,
                                                 guide_until=self.guide_until, push_step=self.push_step,
                                                 push_margin=self.push_margin)
            else:
                self.driver.client.timeout = self.nc.get("reset_timeout", 60.0)
                self.driver.intrinsic = Ks
                self.driver.reset()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{name} server at {server} did not answer the reset ({type(e).__name__}: {e}). "
                               f"Is it running ({script})? Not paused with Ctrl+Z?") from None
        self.driver.client.timeout = self.nc.get("request_timeout", 5.0)
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.error = ""
        self.status = {"state": "planning"}

    def cancel(self, state="idle"):
        self.goal = None
        self.status = {"state": state}

    # ------------------------------------------------------------------ planning
    def want_plan(self, now):
        busy = self.thread is not None and self.thread.is_alive()
        return self.goal is not None and not busy and now - self.last_plan_start >= 1.0 / self.nc.get("plan_hz", 5.0)

    def plan(self, rgb, depth, pose, t, world_map):
        """Blocking plan from one image. pose = (x, y, yaw) of the robot when the image was taken, t its time,
        world_map the map then (nav_map)."""
        goal = self.goal
        if goal is None:
            return None
        self.map = world_map
        if self.scale != 1.0:
            h, w = depth.shape
            size = (int(round(w * self.scale)), int(round(h * self.scale)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        t0 = time.perf_counter()
        try:
            result = self.driver.plan(rgb, depth, goal, pose, world_map, t)
            self.error = ""
        except Exception as e:  # noqa: BLE001  (server down, timeout): no new plan; command() stops when stale
            self.error = f"{type(e).__name__}: {e}"[:220]
            return None
        self.plan_ms = 1000 * (time.perf_counter() - t0)
        return result

    def plan_async(self, rgb, depth, pose, t, world_map, now=None):
        now = time.monotonic() if now is None else now
        if not self.want_plan(now):
            return False
        self.last_plan_start = now
        self.thread = threading.Thread(target=self.plan, args=(rgb.copy(), depth.copy(), tuple(pose), t, world_map),
                                       daemon=True)
        self.thread.start()
        return True

    def _wait(self):
        if self.thread is not None:
            self.thread.join(timeout=10.0)

    # ------------------------------------------------------------------ control
    def goal_distance(self, pose):
        return None if self.goal is None else math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])

    def command(self, pose, now):
        """-> (v m/s, omega rad/s, status). Zero unless there is a goal and a fresh plan."""
        if self.goal is None:
            return 0.0, 0.0, self.status
        d = self.goal_distance(pose)
        if d <= self.nc.get("stop_distance", 0.4):
            self.cancel("arrived")
            self.status["goal_dist"] = d
            return 0.0, 0.0, self.status
        drv = self.driver
        if drv.plan_pose is None or now - drv.plan_time > self.nc.get("plan_timeout", 1.0):
            self.status = {"state": "no plan" + (f" ({self.error})" if self.error else ""), "goal_dist": d}
            return 0.0, 0.0, self.status
        v, w, info = drv.command(pose, now)
        res = drv.result
        tube = res.tube or {}
        self.status = {"state": "driving", "pure": self.pure, "planner": self.planner, "goal_dist": d, "v": v, "w": w,
                       "inside": info["inside"], "saturated": info["saturated"], "fallback": info["fallback"],
                       "certified": tube.get("chosen_certified"), "choice": getattr(res, "choice_kind", ""),
                       "n_tubes": len(tube.get("tubes", [])),
                       "n_valid": sum(bool(t.get("feasible")) for t in tube.get("tubes", [])),
                       "n_cells": getattr(res, "sent_cells", 0), "plan_ms": self.plan_ms,
                       "tube_radius": getattr(res, "tube_radius", None)}
        return v, w, self.status

    def close(self):
        self._wait()
        if self.driver is not None:
            self.driver.close()


# ---------------------------------------------------------------------- drawing
CAND, CHOSEN_OK, CHOSEN_BAD = (170, 170, 170), (60, 200, 60), (0, 140, 255)   # BGR
OBST, MARGIN, TUBE, GOAL = (40, 40, 230), (150, 150, 255), (220, 200, 0), (0, 200, 0)
WAY_OK, WAY_IN = (255, 255, 255), (255, 0, 255)   # chosen-path waypoints: clear / inside the grown zone


def draw_nav(img, grid, scale, nav, robot_xy=None):
    """Draw the planner's map (its occupied cells as mapped, and round them the inflation margin nav.inflation_margin:
    with the robot circle drawn at its own size robot_radius, the circle touching the margin = at the planner's limit,
    the inflation radius; touching the cells = contact), goal + cleared disk, NavDP's paths and the tube."""
    px = lambda x, y: world_to_px(grid, scale, x, y)
    m = lambda r: max(1, int(round(r / grid.res * scale)))
    nc = nav.nc
    pure = getattr(nav, "pure", False)               # pure NavDP: only the raw map, NavDP's paths and its choice
    if nav.map is not None and nav.map.occupied.any():
        occ = nav.map.occupied
        margin = 0.0 if pure else float(nc.get("inflation_margin", 0.0))
        layers = ([(grow(nav.map, margin).occupied & ~occ, MARGIN)] if margin > 0 else []) + [(occ, OBST)]
        for mask, colour in layers:
            big = cv2.resize(np.flipud(mask).astype(np.uint8), (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0
            img[big] = colour
    if nav.goal is not None:
        g = px(*nav.goal)
        if not pure:
            cv2.circle(img, g, m(nc.get("goal_clear_radius", 0.4)), GOAL, 1, cv2.LINE_AA)
        cv2.drawMarker(img, g, GOAL, cv2.MARKER_STAR, 18, 2)
    drv = nav.driver
    if drv is None or drv.result is None or nav.goal is None:
        return img
    with drv.lock:
        res, pose = drv.result, drv.plan_pose
    if res.candidates is not None:
        for c in res.candidates:
            pts = np.array([px(*p) for p in body_to_world(np.vstack([[0, 0], c[:, :2]]), pose)], np.int32)
            cv2.polylines(img, [pts], False, CAND, 1, cv2.LINE_AA)
    tube = executed_tube(res.tube or {})
    if tube is not None:
        for lo, hi in list(zip(tube["lo"], tube["hi"]))[1::3]:
            box = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
            pts = np.array([px(*p) for p in body_to_world(box, pose)], np.int32)
            cv2.polylines(img, [pts], True, TUBE, 1, cv2.LINE_AA)
    ok = pure or getattr(res, "choice_kind", "") == "certified"   # pure: one colour, NavDP's own pick
    dwa = getattr(getattr(nav, "dwa", None), "best", None)        # A*+DWA / NavDP+DWA: the chosen rollout
    if dwa is not None and len(dwa):
        cv2.polylines(img, [np.array([px(*p) for p in dwa], np.int32)], False, TUBE, 2, cv2.LINE_AA)
    pts = np.array([px(*p) for p in body_to_world(np.vstack([[0, 0], res.trajectory[:, :2]]), pose)], np.int32)
    cv2.polylines(img, [pts], False, CHOSEN_OK if ok else CHOSEN_BAD, 2, cv2.LINE_AA)
    # the chosen path's waypoints: white outside the planner's grown zone (obstacles grown by the inflation radius),
    # magenta inside it; every 6th numbered (1-based) to show where it is along the path
    way = body_to_world(res.trajectory[:, :2], pose)
    inside = (occupied_at(grow(nav.map, inflation_radii(nc)[0]), way) if nav.map is not None and nav.map.occupied.any()
              else np.zeros(len(way), bool))
    for j, (p, bad) in enumerate(zip(way, inside)):
        c = px(*p)
        cv2.circle(img, c, 3, WAY_IN if bad else WAY_OK, -1, cv2.LINE_AA)
        cv2.circle(img, c, 3, (0, 0, 0), 1, cv2.LINE_AA)
        if j % 6 == 5 or j == 0:
            cv2.putText(img, str(j + 1), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(j + 1), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return img


def status_text(st):
    s = st.get("state", "")
    if "goal_dist" in st and st["goal_dist"] is not None:
        s += f"  goal {st['goal_dist']:.2f} m"
    if s.startswith("driving") and st.get("pure"):
        name = {"xnavdp": "X-NavDP", "sandplanner": "SanD-Planner", "astar": "A*", "astar_dwa": "A*+DWA",
                "navdp_dwa": "NavDP+DWA"}.get(st.get("planner"), "PURE NavDP")
        s += f"  v={st['v']:.2f} w={st['w']:+.2f}  {name}  {st['plan_ms']:.0f} ms"
        if st.get("info"):
            s += f"  {st['info']}"
    elif s.startswith("driving"):
        s += (f"  v={st['v']:.2f} w={st['w']:+.2f}  tubes {st['n_valid']}/{st['n_tubes']}"
              f"  {st['choice'] or ('CERT' if st['certified'] else 'uncert')}{'  FALLBACK' if st['fallback'] else ''}"
              f"{'  SAT' if st['saturated'] else ''}"
              f"{'  r' + format(st['tube_radius'], '.2f') if st.get('tube_radius') else ''}"
              f"  cells {st['n_cells']}  {st['plan_ms']:.0f} ms")
    return s
