"""Closed-loop point-goal test of NavDP / X-NavDP / SanD-Planner on the Mars yard, in Habitat.

The rover is kinematic: it moves exactly as commanded (unicycle), its camera rides camera_height above the terrain
under it and stays level. Each camera frame (Habitat RGB + true depth) goes to the planner server, which returns
a path in the rover's frame; the same PathFollower as in the MuJoCo sim (../tube_nav.py) drives it. No BEV map:
the three planners run by themselves (as --modes pure / xnavdp / sandplanner in ../sim).

Collisions: the rocks' and flags' footprints (mars_scene.py) are solid. A step that would put the rover's circle
(--robot-radius) into one is not taken (the rover may turn in place), and counts as collision time.
Goal: a flag; reached when the rover's circle is --reach-edge from the pole.

    conda activate habitat
    python run_mars_habitat.py --planner navdp --goal flag_red          # server: ../sim/start_tube_server.sh
    python run_mars_habitat.py --planner xnavdp --goal flag_blue        # ../sim/start_xnavdp_server.sh
    python run_mars_habitat.py --planner sandplanner --goal 6,10        # ../sim/start_sandplanner_server.sh
    python run_mars_habitat.py --planner ours --test behind_1           # ours (guidance, no tubes): tube server
    python run_mars_habitat.py --list                                   # the tests (flags)

Writes runs/<name>/video.mp4, final.png, metrics.json (same keys as ../sim; ../sim/summarize_tests.py reads them).
"""

import argparse
import json
import math
import os
import sys
import time

try:                                     # not in the habitat env (e.g. conda base): run again with its python
    import habitat_sim  # noqa: F401
except ImportError:
    _py = os.path.join(os.environ.get("CONDA_EXE", os.path.expanduser("~/miniconda3/bin/conda")).rsplit("/bin/", 1)[0],
                       "envs", "habitat", "bin", "python")
    if os.path.exists(_py) and os.path.realpath(sys.executable) != os.path.realpath(_py):
        print(f"[env] habitat_sim not found in {sys.executable}; running with {_py}", flush=True)
        os.execv(_py, [_py] + sys.argv)
    sys.exit("habitat_sim not found: conda activate habitat")

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bev_pipeline import load_config  # noqa: E402
from mars_scene import MarsScene, stage_config  # noqa: E402
from seg_bev_map import SegBEVMap, add_args as add_seg_args  # noqa: E402
from tube_nav import TubeNavigator, WorldGrid, body_to_world, status_text  # noqa: E402

DEFAULT_GLB = os.path.expanduser("~/Desktop/mars_environment/marsyard_with_rocks.glb")
FLAGS = ("flag_red", "flag_blue", "flag_yellow", "flag_green", "flag_purple")
# rock right in front of the rover: start behind the rock, facing the flag (--test behind_N, or --behind ROCK)
BEHIND = {"behind_0": ("boulder_0", "flag_red"), "behind_1": ("boulder_1", "flag_yellow"),
          "behind_2": ("boulder_2", "flag_yellow"), "behind_3": ("boulder_3", "flag_green"),
          "behind_4": ("boulder_4", "flag_blue"), "behind_5": ("boulder_5", "flag_yellow")}
PLANNERS = {   # --planner -> nav settings (as run_sim_tests.sh --modes pure / xnavdp / sandplanner)
    "navdp": ["nav.planner=navdp", "nav.mode=navdp", "nav.pure_navdp=true", "nav.controller=track"],
    "xnavdp": ["nav.planner=xnavdp", "nav.mode=navdp", "nav.pure_navdp=true", "nav.controller=track"],
    "sandplanner": ["nav.planner=sandplanner", "nav.mode=navdp", "nav.pure_navdp=true", "nav.controller=track"],
    # ours: NavDP score-guided away from the map's obstacles, no tubes (= ../sim run_smooth_tests.sh --with-obstacle
    # --no-smooth, i.e. --modes obstacle --split 1 --lyapunov min --beta 2); needs the map (--map)
    "ours": ["nav.planner=navdp", "nav.mode=obstacle", "nav.pure_navdp=false", "nav.controller=track",
             "nav.tube_split=1", "nav.tube_lyapunov=min", "nav.tube_smooth=2"],
}
MODE_NAME = {"navdp": "pure", "xnavdp": "xnavdp", "sandplanner": "sandplanner",   # metrics "mode", as in ../sim
             "ours": "obstacle+b2+s1+lmin"}


def make_sim(scene_id, width, height, hfov):
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = scene_id
    sc.enable_physics = False
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.hfov = uuid, kind, [height, width], hfov
        s.position = np.zeros(3, np.float32)       # the agent's own position is the camera
        specs.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [ac]))
    return sim, sim.initialize_agent(0)


def place_camera(agent, x, y, yaw, cam_h):
    """Planner pose (x, y, yaw ccw from +x) -> Habitat agent at (X=x, Y=cam_h, Z=-y). Habitat looks along -Z at
    yaw 0 of its own, i.e. along planner +y: rotate about +Y by yaw - 90 deg."""
    st = agent.get_state()
    st.position = np.array([x, cam_h, -y], np.float32)
    th = yaw - math.pi / 2
    st.rotation = np.quaternion(math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
    agent.set_state(st)


def intrinsics(width, height, hfov_deg):
    f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))   # Habitat: square pixels
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


class ObstacleMap:
    """The map our method plans on (world frame, the scene's 5 cm grid), no robot size (the driver adds it).

    depth: each frame's depth pixels are put in the world; a point 10 cm to 1 m above the known terrain is an
           obstacle hit, and a cell with min_hits hits is occupied (kept for the rest of the run). Only what the
           camera has seen, up to max_range.
    (seg, the SAM2 segmentation BEV map, is seg_bev_map.SegBEVMap.)
    truth: the true rock / pole footprints (mars_scene), to check the method with a perfect map.
    Either way, the planner gets the cells within obstacle_range of the rover, the goal disk erased."""

    def __init__(self, scene, kind, K, max_range=6.0, min_h=0.10, max_h=1.0, min_hits=2, stride=2):
        self.s, self.kind, self.K = scene, kind, K
        self.max_range, self.min_h, self.max_h, self.min_hits, self.stride = max_range, min_h, max_h, min_hits, stride
        self.hits = np.zeros(scene.labels.shape, np.int32)
        g = scene.grid
        self.cx = g.x_min + (np.arange(g.nx) + 0.5) * g.res
        self.cy = g.y_min + (np.arange(g.ny) + 0.5) * g.res

    def add(self, depth, x, y, yaw, cam_h):
        if self.kind != "depth":
            return
        z = depth[::self.stride, ::self.stride]
        h, w = z.shape
        u = (np.arange(w) * self.stride)[None, :]
        v = (np.arange(h) * self.stride)[:, None]
        f, cx, cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
        ok = (z > 0.1) & (z < self.max_range)
        xr = ((u - cx) * z / f)[ok]                  # right of the optical axis
        yd = ((v - cy) * z / f)[ok]                  # down
        z = z[ok]
        c, s = math.cos(yaw), math.sin(yaw)
        px, py, ph = x + z * c + xr * s, y + z * s - xr * c, cam_h - yd
        above = ph - self.s.height_at(px, py)
        sel = (above >= self.min_h) & (above <= self.max_h)
        iy, ix = self.s.grid.index(px[sel], py[sel])
        np.add.at(self.hits, (iy, ix), 1)

    def occupied(self):
        return self.s.occ.copy() if self.kind == "truth" else self.hits >= self.min_hits

    def world_map(self, x, y, goal, nc):
        occ = self.occupied()
        g = self.s.grid
        X, Y = np.meshgrid(self.cx, self.cy)
        occ &= np.hypot(X - x, Y - y) <= nc.get("obstacle_range", 4.0)
        occ &= np.hypot(X - goal[0], Y - goal[1]) > nc.get("goal_clear_radius", 0.4)
        return WorldGrid(occ, g.res, (g.x_min, g.y_min))


class TopView:
    """Top-down map around start and goal: terrain shading, obstacles, trail, NavDP's paths."""

    def __init__(self, scene, a, b, margin=4.0, px_per_m=24):
        self.s, self.k = scene, px_per_m
        self.x0, self.x1 = min(a[0], b[0]) - margin, max(a[0], b[0]) + margin
        self.y0, self.y1 = min(a[1], b[1]) - margin, max(a[1], b[1]) + margin
        w, h = int((self.x1 - self.x0) * px_per_m), int((self.y1 - self.y0) * px_per_m)
        xs = self.x0 + (np.arange(w) + 0.5) / px_per_m
        ys = self.y1 - (np.arange(h) + 0.5) / px_per_m
        X, Y = np.meshgrid(xs, ys)
        H = cv2.GaussianBlur(scene.height_at(X, Y).astype(np.float32), (0, 0), 0.15 * px_per_m)
        gy, gx = np.gradient(H)
        shade = np.clip(0.6 + 4.0 * (gx - gy), 0.15, 1.0)
        base = (np.stack([shade * 150, shade * 175, shade * 215], -1)).astype(np.uint8)   # BGR sand
        iy, ix = scene.grid.index(X, Y)
        base[scene.occ[iy, ix]] = (60, 60, 60)
        self.base = base

    def px(self, x, y):
        return int((x - self.x0) * self.k), int((self.y1 - y) * self.k)

    def draw(self, trail, pose, radius, goal, goal_name, result=None, world_map=None, inflate=0.0):
        img = self.base.copy()
        if world_map is not None and world_map.occupied.any():      # our map: cells (red), inflation (pink)
            g = world_map
            iy, ix = np.nonzero(g.occupied)
            xs, ys = g.origin[0] + (ix + 0.5) * g.resolution, g.origin[1] + (iy + 0.5) * g.resolution
            keep = (xs > self.x0) & (xs < self.x1) & (ys > self.y0) & (ys < self.y1)
            pts = [self.px(a, b) for a, b in zip(xs[keep], ys[keep])]
            r = max(1, int(inflate * self.k))
            over = img.copy()
            for p in pts:
                cv2.circle(over, p, r, (180, 170, 255), -1)
            img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
            for p in pts:
                cv2.circle(img, p, max(1, int(0.5 * g.resolution * self.k)), (40, 40, 230), -1)
        if result is not None and result.candidates is not None:
            for c in result.candidates:
                pts = body_to_world(np.vstack([[0, 0], c[:, :2]]), pose)
                cv2.polylines(img, [np.array([self.px(*p) for p in pts], np.int32)], False, (170, 170, 170), 1)
        if result is not None and result.trajectory is not None:
            pts = body_to_world(np.vstack([[0, 0], result.trajectory[:, :2]]), pose)
            cv2.polylines(img, [np.array([self.px(*p) for p in pts], np.int32)], False, (60, 200, 60), 2)
        if len(trail) > 1:
            cv2.polylines(img, [np.array([self.px(*p) for p in trail], np.int32)], False, (0, 140, 255), 2)
        cv2.circle(img, self.px(*goal), 8, (0, 200, 0), 2)
        cv2.putText(img, goal_name, (self.px(*goal)[0] + 10, self.px(*goal)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 120, 0), 1, cv2.LINE_AA)
        c = self.px(pose[0], pose[1])
        cv2.circle(img, c, max(2, int(radius * self.k)), (200, 60, 20), 2)
        cv2.line(img, c, (int(c[0] + 18 * math.cos(pose[2])), int(c[1] - 18 * math.sin(pose[2]))), (200, 60, 20), 2)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", default="navdp", choices=list(PLANNERS),
                    help="navdp | xnavdp | sandplanner: the planner alone; ours: NavDP score-guided by the map, no tubes")
    ap.add_argument("--goal", default="flag_red", help="a flag (" + ", ".join(FLAGS) + ") or x,y (planner frame)")
    ap.add_argument("--start", default="0,0", help="x,y (planner frame; the yard's open start area is at 0,0)")
    ap.add_argument("--start-yaw", type=float, default=None, help="deg ccw from +x (default: facing the goal)")
    ap.add_argument("--test", default="", choices=[""] + list(BEHIND),
                    help="preset: behind_N = start right behind boulder N, facing a flag (sets --behind, --goal)")
    ap.add_argument("--behind", default="", help="a rock (boulder_0..5): start behind it on the line to the goal, "
                                                 "facing the goal (overrides --start)")
    ap.add_argument("--map", default="seg", choices=["seg", "depth", "truth"],
                    help="ours: the BEV map from the SAM2 terrain segmentation (seg, default: seg_bev_map.py, "
                         "the teleop_seg_bev.py map), from Habitat's depth, or the true rock footprints")
    ap.add_argument("--obstacle-range", type=float, default=4.0, help="ours: map cells within this of the rover (m)")
    ap.add_argument("--gap", type=float, default=1.0, help="--behind: m from the rover's edge to the rock's surface")
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--list", action="store_true", help="print the flags and exit")
    ap.add_argument("--name", default="", help="test name in metrics.json (default: the goal)")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-time", type=float, default=150.0, help="sim seconds")
    ap.add_argument("--stuck-time", type=float, default=20.0, help="stuck: moved < 0.25 m in this many s (0 = off)")
    ap.add_argument("--frame-dt", type=float, default=0.2, help="sim s between camera frames (one plan each)")
    ap.add_argument("--control-dt", type=float, default=0.05)
    ap.add_argument("--camera-height", type=float, default=0.5, help="camera above the terrain (m)")
    ap.add_argument("--robot-radius", type=float, default=0.3, help="rover circle for collisions (m)")
    ap.add_argument("--reach-edge", type=float, default=0.5, help="reached: rover circle this close to the pole (m)")
    ap.add_argument("--max-speed", type=float, default=0.5, help="m/s")
    ap.add_argument("--max-yaw-rate", type=float, default=0.6, help="rad/s")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=79.0, help="deg (79 = SanD-Planner's 640x480, fx 389.6 camera)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. nav.follow_lookahead_m=0.5")
    ap.add_argument("--no-window", action="store_true", help="no live window (default: shown if there is a screen)")
    add_seg_args(ap)
    args = ap.parse_args()

    scene = MarsScene(args.glb)
    if args.list:
        for f in FLAGS:
            if f in scene.origin:
                x, y = scene.origin[f]
                print(f"{f:12s} at ({x:+6.2f}, {y:+6.2f})  {math.hypot(x, y):5.1f} m from (0, 0)")
        for n, (rock, flag) in BEHIND.items():
            sx, sy = scene.start_behind(rock, scene.origin[flag], args.gap, args.robot_radius)
            print(f"{n:12s} start ({sx:+6.2f}, {sy:+6.2f}) behind {rock}, {args.gap:.1f} m from it, facing {flag} "
                  f"({math.hypot(scene.origin[flag][0] - sx, scene.origin[flag][1] - sy):.1f} m)")
        return

    if args.test:
        args.behind, args.goal = BEHIND[args.test]
    if args.goal in scene.origin:
        goal_name, goal = args.goal, scene.origin[args.goal]
    else:
        goal_name, goal = None, tuple(float(v) for v in args.goal.split(","))
    if args.behind:
        if args.behind not in scene.names:
            sys.exit(f"--behind: no rock {args.behind!r} (have {', '.join(n for n in scene.names if 'boulder' in n)})")
        start = scene.start_behind(args.behind, goal, args.gap, args.robot_radius)
    else:
        start = tuple(float(v) for v in args.start.split(","))
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0]) if args.start_yaw is None else math.radians(args.start_yaw)
    test = args.name or args.test or (f"behind_{args.behind}_{goal_name or 'goal'}" if args.behind
                                      else (goal_name or "goal"))

    cfg = load_config(overrides=PLANNERS[args.planner] + [
        f"nav.max_speed={args.max_speed}", f"nav.max_yaw_rate={args.max_yaw_rate}",
        "nav.stop_distance=0.0",           # this script decides "reached" (from the pole, or the goal point)
        f"nav.robot_radius={args.robot_radius}", "nav.follow_lookahead_m=0.5",
        "nav.max_accel=1.0", "nav.max_yaw_accel=3.0", f"nav.obstacle_range={args.obstacle_range}"] + args.set)
    nc = cfg["nav"]

    # obstacle distance fields: to everything but the goal flag, and to the goal flag
    others = scene.distance_field(exclude=(goal_name,) if goal_name else ())
    dist_goal = scene.distance_field(names=(goal_name,)) if goal_name else None

    out = args.out or os.path.join(HERE, "runs", "tests",
                                   time.strftime(f"{test}_{MODE_NAME[args.planner]}_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    sim, agent = make_sim(stage_config(args.glb, os.path.join(HERE, "runs")), args.width, args.height, args.hfov)
    K = intrinsics(args.width, args.height, args.hfov)
    uses_map = args.planner == "ours"
    bev = SegBEVMap(scene, K, args.width, args.height, start, args) if uses_map and args.map == "seg" else None
    omap = ObstacleMap(scene, args.map, K) if uses_map and bev is None else None
    inflate = float(nc.get("robot_radius", 0.16)) + float(nc.get("inflation_margin", 0.0))

    def cam_height(x, y):   # mean terrain under the rover's circle + the camera's height
        a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        xs = np.r_[x, x + args.robot_radius * np.cos(a)]
        ys = np.r_[y, y + args.robot_radius * np.sin(a)]
        return float(np.mean(scene.height_at(xs, ys))) + args.camera_height

    def observe(x, y, yaw):
        place_camera(agent, x, y, yaw, cam_height(x, y))
        o = sim.get_sensor_observations()
        rgb = np.ascontiguousarray(o["rgb"][..., :3])
        depth = np.asarray(o["depth"], np.float32)     # metres along the optical axis; 0 = sky
        return rgb, depth

    nav = TubeNavigator(nc)
    print(f"[nav] {args.planner}: resetting the server (the first time loads the model) ...", flush=True)
    try:
        nav.set_goal(goal, K)
    except RuntimeError as e:
        print(f"[error] {e}", flush=True)
        sim.close()
        sys.exit(1)

    show = bool(os.environ.get("DISPLAY")) and not args.no_window
    top = TopView(scene, start, goal)
    x, y = start
    trail, video = [(x, y)], None
    stats = {"collision_s": 0.0, "collided_with": set(), "path": 0.0, "min_clear": float("inf"),
             "min_edge": float("inf"), "plans": 0, "plan_ms": []}
    stuck_ref = (0.0, (x, y))
    per_frame = max(1, int(round(args.frame_dt / args.control_dt)))
    reached, reason, k, t = False, "timeout", 0, 0.0
    gname = goal_name or f"({goal[0]:.1f}, {goal[1]:.1f})"
    print(f"test {test}: {args.planner} from ({x:.1f}, {y:.1f}) to {gname} at ({goal[0]:.2f}, {goal[1]:.2f}), "
          f"{math.hypot(goal[0] - x, goal[1] - y):.1f} m", flush=True)
    try:
        while t < args.max_time:
            pose = (x, y, yaw)
            if k % per_frame == 0:                      # camera frame: plan (blocking, in sim time)
                rgb, depth = observe(x, y, yaw)
                world_map = None
                if bev is not None:                     # the SAM2 segmentation BEV map (seg_bev_map.py)
                    bev.update(rgb, depth, x, y, yaw, cam_height(x, y))
                    world_map = bev.world_map(x, y, goal, nc)
                elif omap is not None:
                    omap.add(depth, x, y, yaw, cam_height(x, y))
                    world_map = omap.world_map(x, y, goal, nc)
                nav.plan(rgb, depth, pose, t, world_map)
                if nav.error and stats["plans"] == 0:
                    reason = "server error"
                    print(f"[error] t={t:.1f}s  the planner server failed: {nav.error}")
                    break
                if not nav.error:
                    stats["plans"] += 1
                    stats["plan_ms"].append(nav.plan_ms)
                    kind = getattr(nav.driver.result, "choice_kind", "")
                    stats.setdefault("choice", {}).setdefault(kind, 0)
                    stats["choice"][kind] += 1
                res = nav.driver.result
                topimg = top.draw(trail, pose, args.robot_radius, goal, gname, res, world_map, inflate)
                cv2.putText(topimg, f"t={t:5.1f}s  {status_text(nav.status)}"[:80], (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                if bev is not None:                     # camera + segmentation | egocentric map with the paths
                    cam = cv2.resize(bev.seg_view(), (480, 360))

                    def paths(img, to_px):
                        if res is not None and res.candidates is not None:
                            for cnd in res.candidates:
                                p = body_to_world(np.vstack([[0, 0], cnd[:, :2]]), pose)
                                cv2.polylines(img, [to_px(p)], False, (170, 170, 170), 1, cv2.LINE_AA)
                        if res is not None and res.trajectory is not None:
                            p = body_to_world(np.vstack([[0, 0], res.trajectory[:, :2]]), pose)
                            cv2.polylines(img, [to_px(p)], False, (60, 200, 60), 2, cv2.LINE_AA)
                        cv2.circle(img, tuple(int(v) for v in to_px(np.array(goal))), 6, (0, 160, 0), 2)
                    ego = bev.ego_view(x, y, yaw, 360, args.robot_radius, extra=paths)
                    dv = np.full((360, 480, 3), 110, np.uint8)
                    dv[:, 60:420] = ego
                    cv2.putText(dv, "egocentric BEV map (SAM2)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cam = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (480, 360))
                    dv = cv2.applyColorMap((np.clip(depth / 10.0, 0, 1) * 255).astype(np.uint8),
                                           cv2.COLORMAP_TURBO)
                    dv[depth <= 0] = 0
                    dv = cv2.resize(dv, (480, 360))
                left = np.vstack([cam, dv])
                tv = cv2.resize(topimg, (int(topimg.shape[1] * 720 / topimg.shape[0]), 720))
                dash = np.hstack([left, tv])
                if video is None:
                    video = cv2.VideoWriter(os.path.join(out, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            1.0 / args.frame_dt, (dash.shape[1], dash.shape[0]))
                video.write(dash)
                if show:
                    cv2.imshow("Mars yard (Habitat)  Q = stop", dash)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        raise KeyboardInterrupt

            if dist_goal is not None:
                edge = scene.lookup(dist_goal, x, y) - args.robot_radius
                stats["min_edge"] = min(stats["min_edge"], edge)
                if edge <= args.reach_edge:
                    reached, reason = True, "reached"
                    print(f"[arrived] t={t:.1f}s  {max(edge, 0):.2f} m from the {goal_name}")
                    break
            elif math.hypot(goal[0] - x, goal[1] - y) <= args.reach_edge:
                reached, reason = True, "reached"
                print(f"[arrived] t={t:.1f}s")
                break

            v, w, st = nav.command(pose, t)
            dt = args.control_dt
            yaw_n = (yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
            xn = x + v * math.cos(yaw + 0.5 * w * dt) * dt
            yn = y + v * math.sin(yaw + 0.5 * w * dt) * dt
            clear_n = scene.lookup(others, xn, yn) - args.robot_radius
            goal_n = scene.lookup(dist_goal, xn, yn) - args.robot_radius if dist_goal is not None else 1.0
            if (clear_n <= 0 or goal_n <= 0) and abs(v) > 1e-6 or not scene.grid.inside(xn, yn):
                stats["collision_s"] += dt             # blocked: the rover stays, may still turn
                hit = scene.nearest_name(xn, yn, args.robot_radius + scene.grid.res)
                if hit:
                    stats["collided_with"].add(hit)
                xn, yn = x, y
            stats["path"] += math.hypot(xn - x, yn - y)
            x, y, yaw = xn, yn, yaw_n
            trail.append((x, y))
            stats["min_clear"] = min(stats["min_clear"], scene.lookup(others, x, y) - args.robot_radius)
            t += dt
            k += 1
            if args.stuck_time > 0 and t - stuck_ref[0] >= args.stuck_time:
                if math.hypot(x - stuck_ref[1][0], y - stuck_ref[1][1]) < 0.25:
                    reason = "stuck"
                    print(f"[stuck] t={t:.1f}s  moved < 0.25 m in the last {args.stuck_time:.0f} s")
                    break
                stuck_ref = (t, (x, y))
            if k % int(2.0 / dt) == 0:
                print(f"t={t:5.1f}  pose ({x:+6.2f},{y:+6.2f},{math.degrees(yaw):+4.0f}deg)  {status_text(st)}",
                      flush=True)
    except KeyboardInterrupt:
        reason = "stopped"
    finally:
        if video is not None:
            video.release()
            final_map = None
            if bev is not None:           # everything mapped during the run (no range limit)
                final_map = bev.occupied()
            elif omap is not None:
                final_map = WorldGrid(omap.occupied(), scene.grid.res, (scene.grid.x_min, scene.grid.y_min))
            cv2.imwrite(os.path.join(out, "final.png"), top.draw(trail, (x, y, yaw), args.robot_radius, goal, gname,
                                                                 world_map=final_map, inflate=inflate))
        if show:
            cv2.destroyAllWindows()
        nav.close()
        sim.close()

    rnd = lambda v: round(v, 3) if math.isfinite(v) else None  # noqa: E731
    metrics = {
        "test": test, "env": "mars_habitat", "glb": os.path.abspath(args.glb), "mode": MODE_NAME[args.planner],
        "planner": args.planner, "controller": "track", "goal_object": goal_name, "goal": list(goal),
        "behind_rock": args.behind or None, "gap_m": args.gap if args.behind else None,
        "start": list(start), "start_yaw_deg": round(math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0])
                                                                  if args.start_yaw is None else args.start_yaw), 1),
        "reached": reached, "end": reason, "time_s": round(t, 2), "path_length_m": round(stats["path"], 3),
        "straight_line_m": round(math.hypot(goal[0] - start[0], goal[1] - start[1]), 2),
        "final_goal_dist": round(math.hypot(goal[0] - x, goal[1] - y), 3),
        "reach_edge_m": args.reach_edge,
        "final_edge_dist_m": rnd(scene.lookup(dist_goal, x, y) - args.robot_radius) if dist_goal is not None else None,
        "min_edge_dist_m": rnd(stats["min_edge"]) if dist_goal is not None else None,
        "min_clearance_excl_goal_object_m": rnd(stats["min_clear"]),
        "collision_s": round(stats["collision_s"], 2), "collided_with": sorted(stats["collided_with"]),
        "map": args.map if uses_map else None, "obstacle_range_m": args.obstacle_range if uses_map else None,
        "inflation_radius_m": inflate if uses_map else None, "plan_choice": stats.get("choice"),
        "map_cells_final": (int(bev.occupied().occupied.sum()) if bev is not None
                            else int(omap.occupied().sum()) if uses_map else None),
        "seg_map": ({"occupied": bev.occ_c, "free": bev.free_c, "rocks": bev.rock_c, "range_m": args.range,
                     "hidden_free": not args.hidden_unknown} if bev is not None else None),
        "plans_made": stats["plans"], "plan_ms_mean": round(float(np.mean(stats["plan_ms"])), 1) if stats["plan_ms"] else None,
        "robot_radius_m": args.robot_radius, "camera_height_m": args.camera_height, "max_speed": args.max_speed,
        "max_yaw_rate": args.max_yaw_rate, "image": [args.width, args.height], "hfov_deg": args.hfov,
        "frame_dt": args.frame_dt, "control_dt": args.control_dt, "overrides": args.set,
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"[done] {reason}  t={t:.1f}s  path {stats['path']:.1f} m  collision {stats['collision_s']:.1f} s  -> {out}")


if __name__ == "__main__":
    main()
