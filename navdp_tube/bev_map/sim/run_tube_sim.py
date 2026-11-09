#!/usr/bin/env python3
"""Closed-loop test of tube-guided NavDP on the BEV map, in the MuJoCo arena.

The rover maps with Depth Anything (as on the real robot), the map's occupied cells (goal area erased) go to the
planner as they are, tube-guided NavDP plans from the same image + corrected depth, and its (v, omega) drives the
wheels.

    # terminal 1 (conda env navdp)
    cd navdp_tube/tube_navdp && python scripts/tube_navdp_server.py --navdp-root ../NavDP \
        --checkpoint ../NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt --port 8890
    # terminal 2 (conda env yolo)
    cd navdp_tube/bev_map/sim
    python run_tube_sim.py --list                   # all scenarios
    python run_tube_sim.py --scenario duck          # goal = the duck (the goal object is in the map)
    python run_tube_sim.py --scenario box_tv        # a box on the straight line to the TV
    python run_tube_sim.py --scenario box_tv --set nav.mode=navdp  # same without tubes, for comparison
    python run_tube_sim.py --goal thermos --box -0.6,-0.2,0.1      # own goal (object or x,y) / boxes (x,y,half)
    python run_tube_sim.py --scenario duck --web    # watch live in the browser (http://localhost:8090)
    python run_tube_sim.py --scenario duck --threads 2   # fewer CPU threads (default 4)
    python run_tube_sim.py --scenario duck --viewer # also the MuJoCo 3D viewer (needs a screen)

The scene objects (duck, TV, thermos, books) are made solid here (their mesh's convex hull), only in this script's
copy of the scene; the XML file is not changed. Goals are objects: NavDP gets the centre of the object's footprint,
and the run counts as reached when the rover's footprint comes within --reach-edge m of the object's surface
(true shape from the scene). The run also ends when the rover is stuck (moved < 0.10 m in --stuck-time s).

On a machine with a screen the dashboard opens in a window (camera, depth, BEV map with the planner's cells, NavDP's
paths and the tube); --no-window turns it off. Q / ESC in the window stops the run.

Writes runs/tube_<scenario>_<time>/: video.mp4, final.png, metrics.json (reached and why it ended, time, path
length, distance to the goal object's edge, closest approach to any other true obstacle, physical collisions,
how often tubes were valid / the path certified, BEV map quality against the true map).
"""

import argparse
import json
import math
import os
import sys
import time

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if not HAS_DISPLAY:                      # no screen (e.g. ssh): render offscreen; must be set before mujoco loads
    os.environ.setdefault("MUJOCO_GL", "egl")


def _threads_arg(default=4):
    """--threads N / --threads=N, read before numpy/torch load (they size their thread pools at import)."""
    for i, a in enumerate(sys.argv):
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
        if a == "--threads" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return default


THREADS = _threads_arg()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import cv2
import mujoco
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SIM_DIR))
SIM_CONFIG = os.path.join(SIM_DIR, "config_sim.yaml")

from bev_pipeline import BEVPipeline, load_config  # noqa: E402
from bev_render import depth_vis, label, render_global  # noqa: E402
from ground_truth import ground_truth_grid, score  # noqa: E402
from occupancy_grid import OCCUPIED  # noqa: E402
from sim_source import SimSource  # noqa: E402
from classic_nav import CLASSIC, NAMES, make_navigator  # noqa: E402
from tube_nav import draw_nav, nav_map, status_text  # noqa: E402

# Scene objects (geoms of ugv_rover_environment.xml). Footprints in the world (m): duck x 1.00..1.30 y -0.66..-0.44,
# TV x 1.03..1.38 y 0.35..0.75, thermos x 0.00..0.10 y -0.60..-0.46, books x -0.16..0.15 y 0.48..0.79.
# The rover starts at (-1.5, 0) facing +x.
OBJECTS = {
    "duck": ["duck_visual"],
    "tv": ["television_visual"],
    "thermos": ["thermos_visual"],
    "books": ["books_visual_1", "books_visual_2", "books_visual_3"],
}

# goal: an object name (or (x, y)); boxes: extra solid boxes (x, y, half size), 0.30 m tall.
SCENARIOS = {
    # A. object goals, nothing added
    "duck": {"goal": "duck", "boxes": [], "about": "drive to the duck (passes the thermos)"},
    "tv": {"goal": "tv", "boxes": [], "about": "drive to the TV (passes the books)"},
    "thermos": {"goal": "thermos", "boxes": [], "about": "drive to the thermos (small, close)"},
    "books": {"goal": "books", "boxes": [], "about": "drive to the book stack"},
    # B1. a box right on the straight line to the object: go around it, either side
    "box": {"goal": (0.6, 0.0), "boxes": [(-0.6, 0.0, 0.12)],
            "about": "box right on the straight line to the point (0.6, 0): go around it (the original test)"},
    "box_thermos": {"goal": "thermos", "boxes": [(-0.75, -0.26, 0.10)], "about": "box on the line to the thermos"},
    "box_tv": {"goal": "tv", "boxes": [(-0.6, 0.18, 0.10)], "about": "box on the line to the TV"},
    "box_duck": {"goal": "duck", "boxes": [(-0.5, -0.3, 0.12)], "about": "box on the line to the duck"},
    # B2. a box a little to one side: the straight line would clip it by ~5 cm, the free side is the other one
    "side_left": {"goal": "duck", "boxes": [(-0.4, -0.08, 0.10)],
                  "about": "box just left of the line to the duck: pass it on the right"},
    "side_right": {"goal": "duck", "boxes": [(-0.4, -0.38, 0.10)],
                   "about": "box just right of the line to the duck: pass it on the left"},
    # B3. two boxes across the line to the duck with a gap on the line (rover is 0.20 m wide)
    "gap_wide": {"goal": "duck", "boxes": [(-0.5, 0.3125, 0.12), (-0.5, -0.7275, 0.12)],
                 "about": "two boxes, 0.80 m gap on the line to the duck"},
    "gap_tight": {"goal": "duck", "boxes": [(-0.5, 0.1875, 0.12), (-0.5, -0.6025, 0.12)],
                  "about": "two boxes, 0.55 m gap on the line to the duck"},
    # B4. three staggered boxes to the duck: left of them, right of the 2nd, left of the 3rd (thermos in between)
    "slalom": {"goal": "duck", "boxes": [(-1.0, -0.15, 0.10), (-0.45, -0.05, 0.10), (0.55, -0.45, 0.10)],
               "about": "three staggered boxes on the way to the duck"},
    # B7. a small box where the rover would stop in front of the object, just outside the 0.40 m cleared disk
    "near_duck": {"goal": "duck", "boxes": [(0.60, -0.42, 0.08)],
                  "about": "box right in front of the duck: approach from another side"},
    "near_tv": {"goal": "tv", "boxes": [(0.64, 0.43, 0.08)],
                "about": "box right in front of the TV: approach from another side"},
}
WHEEL_R, WHEEL_LIMIT = 0.037, 5.0
FOOT_HALF = (0.125, 0.10)       # rover footprint half length / width with wheels (m)


def build_model(xml, boxes):
    """The scene with the objects made solid (convex hull of each mesh) and the extra boxes. The XML is not changed."""
    spec = mujoco.MjSpec.from_file(xml)
    solid = {n for names in OBJECTS.values() for n in names}
    for g in spec.geoms:
        if g.name in solid:
            g.contype, g.conaffinity = 1, 1
    for i, (x, y, h) in enumerate(boxes):
        g = spec.worldbody.add_geom()
        g.name, g.type = f"test_box_{i}", mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.rgba = [h, h, 0.15], [x, y, 0.15], [0.85, 0.35, 0.1, 1]
    return spec.compile()


class WheelControl:
    """(v, omega) -> wheel speeds. Skid steering slips a lot in this sim (a wheel speed difference gives only
    10-25 % of the kinematic turn rate), so the turn part is scaled up by kff.

    kp / ki add a PI loop on the measured yaw rate. Off by default: the yaw rate measured from 0.05 s pose
    differences spikes to +-4 rad/s with stick-slip, and with kp=8, ki=25 the loop drove the wheels to +-5 rad/s
    in opposite directions 85 % of the time, so the rover lurched (duck run: 39 s, jagged trail). Feed-forward
    only: 22.7 s, smooth trail; the path follower corrects the heading from each plan anyway.
    Try the old loop with --wheel-gains 8,25."""

    def __init__(self, kff=12.0, kp=0.0, ki=0.0):
        self.kff, self.kp, self.ki, self.i = kff, kp, ki, 0.0

    def __call__(self, v, w, w_meas, dt):
        err = w - w_meas
        self.i = float(np.clip(self.i + self.ki * err * dt, -WHEEL_LIMIT, WHEEL_LIMIT))
        if v == 0.0 and w == 0.0:
            self.i = 0.0
            return 0.0, 0.0
        diff = self.kff * w + self.kp * err + self.i
        base = v / WHEEL_R
        return (float(np.clip(base - diff, -WHEEL_LIMIT, WHEEL_LIMIT)),
                float(np.clip(base + diff, -WHEEL_LIMIT, WHEEL_LIMIT)))


def footprint_clearance(dist_m, grid, xy, yaw):
    """Smallest distance (m) from the rover's footprint rectangle to a true obstacle cell (<= 0: touching)."""
    lx, ly = FOOT_HALF
    u = np.linspace(-lx, lx, 11)
    v = np.linspace(-ly, ly, 9)
    edge = np.concatenate([np.stack([u, np.full_like(u, s * ly)], 1) for s in (-1, 1)]
                          + [np.stack([np.full_like(v, s * lx), v], 1) for s in (-1, 1)])
    c, s = math.cos(yaw), math.sin(yaw)
    px = xy[0] + c * edge[:, 0] - s * edge[:, 1]
    py = xy[1] + s * edge[:, 0] + c * edge[:, 1]
    ix, iy, ok = grid.world_to_cell(px, py)
    if not ok.any():
        return float("inf")
    return float(dist_m[iy[ok], ix[ok]].min())


def lyapunov_share(plans, nav):
    """Share of guided plans whose mean dist^2 to the tube fell by at least the factor (1 - delta) at every guided
    step (the Lyapunov condition; it is checked per sample in the sampler, this is the mean), or None when off."""
    if nav.pure or nav.lyapunov in (None, "off"):
        return None
    delta = 0.3 if nav.lyapunov_delta is None else nav.lyapunov_delta
    seqs = [p["lyapunov_v"] for p in plans if p.get("lyapunov_v") and len(p["lyapunov_v"]) > 1]
    if not seqs:
        return None
    ok = [all(b <= (1 - delta) * a + 1e-9 for a, b in zip(v[:-1], v[1:])) for v in seqs]
    return round(sum(ok) / len(ok), 3)


def release_metrics(plans, nav) -> dict:
    """guide_until check: does NavDP keep the guided paths clear once the pull stops? Over the plans: the share whose
    chosen path has a waypoint in an occupied (grown) cell, the mean V (1/2 sum dist^2 to the free cells) of the chosen
    and of all finished paths, and the mean V of the model's own clean-path estimate at every denoising step."""
    if nav.pure:
        return {}
    got = [p for p in plans if p.get("chosen_v") is not None]
    if not got:
        return {}
    mean = lambda xs: round(float(np.mean(xs)), 5)
    steps = [p["model_v"] for p in got if p.get("model_v")]
    width = min((len(v) for v in steps), default=0)
    return {"chosen_in_cells_share": round(sum(bool(p["chosen_in_cells"]) for p in got) / len(got), 3),
            "chosen_v_mean": mean([p["chosen_v"] for p in got]),
            "final_v_mean": mean([p["final_v"] for p in got]),
            "final_in_cells_mean": mean([p["final_in_cells"] for p in got]),
            "model_v_mean": [mean([v[i] for v in steps]) for i in range(width)] if width else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="duck", choices=list(SCENARIOS))
    ap.add_argument("--list", action="store_true", help="print the scenarios and exit")
    ap.add_argument("--name", default="", help="test name written to metrics.json (default: the scenario)")
    ap.add_argument("--goal", default="", help="object name (" + ", ".join(OBJECTS) + ") or x,y; overrides the "
                                               "scenario's goal")
    ap.add_argument("--reach-edge", type=float, default=0.35,
                    help="object goal: reached when the rover's footprint is this close to the object's surface (m). "
                         "Keep >= ~0.3: NavDP turns paths shorter than 0.5 m into 'turn in place'")
    ap.add_argument("--stuck-time", type=float, default=20.0,
                    help="end the run as 'stuck' if the rover moved < 0.10 m in this many sim seconds (0 = off)")
    ap.add_argument("--box", nargs="*", default=None, help="extra boxes x,y,half (overrides the scenario's)")
    ap.add_argument("--max-time", type=float, default=120.0, help="sim seconds")
    ap.add_argument("--frame-dt", type=float, default=0.2, help="sim seconds between camera frames (map + plan)")
    ap.add_argument("--control-dt", type=float, default=0.05)
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1",
                    help="web page interface: 127.0.0.1 = this machine only (use an ssh tunnel), 0.0.0.0 = network")
    ap.add_argument("--out", default="")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. nav.mode=navdp")
    ap.add_argument("--threads", type=int, default=THREADS, help="CPU threads for torch / OpenCV / BLAS (default 4)")
    ap.add_argument("--no-window", action="store_true", help="no dashboard window (default: shown if there is a screen)")
    ap.add_argument("--no-top", action="store_true", help="no top-down camera panel in the dashboard / video")
    ap.add_argument("--viewer", action="store_true", help="also open the MuJoCo 3D viewer (needs a screen)")
    ap.add_argument("--wheel-gains", default="0,0",
                    help="kp,ki of the wheel yaw-rate PI loop (default 0,0 = feed-forward only; old setting 8,25)")
    args = ap.parse_args()
    if args.list:
        for n, s in SCENARIOS.items():
            print(f"{n:12s} goal {s['goal']!s:8s} boxes {len(s['boxes'])}  {s['about']}")
        return
    show_window = HAS_DISPLAY and not args.no_window
    if args.viewer and not HAS_DISPLAY:
        print("[info] no screen: --viewer ignored (use --web to watch)")
        args.viewer = False
    cv2.setNumThreads(THREADS)
    import torch
    torch.set_num_threads(THREADS)

    goal, boxes = SCENARIOS[args.scenario]["goal"], SCENARIOS[args.scenario]["boxes"]
    if args.goal:
        goal = args.goal if args.goal in OBJECTS else tuple(float(v) for v in args.goal.split(","))
    goal_object = goal if isinstance(goal, str) else None
    if args.box is not None:
        boxes = [tuple(float(v) for v in b.split(",")) for b in args.box]

    cfg = load_config(overrides=args.set, extra=[SIM_CONFIG])
    nc, rc = cfg["nav"], cfg["render"]
    xml = cfg["sim"]["xml_path"]
    xml = xml if os.path.isabs(xml) else os.path.normpath(os.path.join(SIM_DIR, xml))
    src = SimSource(cfg, build_model(xml, boxes))
    pipe = BEVPipeline(cfg)
    src.settle(0.5)

    gt, gt_ids = ground_truth_grid(src.model, pipe.grid, cfg["sim"]["rover_body"], cfg["classify"]["obstacle_min_z"],
                                   return_ids=True)
    dt = lambda occ: cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * pipe.grid.res  # noqa: E731
    dist_m = dt(gt)
    dist_goal = None
    if goal_object is not None:          # the object's true cells; NavDP's goal = the centre of their bounding box
        on_goal = np.isin(gt_ids, [src.model.geom(n).id for n in OBJECTS[goal_object]])
        gx, gy = pipe.grid.cell_x[on_goal], pipe.grid.cell_y[on_goal]
        goal = (round(float(gx.min() + gx.max()) / 2, 3), round(float(gy.min() + gy.max()) / 2, 3))
        dist_goal = dt(on_goal)
        nc["stop_distance"] = 0.0        # this script decides "reached" from the object's edge
    else:
        on_goal = np.hypot(pipe.grid.cell_x - goal[0], pipe.grid.cell_y - goal[1]) <= nc["goal_clear_radius"]
    dist_other = dt(gt & ~on_goal)
    floor_id = src.model.geom("floor").id
    rover_bodies = {b for b in range(src.model.nbody) if src.model.body_rootid[b] == src.rover_id}

    out = args.out or os.path.join(SIM_DIR, "runs", time.strftime(f"tube_{args.scenario}_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    video = None
    web = None
    if args.web:
        from web_view import WebView
        web = WebView(args.host, args.port)
        url = web.url
        if args.host == "0.0.0.0":
            import socket
            url = f"http://{socket.gethostbyname(socket.gethostname())}:{args.port}  (or http://<this machine's IP>:{args.port})"
        print("=" * 60 + f"\n  Open {url} in your browser\n" + "=" * 60, flush=True)

    nav = make_navigator(nc)
    first = src.get_frame()
    print("[nav] resetting NavDP on the server (the first time loads the model) ...", flush=True)
    try:
        nav.set_goal(goal, first.K)
    except RuntimeError as e:
        print(f"[error] {e}", flush=True)
        if web is not None:
            web.close()
        sys.exit(1)
    kp, ki = (float(g) for g in args.wheel_gains.split(","))
    wheels = WheelControl(kp=kp, ki=ki)

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(src.model, src.data)
    WIN = "tube NavDP sim  (Q = stop)"

    steps = max(1, int(round(args.control_dt / src.model.opt.timestep)))
    per_frame = max(1, int(round(args.frame_dt / args.control_dt)))
    trail, plans = [], []
    stats = {"contact_s": 0.0, "min_clear": float("inf"), "min_clear_other": float("inf"),
             "collision_s": 0.0, "collided_with": set(), "min_edge": float("inf"), "path": 0.0}
    xy0, yaw_prev = src.robot_pose()
    stuck_ref = (0.0, xy0.copy())
    v = w = 0.0
    k = 0
    reached, reason = False, "timeout"
    goal_txt = f"{goal_object} at {goal}" if goal_object else f"{goal}"
    print(f"scenario {args.scenario}: goal {goal_txt}, boxes {boxes}, mode {NAMES.get(nav.planner, 'pure NavDP') if nav.pure else nc['mode'] + '/' + nc['controller'] + ' push ' + str(nav.push or 'server default') + f' consistency {nav.consistency} smooth {nav.smooth}'}")
    try:
        while src.data.time < args.max_time:
            t = float(src.data.time)
            xy, yaw = src.robot_pose()
            pose = (float(xy[0]), float(xy[1]), yaw)
            if k % per_frame == 0:                           # camera frame: map, planner's cells, plan (blocking)
                f = src.get_frame()
                r = pipe.process(f)
                state = pipe.grid.state()
                world_map = nav_map(pipe.grid, state, xy, nav.goal or goal, nc)
                if nav.goal is not None:
                    nav.plan(f.rgb, r["depth"], pose, t, world_map)
                    st = nav.status
                    if nav.error and not plans:          # the very first plan failed: say why, do not wait 20 s
                        reason = "server error"
                        print(f"[error] t={t:.1f}s  the planner server failed: {nav.error}")
                        break
                    if nav.driver.result is not None and (nav.driver.result.tube is not None or nav.pure):
                        tb = nav.driver.result.tube or {}
                        plans.append({"t": t, "certified": bool(tb.get("chosen_certified")), "push": tb.get("push"),
                                      "choice": getattr(nav.driver.result, "choice_kind", ""),
                                      "clear_m": getattr(nav.driver.result, "clear_m", None),
                                      "valid": sum(bool(x.get("feasible")) for x in tb.get("tubes", [])),
                                      "tubes": len(tb.get("tubes", [])), "cells": int(world_map.occupied.sum()),
                                      "lyapunov_v": tb.get("lyapunov_v"), "builds": tb.get("tube_builds"),
                                      "model_v": tb.get("model_v"), "final_v": tb.get("final_v"),
                                      "final_in_cells": tb.get("final_in_cells"), "chosen_v": tb.get("chosen_v"),
                                      "chosen_in_cells": tb.get("chosen_in_cells")})
                else:
                    nav.map = world_map
                gimg = render_global(pipe.grid, state, rc["scale"], xy, yaw, nc["robot_radius"], gt_occ=gt,
                                     trail=trail)
                draw_nav(gimg, pipe.grid, rc["scale"], nav)
                label(gimg, f"t={t:5.1f}s  {status_text(nav.status)}"[:110], (8, 22))
                cam = cv2.resize(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), (480, 270))
                dep = cv2.resize(depth_vis(r["depth"], cfg["depth"]["max_range"]), (480, 270))
                left = np.vstack([cam, dep])
                g = cv2.resize(gimg, (int(gimg.shape[1] * 540 / gimg.shape[0]), 540))
                dash = np.hstack([left, g])
                if not args.no_top:      # the real scene from straight above, beside the map
                    dash = np.hstack([dash, cv2.cvtColor(src.render_top(540), cv2.COLOR_RGB2BGR)])
                if video is None:
                    video = cv2.VideoWriter(os.path.join(out, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            1.0 / args.frame_dt, (dash.shape[1], dash.shape[0]))
                video.write(dash)
                if web is not None:
                    web.status = status_text(nav.status)
                    web.show(dash)
                    for key in web.pop_keys():
                        if key == "q":
                            raise KeyboardInterrupt
                if show_window:
                    cv2.imshow(WIN, dash)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        raise KeyboardInterrupt

            if dist_goal is not None:
                edge = footprint_clearance(dist_goal, pipe.grid, xy, yaw)
                stats["min_edge"] = min(stats["min_edge"], edge)
                if edge <= args.reach_edge:
                    reached, reason = True, "reached"
                    print(f"[arrived] t={t:.1f}s  {edge:.2f} m from the {goal_object}'s edge")
                    break
            v, w, st = nav.command(pose, t)
            if st.get("state") == "arrived":
                reached, reason = True, "reached"
                print(f"[arrived] t={t:.1f}s  goal distance {st.get('goal_dist', 0):.2f} m")
                break
            if args.stuck_time > 0 and t - stuck_ref[0] >= args.stuck_time:
                if np.hypot(*(xy - stuck_ref[1])) < 0.10:
                    reason = "stuck"
                    print(f"[stuck] t={t:.1f}s  moved < 0.10 m in the last {args.stuck_time:.0f} s")
                    break
                stuck_ref = (t, xy.copy())
            w_meas = ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / args.control_dt
            yaw_prev = yaw
            src.set_wheels(*wheels(v, w, w_meas, args.control_dt))
            src.step(steps)
            k += 1
            if viewer is not None:
                if not viewer.is_running():          # viewer window closed -> stop the run
                    raise KeyboardInterrupt
                viewer.sync()

            if trail:
                stats["path"] += float(np.hypot(*(xy - trail[-1])))
            trail.append(xy.copy())
            hit = False                                  # physical contact of the rover with anything but the floor
            for c in src.data.contact[:src.data.ncon]:
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if (src.model.geom_bodyid[a] in rover_bodies and b != floor_id
                            and src.model.geom_bodyid[b] not in rover_bodies):
                        hit = True
                        stats["collided_with"].add(src.model.geom(b).name or f"geom{b}")
            if hit:
                stats["collision_s"] += args.control_dt
            clear = footprint_clearance(dist_m, pipe.grid, xy, yaw)
            stats["min_clear"] = min(stats["min_clear"], clear)
            stats["min_clear_other"] = min(stats["min_clear_other"],
                                           footprint_clearance(dist_other, pipe.grid, xy, yaw))
            if clear <= pipe.grid.res:
                stats["contact_s"] += args.control_dt
            if k % int(2.0 / args.control_dt) == 0:
                print(f"t={t:5.1f}  pose ({xy[0]:+.2f},{xy[1]:+.2f},{math.degrees(yaw):+4.0f}deg)  "
                      f"{status_text(st)}")
    except KeyboardInterrupt:
        reason = "stopped"
    finally:
        src.set_wheels(0, 0)
        if viewer is not None:
            viewer.close()
        if show_window:
            cv2.destroyAllWindows()
        if video is not None:
            video.release()
            if "gimg" in locals():
                cv2.imwrite(os.path.join(out, "final.png"), gimg)
        nav.close()

    xy, yaw = src.robot_pose()
    rnd = lambda x: round(x, 3) if math.isfinite(x) else None  # noqa: E731
    map_q = score(pipe.grid.state(), gt)
    metrics = {
        "test": args.name or args.scenario,
        "scenario": args.scenario, "goal_object": goal_object, "goal": goal, "boxes": boxes,
        "mode": nc["mode"], "controller": "dwa" if "dwa" in nav.planner else nc["controller"], "planner": nav.planner, "push": None if nav.pure else nav.push,
        "push_used_by_server": sorted({str(p.get("push")) for p in plans}) if plans and not nav.pure else None,
        "consistency": None if nav.pure else nav.consistency, "smooth_beta": None if nav.pure else nav.smooth,
        "split": None if nav.pure else nav.split, "rebuild": None if nav.pure else nav.rebuild,
        "lyapunov": None if nav.pure else nav.lyapunov, "lyapunov_delta": None if nav.pure else nav.lyapunov_delta,
        "lyapunov_v_last_plan": (plans[-1].get("lyapunov_v") if plans else None),
        "lyapunov_decrease_ok": lyapunov_share(plans, nav),
        "guide_until": None if nav.pure else nav.guide_until,
        "push_step": None if nav.pure else nav.push_step, "push_margin": None if nav.pure else nav.push_margin,
        **release_metrics(plans, nav),
        "reached": reached, "end": reason, "time_s": round(float(src.data.time), 2),
        "path_length_m": round(stats["path"], 2),
        "straight_line_m": round(math.hypot(goal[0] - xy0[0], goal[1] - xy0[1]), 2),
        "final_goal_dist": round(math.hypot(goal[0] - xy[0], goal[1] - xy[1]), 3),
        "reach_edge_m": args.reach_edge if goal_object else None,
        "final_edge_dist_m": rnd(footprint_clearance(dist_goal, pipe.grid, xy, yaw)) if goal_object else None,
        "min_edge_dist_m": rnd(stats["min_edge"]) if goal_object else None,
        "min_footprint_clearance_m": rnd(stats["min_clear"]),
        "min_clearance_excl_goal_object_m": rnd(stats["min_clear_other"]),
        "contact_s": round(stats["contact_s"], 2),
        "collision_s": round(stats["collision_s"], 2),
        "collided_with": sorted(stats["collided_with"]),
        "map_precision": round(map_q["precision"], 3), "map_recall": round(map_q["recall"], 3),
        "map_false_free": round(map_q["false_free"], 3),
        "plans": len(plans),
        "plans_certified": sum(p["certified"] for p in plans),
        "plans_with_valid_tube": sum(p["valid"] > 0 for p in plans),
        "plans_farthest": sum(p["choice"] == "farthest" for p in plans),
        "farthest_clear_m_mean": (round(float(np.mean([p["clear_m"] for p in plans if p.get("clear_m") is not None])), 3)
                                  if any(p.get("clear_m") is not None for p in plans) else None),
        "mean_obstacle_cells": round(float(np.mean([p["cells"] for p in plans])), 1) if plans else 0,
        "overrides": args.set,
    }
    if nav.pure:                      # pure NavDP: no tubes or map fallbacks, so drop their counters
        for key in ("plans_certified", "plans_with_valid_tube", "plans_farthest", "farthest_clear_m_mean",
                    "mean_obstacle_cells"):
            metrics.pop(key, None)
        metrics["mode"] = nav.planner if nav.planner in ("xnavdp", "sandplanner") + CLASSIC else "pure"
    elif nc["mode"] == "tube" and nav.push == "smooth":
        metrics["mode"] = "smooth"    # tubes with the smooth push: its own row in the summary
    if not nav.pure:                  # consistency / smooth projection on: own rows too (e.g. smooth+c0.3+b2)
        metrics["mode"] += (f"+c{nav.consistency:g}" if nav.consistency else "") + (f"+b{nav.smooth:g}" if nav.smooth else "")
        metrics["mode"] += ((f"+s{nav.split}" if nav.split is not None else "") + ("+r" if nav.rebuild else "")
                            + (f"+u{nav.guide_until}" if nav.guide_until else "")
                            + (f"+p{nav.push_step:g}" if nav.push_step and nav.push_step != 0.25 else "")
                            + (f"+m{nav.push_margin:g}" if nav.push_margin else "")
                            + (f"+l{nav.lyapunov}" if nav.lyapunov not in (None, "off") else "")
                            + (f"{nav.lyapunov_delta:g}" if nav.lyapunov not in (None, "off") and nav.lyapunov_delta
                               else ""))
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
    print(json.dumps(metrics, indent=1))
    print(f"-> {out}")
    if reason == "stopped":           # quit by hand (Ctrl+C / q): tell run_sim_tests.sh to stop, not run the next one
        sys.exit(130)


if __name__ == "__main__":
    main()
