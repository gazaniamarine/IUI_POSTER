"""Data collection on the Mars yard (Habitat): NavDP and X-NavDP on identical random episodes, logged every step.

Each episode (seed = --seed + i, episode_gen.make_episode):
  - the yard of marsyard_with_rocks.glb (terrain, 5 flags) with its 6 boulders moved to random spots 1..6 m from the
    flags (1..3 of them round the goal flag), each turned at random and seated in the ground
  - goal: a flag; start: a spawn mode (random, facing_goal, facing_away, behind_rock, near_rock), seeded
  - oracle: A* on a 0.1 m grid over the true rock / pole footprints grown by the robot radius (slopes and craters do
    not block), shortened by line of sight; its length is the reference (oracle_length_m)
  - every planner (--planners navdp,xnavdp) drives the same episode, with the same per-step sampling seeds

Writes --out/ep_<seed>/:
  episode.json         seed, spawn, start pose, goal, boulder placements, oracle route, obstacle names
  top.png              the episode from above: route (white), each planner's trail, its events
  <planner>/steps.npz  one row per plan step (0.2 s), see STEP_FIELDS below
  <planner>/events.json  result (reached, time, SPL, ...) + the automatic events (episode_events.py)
  <planner>/acts.npz   every --act-every steps: scene tokens, decoder layer, critic features (navdp/tube/acts.py)
and one line per run in --out/index.jsonl. Finished episodes are skipped when run again (resume).

    conda activate habitat
    # servers (from ../sim, after any change to their code): ./start_tube_server.sh ; ./start_xnavdp_server.sh
    python collect_episodes.py --episodes 200 --out runs/collect
    python collect_episodes.py --episodes 2 --seed 1000 --out runs/collect_test --save-frames
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np

from run_mars_habitat import (DEFAULT_GLB, PLANNERS, HERE, intrinsics, make_sim,  # noqa: E402  (re-execs into habitat)
                              place_camera)
from bev_pipeline import load_config  # noqa: E402
from episode_events import DEFAULTS as EVENT_DEFAULTS, detect  # noqa: E402
from episode_gen import (KINDS, SPAWN_MODES, Route, Yard, draw_episode, make_episode, prepare_assets,  # noqa: E402
                         save_json)
from mars_scene import stage_config  # noqa: E402
from tube_nav import TubeNavigator  # noqa: E402

COLORS = {"navdp": (0, 140, 255), "xnavdp": (230, 60, 200)}     # BGR trails
STEP_FIELDS = """
  t [N]                 sim time of the plan step (s)
  pose [N, 3]           x, y, yaw (planner frame, rad) when the image was taken
  togo_route [N]        distance to go along the oracle route from the nearest route point (m)
  togo_geo [N]          shortest safe distance to the goal from here (Dijkstra on the oracle grid; inf if blocked)
  route_s [N]           arclength of the nearest route point (m)
  cross_track [N]       signed distance from the route (+ = left of it, travelling to the goal) (m)
  route_turn_deg [N]    where the route goes 2 m ahead of the nearest route point, relative to the heading (+ left)
  goal_edge [N]         rover circle to the goal pole (m)
  clear_solid [N]       rover centre to the nearest rock / pole footprint, goal pole excluded (true geometry) (m)
  nearest_solid [N]     its index in episode.json "obstacles" (-1 none); nearest_solid_kind [N]: KINDS code
  clear_hazard [N]      to the nearest steep slope / crater cell (0 = on one) (m); hazard_kind [N]: KINDS code
  clearance [N], nearest_kind [N]   the nearer of the two (rock, flag, slope or crater)
  facing_obstacle [N], facing_dist [N]   first footprint straight ahead within 5 m (-1 / 5 = none)
  v_cmd, w_cmd [N]      mean commanded speed (m/s) and turn rate (rad/s) over the step; v_first, w_first: its first tick
  blocked_ticks [N]     control ticks whose move was blocked (would enter a footprint or leave the yard); ticks [N]
  moved [N]             distance moved during the step (m)
  candidates [N, S, 24, 3]  all sampled paths (rover frame: forward, left, yaw); values [N, S]: the critic's scores
  trajectory [N, 24, 3] the chosen (executed) path; choice [N]: its index in candidates (choice_exact: matched exactly;
                        false when the stop rule rewrote it)
  stop_fired [N]        NavDP's stop rule (best critic value < nav.stop_threshold; never for X-NavDP, it has none)
  turn_in_place [N]     the chosen path has no forward motion (NavDP's short-path rule or its stop)
  planner_stuck [N]     X-NavDP's own stuck flag (it then executes a random sample); -1 = not reported
  goal_body [N, 2]      the goal in the rover frame sent to the planner; plan_ms [N]; plan_ok [N]
  act_index [N]         row in acts.npz for this step (-1 = none)
"""
KIND_CODE = {k: i for i, k in enumerate(KINDS)}


def nav_config(planner, args):
    return load_config(overrides=PLANNERS[planner] + [
        f"nav.max_speed={args.max_speed}", f"nav.max_yaw_rate={args.max_yaw_rate}", "nav.stop_distance=0.0",
        f"nav.robot_radius={args.robot_radius}", "nav.follow_lookahead_m=0.5", "nav.max_accel=1.0",
        "nav.max_yaw_accel=3.0"] + args.set)["nav"]


class Boulders:
    """The 6 boulders as Habitat objects, moved for every episode."""

    def __init__(self, sim, rock_glbs):
        import magnum as mn
        self.mn = mn
        otm, rom = sim.get_object_template_manager(), sim.get_rigid_object_manager()
        self.obj = {}
        for b, path in rock_glbs.items():
            t = otm.create_new_template(path)
            t.compute_COM_from_shape = False      # the baked .glb is centred already: keep its frame
            self.obj[b] = rom.add_object_by_template_id(otm.register_template(t, f"collect_{b}"))

    def move(self, translations, placements):
        mn = self.mn
        for b, o in self.obj.items():
            o.translation = mn.Vector3(*translations[b])
            o.rotation = mn.Quaternion.rotation(mn.Rad(placements[b][2]), mn.Vector3(0.0, 1.0, 0.0))


def pack_acts(tmp_dir, rows, out_path):
    """Stack the per-step activation files into one acts.npz (float16), then remove them. -> rows written."""
    data, steps = {}, []
    for step, path in rows:
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
        steps.append(step)
    if steps:
        out = {"step": np.array(steps, np.int32)}
        for k, v in data.items():
            try:
                out[k] = np.stack(v)
            except ValueError:                    # ragged (should not happen): keep as objects
                out[k] = np.array(v, dtype=object)
        np.savez(out_path, **out)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return len(steps)


def run_planner(planner, nav, sim, agent, K, yard, ep, oracle, route, run_dir, args):
    """One planner on one episode. -> (summary dict, step arrays dict)."""
    s = yard.scene
    goal_name = ep["goal"]
    goal = tuple(ep["goal_xy"])
    r = args.robot_radius
    goal_field = s.distance_field(names=(goal_name,))
    others = s.distance_field(exclude=(goal_name,))
    obstacles = ep["obstacles"]
    label_to_obstacle = {i: obstacles.index(n) for i, n in enumerate(s.names) if n in obstacles}
    os.makedirs(run_dir, exist_ok=True)
    tmp_acts = os.path.join(run_dir, "_acts")
    frames_dir = os.path.join(run_dir, "frames")

    def cam_height(x, y):
        a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        return float(np.mean(s.height_at(np.r_[x, x + r * np.cos(a)], np.r_[y, y + r * np.sin(a)]))) + args.camera_height

    nav.set_goal(goal, K)
    client = nav.driver.client
    nc = nav.nc
    stop_thr = float(nc.get("stop_threshold", -3.0))
    max_time = args.max_time or (args.time_factor * ep["oracle_length_m"] / args.max_speed + 30.0)
    per_frame = max(1, int(round(args.frame_dt / args.control_dt)))
    x, y = ep["start"]
    yaw = ep["yaw"]
    rows, acts_rows = [], []
    state = {"cur": None}       # the plan step being filled (its commands come in the control ticks after it)
    reached, reason, k, t, path_len = False, "timeout", 0, 0.0, 0.0

    def close_step():
        cur = state.pop("cur", None)
        state["cur"] = None
        if cur is not None:
            n = max(cur["ticks"], 1)
            cur.update(v_cmd=cur.pop("v_sum") / n, w_cmd=cur.pop("w_sum") / n)
            rows.append(cur)

    try:
        while t < max_time:
            pose = (x, y, yaw)
            edge = s.lookup(goal_field, x, y) - r
            if edge <= args.reach_edge:
                reached, reason = True, "reached"
                break
            if k % per_frame == 0:
                close_step()
                step = len(rows)
                # stuck long enough: end the run (the stuck event itself is marked at stuck_steps)
                if args.stuck_end and step > args.stuck_end:
                    win = np.array([[q["pose"][0], q["pose"][1]] for q in rows[-args.stuck_end - 1:]])
                    if np.max(np.linalg.norm(win - win[0], axis=1)) < args.stuck_dist:
                        reason = "stuck"
                        break
                place_camera(agent, x, y, yaw, cam_height(x, y))
                o = sim.get_sensor_observations()
                rgb = np.ascontiguousarray(o["rgb"][..., :3])
                depth = np.asarray(o["depth"], np.float32)
                want_acts = args.act_every > 0 and step % args.act_every == 0
                act_path = os.path.join(tmp_acts, f"{step:06d}.npz") if want_acts else None
                client.acts_path = act_path
                client.step_seed = (ep["seed"] * 100003 + step * 31) % (2 ** 31 - 1)   # same for every planner
                res = nav.plan(rgb, depth, pose, t, None)
                client.acts_path = None
                if res is None and step == 0:
                    raise RuntimeError(f"{planner} server failed: {nav.error}")
                if want_acts and res is not None and getattr(res, "acts_saved", False):
                    acts_rows.append((step, act_path))
                    act_index = len(acts_rows) - 1
                else:
                    act_index = -1
                if args.save_frames and want_acts:
                    os.makedirs(frames_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(frames_dir, f"{step:06d}_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(os.path.join(frames_dir, f"{step:06d}_depth.png"),
                                np.clip(depth * 1000, 0, 65535).astype(np.uint16))
                state["cur"] = log_step(yard, oracle, route, ep, pose, t, edge, res, stop_thr, planner, act_index, r,
                               label_to_obstacle, nav.plan_ms)

            v, w, _ = nav.command(pose, t)
            dt = args.control_dt
            yaw_n = (yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
            xn = x + v * math.cos(yaw + 0.5 * w * dt) * dt
            yn = y + v * math.sin(yaw + 0.5 * w * dt) * dt
            blocked = ((s.lookup(others, xn, yn) - r <= 0 or s.lookup(goal_field, xn, yn) - r <= 0) and abs(v) > 1e-6
                       or not s.grid.inside(xn, yn))
            if blocked:
                xn, yn = x, y
            d = math.hypot(xn - x, yn - y)
            path_len += d
            cur = state["cur"]
            if cur is not None:
                if cur["ticks"] == 0:
                    cur.update(v_first=float(v), w_first=float(w))
                cur["ticks"] += 1
                cur["v_sum"] += float(v)
                cur["w_sum"] += float(w)
                cur["blocked_ticks"] += int(blocked)
                cur["moved"] += d
            x, y, yaw = xn, yn, yaw_n
            t += dt
            k += 1
            if k % int(round(10.0 / dt)) == 0:
                print(f"    {planner} t={t:5.1f}s ({x:+6.2f},{y:+6.2f}) to go {rows[-1]['togo_route'] if rows else 0:5.1f} m",
                      flush=True)
    except KeyboardInterrupt:
        reason = "stopped"
        raise
    finally:
        close_step()
        n_acts = pack_acts(tmp_acts, acts_rows, os.path.join(run_dir, "acts.npz"))

    st = stack_rows(rows)
    np.savez_compressed(os.path.join(run_dir, "steps.npz"), **st)
    L = ep["oracle_length_m"]
    summary = {"planner": planner, "mode": nc.get("planner"), "reached": reached, "end": reason, "time_s": round(t, 2),
               "steps": len(rows), "path_length_m": round(path_len, 3), "oracle_length_m": round(L, 3),
               "spl": round(float(reached) * L / max(L, path_len), 4) if path_len > 0 else 0.0,
               "final_goal_edge_m": round(s.lookup(goal_field, x, y) - r, 3),
               "final_togo_route_m": round(float(st["togo_route"][-1]), 3) if len(rows) else None,
               "min_clearance_m": round(float(np.min(st["clear_solid"])), 3) if len(rows) else None,
               "blocked_s": round(float(st["blocked_ticks"].sum()) * args.control_dt, 2) if len(rows) else 0.0,
               "stop_fired_steps": int(st["stop_fired"].sum()) if len(rows) else 0,
               "act_rows": n_acts, "max_time_s": round(max_time, 1)}
    return summary, st


def log_step(yard, oracle, route, ep, pose, t, edge, res, stop_thr, planner, act_index, r, label_to_obstacle, plan_ms):
    x, y, yaw = pose
    s_arc, cross, tan = route.project(x, y)
    ahead = route.at(s_arc + 2.0)
    turn = math.degrees(math.atan2(ahead[1] - y, ahead[0] - x) - yaw)
    turn = (turn + 180.0) % 360.0 - 180.0
    cs, lab = yard.solid_at(x, y)
    ch, hk = yard.hazard_at(x, y)
    hit, hit_d = yard.ray(x, y, yaw)
    kind_s = yard.kind_of(lab)
    row = {"t": t, "pose": [x, y, yaw], "togo_route": route.length - s_arc, "togo_geo": oracle.togo_at(x, y),
           "route_s": s_arc, "cross_track": cross, "route_turn_deg": turn, "goal_edge": edge,
           "clear_solid": cs, "nearest_solid": label_to_obstacle.get(lab, -1), "nearest_solid_kind": kind_s,
           "clear_hazard": ch, "hazard_kind": hk,
           "clearance": min(cs, ch), "nearest_kind": kind_s if cs <= ch else hk,
           "facing_obstacle": label_to_obstacle.get(hit, -1), "facing_dist": hit_d,
           "ticks": 0, "v_sum": 0.0, "w_sum": 0.0, "v_first": 0.0, "w_first": 0.0, "blocked_ticks": 0, "moved": 0.0,
           "act_index": act_index, "plan_ms": plan_ms, "plan_ok": res is not None}
    c, sn = math.cos(yaw), math.sin(yaw)
    gx, gy = ep["goal_xy"][0] - x, ep["goal_xy"][1] - y
    row["goal_body"] = [c * gx + sn * gy, -sn * gx + c * gy]
    if res is not None and res.candidates is not None:
        cand = np.asarray(res.candidates, np.float32)
        vals = np.asarray(res.values, np.float32).ravel()
        traj = np.asarray(res.trajectory, np.float32)
        err = np.abs(cand - traj[None]).reshape(len(cand), -1).max(1)
        row.update(candidates=cand, values=vals, trajectory=traj, choice=int(np.argmin(err)),
                   choice_exact=bool(err.min() < 1e-4),
                   stop_fired=bool(planner == "navdp" and vals.max() < stop_thr),
                   turn_in_place=bool(np.linalg.norm(traj[-1, :2]) < 0.05),
                   planner_stuck=int(res.stuck) if getattr(res, "stuck", None) is not None else -1)
    else:
        row.update(candidates=None, values=None, trajectory=None, choice=-1, choice_exact=False, stop_fired=False,
                   turn_in_place=False, planner_stuck=-1)
    return row


def stack_rows(rows):
    if not rows:
        return {k: np.zeros(0) for k in ("t", "pose", "togo_route", "clear_solid", "stop_fired", "blocked_ticks")}
    shape = next((np.shape(q["candidates"]) for q in rows if q["candidates"] is not None), (0, 24, 3))
    out = {}
    for key in rows[0]:
        if key in ("candidates", "values", "trajectory"):
            sh = {"candidates": shape, "values": shape[:1], "trajectory": shape[1:]}[key]
            out[key] = np.stack([np.full(sh, np.nan, np.float32) if q[key] is None else q[key] for q in rows])
            if key == "candidates":
                out[key] = out[key].astype(np.float16)
        else:
            out[key] = np.array([q[key] for q in rows])
    out["pose"] = out["pose"].astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__ + "\nsteps.npz fields:" + STEP_FIELDS,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="episode i uses seed --seed + i")
    ap.add_argument("--planners", default="navdp,xnavdp", help="comma list of: " + ", ".join(PLANNERS))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "collect"))
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--spawn-modes", default=",".join(SPAWN_MODES), help="drawn uniformly from these")
    ap.add_argument("--min-len", type=float, default=4.0, help="oracle route length range (m)")
    ap.add_argument("--max-len", type=float, default=30.0)
    ap.add_argument("--max-slope", type=float, default=25.0, help="hazard map: slope above this (deg)")
    ap.add_argument("--crater-depth", type=float, default=0.3, help="hazard map: hollow deeper than this (m)")
    ap.add_argument("--act-every", type=int, default=4, help="save activations every this many plan steps (0 = off)")
    ap.add_argument("--save-frames", action="store_true", help="also save RGB (jpg) + depth (png, mm) at those steps")
    ap.add_argument("--max-time", type=float, default=0.0,
                    help="sim s per run (0 = time_factor x oracle length / max speed + 30)")
    ap.add_argument("--time-factor", type=float, default=3.0)
    ap.add_argument("--stuck-end", type=int, default=200,
                    help="end the run after this many plan steps within --stuck-dist (0 = never)")
    for k, v in EVENT_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v, help="event threshold (episode_events.py)")
    ap.add_argument("--frame-dt", type=float, default=0.2)
    ap.add_argument("--control-dt", type=float, default=0.05)
    ap.add_argument("--camera-height", type=float, default=0.5)
    ap.add_argument("--robot-radius", type=float, default=0.3)
    ap.add_argument("--reach-edge", type=float, default=0.5)
    ap.add_argument("--max-speed", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate", type=float, default=0.6)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=79.0)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides for every planner")
    args = ap.parse_args()
    planners = [p for p in args.planners.split(",") if p]
    args.stuck_end = max(args.stuck_end, args.stuck_steps) if args.stuck_end else 0   # the stuck mark comes first
    bad = [p for p in planners if p not in PLANNERS or p == "ours"]
    if bad:
        sys.exit(f"--planners: unknown or map-based {bad}")
    modes = tuple(m for m in args.spawn_modes.split(",") if m)
    if not set(modes) <= set(SPAWN_MODES):
        sys.exit(f"--spawn-modes: one of {SPAWN_MODES}")

    args.out = os.path.abspath(args.out)        # the servers write the activation files: no relative paths
    os.makedirs(args.out, exist_ok=True)
    stage, rocks = prepare_assets(args.glb, os.path.join(HERE, "runs", "assets"))
    yard = Yard(stage, rocks, args.max_slope, args.crater_depth)
    sim, agent = make_sim(stage_config(stage, os.path.join(HERE, "runs", "assets")), args.width, args.height, args.hfov)
    boulders = Boulders(sim, rocks)
    K = intrinsics(args.width, args.height, args.hfov)
    navs = {p: TubeNavigator(nav_config(p, args)) for p in planners}
    event_kw = {k: getattr(args, k) for k in EVENT_DEFAULTS}
    index = os.path.join(args.out, "index.jsonl")
    started = time.time()
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            ep_dir = os.path.join(args.out, f"ep_{seed:06d}")
            if os.path.exists(os.path.join(ep_dir, "done")):
                print(f"[skip] seed {seed}: done")
                continue
            ep, oracle = make_episode(yard, seed, args.robot_radius, args.reach_edge, modes, args.min_len, args.max_len)
            if ep is None:
                print(f"[skip] seed {seed}: no valid episode")
                continue
            boulders.move({b: yard.boulder_pose(b, *p)[0] for b, p in yard.placements.items()}, yard.placements)
            yard.refresh(exclude=(ep["goal"],))
            route = Route(ep["route"])
            ep["obstacles"] = list(yard.scene.names)
            ep["obstacle_kinds"] = [KINDS[yard.kind_of(j)] for j in range(len(yard.scene.names))]
            ep["obstacle_xy"] = {n: list(yard.scene.origin[n]) for n in yard.scene.names}
            ep["kinds"] = list(KINDS)
            ep["settings"] = {k: getattr(args, k) for k in ("robot_radius", "reach_edge", "max_speed", "max_yaw_rate",
                                                          "frame_dt", "control_dt", "camera_height", "width",
                                                          "height", "hfov", "max_slope", "crater_depth", "act_every")}
            ep["oracle_start_togo_geo_m"] = oracle.togo_at(*ep["start"])
            shutil.rmtree(ep_dir, ignore_errors=True)
            os.makedirs(ep_dir)
            save_json(os.path.join(ep_dir, "episode.json"), ep)
            print(f"[ep {seed}] {ep['spawn']} -> {ep['goal']}  oracle {ep['oracle_length_m']:.1f} m", flush=True)
            trails, marks = [], []
            for p in planners:
                run_dir = os.path.join(ep_dir, p)
                summary, st = run_planner(p, navs[p], sim, agent, K, yard, ep, oracle, route, run_dir, args)
                events = detect(st, ep, **event_kw) if len(st["t"]) else []
                summary["events"] = {kind: sum(e["kind"] == kind for e in events)
                                     for kind in ("stuck", "off_route", "collision", "brush", "stopped_short",
                                                  "wrong_way")}
                save_json(os.path.join(run_dir, "events.json"), {**summary, "event_params": event_kw,
                                                                 "events": events})
                with open(index, "a") as f:
                    f.write(json.dumps({"seed": seed, "goal": ep["goal"], "spawn": ep["spawn"], **summary}) + "\n")
                trails.append((p, st["pose"][:, :2] if len(st["t"]) else np.zeros((0, 2)), COLORS.get(p, (255, 255, 0))))
                marks += [(e["x"], e["y"], f"{p[0]}:{e['kind']}", COLORS.get(p, (255, 255, 0))) for e in events
                          if e["kind"] in ("stuck", "collision", "stopped_short")]
                print(f"  {p}: {summary['end']} t={summary['time_s']:.0f}s path {summary['path_length_m']:.1f} m "
                      f"SPL {summary['spl']:.2f}  events {summary['events']}", flush=True)
            cv2.imwrite(os.path.join(ep_dir, "top.png"), draw_episode(yard, ep, trails, marks))
            open(os.path.join(ep_dir, "done"), "w").close()
            print(f"  ({(time.time() - started) / 60:.1f} min so far)", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        for nav in navs.values():
            nav.close()
        sim.close()


if __name__ == "__main__":
    main()
