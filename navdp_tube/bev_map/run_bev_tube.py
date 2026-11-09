#!/usr/bin/env python3
"""BEV map + tube-guided NavDP on the REAL rover: click a goal, the rover drives there around mapped obstacles.

    # GPU server, terminal 1 (conda env navdp): the tube NavDP server
    cd navdp_tube/tube_navdp && python scripts/tube_navdp_server.py --navdp-root ../NavDP \
        --checkpoint ../NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt --port 8890
    # rover: camera stream, rover_state_bridge.py, navdp_cmdvel_bridge.py (as for run_bev_real.py --drive)
    # GPU server, terminal 2 (conda env yolo)
    cd navdp_tube/bev_map && python run_bev_tube.py                 # browser at http://localhost:8090
    python run_bev_tube.py --goal 1.5,0.3                   # start with a goal (odom x, y)
    python run_bev_tube.py --dry-run                        # plan and show, but never send cmd_vel
    python run_bev_tube.py --replay runs/real_001           # plan on a recording (no rover)

Goal: click the object in the camera view (its 3D point from the corrected depth) or a spot on the global map,
or send JSON to nav.goal_bind (ZMQ PUB -> this SUB): {"x": .., "y": ..} in odom, or {"forward": .., "left": ..}
relative to the rover now. Keys: X / SPACE cancel the goal and stop, arrows / WASD drive by hand (cancels the
goal), R clear map, P save, Q quit.

What is sent to the planner (config.yaml nav:): the map's occupied cells within nav.obstacle_range, WITHOUT the
goal's surroundings (nav.goal_clear_radius) and without the robot's size (the driver adds nav.robot_radius). The
rover stops nav.stop_distance from the goal point, when the plan is older than nav.plan_timeout, when telemetry
is older than real.telemetry_timeout, and drives no further forward when a map obstacle is right ahead.
"""

import argparse
import csv
import json
import os
import threading
import time

import cv2
import numpy as np

from bev_pipeline import BEVPipeline
from bev_render import depth_vis, draw_robot, label, render_global, render_local
from real_source import Frame, LiveSource, Recorder, ReplaySource, load_calibration
from rover_kinematics import RoverCamera, camera_pitch_deg
from run_bev_real import HEADLESS, CmdVel, fit, motion_gate, pad_to, real_config
from classic_nav import make_navigator
from tube_nav import draw_nav, nav_map, status_text


def pixel_to_world(u, v, depth, K, cam_pos, cam_rot, win=3):
    """World point at pixel (u, v) from the median valid depth around it, or None."""
    h, w = depth.shape
    u, v = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    patch = depth[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    d = float(np.median(patch))
    ray = np.array([(u + 0.5 - K[0, 2]) / K[0, 0], -(v + 0.5 - K[1, 2]) / K[1, 1], -1.0]) @ cam_rot.T
    return cam_pos + ray * d


def set_goal_safe(nav, goal, K):
    """Set a goal; a server problem is printed (and shown) instead of stopping the program."""
    try:
        nav.set_goal(goal, K)
        return True
    except RuntimeError as e:
        nav.cancel()
        nav.error = str(e)[:120]
        print(f"[error] {e}")
        return False


class GoalInbox:
    """Goals from other programs (e.g. the gaze / VLM pipeline) over ZMQ."""

    def __init__(self, bind):
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.bind(bind)

    def poll(self):
        out = []
        while True:
            try:
                out.append(json.loads(self.sock.recv_string(self.zmq.NOBLOCK)))
            except self.zmq.Again:
                return out
            except ValueError:
                continue

    def close(self):
        self.sock.close(0)
        self.ctx.term()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", default="", help="initial goal x,y in the odom frame")
    ap.add_argument("--dry-run", action="store_true", help="plan and draw, never send cmd_vel")
    ap.add_argument("--record", default="")
    ap.add_argument("--replay", default="")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. nav.max_speed=0.08")
    args = ap.parse_args()

    cfg = real_config(args.set)
    rc, rcfg, nc = cfg["real"], cfg["render"], cfg["nav"]
    calib = load_calibration(rc)
    cam_model = RoverCamera(rc, calib)
    src = ReplaySource(cfg, args.replay) if args.replay else LiveSource(cfg, calib, telemetry=True)
    if not args.replay and not src.undistort.calibrated:
        print("[warn] no camera calibration: depth / goal positions will be off. Run calibrate_camera.py.")
    pipe = BEVPipeline(cfg)
    nav = make_navigator(nc)
    live = not args.replay
    cmd = CmdVel(rc["cmdvel_bind"], rc["drive"]["linear"], rc["drive"]["angular"], rc["drive"]["deadman"]) \
        if live and not args.dry_run else None
    inbox = GoalInbox(nc["goal_bind"]) if live and nc.get("goal_bind") else None

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                       time.strftime("tube_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "nav_log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["t", "x", "y", "yaw", "mapped", "goal_x", "goal_y", "state", "choice", "v", "w", "n_cells",
                  "n_tubes", "n_valid", "plan_ms"])

    from web_view import Display
    disp = Display("BEV tube", web=args.web or HEADLESS, host=args.host, port=args.port)
    if disp.url:
        print("=" * 60 + f"\n  Open {disp.url} in your browser\n" + "=" * 60)

    # ---------------------------------------------------------------- control thread (live only)
    running = True
    pending_goal = [tuple(float(v) for v in args.goal.split(","))] if args.goal else []

    def latest_pose():
        s = src.telemetry.at(time.monotonic()) if getattr(src, "telemetry", None) else None
        if s is None or s.get("x") is None or s.get("_age", 1e9) > rc.get("telemetry_timeout", 0.5):
            return None
        return (float(s["x"]), float(s["y"]), float(s["yaw"]))

    def control():
        period = 1.0 / nc.get("control_hz", 20.0)
        while running:
            if nav.goal is not None:
                pose = latest_pose()
                if pose is None:
                    nav.status = {"state": "no telemetry"}
                    v = w = 0.0
                else:
                    v, w, _ = nav.command(pose, time.monotonic())
                if cmd is not None:
                    cmd.set(v, w)
            time.sleep(period)

    ctl = threading.Thread(target=control, daemon=True) if live else None
    if ctl is not None:
        ctl.start()

    def stop():
        nav.cancel()
        if cmd is not None:
            cmd.set(0.0, 0.0)

    trail, gimg, dash, fps, layout = [], None, None, 0.0, {}
    last = None          # (frame, r) shown on the dashboard, for clicks
    try:
        while True:
            t0 = time.perf_counter()
            item = src.next()
            if item is None:
                if args.replay:
                    print("[replay] finished")
                    break
                print("[camera] no frame for 5 s")
                continue
            rgb, K, t, s = item
            if args.record and live and not hasattr(src, "_rec"):
                src._rec = Recorder(args.record, K, src.telemetry)
            if hasattr(src, "_rec"):
                src._rec.add(rgb, t)

            reason = ""
            if s is None or s.get("x") is None or s.get("_age", 0) > rc.get("telemetry_timeout", 0.5):
                reason, s = "no_telemetry", (s or {})
            else:
                reason = motion_gate(s, rc)
            cam_pos, cam_rot = cam_model.pose(s)
            frame = Frame(rgb=rgb, K=K, cam_pos=cam_pos, cam_rot=cam_rot,
                          robot_xy=np.array([s.get("x") or 0.0, s.get("y") or 0.0]),
                          robot_yaw=float(s.get("yaw") or 0.0), time=t)
            r = pipe.process(frame, fuse=(reason == ""))
            trail.append(frame.robot_xy.copy())
            last = (frame, r)
            state = pipe.grid.state()

            # ---- goals
            for g in pending_goal:
                set_goal_safe(nav, g, K)
                print(f"[goal] {g[0]:+.2f}, {g[1]:+.2f}")
            pending_goal = []
            for msg in (inbox.poll() if inbox else []):
                pose = latest_pose()
                if "x" in msg and "y" in msg:
                    g = (float(msg["x"]), float(msg["y"]))
                elif "forward" in msg and pose is not None:
                    c, sn = np.cos(pose[2]), np.sin(pose[2])
                    g = (pose[0] + c * msg["forward"] - sn * msg.get("left", 0.0),
                         pose[1] + sn * msg["forward"] + c * msg.get("left", 0.0))
                else:
                    continue
                set_goal_safe(nav, g, K)
                print(f"[goal from {nc['goal_bind']}] {g[0]:+.2f}, {g[1]:+.2f}")

            # ---- plan (background; pose = robot pose when the image was taken)
            world_map = nav_map(pipe.grid, state, frame.robot_xy, nav.goal, nc)
            nav.map = world_map                       # newest map for the safety check and the drawing
            if nav.goal is not None and reason != "no_telemetry":
                pose_img = (float(frame.robot_xy[0]), float(frame.robot_xy[1]), frame.robot_yaw)
                nav.plan_async(rgb, r["depth"], pose_img, t - getattr(src, "latency", 0.0), world_map)
            if not live and nav.goal is not None:      # replay: no control loop, only the planner's view
                nav.status = {"state": "replay (no control)"}

            # ---- keys / clicks
            for k in disp.keys():
                if k == "q":
                    raise KeyboardInterrupt
                if k in ("x", "stop"):
                    stop()
                    print("[goal] cancelled")
                elif k == "r":
                    pipe.reset(); trail.clear()
                elif k == "p" and gimg is not None:
                    stem = os.path.join(out_dir, f"map_{time.strftime('%H%M%S')}")
                    pipe.grid.save(stem)
                    cv2.imwrite(stem + "_dashboard.png", dash)
                elif k in ("up", "down", "left", "right"):
                    nav.cancel()
                    if cmd is not None:
                        cmd.key(k)
                elif k.startswith("click:") and dash is not None:
                    fx, fy = (float(v) for v in k[6:].split(","))
                    goal = click_to_goal(fx * dash.shape[1], fy * dash.shape[0], layout, last, pipe.grid,
                                         rcfg["scale"])
                    if goal is None:
                        print("[goal] no valid depth / not on the camera or map view")
                    else:
                        set_goal_safe(nav, goal, K)
                        print(f"[goal] clicked {goal[0]:+.2f}, {goal[1]:+.2f}")

            # ---- draw
            gimg = render_global(pipe.grid, state, rcfg["scale"], frame.robot_xy, frame.robot_yaw,
                                 nc["robot_radius"], trail=trail)
            draw_nav(gimg, pipe.grid, rcfg["scale"], nav)
            fov_h = 2 * np.arctan(rgb.shape[1] / 2 / K[0, 0])
            cam_fwd = -cam_rot[:, 2]
            draw_robot(gimg, pipe.grid, rcfg["scale"], frame.robot_xy, float(np.arctan2(cam_fwd[1], cam_fwd[0])),
                       0.0, fov_h, cfg["grid"].get("obstacle_max_range", cfg["depth"]["max_range"]))
            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)
            cam_img = fit(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 560)
            dimg = fit(depth_vis(r["depth"], cfg["depth"]["max_range"]), 560)
            label(cam_img, f"camera  h={cam_pos[2] * 100:.1f}cm  pitch={camera_pitch_deg(cam_rot):+.1f}deg  "
                           "(click = goal)")
            label(dimg, "MAPPED" if r["fused"] else f"SKIPPED: {reason or 'fit'}", (8, 22),
                  (120, 255, 120) if r["fused"] else (80, 80, 255))
            left = np.vstack([cam_img, dimg])
            h = left.shape[0]
            gview = fit(gimg, 760, h)
            label(gview, f"{fps:.1f} FPS  {status_text(nav.status)}"[:100])
            if nav.error:
                label(gview, nav.error[:90], (8, 46), (80, 80, 255))
            lview = fit(render_local(gimg, pipe.grid, rcfg["scale"], frame.robot_xy, frame.robot_yaw,
                                     rcfg["local_size"]), 380, h)
            dash = np.hstack([left, pad_to(gview, gview.shape[1], h), pad_to(lview, lview.shape[1], h)])
            layout = {"cam": (0, 0, cam_img.shape[1], cam_img.shape[0], rgb.shape[1] / cam_img.shape[1]),
                      "map": (left.shape[1], 0, gview.shape[1], gview.shape[0], gimg.shape[1] / gview.shape[1])}
            disp.show(dash, status_text(nav.status))

            st = nav.status
            log.writerow([f"{t:.3f}", f"{frame.robot_xy[0]:.3f}", f"{frame.robot_xy[1]:.3f}",
                          f"{frame.robot_yaw:.3f}", int(r["fused"]),
                          *(f"{v:.3f}" for v in (nav.goal or (np.nan, np.nan))), st.get("state", ""),
                          st.get("choice", ""), f"{st.get('v', 0):.3f}", f"{st.get('w', 0):.3f}",
                          st.get("n_cells", int(world_map.occupied.sum())), st.get("n_tubes", ""), st.get("n_valid", ""),
                          f"{st.get('plan_ms', 0):.0f}"])
    except KeyboardInterrupt:
        print("\n[quit]")
    finally:
        running = False
        stop()
        if ctl is not None:
            ctl.join(timeout=1.0)
        if cmd is not None:
            cmd.close()
        nav.close()
        if inbox is not None:
            inbox.close()
        log_f.close()
        if dash is not None:
            cv2.imwrite(os.path.join(out_dir, "last_dashboard.png"), dash)
        if hasattr(src, "_rec"):
            src._rec.close()
        src.close()
        disp.close()
        print(f"log: {os.path.join(out_dir, 'nav_log.csv')}")


def click_to_goal(x, y, layout, last, grid, scale):
    """Dashboard pixel -> odom goal: camera panel = 3D point of the clicked pixel, map panel = that map spot."""
    if "cam" in layout and last is not None:
        x0, y0, w, h, k = layout["cam"]
        if x0 <= x < x0 + w and y0 <= y < y0 + h:
            frame, r = last
            p = pixel_to_world((x - x0) * k, (y - y0) * k, r["depth"], frame.K, frame.cam_pos, frame.cam_rot)
            return None if p is None else (float(p[0]), float(p[1]))
    if "map" in layout:
        x0, y0, w, h, k = layout["map"]
        if x0 <= x < x0 + w and y0 <= y < y0 + h:
            px, py = (x - x0) * k, (y - y0) * k
            return (grid.x_min + px / scale * grid.res, grid.y_max - py / scale * grid.res)
    return None


if __name__ == "__main__":
    main()
