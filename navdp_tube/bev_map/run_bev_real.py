#!/usr/bin/env python3
"""Live BEV occupancy map on the REAL rover (same pipeline as run_bev_sim.py).

    cd navdp_tube/bev_map && conda activate yolo
    python run_bev_real.py                       # map; drive the rover with something else
    python run_bev_real.py --drive               # also drive with the keyboard (publishes cmd_vel)
    python run_bev_real.py --record runs/real_a  # save the session for offline replay
    python run_bev_real.py --replay runs/real_a  # replay a session (no rover), e.g. to tune settings
    python run_bev_real.py --no-telemetry        # no odom/IMU: per-frame map around the robot only
    python run_bev_real.py --set real.camera_latency=0.15 depth.scale_gate_rel=0.15

On the rover first:  ./start_camera_stream.sh <SERVER_IP>   and
                     python3 rover_state_bridge.py --server <SERVER_IP>   (in the ROS container)
For --drive also the existing cmd_vel bridge (navdp_cmdvel_bridge.py). See readme.md "Real rover".

Keys (window or browser page): arrows/WASD drive (hold), SPACE stop, R clear map, P save, Q quit.

How each frame is corrected (details in readme.md):
  pose    odometry x, y, yaw + IMU roll, pitch + gimbal pan/tilt -> camera pose (rover_kinematics.py)
          calibrated mount (height, tilt) from calibrate_camera.py
  depth   Depth Anything, then per-frame floor fit: scale + offset + roll (projection.JointFloorFit)
  gating  frames are NOT added to the map when the floor fit fails / jumps, telemetry is missing,
          or the rover turns or bounces too fast (pose and timing errors are largest then)
"""

import argparse
import csv
import os
import threading
import time

import cv2
import numpy as np

from bev_pipeline import BEVPipeline, load_config
from bev_render import depth_vis, draw_robot, label, render_global, render_local
from real_source import Frame, LiveSource, Recorder, ReplaySource, load_calibration
from rover_kinematics import RoverCamera, camera_pitch_deg

HEADLESS = not os.environ.get("DISPLAY")


class CmdVel:
    """Publishes cmd_vel ({"linear", "angular"}) like the NavDP scripts, with a dead-man stop:
    without a fresh drive key for `deadman` seconds the rover is told to stop."""

    def __init__(self, bind, linear, angular, deadman):
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(bind)
        self.v, self.w = linear, angular
        self.deadman = deadman
        self.cmd, self.t_cmd = (0.0, 0.0), 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def key(self, k):
        m = {"up": (self.v, 0.0), "down": (-self.v, 0.0), "left": (0.0, self.w), "right": (0.0, -self.w),
             "stop": (0.0, 0.0)}
        if k in m:
            with self.lock:
                self.cmd, self.t_cmd = m[k], time.monotonic()

    def set(self, linear, angular):
        """Velocity from a controller (e.g. navigation); also subject to the dead-man timeout."""
        with self.lock:
            self.cmd, self.t_cmd = (float(linear), float(angular)), time.monotonic()

    def _loop(self):
        while self.running:
            with self.lock:
                cmd = self.cmd if time.monotonic() - self.t_cmd < self.deadman else (0.0, 0.0)
            try:
                self.sock.send_json({"linear": float(cmd[0]), "angular": float(cmd[1])}, self.zmq.NOBLOCK)
            except self.zmq.Again:
                pass
            time.sleep(0.1)

    def close(self):
        with self.lock:
            self.cmd = (0.0, 0.0)
        time.sleep(0.3)
        self.running = False
        self.thread.join(timeout=1.0)
        self.sock.close(0)
        self.ctx.term()


def real_config(overrides):
    """config.yaml with the real-rover settings applied."""
    cfg = load_config(overrides=overrides)
    rc = cfg["real"]
    if cfg["depth"]["source"] not in ("depth_anything", "sam2"):   # sam2: --set depth.source=sam2
        cfg["depth"]["source"] = "depth_anything"
    cfg["classify"]["pitch_uncertainty_deg"] = rc.get("pitch_uncertainty_deg", 0.5)
    cfg["grid"].update(rc.get("grid", {}))
    return cfg


def motion_gate(s, rc):
    """Reason not to map this frame because of motion, or ''."""
    wz = s.get("gz") if s.get("gz") is not None else s.get("wz")
    if wz is not None and abs(wz) > rc.get("max_yaw_rate", 0.6):
        return "turning"
    gx, gy = s.get("gx"), s.get("gy")
    if gx is not None and gy is not None and np.hypot(gx, gy) > rc.get("max_tilt_rate", 0.35):
        return "bouncing"
    return ""


def fit(img, w=None, h=None):
    H, W = img.shape[:2]
    s = min((w or 1e9) / W, (h or 1e9) / H)
    return cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)


def pad_to(img, w, h, color=(40, 40, 40)):
    out = np.full((h, w, 3), color, np.uint8)
    out[:min(h, img.shape[0]), :min(w, img.shape[1])] = img[:h, :w]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", action="store_true", help="drive with the keyboard (publishes cmd_vel)")
    ap.add_argument("--record", default="", help="save the session to this folder")
    ap.add_argument("--replay", default="", help="replay a recorded session instead of the live rover")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="no odom/IMU: rebuild the map every frame around the robot (no fusion over time)")
    ap.add_argument("--web", action="store_true", help="show in the browser (default without DISPLAY)")
    ap.add_argument("--no-gui", action="store_true", help="no display at all (e.g. batch replay)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="", help="where maps/logs go (default runs/real_<time>)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. real.camera_latency=0.15")
    args = ap.parse_args()

    cfg = real_config(args.set)
    rc, rcfg = cfg["real"], cfg["render"]
    calib = load_calibration(rc)
    cam_model = RoverCamera(rc, calib)
    no_tel = args.no_telemetry

    if args.replay:
        src = ReplaySource(cfg, args.replay)
        no_tel = no_tel or not src.tel
    else:
        src = LiveSource(cfg, calib, telemetry=not no_tel)
        if not src.undistort.calibrated:
            print(f"[warn] no camera calibration - using a {rc['fallback_hfov_deg']} deg FOV guess. "
                  "Heights will be off. Run calibrate_camera.py.")
    if not cam_model.calibrated:
        print("[warn] no mount calibration - camera height/tilt from the CAD model only. "
              "Run calibrate_camera.py mount.")
    if no_tel:
        print("[info] no telemetry: robot assumed at the origin, map rebuilt every frame.")

    pipe = BEVPipeline(cfg)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                       time.strftime("real_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    rec = None
    show = not args.no_gui
    disp = None
    if show:
        from web_view import Display
        disp = Display("BEV real", web=args.web or HEADLESS, host=args.host, port=args.port)
        if disp.url:
            print("=" * 60 + f"\n  Open {disp.url} in your browser\n" + "=" * 60)
    drive = CmdVel(rc["cmdvel_bind"], rc["drive"]["linear"], rc["drive"]["angular"],
                   rc["drive"]["deadman"]) if args.drive and not args.replay else None

    log_f = open(os.path.join(out_dir, "frames_log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["t", "fused", "skip_reason", "a", "b", "roll_fix_deg", "floor_inliers", "imu_roll_deg",
                  "imu_pitch_deg", "cam_height", "cam_pitch_deg", "pan_deg", "tilt_deg", "yaw_rate",
                  "x", "y", "yaw", "tel_age"])

    trail, gimg, dash, fps = [], None, None, 0.0
    counts = {"frames": 0, "fused": 0}
    skip_counts = {}

    def save():
        stem = os.path.join(out_dir, f"map_{time.strftime('%H%M%S')}")
        pipe.grid.save(stem)
        if gimg is not None:
            cv2.imwrite(stem + "_bev.png", gimg)
        if dash is not None:
            cv2.imwrite(stem + "_dashboard.png", dash)
        print(f"[saved] {stem}.npy/.pgm/.yaml/_bev.png")

    try:
        while True:
            if args.max_frames and counts["frames"] >= args.max_frames:
                break
            t0 = time.perf_counter()
            item = src.next()
            if item is None:
                if args.replay:
                    print("[replay] finished")
                    break
                print("[camera] no frame for 5 s - is start_camera_stream.sh running?")
                continue
            rgb, K, t, s = item
            counts["frames"] += 1
            if rec is None and args.record and not args.replay:
                rec = Recorder(args.record, K, getattr(src, "telemetry", None))
            if rec is not None:
                rec.add(rgb, t)

            # ---- pose for this frame
            reason = ""
            if no_tel:
                s = {}
                pipe.grid.reset()                          # no odometry -> no fusion over time
            elif s is None or s.get("_age", 0) > rc.get("telemetry_timeout", 0.5) or s.get("x") is None:
                reason = "no_telemetry"
                s = s or {}
            else:
                reason = motion_gate(s, rc)
            cam_pos, cam_rot = cam_model.pose(s)
            frame = Frame(rgb=rgb, K=K, cam_pos=cam_pos, cam_rot=cam_rot,
                          robot_xy=np.array([s.get("x") or 0.0, s.get("y") or 0.0]),
                          robot_yaw=float(s.get("yaw") or 0.0), time=t)

            r = pipe.process(frame, fuse=(reason == ""))
            if not r["fit_ok"] and not reason:
                reason = "fit:" + (pipe.joint.last.get("reason", "") if pipe.joint else "failed")
            if r["fused"]:
                counts["fused"] += 1
            else:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
            trail.append(frame.robot_xy.copy())

            jl = pipe.joint.last if pipe.joint is not None else {}
            roll_i, pitch_i = cam_model.base_tilt(s)
            pan_j, tilt_j = cam_model.joint_angles(s)
            log.writerow([f"{t:.3f}", int(r["fused"]), reason, f"{r['scale'] or 0:.4f}", f"{r['offset'] or 0:.4f}",
                          f"{jl.get('roll_deg', 0):.3f}", f"{r['floor_inliers']:.3f}",
                          f"{np.degrees(roll_i):.3f}", f"{np.degrees(pitch_i):.3f}", f"{cam_pos[2]:.4f}",
                          f"{camera_pitch_deg(cam_rot):.2f}", f"{np.degrees(pan_j):.1f}", f"{np.degrees(tilt_j):.1f}",
                          f"{s.get('gz') if s.get('gz') is not None else (s.get('wz') or 0):.3f}",
                          f"{frame.robot_xy[0]:.3f}", f"{frame.robot_xy[1]:.3f}", f"{frame.robot_yaw:.3f}",
                          f"{s.get('_age', -1):.3f}"])

            # ---- keys
            if disp is not None:
                for k in disp.keys():
                    if k == "q":
                        raise KeyboardInterrupt
                    if k == "r":
                        pipe.reset(); trail.clear(); print("[map] reset")
                    elif k == "p":
                        save()
                    elif drive is not None:
                        drive.key(k)

            # ---- draw
            state = pipe.grid.state()
            fov_h = 2 * np.arctan(rgb.shape[1] / 2 / K[0, 0])
            cam_fwd = -cam_rot[:, 2]
            cam_yaw = float(np.arctan2(cam_fwd[1], cam_fwd[0]))
            gimg = render_global(pipe.grid, state, rcfg["scale"], frame.robot_xy, frame.robot_yaw,
                                 rcfg["robot_radius"], inflated=pipe.grid.inflate(state, rcfg["robot_radius"]),
                                 trail=trail)
            draw_robot(gimg, pipe.grid, rcfg["scale"], frame.robot_xy, cam_yaw, 0.0, fov_h,
                       cfg["grid"].get("obstacle_max_range", cfg["depth"]["max_range"]))
            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)
            if disp is not None:
                limg = render_local(gimg, pipe.grid, rcfg["scale"], frame.robot_xy, frame.robot_yaw,
                                    rcfg["local_size"])
                cam_img = fit(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 560)
                dimg = fit(depth_vis(r["depth"], cfg["depth"]["max_range"]), 560)
                label(cam_img, f"camera  h={cam_pos[2] * 100:.1f}cm  pitch={camera_pitch_deg(cam_rot):+.1f}deg")
                fit_txt = (f"a={r['scale']:.2f} b={r['offset']:+.2f} roll fix {jl.get('roll_deg', 0):+.2f}deg"
                           if r["scale"] is not None else "no fit yet")
                label(dimg, fit_txt)
                col = (120, 255, 120) if r["fused"] else (80, 80, 255)
                label(dimg, "MAPPED" if r["fused"] else f"SKIPPED: {reason}", (8, 46), col)
                left = np.vstack([cam_img, dimg])
                h = left.shape[0]
                gview = fit(gimg, 760, h)
                label(gview, f"global map (odom)  {fps:.1f} FPS  mapped {counts['fused']}/{counts['frames']}")
                lview = fit(limg, 380, h)
                label(lview, "robot view")
                dash = np.hstack([left, pad_to(gview, gview.shape[1], h), pad_to(lview, lview.shape[1], h)])
                disp.show(dash, f"{fps:.1f} FPS  {'MAPPED' if r['fused'] else 'skipped: ' + reason}")
    except KeyboardInterrupt:
        print("\n[quit]")
    finally:
        if drive is not None:
            drive.close()
        try:
            if gimg is not None:
                save()
        except Exception as e:  # noqa: BLE001
            print(f"[save failed] {e}")
        log_f.close()
        if rec is not None:
            rec.close()
            print(f"[record] {rec.n} frames -> {args.record}")
        src.close()
        if disp is not None:
            disp.close()
        print(f"frames {counts['frames']}, mapped {counts['fused']}, skipped {skip_counts}")
        print(f"log: {os.path.join(out_dir, 'frames_log.csv')}")


if __name__ == "__main__":
    main()
