#!/usr/bin/env python3
"""Live BEV occupancy map in the MuJoCo simulation, built from Depth Anything depth.

    cd navdp_tube/bev_map && conda activate yolo
    python run_bev_sim.py                      # drive with the keyboard (focus the "BEV" window)
    python run_bev_sim.py --auto               # scripted drive through the arena, hands-free
    python run_bev_sim.py --depth sim          # MuJoCo ground-truth depth (reference map)
    python run_bev_sim.py --viewer             # also open the MuJoCo 3D viewer
    python run_bev_sim.py --set grid.resolution=0.05 depth.max_range=3.0
    python run_bev_sim.py --web                # in the browser (automatic when there is no display,
                                               # e.g. over SSH): open http://localhost:8090

Keys (in the BEV window, the browser page, or the MuJoCo viewer with --viewer):
    arrows / W A S D   drive         SPACE  stop
    J / L              pan camera    I / K  tilt camera
    G                  toggle ground-truth outline       R  reset map
    P                  save map + screenshot             Q / ESC  quit

Each loop advances the simulation by --dt seconds and then maps one frame, so the map stays
consistent with the robot motion no matter how long Depth Anything takes per frame.
Saved maps go to runs/live_<time>/ as .png, .npy (ROS int8) and .pgm/.yaml (ROS map_server).
"""

import argparse
import os
import sys
import time

# No display (e.g. SSH) -> render offscreen with EGL and show the dashboard in the browser.
HEADLESS = "--no-gui" in sys.argv or "--web" in sys.argv or not os.environ.get("DISPLAY")
if HEADLESS:                         # must be set before mujoco is imported
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SIM_DIR))   # core pipeline files live in bev_map/
SIM_CONFIG = os.path.join(SIM_DIR, "config_sim.yaml")

from bev_pipeline import BEVPipeline, load_config
from bev_render import depth_vis, draw_robot, label, render_global, render_local
from sim_source import SimSource

WHEEL = 5.0          # wheel actuator limit
TURN = 3.0

# cv2.waitKeyEx codes for arrow keys: GTK/X11, Windows, Qt backend
K_LEFT = (65361, 2424832, 16777234)
K_UP = (65362, 2490368, 16777235)
K_RIGHT = (65363, 2555904, 16777236)
K_DOWN = (65364, 2621440, 16777237)


def fit(img, w=None, h=None):
    H, W = img.shape[:2]
    s = min((w or 1e9) / W, (h or 1e9) / H)
    return cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)


def pad_to(img, w, h, color=(40, 40, 40)):
    out = np.full((h, w, 3), color, np.uint8)
    out[:img.shape[0], :img.shape[1]] = img[:h, :w]
    return out


class Teleop:
    def __init__(self, src):
        self.src = src
        self.flags = {"quit": False, "reset": False, "save": False, "gt": False}

    def key(self, k):
        s = self.src
        if k < 0:
            return
        c = chr(k & 0xFF).lower() if k < 256 else ""
        if k in K_UP or c == "w":
            s.set_wheels(WHEEL, WHEEL)
        elif k in K_DOWN or c == "s":
            s.set_wheels(-WHEEL, -WHEEL)
        elif k in K_LEFT or c == "a":
            s.set_wheels(-TURN, TURN)
        elif k in K_RIGHT or c == "d":
            s.set_wheels(TURN, -TURN)
        elif c == " ":
            s.set_wheels(0, 0)
        elif c == "j":
            s.pan_cmd = float(np.clip(s.pan_cmd + 0.15, -3.14, 3.14))
        elif c == "l":
            s.pan_cmd = float(np.clip(s.pan_cmd - 0.15, -3.14, 3.14))
        elif c == "i":
            s.tilt_cmd = float(np.clip(s.tilt_cmd - 0.1, -0.52, 1.57))
        elif c == "k":
            s.tilt_cmd = float(np.clip(s.tilt_cmd + 0.1, -0.52, 1.57))
        elif c == "r":
            self.flags["reset"] = True
        elif c == "p":
            self.flags["save"] = True
        elif c == "g":
            self.flags["gt"] = not self.flags["gt"]
        elif c == "q" or k == 27:
            self.flags["quit"] = True

    def web_key(self, k):
        """Keys from the browser page: up/down/left/right/stop or a single letter."""
        m = {"up": K_UP[0], "down": K_DOWN[0], "left": K_LEFT[0], "right": K_RIGHT[0], "stop": ord(" ")}
        if k in m:
            self.key(m[k])
        elif len(k) == 1:
            self.key(ord(k.lower()))

    def glfw_key(self, keycode):
        """MuJoCo viewer key callback (GLFW key codes)."""
        from mujoco.glfw import glfw
        m = {glfw.KEY_UP: K_UP[0], glfw.KEY_DOWN: K_DOWN[0], glfw.KEY_LEFT: K_LEFT[0],
             glfw.KEY_RIGHT: K_RIGHT[0], glfw.KEY_SPACE: ord(" ")}
        if keycode in m:
            self.key(m[keycode])
        elif 32 < keycode < 127:
            self.key(ord(chr(keycode).lower()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", choices=["depth_anything", "sam2", "sim"], default=None,
                    help="depth source (default from config.yaml)")
    ap.add_argument("--auto", action="store_true", help="drive the scripted route from record_sim.py")
    ap.add_argument("--viewer", action="store_true", help="also open the MuJoCo 3D viewer")
    ap.add_argument("--dt", type=float, default=0.1, help="sim seconds between mapped frames")
    ap.add_argument("--no-gui", action="store_true", help="headless (use with --auto); saves at the end")
    ap.add_argument("--web", action="store_true",
                    help="show the dashboard in the browser (default when there is no DISPLAY)")
    ap.add_argument("--port", type=int, default=8090, help="port for --web")
    ap.add_argument("--host", default="127.0.0.1", help="interface for --web (0.0.0.0 = whole network)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many mapped frames (0 = no limit)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. grid.resolution=0.05")
    args = ap.parse_args()

    web_mode = not args.no_gui and (args.web or HEADLESS)
    if web_mode and args.viewer:
        print("[info] no display: the MuJoCo viewer is replaced by a chase-camera panel in the browser")
        args.viewer = False

    cfg = load_config(overrides=args.set, extra=[SIM_CONFIG])
    if args.depth:
        cfg["depth"]["source"] = args.depth
    rc = cfg["render"]

    src = SimSource(cfg)
    pipe = BEVPipeline(cfg)
    tele = Teleop(src)
    src.settle(0.5)

    gt = None
    gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                           f"gt_occ_{cfg['grid']['resolution']}_{cfg['grid']['x_min']}_{cfg['grid']['y_min']}.npy")
    def ground_truth():
        nonlocal gt
        if gt is None:
            if os.path.exists(gt_path):
                gt = np.load(gt_path)
            else:
                from ground_truth import ground_truth_grid
                print("[GT] computing ground-truth outline (one-off) ...")
                gt = ground_truth_grid(src.model, pipe.grid, cfg["sim"]["rover_body"],
                                       cfg["classify"]["obstacle_min_z"])
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                np.save(gt_path, gt)
        return gt

    route = None
    if args.auto:
        from record_sim import ROUTE, controller
        route = list(ROUTE)
        step_kind, step_target, step_end = None, None, 0.0

    web = None
    if web_mode:
        from web_view import WebView
        web = WebView(args.host, args.port)

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(src.model, src.data, key_callback=tele.glfw_key)

    out_dir = os.path.join(SIM_DIR, "runs", time.strftime("live_%Y%m%d_%H%M%S"))
    steps = max(1, int(round(args.dt / src.model.opt.timestep)))
    trail, fps, dash, gimg = [], 0.0, None, None

    def save():
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"map_{time.strftime('%H%M%S')}")
        pipe.grid.save(stem)
        if dash is not None:
            cv2.imwrite(stem + "_dashboard.png", dash)
        cv2.imwrite(stem + "_bev.png", gimg)
        print(f"[saved] {stem}.npy/.pgm/.yaml/_bev.png")

    print(__doc__.split("Each loop")[0])
    if web is not None:
        print("=" * 60)
        print(f"  Open {web.url} in your browser")
        print("  (VS Code forwards the port automatically; otherwise:")
        print(f"   ssh -L {args.port}:localhost:{args.port} <user>@<this-machine>)")
        print("  Ctrl+C here or Q in the page to quit.")
        print("=" * 60)
    try:
        n_frames = 0
        while not tele.flags["quit"]:
            n_frames += 1
            if args.max_frames and n_frames > args.max_frames:
                break
            t0 = time.perf_counter()

            # ---- scripted route
            if route is not None:
                if step_kind is None:
                    if not route:
                        src.set_wheels(0, 0)
                        print("[auto] route finished")
                        if args.no_gui:
                            break
                        route = None
                    else:
                        step_kind, step_target = route.pop(0)
                        step_end = src.data.time + (step_target if step_kind == "wait" else 30.0)
                if step_kind is not None:
                    cmd = (0.0, 0.0) if step_kind == "wait" else controller(src, step_kind, step_target)
                    if cmd is None or src.data.time >= step_end:
                        src.set_wheels(0, 0); step_kind = None
                    else:
                        src.set_wheels(*cmd)

            # ---- simulate, then map one frame
            src.step(steps)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            if web is not None:
                for k in web.pop_keys():
                    tele.web_key(k)
            if tele.flags["reset"]:
                pipe.reset(); trail.clear(); tele.flags["reset"] = False
                print("[map] reset")

            frame = src.get_frame()
            r = pipe.process(frame)
            trail.append(frame.robot_xy.copy())
            state = pipe.grid.state()

            # ---- draw
            g = ground_truth() if tele.flags["gt"] else None
            fov_h = 2 * np.arctan(frame.rgb.shape[1] / 2 / frame.K[0, 0])
            cam_fwd = -frame.cam_rot[:, 2]
            cam_yaw = float(np.arctan2(cam_fwd[1], cam_fwd[0]))
            gimg = render_global(pipe.grid, state, rc["scale"], frame.robot_xy, frame.robot_yaw,
                                 rc["robot_radius"], inflated=pipe.grid.inflate(state, rc["robot_radius"]),
                                 gt_occ=g, trail=trail)
            draw_robot(gimg, pipe.grid, rc["scale"], frame.robot_xy, cam_yaw, 0.0, fov_h,
                       cfg["grid"].get("obstacle_max_range", cfg["depth"]["max_range"]))
            limg = render_local(gimg, pipe.grid, rc["scale"], frame.robot_xy, frame.robot_yaw, rc["local_size"])

            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)

            if not args.no_gui:
                rgb = fit(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), 560)
                dimg = fit(depth_vis(r["depth"], cfg["depth"]["max_range"]), 560)
                label(rgb, f"camera  t={frame.time:.1f}s")
                corr = "" if r["scale"] is None else f"  a={r['scale']:.2f} b={r['offset']:+.2f}"
                label(dimg, f"{cfg['depth']['source']} depth{corr}")
                left = np.vstack([rgb, dimg])
                gview = fit(gimg, 760, left.shape[0])
                label(gview, f"global map  {fps:.1f} FPS" + ("  [GT outline]" if g is not None else ""))
                lview = fit(limg, 380, left.shape[0])
                label(lview, "robot view")
                h = left.shape[0]
                if web is not None:     # chase camera under the robot view (replaces the 3D viewer)
                    chase = fit(cv2.cvtColor(src.render_chase(), cv2.COLOR_RGB2BGR), lview.shape[1])
                    label(chase, "chase camera")
                    lview = np.vstack([lview, chase])[:h]
                dash = np.hstack([left, pad_to(gview, gview.shape[1], h), pad_to(lview, lview.shape[1], h)])
                if web is not None:
                    mode = "TRUE OUTLINE ON" if tele.flags["gt"] else ""
                    web.status = f"t={frame.time:.1f}s  {fps:.1f} FPS  {mode}"
                    web.show(dash)
                else:
                    cv2.imshow("BEV", dash)
                    tele.key(cv2.waitKeyEx(1))

            if tele.flags["save"]:
                save(); tele.flags["save"] = False
    finally:
        try:
            if gimg is not None:
                save()
        except Exception as e:  # noqa: BLE001
            print(f"[save failed] {e}")
        if viewer is not None:
            viewer.close()
        if web is not None:
            web.close()
        else:
            cv2.destroyAllWindows()
        src.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[quit]")
