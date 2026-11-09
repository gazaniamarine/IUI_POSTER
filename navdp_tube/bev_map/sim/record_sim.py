#!/usr/bin/env python3
"""Drive the rover along a scripted route in MuJoCo and record frames.

Writes <out>/:
  rgb/000123.png       camera image
  depth_gt/000123.npy  MuJoCo ground-truth depth (metres, float16)
  mask/000123.png      255 where the pixel shows scene geometry (0 = empty void)
  poses.csv            file,x,y,theta
  frames.jsonl         per frame: exact camera position/rotation, robot pose, time
  camera.json          intrinsics + mount (height, pitch)

    MUJOCO_GL=egl python record_sim.py --out runs/sim_route
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SIM_DIR))   # core pipeline files live in bev_map/
SIM_CONFIG = os.path.join(SIM_DIR, "config_sim.yaml")

from bev_pipeline import load_config
from sim_source import SimSource

# (kind, target): drive_to (x, y) | turn_to yaw(rad) | wait seconds
ROUTE = [
    ("wait", 0.5),
    ("drive_to", (-0.7, 0.0)),
    ("turn_to", 0.9),
    ("turn_to", -0.9),
    ("turn_to", 0.0),
    ("drive_to", (0.55, 0.0)),
    ("turn_to", 1.57),
    ("turn_to", 3.14),
    ("turn_to", -1.57),
    ("turn_to", 0.0),
]

WHEEL_MAX = 5.0          # actuator ctrlrange
FRAME_EVERY = 0.1        # seconds of sim time between recorded frames


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def controller(src, kind, target):
    """Return wheel commands (left, right) or None when the step is done."""
    xy, yaw = src.robot_pose()
    if kind == "turn_to":
        err = wrap(target - yaw)
        if abs(err) < 0.03:
            return None
        w = np.clip(3.0 * err, -1, 1) * WHEEL_MAX * 0.6
        if abs(w) < 1.2:
            w = np.copysign(1.2, w)
        return -w, w
    if kind == "drive_to":
        d = np.asarray(target) - xy
        dist = np.hypot(*d)
        if dist < 0.03:
            return None
        err = wrap(np.arctan2(d[1], d[0]) - yaw)
        if abs(err) > 0.35:
            return controller(src, "turn_to", yaw + err)
        v = WHEEL_MAX * min(1.0, dist / 0.15)
        steer = np.clip(4.0 * err, -1, 1) * 0.5 * v
        return v - steer, v + steer
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(SIM_DIR, "runs", "sim_route"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, extra=[SIM_CONFIG])
    src = SimSource(cfg)
    os.makedirs(f"{args.out}/rgb", exist_ok=True)
    os.makedirs(f"{args.out}/depth_gt", exist_ok=True)
    os.makedirs(f"{args.out}/mask", exist_ok=True)

    src.settle(1.0)
    f0 = src.get_frame()
    R = f0.cam_rot
    fwd = -R[:, 2]                                         # optical axis in world
    pitch_deg = float(np.degrees(np.arcsin(-fwd[2])))     # +ve = looking down
    json.dump({"K": f0.K.tolist(), "width": src.width, "height": src.height,
               "cam_height": float(f0.cam_pos[2]), "pitch_deg": pitch_deg},
              open(f"{args.out}/camera.json", "w"), indent=2)

    dt = src.model.opt.timestep
    steps_per_frame = max(1, int(round(FRAME_EVERY / dt)))
    poses = open(f"{args.out}/poses.csv", "w", newline="")
    pw = csv.writer(poses); pw.writerow(["file", "x", "y", "theta"])
    meta = open(f"{args.out}/frames.jsonl", "w")

    idx = 0
    for kind, target in ROUTE:
        t_end = src.data.time + target if kind == "wait" else src.data.time + 30.0
        while src.data.time < t_end:
            if kind == "wait":
                cmd = (0.0, 0.0)
            else:
                cmd = controller(src, kind, target)
                if cmd is None:
                    break
            src.set_wheels(*cmd)
            src.step(steps_per_frame)

            f = src.get_frame()
            name = f"{idx:06d}"
            cv2.imwrite(f"{args.out}/rgb/{name}.png", cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
            np.save(f"{args.out}/depth_gt/{name}.npy", f.depth.astype(np.float16))
            if f.mask is not None:
                cv2.imwrite(f"{args.out}/mask/{name}.png", f.mask.astype(np.uint8) * 255)
            pw.writerow([f"{name}.png", f"{f.robot_xy[0]:.5f}", f"{f.robot_xy[1]:.5f}", f"{f.robot_yaw:.5f}"])
            meta.write(json.dumps({"file": f"{name}.png", "t": f.time,
                                   "cam_pos": f.cam_pos.tolist(), "cam_rot": f.cam_rot.tolist(),
                                   "robot_xy": f.robot_xy.tolist(), "robot_yaw": f.robot_yaw}) + "\n")
            idx += 1
        src.set_wheels(0, 0)
        xy, yaw = src.robot_pose()
        print(f"{kind:9s} {target}  -> at ({xy[0]:.2f}, {xy[1]:.2f}) yaw {yaw:.2f}  frames {idx}")

    poses.close(); meta.close(); src.close()
    print(f"recorded {idx} frames to {args.out}")


if __name__ == "__main__":
    main()
