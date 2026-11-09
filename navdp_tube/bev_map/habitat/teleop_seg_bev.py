"""Drive the rover through the Mars yard with the keyboard and build the BEV occupancy map on the fly from the
SAM2 terrain segmentation (best.pth): the original BEVPipeline / OccupancyGrid / bev_render, depth.source = sam2.

    conda activate habitat
    python teleop_seg_bev.py                                   # free: soil; occupied: sand, bedrock, rocks
    python teleop_seg_bev.py --occupied bedrock --free soil,sand
    python teleop_seg_bev.py --start 12,-3 --yaw 20
    python teleop_seg_bev.py --keys wwwwaaawwwwdddddwwww         # scripted drive, no window: saves the last view

Keys (window focused):  W / S  forward / back     A / D  turn left / right     SPACE  map this view again
                        R  clear the map          P  save the map + a screenshot     Q / ESC  quit
Window: camera + segmentation (boulders outlined cyan) | egocentric map (robot at the bottom, facing up).

Class roles:
  --occupied  the ground these classes cover is occupied where it lies (flat)  (seg.ground_occupied_classes)
  --free      free ground                                                       (seg.free_classes)
  --rocks     obstacles standing on the ground: occupied from where they meet it to --rock-depth behind;
              the ground hidden behind them stays unknown                       (seg.obstacle_classes)
Each map cell takes the class of the pixel that sees it; ground hidden by the terrain is free, behind rocks unknown.
The map itself: seg_bev_map.py (the same map run_mars_habitat.py --map seg gives the planners).
The rover is blocked by the big rocks.
"""

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

try:                                     # not in the habitat env: run again with its python
    import habitat_sim  # noqa: F401
except ImportError:
    _py = os.path.join(os.environ.get("CONDA_EXE", os.path.expanduser("~/miniconda3/bin/conda")).rsplit("/bin/", 1)[0],
                       "envs", "habitat", "bin", "python")
    if os.path.exists(_py) and os.path.realpath(sys.executable) != os.path.realpath(_py):
        os.execv(_py, [_py] + sys.argv)
    sys.exit("habitat_sim not found: conda activate habitat")

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bev_render import label  # noqa: E402
from mars_scene import MarsScene, stage_config  # noqa: E402
from run_mars_habitat import DEFAULT_GLB, intrinsics, make_sim, place_camera  # noqa: E402
from seg_bev_map import SegBEVMap, add_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--start", default="12.5,-3.4", help="x,y in the yard (planner frame; default: open ground)")
    ap.add_argument("--yaw", type=float, default=18.0, help="start heading, deg ccw from +x")
    ap.add_argument("--step", type=float, default=0.25, help="m per W / S press")
    ap.add_argument("--turn", type=float, default=10.0, help="deg per A / D press")
    ap.add_argument("--camera-height", type=float, default=0.5)
    ap.add_argument("--robot-radius", type=float, default=0.3)
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "teleop"))
    ap.add_argument("--keys", default="", help="drive these keys without a window (e.g. wwwaawww), save the last view")
    ap.add_argument("--set", nargs="*", default=[], help="more config overrides, e.g. grid.resolution=0.05")
    args = ap.parse_args()
    if not args.keys and not os.environ.get("DISPLAY"):
        sys.exit("no DISPLAY: teleop needs a screen (or use --keys)")

    scene = MarsScene(args.glb)
    sx, sy = (float(v) for v in args.start.split(","))
    W, H, HFOV = 640, 480, 79.0
    sim, agent = make_sim(stage_config(args.glb, os.path.join(HERE, "runs")), W, H, HFOV)
    K = intrinsics(W, H, HFOV)
    bev = SegBEVMap(scene, K, W, H, (sx, sy), args, args.set)
    clear = scene.distance_field()
    x, y, yaw = sx, sy, math.radians(args.yaw)
    os.makedirs(args.out, exist_ok=True)
    print("Keys: W/S forward/back  A/D turn  SPACE map again  R clear  P save  Q quit", flush=True)
    print(f"occupied: {bev.occ_c}  free: {bev.free_c}  rocks: {bev.rock_c}", flush=True)

    def step():
        cam_h = float(scene.height_at(x, y)) + args.camera_height
        place_camera(agent, x, y, yaw, cam_h)
        o = sim.get_sensor_observations()
        return bev.update(np.ascontiguousarray(o["rgb"][..., :3]), np.asarray(o["depth"], np.float32),
                          x, y, yaw, cam_h)

    def draw(fr, r, ms):
        cam = bev.seg_view(fr, r)
        ego = bev.ego_view(x, y, yaw, H, args.robot_radius)
        label(cam, f"({x:+.1f}, {y:+.1f})  {math.degrees(yaw):+.0f} deg   seg+map {ms:.0f} ms")
        label(ego, "egocentric map")
        leg = np.full((44, W + H, 3), 255, np.uint8)
        t1, t2 = bev.legend()
        cv2.putText(leg, t1, (6, 17), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(leg, t2 + " | W/S A/D drive, SPACE map, R clear, P save, Q quit", (6, 37), 0, 0.45, (0, 0, 0),
                    1, cv2.LINE_AA)
        return np.vstack([leg, np.hstack([cam, ego])])

    fr, r, ms = step()
    win = "Mars yard teleop: SAM2 BEV map"
    script = list(args.keys.lower())
    if not script:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            dash = draw(fr, r, ms)
            if script is not None and args.keys:
                if not script:
                    stem = os.path.join(args.out, time.strftime("keys_%Y%m%d_%H%M%S"))
                    bev.grid.save(stem)
                    cv2.imwrite(stem + "_view.png", dash)
                    print(f"[saved] {stem}_view.png", flush=True)
                    break
                k = ord(script.pop(0))
            else:
                cv2.imshow(win, dash)
                k = cv2.waitKey(0) & 0xFF
            if k in (ord("q"), 27):
                break
            moved = True
            if k in (ord("w"), ord("s")):
                d = args.step * (1 if k == ord("w") else -1)
                nx, ny = x + d * math.cos(yaw), y + d * math.sin(yaw)
                if scene.grid.inside(nx, ny) and scene.lookup(clear, nx, ny) > args.robot_radius:
                    x, y = nx, ny
                else:
                    print("[blocked] a rock / the edge of the yard", flush=True)
                    moved = False
            elif k in (ord("a"), ord("d")):
                yaw += math.radians(args.turn) * (1 if k == ord("a") else -1)
                yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            elif k == ord("r"):
                bev.reset()
            elif k == ord("p"):
                stem = os.path.join(args.out, time.strftime("map_%Y%m%d_%H%M%S"))
                bev.grid.save(stem)
                cv2.imwrite(stem + "_view.png", dash)
                print(f"[saved] {stem}.pgm/.yaml/.npy + _view.png", flush=True)
                moved = False
            elif k != ord(" "):
                moved = False
            if moved:
                fr, r, ms = step()
    finally:
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()
