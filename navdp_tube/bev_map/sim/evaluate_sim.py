
#!/usr/bin/env python3
"""Score the BEV pipeline on a recorded simulation run against MuJoCo ground truth.

    MUJOCO_GL=egl python evaluate_sim.py --run runs/sim_route

Modes (all through the same live pipeline):
  sim        MuJoCo ground-truth depth          -> checks geometry / grid code
  da_none    Depth Anything, no correction      -> what raw DA gives
  da_scale   Depth Anything * s (floor fit)
  da_affine  Depth Anything * a + b (floor fit) -> default
Outputs <run>/eval/: per-mode map PNGs (red outline = true obstacles), a side-by-side
comparison, depth error stats, and metrics.json.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SIM_DIR))   # core pipeline files live in bev_map/
SIM_CONFIG = os.path.join(SIM_DIR, "config_sim.yaml")

from bev_pipeline import BEVPipeline, load_config
from bev_render import label, render_global
from ground_truth import ground_truth_grid, score
from sim_source import SimSource

MODES = {
    "sim": ("sim", "none"),
    "da_none": ("depth_anything", "none"),
    "da_scale": ("depth_anything", "scale"),
    "da_affine": ("depth_anything", "affine"),
}


def load_run(run):
    cam = json.load(open(f"{run}/camera.json"))
    metas = [json.loads(l) for l in open(f"{run}/frames.jsonl")]
    return cam, metas


def frame_at(run, cam, m, with_rgb=True):
    name = os.path.splitext(m["file"])[0]
    rgb = cv2.cvtColor(cv2.imread(f"{run}/rgb/{m['file']}"), cv2.COLOR_BGR2RGB) if with_rgb else None
    mpath = f"{run}/mask/{name}.png"
    mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 0 if os.path.exists(mpath) else None
    return SimpleNamespace(
        name=name, rgb=rgb, mask=mask, depth=np.load(f"{run}/depth_gt/{name}.npy").astype(np.float32),
        K=np.asarray(cam["K"]), cam_pos=np.asarray(m["cam_pos"]), cam_rot=np.asarray(m["cam_rot"]),
        robot_xy=np.asarray(m["robot_xy"]), robot_yaw=m["robot_yaw"],
    )


def cache_da(run, cam, metas, cfg):
    """Run Depth Anything once per frame; reuse for every correction mode."""
    out = f"{run}/depth_da"
    os.makedirs(out, exist_ok=True)
    todo = [m for m in metas if not os.path.exists(f"{out}/{os.path.splitext(m['file'])[0]}.npy")]
    if not todo:
        return
    from depth_estimator import DepthAnything
    model = DepthAnything(cfg["depth"]["model_id"])
    for i in range(0, len(todo), 8):
        batch = [frame_at(run, cam, m) for m in todo[i:i + 8]]
        for f, d in zip(batch, model([f.rgb for f in batch])):
            np.save(f"{out}/{f.name}.npy", d.astype(np.float32))
        print(f"  depth anything {min(i + 8, len(todo))}/{len(todo)}", end="\r")
    print()


class CachedDA:
    """Stands in for the model: returns the cached prediction for the current frame."""
    def __init__(self, run):
        self.run, self.name = run, None

    def __call__(self, rgbs):
        return [np.load(f"{self.run}/depth_da/{self.name}.npy")]


def perturb(cam_rot, pitch_deg, roll_deg):
    """Tilt the camera rotation by pitch (about camera x) and roll (about camera optical axis)."""
    def rot(axis, deg):
        a = np.radians(deg); axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    R = rot(cam_rot[:, 0], -pitch_deg) @ cam_rot          # +pitch = camera looks further down
    return rot(R[:, 2], roll_deg) @ R


def depth_errors(depth, gt, max_range):
    v = np.isfinite(gt) & (gt > 0.15) & (gt < max_range) & np.isfinite(depth) & (depth > 0)
    rel = np.abs(depth[v] - gt[v]) / gt[v]
    return float(np.median(rel)), float(np.mean(rel < 0.10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.path.join(SIM_DIR, "runs", "sim_route"))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. grid.resolution=0.05")
    ap.add_argument("--tag", default="", help="suffix for output folder")
    ap.add_argument("--pitch-offset", type=float, default=0.0,
                    help="deg: constant camera pitch error given to the pipeline (wrong mount angle)")
    ap.add_argument("--pitch-noise", type=float, default=0.0,
                    help="deg: random per-frame pitch error (std), e.g. bouncing while driving")
    ap.add_argument("--roll-noise", type=float, default=0.0, help="deg: random per-frame roll error (std)")
    args = ap.parse_args()

    cfg = load_config(overrides=args.set, extra=[SIM_CONFIG])
    cam, metas = load_run(args.run)
    metas = metas[::args.every]
    out = f"{args.run}/eval{args.tag}"
    os.makedirs(out, exist_ok=True)

    # ground truth occupancy (cached)
    src = SimSource(cfg)
    probe = BEVPipeline({**cfg, "depth": {**cfg["depth"], "source": "sim"}})
    gt_path = f"{args.run}/gt_occ_{cfg['grid']['resolution']}.npy"
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
    else:
        print("computing ground-truth occupancy ...")
        gt = ground_truth_grid(src.model, probe.grid, cfg["sim"]["rover_body"],
                               cfg["classify"]["obstacle_min_z"])
        np.save(gt_path, gt)
    src.close()

    modes = args.modes.split(",")
    if any(MODES[m][0] == "depth_anything" for m in modes):
        cache_da(args.run, cam, metas, cfg)

    results, panels = {}, []
    for mode in modes:
        source, corr = MODES[mode]
        mcfg = {**cfg, "depth": {**cfg["depth"], "source": source, "floor_correction": corr}}
        da = CachedDA(args.run)
        pipe = BEVPipeline(mcfg, depth_model=da)
        rel_errs, within10, scales, trail = [], [], [], []
        noise_rng = np.random.default_rng(1)
        for k, m in enumerate(metas):
            f = frame_at(args.run, cam, m, with_rgb=False)
            if args.pitch_offset or args.pitch_noise or args.roll_noise:
                f.cam_rot = perturb(f.cam_rot, args.pitch_offset + noise_rng.normal(0, args.pitch_noise),
                                    noise_rng.normal(0, args.roll_noise))
            da.name = f.name
            r = pipe.process(f)
            trail.append(f.robot_xy)
            if k % 10 == 0:
                e, w = depth_errors(r["depth"], f.depth, cfg["depth"]["max_range"])
                rel_errs.append(e); within10.append(w)
                if r["scale"] is not None:
                    scales.append((r["scale"], r["offset"]))
        state = pipe.grid.state()
        s = score(state, gt)
        s["depth_median_rel_err"] = float(np.median(rel_errs))
        s["depth_within_10pct"] = float(np.median(within10))
        if scales:
            s["correction_a_b_median"] = np.median(np.array(scales), 0).round(3).tolist()
        results[mode] = s
        print(f"{mode:10s} precision {s['precision']:.3f}  recall {s['recall']:.3f}  "
              f"iou {s['iou_strict']:.3f}  noise cells {s['false_occ_cells']:4d}  "
              f"depth err {s['depth_median_rel_err']:.3f}")

        sc = cfg["render"]["scale"]
        img = render_global(pipe.grid, state, sc, f.robot_xy, f.robot_yaw, cfg["render"]["robot_radius"],
                            gt_occ=gt, trail=trail)
        label(img, f"{mode}  P {s['precision']:.2f}  R {s['recall']:.2f}  IoU {s['iou_strict']:.2f}")
        cv2.imwrite(f"{out}/map_{mode}.png", img)
        np.save(f"{out}/grid_{mode}.npy", pipe.grid.to_ros_prob())
        panels.append(img)

    if panels:
        rows = [np.hstack(panels[i:i + 2]) if i + 1 < len(panels) else
                np.hstack([panels[i], np.zeros_like(panels[i])]) for i in range(0, len(panels), 2)]
        cv2.imwrite(f"{out}/compare.png", np.vstack(rows))
    json.dump(results, open(f"{out}/metrics.json", "w"), indent=2)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
