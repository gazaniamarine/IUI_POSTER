"""Tube-guided NavDP server (port 8890): NavDP's point-goal request plus a BEV occupancy grid and the mode.

The grid (``tube_data["bev"]``, from navdp.tube.stt.Bev.encode) is in the rover's body frame and already holds the
rover's size; navdp/tube/driver.py makes it from a world-frame map. No grid = no obstacles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdp.tube.acts import ActRecorder  # noqa: E402
from navdp.tube.agent import make_agent  # noqa: E402
from navdp.tube.guided_sampler import LYAPUNOV_MODES, MODES, GuideConfig  # noqa: E402
from navdp.tube.stt import PUSH_MODES, Bev, TubeConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--navdp-root", type=Path, required=True, help="The NavDP repository (has baselines/navdp)") 
    p.add_argument("--checkpoint", type=Path, required=True) 
    p.add_argument("--port", type=int, default=8890)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=16, help="Candidates per call (official: 16)")
    p.add_argument("--split", type=int, default=4, help="Scheduler step (0-9) at which the tubes are built")
    p.add_argument("--alpha0", type=float, default=0.3)
    p.add_argument("--project-last", type=int, default=2)
    p.add_argument("--tube-degree", type=int, default=6)
    p.add_argument("--tube-min-width", type=float, default=0.3)
    p.add_argument("--tube-margin", type=float, default=0.05, help="Bounds stay this far inside their free boxes")
    p.add_argument("--tube-samples-per-step", type=int, default=6, help="Sampled times per waypoint step")
    p.add_argument("--consistency", type=float, default=0.0,
                   help="mu: pull toward the previous plan sent by the driver (0 = off; at most 1). Default for "
                        "requests; a request's tube_data['consistency'] overrides it")
    p.add_argument("--consistency-decay", type=float, default=0.9, help="gamma: weight mu gamma^j on waypoint j")
    p.add_argument("--consistency-from", type=int, default=0, help="Scheduler step (0-9) from which it pulls")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="beta: smooth projection into the boxes (0 = the per-waypoint clamp, as before). Default "
                        "for requests; a request's tube_data['smooth'] overrides it")
    p.add_argument("--smooth-iters", type=int, default=100, help="ADMM iterations of the smooth projection")
    p.add_argument("--tube-push", choices=PUSH_MODES, default="nearest",
                   help="Step 1 push: nearest (each waypoint to its nearest free cell, the first version) or smooth "
                        "(the nearest push, then smoothed inside free boxes round the pushed waypoints). Default for requests; a request's tube_data['push'] "
                        "overrides it")
    p.add_argument("--rebuild-tubes", action="store_true",
                   help="Build the tubes again at every guided step (default: once, at --split). Per request: "
                        "tube_data['rebuild']")
    p.add_argument("--lyapunov", choices=LYAPUNOV_MODES, default="off",
                   help="Lyapunov-guided pull (LyaGuide): off (the alpha ramp) | min (only the pull the condition "
                        "needs) | ramp (the alpha ramp, raised where needed). Tube and obstacle modes. Per request: "
                        "tube_data['lyapunov']")
    p.add_argument("--lyapunov-delta", type=float, default=0.3,
                   help="delta: dist^2 to the boxes (obstacle mode: to the free cells) must fall by this fraction "
                        "per guided step")
    p.add_argument("--guide-until", type=int, default=0,
                   help="Pull (tube / obstacle) only before this denoising step, then NavDP denoises by itself "
                        "(0 = to the last step, as before; else in [split + 1, 10]). The report then gives model_v, "
                        "final_v, final_in_cells and chosen_v to see if the model keeps the paths clear. Per request: "
                        "tube_data['guide_until']")
    p.add_argument("--push-step", type=float, default=0.25,
                   help="obstacle mode: most a pushed path moves per waypoint and axis, m (0.25 = NavDP's own, as "
                        "before; up to 1). Above 0.25 the moves leave NavDP's [-1, 1]. Per request: "
                        "tube_data['push_step']")
    p.add_argument("--push-margin", type=float, default=0.0,
                   help="obstacle mode: push waypoints this far past the grown obstacles, m (0 = to their edge, as "
                        "before; try 0.05). Per request: tube_data['push_margin']")
    p.add_argument("--act-layer", type=int, default=None,
                   help="decoder layer whose output a request with tube_data['acts_path'] saves (default: the middle "
                        "one; navdp/tube/acts.py)")
    return p.parse_args()


def decode(request, batch_rows: int = 1):
    import cv2
    from PIL import Image
    image = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).reshape((batch_rows, -1, image.shape[1], 3))
    depth = np.asarray(Image.open(request.files["depth"].stream).convert("I"))[:, :, None].astype(np.float32) / 10000.0
    return image, depth.reshape((batch_rows, -1, depth.shape[1], 1))


def main():
    from flask import Flask, jsonify, request
    args = parse_args()
    app = Flask(__name__)
    state = {"agent": None, "acts": None}
    base = GuideConfig(split=args.split, alpha0=args.alpha0, project_last=args.project_last,
                       consistency=args.consistency, consistency_decay=args.consistency_decay,
                       consistency_from=args.consistency_from, smooth=args.smooth, smooth_iters=args.smooth_iters,
                       rebuild=args.rebuild_tubes, lyapunov=args.lyapunov, lyapunov_delta=args.lyapunov_delta,
                       guide_until=args.guide_until, push_step=args.push_step, push_margin=args.push_margin,
                       tube=TubeConfig(degree=args.tube_degree, min_width=args.tube_min_width, margin=args.tube_margin,
                                       samples_per_step=args.tube_samples_per_step,
                                       push_mode=args.tube_push))

    @app.route("/navigator_reset", methods=["POST"])
    def reset():
        body = request.get_json()
        if int(body.get("batch_size", 1)) != 1:
            return jsonify({"error": "batch size 1 only"}), 400
        if state["agent"] is None:
            state["agent"] = make_agent(args.navdp_root, np.array(body["intrinsic"]), str(args.checkpoint),
                                        args.device, float(body["stop_threshold"]))
            state["acts"] = ActRecorder(state["agent"].navi_former, args.act_layer)
        else:
            state["agent"].reset(1, float(body["stop_threshold"]))
        return jsonify({"algo": "navdp"})

    @app.route("/pointgoal_step", methods=["POST"])
    def step():
        started = time.time()
        goal = json.loads(request.form.get("goal_data"))
        goal = np.array([[goal["goal_x"][0], goal["goal_y"][0], 0.0]])
        tube = json.loads(request.form.get("tube_data") or "{}")
        mode = tube.get("mode", "navdp")
        if mode not in MODES:
            return jsonify({"error": f"mode must be one of {MODES}"}), 400
        bev = Bev.decode(tube["bev"]) if "bev" in tube else None
        split = int(tube.get("split", base.split))
        if not 0 <= split <= 9:
            return jsonify({"error": "split must be in [0, 9]"}), 400
        push = tube.get("push", base.tube.push_mode)
        if push not in PUSH_MODES:
            return jsonify({"error": f"push must be one of {PUSH_MODES}"}), 400
        consistency = float(tube.get("consistency", base.consistency))   # a request overrides --consistency
        smooth = float(tube.get("smooth", base.smooth))                  # and --smooth
        if not 0.0 <= consistency <= 1.0 or smooth < 0.0:
            return jsonify({"error": "consistency must be in [0, 1] and smooth >= 0"}), 400
        rebuild = bool(tube.get("rebuild", base.rebuild))                # a request overrides --rebuild-tubes,
        lyapunov = str(tube.get("lyapunov", base.lyapunov))              # --lyapunov
        lyapunov_delta = float(tube.get("lyapunov_delta", base.lyapunov_delta))   # and --lyapunov-delta
        if lyapunov not in LYAPUNOV_MODES or not 0.0 < lyapunov_delta < 1.0:
            return jsonify({"error": f"lyapunov must be one of {LYAPUNOV_MODES} and lyapunov_delta in (0, 1)"}), 400
        guide_until = int(tube.get("guide_until", base.guide_until))     # a request overrides --guide-until
        if guide_until and not split < guide_until <= 10:
            return jsonify({"error": "guide_until must be 0 (off) or in [split + 1, 10]"}), 400
        push_step = float(tube.get("push_step", base.push_step))         # a request overrides --push-step
        push_margin = float(tube.get("push_margin", base.push_margin))   # and --push-margin
        if not 0.25 <= push_step <= 1.0 or not 0.0 <= push_margin <= 0.5:
            return jsonify({"error": "push_step must be in [0.25, 1] and push_margin in [0, 0.5]"}), 400
        cfg = GuideConfig(**{**base.__dict__, "mode": mode, "split": split, "consistency": consistency,
                             "smooth": smooth, "rebuild": rebuild, "lyapunov": lyapunov,
                             "lyapunov_delta": lyapunov_delta, "guide_until": guide_until,
                             "push_step": push_step, "push_margin": push_margin,
                             "tube": replace(base.tube, start_forward=float(tube.get("start_forward", 0.0)),
                                             push_mode=push)})
        prev = np.asarray(tube["prev"], np.float64).reshape(-1, 2) if tube.get("prev") else None
        image, depth = decode(request)
        acts_path = tube.get("acts_path")       # save the network's activations of this step (acts.py)
        if tube.get("seed") is not None:        # reproducible sampling noise for this step
            import torch
            torch.manual_seed(int(tube["seed"]))
        if acts_path:
            state["acts"].arm()
        out = state["agent"].step_pointgoal_tube(goal, image, depth, bev, cfg, args.samples,
                                                 record=bool(tube.get("record_x0", False)), prev=prev)
        acts_saved = bool(acts_path) and state["acts"].save(acts_path, values=out["values"], choice=out["choice"])
        return jsonify({"trajectory": out["trajectory"][None].tolist(), "acts_saved": acts_saved,
                        "all_trajectory": out["candidates"][None].tolist(), "all_values": out["values"][None].tolist(),
                        "tube": {"mode": mode, "push": push, "consistency": consistency, "smooth": smooth,
                                 "rebuild": rebuild, "lyapunov": lyapunov, "lyapunov_delta": lyapunov_delta,
                                 "lyapunov_v": out["lyapunov_v"], "tube_builds": out["tube_builds"],
                                 "guide_until": guide_until, "model_v": out["model_v"],
                                 "push_step": push_step, "push_margin": push_margin,
                                 "final_v": float(out["final_v"].mean()),
                                 "final_in_cells": int(out["final_in_cells"].sum()),
                                 "chosen_v": float(out["final_v"][out["choice"]]),
                                 "chosen_in_cells": bool(out["final_in_cells"][out["choice"]]),
                                 "choice": out["choice"], "chosen_certified": out["chosen_certified"],
                                 "certified": out["certified"].tolist(), "group": out["group"].tolist(),
                                 "tubes": out["tubes"],
                                 "occupied_cells": 0 if bev is None else int(bev.occupied.sum()),
                                 "split_paths": None if out["split_paths"] is None else out["split_paths"].tolist(),
                                 "split": split, "prev_waypoints": 0 if prev is None else len(prev),
                                 "x0_steps": None if out["x0_steps"] is None else out["x0_steps"].round(4).tolist(),
                                 "server_s": time.time() - started}})

    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
