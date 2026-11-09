"""SanD-Planner server (port 8892): the upstream SanD-Planner agent (github.com/WangJinCheng1998/sandplanner, cloned
next to this repo in ~/Desktop/sandplanner, not changed) behind the NavDP HTTP API (/navigator_reset,
/pointgoal_step), with settings for this rover. The client is navdp.tube.client.SandPlannerClient, which sends
depth in SanD-Planner's training camera (640x480, fx 389.55).

Changed from upstream's simple_server.py:
  - clearance_height: SanD-Planner checks its paths on a horizontal slice of its distance field this far below the
    camera (upstream 0.36 m, for a taller robot). This rover's camera is 0.21 m above the floor, so the default
    here is 0.11 m: a slice 10 cm above the floor, through the 0.30 m test boxes.
  - nothing is written to disk per step (upstream saves every depth / RGB frame and an fps video).

    conda activate navdp
    PYTHONPATH=~/Desktop/sandplanner/.deps python scripts/sandplanner_server.py   # .deps: timm (see start script)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAND_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "sandplanner"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sand-root", default=SAND_ROOT, help="the sandplanner clone")
    ap.add_argument("--checkpoint", default=None, help="default: <sand-root>/checkpoints/NoMax.pth")
    ap.add_argument("--clearance-height", type=float, default=0.11,
                    help="m below the camera of the slice its evaluator checks (upstream 0.36)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8892)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.sand_root)
    checkpoint = args.checkpoint or os.path.join(root, "checkpoints", "NoMax.pth")
    if not os.path.isfile(checkpoint):
        sys.exit(f"checkpoint not found: {checkpoint}")
    sys.path.insert(0, root)
    os.environ.setdefault("SAND_PLANNER_BASE_DIR", root)

    import cv2
    from flask import Flask, jsonify, request
    from PIL import Image
    from sand_planner.agent.sand_planner_agent import SandPlannerAgent
    from sand_planner.config import InferenceConfig

    app = Flask(__name__)
    state = {"agent": None}

    @app.route("/navigator_reset", methods=["POST"])
    def navigator_reset():
        body = request.get_json()
        intrinsic = np.array(body.get("intrinsic"), np.float64)
        threshold = np.array(body.get("stop_threshold", -3.0))
        batch = int(body.get("batch_size", 1))
        if state["agent"] is None:
            config = InferenceConfig(checkpoint_path=checkpoint, device=args.device, save_visualizations=False,
                                     save_data=False, show_verbose=args.verbose,
                                     clearance_height=args.clearance_height)
            agent = SandPlannerAgent(image_intrinsic=intrinsic, config=config, verbose=args.verbose)
            agent.reset(batch, threshold)
            started = time.time()           # warm-up: the first calls compile the model (upstream does the same)
            dummy_dep = np.random.rand(1, 480, 640, 1).astype(np.float32) * 5.0
            for _ in range(3):
                agent.step_pointgoal(np.array([[3.0, 0.0, 0.0]]), np.zeros((1, 480, 640, 3), np.uint8), dummy_dep)
            print(f"[sandplanner] warmed up in {time.time() - started:.1f} s")
            state["agent"] = agent
        agent = state["agent"]
        if not np.array_equal(agent.image_intrinsic, intrinsic):
            agent.image_intrinsic = intrinsic
            agent.update_camera_config(intrinsic)
        agent.reset(batch, threshold)
        return jsonify({"algo": "sand_planner"})

    @app.route("/pointgoal_step", methods=["POST"])
    def pointgoal_step():
        agent = state["agent"]
        if agent is None:
            return jsonify({"error": "reset first"}), 400
        goal = json.loads(request.form.get("goal_data"))
        goal = np.stack([np.array(goal["goal_x"]), np.array(goal["goal_y"]), np.zeros(len(goal["goal_x"]))], axis=1)
        image = cv2.cvtColor(np.asarray(Image.open(request.files["image"].stream).convert("RGB")), cv2.COLOR_RGB2BGR)
        depth = np.asarray(Image.open(request.files["depth"].stream).convert("I")).astype(np.float32) / 10000.0
        best, cands, costs, _ = agent.step_pointgoal(goal, image[None], depth[None, :, :, None])
        best, cands, costs = np.asarray(best), np.asarray(cands), np.asarray(costs, np.float64)
        best = best[0] if best.ndim == 3 else best
        cands = cands[0] if cands.ndim == 4 else cands
        costs = costs.reshape(-1)
        return jsonify({"trajectory": best[None].tolist(), "all_trajectory": cands[None].tolist(),
                        "all_values": costs[None].tolist()})      # costs: lower = better (the client negates)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "agent_loaded": state["agent"] is not None, "checkpoint": checkpoint})

    print(f"SanD-Planner server on http://{args.host}:{args.port}  checkpoint {checkpoint}  "
          f"clearance_height {args.clearance_height}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
