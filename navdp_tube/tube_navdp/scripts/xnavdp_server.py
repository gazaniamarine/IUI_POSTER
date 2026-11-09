"""X-NavDP server (port 8891): the upstream X-NavDP policy server, unchanged, with settings for this rover.

Runs NavDP/baselines/x-navdp/eval/src/policy_server.py (same HTTP API as NavDP: /navigator_reset, /pointgoal_step)
with the wheeled embodiment and no video writing. The client is navdp.tube.client.XNavDPClient.

Only one thing is changed, the stuck detector's distance: X-NavDP calls the robot stuck when it moved less than
stuck_xy m over the last stuck_window plans, and then executes a random sample. Upstream uses 0.25 m, tuned for its
eval (0.5 m/s, ~6 plans/s); at this rover's 0.10 m/s that would flag "stuck" all the time. Default here 0.05 m
(0.25 scaled by 0.10 / 0.5).

Added (no change to X-NavDP itself): every /pointgoal_step answer also carries ``stuck`` (X-NavDP's own stuck flag:
true when it executed a random sample) and ``acts_saved``; a request with form field ``acts_path`` saves the network's
activations of that step there (navdp/tube/acts.py: scene tokens, decoder layer --act-layer, critic); form field
``seed`` seeds that step's sampling noise.

    conda activate navdp
    python scripts/xnavdp_server.py --checkpoint ../NavDP/baselines/x-navdp/pretrain_model/x-navdp_posttrain.ckpt
"""
import argparse
import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
XNAVDP_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "NavDP", "baselines", "x-navdp"))


def wrap_step(policy_server, act_layer):
    """Add stuck / acts_saved to the step answer, and save activations when the request asks (form acts_path)."""
    import json
    sys.path.insert(0, os.path.dirname(HERE))
    from flask import jsonify, request
    from navdp.tube.acts import ActRecorder

    app = policy_server.app
    endpoint = next(e for e, r in ((r.endpoint, r) for r in app.url_map.iter_rules()) if r.rule == "/pointgoal_step")
    step = app.view_functions[endpoint]
    state = {"acts": None, "net": None}

    def wrapped():
        nav = policy_server.navdp_navigator
        acts_path = request.form.get("acts_path")
        if request.form.get("seed") is not None:     # reproducible sampling noise (and stuck pick) for this step
            import numpy as np
            import torch
            torch.manual_seed(int(request.form["seed"]))
            np.random.seed(int(request.form["seed"]) % 2 ** 32)
        if acts_path and nav is not None and state["net"] is not nav.navi_former:
            state["acts"], state["net"] = ActRecorder(nav.navi_former, act_layer), nav.navi_former
        if acts_path and state["acts"] is not None:
            state["acts"].arm()
        response = step()
        payload = json.loads(response.get_data())
        nav = policy_server.navdp_navigator
        stuck = bool(nav.is_stuck[0]) if nav is not None else False
        saved = False
        if acts_path and state["acts"] is not None:
            saved = state["acts"].save(acts_path, values=payload["all_values"][0], stuck=stuck)
        payload.update(stuck=stuck, acts_saved=saved)
        return jsonify(payload)

    app.view_functions[endpoint] = wrapped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(XNAVDP_ROOT, "pretrain_model", "x-navdp_posttrain.ckpt"))
    ap.add_argument("--xnavdp-root", default=XNAVDP_ROOT)
    ap.add_argument("--embodiment", default="wheeled", choices=("wheeled", "humanoid", "quadruped"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8891)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--stuck-window", type=int, default=5, help="plans looked back for the stuck test (upstream 5)")
    ap.add_argument("--stuck-xy", type=float, default=0.05,
                    help="stuck = moved less than this (m) over --stuck-window plans (upstream 0.25)")
    ap.add_argument("--act-layer", type=int, default=None,
                    help="decoder layer saved for a request with acts_path (default: the middle one)")
    ap.add_argument("--visualization", action="store_true", help="write X-NavDP's own fps videos to vis_output/")
    args = ap.parse_args()
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    root = os.path.abspath(args.xnavdp_root)
    sys.path.insert(0, root)
    os.chdir(root)  # the server writes vis_output/ under the working directory (ignored by x-navdp/.gitignore)
    from eval.src import policy_agent, policy_server  # noqa: E402

    reset = policy_agent.NavDP_Agent.reset
    policy_agent.NavDP_Agent.reset = functools.partialmethod(
        reset, stuck_window=args.stuck_window, stuck_xy_threshold=args.stuck_xy)

    wrap_step(policy_server, args.act_layer)
    policy_server.init_app(args.embodiment, no_visualization=not args.visualization, device=args.device,
                           checkpoint=os.path.abspath(args.checkpoint))
    print(f"X-NavDP server on http://{args.host}:{args.port}  embodiment {args.embodiment}, "
          f"stuck < {args.stuck_xy} m over {args.stuck_window} plans  (Ctrl+C to stop)")
    policy_server.run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
