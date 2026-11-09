"""
Spike/benchmark for the "batched re-window" SAM3 integration pattern.

Not part of the runtime path. Standalone — run in the `sam3` conda env,
which has the right torch/cuda for the vendored `packages/sam3` package.
Does not import anything from `sam_vla.env` (that pulls in `habitat_sim`,
which lives in a different conda env), so RGB_WIDTH/RGB_HEIGHT below are
hardcoded to match sam_vla/env/habitat_env.py's real sensor resolution
rather than imported.

Why this pattern exists: `Sam3BasePredictor.start_session` calls
`model.init_state()`, which loads a *fixed* list of already-written frame
files off disk (`load_video_frames`, packages/sam3/sam3/model/io_utils.py:118)
at session-start. There is no public API to append one live frame to an
already-open session. So "run segmentation every second" on a live rollout
has to be simulated as: keep a ring buffer of recent RGB frames, every
`--seg-interval-steps` write the buffer to a scratch folder, open a *fresh*
SAM3 session on that folder, `add_prompt(text=term)` once per vocabulary
term at frame 0, `propagate_in_video(propagation_direction="forward")` to
the last frame in the window (short local propagation, not a from-scratch
detection), read that frame's masks, close the session.

This script benchmarks exactly that cycle — session open -> N prompts ->
forward propagate -> read -> close — against synthetic video windows at
several sizes, to pick a `--sam3-window-frames` / `--seg-interval-steps`
default before wiring it into the live rollout loop.

Usage:
  conda run -n sam3 python -m sam_vla.perception.bench_sam3_window
  conda run -n sam3 python -m sam_vla.perception.bench_sam3_window \\
      --window-frames 5 10 20 30 --trials 5 --vocab-terms "small rock,big rock"
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

# sam_vla/env/habitat_env.py:19-20 — kept in sync manually, not imported
# (see module docstring for why).
RGB_HEIGHT = 480
RGB_WIDTH = 640

# sam_vla/run_navdp_rollout.py:54 — the live rollout's control-loop period
# ("run segmentation every second" in next.md == every 1/DEFAULT_DT steps).
DEFAULT_DT = 0.1


def synthesize_window(
    out_dir: str,
    n_frames: int,
    width: int,
    height: int,
    num_small: int,
    num_big: int,
    small_radius: int,
    big_radius: int,
    speed: int,
):
    """Write an n_frames folder of moving circles standing in for a live
    RGB ring buffer: `num_small` small circles + `num_big` big circles,
    mirroring the plan's two-term goal vocabulary (small rock / big rock)."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    objects = [small_radius] * num_small + [big_radius] * num_big
    colors = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in objects]
    positions = []
    velocities = []
    for radius in objects:
        px = float(np.random.randint(radius, max(width - radius, radius + 1)))
        py = float(np.random.randint(radius, max(height - radius, radius + 1)))
        vx = np.random.choice([-1, 1]) * speed
        vy = np.random.choice([-1, 1]) * speed
        positions.append([px, py])
        velocities.append([vx, vy])

    for i in range(n_frames):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for idx, radius in enumerate(objects):
            x, y = positions[idx]
            rx, ry = round(x), round(y)
            draw.ellipse(
                [(rx - radius, ry - radius), (rx + radius, ry + radius)],
                fill=colors[idx],
            )
            vx, vy = velocities[idx]
            x += vx
            y += vy
            positions[idx] = [
                float(np.clip(x, radius, width - radius)),
                float(np.clip(y, radius, height - radius)),
            ]
            if x - radius < 0 or x + radius > width:
                vx *= -1
            if y - radius < 0 or y + radius > height:
                vy *= -1
            velocities[idx] = [vx, vy]
        img.save(os.path.join(out_dir, f"{i:03d}.jpg"))


def build_predictor(
    version: str,
    checkpoint_path: str | None,
    compile_: bool,
    num_objects: int,
    use_fa3: bool,
):
    from sam3 import build_sam3_predictor

    build_kwargs = dict(
        version=version, compile=compile_, async_loading_frames=False, use_fa3=use_fa3
    )
    if checkpoint_path:
        build_kwargs["checkpoint_path"] = checkpoint_path
    if version == "sam3.1":
        build_kwargs["warm_up"] = compile_
        build_kwargs["max_num_objects"] = num_objects
    return build_sam3_predictor(**build_kwargs)


def run_one_window(
    predictor, window_dir: str, vocab_terms: list[str], output_prob_thresh: float
):
    """One batched-re-window cycle. Returns (total_latency_s, breakdown_dict, num_masks_last_frame)."""
    t_open0 = time.perf_counter()
    resp = predictor.handle_request(
        {"type": "start_session", "resource_path": window_dir}
    )
    session_id = resp["session_id"]
    t_open1 = time.perf_counter()

    for term in vocab_terms:
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": term,
                "output_prob_thresh": output_prob_thresh,
            }
        )
    t_prompt1 = time.perf_counter()

    last_outputs = None
    last_frame_idx = -1
    for step in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
        }
    ):
        if step["frame_index"] >= last_frame_idx:
            last_frame_idx = step["frame_index"]
            last_outputs = step["outputs"]
    torch.cuda.synchronize()
    t_propagate1 = time.perf_counter()

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    t_close1 = time.perf_counter()

    num_masks = 0
    if last_outputs is not None:
        num_masks = len(last_outputs.get("out_obj_ids", []))

    breakdown = {
        "open_s": t_open1 - t_open0,
        "prompt_s": t_prompt1 - t_open1,
        "propagate_s": t_propagate1 - t_prompt1,
        "close_s": t_close1 - t_propagate1,
    }
    return t_close1 - t_open0, breakdown, num_masks


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 batched re-window spike/benchmark"
    )
    parser.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3.1")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--window-frames", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument(
        "--trials", type=int, default=5, help="timed trials per window size"
    )
    parser.add_argument("--warmup-trials", type=int, default=2)
    parser.add_argument("--width", type=int, default=RGB_WIDTH)
    parser.add_argument("--height", type=int, default=RGB_HEIGHT)
    parser.add_argument(
        "--num-small", type=int, default=2, help="small-rock stand-in circles"
    )
    parser.add_argument(
        "--num-big", type=int, default=2, help="big-rock stand-in circles"
    )
    parser.add_argument("--small-radius", type=int, default=15)
    parser.add_argument("--big-radius", type=int, default=45)
    parser.add_argument("--speed", type=int, default=15)
    parser.add_argument(
        "--vocab-terms",
        type=str,
        default="small rock,big rock",
        help="comma-separated text prompts, one add_prompt call per term",
    )
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention-3 (Hopper-only; off by default since it needs "
        "the flash_attn_interface package, which isn't installed/buildable on "
        "non-Hopper GPUs like Blackwell)",
    )
    parser.add_argument(
        "--scratch-dir", type=str, default="/tmp/segment-anything-3/bench_window"
    )
    parser.add_argument(
        "--assumed-dt",
        type=float,
        default=DEFAULT_DT,
        help="live rollout's control-loop period in seconds, for translating "
        "measured latency into a minimum --seg-interval-steps",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()

    vocab_terms = [t.strip() for t in args.vocab_terms.split(",") if t.strip()]
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    print(f"Building {args.version} predictor...")
    predictor = build_predictor(
        version=args.version,
        checkpoint_path=args.checkpoint,
        compile_=args.compile,
        num_objects=max(len(vocab_terms), args.num_small + args.num_big),
        use_fa3=args.use_fa3,
    )

    results = []
    for n_frames in args.window_frames:
        window_dir = os.path.join(args.scratch_dir, f"w{n_frames}")
        synthesize_window(
            window_dir,
            n_frames=n_frames,
            width=args.width,
            height=args.height,
            num_small=args.num_small,
            num_big=args.num_big,
            small_radius=args.small_radius,
            big_radius=args.big_radius,
            speed=args.speed,
        )

        for _ in range(args.warmup_trials):
            run_one_window(predictor, window_dir, vocab_terms, args.output_prob_thresh)

        latencies = []
        breakdown_sum = {
            "open_s": 0.0,
            "prompt_s": 0.0,
            "propagate_s": 0.0,
            "close_s": 0.0,
        }
        last_num_masks = 0
        for _ in range(args.trials):
            latency, breakdown, last_num_masks = run_one_window(
                predictor, window_dir, vocab_terms, args.output_prob_thresh
            )
            latencies.append(latency)
            for k in breakdown_sum:
                breakdown_sum[k] += breakdown[k]

        n = len(latencies)
        mean_latency = sum(latencies) / n
        min_seg_interval_steps = max(1, int(np.ceil(mean_latency / args.assumed_dt)))
        results.append(
            {
                "window_frames": n_frames,
                "mean_latency_s": mean_latency,
                "min_latency_s": min(latencies),
                "max_latency_s": max(latencies),
                "mean_breakdown_s": {k: v / n for k, v in breakdown_sum.items()},
                "last_num_masks": last_num_masks,
                "min_seg_interval_steps": min_seg_interval_steps,
            }
        )

        if not args.keep_scratch:
            shutil.rmtree(window_dir, ignore_errors=True)

    print(
        f"\n{'window':>8}  {'mean_s':>8}  {'min_s':>8}  {'max_s':>8}  "
        f"{'open':>7}  {'prompt':>7}  {'prop':>7}  {'close':>7}  "
        f"{'masks':>6}  {'min_seg_interval_steps@dt=' + str(args.assumed_dt):>28}"
    )
    for r in results:
        b = r["mean_breakdown_s"]
        print(
            f"{r['window_frames']:>8}  {r['mean_latency_s']:>8.3f}  {r['min_latency_s']:>8.3f}  "
            f"{r['max_latency_s']:>8.3f}  {b['open_s']:>7.3f}  {b['prompt_s']:>7.3f}  "
            f"{b['propagate_s']:>7.3f}  {b['close_s']:>7.3f}  {r['last_num_masks']:>6}  "
            f"{r['min_seg_interval_steps']:>28}"
        )


if __name__ == "__main__":
    main()
