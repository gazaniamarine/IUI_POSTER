"""Tiny standalone benchmark: average per-call inference latency for the
three models the nav pipeline actually calls per-frame -- SAM2 (segment_frame),
GroundingDINO (detect_in_frame), and Qwen VLM (ground_object_verbose).

Loads a handful of real captured frames from an existing output/ run (no sim
needed) and times each model's forward call in isolation, excluding one-time
model/weights load. Qwen is timed over a real qwen_server subprocess (spawned
via QwenServerManager, same as nav/rover_controller.py does), so its number
includes actual socket round-trip + generation, not just a mocked stub.

Run in the sam2 env (has torch + transformers for SAM2/GDINO; the Qwen client
side is plain socket/PIL/numpy so it rides along fine):

    conda run -n sam2 python scripts/vlm_nav_tests/model_timing_bench.py \\
        --frames-dir output/2026-08-01_141535_kb-teleop-goal_manual_obsna_seedna/rgb \\
        --n-frames 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAVDP_ROOT = ROOT / "navdp"
for p in (NAVDP_ROOT, NAVDP_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_frames(frames_dir: Path, n: int) -> list[np.ndarray]:
    paths = sorted(frames_dir.glob("*.png"))[:n]
    if not paths:
        raise FileNotFoundError(f"no .png frames found under {frames_dir}")
    frames = []
    for p in paths:
        bgr = cv2.imread(str(p))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def summarize(name: str, times_ms: list[float]) -> None:
    print(
        f"{name:>14}: mean={statistics.mean(times_ms):7.1f}ms  "
        f"median={statistics.median(times_ms):7.1f}ms  "
        f"min={min(times_ms):7.1f}ms  max={max(times_ms):7.1f}ms  n={len(times_ms)}"
    )


def bench_sam2(frames: list[np.ndarray]) -> list[float]:
    from sam_vla.perception import sam_segmenter

    sam_segmenter.segment_frame(frames[0])  # warm up: load weights, first-call cuda init
    times = []
    for rgb in frames:
        t0 = time.perf_counter()
        sam_segmenter.segment_frame(rgb)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_gdino(frames: list[np.ndarray]) -> list[float]:
    from sam_vla.goal_resolution import dino_grounding_resolver as dgr

    depth = np.full(frames[0].shape[:2], 5.0, dtype=np.float32)  # dummy: only depth's median is read
    queries = ["a rock"]
    dgr.detect_in_frame(frames[0], depth, queries, hfov_deg=90.0)  # warm up: load weights
    times = []
    for rgb in frames:
        t0 = time.perf_counter()
        dgr.detect_in_frame(rgb, depth, queries, hfov_deg=90.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_qwen(frames: list[np.ndarray]) -> list[float]:
    from sam_vla.vlm import qwen_client
    from sam_vla.vlm.qwen_server_manager import QwenServerManager

    manager = QwenServerManager()
    manager.start()
    print(f"[qwen] server load_ms={manager.load_ms} (0.0 means one was already running)")
    try:
        qwen_client.ground_object_verbose(frames[0], "a rock")  # warm up: first-call model load inside server
        times = []
        for rgb in frames:
            t0 = time.perf_counter()
            qwen_client.ground_object_verbose(rgb, "a rock")
            times.append((time.perf_counter() - t0) * 1000.0)
        return times
    finally:
        manager.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frames-dir",
        default=str(
            ROOT
            / "output"
            / "2026-08-01_141535_kb-teleop-goal_manual_obsna_seedna"
            / "rgb"
        ),
    )
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["sam2", "gdino", "qwen"],
        choices=["sam2", "gdino", "qwen"],
    )
    args = ap.parse_args()

    frames = load_frames(Path(args.frames_dir), args.n_frames)
    print(f"loaded {len(frames)} frames from {args.frames_dir}\n")

    results = {}
    if "sam2" in args.models:
        print("timing SAM2 (segment_frame)...")
        results["sam2"] = bench_sam2(frames)
    if "gdino" in args.models:
        print("timing GroundingDINO (detect_in_frame)...")
        results["gdino"] = bench_gdino(frames)
    if "qwen" in args.models:
        print("timing Qwen VLM (ground_object_verbose)...")
        results["qwen"] = bench_qwen(frames)

    print("\n=== average per-call inference latency ===")
    for name, times in results.items():
        summarize(name, times)


if __name__ == "__main__":
    main()
