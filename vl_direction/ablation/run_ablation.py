"""
Ablation driver: for each ModelSpec in model_specs.py, spawns that model's
server (InternVLServerManager or QwenAblationServerManager), replays the
frozen benchmark frame set from capture_benchmark_frames.py through the
*real* vl_direction.directive_engine.query() call (mirroring kb_teleop_vl.
py's cbf/exploration/uncertainty mode selection exactly), and records
latency, parse success, and GPU memory footprint. One model is ever loaded
at a time -- each is fully torn down before the next is spawned -- so this
does not require enough GPU memory to hold every candidate simultaneously.

Uses dedicated ports (not vl_direction/config.py's INTERNVL_SERVER_PORT)
so this never collides with a live kb_teleop_vl.py session that happens to
already have a server running.

Run in the "habitat" conda env (needs kb_teleop_vl's obstacle/projection
helpers, which import habitat_sim transitively), from the repo root:
    conda activate habitat && python -m vl_direction.ablation.run_ablation [--reps N]
"""

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import kb_teleop_vl as kbvl
from vl_direction import config as vl_dir_config
from vl_direction.client import InternVLSocketClient
from vl_direction.directive_engine import query as vl_query
from vl_direction.internvl_server_manager import InternVLServerManager
from vl_direction.ablation.qwen_ablation_server_manager import QwenAblationServerManager
from vl_direction.ablation.model_specs import MODEL_SPECS
from vl_direction.schemas import CBFContext, ExplorationContext, UncertaintyContext

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE / "benchmark_frames"
RESULTS_DIR = HERE / "results"

ABLATION_INTERNVL_PORT = 8790
ABLATION_QWEN_PORT = 8791

WARMUP_REPS = 1
DEFAULT_MEASURED_REPS = 3
GPU_TEARDOWN_POLL_S = 2.0
GPU_TEARDOWN_MAX_WAIT_S = 30.0
UNCERTAINTY_COVARIANCE_VALUE = (
    2.5  # arbitrary "well past threshold" value for the request-phase call
)

EPISODE_ID = f"vl-ablation-{int(time.time())}"


def _load_benchmark_set():
    manifest_path = BENCHMARK_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run capture_benchmark_frames.py first "
            "(in the habitat conda env)"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    scenarios = []
    for entry in manifest:
        frame = np.array(Image.open(BENCHMARK_DIR / entry["frame_file"]).convert("RGB"))
        scenarios.append((entry, frame))
    return scenarios


def _build_query_args(entry):
    """Mirrors kb_teleop_vl.VLTeleopApp._dispatch_vl_query's mode/context
    selection exactly, reconstructed from the manifest's saved obstacle
    geometry (frames are pre-rendered, not live, so this can't call the
    original method directly)."""
    if entry["label"].startswith("uncertainty"):
        context = UncertaintyContext(
            covariance_value=UNCERTAINTY_COVARIANCE_VALUE,
            threshold_used=vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD,
        )
        return "uncertainty", context, ""

    nearest_any_edge = entry["nearest_any_edge_distance_m"]
    nearest_visible_bbox = entry["nearest_visible_bbox_xyxy"]

    mode = (
        "cbf"
        if nearest_any_edge is not None
        and nearest_any_edge <= kbvl.CBF_DISTANCE_THRESHOLD_M
        else "exploration"
    )
    fallback_note = ""

    if mode == "cbf" and nearest_visible_bbox is not None:
        context = CBFContext(
            bbox_xyxy=tuple(nearest_visible_bbox), frame_wh=tuple(entry["frame_wh"])
        )
    else:
        if mode == "cbf":
            fallback_note = " (nearest obstacle not in view, fell back to exploration)"
        mode = "exploration"
        hint = (
            f"nearest obstacle is {nearest_any_edge:.1f}m away"
            if nearest_any_edge is not None
            else None
        )
        context = ExplorationContext(
            task_str=kbvl.EXPLORATION_TASK_STR, vague_hint=hint
        )

    return mode, context, fallback_note


def _gpu_memory_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sum(int(line.strip()) for line in out.splitlines() if line.strip())


def _wait_for_gpu_teardown(baseline_mib: int) -> int:
    deadline = time.time() + GPU_TEARDOWN_MAX_WAIT_S
    last = _gpu_memory_used_mib()
    while time.time() < deadline:
        last = _gpu_memory_used_mib()
        if last <= baseline_mib + 512:  # small slack for driver bookkeeping
            return last
        time.sleep(GPU_TEARDOWN_POLL_S)
    print(
        f"[run_ablation] warning: GPU memory still {last}MiB vs baseline {baseline_mib}MiB after teardown wait"
    )
    return last


def _make_manager(spec):
    if spec.kind == "internvl":
        return InternVLServerManager(
            port=ABLATION_INTERNVL_PORT,
            model_path=spec.model_path,
            startup_timeout=spec.startup_timeout_s,
        )
    if spec.kind == "qwen":
        return QwenAblationServerManager(
            port=ABLATION_QWEN_PORT,
            model_path=spec.model_path,
            startup_timeout=spec.startup_timeout_s,
        )
    raise ValueError(f"unknown ModelSpec.kind {spec.kind!r}")


def _run_one_model(spec, scenarios, measured_reps, raw_rows):
    print(f"\n=== {spec.name} ({spec.model_path}) ===")
    baseline_mib = _gpu_memory_used_mib()
    manager = _make_manager(spec)

    result_meta = None
    try:
        t0 = time.monotonic()
        try:
            manager.start()
        except Exception as e:
            print(f"[run_ablation] {spec.name}: FAILED to start -- {e}")
            raw_rows.append(
                {
                    "model": spec.name,
                    "scenario": "",
                    "mode": "",
                    "rep": -1,
                    "latency_ms": "",
                    "parse_ok": "",
                    "direction": "",
                    "raw_response": f"START FAILED: {e}",
                }
            )
            return None
        load_time_s = time.monotonic() - t0
        post_load_mib = _gpu_memory_used_mib()
        vram_delta_mib = post_load_mib - baseline_mib
        print(
            f"[run_ablation] {spec.name}: loaded in {load_time_s:.1f}s, +{vram_delta_mib}MiB GPU"
        )

        client = InternVLSocketClient(host="127.0.0.1", port=manager.port)

        for entry, frame in scenarios:
            mode, context, fallback_note = _build_query_args(entry)
            for rep in range(WARMUP_REPS + measured_reps):
                is_warmup = rep < WARMUP_REPS
                try:
                    result = vl_query(mode, [frame], context, EPISODE_ID, client=client)
                    row = {
                        "model": spec.name,
                        "scenario": entry["label"],
                        "mode": mode,
                        "rep": -1 if is_warmup else rep - WARMUP_REPS,
                        "latency_ms": result.latency_ms,
                        "parse_ok": result.parse_ok,
                        "direction": (
                            result.direction.value
                            if result.direction is not None
                            else ""
                        ),
                        "raw_response": result.raw_response + fallback_note,
                    }
                except Exception as e:
                    row = {
                        "model": spec.name,
                        "scenario": entry["label"],
                        "mode": mode,
                        "rep": -1 if is_warmup else rep - WARMUP_REPS,
                        "latency_ms": "",
                        "parse_ok": False,
                        "direction": "",
                        "raw_response": f"ERROR: {e}",
                    }
                if not is_warmup:
                    raw_rows.append(row)
                print(
                    f"  [{entry['label']}{'(warmup)' if is_warmup else ''}] {mode} -> "
                    f"{row['direction'] or 'NONE'} -> {row['latency_ms']} -> {row['raw_response']!r}"
                )

        result_meta = {
            "name": spec.name,
            "load_time_s": load_time_s,
            "vram_delta_mib": vram_delta_mib,
        }
    finally:
        # Always torn down, even if manager.start() itself raised (e.g. a
        # timed-out download/load) -- start() sets _owns_process=True right
        # after spawning, before the health-check loop, so the subprocess is
        # still reachable here and would otherwise leak as an orphaned,
        # GPU-memory-holding process with no driver left to stop it.
        manager.stop()
        _wait_for_gpu_teardown(baseline_mib)

    return result_meta


def _summarize(raw_rows, model_meta):
    by_model = {}
    for row in raw_rows:
        by_model.setdefault(row["model"], []).append(row)

    summary = []
    for model_name, rows in by_model.items():
        latencies = [
            r["latency_ms"] for r in rows if isinstance(r["latency_ms"], (int, float))
        ]
        parse_oks = [r["parse_ok"] for r in rows if r["parse_ok"] in (True, False)]
        meta = model_meta.get(model_name, {})
        summary.append(
            {
                "model": model_name,
                "n_calls": len(rows),
                "n_ok": sum(1 for p in parse_oks if p),
                "parse_success_rate": (
                    (sum(1 for p in parse_oks if p) / len(parse_oks))
                    if parse_oks
                    else 0.0
                ),
                "latency_mean_ms": statistics.mean(latencies) if latencies else None,
                "latency_p50_ms": statistics.median(latencies) if latencies else None,
                "latency_p95_ms": (
                    sorted(latencies)[int(0.95 * (len(latencies) - 1))]
                    if latencies
                    else None
                ),
                "load_time_s": meta.get("load_time_s"),
                "vram_delta_mib": meta.get("vram_delta_mib"),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_MEASURED_REPS,
        help="measured reps per scenario",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="subset of ModelSpec.name values to run (default: all)",
    )
    args = parser.parse_args()

    scenarios = _load_benchmark_set()
    print(f"[run_ablation] loaded {len(scenarios)} benchmark scenarios")

    specs = MODEL_SPECS
    if args.models:
        wanted = set(args.models)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise ValueError(
                f"unknown model name(s): {missing}; available: {[s.name for s in MODEL_SPECS]}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    raw_path = RESULTS_DIR / f"ablation_raw_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"ablation_summary_{timestamp}.csv"
    raw_fieldnames = [
        "model",
        "scenario",
        "mode",
        "rep",
        "latency_ms",
        "parse_ok",
        "direction",
        "raw_response",
    ]
    summary_fieldnames = [
        "model",
        "n_calls",
        "n_ok",
        "parse_success_rate",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "load_time_s",
        "vram_delta_mib",
    ]
    print(
        f"[run_ablation] writing incrementally to {raw_path} and {summary_path} as each model finishes"
    )

    raw_rows = []
    model_meta = {}

    for spec in specs:
        rows_before = len(raw_rows)
        meta = _run_one_model(spec, scenarios, args.reps, raw_rows)
        if meta is not None:
            model_meta[spec.name] = meta

        # Append just this model's new rows -- and rewrite the (small)
        # summary from everything accumulated so far -- right after each
        # model finishes, so a still-running sweep is visible on disk
        # instead of only appearing once every model has completed.
        write_header = not raw_path.exists()
        with open(raw_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(raw_rows[rows_before:])

        summary = _summarize(raw_rows, model_meta)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary)

        print(f"[run_ablation] {spec.name} done -- updated {summary_path.name}")
        print(
            f"{'model':<24} {'parse%':>7} {'mean_ms':>9} {'p50_ms':>8} {'p95_ms':>8} {'load_s':>7} {'vram_MiB':>9}"
        )
        for row in summary:
            print(
                f"{row['model']:<24} "
                f"{row['parse_success_rate'] * 100:>6.1f}% "
                f"{row['latency_mean_ms'] or 0:>9.1f} "
                f"{row['latency_p50_ms'] or 0:>8.1f} "
                f"{row['latency_p95_ms'] or 0:>8.1f} "
                f"{row['load_time_s'] or 0:>7.1f} "
                f"{row['vram_delta_mib'] or 0:>9}"
            )

    print(f"\n[run_ablation] sweep complete. Final results: {raw_path}, {summary_path}")


if __name__ == "__main__":
    main()
