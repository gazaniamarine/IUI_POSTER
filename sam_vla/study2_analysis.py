"""Study 2 (next.md) analysis: reads the manifest.json/rollout.npz that each
sam_vla.study2_noise_sweep run left behind and computes, per (episode,
noise_level) run, task-performance metrics (success, steps/time to goal, CBF-
trigger rate) plus the belief tracker's own uncertainty_value() trajectory
(mean/max), then aggregates by noise_level -- the actual "does the offline
numpy approximation hold up in the real sim" comparison next.md's Study 2
section asks for.

`success` is derived the same way study1_analysis.py does it (single-goal
loop has no goal-reached break): distance_to_goal came within
--success-radius at some point, read from rollout.npz's distances_to_goal.

CBF-trigger rate is the fraction of steps with vla_result["blocked"] True
(CbfObstacleAvoidance.apply's per-step "blocked" flag, merged into vla_result
via cbf_info -- see run_navdp_rollout.py). Only meaningful for runs actually
started with --cbf.

uncertainty trajectory is read from vla_result["uncertainty_value"], logged
every step by run_navdp_rollout.py's single-goal path (added alongside Study
2's other wiring -- absent on older runs/multi-goal or base-station runs,
in which case mean/max are None).

Usage:
    python -m sam_vla.study2_analysis --sweep-manifest study2_run1/sweep_manifest.json \\
        [--success-radius 1.0] [--out-csv study2_run1/analysis.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_run_metrics(run_dir: Path, success_radius: float) -> dict:
    """Reads one run_navdp_rollout out_dir (one leg of one sweep point) and
    returns its task-performance + belief-uncertainty metrics. Raises
    FileNotFoundError if manifest.json/rollout.npz are missing (e.g. the run
    failed before flush())."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    npz = np.load(run_dir / "rollout.npz")

    steps = manifest["steps"]
    n_steps = len(steps)

    if n_steps >= 2:
        t0 = _parse_iso(steps[0]["timestamp"])
        t1 = _parse_iso(steps[-1]["timestamp"])
        total_episode_time_s = (t1 - t0).total_seconds()
    else:
        total_episode_time_s = 0.0

    distances = npz["distances_to_goal"]
    finite = distances[np.isfinite(distances)]
    success = bool(finite.size > 0 and np.min(finite) <= success_radius)
    steps_to_goal = (
        int(np.argmax(distances <= success_radius)) if success else None
    )
    final_distance_to_goal = float(distances[-1]) if distances.size else None

    n_blocked = 0
    n_hard_gate = 0
    uncertainty_values = []
    for step in steps:
        vla_result = step.get("vla_result") or {}
        if vla_result.get("blocked"):
            n_blocked += 1
        if vla_result.get("hard_gate_fired"):
            n_hard_gate += 1
        uv = vla_result.get("uncertainty_value")
        if uv is not None:
            uncertainty_values.append(uv)

    cbf_trigger_rate = (n_blocked / n_steps) if n_steps else None

    return {
        "n_steps": n_steps,
        "total_episode_time_s": total_episode_time_s,
        "success": success,
        "steps_to_goal": steps_to_goal,
        "final_distance_to_goal": final_distance_to_goal,
        "cbf_blocked_steps": n_blocked,
        "cbf_hard_gate_steps": n_hard_gate,
        "cbf_trigger_rate": cbf_trigger_rate,
        "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else None,
        "max_uncertainty": float(np.max(uncertainty_values)) if uncertainty_values else None,
    }


def analyze_sweep(sweep_manifest: dict, success_radius: float) -> List[dict]:
    """One row per (episode, noise_level) run that actually completed
    (returncode 0 and its out_dir has a flushed manifest/npz) -- rows for
    failed or not-yet-run legs are skipped with a printed warning rather than
    raising, since a sweep is often analyzed while still in flight or after a
    partial failure."""
    rows = []
    for ep in sweep_manifest["episodes"]:
        for noise_level_str, run_record in ep["runs"].items():
            run_dir = Path(run_record["out_dir"])
            if run_record.get("returncode") not in (0, None):
                print(
                    f"[analysis] skipping {ep['id']}/noise={noise_level_str}: "
                    f"returncode={run_record.get('returncode')}"
                )
                continue
            try:
                metrics = load_run_metrics(run_dir, success_radius)
            except FileNotFoundError:
                print(
                    f"[analysis] skipping {ep['id']}/noise={noise_level_str}: "
                    f"no manifest/npz at {run_dir}"
                )
                continue

            rows.append(
                {
                    "episode_id": ep["id"],
                    "noise_level": float(noise_level_str),
                    "seed": ep.get("seed"),
                    **metrics,
                }
            )
    return rows


CSV_FIELDS = [
    "episode_id",
    "noise_level",
    "seed",
    "n_steps",
    "total_episode_time_s",
    "success",
    "steps_to_goal",
    "final_distance_to_goal",
    "cbf_blocked_steps",
    "cbf_hard_gate_steps",
    "cbf_trigger_rate",
    "mean_uncertainty",
    "max_uncertainty",
]


def write_csv(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def summarize_by_noise_level(rows: List[dict]) -> List[dict]:
    """One row per swept noise_level: mean success rate, mean steps/time to
    goal (successes only), mean CBF-trigger rate, mean of each run's mean
    uncertainty -- the actual noise-vs-performance/calibration curve this
    study is after. Sorted by noise_level ascending."""
    by_level: dict = {}
    for row in rows:
        by_level.setdefault(row["noise_level"], []).append(row)

    summary = []
    for noise_level in sorted(by_level):
        group = by_level[noise_level]
        n = len(group)
        successes = [r for r in group if r["success"]]
        steps_to_goal = [r["steps_to_goal"] for r in successes if r["steps_to_goal"] is not None]
        cbf_rates = [r["cbf_trigger_rate"] for r in group if r["cbf_trigger_rate"] is not None]
        mean_uncs = [r["mean_uncertainty"] for r in group if r["mean_uncertainty"] is not None]
        max_uncs = [r["max_uncertainty"] for r in group if r["max_uncertainty"] is not None]
        summary.append(
            {
                "noise_level": noise_level,
                "n_runs": n,
                "success_rate": float(np.mean([r["success"] for r in group])) if n else None,
                "mean_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else None,
                "mean_cbf_trigger_rate": float(np.mean(cbf_rates)) if cbf_rates else None,
                "mean_uncertainty": float(np.mean(mean_uncs)) if mean_uncs else None,
                "mean_max_uncertainty": float(np.mean(max_uncs)) if max_uncs else None,
            }
        )
    return summary


def print_summary(summary: List[dict]) -> None:
    print("[analysis] performance/calibration vs. noise_level:")
    for row in summary:
        print(f"  {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep-manifest", required=True, help="sweep_manifest.json from sam_vla.study2_noise_sweep")
    ap.add_argument("--success-radius", type=float, default=1.0, help="distance (m) counted as goal reached")
    ap.add_argument("--out-csv", default=None, help="default: <sweep_manifest's dir>/analysis.csv")
    args = ap.parse_args()

    manifest_path = Path(args.sweep_manifest)
    sweep_manifest = json.loads(manifest_path.read_text())

    rows = analyze_sweep(sweep_manifest, args.success_radius)
    out_csv = Path(args.out_csv) if args.out_csv else manifest_path.parent / "analysis.csv"
    write_csv(rows, out_csv)
    print(f"[analysis] wrote {len(rows)} rows -> {out_csv}")

    print_summary(summarize_by_noise_level(rows))
