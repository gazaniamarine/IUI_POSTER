"""Study 1 (next.md) Phase 4: analysis. Reads the manifest.json/rollout.npz
that each sam_vla.study1_paired_runs pair left behind and computes, per
episode/condition, the four Phase 0 timing buckets plus success -- then
performs the timing-subtraction comparison next.md's "Timing decomposition"
section describes:

    total_episode_time - sum(human_decision_ms) - model_load_ms
        == Condition A's VLM-and-driving-only time,
        directly comparable to Condition B's total time.

`total_episode_time` here is measured from the first logged step's timestamp
to the last (i.e. the step loop's own wall time), not manifest["start_time"]
(which is set before scene/policy loading -- that setup cost is shared by
both conditions and isn't one of the four buckets the ablation is about, so
folding it in would just add a common-mode offset to both sides without
changing the comparison).

`success` is not logged anywhere today (the single-goal rollout loop has no
goal-reached break, see next.md) -- it's derived here as "distance_to_goal
came within --success-radius at some point in the episode", read from
rollout.npz's distances_to_goal array.

Usage:
    python -m sam_vla.study1_analysis --pairs-manifest study1_run1/pairs_manifest.json \
        [--success-radius 1.0] [--out-csv study1_run1/analysis.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_episode_metrics(run_dir: Path, success_radius: float) -> dict:
    """Reads one run_navdp_rollout out_dir (one leg of one pair) and returns
    its Phase 0-4 metrics. Raises FileNotFoundError if manifest.json/
    rollout.npz are missing (e.g. the run failed before flush())."""
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

    n_triggers = 0
    sum_vlm_inference_ms = 0.0
    sum_human_decision_ms = 0.0
    sum_drive_ms = 0.0
    sum_model_load_ms = 0.0
    for step in steps:
        event = step.get("uncertainty_event")
        if event is None:
            continue
        n_triggers += 1
        sum_vlm_inference_ms += event.get("vlm_inference_ms") or 0.0
        sum_human_decision_ms += event.get("human_decision_ms") or 0.0
        sum_drive_ms += event.get("drive_ms") or 0.0
        sum_model_load_ms += event.get("model_load_ms") or 0.0

    return {
        "n_steps": n_steps,
        "total_episode_time_s": total_episode_time_s,
        "success": success,
        "n_uncertainty_triggers": n_triggers,
        "sum_vlm_inference_ms": sum_vlm_inference_ms,
        "sum_human_decision_ms": sum_human_decision_ms,
        "sum_drive_ms": sum_drive_ms,
        "model_load_ms": sum_model_load_ms,
    }


def analyze_pairs(pairs_manifest: dict, success_radius: float) -> List[dict]:
    """One row per (episode, condition) that actually completed (returncode
    0 and its out_dir has a flushed manifest/npz) -- rows for failed or
    not-yet-run legs are skipped with a printed warning rather than raising,
    since a paired-run batch is often analyzed while still in flight or
    after a partial failure."""
    rows = []
    for ep in pairs_manifest["episodes"]:
        for condition, run_record in ep["runs"].items():
            run_dir = Path(run_record["out_dir"])
            if run_record.get("returncode") not in (0, None):
                print(
                    f"[analysis] skipping {ep['id']}/{condition}: "
                    f"returncode={run_record.get('returncode')}"
                )
                continue
            try:
                metrics = load_episode_metrics(run_dir, success_radius)
            except FileNotFoundError:
                print(f"[analysis] skipping {ep['id']}/{condition}: no manifest/npz at {run_dir}")
                continue

            row = {"episode_id": ep["id"], "condition": condition, **metrics}
            if condition == "human":
                row["vlm_and_driving_only_time_s"] = (
                    metrics["total_episode_time_s"]
                    - metrics["sum_human_decision_ms"] / 1000.0
                    - metrics["model_load_ms"] / 1000.0
                )
            else:
                row["vlm_and_driving_only_time_s"] = None
            rows.append(row)
    return rows


CSV_FIELDS = [
    "episode_id",
    "condition",
    "n_steps",
    "total_episode_time_s",
    "success",
    "n_uncertainty_triggers",
    "sum_vlm_inference_ms",
    "sum_human_decision_ms",
    "sum_drive_ms",
    "model_load_ms",
    "vlm_and_driving_only_time_s",
]


def write_csv(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def summarize(rows: List[dict]) -> dict:
    """Condition A's derived (VLM-and-driving-only) time vs Condition B's
    raw total time, and success rate A vs B -- the actual ablation result
    per next.md's Phase 4 description. Only counts episodes with BOTH
    conditions present, so the comparison stays paired."""
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], {})[row["condition"]] = row

    paired = {
        ep_id: legs
        for ep_id, legs in by_episode.items()
        if "human" in legs and "autonomous" in legs
    }

    a_times = [legs["human"]["vlm_and_driving_only_time_s"] for legs in paired.values()]
    b_times = [legs["autonomous"]["total_episode_time_s"] for legs in paired.values()]
    a_success = [legs["human"]["success"] for legs in paired.values()]
    b_success = [legs["autonomous"]["success"] for legs in paired.values()]

    n = len(paired)
    return {
        "n_paired_episodes": n,
        "condition_a_mean_derived_time_s": float(np.mean(a_times)) if n else None,
        "condition_b_mean_raw_time_s": float(np.mean(b_times)) if n else None,
        "condition_a_success_rate": float(np.mean(a_success)) if n else None,
        "condition_b_success_rate": float(np.mean(b_success)) if n else None,
        "a_faster_than_b_count": sum(a < b for a, b in zip(a_times, b_times)),
    }


def print_summary(summary: dict) -> None:
    print("[analysis] paired-episode comparison (Condition A derived vs Condition B raw):")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pairs-manifest", required=True, help="pairs_manifest.json from sam_vla.study1_paired_runs")
    ap.add_argument("--success-radius", type=float, default=1.0, help="distance (m) counted as goal reached")
    ap.add_argument("--out-csv", default=None, help="default: <pairs_manifest's dir>/analysis.csv")
    args = ap.parse_args()

    manifest_path = Path(args.pairs_manifest)
    pairs_manifest = json.loads(manifest_path.read_text())

    rows = analyze_pairs(pairs_manifest, args.success_radius)
    out_csv = Path(args.out_csv) if args.out_csv else manifest_path.parent / "analysis.csv"
    write_csv(rows, out_csv)
    print(f"[analysis] wrote {len(rows)} rows -> {out_csv}")

    print_summary(summarize(rows))
