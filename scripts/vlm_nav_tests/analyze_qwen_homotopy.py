#!/usr/bin/env python3
"""Audit Qwen homotopy validity, repeatability, and candidate conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Run directory, rollout.npz, or manifest.json")
    parser.add_argument("--minimum-consistency", type=float, default=1.0)
    args = parser.parse_args()

    supplied = Path(args.run).expanduser().resolve()
    run_directory = supplied if supplied.is_dir() else supplied.parent
    archive_path = supplied if supplied.suffix == ".npz" else run_directory / "rollout.npz"
    manifest_path = (
        supplied if supplied.name == "manifest.json" else run_directory / "manifest.json"
    )
    if not archive_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run must contain rollout.npz and manifest.json")

    data = np.load(archive_path, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = manifest.get("qwen_homotopy_events", [])

    start = np.asarray(data["start_position_xz"], dtype=np.float64)
    goal = np.asarray(data["goal_position"], dtype=np.float64)[[0, 2]]
    obstacles = np.asarray(data["obstacle_positions"], dtype=np.float64)
    obstacle = obstacles[0, [0, 2]] if len(obstacles) else np.full(2, np.nan)
    goal_vector = goal - start
    obstacle_vector = obstacle - start
    denominator = max(float(np.dot(goal_vector, goal_vector)), 1e-12)
    projection = float(np.dot(obstacle_vector, goal_vector) / denominator)
    cross_track = abs(
        goal_vector[0] * obstacle_vector[1]
        - goal_vector[1] * obstacle_vector[0]
    ) / max(float(np.linalg.norm(goal_vector)), 1e-12)
    collinear = bool(cross_track <= 1e-3 and 0.0 < projection < 1.0)

    print("\nGeometry")
    print(f"  start       : {start.tolist()}")
    print(f"  obstacle    : {obstacle.tolist()}")
    print(f"  goal        : {goal.tolist()}")
    print(f"  cross-track : {cross_track:.6f} m")
    print(f"  collinear   : {collinear}")

    print("\nQwen decisions")
    if not events:
        print("  ERROR: Qwen was never queried. Check metric depth/mask thresholds.")
    for index, event in enumerate(events):
        repeats = event.get("repeat_sides", [])
        consistency = float(event.get("consistency_rate", 0.0))
        print(
            f"  event {index}: step={event.get('step')} side={event.get('side')} "
            f"sign={event.get('circulation_sign'):+.0f} repeats={repeats} "
            f"consistency={consistency:.1%} fallback={event.get('used_fallback')}"
        )

    forced = np.asarray(data["qwen_homotopy_sign"], dtype=np.float32)
    candidates = np.asarray(data["candidate_circulation_signs"], dtype=np.float32)
    active = np.abs(forced) > 0.5
    candidate_match = bool(
        np.all(candidates[active] == forced[active, None]) if np.any(active) else False
    )
    valid_sides = all(event.get("side") in {"LEFT", "RIGHT"} for event in events)
    no_fallback = all(not bool(event.get("used_fallback")) for event in events)
    consistency_ok = bool(
        events
        and all(
            float(event.get("consistency_rate", 0.0)) >= args.minimum_consistency
            for event in events
        )
    )
    collision_free = not bool(np.any(data["geometric_collision"]))

    print("\nChecks")
    checks = {
        "straight-line layout": collinear,
        "Qwen returned LEFT/RIGHT": bool(events) and valid_sides,
        "identical-frame consistency": consistency_ok,
        "no deterministic fallback": bool(events) and no_fallback,
        "all candidate signs match Qwen": candidate_match,
        "executed rollout collision-free": collision_free,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    report = {
        "checks": checks,
        "events": events,
        "cross_track_distance_m": cross_track,
        "note": (
            "For a symmetric centred obstacle, LEFT and RIGHT are both valid. "
            "Correctness means a valid, consistently forced, collision-free side; "
            "there is no unique semantic ground-truth side."
        ),
    }
    report_path = run_directory / "qwen_homotopy_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {report_path}")


if __name__ == "__main__":
    main()
