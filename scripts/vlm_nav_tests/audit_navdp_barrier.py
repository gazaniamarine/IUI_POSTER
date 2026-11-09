#!/usr/bin/env python3
"""Audit executed NavDP rollouts against a discrete geometric barrier condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_run(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("run must be LABEL=/path/to/rollout.npz")
    label, path = specification.split("=", 1)
    return label, Path(path).expanduser().resolve()


def audit(path: Path, barrier_rate: float, tolerance: float) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "executed_center_clearance" not in data:
            raise ValueError(f"missing executed_center_clearance: {path}")
        center = np.asarray(data["executed_center_clearance"], dtype=np.float64).reshape(-1)
        radius = float(np.asarray(data.get("robot_radius", 0.24)))
        collisions = np.asarray(
            data.get("geometric_collision", center <= radius), dtype=bool
        ).reshape(-1)
        finite = np.isfinite(center)
        center = center[finite]
        collisions = collisions[finite]
        if center.size == 0:
            raise ValueError(f"no finite executed clearances: {path}")

        barrier = np.square(center) - radius**2
        if barrier.size >= 2:
            residual = (1.0 - barrier_rate) * barrier[:-1] - barrier[1:]
            violations = residual > tolerance
        else:
            residual = np.zeros(0, dtype=np.float64)
            violations = np.zeros(0, dtype=bool)
        surface = np.maximum(center - radius, 0.0)
        return {
            "frames": int(center.size),
            "transitions": int(residual.size),
            "collision": bool(collisions.any()),
            "collision_frames": int(collisions.sum()),
            "minimum_center_clearance_m": float(center.min()),
            "minimum_surface_clearance_m": float(surface.min()),
            "minimum_barrier_value": float(barrier.min()),
            "barrier_transition_satisfaction_rate": (
                float((~violations).mean()) if violations.size else 1.0
            ),
            "barrier_violation_count": int(violations.sum()),
            "maximum_barrier_residual": (
                float(np.maximum(residual, 0.0).max()) if residual.size else 0.0
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--barrier-rate", type=float, default=0.15)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 < args.barrier_rate <= 1.0:
        raise ValueError("barrier-rate must be in (0,1]")

    episodes = []
    for label, path in args.run:
        episodes.append(
            {"label": label, "path": str(path), **audit(path, args.barrier_rate, args.tolerance)}
        )
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        by_label.setdefault(str(row["label"]), []).append(row)
    summary = []
    for label, rows in by_label.items():
        total_transitions = sum(int(row["transitions"]) for row in rows)
        total_violations = sum(int(row["barrier_violation_count"]) for row in rows)
        summary.append(
            {
                "label": label,
                "episodes": len(rows),
                "collision_rate": float(np.mean([row["collision"] for row in rows])),
                "minimum_surface_clearance_m": float(
                    min(row["minimum_surface_clearance_m"] for row in rows)
                ),
                "barrier_transition_satisfaction_rate": (
                    1.0 - total_violations / total_transitions
                    if total_transitions
                    else 1.0
                ),
                "barrier_violations": total_violations,
                "transitions": total_transitions,
            }
        )

    payload = {
        "scope": "empirical executed-transition audit; this is not a formal safety proof",
        "barrier": "h_t = center_clearance_t^2 - robot_radius^2",
        "condition": "h_{t+1} >= (1-alpha) h_t",
        "barrier_rate": args.barrier_rate,
        "tolerance": args.tolerance,
        "episodes": episodes,
        "summary": summary,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for row in summary:
        print(
            f"{row['label']}: collision_rate={row['collision_rate']:.4f} "
            f"min_surface={row['minimum_surface_clearance_m']:.4f}m "
            f"barrier_satisfaction={row['barrier_transition_satisfaction_rate']:.4f} "
            f"({row['barrier_violations']}/{row['transitions']} violations)"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()