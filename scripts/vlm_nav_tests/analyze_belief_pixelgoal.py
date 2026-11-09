#!/usr/bin/env python3
"""Summarize live/ghost PixelGoal belief behavior from a rollout archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Run directory or rollout.npz")
    args = parser.parse_args()
    supplied = Path(args.run).expanduser().resolve()
    archive = supplied if supplied.suffix == ".npz" else supplied / "rollout.npz"
    data = np.load(archive, allow_pickle=False)

    required = {
        "belief_goal_mu",
        "belief_goal_covariance",
        "belief_goal_pixel",
        "belief_goal_visible",
        "belief_goal_source",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"rollout lacks belief fields: {missing}")

    visible = np.asarray(data["belief_goal_visible"], dtype=bool)
    sources = np.asarray(data["belief_goal_source"]).astype(str)
    covariance = np.asarray(data["belief_goal_covariance"], dtype=np.float64)
    traces = np.trace(covariance, axis1=-2, axis2=-1)
    ghost = sources == "GHOST"
    bootstrapped = sources == "WORLD_BOOTSTRAP"

    print("Belief PixelGoal summary")
    print(f"  steps                 : {len(sources)}")
    print(f"  live-mask steps       : {int(visible.sum())}")
    print(f"  ghost-belief steps    : {int(ghost.sum())}")
    print(f"  world-bootstrap steps : {int(bootstrapped.sum())}")
    print(f"  maximum covariance tr.: {float(np.nanmax(traces)):.6f}")
    print(f"  final covariance tr.  : {float(traces[-1]):.6f}")
    print(f"  final mean [fwd,left] : {data['belief_goal_mu'][-1].tolist()}")
    print(f"  final PixelGoal [u,v] : {data['belief_goal_pixel'][-1].tolist()}")
    print(f"  collision frames      : {int(np.asarray(data['geometric_collision']).sum())}")

    if not np.any(visible):
        print("  NOTE: no live goal sighting occurred; this run used only its bootstrap/prior.")
    if not np.any(ghost):
        print("  NOTE: no post-sighting occlusion occurred, so ghost tracking was not exercised.")


if __name__ == "__main__":
    main()
