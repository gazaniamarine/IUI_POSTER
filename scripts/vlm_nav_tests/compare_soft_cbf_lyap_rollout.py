#!/usr/bin/env python3
"""Compare one pure-NavDP rollout with one soft CBF+Lyapunov rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar(data: np.lib.npyio.NpzFile, key: str, default):
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.reshape(-1)[0].item() if value.size else default


def summarize(path: Path, *, verify_soft_only: bool) -> dict[str, float | bool | str]:
    with np.load(path, allow_pickle=False) as data:
        pose = np.asarray(data["pose"], dtype=np.float64)
        actions = np.asarray(data["action_3d"], dtype=np.float64)
        goal_distance = np.asarray(data["goal_distance"], dtype=np.float64)
        fallback = np.asarray(data["fallback_stop"], dtype=bool)
        escape = np.asarray(data["escape_turn"], dtype=bool)
        collision = np.asarray(data["geometric_collision"], dtype=bool)
        clearance = np.asarray(data["executed_surface_clearance"], dtype=np.float64)
        barrier = np.asarray(data["selected_barrier_energy"], dtype=np.float64)
        lyapunov = np.asarray(data["selected_lyapunov_energy"], dtype=np.float64)
        correction = np.asarray(
            data["mean_guidance_noise_correction"], dtype=np.float64
        )
        if "selected_energy_valid" in data:
            energy_valid = np.asarray(data["selected_energy_valid"], dtype=bool)
        else:
            energy_valid = ~(fallback | escape)

        if verify_soft_only:
            expected = {
                "particle_collision_mask": False,
                "hard_collision_rejection": False,
                "deterministic_escape": False,
            }
            for key, wanted in expected.items():
                actual = bool(scalar(data, key, not wanted))
                if actual != wanted:
                    raise RuntimeError(
                        f"{path}: expected {key}={wanted}, found {actual}"
                    )
            zero_weights = (
                "hard_collision_penalty",
                "safety_weight",
                "terminal_goal_weight",
                "nominal_weight",
                "smoothness_weight",
                "step_weight",
                "circulation_weight",
                "circulation_switch_weight",
                "lyapunov_weight",
            )
            for key in zero_weights:
                value = float(scalar(data, key, np.nan))
                if not np.isfinite(value) or abs(value) > 1.0e-12:
                    raise RuntimeError(f"{path}: expected {key}=0, found {value}")
            if fallback.any() or escape.any():
                raise RuntimeError(
                    f"{path}: deterministic override occurred despite being disabled"
                )

        positions = pose[:, [0, 2]]
        segments = np.diff(positions, axis=0)
        path_length = float(np.linalg.norm(segments, axis=1).sum())
        start = positions[0]
        goal = np.asarray(data["goal_position"], dtype=np.float64)[[0, 2]]
        line = goal - start
        line_norm = max(float(np.linalg.norm(line)), 1.0e-12)
        cross_track = (
            np.abs(
                line[0] * (start[1] - positions[:, 1])
                - (start[0] - positions[:, 0]) * line[1]
            )
            / line_norm
        )

        valid_clearance = clearance[np.isfinite(clearance)]
        valid_barrier = barrier[energy_valid & np.isfinite(barrier)]
        valid_lyapunov = lyapunov[energy_valid & np.isfinite(lyapunov)]
        valid_correction = correction[np.isfinite(correction)]

        barrier_weight = float(scalar(data, "barrier_weight", np.nan))
        lyapunov_weight = float(scalar(data, "lyapunov_weight", np.nan))

        return {
            "path": str(path),
            "success": bool(scalar(data, "success", False)),
            "steps": int(len(goal_distance)),
            "collision": bool(collision.any()),
            "collision_fraction": float(collision.mean()),
            "minimum_clearance_m": (
                float(valid_clearance.min()) if valid_clearance.size else np.nan
            ),
            "final_goal_distance_m": float(goal_distance[-1]),
            "path_length_m": path_length,
            "maximum_cross_track_m": float(cross_track.max()),
            "mean_forward_speed_mps": float(np.abs(actions[:, 0]).mean()),
            "fallback_fraction": float(fallback.mean()),
            "escape_fraction": float(escape.mean()),
            "energy_valid_fraction": float(energy_valid.mean()),
            "barrier_active_fraction": (
                float((valid_barrier > 1.0e-8).mean()) if valid_barrier.size else np.nan
            ),
            "lyapunov_active_fraction": (
                float((valid_lyapunov > 1.0e-8).mean())
                if valid_lyapunov.size
                else np.nan
            ),
            "mean_barrier_energy": (
                float(valid_barrier.mean()) if valid_barrier.size else np.nan
            ),
            "mean_lyapunov_energy": (
                float(valid_lyapunov.mean()) if valid_lyapunov.size else np.nan
            ),
            "weighted_barrier_contribution": (
                float(valid_barrier.mean() * barrier_weight)
                if valid_barrier.size
                else np.nan
            ),
            "weighted_lyapunov_contribution": (
                float(valid_lyapunov.mean() * lyapunov_weight)
                if valid_lyapunov.size
                else np.nan
            ),
            "mean_guidance_correction": (
                float(valid_correction.mean()) if valid_correction.size else np.nan
            ),
            "maximum_guidance_correction": float(
                np.nanmax(np.asarray(data["maximum_guidance_noise_correction"]))
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure", type=Path, required=True)
    parser.add_argument("--guided", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pure = summarize(args.pure, verify_soft_only=False)
    guided = summarize(args.guided, verify_soft_only=True)

    delta_keys = (
        "minimum_clearance_m",
        "final_goal_distance_m",
        "path_length_m",
        "maximum_cross_track_m",
        "mean_forward_speed_mps",
    )
    delta = {key: float(guided[key]) - float(pure[key]) for key in delta_keys}
    result = {"pure_navdp": pure, "cbf_only": guided, "guided_minus_pure": delta}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    columns = (
        ("success", "success"),
        ("collision", "collision"),
        ("minimum_clearance_m", "min_clear"),
        ("final_goal_distance_m", "goal_final"),
        ("path_length_m", "path_len"),
        ("maximum_cross_track_m", "lateral"),
        ("barrier_active_fraction", "CBF_active"),
        ("lyapunov_active_fraction", "Lyap_active"),
        ("mean_guidance_correction", "guide_rms"),
        ("escape_fraction", "escape"),
    )
    print(f"{'model':>20}  " + "  ".join(f"{title:>10}" for _, title in columns))
    for name, row in (("pure_navdp", pure), ("cbf_only", guided)):
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, bool):
                values.append(f"{int(value):10d}")
            else:
                values.append(f"{float(value):10.4f}")
        print(f"{name:>20}  " + "  ".join(values))

    print("\nGuided minus pure:")
    for key, value in delta.items():
        print(f"  {key}: {value:+.4f}")
    print("\nSoft-guidance contributions:")
    print(f"  CBF active:       {float(guided['barrier_active_fraction']):.2%}")
    print(
        f"  Lyapunov residual positive: {float(guided['lyapunov_active_fraction']):.2%}"
    )
    print(f"  weighted CBF:     {float(guided['weighted_barrier_contribution']):.6f}")
    print(f"  weighted Lyap:    {float(guided['weighted_lyapunov_contribution']):.6f}")
    print(
        f"  deterministic overrides: {float(guided['escape_fraction'] + guided['fallback_fraction']):.2%}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
