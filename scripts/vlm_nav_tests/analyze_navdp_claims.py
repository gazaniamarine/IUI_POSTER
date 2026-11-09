#!/usr/bin/env python3
"""Evaluate whether NavDP/HLC-S2Diff paper claims are supported by experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HIGHER_BETTER = {"success", "minimum_clearance_m", "path_efficiency", "euclidean_spl"}
LOWER_BETTER = {"collision", "deadlock", "stopped_fraction", "latency_p50_ms"}


def load_results(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["_path"] = str(resolved)
    return payload


def summaries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in payload.get("summary", [])}


def effects(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["comparator"]), str(row["metric"])): row
        for row in payload.get("paired_effects", [])
    }


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def significant_advantage(
    effect_map: dict[tuple[str, str], dict[str, Any]], comparator: str, metric: str
) -> bool:
    row = effect_map.get((comparator, metric))
    if row is None:
        return False
    low = finite(row.get("reference_advantage_ci95_low"))
    return low is not None and low > 0.0


def adequately_sampled(
    effect_map: dict[tuple[str, str], dict[str, Any]],
    comparator: str,
    metric: str,
    *,
    minimum_layouts: int = 5,
    minimum_episodes: int = 30,
) -> bool:
    row = effect_map.get((comparator, metric))
    return bool(
        row
        and int(row.get("paired_layouts", 0)) >= minimum_layouts
        and int(row.get("paired_episodes", 0)) >= minimum_episodes
    )


def effect_evidence(
    effect_map: dict[tuple[str, str], dict[str, Any]], comparator: str, metric: str
) -> dict[str, Any] | None:
    row = effect_map.get((comparator, metric))
    if row is None:
        return None
    return {
        "metric": metric,
        "comparator": comparator,
        "advantage_mean": row.get("reference_advantage_mean"),
        "ci95_low": row.get("reference_advantage_ci95_low"),
        "ci95_high": row.get("reference_advantage_ci95_high"),
        "paired_episodes": row.get("paired_episodes"),
        "paired_layouts": row.get("paired_layouts"),
        "significant": significant_advantage(effect_map, comparator, metric),
    }


def build_report(
    static: dict[str, Any],
    escape: dict[str, Any],
    moving: dict[str, Any],
    minimum_moving_success: float,
    maximum_moving_collision: float,
) -> dict[str, Any]:
    static_summary = summaries(static)
    escape_summary = summaries(escape)
    moving_summary = summaries(moving)
    static_effects = effects(static)
    escape_effects = effects(escape)
    moving_effects = effects(moving)

    escape_on = escape_summary.get("hlc_escape_on", {})
    escape_activations = finite(escape_on.get("escape_count_mean")) or 0.0
    escape_success = significant_advantage(
        escape_effects, "hlc_escape_off", "success"
    ) and adequately_sampled(
        escape_effects,
        "hlc_escape_off",
        "success",
        minimum_layouts=4,
        minimum_episodes=20,
    )
    escape_deadlock = significant_advantage(
        escape_effects, "hlc_escape_off", "deadlock"
    ) and adequately_sampled(
        escape_effects,
        "hlc_escape_off",
        "deadlock",
        minimum_layouts=4,
        minimum_episodes=20,
    )
    claim_escape_supported = escape_activations > 0.0 and (escape_success or escape_deadlock)

    static_full = static_summary.get("hlc_full_k4", {})
    collision_mean = finite(static_full.get("collision_mean"))
    collision_high = finite(static_full.get("collision_ci95_high"))

    base_metrics = [
        effect_evidence(static_effects, "s2diff_base_k4", metric)
        for metric in (
            "success",
            "collision",
            "deadlock",
            "minimum_clearance_m",
            "path_efficiency",
        )
    ]
    base_metrics = [item for item in base_metrics if item is not None]
    significant_base_metrics = [
        item
        for item in base_metrics
        if item["significant"]
        and int(item.get("paired_layouts") or 0) >= 5
        and int(item.get("paired_episodes") or 0) >= 30
    ]

    moving_full = moving_summary.get("hlc_full_k4", {})
    moving_success_low = finite(moving_full.get("success_ci95_low"))
    moving_collision_high = finite(moving_full.get("collision_ci95_high"))
    moving_sample_ok = (
        int(moving_full.get("layouts", 0)) >= 5
        and int(moving_full.get("episodes", 0)) >= 30
    )
    moving_supported = (
        moving_sample_ok
        and moving_success_low is not None
        and moving_success_low >= minimum_moving_success
        and moving_collision_high is not None
        and moving_collision_high <= maximum_moving_collision
    )

    pure_metrics = [
        effect_evidence(static_effects, "pure_navdp", metric)
        for metric in (
            "success",
            "collision",
            "deadlock",
            "minimum_clearance_m",
            "path_efficiency",
        )
    ]
    pure_metrics = [item for item in pure_metrics if item is not None]
    significant_pure_metrics = [
        item
        for item in pure_metrics
        if item["significant"]
        and int(item.get("paired_layouts") or 0) >= 5
        and int(item.get("paired_episodes") or 0) >= 30
    ]
    pure_outperformance = bool(significant_pure_metrics)

    return {
        "claims": [
            {
                "claim": "The escape mechanism successfully resolves deadlocks.",
                "status": "SUPPORTED" if claim_escape_supported else "NOT_SUPPORTED_YET",
                "criteria": {
                    "escape_activated": escape_activations > 0.0,
                    "significant_success_advantage": escape_success,
                    "significant_deadlock_advantage": escape_deadlock,
                },
                "evidence": {
                    "mean_escape_activations": escape_activations,
                    "success_effect": effect_evidence(
                        escape_effects, "hlc_escape_off", "success"
                    ),
                    "deadlock_effect": effect_evidence(
                        escape_effects, "hlc_escape_off", "deadlock"
                    ),
                },
            },
            {
                "claim": "HLC-S2Diff guarantees collision avoidance.",
                "status": "CANNOT_BE_ESTABLISHED_BY_EXPERIMENT",
                "criteria": {
                    "formal_forward_invariance_proof_required": True,
                    "current_barrier_is_soft_energy": True,
                },
                "empirical_evidence": {
                    "observed_collision_rate": collision_mean,
                    "layout_bootstrap_ci95_high": collision_high,
                    "permitted_wording": (
                        "No geometric collisions were observed under the evaluated conditions."
                        if collision_mean == 0.0
                        else "Report the measured empirical collision rate."
                    ),
                },
            },
            {
                "claim": "The improvement over base S2Diff is statistically significant.",
                "status": "SUPPORTED_FOR_LISTED_METRICS" if significant_base_metrics else "NOT_SUPPORTED_YET",
                "significant_metrics": significant_base_metrics,
                "all_tested_metrics": base_metrics,
            },
            {
                "claim": "The method handles moving obstacles.",
                "status": "EMPIRICALLY_SUPPORTED" if moving_supported else "NOT_SUPPORTED_YET",
                "criteria": {
                    "minimum_layouts": 5,
                    "minimum_episodes": 30,
                    "minimum_success_ci95_low": minimum_moving_success,
                    "maximum_collision_ci95_high": maximum_moving_collision,
                },
                "evidence": {
                    "layouts": moving_full.get("layouts"),
                    "episodes": moving_full.get("episodes"),
                    "success_mean": moving_full.get("success_mean"),
                    "success_ci95_low": moving_success_low,
                    "collision_mean": moving_full.get("collision_mean"),
                    "collision_ci95_high": moving_collision_high,
                    "pure_navdp_comparison": [
                        effect_evidence(moving_effects, "pure_navdp", metric)
                        for metric in ("success", "collision", "minimum_clearance_m")
                    ],
                },
                "scope": "Reactive frame-by-frame moving-obstacle avoidance; no velocity prediction.",
            },
            {
                "claim": "HLC-S2Diff significantly outperforms pure NavDP.",
                "status": "SUPPORTED_FOR_LISTED_METRICS" if pure_outperformance else "NOT_SUPPORTED_YET",
                "significant_metrics": significant_pure_metrics,
                "all_tested_metrics": pure_metrics,
                "warning": "Use metric-specific wording rather than a universal outperformance claim.",
            },
        ],
        "sources": {
            "static": static["_path"],
            "escape": escape["_path"],
            "moving": moving["_path"],
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# NavDP/HLC-S2Diff Claim Verification", ""]
    for index, claim in enumerate(report["claims"], 1):
        lines.extend(
            [
                f"## {index}. {claim['claim']}",
                "",
                f"**Status: {claim['status']}**",
                "",
                "```json",
                json.dumps(claim, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-results", required=True)
    parser.add_argument("--escape-results", required=True)
    parser.add_argument("--moving-results", required=True)
    parser.add_argument("--minimum-moving-success", type=float, default=0.80)
    parser.add_argument("--maximum-moving-collision", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        load_results(args.static_results),
        load_results(args.escape_results),
        load_results(args.moving_results),
        minimum_moving_success=args.minimum_moving_success,
        maximum_moving_collision=args.maximum_moving_collision,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"Saved {output.with_suffix('.json')}")
    print(f"Saved {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()