#!/usr/bin/env python3
"""Plot the particle-only NavDP ablation and static hard-case breakdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MECHANISMS = [
    "pure_navdp",
    "particle_full_p2",
    "particle_no_cbf_barrier",
    "particle_no_lyapunov",
    "particle_no_cbf_no_lyapunov",
    "particle_no_denoise_feedback",
    "particle_p1",
    "particle_uniform_weights",
    "particle_sigma0",
    "particle_no_collision_mask",
    "particle_no_anchor",
    "particle_fixed_noise",
    "particle_constant_guidance",
]
DISPLAY = {
    "pure_navdp": "Pure NavDP",
    "particle_full_p2": "Full (P=2)",
    "particle_no_cbf_barrier": "No CBF barrier",
    "particle_no_lyapunov": "No Lyapunov",
    "particle_no_cbf_no_lyapunov": "No CBF / Lyap.",
    "particle_p1": "P=1",
    "particle_sigma0": "No exploration",
    "particle_uniform_weights": "Uniform weights",
    "particle_no_collision_mask": "No collision mask",
    "particle_no_anchor": "No anchor",
    "particle_fixed_noise": "Fixed spread",
    "particle_constant_guidance": "No ramp",
    "particle_no_denoise_feedback": "No feedback",
}


def number(value: Any) -> float:
    return np.nan if value is None else float(value)


def ci_error(row: dict[str, Any], metric: str, scale: float = 1.0) -> np.ndarray:
    mean = number(row.get(f"{metric}_mean")) * scale
    low = number(row.get(f"{metric}_ci95_low")) * scale
    high = number(row.get(f"{metric}_ci95_high")) * scale
    if not np.all(np.isfinite([mean, low, high])):
        return np.asarray([[0.0], [0.0]])
    return np.asarray([[max(mean - low, 0.0)], [max(high - mean, 0.0)]])


def save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(figure)


def mechanism_plot(summary: dict[str, dict[str, Any]], output: Path) -> None:
    labels = [label for label in MECHANISMS if label in summary]
    if not labels:
        return
    metrics = [
        ("success", "Success (%)", 100.0),
        ("collision", "Collision (%)", 100.0),
        ("minimum_clearance_m", "Minimum clearance (m)", 1.0),
        ("latency_p50_ms", "Planning latency p50 (ms)", 1.0),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 7.5))
    x = np.arange(len(labels))
    colors = [
        "#e45756" if label == "particle_full_p2" else "#4c78a8" for label in labels
    ]
    for axis, (metric, title, scale) in zip(axes.flat, metrics):
        values = [
            number(summary[label].get(f"{metric}_mean")) * scale for label in labels
        ]
        errors = np.concatenate(
            [ci_error(summary[label], metric, scale) for label in labels], axis=1
        )
        axis.bar(
            x,
            values,
            yerr=errors,
            color=colors,
            capsize=2,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_title(title)
        axis.set_xticks(
            x, [DISPLAY.get(label, label) for label in labels], rotation=38, ha="right"
        )
        axis.grid(axis="y", alpha=0.25)
        if metric in {"success", "collision"}:
            axis.set_ylim(0.0, 105.0)
    figure.tight_layout()
    save(figure, output / "particle_mechanisms")


def particle_count_plot(summary: dict[str, dict[str, Any]], output: Path) -> None:
    points: dict[int, dict[str, Any]] = {}
    for label, row in summary.items():
        if label == "particle_full_p2":
            points[2] = row
        else:
            match = re.fullmatch(r"particle_p(\d+)", label)
            if match:
                points[int(match.group(1))] = row
    if len(points) < 2:
        return
    counts = sorted(points)
    metrics = [
        ("success", "Success (%)", 100.0),
        ("minimum_clearance_m", "Minimum clearance (m)", 1.0),
        ("latency_p50_ms", "Planning latency p50 (ms)", 1.0),
        ("normalized_particle_ess", "Normalized ESS", 1.0),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(14.0, 3.2))
    for axis, (metric, title, scale) in zip(axes, metrics):
        values = [
            number(points[count].get(f"{metric}_mean")) * scale for count in counts
        ]
        axis.plot(counts, values, marker="o", color="#e45756")
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts, [str(count) for count in counts])
        axis.set_xlabel("Particles per candidate")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if metric == "success":
            axis.set_ylim(0.0, 105.0)
    figure.tight_layout()
    save(figure, output / "particle_count_tradeoff")


def hardcase_heatmap(layout_rows: list[dict[str, Any]], output: Path) -> None:
    wanted = [
        "pure_navdp",
        "particle_full_p2",
        "particle_no_cbf_barrier",
        "particle_uniform_weights",
        "particle_no_collision_mask",
    ]
    layouts = sorted({str(row["layout"]) for row in layout_rows})
    labels = [
        label for label in wanted if any(row["label"] == label for row in layout_rows)
    ]
    if not layouts or not labels:
        return
    lookup = {(str(row["label"]), str(row["layout"])): row for row in layout_rows}
    values = np.full((len(labels), len(layouts)), np.nan)
    for row_index, label in enumerate(labels):
        for column_index, layout in enumerate(layouts):
            row = lookup.get((label, layout))
            if row is not None:
                values[row_index, column_index] = (
                    number(row.get("success_mean")) * 100.0
                )
    figure, axis = plt.subplots(figsize=(max(8.0, 0.75 * len(layouts)), 3.8))
    image = axis.imshow(values, vmin=0.0, vmax=100.0, cmap="RdYlGn", aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            if np.isfinite(values[row_index, column_index]):
                axis.text(
                    column_index,
                    row_index,
                    f"{values[row_index, column_index]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    axis.set_xticks(np.arange(len(layouts)), layouts, rotation=40, ha="right")
    axis.set_yticks(
        np.arange(len(labels)), [DISPLAY.get(label, label) for label in labels]
    )
    axis.set_title("Success rate by static hard case (%)")
    figure.colorbar(image, ax=axis, label="Success (%)")
    figure.tight_layout()
    save(figure, output / "hardcase_success_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with Path(args.results).expanduser().resolve().open(encoding="utf-8") as stream:
        payload = json.load(stream)
    summary = {str(row["label"]): row for row in payload["summary"]}
    output = Path(args.output_dir).expanduser().resolve()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    mechanism_plot(summary, output)
    particle_count_plot(summary, output)
    hardcase_heatmap(payload.get("layout_summary", []), output)


if __name__ == "__main__":
    main()
