#!/usr/bin/env python3
"""sigma_min_sweep.py -- for sigma_visible and odom_noise (BankConfig's two belief-tightness
knobs: the post-sighting reset value, and the Sigma growth rate while occluded), grid-scan a
2D log-spaced search to find the smallest JOINTLY viable pair, holding every other
inspect_one.py param at its default. Repeat that search once per grid value of each OTHER
param (one-at-a-time sensitivity), so the minimum-viable-sigma curve as a function of each
other param can be read straight off the summary CSV.

"Viable" = combined calibration + task criterion on metrics.py's own outputs:
    coverage_deviation    <= --coverage-dev-tol
    AND advance_rate      >= --min-advance-rate
    AND false_advance_rate <= --max-false-advance-rate

Among viable (sigma_visible, odom_noise) pairs, the one minimizing their product (both are
variance-like quantities) is reported as the minimum viable point.

The aleatoric-sigma pause/scan gate is disabled for every run here (GateConfig with an
infinite threshold), so results reflect pure belief-tightness effects rather than gate
interaction -- this is a caller-side GateConfig choice only; scenario.py and navdp/ are
untouched. The continuous speed-caution scaling (strength_sigma_low/high) is a separate
always-on mechanism and stays active.

Smoke test:
   conda run -n sam2 --no-capture-output python belief_exp/sigma_min_sweep.py --episodes-per-config 5  --sigma-visible-grid 1e-3 0.2 4 --odom-noise-grid 1e-3 0.1 4 --other-grid-n 2  --params decay_factor,success_radius

Full run (~48 config-points x 144 sigma pairs x 20 episodes):
    conda run -n sam2 --no-capture-output python belief_exp/sigma_min_sweep.py --seed 0
"""

from __future__ import annotations

import argparse
import csv
import datetime
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from metrics import compute_metrics
from scenario import BankConfig, EnvConfig, GateConfig, RouteConfig, run_episode
from sweep import METRIC_COLUMNS

DISABLED_GATE = GateConfig(sigma_ale_threshold=float("inf"))

# One-at-a-time sensitivity dimensions. Each holds every OTHER param at its inspect_one.py
# default and varies just this one across a grid. sigma_init is intentionally excluded (the
# README notes it's essentially dead / rarely hit in practice); sigma_ale_threshold is
# excluded because the gate is disabled for this whole script.
OTHER_PARAMS: Dict[str, dict] = {
    "decay_factor": dict(
        kind="bank", field="decay_factor", range=(0.8, 0.999), spacing="linear"
    ),
    "large_uncertainty": dict(
        kind="bank", field="large_uncertainty", range=(50.0, 5000.0), spacing="log"
    ),
    "success_radius": dict(
        kind="route", field="success_radius", range=(0.2, 1.0), spacing="linear"
    ),
    "env_obs_noise": dict(
        kind="env", field="obs_noise_std", range=(0.02, 0.3), spacing="log"
    ),
    "env_odom_noise": dict(
        kind="env", field="odom_noise_std", range=(0.0, 0.15), spacing="linear"
    ),
    "occlusion_mode": dict(
        kind="env",
        field="occlusion_mode",
        values=["markov", "bernoulli"],
        spacing="categorical",
    ),
    "p_visible": dict(
        kind="env",
        field="p_visible",
        range=(0.2, 0.8),
        spacing="linear",
        forces={
            "occlusion_mode": "bernoulli"
        },  # p_visible is a no-op under the default "markov" mode
    ),
    "mean_streak_len": dict(
        kind="env", field="mean_streak_len", range=(3.0, 12.0), spacing="linear"
    ),
    "bearing0_deg": dict(
        kind="env", field="bearing0_deg", range=(30.0, 90.0), spacing="linear"
    ),
    "range0": dict(
        kind="env",
        field="range0",
        range=(4.0, 20.0),
        spacing="linear",
        to_field=lambda upper: (
            2.0,
            upper,
        ),  # lower bound fixed at 2.0; sweep the upper bound
    ),
    "dt": dict(kind="env", field="dt", range=(0.2, 1.0), spacing="linear"),
}


def _linspace_grid(lo: float, hi: float, n: int) -> List[float]:
    return [float(x) for x in np.linspace(lo, hi, n)]


def _logspace_grid(lo: float, hi: float, n: int) -> List[float]:
    return [float(x) for x in np.exp(np.linspace(np.log(lo), np.log(hi), n))]


def _param_grid_values(name: str, spec: dict, args: argparse.Namespace) -> List:
    if spec["spacing"] == "categorical":
        return list(spec["values"])
    lo, hi = getattr(args, f"{name}_range")
    n = int(args.other_grid_n)
    if spec["spacing"] == "log":
        return _logspace_grid(lo, hi, n)
    return _linspace_grid(lo, hi, n)


def iter_config_points(args: argparse.Namespace) -> Iterator[dict]:
    yield {
        "sweep_param": "baseline",
        "sweep_value": None,
        "bank": {},
        "route": {},
        "env": {},
    }
    selected = args.params
    for name, spec in OTHER_PARAMS.items():
        if selected is not None and name not in selected:
            continue
        to_field = spec.get("to_field", lambda v: v)
        for v in _param_grid_values(name, spec, args):
            overrides = {"bank": {}, "route": {}, "env": {}}
            overrides[spec["kind"]][spec["field"]] = to_field(v)
            if spec.get("forces"):
                overrides["env"].update(spec["forces"])
            yield {"sweep_param": name, "sweep_value": v, **overrides}


def is_viable(metrics: Dict[str, float], args: argparse.Namespace) -> bool:
    cov_ok = metrics["coverage_deviation"] <= args.coverage_dev_tol
    adv_ok = metrics["advance_rate"] >= args.min_advance_rate
    false_adv = metrics["false_advance_rate"]
    false_adv_ok = (
        not np.isnan(false_adv)
    ) and false_adv <= args.max_false_advance_rate
    return cov_ok and adv_ok and false_adv_ok


def evaluate_sigma_grid(
    bank_base: BankConfig,
    route_cfg: RouteConfig,
    env_cfg: EnvConfig,
    args: argparse.Namespace,
    sv_grid: List[float],
    on_grid: List[float],
    progress_prefix: str = "",
) -> Tuple[
    List[Tuple[float, float, Dict[str, float], bool]],
    Optional[Tuple[float, float, float, Dict[str, float]]],
]:
    detail_rows = []
    best: Optional[Tuple[float, float, float, Dict[str, float]]] = None
    total_pairs = len(sv_grid) * len(on_grid)
    report_every = max(1, total_pairs // 10)
    t0 = time.time()
    for idx, (sv, on) in enumerate(
        ((sv, on) for sv in sv_grid for on in on_grid), start=1
    ):
        bank_cfg = replace(bank_base, sigma_visible=sv, odom_noise=on)
        logs = [
            run_episode(
                bank_cfg,
                route_cfg,
                env_cfg,
                DISABLED_GATE,
                np.random.default_rng(args.seed + i),
                args.max_steps,
            )
            for i in range(args.episodes_per_config)
        ]
        metrics = compute_metrics(logs, success_radius=route_cfg.success_radius)
        viable = is_viable(metrics, args)
        detail_rows.append((sv, on, metrics, viable))
        if viable:
            score = sv * on
            if best is None or score < best[0]:
                best = (score, sv, on, metrics)
        if idx % report_every == 0 or idx == total_pairs:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else float("inf")
            print(
                f"    {progress_prefix}sigma pair {idx}/{total_pairs} "
                f"({elapsed:.1f}s elapsed, {rate:.1f} pairs/s)",
                flush=True,
            )
    return detail_rows, best


def print_summary_table(rows: List[Dict]) -> None:
    cols = [
        "sweep_param",
        "sweep_value",
        "min_sigma_visible",
        "min_odom_noise",
        "found_viable",
    ]
    header = " ".join(f"{c:>18}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for c in cols:
            v = row[c]
            parts.append(f"{v:>18.4g}" if isinstance(v, float) else f"{str(v):>18}")
        print(" ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sigma-visible-grid",
        type=float,
        nargs=3,
        default=[1e-4, 0.5, 12],
        metavar=("MIN", "MAX", "N"),
    )
    ap.add_argument(
        "--odom-noise-grid",
        type=float,
        nargs=3,
        default=[1e-4, 0.3, 12],
        metavar=("MIN", "MAX", "N"),
    )
    ap.add_argument("--episodes-per-config", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coverage-dev-tol", type=float, default=0.15)
    ap.add_argument("--min-advance-rate", type=float, default=0.7)
    ap.add_argument("--max-false-advance-rate", type=float, default=0.1)
    ap.add_argument("--other-grid-n", type=int, default=5)
    ap.add_argument(
        "--params",
        type=str,
        default=None,
        help="comma-separated subset of OTHER_PARAMS names to sweep (default: all)",
    )
    ap.add_argument("--out-details", type=str, default=None)
    ap.add_argument("--out-summary", type=str, default=None)

    for name, spec in OTHER_PARAMS.items():
        if spec["spacing"] == "categorical":
            continue
        flag = f"--{name.replace('_', '-')}-range"
        ap.add_argument(flag, type=float, nargs=2, default=list(spec["range"]))

    args = ap.parse_args()
    args.params = args.params.split(",") if args.params else None
    if args.params is not None:
        unknown = sorted(set(args.params) - set(OTHER_PARAMS))
        if unknown:
            ap.error(
                f"unknown --params entries: {unknown}; choices are {sorted(OTHER_PARAMS)}"
            )

    sv_lo, sv_hi, sv_n = args.sigma_visible_grid
    on_lo, on_hi, on_n = args.odom_noise_grid
    sv_grid = _logspace_grid(sv_lo, sv_hi, int(sv_n))
    on_grid = _logspace_grid(on_lo, on_hi, int(on_n))

    timestamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_details = (
        Path(args.out_details)
        if args.out_details
        else results_dir / f"sigma_min_details_{timestamp}.csv"
    )
    out_summary = (
        Path(args.out_summary)
        if args.out_summary
        else results_dir / f"sigma_min_summary_{timestamp}.csv"
    )

    points = list(iter_config_points(args))
    pairs_per_point = len(sv_grid) * len(on_grid)
    print(
        f"evaluating {len(points)} config-point(s), {len(sv_grid)}x{len(on_grid)} sigma pairs each "
        f"({pairs_per_point * args.episodes_per_config} episodes per config-point)"
    )

    detail_fieldnames = (
        ["sweep_param", "sweep_value", "sigma_visible", "odom_noise"]
        + METRIC_COLUMNS
        + ["n_episodes", "viable"]
    )
    other_param_names = list(OTHER_PARAMS.keys())
    summary_fieldnames = (
        ["sweep_param", "sweep_value"]
        + other_param_names
        + ["min_sigma_visible", "min_odom_noise", "found_viable"]
        + METRIC_COLUMNS
        + ["n_episodes"]
    )

    summary_rows: List[Dict] = []
    total_detail_rows = 0
    run_t0 = time.time()
    with out_details.open("w", newline="") as f_details:
        writer = csv.DictWriter(f_details, fieldnames=detail_fieldnames)
        writer.writeheader()

        for i, point in enumerate(points):
            bank_base = replace(BankConfig(), **point["bank"])
            route_cfg = replace(RouteConfig(), **point["route"])
            env_cfg = replace(EnvConfig(), **point["env"])

            print(
                f"[{i + 1}/{len(points)}] {point['sweep_param']}={point['sweep_value']} "
                f"-- starting {pairs_per_point} sigma pairs...",
                flush=True,
            )
            detail_rows, best = evaluate_sigma_grid(
                bank_base,
                route_cfg,
                env_cfg,
                args,
                sv_grid,
                on_grid,
                progress_prefix=f"[{i + 1}/{len(points)}] ",
            )
            total_detail_rows += len(detail_rows)

            for sv, on, metrics, viable in detail_rows:
                row = {
                    "sweep_param": point["sweep_param"],
                    "sweep_value": point["sweep_value"],
                    "sigma_visible": sv,
                    "odom_noise": on,
                    "viable": viable,
                }
                row.update({k: metrics[k] for k in METRIC_COLUMNS})
                row["n_episodes"] = metrics["n_episodes"]
                writer.writerow(row)

            other_values = {}
            for name, spec in OTHER_PARAMS.items():
                cfg_obj = {"bank": bank_base, "route": route_cfg, "env": env_cfg}[
                    spec["kind"]
                ]
                val = getattr(cfg_obj, spec["field"])
                other_values[name] = val[1] if name == "range0" else val

            summary_row = {
                "sweep_param": point["sweep_param"],
                "sweep_value": point["sweep_value"],
                **other_values,
            }
            if best is not None:
                _, best_sv, best_on, best_metrics = best
                summary_row["min_sigma_visible"] = best_sv
                summary_row["min_odom_noise"] = best_on
                summary_row["found_viable"] = True
                summary_row.update({k: best_metrics[k] for k in METRIC_COLUMNS})
                summary_row["n_episodes"] = best_metrics["n_episodes"]
                progress_msg = f"min_sv={best_sv:.4g} min_on={best_on:.4g}"
            else:
                summary_row["min_sigma_visible"] = float("nan")
                summary_row["min_odom_noise"] = float("nan")
                summary_row["found_viable"] = False
                summary_row.update({k: float("nan") for k in METRIC_COLUMNS})
                summary_row["n_episodes"] = float("nan")
                progress_msg = "NO VIABLE PAIR"
            summary_rows.append(summary_row)

            elapsed_total = time.time() - run_t0
            print(
                f"[{i + 1}/{len(points)}] {point['sweep_param']}={point['sweep_value']} -> {progress_msg} "
                f"(run elapsed: {elapsed_total:.1f}s)",
                flush=True,
            )

    with out_summary.open("w", newline="") as f_summary:
        writer = csv.DictWriter(f_summary, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nwrote {total_detail_rows} detail rows to {out_details}")
    print(f"wrote {len(summary_rows)} summary rows to {out_summary}")
    print(f"total wall time: {time.time() - run_t0:.1f}s")
    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
