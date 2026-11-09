#!/usr/bin/env python3
"""multi_goal_sweep.py -- sibling to sigma_min_sweep.py for the N-goal route
(multi_goal_scenario.run_multi_episode) instead of the single-goal scenario.
Same grid-sweep CLI pattern: for sigma_visible and odom_noise (BankConfig's two
belief-tightness knobs), grid-scan a 2D log-spaced search to find the smallest
jointly viable pair, holding every other param at its default, one config-point
per grid value of each OTHER param (one-at-a-time sensitivity) -- plus a new
--n-goals dimension, swept as an OUTER loop (every OTHER param and every sigma
pair is shared across all slots in one bank, so n_goals is the only genuinely
new axis; it needs no new per-goal config, per the plan).

"Viable" here uses multi_goal_metrics' route-level output instead of
scenario.py's single-leg advance_rate/false_advance_rate (which don't apply to
an N-leg route the same way):
    coverage_deviation <= --coverage-dev-tol
    AND completion_rate  >= --min-completion-rate

Among viable (sigma_visible, odom_noise) pairs, the one minimizing their
product (both are variance-like quantities) is reported as the minimum viable
point, same as sigma_min_sweep.py.

The aleatoric-sigma pause/scan gate is disabled for every run here (GateConfig
with an infinite threshold), same reasoning as sigma_min_sweep.py.

Smoke test:
   conda run -n sam2 --no-capture-output python belief_exp/multi_goal_sweep.py --n-goals 2 3 --episodes-per-config 5 --sigma-visible-grid 1e-3 0.2 4 --odom-noise-grid 1e-3 0.1 4 --other-grid-n 2 --params decay_factor,success_radius

Full run:
    conda run -n sam2 --no-capture-output python belief_exp/multi_goal_sweep.py --seed 0
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

from multi_goal_metrics import METRIC_COLUMNS, compute_multi_goal_metrics
from multi_goal_scenario import run_multi_episode
from scenario import BankConfig, EnvConfig, GateConfig, RouteConfig
from sigma_min_sweep import (
    OTHER_PARAMS,
    _linspace_grid,
    _logspace_grid,
    _param_grid_values,
)

DISABLED_GATE = GateConfig(sigma_ale_threshold=float("inf"))


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
    comp_ok = metrics["completion_rate"] >= args.min_completion_rate
    return cov_ok and comp_ok


def evaluate_sigma_grid(
    bank_base: BankConfig,
    route_cfg: RouteConfig,
    env_cfg: EnvConfig,
    args: argparse.Namespace,
    n_goals: int,
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
            run_multi_episode(
                bank_cfg,
                route_cfg,
                env_cfg,
                DISABLED_GATE,
                np.random.default_rng(args.seed + i),
                n_goals=n_goals,
                max_steps=args.max_steps,
            )
            for i in range(args.episodes_per_config)
        ]
        metrics = compute_multi_goal_metrics(logs)
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
                f"    {progress_prefix}sigma pair {idx}/{total_pairs} ({elapsed:.1f}s elapsed, {rate:.1f} pairs/s)",
                flush=True,
            )
    return detail_rows, best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-goals", type=int, nargs="+", default=[2, 3, 5])
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
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coverage-dev-tol", type=float, default=0.15)
    ap.add_argument("--min-completion-rate", type=float, default=0.7)
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
        else results_dir / f"multi_goal_sigma_min_details_{timestamp}.csv"
    )
    out_summary = (
        Path(args.out_summary)
        if args.out_summary
        else results_dir / f"multi_goal_sigma_min_summary_{timestamp}.csv"
    )

    points = list(iter_config_points(args))
    pairs_per_point = len(sv_grid) * len(on_grid)
    total_points = len(points) * len(args.n_goals)
    print(
        f"evaluating {len(args.n_goals)} n_goals value(s) x {len(points)} config-point(s), "
        f"{len(sv_grid)}x{len(on_grid)} sigma pairs each "
        f"({pairs_per_point * args.episodes_per_config} episodes per config-point)"
    )

    detail_fieldnames = (
        ["n_goals", "sweep_param", "sweep_value", "sigma_visible", "odom_noise"]
        + METRIC_COLUMNS
        + ["viable"]
    )
    other_param_names = list(OTHER_PARAMS.keys())
    summary_fieldnames = (
        ["n_goals", "sweep_param", "sweep_value"]
        + other_param_names
        + ["min_sigma_visible", "min_odom_noise", "found_viable"]
        + METRIC_COLUMNS
    )

    summary_rows: List[Dict] = []
    total_detail_rows = 0
    run_t0 = time.time()
    point_i = 0
    with out_details.open("w", newline="") as f_details:
        writer = csv.DictWriter(f_details, fieldnames=detail_fieldnames)
        writer.writeheader()

        for n_goals in args.n_goals:
            for point in points:
                point_i += 1
                bank_base = replace(BankConfig(), **point["bank"])
                route_cfg = replace(RouteConfig(), **point["route"])
                env_cfg = replace(EnvConfig(), **point["env"])

                print(
                    f"[{point_i}/{total_points}] n_goals={n_goals} {point['sweep_param']}={point['sweep_value']} "
                    f"-- starting {pairs_per_point} sigma pairs...",
                    flush=True,
                )
                detail_rows, best = evaluate_sigma_grid(
                    bank_base,
                    route_cfg,
                    env_cfg,
                    args,
                    n_goals,
                    sv_grid,
                    on_grid,
                    progress_prefix=f"[{point_i}/{total_points}] ",
                )
                total_detail_rows += len(detail_rows)

                for sv, on, metrics, viable in detail_rows:
                    row = {
                        "n_goals": n_goals,
                        "sweep_param": point["sweep_param"],
                        "sweep_value": point["sweep_value"],
                        "sigma_visible": sv,
                        "odom_noise": on,
                        "viable": viable,
                    }
                    row.update({k: metrics[k] for k in METRIC_COLUMNS})
                    writer.writerow(row)

                other_values = {}
                for name, spec in OTHER_PARAMS.items():
                    cfg_obj = {"bank": bank_base, "route": route_cfg, "env": env_cfg}[
                        spec["kind"]
                    ]
                    val = getattr(cfg_obj, spec["field"])
                    other_values[name] = val[1] if name == "range0" else val

                summary_row = {
                    "n_goals": n_goals,
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
                    progress_msg = f"min_sv={best_sv:.4g} min_on={best_on:.4g}"
                else:
                    summary_row["min_sigma_visible"] = float("nan")
                    summary_row["min_odom_noise"] = float("nan")
                    summary_row["found_viable"] = False
                    summary_row.update({k: float("nan") for k in METRIC_COLUMNS})
                    progress_msg = "NO VIABLE PAIR"
                summary_rows.append(summary_row)

                elapsed_total = time.time() - run_t0
                print(
                    f"[{point_i}/{total_points}] n_goals={n_goals} {point['sweep_param']}={point['sweep_value']} -> {progress_msg} "
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


if __name__ == "__main__":
    main()
