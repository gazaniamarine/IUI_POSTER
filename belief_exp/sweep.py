#!/usr/bin/env python3
"""Paired random-search sweep over navdp.SubgoalBeliefBank's covariance/
confidence parameters (+ RouteManager.success_radius, + the scan-gate
threshold), ranked by calibration + closed-loop task performance. "Paired"
means the same seeded list of randomized scenarios is replayed against every
sampled config, so leaderboard differences reflect the config, not which
random episodes it happened to see.

Usage:
    conda run -n sam2 python belief_exp/sweep.py \\
        --configs-n 200 --episodes-per-config 60 --seed 0 \\
        --out belief_exp/results/sweep_001.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from metrics import compute_metrics
from scenario import BankConfig, EnvConfig, GateConfig, RouteConfig, run_episode

PARAM_COLUMNS = [
    "sigma_init",
    "sigma_visible",
    "odom_noise",
    "decay_factor",
    "large_uncertainty",
    "success_radius",
    "sigma_ale_threshold",
]
METRIC_COLUMNS = [
    "mean_err_visible",
    "mean_err_occluded",
    "calibration_nll",
    "coverage_1sigma",
    "coverage_2sigma",
    "coverage_deviation",
    "mean_final_dist",
    "advance_rate",
    "mean_steps_to_advance",
    "false_advance_rate",
]


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_config(
    rng: np.random.Generator, args: argparse.Namespace
) -> Tuple[BankConfig, RouteConfig, GateConfig]:
    bank_cfg = BankConfig(
        sigma_init=_log_uniform(rng, *args.sigma_init_range),
        sigma_visible=_log_uniform(rng, *args.sigma_visible_range),
        odom_noise=_log_uniform(rng, *args.odom_noise_range),
        decay_factor=float(rng.uniform(*args.decay_factor_range)),
        large_uncertainty=_log_uniform(rng, *args.large_uncertainty_range),
    )
    route_cfg = RouteConfig(
        success_radius=float(rng.uniform(*args.success_radius_range))
    )
    gate_cfg = GateConfig(
        sigma_ale_threshold=float(rng.uniform(*args.gate_threshold_range))
    )
    return bank_cfg, route_cfg, gate_cfg


def sample_env_scenarios(
    n: int, rng: np.random.Generator, args: argparse.Namespace
) -> List[Tuple[EnvConfig, int]]:
    """Environment (ground-truth noise / occlusion regime) scenarios, generated ONCE and
    replayed identically against every sampled bank/route/gate config below."""
    scenarios = []
    for _ in range(n):
        env_cfg = EnvConfig(
            bearing0_deg=args.bearing0_deg,
            range0=tuple(args.range0),
            obs_noise_std=float(rng.uniform(*args.env_obs_noise_range)),
            odom_noise_std=float(rng.uniform(*args.env_odom_noise_range)),
            occlusion_mode="markov" if rng.random() < 0.5 else "bernoulli",
            p_visible=float(rng.uniform(0.2, 0.8)),
            mean_streak_len=float(rng.uniform(3.0, 12.0)),
            dt=args.dt,
            turn_kp=args.turn_kp,
            base_forward=args.base_forward,
            max_yaw_rate=args.max_yaw_rate,
        )
        seed = int(rng.integers(0, 2**31 - 1))
        scenarios.append((env_cfg, seed))
    return scenarios


def evaluate_config(
    bank_cfg: BankConfig,
    route_cfg: RouteConfig,
    gate_cfg: GateConfig,
    scenarios: List[Tuple[EnvConfig, int]],
    max_steps: int,
) -> Dict[str, float]:
    logs = [
        run_episode(
            bank_cfg,
            route_cfg,
            env_cfg,
            gate_cfg,
            np.random.default_rng(seed),
            max_steps,
        )
        for env_cfg, seed in scenarios
    ]
    return compute_metrics(logs, success_radius=route_cfg.success_radius)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    return (x - float(np.mean(x))) / (std + 1e-8)


def add_combined_scores(
    rows: List[Dict[str, float]], calibration_weight: float, task_weight: float
) -> None:
    """In-place: z-normalize metrics ACROSS the leaderboard and add score columns.

    This is the only place metrics from different configs are compared against each
    other -- every metric in `rows` is otherwise self-contained (see metrics.py)."""
    nll = np.array([r["calibration_nll"] for r in rows])
    covdev = np.array([r["coverage_deviation"] for r in rows])
    dist = np.array([r["mean_final_dist"] for r in rows])
    adv_rate = np.array([r["advance_rate"] for r in rows])
    false_adv = np.array(
        [
            0.0 if np.isnan(r["false_advance_rate"]) else r["false_advance_rate"]
            for r in rows
        ]
    )

    calibration_score = (_zscore(-nll) + _zscore(-covdev)) / 2.0
    task_score = (_zscore(-dist) + _zscore(adv_rate) + _zscore(-false_adv)) / 3.0
    combined = calibration_weight * calibration_score + task_weight * task_score

    for i, row in enumerate(rows):
        row["calibration_score"] = float(calibration_score[i])
        row["task_score"] = float(task_score[i])
        row["combined_score"] = float(combined[i])


def print_leaderboard(rows: List[Dict[str, float]], top_n: int = 10) -> None:
    cols = ["combined_score", "calibration_score", "task_score"] + PARAM_COLUMNS
    header = " ".join(f"{c[:14]:>14}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for row in rows[:top_n]:
        print(" ".join(f"{row[c]:>14.4f}" for c in cols))

    best = rows[0]
    print("\nBest config:")
    print(
        "SubgoalBeliefBank([goal_id], "
        f"sigma_init={best['sigma_init']:.4g}, sigma_visible={best['sigma_visible']:.4g}, "
        f"odom_noise={best['odom_noise']:.4g}, decay_factor={best['decay_factor']:.4g}, "
        f"large_uncertainty={best['large_uncertainty']:.4g})"
    )
    print(f"RouteManager(route, success_radius={best['success_radius']:.4g})")
    print(f"gate.sigma_ale_threshold = {best['sigma_ale_threshold']:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--configs-n", type=int, default=200)
    ap.add_argument("--episodes-per-config", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--calibration-weight", type=float, default=0.5)
    ap.add_argument("--task-weight", type=float, default=0.5)

    # bank/route/gate param ranges (random search)
    ap.add_argument("--sigma-init-range", type=float, nargs=2, default=[0.1, 5.0])
    ap.add_argument("--sigma-visible-range", type=float, nargs=2, default=[0.01, 0.5])
    ap.add_argument("--odom-noise-range", type=float, nargs=2, default=[0.001, 0.2])
    ap.add_argument("--decay-factor-range", type=float, nargs=2, default=[0.8, 0.999])
    ap.add_argument(
        "--large-uncertainty-range", type=float, nargs=2, default=[50.0, 5000.0]
    )
    ap.add_argument("--success-radius-range", type=float, nargs=2, default=[0.2, 1.0])
    ap.add_argument("--gate-threshold-range", type=float, nargs=2, default=[0.1, 2.0])

    # scenario (ground-truth) ranges/fixed values
    ap.add_argument("--env-obs-noise-range", type=float, nargs=2, default=[0.02, 0.3])
    ap.add_argument("--env-odom-noise-range", type=float, nargs=2, default=[0.0, 0.15])
    ap.add_argument("--bearing0-deg", type=float, default=60.0)
    ap.add_argument("--range0", type=float, nargs=2, default=[2.0, 10.0])
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--turn-kp", type=float, default=1.4)
    ap.add_argument("--base-forward", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    args = ap.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent
        / "results"
        / (f"sweep_{datetime.datetime.now():%y%m%d_%H%M%S}.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    scenarios = sample_env_scenarios(args.episodes_per_config, master_rng, args)
    print(f"generated {len(scenarios)} paired environment scenarios (seed={args.seed})")

    rows: List[Dict[str, float]] = []
    for i in range(args.configs_n):
        bank_cfg, route_cfg, gate_cfg = sample_config(master_rng, args)
        metrics = evaluate_config(
            bank_cfg, route_cfg, gate_cfg, scenarios, args.max_steps
        )
        row = {
            "sigma_init": bank_cfg.sigma_init,
            "sigma_visible": bank_cfg.sigma_visible,
            "odom_noise": bank_cfg.odom_noise,
            "decay_factor": bank_cfg.decay_factor,
            "large_uncertainty": bank_cfg.large_uncertainty,
            "success_radius": route_cfg.success_radius,
            "sigma_ale_threshold": gate_cfg.sigma_ale_threshold,
            **metrics,
        }
        rows.append(row)
        if (i + 1) % max(args.configs_n // 20, 1) == 0:
            print(f"[{i + 1}/{args.configs_n}] configs evaluated", flush=True)

    add_combined_scores(rows, args.calibration_weight, args.task_weight)
    rows.sort(key=lambda r: r["combined_score"], reverse=True)

    fieldnames = (
        PARAM_COLUMNS
        + METRIC_COLUMNS
        + ["n_episodes", "calibration_score", "task_score", "combined_score"]
    )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nwrote {len(rows)} configs to {out_path}")

    print_leaderboard(rows)


if __name__ == "__main__":
    main()
