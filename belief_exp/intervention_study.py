#!/usr/bin/env python3
"""Reachability study: belief-only steering vs. human-intervention steering under
IRREDUCIBLE occlusion/odometry noise.

Thesis being tested: once sigma_ale (the real navdp Sigma-derived uncertainty,
see common.sigma_ale_from_bank) climbs high enough, it's because the belief mean
`mu` has been dead-reckoned through a long occlusion streak with real odometry
error -- a property of the SCENARIO, not of any SubgoalBeliefBank covariance
knob (see README's "why mean has no sweep range" and sigma_min_sweep.py, which
already establish that sigma_visible/odom_noise tuning only changes how honestly
Sigma REPORTS that drift, not how much mu actually drifts). So past some
severity, no bank tuning saves a controller that blindly steers off mu: it
follows the drifted belief and fails to reach the true goal. A human watching
the raw feed can still supply a roughly-correct bearing without having
integrated that drift, so switching steering to a human-provided bearing
whenever sigma_ale is high should recover reachability that no belief-only
config can.

Both conditions here share the SAME BankConfig/RouteConfig/GateConfig and the
SAME paired scenario seeds per severity level -- the only difference is
scenario.InterventionConfig.enabled, isolating "who supplies the steering
bearing" as the single variable. See scenario.run_episode's human_active branch
for the actual control-law difference; navdp's SubgoalBeliefBank/RouteManager
are exercised identically (and untouched) in both conditions.

Usage:
    # smoke test
    conda run -n sam2 python belief_exp/intervention_study.py \\
        --severity-levels 3 --episodes-per-level 10 --trace

    # full study
    conda run -n sam2 python belief_exp/intervention_study.py \\
        --out belief_exp/results/intervention_study_001.csv --trace
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from metrics import compute_metrics
from scenario import (
    BankConfig,
    EnvConfig,
    GateConfig,
    InterventionConfig,
    RouteConfig,
    run_episode,
)
from sweep import METRIC_COLUMNS

STUDY_METRIC_COLUMNS = METRIC_COLUMNS + ["true_success_rate", "human_active_frac"]

CONDITIONS = ["belief_only", "human_intervention"]


def make_env_cfg(base: EnvConfig, mean_streak_len: float, odom_noise_std: float) -> EnvConfig:
    from dataclasses import replace

    return replace(
        base,
        occlusion_mode="markov",  # long blind streaks are the whole point of "irreducible"
        mean_streak_len=mean_streak_len,
        odom_noise_std=odom_noise_std,
    )


def severity_levels(args: argparse.Namespace) -> List[Tuple[float, float, float]]:
    """Returns [(severity_frac, mean_streak_len, env_odom_noise_std), ...],
    linearly interpolated from mild to severe."""
    fracs = np.linspace(0.0, 1.0, args.severity_levels)
    out = []
    for f in fracs:
        msl = args.mild_mean_streak_len + f * (
            args.severe_mean_streak_len - args.mild_mean_streak_len
        )
        odom = args.mild_env_odom_noise + f * (
            args.severe_env_odom_noise - args.mild_env_odom_noise
        )
        out.append((float(f), float(msl), float(odom)))
    return out


def run_condition(
    bank_cfg: BankConfig,
    route_cfg: RouteConfig,
    env_cfg: EnvConfig,
    gate_cfg: GateConfig,
    intervention_cfg: InterventionConfig,
    seeds: List[int],
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
            intervention_cfg=intervention_cfg,
        )
        for seed in seeds
    ]
    return compute_metrics(logs, success_radius=route_cfg.success_radius)


def print_trace(label: str, log) -> None:
    print(f"\n--- {label} ---")
    header = (
        f"{'t':>3} {'vis':>4} {'human':>6} {'true_x':>8} {'true_y':>8} "
        f"{'mu_x':>8} {'mu_y':>8} {'err':>7} {'sig_ale':>7}"
    )
    print(header)
    print("-" * len(header))
    for i in range(len(log.t)):
        tg = log.true_goal[i]
        mu = log.mu[i]
        err = float(np.linalg.norm(mu - tg))
        sig_ale = float(np.sqrt(max(log.sigma_diag[i][0], log.sigma_diag[i][1])))
        print(
            f"{log.t[i]:>3} {'Y' if log.visible[i] else '.':>4} "
            f"{'H' if log.human_active[i] else '':>6} "
            f"{tg[0]:>8.3f} {tg[1]:>8.3f} {mu[0]:>8.3f} {mu[1]:>8.3f} "
            f"{err:>7.3f} {sig_ale:>7.3f}"
        )
    if log.advanced:
        print(
            f"RouteManager advanced at step {log.steps_to_advance}; "
            f"final true dist = {log.final_true_dist:.3f}"
        )
    else:
        print(f"never advanced; final true dist = {log.final_true_dist:.3f}")


def print_summary(rows: List[Dict]) -> None:
    cols = [
        "severity",
        "mean_streak_len",
        "env_odom_noise",
        "condition",
        "true_success_rate",
        "advance_rate",
        "false_advance_rate",
        "mean_final_dist",
        "human_active_frac",
    ]
    header = " ".join(f"{c[:16]:>16}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for c in cols:
            v = row[c]
            parts.append(f"{v:>16.4g}" if isinstance(v, float) else f"{str(v):>16}")
        print(" ".join(parts))

    print("\ntrue_success_rate delta (human_intervention - belief_only) by severity:")
    by_sev: Dict[float, Dict[str, float]] = {}
    for row in rows:
        by_sev.setdefault(row["severity"], {})[row["condition"]] = row[
            "true_success_rate"
        ]
    for sev in sorted(by_sev):
        d = by_sev[sev]
        if "belief_only" in d and "human_intervention" in d:
            delta = d["human_intervention"] - d["belief_only"]
            print(
                f"  severity={sev:.2f}: belief_only={d['belief_only']:.3f} "
                f"human_intervention={d['human_intervention']:.3f} delta={delta:+.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--severity-levels", type=int, default=6)
    ap.add_argument("--episodes-per-level", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--trace", action="store_true", help="print one paired episode trace at the most severe level")
    ap.add_argument(
        "--trace-candidates",
        type=int,
        default=30,
        help="seeds to search for a trace pair where belief_only fails and human_intervention succeeds",
    )

    # scenario severity range: longer occlusion streaks + noisier odometry during
    # them is what makes mu drift irrecoverably far from the true goal.
    ap.add_argument("--mild-mean-streak-len", type=float, default=3.0)
    ap.add_argument("--severe-mean-streak-len", type=float, default=30.0)
    ap.add_argument("--mild-env-odom-noise", type=float, default=0.02)
    ap.add_argument("--severe-env-odom-noise", type=float, default=0.30)
    ap.add_argument("--env-obs-noise", type=float, default=0.1)
    ap.add_argument("--bearing0-deg", type=float, default=60.0)
    ap.add_argument("--range0", type=float, nargs=2, default=[2.0, 10.0])
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--turn-kp", type=float, default=1.4)
    ap.add_argument("--base-forward", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)

    # bank/route/gate: fixed, reasonably-tuned defaults shared by BOTH conditions
    # (the point is that no fixed bank tuning saves belief-only steering here).
    ap.add_argument("--sigma-init", type=float, default=1.0)
    ap.add_argument("--sigma-visible", type=float, default=0.05)
    ap.add_argument("--odom-noise", type=float, default=0.02)
    ap.add_argument("--decay-factor", type=float, default=0.95)
    ap.add_argument("--large-uncertainty", type=float, default=1000.0)
    ap.add_argument("--success-radius", type=float, default=0.5)
    ap.add_argument(
        "--gate-threshold",
        type=float,
        default=0.5,
        help="sigma_ale trigger for switching to human steering (GateConfig.sigma_ale_threshold)",
    )
    ap.add_argument("--human-bearing-noise-deg", type=float, default=8.0)

    args = ap.parse_args()

    bank_cfg = BankConfig(
        sigma_init=args.sigma_init,
        sigma_visible=args.sigma_visible,
        odom_noise=args.odom_noise,
        decay_factor=args.decay_factor,
        large_uncertainty=args.large_uncertainty,
    )
    route_cfg = RouteConfig(success_radius=args.success_radius)
    gate_cfg = GateConfig(sigma_ale_threshold=args.gate_threshold)
    base_env_cfg = EnvConfig(
        bearing0_deg=args.bearing0_deg,
        range0=tuple(args.range0),
        obs_noise_std=args.env_obs_noise,
        dt=args.dt,
        turn_kp=args.turn_kp,
        base_forward=args.base_forward,
        max_yaw_rate=args.max_yaw_rate,
    )
    intervention_cfgs = {
        "belief_only": InterventionConfig(enabled=False),
        "human_intervention": InterventionConfig(
            enabled=True, human_bearing_noise_std_deg=args.human_bearing_noise_deg
        ),
    }

    master_rng = np.random.default_rng(args.seed)
    levels = severity_levels(args)

    rows: List[Dict] = []
    for severity, msl, odom in levels:
        env_cfg = make_env_cfg(base_env_cfg, msl, odom)
        seeds = [
            int(master_rng.integers(0, 2**31 - 1))
            for _ in range(args.episodes_per_level)
        ]
        for condition in CONDITIONS:
            metrics = run_condition(
                bank_cfg,
                route_cfg,
                env_cfg,
                gate_cfg,
                intervention_cfgs[condition],
                seeds,
                args.max_steps,
            )
            row = {
                "severity": severity,
                "mean_streak_len": msl,
                "env_odom_noise": odom,
                "condition": condition,
                **{k: metrics[k] for k in STUDY_METRIC_COLUMNS},
                "n_episodes": metrics["n_episodes"],
            }
            rows.append(row)
        print(
            f"severity={severity:.2f} (mean_streak_len={msl:.1f}, "
            f"env_odom_noise={odom:.3f}) done",
            flush=True,
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent
        / "results"
        / f"intervention_study_{datetime.datetime.now():%y%m%d_%H%M%S}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "severity",
        "mean_streak_len",
        "env_odom_noise",
        "condition",
    ] + STUDY_METRIC_COLUMNS + ["n_episodes"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nwrote {len(rows)} rows to {out_path}")

    print_summary(rows)

    if args.trace:
        severity, msl, odom = levels[-1]
        env_cfg = make_env_cfg(base_env_cfg, msl, odom)
        # Search a handful of candidate seeds for one that actually illustrates the
        # thesis: belief_only fails to reach the true goal (or false-advances) while
        # human_intervention, replayed on the SAME seed, reaches it -- rather than
        # printing whatever seed happens to land next, which may see no occlusion
        # at all and trigger neither condition's failure mode.
        chosen = None
        for _ in range(args.trace_candidates):
            candidate_seed = int(master_rng.integers(0, 2**31 - 1))
            belief_log = run_episode(
                bank_cfg,
                route_cfg,
                env_cfg,
                gate_cfg,
                np.random.default_rng(candidate_seed),
                args.max_steps,
                intervention_cfg=intervention_cfgs["belief_only"],
            )
            human_log = run_episode(
                bank_cfg,
                route_cfg,
                env_cfg,
                gate_cfg,
                np.random.default_rng(candidate_seed),
                args.max_steps,
                intervention_cfg=intervention_cfgs["human_intervention"],
            )
            belief_failed = belief_log.final_true_dist > route_cfg.success_radius
            human_succeeded = human_log.final_true_dist <= route_cfg.success_radius
            if any(human_log.human_active) and belief_failed and human_succeeded:
                chosen = (belief_log, human_log)
                break
        if chosen is None:
            print(
                f"\n(no trace candidate among {args.trace_candidates} seeds showed "
                "both belief_only failing and human_intervention succeeding -- "
                "showing the last candidate instead)"
            )
            chosen = (belief_log, human_log)
        belief_log, human_log = chosen
        print_trace(f"belief_only @ severity={severity:.2f} (most severe level)", belief_log)
        print_trace(
            f"human_intervention @ severity={severity:.2f} (most severe level)",
            human_log,
        )


if __name__ == "__main__":
    main()
