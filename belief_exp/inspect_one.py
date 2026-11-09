#!/usr/bin/env python3
"""Runs ONE belief-bank config over one or more episodes and prints a
step-by-step trace, for eyeballing behavior directly (mu tracking, Sigma
growth on occlusion / reset on re-sighting, gate pauses) before committing to
a full sweep.py run.

Usage:
    conda run -n sam2 python belief_exp/inspect_one.py
    conda run -n sam2 python belief_exp/inspect_one.py --sigma-visible 1e-4 --env-obs-noise 0.3
"""

from __future__ import annotations

import argparse

import numpy as np

from metrics import compute_metrics
from multi_goal_metrics import compute_multi_goal_metrics
from multi_goal_scenario import run_multi_episode
from scenario import BankConfig, EnvConfig, GateConfig, RouteConfig, run_episode


def print_multi_trace(log) -> None:
    """Per-goal mu/Sigma/confidence trace plus a route advance-step timeline,
    the multi-goal analogue of print_trace."""
    header = f"{'t':>3}  {'active':>8}  " + "  ".join(
        f"{gid + '(vis,err,conf)':>28}" for gid in log.route_order
    )
    print(header)
    print("-" * len(header))
    for i in range(len(log.t)):
        row = [f"{log.t[i]:>3}", f"{(log.active_goal_id[i] or '-'):>8}"]
        for gid in log.route_order:
            mu = log.mu[i][gid]
            tg = log.true_goals[i][gid]
            err = float(np.linalg.norm(mu - tg))
            vis = "Y" if log.visible[i][gid] else "."
            conf = log.confidence[i][gid]
            row.append(f"{vis:>3} err={err:>6.2f} conf={conf:>4.2f}".rjust(28))
        print("  ".join(row))

    print(f"\nroute_order: {log.route_order}")
    if log.finished:
        print(
            f"route finished; final_route_index={log.final_route_index}/{len(log.route_order)}"
        )
    else:
        print(
            f"route NOT finished within max_steps; final_route_index={log.final_route_index}/{len(log.route_order)}"
        )
    for gid in log.route_order:
        step = log.advance_steps.get(gid)
        print(
            f"  {gid}: {'advanced at step ' + str(step) if step is not None else 'never reached'}"
        )


def print_trace(log) -> None:
    header = f"{'t':>3} {'vis':>4} {'pause':>6} {'true_x':>8} {'true_y':>8} {'mu_x':>8} {'mu_y':>8} {'err':>7} {'sig_x':>7} {'sig_y':>7} {'conf':>5}"
    print(header)
    print("-" * len(header))
    for i in range(len(log.t)):
        tg = log.true_goal[i]
        mu = log.mu[i]
        err = float(np.linalg.norm(mu - tg))
        sd = log.sigma_diag[i]
        print(
            f"{log.t[i]:>3} {'Y' if log.visible[i] else '.':>4} "
            f"{'PAUSE' if log.should_pause[i] else '':>6} "
            f"{tg[0]:>8.3f} {tg[1]:>8.3f} {mu[0]:>8.3f} {mu[1]:>8.3f} {err:>7.3f} "
            f"{np.sqrt(sd[0]):>7.3f} {np.sqrt(sd[1]):>7.3f} {log.confidence[i]:>5.2f}"
        )
    if log.advanced:
        print(
            f"\nRouteManager advanced at step {log.steps_to_advance}; final true dist = {log.final_true_dist:.3f}"
        )
    else:
        print(
            f"\nnever advanced within max_steps; final true dist = {log.final_true_dist:.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sigma-init", type=float, default=1.0)
    ap.add_argument("--sigma-visible", type=float, default=0.05)
    ap.add_argument("--odom-noise", type=float, default=0.02)
    ap.add_argument("--decay-factor", type=float, default=0.95)
    ap.add_argument("--large-uncertainty", type=float, default=1000.0)
    ap.add_argument("--success-radius", type=float, default=0.5)
    ap.add_argument("--gate-threshold", type=float, default=0.6)

    ap.add_argument("--env-obs-noise", type=float, default=0.1)
    ap.add_argument("--env-odom-noise", type=float, default=0.05)
    ap.add_argument(
        "--occlusion-mode", choices=["markov", "bernoulli"], default="markov"
    )
    ap.add_argument("--p-visible", type=float, default=0.5)
    ap.add_argument("--mean-streak-len", type=float, default=6.0)
    ap.add_argument("--bearing0-deg", type=float, default=60.0)
    ap.add_argument("--range0", type=float, nargs=2, default=[2.0, 10.0])
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=40)

    ap.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="episodes to run; only the first is traced",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--multi-goal",
        action="store_true",
        help="run multi_goal_scenario.run_multi_episode (N goals, fixed visiting order) "
        "instead of the single-goal scenario.run_episode",
    )
    ap.add_argument(
        "--n-goals",
        type=int,
        default=3,
        help="--multi-goal only: number of goals in the route",
    )
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
    env_cfg = EnvConfig(
        bearing0_deg=args.bearing0_deg,
        range0=tuple(args.range0),
        obs_noise_std=args.env_obs_noise,
        odom_noise_std=args.env_odom_noise,
        occlusion_mode=args.occlusion_mode,
        p_visible=args.p_visible,
        mean_streak_len=args.mean_streak_len,
        dt=args.dt,
    )

    if args.multi_goal:
        logs = [
            run_multi_episode(
                bank_cfg,
                route_cfg,
                env_cfg,
                gate_cfg,
                np.random.default_rng(args.seed + i),
                n_goals=args.n_goals,
                max_steps=args.max_steps,
            )
            for i in range(args.episodes)
        ]
        print_multi_trace(logs[0])

        metrics = compute_multi_goal_metrics(logs)
        print(f"\nmetrics over {args.episodes} episode(s):")
        for k, v in metrics.items():
            print(f"  {k:>22} = {v:.4f}")
        return

    logs = [
        run_episode(
            bank_cfg,
            route_cfg,
            env_cfg,
            gate_cfg,
            np.random.default_rng(args.seed + i),
            args.max_steps,
        )
        for i in range(args.episodes)
    ]
    print_trace(logs[0])

    metrics = compute_metrics(logs, success_radius=route_cfg.success_radius)
    print(f"\nmetrics over {args.episodes} episode(s):")
    for k, v in metrics.items():
        print(f"  {k:>22} = {v:.4f}")


if __name__ == "__main__":
    main()
