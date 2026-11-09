"""Score a batch of EpisodeLogs (all belief numbers sourced from navdp's own
SubgoalBeliefBank / RouteManager via scenario.run_episode) into calibration and
task-performance metrics for one (bank_cfg, route_cfg, gate_cfg) combination.

Combining metrics across DIFFERENT configs (z-normalized "combined_score" for the
leaderboard) happens in sweep.py, not here -- this module only ever looks at one
config's episodes at a time, so its numbers are always directly interpretable on
their own (e.g. "68% one-sigma coverage" means the same thing regardless of what
else is in the sweep).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from scenario import EpisodeLog

_NOMINAL_1SIGMA = 0.6827
_NOMINAL_2SIGMA = 0.9545
_MIN_VAR = 1e-8


def compute_metrics(logs: List[EpisodeLog], success_radius: float) -> Dict[str, float]:
    if not logs:
        raise ValueError("compute_metrics requires at least one episode log")

    err_visible: List[float] = []
    err_occluded: List[float] = []
    nll_terms: List[float] = []
    hit_1sigma: List[float] = []
    hit_2sigma: List[float] = []

    for log in logs:
        for mu, true_goal, sigma_diag, visible in zip(
            log.mu, log.true_goal, log.sigma_diag, log.visible
        ):
            err = np.asarray(mu, dtype=np.float64) - np.asarray(
                true_goal, dtype=np.float64
            )
            var = np.clip(np.asarray(sigma_diag, dtype=np.float64), _MIN_VAR, None)

            (err_visible if visible else err_occluded).append(
                float(np.linalg.norm(err))
            )

            nll = 0.5 * np.sum(err**2 / var + np.log(var))
            nll_terms.append(float(nll))

            std = np.sqrt(var)
            hit_1sigma.extend((np.abs(err) < std).astype(np.float64).tolist())
            hit_2sigma.extend((np.abs(err) < 2.0 * std).astype(np.float64).tolist())

    n_episodes = len(logs)
    advanced = [log for log in logs if log.advanced]
    n_advanced = len(advanced)
    false_advances = [
        log for log in advanced if log.final_true_dist > 2.0 * float(success_radius)
    ]

    coverage_1sigma = float(np.mean(hit_1sigma)) if hit_1sigma else float("nan")
    coverage_2sigma = float(np.mean(hit_2sigma)) if hit_2sigma else float("nan")

    # Ground-truth reachability: was the robot actually within success_radius of
    # the true goal at episode end, regardless of whether RouteManager's
    # belief-mean-based "advanced" flag agreed. advance_rate/false_advance_rate
    # above are about the belief being fooled; this is about the robot itself.
    true_success_rate = float(
        np.mean(
            [log.final_true_dist <= float(success_radius) for log in logs]
        )
    )
    human_active_frac = float(
        np.mean(
            [
                float(np.mean(log.human_active)) if log.human_active else 0.0
                for log in logs
            ]
        )
    )

    return {
        "mean_err_visible": (
            float(np.mean(err_visible)) if err_visible else float("nan")
        ),
        "mean_err_occluded": (
            float(np.mean(err_occluded)) if err_occluded else float("nan")
        ),
        "calibration_nll": float(np.mean(nll_terms)) if nll_terms else float("nan"),
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
        "coverage_deviation": abs(coverage_1sigma - _NOMINAL_1SIGMA)
        + abs(coverage_2sigma - _NOMINAL_2SIGMA),
        "mean_final_dist": float(np.mean([log.final_true_dist for log in logs])),
        "advance_rate": n_advanced / n_episodes,
        "mean_steps_to_advance": (
            float(np.mean([log.steps_to_advance for log in advanced]))
            if advanced
            else float("nan")
        ),
        "false_advance_rate": (
            (len(false_advances) / n_advanced) if n_advanced else float("nan")
        ),
        "true_success_rate": true_success_rate,
        "human_active_frac": human_active_frac,
        "n_episodes": float(n_episodes),
    }


__all__ = ["compute_metrics"]
