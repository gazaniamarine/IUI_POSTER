"""Score a batch of MultiEpisodeLogs (all belief numbers sourced from navdp's own
SubgoalBeliefBank / RouteManager via multi_goal_scenario.run_multi_episode) into
calibration and route-level task metrics for one (bank_cfg, route_cfg, gate_cfg,
n_goals) combination.

Per-goal calibration formulas are identical to metrics.compute_metrics, just
pooled over every goal x every step x every episode instead of one goal x every
step x every episode -- the same numbers mean the same thing (e.g. "68% one-sigma
coverage" is still a per-(goal, step) sample statistic), just with more samples
per episode. Route-level metrics (completion_rate, mean_steps_per_leg) are new,
since scenario.py's single-goal route only ever has one leg.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from multi_goal_scenario import MultiEpisodeLog

_NOMINAL_1SIGMA = 0.6827
_NOMINAL_2SIGMA = 0.9545
_MIN_VAR = 1e-8


def compute_multi_goal_metrics(logs: List[MultiEpisodeLog]) -> Dict[str, float]:
    if not logs:
        raise ValueError("compute_multi_goal_metrics requires at least one episode log")

    err_visible: List[float] = []
    err_occluded: List[float] = []
    nll_terms: List[float] = []
    hit_1sigma: List[float] = []
    hit_2sigma: List[float] = []
    leg_steps: List[int] = []

    for log in logs:
        for mu_by_goal, true_by_goal, sigma_by_goal, visible_by_goal in zip(
            log.mu, log.true_goals, log.sigma_diag, log.visible
        ):
            for gid in log.route_order:
                mu = np.asarray(mu_by_goal[gid], dtype=np.float64)
                true_goal = np.asarray(true_by_goal[gid], dtype=np.float64)
                sigma_diag = np.asarray(sigma_by_goal[gid], dtype=np.float64)
                visible = bool(visible_by_goal[gid])

                err = mu - true_goal
                var = np.clip(sigma_diag, _MIN_VAR, None)

                (err_visible if visible else err_occluded).append(
                    float(np.linalg.norm(err))
                )

                nll = 0.5 * np.sum(err**2 / var + np.log(var))
                nll_terms.append(float(nll))

                std = np.sqrt(var)
                hit_1sigma.extend((np.abs(err) < std).astype(np.float64).tolist())
                hit_2sigma.extend((np.abs(err) < 2.0 * std).astype(np.float64).tolist())

        prev_step = 0
        for gid in log.route_order:
            if gid not in log.advance_steps:
                break  # route not completed past this point; remaining legs never happened
            leg_steps.append(log.advance_steps[gid] - prev_step)
            prev_step = log.advance_steps[gid]

    n_episodes = len(logs)
    n_finished = sum(1 for log in logs if log.finished)

    coverage_1sigma = float(np.mean(hit_1sigma)) if hit_1sigma else float("nan")
    coverage_2sigma = float(np.mean(hit_2sigma)) if hit_2sigma else float("nan")

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
        "completion_rate": n_finished / n_episodes,
        "mean_steps_per_leg": float(np.mean(leg_steps)) if leg_steps else float("nan"),
        "n_episodes": float(n_episodes),
    }


METRIC_COLUMNS = [
    "mean_err_visible",
    "mean_err_occluded",
    "calibration_nll",
    "coverage_1sigma",
    "coverage_2sigma",
    "coverage_deviation",
    "completion_rate",
    "mean_steps_per_leg",
]

__all__ = ["compute_multi_goal_metrics", "METRIC_COLUMNS"]
