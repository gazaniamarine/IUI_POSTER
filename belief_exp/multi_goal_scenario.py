"""N-goal sibling to scenario.py: N independent synthetic goals sharing one
SubgoalBeliefBank + one RouteManager, visited in a fixed order (definition
order). Every belief number (mu, Sigma, confidence, RouteManager's advance
decisions) comes directly from navdp's own SubgoalBeliefBank / RouteManager
classes, same as scenario.py -- this file only adds the N-goal bookkeeping
scenario.py doesn't need for a single goal. scenario.py itself is untouched,
so all 21 existing exp.py experiments and their CSV schemas stay unaffected.

Goals are STATIONARY in the world (unlike scenario.py's single
ego_motion_true-driven goal, which only exists because there's nothing else
to track): each step the robot steers via p_controller toward whichever goal
is currently active, then that SAME executed motion is applied via
ego_motion_true to EVERY goal's true position -- all fixed world points shift
identically in the robot's own frame as it drives, exactly like a real
rollout would see several stationary rocks' apparent positions change
together as the rover moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from common import (
    RouteManager,
    SubgoalBeliefBank,
    ego_motion_true,
    p_controller,
    sigma_ale_from_bank,
    strength_from_sigma_ale,
)
from scenario import BankConfig, EnvConfig, GateConfig, RouteConfig, _OcclusionProcess


def goal_id_for(i: int) -> str:
    return f"goal_{i}"


@dataclass
class MultiEpisodeLog:
    t: List[int] = field(default_factory=list)
    active_goal_id: List[str] = field(default_factory=list)
    true_goals: List[Dict[str, np.ndarray]] = field(default_factory=list)
    mu: List[Dict[str, np.ndarray]] = field(default_factory=list)
    sigma_diag: List[Dict[str, np.ndarray]] = field(default_factory=list)
    confidence: List[Dict[str, float]] = field(default_factory=list)
    visible: List[Dict[str, bool]] = field(default_factory=list)
    route_order: List[str] = field(default_factory=list)
    advance_steps: Dict[str, int] = field(default_factory=dict)
    finished: bool = False
    final_route_index: int = 0
    final_true_dists: Dict[str, float] = field(default_factory=dict)


def run_multi_episode(
    bank_cfg: BankConfig,
    route_cfg: RouteConfig,
    env_cfg: EnvConfig,
    gate_cfg: GateConfig,
    rng: np.random.Generator,
    n_goals: int = 3,
    max_steps: int = 200,
) -> MultiEpisodeLog:
    goal_ids = [goal_id_for(i) for i in range(n_goals)]

    true_goals: Dict[str, np.ndarray] = {}
    occlusions: Dict[str, _OcclusionProcess] = {}
    for gid in goal_ids:
        bearing0 = math.radians(
            rng.uniform(-env_cfg.bearing0_deg, env_cfg.bearing0_deg)
        )
        range0 = float(rng.uniform(*env_cfg.range0))
        true_goals[gid] = np.array(
            [range0 * math.cos(bearing0), range0 * math.sin(bearing0)], dtype=np.float32
        )
        occlusions[gid] = _OcclusionProcess(env_cfg, rng)

    bank = SubgoalBeliefBank(
        goal_ids,
        sigma_init=bank_cfg.sigma_init,
        sigma_visible=bank_cfg.sigma_visible,
        odom_noise=bank_cfg.odom_noise,
        decay_factor=bank_cfg.decay_factor,
        large_uncertainty=bank_cfg.large_uncertainty,
    )
    route = RouteManager(goal_ids, success_radius=route_cfg.success_radius)

    log = MultiEpisodeLog(route_order=list(goal_ids))
    prev_dx, prev_dy, prev_dtheta = 0.0, 0.0, 0.0

    for t in range(max_steps):
        odom_noise_xy = float(env_cfg.odom_noise_std)
        odom_noise_th = float(env_cfg.odom_noise_std) * 0.5
        noisy_odom = [
            prev_dx + float(rng.normal(0.0, odom_noise_xy)),
            prev_dy + float(rng.normal(0.0, odom_noise_xy)),
            prev_dtheta + float(rng.normal(0.0, odom_noise_th)),
        ]

        obs = {}
        visible_this_step: Dict[str, bool] = {}
        for gid in goal_ids:
            visible = occlusions[gid].step()
            visible_this_step[gid] = visible
            if visible:
                noise = rng.normal(0.0, env_cfg.obs_noise_std, size=2).astype(
                    np.float32
                )
                obs[gid] = {
                    "visible": True,
                    "position": true_goals[gid] + noise,
                    "confidence": 1.0,
                }
            else:
                obs[gid] = {"visible": False, "position": None, "confidence": 0.0}

        bank.update(obs, odom_delta=noisy_odom, step=t)
        status = route.update(robot_position=[0.0, 0.0], belief_bank=bank)

        log.t.append(t)
        log.true_goals.append({gid: true_goals[gid].copy() for gid in goal_ids})
        log.mu.append(
            {
                gid: np.asarray(bank.get(gid).mu[:2], dtype=np.float32).copy()
                for gid in goal_ids
            }
        )
        log.sigma_diag.append(
            {
                gid: np.array(
                    [bank.get(gid).Sigma[0, 0], bank.get(gid).Sigma[1, 1]],
                    dtype=np.float32,
                )
                for gid in goal_ids
            }
        )
        log.confidence.append(
            {gid: float(bank.get(gid).confidence) for gid in goal_ids}
        )
        log.visible.append(visible_this_step)

        # RouteManager.update's "active_goal" is the goal that was being
        # steered toward WHEN this step's advance decision was evaluated
        # (captured before any advance) -- the right thing to log.
        active_goal_id = status["active_goal"]
        log.active_goal_id.append(active_goal_id)
        if bool(status["advanced"]) and active_goal_id is not None:
            log.advance_steps[active_goal_id] = t

        if route.is_finished():
            log.finished = True
            log.final_route_index = route.get_route_index()
            log.final_true_dists = {
                gid: float(np.linalg.norm(true_goals[gid])) for gid in goal_ids
            }
            return log

        # "next_active_goal" is post-advance-aware: whichever goal is now
        # current, so a same-step advance re-targets immediately instead of
        # wasting a step steering at the goal just reached.
        steer_goal_id = status["next_active_goal"]
        steer_slot = bank.get(steer_goal_id)
        steer_mu = np.asarray(steer_slot.mu[:2], dtype=np.float32)
        sigma_ale = sigma_ale_from_bank(bank, steer_goal_id)
        v_fwd, v_lat, yaw = p_controller(
            steer_mu, env_cfg.turn_kp, env_cfg.base_forward, env_cfg.max_yaw_rate
        )
        strength = strength_from_sigma_ale(
            sigma_ale,
            sigma_low=gate_cfg.strength_sigma_low,
            sigma_high=gate_cfg.strength_sigma_high,
            strength_min=gate_cfg.strength_min,
            strength_max=gate_cfg.strength_max,
        )
        v_fwd *= 1.0 - strength

        for gid in goal_ids:
            true_goals[gid] = ego_motion_true(
                true_goals[gid], v_fwd, v_lat, yaw, env_cfg.dt
            )
        prev_dx, prev_dy, prev_dtheta = (
            v_fwd * env_cfg.dt,
            v_lat * env_cfg.dt,
            yaw * env_cfg.dt,
        )

    log.final_route_index = route.get_route_index()
    log.final_true_dists = {
        gid: float(np.linalg.norm(true_goals[gid])) for gid in goal_ids
    }
    return log
