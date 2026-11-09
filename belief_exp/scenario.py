"""Closed-loop synthetic episode simulator driving the navdp.SubgoalBeliefBank.

Every belief number in an EpisodeLog (mu, Sigma, confidence, RouteManager's advance
decision) comes directly from navdp's own SubgoalBeliefBank / RouteManager classes.
The only harness-authored logic is generic robot control (common.p_controller) and the
"pause and scan when sigma_ale is high" decision threshold -- a policy that consumes
the bank's own uncertainty output, since navdp has no such policy defined for the
no-RelationalBelief case (see plan Context, point 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from common import (
    GOAL_ID,
    RouteManager,
    SubgoalBeliefBank,
    ego_motion_true,
    p_controller,
    sigma_ale_from_bank,
    strength_from_sigma_ale,
)


@dataclass
class BankConfig:
    """The tunable surface of SubgoalBeliefBank -- covariance/confidence dynamics."""

    sigma_init: float = 1.0
    sigma_visible: float = 0.05
    odom_noise: float = 0.02
    decay_factor: float = 0.95
    large_uncertainty: float = 1000.0


@dataclass
class RouteConfig:
    success_radius: float = 0.5


@dataclass
class GateConfig:
    sigma_ale_threshold: float = 0.6
    scan_yaw_rate: float = 0.6
    strength_sigma_low: float = 0.05
    strength_sigma_high: float = 1.0
    strength_min: float = 0.05
    strength_max: float = 0.65


@dataclass
class EnvConfig:
    """Ground-truth scenario randomization -- NOT belief-bank parameters.

    env_odom_noise_std is the true noise corrupting the odometry the bank is TOLD about;
    bank_cfg.odom_noise is the bank's own belief about how noisy that is (used only to
    inflate Sigma). Sweeping the two independently is the point of the calibration test.
    """

    bearing0_deg: float = 60.0
    range0: tuple = (2.0, 10.0)
    obs_noise_std: float = 0.1
    odom_noise_std: float = 0.05
    occlusion_mode: str = "markov"  # "markov" or "bernoulli"
    p_visible: float = 0.5  # bernoulli mode
    mean_streak_len: float = 6.0  # markov mode: mean steps per visible/occluded streak
    dt: float = 0.5
    turn_kp: float = 1.4
    base_forward: float = 0.5
    max_yaw_rate: float = 1.0
    seed: int = 0


@dataclass
class InterventionConfig:
    """Switches steering from the belief mean to a human-provided bearing once
    the bank's own sigma_ale crosses gate_cfg.sigma_ale_threshold -- reusing that
    existing (previously dead, see the should_pause comment below) threshold field
    rather than adding a second one.

    The point of this study: sigma_ale climbing under heavy occlusion + odometry
    noise reflects IRREDUCIBLE scenario noise (see README's "why mean has no sweep
    range") -- no sigma_init/sigma_visible/odom_noise/decay_factor tuning fixes a
    dead-reckoned mu that has drifted during a long occlusion streak. A human
    watching the raw feed can still roughly judge the true bearing without
    integrating that drift, so this models a teleoperator taking over steering
    only while the belief is untrustworthy, not a better sensor or a smarter bank.
    """

    enabled: bool = False
    human_bearing_noise_std_deg: float = 8.0


@dataclass
class EpisodeLog:
    t: List[int] = field(default_factory=list)
    true_goal: List[np.ndarray] = field(default_factory=list)
    mu: List[np.ndarray] = field(default_factory=list)
    sigma_diag: List[np.ndarray] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    visible: List[bool] = field(default_factory=list)
    should_pause: List[bool] = field(default_factory=list)
    human_active: List[bool] = field(default_factory=list)
    advanced: bool = False
    steps_to_advance: Optional[int] = None
    final_true_dist: float = float("nan")


class _OcclusionProcess:
    """Per-step visibility sampler: independent Bernoulli or bursty Markov streaks."""

    def __init__(self, env_cfg: EnvConfig, rng: np.random.Generator):
        self.mode = env_cfg.occlusion_mode
        self.p_visible = float(env_cfg.p_visible)
        self.mean_streak_len = max(float(env_cfg.mean_streak_len), 1.0)
        self.rng = rng
        self._state_visible = True  # episodes start on a real sighting

    def step(self) -> bool:
        if self.mode == "bernoulli":
            return bool(self.rng.random() < self.p_visible)
        # markov: geometric holding time with mean `mean_streak_len` in either state
        if self.rng.random() < (1.0 / self.mean_streak_len):
            self._state_visible = not self._state_visible
        return self._state_visible


def run_episode(
    bank_cfg: BankConfig,
    route_cfg: RouteConfig,
    env_cfg: EnvConfig,
    gate_cfg: GateConfig,
    rng: np.random.Generator,
    max_steps: int = 40,
    intervention_cfg: Optional[InterventionConfig] = None,
) -> EpisodeLog:
    bearing0 = math.radians(rng.uniform(-env_cfg.bearing0_deg, env_cfg.bearing0_deg))
    range0 = float(rng.uniform(*env_cfg.range0))
    true_goal = np.array(
        [range0 * math.cos(bearing0), range0 * math.sin(bearing0)], dtype=np.float32
    )

    bank = SubgoalBeliefBank(
        [GOAL_ID],
        sigma_init=bank_cfg.sigma_init,
        sigma_visible=bank_cfg.sigma_visible,
        odom_noise=bank_cfg.odom_noise,
        decay_factor=bank_cfg.decay_factor,
        large_uncertainty=bank_cfg.large_uncertainty,
    )
    route = RouteManager([GOAL_ID], success_radius=route_cfg.success_radius)
    occlusion = _OcclusionProcess(env_cfg, rng)

    log = EpisodeLog()
    prev_dx, prev_dy, prev_dtheta = (
        0.0,
        0.0,
        0.0,
    )  # first step's odom_delta: no prior motion

    for t in range(max_steps):
        odom_noise_xy = float(env_cfg.odom_noise_std)
        odom_noise_th = float(env_cfg.odom_noise_std) * 0.5
        noisy_odom = [
            prev_dx + float(rng.normal(0.0, odom_noise_xy)),
            prev_dy + float(rng.normal(0.0, odom_noise_xy)),
            prev_dtheta + float(rng.normal(0.0, odom_noise_th)),
        ]

        visible = occlusion.step()
        if visible:
            noise = rng.normal(0.0, env_cfg.obs_noise_std, size=2).astype(np.float32)
            obs = {
                GOAL_ID: {
                    "visible": True,
                    "position": true_goal + noise,
                    "confidence": 1.0,
                }
            }
        else:
            obs = {GOAL_ID: {"visible": False, "position": None, "confidence": 0.0}}

        bank.update(obs, odom_delta=noisy_odom, step=t)
        status = route.update(robot_position=[0.0, 0.0], belief_bank=bank)

        slot = bank.get(GOAL_ID)
        mu = np.asarray(slot.mu[:2], dtype=np.float32)
        sigma_diag = np.array([slot.Sigma[0, 0], slot.Sigma[1, 1]], dtype=np.float32)
        sigma_ale = sigma_ale_from_bank(bank, GOAL_ID)

        log.t.append(t)
        log.true_goal.append(true_goal.copy())
        log.mu.append(mu.copy())
        log.sigma_diag.append(sigma_diag.copy())
        log.confidence.append(float(slot.confidence))
        log.visible.append(visible)

        should_pause = (
            False  # sigma_ale > gate_cfg.sigma_ale_threshold # exchange this later ?
        )
        log.should_pause.append(should_pause)

        human_active = bool(
            intervention_cfg
            and intervention_cfg.enabled
            and sigma_ale > gate_cfg.sigma_ale_threshold
        )
        log.human_active.append(human_active)

        if bool(status["advanced"]):
            log.advanced = True
            log.steps_to_advance = t
            log.final_true_dist = float(np.linalg.norm(true_goal))
            return log

        if should_pause:
            v_fwd, v_lat, yaw = 0.0, 0.0, float(gate_cfg.scan_yaw_rate)
        else:
            if human_active:
                # Human steers off the raw true bearing (own small, non-accumulating
                # noise) instead of the possibly drift-corrupted belief mean.
                true_bearing = math.atan2(float(true_goal[1]), float(true_goal[0]))
                noisy_bearing = true_bearing + rng.normal(
                    0.0, math.radians(intervention_cfg.human_bearing_noise_std_deg)
                )
                steer_target = np.array(
                    [math.cos(noisy_bearing), math.sin(noisy_bearing)],
                    dtype=np.float32,
                )
            else:
                steer_target = mu
            v_fwd, v_lat, yaw = p_controller(
                steer_target, env_cfg.turn_kp, env_cfg.base_forward, env_cfg.max_yaw_rate
            )
            strength = strength_from_sigma_ale(
                sigma_ale,
                sigma_low=gate_cfg.strength_sigma_low,
                sigma_high=gate_cfg.strength_sigma_high,
                strength_min=gate_cfg.strength_min,
                strength_max=gate_cfg.strength_max,
            )
            v_fwd *= 1.0 - strength  # higher uncertainty -> more caution -> slower

        true_goal = ego_motion_true(true_goal, v_fwd, v_lat, yaw, env_cfg.dt)
        prev_dx, prev_dy, prev_dtheta = (
            v_fwd * env_cfg.dt,
            v_lat * env_cfg.dt,
            yaw * env_cfg.dt,
        )

    log.final_true_dist = float(np.linalg.norm(true_goal))
    return log
