"""Autonomous exploration policy driven by vl_direction's "exploration" mode.

vl_direction.directive_engine.query("exploration", ...) is a pure function -- it
returns a LEFT/RIGHT/FRONT/BACK prior but never moves anything (see
vl_direction/DESIGN.md's hard interface boundary). This module is the caller-side
orchestration that turns that prior into actual motion: a single repeating "leg"
(turn to a VLM-chosen heading, cruise forward a jittered distance, repeat) shared
by all three commandable patterns (explore_area/explore_left/explore_right), plus
a visited-cell tracker feeding the "this area's explored" hint that
vl_direction/prompts/exploration_prompt.py's own few-shot example already expects,
and stall-triggered backtracking for dead ends. Obstacle avoidance is deliberately
NOT implemented here -- per the composition pattern already used in
run_navdp_rollout.py (policy.act_verbose -> safety_filter -> CbfObstacleAvoidance),
this policy only needs to produce a plain forward-biased Action for CBF to override.
"""

import math
import random
import typing
from collections import deque
from dataclasses import dataclass
from typing import Optional

from sam_vla.core.types import Action, GoalSpec, Observation, Pose
from vl_direction.directive_engine import query as vl_query
from vl_direction.schemas import Direction, ExplorationContext

Command = typing.Literal["area", "left", "right"]
LegKind = typing.Literal["normal", "backtrack"]


def _wrap_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _project_ahead(pose: Pose, distance: float) -> tuple[float, float]:
    """Point `distance` ahead of pose along its heading, in pose_integrator's
    forward-direction convention (-sin(yaw), -cos(yaw)) -- used only to decide
    whether the ground the rover is about to enter is already marked explored."""
    return pose.x - distance * math.sin(pose.yaw), pose.z - distance * math.cos(pose.yaw)


@dataclass
class ExplorationConfig:
    task_str: str
    leg_length_m: float = 3.0
    leg_length_jitter_m: float = 1.0
    cruise_speed: float = 1.0
    turn_deg: float = 60.0
    turn_jitter_deg: float = 20.0
    turn_rate_deg_per_step: float = 30.0
    cell_size_m: float = 1.5
    stall_window_steps: int = 15
    stall_distance_m: float = 0.5
    backtrack_distance_m: float = 2.0
    seed: Optional[int] = None


@dataclass
class _LegState:
    direction: Direction = Direction.FRONT
    remaining_m: float = 0.0
    turning: bool = False
    target_yaw: Optional[float] = None


class ExplorationPolicy:
    """NavigationPolicy-shaped (act/act_verbose), same (Action, dict) convention as
    NavdpPolicy and QwenDiscreteDirectionPolicy. `client` is a vl_direction
    InternVLClient (e.g. vl_direction.client.get_client("qwen") or
    MockInternVLClient for tests); None defers to vl_direction's own default."""

    def __init__(self, config: ExplorationConfig, episode_id: str, client=None):
        self.config = config
        self.episode_id = episode_id
        self.client = client

        self._rng = random.Random(config.seed)
        self._visited: set[tuple[int, int]] = set()
        self._pose_history: deque = deque(maxlen=config.stall_window_steps)
        self._last_pose: Optional[tuple[float, float]] = None
        self._leg = _LegState()
        self._leg_kind: LegKind = "normal"
        self._last_failed_direction: Optional[Direction] = None
        self._last_query_hint: Optional[str] = None
        self._command: Command = "area"

    # -- visited-area tracking -------------------------------------------------

    def _cell(self, x: float, z: float) -> tuple[int, int]:
        c = self.config.cell_size_m
        return (round(x / c), round(z / c))

    def mark_explored(self, x: float, z: float) -> None:
        self._visited.add(self._cell(x, z))

    def is_explored(self, x: float, z: float) -> bool:
        return self._cell(x, z) in self._visited

    # -- commandable patterns, all funnel into the shared _step() leg engine --

    def set_command(self, command: Command) -> None:
        if command not in ("area", "left", "right"):
            raise ValueError(
                f"unknown command {command!r}, expected 'area', 'left', or 'right'"
            )
        self._command = command

    def explore_area(self, obs: Observation) -> Action:
        self._command = "area"
        return self._step(obs)

    def explore_left(self, obs: Observation) -> Action:
        self._command = "left"
        return self._step(obs)

    def explore_right(self, obs: Observation) -> Action:
        self._command = "right"
        return self._step(obs)

    def backtrack(self, obs: Observation) -> Action:
        """Manual/forced backtrack trigger (mirrors the automatic stall-triggered
        path in _step) -- exposed for callers that detect a dead end some other
        way (e.g. a CBF hard-gate stuck at zero forward velocity for too long)."""
        self._last_failed_direction = self._leg.direction
        self._pose_history.clear()
        self._start_leg(obs, forced_direction=Direction.BACK, kind="backtrack")
        return self._advance_leg(obs, moved=0.0)

    # -- VLM query ---------------------------------------------------------

    def _next_hint(self, obs: Observation) -> Optional[str]:
        if self._last_failed_direction is not None:
            return (
                f"the {self._last_failed_direction.value.lower()} direction was "
                "blocked, choose a different direction"
            )
        ahead_x, ahead_z = _project_ahead(obs.pose, self.config.leg_length_m * 0.5)
        if self.is_explored(ahead_x, ahead_z):
            return "this area's explored"
        if self._command == "left":
            return "explore the left"
        if self._command == "right":
            return "explore the right"
        return None

    def _query_direction(self, obs: Observation) -> Direction:
        hint = self._next_hint(obs)
        self._last_query_hint = hint
        context = ExplorationContext(task_str=self.config.task_str, vague_hint=hint)
        result = vl_query(
            "exploration", [obs.rgb], context, self.episode_id, client=self.client
        )
        self._last_failed_direction = None
        if not result.parse_ok or result.direction is None:
            return Direction.FRONT
        return result.direction

    # -- leg state machine ---------------------------------------------------

    def _start_leg(
        self,
        obs: Observation,
        forced_direction: Optional[Direction] = None,
        kind: LegKind = "normal",
    ) -> None:
        direction = (
            forced_direction if forced_direction is not None else self._query_direction(obs)
        )
        turn_jitter = self._rng.uniform(-self.config.turn_jitter_deg, self.config.turn_jitter_deg)
        if direction == Direction.LEFT:
            delta_deg = self.config.turn_deg + turn_jitter
        elif direction == Direction.RIGHT:
            delta_deg = -(self.config.turn_deg + turn_jitter)
        elif direction == Direction.BACK:
            delta_deg = 180.0 + turn_jitter
        else:
            delta_deg = turn_jitter * 0.3  # FRONT: small organic wander only

        if kind == "backtrack":
            remaining = max(0.5, self.config.backtrack_distance_m)
        else:
            length_jitter = self._rng.uniform(
                -self.config.leg_length_jitter_m, self.config.leg_length_jitter_m
            )
            remaining = max(0.5, self.config.leg_length_m + length_jitter)

        self._leg = _LegState(
            direction=direction,
            remaining_m=remaining,
            turning=abs(delta_deg) > 1.0,
            target_yaw=_wrap_rad(obs.pose.yaw + math.radians(delta_deg)),
        )
        self._leg_kind = kind

    def _advance_leg(self, obs: Observation, moved: float) -> Action:
        if not (self._leg.remaining_m <= 0.0 and not self._leg.turning):
            self._leg.remaining_m -= moved

        if self._leg.turning:
            yaw_err = _wrap_rad(self._leg.target_yaw - obs.pose.yaw)
            if abs(yaw_err) <= math.radians(3.0):
                self._leg.turning = False
                return Action(v_fwd=self.config.cruise_speed, v_lat=0.0, yaw_rate=0.0)
            max_rate = math.radians(self.config.turn_rate_deg_per_step)
            yaw_rate = max(-max_rate, min(max_rate, yaw_err))
            return Action(v_fwd=0.2, v_lat=0.0, yaw_rate=yaw_rate)

        return Action(v_fwd=self.config.cruise_speed, v_lat=0.0, yaw_rate=0.0)

    def _step(self, obs: Observation) -> Action:
        self.mark_explored(obs.pose.x, obs.pose.z)

        if self._last_pose is None:
            moved = 0.0
        else:
            moved = math.hypot(
                obs.pose.x - self._last_pose[0], obs.pose.z - self._last_pose[1]
            )
        self._last_pose = (obs.pose.x, obs.pose.z)

        if self._leg_kind == "normal":
            self._pose_history.append((obs.frame_idx, obs.pose.x, obs.pose.z))
            if len(self._pose_history) == self._pose_history.maxlen:
                _, x0, z0 = self._pose_history[0]
                _, x1, z1 = self._pose_history[-1]
                if math.hypot(x1 - x0, z1 - z0) < self.config.stall_distance_m:
                    self._last_failed_direction = self._leg.direction
                    self._pose_history.clear()
                    self._start_leg(obs, forced_direction=Direction.BACK, kind="backtrack")
                    return self._advance_leg(obs, moved=0.0)

        if self._leg.remaining_m <= 0.0 and not self._leg.turning:
            self._start_leg(obs, kind="normal")
            return self._advance_leg(obs, moved=0.0)

        return self._advance_leg(obs, moved=moved)

    # -- NavigationPolicy-shaped entrypoint ----------------------------------

    def act_verbose(
        self, obs: Observation, semantic, goal_spec: GoalSpec, step: int
    ) -> tuple[Action, dict]:
        action = self._step(obs)
        info = {
            "command": self._command,
            "leg_kind": self._leg_kind,
            "leg_direction": self._leg.direction.value,
            "leg_remaining_m": round(self._leg.remaining_m, 3),
            "leg_turning": self._leg.turning,
            "visited_cells": len(self._visited),
            "last_query_hint": self._last_query_hint,
        }
        return action, info

    def act(self, obs: Observation, goal_spec: GoalSpec) -> Action:
        return self.act_verbose(obs, None, goal_spec, obs.frame_idx)[0]


if __name__ == "__main__":
    from vl_direction.client import MockInternVLClient

    config = ExplorationConfig(task_str="find the sample site", seed=0)
    policy = ExplorationPolicy(config, episode_id="demo", client=MockInternVLClient(canned_response="LEFT"))

    pose = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    for step in range(20):
        obs = Observation(rgb=__import__("numpy").zeros((4, 4, 3), dtype="uint8"), depth=None, pose=pose, frame_idx=step)
        action, info = policy.act_verbose(obs, None, None, step)
        print(step, info, action)
        from sam_vla.core.pose_integrator import integrate_mars
        pose = integrate_mars(pose, action, dt=1.0)
