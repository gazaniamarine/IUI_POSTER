"""
Standalone test for sam_vla.policy.exploration_policy.ExplorationPolicy: the leg
state machine (turn-to-heading -> cruise -> repeat), visited-cell tracking, VLM
hint routing (explore left/right, "this area's explored", blocked-direction after
backtrack), and stall-triggered backtracking -- all exercised with
vl_direction.client.MockInternVLClient, no Habitat-Sim / live model needed.

Usage:
    python scripts/vlm_nav_tests/test_exploration_policy.py
"""

import math

import numpy as np

from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Observation, Pose
from sam_vla.policy.exploration_policy import (
    ExplorationConfig,
    ExplorationPolicy,
    _project_ahead,
)
from vl_direction.client import MockInternVLClient

_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def _obs(pose: Pose, frame_idx: int) -> Observation:
    return Observation(rgb=_FRAME, depth=None, pose=pose, frame_idx=frame_idx)


def test_mark_and_is_explored():
    config = ExplorationConfig(task_str="find rocks", cell_size_m=1.5)
    policy = ExplorationPolicy(config, "ep-mark", client=MockInternVLClient("FRONT"))

    policy.mark_explored(0.0, 0.0)
    assert policy.is_explored(0.4, 0.4), "nearby point in the same cell should read explored"
    assert not policy.is_explored(10.0, 10.0), "far point should not read explored"
    print("[test] mark_and_is_explored PASSED")


def test_leg_turns_and_cruises():
    client = MockInternVLClient(canned_response="LEFT")
    config = ExplorationConfig(task_str="find rocks", leg_length_m=3.0, seed=0)
    policy = ExplorationPolicy(config, "ep-leg", client=client)

    pose = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    for step in range(40):
        obs = _obs(pose, step)
        action, info = policy.act_verbose(obs, None, None, step)
        pose = integrate_mars(pose, action, dt=1.0)

    assert len(client.calls) >= 3, "expected multiple leg-start VLM queries over 40 steps"
    assert math.hypot(pose.x, pose.z) > 2.0, "rover should have made real forward progress"
    assert abs(pose.yaw) > math.radians(5.0), "LEFT directives should have turned the rover"
    assert info["visited_cells"] > 1
    print(
        f"[test] leg_turns_and_cruises PASSED "
        f"(vl_calls={len(client.calls)}, final_pose=({pose.x:.2f},{pose.z:.2f}), "
        f"yaw_deg={math.degrees(pose.yaw):.1f}, visited_cells={info['visited_cells']})"
    )


def test_explored_area_hint():
    config = ExplorationConfig(task_str="find rocks", leg_length_m=2.0, seed=1)
    client = MockInternVLClient(canned_response="FRONT")
    policy = ExplorationPolicy(config, "ep-hint-explored", client=client)

    pose = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    ahead_x, ahead_z = _project_ahead(pose, config.leg_length_m * 0.5)
    policy.mark_explored(ahead_x, ahead_z)

    policy._query_direction(_obs(pose, 0))
    prompt = client.calls[-1]["prompt"]
    assert "this area's explored" in prompt, f"expected explored hint in prompt, got: {prompt!r}"
    print("[test] explored_area_hint PASSED")


def test_command_hints():
    for command, expected in (("left", "explore the left"), ("right", "explore the right")):
        config = ExplorationConfig(task_str="find rocks", seed=2)
        client = MockInternVLClient(canned_response="FRONT")
        policy = ExplorationPolicy(config, f"ep-hint-{command}", client=client)
        policy.set_command(command)

        policy._query_direction(_obs(Pose(x=0.0, y=0.0, z=0.0, yaw=0.0), 0))
        prompt = client.calls[-1]["prompt"]
        assert expected in prompt, f"expected {expected!r} in prompt, got: {prompt!r}"
    print("[test] command_hints PASSED")


def test_stall_triggers_backtrack_with_blocked_hint():
    config = ExplorationConfig(
        task_str="find rocks",
        stall_window_steps=5,
        stall_distance_m=0.5,
        leg_length_m=10.0,
        seed=3,
    )
    client = MockInternVLClient(canned_response="FRONT")
    policy = ExplorationPolicy(config, "ep-stall", client=client)

    frozen_pose = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    info = None
    for step in range(config.stall_window_steps):
        obs = _obs(frozen_pose, step)
        _, info = policy.act_verbose(obs, None, None, step)

    assert policy._leg_kind == "backtrack", f"expected backtrack after stalling, got info={info}"
    assert policy._leg.direction.value == "BACK", "backtrack leg should be a BACK-direction leg"

    hint = policy._next_hint(_obs(frozen_pose, config.stall_window_steps))
    assert hint is not None and "blocked" in hint and "choose a different direction" in hint, (
        f"expected a blocked-direction hint queued after backtrack, got: {hint!r}"
    )
    print(f"[test] stall_triggers_backtrack_with_blocked_hint PASSED (hint={hint!r})")


if __name__ == "__main__":
    test_mark_and_is_explored()
    test_leg_turns_and_cruises()
    test_explored_area_hint()
    test_command_hints()
    test_stall_triggers_backtrack_with_blocked_hint()
    print("[test] ALL PASSED")
