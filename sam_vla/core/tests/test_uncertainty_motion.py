import math

import numpy as np
import pytest

from sam_vla.core.types import Observation, Pose
from sam_vla.core.uncertainty_motion import (
    drive_toward_heading,
    resolve_absolute_heading_deg,
    rotate_sweep,
    yaw_rate_toward_heading,
)


class FakeEnv:
    """Duck-types MarsHabitatEnv's step/get_observation/get_full_observation
    with no sim dependency: step() just records the pose it was teleported
    to, get_observation/get_full_observation report it back with a synthetic
    semantic buffer that flips "visible" after a configurable step count."""

    def __init__(self, semantic_after_step: int = 10_000):
        self.pose = None
        self.step_count = 0
        self.semantic_after_step = semantic_after_step

    def step(self, pose: Pose) -> None:
        self.pose = pose
        self.step_count += 1

    def get_observation(self, frame_idx: int) -> Observation:
        return Observation(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=None,
            pose=self.pose,
            frame_idx=frame_idx,
        )

    def get_full_observation(self, frame_idx: int) -> Observation:
        visible = self.step_count >= self.semantic_after_step
        semantic = np.full((4, 4), 1 if visible else 0, dtype=np.int32)
        return Observation(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=None,
            pose=self.pose,
            frame_idx=frame_idx,
            semantic=semantic,
        )


def _goal_visible(obs: Observation) -> bool:
    return bool(np.any(obs.semantic == 1))


@pytest.mark.parametrize(
    "current_yaw,target_yaw,expected_sign",
    [
        (0.0, math.pi / 2, 1.0),  # target is CCW of current -> positive yaw_rate
        (0.0, -math.pi / 2, -1.0),  # target is CW of current -> negative yaw_rate
    ],
)
def test_yaw_rate_toward_heading_turns_the_short_way(
    current_yaw, target_yaw, expected_sign
):
    rate = yaw_rate_toward_heading(
        current_yaw, target_yaw, turn_kp=1.0, max_yaw_rate=10.0
    )
    assert rate * expected_sign > 0.0


def test_yaw_rate_toward_heading_wraps_across_pi_boundary():
    # current just past +pi, target just past -pi: true error is small and
    # positive, not the ~2pi it'd be without wrapping.
    rate = yaw_rate_toward_heading(
        math.pi - 0.05, -math.pi + 0.05, turn_kp=1.0, max_yaw_rate=10.0
    )
    assert 0.0 < rate < 1.0


def test_yaw_rate_toward_heading_clamps_to_max():
    rate = yaw_rate_toward_heading(0.0, math.pi / 2, turn_kp=100.0, max_yaw_rate=0.5)
    assert rate == pytest.approx(0.5)


def test_yaw_rate_toward_heading_zero_at_target():
    rate = yaw_rate_toward_heading(1.2, 1.2, turn_kp=5.0, max_yaw_rate=10.0)
    assert rate == pytest.approx(0.0)


def test_resolve_absolute_heading_deg_front_is_reference_yaw():
    result = resolve_absolute_heading_deg(math.radians(30.0), 0.0)
    assert result == pytest.approx(30.0)


def test_resolve_absolute_heading_deg_right_subtracts():
    # Facing world yaw=0 (i.e. +x), human says "turn right 90" -> since
    # Pose.yaw is CCW-positive, a rightward/clockwise turn is a DEcrease.
    result = resolve_absolute_heading_deg(0.0, 90.0)
    assert result == pytest.approx(-90.0)


def test_resolve_absolute_heading_deg_left_adds():
    result = resolve_absolute_heading_deg(0.0, -90.0)
    assert result == pytest.approx(90.0)


def test_rotate_sweep_returns_one_frame_per_step():
    env = FakeEnv()
    start = Pose(x=1.0, y=0.0, z=2.0, yaw=0.0)
    frames = rotate_sweep(env, start, degrees_per_step=30.0, n_steps=5, dt=0.1)
    assert len(frames) == 5


def test_rotate_sweep_leaves_position_unchanged():
    env = FakeEnv()
    start = Pose(x=1.0, y=0.0, z=2.0, yaw=0.0)
    frames = rotate_sweep(env, start, degrees_per_step=45.0, n_steps=8, dt=0.1)
    final_pose, _ = frames[-1]
    assert final_pose.x == pytest.approx(start.x, abs=1e-6)
    assert final_pose.z == pytest.approx(start.z, abs=1e-6)


def test_rotate_sweep_accumulates_yaw():
    env = FakeEnv()
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    frames = rotate_sweep(env, start, degrees_per_step=45.0, n_steps=2, dt=0.1)
    final_pose, _ = frames[-1]
    assert final_pose.yaw == pytest.approx(math.radians(90.0), abs=1e-3)


def test_drive_toward_heading_stops_early_when_goal_becomes_visible():
    env = FakeEnv(semantic_after_step=3)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    final_pose, units, found = drive_toward_heading(
        env,
        start,
        heading_deg=90.0,
        max_units=5.0,
        goal_visible_fn=_goal_visible,
        v_fwd=0.5,
        dt=0.1,
    )
    assert found is True
    assert units == pytest.approx(0.5 * 0.1 * 3, abs=1e-6)


def test_drive_toward_heading_caps_at_max_units_when_never_found():
    env = FakeEnv(semantic_after_step=10_000)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    final_pose, units, found = drive_toward_heading(
        env,
        start,
        heading_deg=90.0,
        max_units=1.0,
        goal_visible_fn=_goal_visible,
        v_fwd=0.5,
        dt=0.1,
    )
    assert found is False
    assert units >= 1.0


def test_drive_toward_heading_steers_yaw_toward_target():
    env = FakeEnv(semantic_after_step=10_000)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    final_pose, _, _ = drive_toward_heading(
        env,
        start,
        heading_deg=90.0,
        max_units=2.0,
        goal_visible_fn=_goal_visible,
        v_fwd=0.5,
        turn_kp=1.4,
        dt=0.1,
    )
    # Started facing yaw=0, target is +90deg (pi/2) -- final yaw should have
    # moved toward (not away from) the target.
    assert 0.0 < final_pose.yaw <= math.pi / 2 + 1e-6
