"""Study 1 (next.md) Phase 2: exercises _run_uncertainty_handoff's
orchestration logic (rotate/ask/drive/retry loop, SessionMode flip,
attempt/resolved/timing bookkeeping) against a fake env and
MockInternVLClient -- no real display, sim, or live VLM needed. The Tk
popup itself (_prompt_human_heading) is not exercised here; prompt_fn is
injected instead, per the docstring's stated reason for making it
injectable.
"""

import numpy as np

from sam_vla.core.goal_geometry import MESH_GOAL_ID
from sam_vla.core.types import Observation, Pose
from sam_vla.run_navdp_rollout import _run_uncertainty_handoff
from vl_direction.client import MockInternVLClient
from vl_direction.intervention import mode_flag
from vl_direction.intervention.mode_flag import SessionMode


class FakeEnv:
    """Duck-types MarsHabitatEnv's step/get_observation/get_full_observation
    (see sam_vla/core/tests/test_uncertainty_motion.py's FakeEnv for the
    same pattern) with a semantic buffer that's always-visible or
    never-visible, controlled at construction."""

    def __init__(self, goal_always_visible: bool):
        self.pose = None
        self.goal_always_visible = goal_always_visible

    def step(self, pose: Pose) -> None:
        self.pose = pose

    def get_observation(self, frame_idx: int) -> Observation:
        return Observation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=None,
            pose=self.pose,
            frame_idx=frame_idx,
        )

    def get_full_observation(self, frame_idx: int) -> Observation:
        semantic = np.full(
            (2, 2), MESH_GOAL_ID if self.goal_always_visible else 0, dtype=np.int32
        )
        return Observation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=None,
            pose=self.pose,
            frame_idx=frame_idx,
            semantic=semantic,
        )


def _fixed_heading_prompt(angle_deg):
    def _prompt(frame, sweep_description, attempt, max_retries):
        return angle_deg

    return _prompt


def test_handoff_resolves_on_first_attempt_when_goal_immediately_visible():
    env = FakeEnv(goal_always_visible=True)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)

    result = _run_uncertainty_handoff(
        env=env,
        pose=start,
        step=0,
        episode_id="test-episode",
        client=MockInternVLClient(canned_response="rocky terrain ahead"),
        current_uncertainty=0.5,
        uncertainty_threshold=0.3,
        dt=0.1,
        sweep_degrees_per_step=45.0,
        sweep_steps=2,
        max_units=2.0,
        max_retries=3,
        drive_v_fwd=0.5,
        lost_goal_min_px=1,
        prompt_fn=_fixed_heading_prompt(0.0),
    )

    assert result["resolved"] is True
    assert result["attempt"] == 1
    assert result["vlm_inference_ms"] >= 0.0
    assert result["human_decision_ms"] >= 0.0
    assert result["drive_ms"] >= 0.0
    assert isinstance(result["final_pose"], Pose)


def test_handoff_gives_up_after_max_retries_when_goal_never_visible():
    env = FakeEnv(goal_always_visible=False)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)

    result = _run_uncertainty_handoff(
        env=env,
        pose=start,
        step=0,
        episode_id="test-episode",
        client=MockInternVLClient(canned_response="no landmarks visible"),
        current_uncertainty=0.5,
        uncertainty_threshold=0.3,
        dt=0.1,
        sweep_degrees_per_step=45.0,
        sweep_steps=2,
        max_units=0.2,
        max_retries=1,
        drive_v_fwd=0.5,
        lost_goal_min_px=1,
        prompt_fn=_fixed_heading_prompt(90.0),
    )

    assert result["resolved"] is False
    assert result["attempt"] == 2  # initial attempt (0) + 1 retry, both exhausted


def test_handoff_flips_session_mode_during_call_and_resets_after():
    env = FakeEnv(goal_always_visible=True)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
    observed_mode_during_prompt = {}

    def _prompt_and_record(frame, sweep_description, attempt, max_retries):
        observed_mode_during_prompt["mode"] = mode_flag.get_current_mode()
        return 0.0

    assert mode_flag.get_current_mode() == SessionMode.AUTONOMOUS

    _run_uncertainty_handoff(
        env=env,
        pose=start,
        step=0,
        episode_id="test-episode",
        client=MockInternVLClient(canned_response="clear path"),
        current_uncertainty=0.5,
        uncertainty_threshold=0.3,
        dt=0.1,
        sweep_degrees_per_step=45.0,
        sweep_steps=2,
        max_units=2.0,
        max_retries=3,
        drive_v_fwd=0.5,
        lost_goal_min_px=1,
        prompt_fn=_prompt_and_record,
    )

    assert observed_mode_during_prompt["mode"] == SessionMode.HUMAN_INTERVENED
    assert mode_flag.get_current_mode() == SessionMode.AUTONOMOUS


def test_handoff_resets_session_mode_even_if_prompt_fn_raises():
    env = FakeEnv(goal_always_visible=True)
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)

    def _prompt_that_raises(frame, sweep_description, attempt, max_retries):
        raise RuntimeError("simulated UI failure")

    assert mode_flag.get_current_mode() == SessionMode.AUTONOMOUS
    try:
        _run_uncertainty_handoff(
            env=env,
            pose=start,
            step=0,
            episode_id="test-episode",
            client=MockInternVLClient(canned_response="clear path"),
            current_uncertainty=0.5,
            uncertainty_threshold=0.3,
            dt=0.1,
            sweep_degrees_per_step=45.0,
            sweep_steps=2,
            max_units=2.0,
            max_retries=3,
            drive_v_fwd=0.5,
            lost_goal_min_px=1,
            prompt_fn=_prompt_that_raises,
        )
    except RuntimeError:
        pass
    assert mode_flag.get_current_mode() == SessionMode.AUTONOMOUS
