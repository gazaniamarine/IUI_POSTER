"""Integration project (next.md) Phase 3: exercises NavdpUpstreamPolicy's
replan/lookahead state machine and goal-body plumbing against a mocked
pointgoal_step and a server manager whose start()/stop() are stubbed out --
no real navdp_server subprocess, checkpoint, or vendored checkout needed."""

from unittest.mock import patch

import numpy as np

from sam_vla.core.types import GoalSpec, Observation, Pose
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy


def _make_policy(**kwargs) -> NavdpUpstreamPolicy:
    with patch(
        "sam_vla.vlm.navdp_upstream_server_manager.resolve_navdp_upstream_root",
        return_value="/fake/navdp_upstream",
    ):
        policy = NavdpUpstreamPolicy(checkpoint_path="/fake/ckpt.ckpt", **kwargs)
    policy._server.start = lambda: setattr(policy._server, "load_ms", 0.0)
    policy._server.stop = lambda: None
    policy._server.base_url = "http://127.0.0.1:8766"
    return policy


def _obs(step: int) -> Observation:
    return Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        frame_idx=step,
    )


_GOAL_SPEC = GoalSpec(
    goal_bbox_norm=(0.4, 0.4, 0.6, 0.6), obstacle_bboxes_norm=[], instruction_text="x"
)


def test_act_verbose_replans_every_step_by_default_and_forwards_goal_body():
    policy = _make_policy(replan_every=1, lookahead=0)
    policy.set_goal_body(forward=3.0, left=-1.0)

    calls = []

    def fake_pointgoal_step(
        base_url, rgb, depth, goal_forward, goal_left, timeout, obstacle_pixels=None
    ):
        calls.append((goal_forward, goal_left))
        trajectory = np.tile(np.array([1.0, 0.0], dtype=np.float32), (4, 1))
        return trajectory, np.array([0.5], dtype=np.float32)

    with patch(
        "sam_vla.policy.navdp_upstream_policy.pointgoal_step", fake_pointgoal_step
    ):
        for step in range(3):
            action, vla_result = policy.act_verbose(
                _obs(step), semantic=None, goal_spec=_GOAL_SPEC, step=step
            )
            assert vla_result["replanned"] is True

    assert calls == [(3.0, -1.0)] * 3


def test_act_verbose_replan_every_caches_trajectory_between_calls():
    policy = _make_policy(replan_every=3, lookahead=0)
    policy.set_goal_body(forward=2.0, left=0.0)

    call_count = {"n": 0}

    def fake_pointgoal_step(
        base_url, rgb, depth, goal_forward, goal_left, timeout, obstacle_pixels=None
    ):
        call_count["n"] += 1
        trajectory = np.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32
        )
        return trajectory, np.array([0.5], dtype=np.float32)

    with patch(
        "sam_vla.policy.navdp_upstream_policy.pointgoal_step", fake_pointgoal_step
    ):
        replanned_flags = []
        for step in range(6):
            _, vla_result = policy.act_verbose(
                _obs(step), semantic=None, goal_spec=_GOAL_SPEC, step=step
            )
            replanned_flags.append(vla_result["replanned"])

    # steps 0 and 3 trigger a fresh server call (step % replan_every == 0);
    # 1,2,4,5 reuse the cached trajectory, walking the lookahead index forward.
    assert replanned_flags == [True, False, False, True, False, False]
    assert call_count["n"] == 2


def test_act_verbose_requires_depth():
    policy = _make_policy()
    obs_no_depth = Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        frame_idx=0,
    )
    try:
        policy.act_verbose(obs_no_depth, semantic=None, goal_spec=_GOAL_SPEC, step=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_start_is_idempotent():
    policy = _make_policy()
    starts = {"n": 0}

    def counting_start():
        starts["n"] += 1

    policy._server.start = counting_start
    policy.start()
    policy.start()
    assert starts["n"] == 1
