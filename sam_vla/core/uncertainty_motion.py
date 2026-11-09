"""Study 1 (next.md) Phase 1: autonomous motion primitives shared by both
uncertainty-handoff conditions' plumbing (Phase 2 human-in-the-loop, Phase 3
autonomous baseline) -- a rotation sweep for capturing frames to show a
human/VLM, and a "drive toward a target heading" primitive. Neither exists
anywhere else in the repo: kb_teleop_vl.py only ever resumes manual WASD
after submit_heading(), and no rollout script does anything but keep its own
policy running through an uncertainty spike.

Pure/sim-facing: env is duck-typed (step(pose), get_observation(frame_idx),
get_full_observation(frame_idx), matching MarsHabitatEnv) so these are
testable against a fake env with no real habitat-sim dependency. No VLM or
UncertaintySession import here -- Phase 2 wires those in around these.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, Pose


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_rate_toward_heading(
    current_yaw: float,
    target_yaw: float,
    turn_kp: float,
    max_yaw_rate: float,
) -> float:
    """Proportional yaw-rate command steering current_yaw toward target_yaw
    (both radians, both Pose.yaw's CCW-from-+x convention). Wraps the error
    to (-pi, pi] first so it always turns the short way around. Pulled out
    of drive_toward_heading so the steering law itself is unit-testable
    without an env/sim dependency."""
    err = _wrap_to_pi(target_yaw - current_yaw)
    return float(np.clip(turn_kp * err, -max_yaw_rate, max_yaw_rate))


def resolve_absolute_heading_deg(reference_yaw: float, human_angle_deg: float) -> float:
    """Converts a human-supplied rover-front-relative heading (the convention
    UncertaintySession.submit_heading's angle_deg uses: 0=front, positive=
    clockwise/right -- matching kb_teleop_vl.py's numpad mapping,
    UNCERTAINTY_HEADING_KEYS) into an absolute world heading in degrees
    (Pose.yaw's CCW-positive-from-+x convention), anchored at reference_yaw
    (radians) -- the world yaw captured BEFORE rotate_sweep ran, i.e. the
    human's "front" when they were asked. Pose.yaw is CCW-positive, so a
    clockwise/rightward human angle subtracts from the reference yaw."""
    return math.degrees(reference_yaw) - human_angle_deg


def rotate_sweep(
    env,
    pose: Pose,
    degrees_per_step: float,
    n_steps: int,
    dt: float = 0.1,
    frame_idx_start: int = 0,
) -> list[tuple[Pose, np.ndarray]]:
    """In-place rotation sweep: n_steps steps of degrees_per_step each (sign
    picks direction; Pose.yaw is CCW-positive, see pose_integrator.py),
    capturing (pose, rgb) at each step -- frames to feed
    UncertaintySession.request_human_heading()/retry(). Pure rotation
    (v_fwd=v_lat=0), so the rover's (x, z) is unchanged when this returns;
    only its yaw and the captured frames matter to the caller."""
    action = Action(
        v_fwd=0.0, v_lat=0.0, yaw_rate=math.radians(degrees_per_step) / dt
    )
    cur_pose = pose
    frames: list[tuple[Pose, np.ndarray]] = []
    for i in range(n_steps):
        cur_pose = integrate_mars(cur_pose, action, dt)
        env.step(cur_pose)
        obs = env.get_observation(frame_idx=frame_idx_start + i)
        cur_pose = obs.pose
        frames.append((obs.pose, obs.rgb))
    return frames


def drive_toward_heading(
    env,
    pose: Pose,
    heading_deg: float,
    max_units: float,
    goal_visible_fn: Callable[..., bool],
    v_fwd: float = 0.5,
    turn_kp: float = 1.4,
    max_yaw_rate: float = 1.0,
    dt: float = 0.1,
    frame_idx_start: int = 0,
) -> tuple[Pose, float, bool]:
    """Drive forward at v_fwd, steering yaw toward the ABSOLUTE world heading
    heading_deg (degrees, Pose.yaw's convention) via yaw_rate_toward_heading,
    stopping early once goal_visible_fn(obs) reports the goal in view, or
    once max_units (m) of forward distance has been covered -- whichever
    comes first.

    heading_deg is absolute-world, not rover-front-relative: Phase 2 must
    resolve the human's UncertaintySession angle_deg (given relative to the
    reference yaw captured before rotate_sweep ran) into this frame before
    calling in here. goal_visible_fn receives the Observation from
    env.get_full_observation() (rgb+depth+pose+semantic) so callers can
    reuse the same MESH_GOAL_ID-mask check the main rollout loop uses.

    Returns (final_pose, units_covered, found).
    """
    target_yaw = math.radians(heading_deg)
    cur_pose = pose
    units_covered = 0.0
    found = False
    step = 0
    while units_covered < max_units and not found:
        yaw_rate = yaw_rate_toward_heading(
            cur_pose.yaw, target_yaw, turn_kp, max_yaw_rate
        )
        action = Action(v_fwd=v_fwd, v_lat=0.0, yaw_rate=yaw_rate)
        cur_pose = integrate_mars(cur_pose, action, dt)
        env.step(cur_pose)
        obs = env.get_full_observation(frame_idx=frame_idx_start + step)
        cur_pose = obs.pose
        units_covered += v_fwd * dt
        found = bool(goal_visible_fn(obs))
        step += 1
    return cur_pose, units_covered, found


if __name__ == "__main__":

    class _FakeEnv:
        """Minimal env stub duck-typing MarsHabitatEnv's step/get_observation/
        get_full_observation for exercising these primitives with no sim."""

        def __init__(self, semantic_after_step: int = 3):
            self.pose = None
            self._step_count = 0
            self._semantic_after_step = semantic_after_step

        def step(self, pose: Pose) -> None:
            self.pose = pose
            self._step_count += 1

        def get_observation(self, frame_idx: int):
            from sam_vla.core.types import Observation

            return Observation(
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                depth=None,
                pose=self.pose,
                frame_idx=frame_idx,
            )

        def get_full_observation(self, frame_idx: int):
            from sam_vla.core.types import Observation

            visible = self._step_count >= self._semantic_after_step
            semantic = np.full((4, 4), 1 if visible else 0, dtype=np.int32)
            return Observation(
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                depth=None,
                pose=self.pose,
                frame_idx=frame_idx,
                semantic=semantic,
            )

    env = _FakeEnv()
    start = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)

    frames = rotate_sweep(env, start, degrees_per_step=45.0, n_steps=8, dt=0.1)
    print(f"rotate_sweep: {len(frames)} frames, final yaw={frames[-1][0].yaw:.3f}")
    assert len(frames) == 8
    assert math.isclose(frames[-1][0].x, start.x, abs_tol=1e-6)
    assert math.isclose(frames[-1][0].z, start.z, abs_tol=1e-6)

    env2 = _FakeEnv(semantic_after_step=3)
    final_pose, units, found = drive_toward_heading(
        env2,
        start,
        heading_deg=90.0,
        max_units=5.0,
        goal_visible_fn=lambda obs: bool(np.any(obs.semantic == 1)),
        v_fwd=0.5,
        dt=0.1,
    )
    print(f"drive_toward_heading: units={units:.2f} found={found} pose={final_pose}")
    assert found is True
    assert math.isclose(units, 0.5 * 0.1 * 3, abs_tol=1e-6)

    env3 = _FakeEnv(semantic_after_step=10_000)  # never becomes visible
    final_pose3, units3, found3 = drive_toward_heading(
        env3,
        start,
        heading_deg=90.0,
        max_units=1.0,
        goal_visible_fn=lambda obs: bool(np.any(obs.semantic == 1)),
        v_fwd=0.5,
        dt=0.1,
    )
    print(f"drive_toward_heading (never found): units={units3:.2f} found={found3}")
    assert found3 is False
    assert units3 >= 1.0

    print("OK: uncertainty_motion smoke checks passed.")
