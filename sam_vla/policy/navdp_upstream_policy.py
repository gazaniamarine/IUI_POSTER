"""Policy backend wrapping upstream NavDP's HTTP server -- either the real,
published NavDP model (Cai et al., arXiv 2505.08712,
github.com/InternRobotics/NavDP) or this project's S2Diff-guided fork of it,
selected via server_variant (see navdp_upstream_server_manager.py's
docstring for how the two differ) -- next.md's "Integration project" Phase
3. Architecturally unrelated to this repo's own navdp/ package beyond
sharing a name; see navdp_policy.py's NavdpPolicy for that one (an in-house
S2DiT/VL3-DP model trained on this project's own data, action-chunk-of-
velocities output). This class is additive, not a replacement -- both are
selectable via run_navdp_rollout.py's --policy-backend flag.

Same act_verbose(obs, semantic, goal_spec, step) -> (Action, dict) shape as
NavdpPolicy for interface parity (NavigationPolicy, base_policy.py), but
upstream NavDP's point-goal mode needs one more per-step input its signature
has no room for: a body-frame (forward, left) goal point (matching
BeliefGoalTracker.belief_g exactly -- see next.md's building-blocks table).
Rather than changing the shared protocol, the caller sets it via
set_goal_body() immediately before each act_verbose() call, same idea as
kb_teleop_vl.py setting state on a policy/session object between ticks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sam_vla.core.types import Action, GoalSpec, Observation
from sam_vla.policy.navdp_upstream_client import (
    pointgoal_step,
    select_action_from_trajectory,
)
from sam_vla.vlm.navdp_upstream_server_manager import NavdpUpstreamServerManager


class NavdpUpstreamPolicy:  # implements NavigationPolicy via act_verbose
    def __init__(
        self,
        checkpoint_path: str,
        navdp_upstream_root: Optional[str] = None,
        port: Optional[int] = None,
        stop_threshold: float = 0.0,
        image_hw: tuple[int, int] = (480, 640),
        hfov_deg: float = 90.0,
        lookahead: int = 3,
        replan_every: int = 1,
        max_forward_speed: float = 1.0,
        turn_kp: float = 1.4,
        max_yaw_rate: float = 1.0,
        request_timeout: float = 30.0,
        default_goal_forward: float = 5.0,
        default_goal_left: float = 0.0,
        server_variant: str = "navdp",
        device: str = "cuda:0",
        planner_mode: str = "s2diff",
        remove_critic: bool = True,
        s2diff_extra_args: Optional[dict] = None,
    ):
        self._server = NavdpUpstreamServerManager(
            checkpoint_path=checkpoint_path,
            navdp_upstream_root=navdp_upstream_root,
            port=port,
            stop_threshold=stop_threshold,
            batch_size=1,
            image_hw=image_hw,
            hfov_deg=hfov_deg,
            server_variant=server_variant,
            device=device,
            planner_mode=planner_mode,
            remove_critic=remove_critic,
            s2diff_extra_args=s2diff_extra_args,
        )
        self._started = False
        self._obstacle_pixels: Optional[list] = None
        # Lookahead/replan cadence: next.md's own open question flags there is
        # no principled default yet ("start from whatever NavdpPolicy's own
        # replan_every default is and tune empirically") -- these mirror
        # NavdpPolicy's replan_every=1 default so the two backends are
        # comparably chatty with their model out of the box.
        self.lookahead = max(int(lookahead), 0)
        self.replan_every = max(int(replan_every), 1)
        self.max_forward_speed = float(max_forward_speed)
        self.turn_kp = float(turn_kp)
        self.max_yaw_rate = float(max_yaw_rate)
        self.request_timeout = float(request_timeout)

        self._goal_forward = float(default_goal_forward)
        self._goal_left = float(default_goal_left)
        self._last_trajectory: Optional[np.ndarray] = None
        self._last_all_values: np.ndarray = np.zeros(0, dtype=np.float32)
        self._steps_since_replan = 0

    def set_goal_body(self, forward: float, left: float) -> None:
        """Caller (run_navdp_rollout.py's loop) calls this with
        belief_tracker.belief_g's current [forward, left] estimate right
        before act_verbose() each step -- see this module's docstring for why
        it isn't threaded through act_verbose's own signature instead."""
        self._goal_forward = float(forward)
        self._goal_left = float(left)

    def set_obstacle_pixels(self, obstacle_pixels: Optional[list]) -> None:
        """Optional per-step [[u, v], ...] obstacle pixels for
        server_variant="s2diff"'s S2Diff guidance (see
        navdp_upstream_client.pointgoal_step's docstring); ignored by the
        plain "navdp" variant. Unset (None) means unguided sampling."""
        self._obstacle_pixels = obstacle_pixels

    def start(self) -> None:
        """Explicit early start, for callers that want the (slow, one-time)
        model load to happen before the rollout loop starts rather than on
        the first act_verbose() call. Idempotent."""
        if not self._started:
            self._server.start()
            self._started = True

    def stop(self) -> None:
        self._server.stop()

    def act_verbose(
        self, obs: Observation, semantic: np.ndarray, goal_spec: GoalSpec, step: int
    ) -> tuple[Action, dict]:
        """semantic/goal_spec are accepted for interface parity with
        NavdpPolicy/QwenDiscreteDirectionPolicy but unused -- upstream NavDP's
        point-goal mode only wants the raw RGB-D frame plus the body-frame
        goal point set via set_goal_body()."""
        if obs.depth is None:
            raise ValueError("NavdpUpstreamPolicy requires depth in the observation")
        self.start()

        do_replan = (step % self.replan_every == 0) or (self._last_trajectory is None)
        if do_replan:
            trajectory_xy, all_values = pointgoal_step(
                self._server.base_url,
                obs.rgb,
                obs.depth,
                self._goal_forward,
                self._goal_left,
                timeout=self.request_timeout,
                obstacle_pixels=self._obstacle_pixels,
            )
            self._last_trajectory = trajectory_xy
            self._last_all_values = all_values
            self._steps_since_replan = 0
        else:
            self._steps_since_replan += 1

        action = select_action_from_trajectory(
            self._last_trajectory,
            waypoint_index=self._steps_since_replan + self.lookahead,
            max_forward_speed=self.max_forward_speed,
            turn_kp=self.turn_kp,
            max_yaw_rate=self.max_yaw_rate,
        )
        vla_result = {
            "goal_forward": self._goal_forward,
            "goal_left": self._goal_left,
            "replanned": bool(do_replan),
            "trajectory_len": int(self._last_trajectory.shape[0]),
            "critic_value_max": (
                float(self._last_all_values.max())
                if self._last_all_values.size
                else None
            ),
        }
        return action, vla_result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Smoke-test NavdpUpstreamPolicy against a synthetic observation. "
            "Requires the vendored InternRobotics/NavDP checkout (--navdp-upstream-root "
            "or $NAVDP_UPSTREAM_ROOT) and a real checkpoint (gated behind upstream's "
            "Google Form, see next.md's Integration project Phase 0) -- this will fail "
            "fast with FileNotFoundError until Phase 0 is done by hand."
        )
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--navdp-upstream-root", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--server-variant", choices=["navdp", "s2diff"], default="navdp"
    )
    ap.add_argument("--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff")
    args = ap.parse_args()

    from sam_vla.core.types import Pose

    policy = NavdpUpstreamPolicy(
        checkpoint_path=args.checkpoint,
        navdp_upstream_root=args.navdp_upstream_root,
        port=args.port,
        server_variant=args.server_variant,
        planner_mode=args.planner_mode,
    )
    policy.set_goal_body(forward=3.0, left=0.5)
    obs = Observation(
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.full((480, 640), 3.0, dtype=np.float32),
        pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        frame_idx=0,
    )
    goal_spec = GoalSpec(
        goal_bbox_norm=(0.4, 0.4, 0.6, 0.6),
        obstacle_bboxes_norm=[],
        instruction_text="smoke test",
    )
    action, vla_result = policy.act_verbose(obs, semantic=None, goal_spec=goal_spec, step=0)
    print(f"action={action} vla_result={vla_result}")
    policy.stop()
