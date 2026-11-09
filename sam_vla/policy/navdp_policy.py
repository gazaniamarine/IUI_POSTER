"""NavDP (S2DiT diffusion) driving policy, ported from rollout_navdp_policy.py.

Qwen-VLA policies (qwen_vla_policy.py, qwen_discrete_direction_policy.py) drive
by asking a VLM what to do. This is a different implementation of the same
swap point (see base_policy.NavigationPolicy): a trained S2DiT diffusion model
samples a chunk of velocity actions directly from depth + a goal/obstacle mask
+ proprioception -- no language in the loop. Everything below the model call
(CBF safety, language-conditioned adapters, belief-return) is deliberately
left out; this is the base learned controller, and safety/steering concerns
live in sam_vla.safety and the Qwen policies respectively.

Like QwenDiscreteDirectionPolicy, this needs the rendered goal/obstacle
semantic mask (MarsHabitatEnv.get_semantic_frame(), painted by
register_object_mask with MESH_GOAL_ID/MESH_OBST_ID) alongside the frame, so
it only exposes act_verbose(), not the plain act(obs, goal_spec) signature.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from sam_vla.core.goal_geometry import MESH_GOAL_ID, MESH_OBST_ID
from sam_vla.core.types import Action, GoalSpec, Observation

HERE = Path(__file__).resolve().parent


def _resolve_navdp_root(raw: Optional[str]) -> Path:
    candidates = []
    if raw:
        candidates.append(Path(raw))
    env = os.environ.get("NAVDP_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(HERE.parent.parent / "navdp")
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "model_s2_dit.py").exists() and (
            c / "scripts" / "rollout_habitat_policy.py"
        ).exists():
            return c
    raise FileNotFoundError(
        "Could not find the navdp repo (expected model_s2_dit.py + "
        "scripts/rollout_habitat_policy.py). Pass navdp_root=/path/to/navdp "
        "or set NAVDP_ROOT."
    )


def _add_navdp_to_path(navdp_root: Path) -> None:
    for p in (navdp_root, navdp_root / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _intrinsics_from_hfov(height: int, width: int, hfov_deg: float) -> Dict[str, float]:
    hfov = math.radians(float(hfov_deg))
    fx = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    return {"fx": fx, "fy": fx, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.asarray([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float32)


class NavdpPolicy:  # implements NavigationPolicy (via act_verbose, like QwenDiscreteDirectionPolicy)
    def __init__(
        self,
        ckpt_path: str,
        navdp_root: Optional[str] = None,
        device: str = "cuda",
        weights: str = "model",
        sample_steps: int = 20,
        image_size: Optional[int] = None,
        hfov_deg: float = 90.0,
        proprio_mode: Optional[str] = None,
        action_mode: Optional[str] = None,
        yaw_axis: Optional[str] = None,
        use_obstacle_channel: Optional[bool] = None,
        use_obstacle_depth_map: bool = False,
        max_forward_speed: float = 1.0,
        max_lateral_speed: float = 1.0,
        max_yaw_rate: float = 1.0,
        action_smoothing: str = "none",
        ensemble_decay: float = 0.5,
        ema_alpha: float = 0.6,
        replan_every: int = 1,
    ):
        navdp_root_path = _resolve_navdp_root(navdp_root)
        _add_navdp_to_path(navdp_root_path)

        from navdp.data.habitat_route_dataset import (
            _empty_belief_tensor,
            _proprio_from_pose,
        )
        from navdp.extensions import DepthObstacleMap
        from rollout_habitat_policy import (
            ActionSmoother,
            action_to_control,
            frame_to_spatial,
            load_model,
        )

        self._empty_belief_tensor = _empty_belief_tensor
        self._proprio_from_pose = _proprio_from_pose
        self._frame_to_spatial = frame_to_spatial
        self._action_to_control = action_to_control

        self.device = device
        self.hfov_deg = float(hfov_deg)
        self.sample_steps = int(sample_steps)
        self.max_forward_speed = float(max_forward_speed)
        self.max_lateral_speed = float(max_lateral_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.replan_every = max(int(replan_every), 1)
        self.use_obstacle_depth_map = bool(use_obstacle_depth_map)

        self.model, train_args = load_model(
            Path(ckpt_path).expanduser().resolve(), device, weights
        )

        self.modes = {
            "proprio_mode": str(
                proprio_mode or train_args.get("habitat_proprio_mode", "planar3")
            ),
            "action_mode": str(
                action_mode or train_args.get("habitat_action_mode", "action3d")
            ),
            "yaw_axis": str(yaw_axis or train_args.get("habitat_yaw_axis", "y")),
        }
        if self.modes["action_mode"] == "waypoint":
            raise ValueError(
                "NavdpPolicy executes velocity actions; use an action3d or action2d checkpoint/mode."
            )

        if use_obstacle_channel is not None:
            self.use_obstacle_channel = bool(use_obstacle_channel)
        else:
            self.use_obstacle_channel = (
                bool(train_args.get("habitat_use_obstacle_channel", False))
                or int(train_args.get("spatial_channels", 2)) >= 3
            )

        self.image_size = int(image_size or train_args.get("image_size", 224))
        self._obstacle_builder = DepthObstacleMap(camera_intrinsics=None)
        self._smoother = ActionSmoother(action_smoothing, ensemble_decay, ema_alpha)
        self._last_pred_chunk = None

    def act_verbose(
        self, obs: Observation, semantic: np.ndarray, goal_spec: GoalSpec, step: int
    ) -> tuple[Action, dict]:
        """semantic is MarsHabitatEnv.get_semantic_frame() for the same step, carrying
        MESH_GOAL_ID/MESH_OBST_ID wherever the registered goal/obstacle masks render.
        goal_spec is accepted for interface parity with the other policies but unused --
        the mask, not the first-frame bbox, is what conditions the model every step."""
        if obs.depth is None:
            raise ValueError("NavdpPolicy requires depth in the observation")
        depth = np.asarray(obs.depth)
        sem = np.asarray(semantic)
        goal_mask = np.where(sem == MESH_GOAL_ID, 255, 0).astype(np.uint8)
        obstacle_mask = np.where(sem == MESH_OBST_ID, 255, 0).astype(np.uint8)

        spatial = self._frame_to_spatial(
            depth,
            goal_mask,
            self.image_size,
            obstacle_mask,
            include_obstacle_channel=self.use_obstacle_channel,
        ).to(self.device)

        if self.use_obstacle_depth_map:
            height, width = depth.shape[:2]
            self._obstacle_builder.camera_intrinsics = _intrinsics_from_hfov(
                height, width, self.hfov_deg
            )
            obstacle_map = self._obstacle_builder.build(depth)
        else:
            obstacle_map = np.zeros((96, 96), dtype=np.float32)
        obstacle_t = torch.from_numpy(obstacle_map[None]).float().to(self.device)

        qx, qy, qz, qw = _yaw_quat_xyzw(obs.pose.yaw)
        pose7 = np.asarray(
            [obs.pose.x, obs.pose.y, obs.pose.z, qx, qy, qz, qw], dtype=np.float32
        )
        proprio = self._proprio_from_pose(
            pose7,
            self.modes["proprio_mode"],
            planar_axes=(0, 2),
            yaw_axis=self.modes["yaw_axis"],
        )
        proprio_t = torch.from_numpy(proprio[None]).float().to(self.device)

        belief_t = (
            torch.from_numpy(self._empty_belief_tensor()[None]).float().to(self.device)
        )
        route_index = torch.zeros(1, dtype=torch.long, device=self.device)
        active_goal_index = torch.zeros(1, dtype=torch.long, device=self.device)

        do_replan = (step % self.replan_every == 0) or (self._last_pred_chunk is None)
        if do_replan:
            pred = self.model.sample(
                spatial,
                proprio_t,
                steps=self.sample_steps,
                belief_tensor=belief_t,
                obstacle_map=obstacle_t,
                route_index=route_index,
                active_goal_index=active_goal_index,
            )
            pred_chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
            chunk_ctrl = np.stack(
                [
                    self._action_to_control(
                        a,
                        action_mode=self.modes["action_mode"],
                        max_forward_speed=self.max_forward_speed,
                        max_lateral_speed=self.max_lateral_speed,
                        max_yaw_rate=self.max_yaw_rate,
                    )
                    for a in pred_chunk
                ]
            ).astype(np.float32)
            self._smoother.add(step, chunk_ctrl)
            self._last_pred_chunk = pred_chunk

        v_fwd, v_lat, yaw_rate = self._smoother.get(step)
        action = Action(
            v_fwd=float(v_fwd), v_lat=float(v_lat), yaw_rate=float(yaw_rate)
        )
        vla_result = {
            "goal_visible_pixels": int((goal_mask > 0).sum()),
            "obstacle_visible_pixels": int((obstacle_mask > 0).sum()),
            "replanned": bool(do_replan),
        }
        return action, vla_result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Smoke-load a NavDP checkpoint outside a rollout."
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--navdp-root", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    policy = NavdpPolicy(
        ckpt_path=args.ckpt, navdp_root=args.navdp_root, device=args.device
    )
    print(
        f"loaded NavdpPolicy: modes={policy.modes} image_size={policy.image_size} "
        f"use_obstacle_channel={policy.use_obstacle_channel}"
    )
