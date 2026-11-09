"""MultiTargetRunner -- navigate to ANY belief-tracked object, switchable at runtime.

Extends the object-agnostic policy to multiple named targets, each with its OWN
persistent belief that is tracked every step (so it stays valid even while that
object is out of view). At any moment ONE target is "active": its mask goes in
the goal channel and its belief is fed to the policy, so the robot navigates to
it. Switch the active target (e.g. goal -> obstacle) and the robot heads to the
new one -- NO retraining, because the policy navigates to whatever the active
belief points at.

Use case: reach the goal, then make the obstacle the new goal and return to it.

  r = MultiTargetRunner(ckpt, device="cuda")
  r.reset(["goal", "obstacle"])               # two persistent beliefs
  ... each step ...
  a = r.step(depth, masks={"goal": gm, "obstacle": om}, pose=(x,z,yaw),
             intrinsics=K)                     # both beliefs updated; active = "goal"
  if reached: r.set_active("obstacle")         # now return to the obstacle

Training consistency: only the ACTIVE slot's belief is fed to the policy (a
single [1,11] row, exactly as trained); the other beliefs are tracked silently.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from navdp.extensions import DepthObstacleMap, SAMDepthTargetExtractor, SubgoalBeliefBank
from navdp.deploy.policy_runner import _odom_from_poses
from scripts.rollout_habitat_policy import (
    ActionSmoother,
    action_to_control,
    default_intrinsics,
    frame_to_spatial,
    make_belief_observation,
)
from scripts.train_pixel_goal_policy import build_model_from_ckpt


class MultiTargetRunner:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        weights: str = "model",
        sample_steps: int = 20,
        smoothing: str = "ensemble",
        max_forward_speed: float = 1.0,
        max_lateral_speed: float = 1.0,
        max_yaw_rate: float = 1.0,
        min_visible_pixels: int = 20,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model, self.targs, self.dims = build_model_from_ckpt(ckpt, device, weights)
        self.model.eval()
        self.device = device
        self.image_size = int(self.targs.get("image_size", 224))
        self.action_mode = str(self.targs.get("habitat_action_mode", "action3d"))
        self.use_obstacle_channel = bool(
            self.targs.get("habitat_use_obstacle_channel", self.dims["spatial_channels"] >= 3)
        )
        self.sample_steps = int(sample_steps)
        self.smoothing = smoothing
        self.max_fwd, self.max_lat, self.max_yaw = max_forward_speed, max_lateral_speed, max_yaw_rate
        self.min_visible = int(min_visible_pixels)
        self._intr: Optional[Dict[str, float]] = None
        self._extractor = self._obstacle = None
        self.reset(["goal"])

    def reset(self, targets: Sequence[str]) -> None:
        self.targets = list(targets)
        self.active = self.targets[0]
        self.bank = SubgoalBeliefBank(self.targets, sigma_visible=0.05, odom_noise=0.02)
        self.smoother = ActionSmoother(mode=self.smoothing)
        self._prev_pose: Optional[tuple] = None
        self._step = 0

    def set_active(self, target: str) -> None:
        """Switch which tracked target the robot navigates to (must be in reset() list)."""
        if target not in self.targets:
            raise ValueError(f"{target!r} not in tracked targets {self.targets}")
        self.active = target
        self.smoother = ActionSmoother(mode=self.smoothing)  # fresh smoothing on switch

    @torch.no_grad()
    def step(
        self,
        depth: np.ndarray,
        masks: Dict[str, np.ndarray],
        pose: Sequence[float],
        avoid_mask: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        hz: float = 30.0,
    ) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        h, w = depth.shape[:2]
        if self._intr is None:
            self._intr = dict(intrinsics) if intrinsics else default_intrinsics(h, w)
            self._extractor = SAMDepthTargetExtractor(self._intr, min_mask_area=50, depth_scale=1.0, position_dim=2)
            self._obstacle = DepthObstacleMap(grid_size=96, resolution=0.05, camera_intrinsics=self._intr, depth_scale=1.0)
        dt = 1.0 / float(hz)

        cur = (float(pose[0]), float(pose[1]), float(pose[2]))
        odom = [0.0, 0.0, 0.0] if self._prev_pose is None else _odom_from_poses(self._prev_pose, cur)
        self._prev_pose = cur

        # Update EVERY target's belief (observed or propagated) so all stay valid.
        obs_all: Dict[str, dict] = {}
        for tid in self.targets:
            m = masks.get(tid)
            if m is None:
                obs_all[tid] = {"visible": False, "position": None, "confidence": 0.0}
            else:
                m = np.asarray(m)
                obs_all.update(make_belief_observation(
                    self._extractor, tid, m, depth, int((m > 0).sum()), self.min_visible))
        self.bank.update(obs_all, odom_delta=odom, step=self._step)

        # Feed ONLY the active target to the policy (1 slot, as trained).
        belief = self.bank.as_tensor([self.active], active_goal_id=self.active,
                                     route_index=0, route_length=1).cpu().numpy()
        goal_channel = masks.get(self.active)
        if goal_channel is None:
            goal_channel = np.zeros((h, w), dtype=np.uint8)  # gone from view -> belief carries it

        spatial = frame_to_spatial(
            depth, goal_channel, self.image_size,
            obstacle_mask=avoid_mask, include_obstacle_channel=self.use_obstacle_channel,
        ).to(self.device)
        proprio = torch.tensor([[cur[0], cur[1], cur[2]]], dtype=torch.float32, device=self.device)
        obstacle_t = torch.from_numpy(self._obstacle.build(depth)[None]).float().to(self.device)
        belief_t = torch.from_numpy(belief[None]).float().to(self.device)
        ri = torch.zeros(1, dtype=torch.long, device=self.device)
        ai = torch.zeros(1, dtype=torch.long, device=self.device)

        pred = self.model.sample(spatial, proprio, steps=self.sample_steps,
                                 belief_tensor=belief_t, obstacle_map=obstacle_t,
                                 route_index=ri, active_goal_index=ai)
        chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        ctrl = np.stack([action_to_control(chunk[k], action_mode=self.action_mode,
                                            max_forward_speed=self.max_fwd, max_lateral_speed=self.max_lat,
                                            max_yaw_rate=self.max_yaw) for k in range(chunk.shape[0])], axis=0)
        self.smoother.add(self._step, ctrl)
        action = self.smoother.get(self._step)
        self._step += 1
        return np.asarray(action, dtype=np.float32)

    def belief_range(self, target: str) -> float:
        """Current belief distance to a target (m) -- handy for the phase switch."""
        row = self.bank.as_tensor([target], active_goal_id=target).cpu().numpy()[0]
        return float(np.hypot(row[0], row[1]))
