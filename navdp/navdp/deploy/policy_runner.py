"""PolicyRunner -- run the trained policy on ANY platform (real robot, Mars sim).

The checkpoint is the *fast policy*; it consumes a structured per-step state, not
raw frames. This wrapper builds that state from the four signals every platform
can provide and returns a body-frame velocity command. No habitat, no simulator.

Per step you provide:
  depth         : HxW float32, METERS (depth/stereo camera)
  goal_mask     : HxW bool/uint8, your goal segmentation (SAM / detector / template)
  pose          : (x, z, yaw)  in your odometry frame  (planar; yaw in radians)
  obstacle_mask : HxW bool/uint8, optional (enables the obstacle channel + CBF)
  intrinsics    : {"fx","fy","cx","cy"} for the depth camera (once)

You get back:
  action = [v_fwd, v_lat, yaw_rate]   (m/s, m/s, rad/s) in the robot body frame.

CONVENTIONS (must match, or transform your inputs to them):
  * Camera optical [right, down, forward] -> robot body [forward, left, up].
  * depth is metric (meters). If yours is mm, divide by 1000 before passing.
  * body frame: +x forward, +y left; +yaw turns left.
  * pose is your integrated odometry (origin at episode start is fine -- the
    policy navigates relative to the belief, not an absolute map).

The belief is propagated by your odometry between frames, so it keeps pointing at
the goal even when the goal mask is empty (occluded / out of frame). Odometry
quality matters: the SE(2) propagation is exact given exact odom; drift in your
odom drifts the belief over long blackouts.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navdp.extensions import (
    DepthObstacleMap,
    SAMDepthTargetExtractor,
    SubgoalBeliefBank,
    build_cbf_guidance,
    estimate_obstacle_velocity,
    nearest_obstacle_point,
)
# Reuse the exact input builders the rollout/eval used (sim-free helpers).
from scripts.rollout_habitat_policy import (  # noqa: E402
    ActionSmoother,
    action_to_control,
    default_intrinsics,
    frame_to_spatial,
    make_belief_observation,
)
from scripts.train_pixel_goal_policy import build_model_from_ckpt  # noqa: E402


def _odom_from_poses(prev: Sequence[float], cur: Sequence[float]) -> list:
    """Body-frame ego-motion (dx, dy, dtheta) from two planar poses (x, z, yaw).

    Matches navdp.data.habitat_route_dataset._odom_delta (planar axes x,z)."""
    dwx, dwz = cur[0] - prev[0], cur[1] - prev[1]
    c, s = math.cos(-prev[2]), math.sin(-prev[2])
    dth = (cur[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi
    return [c * dwx - s * dwz, s * dwx + c * dwz, dth]


class PolicyRunner:
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
        use_cbf: bool = False,
        cbf_d_safe: float = 0.5,
        cbf_gamma: float = 0.3,
        cbf_guidance_scale: float = 0.4,
        cbf_steps: int = 2,
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
        self.min_visible_pixels = int(min_visible_pixels)
        self.use_cbf = bool(use_cbf)
        self.cbf = dict(d_safe=cbf_d_safe, gamma=cbf_gamma, guidance_scale=cbf_guidance_scale, n_steps=cbf_steps)
        self._intr: Optional[Dict[str, float]] = None
        self._extractor = self._obstacle = None
        self.reset("goal")

    def reset(self, goal: str) -> None:
        """Start a new episode for goal `goal` (the category your segmenter targets)."""
        self.goal = str(goal)
        self.bank = SubgoalBeliefBank([self.goal], sigma_visible=0.05, odom_noise=0.02)
        self.smoother = ActionSmoother(mode=self.smoothing)
        self._prev_pose: Optional[tuple] = None
        self._prev_obs_point = None
        self._step = 0

    @torch.no_grad()
    def step(
        self,
        depth: np.ndarray,
        goal_mask: np.ndarray,
        pose: Sequence[float],
        obstacle_mask: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        hz: float = 30.0,
    ) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        goal_mask = np.asarray(goal_mask)
        h, w = depth.shape[:2]
        if self._intr is None:
            self._intr = dict(intrinsics) if intrinsics else default_intrinsics(h, w)
            self._extractor = SAMDepthTargetExtractor(self._intr, min_mask_area=50, depth_scale=1.0, position_dim=2)
            self._obstacle = DepthObstacleMap(grid_size=96, resolution=0.05, camera_intrinsics=self._intr, depth_scale=1.0)
        dt = 1.0 / float(hz)

        cur = (float(pose[0]), float(pose[1]), float(pose[2]))
        odom = [0.0, 0.0, 0.0] if self._prev_pose is None else _odom_from_poses(self._prev_pose, cur)
        self._prev_pose = cur

        # belief update (goal observation from mask+depth, propagated by odom)
        vis_px = int((goal_mask > 0).sum())
        obs = make_belief_observation(self._extractor, self.goal, goal_mask, depth, vis_px, self.min_visible_pixels)
        self.bank.update(obs, odom_delta=odom, step=self._step)
        belief = self.bank.as_tensor([self.goal], active_goal_id=self.goal, route_index=0, route_length=1).cpu().numpy()

        # policy inputs
        spatial = frame_to_spatial(
            depth, goal_mask, self.image_size,
            obstacle_mask=obstacle_mask, include_obstacle_channel=self.use_obstacle_channel,
        ).to(self.device)
        proprio = torch.tensor([[cur[0], cur[1], cur[2]]], dtype=torch.float32, device=self.device)
        obstacle_t = torch.from_numpy(self._obstacle.build(depth)[None]).float().to(self.device)
        belief_t = torch.from_numpy(belief[None]).float().to(self.device)
        ri = torch.zeros(1, dtype=torch.long, device=self.device)
        ai = torch.zeros(1, dtype=torch.long, device=self.device)

        guidance = None
        if self.use_cbf and obstacle_mask is not None:
            op = nearest_obstacle_point(np.asarray(obstacle_mask), depth, self._intr)
            if op is not None:
                vo = estimate_obstacle_velocity(self._prev_obs_point, op, odom, dt)
                guidance = build_cbf_guidance(
                    p0=op, v_o=vo, d_safe=self.cbf["d_safe"], gamma=self.cbf["gamma"], dt=dt,
                    guidance_scale=self.cbf["guidance_scale"], n_steps=self.cbf["n_steps"],
                )
            self._prev_obs_point = op

        pred = self.model.sample(
            spatial, proprio, steps=self.sample_steps,
            belief_tensor=belief_t, obstacle_map=obstacle_t,
            route_index=ri, active_goal_index=ai, guidance_fn=guidance,
        )
        chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        ctrl = np.stack(
            [action_to_control(chunk[k], action_mode=self.action_mode,
                               max_forward_speed=self.max_fwd, max_lateral_speed=self.max_lat,
                               max_yaw_rate=self.max_yaw) for k in range(chunk.shape[0])],
            axis=0,
        )
        self.smoother.add(self._step, ctrl)
        action = self.smoother.get(self._step)
        self._step += 1
        return np.asarray(action, dtype=np.float32)  # [v_fwd, v_lat, yaw_rate]
