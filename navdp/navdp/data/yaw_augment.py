"""Viewpoint (yaw) augmentation for belief-conditioned navigation.

Goal: teach the policy to TURN BACK toward a goal that has left the frame, using
the belief as the only remaining cue. The expert keeps the goal centered, so it
never demonstrates a large turn toward an off-frame goal -- this augmentation
manufactures exactly those examples.

For a random rotation of the robot's heading by delta (delta>0 = rotate LEFT):
  * Observation: horizontal image shift RIGHT by shift_px = focal * tan(delta).
    For delta beyond the half-FOV the goal mask slides OFF the edge -> goal is
    genuinely out of frame, and only the belief points to it.
  * Belief: mu -> R(-delta) mu, Sigma -> R(-delta) Sigma R(-delta)^T (the goal's
    relative bearing decreases by delta).
  * Proprio (planar3 [x, z, yaw]): yaw += delta.
  * Expert action chunk: rotate (v_fwd, v_lat) by R(-delta) and add a corrective
    yaw of -delta spread over the horizon (clipped) -> "turn back toward belief".
  * Obstacle BEV map: rotate about the robot origin (bottom-centre) by -delta.

All sign conventions are gathered in YawAugConfig so the verifier can flip any
that render wrong BEFORE training.

Frame: body = [forward, left]; image column increases to the RIGHT; an image-right
goal has mu_y < 0 and needs NEGATIVE yaw (turn right) to re-centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class YawAugConfig:
    hfov_deg: float = 90.0          # camera horizontal field of view
    dt: float = 1.0 / 30.0          # control timestep (for the yaw-rate correction)
    max_yaw: float = 1.0            # clip for the corrected yaw rate
    aug_prob: float = 0.5           # fraction of samples to augment
    max_deg: float = 60.0           # |delta| sampled uniformly in [-max_deg, max_deg]
    yaw_gain: float = 0.67          # corrective yaw = yaw_gain*delta (rad/s per rad), clipped.
                                    # PROPORTIONAL (not bang-bang): 60deg -> ~0.7, 20deg -> ~0.23.
    depth_channel: int = -1         # which spatial channel is depth (edge-filled, not zeroed)
    proprio_yaw_index: int = 2      # planar3 -> [x, z, yaw]
    yaw_action_index: int = 2       # action3d -> [v_fwd, v_lat, yaw_rate]
    augment_obstacle_map: bool = True
    # sign flips (verifier sets these if a render looks mirrored)
    image_shift_sign: float = 1.0   # +1: delta>0 shifts image RIGHT
    belief_rot_sign: float = -1.0   # mu -> R(belief_rot_sign*delta) mu
    proprio_yaw_sign: float = 1.0   # yaw += proprio_yaw_sign*delta
    action_yaw_sign: float = -1.0   # corrective yaw = action_yaw_sign*delta/(H*dt)
    action_trans_sign: float = -1.0 # (vf,vl) -> R(action_trans_sign*delta)
    bev_rot_sign: float = -1.0      # obstacle BEV rotated by bev_rot_sign*delta


def focal_px(width: int, hfov_rad: float) -> float:
    return (width * 0.5) / math.tan(hfov_rad * 0.5)


def _rot2(c: float, s: float, x: torch.Tensor, y: torch.Tensor):
    """Apply R = [[c,-s],[s,c]] to (x, y)."""
    return c * x - s * y, s * x + c * y


def shift_horizontal(spatial: torch.Tensor, shift_px: int, depth_channel: int) -> torch.Tensor:
    """Shift [C,H,W] horizontally. Mask channels fill 0; depth channel replicates edge.

    shift_px > 0 shifts content to the RIGHT (exposes a strip on the LEFT).
    """
    c, h, w = spatial.shape
    out = torch.zeros_like(spatial)
    dch = depth_channel % c
    s = int(shift_px)
    if s == 0:
        return spatial.clone()
    if s > 0:
        s = min(s, w)
        if s < w:
            out[:, :, s:] = spatial[:, :, : w - s]
        if s > 0:
            edge = out[dch, :, s : s + 1] if s < w else spatial[dch, :, :1]
            out[dch, :, :s] = edge  # depth: replicate the new left edge
    else:
        a = min(-s, w)
        if a < w:
            out[:, :, : w - a] = spatial[:, :, a:]
        if a > 0:
            edge = out[dch, :, w - a - 1 : w - a] if a < w else spatial[dch, :, -1:]
            out[dch, :, w - a :] = edge
    return out


def rotate_belief(belief: torch.Tensor, delta: float, sign: float) -> torch.Tensor:
    """Rotate mu (cols 0,1) and Sigma (cols 2,3,4) of a [N,11] belief by R(sign*delta)."""
    out = belief.clone().float()
    ang = sign * delta
    c, s = math.cos(ang), math.sin(ang)
    mux, muy = out[:, 0].clone(), out[:, 1].clone()
    out[:, 0] = c * mux - s * muy
    out[:, 1] = s * mux + c * muy
    a, b, d = out[:, 2].clone(), out[:, 3].clone(), out[:, 4].clone()  # Sxx, Sxy, Syy
    # Sigma' = R Sigma R^T with R = [[c,-s],[s,c]]
    out[:, 2] = c * c * a - 2 * c * s * b + s * s * d
    out[:, 3] = c * s * a + (c * c - s * s) * b - c * s * d
    out[:, 4] = s * s * a + 2 * c * s * b + c * c * d
    return out


def correct_action(actions: torch.Tensor, delta: float, cfg: YawAugConfig) -> torch.Tensor:
    """Rotate translation by R(action_trans_sign*delta) and add corrective yaw."""
    out = actions.clone().float()
    h, a_dim = out.shape
    yaw_i = cfg.yaw_action_index
    # translation rotation
    ct = math.cos(cfg.action_trans_sign * delta)
    st = math.sin(cfg.action_trans_sign * delta)
    vf = out[:, 0].clone()
    vl = out[:, 1].clone() if a_dim >= 3 else torch.zeros_like(vf)
    out[:, 0] = ct * vf - st * vl
    if a_dim >= 3:
        out[:, 1] = st * vf + ct * vl
    # corrective yaw: PROPORTIONAL to the offset (graded turns, not bang-bang),
    # so moderate offsets get moderate turns and the policy stops over-rotating.
    dw = cfg.action_yaw_sign * cfg.yaw_gain * delta
    out[:, yaw_i] = torch.clamp(out[:, yaw_i] + dw, -cfg.max_yaw, cfg.max_yaw)
    return out


def adjust_proprio(proprio: torch.Tensor, delta: float, cfg: YawAugConfig) -> torch.Tensor:
    out = proprio.clone().float()
    i = cfg.proprio_yaw_index
    if out.numel() > i:
        out[i] = out[i] + cfg.proprio_yaw_sign * delta
    return out


def rotate_bev(grid: torch.Tensor, delta: float, sign: float) -> torch.Tensor:
    """Rotate a [H,W] egocentric occupancy grid about the robot origin (bottom-centre)."""
    h, w = grid.shape
    ang = sign * delta
    cr, cc = float(h - 1), float(w - 1) * 0.5  # robot at (row=H-1, col=W/2)
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    yc, xc = ys - cr, xs - cc
    ca, sa = math.cos(ang), math.sin(ang)
    src_x = ca * xc + sa * yc + cc       # inverse-rotate to find the source pixel
    src_y = -sa * xc + ca * yc + cr
    gx = 2.0 * src_x / max(w - 1, 1) - 1.0
    gy = 2.0 * src_y / max(h - 1, 1) - 1.0
    samp = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(grid[None, None].float(), samp, align_corners=True, mode="nearest")
    return out[0, 0]


def apply_yaw_augmentation(sample: Dict[str, object], delta: float, cfg: YawAugConfig) -> Dict[str, object]:
    """Return a NEW sample dict with a heading rotation of `delta` radians applied."""
    out = dict(sample)
    spatial = sample["spatial_semantic"]
    w = spatial.shape[-1]
    hfov = math.radians(cfg.hfov_deg)
    shift_px = int(round(cfg.image_shift_sign * focal_px(w, hfov) * math.tan(delta)))

    out["spatial_semantic"] = shift_horizontal(spatial, shift_px, cfg.depth_channel)
    if "goal_mask" in sample and torch.is_tensor(sample["goal_mask"]):
        gm = sample["goal_mask"]
        out["goal_mask"] = shift_horizontal(gm[None], shift_px, depth_channel=99)[0]
        out["semantic"] = out["goal_mask"][None]
    if "belief_tensor" in sample:
        out["belief_tensor"] = rotate_belief(sample["belief_tensor"], delta, cfg.belief_rot_sign)
    if "proprio" in sample:
        out["proprio"] = adjust_proprio(sample["proprio"], delta, cfg)
    if "expert_waypoints" in sample:
        out["expert_waypoints"] = correct_action(sample["expert_waypoints"], delta, cfg)
    if cfg.augment_obstacle_map and "obstacle_map" in sample and torch.is_tensor(sample["obstacle_map"]):
        om = sample["obstacle_map"]
        if om.dim() == 2:
            out["obstacle_map"] = rotate_bev(om, delta, cfg.bev_rot_sign)
    # NOTE: do NOT add new keys here -- augmented and un-augmented samples must
    # share an identical key set or habitat_route_collate (mixed batch) KeyErrors.
    return out


class YawAugmentationDataset(Dataset):
    """Wrap a base dataset; with prob aug_prob apply a random heading rotation."""

    def __init__(self, base: Dataset, cfg: Optional[YawAugConfig] = None, seed: Optional[int] = None):
        self.base = base
        self.cfg = cfg or YawAugConfig()
        self._seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Dict[str, object]:
        sample = self.base[i]
        rng = np.random.default_rng(None if self._seed is None else self._seed + i)
        if rng.random() >= self.cfg.aug_prob:
            return sample
        delta = math.radians(float(rng.uniform(-self.cfg.max_deg, self.cfg.max_deg)))
        return apply_yaw_augmentation(sample, delta, self.cfg)
