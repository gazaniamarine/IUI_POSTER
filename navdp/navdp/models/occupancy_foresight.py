"""Occupancy foresight: predict the next egocentric occupancy map from ego-motion.

This module is aligned with how obstacles are actually represented in this repo:
:class:`navdp.extensions.DepthObstacleMap` produces a per-frame, **egocentric**
binary occupancy grid (robot at row=grid-1, col=grid/2; +x forward decreases the
row, +y left increases the column). Because the grid is robot-centred, a *static*
obstacle's cell moves deterministically as the rover moves -- by the very same
SE(2) transform the belief bank uses in ``ego_motion_update``.

So instead of decoding the next map from scratch, ``OccupancyForesightHead``:

1. Analytically **warps** the current grid by the candidate's first-step
   ego-motion ``delta_pose = (dx, dy, dtheta)`` (robot frame). This is the
   physics prior -- "where the obstacle goes when the rover moves" -- and needs
   no learning.
2. Learns only a small **residual** on top of the warp (newly-revealed geometry,
   occluded fill) via a tiny conv encoder/decoder fused with the action.
3. Reports an aleatoric uncertainty ``u_occ`` whose floor is the fraction of the
   robot's predicted footprint that is **hallucinated** (warped from unknown /
   out-of-bounds cells). That is the occupancy analogue of ``sigma_aleatoric``
   from the belief bank, and is exactly large for blind-corner forecasts.

The map can also carry a second "known/observed" channel (1 where depth gave a
return, 0 where unobserved); blind corners are unobserved regions. If omitted it
is assumed fully observed.

Set ``use_egomotion_warp=False`` to fall back to a literal from-scratch
fuse-and-decode head (no physics prior) for ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def footprint_mask(
    grid_size: int,
    forward_cells: int,
    half_width: int,
    origin_row: Optional[int] = None,
    origin_col: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Boolean [H,W] mask of the cells the robot body sweeps just ahead.

    Because the predicted map is re-centred on the robot, the footprint is a
    fixed forward strip from the robot origin (bottom-centre) regardless of the
    candidate -- the candidate's motion is already baked into the warped map.
    """
    row0 = grid_size - 1 if origin_row is None else int(origin_row)
    col0 = grid_size * 0.5 if origin_col is None else float(origin_col)
    rows = torch.arange(grid_size, device=device)
    cols = torch.arange(grid_size, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    ahead = (rr <= row0) & (rr >= row0 - int(forward_cells))
    within = (cc >= col0 - int(half_width)) & (cc <= col0 + int(half_width))
    return ahead & within


def egomotion_warp(
    grid: torch.Tensor,
    delta_pose: torch.Tensor,
    resolution: float,
    origin_row: Optional[int] = None,
    origin_col: Optional[float] = None,
) -> torch.Tensor:
    """Warp an egocentric grid into the next frame given robot motion.

    ``grid``: [B,C,H,W]. ``delta_pose``: [B,3] = (dx, dy, dtheta) robot-frame
    forward/left/yaw motion from this frame into the next. Uses the inverse SE(2)
    transform consistent with ``SubgoalBeliefBank.ego_motion_update``:
    a point at new-frame position ``p_new`` was at ``p_old = R(dtheta) p_new + t``
    in the current frame, so we sample the current grid there.

    Out-of-bounds samples are filled with zeros (``padding_mode="zeros"``), which
    is how "unknown" propagates: warp a ones channel and the zeros mark cells with
    no observed source.
    """
    if grid.dim() != 4:
        raise ValueError("grid must be [B,C,H,W]")
    b, _c, h, w = grid.shape
    device = grid.device
    dtype = grid.dtype
    row0 = (h - 1) if origin_row is None else float(origin_row)
    col0 = (w * 0.5) if origin_col is None else float(origin_col)

    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")  # [H,W]
    # New-frame cell -> metric (x forward, y left).
    x_new = ((h - 1) - rr) * resolution
    y_new = (cc - col0) * resolution
    x_new = x_new[None].expand(b, h, w)
    y_new = y_new[None].expand(b, h, w)

    dx = delta_pose[:, 0].to(dtype).view(b, 1, 1)
    dy = delta_pose[:, 1].to(dtype).view(b, 1, 1)
    dtheta = delta_pose[:, 2].to(dtype).view(b, 1, 1)
    cd = torch.cos(dtheta)
    sd = torch.sin(dtheta)
    # p_old = R(dtheta) p_new + t
    x_old = cd * x_new - sd * y_new + dx
    y_old = sd * x_new + cd * y_new + dy

    r_old = (h - 1) - x_old / resolution
    c_old = y_old / resolution + col0
    # Normalize to [-1, 1] for grid_sample (x=cols=width, y=rows=height).
    gx = 2.0 * c_old / max(w - 1, 1) - 1.0
    gy = 2.0 * r_old / max(h - 1, 1) - 1.0
    sample_grid = torch.stack([gx, gy], dim=-1)  # [B,H,W,2]
    return F.grid_sample(
        grid, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def mask_random_rectangle(
    grid: torch.Tensor,
    known: Optional[torch.Tensor] = None,
    max_frac: float = 0.4,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blank a random rectangle of the current map (simulate a blind/occluded zone).

    The occupancy in the rectangle is zeroed and, if a ``known`` channel is given,
    it is marked unobserved there. The *target* next map is left untouched, which
    forces the model to hallucinate plausible geometry from ego-motion alone --
    the nav-domain analogue of relational occlusion-dropout in the belief bank.
    Returns ``(masked_grid, masked_known)``.
    """
    if grid.dim() != 4:
        raise ValueError("grid must be [B,C,H,W]")
    b, _c, h, w = grid.shape
    out = grid.clone()
    known_out = (torch.ones(b, 1, h, w, device=grid.device, dtype=grid.dtype)
                 if known is None else known.clone())

    def _rand(n: int) -> torch.Tensor:
        return torch.rand(n, generator=generator, device=grid.device)

    rh = (max_frac * h * _rand(b)).clamp(min=1).long()
    rw = (max_frac * w * _rand(b)).clamp(min=1).long()
    for i in range(b):
        ph = int(rh[i].item())
        pw = int(rw[i].item())
        r0 = int(((h - ph) * _rand(1)).item())
        c0 = int(((w - pw) * _rand(1)).item())
        out[i, :, r0 : r0 + ph, c0 : c0 + pw] = 0.0
        known_out[i, :, r0 : r0 + ph, c0 : c0 + pw] = 0.0
    return out, known_out


@dataclass
class ForesightOutput:
    predicted_next_map: torch.Tensor  # [B,1,H,W] occupancy (raw; clamp for scoring)
    u_occ: torch.Tensor               # [B] aleatoric uncertainty (>=0)
    warped_known: torch.Tensor        # [B,1,H,W] observed-coverage after warp
    footprint_unknown: torch.Tensor   # [B] fraction of footprint that is hallucinated


class OccupancyForesightHead(nn.Module):
    """Predict next egocentric occupancy + aleatoric uncertainty from ego-motion."""

    def __init__(
        self,
        grid_size: int = 96,
        resolution: float = 0.05,
        bottleneck: int = 128,
        enc_width: int = 32,
        action_dim: int = 3,
        forward_cells: int = 10,
        half_width: int = 6,
        unknown_uncertainty_coef: float = 2.0,
        use_egomotion_warp: bool = True,
        origin_row: Optional[int] = None,
        origin_col: Optional[float] = None,
    ):
        super().__init__()
        if grid_size % 4 != 0:
            raise ValueError("grid_size must be divisible by 4")
        self.grid_size = int(grid_size)
        self.resolution = float(resolution)
        self.bottleneck = int(bottleneck)
        self.forward_cells = int(forward_cells)
        self.half_width = int(half_width)
        self.unknown_uncertainty_coef = float(unknown_uncertainty_coef)
        self.use_egomotion_warp = bool(use_egomotion_warp)
        self.origin_row = origin_row
        self.origin_col = origin_col

        w = int(enc_width)
        # Encoder: 2 conv layers, /4 spatial. Input = [occupancy, known].
        self.encoder = nn.Sequential(
            nn.Conv2d(2, w, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(w), w),
            nn.SiLU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(2 * w), 2 * w),
            nn.SiLU(),
        )
        self.enc_channels = 2 * w
        self.feat_hw = self.grid_size // 4
        self.to_bottleneck = nn.Linear(self.enc_channels, self.bottleneck)
        # Action MLP: depth 2.
        self.action_mlp = nn.Sequential(
            nn.Linear(int(action_dim), self.bottleneck),
            nn.SiLU(),
            nn.Linear(self.bottleneck, self.bottleneck),
        )
        # Uncertainty head on the fused bottleneck.
        self.uncertainty_head = nn.Linear(self.bottleneck, 1)
        # Decoder: project bottleneck back to a small map, 2 transposed convs.
        self.from_bottleneck = nn.Linear(self.bottleneck, self.enc_channels * self.feat_hw * self.feat_hw)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.enc_channels, w, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(w), w),
            nn.SiLU(),
            nn.ConvTranspose2d(w, 1, 4, stride=2, padding=1),
        )
        self.register_buffer(
            "_footprint",
            footprint_mask(
                self.grid_size, self.forward_cells, self.half_width,
                origin_row=self.origin_row, origin_col=self.origin_col,
            ),
            persistent=False,
        )

    def forward(
        self,
        occupancy: torch.Tensor,
        delta_pose: torch.Tensor,
        known: Optional[torch.Tensor] = None,
    ) -> ForesightOutput:
        occ = self._as_bchw(occupancy)
        b, _c, h, w = occ.shape
        if (h, w) != (self.grid_size, self.grid_size):
            raise ValueError(f"expected {self.grid_size}x{self.grid_size} grid, got {h}x{w}")
        if known is None:
            known_in = torch.ones(b, 1, h, w, device=occ.device, dtype=occ.dtype)
        else:
            known_in = self._as_bchw(known)
        delta_pose = delta_pose.to(occ.dtype)
        if delta_pose.dim() == 1:
            delta_pose = delta_pose[None]

        enc_in = torch.cat([occ, known_in], dim=1)
        feat = self.encoder(enc_in)
        pooled = feat.mean(dim=(2, 3))  # [B, enc_channels]
        z_obs = self.to_bottleneck(pooled)
        z_act = self.action_mlp(delta_pose)
        z = z_obs + z_act

        residual = self.decoder(
            self.from_bottleneck(z).view(b, self.enc_channels, self.feat_hw, self.feat_hw)
        )

        if self.use_egomotion_warp:
            warped_occ = egomotion_warp(
                occ, delta_pose, self.resolution,
                origin_row=self.origin_row, origin_col=self.origin_col,
            )
            warped_known = egomotion_warp(
                known_in, delta_pose, self.resolution,
                origin_row=self.origin_row, origin_col=self.origin_col,
            )
            predicted = warped_occ + residual
        else:
            warped_known = known_in
            predicted = residual

        fp = self._footprint.to(occ.device)
        fp_area = fp.sum().clamp(min=1).to(occ.dtype)
        footprint_unknown = 1.0 - (warped_known.squeeze(1) * fp[None]).sum(dim=(1, 2)) / fp_area
        footprint_unknown = footprint_unknown.clamp(0.0, 1.0)

        u_learned = F.softplus(self.uncertainty_head(z)).squeeze(-1)  # [B]
        u_occ = u_learned + self.unknown_uncertainty_coef * footprint_unknown

        return ForesightOutput(
            predicted_next_map=predicted,
            u_occ=u_occ,
            warped_known=warped_known,
            footprint_unknown=footprint_unknown,
        )

    def footprint_occupancy(self, predicted_next_map: torch.Tensor) -> torch.Tensor:
        """Sum of predicted occupancy over the robot footprint -> [B]."""
        pred = self._as_bchw(predicted_next_map).clamp(0.0, 1.0).squeeze(1)
        fp = self._footprint.to(pred.device)
        return (pred * fp[None]).sum(dim=(1, 2))

    @staticmethod
    def _as_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x[None, None]
        if x.dim() == 3:
            return x[:, None]
        if x.dim() == 4:
            return x
        raise ValueError("map must be [H,W], [B,H,W], or [B,1,H,W]")


def foresight_loss(
    output: ForesightOutput,
    target_next_map: torch.Tensor,
    mse_weight: float = 1.0,
    nll_weight: float = 0.05,
    eps: float = 0.01,
    u_min: float = 0.05,
    mask_indicator: Optional[torch.Tensor] = None,
    cal_weight: float = 2.0,
    u_masked_target: float = 0.4,
    u_observed_target: float = 0.07,
) -> dict:
    """MSE + NLL + calibration loss.

    The calibration term directly supervises u_occ to be high for masked
    (blind-corner) frames and low for observed frames. This is necessary because
    the SE(2) warp makes mse tiny (~0.005) for all frames, so the NLL alone drives
    u_occ → 0 regardless of occlusion status. The calibration term breaks that
    degeneracy by giving the uncertainty head an explicit target that varies with
    the input.

    mask_indicator: [B] bool tensor, True for samples whose input was masked
    by mask_random_rectangle (blind-corner augmentation). Pass from the
    training loop when mask_prob > 0.
    """
    pred = OccupancyForesightHead._as_bchw(output.predicted_next_map)
    target = OccupancyForesightHead._as_bchw(target_next_map).to(pred.dtype)
    per_sample_se = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]
    mse = per_sample_se.mean()
    u_clamped = output.u_occ.clamp(min=float(u_min))
    var = (u_clamped ** 2) + float(eps)
    nll = 0.5 * (per_sample_se / var + torch.log(var))
    nll = nll.mean()

    cal = output.u_occ.new_tensor(0.0)
    if mask_indicator is not None and float(cal_weight) > 0:
        ind = mask_indicator.to(dtype=torch.bool, device=output.u_occ.device)
        u_target = torch.where(
            ind,
            output.u_occ.new_full(output.u_occ.shape, float(u_masked_target)),
            output.u_occ.new_full(output.u_occ.shape, float(u_observed_target)),
        )
        cal = F.mse_loss(output.u_occ, u_target)

    total = mse_weight * mse + nll_weight * nll + cal_weight * cal
    return {
        "loss": total,
        "mse": mse,
        "nll": nll,
        "cal": cal,
    }
