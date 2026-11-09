from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_belief_features(belief_tensor: torch.Tensor) -> torch.Tensor:
    """Map raw 11-dim belief slots to a bounded, scale-stable range.

    Raw layout (see SubgoalBeliefBank.as_tensor):
        [mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
         visible, initialized, time_since_seen, confidence,
         is_active, route_index_normalized]

    The unbounded entries (mu, Sigma, time_since_seen) are the ones that drift
    far out of the training distribution once a goal has been hidden for a long
    time: Sigma grows at +odom_noise/step and time_since_seen counts frames.
    Feeding those raw into the first Linear saturates it, which is why a stale
    belief produced near-random conditioning. We squash them into ~[-1, 1] with
    smooth, monotonic transforms so a long occlusion stays in-distribution.
    """
    if belief_tensor.shape[-1] < 11:
        return belief_tensor.float()
    b = belief_tensor.float()
    mu = b[..., 0:2] / 5.0  # metres -> ~[-1, 1] for typical indoor ranges
    sigma_diag = torch.log1p(b[..., 2:5:2].clamp_min(0.0)) / 10.0  # xx, yy
    sigma_xy = torch.tanh(b[..., 3:4])
    flags = b[..., 5:7]  # visible, initialized (already 0/1)
    time_since = torch.tanh(b[..., 7:8].clamp_min(0.0) / 20.0)
    tail = b[..., 8:]  # confidence, is_active, route_index (already bounded)
    return torch.cat(
        [
            mu,
            sigma_diag[..., 0:1],
            sigma_xy,
            sigma_diag[..., 1:2],
            flags,
            time_since,
            tail,
        ],
        dim=-1,
    )


class BeliefEncoder(nn.Module):
    """Encode per-goal belief slots into condition tokens.

    Input:
        belief_tensor: [B, N_goals, belief_dim]

    Expected feature layout for the default belief_dim=11:
        [mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
         visible, initialized, time_since_seen, confidence,
         is_active, route_index_normalized]

    Output:
        belief_tokens: [B, N_goals, embed_dim]
        active_belief_token: [B, 1, embed_dim]
    """

    def __init__(
        self,
        belief_dim: int,
        embed_dim: int,
        max_goals: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_transformer: bool = True,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.embed_dim = int(embed_dim)
        self.max_goals = int(max_goals)
        self.normalize_features = bool(normalize_features)
        self.slot_mlp = nn.Sequential(
            nn.Linear(belief_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.goal_pos = nn.Parameter(torch.randn(max_goals, embed_dim) * 0.02)
        if use_transformer and num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.encoder = None
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        belief_tensor: torch.Tensor,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if belief_tensor.dim() == 2:
            belief_tensor = belief_tensor.unsqueeze(0)
        if belief_tensor.dim() != 3:
            raise ValueError("belief_tensor must have shape [B,N,F] or [N,F]")
        b, n, f = belief_tensor.shape
        if f != self.belief_dim:
            raise ValueError(f"belief_tensor has {f} features, expected {self.belief_dim}")
        if n > self.max_goals:
            raise ValueError(f"belief_tensor has {n} goals, max_goals={self.max_goals}")

        feats = normalize_belief_features(belief_tensor) if self.normalize_features else belief_tensor.float()
        x = self.slot_mlp(feats)
        x = x + self.goal_pos[:n][None].to(device=x.device, dtype=x.dtype)
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.out_norm(x)

        if active_goal_index is None:
            if f >= 10:
                active_goal_index = belief_tensor[..., 9].argmax(dim=1)
            else:
                active_goal_index = torch.zeros(b, dtype=torch.long, device=x.device)
        active_goal_index = active_goal_index.to(device=x.device, dtype=torch.long).clamp(0, n - 1)
        gather_idx = active_goal_index[:, None, None].expand(-1, 1, self.embed_dim)
        active = x.gather(1, gather_idx)
        return x, active


class ObstacleMapEncoder(nn.Module):
    """Encode local obstacle maps into condition tokens.

    Input:
        obstacle_map: [B, H, W] or [B, 1, H, W], values in [0, 1]
    Output:
        obstacle_tokens: [B, token_grid**2, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 1,
        width: int = 32,
        num_tokens: int = 16,
    ):
        super().__init__()
        token_grid = int(math.sqrt(num_tokens))
        if token_grid * token_grid != num_tokens:
            raise ValueError("num_tokens must be a perfect square")
        self.token_grid = token_grid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(width * 2), width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, embed_dim, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obstacle_map: torch.Tensor) -> torch.Tensor:
        if obstacle_map.dim() == 3:
            obstacle_map = obstacle_map[:, None]
        if obstacle_map.dim() != 4:
            raise ValueError("obstacle_map must have shape [B,H,W] or [B,1,H,W]")
        x = obstacle_map.float()
        feat = self.pool(self.net(x))
        tokens = feat.flatten(2).transpose(1, 2)
        return self.norm(tokens)


class RouteTokenEncoder(nn.Module):
    """Embed the current route pointer as one condition token."""

    def __init__(self, embed_dim: int, max_route_len: int = 32):
        super().__init__()
        self.max_route_len = int(max_route_len)
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, route_index: torch.Tensor) -> torch.Tensor:
        if route_index.dim() == 0:
            route_index = route_index[None]
        route_index = route_index.float().view(-1, 1)
        denom = max(self.max_route_len - 1, 1)
        if route_index.max().item() > 1.0:
            route_index = route_index / float(denom)
        return self.mlp(route_index.clamp(0.0, 1.0))[:, None, :]


class NavDPConditionAdapter(nn.Module):
    """Append optional route-belief tokens to existing NavDP condition tokens."""

    def __init__(
        self,
        embed_dim: int,
        use_belief_bank: bool = False,
        use_obstacle_map: bool = False,
        use_route_token: bool = False,
        belief_dim: int = 11,
        max_goals: int = 16,
        obstacle_tokens: int = 16,
        max_route_len: int = 32,
        normalize_belief: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.use_belief_bank = bool(use_belief_bank)
        self.use_obstacle_map = bool(use_obstacle_map)
        self.use_route_token = bool(use_route_token)
        self.debug = bool(debug)
        self.belief_encoder = (
            BeliefEncoder(
                belief_dim=belief_dim,
                embed_dim=embed_dim,
                max_goals=max_goals,
                normalize_features=normalize_belief,
            )
            if self.use_belief_bank
            else None
        )
        self.obstacle_encoder = (
            ObstacleMapEncoder(embed_dim=embed_dim, num_tokens=obstacle_tokens)
            if self.use_obstacle_map
            else None
        )
        self.route_encoder = (
            RouteTokenEncoder(embed_dim=embed_dim, max_route_len=max_route_len)
            if self.use_route_token
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.use_belief_bank or self.use_obstacle_map or self.use_route_token

    def forward(
        self,
        condition_tokens: torch.Tensor,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        aux = self.encode_aux_tokens(
            condition_tokens,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        if aux is None:
            out = condition_tokens
        else:
            out = torch.cat([condition_tokens, aux], dim=1)
        if self.debug:
            print(
                "[NavDPConditionAdapter]",
                "base", tuple(condition_tokens.shape),
                "aux", None if aux is None else tuple(aux.shape),
                "out", tuple(out.shape),
            )
        return out

    def encode_aux_tokens(
        self,
        condition_tokens: torch.Tensor,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return only the new route-belief/obstacle tokens.

        This lets a pretrained denoiser keep its original context branch frozen
        while a separate trainable memory-attention branch consumes the new
        tokens.
        """
        tokens = []
        if self.belief_encoder is not None:
            if belief_tensor is None:
                raise ValueError("belief_tensor is required when use_belief_bank=True")
            belief_tensor = belief_tensor.to(device=condition_tokens.device)
            belief_tokens, active_token = self.belief_encoder(belief_tensor, active_goal_index)
            tokens.extend([belief_tokens, active_token])
        if self.obstacle_encoder is not None:
            if obstacle_map is None:
                raise ValueError("obstacle_map is required when use_obstacle_map=True")
            obstacle_map = obstacle_map.to(device=condition_tokens.device)
            tokens.append(self.obstacle_encoder(obstacle_map))
        if self.route_encoder is not None:
            if route_index is None:
                raise ValueError("route_index is required when use_route_token=True")
            tokens.append(self.route_encoder(route_index.to(condition_tokens.device)))
        if not tokens:
            return None
        return torch.cat(tokens, dim=1)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
