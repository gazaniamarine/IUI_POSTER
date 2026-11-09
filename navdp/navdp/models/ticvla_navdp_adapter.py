"""Learned TIC-VLA -> NavDP semantic-control adapter.

This is the architecture-level bridge, separate from rollout glue.

TIC-VLA original:
    delayed VLM KV/reasoning + current image + robot state
        -> Transformer Action Expert
        -> direct waypoints/actions

This adapter version:
    delayed VLM KV/reasoning + TIC-VLA waypoints + ego-motion/latency
        -> semantic-control tokens
        -> NavDP DiT extra_cond_tokens

It also predicts a compact belief delta and a robot-frame ghost waypoint for
backward-compatible mask/belief conditioning. The existing S2DiTPolicy already
accepts `extra_cond_tokens` in forward/sample, so this module can be trained and
used without editing the working NavDP model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class TICVLANavDPAdapterOutput:
    """Adapter outputs consumed by a NavDP wrapper/training script."""

    extra_cond_tokens: torch.Tensor       # [B, K, D], appended to NavDP DiT condition tokens
    belief_delta: torch.Tensor            # [B, 1, belief_dim], bounded correction
    ghost_waypoint: torch.Tensor          # [B, 2], robot-frame forward/left target
    confidence: torch.Tensor              # [B, 1], 0..1


def pool_ticvla_kv_cache(
    past_key_values,
    *,
    layer: int = -1,
    use_values: bool = True,
) -> torch.Tensor:
    """Pool TIC-VLA/InternVL KV cache into [B, H] state.

    Expected TIC-VLA cache item shape is `(key, value)`, with value shaped
    `[B, num_heads, seq_len, head_dim]`. The function also tolerates already
    pooled tensors `[B, H]` and token tensors `[B, L, H]`.
    """
    if past_key_values is None:
        raise ValueError("past_key_values is None")
    if torch.is_tensor(past_key_values):
        x = past_key_values
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 4:
            return x.permute(0, 2, 1, 3).flatten(2).mean(dim=1)
        raise ValueError(f"unsupported tensor KV shape: {tuple(x.shape)}")
    if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) == 0:
        raise ValueError("past_key_values must be a non-empty tuple/list or tensor")
    key, value = past_key_values[int(layer)]
    x = value if use_values else key
    if not torch.is_tensor(x):
        raise ValueError("selected KV entry is not a tensor")
    if x.dim() != 4:
        raise ValueError(f"expected KV tensor [B,heads,seq,dim], got {tuple(x.shape)}")
    return x.permute(0, 2, 1, 3).flatten(2).mean(dim=1)


class TICVLANavDPAdapter(nn.Module):
    """Map TIC-VLA delayed reasoning state into NavDP conditioning.

    Inputs:
      kv_state: pooled TIC-VLA reasoning state [B, kv_dim], or token state.
      waypoints: TIC-VLA relative waypoints [B, T, 2] or [B, T, 3].
      robot_state: compact robot state [B, R], e.g. velocity/yaw/offset.
      ego_delta: ego-motion offset since VLM started [B, 3] = dx, dy, dtheta.
      latency: scalar delay [B, 1].

    Outputs:
      extra_cond_tokens can be passed straight to S2DiTPolicy.sample(...).
    """

    def __init__(
        self,
        *,
        kv_dim: int = 128,
        waypoint_dim: int = 2,
        robot_state_dim: int = 5,
        ego_delta_dim: int = 3,
        belief_dim: int = 11,
        navdp_dim: int = 512,
        num_tokens: int = 4,
        max_waypoints: int = 30,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kv_dim = int(kv_dim)
        self.waypoint_dim = int(waypoint_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.ego_delta_dim = int(ego_delta_dim)
        self.belief_dim = int(belief_dim)
        self.navdp_dim = int(navdp_dim)
        self.num_tokens = int(num_tokens)
        self.max_waypoints = int(max_waypoints)

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(self.kv_dim),
            nn.Linear(self.kv_dim, hidden_dim),
            nn.SiLU(),
        )
        self.waypoint_proj = nn.Sequential(
            nn.LayerNorm(self.max_waypoints * self.waypoint_dim),
            nn.Linear(self.max_waypoints * self.waypoint_dim, hidden_dim),
            nn.SiLU(),
        )
        aux_dim = self.robot_state_dim + self.ego_delta_dim + 1
        self.aux_proj = nn.Sequential(
            nn.LayerNorm(aux_dim),
            nn.Linear(aux_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.token_head = nn.Linear(hidden_dim, self.num_tokens * self.navdp_dim)
        self.token_type = nn.Parameter(torch.randn(self.num_tokens, self.navdp_dim) * 0.02)
        self.token_norm = nn.LayerNorm(self.navdp_dim)

        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, self.belief_dim),
            nn.Tanh(),
        )
        self.ghost_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        *,
        kv_state: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None,
        ego_delta: Optional[torch.Tensor] = None,
        latency: Optional[torch.Tensor] = None,
    ) -> TICVLANavDPAdapterOutput:
        kv_state = self._normalize_kv(kv_state)
        b = kv_state.shape[0]
        device = kv_state.device
        dtype = kv_state.dtype

        wp = self._pack_waypoints(waypoints, b, device, dtype)
        robot = self._zeros_if_none(robot_state, (b, self.robot_state_dim), device, dtype)
        ego = self._zeros_if_none(ego_delta, (b, self.ego_delta_dim), device, dtype)
        lat = self._zeros_if_none(latency, (b, 1), device, dtype)
        aux = torch.cat([robot, ego, lat], dim=-1)

        h = self.fuse(torch.cat([self.kv_proj(kv_state), self.waypoint_proj(wp), self.aux_proj(aux)], dim=-1))
        tokens = self.token_head(h).view(b, self.num_tokens, self.navdp_dim)
        tokens = self.token_norm(tokens + self.token_type[None].to(dtype=tokens.dtype))
        belief_delta = self.belief_head(h)[:, None]
        ghost_waypoint = self.ghost_head(h)
        confidence = self.conf_head(h)
        return TICVLANavDPAdapterOutput(
            extra_cond_tokens=tokens,
            belief_delta=belief_delta,
            ghost_waypoint=ghost_waypoint,
            confidence=confidence,
        )

    def _normalize_kv(self, kv_state: torch.Tensor) -> torch.Tensor:
        if kv_state.dim() == 3:
            kv_state = kv_state.mean(dim=1)
        elif kv_state.dim() == 4:
            kv_state = kv_state.permute(0, 2, 1, 3).flatten(2).mean(dim=1)
        if kv_state.dim() != 2:
            raise ValueError(f"kv_state must reduce to [B,D], got {tuple(kv_state.shape)}")
        if kv_state.shape[-1] != self.kv_dim:
            raise ValueError(f"kv_state dim mismatch: expected {self.kv_dim}, got {kv_state.shape[-1]}")
        return kv_state

    def _pack_waypoints(
        self,
        waypoints: Optional[torch.Tensor],
        batch: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if waypoints is None:
            return torch.zeros(batch, self.max_waypoints * self.waypoint_dim, device=device, dtype=dtype)
        wp = waypoints.to(device=device, dtype=dtype)
        if wp.dim() == 2:
            wp = wp[None].expand(batch, -1, -1)
        if wp.dim() != 3:
            raise ValueError(f"waypoints must be [B,T,A], got {tuple(wp.shape)}")
        wp = wp[..., : self.waypoint_dim]
        if wp.shape[0] != batch:
            raise ValueError(f"waypoint batch mismatch: expected {batch}, got {wp.shape[0]}")
        if wp.shape[1] < self.max_waypoints:
            pad = torch.zeros(batch, self.max_waypoints - wp.shape[1], self.waypoint_dim, device=device, dtype=dtype)
            wp = torch.cat([wp, pad], dim=1)
        else:
            wp = wp[:, : self.max_waypoints]
        return wp.flatten(1)

    @staticmethod
    def _zeros_if_none(
        value: Optional[torch.Tensor],
        shape: Tuple[int, int],
        device,
        dtype,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(*shape, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 1:
            value = value[None]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"expected shape {shape}, got {tuple(value.shape)}")
        return value


def apply_belief_delta(
    base_belief: torch.Tensor,
    belief_delta: torch.Tensor,
    confidence: torch.Tensor,
    *,
    scale: float = 0.25,
) -> torch.Tensor:
    """Blend adapter belief correction into an existing NavDP belief tensor."""
    if base_belief.dim() != 3:
        raise ValueError("base_belief must be [B,G,D]")
    if belief_delta.dim() != 3:
        raise ValueError("belief_delta must be [B,1,D]")
    conf = confidence.view(confidence.shape[0], 1, 1).to(base_belief.dtype)
    delta = belief_delta.to(base_belief.dtype)
    return base_belief + float(scale) * conf * delta
