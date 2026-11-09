"""
model.py
========
VL3-DP: Vision-Language-3D Diffusion Policy.

Architecture (flow-matching variant):
  1. Point cloud  ──► DP3Encoder + GeoTokenizer    ──► K geometric tokens
  2. Qwen features (cached, frozen) ──► VLMProjector ──► L semantic tokens
  3. Proprioception ──► MLP                         ──► 1 proprio token

  All tokens ──► MetaQueryFusion (16 soft queries, cross-attention)
             ──► conditioning buffer [B, Q, dim]

  Conditioning buffer + noised action + timestep
             ──► FlowMatchingDiT (self-attn + cross-attn + adaLN)
             ──► predicted velocity  (rectified flow)

Training losses:
  L = MSE(v_pred, v_target)  +  freq_weight * FrequencyDomainLoss(x0_hat, x0)

VLANeXt findings implemented:
  - Soft connection (finding #6):  MetaQueryFusion is a learnable latent buffer
  - Flow matching  (finding #4):  rectified-flow velocity prediction
  - Action chunking(finding #3):  horizon > 1 predicted jointly
  - Freq-domain loss(finding #12): DCT regulariser, low-freq weighted higher
  - No temporal history(finding #7): single current observation only
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from navdp.models import BeliefConditionedCocosSource, NavDPConditionAdapter


# ---------------------------------------------------------------------------
# DP3 point encoder  (faithful to the RSS'24 "simple" encoder, extended to
# also return per-point features so the tokeniser can attend to local geometry)
# ---------------------------------------------------------------------------
class DP3Encoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden=(64, 128, 256), out_dim: int = 256):
        super().__init__()
        dims = [in_channels, *hidden]
        layers: list = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.GELU()]
        self.point_mlp = nn.Sequential(*layers)
        self.global_proj = nn.Sequential(nn.Linear(hidden[-1], out_dim), nn.LayerNorm(out_dim))
        self.point_dim = hidden[-1]   # dim of per-point features before pooling
        self.out_dim = out_dim        # dim of global feature

    def forward(self, pc: torch.Tensor):
        """pc: [B, N, in_channels]  →  (point_feats [B,N,point_dim], global [B,out_dim])"""
        feat = self.point_mlp(pc)            # [B, N, point_dim]
        g    = feat.max(dim=1).values        # [B, point_dim]  PointNet max-pool
        return feat, self.global_proj(g)     # [B,N,point_dim], [B,out_dim]


# ---------------------------------------------------------------------------
# Cross-attention building block
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: Optional[int] = None, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        kv_dim = kv_dim or dim
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.scale = self.dh ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        """x:[B,Q,dim]  ctx:[B,K,kv_dim]  mask:[B,K] True=ignore"""
        B, Q, D = x.shape
        K = ctx.shape[1]
        q = self.q(x).view(B, Q, self.heads, self.dh).transpose(1, 2)
        k = self.k(ctx).view(B, K, self.heads, self.dh).transpose(1, 2)
        v = self.v(ctx).view(B, K, self.heads, self.dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale           # [B,H,Q,K]
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)   # guard: all-masked row → NaN → zero
        out = (attn @ v).transpose(1, 2).reshape(B, Q, D)
        return self.proj(self.drop(out))


# ---------------------------------------------------------------------------
# Geometric tokeniser: K learnable queries attention-pool point features
# into K geometric tokens; prepend one projected global token → K+1 total.
# ---------------------------------------------------------------------------
class GeoTokenizer(nn.Module):
    def __init__(self, point_dim: int, global_dim: int, dim: int,
                 num_tokens: int = 8, heads: int = 8):
        super().__init__()
        self.queries    = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attn       = CrossAttention(dim, kv_dim=point_dim, heads=heads)
        self.norm       = nn.LayerNorm(dim)
        self.global_proj= nn.Linear(global_dim, dim)

    def forward(self, point_feats: torch.Tensor, global_feat: torch.Tensor):
        B = point_feats.shape[0]
        q   = self.queries[None].expand(B, -1, -1)
        geo = self.norm(q + self.attn(q, point_feats))       # [B, K, dim]
        g   = self.global_proj(global_feat)[:, None, :]      # [B, 1, dim]
        return torch.cat([g, geo], dim=1)                    # [B, K+1, dim]


# ---------------------------------------------------------------------------
# Soft connection: learnable meta-queries cross-attend to the union of all
# observation tokens, acting as a latent buffer before the policy.
# ---------------------------------------------------------------------------
class _FusionLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, q, ctx, mask=None):
        q = q + self.cross(self.norm1(q), ctx, mask)
        q = q + self.self_attn(self.norm2(q), self.norm2(q), self.norm2(q), need_weights=False)[0]
        q = q + self.mlp(self.norm3(q))
        return q


class MetaQueryFusion(nn.Module):
    def __init__(self, dim: int, vlm_dim: int, num_queries: int = 16,
                 depth: int = 4, heads: int = 8):
        super().__init__()
        self.queries  = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.vlm_proj = nn.Sequential(nn.Linear(vlm_dim, dim), nn.LayerNorm(dim))
        self.layers   = nn.ModuleList([_FusionLayer(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, vlm_tokens, geo_tokens, proprio_token, vlm_mask=None):
        B = geo_tokens.shape[0]
        vlm = self.vlm_proj(vlm_tokens)
        ctx = torch.cat([vlm, geo_tokens, proprio_token], dim=1)  # all tokens as KV
        if vlm_mask is not None:
            extra = torch.zeros(B, geo_tokens.shape[1] + 1,
                                dtype=torch.bool, device=ctx.device)
            mask = torch.cat([vlm_mask, extra], dim=1)
        else:
            mask = None
        q = self.queries[None].expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, ctx, mask)
        return self.out_norm(q)                                   # [B, num_queries, dim]


# ---------------------------------------------------------------------------
# Flow-matching DiT action head
# ---------------------------------------------------------------------------
def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half  = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args  = t[:, None].float() * freqs[None]
    emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        sc, sh = self.proj(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + sc[:, None]) + sh[:, None]


class _DiTBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.ada1 = _AdaLN(dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ada2 = _AdaLN(dim, cond_dim)
        self.cross = CrossAttention(dim, heads=heads)
        self.ada3 = _AdaLN(dim, cond_dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, cond_vec, ctx):
        h = self.ada1(x, cond_vec)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross(self.ada2(x, cond_vec), ctx)
        x = x + self.mlp(self.ada3(x, cond_vec))
        return x


class FlowMatchingDiT(nn.Module):
    """Conditional denoiser (shared between flow-matching and diffusion variants)."""
    def __init__(self, action_dim: int, horizon: int, dim: int = 512,
                 depth: int = 8, heads: int = 8, cond_dim: int = 512):
        super().__init__()
        self.dim, self.horizon, self.action_dim = dim, horizon, action_dim
        self.in_proj  = nn.Linear(action_dim, dim)
        self.pos      = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)
        self.t_embed  = nn.Sequential(nn.Linear(dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.c_pool   = nn.Sequential(nn.Linear(dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.blocks   = nn.ModuleList([_DiTBlock(dim, cond_dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor):
        """x_t:[B,T,A]  t:[B] normalised to ~[0,1]  cond_tokens:[B,Q,dim]"""
        h        = self.in_proj(x_t) + self.pos
        temb     = self.t_embed(_sinusoidal_embedding(t * 1000.0, self.dim))
        cpool    = self.c_pool(cond_tokens.mean(dim=1))
        cond_vec = temb + cpool                                   # [B, cond_dim]
        for blk in self.blocks:
            h = blk(h, cond_vec, cond_tokens)
        return self.out_proj(self.out_norm(h))                    # [B, T, action_dim]


# ---------------------------------------------------------------------------
# Frequency-domain auxiliary loss  (DCT-II along time axis)
# ---------------------------------------------------------------------------
class FrequencyDomainLoss(nn.Module):
    def __init__(self, horizon: int, low_w: float = 1.0, high_w: float = 0.2):
        super().__init__()
        n = horizon
        k = torch.arange(n).float()
        basis = torch.cos(math.pi / n * k[:, None] * (k[None, :] + 0.5))  # [freq, time]
        self.register_buffer("basis",  basis)
        self.register_buffer("freq_w", torch.linspace(low_w, high_w, n))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, T, A]"""
        P = torch.einsum("ft,bta->bfa", self.basis, pred)
        G = torch.einsum("ft,bta->bfa", self.basis, target)
        return ((P - G) ** 2 * self.freq_w[None, :, None]).mean()


# ---------------------------------------------------------------------------
# Full policy (flow-matching)
# ---------------------------------------------------------------------------
class VL3DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim:      int   = 7,
        horizon:         int   = 8,
        proprio_dim:     int   = 9,
        vlm_dim:         int   = 2048,
        pc_in:           int   = 6,
        dim:             int   = 512,
        num_meta_queries:int   = 16,
        fusion_depth:    int   = 4,
        dit_depth:       int   = 8,
        heads:           int   = 8,
        num_geo_tokens:  int   = 8,
        low_freq_w:      float = 1.0,
        high_freq_w:     float = 0.2,
        use_belief_bank: bool  = False,
        use_obstacle_map: bool = False,
        use_route_token: bool  = False,
        use_cocos_source: bool = False,
        belief_dim:      int   = 11,
        max_goals:       int   = 16,
        obstacle_tokens: int   = 16,
        max_route_len:   int   = 32,
        cocos_alpha:     float = 1.0,
        cocos_beta:      float = 0.2,
        mean_loss_weight:float = 0.1,
        debug_condition: bool  = False,
    ):
        super().__init__()
        self.encoder    = DP3Encoder(in_channels=pc_in, out_dim=256)
        self.geo_tok    = GeoTokenizer(self.encoder.point_dim, self.encoder.out_dim,
                                       dim, num_geo_tokens, heads)
        self.prop_proj  = nn.Sequential(
            nn.Linear(proprio_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.fusion     = MetaQueryFusion(dim, vlm_dim, num_meta_queries, fusion_depth, heads)
        self.condition_adapter = NavDPConditionAdapter(
            embed_dim=dim,
            use_belief_bank=use_belief_bank,
            use_obstacle_map=use_obstacle_map,
            use_route_token=use_route_token,
            belief_dim=belief_dim,
            max_goals=max_goals,
            obstacle_tokens=obstacle_tokens,
            max_route_len=max_route_len,
            debug=debug_condition,
        )
        self.dit        = FlowMatchingDiT(action_dim, horizon, dim, dit_depth, heads, cond_dim=dim)
        self.cocos_source = (
            BeliefConditionedCocosSource(
                condition_dim=dim,
                action_dim=action_dim,
                horizon=horizon,
                alpha=cocos_alpha,
                beta=cocos_beta,
            )
            if use_cocos_source
            else None
        )
        self.mean_loss_weight = float(mean_loss_weight)
        self.freq_loss  = FrequencyDomainLoss(horizon, low_freq_w, high_freq_w)
        self.horizon    = horizon
        self.action_dim = action_dim

    def encode(
        self,
        pc,
        vlm,
        prop,
        vlm_mask=None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ):
        pf, g  = self.encoder(pc)
        geo    = self.geo_tok(pf, g)
        prop_t = self.prop_proj(prop)[:, None, :]      # [B, 1, dim]
        cond = self.fusion(vlm, geo, prop_t, vlm_mask) # [B, Q, dim]
        return self.condition_adapter(
            cond,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )

    def forward(self, pc, vlm, prop, actions, vlm_mask=None,
                freq_weight: float = 0.5,
                belief_tensor: Optional[torch.Tensor] = None,
                obstacle_map: Optional[torch.Tensor] = None,
                route_index: Optional[torch.Tensor] = None,
                active_goal_index: Optional[torch.Tensor] = None,
                mean_loss_weight: Optional[float] = None) -> Dict[str, torch.Tensor]:
        B    = actions.shape[0]
        cond = self.encode(
            pc,
            vlm,
            prop,
            vlm_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        t    = torch.rand(B, device=actions.device)
        mean_loss = actions.new_tensor(0.0)
        if self.cocos_source is not None:
            x0, source_mean = self.cocos_source.sample_source(cond)
            mean_loss = F.mse_loss(source_mean, actions)
        else:
            x0 = torch.randn_like(actions)
        xt   = (1 - t[:, None, None]) * x0 + t[:, None, None] * actions
        v_pred   = self.dit(xt, t, cond)
        v_target = actions - x0
        flow_loss = F.mse_loss(v_pred, v_target)
        x0_hat    = xt + (1 - t[:, None, None]) * v_pred   # reconstruct clean action
        fq        = self.freq_loss(x0_hat, actions)
        w_mean = self.mean_loss_weight if mean_loss_weight is None else float(mean_loss_weight)
        total = flow_loss + freq_weight * fq + w_mean * mean_loss
        return {
            "loss": total,
            "flow_loss": flow_loss,
            "freq_loss": fq,
            "mean_loss": mean_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        pc,
        vlm,
        prop,
        vlm_mask=None,
        steps: int = 10,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B    = pc.shape[0]
        cond = self.encode(
            pc,
            vlm,
            prop,
            vlm_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        if self.cocos_source is not None:
            x, _ = self.cocos_source.sample_source(cond)
        else:
            x = torch.randn(B, self.horizon, self.action_dim, device=pc.device)
        dt   = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=x.device)
            x = x + dt * self.dit(x, t, cond)
        return x     # [B, horizon, action_dim] in normalised space


if __name__ == "__main__":
    m  = VL3DiffusionPolicy()
    pc = torch.randn(2, 1024, 6)
    vl = torch.randn(2, 64, 2048)
    pr = torch.randn(2, 9)
    ac = torch.randn(2, 8, 7)
    out = m(pc, vl, pr, ac)
    print("losses:", {k: round(v.item(), 4) for k, v in out.items()})
    print("sample:", tuple(m.sample(pc, vl, pr, steps=6).shape))