from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from navdp.models import BeliefConditionedCocosSource, NavDPConditionAdapter
from model import CrossAttention, FlowMatchingDiT, FrequencyDomainLoss


def normalize_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    
    if depth.dim() == 3:
        depth = depth[:, None]
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B,H,W] or [B,1,H,W]")
    flat = depth.flatten(1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1)
    return (depth - lo) / (hi - lo + eps)


def build_spatial_semantic(
    semantic: torch.Tensor,
    depth: torch.Tensor,
    normalize_depth_input: bool = True,
) -> torch.Tensor:
    if semantic.dim() == 3:
        semantic = semantic[:, None]
    if semantic.dim() != 4:
        raise ValueError("semantic must have shape [B,H,W] or [B,C,H,W]")
    depth = normalize_depth(depth) if normalize_depth_input else depth
    if depth.dim() == 3:
        depth = depth[:, None]
    if semantic.shape[0] != depth.shape[0] or semantic.shape[-2:] != depth.shape[-2:]:
        raise ValueError("semantic and depth must share batch, height, and width")
    return torch.cat([semantic.float(), depth.float()], dim=1)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return betas.clamp(1e-4, 0.999)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        else:
            self.skip = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


def _sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    half = max(dim // 2, 1)
    omega = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=pos.device, dtype=torch.float32)
        / half
    )
    out = pos.float()[:, None] * omega[None]
    emb = torch.cat([out.sin(), out.cos()], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def _sincos_2d(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    y_dim = dim // 2
    x_dim = dim - y_dim
    pos = torch.cat(
        [
            _sincos_1d(yy.flatten(), y_dim),
            _sincos_1d(xx.flatten(), x_dim),
        ],
        dim=1,
    )
    return pos.to(dtype=dtype)[None]


class SpatialSemanticEncoder(nn.Module):
    """CNN encoder plus learnable query pooling for S2 observations."""

    def __init__(
        self,
        in_channels: int = 2,
        dim: int = 512,
        width: int = 64,
        num_queries: int = 16,
        heads: int = 8,
        proprio_dim: int = 9,
    ):
        super().__init__()
        self.visual = nn.Sequential(
            _ResBlock(in_channels, width, stride=2),
            _ResBlock(width, width, stride=1),
            _ResBlock(width, width * 2, stride=2),
            _ResBlock(width * 2, width * 2, stride=1),
            _ResBlock(width * 2, width * 4, stride=2),
            _ResBlock(width * 4, width * 4, stride=1),
            _ResBlock(width * 4, dim, stride=2),
        )
        self.token_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.pool = CrossAttention(dim, kv_dim=dim, heads=heads)
        self.out_norm = nn.LayerNorm(dim)
        self.proprio_dim = proprio_dim
        if proprio_dim > 0:
            self.proj_proprio = nn.Sequential(
                nn.Linear(proprio_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        else:
            self.proj_proprio = None

    def forward(
        self,
        spatial_semantic: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.visual(spatial_semantic)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + _sincos_2d(h, w, c, feat.device, feat.dtype)
        tokens = self.token_norm(tokens)

        q = self.queries[None].expand(b, -1, -1)
        cond = self.out_norm(q + self.pool(q, tokens))

        if self.proj_proprio is not None:
            if proprio is None:
                raise ValueError("proprio is required when proprio_dim > 0")
            prop_token = self.proj_proprio(proprio)[:, None]
            cond = torch.cat([cond, prop_token], dim=1)
        return cond


class S2DiTPolicy(nn.Module):
    """Open-vocabulary spatial-semantic diffusion policy with a DiT denoiser."""

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 8,
        proprio_dim: int = 9,
        spatial_channels: int = 2,
        dim: int = 512,
        encoder_width: int = 64,
        num_cond_queries: int = 16,
        dit_depth: int = 8,
        heads: int = 8,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        low_freq_w: float = 1.0,
        high_freq_w: float = 0.2,
        clip_sample: bool = True,
        use_belief_bank: bool = False,
        use_obstacle_map: bool = False,
        use_route_token: bool = False,
        use_cocos_source: bool = False,
        belief_dim: int = 11,
        max_goals: int = 16,
        obstacle_tokens: int = 16,
        max_route_len: int = 32,
        cocos_alpha: float = 1.0,
        cocos_beta: float = 0.2,
        mean_loss_weight: float = 0.1,
        normalize_belief: bool = True,
        debug_condition: bool = False,
    ):
        super().__init__()
        self.encoder = SpatialSemanticEncoder(
            in_channels=spatial_channels,
            dim=dim,
            width=encoder_width,
            num_queries=num_cond_queries,
            heads=heads,
            proprio_dim=proprio_dim,
        )
        self.condition_adapter = NavDPConditionAdapter(
            embed_dim=dim,
            use_belief_bank=use_belief_bank,
            use_obstacle_map=use_obstacle_map,
            use_route_token=use_route_token,
            belief_dim=belief_dim,
            max_goals=max_goals,
            obstacle_tokens=obstacle_tokens,
            max_route_len=max_route_len,
            normalize_belief=normalize_belief,
            debug=debug_condition,
        )
        self.dit = FlowMatchingDiT(
            action_dim=action_dim,
            horizon=horizon,
            dim=dim,
            depth=dit_depth,
            heads=heads,
            cond_dim=dim,
        )
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
        self.freq_loss = FrequencyDomainLoss(horizon, low_freq_w, high_freq_w)
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.clip_sample = clip_sample

        betas = _cosine_beta_schedule(num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - alphas_cumprod).sqrt(),
        )

    def encode(
        self,
        spatial_semantic: torch.Tensor,
        proprio: torch.Tensor,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        extra_cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self.encoder(spatial_semantic, proprio)
        cond = self.condition_adapter(
            cond,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        if extra_cond_tokens is not None:
            # Append e.g. a language token to the conditioning set the DiT cross-attends to.
            cond = torch.cat([cond, extra_cond_tokens.to(cond.dtype)], dim=1)
        return cond

    def add_noise(
        self,
        clean_actions: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        return sqrt_alpha * clean_actions + sqrt_one_minus * noise

    def forward(
        self,
        spatial_semantic: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        freq_weight: float = 0.5,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        mean_loss_weight: Optional[float] = None,
        extra_cond_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = actions.shape[0]
        cond = self.encode(
            spatial_semantic,
            proprio,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
            extra_cond_tokens=extra_cond_tokens,
        )
        noise = torch.randn_like(actions)
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (b,),
            device=actions.device,
            dtype=torch.long,
        )
        noisy = self.add_noise(actions, noise, t)
        t_norm = t.float() / self.num_train_timesteps

        pred_x0 = self.dit(noisy, t_norm, cond)
        denoise_loss = F.mse_loss(pred_x0, actions)
        fq = self.freq_loss(pred_x0, actions)
        mean_loss = actions.new_tensor(0.0)
        if self.cocos_source is not None:
            source_mean = self.cocos_source(cond)
            mean_loss = F.mse_loss(source_mean, actions)
        w_mean = self.mean_loss_weight if mean_loss_weight is None else float(mean_loss_weight)
        return {
            "loss": denoise_loss + freq_weight * fq + w_mean * mean_loss,
            "denoise_loss": denoise_loss,
            "freq_loss": fq,
            "mean_loss": mean_loss,
        }

    def _ddim_prev(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        t: int,
        prev_t: int,
    ) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t].view(1, 1, 1)
        if prev_t >= 0:
            alpha_prev = self.alphas_cumprod[prev_t].view(1, 1, 1)
        else:
            alpha_prev = torch.ones_like(alpha_t)

        eps = (x_t - alpha_t.sqrt() * pred_x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-6)
        return alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * eps

    @torch.no_grad()
    def sample(
        self,
        spatial_semantic: torch.Tensor,
        proprio: torch.Tensor,
        steps: Optional[int] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        guidance_fn: Optional[callable] = None,
        extra_cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = spatial_semantic.shape[0]
        cond = self.encode(
            spatial_semantic,
            proprio,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
            extra_cond_tokens=extra_cond_tokens,
        )
        if noise is not None:
            # Fixed initial noise keeps DDIM deterministic across control steps,
            # so the sampled chunk drifts smoothly with the conditioning instead
            # of jumping between independent diffusion modes each step.
            x = noise.to(device=spatial_semantic.device, dtype=spatial_semantic.dtype)
        elif self.cocos_source is not None:
            x, _ = self.cocos_source.sample_source(cond)
        else:
            x = torch.randn(
                b,
                self.horizon,
                self.action_dim,
                device=spatial_semantic.device,
                dtype=spatial_semantic.dtype,
            )
        n = steps or self.num_inference_steps
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            n,
            device=spatial_semantic.device,
        ).long()

        for i, t_tensor in enumerate(timesteps):
            t = int(t_tensor.item())
            tb = torch.full((b,), t, device=x.device, dtype=torch.float32)
            pred_x0 = self.dit(x, tb / self.num_train_timesteps, cond)
            if guidance_fn is not None:
                # Safety guidance: nudge the predicted clean action chunk toward a
                # safe trajectory (e.g. horizon-CBF) at each denoising step, so the
                # final sample is safe by construction. The callback takes and
                # returns [B, horizon, action_dim].
                pred_x0 = guidance_fn(pred_x0)
            if self.clip_sample:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
            prev_t = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            x = self._ddim_prev(x, pred_x0, t, prev_t)
        return x


S2DiffusionDiT = S2DiTPolicy


if __name__ == "__main__":
    model = S2DiTPolicy(spatial_channels=2, dim=128, encoder_width=32, dit_depth=2, heads=4)
    sem = torch.rand(2, 1, 128, 128)
    dep = torch.rand(2, 1, 128, 128)
    z = build_spatial_semantic(sem, dep)
    prop = torch.randn(2, 9)
    act = torch.randn(2, 8, 7).clamp(-1, 1)
    out = model(z, prop, act)
    print("losses:", {k: round(v.item(), 4) for k, v in out.items()})
    print("sample:", tuple(model.sample(z, prop, steps=4).shape))
