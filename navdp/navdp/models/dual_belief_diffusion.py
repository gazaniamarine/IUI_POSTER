from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .belief_augmented_traj_dit import (
    BeliefAugmentedTrajectoryDiT,
    DiTCrossAttnBlock,
    SinusoidalPE,
)


class DualHeadConditionedDiT(nn.Module):
    """Conditioned DiT with noise and log-variance heads.

    This is the backward/context diffusion in the dual-diffusion stack. It
    reconstructs an unobserved belief/history trajectory and estimates
    aleatoric uncertainty in one pass.

    Shapes:
        x_noisy: [B, H_back, state_dim]
        condition_tokens: [B, Q, dit_dim]
        eps_pred: [B, H_back, state_dim]
        logvar_pred: [B, H_back, state_dim]
    """

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        dit_dim: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        T: int = 100,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.dit_dim = int(dit_dim)
        self.T = int(T)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.in_proj = nn.Linear(state_dim, dit_dim)
        self.pos = nn.Parameter(torch.randn(1, horizon, dit_dim) * 0.02)
        self.t_embed = nn.Sequential(
            SinusoidalPE(dit_dim),
            nn.Linear(dit_dim, dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim),
        )
        self.blocks = nn.ModuleList(
            [DiTCrossAttnBlock(dit_dim, n_heads, ctx_dim=dit_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dit_dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dit_dim, 2 * dit_dim))
        self.eps_head = nn.Linear(dit_dim, state_dim)
        self.logvar_head = nn.Linear(dit_dim, state_dim)
        nn.init.zeros_(self.eps_head.weight)
        nn.init.zeros_(self.eps_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -4.0)

        self.register_buffer("alphas_cumprod", self._cosine_schedule(T))

    @staticmethod
    def _cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
        steps = torch.arange(T + 1) / T
        ft = torch.cos((steps + s) / (1.0 + s) * torch.pi / 2.0) ** 2
        return torch.clamp(ft / ft[0], 1e-5, 1.0)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.in_proj(x_noisy) + self.pos
        c = self.t_embed(t)
        for block in self.blocks:
            z = block(z, c, condition_tokens, ctx_key_padding_mask=condition_mask)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        z = self.norm(z) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        eps = self.eps_head(z)
        logvar = self.logvar_head(z).clamp(self.logvar_min, self.logvar_max)
        return eps, logvar

    def loss(
        self,
        clean_state: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        nll_weight: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        b = clean_state.shape[0]
        t = torch.randint(0, self.T, (b,), device=clean_state.device)
        acp = self.alphas_cumprod[t].view(-1, 1, 1)
        eps = torch.randn_like(clean_state)
        x_t = acp.sqrt() * clean_state + (1.0 - acp).sqrt() * eps
        eps_pred, logvar = self.forward(x_t, t, condition_tokens, condition_mask)

        eps_loss = F.mse_loss(eps_pred, eps)
        residual = (eps_pred - eps).square()
        nll = 0.5 * (torch.exp(-logvar) * residual + logvar).mean()
        loss = eps_loss + float(nll_weight) * nll
        uncertainty = torch.exp(logvar).detach()
        return {
            "loss": loss,
            "backward_eps_loss": eps_loss,
            "backward_nll_loss": nll,
            "uncertainty": uncertainty,
            "eps_pred": eps_pred,
            "logvar": logvar,
        }

    @torch.no_grad()
    def sample(
        self,
        condition_tokens: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        ddim_steps: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = condition_tokens.shape[0]
        device = condition_tokens.device
        x = torch.randn(b, self.horizon, self.state_dim, device=device)
        last_logvar = torch.zeros_like(x)
        steps = torch.linspace(self.T - 1, 0, ddim_steps, device=device).long()
        for i, t_now in enumerate(steps):
            t_prev = steps[i + 1] if i + 1 < ddim_steps else torch.tensor(0, device=device)
            tb = torch.full((b,), int(t_now.item()), device=device, dtype=torch.long)
            eps, last_logvar = self.forward(x, tb, condition_tokens, condition_mask)
            acp_now = self.alphas_cumprod[t_now]
            acp_prev = self.alphas_cumprod[t_prev]
            x0 = ((x - (1.0 - acp_now).sqrt() * eps) / acp_now.sqrt()).clamp(-4.0, 4.0)
            x = (
                acp_prev.sqrt() * x0 + (1.0 - acp_prev).sqrt() * eps
                if int(t_prev.item()) > 0
                else x0
            )
        return x, torch.exp(last_logvar)


class AdaptiveNoiseSchedule(nn.Module):
    """Learn uncertainty-conditioned noise scale for forward waypoint diffusion.

    Input uncertainty can be [B, H_back, D] or [B, D]. The output scale is
    bounded and shaped [B, H_forward, action_dim].
    """

    def __init__(
        self,
        uncertainty_dim: int,
        horizon: int,
        action_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 128,
        min_scale: float = 0.5,
        max_scale: float = 1.8,
    ):
        super().__init__()
        self.uncertainty_dim = int(uncertainty_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.t_embed = SinusoidalPE(time_dim)
        self.net = nn.Sequential(
            nn.Linear(uncertainty_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, uncertainty: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if uncertainty.dim() == 3:
            u = uncertainty.mean(dim=1)
        elif uncertainty.dim() == 2:
            u = uncertainty
        else:
            raise ValueError("uncertainty must have shape [B,H,D] or [B,D]")
        u = _fit_last_dim(u, self.uncertainty_dim)
        h = torch.cat([u, self.t_embed(t.float())], dim=-1)
        raw = self.net(h).view(-1, self.horizon, self.action_dim)
        return self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(raw)


class DualBeliefDiffusionPolicy(nn.Module):
    """Dual-diffusion novelty layer for route-belief navigation.

    Stage 1 (backward/context diffusion):
        infer a missing belief/history trace and uncertainty.

    Stage 2 (forward waypoint diffusion):
        append the inferred trace as condition tokens and train/sample with
        uncertainty-adaptive noise.
    """

    def __init__(
        self,
        forward_policy: BeliefAugmentedTrajectoryDiT,
        belief_history_dim: int,
        belief_history_steps: int = 8,
        dit_dim: int = 256,
        backward_depth: int = 4,
        heads: int = 4,
        T: int = 100,
        history_token_dim: Optional[int] = None,
    ):
        super().__init__()
        self.forward_policy = forward_policy
        self.belief_history_dim = int(belief_history_dim)
        self.belief_history_steps = int(belief_history_steps)
        self.dit_dim = int(dit_dim)

        self.backward_diffusion = DualHeadConditionedDiT(
            state_dim=belief_history_dim,
            horizon=belief_history_steps,
            dit_dim=dit_dim,
            depth=backward_depth,
            n_heads=heads,
            T=T,
        )
        hidden = int(history_token_dim or dit_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(2 * belief_history_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, dit_dim),
        )
        self.adaptive_noise = AdaptiveNoiseSchedule(
            uncertainty_dim=belief_history_dim,
            horizon=forward_policy.h_steps,
            action_dim=forward_policy.act_dim,
        )

    def encode_base_condition(
        self,
        feat_ctx: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        prior_ctx_tokens: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward_policy.encode_condition(
            feat_ctx=feat_ctx,
            proprio_ctx=proprio_ctx,
            prior_ctx_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )

    def encode_history_tokens(
        self,
        belief_history: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat([belief_history, uncertainty], dim=-1)
        return self.history_encoder(h)

    def forward_loss(
        self,
        traj: torch.Tensor,
        prior_ctx_tokens: torch.Tensor,
        uncertainty: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = traj.shape[0]
        x0 = self.forward_policy.norm_traj(traj)
        t = torch.randint(0, self.forward_policy.T, (b,), device=traj.device)
        acp = self.forward_policy.alphas_cumprod[t].view(-1, 1, 1)
        eps = torch.randn_like(x0)
        noise_scale = self.adaptive_noise(uncertainty, t)
        x_t = acp.sqrt() * x0 + (1.0 - acp).sqrt() * noise_scale * eps

        eps_pred = self.forward_policy.denoise_with_condition_tokens(
            x_t,
            t,
            condition_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
        )
        forward_loss = F.mse_loss(eps_pred, eps)
        return {
            "forward_loss": forward_loss,
            "noise_scale_mean": noise_scale.mean().detach(),
        }

    def loss(
        self,
        traj: torch.Tensor,
        belief_history_target: torch.Tensor,
        feat_ctx: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        prior_ctx_tokens: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        backward_weight: float = 1.0,
        forward_weight: float = 1.0,
        nll_weight: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        base_tokens, base_mask = self.encode_base_condition(
            feat_ctx=feat_ctx,
            proprio_ctx=proprio_ctx,
            prior_ctx_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        back = self.backward_diffusion.loss(
            belief_history_target,
            base_tokens,
            condition_mask=base_mask,
            nll_weight=nll_weight,
        )
        # During supervised training, use the ground-truth hidden belief trace
        # but the predicted uncertainty. At inference both are sampled.
        hist_tokens = self.encode_history_tokens(belief_history_target, back["uncertainty"])
        aug_tokens = torch.cat([base_tokens, hist_tokens], dim=1)
        aug_mask = _append_valid_tokens(base_mask, hist_tokens.shape[1])
        fwd = self.forward_loss(traj, aug_tokens, back["uncertainty"], ctx_mask=aug_mask)
        total = float(backward_weight) * back["loss"] + float(forward_weight) * fwd["forward_loss"]
        return {
            "loss": total,
            "backward_loss": back["loss"],
            "backward_eps_loss": back["backward_eps_loss"],
            "backward_nll_loss": back["backward_nll_loss"],
            "forward_loss": fwd["forward_loss"],
            "noise_scale_mean": fwd["noise_scale_mean"],
        }

    @torch.no_grad()
    def sample(
        self,
        feat_ctx: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        prior_ctx_tokens: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        backward_steps: int = 10,
        forward_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        base_tokens, base_mask = self.encode_base_condition(
            feat_ctx=feat_ctx,
            proprio_ctx=proprio_ctx,
            prior_ctx_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        belief_history, uncertainty = self.backward_diffusion.sample(
            base_tokens,
            condition_mask=base_mask,
            ddim_steps=backward_steps,
        )
        hist_tokens = self.encode_history_tokens(belief_history, uncertainty)
        aug_tokens = torch.cat([base_tokens, hist_tokens], dim=1)
        aug_mask = _append_valid_tokens(base_mask, hist_tokens.shape[1])
        traj = self._sample_forward_with_adaptive_noise(
            aug_tokens,
            uncertainty,
            ctx_mask=aug_mask,
            ddim_steps=forward_steps,
        )
        return {
            "traj": traj,
            "belief_history": belief_history,
            "uncertainty": uncertainty,
        }

    def _sample_forward_with_adaptive_noise(
        self,
        condition_tokens: torch.Tensor,
        uncertainty: torch.Tensor,
        ctx_mask: Optional[torch.Tensor],
        ddim_steps: int,
    ) -> torch.Tensor:
        b = condition_tokens.shape[0]
        device = condition_tokens.device
        x = torch.randn(b, self.forward_policy.h_steps, self.forward_policy.act_dim, device=device)
        steps = torch.linspace(self.forward_policy.T - 1, 0, ddim_steps, device=device).long()
        for i, t_now in enumerate(steps):
            t_prev = steps[i + 1] if i + 1 < ddim_steps else torch.tensor(0, device=device)
            tb = torch.full((b,), int(t_now.item()), device=device, dtype=torch.long)
            scale_now = self.adaptive_noise(uncertainty, tb)
            eps = self.forward_policy.denoise_with_condition_tokens(
                x,
                tb,
                condition_tokens=condition_tokens,
                ctx_mask=ctx_mask,
            )
            acp_now = self.forward_policy.alphas_cumprod[t_now]
            acp_prev = self.forward_policy.alphas_cumprod[t_prev]
            x0 = ((x - (1.0 - acp_now).sqrt() * scale_now * eps) / acp_now.sqrt()).clamp(-4.0, 4.0)
            if int(t_prev.item()) > 0:
                tb_prev = torch.full((b,), int(t_prev.item()), device=device, dtype=torch.long)
                scale_prev = self.adaptive_noise(uncertainty, tb_prev)
                x = acp_prev.sqrt() * x0 + (1.0 - acp_prev).sqrt() * scale_prev * eps
            else:
                x = x0
        return self.forward_policy.denorm_traj(x)


def _append_valid_tokens(mask: Optional[torch.Tensor], num_extra: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    extra = torch.zeros(mask.shape[0], num_extra, dtype=torch.bool, device=mask.device)
    return torch.cat([mask, extra], dim=1)


def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    return F.pad(x, (0, dim - x.shape[-1]))
