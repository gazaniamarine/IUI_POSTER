from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .belief_encoder import NavDPConditionAdapter
from .relational_belief import RelationalBelief, RelationalBeliefOutput


class SinusoidalPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


class DiTCrossAttnBlock(nn.Module):
    """DiT block that denoises trajectory tokens against condition tokens."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True
        )
        self.norm_mem = nn.LayerNorm(dim)
        self.memory_cross_attn = nn.MultiheadAttention(
            dim, n_heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True
        )
        self.memory_gate = nn.Parameter(torch.zeros(()))

        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: torch.Tensor,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
        memory_ctx: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sm, ss, gm, fm, fs, gf = self.ada(c).chunk(6, dim=-1)

        def mod(z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return z * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x_mod = mod(self.norm_sa(x), sm, ss)
        x = x + gm.unsqueeze(1) * self.self_attn(x_mod, x_mod, x_mod, need_weights=False)[0]

        x = x + self.cross_attn(
            self.norm_ca(x),
            ctx,
            ctx,
            key_padding_mask=ctx_key_padding_mask,
            need_weights=False,
        )[0]

        if memory_ctx is not None:
            memory_delta = self.memory_cross_attn(
                self.norm_mem(x),
                memory_ctx,
                memory_ctx,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
            x = x + torch.tanh(self.memory_gate) * memory_delta

        x = x + gf.unsqueeze(1) * self.ff(mod(self.norm_ff(x), fm, fs))
        return x


class BeliefAnchoredSource(nn.Module):
    """Map active belief state to a source mean and uncertainty scale.

    The source is used to initialize diffusion sampling and to train the denoiser
    across memory-reliability-conditioned noise levels.

    Inputs:
        belief_tensor: [B, N_goals, belief_dim]

    Outputs:
        source_mean: [B, H, A]
        beta: [B, 1, 1]
        uncertainty: [B, 1]
    """

    def __init__(
        self,
        belief_dim: int,
        horizon: int,
        action_dim: int,
        hidden_dim: int,
        beta_min: float = 0.2,
        beta_max: float = 1.5,
    ):
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.mean_head = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(belief_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_token = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)
        target = (1.0 - self.beta_min) / max(self.beta_max - self.beta_min, 1e-6)
        target = float(min(max(target, 1e-4), 1.0 - 1e-4))
        nn.init.zeros_(self.beta_head[-1].weight)
        nn.init.constant_(self.beta_head[-1].bias, math.log(target / (1.0 - target)))

    def forward(
        self,
        belief_tensor: torch.Tensor,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        active = _select_active_belief(belief_tensor.float(), active_goal_index)
        uncertainty = _belief_uncertainty_scalar(active)
        source_mean = self.mean_head(active).view(-1, self.horizon, self.action_dim)
        beta_raw = self.beta_head(torch.cat([active, uncertainty], dim=-1))
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta_raw)
        return {
            "source_mean": source_mean,
            "beta": beta[:, None],
            "uncertainty": uncertainty,
            "uncertainty_token": self.uncertainty_token(uncertainty)[:, None, :],
        }


@dataclass
class PriorLoadReport:
    loaded: int
    skipped: int
    missing_in_target: int
    shape_mismatch: int


class BeliefAugmentedTrajectoryDiT(nn.Module):
    """Pretrained-condition Trajectory DiT with persistent belief tokens.

    Inputs:
        feat_ctx: [B, K, feat_dim], optional raw SAM2Act feature history
        proprio_ctx: [B, K, proprio_dim], optional proprio history
        prior_ctx_tokens: [B, Q, dit_dim], optional precomputed prior tokens
        belief_tensor: [B, N_goals, 11]
        obstacle_map: [B, H_map, W_map] or [B, 1, H_map, W_map]
        route_index: [B]
        traj: [B, H, act_dim]

    The old prior context is preserved and the belief/route/obstacle features
    are appended as extra cross-attention tokens.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        proprio_dim: int = 8,
        h_steps: int = 20,
        act_dim: int = 3,
        dit_dim: int = 256,
        depth: int = 6,
        n_heads: int = 4,
        max_ctx: int = 8,
        T: int = 100,
        use_belief_bank: bool = True,
        use_obstacle_map: bool = True,
        use_route_token: bool = True,
        belief_dim: int = 11,
        max_goals: int = 16,
        obstacle_tokens: int = 16,
        max_route_len: int = 32,
        use_relational_belief: bool = True,
        relational_depth: int = 2,
        use_belief_source: bool = True,
        source_beta_min: float = 0.2,
        source_beta_max: float = 1.5,
        debug_condition: bool = False,
    ):
        super().__init__()
        self.h_steps = int(h_steps)
        self.act_dim = int(act_dim)
        self.dit_dim = int(dit_dim)
        self.max_ctx = int(max_ctx)
        self.T = int(T)
        self.raw_belief_dim = int(belief_dim)
        self.refined_belief_dim = int(belief_dim + 2) if use_relational_belief else int(belief_dim)
        self.use_relational_belief = bool(use_relational_belief)

        self.ctx_proj = nn.Sequential(
            nn.Linear(feat_dim + proprio_dim, dit_dim),
            nn.LayerNorm(dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim),
        )
        self.ctx_pos = nn.Parameter(torch.randn(1, max_ctx, dit_dim) * 0.02)

        self.relational_belief = (
            RelationalBelief(
                input_dim=belief_dim,
                hidden_dim=dit_dim,
                max_goals=max_goals,
                depth=relational_depth,
                heads=n_heads,
            )
            if self.use_relational_belief
            else None
        )
        self.condition_adapter = NavDPConditionAdapter(
            embed_dim=dit_dim,
            use_belief_bank=use_belief_bank,
            use_obstacle_map=use_obstacle_map,
            use_route_token=use_route_token,
            belief_dim=self.refined_belief_dim,
            max_goals=max_goals,
            obstacle_tokens=obstacle_tokens,
            max_route_len=max_route_len,
            debug=debug_condition,
        )
        self.use_belief_source = bool(use_belief_source)
        self.belief_source = (
            BeliefAnchoredSource(
                belief_dim=self.refined_belief_dim,
                horizon=h_steps,
                action_dim=act_dim,
                hidden_dim=dit_dim,
                beta_min=source_beta_min,
                beta_max=source_beta_max,
            )
            if self.use_belief_source
            else None
        )

        self.traj_embed = nn.Linear(act_dim, dit_dim)
        self.traj_pos = nn.Parameter(torch.randn(1, h_steps, dit_dim) * 0.02)

        self.t_embed = nn.Sequential(
            SinusoidalPE(dit_dim),
            nn.Linear(dit_dim, dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim),
        )

        self.blocks = nn.ModuleList(
            [DiTCrossAttnBlock(dit_dim, n_heads, ctx_dim=dit_dim) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(dit_dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dit_dim, 2 * dit_dim))
        self.out_proj = nn.Linear(dit_dim, act_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.register_buffer("alphas_cumprod", self._cosine_schedule(T))
        self.register_buffer("traj_mean", torch.zeros(act_dim))
        self.register_buffer("traj_std", torch.ones(act_dim))

    @staticmethod
    def _cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
        steps = torch.arange(T + 1) / T
        ft = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
        return torch.clamp(ft / ft[0], 1e-5, 1.0)

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.traj_mean.copy_(mean.detach().to(self.traj_mean.device))
        self.traj_std.copy_(std.detach().to(self.traj_std.device).clamp(min=1e-3))

    def norm_traj(self, traj: torch.Tensor) -> torch.Tensor:
        return (traj - self.traj_mean) / self.traj_std

    def denorm_traj(self, traj: torch.Tensor) -> torch.Tensor:
        return traj * self.traj_std + self.traj_mean

    def encode_condition(
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
        """Return condition tokens [B, Q_total, D] and key padding mask."""
        base, base_mask, memory, memory_mask = self.encode_condition_split(
            feat_ctx=feat_ctx,
            proprio_ctx=proprio_ctx,
            prior_ctx_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        if memory is None:
            return base, base_mask
        tokens = torch.cat([base, memory], dim=1)
        if base_mask is None:
            return tokens, None
        if memory_mask is None:
            memory_mask = torch.zeros(
                base_mask.shape[0],
                memory.shape[1],
                dtype=torch.bool,
                device=base_mask.device,
            )
        return tokens, torch.cat([base_mask, memory_mask], dim=1)

    def refine_belief(
        self,
        belief_tensor: Optional[torch.Tensor],
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[RelationalBeliefOutput]]:
        if belief_tensor is None:
            return None, None
        if self.relational_belief is None:
            return belief_tensor, None
        if belief_tensor.shape[-1] != self.raw_belief_dim:
            return belief_tensor, None
        out = self.relational_belief(
            belief_tensor,
            active_goal_index=active_goal_index,
        )
        return out.belief_tensor, out

    def encode_condition_split(
        self,
        feat_ctx: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        prior_ctx_tokens: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return prior tokens and route-belief memory tokens separately."""
        if prior_ctx_tokens is not None:
            ctx_tokens = prior_ctx_tokens
            key_padding_mask = (~ctx_mask) if ctx_mask is not None else None
        else:
            if feat_ctx is None or proprio_ctx is None:
                raise ValueError("pass either prior_ctx_tokens or both feat_ctx and proprio_ctx")
            ctx_tokens = self._encode_raw_ctx(feat_ctx, proprio_ctx)
            key_padding_mask = (~ctx_mask) if ctx_mask is not None else None

        refined_belief, _ = self.refine_belief(belief_tensor, active_goal_index)
        memory_tokens = self.condition_adapter.encode_aux_tokens(
            ctx_tokens,
            belief_tensor=refined_belief,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        source = self._belief_source(
            belief_tensor=refined_belief,
            active_goal_index=active_goal_index,
        )
        if source is not None:
            memory_tokens = (
                source["uncertainty_token"]
                if memory_tokens is None
                else torch.cat([memory_tokens, source["uncertainty_token"]], dim=1)
            )
        memory_mask = None
        if memory_tokens is not None:
            memory_mask = torch.zeros(
                memory_tokens.shape[0],
                memory_tokens.shape[1],
                dtype=torch.bool,
                device=memory_tokens.device,
            )
        return ctx_tokens, key_padding_mask, memory_tokens, memory_mask

    def _encode_raw_ctx(self, feat_ctx: torch.Tensor, proprio_ctx: torch.Tensor) -> torch.Tensor:
        b, k = feat_ctx.shape[:2]
        x = torch.cat([feat_ctx, proprio_ctx], dim=-1)
        tokens = self.ctx_proj(x)
        pos_idx = torch.arange(k - 1, -1, -1, device=feat_ctx.device).clamp(0, self.max_ctx - 1)
        return tokens + self.ctx_pos[:, pos_idx, :]

    def forward_net(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        feat_ctx: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        prior_ctx_tokens: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx, key_padding_mask, memory, memory_mask = self.encode_condition_split(
            feat_ctx=feat_ctx,
            proprio_ctx=proprio_ctx,
            prior_ctx_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
            belief_tensor=belief_tensor,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        return self.denoise_with_condition_tokens(
            x_noisy,
            t,
            ctx,
            ctx_mask=key_padding_mask,
            memory_tokens=memory,
            memory_mask=memory_mask,
        )

    def denoise_with_condition_tokens(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition_tokens: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise using already-built condition tokens.

        Use this when another module has already appended belief/history tokens
        and should not run the condition adapter a second time.
        """
        z = self.traj_embed(x_noisy) + self.traj_pos
        c = self.t_embed(t)
        for block in self.blocks:
            z = block(
                z,
                c,
                condition_tokens,
                ctx_key_padding_mask=ctx_mask,
                memory_ctx=memory_tokens,
                memory_key_padding_mask=memory_mask,
            )
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        z = self.final_norm(z) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out_proj(z)

    def loss(
        self,
        traj: torch.Tensor,
        feat_ctx: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        prior_ctx_tokens: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        belief_tensor: Optional[torch.Tensor] = None,
        obstacle_map: Optional[torch.Tensor] = None,
        route_index: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        teacher_eps: Optional[torch.Tensor] = None,
        teacher_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        b = traj.shape[0]
        x0 = self.norm_traj(traj)
        t = torch.randint(0, self.T, (b,), device=traj.device)
        acp = self.alphas_cumprod[t].view(-1, 1, 1)
        eps = torch.randn_like(x0)
        refined_belief, relational_out = self.refine_belief(belief_tensor, active_goal_index)
        source = self._belief_source(refined_belief, active_goal_index)
        beta = source["beta"] if source is not None else 1.0
        x_t = x0 * acp.sqrt() + eps * (1.0 - acp).sqrt() * beta

        eps_pred = self.forward_net(
            x_t,
            t,
            feat_ctx=feat_ctx,
            proprio_ctx=proprio_ctx,
            prior_ctx_tokens=prior_ctx_tokens,
            ctx_mask=ctx_mask,
            belief_tensor=refined_belief,
            obstacle_map=obstacle_map,
            route_index=route_index,
            active_goal_index=active_goal_index,
        )
        diffusion_loss = F.mse_loss(eps_pred, eps)
        source_loss = traj.new_tensor(0.0)
        if source is not None:
            source_loss = F.mse_loss(source["source_mean"], x0)
        relational_reg = traj.new_tensor(0.0)
        if relational_out is not None:
            # Keep the zero-init relational correction conservative unless the
            # policy or reconstruction losses justify moving it.
            relational_reg = relational_out.delta_mu.square().mean()
        teacher_loss = traj.new_tensor(0.0)
        if teacher_eps is not None and teacher_weight > 0.0:
            teacher_loss = F.mse_loss(eps_pred, teacher_eps.detach())
        total = diffusion_loss + 0.1 * source_loss + 1e-3 * relational_reg + float(teacher_weight) * teacher_loss
        return {
            "loss": total,
            "diffusion_loss": diffusion_loss,
            "source_loss": source_loss,
            "relational_reg": relational_reg,
            "teacher_loss": teacher_loss,
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
        warm_x: Optional[torch.Tensor] = None,
        warm_sigma: float = 0.05,
        strength: Optional[float | torch.Tensor] = None,
        ddim_steps: int = 10,
    ) -> torch.Tensor:
        b = _infer_batch_size(feat_ctx, prior_ctx_tokens, belief_tensor, obstacle_map)
        device = _infer_device(feat_ctx, prior_ctx_tokens, belief_tensor, obstacle_map, self.traj_mean)
        refined_belief, _ = self.refine_belief(belief_tensor, active_goal_index)
        source = self._belief_source(refined_belief, active_goal_index)
        t_start = self.T - 1
        if warm_x is not None:
            warm = self.norm_traj(warm_x.to(device))
            if strength is None:
                x = warm + warm_sigma * torch.randn(b, self.h_steps, self.act_dim, device=device)
            else:
                t_start = _strength_to_timestep(strength, self.T)
                acp = self.alphas_cumprod[t_start].view(1, 1, 1)
                beta = source["beta"] if source is not None else 1.0
                x = acp.sqrt() * warm + (1.0 - acp).sqrt() * beta * torch.randn_like(warm)
        elif source is not None:
            x = source["source_mean"] + source["beta"] * torch.randn(
                b, self.h_steps, self.act_dim, device=device
            )
        else:
            x = torch.randn(b, self.h_steps, self.act_dim, device=device)

        steps = torch.linspace(t_start, 0, ddim_steps, device=device).long()
        for i, t_now in enumerate(steps):
            t_prev = steps[i + 1] if i + 1 < ddim_steps else torch.tensor(0, device=device)
            tb = torch.full((b,), int(t_now.item()), device=device, dtype=torch.long)
            eps = self.forward_net(
                x,
                tb,
                feat_ctx=feat_ctx,
                proprio_ctx=proprio_ctx,
                prior_ctx_tokens=prior_ctx_tokens,
                ctx_mask=ctx_mask,
                belief_tensor=refined_belief,
                obstacle_map=obstacle_map,
                route_index=route_index,
                active_goal_index=active_goal_index,
            )
            acp_now = self.alphas_cumprod[t_now]
            acp_prev = self.alphas_cumprod[t_prev]
            beta = source["beta"] if source is not None else 1.0
            x0_pred = ((x - (1.0 - acp_now).sqrt() * beta * eps) / acp_now.sqrt()).clamp(-4.0, 4.0)
            x = (
                acp_prev.sqrt() * x0_pred + (1.0 - acp_prev).sqrt() * beta * eps
                if int(t_prev.item()) > 0
                else x0_pred
            )
        return self.denorm_traj(x)

    def _belief_source(
        self,
        belief_tensor: Optional[torch.Tensor],
        active_goal_index: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.belief_source is None or belief_tensor is None:
            return None
        return self.belief_source(belief_tensor, active_goal_index)

    def load_prior_weights(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict_shapes: bool = True,
    ) -> PriorLoadReport:
        """Load all checkpoint tensors whose names/shapes match this model."""
        target = self.state_dict()
        merged = dict(target)
        loaded = 0
        missing = 0
        mismatch = 0
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                missing += 1
                continue
            if key not in target:
                missing += 1
                continue
            if strict_shapes and tuple(target[key].shape) != tuple(value.shape):
                mismatch += 1
                continue
            merged[key] = value
            loaded += 1
        self.load_state_dict(merged)
        return PriorLoadReport(
            loaded=loaded,
            skipped=len(state_dict) - loaded,
            missing_in_target=missing,
            shape_mismatch=mismatch,
        )

    def set_train_stage(self, stage: str) -> None:
        """Freeze schedule for warm-start training.

        stage="adapter": train only belief/route/obstacle adapter, memory
            attention branch, and belief-anchored source.
        stage="top": train adapter plus final denoising layers.
        stage="all": train everything.
        """
        if stage not in {"adapter", "top", "all"}:
            raise ValueError("stage must be 'adapter', 'top', or 'all'")
        for param in self.parameters():
            param.requires_grad_(stage == "all")
        if stage == "adapter":
            self._enable_adapter_parameters()
            return
        if stage == "top":
            self._enable_adapter_parameters()
            for module in (self.blocks[-1], self.final_ada, self.out_proj):
                for param in module.parameters():
                    param.requires_grad_(True)

    def _enable_adapter_parameters(self) -> None:
        for param in self.condition_adapter.parameters():
            param.requires_grad_(True)
        if self.belief_source is not None:
            for param in self.belief_source.parameters():
                param.requires_grad_(True)
        for block in self.blocks:
            for module in (block.norm_mem, block.memory_cross_attn):
                for param in module.parameters():
                    param.requires_grad_(True)
            block.memory_gate.requires_grad_(True)


def load_checkpoint_state(path: str, map_location: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("ema", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return _strip_common_prefixes(ckpt[key])
    return _strip_common_prefixes(ckpt)


def _strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "ema.")
    out = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def _infer_batch_size(*tensors: Optional[torch.Tensor]) -> int:
    for tensor in tensors:
        if tensor is not None:
            return int(tensor.shape[0])
    raise ValueError("cannot infer batch size without an input tensor")


def _infer_device(*tensors: Optional[torch.Tensor]) -> torch.device:
    for tensor in tensors:
        if tensor is not None:
            return tensor.device
    return torch.device("cpu")


def _select_active_belief(
    belief_tensor: torch.Tensor,
    active_goal_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, n, _ = belief_tensor.shape
    if active_goal_index is None:
        active_goal_index = belief_tensor[..., 9].argmax(dim=1)
    active_goal_index = active_goal_index.to(device=belief_tensor.device, dtype=torch.long).clamp(0, n - 1)
    gather_idx = active_goal_index[:, None, None].expand(-1, 1, belief_tensor.shape[-1])
    return belief_tensor.gather(1, gather_idx).squeeze(1)


def _belief_uncertainty_scalar(active_belief: torch.Tensor) -> torch.Tensor:
    # Features 2 and 4 are Sigma_xx and Sigma_yy in the default 11D layout.
    if active_belief.shape[-1] >= 9:
        sigma_trace = 0.5 * (active_belief[:, 2:3].abs() + active_belief[:, 4:5].abs())
        time_since_seen = active_belief[:, 7:8].clamp_min(0.0) / 20.0
        confidence = active_belief[:, 8:9].clamp(0.0, 1.0)
        uncertainty = torch.log1p(sigma_trace) + time_since_seen + (1.0 - confidence)
        return uncertainty.clamp(0.0, 10.0)
    return active_belief.new_zeros(active_belief.shape[0], 1)


def _strength_to_timestep(strength: float | torch.Tensor, T: int) -> int:
    if torch.is_tensor(strength):
        value = float(strength.detach().float().mean().clamp(0.0, 1.0).item())
    else:
        value = float(min(max(strength, 0.0), 1.0))
    return int(round((T - 1) * value))
