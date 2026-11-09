from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RelationalBeliefOutput:
    """Output of the relational amortized belief layer.

    Shapes:
        mu: [B, N, 2]
        chol: [B, N, 3] lower-triangular covariance factors [l00, l10, l11]
        sigma_ale: [B, N, 2]
        sigma_epi: [B, N, 2]
        belief_tensor: [B, N, 13]
    """

    mu: torch.Tensor
    chol: torch.Tensor
    sigma_ale: torch.Tensor
    sigma_epi: torch.Tensor
    delta_mu: torch.Tensor
    belief_tensor: torch.Tensor
    visible: torch.Tensor
    active: torch.Tensor


class RelationalBelief(nn.Module):
    """Permutation-equivariant relational belief refinement.

    Input bank tensor layout, default [B, N, 11]:
        [mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
         visible, initialized, time_since_seen, confidence,
         is_active, route_index_normalized]

    Output refined tensor layout, default [B, N, 13]:
        [mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
         visible, initialized, time_since_seen, confidence,
         is_active, route_index_normalized, sigma_ale_mean, sigma_epi_mean]

    The mean correction head is zero-initialized and gated by occlusion/recency,
    so the module starts as an exact identity on the deterministic belief means.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        max_goals: int = 16,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        max_correction: float = 2.0,
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_goals = int(max_goals)
        self.max_correction = float(max_correction)
        self.min_sigma = float(min_sigma)
        self._sigma_head_init = -8.0

        self.id_embed = nn.Embedding(max_goals, hidden_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.delta_head = nn.Linear(hidden_dim, 2)
        self.ale_head = nn.Linear(hidden_dim, 2)
        self.epi_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        # Start with small learned additions, so aleatoric uncertainty is the
        # bank covariance at initialization and epistemic is near zero.
        nn.init.zeros_(self.ale_head.weight)
        nn.init.constant_(self.ale_head.bias, self._sigma_head_init)
        nn.init.zeros_(self.epi_head.weight)
        nn.init.constant_(self.epi_head.bias, self._sigma_head_init)

    def forward(
        self,
        bank_tensor: torch.Tensor,
        goal_ids: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
    ) -> RelationalBeliefOutput:
        if bank_tensor.dim() == 2:
            bank_tensor = bank_tensor.unsqueeze(0)
        if bank_tensor.dim() != 3:
            raise ValueError("bank_tensor must have shape [B,N,F] or [N,F]")
        b, n, f = bank_tensor.shape
        if f < 11:
            raise ValueError("bank_tensor must contain at least the default 11 features")
        if n > self.max_goals:
            raise ValueError(f"got {n} goals but max_goals={self.max_goals}")

        x = bank_tensor.float()
        if goal_ids is None:
            goal_ids = torch.arange(n, device=x.device)[None].expand(b, -1)
        goal_ids = goal_ids.to(device=x.device, dtype=torch.long).clamp(0, self.max_goals - 1)

        if active_goal_index is not None:
            active = torch.zeros(b, n, 1, device=x.device, dtype=x.dtype)
            active_idx = active_goal_index.to(device=x.device, dtype=torch.long).clamp(0, n - 1)
            active.scatter_(1, active_idx[:, None, None], 1.0)
            x = x.clone()
            x[..., 9:10] = active

        id_emb = self.id_embed(goal_ids)
        h = self.slot_mlp(torch.cat([x[..., : self.input_dim], id_emb], dim=-1))
        h = self.out_norm(self.set_encoder(h))

        visible = x[..., 5:6].clamp(0.0, 1.0)
        initialized = x[..., 6:7].clamp(0.0, 1.0)
        recency = x[..., 7:8].clamp_min(0.0)
        confidence = x[..., 8:9].clamp(0.0, 1.0)
        active = x[..., 9:10].clamp(0.0, 1.0)

        # Visible, high-confidence measurements should stay close to the sensor
        # update. Stale/invisible slots are allowed larger relational correction.
        correction_gate = initialized * (1.0 - visible * confidence) * (1.0 - torch.exp(-recency / 5.0))
        delta_mu = self.max_correction * torch.tanh(self.delta_head(h)) * correction_gate
        mu = x[..., :2] + delta_mu

        sigma_base = _sigma_from_bank(x)
        ale_delta = _zero_at_init_softplus(self.ale_head(h), self._sigma_head_init)
        epi_delta = _zero_at_init_softplus(self.epi_head(h), self._sigma_head_init)
        sigma_ale = (sigma_base + ale_delta * initialized).clamp_min(self.min_sigma)
        sigma_epi = (_epi_from_bank(x) + epi_delta * initialized).clamp_min(0.0)
        chol = _chol_from_diag_sigma(sigma_ale)
        cov = torch.cat([sigma_ale[..., 0:1].square(), x[..., 3:4], sigma_ale[..., 1:2].square()], dim=-1)
        belief_tensor = torch.cat(
            [
                mu,
                cov[..., 0:1],
                cov[..., 1:2],
                cov[..., 2:3],
                x[..., 5:11],
                sigma_ale.mean(dim=-1, keepdim=True),
                sigma_epi.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return RelationalBeliefOutput(
            mu=mu,
            chol=chol,
            sigma_ale=sigma_ale,
            sigma_epi=sigma_epi,
            delta_mu=delta_mu,
            belief_tensor=belief_tensor,
            visible=visible,
            active=active,
        )

    @staticmethod
    def occlusion_dropout(
        bank_tensor: torch.Tensor,
        drop_prob: float = 0.3,
        visible_only: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hide visible slots for self-supervised reconstruction.

        Returns:
            corrupted_bank: same shape as input
            drop_mask: [B, N] True where a target was hidden
        """
        if bank_tensor.dim() == 2:
            bank_tensor = bank_tensor.unsqueeze(0)
        x = bank_tensor.clone()
        visible = x[..., 5] > 0.5
        rand = torch.rand(x.shape[:2], device=x.device, generator=generator)
        drop_mask = rand < float(drop_prob)
        if visible_only:
            drop_mask = drop_mask & visible
        visible_feat = x[..., 5]
        recency_feat = x[..., 7]
        conf_feat = x[..., 8]
        visible_feat[drop_mask] = 0.0
        recency_feat[drop_mask] = recency_feat[drop_mask] + 1.0
        conf_feat[drop_mask] = 0.0
        return x, drop_mask

    def reconstruction_loss(
        self,
        corrupted_bank: torch.Tensor,
        target_bank: torch.Tensor,
        drop_mask: torch.Tensor,
        goal_ids: Optional[torch.Tensor] = None,
        active_goal_index: Optional[torch.Tensor] = None,
        epi_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Self-supervised occlusion-dropout loss.

        Gaussian NLL trains mean and aleatoric uncertainty. The epistemic head
        regresses reconstruction error magnitude.
        """
        out = self(corrupted_bank, goal_ids=goal_ids, active_goal_index=active_goal_index)
        target_mu = target_bank[..., :2].float()
        mask = drop_mask.to(device=target_mu.device, dtype=torch.bool)
        if mask.sum() == 0:
            zero = target_mu.new_tensor(0.0)
            return {"loss": zero, "nll_loss": zero, "epi_loss": zero, "num_dropped": zero}

        mu = out.mu[mask]
        sigma_ale = out.sigma_ale[mask].clamp_min(self.min_sigma)
        target = target_mu[mask]
        sq = (mu - target).square()
        nll = 0.5 * (sq / sigma_ale.square() + 2.0 * sigma_ale.log()).mean()
        err = (mu.detach() - target).abs()
        sigma_epi = out.sigma_epi[mask]
        epi_loss = F.smooth_l1_loss(sigma_epi, err)
        return {
            "loss": nll + float(epi_weight) * epi_loss,
            "nll_loss": nll,
            "epi_loss": epi_loss,
            "num_dropped": mask.sum().float(),
        }


def _sigma_from_bank(x: torch.Tensor) -> torch.Tensor:
    sx = x[..., 2:3].abs().clamp_min(1e-6).sqrt()
    sy = x[..., 4:5].abs().clamp_min(1e-6).sqrt()
    return torch.cat([sx, sy], dim=-1)


def _epi_from_bank(x: torch.Tensor) -> torch.Tensor:
    initialized = x[..., 6:7].clamp(0.0, 1.0)
    time_since_seen = x[..., 7:8].clamp_min(0.0)
    confidence = x[..., 8:9].clamp(0.0, 1.0)
    base = initialized * (time_since_seen / 20.0 + (1.0 - confidence))
    return base.expand(-1, -1, 2)


def _chol_from_diag_sigma(sigma: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(sigma[..., :1])
    return torch.cat([sigma[..., 0:1], zero, sigma[..., 1:2]], dim=-1)


def _cov_from_chol(chol: torch.Tensor) -> torch.Tensor:
    l00 = chol[..., 0:1]
    l10 = chol[..., 1:2]
    l11 = chol[..., 2:3]
    c00 = l00.square()
    c01 = l00 * l10
    c11 = l10.square() + l11.square()
    return torch.cat([c00, c01, c11], dim=-1)


def _zero_at_init_softplus(raw: torch.Tensor, init_bias: float) -> torch.Tensor:
    return F.softplus(raw) - F.softplus(raw.new_tensor(init_bias))
