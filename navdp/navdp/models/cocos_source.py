from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefConditionedCocosSource(nn.Module):
    """Conditioned source distribution for action diffusion.

    Given condition tokens [B, Q, D], predicts a source mean [B, H, A].
    Sampling uses:
        x_source = alpha * mean + beta * epsilon
    """

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        horizon: int,
        beta: float = 0.2,
        alpha: float = 1.0,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.beta = float(beta)
        self.alpha = float(alpha)
        hidden = int(hidden_dim or condition_dim)
        self.mean_head = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, horizon * action_dim),
        )

    def forward(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        pooled = condition_tokens.mean(dim=1)
        mean = self.mean_head(pooled)
        return mean.view(condition_tokens.shape[0], self.horizon, self.action_dim)

    def sample_source(
        self,
        condition_tokens: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward(condition_tokens)
        if noise is None:
            noise = torch.randn_like(mean)
        return self.alpha * mean + self.beta * noise, mean

    def mean_loss(self, condition_tokens: torch.Tensor, expert_waypoints: torch.Tensor) -> torch.Tensor:
        mean = self.forward(condition_tokens)
        return F.mse_loss(mean, expert_waypoints)

