"""Anticipatory foresight ranking layered on the hard collision gate.

Ordering of authority (never violated):

1. The **hard collision gate** on the *real current depth* runs FIRST. Any
   candidate it rejects is dropped and can never come back.
2. The :class:`OccupancyForesightHead` forecast is a **secondary ranking signal**
   applied only to survivors. Predicted free space is never permission to drive;
   predicted occupancy is only a reason for extra caution.

The frozen DiT is not touched -- this operates on its sampled candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from navdp.models.occupancy_foresight import OccupancyForesightHead


def action_to_delta_pose(action_3d, dt: float) -> torch.Tensor:
    """First-step robot-frame motion (dx, dy, dtheta) from an action chunk's head.

    ``action_3d`` is [..., 3] = (v_fwd, v_lat, yaw_rate). In the robot's own frame
    the forward/left displacement is simply velocity*dt regardless of the world
    motion convention, which is exactly "how much the rover moved" -- the quantity
    the egocentric warp needs and the same body-frame convention the belief bank's
    odometry uses.
    """
    a = torch.as_tensor(action_3d, dtype=torch.float32)
    if a.dim() == 1:
        a = a[None]
    if a.shape[-1] < 3:
        raise ValueError("action_3d must have at least 3 components (v_fwd, v_lat, yaw_rate)")
    return torch.stack([a[:, 0] * dt, a[:, 1] * dt, a[:, 2] * dt], dim=-1)


@dataclass
class ForesightResult:
    best_index: int                    # index into the ORIGINAL candidate array
    best_action: torch.Tensor          # the winning candidate action
    u_occ_active: float                # uncertainty of the active candidate (-> epistemic gate)
    ranked_indices: list               # survivors, best-first (original indices)
    penalties: np.ndarray              # foresight penalty per survivor (aligned to ranked_indices)
    admissible: bool                   # False if the hard gate rejected everything


class ForesightGate:
    """Rank candidates with foresight, strictly subordinate to the hard gate."""

    def __init__(
        self,
        head: OccupancyForesightHead,
        dt: float,
        u_ref: float = 1.0,
        device: Optional[str] = None,
    ):
        self.head = head
        self.dt = float(dt)
        self.u_ref = float(u_ref)
        self.device = device or next(head.parameters()).device

    @torch.no_grad()
    def rank(
        self,
        occupancy_current,
        candidate_actions,
        real_depth_pass: Sequence[bool],
        known_current=None,
    ) -> ForesightResult:
        actions = torch.as_tensor(candidate_actions, dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions[None]
        n = actions.shape[0]
        passed = np.asarray(real_depth_pass, dtype=bool).reshape(-1)
        if passed.shape[0] != n:
            raise ValueError("real_depth_pass must have one flag per candidate")

        survivors = [i for i in range(n) if passed[i]]
        rejected = [i for i in range(n) if not passed[i]]
        if not survivors:
            return ForesightResult(
                best_index=-1, best_action=actions.new_zeros(actions.shape[-1]),
                u_occ_active=float("inf"), ranked_indices=[], penalties=np.empty(0),
                admissible=False,
            )

        m = len(survivors)
        occ = self.head._as_bchw(torch.as_tensor(occupancy_current, dtype=torch.float32)).to(self.device)
        if occ.shape[0] == 1 and m > 1:
            occ = occ.expand(m, -1, -1, -1)
        known = None
        if known_current is not None:
            known = self.head._as_bchw(torch.as_tensor(known_current, dtype=torch.float32)).to(self.device)
            if known.shape[0] == 1 and m > 1:
                known = known.expand(m, -1, -1, -1)

        deltas = action_to_delta_pose(actions[survivors], self.dt).to(self.device)
        out = self.head(occ, deltas, known)
        footprint_occ = self.head.footprint_occupancy(out.predicted_next_map)  # [M]
        # Uncertain forecasts apply a weaker penalty (don't over-trust a guess).
        u_norm = (out.u_occ / (out.u_occ + self.u_ref)).clamp(0.0, 1.0)
        penalty = (footprint_occ * (1.0 - u_norm)).detach().cpu().numpy()

        order = np.argsort(penalty, kind="stable")
        ranked_indices = [survivors[k] for k in order]
        best_index = ranked_indices[0]

        # Safety invariant: the hard gate is final. No rejected candidate may be
        # re-admitted by any foresight score.
        assert set(ranked_indices).isdisjoint(rejected), (
            "foresight re-admitted a candidate the hard depth gate rejected"
        )
        assert best_index in survivors

        return ForesightResult(
            best_index=int(best_index),
            best_action=actions[best_index].detach().cpu(),
            u_occ_active=float(out.u_occ[order[0]].item()),
            ranked_indices=[int(i) for i in ranked_indices],
            penalties=penalty[order],
            admissible=True,
        )
