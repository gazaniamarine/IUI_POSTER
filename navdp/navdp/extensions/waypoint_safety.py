from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class WaypointSafetySelector:
    """Select the safest, most goal-progressing waypoint chunk."""

    def __init__(
        self,
        d_safe: float = 0.25,
        goal_weight: float = 1.0,
        clearance_weight: float = 1.0,
        collision_weight: float = 5.0,
        resolution: float = 0.05,
    ):
        self.d_safe = float(d_safe)
        self.goal_weight = float(goal_weight)
        self.clearance_weight = float(clearance_weight)
        self.collision_weight = float(collision_weight)
        self.resolution = float(resolution)

    def __call__(
        self,
        candidate_waypoints,
        active_goal_mu: np.ndarray,
        clearance_map: np.ndarray,
    ):
        return self.select(candidate_waypoints, active_goal_mu, clearance_map)

    def select(
        self,
        candidate_waypoints,
        active_goal_mu: np.ndarray,
        clearance_map: np.ndarray,
        return_info: bool = False,
    ) -> Any:
        """Score candidates [N, K, 3] and return the best waypoint chunk."""
        if candidate_waypoints.ndim != 3 or candidate_waypoints.shape[-1] < 2:
            raise ValueError("candidate_waypoints must have shape [N,K,>=2]")

        if hasattr(candidate_waypoints, "detach"):
            way_np = candidate_waypoints.detach().cpu().float().numpy()
        else:
            way_np = np.asarray(candidate_waypoints, dtype=np.float32)
        goal = np.asarray(active_goal_mu, dtype=np.float32).reshape(-1)[:2]
        clr = np.asarray(clearance_map, dtype=np.float32)
        scores = []
        details = []
        for i, chunk in enumerate(way_np):
            final_xy = chunk[-1, :2]
            goal_dist = float(np.linalg.norm(final_xy - goal))
            clearances = np.asarray([self._lookup_clearance(xy, clr) for xy in chunk[:, :2]], dtype=np.float32)
            unsafe = int((clearances < self.d_safe).sum())
            mean_clearance = float(clearances.mean()) if clearances.size else 0.0
            score = (
                -self.goal_weight * goal_dist
                + self.clearance_weight * mean_clearance
                - self.collision_weight * unsafe
            )
            scores.append(score)
            details.append(
                {
                    "index": i,
                    "score": score,
                    "goal_distance": goal_dist,
                    "mean_clearance": mean_clearance,
                    "num_unsafe_waypoints": unsafe,
                }
            )

        best_idx = int(np.argmax(np.asarray(scores)))
        best = candidate_waypoints[best_idx]
        if return_info:
            return best, {"best_index": best_idx, "candidates": details}
        return best

    def _lookup_clearance(self, xy: np.ndarray, clearance_map: np.ndarray) -> float:
        grid = clearance_map.shape[0]
        x_forward = float(xy[0])
        y_left = float(xy[1])
        row = grid - 1 - int(np.floor(x_forward / self.resolution))
        col = int(np.floor(y_left / self.resolution + grid * 0.5))
        if row < 0 or row >= clearance_map.shape[0] or col < 0 or col >= clearance_map.shape[1]:
            return 0.0
        return float(clearance_map[row, col])
