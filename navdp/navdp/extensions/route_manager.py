from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


class RouteManager:
    """Route pointer for ordered, repeatable subgoal navigation.

    The manager advances by index, not by a persistent "goal was reached" flag,
    so routes such as ["A", "B", "A"] remain well-defined.
    """

    def __init__(self, route: Sequence[str], success_radius: float = 0.5):
        if success_radius <= 0:
            raise ValueError("success_radius must be positive")
        self.route = list(route)
        self.success_radius = float(success_radius)
        self.k = 0

    def reset(self) -> None:
        self.k = 0

    def get_route_index(self) -> int:
        return self.k

    def is_finished(self) -> bool:
        return self.k >= len(self.route)

    def get_active_goal(self) -> str:
        if self.is_finished():
            raise RuntimeError("route is finished; there is no active goal")
        return self.route[self.k]

    def update(self, robot_position: Sequence[float], belief_bank: Any) -> Dict[str, Any]:
        """Advance the route pointer if the active belief is within radius.

        Args:
            robot_position: Position in the same frame as the belief means. For
                local-frame beliefs this is typically [0, 0].
            belief_bank: Either a SubgoalBeliefBank or a mapping from goal id to
                an object/dict with a ``mu`` field.

        Returns:
            A status dictionary containing the pre-update active goal, the new
            route index, and whether the pointer advanced.
        """
        if self.is_finished():
            return {
                "active_goal": None,
                "previous_active_goal": None,
                "next_active_goal": None,
                "route_index": self.k,
                "advanced": False,
                "finished": True,
                "distance": None,
            }

        active_goal = self.get_active_goal()
        slot = _get_slot(belief_bank, active_goal)
        mu = _get_mu(slot)
        initialized = bool(getattr(slot, "initialized", True))
        if isinstance(slot, dict):
            initialized = bool(slot.get("initialized", initialized))

        distance: Optional[float] = None
        advanced = False
        if initialized and mu is not None:
            robot = np.asarray(robot_position, dtype=np.float32).reshape(-1)
            goal = np.asarray(mu, dtype=np.float32).reshape(-1)
            d = min(robot.shape[0], goal.shape[0], 2)
            if d <= 0:
                raise ValueError("robot_position and belief mu must have at least one dimension")
            distance = float(np.linalg.norm(robot[:d] - goal[:d]))
            if distance < self.success_radius:
                self.k += 1
                advanced = True

        next_goal = None if self.is_finished() else self.route[self.k]
        return {
            "active_goal": active_goal,
            "previous_active_goal": active_goal,
            "next_active_goal": next_goal,
            "route_index": self.k,
            "advanced": advanced,
            "finished": self.is_finished(),
            "distance": distance,
        }


def _get_slot(belief_bank: Any, goal_id: str) -> Any:
    if hasattr(belief_bank, "get"):
        return belief_bank.get(goal_id)
    return belief_bank[goal_id]


def _get_mu(slot: Any) -> Optional[np.ndarray]:
    if slot is None:
        return None
    if isinstance(slot, dict):
        return slot.get("mu")
    return getattr(slot, "mu", None)

