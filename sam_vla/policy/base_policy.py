"""
Swap point between different driving strategies. Qwen-VLA implements this
today; NavDP or anything else can implement it later — each is an
independent implementation of act() and the rest of the system doesn't care
which one is plugged in.
"""

import typing

from sam_vla.core.types import Action, GoalSpec, Observation


class NavigationPolicy(typing.Protocol):
    def act(self, obs: "Observation", goal_spec: "GoalSpec") -> "Action": ...
