"""Free-text nav-command decomposition into an ordered sequence of sub-goals,
consumed by RoverController's mission stepper (see rover_controller.py's
`_start_mission_subgoal`) to drive nav/gui.py's Command panel end to end.

Splitting the raw instruction is delegated to the Qwen VLM
(`sam_vla.vlm.qwen_client.parse_nav_command`, see
`qwen_prompts.build_parse_nav_command_prompt`) rather than done here --
that call needs a camera frame and is slow, so RoverController runs it on a
background thread (`_dispatch_nav_command`/`_nav_command_worker`) and hands
this module only the resulting `(directions, goals)` two-list split. This
module's job is turning that split into an ordered `SubGoal` sequence:
`parse_parts` below, wrapped by `Mission`.

`reached_current_goal`-style green-pixel/VLM heuristics from the original
standalone prototype (`check.py`, kept for reference at the repo root) were
dropped rather than ported -- RoverController already has a strictly better
reached-goal signal (ground-truth distance for MODE_POINT,
BeliefGoalTracker.distance() for MODE_RESOLVE, see its `display.goal_reached`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class GoalKind(Enum):
    GO_TO = auto()  # navigate to a landmark (open-vocabulary Qwen grounding)
    RETURN = auto()  # return to the spawn/home point
    TURN = auto()  # in-place rotation, no forward movement
    ADVANCE = auto()  # drive forward in the current facing direction, no turn
    FIND = auto()  # same handling as GO_TO -- search/scan until a target appears
    DONE = auto()


@dataclass
class SubGoal:
    kind: GoalKind
    target: str  # e.g. "flag", "home station", "right"
    raw: str  # the original phrase, for status display


# Direction tokens qwen_prompts.build_parse_nav_command_prompt's closed
# vocabulary emits, split by verb: "turn_*" is an explicit in-place
# turn/face command with no forward movement ("turn left", "turn around");
# "go_*" implies actually moving that way ("go left", "head right", "keep
# going straight") -- turn to face the heading first (skipped for go_front,
# already facing it), THEN drive forward, see _direction_subgoals. Bare
# legacy tokens ("left"/"right"/"back", no verb prefix) are accepted too, as
# a fallback if the VLM ever emits one anyway -- treated as the safer
# turn-only reading rather than guessing it meant to drive. The actual
# turn-heading-degrees mapping lives once, in rover_controller.py's
# MODE_TURN handling.
_TURN_HEADINGS = {
    "turn_left": "left",
    "turn_right": "right",
    "turn_back": "back",
    "left": "left",
    "right": "right",
    "back": "back",
}
_GO_HEADINGS = {
    "go_left": "left",
    "go_right": "right",
    "go_front": None,  # already facing front -- no turn needed, just drive
}

# Goal phrases naming the spawn point rather than a new landmark -- e.g.
# "home base", "the spawn point", "go back", "return" -- become RETURN, not
# GO_TO. A safety net independent of qwen_prompts.build_parse_nav_command_prompt's
# own "come back"/"go back" -> "home base" instruction, in case the VLM ever
# emits one of those phrases verbatim as a goal instead of normalizing it.
_RETURN_WORDS = re.compile(r"\b(home|base|spawn|return|come back|go back)\b", re.I)

# A second, outer safety net: unlike _RETURN_WORDS above (checked against
# each already-emitted goal phrase), this is checked against the raw
# instruction text itself, to catch the VLM dropping the return leg from
# "goals" entirely (observed in practice -- "turn right go to the flag and
# come back to base" parsed to goals=["flag"], silently losing "come back to
# base"). Deliberately narrower than _RETURN_WORDS -- no bare "back"/"home"/
# "base"/"return" alone, since e.g. "turn back" (in-place turn, handled by
# "directions") legitimately contains "back" with no return-to-spawn sense
# at all. Only multi-word phrases that are unambiguous in this domain.
_RETURN_PHRASE = re.compile(
    r"come back|go back|head back|back to (?:the )?(?:home )?base|"
    r"back to (?:the )?spawn|home base|return (?:to|home)",
    re.I,
)


def _direction_subgoals(direction: str) -> List[SubGoal]:
    """One VLM-emitted direction token -> zero, one, or two sub-goals.

    "go_*" is "turn left/right means turn in place, go left/right means
    turn AND move forward": emits a TURN (skipped for go_front, which needs
    none) followed by an ADVANCE that actually drives the rover forward
    once it's facing the right way. "turn_*"/bare legacy tokens are a
    single in-place TURN with no following drive. Anything unrecognized
    (e.g. bare "front") -> no sub-goal, same as before this distinction
    existed."""
    if direction in _GO_HEADINGS:
        heading = _GO_HEADINGS[direction]
        steps: List[SubGoal] = []
        if heading is not None:
            steps.append(SubGoal(GoalKind.TURN, heading, direction))
        steps.append(SubGoal(GoalKind.ADVANCE, heading or "front", direction))
        return steps
    if direction in _TURN_HEADINGS:
        return [SubGoal(GoalKind.TURN, _TURN_HEADINGS[direction], direction)]
    return []


def _goal_subgoal(goal: str) -> SubGoal:
    kind = GoalKind.RETURN if _RETURN_WORDS.search(goal) else GoalKind.GO_TO
    return SubGoal(kind, goal, goal)


def parse_parts(
    directions: List[str], goals: List[str], instruction: str = ""
) -> List[SubGoal]:
    """Turn the VLM's (directions, goals) split into an ordered sub-goal
    sequence, in two parts: every direction first (each a TURN and/or
    ADVANCE, see _direction_subgoals), then every goal (GO_TO, or RETURN for
    phrases naming home/base/spawn). Each part preserves the VLM's own
    within-list ordering (see qwen_prompts.build_parse_nav_command_prompt),
    but the VLM never orders directions relative to goals across the two
    lists -- "part 1 then part 2" is the closest ordering available, not a
    reconstruction of the original interleaving.

    `instruction` (the original raw command text, if given) backstops a
    return leg the VLM dropped from `goals` altogether -- see
    _RETURN_PHRASE's docstring. Only fires when nothing already in `goals`
    reads as a return (checked via _RETURN_WORDS, not _RETURN_PHRASE, so an
    already-present "home base"/"spawn point" goal from the VLM counts)."""
    if (
        instruction
        and _RETURN_PHRASE.search(instruction)
        and not any(_RETURN_WORDS.search(g) for g in goals)
    ):
        goals = [*goals, "home base"]
    steps: List[SubGoal] = []
    for direction in directions:
        steps.extend(_direction_subgoals(direction))
    for goal in goals:
        steps.append(_goal_subgoal(goal))
    steps.append(SubGoal(GoalKind.DONE, "", "mission complete"))
    return steps


@dataclass
class Mission:
    instruction: str
    directions: List[str] = field(default_factory=list)
    goal_texts: List[str] = field(default_factory=list)
    goals: List[SubGoal] = field(default_factory=list)
    idx: int = 0

    def __post_init__(self):
        if not self.goals:
            self.goals = parse_parts(self.directions, self.goal_texts, self.instruction)

    @property
    def current(self) -> SubGoal:
        return self.goals[self.idx]

    @property
    def finished(self) -> bool:
        return self.current.kind == GoalKind.DONE

    def status(self) -> str:
        cur = self.current
        return (
            f"[step {self.idx + 1}/{len(self.goals)}] "
            f"{cur.kind.name} -> '{cur.target}'  ({cur.raw!r})"
        )

    def advance(self):
        if not self.finished:
            self.idx += 1
