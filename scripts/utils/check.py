from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 0. Plug your Qwen call in here
# ---------------------------------------------------------------------------
def query_qwen(image: np.ndarray, prompt: str) -> str:
    """Return Qwen's raw text answer for (image, prompt).

    Replace the body with your actual Qwen2.5-VL call (HF transformers, an
    HTTP endpoint, vLLM, etc.). `image` is an RGB uint8 HxWx3 array.
    """
    raise NotImplementedError("Wire up your Qwen2.5-VL inference here.")


def _qwen_yes(image: np.ndarray, prompt: str) -> bool:
    """Ask Qwen a yes/no question and parse the answer robustly."""
    ans = query_qwen(image, prompt).strip().lower()
    # take the first word-ish token so 'yes, because...' still counts
    return bool(re.match(r"^\s*(yes|true|reached|arrived|1)\b", ans))


# ---------------------------------------------------------------------------
# 1. Instruction -> ordered sub-goals (states)
# ---------------------------------------------------------------------------
class GoalKind(Enum):
    GO_TO = auto()  # navigate to a landmark (has a green goal mask)
    RETURN = auto()  # return to a previously visited place
    TURN = auto()  # in-place rotation
    FIND = auto()  # search/scan until a target appears
    DONE = auto()


@dataclass
class SubGoal:
    kind: GoalKind
    target: str  # e.g. "flag", "home station", "right"
    raw: str  # the original phrase, for prompting Qwen


# connective words that separate steps in the instruction
_SPLIT = re.compile(r"\bthen\b|\band then\b|,\s*then\b|;", flags=re.I)


def parse_instruction(instruction: str) -> List[SubGoal]:
    """Rule-based decomposition. Good enough for 'go ... then return ... then turn ...'.

    For truly free-form language, swap this for a single Qwen call that returns
    a JSON list of {kind, target}. The rest of the file doesn't care how the
    list was produced.
    """
    steps: List[SubGoal] = []
    for chunk in _SPLIT.split(instruction):
        phrase = chunk.strip().rstrip(".")
        if not phrase:
            continue
        low = phrase.lower()

        if "return" in low or "back to" in low or "go back" in low:
            target = (
                re.sub(r".*(return|back)\s*(to)?\s*", "", low).strip() or "home station"
            )
            steps.append(SubGoal(GoalKind.RETURN, target, phrase))
        elif low.startswith("turn") or " turn " in low:
            direction = (
                "right" if "right" in low else "left" if "left" in low else "around"
            )
            steps.append(SubGoal(GoalKind.TURN, direction, phrase))
        elif low.startswith("find") or "look for" in low or "search" in low:
            target = re.sub(r"^(find|look for|search for|search)\s+", "", low).strip()
            steps.append(SubGoal(GoalKind.FIND, target, phrase))
        else:  # default: navigate somewhere ("go to a flag stop")
            target = re.sub(
                r"^(go to|navigate to|move to|reach|go)\s+(a|an|the)?\s*", "", low
            ).strip()
            steps.append(SubGoal(GoalKind.GO_TO, target or low, phrase))

    steps.append(SubGoal(GoalKind.DONE, "", "mission complete"))
    return steps


# ---------------------------------------------------------------------------
# 2. Green goal-mask analysis (cheap geometric closeness pre-check)
# ---------------------------------------------------------------------------
@dataclass
class MaskInfo:
    present: bool
    coverage: float  # fraction of frame occupied by green (0..1)
    cx: float  # centroid x in [0,1] (0=left, 1=right)
    cy: float  # centroid y in [0,1] (0=top,  1=bottom)
    bottom: float  # lowest green pixel, normalized (1=frame bottom)


def analyze_green_mask(image_rgb: np.ndarray) -> MaskInfo:
    """Find the green goal overlay and summarize its geometry.

    Intuition for a ground rover: as you approach the goal the green blob grows
    (coverage up) and slides toward the BOTTOM-CENTER of the view (cy, bottom up).
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # green overlay band; widen S/V lower bounds if your overlay is translucent
    mask = cv2.inRange(hsv, (40, 60, 60), (85, 255, 255))

    area = int(mask.sum() // 255)
    if area == 0:
        return MaskInfo(False, 0.0, 0.5, 0.5, 0.0)

    ys, xs = np.where(mask > 0)
    return MaskInfo(
        present=True,
        coverage=area / float(h * w),
        cx=float(xs.mean()) / w,
        cy=float(ys.mean()) / h,
        bottom=float(ys.max()) / h,
    )


def geometric_close_enough(
    m: MaskInfo,
    min_coverage: float = 0.06,
    min_bottom: float = 0.75,
    center_tol: float = 0.30,
) -> bool:
    """Fast gate BEFORE bothering the VLM. Tune the three thresholds to Mars.

    True when the goal blob is big enough, reaches low in the frame (near the
    rover), and is roughly centered ahead.
    """
    if not m.present:
        return False
    return (
        m.coverage >= min_coverage
        and m.bottom >= min_bottom
        and abs(m.cx - 0.5) <= center_tol
    )


# ---------------------------------------------------------------------------
# 3. The state machine
# ---------------------------------------------------------------------------
@dataclass
class Mission:
    instruction: str
    goals: List[SubGoal] = field(default_factory=list)
    idx: int = 0

    def __post_init__(self):
        if not self.goals:
            self.goals = parse_instruction(self.instruction)

    @property
    def current(self) -> SubGoal:
        return self.goals[self.idx]

    @property
    def finished(self) -> bool:
        return self.current.kind == GoalKind.DONE

    def status(self) -> str:
        cur = self.current
        return (
            f"[state {self.idx + 1}/{len(self.goals)}] "
            f"{cur.kind.name} -> '{cur.target}'  ({cur.raw!r})"
        )

    def advance(self):
        if not self.finished:
            self.idx += 1


def reached_current_goal(
    mission: Mission, image_rgb: np.ndarray, use_vlm: bool = True
) -> bool:
    """Has the rover completed the CURRENT sub-goal in this frame?

    Strategy: cheap geometry gate first; if it passes, let Qwen make the call.
    This keeps VLM invocations rare (only when you're plausibly there).
    """
    goal = mission.current
    m = analyze_green_mask(image_rgb)

    # TURN goals have no green mask; confirm heading with the VLM directly.
    if goal.kind == GoalKind.TURN:
        if not use_vlm:
            return True
        return _qwen_yes(
            image_rgb,
            f"The rover was asked to turn {goal.target}. "
            f"Has it finished turning {goal.target} and is now facing a new direction? "
            f"Answer yes or no.",
        )

    # Navigation-type goals ride on the green goal mask.
    if not geometric_close_enough(m):
        return False
    if not use_vlm:
        return True

    prompt = (
        f"The current goal is: '{goal.raw}'. "
        f"The goal location is highlighted with a GREEN overlay in the image. "
        f"Is the rover now RIGHT NEXT TO / directly at the green-highlighted goal "
        f"(close enough to consider it reached)? Answer strictly yes or no."
    )
    return _qwen_yes(image_rgb, prompt)


# ---------------------------------------------------------------------------
# 4. Example driver loop
# ---------------------------------------------------------------------------
def run(instruction: str, get_frame: Callable[[], Optional[np.ndarray]]):
    """`get_frame()` returns the latest RGB observation, or None to stop."""
    mission = Mission(instruction)
    print("Decomposed mission:")
    for i, g in enumerate(mission.goals):
        print(f"  {i + 1}. {g.kind.name:<7} {g.target}")

    while not mission.finished:
        frame = get_frame()
        if frame is None:
            break

        print(mission.status())
        # ... your policy picks/executes an action toward mission.current here ...

        if reached_current_goal(mission, frame):
            print(f"  ✔ reached: {mission.current.raw!r}")
            mission.advance()

    print("Mission complete." if mission.finished else "Stopped early.")


if __name__ == "__main__":
    demo = "go to a flag stop then return back to the homestation then turn right then find flag and then return to homestation"
    for i, g in enumerate(parse_instruction(demo)):
        print(f"{i + 1}. {g.kind.name:<7} target={g.target!r}")
