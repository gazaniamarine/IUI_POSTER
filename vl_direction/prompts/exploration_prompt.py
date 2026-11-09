"""
Prompt templates for EXPLORATION mode (next.md sec 1.2): four-way directional
prior given the high-level task and an optional vague human hint.
"""

from vl_direction.schemas import ExplorationContext

_SYSTEM_FRAMING = "You are a directional assistant for a Mars rover. You never explain, you only answer."

_FEW_SHOT = (
    'Example: task is "find the sample collection site", scene shows open '
    "terrain ahead and a ridge to the right -> FRONT\n"
    'Example: hint is "this area\'s explored", scene shows a previously '
    "visited crater dead ahead -> RIGHT"
)


def build_exploration_prompt(context: ExplorationContext, frame_count: int) -> str:
    hint_block = f'Human hint: "{context.vague_hint}"\n' if context.vague_hint else ""
    hint_clause = " and the hint" if context.vague_hint else ""
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f'Task: "{context.task_str}"\n'
        f"{hint_block}"
        f"You are shown {frame_count} frame(s) of the current scene.\n"
        f"Given the scene and the task{hint_clause}, which direction should "
        "exploration continue: left, right, front, or back?\n\n"
        f"{_FEW_SHOT}\n\n"
        "Respond with exactly one word: LEFT, RIGHT, FRONT, or BACK."
    )


def build_exploration_reprompt(context: ExplorationContext, frame_count: int) -> str:
    return (
        build_exploration_prompt(context, frame_count)
        + "\n\nYou must answer with exactly one of: LEFT, RIGHT, FRONT, BACK."
    )


if __name__ == "__main__":
    demo = ExplorationContext(
        task_str="find the sample collection site", vague_hint="this area's explored"
    )
    print(build_exploration_prompt(demo, frame_count=3))
