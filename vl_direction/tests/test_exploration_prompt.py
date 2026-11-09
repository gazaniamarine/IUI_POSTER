from vl_direction.prompts.exploration_prompt import build_exploration_prompt, build_exploration_reprompt
from vl_direction.schemas import ExplorationContext


def test_prompt_contains_task_string():
    context = ExplorationContext(task_str="find the sample collection site")
    prompt = build_exploration_prompt(context, frame_count=1)
    assert "find the sample collection site" in prompt


def test_prompt_omits_hint_block_when_hint_is_none():
    context = ExplorationContext(task_str="explore the crater rim")
    prompt = build_exploration_prompt(context, frame_count=1)
    assert "Human hint" not in prompt


def test_prompt_includes_hint_when_present():
    context = ExplorationContext(task_str="explore the crater rim", vague_hint="this area's explored")
    prompt = build_exploration_prompt(context, frame_count=1)
    assert "this area's explored" in prompt


def test_prompt_contains_all_four_directions():
    context = ExplorationContext(task_str="explore")
    prompt = build_exploration_prompt(context, frame_count=1)
    for word in ("LEFT", "RIGHT", "FRONT", "BACK"):
        assert word in prompt


def test_reprompt_adds_corrective_instruction():
    context = ExplorationContext(task_str="explore")
    prompt = build_exploration_prompt(context, frame_count=1)
    reprompt = build_exploration_reprompt(context, frame_count=1)
    assert reprompt.startswith(prompt)
    assert "must answer" in reprompt.lower()
