"""
Resolves the goal vocabulary (text terms + instruction) that seeds SAM3's
per-term prompts and CLIP's text-embedding bank in the multi-goal path.
Mirrors first_frame_resolver's role in the single-goal path, but produces a
vocabulary instead of picking one detection.
"""

import numpy as np

from sam_vla.vlm import qwen_client


def resolve_goal_vocabulary(
    rgb0: np.ndarray, cli_vocab: list[str] | None, use_qwen: bool = True
) -> tuple[list[str], str]:
    """cli_vocab (--goal-vocab, comma-separated terms) short-circuits Qwen
    when given — both as a fallback and for belief_exp-style/no-network
    testing. Otherwise calls qwen_client.describe_goal_vocabulary."""
    if cli_vocab:
        terms = [t.strip() for t in cli_vocab if t.strip()]
        if not terms:
            raise ValueError("cli_vocab was given but contained no non-empty terms")
        instruction_text = "Navigate to each of: " + ", ".join(terms) + "."
        return terms, instruction_text

    if not use_qwen:
        raise ValueError("cli_vocab is required when use_qwen is False")

    return qwen_client.describe_goal_vocabulary(rgb0)
