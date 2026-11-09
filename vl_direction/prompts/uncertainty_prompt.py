"""
Prompt template for UNCERTAINTY mode's sweep-description ask (next.md sec
1.3/5.3). This mode never emits LEFT/RIGHT/FRONT/BACK -- it asks the model to
describe the sweep so a human operator has context when picking a heading,
so this prompt has no closed-vocabulary output-format constraint or
few-shot exemplar.
"""

_SYSTEM_FRAMING = "You are a directional assistant for a Mars rover. You never explain beyond what is asked."


def build_sweep_description_prompt(rover_front_reference_deg: float = 0.0) -> str:
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f"The rover's front is {rover_front_reference_deg:.0f} degrees. "
        "Describe what you see around the rover during this look-around sweep "
        "in one or two sentences, noting any landmarks and their approximate "
        "heading relative to the rover's front, to help a human operator pick "
        "a direction to search for the goal."
    )


if __name__ == "__main__":
    print(build_sweep_description_prompt(0.0))
