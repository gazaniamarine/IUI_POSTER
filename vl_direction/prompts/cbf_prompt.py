"""
Prompt templates for CBF mode (next.md sec 1.1): binary LEFT/RIGHT go-around
decision given an obstacle bbox. Free functions building plain strings, no
classes -- mirrors sam_vla/vlm/qwen_prompts.py's module-level template style.
"""

from vl_direction.schemas import CBFContext

_SYSTEM_FRAMING = "You are a directional assistant for a Mars rover. You never explain, you only answer."

_FEW_SHOT = (
    "Example: obstacle bbox (300, 200)-(420, 340) in a 640x480 frame, "
    "obstacle centered slightly right of frame center -> LEFT\n"
    "Example: obstacle bbox (40, 180)-(160, 300) in a 640x480 frame, "
    "obstacle centered on the left side of the frame -> RIGHT"
)


def build_cbf_prompt(context: CBFContext) -> str:
    x1, y1, x2, y2 = context.bbox_xyxy
    w, h = context.frame_wh
    center_x_norm = ((x1 + x2) / 2.0) / w
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f"An obstacle is detected at pixel bbox ({x1}, {y1})-({x2}, {y2}) "
        f"in a {w}x{h} frame (obstacle horizontal center at {center_x_norm:.2f} "
        f"of frame width, 0=left edge, 1=right edge).\n"
        "Should the rover pass on the left or the right of this obstacle?\n\n"
        f"{_FEW_SHOT}\n\n"
        "Respond with exactly one word: LEFT or RIGHT."
    )


def build_cbf_reprompt(context: CBFContext) -> str:
    return (
        build_cbf_prompt(context)
        + "\n\nYou must answer with exactly one of: LEFT, RIGHT."
    )


if __name__ == "__main__":
    demo = CBFContext(bbox_xyxy=(300, 200, 420, 340), frame_wh=(640, 480))
    print(build_cbf_prompt(demo))
