"""Discrete-direction VLA query.

Instead of letting the VLA choose a free continuous action (v_fwd, v_lat,
yaw_rate), this constrains it to a single categorical choice -- forward,
turn_left, or turn_right -- over the current frame with the goal/obstacle
semantic masks overlaid, then maps that choice onto a fixed Action. This is
additive: QwenVlaPolicy and qwen_client.drive_action still exist unchanged
for continuous-action use.
"""

import numpy as np

from sam_vla.core.types import Action, GoalSpec
from sam_vla.perception.semantic_overlay import overlay_semantic_masks
from sam_vla.vlm import qwen_client

# Fixed actions a discrete direction choice maps onto. Turns keep some forward
# speed so the rover arcs rather than spinning in place.
FORWARD_ACTION = Action(v_fwd=1.0, v_lat=0.0, yaw_rate=0.0)
TURN_LEFT_ACTION = Action(v_fwd=0.5, v_lat=0.0, yaw_rate=-1.0)
TURN_RIGHT_ACTION = Action(v_fwd=0.5, v_lat=0.0, yaw_rate=1.0)

_DIRECTION_TO_ACTION = {
    "forward": FORWARD_ACTION,
    "turn_left": TURN_LEFT_ACTION,
    "turn_right": TURN_RIGHT_ACTION,
}


def direction_to_action(direction: str) -> Action:
    """Map a discrete VLA direction choice (qwen_response_parser.DIRECTIONS)
    onto its fixed Action."""
    try:
        return _DIRECTION_TO_ACTION[direction]
    except KeyError:
        raise ValueError(
            f"unknown direction {direction!r}, expected one of {sorted(_DIRECTION_TO_ACTION)}"
        ) from None


class QwenDiscreteDirectionPolicy:
    """Queries the VLA for a single discrete direction over a semantic-mask
    overlay of the current frame, then maps it onto a fixed Action."""

    def act_verbose(
        self, rgb: np.ndarray, semantic: np.ndarray, goal_spec: GoalSpec, frame_idx: int
    ) -> tuple[Action, dict]:
        """rgb/semantic are the raw camera frame and its per-pixel semantic-id
        frame (MarsHabitatEnv.get_semantic_frame()) for the same step. Returns
        the mapped Action and the raw VLA result dict for logging."""
        overlaid_rgb = overlay_semantic_masks(rgb, semantic)
        direction, vla_result = qwen_client.drive_direction_verbose(
            overlaid_rgb, goal_spec, frame_idx
        )
        action = direction_to_action(direction)
        return action, vla_result


if __name__ == "__main__":
    for direction in ("forward", "turn_left", "turn_right"):
        print(direction, "->", direction_to_action(direction))

    try:
        direction_to_action("reverse")
    except ValueError as e:
        print(f"caught ValueError: {e}")
