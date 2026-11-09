"""Open-vocabulary goal resolution via direct Qwen grounding, for object
classes SAM2/SAM2-LoRA wasn't trained on and that first_frame_resolver
deliberately never sees as goal/obstacle candidates (the placed flag
markers, the home-base ghost cuboid -- see FLAG_SEMANTIC_ID/
HOME_BASE_SEMANTIC_ID in sam_vla.core.goal_geometry). Mirrors
first_frame_resolver's resolve/resolve_verbose -> GoalSpec shape exactly so
nav/rover_controller.py's _do_resolve can swap between the two by whether a
target_text was given, but the goal position here comes straight from a
single point qwen_client.ground_object_verbose reports (see
qwen_prompts.build_ground_object_prompt for why a point, not a bbox),
instead of a SAM2 detection Qwen picks among. SAM2 still runs in this
module -- only to seed obstacle_bboxes_norm (rocks etc.) via
first_frame_resolver.resolve_obstacles -- per the reason this module exists
at all: keep SAM2 for rocks, use Qwen grounding only for open-vocabulary
targets.
"""

from typing import Optional

import numpy as np

from sam_vla.core.types import Detection, GoalSpec
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.vlm import qwen_client

# Half-width (px) of the small bbox synthesized around the grounded point --
# purely so the existing bbox-shaped GoalSpec/bbox_to_world depth-backprojection
# pipeline can still anchor a world position from a single point. Matches
# bbox_to_world's own pad_px default, so the two paddings compose into one
# reasonably sized depth-sample patch rather than either dominating.
_POINT_PATCH_MARGIN_PX = 6.0


def _point_to_small_bbox_norm(
    u: float, v: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Turns a single grounded (u, v) point into a small bbox_norm patch
    around it -- mirrors nav/rover_controller.py's _handle_pixel_click,
    which does the same small-patch trick for a manually clicked point, for
    the same single-pixel-depth-discontinuity robustness reason."""
    px, py = u * width, v * height
    return (
        max(0.0, (px - _POINT_PATCH_MARGIN_PX) / width),
        max(0.0, (py - _POINT_PATCH_MARGIN_PX) / height),
        min(1.0, (px + _POINT_PATCH_MARGIN_PX) / width),
        min(1.0, (py + _POINT_PATCH_MARGIN_PX) / height),
    )


def resolve_verbose(
    rgb: np.ndarray,
    target_text: str,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> tuple[GoalSpec, dict, list[Detection]]:
    """Grounds `target_text` directly via Qwen (no SAM2 candidate list
    involved in picking the goal). Raises RuntimeError if Qwen doesn't find
    the object in the current frame -- the same "no usable goal this
    resolve" contract first_frame_resolver.resolve_verbose has when SAM2
    finds zero detections."""
    point_norm, vlm_result = qwen_client.ground_object_verbose(rgb, target_text)
    if point_norm is None:
        raise RuntimeError(
            f'Qwen did not find "{target_text}" in the current frame '
            f'(reasoning: {vlm_result.get("reasoning", "")!r})'
        )
    height, width = rgb.shape[:2]
    goal_bbox_norm = _point_to_small_bbox_norm(point_norm[0], point_norm[1], width, height)

    try:
        obstacle_detections = first_frame_resolver.resolve_obstacles(
            rgb,
            detect_rgb=detect_rgb,
            backend=backend,
            checkpoint_path=checkpoint_path,
        )
    except RuntimeError:
        # Unlike first_frame_resolver's own path, the goal here doesn't
        # depend on SAM2 finding anything -- only obstacles do, so zero
        # detections is a legitimate (obstacle-free) outcome, not a failure.
        obstacle_detections = []

    goal_spec = GoalSpec(
        goal_bbox_norm=goal_bbox_norm,
        obstacle_bboxes_norm=[d.bbox_norm for d in obstacle_detections],
        instruction_text=f"Navigate to the {target_text}.",
    )
    return goal_spec, vlm_result, obstacle_detections


def resolve(
    rgb: np.ndarray,
    target_text: str,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> GoalSpec:
    goal_spec, _vlm_result, _dets = resolve_verbose(
        rgb,
        target_text,
        detect_rgb=detect_rgb,
        backend=backend,
        checkpoint_path=checkpoint_path,
    )
    return goal_spec
