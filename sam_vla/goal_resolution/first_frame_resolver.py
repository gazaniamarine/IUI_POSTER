# This module's entire job is chaining sam_segmenter -> sam_output_adapter ->
# qwen_client into a single first-frame goal resolution call, so importing all
# three (plus core.types) is intentional, not a layering violation.

import sys
from typing import Optional

import numpy as np
from PIL import Image

from sam_vla.core.types import Detection, GoalSpec
from sam_vla.perception import sam_output_adapter, sam_segmenter
from sam_vla.vlm import qwen_client


def _detect(
    rgb: np.ndarray,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> list[Detection]:
    """Segments `detect_rgb` if given (e.g. a mesh-overlay frame meant only
    for the segmentation model, see MarsHabitatEnv.get_mesh_overlay_rgb),
    else `rgb` itself. `backend`/`checkpoint_path` select which segmentation
    checkpoint runs (sam_segmenter.segment_frame's own defaults if omitted)."""
    frame = detect_rgb if detect_rgb is not None else rgb
    seg_kwargs = {}
    if backend is not None:
        seg_kwargs["backend"] = backend
    if checkpoint_path is not None:
        seg_kwargs["checkpoint_path"] = checkpoint_path
    raw_detections = sam_segmenter.segment_frame(frame, **seg_kwargs)
    # Logged pre-filter (includes "bedrock", which to_detections always drops)
    # so a class-confusion bug -- e.g. ground pixels classified "bigrock" --
    # is visible here rather than only as an unexplained obstacle box later.
    print(
        f"[sam detect] {len(raw_detections)} raw detection(s) on "
        f"{'mesh-overlay' if detect_rgb is not None else 'plain'} frame "
        f"({frame.shape[1]}x{frame.shape[0]}): "
        + (
            ", ".join(
                f"{d['class_name']}@px({d['x']:.0f},{d['y']:.0f} "
                f"{d['width']:.0f}x{d['height']:.0f}) score={d['score']:.2f}"
                for d in raw_detections
            )
            or "none"
        )
    )
    detections: list[Detection] = sam_output_adapter.to_detections(
        raw_detections, frame.shape[1], frame.shape[0]
    )
    print(
        f"[sam detect] {len(detections)}/{len(raw_detections)} survive class "
        "filter (bigrock only)"
    )

    if not detections:
        raise RuntimeError("no detections found on first frame — cannot resolve a goal")
    return detections


def resolve(
    rgb: np.ndarray,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> GoalSpec:
    detections = _detect(
        rgb, detect_rgb=detect_rgb, backend=backend, checkpoint_path=checkpoint_path
    )
    return qwen_client.select_goal(rgb, detections)


def resolve_verbose(
    rgb: np.ndarray,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> tuple[GoalSpec, dict, list[Detection]]:
    """Same as resolve, but also returns the raw VLM goal-selection result dict and
    the SAM detections used to produce it, for logging."""
    detections = _detect(
        rgb, detect_rgb=detect_rgb, backend=backend, checkpoint_path=checkpoint_path
    )
    goal_spec, vlm_result = qwen_client.select_goal_verbose(rgb, detections)
    return goal_spec, vlm_result, detections


def resolve_obstacles(
    rgb: np.ndarray,
    *,
    detect_rgb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> list[Detection]:
    """Just the SAM2+adapter detection half of resolve(), factored out for the
    multi-goal path: there, goals come from SAM3+CLIP instead, so every
    detection here is treated as an obstacle rather than one being picked out
    as the goal by Qwen."""
    return _detect(
        rgb, detect_rgb=detect_rgb, backend=backend, checkpoint_path=checkpoint_path
    )


def resolve_from_path(image_path: str) -> GoalSpec:
    # PIL, to match qwen_client's own __main__ image loading convention.
    rgb = np.array(Image.open(image_path).convert("RGB"))
    return resolve(rgb)


if __name__ == "__main__":
    goal_spec = resolve_from_path(sys.argv[1])
    print("goal_bbox_norm:", goal_spec.goal_bbox_norm)
    print("num_obstacles:", len(goal_spec.obstacle_bboxes_norm))
    print("instruction_text:", goal_spec.instruction_text)
