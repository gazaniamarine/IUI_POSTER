"""
Converts sam_segmenter.segment_frame's raw pixel-space dicts into Detections.

"""

from sam_vla.core.types import Detection

# bigrock is the only discrete-object class the goal/obstacle VLM selection
# should ever see. bedrock is terrain/background segmentation (its contours
# can span most of the frame), not a candidate rock - surfacing it here lets
# the VLM pick a bedrock region as the "goal" or "obstacle", which is wrong.
# Any class not in this map (bedrock included) is dropped in to_detections.
#
# The mapped value is a neutral role-free label, not "obstacle" -- it flows
# straight into qwen_prompts.build_select_goal_prompt's per-detection
# `class="..."` line and qwen_client's instruction_text ("Navigate to the
# {class_name} target..."), both *before* goal_index picks which detection
# is actually the goal. Mapping to "obstacle" here pre-labels every
# candidate (including whichever one ends up chosen as the goal) as an
# obstacle in the VLM's own prompt/output, which both biases the selection
# and makes the resolved goal's status text misleadingly read "Navigate to
# the obstacle target".
_CLASS_MAP = {
    "bigrock": "rock",
}

# NOTE: the LoRA checkpoints trained on mesh_annotation_tool's hull masks
# predict thin silhouette-line masks rather than filled rock regions (see
# CLAUDE.md's "Annotation masks are thin silhouette slivers" known issue),
# so bounding rectangles here are routinely just 1-3px tall regardless of
# whether depth backprojection later succeeds or fails for that particular
# box -- a min-size filter was tried here and reverted, since it rejected
# the *same shape* of box that succeeds as often as it rejects ones that
# fail (bbox size isn't what predicts a bad depth sample; see
# goal_geometry.bbox_to_world's padding instead, which is where this is
# actually handled). Don't reintroduce a size floor here without evidence
# it's not just discarding real detections wholesale.


def to_detections(
    raw_detections: list[dict], image_width: int, image_height: int
) -> list[Detection]:
    detections = []
    for raw in raw_detections:
        if raw["class_name"] not in _CLASS_MAP:
            continue
        class_name = _CLASS_MAP[raw["class_name"]]
        x0 = raw["x"] / image_width
        y0 = raw["y"] / image_height
        x1 = (raw["x"] + raw["width"]) / image_width
        y1 = (raw["y"] + raw["height"]) / image_height

        det = Detection(
            class_name=class_name,
            bbox_norm=(x0, y0, x1, y1),
            confidence=raw["score"],
        )
        try:
            det.validate()
        except ValueError as e:
            print(f"Warning: skipping invalid detection {raw}: {e}")
            continue
        detections.append(det)

    return detections


if __name__ == "__main__":
    raw_examples = [
        {
            "class_name": "bedrock",
            "x": 100.0,
            "y": 50.0,
            "width": 200.0,
            "height": 150.0,
            "score": 0.92,
        },
        {
            "class_name": "bigrock",
            "x": 400.0,
            "y": 300.0,
            "width": 80.0,
            "height": 80.0,
            "score": 0.77,
        },
        # Invalid: zero width -> x0 == x1, fails validate()
        {
            "class_name": "bedrock",
            "x": 600.0,
            "y": 200.0,
            "width": 0.0,
            "height": 40.0,
            "score": 0.5,
        },
    ]

    result = to_detections(raw_examples, image_width=1280, image_height=720)
    print(f"\n{len(result)} valid detections:")
    for d in result:
        print(d)
