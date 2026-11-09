"""Adapter: runs SAM (sam/sam/inference.py) on a single already-captured frame
and converts its per-class boxes into the labelme-style annotation JSON that
vlm_nav_interactive.py's resolve_vlm_selection() expects. Maps SAM's `bigrock`
class to the pipeline's "rock" label (the only obstacle/goal candidate the VLM
prompt understands) and drops `bedrock` entirely (background surface, not a
discrete object). vlm_nav_interactive imports are done lazily so this stays
importable without habitat_sim/quaternion/a real scene on disk.

Usage:
    python scripts/utils/sam_annotation_adapter.py --full <source_frame_idx> <target_frame_idx>
"""

import sys
import json
import shutil
from pathlib import Path

import cv2

HERE = Path(__file__).parent
SAM_DIR = HERE / "sam" / "sam"
if str(SAM_DIR) not in sys.path:
    sys.path.insert(0, str(SAM_DIR))

from inference import run_inference_on_frame  # noqa: E402

SAM_LABEL_MAP = {
    "bigrock": "rock",
}

LABELME_VERSION = "5.4.1"


def sam_boxes_to_shapes(sam_boxes):
    """
    Convert run_inference_on_frame()'s {'bedrock': [...], 'bigrock': [...]}
    output into a list of labelme-style shape dicts (no "id" yet - that's
    assigned by assign_object_ids() once the shapes are on disk). bedrock
    is ignored - it's terrain/background, not a selectable rock.
    """
    shapes = []
    for class_name in ("bigrock",):
        label = SAM_LABEL_MAP[class_name]
        for box in sam_boxes.get(class_name, []):
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            shapes.append(
                {
                    "label": label,
                    "points": [[x, y], [x + w, y + h]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )
    return shapes


def build_annotation_json(image_path, sam_boxes, image_height, image_width):
    """Assemble a full labelme-shaped annotation dict (sans per-shape "id")."""
    image_path = Path(image_path)
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": sam_boxes_to_shapes(sam_boxes),
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def sam_frame_to_annotation(image_path, annotation_out_path, model=None):
    """
    Run SAM on a single frame and write a resolve_vlm_selection-compatible
    annotation JSON to annotation_out_path.

    Args:
        image_path: path to the source RGB frame.
        annotation_out_path: where to write the annotation JSON.
        model: preloaded SAM model (load_best_model()); loaded fresh if None.

    Returns:
        (annotation_out_path, is_valid, status_message).
    """
    from vlm_nav_interactive import assign_object_ids, validate_annotation_json

    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = frame.shape[:2]

    sam_boxes = run_inference_on_frame(frame, model=model)
    annotation = build_annotation_json(image_path, sam_boxes, height, width)

    annotation_out_path = Path(annotation_out_path)
    annotation_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotation_out_path, "w") as f:
        json.dump(annotation, f, indent=2)

    assign_object_ids(str(annotation_out_path))

    is_valid, status = validate_annotation_json(str(annotation_out_path))
    if is_valid and len(sam_boxes.get("bigrock", [])) < 2:
        # The mission needs two distinct rocks (one goal, one obstacle).
        # Fewer than that and the VLM has no real rock to pick from -- it
        # will either pick nothing or hallucinate a "rock" label to satisfy
        # VLM_PROMPT's schema, so fail loudly here instead.
        is_valid = False
        status = (
            f"only {len(sam_boxes.get('bigrock', []))} bigrock detection(s) "
            "in frame, need >=2 for goal+obstacle"
        )
    return str(annotation_out_path), is_valid, status


def duplicate_frame_files(source_frame_idx, target_frame_idx, out_dir):
    """
    Copy an already-captured frame's rgb/depth/pose files (as written by
    vlm_nav_interactive.save_frame/save_pose) to a new frame index.

    resolve_vlm_selection() keys everything - the annotation JSON path,
    the raw RGB it sends to Qwen, and the depth/pose it back-projects
    through - off a single frame_idx. Giving the SAM run its own
    target_frame_idx (rather than reusing source_frame_idx) means the
    SAM-sourced annotation lands at its own
    ANNOTATIONS_DIR/rgb_{target_frame_idx:04d}.json instead of
    overwriting an existing labelme annotation for the same image, so
    both can be diffed side by side.

    Returns target_frame_idx.
    """
    out_dir = Path(out_dir)
    for name, required in (
        (f"rgb_{source_frame_idx:04d}.png", True),
        (f"depth_{source_frame_idx:04d}.npy", False),
        (f"depth_{source_frame_idx:04d}.png", False),
        (f"pose_{source_frame_idx:04d}.json", True),
    ):
        src = out_dir / name
        if not src.exists():
            if required:
                raise FileNotFoundError(f"missing required source frame file: {src}")
            continue
        dst_name = name.replace(f"{source_frame_idx:04d}", f"{target_frame_idx:04d}")
        shutil.copy2(src, out_dir / dst_name)
    return target_frame_idx


def run_sam_vlm_on_frame(source_frame_idx, target_frame_idx, model=None):
    """
    Run SAM on an already-captured frame and feed the result through the
    exact same resolve_vlm_selection() path the manual labelme flow uses.

    Duplicates source_frame_idx's rgb/depth/pose to target_frame_idx (see
    duplicate_frame_files), runs SAM on that copy, writes its annotation
    to ANNOTATIONS_DIR/rgb_{target_frame_idx:04d}.json - the same on-disk
    location resolve_vlm_selection() already reads from - then hands off
    to vlm_nav_interactive.run_vlm_on_frame(), which draws the overlay,
    queries the VLM, resolves goal/obstacle meshes, and saves mission
    metadata exactly as the manual --vlm CLI path does. No logic in
    resolve_vlm_selection or run_vlm_on_frame is touched; this only
    arranges for SAM's output to be sitting where they already look.

    target_frame_idx must differ from source_frame_idx (enforced) so a
    labelme annotation already captured for source_frame_idx is never
    overwritten.
    """
    if target_frame_idx == source_frame_idx:
        raise ValueError(
            "target_frame_idx must differ from source_frame_idx, to avoid "
            "overwriting an existing labelme annotation for that frame"
        )

    from vlm_nav_interactive import OUT_DIR, ANNOTATIONS_DIR, run_vlm_on_frame

    duplicate_frame_files(source_frame_idx, target_frame_idx, OUT_DIR)

    rgb_path = f"{OUT_DIR}/rgb_{target_frame_idx:04d}.png"
    annotation_path = f"{ANNOTATIONS_DIR}/rgb_{target_frame_idx:04d}.json"
    annotation_path, is_valid, status = sam_frame_to_annotation(
        rgb_path, annotation_path, model=model
    )
    print(f"[sam] wrote {annotation_path} valid={is_valid} status={status!r}")
    if not is_valid:
        raise RuntimeError(f"SAM annotation failed validate_annotation_json: {status}")

    run_vlm_on_frame(target_frame_idx)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full routing: SAM -> adapter -> resolve_vlm_selection, targeting
        # a fresh frame slot so an existing labelme annotation is untouched.
        # usage: python sam_annotation_adapter.py --full <source_frame_idx> <target_frame_idx>
        if len(sys.argv) < 4:
            print(
                "usage: python sam_annotation_adapter.py --full <source_frame_idx> <target_frame_idx>"
            )
            sys.exit(1)
        run_sam_vlm_on_frame(int(sys.argv[2]), int(sys.argv[3]))
    else:
        image_path = sys.argv[1] if len(sys.argv) > 1 else "vlm_nav_out/rgb_0000.png"
        out_path = sys.argv[2] if len(sys.argv) > 2 else "annotations/rgb_0000_sam.json"

        annotation_path, is_valid, status = sam_frame_to_annotation(
            image_path, out_path
        )
        print(f"[adapter] wrote {annotation_path}")
        print(
            f"[adapter] validate_annotation_json -> valid={is_valid} status={status!r}"
        )

        with open(annotation_path) as f:
            data = json.load(f)
        for shape in data["shapes"]:
            print(
                f"  id={shape['id']} label={shape['label']!r} points={shape['points']}"
            )
