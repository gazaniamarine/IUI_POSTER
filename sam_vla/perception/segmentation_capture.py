"""Turns a raw per-pixel semantic/mesh-id buffer (MarsHabitatEnv's semantic
sensor -- see sam_vla.env.habitat_env.get_full_observation) into a labeled
category mask + per-object metadata, using the mesh_id -> category registry
authored by mesh_annotation_tool.py (next.md Steps 3B/4/5-collapse).

capture_frame_record is deliberately pure and I/O-free: it doesn't know or
care whether it's called from an offline sweep script or inline in a live
rollout step -- that's what keeps this module agnostic to what consumes its
output (dataset export vs. runtime VLM/VLA feed). write_segmentation_assets
is the separate, thin persistence layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class ObjectRecord:
    mesh_id: int
    category: str
    pixel_count: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h), pixel space


def build_category_lut(
    mesh_id_map: Dict[str, Dict[str, str]],
    categories: Sequence[str],
    background_category: str = "background",
) -> Tuple[np.ndarray, List[str]]:
    """Build ONCE per run (not per frame): a LUT array where LUT[id] is the
    class index for semantic id `id`. id 0, any id absent from
    mesh_id_map (including the reserved goal/obstacle/rock ids 1/2/3), AND
    any id whose registry category is not in `categories` all fold into
    background_category -- the id-0/unlabeled mapping is a configurable
    parameter here rather than hardcoded, and `categories` doubles as an
    include-filter (pass the same list given to
    annotation_meshes.register_annotation_meshes so the mask and the
    registered scene agree). Returns (lut, class_names) where
    class_names = [background_category, *categories]."""
    class_names = [background_category, *categories]
    category_index = {name: i for i, name in enumerate(class_names)}
    included = set(categories)

    max_id = max((int(k) for k in mesh_id_map), default=0)
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for mesh_id_str, entry in mesh_id_map.items():
        category = entry["category"]
        if category in included:
            lut[int(mesh_id_str)] = category_index[category]
    return lut, class_names


def capture_frame_record(
    rgb: np.ndarray,
    semantic_id_buffer: np.ndarray,
    mesh_id_map: Dict[str, Dict[str, str]],
    lut: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, List[ObjectRecord]]:
    """Pure, I/O-free core of the capture pipeline. Given one frame's raw
    semantic id buffer plus the mesh_id->category registry, returns
    (category_mask, per-object records). `lut`/`class_names` must come from
    build_category_lut(...) called ONCE outside any per-frame loop -- it's
    identical for every frame of a run, so rebuilding it per frame would be
    pure waste.

    category_mask is a single vectorized gather over the whole (H, W)
    buffer -- O(H*W), no Python loop. Per-object pixel_count/bbox come from
    np.unique(..., return_counts=True) + scipy.ndimage.find_objects, each a
    single call over the ids actually present in THIS frame (typically a
    handful, never all registered hulls) -- not a per-object
    np.argwhere/masking pass."""
    if rgb.shape[:2] != semantic_id_buffer.shape:
        raise ValueError(
            f"rgb {rgb.shape[:2]} and semantic buffer {semantic_id_buffer.shape} size mismatch"
        )
    included_categories = set(class_names[1:])  # class_names[0] is background_category

    category_mask = np.take(lut, semantic_id_buffer, mode="clip").astype(np.uint8)

    semantic_id_buffer = np.asarray(semantic_id_buffer)
    ids, counts = np.unique(semantic_id_buffer, return_counts=True)
    max_id = int(ids.max()) if ids.size else 0
    slices = (
        ndimage.find_objects(semantic_id_buffer, max_label=max_id) if max_id > 0 else []
    )

    objects: List[ObjectRecord] = []
    for mesh_id, count in zip(ids.tolist(), counts.tolist()):
        entry = mesh_id_map.get(str(mesh_id))
        if entry is None or entry["category"] not in included_categories:
            continue  # background / reserved ids / unregistered / filtered-out category
        y_slice, x_slice = slices[mesh_id - 1]
        objects.append(
            ObjectRecord(
                mesh_id=mesh_id,
                category=entry["category"],
                pixel_count=count,
                bbox=(
                    x_slice.start,
                    y_slice.start,
                    x_slice.stop - x_slice.start,
                    y_slice.stop - y_slice.start,
                ),
            )
        )
    return category_mask, objects


def write_segmentation_assets(
    out_dir: Path,
    frame_id: str,
    rgb: np.ndarray,
    instance_mask: np.ndarray,
    category_mask: np.ndarray,
) -> Dict[str, str]:
    """Persist one frame's three PNGs under out_dir/{rgb,masks_instance,masks_category}/
    and return their paths (relative to out_dir) for
    EpisodeLogger.log_segmentation_frame. instance_mask is saved as a 16-bit
    PNG (mesh_ids overflow uint8) -- keeping the raw instance buffer is a
    deliberate, cheap-now choice (next.md: instance masks are cheap to store
    and expensive to regenerate later if only category masks were kept).
    category_mask is saved as 8-bit PNG (few classes). rgb is saved uint8,
    matching sim_utils.rgb_depth's convention."""
    import imageio.v3 as iio

    out_dir = Path(out_dir)
    rgb_dir = out_dir / "rgb"
    instance_dir = out_dir / "masks_instance"
    category_dir = out_dir / "masks_category"
    for d in (rgb_dir, instance_dir, category_dir):
        d.mkdir(parents=True, exist_ok=True)

    rgb_path = rgb_dir / f"{frame_id}.png"
    instance_path = instance_dir / f"{frame_id}.png"
    category_path = category_dir / f"{frame_id}.png"

    iio.imwrite(rgb_path, np.asarray(rgb, dtype=np.uint8))
    iio.imwrite(instance_path, np.asarray(instance_mask, dtype=np.uint16))
    iio.imwrite(category_path, np.asarray(category_mask, dtype=np.uint8))

    return {
        "rgb_path": str(rgb_path.relative_to(out_dir)),
        "instance_mask_path": str(instance_path.relative_to(out_dir)),
        "category_mask_path": str(category_path.relative_to(out_dir)),
    }


if __name__ == "__main__":
    h, w = 8, 8
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    semantic = np.zeros((h, w), dtype=np.int32)
    semantic[0:3, 0:3] = 1024  # small_rock
    semantic[5:8, 5:8] = 1025  # big_rock
    semantic[3:5, 3:5] = 3  # ROCK_SEMANTIC_ID, not in registry -> background

    mesh_id_map = {
        "1024": {"category": "small_rock", "name": "rock_a"},
        "1025": {"category": "big_rock", "name": "rock_b"},
    }
    categories = ["small_rock", "big_rock", "bedrock", "hole_in_ground"]
    lut, class_names = build_category_lut(mesh_id_map, categories)
    assert class_names == [
        "background",
        "small_rock",
        "big_rock",
        "bedrock",
        "hole_in_ground",
    ]

    category_mask, objects = capture_frame_record(
        rgb, semantic, mesh_id_map, lut, class_names
    )

    small_rock_idx = class_names.index("small_rock")
    big_rock_idx = class_names.index("big_rock")
    assert (category_mask[0:3, 0:3] == small_rock_idx).all()
    assert (category_mask[5:8, 5:8] == big_rock_idx).all()
    assert (category_mask[3:5, 3:5] == 0).all()  # unregistered id -> background

    by_id = {o.mesh_id: o for o in objects}
    assert by_id[1024].category == "small_rock" and by_id[1024].pixel_count == 9
    assert by_id[1025].category == "big_rock" and by_id[1025].pixel_count == 9
    assert 3 not in by_id  # reserved/unregistered id never becomes an object record

    background_pixels = int((category_mask == 0).sum())
    assert sum(o.pixel_count for o in objects) + background_pixels == h * w

    # category filter: excluding big_rock folds it into background too
    filtered_lut, filtered_names = build_category_lut(mesh_id_map, ["small_rock"])
    filtered_mask, filtered_objects = capture_frame_record(
        rgb, semantic, mesh_id_map, filtered_lut, filtered_names
    )
    assert filtered_names == ["background", "small_rock"]
    assert [o.category for o in filtered_objects] == ["small_rock"]
    assert (filtered_mask[5:8, 5:8] == 0).all()  # big_rock now background

    print("OK")
