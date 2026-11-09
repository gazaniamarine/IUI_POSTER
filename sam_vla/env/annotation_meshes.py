"""Batch-loads the manually-annotated convex-hull sub-meshes (mesh_annotation_tool.py,
next.md Steps 1-2) into a live sim as render-only semantic objects -- the runtime
counterpart of register_rocks (sam_vla.env.rock_generation), but keyed by the
persistent mesh_id already assigned in mesh_id_map.json rather than positional index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from sam_vla.core.goal_geometry import (
    FLAG_SEMANTIC_ID,
    HOME_BASE_SEMANTIC_ID,
    MESH_GOAL_ID,
    MESH_OBST_ID,
    ROCK_SEMANTIC_ID,
)
from sam_vla.env.sim_utils import register_semantic_mesh

RESERVED_SEMANTIC_IDS = {
    0,
    MESH_GOAL_ID,
    MESH_OBST_ID,
    ROCK_SEMANTIC_ID,
    FLAG_SEMANTIC_ID,
    HOME_BASE_SEMANTIC_ID,
}

# Annotation hulls are terrain-following patches whose vertices already sit
# exactly at the heightmap's sampled ground height -- coplanar with the real
# render terrain, which z-fights almost everywhere instead of rendering the
# semantic id (see sim_utils.register_semantic_mesh). Lifting the whole
# object clears that. habitat_env.get_full_observation renders these meshes
# only for the semantic pass (never the RGB pass, via sim_utils.set_objects_hidden),
# so this only has to survive z-fighting on the semantic camera -- not stay
# small enough to be an unnoticeable RGB artifact. Keep it as small as
# possible anyway: a larger lift shifts the mesh's projected 2D footprint
# under oblique-camera parallax, which can offset masks/bboxes from the true
# rock silhouette. Verify empirically (inspect the semantic buffer for
# z-fighting holes/speckle) before changing this value.
DEFAULT_Y_LIFT = 0.005


def load_mesh_id_map(annotations_dir: Path) -> Dict[str, Dict[str, str]]:
    """Read annotations_dir/mesh_id_map.json -> {"1024": {"category": ..., "name": ...}, ...}."""
    path = Path(annotations_dir) / "mesh_id_map.json"
    payload = json.loads(path.read_text())
    return payload["mesh_id_map"]


def register_annotation_meshes(
    sim,
    annotations_dir: str,
    categories: Optional[Sequence[str]] = None,
    y_lift: float = DEFAULT_Y_LIFT,
) -> Dict[int, Any]:
    """Load subobjects/mesh_{id}_object_{id}.obj as render-only semantic
    objects tagged with their registry mesh_id, via
    sim_utils.register_semantic_mesh. If `categories` is given, only meshes
    whose registry category is in that set are registered at all -- e.g.
    categories=["small_rock"] loads only the small_rock hulls, so
    big_rock/bedrock/hole_in_ground hulls are never added to the scene (not
    just excluded from the mask). This is both a correctness knob (isolate
    exactly one category for a focused test/dataset) and an efficiency knob
    (fewer registered objects -> cheaper render when testing a subset).
    `categories=None` registers every entry. Raises if a registry mesh_id
    collides with RESERVED_SEMANTIC_IDS or if a referenced OBJ is missing --
    fail loudly at load time, not silently mid-sweep."""
    annotations_dir = Path(annotations_dir)
    mesh_id_map = load_mesh_id_map(annotations_dir)
    subobjects_dir = annotations_dir / "subobjects"
    category_filter = set(categories) if categories is not None else None

    objects: Dict[int, Any] = {}
    for mesh_id_str, entry in mesh_id_map.items():
        mesh_id = int(mesh_id_str)
        if mesh_id in RESERVED_SEMANTIC_IDS:
            raise ValueError(
                f"annotation mesh_id {mesh_id} ({entry.get('name')}) collides with a "
                f"reserved semantic id {RESERVED_SEMANTIC_IDS}"
            )
        if category_filter is not None and entry["category"] not in category_filter:
            continue

        mesh_path = subobjects_dir / f"mesh_{mesh_id}_object_{mesh_id}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"mesh_id_map references missing hull mesh {mesh_path} for mesh_id {mesh_id}"
            )
        objects[mesh_id] = register_semantic_mesh(
            sim, str(mesh_path), mesh_id, y_offset=y_lift
        )

    return objects
