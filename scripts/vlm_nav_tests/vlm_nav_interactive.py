"""Interactive frame capture + labelme annotation handoff for VLM navigation:
loads the marsyard2022 scene from vlm_nav_demo.py's config, captures a frame
on spacebar, hands it to labelme for annotation, and validates the resulting
annotation JSON for structural correctness.

Usage:
    python scripts/vlm_nav_tests/vlm_nav_interactive.py
"""

import os
import re
import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import quaternion
import tkinter as tk

import habitat_sim
from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_rotate_vector

from vlm_nav_demo import GoToGoalController, DT as NAV_DT, MAX_STEPS as NAV_MAX_STEPS
from conda_env import resolve_conda_base

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "marsyard2022.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm_1025.tif")

OUT_DIR = str(HERE / "vlm_nav_out")
ANNOTATIONS_DIR = str(HERE / "annotations")
MASKS_DIR = str(HERE / "masks")

SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566

# Start pose
START_X = 12.2
START_Z = 10.0
START_YAW_DEG = 10.0
INITIAL_CLEARANCE = 0.9

# display
RGBD_RESOLUTION = [480, 640]

# camera intrinisics
HFOV_DEG = 90.0

# Semantic categories for dynamically generated goal/obstacle meshes.
# 0 (environment/default scene) is never assigned explicitly - it's what
# Habitat-Sim reports for the static stage and any object nobody has
# touched.
SEMANTIC_ID_ENVIRONMENT = 0
SEMANTIC_ID_GOAL = 1
SEMANTIC_ID_OBSTACLE = 2

# hm correction
FLIP_HEIGHTMAP_X = False
FLIP_HEIGHTMAP_Z = True
SWAP_HEIGHTMAP_XZ = False

# annotation
LABELS_FILE = str(HERE / "labels.txt")

CONDA_BASE = resolve_conda_base()
LABELME_BIN = f"{CONDA_BASE}/envs/annotate/bin/labelme"

VLM_PYTHON_BIN = f"{CONDA_BASE}/envs/qwen_vlm/bin/python"
VLM_SCRIPT = str(HERE / "vlm_query.py")
VLM_PROMPT = """You must output ONLY valid JSON with no other text.

1. Goal Object: Identify the single most scientifically significant detected object (can be rock or sand). This is your primary target. Select exactly ONE.
2. Obstacles: every OTHER detected object (i.e. every detected object except the one you picked as the goal) is an obstacle the rover must drive to/around before it heads to the goal. List ALL of them, not a subset.

Output format:
{
  "goal_object": {
    "object_id": <int>,
    "label": <string>,
    "coordinates2D": <array>,
    "reasoning": <string>
  },
  "obstacles": [
    {
      "object_id": <int>,
      "label": <string>,
      "coordinates2D": <array>,
      "reasoning": <string>
    }
  ]
}

Requirements:
- Exactly one goal_object, chosen from the detected objects
- "obstacles" must contain every detected object except the goal_object -- none may be omitted
- No obstacle may share an object_id with the goal_object
- "obstacles" must always be a JSON array (never a bare object), even when it holds a single entry
- Output ONLY JSON, no preamble or explanation
- If there are no other detected objects besides the goal, output "obstacles": []"""

"""VLM_PROMPT = "Identify the most scientifically significant test subject among the detected objects. Output strictly in JSON format with keys: 'object_id', 'label', 'coordinates2Darray', 'reasoning'."""


# Heightmap utilities
def load_heightmap(path):
    """Load and normalize heightmap from PNG."""
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)

    y = arr * SIZE_Y
    y = y - np.mean(y)

    return y


HEIGHT = load_heightmap(HEIGHTMAP)
HM_H, HM_W = HEIGHT.shape


def terrain_height_at(x, z):
    """Bilinear interpolation of terrain height at (x, z)."""
    if SWAP_HEIGHTMAP_XZ:
        x, z = z, x
    u = (x + SIZE_X / 2.0) / SIZE_X
    v = (z + SIZE_Z / 2.0) / SIZE_Z
    if FLIP_HEIGHTMAP_X:
        u = 1.0 - u
    if FLIP_HEIGHTMAP_Z:
        v = 1.0 - v
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    px = u * (HM_W - 1)
    py = v * (HM_H - 1)
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, HM_W - 1)
    y1 = min(y0 + 1, HM_H - 1)
    dx = px - x0
    dy = py - y0
    h00 = HEIGHT[y0, x0]
    h10 = HEIGHT[y0, x1]
    h01 = HEIGHT[y1, x0]
    h11 = HEIGHT[y1, x1]
    h0 = h00 * (1.0 - dx) + h10 * dx
    h1 = h01 * (1.0 - dx) + h11 * dx
    return float(h0 * (1.0 - dy) + h1 * dy)


# Default heightmap sampler, captured before any local shadowing (e.g. the
# `terrain_height_at` override parameter on make_local_height_patch_mesh).
_DEFAULT_TERRAIN_HEIGHT_AT = terrain_height_at


# Sensor and simulator setup
def make_sensor(uuid, sensor_type):
    """Create a camera sensor spec (RGB or depth)."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = RGBD_RESOLUTION
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = HFOV_DEG
    return spec


def make_sim():
    """Initialize Habitat simulator with scene and sensors."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.enable_physics = False

    rgb = make_sensor("rgb", habitat_sim.SensorType.COLOR)
    depth = make_sensor("depth", habitat_sim.SensorType.DEPTH)
    semantic = make_sensor("semantic", habitat_sim.SensorType.SEMANTIC)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth, semantic]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def rgb_depth_from_obs(obs):
    """Extract and process RGB and depth from sensor observation."""
    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb.astype(np.uint8)

    depth = obs["depth"]
    return rgb, depth


def semantic_from_obs(obs):
    """
    Extract the per-pixel semantic ID image from a sensor observation.

    Habitat-Sim's semantic sensor reports obj.semantic_id (see
    SemanticMeshRegistry.register) for pixels covered by a registered
    goal/obstacle mesh, and 0 (SEMANTIC_ID_ENVIRONMENT) for the static
    stage and everything else - see SEMANTIC_ID_* above. Values are
    always in that small {0, 1, 2, ...} range, so an 8-bit image is
    lossless and keeps the file directly viewable/loadable as a label map.
    """
    semantic = np.asarray(obs["semantic"])
    return semantic.astype(np.uint8)


def semantic_to_masks(semantic_img):
    """
    Convert a per-pixel semantic ID image into binary goal/obstacle masks.

    Args:
        semantic_img: HxW array of semantic category ids (see SEMANTIC_ID_*).

    Returns:
        (goal_mask, obstacle_mask): HxW uint8 arrays, 0=background,
        255=pixels belonging to that category. Derived straight from the
        renderer's semantic buffer, so partial visibility/occlusion at the
        current camera pose is reflected automatically - no bbox involved.
    """
    goal_mask = np.where(semantic_img == SEMANTIC_ID_GOAL, 255, 0).astype(np.uint8)
    obstacle_mask = np.where(semantic_img == SEMANTIC_ID_OBSTACLE, 255, 0).astype(
        np.uint8
    )
    return goal_mask, obstacle_mask


def save_frame(obs, frame_idx):
    """Save RGB, depth, semantic, and derived goal/obstacle mask frames to disk."""
    os.makedirs(OUT_DIR, exist_ok=True)

    rgb, depth = rgb_depth_from_obs(obs)
    semantic = semantic_from_obs(obs)

    rgb_path = f"{OUT_DIR}/rgb_{frame_idx:04d}.png"
    depth_path = f"{OUT_DIR}/depth_{frame_idx:04d}.png"
    semantic_path = f"{OUT_DIR}/semantic_{frame_idx:04d}.png"
    goal_mask_path = f"{OUT_DIR}/goal_mask_{frame_idx:04d}.png"
    obstacle_mask_path = f"{OUT_DIR}/obstacle_mask_{frame_idx:04d}.png"

    Image.fromarray(rgb).save(rgb_path)

    # Normalize depth for visualization
    depth_clip = np.clip(depth, 0.0, 10.0)
    depth_vis = (depth_clip / 10.0 * 255.0).astype(np.uint8)
    Image.fromarray(depth_vis).save(depth_path)

    # Raw metric depth, kept alongside the 8-bit visualization so bbox->world back-projection isn't limited to the 0-10m visualization clip range.
    depth_npy_path = f"{OUT_DIR}/depth_{frame_idx:04d}.npy"
    np.save(depth_npy_path, depth.astype(np.float32))

    Image.fromarray(semantic).save(semantic_path)

    goal_mask, obstacle_mask = semantic_to_masks(semantic)
    Image.fromarray(goal_mask).save(goal_mask_path)
    Image.fromarray(obstacle_mask).save(obstacle_mask_path)

    print(f"[captured] {rgb_path}")
    print(f"[captured] {depth_path}")
    print(f"[captured] {depth_npy_path}")
    print(f"[captured] {semantic_path}")
    print(f"[captured] {goal_mask_path}")
    print(f"[captured] {obstacle_mask_path}")

    return rgb_path


def save_pose(frame_idx, x, y, z, yaw):
    """Persist the agent pose active when a frame was captured, for later back-projection."""
    pose_path = f"{OUT_DIR}/pose_{frame_idx:04d}.json"
    with open(pose_path, "w") as f:
        json.dump({"x": x, "y": y, "z": z, "yaw": yaw}, f, indent=2)
    return pose_path


def load_pose(frame_idx):
    """Load the pose saved for a frame, falling back to the fixed start pose for older captures."""
    pose_path = f"{OUT_DIR}/pose_{frame_idx:04d}.json"
    if os.path.exists(pose_path):
        with open(pose_path, "r") as f:
            return json.load(f)

    print(f"[warn] no saved pose for frame {frame_idx}, assuming the fixed start pose")
    x, z = START_X, START_Z
    return {
        "x": x,
        "y": terrain_height_at(x, z) + INITIAL_CLEARANCE,
        "z": z,
        "yaw": float(np.deg2rad(START_YAW_DEG)),
    }


def load_depth_frame(frame_idx):
    """Load raw metric depth for a frame, reconstructing from the 8-bit visualization if needed."""
    npy_path = f"{OUT_DIR}/depth_{frame_idx:04d}.npy"
    if os.path.exists(npy_path):
        return np.load(npy_path)

    png_path = f"{OUT_DIR}/depth_{frame_idx:04d}.png"
    print(
        f"[warn] no raw depth for frame {frame_idx}, reconstructing from {png_path} "
        f"(precision limited to the 0-10m visualization clip range)"
    )
    depth_vis = np.array(Image.open(png_path)).astype(np.float32)
    return depth_vis / 255.0 * 10.0


def load_detected_shapes(frame_idx):
    """
    Load the SAM/labelme annotation for a frame, keyed by the object id
    `assign_object_ids` stamped onto each shape.

    This is the ground truth for where each detected object actually is.
    The VLM is only asked to pick *which* object_id is the goal/each
    obstacle - its own echoed "coordinates2D" is not reliable geometry (it
    has been observed to just copy one corner of the annotation instead of
    describing the object), so mesh/position resolution must look the real
    bounding box up here by object_id rather than trust the VLM's reply.

    Returns:
        {object_id: shape_dict}, shape_dict has "label" and "points" (the
        labelme rectangle's two corners, in pixels).
    """
    annotation_path = f"{ANNOTATIONS_DIR}/rgb_{frame_idx:04d}.json"
    with open(annotation_path, "r") as f:
        data = json.load(f)
    return {shape["id"]: shape for shape in data["shapes"]}


def validate_annotation_json(json_path):
    """
    Validate labelme annotation JSON structure.

    Args:
        json_path: Path to the annotation JSON file

    Returns:
        (is_valid, status_message)
    """
    if not os.path.exists(json_path):
        return False, f"annotation file not found: {json_path}"

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    except Exception as e:
        return False, f"error reading file: {e}"

    # Validate required top-level keys
    required_keys = [
        "version",
        "flags",
        "shapes",
        "imagePath",
        "imageHeight",
        "imageWidth",
    ]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        return False, f"missing required keys: {missing_keys}"

    # Validate shapes structure
    if not isinstance(data["shapes"], list):
        return False, f"'shapes' must be a list, got {type(data['shapes']).__name__}"

    for i, shape in enumerate(data["shapes"]):
        if not isinstance(shape, dict):
            return False, f"shape {i}: expected dict, got {type(shape).__name__}"

        shape_required = ["label", "points"]
        shape_missing = [k for k in shape_required if k not in shape]
        if shape_missing:
            return False, f"shape {i}: missing required keys {shape_missing}"

        if not isinstance(shape["label"], str) or not shape["label"]:
            return False, f"shape {i}: 'label' must be a non-empty string"

        if not isinstance(shape["points"], list):
            return False, f"shape {i}: 'points' must be a list"

        if len(shape["points"]) < 2:
            return False, f"shape {i}: 'points' must have at least 2 coordinates"

        for j, point in enumerate(shape["points"]):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return False, f"shape {i}, point {j}: expected [x, y], got {point}"

    return True, "accepted"


# Bounding box overlay
BOX_COLOR_PALETTE = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
]


def load_labels(labels_path=LABELS_FILE):
    """Load ordered label list; list index is the class id."""
    with open(labels_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def color_for_class(class_id):
    """Deterministic color per class id."""
    return BOX_COLOR_PALETTE[class_id % len(BOX_COLOR_PALETTE)]


def assign_object_ids(annotation_path):
    """
    Assign a unique incrementing object id (0, 1, 2, ...) to each shape in a
    labelme annotation and persist it back to the JSON.

    This lets a VLM reading the RGB frame, the annotation JSON, and the overlay
    PNG together refer to "object 3" and have that resolve to the exact same
    shape across all three artifacts.

    Args:
        annotation_path: Path to the labelme annotation JSON (modified in place)

    Returns:
        The updated annotation dict.
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)

    for i, shape in enumerate(data["shapes"]):
        shape["id"] = i

    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def draw_annotation_overlay(image_path, annotation_path, output_path):
    """
    Assign per-object ids to a labelme annotation, then draw its labeled
    bounding boxes (tagged with those ids) onto the source image.

    Args:
        image_path: Path to the source RGB image
        annotation_path: Path to the validated labelme annotation JSON
        output_path: Path to write the overlaid image

    Returns:
        output_path
    """
    data = assign_object_ids(annotation_path)

    labels = load_labels()
    label_to_id = {name: i for i, name in enumerate(labels)}

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for shape in data["shapes"]:
        object_id = shape["id"]
        label = shape["label"]
        class_id = label_to_id.get(label, -1)
        color = color_for_class(max(class_id, 0))
        points = [tuple(p) for p in shape["points"]]

        if shape.get("shape_type") == "rectangle":
            (x0, y0), (x1, y1) = points
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            tag_xy = (x0, max(y0 - 12, 0))
        else:
            draw.polygon(points, outline=color)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            tag_xy = (min(xs), max(min(ys) - 12, 0))

        tag = f"#{object_id} {label}"
        tag_bbox = draw.textbbox(tag_xy, tag)
        draw.rectangle(tag_bbox, fill=color)
        draw.text(tag_xy, tag, fill=(255, 255, 255))

    img.save(output_path)
    print(f"[overlay] {output_path}")

    return output_path


# Labelme launch
def launch_labelme_on_frame(rgb_path, frame_idx):
    """
    Launch labelme on the captured frame and wait for annotation.

    Args:
        rgb_path: Path to the RGB image
        frame_idx: Frame index (for naming convention)

    Returns:
        (success, annotation_json_path, status_message)
    """
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    cmd = [
        LABELME_BIN,
        rgb_path,
        "--labels",
        LABELS_FILE,
        "--output",
        ANNOTATIONS_DIR,
        "--no-sort-labels",
    ]

    print(f"\n[info] Launching labelme...")
    print(f"[cmd] {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return False, None, f"labelme process exited with code {result.returncode}"
    except FileNotFoundError:
        return False, None, f"labelme binary not found at {LABELME_BIN}"
    except Exception as e:
        return False, None, f"error launching labelme: {e}"

    # Determine expected annotation path (labelme uses input filename)
    annotation_path = f"{ANNOTATIONS_DIR}/rgb_{frame_idx:04d}.json"

    # Validate the annotation
    is_valid, status_msg = validate_annotation_json(annotation_path)

    return is_valid, annotation_path, status_msg


# VLM query
def query_vlm(rgb_path, overlay_path, annotation_path, frame_idx, prompt=VLM_PROMPT):
    """
    Feed the raw RGB frame, its annotated overlay, and the labelme annotation
    JSON to the Qwen2.5-VL model and print/save its response.

    Runs as a subprocess in the `qwen_vlm` conda env (separate from the env
    this script runs in), same pattern as launch_labelme_on_frame.

    Args:
        rgb_path: Path to the raw RGB image
        overlay_path: Path to the annotated overlay image
        annotation_path: Path to the validated labelme annotation JSON
        frame_idx: Frame index (for naming the output file)
        prompt: Question to ask the VLM about the detected objects

    Returns:
        (success, output_path_or_None, status_message)
    """
    out_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_vlm.txt"

    cmd = [
        VLM_PYTHON_BIN,
        VLM_SCRIPT,
        "--rgb",
        rgb_path,
        # "--overlay", overlay_path,
        "--annotation",
        annotation_path,
        "--prompt",
        prompt,
        "--out",
        out_path,
    ]

    print(f"\n[info] Querying VLM")
    print(f"[cmd] {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return False, None, f"VLM process exited with code {result.returncode}"
    except FileNotFoundError:
        return False, None, f"VLM python binary not found at {VLM_PYTHON_BIN}"
    except Exception as e:
        return False, None, f"error launching VLM: {e}"

    return True, out_path, "accepted"


# Bbox -> world position, and facing the chosen object
def parse_vlm_response(vlm_out_path):
    """Parse the VLM's JSON reply, tolerating markdown code fences or stray prose around it."""
    with open(vlm_out_path, "r") as f:
        text = f.read().strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in {vlm_out_path}")
    return json.loads(match.group(0))


def coords2d_to_bbox_xyxy(coords2d):
    """
    Reduce the VLM's 2D coordinates into an axis-aligned (x1, y1, x2, y2)
    bbox. Handles the rectangle case (two corner points, in either order,
    as stored by labelme) as well as denser polygons, by taking the
    point cloud's bounding box. A bare [u, v] pair degenerates to a
    zero-area box at that point.
    """
    pts = np.array(coords2d, dtype=np.float32)
    if pts.ndim == 1:
        u, v = pts
        return float(u), float(v), float(u), float(v)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def camera_intrinsics():
    """Pinhole intrinsics for the RGB/depth sensors (square pixels, matching hfov below)."""
    h, w = RGBD_RESOLUTION
    f = (w / 2.0) / np.tan(np.deg2rad(HFOV_DEG) / 2.0)
    return f, f, w / 2.0, h / 2.0


def pixel_to_world(u, v, depth_value, pose):
    """
    Back-project a pixel + depth reading into a world-space point.

    Habitat's pinhole camera sits at the agent's origin looking down -Z, with +X right and +Y up; depth is the planar (z-axis) distance, not radial range. The camera-space point is rotated into world space with the same yaw-about-Y construction used by set_agent_pose, then offset by the agent's world position.
    """
    fx, fy, cx, cy = camera_intrinsics()
    x_cam = (u - cx) * depth_value / fx
    y_cam = -(v - cy) * depth_value / fy
    z_cam = -depth_value

    q = quaternion.from_rotation_vector([0.0, pose["yaw"], 0.0])
    world_offset = quat_rotate_vector(q, np.array([x_cam, y_cam, z_cam]))

    agent_pos = np.array([pose["x"], pose["y"], pose["z"]])
    return agent_pos + world_offset


# Local mesh patch export
def make_local_height_patch_mesh(
    center_world,
    radius=0.5,
    terrain_height_at=None,
    resolution=0.03,
):
    """
    Rebuild a local circular terrain patch as a triangle mesh, centered on
    `center_world` in the Habitat world frame (X right, Y up, Z per the
    scene's ground plane). Vertices are sampled on a regular XZ grid at
    `resolution` spacing, masked to a disc of `radius`, with Y taken from
    the terrain heightmap at each sample; grid cells are only triangulated
    when all four corners fall inside the disc, so the boundary is a
    (stairstepped) circle rather than a square.

    Args:
        center_world: (x, y, z) world-space point to center the patch on;
            y is unused (each vertex's height is resampled from the
            heightmap) but accepted for convenience since callers already
            have a full seed point.
        radius: patch radius in meters.
        terrain_height_at: fn(x, z) -> y; defaults to the module's MarsYard
            heightmap sampler.
        resolution: grid spacing in meters between adjacent samples.

    Returns:
        (verts, faces): verts is an (N, 3) float array of world-space
        [x, y, z] positions; faces is an (M, 3) int array of 0-based
        vertex indices, wound so the normal points toward +Y (up).
    """
    height_fn = (
        terrain_height_at
        if terrain_height_at is not None
        else _DEFAULT_TERRAIN_HEIGHT_AT
    )

    cx, _, cz = center_world
    n = max(int(np.ceil(radius / resolution)), 1)
    offsets = np.arange(-n, n + 1) * resolution

    index_grid = -np.ones((len(offsets), len(offsets)), dtype=int)
    verts = []
    for j, dz in enumerate(offsets):
        for i, dx in enumerate(offsets):
            if dx * dx + dz * dz > radius * radius:
                continue
            wx = cx + dx
            wz = cz + dz
            wy = height_fn(wx, wz)
            index_grid[j, i] = len(verts)
            verts.append((wx, wy, wz))

    faces = []
    for j in range(len(offsets) - 1):
        for i in range(len(offsets) - 1):
            v00 = index_grid[j, i]
            v10 = index_grid[j, i + 1]
            v01 = index_grid[j + 1, i]
            v11 = index_grid[j + 1, i + 1]
            if v00 < 0 or v10 < 0 or v01 < 0 or v11 < 0:
                continue
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))

    verts = (
        np.array(verts, dtype=np.float64)
        if verts
        else np.empty((0, 3), dtype=np.float64)
    )
    faces = (
        np.array(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
    )
    return verts, faces


def save_obj(path, verts, faces):
    """Write a triangle mesh (0-based vertex indices) to a Wavefront OBJ file."""
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
    return path


# Semantic object registration
def semantic_id_for_role(role):
    """Map a mesh role ("goal", "obstacle", "obstacle1", ...) to its semantic category id."""
    if role == "goal":
        return SEMANTIC_ID_GOAL
    if role.startswith("obstacle"):
        return SEMANTIC_ID_OBSTACLE
    return SEMANTIC_ID_ENVIRONMENT


class SemanticMeshRegistry:
    """
    Loads dynamically generated mesh patches (goal/obstacle terrain regions)
    into a Habitat-Sim scene as temporary, render-only semantic objects.

    Every object this registers is forced kinematic and non-collidable, so
    it exists purely for the semantic/RGB/depth sensors to see - it never
    enters collision detection, never gets a navmesh presence, and the
    GoToGoalController's depth-based obstacle_avoidance_hook is untouched
    (it only ever reads the depth sensor, which these objects still
    contribute to exactly like the terrain patch they represent, since the
    mesh geometry is the same local terrain resampled).
    """

    def __init__(self, sim):
        self.sim = sim
        self.object_template_manager = sim.get_object_template_manager()
        self.rigid_object_manager = sim.get_rigid_object_manager()
        self._object_ids = []  # ManagedRigidObject.object_id, registration order

    def register(self, mesh_path, role, semantic_id, transform=None):
        """
        Register one mesh patch as a semantic object.

        Args:
            mesh_path: path to the mesh file (e.g. an OBJ produced by
                make_local_height_patch_mesh + save_obj).
            role: "goal", "obstacle", "obstacle1", ... - used only to build
                a unique object template handle; purely for bookkeeping.
            semantic_id: integer semantic category id (see SEMANTIC_ID_*).
            transform: optional (translation, rotation) pair placing the
                object in world coordinates - translation is an (x, y, z)
                world position and rotation a quaternion.quaternion. Left
                as None (the default) for meshes like ours that already
                bake world-space vertex positions in: Habitat-Sim derives
                a newly added object's initial translation/rotation from
                its own asset bounding box, which already lines up with
                those baked-in coordinates, so touching it would only
                relocate the object away from where its mesh actually is.
                Pass an explicit transform for meshes authored in local
                (object-relative) coordinates instead.

        Returns:
            The newly created habitat_sim.physics.ManagedRigidObject.
        """
        template = self.object_template_manager.create_new_template(mesh_path)
        template.render_asset_handle = mesh_path
        template.collision_asset_handle = mesh_path
        template.is_collidable = False

        template_name = (
            f"semantic_{role}_{len(self._object_ids)}_{os.path.basename(mesh_path)}"
        )
        template_id = self.object_template_manager.register_template(
            template, template_name
        )
        template_handle = self.object_template_manager.get_template_handle_by_id(
            template_id
        )

        obj = self.rigid_object_manager.add_object_by_template_handle(template_handle)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.collidable = False
        obj.semantic_id = int(semantic_id)
        if transform is not None:
            translation, rotation = transform
            obj.translation = np.array(translation, dtype=np.float32)
            if rotation is not None:
                obj.rotation = rotation

        self._object_ids.append(obj.object_id)
        print(
            f"[semantic] registered {role} mesh '{mesh_path}' as "
            f"object_id={obj.object_id} semantic_id={semantic_id}"
        )
        return obj

    def clear(self):
        """Remove every object this registry has added so far."""
        for obj_id in self._object_ids:
            if self.rigid_object_manager.get_library_has_id(obj_id):
                self.rigid_object_manager.remove_object_by_id(obj_id)
        self._object_ids.clear()


def selected_bbox_to_object_mesh(
    selected_obj,
    depth,
    pose,
    frame_idx,
    role,
    pixel_to_world,
    terrain_height_at,
    output_dir,
    radius=0.5,
    detected_shapes=None,
):
    """
    Resolve one VLM-selected object entry (goal or obstacle) to a world seed
    point via bbox back-projection, extract its local terrain mesh patch,
    save it to disk, and bundle both into a single metadata dict.

    This is the one place bbox->world seeding and mesh extraction meet, so
    every VLM-selected goal/obstacle - whether resolved live during the
    interactive loop or replayed via --vlm/--goto - gets an on-disk mesh
    and consistent metadata without re-deriving the position twice.

    Args:
        selected_obj: VLM object entry (goal_object or one obstacle), with
            "object_id", "label" and "coordinates2D" keys.
        depth: HxW metric depth array for the frame.
        pose: agent pose dict active when the frame was captured.
        frame_idx: frame index the object was resolved from (for naming).
        role: "goal", "obstacle", "obstacle1", etc.
        pixel_to_world: fn(u, v, depth_value, pose) -> array-like [x, y, z].
        terrain_height_at: fn(x, z) -> y.
        output_dir: directory to save the mesh OBJ into.
        radius: patch radius in meters.
        detected_shapes: {object_id: shape} from `load_detected_shapes`, the
            SAM/labelme ground truth to resolve the bbox from instead of
            the VLM's own (unreliable) coordinates2D.

    Returns:
        {
            "role": role, "label": label, "bbox": [x1, y1, x2, y2],
            "seed_world": [x, y, z], "mesh_path": path, "radius": radius,
            "num_vertices": int, "num_faces": int,
        }
    """
    seed_world, bbox_xyxy, label = _resolve_entry_seed_world(
        selected_obj, depth, pose, pixel_to_world, terrain_height_at, detected_shapes
    )

    safe_label = (
        re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_") or "object"
    )
    mesh_path = os.path.join(
        output_dir, f"rgb_{frame_idx:04d}_{role}_{safe_label}_r{radius:.2f}.obj"
    )

    verts, faces = make_local_height_patch_mesh(
        seed_world, radius=radius, terrain_height_at=terrain_height_at
    )
    save_obj(mesh_path, verts, faces)
    print(f"[mesh] saved {mesh_path} ({len(verts)} verts, {len(faces)} faces)")

    return {
        "role": role,
        "label": label,
        "bbox": [round(float(c), 2) for c in bbox_xyxy],
        "seed_world": [round(float(c), 3) for c in seed_world],
        "mesh_path": mesh_path,
        "radius": radius,
        "num_vertices": int(len(verts)),
        "num_faces": int(len(faces)),
    }


def extract_obstacle_entries(response, goal_entry=None):
    """
    Normalize the VLM response's obstacle field into a list of entries.

    Tolerates the current schema ("obstacles": [...]), a single dict under
    that same key, and the older single-"obstacle" schema, so a stale
    rgb_*_vlm.txt captured before the prompt grew a list still works. Any
    null/empty entries are dropped so a missing-detection case just yields
    zero obstacles instead of a crash.

    If `goal_entry` is given, any entry sharing its "object_id" is dropped
    too -- a defensive backstop (independent of the VLM prompt's own "no
    obstacle may share the goal's object_id" instruction) so the goal can
    never end up double-booked as an obstacle regardless of how many
    obstacles get selected downstream.
    """
    entries = response.get("obstacles")
    if entries is None:
        legacy = response.get("obstacle")
        entries = [legacy] if legacy else []
    elif isinstance(entries, dict):
        entries = [entries]

    entries = [e for e in entries if e]

    if goal_entry:
        goal_id = goal_entry.get("object_id")
        entries = [e for e in entries if e.get("object_id") != goal_id]

    return entries


def _resolve_entry_seed_world(
    entry, depth, pose, pixel_to_world, terrain_height_at, detected_shapes=None
):
    """
    Resolve a single VLM object entry's bbox to a world-space seed point.
    Shared by `entry_world_position` (plain position lookups) and
    `selected_bbox_to_object_mesh` (position + mesh extraction) so the
    resolution logic only lives in one place.

    The bbox comes from `detected_shapes` (the SAM/labelme annotation),
    looked up by the entry's "object_id" - NOT from the VLM's own
    "coordinates2D", which has been observed to just echo one corner of
    the annotation rather than describe the object, silently detaching the
    resolved mesh from what SAM actually detected. `detected_shapes` is
    optional only so this still degrades gracefully (with a loud warning)
    for callers that haven't been updated to load it.

    Only that bbox's exact center pixel is used (no grid sampling across
    the box), so a loose annotation box can't pull the seed toward its own
    background. The depth image is read at that single pixel and
    back-projected via `pixel_to_world`; y is then re-derived from the
    terrain heightmap at the resulting (x, z), since the depth reading
    lands on the object's surface, not the ground it's resting on.

    Returns:
        (seed_world, bbox_xyxy, label) - seed_world is an np.array([x, y, z]).
    """
    object_id = entry.get("object_id")
    shape = detected_shapes.get(object_id) if detected_shapes is not None else None

    if shape is not None:
        label = shape.get("label", entry.get("label", "?"))
        bbox_xyxy = coords2d_to_bbox_xyxy(shape["points"])
    else:
        print(
            f"[warn] object_id={object_id} not found in SAM annotation; "
            f"falling back to the VLM's own coordinates2D (unreliable)"
        )
        label = entry.get("label", "?")
        bbox_xyxy = coords2d_to_bbox_xyxy(entry["coordinates2D"])

    x1, y1, x2, y2 = bbox_xyxy
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    h, w = depth.shape[:2]
    px = int(np.clip(round(u), 0, w - 1))
    py = int(np.clip(round(v), 0, h - 1))
    depth_value = float(depth[py, px])
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        print(
            f"[warn] non-positive/invalid depth ({depth_value}) at bbox-center pixel ({px},{py}) "
            f"for '{label}'; result will be unreliable"
        )
    world = pixel_to_world(u, v, depth_value, pose)
    grounded_y = terrain_height_at(world[0], world[2])
    seed_world = np.array([world[0], grounded_y, world[2]], dtype=np.float64)

    print(
        f"[object] '{label}' object_id={object_id} bbox={tuple(round(c, 1) for c in bbox_xyxy)} "
        f"center=({u:.1f},{v:.1f}) depth={depth_value:.2f}m "
        f"seed=({seed_world[0]:.2f}, {seed_world[1]:.2f}, {seed_world[2]:.2f})"
    )

    return seed_world, bbox_xyxy, label


def entry_world_position(entry, frame_idx):
    """
    Resolve a single VLM object entry (goal_object or one obstacle) to a
    world position, using the depth/pose/annotation captured alongside
    frame_idx.

    Returns:
        (x, y, z, label)
    """
    depth = load_depth_frame(frame_idx)
    pose = load_pose(frame_idx)
    detected_shapes = load_detected_shapes(frame_idx)

    seed_world, _, label = _resolve_entry_seed_world(
        entry, depth, pose, pixel_to_world, terrain_height_at, detected_shapes
    )

    return float(seed_world[0]), float(seed_world[1]), float(seed_world[2]), label


def object_world_position(frame_idx):
    """
    Resolve the VLM-chosen goal object (from rgb_{idx}_vlm.txt) to a world
    position. Kept as a convenience for the single-object --face CLI path;
    the full obstacles-then-goal mission uses resolve_mission_meshes directly.

    Returns:
        (x, y, z, label)
    """
    vlm_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_vlm.txt"
    response = parse_vlm_response(vlm_path)
    goal_entry = response.get("goal_object")
    if not goal_entry:
        raise ValueError(f"no goal_object in {vlm_path}")

    return entry_world_position(goal_entry, frame_idx)


def resolve_mission_meshes(frame_idx, response, radius=0.5, obstacle_limit=None):
    """
    Resolve every VLM-selected object in a frame's response (goal +
    obstacles) to world positions and local terrain meshes, without driving
    the rover. Shared by the live `navigate_mission` drive and the
    standalone --vlm/--goto CLI reruns, so mesh generation happens exactly
    once per object regardless of which path triggers it.

    Obstacle selection runs entirely separately from (and after) goal
    selection: the goal is resolved first from `response["goal_object"]`,
    then obstacle candidates are drawn from the remaining detections with
    the goal's own object_id excluded (see `extract_obstacle_entries`), and
    only then is `obstacle_limit` applied to that goal-free pool.

    Args:
        obstacle_limit: None keeps every resolved obstacle (default,
            matches prior behavior). An int caps how many of the VLM's
            obstacle entries are resolved, taken in the order the VLM
            returned them (e.g. 1 for "single", 5 for "up to 5", or a
            number >= the detection count for "everything except the
            goal").

    Returns:
        (goal_mesh, obstacle_meshes) - goal_mesh is None if there's no
        goal_object in `response`; obstacle_meshes is a list of metadata
        dicts, skipping any obstacle entry that fails to resolve.
    """
    depth = load_depth_frame(frame_idx)
    pose = load_pose(frame_idx)
    detected_shapes = load_detected_shapes(frame_idx)

    goal_entry = response.get("goal_object")
    goal_mesh = None
    if goal_entry:
        goal_mesh = selected_bbox_to_object_mesh(
            goal_entry,
            depth,
            pose,
            frame_idx,
            "goal",
            pixel_to_world,
            terrain_height_at,
            OUT_DIR,
            radius=radius,
            detected_shapes=detected_shapes,
        )

    candidate_obstacles = extract_obstacle_entries(response, goal_entry=goal_entry)
    if obstacle_limit is not None:
        candidate_obstacles = candidate_obstacles[:obstacle_limit]

    obstacle_meshes = []
    for i, obstacle in enumerate(candidate_obstacles):
        role = "obstacle" if i == 0 else f"obstacle{i}"
        try:
            mesh_meta = selected_bbox_to_object_mesh(
                obstacle,
                depth,
                pose,
                frame_idx,
                role,
                pixel_to_world,
                terrain_height_at,
                OUT_DIR,
                radius=radius,
                detected_shapes=detected_shapes,
            )
        except Exception as e:
            oid = obstacle.get("object_id", "?") if isinstance(obstacle, dict) else "?"
            print(
                f"[warn] skipping obstacle {i} (object_id={oid}): could not resolve position ({e})"
            )
            continue
        obstacle_meshes.append(mesh_meta)

    return goal_mesh, obstacle_meshes


def resolve_vlm_selection(
    rgb_path,
    overlay_path,
    annotation_path,
    frame_idx,
    prompt=VLM_PROMPT,
    radius=0.5,
    obstacle_limit=None,
):
    """
    Query the VLM for its goal/obstacle object selection on an annotated
    frame, then back-project the chosen bbox(es) through depth + camera
    intrinsics into world-space 3D positions (with local terrain meshes).

    Combines query_vlm + parse_vlm_response + resolve_mission_meshes into
    one call with no Tk/labelme/spacebar dependency, so it's reusable
    outside the interactive capture loop (e.g. from a NavDP rollout script).

    Args:
        rgb_path: Path to the raw RGB frame.
        overlay_path: Path to the annotated overlay image.
        annotation_path: Path to the validated labelme annotation JSON.
        frame_idx: Frame index (for naming the VLM output file and loading
            the frame's saved depth/pose).
        prompt: Question to ask the VLM about the detected objects.
        radius: Local terrain patch radius, in meters, for the resolved mesh(es).
        obstacle_limit: forwarded to `resolve_mission_meshes` -- None (default)
            resolves every obstacle the VLM flagged; an int caps the count.

    Returns:
        (success, result, status_message). On success, result is
        (response, goal_mesh, obstacle_meshes); on failure, result is None.
    """
    vlm_success, vlm_out_path, vlm_status = query_vlm(
        rgb_path, overlay_path, annotation_path, frame_idx, prompt
    )
    if not vlm_success:
        return False, None, vlm_status

    try:
        response = parse_vlm_response(vlm_out_path)
    except Exception as e:
        return False, None, f"could not parse VLM response {vlm_out_path}: {e}"

    goal_mesh, obstacle_meshes = resolve_mission_meshes(
        frame_idx, response, radius=radius, obstacle_limit=obstacle_limit
    )
    if goal_mesh is None:
        return False, None, f"no goal_object in {vlm_out_path}"

    return True, (response, goal_mesh, obstacle_meshes), "accepted"


def save_mission_metadata(
    frame_idx,
    vlm_response,
    goal_mesh,
    obstacle_meshes,
    goal_target_world,
    semantic_paths=None,
):
    """
    Persist the resolved mission for a frame - the VLM reply, every
    generated object mesh (goal + obstacles) enriched with its semantic id
    and rendered mask path, and the final navigation target - as
    rgb_{idx}_mission.json, alongside the raw rgb_{idx}_vlm.txt reply. This
    lets later steps load the chosen goal/obstacle mesh and its mask
    straight from disk without rerunning annotation + VLM + bbox
    back-projection + semantic rendering.

    `semantic_paths` is the dict returned by `capture_mission_semantics`
    (semantic_path/goal_mask_path/obstacle_mask_path); obstacles all share
    one obstacle_mask_path since SemanticMeshRegistry only distinguishes
    the goal/obstacle *category* (SEMANTIC_ID_GOAL/OBSTACLE), not
    per-instance ids.

    Returns:
        Path to the saved JSON file.
    """
    semantic_paths = semantic_paths or {}
    goal_mask_path = semantic_paths.get("goal_mask_path")
    obstacle_mask_path = semantic_paths.get("obstacle_mask_path")

    goal = None
    if goal_mesh is not None:
        goal = dict(goal_mesh)
        goal["semantic_id"] = SEMANTIC_ID_GOAL
        goal["mask_path"] = goal_mask_path

    obstacles = []
    for mesh_meta in obstacle_meshes:
        obstacle = dict(mesh_meta)
        obstacle["semantic_id"] = SEMANTIC_ID_OBSTACLE
        obstacle["mask_path"] = obstacle_mask_path
        obstacles.append(obstacle)

    meta_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_mission.json"
    metadata = {
        "frame_id": frame_idx,
        "vlm_response": vlm_response,
        "goal": goal,
        "obstacles": obstacles,
        "goal_target_world": [float(c) for c in goal_target_world],
        "semantic_render_path": semantic_paths.get("semantic_path"),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[mission] metadata saved: {meta_path}")
    return meta_path


def yaw_to_face(agent_x, agent_z, target_x, target_z):
    """Yaw (about Y, same convention as START_YAW_DEG/set_agent_pose) that points the agent at a target."""
    dx = target_x - agent_x
    dz = target_z - agent_z
    return float(np.arctan2(-dx, -dz))


# Interactive capture loop
class InteractiveCapture:
    """Main interactive capture and annotation class."""

    def __init__(self):
        print("[info] Initializing sim")
        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)
        self.semantic_registry = SemanticMeshRegistry(self.sim)

        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)

        # Compute Y from heightmap
        self.terrain_y = terrain_height_at(self.x, self.z)
        self.y = self.terrain_y + INITIAL_CLEARANCE

        self.set_agent_pose()
        self.frame_idx = 0
        self.root = None  # set by run(); navigate_to_target() pumps it if present

        # Initialize output directory
        os.makedirs(OUT_DIR, exist_ok=True)

        print(f"[info] Simulator ready.")
        print(
            f"[info] Start pose: x={self.x:.2f}, z={self.z:.2f}, yaw={np.rad2deg(self.yaw):.1f}°"
        )
        print(f"\n[info] Controls:")
        print(f"  SPACE - Capture frame and launch labelme")
        print(f"  Q     - Quit")
        print(
            f"\n[info] Window will display RGB stream. Annotations saved to '{ANNOTATIONS_DIR}/'.\n"
        )

    def set_agent_pose(self):
        """Update agent pose in simulator."""
        state = self.agent.get_state()
        state.position = np.array([self.x, self.y, self.z], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0.0, self.yaw, 0.0])
        self.agent.set_state(state)

    def get_observation(self):
        """Get current RGB observation."""
        obs = self.sim.get_sensor_observations()
        rgb, _ = rgb_depth_from_obs(obs)
        return rgb

    def on_spacebar(self):
        """Handle spacebar: capture frame and launch labelme."""
        obs = self.sim.get_sensor_observations()

        # Save RGB and depth
        rgb_path = save_frame(obs, self.frame_idx)
        save_pose(self.frame_idx, self.x, self.y, self.z, self.yaw)

        # Launch labelme
        success, annotation_path, status_msg = launch_labelme_on_frame(
            rgb_path, self.frame_idx
        )

        print(f"\n[annotation status] {status_msg}")

        if success:
            print(f"[info] Annotation saved: {annotation_path}\n")
            overlay_path = f"{OUT_DIR}/rgb_{self.frame_idx:04d}_at.png"
            draw_annotation_overlay(rgb_path, annotation_path, overlay_path)

            vlm_success, vlm_result, vlm_status = resolve_vlm_selection(
                rgb_path, overlay_path, annotation_path, self.frame_idx
            )
            if vlm_success:
                response, goal_mesh, obstacle_meshes = vlm_result
                print(
                    f"[info] VLM response saved: {OUT_DIR}/rgb_{self.frame_idx:04d}_vlm.txt\n"
                )
                self.navigate_mission(
                    self.frame_idx, response, goal_mesh, obstacle_meshes
                )
            else:
                print(f"[error] VLM query failed: {vlm_status}\n")
        else:
            print(f"[error] Annotation validation failed\n")

        self.frame_idx += 1

    def face_object(self, frame_idx):
        """Rotate in place to face the VLM-chosen object from the given frame."""
        try:
            obj_x, obj_y, obj_z, label = object_world_position(frame_idx)
        except Exception as e:
            print(f"[error] could not resolve object position: {e}")
            return

        self.yaw = yaw_to_face(self.x, self.z, obj_x, obj_z)
        self.set_agent_pose()
        print(
            f"[face] turned to face '{label}' at ({obj_x:.2f}, {obj_y:.2f}, {obj_z:.2f}), "
            f"yaw={np.rad2deg(self.yaw):.1f}°"
        )

    def navigate_to_target(self, target_x, target_z, label="target"):
        """
        Drive to (target_x, target_z) using vlm_nav_demo's proportional
        go-to-goal controller, integrating pose the same way RoverSim does.
        """
        target_y = terrain_height_at(target_x, target_z) + INITIAL_CLEARANCE
        controller = GoToGoalController(target_x, target_y, target_z)

        print(f"[nav] driving to '{label}' at ({target_x:.2f}, {target_z:.2f})")

        step = 0
        for step in range(NAV_MAX_STEPS):
            obs = self.sim.get_sensor_observations()
            _, depth = rgb_depth_from_obs(obs)

            linear_x, angular_y, distance, heading_error = controller.update(
                self.x, self.y, self.z, self.yaw
            )
            linear_x, angular_y = controller.obstacle_avoidance_hook(
                linear_x, angular_y, depth
            )

            self.x += linear_x * (-np.sin(self.yaw)) * NAV_DT
            self.z += linear_x * (-np.cos(self.yaw)) * NAV_DT
            self.yaw += angular_y * NAV_DT

            self.terrain_y = terrain_height_at(self.x, self.z)
            self.y = self.terrain_y + INITIAL_CLEARANCE
            self.set_agent_pose()

            # Pump the tk event loop so the display doesn't freeze while driving
            if self.root is not None:
                self.root.update()

            if step % 10 == 0:
                print(
                    f"[nav] step={step:4d} pos=({self.x:.2f},{self.z:.2f}) "
                    f"dist={distance:.2f}m heading_err={np.rad2deg(heading_error):.1f}°"
                )

            if controller.at_target:
                print(
                    f"[nav] reached '{label}' in {step} steps, "
                    f"pos=({self.x:.2f},{self.z:.2f})"
                )
                break
        else:
            print(
                f"[nav] max steps ({NAV_MAX_STEPS}) reached before arriving at '{label}'"
            )

    def register_mission_semantics(self, goal_mesh, obstacle_meshes):
        """
        Load the resolved goal/obstacle meshes into the sim as semantic
        objects, replacing whatever the previous mission (if any) had
        registered. Purely a rendering/perception aid - see
        SemanticMeshRegistry for why this can't affect physics, collision,
        or navigation.
        """
        self.semantic_registry.clear()
        if goal_mesh is not None:
            self.semantic_registry.register(
                goal_mesh["mesh_path"], "goal", semantic_id_for_role("goal")
            )
        for mesh_meta in obstacle_meshes:
            self.semantic_registry.register(
                mesh_meta["mesh_path"],
                mesh_meta["role"],
                semantic_id_for_role(mesh_meta["role"]),
            )

    def capture_mission_semantics(self, frame_idx, pose):
        """
        Move to the pose the frame was originally captured at (a no-op
        during the live capture loop, since the agent hasn't moved since
        save_frame; required for standalone --vlm reruns, which start from
        a fresh sim at START_*) and re-render the semantic sensor now that
        this mission's goal/obstacle meshes are registered. Overwrites the
        semantic/goal_mask/obstacle_mask files save_frame wrote at capture
        time (all-environment, since no semantic objects existed yet) with
        the real render, and resyncs self.x/y/z/yaw so any navigation that
        follows starts from the frame's actual pose.
        """
        self.x, self.y, self.z, self.yaw = pose["x"], pose["y"], pose["z"], pose["yaw"]
        self.set_agent_pose()

        obs = self.sim.get_sensor_observations()
        semantic = semantic_from_obs(obs)

        semantic_path = f"{OUT_DIR}/semantic_{frame_idx:04d}.png"
        goal_mask_path = f"{OUT_DIR}/goal_mask_{frame_idx:04d}.png"
        obstacle_mask_path = f"{OUT_DIR}/obstacle_mask_{frame_idx:04d}.png"

        Image.fromarray(semantic).save(semantic_path)
        goal_mask, obstacle_mask = semantic_to_masks(semantic)
        Image.fromarray(goal_mask).save(goal_mask_path)
        Image.fromarray(obstacle_mask).save(obstacle_mask_path)

        print(f"[semantic] rendered {semantic_path}")
        print(f"[semantic] {goal_mask_path}")
        print(f"[semantic] {obstacle_mask_path}")

        return {
            "semantic_path": semantic_path,
            "goal_mask_path": goal_mask_path,
            "obstacle_mask_path": obstacle_mask_path,
        }

    def navigate_to_obstacles(self, obstacle_meshes):
        """
        Drive to each already-resolved obstacle mesh in turn. Takes
        pre-resolved metadata (from `resolve_mission_meshes`) rather than raw
        VLM entries, so position resolution and mesh extraction happen
        exactly once per obstacle regardless of caller.
        """
        for i, mesh_meta in enumerate(obstacle_meshes):
            obs_x, _, obs_z = mesh_meta["seed_world"]
            self.navigate_to_target(
                obs_x, obs_z, label=f"obstacle[{i}]:{mesh_meta['label']}"
            )

    def navigate_to_goal(self, goal_mesh):
        """Drive to the already-resolved goal mesh, then rotate in place to square up and face it."""
        goal_x, goal_y, goal_z = goal_mesh["seed_world"]
        label = goal_mesh["label"]
        self.navigate_to_target(goal_x, goal_z, label=f"goal:{label}")

        self.yaw = yaw_to_face(self.x, self.z, goal_x, goal_z)
        self.set_agent_pose()
        print(
            f"[nav] facing goal '{label}' at ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}), "
            f"yaw={np.rad2deg(self.yaw):.1f}°"
        )

        return goal_x, goal_y, goal_z, label

    def navigate_mission(
        self, frame_idx, response=None, goal_mesh=None, obstacle_meshes=None
    ):
        """
        Full mission for a queried frame: resolve the VLM's goal + obstacles
        (including local terrain meshes), drive to each obstacle first, then
        drive to the goal and stand facing it at the controller's standoff
        distance. Mission metadata (VLM reply, meshes, final target) is
        saved to rgb_{idx}_mission.json on completion.

        response/goal_mesh/obstacle_meshes may be passed in already resolved
        (on_spacebar does this via resolve_vlm_selection, to avoid re-parsing
        what it just wrote); when omitted, they're parsed/resolved fresh from
        the frame's saved rgb_*_vlm.txt, as the --goto CLI replay path does.
        """
        vlm_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_vlm.txt"
        if response is None:
            try:
                response = parse_vlm_response(vlm_path)
            except Exception as e:
                print(f"[error] could not parse VLM response {vlm_path}: {e}")
                return

        if goal_mesh is None and obstacle_meshes is None:
            goal_mesh, obstacle_meshes = resolve_mission_meshes(frame_idx, response)
        if goal_mesh is None:
            print(f"[error] no goal_object in {vlm_path}")
            return

        self.register_mission_semantics(goal_mesh, obstacle_meshes)
        pose = load_pose(frame_idx)
        semantic_paths = self.capture_mission_semantics(frame_idx, pose)

        print(f"[mission] {len(obstacle_meshes)} obstacle(s) to visit before the goal")

        self.navigate_to_obstacles(obstacle_meshes)

        try:
            goal_x, goal_y, goal_z, _ = self.navigate_to_goal(goal_mesh)
        except Exception as e:
            print(f"[error] could not reach goal position: {e}")
            return

        save_mission_metadata(
            frame_idx,
            response,
            goal_mesh,
            obstacle_meshes,
            (goal_x, goal_y, goal_z),
            semantic_paths,
        )
        print(f"[mission] complete")

    def close(self):
        """Cleanup."""
        try:
            self.sim.close()
        except Exception:
            pass

    def run(self):
        """Main interactive loop using tkinter for windowed display and input."""
        root = tk.Tk()
        root.title("VLM Nav Capture")
        root.geometry(f"{RGBD_RESOLUTION[1]}x{RGBD_RESOLUTION[0]}")
        self.root = root

        canvas = tk.Canvas(root, width=RGBD_RESOLUTION[1], height=RGBD_RESOLUTION[0])
        canvas.pack()

        info_label = tk.Label(
            root, text="SPACE: capture & annotate | Q: quit", bg="black", fg="white"
        )
        info_label.pack(side=tk.BOTTOM, fill=tk.X)

        running = True

        def update_frame():
            """Update display with current observation."""
            if not running:
                return

            rgb = self.get_observation()
            pil_image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_image)

            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo  # Keep a reference

            root.after(33, update_frame)

        def on_key_press(event):
            """Handle key press events."""
            nonlocal running
            if event.keysym == "space":
                self.on_spacebar()
            elif event.keysym == "q":
                running = False
                root.quit()

        root.bind("<KeyPress>", on_key_press)
        root.focus_set()

        try:
            update_frame()
            root.mainloop()
        finally:
            self.close()


def regenerate_overlays():
    """Rebuild rgb_*_at.png for every existing annotation, without touching the sim."""
    for json_path in sorted(Path(ANNOTATIONS_DIR).glob("rgb_*.json")):
        frame_idx = int(json_path.stem.split("_")[1])
        rgb_path = f"{OUT_DIR}/rgb_{frame_idx:04d}.png"
        out_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_at.png"

        if not os.path.exists(rgb_path):
            print(f"[warn] missing image for {json_path}, skipping")
            continue

        is_valid, status_msg = validate_annotation_json(str(json_path))
        if not is_valid:
            print(f"[warn] {json_path}: {status_msg}, skipping")
            continue

        draw_annotation_overlay(rgb_path, str(json_path), out_path)


def run_vlm_on_frame(frame_idx, prompt=VLM_PROMPT):
    """
    Query the VLM on an already-captured/annotated frame, then resolve the
    chosen goal/obstacles to world-space meshes, register them as semantic
    objects, re-render the semantic sensor from the frame's saved pose, and
    save the resulting goal/obstacle masks plus mission metadata - the same
    semantic pipeline the live interactive loop runs (minus navigation), so
    a --vlm rerun alone is enough to regenerate everything for a previously
    captured frame without launching a new interactive capture.
    """
    rgb_path = f"{OUT_DIR}/rgb_{frame_idx:04d}.png"
    annotation_path = f"{ANNOTATIONS_DIR}/rgb_{frame_idx:04d}.json"
    overlay_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_at.png"

    for path in (rgb_path, annotation_path):
        if not os.path.exists(path):
            print(f"[error] missing file: {path}")
            return

    is_valid, status_msg = validate_annotation_json(annotation_path)
    if not is_valid:
        print(f"[error] {annotation_path}: {status_msg}")
        return

    if not os.path.exists(overlay_path):
        draw_annotation_overlay(rgb_path, annotation_path, overlay_path)

    success, result, status_msg = resolve_vlm_selection(
        rgb_path, overlay_path, annotation_path, frame_idx, prompt
    )
    if not success:
        print(f"[error] {status_msg}")
        return
    response, goal_mesh, obstacle_meshes = result

    pose = load_pose(frame_idx)
    capture = InteractiveCapture()
    try:
        capture.register_mission_semantics(goal_mesh, obstacle_meshes)
        semantic_paths = capture.capture_mission_semantics(frame_idx, pose)
    finally:
        capture.close()

    save_mission_metadata(
        frame_idx,
        response,
        goal_mesh,
        obstacle_meshes,
        goal_mesh["seed_world"],
        semantic_paths,
    )


def run_face_on_frame(frame_idx):
    """Turn the agent to face the VLM-chosen object from an already-queried frame, then save a check frame."""
    vlm_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_vlm.txt"
    if not os.path.exists(vlm_path):
        print(f"[error] missing file: {vlm_path}")
        return

    capture = InteractiveCapture()
    try:
        capture.face_object(frame_idx)
        obs = capture.sim.get_sensor_observations()
        rgb, _ = rgb_depth_from_obs(obs)
        check_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_faced.png"
        Image.fromarray(rgb).save(check_path)
        print(f"[info] saved post-turn view: {check_path}")
    finally:
        capture.close()


def run_goto_on_frame(frame_idx):
    """Run the full obstacles-then-goal mission for an already-queried frame."""
    vlm_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_vlm.txt"
    if not os.path.exists(vlm_path):
        print(f"[error] missing file: {vlm_path}")
        return

    capture = InteractiveCapture()
    try:
        capture.navigate_mission(frame_idx)
    finally:
        capture.close()


def validate_semantic_masks(frame_idx):
    """
    Sanity-check a frame's semantic pipeline output: the goal and obstacle masks must be mutually exclusive (no pixel belongs to both), each mask's 255 pixels must exactly match the corresponding semantic id in the raw semantic render, and every pixel neither mask claims must be SEMANTIC_ID_ENVIRONMENT. This is the pixel-level demonstration that goal pixels only cover the selected goal region, obstacle pixels only cover the selected obstacle region(s), and everything else is environment.

    Returns:
        True if every check passes, False otherwise.
    """
    semantic_path = f"{OUT_DIR}/semantic_{frame_idx:04d}.png"
    goal_mask_path = f"{OUT_DIR}/goal_mask_{frame_idx:04d}.png"
    obstacle_mask_path = f"{OUT_DIR}/obstacle_mask_{frame_idx:04d}.png"

    for path in (semantic_path, goal_mask_path, obstacle_mask_path):
        if not os.path.exists(path):
            print(f"[error] missing file: {path}")
            return False

    semantic = np.array(Image.open(semantic_path))
    goal_mask = np.array(Image.open(goal_mask_path)) > 0
    obstacle_mask = np.array(Image.open(obstacle_mask_path)) > 0

    overlap = np.logical_and(goal_mask, obstacle_mask)
    goal_matches = np.array_equal(goal_mask, semantic == SEMANTIC_ID_GOAL)
    obstacle_matches = np.array_equal(obstacle_mask, semantic == SEMANTIC_ID_OBSTACLE)
    environment_matches = np.array_equal(
        semantic == SEMANTIC_ID_ENVIRONMENT,
        ~np.logical_or(goal_mask, obstacle_mask),
    )

    ok = not overlap.any() and goal_matches and obstacle_matches and environment_matches
    status = "PASS" if ok else "FAIL"
    print(
        f"[validate] frame {frame_idx}: {status} "
        f"(goal_px={int(goal_mask.sum())}, obstacle_px={int(obstacle_mask.sum())}, "
        f"overlap_px={int(overlap.sum())}, "
        f"goal_matches_render={goal_matches}, obstacle_matches_render={obstacle_matches}, "
        f"remainder_is_environment={environment_matches})"
    )
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--overlay":
        regenerate_overlays()
    elif len(sys.argv) > 1 and sys.argv[1] == "--vlm":
        if len(sys.argv) < 3:
            print("usage: python vlm_nav_interactive.py --vlm <frame_idx>")
            sys.exit(1)
        run_vlm_on_frame(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--face":
        if len(sys.argv) < 3:
            print("usage: python vlm_nav_interactive.py --face <frame_idx>")
            sys.exit(1)
        run_face_on_frame(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--goto":
        if len(sys.argv) < 3:
            print("usage: python vlm_nav_interactive.py --goto <frame_idx>")
            sys.exit(1)
        run_goto_on_frame(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--validate":
        if len(sys.argv) < 3:
            print("usage: python vlm_nav_interactive.py --validate <frame_idx>")
            sys.exit(1)
        ok = validate_semantic_masks(int(sys.argv[2]))
        sys.exit(0 if ok else 1)
    else:
        capture = InteractiveCapture()
        capture.run()
        print("[info] khel khatam.")
