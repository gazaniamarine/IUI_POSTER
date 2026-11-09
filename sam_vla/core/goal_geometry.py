"""Pinhole backprojection of the (image-space, first-frame-only) goal bbox into a
static world-frame goal position, so rollouts can log a per-frame distance-to-goal.

GoalSpec only ever carries a normalized bbox in the frame it was resolved from —
there is no separate goal actor in the sim to query a world position from. This
module anchors one by backprojecting the bbox center through the depth map of
that same frame, using the pinhole model and the yaw convention pose_integrator
already commits to (heading = (-sin_yaw, -cos_yaw) in the x-z plane --
habitat-sim's actual agent-forward direction, verified against
kb_teleop.py; see pose_integrator.py's docstring for the front/back
navigation bug this convention fixes).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from sam_vla.core.types import GoalSpec, Observation, Pose

GoalPosition = tuple[float, float, float]

# Semantic ids painted onto goal/obstacle mask meshes (see disc_mesh + the
# env's register_object_mask), so the semantic sensor can tell them apart.
MESH_GOAL_ID = 1
MESH_OBST_ID = 2

# Semantic id for procedurally generated rock obstacles (sam_vla.env.rock_generation),
# registered into the scene before goal/obstacle roles are assigned by SAM/VLM --
# distinct from MESH_GOAL_ID/MESH_OBST_ID, which mark the *chosen* goal/obstacles.
ROCK_SEMANTIC_ID = 3

# Semantic id for decorative flag markers (sam_vla.env.flag_placement) -- render-only,
# never a goal/obstacle candidate, distinct from ROCK_SEMANTIC_ID.
FLAG_SEMANTIC_ID = 4

# Semantic id for the ghost home-base marker (sam_vla.env.home_base) -- render-only,
# non-collidable, never a goal/obstacle candidate, distinct from FLAG_SEMANTIC_ID.
HOME_BASE_SEMANTIC_ID = 5


def intrinsics_from_hfov(height: int, width: int, hfov_deg: float) -> dict[str, float]:
    """Pinhole intrinsics (fx, fy, cx, cy) for a symmetric camera with the given
    horizontal FOV -- shared by every module that needs to unproject a pixel or
    mask through depth (belief_tracking.mask_to_body, the CBF's nearest-obstacle
    lookup, ...), so they all agree on the same camera model as bbox_to_world below.
    """
    fx = (width * 0.5) / max(math.tan(math.radians(float(hfov_deg)) * 0.5), 1e-6)
    return {"fx": fx, "fy": fx, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


def bbox_to_world(
    obs: Observation,
    bbox_norm: tuple[float, float, float, float],
    hfov_deg: float,
    pad_px: int = 6,
) -> GoalPosition | None:
    """Backproject a normalized bbox into a world-frame (x, y, z) point.

    Bearing/elevation come from the bbox's center pixel; range comes from the
    MEDIAN depth over the bbox's interior (expanded by `pad_px` on every
    side), mirroring rollout_navdp_policy's bbox_to_body robustness -- a
    single center pixel can land on a depth discontinuity (a rock's
    silhouette edge, a gap between the object and the background behind it)
    and seed a badly wrong range.

    The padding matters in practice: the current segmentation checkpoints
    (see CLAUDE.md's "Annotation masks are thin silhouette slivers" known
    issue) routinely predict masks -- and therefore bboxes -- just 1-3px
    tall, whose *unpadded* interior can land squarely on a depth seam even
    though it's centered on real rock (bbox size alone doesn't predict this;
    same-shaped slivers have been observed to both succeed and fail
    unpadded, see nav/rover_controller.py's goal-resolve depth-patch
    logging). Padding trades a little precision for a wider, more robust
    depth sample around that center.

    Returns None if depth is unavailable, or no pixel in the (padded) patch
    has valid depth (e.g. the bbox is entirely a sky/void hit).
    """
    if obs.depth is None:
        return None

    height, width = obs.depth.shape[:2]
    x0, y0, x1, y1 = bbox_norm
    fx0, fx1 = x0 * width, x1 * width
    fy0, fy1 = y0 * height, y1 * height
    ix0, ix1 = sorted(
        (min(max(int(fx0), 0), width - 1), min(max(int(fx1), 0), width - 1))
    )
    iy0, iy1 = sorted(
        (min(max(int(fy0), 0), height - 1), min(max(int(fy1), 0), height - 1))
    )
    ix0, ix1 = max(ix0 - pad_px, 0), min(ix1 + pad_px, width - 1)
    iy0, iy1 = max(iy0 - pad_px, 0), min(iy1 + pad_px, height - 1)

    patch = np.asarray(obs.depth)[iy0 : iy1 + 1, ix0 : ix1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid))

    # Bearing/elevation from the bbox's continuous center (not the rounded patch
    # bounds above) -- mirrors bbox_to_body's `u = 0.5 * (x1 + x2)`.
    px = 0.5 * (fx0 + fx1)
    py = 0.5 * (fy0 + fy1)

    # Depth sensor reports z-depth (perpendicular to the image plane), which is
    # exactly what a pinhole model expects as z_cam.
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    x_cam = (px - width / 2.0) * depth_m / fx
    y_cam = (py - height / 2.0) * depth_m / fy
    z_cam = depth_m

    pose = obs.pose
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    world_x = pose.x - z_cam * sin_yaw + x_cam * cos_yaw
    world_z = pose.z - z_cam * cos_yaw - x_cam * sin_yaw
    world_y = pose.y - y_cam

    return (world_x, world_y, world_z)


def pixel_to_world(
    obs: Observation,
    u_norm: float,
    v_norm: float,
    hfov_deg: float,
    pad_px: int = 6,
) -> GoalPosition | None:
    """Inverse of project_world_to_pixel: backproject a normalized (u, v) point
    (0..1, top-left origin -- qwen_prompts.build_ground_object_prompt's
    convention) into a world-frame (x, y, z) point, given `obs`'s depth frame
    and pose. Same median-depth-over-a-padded-window sampling as
    bbox_to_world (a single center pixel can land on a depth discontinuity),
    just centered on one point instead of a bbox interior -- this is that
    same robustness argument applied to a VLM-predicted point goal (see
    nav.vlm_point_navigator) instead of a detector bbox.

    Returns None if depth is unavailable, or no pixel in the padded patch
    around (u, v) has valid depth.
    """
    if obs.depth is None:
        return None

    height, width = obs.depth.shape[:2]
    px = float(u_norm) * width
    py = float(v_norm) * height
    ix = min(max(int(px), 0), width - 1)
    iy = min(max(int(py), 0), height - 1)
    ix0, ix1 = max(ix - pad_px, 0), min(ix + pad_px, width - 1)
    iy0, iy1 = max(iy - pad_px, 0), min(iy + pad_px, height - 1)

    patch = np.asarray(obs.depth)[iy0 : iy1 + 1, ix0 : ix1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid))

    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    x_cam = (px - width / 2.0) * depth_m / fx
    y_cam = (py - height / 2.0) * depth_m / fy
    z_cam = depth_m

    pose = obs.pose
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    world_x = pose.x - z_cam * sin_yaw + x_cam * cos_yaw
    world_z = pose.z - z_cam * cos_yaw - x_cam * sin_yaw
    world_y = pose.y - y_cam

    return (world_x, world_y, world_z)


def project_world_to_pixel(
    pose: Pose,
    world_xyz: GoalPosition,
    hfov_deg: float,
    width: int,
    height: int,
) -> Optional[tuple[int, int]]:
    """Inverse of bbox_to_world: project a world-frame (x, y, z) point into
    pixel coords for `pose`'s camera, or None if it's behind the camera or
    outside the frame. Lets a caller keep a *fixed* world point (e.g. a
    click-selected goal) visually anchored in the live camera view every
    frame via reprojection, instead of relying on a rendered scene marker --
    dynamically registered objects (register_object_mask and friends) render
    zero pixels on this machine, see CLAUDE.md's dynamic-object-render-bug
    note, so a projected overlay is the only reliable way to mark a fixed 3D
    point in the environment view.
    """
    wx, wy, wz = world_xyz
    dx = wx - pose.x
    dz = wz - pose.z
    cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
    z_cam = -dx * sin_yaw - dz * cos_yaw
    if z_cam <= 1e-3:
        return None
    x_cam = dx * cos_yaw - dz * sin_yaw
    y_cam = pose.y - wy

    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    px = x_cam * fx / z_cam + width / 2.0
    py = y_cam * fy / z_cam + height / 2.0
    if not (0 <= px < width and 0 <= py < height):
        return None
    return (int(round(px)), int(round(py)))


def project_world_point_with_pixel_radius(
    pose: Pose,
    world_xyz: GoalPosition,
    world_radius_m: float,
    hfov_deg: float,
    width: int,
    height: int,
) -> Optional[tuple[int, int, float]]:
    """Same projection as project_world_to_pixel, plus the on-screen pixel
    radius a sphere of `world_radius_m` centered at world_xyz would project
    to at this depth (pinhole small-object approximation: radius_px = fx *
    world_radius_m / z_cam) -- so a caller drawing a ghost mask (see
    sam_vla.core.ghost_mask.draw_ghost_mask) at a fixed real-world radius
    gets a circle whose apparent size tracks depth the way a real object's
    would, instead of a constant pixel radius that reads too big up close or
    too small far away. Returns None under the same out-of-frame/
    behind-camera conditions as project_world_to_pixel."""
    wx, wy, wz = world_xyz
    dx = wx - pose.x
    dz = wz - pose.z
    cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
    z_cam = -dx * sin_yaw - dz * cos_yaw
    if z_cam <= 1e-3:
        return None
    x_cam = dx * cos_yaw - dz * sin_yaw
    y_cam = pose.y - wy

    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    px = x_cam * fx / z_cam + width / 2.0
    py = y_cam * fy / z_cam + height / 2.0
    if not (0 <= px < width and 0 <= py < height):
        return None
    radius_px = fx * float(world_radius_m) / z_cam
    return (int(round(px)), int(round(py)), radius_px)


def backproject_goal_position(
    obs: Observation, goal_spec: GoalSpec, hfov_deg: float
) -> GoalPosition | None:
    """Backproject the goal bbox into a world-frame (x, y, z) point."""
    return bbox_to_world(obs, goal_spec.goal_bbox_norm, hfov_deg)


def disc_mesh(
    center: GoalPosition, radius: float, segments: int = 16, lift: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Flat circular fan of triangles in the x-z plane, centered at `center`
    and lifted slightly above it so the render-only mesh doesn't z-fight
    whatever surface it's marking. Output shape (verts, faces) mirrors
    rollout_navdp_policy's depth_patch_mesh, so it can be handed to the same
    _save_obj / register_semantic_mesh path to become a goal/obstacle mask.
    """
    cx, cy, cz = center
    cy = float(cy) + float(lift)
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    ring = [(cx + radius * math.cos(a), cy, cz + radius * math.sin(a)) for a in angles]
    verts = np.asarray([(cx, cy, cz)] + ring, dtype=np.float64)
    n = len(ring)
    faces = np.asarray([(0, i + 1, (i + 1) % n + 1) for i in range(n)], dtype=np.int64)
    return verts, faces


def terrain_patch_mesh(
    center: GoalPosition,
    radius: float,
    terrain_height_at: Callable[[float, float], float],
    resolution: float = 0.03,
    lift: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Terrain-following circular patch of triangles, resampled from the heightmap on
    an (x, z) grid at `resolution` spacing -- each vertex's y comes from
    `terrain_height_at`, so the mesh follows the ground under it instead of sitting on
    one flat plane. Ported from vlm_nav_interactive.make_local_height_patch_mesh, which
    rollout_navdp_policy's --goal-from-vlm path uses to mark VLM-selected goal/obstacle
    objects; disc_mesh's constant-y disc plays the same role here but comes out visibly
    flat -- and floating off the surface wherever the yard's terrain isn't level under
    `center` -- which is what register_object_mask was using before this replaced it.
    `center`'s own y is ignored (only x, z anchor the patch); grid cells are only
    triangulated when all four corners fall inside the disc, so the boundary comes out
    a (stairstepped) circle rather than a square, matching disc_mesh's footprint.
    """
    cx, _cy, cz = center
    n = max(int(math.ceil(float(radius) / float(resolution))), 1)
    offsets = np.arange(-n, n + 1) * float(resolution)

    index_grid = -np.ones((len(offsets), len(offsets)), dtype=int)
    verts = []
    for j, dz in enumerate(offsets):
        for i, dx in enumerate(offsets):
            if dx * dx + dz * dz > radius * radius:
                continue
            wx = cx + dx
            wz = cz + dz
            wy = float(terrain_height_at(wx, wz)) + float(lift)
            index_grid[j, i] = len(verts)
            verts.append((wx, wy, wz))

    faces = []
    for j in range(len(offsets) - 1):
        for i in range(len(offsets) - 1):
            v00, v10 = index_grid[j, i], index_grid[j, i + 1]
            v01, v11 = index_grid[j + 1, i], index_grid[j + 1, i + 1]
            if v00 < 0 or v10 < 0 or v01 < 0 or v11 < 0:
                continue
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))

    verts_arr = (
        np.asarray(verts, dtype=np.float64)
        if verts
        else np.empty((0, 3), dtype=np.float64)
    )
    faces_arr = (
        np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
    )
    return verts_arr, faces_arr


def distance_to_goal(pose: Pose, goal_position: GoalPosition) -> float:
    gx, gy, gz = goal_position
    return math.sqrt((pose.x - gx) ** 2 + (pose.y - gy) ** 2 + (pose.z - gz) ** 2)


def goal_pixel_center(
    goal_bbox_norm: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int]:
    """Pixel coords of the (first-frame-resolved) goal bbox's center, scaled to
    the given frame's resolution."""
    x0, y0, x1, y1 = goal_bbox_norm
    return (int(0.5 * (x0 + x1) * width), int(0.5 * (y0 + y1) * height))


def mask_pixel_center(mask: np.ndarray) -> tuple[int, int] | None:
    """Pixel coords of a live rendered mask's centroid (mirrors mask_to_body's
    `u = xs.mean()` bearing pixel), or None if the mask is empty this frame."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


if __name__ == "__main__":
    import numpy as np

    depth = np.full((4, 4), 5.0, dtype=np.float32)
    obs = Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        frame_idx=0,
    )
    goal_spec = GoalSpec(
        goal_bbox_norm=(0.4, 0.4, 0.6, 0.6),
        obstacle_bboxes_norm=[],
        instruction_text="Navigate to the rock target.",
    )

    goal_position = backproject_goal_position(obs, goal_spec, hfov_deg=90.0)
    print("goal_position:", goal_position)
    assert goal_position is not None

    dist = distance_to_goal(obs.pose, goal_position)
    print("distance_to_goal:", dist)
    assert math.isclose(dist, 5.0, rel_tol=1e-3)
    print("OK")
