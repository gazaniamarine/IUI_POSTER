"""Low-level habitat_sim helpers: sensor specs, agent pose, RGB-D extraction,
and render-only semantic meshes (used to paint goal/obstacle masks that the
semantic sensor picks up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import magnum as mn
import numpy as np
import quaternion
import habitat_sim

from sam_vla.core.types import Pose


def make_sensor(uuid: str, sensor_type, height: int, width: int, hfov_deg: float):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [int(height), int(width)]
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = float(hfov_deg)
    return spec


def set_agent_pose(agent, x: float, y: float, z: float, yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray([x, y, z], dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    agent.set_state(state)


# -pi/2 about world +X pitches the agent's default forward (-Z) to straight
# down (-Y) -- the standard habitat-sim trick for a bird's-eye camera. See
# set_agent_pose_topdown.
_TOPDOWN_ROTATION_VECTOR = [-1.5707963267948966, 0.0, 0.0]


def set_agent_pose_topdown(agent, x: float, y: float, z: float) -> None:
    """Point a free-standing (non-driving) agent's camera straight down at
    (x, z) from height y, for a fixed top-down viz camera -- unlike
    set_agent_pose, there is no yaw: this agent never steers, so only its
    world position and downward pitch matter."""
    state = agent.get_state()
    state.position = np.asarray([x, y, z], dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(_TOPDOWN_ROTATION_VECTOR)
    agent.set_state(state)


def rgb_depth(obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(obs["rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb.astype(np.uint8), depth.astype(np.float32)


def save_obj(
    path: str,
    verts: np.ndarray,
    faces: np.ndarray,
    diffuse_rgb: Optional[Tuple[float, float, float]] = None,
    alpha: float = 1.0,
) -> None:
    """Write an .obj mesh. If `diffuse_rgb` is given, also write a companion .mtl
    with that diffuse color and zero specular (matte, non-shiny) -- the same
    `Ks 0.000 0.000 0.000` convention the marsyard ground material uses -- and
    reference it from the .obj so importers pick it up instead of falling back
    to a shiny default gray material. `alpha` (1.0 = fully opaque) is written
    as both `d` and `Tr` (dissolve/its inverse -- different importers honor
    one or the other) for whatever picks up the .mtl; unverified whether this
    repo's habitat-sim/GPU combination actually alpha-blends it rather than
    just rendering opaque (see CLAUDE.md's "Known issues" for other render
    quirks specific to this setup) -- worth a visual spot-check."""
    path = Path(path)
    with open(path, "w") as f:
        if diffuse_rgb is not None:
            mtl_name = path.with_suffix(".mtl").name
            f.write(f"mtllib {mtl_name}\n")
            f.write("usemtl mesh_mat\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")

    if diffuse_rgb is not None:
        r, g, b = diffuse_rgb
        mtl_path = path.with_suffix(".mtl")
        with open(mtl_path, "w") as f:
            f.write("newmtl mesh_mat\n")
            f.write(f"Ka {r:.3f} {g:.3f} {b:.3f}\n")
            f.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("Ns 1.0\n")
            f.write(f"d {alpha:.3f}\n")
            f.write(f"Tr {1.0 - alpha:.3f}\n")
            f.write("illum 1\n")


def distance_to_goal(pose: Pose, goal_position: Tuple[float, float, float]) -> float:
    """Planar (x-z) distance from `pose` to `goal_position`, ignoring elevation --
    mirrors rollout_navdp_policy's `goal_dist_now` (terrain height makes the y
    component noisy/uninformative for a ground rover)."""
    gx, _gy, gz = goal_position
    return float(
        np.linalg.norm(np.asarray([pose.x - gx, pose.z - gz], dtype=np.float32))
    )


def register_semantic_mesh(
    sim, mesh_path: str, semantic_id: int, y_offset: float = 0.0
):
    """Add a render-only (kinematic, non-collidable) mesh carrying a semantic
    id, so the semantic sensor renders it as a distinct mask.

    `y_offset` lifts the object straight up in world Y after it's added
    (default 0, no change). These meshes are terrain-following patches whose
    vertices already sit exactly at the heightmap's sampled ground height
    (see mesh_annotation_tool.compute_tight_boundary_mesh / sim_utils'
    terrain_patch_mesh callers) -- coplanar with the real render terrain, a
    classic z-fighting setup where the semantic id randomly loses the depth
    test almost everywhere it should render. A small uniform lift (e.g.
    0.05m) is cheap insurance: since every vertex already encodes
    terrain_height(x, z), translating the whole object gives
    terrain_height(x, z) + y_offset for free, no per-vertex resampling
    needed."""
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    template = otm.create_new_template(mesh_path)
    template.render_asset_handle = mesh_path
    template.collision_asset_handle = mesh_path
    template.is_collidable = False
    template_id = otm.register_template(
        template, f"sem_{semantic_id}_{Path(mesh_path).name}"
    )
    obj = rom.add_object_by_template_handle(otm.get_template_handle_by_id(template_id))
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.collidable = False
    obj.semantic_id = int(semantic_id)
    if y_offset:
        obj.translation = obj.translation + mn.Vector3(0.0, float(y_offset), 0.0)
    return obj


def register_glb_object(
    sim,
    glb_path: str,
    position: Tuple[float, float, float],
    yaw: float = 0.0,
    semantic_id: int = 0,
    template_name: Optional[str] = None,
    scale: float = 1.0,
):
    """Add a fixed-geometry .glb asset (e.g. a flag marker) as a render-only,
    non-collidable kinematic object at an explicit world (x, y, z) / yaw.

    Unlike register_semantic_mesh's rocks/annotation hulls -- each a unique
    .obj already baked into its final world position/orientation -- the same
    .glb geometry here is shared across instances (e.g. every red flag reuses
    red_flag.glb), so position/rotation are set explicitly post-registration
    instead of being pre-baked into the mesh, and each call needs a unique
    template_name (defaults to one derived from the object id) so repeated
    placements of the same asset don't collide in the template registry.

    `scale` is a uniform multiplier on the asset's native size (1.0 = as
    authored in the .glb), set on the template before registration since
    ManagedRigidObject has no post-creation scale setter."""
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    template = otm.create_new_template(str(glb_path))
    template.render_asset_handle = str(glb_path)
    template.collision_asset_handle = str(glb_path)
    template.is_collidable = False
    template.scale = mn.Vector3(float(scale), float(scale), float(scale))
    name = template_name or f"glb_{semantic_id}_{Path(glb_path).stem}_{id(template)}"
    template_id = otm.register_template(template, name)
    obj = rom.add_object_by_template_handle(otm.get_template_handle_by_id(template_id))
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.collidable = False
    obj.semantic_id = int(semantic_id)
    x, y, z = position
    obj.translation = mn.Vector3(float(x), float(y), float(z))
    obj.rotation = mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3.y_axis())
    return obj


# Far outside DEPTH_MAX_RANGE_M (30m) or any sensor's far clip, so an object
# translated down by this much is guaranteed invisible to every camera
# regardless of pose.
ANNOTATION_HIDE_DROP_M = 10_000.0


def set_objects_hidden(
    objects: Dict[int, Any], onstage: Dict[int, mn.Vector3], hidden: bool
) -> None:
    """Toggle a set of registered objects between their real ("onstage")
    world position and a position dropped ANNOTATION_HIDE_DROP_M straight
    down, so a render pass can exclude them without the per-frame cost of
    removing/re-adding them (habitat_sim has no per-sensor visibility flag).
    Always sets translation from the cached `onstage` value rather than
    applying a relative delta, so repeated toggles across a long capture
    run can't drift. `objects`/`onstage` are keyed the same (e.g. mesh_id)."""
    for key, obj in objects.items():
        base = onstage[key]
        obj.translation = (
            base - mn.Vector3(0.0, ANNOTATION_HIDE_DROP_M, 0.0) if hidden else base
        )
