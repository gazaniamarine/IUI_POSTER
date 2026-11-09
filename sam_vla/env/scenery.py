"""Static scenery: a .glb model (e.g. the 25042_Perseverance rover) placed at a
fixed world (x, z) on the terrain. Render-only and non-collidable, same as
the flags and home base (sim_utils.register_glb_object) -- the rover's depth
camera still sees it, so NavDP/CBF steer around it, but it has no physics
body.

The Perseverance GLBs in assets/ are authored at true 1:1 meter scale
(2.7 x 2.3 x 4.5 m, measured in habitat-sim) with their origin at the
bounding-box centre, so the object is lifted by half its height (1.151 m x
scale) to sit on the ground instead of half-buried.
"""

from __future__ import annotations

from sam_vla.env.sim_utils import register_glb_object

PERSEVERANCE_HALF_HEIGHT_M = 1.151
# Not a goal/obstacle candidate -- same "decoration" semantic id 0 as unlabeled.
SCENERY_SEMANTIC_ID = 0


def register_scenery(
    sim,
    terrain,
    glb: str,
    x: float,
    z: float,
    yaw: float = 0.0,
    scale: float = 1.0,
    half_height: float = PERSEVERANCE_HALF_HEIGHT_M,
    footprint_radius: float = 1.5,
):
    ground = terrain.local_height_max(x, z, footprint_radius)
    y = ground + half_height * scale
    return register_glb_object(
        sim,
        str(glb),
        position=(x, y, z),
        yaw=yaw,
        semantic_id=SCENERY_SEMANTIC_ID,
        template_name=f"scenery_{abs(hash((glb, x, z))) % 10**8}",
        scale=scale,
    )
