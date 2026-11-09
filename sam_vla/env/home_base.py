"""Ghost home-base marker: the assets/base_station_antenna.glb model placed a
fixed distance directly behind the rover's spawn pose -- a visual anchor for
"home" (nav/rover_controller.py's go_home already drives back to the spawn
(x, z); this only adds something to see there), not a goal/obstacle
candidate (see HOME_BASE_SEMANTIC_ID in sam_vla.core.goal_geometry).

"Ghost" because, like sam_vla.env.flag_placement's markers, it's render-only
and non-collidable -- the rover can drive straight through it. Registered via
sim_utils.register_glb_object the same way flag_placement places its shared
.glb assets -- no per-instance mesh to bake, position/yaw set explicitly
post-registration.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

from sam_vla.core.goal_geometry import HOME_BASE_SEMANTIC_ID
from sam_vla.env.sim_utils import register_glb_object

HOME_BASE_DISTANCE_M = 2.0
HOME_BASE_FOOTPRINT_RADIUS_M = 1.0
HOME_BASE_SCALE = 0.35
BASE_STATION_GLB = (
    Path(__file__).resolve().parent.parent.parent
    / "assets"
    / "base_station_antenna.glb"
)


def home_base_xz(
    start_x: float,
    start_z: float,
    start_yaw: float,
    distance: float = HOME_BASE_DISTANCE_M,
) -> Tuple[float, float]:
    """World (x, z) `distance` meters directly behind the rover's spawn pose.

    Uses pose_integrator's heading convention (forward = (-sin(yaw),
    -cos(yaw)) in the x-z plane) -- behind is that vector's negation."""
    return (
        start_x + distance * math.sin(start_yaw),
        start_z + distance * math.cos(start_yaw),
    )


def register_home_base(
    sim,
    terrain,
    start_x: float,
    start_z: float,
    start_yaw: float,
    distance: float = HOME_BASE_DISTANCE_M,
    scale: float = HOME_BASE_SCALE,
):
    """Register the base_station_antenna.glb model `distance` meters behind
    (start_x, start_z, start_yaw), sitting on the terrain height under it,
    facing the same heading as the rover's spawn yaw, scaled by `scale`
    (uniform multiplier on the .glb's native size -- see HOME_BASE_SCALE).
    Returns the registered rigid object."""
    x, z = home_base_xz(start_x, start_z, start_yaw, distance)
    y = terrain.local_height_max(x, z, HOME_BASE_FOOTPRINT_RADIUS_M)
    return register_glb_object(
        sim,
        str(BASE_STATION_GLB),
        position=(x, y, z),
        yaw=start_yaw,
        semantic_id=HOME_BASE_SEMANTIC_ID,
        template_name="home_base_antenna",
        scale=scale,
    )
