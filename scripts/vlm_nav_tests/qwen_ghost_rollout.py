"""Shared ghost-goal helper for qwen_search_rollout.py and qwen_search_dino.py: turns a
body-frame (forward, left) offset from the agent's current pose into a world-space point that
can be fed straight into rollout_navdp_policy.project_goal_mask, the same way a manual search
direction ('left'/'right'/'straight'/'back') becomes a ghost goal to steer toward while the real
goal hasn't been found yet.

Coordinate and height conventions mirror the --goal-out-of-view placement in
rollout_navdp_policy.py's main() exactly (forward/left body vectors derived from yaw, terrain
height lookup + a fixed clearance) so a search ghost renders and drives identically to how the
real out-of-view goal would.
"""

from __future__ import annotations

import math

import numpy as np

# Matches rollout_navdp_policy.py's --goal-height default: how far above the local terrain
# height a goal marker sits, so a ghost goal looks and drives the same as a real placed goal.
GHOST_GOAL_HEIGHT = 1.2


def body_offset_to_world(position, yaw, fwd, left, terrain):
    """Body-frame (forward, left) offset from `position`/`yaw` -> world xyz point.

    Generalizes the --goal-out-of-view bearing/range placement in rollout_navdp_policy.py's
    main() to take a forward/left offset directly instead of bearing+range. `terrain` is a
    SceneMappedTerrain (or HeightmapGrid)-like object exposing `local_height_max(x, z, radius)`;
    radius=0 is a plain point lookup since a ghost goal isn't a physical footprint that needs a
    local-max clearance.
    """
    fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    gx = float(position[0]) + fwd * fwd_x + left * left_x
    gz = float(position[2]) + fwd * fwd_z + left * left_z
    gy = terrain.local_height_max(gx, gz, 0.0) + GHOST_GOAL_HEIGHT
    return np.asarray([gx, gy, gz], dtype=np.float32)
