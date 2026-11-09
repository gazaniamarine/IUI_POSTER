"""Policy-independent camera pose sampling for offline segmentation-dataset
capture (next.md Steps 3-6). Pose sampling only needs the scene bounds (see
sam_vla.env.terrain) -- no habitat_sim/GPU required, mirroring
rock_generation.py's split between pure generation and sim-attached
registration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from sam_vla.env.terrain import SIZE_X, SIZE_Z

SweepPose = Tuple[float, float, float]  # (x, z, yaw)


@dataclass
class PoseSweepConfig:
    mode: str = "grid"  # "grid" | "random"
    grid_spacing_m: float = 2.0
    num_yaws_per_cell: int = 4  # grid mode: viewpoint diversity per (x, z) cell
    num_random_poses: int = 500
    boundary_margin: float = 2.0  # keep every pose this far from the scene bound
    seed: int = 0


def _bounds(boundary_margin: float) -> Tuple[float, float]:
    half_x = SIZE_X / 2.0 - boundary_margin
    half_z = SIZE_Z / 2.0 - boundary_margin
    if half_x <= 0.0 or half_z <= 0.0:
        raise ValueError(
            f"boundary_margin ({boundary_margin}) leaves no room inside the "
            f"{SIZE_X}x{SIZE_Z}m scene"
        )
    return half_x, half_z


def sample_sweep_poses(config: PoseSweepConfig) -> List[SweepPose]:
    """Deterministic (seeded), policy-independent (x, z, yaw) samples covering
    the scene for dataset diversity -- the sole pose source for offline
    capture (no rollout-log replay). y is intentionally NOT computed here --
    MarsHabitatEnv.step() already recomputes it via get_height_at_xz
    (local-max + clearance) and ignores the incoming pose's y, so returning
    (x, z, yaw) and letting env.step() place the camera keeps one height
    convention instead of two. 'In-bounds' here means within
    boundary_margin of the scene edge, the same margin notion
    RockFieldConfig already uses -- there is no physics collision check
    available (physics is disabled scene-wide), so this is not a
    collision/raycast test, just bounds."""
    half_x, half_z = _bounds(config.boundary_margin)

    if config.mode == "grid":
        xs = _axis_positions(half_x, config.grid_spacing_m)
        zs = _axis_positions(half_z, config.grid_spacing_m)
        yaws = [
            2.0 * math.pi * i / config.num_yaws_per_cell
            for i in range(config.num_yaws_per_cell)
        ]
        return [(x, z, yaw) for x in xs for z in zs for yaw in yaws]

    if config.mode == "random":
        rng = random.Random(config.seed)
        return [
            (
                rng.uniform(-half_x, half_x),
                rng.uniform(-half_z, half_z),
                rng.uniform(0.0, 2.0 * math.pi),
            )
            for _ in range(config.num_random_poses)
        ]

    raise ValueError(
        f"unknown pose sweep mode {config.mode!r} (expected 'grid' or 'random')"
    )


def _axis_positions(half_extent: float, spacing: float) -> List[float]:
    n = max(int(math.floor(2.0 * half_extent / spacing)), 0)
    return [-half_extent + i * spacing for i in range(n + 1)]


def sample_in_place_sweep_yaws(num_yaws: int = 8, start_yaw: float = 0.0) -> List[float]:
    """`num_yaws` yaws (radians) evenly spaced around one full 360deg turn,
    starting at `start_yaw` -- the single-position counterpart to
    sample_sweep_poses' scene-wide (x, z, yaw) grid, for a caller that wants
    to rotate in place at one already-chosen (x, z) and capture every
    heading rather than move around the scene."""
    if num_yaws < 1:
        raise ValueError(f"num_yaws must be >= 1, got {num_yaws}")
    return [start_yaw + 2.0 * math.pi * i / num_yaws for i in range(num_yaws)]
