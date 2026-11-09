"""Terrain heights read from the scene GLB itself (a node named
`marsyard_terrain`), for yards like marsyard_with_rocks.glb that ship the real
ground mesh instead of a separate DEM heightmap.

Duck-types sam_vla.env.terrain.Terrain (`__call__(x, z)`, `local_height_max`)
so MarsHabitatEnv can swap it in. Parsing/rasterising is done by navdp_tube's
MarsScene (loaded by path; pure numpy/scipy/cv2). That scene is glTF Y-up with
the planner frame y = -Z, hence height(x, z) = scene.height_at(x, -z).

Habitat treats a bare .glb as Z-up and would stand the yard on its side, so
`write_stage_config` makes a stage config with up = +Y to use as scene_id.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HABITAT_DIR = Path(__file__).resolve().parent.parent.parent / "navdp_tube" / "bev_map" / "habitat"


def _mars_scene_module():
    for p in (str(_HABITAT_DIR), str(_HABITAT_DIR.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import mars_scene

    return mars_scene


def write_stage_config(glb: str, out_dir: str) -> str:
    """Stage config (up +Y, front -Z) for `glb`; returns its path (the scene_id)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, Path(glb).stem + ".stage_config.json")
    cfg = {"render_asset": os.path.abspath(glb), "up": [0, 1, 0], "front": [0, 0, -1],
           "origin": [0, 0, 0], "requires_lighting": False}
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class GlbTerrain:
    def __init__(self, glb: str):
        self.scene = _mars_scene_module().MarsScene(str(glb))
        lo_x, hi_x, lo_y, hi_y = self.scene.bounds
        self.x_range = (lo_x, hi_x)
        self.z_range = (-hi_y, -lo_y)

    def __call__(self, x: float, z: float) -> float:
        return float(self.scene.height_at(float(x), -float(z)))

    def local_height_max(self, x: float, z: float, radius: float, samples: int = 5) -> float:
        radius = max(float(radius), 0.0)
        if radius <= 1e-6 or samples <= 1:
            return self(x, z)
        d = np.linspace(-radius, radius, int(samples))
        dx, dz = np.meshgrid(d, d)
        keep = dx * dx + dz * dz <= radius * radius + 1e-8
        h = self.scene.height_at(float(x) + dx[keep], -(float(z) + dz[keep]))
        return float(np.max(h))
