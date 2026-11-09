"""Procedural rock-field generation for the Mars habitat scene.

Rocks are generated once from a seed, written out as real mesh files (one
.obj per rock, already baked into its final world position/orientation) plus
a JSON manifest, and can then be *loaded* back byte-identical for every
ablation run instead of being regenerated -- the whole point being that the
obstacle layout must stay fixed while other things (steering mode, CBF on/off,
goal mode, ...) vary.

Generation only needs a `Terrain` height sampler (see sam_vla.env.terrain) --
no habitat_sim / GPU required. Registering the generated meshes into a live
sim is a separate step (`register_rocks`), called from MarsHabitatEnv once a
Simulator exists.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sam_vla.core.goal_geometry import ROCK_SEMANTIC_ID
from sam_vla.env.sim_utils import save_obj
from sam_vla.env.terrain import SIZE_X, SIZE_Z, Terrain

ExcludeZone = Tuple[float, float, float]  # (x, z, radius)


@dataclass
class RockFieldConfig:
    seed: int = 0
    num_rocks: int = 12
    radius_min: float = 0.25
    radius_max: float = 0.7
    min_spacing: float = 0.4  # extra clearance kept between rock surfaces
    boundary_margin: float = 2.0  # keep every rock's edge this far from the scene bound
    exclude_zones: List[ExcludeZone] = field(
        default_factory=list
    )  # e.g. spawn keep-out
    max_attempts_per_rock: int = 300


@dataclass
class RockSpec:
    id: int
    x: float
    y: float
    z: float
    yaw: float
    radius: float
    mesh_path: str


def _make_rock_mesh(
    rng: random.Random,
    radius: float,
    segments: int = 10,
    rings: int = 6,
    jitter: float = 0.28,
) -> Tuple[np.ndarray, np.ndarray]:
    """A low-poly, irregular rock: a UV-sphere with per-vertex radial jitter and a
    vertical squash, centered at the origin with its lowest vertex at y=0 -- so
    translating it to (x, terrain_height, z) sits it flush on the ground."""
    squash = rng.uniform(0.45, 0.65)
    verts = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings  # 0 (top) .. pi (bottom)
        ring_r = math.sin(phi)
        y = math.cos(phi)
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            jitter_scale = 1.0 + rng.uniform(-jitter, jitter)
            verts.append(
                (
                    ring_r * math.cos(theta) * jitter_scale,
                    y * jitter_scale,
                    ring_r * math.sin(theta) * jitter_scale,
                )
            )
    verts = np.asarray(verts, dtype=np.float64)
    verts[:, 1] *= squash
    verts *= radius
    verts[:, 1] -= verts[:, 1].min()

    faces = []
    for ring in range(rings):
        for seg in range(segments):
            a = ring * segments + seg
            b = ring * segments + (seg + 1) % segments
            c = (ring + 1) * segments + seg
            d = (ring + 1) * segments + (seg + 1) % segments
            faces.append((a, c, d))
            faces.append((a, d, b))
    return verts, np.asarray(faces, dtype=np.int64)


_ROCK_BASE_RGB = (0.9411, 0.8196, 0.72549)
_ROCK_COLOR_JITTER = 0.0  # per-channel +/- jitter so rocks aren't a flat, uniform color


def _rock_diffuse_color(rng: random.Random) -> Tuple[float, float, float]:
    return tuple(
        min(1.0, max(0.0, c + rng.uniform(-_ROCK_COLOR_JITTER, _ROCK_COLOR_JITTER)))
        for c in _ROCK_BASE_RGB
    )


def _too_close(
    x: float,
    z: float,
    r: float,
    placed: Sequence[Tuple[float, float, float]],
    min_spacing: float,
) -> bool:
    for px, pz, pr in placed:
        if math.hypot(x - px, z - pz) < (r + pr + min_spacing):
            return True
    return False


def _in_exclude_zone(
    x: float, z: float, r: float, exclude_zones: Sequence[ExcludeZone]
) -> bool:
    for ex, ez, er in exclude_zones:
        if math.hypot(x - ex, z - ez) < (er + r):
            return True
    return False


def generate_rock_field(
    config: RockFieldConfig, terrain: Terrain, out_dir: Path
) -> List[RockSpec]:
    """Rejection-sample `config.num_rocks` non-overlapping rocks within the scene
    bounds, each dropped onto the terrain height under it, and write each one's
    already-placed mesh to `out_dir/rocks/rock_NNN.obj`. Deterministic given
    `config.seed` -- same config + same terrain => same rocks, every call."""
    rng = random.Random(config.seed)
    mesh_dir = Path(out_dir) / "rocks"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    placed: List[Tuple[float, float, float]] = []  # (x, z, radius) of accepted rocks
    rocks: List[RockSpec] = []

    for i in range(config.num_rocks):
        placement = None
        for _attempt in range(config.max_attempts_per_rock):
            radius = rng.uniform(config.radius_min, config.radius_max)
            half_x = SIZE_X / 2.0 - config.boundary_margin - radius
            half_z = SIZE_Z / 2.0 - config.boundary_margin - radius
            if half_x <= 0.0 or half_z <= 0.0:
                raise ValueError(
                    f"boundary_margin ({config.boundary_margin}) + radius ({radius:.2f}) leaves no "
                    f"room inside the {SIZE_X}x{SIZE_Z}m scene"
                )
            x = rng.uniform(-half_x, half_x)
            z = rng.uniform(-half_z, half_z)
            if _in_exclude_zone(x, z, radius, config.exclude_zones):
                continue
            if _too_close(x, z, radius, placed, config.min_spacing):
                continue
            placement = (x, z, radius)
            break

        if placement is None:
            print(
                f"[rocks] WARN could not place rock {i} after {config.max_attempts_per_rock} attempts; skipping"
            )
            continue

        x, z, radius = placement
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        y = terrain.local_height_max(x, z, radius)

        verts, faces = _make_rock_mesh(rng, radius)
        c, s = math.cos(yaw), math.sin(yaw)
        world = verts.copy()
        world[:, 0] = verts[:, 0] * c - verts[:, 2] * s
        world[:, 2] = verts[:, 0] * s + verts[:, 2] * c
        world[:, 0] += x
        world[:, 1] += y
        world[:, 2] += z

        mesh_path = mesh_dir / f"rock_{i:03d}.obj"
        save_obj(str(mesh_path), world, faces, diffuse_rgb=_rock_diffuse_color(rng))

        placed.append((x, z, radius))
        rocks.append(
            RockSpec(
                id=i, x=x, y=y, z=z, yaw=yaw, radius=radius, mesh_path=str(mesh_path)
            )
        )

    return rocks


def save_rock_field(
    rocks: Sequence[RockSpec], config: RockFieldConfig, path: Path
) -> None:
    payload = {"config": asdict(config), "rocks": [asdict(r) for r in rocks]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def load_rock_field(path: Path) -> Tuple[List[RockSpec], RockFieldConfig]:
    payload = json.loads(Path(path).read_text())
    config = RockFieldConfig(**payload["config"])
    rocks = [RockSpec(**r) for r in payload["rocks"]]
    for rock in rocks:
        if not Path(rock.mesh_path).exists():
            raise FileNotFoundError(
                f"rock field manifest {path} references missing mesh {rock.mesh_path}"
            )
    return rocks, config


def register_rocks(
    sim, rocks: Sequence[RockSpec], semantic_id: int = ROCK_SEMANTIC_ID
) -> List:
    """Add every rock's already-placed mesh into the sim as a render-only,
    non-collidable object (physics is disabled scene-wide; obstacle avoidance
    reads the semantic mask + depth, not physics collisions). All rocks share
    `semantic_id` at registration time -- goal/obstacle roles among them are
    assigned later from SAM detections on the RGB frame, not baked in here."""
    from sam_vla.env.sim_utils import register_semantic_mesh

    return [register_semantic_mesh(sim, rock.mesh_path, semantic_id) for rock in rocks]


if __name__ == "__main__":
    import argparse

    from sam_vla.env.terrain import HeightmapGrid

    HERE = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(
        description="Generate a seeded, non-overlapping rock field on the Mars terrain."
    )
    ap.add_argument(
        "--heightmap", default=str(HERE / "marsyard2022_terrain_hm_1025.tif")
    )
    ap.add_argument("--out-dir", default=str(HERE / "rock_envs" / "default"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-rocks", type=int, default=12)
    ap.add_argument("--radius-min", type=float, default=0.25)
    ap.add_argument("--radius-max", type=float, default=0.7)
    ap.add_argument("--min-spacing", type=float, default=0.4)
    ap.add_argument("--boundary-margin", type=float, default=2.0)
    args = ap.parse_args()

    grid = HeightmapGrid(Path(args.heightmap))
    terrain = Terrain(grid)
    cfg = RockFieldConfig(
        seed=args.seed,
        num_rocks=args.num_rocks,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        min_spacing=args.min_spacing,
        boundary_margin=args.boundary_margin,
    )
    out_dir = Path(args.out_dir)
    rocks = generate_rock_field(cfg, terrain, out_dir)
    save_rock_field(rocks, cfg, out_dir / "rock_field.json")
    print(f"placed {len(rocks)}/{cfg.num_rocks} rocks -> {out_dir / 'rock_field.json'}")
