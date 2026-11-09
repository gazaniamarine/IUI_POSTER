"""Generate a reusable rock-field environment on top of the Mars habitat scene.

Flow (next.md #1/#2):
  1. Load the environment from its glb file (sanity-checked via a throwaway sim).
  2. Generate N random, non-overlapping rocks from a seed (size range, spacing,
     boundary margin, keep-out zones are all configurable).
  3. Save the environment (rock meshes + a JSON manifest of positions/orientations)
     to --out-dir.
  4. Later, experiment scripts (e.g. run_navdp_rollout.py) load this same
     manifest via MarsHabitatEnv(rock_field_path=...) instead of regenerating
     it, so the obstacle layout stays fixed across ablation runs.

Usage:
    python scripts/utils/generate_rock_env.py --out-dir rock_envs/run1 --num-rocks 15 --seed 7
    python scripts/utils/generate_rock_env.py --out-dir rock_envs/run1 --verify   # re-load + open in habitat_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sam_vla.env.rock_generation import (
    RockFieldConfig,
    generate_rock_field,
    load_rock_field,
    register_rocks,
    save_rock_field,
)
from sam_vla.env.terrain import HeightmapGrid, Terrain

HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "assets" / "marsyard2022.glb"
DEFAULT_HEIGHTMAP = HERE / "marsyard2022_terrain_hm.png"


def _parse_zone(raw: str) -> tuple[float, float, float]:
    x, z, r = (float(v) for v in raw.split(","))
    return (x, z, r)


def _check_scene_loads(scene: Path) -> None:
    """Open the glb in a throwaway habitat_sim.Simulator to confirm it's a valid
    scene before spending time on rock placement -- fails fast on a bad --scene
    path instead of writing a manifest for a scene that can't be loaded later."""
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene.expanduser().resolve())
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.close()


def _verify_load(scene: Path, heightmap: Path, manifest: Path) -> None:
    """Re-load the saved manifest and register every rock into a live sim, as
    an experiment script would -- confirms the saved environment is actually
    usable, not just that generation ran without error."""
    import habitat_sim

    rocks, _cfg = load_rock_field(manifest)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene.expanduser().resolve())
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        register_rocks(sim, rocks)
        print(f"[verify] loaded scene + registered {len(rocks)} rocks OK")
    finally:
        sim.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scene", default=str(DEFAULT_SCENE), help="Path to the environment .glb"
    )
    ap.add_argument(
        "--heightmap",
        default=str(DEFAULT_HEIGHTMAP),
        help="Grayscale terrain heightmap for this scene",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output dir for rock meshes + rock_field.json"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed; same seed+params -> identical rock field",
    )
    ap.add_argument("--num-rocks", type=int, default=12)
    ap.add_argument(
        "--radius-min", type=float, default=0.25, help="Min rock radius (m)"
    )
    ap.add_argument("--radius-max", type=float, default=0.7, help="Max rock radius (m)")
    ap.add_argument(
        "--min-spacing",
        type=float,
        default=0.4,
        help="Min clearance (m) kept between rock surfaces",
    )
    ap.add_argument(
        "--boundary-margin",
        type=float,
        default=2.0,
        help="Keep every rock's edge this far (m) from the scene bound",
    )
    ap.add_argument("--max-attempts-per-rock", type=int, default=300)
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="X,Z,RADIUS",
        help="Keep-out circle (world x, world z, radius) rocks may not spawn inside, e.g. the rover "
        "spawn point. Repeatable. Default (if none given): a 3m circle around (0, 8), matching the "
        "rollout scripts' default --start-x/--start-z.",
    )
    ap.add_argument(
        "--skip-scene-check",
        action="store_true",
        help="Skip the up-front glb load sanity check",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="After generating, re-load the manifest and register it into a live sim",
    )
    args = ap.parse_args()

    scene = Path(args.scene)
    heightmap = Path(args.heightmap)
    out_dir = Path(args.out_dir)

    if not scene.exists():
        raise FileNotFoundError(f"scene not found: {scene}")
    if not heightmap.exists():
        raise FileNotFoundError(f"heightmap not found: {heightmap}")

    if not args.skip_scene_check:
        print(f"[1/3] checking scene loads: {scene}")
        _check_scene_loads(scene)

    exclude_zones = [_parse_zone(z) for z in args.exclude] or [(0.0, 8.0, 3.0)]

    print(f"[2/3] generating {args.num_rocks} rocks (seed={args.seed}) ...")
    grid = HeightmapGrid(heightmap)
    terrain = Terrain(grid)
    config = RockFieldConfig(
        seed=args.seed,
        num_rocks=args.num_rocks,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        min_spacing=args.min_spacing,
        boundary_margin=args.boundary_margin,
        exclude_zones=exclude_zones,
        max_attempts_per_rock=args.max_attempts_per_rock,
    )
    rocks = generate_rock_field(config, terrain, out_dir)

    manifest = out_dir / "rock_field.json"
    save_rock_field(rocks, config, manifest)
    print(f"[3/3] placed {len(rocks)}/{config.num_rocks} rocks -> {manifest}")

    if args.verify:
        print("[verify] re-loading manifest into a live sim ...")
        _verify_load(scene, heightmap, manifest)


if __name__ == "__main__":
    main()
