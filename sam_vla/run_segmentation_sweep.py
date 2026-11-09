"""Offline segmentation-dataset capture (next.md Steps 3-6): sweeps camera
poses across the scene, capturing RGB + collapsed category mask + instance
mask + per-object metadata per pose via sam_vla.perception.segmentation_capture.
Pose sourcing is a dedicated policy-independent sweep (sam_vla.env.pose_sweep),
not a rover rollout -- this script only exists to build training data /
validate the capture pipeline, not to drive the rover.

Usage:
    python -m sam_vla.run_segmentation_sweep --out-dir output/
"""

from __future__ import annotations

import argparse
import datetime
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sam_vla.core.types import Pose
from sam_vla.env.habitat_env import MarsHabitatEnv
from sam_vla.env.pose_sweep import PoseSweepConfig, sample_sweep_poses
from sam_vla.logging.episode_logger import EpisodeLogger, make_run_id
from sam_vla.perception.segmentation_capture import (
    build_category_lut,
    capture_frame_record,
    write_segmentation_assets,
)
from sam_vla.perception.spot_check_segmentation import spot_check_run

ALL_CATEGORIES = ["small_rock", "big_rock", "bedrock", "hole_in_ground"]


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    return f"{int(minutes)}m {secs:.1f}s"


def run_sweep(args: argparse.Namespace) -> dict:
    start_time = time.monotonic()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    config = {
        "goal_mode": "segmentation-sweep",
        "steering_mode": args.mode,
        "obstacle_count": "na",
        "obstacle_seed": args.seed,
        "categories": categories,
        "background_category": args.background_category,
        "annotations_dir": args.annotations_dir,
    }
    run_id = make_run_id(config)
    logger = EpisodeLogger(run_id, config, log_root=args.out_dir)

    sweep_config = PoseSweepConfig(
        mode=args.mode,
        grid_spacing_m=args.grid_spacing,
        num_yaws_per_cell=args.num_yaws_per_cell,
        num_random_poses=args.num_random_poses,
        boundary_margin=args.boundary_margin,
        seed=args.seed,
    )
    poses = sample_sweep_poses(sweep_config)

    with MarsHabitatEnv(
        args.scene_path,
        args.heightmap_path,
        with_semantic=True,
        annotations_dir=args.annotations_dir,
        annotation_categories=categories,
        rock_field_path=args.rock_field_path,
    ) as env:
        mesh_id_map = env.annotation_mesh_id_map
        lut, class_names = build_category_lut(
            mesh_id_map, categories, args.background_category
        )

        env.verify_annotation_isolation()  # next.md Step 5: fail loudly, not silently, if this regresses

        for i, (x, z, yaw) in enumerate(poses):
            env.step(Pose(x=x, y=0.0, z=z, yaw=yaw))
            obs = env.get_full_observation(
                frame_idx=i
            )  # one render call: rgb+depth+semantic+pose

            category_mask, objects = capture_frame_record(
                obs.rgb, obs.semantic, mesh_id_map, lut, class_names
            )

            frame_id = f"sweep_{i:06d}"
            paths = write_segmentation_assets(
                logger.run_dir,
                frame_id,
                obs.rgb,
                obs.semantic.astype(np.uint16),
                category_mask,
            )
            logger.log_segmentation_frame(
                frame_id=frame_id,
                objects=[asdict(o) for o in objects],
                camera_pose={
                    "x": obs.pose.x,
                    "y": obs.pose.y,
                    "z": obs.pose.z,
                    "yaw": obs.pose.yaw,
                },
                **paths,
            )
            print(f"[!] wrote -> {i}")

    elapsed_s = time.monotonic() - start_time
    summary = logger.finalize(
        {"total_frames": len(poses), "class_names": class_names, "elapsed_s": elapsed_s}
    )
    print(
        f"generated {len(poses)} images in {_format_duration(elapsed_s)} -> {logger.run_dir}"
    )

    if args.spot_check_n > 0:
        written = spot_check_run(logger.run_dir, n=args.spot_check_n, seed=args.seed)
        print(
            f"wrote {len(written)} spot-check overlays -> {logger.run_dir / 'spot_check'}"
        )

    return summary


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-path", default=str(HERE / "assets" / "marsyard2022.glb"))
    ap.add_argument(
        "--heightmap-path", default=str(HERE / "marsyard2022_terrain_hm_1025.tif")
    )
    ap.add_argument(
        "--annotations-dir", default=str(HERE / "annotations" / "mesh_segmentation")
    )
    ap.add_argument(
        "--out-dir",
        default=f"segmentation_sweep_{datetime.datetime.now().strftime('%d%m%y%H%M')}",
    )
    ap.add_argument(
        "--rock-field-path",
        default=None,
        help="optional procedural rock field, for visual diversity only -- rocks always fold into background in the mask",
    )
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--grid-spacing", type=float, default=2.0)
    ap.add_argument("--num-yaws-per-cell", type=int, default=4)
    ap.add_argument("--num-random-poses", type=int, default=500)
    ap.add_argument("--boundary-margin", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--categories",
        default=",".join(ALL_CATEGORIES),
        help="comma-separated include-list; only these categories are registered into the sim and "
        "given their own class in the mask (e.g. --categories small_rock isolates small rocks, "
        "everything else -- other categories and unlabeled terrain -- becomes background)",
    )
    ap.add_argument("--background-category", default="background")
    ap.add_argument(
        "--spot-check-n",
        type=int,
        default=0,
        help="if > 0, after generation render this many mask-overlay+label images "
        "(sam_vla.perception.spot_check_segmentation) into <out-dir>/spot_check/",
    )
    args = ap.parse_args()

    run_sweep(args)
