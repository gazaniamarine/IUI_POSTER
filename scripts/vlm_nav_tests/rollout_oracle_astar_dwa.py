#!/usr/bin/env python3
"""Run an oracle-map A* + DWA reference in the Mars Habitat scene.

This baseline receives exact world obstacle coordinates and is therefore a
privileged geometric reference, not a perception-matched competitor.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import quaternion
from PIL import Image

from oracle_astar_dwa import (
    AStarConfig,
    BoxObstacle,
    DWAConfig,
    astar_path,
    center_clearance_to_boxes,
    dwa_action,
)
from rollout_navdp_policy import (
    MESH_GOAL_ID,
    MESH_OBSTACLE_ID,
    SIZE_X,
    SIZE_Y,
    SIZE_Z,
    TerrainHeight,
    integrate_mars,
    make_simulator,
    overlay_frame,
    place_world_goal_mesh,
    place_world_obstacle_meshes,
    rgb_depth,
    save_video,
    semantic_from_observation,
    set_agent_pose,
    wrap_angle,
)


def parse_world_xz(specification: str) -> tuple[float, float]:
    values = [float(value) for value in str(specification).split(",")]
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("obstacle center must be finite X,Z")
    return values[0], values[1]


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Privileged static-map A* + DWA Habitat reference"
    )
    argument_parser.add_argument("--scene", required=True)
    argument_parser.add_argument("--terrain-obj", required=True)
    argument_parser.add_argument(
        "--terrain-height-mode", choices=["obj", "flat"], default="obj"
    )
    argument_parser.add_argument("--flat-y", type=float, default=0.0)
    argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
    argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
    argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
    argument_parser.add_argument("--clearance", type=float, default=1.4)
    argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
    argument_parser.add_argument("--height", type=int, default=720)
    argument_parser.add_argument("--width", type=int, default=720)
    argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
    argument_parser.add_argument("--hz", type=float, default=10.0)
    argument_parser.add_argument("--max-steps", type=int, default=800)
    argument_parser.add_argument("--stop-distance", type=float, default=1.0)
    argument_parser.add_argument("--start-x", type=float, default=0.0)
    argument_parser.add_argument("--start-z", type=float, default=8.0)
    argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    argument_parser.add_argument("--goal-x", type=float, required=True)
    argument_parser.add_argument("--goal-z", type=float, required=True)
    argument_parser.add_argument("--goal-height", type=float, default=1.2)
    argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
    argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)
    argument_parser.add_argument(
        "--obstacle-world-xz-item", action="append", default=[], metavar="X,Z"
    )
    argument_parser.add_argument("--world-obstacle-half-extent", type=float, default=0.75)
    argument_parser.add_argument("--world-obstacle-height", type=float, default=1.40)
    argument_parser.add_argument("--robot-radius", type=float, default=0.24)
    argument_parser.add_argument("--astar-resolution", type=float, default=0.10)
    argument_parser.add_argument("--astar-padding", type=float, default=4.0)
    argument_parser.add_argument("--planning-clearance", type=float, default=0.18)
    argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.50)
    argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.80)
    argument_parser.add_argument("--dwa-prediction-horizon", type=float, default=2.0)
    argument_parser.add_argument("--dwa-path-lookahead", type=float, default=1.20)
    argument_parser.add_argument("--dwa-desired-clearance", type=float, default=0.18)
    argument_parser.add_argument("--evaluation-layout", default="default")
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument("--output", required=True)
    argument_parser.add_argument("--save-every", type=int, default=1)
    argument_parser.add_argument(
        "--save-frames", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if not args.obstacle_world_xz_item:
        raise ValueError("at least one --obstacle-world-xz-item=X,Z is required")
    if args.robot_radius < 0.0 or args.planning_clearance < 0.0:
        raise ValueError("robot radius and planning clearance must be non-negative")
    if args.hz <= 0.0 or args.max_steps < 1:
        raise ValueError("hz and max-steps must be positive")

    np.random.seed(args.seed)
    output_directory = Path(args.output).expanduser().resolve()
    frame_directory = output_directory / "frames"
    output_directory.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        frame_directory.mkdir(parents=True, exist_ok=True)

    terrain = TerrainHeight(
        mode=args.terrain_height_mode,
        heightmap=None,
        obj=Path(args.terrain_obj).expanduser().resolve(),
        flat_y=args.flat_y,
        size_x=args.size_x,
        size_z=args.size_z,
        size_y=args.size_y,
        flip_x=False,
        flip_z=True,
        swap_xz=False,
    )
    obstacle_centers = [parse_world_xz(item) for item in args.obstacle_world_xz_item]
    obstacles = [
        BoxObstacle(x, z, args.world_obstacle_half_extent)
        for x, z in obstacle_centers
    ]
    start_xz = np.asarray([args.start_x, args.start_z], dtype=np.float64)
    goal_xz = np.asarray([args.goal_x, args.goal_z], dtype=np.float64)
    initial_goal_distance = float(np.linalg.norm(goal_xz - start_xz))
    global_path = astar_path(
        start_xz,
        goal_xz,
        obstacles,
        args.robot_radius,
        AStarConfig(
            resolution=args.astar_resolution,
            padding=args.astar_padding,
            planning_clearance=args.planning_clearance,
        ),
    )
    dwa_config = DWAConfig(
        maximum_forward_speed=args.maximum_forward_speed,
        maximum_yaw_rate=args.maximum_yaw_rate,
        prediction_horizon=args.dwa_prediction_horizon,
        path_lookahead=args.dwa_path_lookahead,
        desired_surface_clearance=args.dwa_desired_clearance,
    )
    np.save(output_directory / "astar_global_path.npy", global_path)

    simulator = make_simulator(
        Path(args.scene),
        args.height,
        args.width,
        args.hfov_deg,
        with_semantic=True,
    )
    video_frames: list[Image.Image] = []
    rows: dict[str, list[np.ndarray | float | bool]] = {
        "pose": [],
        "action_3d": [],
        "goal_distance": [],
        "planning_time_seconds": [],
        "executed_center_clearance": [],
        "executed_surface_clearance": [],
        "geometric_collision": [],
        "fallback_stop": [],
        "escape_turn": [],
        "selected_circulation_sign": [],
        "selected_barrier_energy": [],
        "selected_lyapunov_energy": [],
        "selected_minimum_clearance": [],
        "mean_guidance_noise_correction": [],
        "mean_final_effective_sample_size": [],
        "selected_trajectory": [],
    }
    success = False
    try:
        place_world_goal_mesh(
            simulator,
            terrain,
            args.goal_x,
            args.goal_z,
            output_directory,
            half_extent=args.goal_mesh_half_extent,
            height=args.goal_mesh_height,
        )
        place_world_obstacle_meshes(
            simulator,
            terrain,
            args.obstacle_world_xz_item,
            output_directory,
            half_extent=args.world_obstacle_half_extent,
            height=args.world_obstacle_height,
        )
        agent = simulator.initialize_agent(0)
        x, z = float(args.start_x), float(args.start_z)
        yaw = math.radians(float(args.start_yaw_deg))
        previous_action = np.zeros(3, dtype=np.float32)
        dt = 1.0 / float(args.hz)

        for step in range(args.max_steps):
            y = terrain.local_height_max(x, z, args.pose_terrain_radius) + args.clearance
            position = np.asarray([x, y, z], dtype=np.float32)
            set_agent_pose(agent, position, yaw)
            observation = simulator.get_sensor_observations()
            rgb, _ = rgb_depth(observation)
            semantic = semantic_from_observation(observation)
            goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
            obstacle_mask = (
                (semantic >= MESH_OBSTACLE_ID)
                & (semantic < MESH_OBSTACLE_ID + len(obstacles))
            ).astype(np.uint8)

            planning_start = time.perf_counter()
            action, predicted_rollout, _ = dwa_action(
                np.asarray([x, z, yaw], dtype=np.float64),
                previous_action,
                global_path,
                goal_xz,
                obstacles,
                args.robot_radius,
                dt,
                dwa_config,
            )
            planning_time = time.perf_counter() - planning_start
            next_position, next_yaw = integrate_mars(position, yaw, action, dt)
            previous_action = action.copy()
            x = float(
                np.clip(next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5)
            )
            z = float(
                np.clip(next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5)
            )
            yaw = wrap_angle(next_yaw)
            goal_distance = float(np.linalg.norm(goal_xz - np.asarray([x, z])))
            center_clearance = center_clearance_to_boxes(np.asarray([x, z]), obstacles)
            surface_clearance = max(center_clearance - args.robot_radius, 0.0)
            collision = bool(center_clearance <= args.robot_radius)
            predicted_clearances = [
                max(center_clearance_to_boxes(point[:2], obstacles) - args.robot_radius, 0.0)
                for point in predicted_rollout
            ]
            predicted_minimum_clearance = float(min(predicted_clearances))
            rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
            pose = np.asarray(
                [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
                dtype=np.float32,
            )
            rows["pose"].append(pose)
            rows["action_3d"].append(action)
            rows["goal_distance"].append(goal_distance)
            rows["planning_time_seconds"].append(planning_time)
            rows["executed_center_clearance"].append(center_clearance)
            rows["executed_surface_clearance"].append(surface_clearance)
            rows["geometric_collision"].append(collision)
            rows["fallback_stop"].append(False)
            rows["escape_turn"].append(False)
            rows["selected_circulation_sign"].append(0.0)
            rows["selected_barrier_energy"].append(np.nan)
            rows["selected_lyapunov_energy"].append(np.nan)
            rows["selected_minimum_clearance"].append(predicted_minimum_clearance)
            rows["mean_guidance_noise_correction"].append(np.nan)
            rows["mean_final_effective_sample_size"].append(np.nan)
            rows["selected_trajectory"].append(predicted_rollout)

            if (args.save_frames or args.save_video) and step % max(args.save_every, 1) == 0:
                label = (
                    f"A*+DWA oracle t={step} goal={goal_distance:.2f}m "
                    f"clear={surface_clearance:.2f}m v={action[0]:.2f} w={action[2]:.2f}"
                )
                frame = overlay_frame(
                    rgb, goal_mask, obstacle_mask, label, show_masks=True
                )
                video_frames.append(frame)
                if args.save_frames:
                    frame.save(frame_directory / f"frame_{step:04d}.png")

            print(
                f"step={step:04d} goal={goal_distance:.2f}m "
                f"actual_clear={surface_clearance:.3f}m collision={collision} "
                f"latency={planning_time * 1000.0:.2f}ms "
                f"action={action.tolist()}",
                flush=True,
            )
            if goal_distance <= args.stop_distance:
                success = True
                break

        if not rows["goal_distance"]:
            raise RuntimeError("oracle rollout produced no steps")
        rollout_path = output_directory / "rollout.npz"
        stacked = {
            key: np.stack(values)
            if isinstance(values[0], np.ndarray)
            else np.asarray(values)
            for key, values in rows.items()
        }
        np.savez_compressed(
            rollout_path,
            **stacked,
            goal_position=np.asarray(
                [
                    args.goal_x,
                    terrain.local_height_max(args.goal_x, args.goal_z, 0.8)
                    + args.goal_height,
                    args.goal_z,
                ],
                dtype=np.float32,
            ),
            obstacle_positions=np.asarray(
                [
                    [
                        obstacle.center_x,
                        terrain.local_height_max(
                            obstacle.center_x,
                            obstacle.center_z,
                            obstacle.half_extent,
                        )
                        + 0.5 * args.world_obstacle_height,
                        obstacle.center_z,
                    ]
                    for obstacle in obstacles
                ],
                dtype=np.float32,
            ),
            success=np.asarray(success),
            hz=np.asarray(args.hz, dtype=np.float32),
            start_position_xz=start_xz,
            initial_goal_distance=np.asarray(initial_goal_distance),
            stop_distance=np.asarray(args.stop_distance),
            robot_radius=np.asarray(args.robot_radius),
            evaluation_layout=np.asarray(args.evaluation_layout),
            seed=np.asarray(args.seed, dtype=np.int64),
            planner_mode=np.asarray("oracle-astar-dwa"),
            goal_mode=np.asarray("privileged-world-goal"),
            particles_per_candidate=np.asarray(np.nan),
            barrier_weight=np.asarray(np.nan),
            lyapunov_weight=np.asarray(np.nan),
            astar_global_path=global_path,
            astar_resolution=np.asarray(args.astar_resolution),
            planning_clearance=np.asarray(args.planning_clearance),
        )
        with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "success": success,
                    "steps": len(rows["goal_distance"]),
                    "planner": "oracle_astar_plus_dwa",
                    "comparison_role": "privileged_geometric_reference",
                    "uses_exact_world_goal": True,
                    "uses_exact_static_obstacle_geometry": True,
                    "uses_learned_perception": False,
                    "uses_navdp": False,
                    "uses_qwen": False,
                    "uses_particles": False,
                    "evaluation_layout": args.evaluation_layout,
                    "seed": args.seed,
                    "robot_radius": args.robot_radius,
                    "planning_clearance": args.planning_clearance,
                    "obstacle_world_xz": args.obstacle_world_xz_item,
                    "geometric_collision": bool(np.any(rows["geometric_collision"])),
                    "minimum_executed_surface_clearance": float(
                        np.min(rows["executed_surface_clearance"])
                    ),
                    "rollout": str(rollout_path),
                },
                file,
                indent=2,
            )
        if args.save_video and video_frames:
            save_video(
                video_frames,
                output_directory / "rollout.mp4",
                fps=max(args.hz / max(args.save_every, 1), 1.0),
            )
        print(f"Saved rollout: {rollout_path}", flush=True)
        print(f"Success: {success}", flush=True)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
