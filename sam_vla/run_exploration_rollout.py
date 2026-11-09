"""Two-phase rollout: drives ExplorationPolicy (LEFT/RIGHT/FRONT/BACK legs)
until BeliefGoalTracker confirms the real, VLM-detected goal mesh has come
into view, then hands off to NavdpPolicy to actually drive to it within
`goal_arrival_radius_m` -- the sim never stops at the handoff. While driving,
if the goal mesh drops out of the semantic render (occlusion/out of FOV),
inject_ghost_goal_mask() paints a filled goal-colored circle at the tracker's
projected pixel so NavdpPolicy (which only ever "sees" goal-colored pixels,
never a language instruction) still has something to steer toward -- unlike
kb_teleop_vl.py's ghost mask, this one is policy-facing, not just advisory.
Needs two separate Qwen server managers (different ports): one for the
one-shot first-frame goal/obstacle detection, one for ExplorationPolicy's
per-leg direction queries.

Usage:
    conda activate habitat
    python -m sam_vla.run_exploration_rollout \
        --scene-path assets/marsyard2022.glb \
        --heightmap-path marsyard2022_terrain_hm_1025.tif \
        --command area --cbf --max-steps 300 \
        --navdp-ckpt <ckpt.pt> \
        --out-dir output/explore_smoke_test
"""

import argparse
import datetime
import math
import time
from pathlib import Path

import numpy as np

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.ghost_mask import (
    project_or_clamp_body_point_to_pixel,
    uncertainty_to_radius_px,
)
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.sim_utils import distance_to_goal
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.logging.rollout_logger import RolloutLogger
from sam_vla.perception.semantic_overlay import overlay_semantic_masks
from sam_vla.policy.exploration_policy import ExplorationConfig, ExplorationPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.vlm.qwen_server_manager import QwenServerManager as SamQwenServerManager
from vl_direction.client import get_client
from vl_direction.qwen_server_manager import QwenServerManager as VlQwenServerManager

GOAL_CONFIRM_STEPS = 5


def register_goal_obstacle_masks(
    env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir
):
    """Give the chosen goal object a goal-mask mesh and every other detected
    object an obstacle-mask mesh -- copy of run_navdp_rollout.py's helper of
    the same name, see module docstring for why this isn't an import."""
    if goal_position is not None:
        env.register_object_mask(
            goal_position, MESH_GOAL_ID, obj_mask_radius, out_dir, "goal"
        )
    else:
        print("[WARN] goal bbox had no valid depth; skipping goal mask", flush=True)

    for i, obstacle_bbox in enumerate(goal_spec.obstacle_bboxes_norm):
        obstacle_position = bbox_to_world(obs0, obstacle_bbox, hfov_deg=HFOV_DEG)
        if obstacle_position is None:
            print(
                f"[WARN] obstacle[{i}] bbox had no valid depth; skipping obstacle mask",
                flush=True,
            )
            continue
        env.register_object_mask(
            obstacle_position, MESH_OBST_ID, obj_mask_radius, out_dir, f"obstacle_{i}"
        )


def inject_ghost_goal_mask(
    semantic,
    obstacle_mask,
    belief_tracker: BeliefGoalTracker,
    hfov_deg: float,
    min_px: float,
    max_px: float,
    radius_scale: float,
):
    """When the goal mesh isn't in this frame's semantic render, paint a filled
    MESH_GOAL_ID circle into a *copy* of `semantic` at BeliefGoalTracker's
    projected pixel, radius grown from its uncertainty -- this is the actual
    pixel NavdpPolicy will drive toward (via its own `sem == MESH_GOAL_ID`
    read), not just a human-facing overlay. Never overwrites obstacle-tagged
    pixels, so CBF's obstacle mask stays intact under the ghost. Returns
    (semantic_or_copy, ghost_info) where ghost_info is safe to merge straight
    into a step's logged vla_result."""
    bearing = belief_tracker.bearing()
    distance = belief_tracker.distance()
    if bearing is None or distance is None:
        return semantic, {"ghost_active": False}

    forward = distance * math.cos(bearing)
    left = distance * math.sin(bearing)
    height, width = semantic.shape[:2]
    u, v = project_or_clamp_body_point_to_pixel(forward, left, hfov_deg, height, width)
    radius_px = uncertainty_to_radius_px(
        belief_tracker.uncertainty_value(), min_px, max_px, radius_scale
    )

    yy, xx = np.mgrid[0:height, 0:width]
    ghost_pixels = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    ghost_pixels &= np.asarray(obstacle_mask) == 0

    ghosted = semantic.copy()
    ghosted[ghost_pixels] = MESH_GOAL_ID
    return ghosted, {
        "ghost_active": True,
        "ghost_u": round(float(u), 1),
        "ghost_v": round(float(v), 1),
        "ghost_radius_px": round(float(radius_px), 1),
    }


def run(
    scene_path: str,
    heightmap_path: str,
    out_dir: str,
    task_str: str = "explore the terrain",
    command: str = "area",
    max_steps: int = 500,
    dt: float = 0.1,
    save_video: bool = False,
    save_frames: bool = False,
    video_fps: int = 10,
    start_x: float = 0.0,
    start_z: float = 8.0,
    start_yaw_deg: float = 0.0,
    randomise_spawn: bool = False,
    obj_mask_radius: float = 0.5,
    cbf: bool = False,
    navdp_root: str = None,
    cbf_d_safe: float = 0.75,
    cbf_gamma: float = 0.3,
    cbf_deadzone: float = 0.6,
    cbf_orbit_kr: float = 0.8,
    cbf_orbit_hyst: float = 0.4,
    cbf_pursuit_kp: float = 1.8,
    cbf_goaround_forward: float = 0.5,
    cbf_escape_yaw: bool = True,
    cbf_hard_gate: bool = True,
    robot_radius: float = 0.25,
    safety_margin: float = 0.15,
    obstacle_radius: float = 0.25,
    max_yaw_rate: float = 1.0,
    zero_lateral: bool = True,
    belief_goal_range: float = 8.0,
    lost_goal_min_px: int = 10,
    belief_odom_noise: float = 0.01,
    belief_odom_noise_growth_rate: float = 0.0,
    leg_length_m: float = 3.0,
    leg_length_jitter_m: float = 1.0,
    cruise_speed: float = 1.0,
    turn_deg: float = 60.0,
    turn_jitter_deg: float = 20.0,
    cell_size_m: float = 1.5,
    stall_window_steps: int = 15,
    stall_distance_m: float = 0.5,
    backtrack_distance_m: float = 2.0,
    seed: int = None,
    navdp_ckpt: str = None,
    navdp_device: str = "cuda",
    navdp_image_size: int = None,
    navdp_sample_steps: int = 20,
    navdp_replan_every: int = 1,
    navdp_max_forward_speed: float = 1.0,
    navdp_max_lateral_speed: float = 1.0,
    goal_arrival_radius_m: float = 1.0,
    ghost_min_px: float = 3.0,
    ghost_max_px: float = 100.0,
    ghost_uncertainty_saturation: float = 0.15,
) -> None:
    if navdp_ckpt is None:
        raise ValueError(
            "--navdp-ckpt is required: after the goal is confirmed this script "
            "drives to it with NavdpPolicy, it doesn't just stop"
        )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sam_qwen_manager = SamQwenServerManager()
    vl_qwen_manager = VlQwenServerManager()
    logger = RolloutLogger()
    episode_id = f"explore-{int(time.time())}"

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        services=[sam_qwen_manager, vl_qwen_manager],
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        randomise_spawn=randomise_spawn,
        with_semantic=True,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)

        goal_spec, goal_vlm_result, sam_detections = (
            first_frame_resolver.resolve_verbose(obs0.rgb)
        )
        goal_position = backproject_goal_position(obs0, goal_spec, hfov_deg=HFOV_DEG)
        logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
        logger.save_sam_first_frame(obs0.rgb, sam_detections, goal_spec, out_dir)
        print(
            f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}"
        )
        register_goal_obstacle_masks(
            env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir
        )

        policy = ExplorationPolicy(
            ExplorationConfig(
                task_str=task_str,
                leg_length_m=leg_length_m,
                leg_length_jitter_m=leg_length_jitter_m,
                cruise_speed=cruise_speed,
                turn_deg=turn_deg,
                turn_jitter_deg=turn_jitter_deg,
                cell_size_m=cell_size_m,
                stall_window_steps=stall_window_steps,
                stall_distance_m=stall_distance_m,
                backtrack_distance_m=backtrack_distance_m,
                seed=seed,
            ),
            episode_id=episode_id,
            client=get_client("qwen"),
        )
        policy.set_command(command)

        belief_tracker = BeliefGoalTracker(
            hfov_deg=HFOV_DEG,
            goal_range=belief_goal_range,
            min_px=lost_goal_min_px,
            odom_noise=belief_odom_noise,
            odom_noise_growth_rate=belief_odom_noise_growth_rate,
        )
        # Ghost radius saturates to ghost_max_px right as uncertainty reaches
        # ghost_uncertainty_saturation -- same scale formula kb_teleop_vl.py uses.
        ghost_radius_scale = ghost_max_px / max(ghost_uncertainty_saturation, 1e-6)

        avoidance = None
        if cbf:
            # CbfObstacleAvoidance.__init__ does `from navdp.extensions import ...`,
            # which needs navdp_root on sys.path first -- normally done by
            # NavdpPolicy's constructor (see its module docstring), but this script
            # has no NavdpPolicy, so do it directly with the same helpers
            # run_navdp_rollout.py uses.
            from sam_vla.policy.navdp_policy import (
                _add_navdp_to_path,
                _resolve_navdp_root,
            )
            from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance

            _add_navdp_to_path(_resolve_navdp_root(navdp_root))

            avoidance = CbfObstacleAvoidance(
                d_safe=cbf_d_safe,
                gamma=cbf_gamma,
                deadzone=cbf_deadzone,
                orbit_kr=cbf_orbit_kr,
                orbit_hyst=cbf_orbit_hyst,
                pursuit_kp=cbf_pursuit_kp,
                goaround_forward=cbf_goaround_forward,
                escape_yaw=cbf_escape_yaw,
                hard_gate=cbf_hard_gate,
                robot_radius=robot_radius,
                safety_margin=safety_margin,
                obstacle_radius=obstacle_radius,
                max_yaw_rate=max_yaw_rate,
            )
        cbf_active_steps = 0
        hard_gate_fired_steps = 0
        goal_seen_steps = 0
        phase = "explore"  # -> "drive" once the goal is confirmed; see module docstring

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            semantic = env.get_semantic_frame()

            goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
            obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255
            goal_visible = belief_tracker.observe(goal_mask, obs.depth)
            goal_bearing = belief_tracker.bearing()

            policy_semantic = semantic
            ghost_info = {}
            if phase == "drive" and not goal_visible:
                policy_semantic, ghost_info = inject_ghost_goal_mask(
                    semantic,
                    obstacle_mask,
                    belief_tracker,
                    HFOV_DEG,
                    ghost_min_px,
                    ghost_max_px,
                    ghost_radius_scale,
                )

            raw_action, vla_result = policy.act_verbose(
                obs, policy_semantic, goal_spec, step
            )
            action = safety_filter_fn(raw_action, obs)

            obstacle_point = None
            if avoidance is not None:
                height, width = obs.depth.shape[:2]
                intr = intrinsics_from_hfov(height, width, HFOV_DEG)
                obstacle_point = avoidance.nearest_obstacle(
                    obstacle_mask, obs.depth, intr
                )

            if zero_lateral and avoidance is not None:
                action = Action(v_fwd=action.v_fwd, v_lat=0.0, yaw_rate=action.yaw_rate)

            cbf_info = {}
            if avoidance is not None:
                action, cbf_info = avoidance.apply(action, obstacle_point, goal_bearing)
                if cbf_info.get("blocked"):
                    cbf_active_steps += 1
                if cbf_info.get("hard_gate_fired"):
                    hard_gate_fired_steps += 1

            new_pose = integrate_mars(obs.pose, action, dt)
            env.step(new_pose)
            belief_tracker.propagate(action, dt)

            dist = (
                distance_to_goal(new_pose, goal_position)
                if goal_position is not None
                else None
            )
            dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
            if phase == "explore":
                overlay_text = (
                    f"t={step} phase={phase} dist={dist_txt} "
                    f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f} "
                    f"{vla_result['command']}/{vla_result['leg_kind']}"
                )
            else:
                overlay_text = (
                    f"t={step} phase={phase} dist={dist_txt} "
                    f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f} "
                    f"ghost={ghost_info.get('ghost_active', False)}"
                )
            vis_rgb = overlay_semantic_masks(
                obs.rgb, policy_semantic, text=overlay_text
            )

            vla_result = {
                **vla_result,
                "phase": phase,
                "goal_visible": goal_visible,
                **ghost_info,
                **cbf_info,
            }
            logger.log_step(
                obs, action, new_pose, vla_result=vla_result, vis_rgb=vis_rgb
            )

            if step % 10 == 0:
                print(
                    f"[traj] step={step} | phase={phase} | distance_to_goal={dist} | "
                    f"action={action} | {vla_result}"
                )

            if phase == "explore":
                if goal_visible:
                    goal_seen_steps += 1
                    if goal_seen_steps >= GOAL_CONFIRM_STEPS:
                        print(
                            f"[explore] goal confirmed visible at step={step}, "
                            "switching to drive phase",
                            flush=True,
                        )
                        from sam_vla.policy.navdp_policy import NavdpPolicy

                        policy = NavdpPolicy(
                            ckpt_path=navdp_ckpt,
                            navdp_root=navdp_root,
                            device=navdp_device,
                            image_size=navdp_image_size,
                            sample_steps=navdp_sample_steps,
                            hfov_deg=HFOV_DEG,
                            replan_every=navdp_replan_every,
                            max_forward_speed=navdp_max_forward_speed,
                            max_lateral_speed=navdp_max_lateral_speed,
                            max_yaw_rate=max_yaw_rate,
                        )
                        phase = "drive"
                else:
                    goal_seen_steps = 0
            else:  # phase == "drive"
                drive_dist = dist if dist is not None else belief_tracker.distance()
                if drive_dist is not None and drive_dist <= goal_arrival_radius_m:
                    print(
                        f"[drive] reached goal at step={step} dist={drive_dist:.2f}m, "
                        "stopping",
                        flush=True,
                    )
                    break

        if avoidance is not None:
            print(
                f"[CBF diag] blocked_steps={cbf_active_steps} hard_gate_fired={hard_gate_fired_steps}",
                flush=True,
            )
        logger.flush(out_dir)
        if save_frames:
            logger.save_frames(out_dir)
        if save_video:
            logger.save_video(out_dir, fps=video_fps)

    print("[explore] qwen managers: stop confirmed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument(
        "--out-dir",
        default=f"explore_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}",
    )
    parser.add_argument(
        "--task-str",
        default="explore around and find the goal",
        help="high-level task string passed to vl_direction's exploration prompt",
    )
    parser.add_argument(
        "--command",
        choices=["area", "left", "right"],
        default="area",
        help="which of ExplorationPolicy's three commandable patterns to run",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--start-x", type=float, default=8.0)
    parser.add_argument("--start-z", type=float, default=10.0)
    parser.add_argument("--start-yaw", dest="start_yaw_deg", type=float, default=0.0)
    parser.add_argument("--randomise-spawn", action="store_true")
    parser.add_argument("--obj-mask-radius", type=float, default=0.5)

    parser.add_argument("--cbf", action="store_true")
    parser.add_argument(
        "--navdp-root",
        default="./navdp",
        help=(
            "Path to the navdp repo, needed by --cbf and always by the drive phase "
            "(default: ./navdp or $NAVDP_ROOT)"
        ),
    )
    parser.add_argument(
        "--navdp-ckpt",
        required=True,
        help="NavDP checkpoint used to drive to the goal once it's confirmed",
    )
    parser.add_argument("--navdp-device", default="cuda")
    parser.add_argument(
        "--navdp-image-size",
        type=int,
        default=None,
        help="default: read from the checkpoint's train_args",
    )
    parser.add_argument("--navdp-sample-steps", type=int, default=20)
    parser.add_argument("--navdp-replan-every", type=int, default=1)
    parser.add_argument("--navdp-max-forward-speed", type=float, default=1.0)
    parser.add_argument("--navdp-max-lateral-speed", type=float, default=1.0)
    parser.add_argument(
        "--goal-arrival-radius-m",
        type=float,
        default=1.0,
        help="drive phase stops once within this planar distance of the goal",
    )
    parser.add_argument("--cbf-d-safe", type=float, default=0.75)
    parser.add_argument("--cbf-gamma", type=float, default=0.3)
    parser.add_argument("--cbf-deadzone", type=float, default=0.6)
    parser.add_argument("--cbf-orbit-kr", type=float, default=0.8)
    parser.add_argument("--cbf-orbit-hyst", type=float, default=0.4)
    parser.add_argument("--cbf-pursuit-kp", type=float, default=1.8)
    parser.add_argument("--cbf-goaround-forward", type=float, default=0.5)
    parser.add_argument(
        "--cbf-escape-yaw", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cbf-hard-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--obstacle-radius", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument(
        "--zero-lateral", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--belief-goal-range", type=float, default=8.0)
    parser.add_argument("--lost-goal-min-px", type=int, default=20)
    parser.add_argument(
        "--belief-odom-noise",
        type=float,
        default=0.001,
        help="per-step uncertainty growth while the goal is unseen (drives ghost radius)",
    )
    parser.add_argument(
        "--belief-odom-noise-growth-rate",
        type=float,
        default=0.0,
        help="accelerating-drift term on top of --belief-odom-noise the longer the goal stays unseen",
    )
    parser.add_argument(
        "--ghost-min-px",
        type=float,
        default=3.0,
        help="min ghost circle radius, in pixels",
    )
    parser.add_argument(
        "--ghost-max-px",
        type=float,
        default=100.0,
        help="max ghost circle radius, in pixels",
    )
    parser.add_argument(
        "--ghost-uncertainty-saturation",
        type=float,
        default=0.15,
        help="uncertainty value at which the ghost radius saturates to --ghost-max-px",
    )

    parser.add_argument("--leg-length-m", type=float, default=3.0)
    parser.add_argument("--leg-length-jitter-m", type=float, default=1.0)
    parser.add_argument("--cruise-speed", type=float, default=1.0)
    parser.add_argument("--turn-deg", type=float, default=60.0)
    parser.add_argument("--turn-jitter-deg", type=float, default=20.0)
    parser.add_argument("--cell-size-m", type=float, default=1.5)
    parser.add_argument("--stall-window-steps", type=int, default=15)
    parser.add_argument("--stall-distance-m", type=float, default=0.5)
    parser.add_argument("--backtrack-distance-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    run(**vars(args))
