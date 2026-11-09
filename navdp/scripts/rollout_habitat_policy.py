"""Closed-loop Habitat rollout for a trained route-belief S2 policy.

This script is meant to be run from the NavDP/SAM repo while pointing at the
separate Habitat dataset-generation folder that contains:

    assets_catalog.py, planner2.py, planner_rrt.py, scene_gen.py

It uses the same scene/spawn machinery as gen_rrt_dataset.py, but the expert
RRT controller is replaced by the learned policy:

    observe -> update belief -> sample action chunk -> execute first action

The output layout mirrors the RRT dataset enough that the existing
HabitatRouteDataset can inspect it, while also saving policy-specific arrays
such as pred_action_chunk and true_seg_masks.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_s2_dit import S2DiTPolicy
from navdp.data.habitat_route_dataset import (
    _empty_belief_tensor,
    _normalize_depth,
    _odom_delta,
    _proprio_from_pose,
    _resize_depth_mask,
)
from navdp.extensions import DepthObstacleMap, SAMDepthTargetExtractor, SubgoalBeliefBank
from navdp.extensions import (
    build_cbf_guidance,
    estimate_obstacle_velocity,
    gc_body_point as _gc_body_point,
    gc_intrinsics as _gc_intrinsics,
    gc_make_mask as _gc_make_mask,
    gc_project as _gc_project,
    horizon_growth_covariance,
    nearest_obstacle_point,
    project_chunk_cone,
    project_forward_velocity_cbf,
    tangential_around_obstacle,
)
# Proven belief -> ghost-subgoal rendering (frame-corrected: motion = -90-2*yaw-belief).
# Validated end-to-end in roundtrip_rollout.py / ablation_belief_rollout.py: the policy
# is MASK-conditioned, so this is the ONLY mechanism that reliably steers it back to a
# goal that has left the frame. --cbf-goal-attract under --cbf-mode cone does NOT do
# this (that path only ever affects during-sampling guidance, which cone mode skips).


GOAL_CATEGORIES = [
    "chair",
    "sofa",
    "lamp",
    "refrigerator",
    "cabinet",
    "indoor plant",
    "rack",
    "stool",
    "beanbag",
    "monitor",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Roll out a trained NavDP/S2 policy in generated Habitat scenes."
    )
    ap.add_argument("--sim-root", required=True, help="Folder containing scene_gen.py/planner2.py")
    ap.add_argument("--ckpt", required=True, help="Policy checkpoint: ckpt_last.pt or ckpt_XXXXXX.pt")
    ap.add_argument("--out", required=True, help="Output dataset root for policy rollouts")
    ap.add_argument("--categories", nargs="*", default=None, help="Goal categories to generate")
    ap.add_argument(
        "--scene-mode",
        choices=["standard", "obstacle"],
        default="standard",
        help="standard uses scene_gen.build_scene; obstacle uses gen_obstacle_dataset._build_obstacle_scene",
    )
    ap.add_argument("--episodes-per-category", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=720)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--max-tries-per-episode", type=int, default=40)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Step budget per episode. At hz=30 and ~0.35 m/s the robot covers "
        "~0.012 m/step, so 300 only reaches ~3.5 m; expert data used up to 1800 "
        "(60 s). Raise this if episodes stop halfway to far goals.",
    )
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--distance-min", type=float, default=2.0)
    ap.add_argument("--distance-max", type=float, default=6.0)
    ap.add_argument("--stop-dist", type=float, default=None)
    ap.add_argument(
        "--goal-radius",
        type=float,
        default=0.0,
        help="Radius (m) of the goal object, so success is measured from its SURFACE not its centre: "
        "success when (dist-to-centre - goal_radius) <= stop_dist. 0 = centre (old behaviour).",
    )
    ap.add_argument(
        "--arrival-brake-dist",
        type=float,
        default=0.0,
        help="[opt-in, default off] Hardcoded deceleration near the goal. Off by default: the success "
        "radius (--stop-dist) ends the episode on approach, so the policy drives the whole way in. 0=off.",
    )
    ap.add_argument(
        "--arrival-min-speed-frac",
        type=float,
        default=0.12,
        help="Floor on the braked speed fraction (only used if --arrival-brake-dist>0).",
    )
    ap.add_argument("--min-start-visible-pixels", type=int, default=12)
    ap.add_argument("--min-visible-pixels", type=int, default=20)
    ap.add_argument(
        "--disable-belief",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed the EMPTY belief tensor (no memory) instead of the real belief. "
        "Runs the policy on mask+depth+proprio only -- the working baseline behavior "
        "from before the belief was added, for use with any checkpoint (belief-aware "
        "checkpoints tolerate this fine since the tensor was already inert in practice).",
    )
    ap.add_argument(
        "--obstacle-pool",
        choices=["generator", "large-footprint"],
        default="generator",
        help="For scene-mode=obstacle: generator uses the original pool; large-footprint selects blockers by bbox size.",
    )
    ap.add_argument(
        "--large-obstacle-min-side",
        type=float,
        default=0.35,
        help="Minimum second-largest bbox side for large-footprint obstacle pool.",
    )
    ap.add_argument(
        "--large-obstacle-min-footprint",
        type=float,
        default=0.16,
        help="Minimum product of the two largest bbox sides for large-footprint obstacle pool.",
    )
    ap.add_argument(
        "--large-obstacle-exclude",
        default="cup,mug,bottle,bowl,plate,choppingboard,spoon,fork,knife,can",
        help="Comma-separated obstacle categories to exclude from the large-footprint pool.",
    )

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", choices=["model", "ema"], default="model")
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--habitat-proprio-mode", choices=["pose7", "planar3", "zero"], default=None)
    ap.add_argument("--habitat-action-mode", choices=["action3d", "action2d", "waypoint"], default=None)
    ap.add_argument("--habitat-yaw-axis", choices=["x", "y", "z"], default=None)
    ap.add_argument(
        "--habitat-use-obstacle-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Feed [goal_mask, obstacle_mask, depth] when enabled; defaults to checkpoint/trained spatial channels",
    )

    ap.add_argument("--max-forward-speed", type=float, default=1.0)
    ap.add_argument("--max-lateral-speed", type=float, default=1.0)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument(
        "--obstacle-mask-dilate",
        type=int,
        default=0,
        help="Inflate the simulator obstacle mask by this many pixels before feeding the policy.",
    )
    ap.add_argument(
        "--obstacle-safety-override",
        action="store_true",
        help="Apply a lightweight reactive dodge when a visible obstacle is close and central.",
    )
    ap.add_argument("--obstacle-safety-min-px", type=int, default=24)
    ap.add_argument("--obstacle-safety-center-frac", type=float, default=0.34)
    ap.add_argument("--obstacle-safety-close-depth", type=float, default=3.0)
    ap.add_argument("--obstacle-safety-forward", type=float, default=0.12)
    ap.add_argument("--obstacle-safety-lateral", type=float, default=0.35)
    ap.add_argument("--obstacle-safety-yaw", type=float, default=0.45)
    ap.add_argument(
        "--motion-convention",
        choices=["habitat", "x-forward", "z-forward"],
        default="habitat",
        help=(
            "How [v_fwd,v_lat] maps into world x/z. Use habitat for Habitat's "
            "default -Z forward; switch if rollouts move in the wrong direction."
        ),
    )
    ap.add_argument("--no-navmesh-step", action="store_true", help="Do not clamp motion with pathfinder.try_step")

    ap.add_argument(
        "--action-smoothing",
        choices=["ensemble", "ema", "none"],
        default="ensemble",
        help=(
            "Temporally smooth executed commands. ensemble=exp-weighted average of "
            "overlapping chunk predictions (ACT-style temporal ensembling); "
            "ema=low-pass on the first action; none=execute raw first action (old behavior)."
        ),
    )
    ap.add_argument(
        "--ensemble-decay",
        type=float,
        default=0.5,
        help="Ensemble weight decay: newer predictions get weight exp(-decay*age). Higher=more responsive.",
    )
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing factor for --action-smoothing ema: a_exec = alpha*a_prev + (1-alpha)*a_pred.",
    )
    ap.add_argument(
        "--fixed-episode-noise",
        action="store_true",
        help="Reuse one fixed diffusion noise per episode so the sampled chunk drifts smoothly with the observation.",
    )

    ap.add_argument(
        "--cbf",
        action="store_true",
        help="Enable horizon-CBF OBSTACLE-SAFETY guidance during diffusion sampling (obstacle from "
        "seg-mask+depth, no grid/VLM). By default this ONLY repels from obstacles; all goal-seeking "
        "and centering come from the policy + belief.",
    )
    ap.add_argument("--cbf-d-safe", type=float, default=0.5, help="CBF clearance radius (m).")
    ap.add_argument("--cbf-gamma", type=float, default=0.3, help="CBF approach rate in (0,1); smaller=brakes earlier.")
    ap.add_argument("--cbf-guidance-scale", type=float, default=0.4, help="Guidance step size per denoising step.")
    ap.add_argument("--cbf-steps", type=int, default=2, help="Guidance gradient steps per denoising step.")
    ap.add_argument("--cbf-vel-scale", type=float, default=1.0, help="Scale mapping action units to m/s for the CBF rollout.")
    ap.add_argument(
        "--cbf-mode",
        choices=["guidance", "project", "cone"],
        default="guidance",
        help="guidance: nudge the sample every denoising step (original; can fight the policy). "
        "project: sample once, then post-hoc BRAKE forward speed (v_lat=0 distance barrier). "
        "cone: sample once, then post-hoc project the whole 8-step chunk OUT of the collision cone "
        "(C3BF) -- steers via yaw over the horizon instead of only braking.",
    )
    ap.add_argument("--cbf-proj-iters", type=int, default=10,
                    help="cone mode: gradient steps of the post-hoc horizon projection.")
    ap.add_argument("--cbf-proj-lr", type=float, default=0.05,
                    help="cone mode: step size of the post-hoc horizon projection.")
    ap.add_argument("--cbf-cone-margin", type=float, default=0.05,
                    help="cone mode: require h >= margin (safety buffer on the cone barrier).")
    ap.add_argument("--cbf-smooth", type=float, default=0.0,
                    help="cone mode: temporal-smoothness weight on the projection correction "
                    "(penalize ||u_{k+1}-u_k||^2) so the avoidance stays jerk-free.")
    ap.add_argument("--cbf-keep-speed", type=float, default=1.0,
                    help="cone mode: penalty for dropping forward speed. The cone is trivially "
                    "satisfied by v=0 (a stopped robot never collides), so WITHOUT this the "
                    "projection brakes to a stop instead of steering. >0 forces it to go AROUND. "
                    "Set 0 to allow braking.")
    ap.add_argument("--cbf-metric", choices=["euclidean", "mahalanobis"], default="euclidean",
                    help="cone mode projection metric. euclidean: plain gradient. mahalanobis: "
                    "covariance-preconditioned step so the correction is spent on the UNCERTAIN "
                    "far-horizon actions and preserves the confident near-term action.")
    ap.add_argument("--cbf-cov-base", type=float, default=1.0,
                    help="mahalanobis: base action variance for step 0.")
    ap.add_argument("--cbf-cov-growth", type=float, default=0.6,
                    help="mahalanobis: how fast action variance changes across the horizon.")
    ap.add_argument("--cbf-cov-mode", choices=["grow", "flat", "shrink"], default="shrink",
                    help="mahalanobis covariance shape. grow: far steps corrected more (diffusion "
                    "'future uncertain' prior, but PROTECTS the executed chunk[0] -> can collide under "
                    "receding horizon). shrink (default): NEAR steps corrected more (emphasizes the "
                    "executed action -> safe). flat: ~euclidean.")
    ap.add_argument("--cbf-deadzone", type=float, default=0.5,
                    help="project mode: extra margin beyond d_safe within which braking activates (m). "
                    "Outside d_safe+deadzone the CBF is a no-op (no free-space thrash).")
    ap.add_argument("--cbf-trust", type=float, default=0.0,
                    help="project mode: max forward-speed cut per step for smoothness (0 = full brake). "
                    "A real breach (d<d_safe) always full-brakes regardless.")
    ap.add_argument(
        "--lost-goal-ghost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When the real goal mask is lost (occluded / out of frame) and the belief is "
        "initialized, render the belief as a GHOST circle in the goal channel (frame-corrected: "
        "motion=-90-2*yaw-belief, clamped to the FOV, at the believed range) so the mask-"
        "conditioned policy has something to steer back to. Without this, cone/project CBF modes "
        "have NO mechanism to return to a lost goal (--cbf-goal-attract only affects guidance mode).",
    )
    ap.add_argument("--ghost-mask-radius", type=int, default=14,
                    help="lost-goal-ghost: pixel radius of the rendered ghost circle.")
    ap.add_argument("--ghost-hfov-deg", type=float, default=90.0,
                    help="lost-goal-ghost: camera HFOV (deg) for ghost projection (Habitat default 90).")
    ap.add_argument(
        "--zero-lateral",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force v_lat=0 (differential-drive / unicycle: forward + yaw only). Required for the "
        "project-mode braking CBF to be the correct model.",
    )

    # --- OPT-IN goal-steering guidance (OFF by default) --------------------------
    # These are hand-built potential-field controllers, NOT the policy. They are
    # disabled (0.0) so goal-seeking + centering stay policy-driven. Set >0 only to
    # experiment; the default rollout is "CBF = obstacle safety only".
    ap.add_argument("--cbf-goal-attract", type=float, default=0.0,
                    help="[opt-in, default off] Goal-attraction weight toward belief-mu. Hardcoded steering.")
    ap.add_argument("--cbf-return-scale", type=float, default=0.25,
                    help="Guidance step size for the obstacle-free goal-attraction phase (only used if --cbf-goal-attract>0).")
    ap.add_argument("--cbf-takeover-attract", type=float, default=0.0,
                    help="[opt-in, default off] Goal-attraction weight when the goal mask is lost. Hardcoded steering.")
    ap.add_argument("--cbf-tangential", type=float, default=0.0,
                    help="[opt-in, default off] Tangential circulation around wide obstacles. Hardcoded steering.")
    ap.add_argument("--cbf-block-range", type=float, default=1.5,
                    help="Circulation trigger range (m) (only used if --cbf-tangential>0).")
    ap.add_argument("--cbf-tangential-speed", type=float, default=0.3,
                    help="Target tangential speed (m/s) (only used if --cbf-tangential>0).")
    ap.add_argument("--cbf-heading", type=float, default=0.0,
                    help="[opt-in, default off] Heading-alignment toward belief-mu (centering). Hardcoded steering.")

    ap.add_argument(
        "--hide-mask-window",
        default=None,
        help="Optional synthetic occlusion window START:END, e.g. 30:120. Hides goal mask from policy/belief.",
    )
    ap.add_argument(
        "--mask-dropout-prob",
        type=float,
        default=0.0,
        help="Per-frame probability of hiding the goal mask from policy/belief.",
    )
    ap.add_argument("--success-only", action="store_true", help="Discard failed rollouts instead of saving them")
    args = ap.parse_args()

    sim_modules = import_sim_modules(Path(args.sim_root))
    planner2 = sim_modules["planner2"]
    planner_rrt = sim_modules["planner_rrt"]
    scene_gen = sim_modules["scene_gen"]
    obstacle_gen = sim_modules.get("gen_obstacle_dataset")
    if args.scene_mode == "obstacle" and args.obstacle_pool == "large-footprint":
        if obstacle_gen is None:
            raise RuntimeError("large-footprint obstacle pool requires gen_obstacle_dataset.py in --sim-root")
        install_large_obstacle_pool(
            obstacle_gen,
            min_side=args.large_obstacle_min_side,
            min_footprint=args.large_obstacle_min_footprint,
            excluded_categories=parse_category_list(args.large_obstacle_exclude),
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[WARN] CUDA unavailable; using CPU", flush=True)

    model, train_args = load_model(Path(args.ckpt), device=device, weights=args.weights)
    modes = resolve_modes(args, train_args)
    if modes["action_mode"] == "waypoint":
        raise ValueError(
            "Closed-loop execution for action_waypoint is ambiguous. "
            "Train/evaluate this rollout script with --habitat-action-mode action3d."
        )

    image_size = args.image_size
    if image_size is None:
        image_size = int(train_args.get("image_size", 224))
    use_obstacle_channel = resolve_obstacle_channel(args, train_args)

    stop_dist = args.stop_dist
    if stop_dist is None:
        # The planner's STOP_DIST_M (~0.6 m) is tighter than the navmesh lets the
        # robot reach for many goal objects: the object body occupies that space, so
        # the robot's closest approach is often ~0.6-0.9 m and success NEVER fires --
        # it overshoots its closest point and drifts/rams. Use a more lenient default
        # success radius so arriving at the object counts as success.
        stop_dist = max(float(getattr(planner_rrt, "STOP_DIST_M", 0.6)), 1.0)
    print(f"Success radius (stop_dist): {stop_dist:.2f} m", flush=True)

    categories = [normalize_category(c) for c in (args.categories or GOAL_CATEGORIES)]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    seed = int(args.seed_base)
    t0 = time.time()

    print(f"Policy rollout output: {out_root}", flush=True)
    print(
        "Modes: "
        f"weights={args.weights} action={modes['action_mode']} proprio={modes['proprio_mode']} "
        f"yaw_axis={modes['yaw_axis']} image_size={image_size} sample_steps={args.sample_steps} "
        f"obstacle_channel={use_obstacle_channel} scene_mode={args.scene_mode}",
        flush=True,
    )
    print(f"Categories: {', '.join(categories)}", flush=True)

    for category in categories:
        cat_slug = slugify(category)
        cat_dir = out_root / cat_slug
        scenes_dir = cat_dir / "scenes"
        cat_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        tries = 0
        while saved < args.episodes_per_category and tries < args.episodes_per_category * args.max_tries_per_episode:
            seed += 1
            tries += 1
            rng = np.random.default_rng(seed)
            ep_name = f"ep_{saved:04d}"
            spec_path = None
            spawn_override = None
            sim = None
            try:
                if args.scene_mode == "obstacle":
                    if obstacle_gen is None:
                        raise RuntimeError(
                            "scene-mode=obstacle requires gen_obstacle_dataset.py in --sim-root"
                        )
                    spec_path, spawn_override = obstacle_gen._build_obstacle_scene(
                        seed,
                        category,
                        rng,
                        str(scenes_dir),
                    )
                else:
                    spec_path = scene_gen.build_scene(seed, goal_category=category, out_dir=str(scenes_dir))
                sim, scene_spec = scene_gen.load_scene(
                    spec_path,
                    camera_height=0.2,
                    resolution=(args.resolution, args.resolution),
                )
                rec = rollout_one_episode(
                    sim=sim,
                    scene_spec=scene_spec,
                    category=category,
                    rng=rng,
                    planner2=planner2,
                    model=model,
                    device=device,
                    image_size=image_size,
                    action_mode=modes["action_mode"],
                    proprio_mode=modes["proprio_mode"],
                    yaw_axis=modes["yaw_axis"],
                    sample_steps=args.sample_steps,
                    hz=args.hz,
                    max_steps=args.max_steps,
                    stop_dist=stop_dist,
                    arrival_brake_dist=args.arrival_brake_dist,
                    arrival_min_speed_frac=args.arrival_min_speed_frac,
                    distance_min=args.distance_min,
                    distance_max=args.distance_max,
                    min_start_visible_pixels=args.min_start_visible_pixels,
                    min_visible_pixels=args.min_visible_pixels,
                    hide_mask_window=parse_window(args.hide_mask_window),
                    mask_dropout_prob=args.mask_dropout_prob,
                    max_forward_speed=args.max_forward_speed,
                    max_lateral_speed=args.max_lateral_speed,
                    max_yaw_rate=args.max_yaw_rate,
                    obstacle_mask_dilate=args.obstacle_mask_dilate,
                    obstacle_safety_override=args.obstacle_safety_override,
                    obstacle_safety_min_px=args.obstacle_safety_min_px,
                    obstacle_safety_center_frac=args.obstacle_safety_center_frac,
                    obstacle_safety_close_depth=args.obstacle_safety_close_depth,
                    obstacle_safety_forward=args.obstacle_safety_forward,
                    obstacle_safety_lateral=args.obstacle_safety_lateral,
                    obstacle_safety_yaw=args.obstacle_safety_yaw,
                    motion_convention=args.motion_convention,
                    use_navmesh_step=not args.no_navmesh_step,
                    spawn_override=spawn_override,
                    use_obstacle_channel=use_obstacle_channel,
                    disable_belief=args.disable_belief,
                    action_smoothing=args.action_smoothing,
                    ensemble_decay=args.ensemble_decay,
                    ema_alpha=args.ema_alpha,
                    fixed_episode_noise=args.fixed_episode_noise,
                    cbf=args.cbf,
                    cbf_mode=args.cbf_mode,
                    cbf_deadzone=args.cbf_deadzone,
                    cbf_trust=args.cbf_trust,
                    cbf_proj_iters=args.cbf_proj_iters,
                    cbf_proj_lr=args.cbf_proj_lr,
                    cbf_cone_margin=args.cbf_cone_margin,
                    cbf_smooth=args.cbf_smooth,
                    cbf_keep_speed=args.cbf_keep_speed,
                    cbf_metric=args.cbf_metric,
                    cbf_cov_base=args.cbf_cov_base,
                    cbf_cov_growth=args.cbf_cov_growth,
                    cbf_cov_mode=args.cbf_cov_mode,
                    goal_radius=args.goal_radius,
                    lost_goal_ghost=args.lost_goal_ghost,
                    ghost_mask_radius=args.ghost_mask_radius,
                    ghost_hfov_deg=args.ghost_hfov_deg,
                    zero_lateral=args.zero_lateral,
                    cbf_d_safe=args.cbf_d_safe,
                    cbf_gamma=args.cbf_gamma,
                    cbf_guidance_scale=args.cbf_guidance_scale,
                    cbf_steps=args.cbf_steps,
                    cbf_vel_scale=args.cbf_vel_scale,
                    cbf_goal_attract=args.cbf_goal_attract,
                    cbf_return_scale=args.cbf_return_scale,
                    cbf_takeover_attract=args.cbf_takeover_attract,
                    cbf_tangential=args.cbf_tangential,
                    cbf_block_range=args.cbf_block_range,
                    cbf_tangential_speed=args.cbf_tangential_speed,
                    cbf_heading=args.cbf_heading,
                )
            except Exception as exc:
                print(f"  [{category}] seed {seed} error: {exc}", flush=True)
                rec = None
            finally:
                if sim is not None:
                    sim.close()

            if rec is None:
                continue
            if not rec["success"] and args.success_only:
                print(
                    f"[{category}] {ep_name}: failed, not saved "
                    f"(steps={len(rec['rgb'])} min_dist={rec['min_goal_distance']:.2f}m seed={seed})",
                    flush=True,
                )
                continue

            npz_path = cat_dir / f"{ep_name}.npz"
            save_npz(npz_path, rec, category, seed, args.resolution, args.hz)
            bbox_path = cat_dir / f"{ep_name}_bboxes.json"
            save_bboxes(bbox_path, ep_name, category, seed, args.resolution, rec["bboxes"])

            manifest_row = {
                "episode": ep_name,
                "category": category,
                "npz_path": str(npz_path.relative_to(out_root)),
                "bbox_path": str(bbox_path.relative_to(out_root)),
                "scene_path": relative_path_if_possible(Path(spec_path), out_root) if spec_path else None,
                "scene_seed": seed,
                "resolution": args.resolution,
                "num_steps": int(len(rec["rgb"])),
                "success": bool(rec["success"]),
                "final_goal_distance": round(float(rec["final_goal_distance"]), 4),
                "min_goal_distance": round(float(rec["min_goal_distance"]), 4),
                "frame0_goal_pixels": int(rec["true_goal_visible_pixels"][0]),
                "frame0_observed_goal_pixels": int(rec["goal_visible_pixels"][0]),
                "synthetic_occlusion_frames": int(np.sum(rec["synthetic_occlusion"])),
                "obstacle_visible_frames": int(np.sum(rec["obstacle_visible_pixels"] > 0)),
                "obstacle_safety_frames": int(np.sum(rec["obstacle_safety_active"])),
                "scene_mode": args.scene_mode,
            }
            manifest.append(manifest_row)
            print(
                f"[{category}] {ep_name}: success={rec['success']} "
                f"steps={len(rec['rgb'])} final={rec['final_goal_distance']:.2f}m "
                f"min={rec['min_goal_distance']:.2f}m seed={seed}",
                flush=True,
            )
            saved += 1

        if saved < args.episodes_per_category:
            print(
                f"[{category}] WARNING: saved {saved}/{args.episodes_per_category} "
                f"after {tries} attempts",
                flush=True,
            )

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = (time.time() - t0) / 60.0
    print(f"\nDone: {len(manifest)} policy rollouts", flush=True)
    print(f"Output: {out_root}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Elapsed: {elapsed:.1f} min", flush=True)


def import_sim_modules(sim_root: Path) -> Dict[str, object]:
    sim_root = sim_root.resolve()
    if not sim_root.exists():
        raise FileNotFoundError(f"--sim-root does not exist: {sim_root}")
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    modules = {}
    for name in ["assets_catalog", "planner2", "planner_rrt", "scene_gen"]:
        modules[name] = importlib.import_module(name)
    try:
        modules["gen_obstacle_dataset"] = importlib.import_module("gen_obstacle_dataset")
    except ModuleNotFoundError:
        pass
    return modules


def install_large_obstacle_pool(
    obstacle_gen,
    *,
    min_side: float,
    min_footprint: float,
    excluded_categories: set[str],
) -> None:
    """Make gen_obstacle_dataset pick blockers by physical bbox, not size_tag.

    The original generator calls scene_gen._small_obstacle_pool(...) internally.
    We replace only that pool selector. Scene creation, path placement, expert
    planning, and recording stay exactly in the user's generator.
    """

    printed = False

    def large_pool(catalog, exclude_category: str = ""):
        nonlocal printed
        exclude = normalize_category(exclude_category)
        rows = []
        fallback = []
        for obj in catalog.get("objects", []):
            category = normalize_category(str(obj.get("category", "")))
            if category == exclude or category in excluded_categories:
                continue
            if not has_required_asset_fields(obj):
                continue
            dims = bbox_dims(obj.get("bbox"))
            if dims is None:
                continue
            dims_sorted = sorted((float(abs(v)) for v in dims), reverse=True)
            if len(dims_sorted) < 2:
                continue
            side0, side1 = dims_sorted[0], dims_sorted[1]
            footprint = side0 * side1
            item = (footprint, side1, obj)
            fallback.append(item)
            if side1 >= float(min_side) and footprint >= float(min_footprint):
                rows.append(item)

        chosen = rows
        if not chosen:
            fallback.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen = fallback[: max(20, min(len(fallback), 80))]
        if not chosen:
            raise RuntimeError(
                "large-footprint obstacle pool found no usable assets; "
                "relax --large-obstacle-min-side or --large-obstacle-min-footprint"
            )
        out = [obj for _footprint, _side1, obj in chosen]
        cats = sorted({normalize_category(str(o.get("category", ""))) for o in out})
        if not printed:
            print(
                "large-footprint obstacle pool: "
                f"{len(out)} assets across {len(cats)} categories "
                f"(min_side={min_side}, min_footprint={min_footprint})",
                flush=True,
            )
            print("large-footprint categories: " + ", ".join(cats[:24]), flush=True)
            printed = True
        return out

    obstacle_gen.scene_gen._small_obstacle_pool = large_pool


def has_required_asset_fields(obj: Mapping[str, object]) -> bool:
    required = {"dataset", "config_path", "stem", "category", "bbox", "size_tag"}
    return required.issubset(set(obj.keys()))


def bbox_dims(bbox: object) -> Optional[Tuple[float, float, float]]:
    """Best-effort bbox extent parser for the habitat_dataget asset catalog."""
    if bbox is None:
        return None
    if isinstance(bbox, Mapping):
        for key in ("extent", "extents", "size", "sizes", "dims", "dimensions", "scale"):
            if key in bbox:
                arr = np.asarray(bbox[key], dtype=np.float32).reshape(-1)
                if arr.size >= 3 and np.isfinite(arr[:3]).all():
                    return float(abs(arr[0])), float(abs(arr[1])), float(abs(arr[2]))
        mins = None
        maxs = None
        for key in ("min", "mins", "minimum"):
            if key in bbox:
                mins = np.asarray(bbox[key], dtype=np.float32).reshape(-1)
        for key in ("max", "maxs", "maximum"):
            if key in bbox:
                maxs = np.asarray(bbox[key], dtype=np.float32).reshape(-1)
        if mins is not None and maxs is not None and mins.size >= 3 and maxs.size >= 3:
            arr = np.abs(maxs[:3] - mins[:3])
            if np.isfinite(arr).all():
                return float(arr[0]), float(arr[1]), float(arr[2])
        return None

    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size >= 6:
        first = arr[:3]
        second = arr[3:6]
        diff = np.abs(second - first)
        if np.isfinite(diff).all() and float(diff.max()) > 0:
            return float(diff[0]), float(diff[1]), float(diff[2])
    if arr.size >= 3 and np.isfinite(arr[:3]).all():
        return float(abs(arr[0])), float(abs(arr[1])), float(abs(arr[2]))
    return None


def load_model(ckpt_path: Path, device: str, weights: str) -> tuple[S2DiTPolicy, Mapping[str, object]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = dict(ckpt.get("args", {}))
    train_args["spatial_channels"] = int(ckpt.get("spatial_channels", train_args.get("spatial_channels", 2)))
    model = S2DiTPolicy(
        action_dim=int(ckpt.get("action_dim", train_args.get("action_dim", 3))),
        horizon=int(ckpt.get("horizon", train_args.get("horizon", 8))),
        proprio_dim=int(ckpt.get("proprio_dim", train_args.get("proprio_dim", 3))),
        spatial_channels=int(ckpt.get("spatial_channels", train_args.get("spatial_channels", 2))),
        dim=int(train_args.get("dim", 512)),
        encoder_width=int(train_args.get("encoder_width", 64)),
        num_cond_queries=int(train_args.get("cond_queries", 16)),
        dit_depth=int(train_args.get("dit_depth", 8)),
        heads=int(train_args.get("heads", 8)),
        use_belief_bank=True,
        use_obstacle_map=True,
        use_route_token=True,
        use_cocos_source=bool(train_args.get("use_cocos_source", False)),
        belief_dim=int(train_args.get("belief_dim", 11)),
        max_goals=int(train_args.get("max_goals", 16)),
        obstacle_tokens=int(train_args.get("obstacle_tokens", 16)),
        max_route_len=int(train_args.get("max_route_len", 32)),
        cocos_alpha=float(train_args.get("cocos_alpha", 1.0)),
        cocos_beta=float(train_args.get("cocos_beta", 0.2)),
        mean_loss_weight=float(train_args.get("mean_loss_weight", 0.1)),
        normalize_belief=bool(train_args.get("belief_normalize", False)),
    ).to(device)
    state = ckpt.get(weights)
    if state is None:
        fallback = "ema" if weights == "model" else "model"
        state = ckpt.get(fallback)
        if state is None:
            raise KeyError(f"checkpoint has neither {weights!r} nor {fallback!r} weights")
        print(f"[WARN] checkpoint missing {weights}; using {fallback}", flush=True)
    model.load_state_dict(state)
    model.eval()
    return model, train_args


def resolve_modes(args: argparse.Namespace, train_args: Mapping[str, object]) -> Dict[str, str]:
    return {
        "proprio_mode": str(args.habitat_proprio_mode or train_args.get("habitat_proprio_mode", "planar3")),
        "action_mode": str(args.habitat_action_mode or train_args.get("habitat_action_mode", "action3d")),
        "yaw_axis": str(args.habitat_yaw_axis or train_args.get("habitat_yaw_axis", "y")),
    }


def resolve_obstacle_channel(args: argparse.Namespace, train_args: Mapping[str, object]) -> bool:
    if args.habitat_use_obstacle_channel is not None:
        return bool(args.habitat_use_obstacle_channel)
    if bool(train_args.get("habitat_use_obstacle_channel", False)):
        return True
    return int(train_args.get("spatial_channels", 2)) >= 3


def rollout_one_episode(
    *,
    sim,
    scene_spec: Mapping[str, object],
    category: str,
    rng: np.random.Generator,
    planner2,
    model: S2DiTPolicy,
    device: str,
    image_size: int,
    action_mode: str,
    proprio_mode: str,
    yaw_axis: str,
    sample_steps: int,
    hz: float,
    max_steps: int,
    stop_dist: float,
    arrival_brake_dist: float,
    arrival_min_speed_frac: float,
    distance_min: float,
    distance_max: float,
    min_start_visible_pixels: int,
    min_visible_pixels: int,
    hide_mask_window: Optional[Tuple[int, int]],
    mask_dropout_prob: float,
    max_forward_speed: float,
    max_lateral_speed: float,
    max_yaw_rate: float,
    obstacle_mask_dilate: int,
    obstacle_safety_override: bool,
    obstacle_safety_min_px: int,
    obstacle_safety_center_frac: float,
    obstacle_safety_close_depth: float,
    obstacle_safety_forward: float,
    obstacle_safety_lateral: float,
    obstacle_safety_yaw: float,
    motion_convention: str,
    use_navmesh_step: bool,
    spawn_override: Optional[Mapping[str, object]] = None,
    use_obstacle_channel: bool = False,
    disable_belief: bool = False,
    action_smoothing: str = "ensemble",
    ensemble_decay: float = 0.5,
    ema_alpha: float = 0.6,
    fixed_episode_noise: bool = False,
    cbf: bool = False,
    cbf_mode: str = "guidance",
    cbf_deadzone: float = 0.5,
    cbf_trust: float = 0.0,
    cbf_proj_iters: int = 10,
    cbf_proj_lr: float = 0.05,
    cbf_cone_margin: float = 0.05,
    cbf_smooth: float = 0.0,
    cbf_keep_speed: float = 1.0,
    cbf_metric: str = "euclidean",
    cbf_cov_base: float = 1.0,
    cbf_cov_growth: float = 0.6,
    cbf_cov_mode: str = "shrink",
    goal_radius: float = 0.0,
    lost_goal_ghost: bool = False,
    ghost_mask_radius: int = 14,
    ghost_hfov_deg: float = 90.0,
    zero_lateral: bool = False,
    cbf_d_safe: float = 0.5,
    cbf_gamma: float = 0.3,
    cbf_guidance_scale: float = 0.4,
    cbf_steps: int = 2,
    cbf_vel_scale: float = 1.0,
    cbf_goal_attract: float = 0.05,
    cbf_return_scale: float = 0.25,
    cbf_takeover_attract: float = 0.30,
    cbf_tangential: float = 0.5,
    cbf_block_range: float = 1.5,
    cbf_tangential_speed: float = 0.3,
    cbf_heading: float = 0.15,
) -> Optional[Dict[str, object]]:
    planner2._rebake_navmesh_strict(sim)
    if spawn_override is None:
        try:
            spawn = planner2.sample_spawn_with_visible_goal(sim, scene_spec, distance_min, distance_max, rng)
        except RuntimeError:
            return None
    else:
        spawn = dict(spawn_override)

    goal_entry = scene_spec["goal_instances"][spawn["goal_instance_idx"]]
    goal_oid = goal_entry["_object_id"]
    obstacle_oids = [
        int(o["_object_id"])
        for o in scene_spec.get("obstacles", [])
        if isinstance(o, Mapping) and "_object_id" in o
    ]
    goal_pos = np.asarray(goal_entry["position"], dtype=np.float32)
    position = np.asarray(spawn["position"], dtype=np.float32).copy()
    yaw = float(spawn["yaw"])
    dt = 1.0 / float(hz)

    h = w = None
    intrinsics = None
    extractor = None
    obstacle_builder = None
    bank = SubgoalBeliefBank([category], sigma_visible=0.05, odom_noise=0.02)
    prev_pose = None
    smoother = ActionSmoother(
        mode=action_smoothing,
        ensemble_decay=ensemble_decay,
        ema_alpha=ema_alpha,
    )
    episode_noise = None
    if fixed_episode_noise:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
        episode_noise = torch.randn(
            1, int(model.horizon), int(model.action_dim), generator=gen, device=device
        )
    prev_obstacle_point = None  # last frame's nearest obstacle point (CBF relative-velocity)

    rows: Dict[str, list] = {
        "rgb": [],
        "depth": [],
        "pose": [],
        "action_2d": [],
        "action_3d": [],
        "action_waypoint": [],
        "seg_masks": [],
        "true_seg_masks": [],
        "goal_visible_pixels": [],
        "true_goal_visible_pixels": [],
        "obstacle_visible_pixels": [],
        "pred_action_chunk": [],
        "belief_tensor": [],
        "synthetic_occlusion": [],
        "obstacle_safety_active": [],
        "bboxes": [],
        "goal_distance": [],
        "belief_range_err": [],
        "belief_goal_lost": [],
    }
    min_goal_distance = float("inf")
    success = False
    # CBF diagnostics: confirm which guidance paths actually engage each frame.
    diag = {
        "cbf_active": 0,        # obstacle detected -> avoidance branch
        "tangential_active": 0, # wide-obstacle circulation fired
        "takeover_active": 0,   # goal mask lost but belief drove the pull
        "belief_init": 0,       # belief slot was initialised (mu available)
        "return_active": 0,     # obstacle-free belief return guidance
        "min_obstacle_dist": float("inf"),
        # Belief-input accuracy: ||mu|| vs true goal distance.
        "belief_err_sum": 0.0,
        "belief_err_n": 0,
        "belief_lost_err_sum": 0.0,
        "belief_lost_err_n": 0,
    }

    with torch.no_grad():
        for step in range(int(max_steps)):
            planner2._set_agent_pose(sim, position, yaw)
            obs = sim.get_sensor_observations()
            rgb = np.asarray(obs["rgb"][..., :3], dtype=np.uint8)
            depth = np.asarray(obs["depth"], dtype=np.float32)
            object_ids = np.asarray(obs["object_id"])
            true_mask = (object_ids == goal_oid).astype(np.uint8)
            obstacle_mask = _object_mask(object_ids, obstacle_oids)
            obstacle_input_mask = dilate_binary(obstacle_mask, obstacle_mask_dilate)

            if h is None:
                h, w = depth.shape[:2]
                intrinsics = default_intrinsics(h, w)
                extractor = SAMDepthTargetExtractor(
                    intrinsics,
                    min_mask_area=50,
                    depth_scale=1.0,
                    position_dim=2,
                )
                obstacle_builder = DepthObstacleMap(
                    grid_size=96,
                    resolution=0.05,
                    camera_intrinsics=intrinsics,
                    depth_scale=1.0,
                )
                ghost_intr = _gc_intrinsics(h, w, ghost_hfov_deg)

            hide_mask = should_hide_mask(step, hide_mask_window, mask_dropout_prob, rng)
            observed_mask = np.zeros_like(true_mask) if hide_mask else true_mask
            true_visible_px = int(true_mask.sum())
            observed_visible_px = int(observed_mask.sum())
            if step == 0 and true_visible_px < min_start_visible_pixels:
                return None

            pose = pose_from_position_yaw(position, yaw, planner2)
            odom = _odom_delta(prev_pose, pose, planar_axes=(0, 2), yaw_axis=yaw_axis)
            belief_obs = make_belief_observation(
                extractor,
                category,
                observed_mask,
                depth,
                observed_visible_px,
                min_visible_pixels,
            )
            bank.update(belief_obs, odom_delta=odom, step=step)
            prev_pose = pose.copy()
            belief = bank.as_tensor([category], active_goal_id=category, route_index=0, route_length=1).cpu().numpy()

            # Belief-input accuracy check (convention-free): the belief's goal RANGE
            # ||mu|| should match the true planar goal distance. If it stays accurate
            # while the mask is gone, the policy is getting a good belief input and any
            # remaining failure is the policy; if it drifts, the belief feed is the bug.
            mu_range = float(np.linalg.norm(belief[0, :2])) if belief.ndim == 2 else 0.0
            true_range = float(planar_distance(position, goal_pos))
            range_err = abs(mu_range - true_range)
            goal_lost_now = observed_visible_px < min_visible_pixels
            rows["belief_range_err"].append(range_err)
            rows["belief_goal_lost"].append(bool(goal_lost_now))
            diag["belief_err_sum"] += range_err
            diag["belief_err_n"] += 1
            if goal_lost_now:
                diag["belief_lost_err_sum"] += range_err
                diag["belief_lost_err_n"] += 1

            # Lost-goal ghost: the policy is MASK-conditioned (proven this session -- the
            # raw belief tensor barely steers it). When the real goal is out of frame,
            # render the belief as a ghost circle in the goal channel instead, so there is
            # something to steer back to. Frame-corrected (belief lives in a reflected
            # frame vs the motion/camera frame): motion = -90 - 2*yaw - belief_bearing.
            goal_channel = observed_mask
            ghost_active = False
            if lost_goal_ghost and goal_lost_now and bool(belief[0, 6] > 0.5):
                mu = belief[0, :2]
                lb = float(np.arctan2(mu[1], mu[0]))
                lr = float(np.hypot(mu[0], mu[1]))
                max_fov = math.radians(ghost_hfov_deg / 2) * 0.82
                pb = wrap_angle(-math.pi / 2 - 2.0 * yaw - lb)
                pb = math.copysign(min(abs(pb), max_fov), pb)
                gp = _gc_body_point(position, yaw, pb, max(lr, 0.8))
                u_g, v_g, _ = _gc_project(gp, position, yaw, ghost_intr)
                ghost_mask = _gc_make_mask(h, w, u_g if u_g is not None else -1.0,
                                           v_g if v_g is not None else -1.0, ghost_mask_radius)
                if ghost_mask.sum() > 0:
                    goal_channel = ghost_mask
                    ghost_active = True
            diag["ghost_active"] = diag.get("ghost_active", 0) + int(ghost_active)

            spatial = frame_to_spatial(
                depth,
                goal_channel,
                image_size,
                obstacle_mask=obstacle_input_mask,
                include_obstacle_channel=use_obstacle_channel,
            ).to(device)
            proprio = _proprio_from_pose(pose, proprio_mode, (0, 2), yaw_axis)
            proprio_t = torch.from_numpy(proprio[None]).float().to(device)
            obstacle_np = obstacle_builder.build(depth)
            obstacle = torch.from_numpy(obstacle_np[None]).float().to(device)
            # --disable-belief: feed the EMPTY belief tensor so the policy runs on
            # mask + depth + proprio only (the "previous model" baseline, no memory).
            belief_feed = _empty_belief_tensor() if disable_belief else belief
            belief_t = torch.from_numpy(belief_feed[None]).float().to(device)
            route_index = torch.zeros(1, dtype=torch.long, device=device)
            active_goal_index = torch.zeros(1, dtype=torch.long, device=device)

            guidance_fn = None
            proj_obstacle_point = None   # for post-hoc project mode
            proj_v_o = None
            if cbf:
                # Obstacle relative state from the segmented mask + depth (no grid).
                obstacle_point = nearest_obstacle_point(obstacle_mask, depth, intrinsics)
                # belief is [N_goals, 11]; belief[0, :2] = [mu_x, mu_y] of active slot.
                # Only seek mu when the slot is initialised (initialized flag = belief[0, 6]).
                _mu = belief[0, :2] if belief.ndim == 2 else None
                _initialized = bool(belief[0, 6] > 0.5) if (_mu is not None) else False
                mu_goal = _mu if _initialized else None

                # Belief takeover: when the goal MASK is lost but the belief still
                # holds the goal, strengthen the pull so the belief drives the final
                # approach instead of the policy stalling/spinning at the obstacle.
                goal_lost = observed_visible_px < min_visible_pixels
                attract_weight = (
                    cbf_takeover_attract if (goal_lost and mu_goal is not None) else cbf_goal_attract
                )

                # Diagnostics.
                if mu_goal is not None:
                    diag["belief_init"] += 1
                if goal_lost and mu_goal is not None:
                    diag["takeover_active"] += 1
                if obstacle_point is not None:
                    diag["cbf_active"] += 1
                    diag["min_obstacle_dist"] = min(
                        diag["min_obstacle_dist"], float(np.linalg.norm(obstacle_point))
                    )
                    proj_v_o = estimate_obstacle_velocity(prev_obstacle_point, obstacle_point, odom, dt)
                    proj_obstacle_point = obstacle_point

                # Only the ORIGINAL mode builds a during-sampling guidance hook.
                # project mode samples clean, then projects the output (below).
                if cbf_mode == "guidance":
                    if obstacle_point is not None:
                        t_hat = (
                            tangential_around_obstacle(obstacle_point, mu_goal, block_range=cbf_block_range)
                            if (mu_goal is not None and cbf_tangential > 0.0)
                            else None
                        )
                        if t_hat is not None:
                            diag["tangential_active"] += 1
                        guidance_fn = build_cbf_guidance(
                            p0=obstacle_point,
                            v_o=proj_v_o,
                            d_safe=cbf_d_safe,
                            gamma=cbf_gamma,
                            dt=dt,
                            vel_scale=cbf_vel_scale,
                            guidance_scale=cbf_guidance_scale,
                            n_steps=cbf_steps,
                            mu_goal=mu_goal,
                            goal_attract_weight=attract_weight,
                            tangential=t_hat,
                            tangential_weight=(cbf_tangential if t_hat is not None else 0.0),
                            tangential_target=cbf_tangential_speed,
                            heading_weight=cbf_heading,
                        )
                    elif mu_goal is not None and attract_weight > 0.0:
                        diag["return_active"] += 1
                        guidance_fn = build_cbf_guidance(
                            p0=None,
                            dt=dt,
                            vel_scale=cbf_vel_scale,
                            guidance_scale=cbf_return_scale,
                            n_steps=cbf_steps,
                            mu_goal=mu_goal,
                            goal_attract_weight=attract_weight,
                            heading_weight=cbf_heading,
                        )
                prev_obstacle_point = obstacle_point

            pred = model.sample(
                spatial,
                proprio_t,
                steps=sample_steps,
                belief_tensor=belief_t,
                obstacle_map=obstacle,
                route_index=route_index,
                active_goal_index=active_goal_index,
                noise=episode_noise,
                guidance_fn=guidance_fn,
            )

            # cone mode: POST-HOC project the whole 8-step chunk OUT of the collision
            # cone (C3BF). For v_lat=0 we zero the lateral column first so the rollout
            # is the true diff-drive model and the projection steers via YAW, not strafe.
            if cbf and cbf_mode == "cone" and proj_obstacle_point is not None:
                if zero_lateral and pred.shape[-1] >= 3:
                    pred = pred.clone()
                    pred[..., 1] = 0.0
                # Which way to go around: away from the side the obstacle leans; if it's
                # ~dead-centre, prefer the belief-goal side; else default right.
                p_lat = float(proj_obstacle_point[1])
                if abs(p_lat) > 0.1:
                    cone_side = -1.0 if p_lat > 0.0 else 1.0
                elif mu_goal is not None:
                    cone_side = 1.0 if float(mu_goal[1]) > 0.0 else -1.0
                else:
                    cone_side = -1.0
                # Mahalanobis metric: horizon-growing covariance so the correction is
                # spent on the UNCERTAIN far-horizon actions, preserving the confident
                # near-term action (smoother, novel projection).
                cone_sigma = None
                if cbf_metric == "mahalanobis":
                    cone_sigma = horizon_growth_covariance(
                        pred.shape[1], pred.shape[2], base=cbf_cov_base, growth=cbf_cov_growth,
                        mode=cbf_cov_mode, device=pred.device, dtype=pred.dtype,
                    )
                pred = project_chunk_cone(
                    pred,
                    proj_obstacle_point,
                    proj_v_o,
                    r=cbf_d_safe,
                    dt=dt,
                    vel_scale=cbf_vel_scale,
                    iters=cbf_proj_iters,
                    lr=cbf_proj_lr,
                    trust=(cbf_trust if cbf_trust > 0.0 else 0.3),
                    margin=cbf_cone_margin,
                    smooth_weight=cbf_smooth,
                    keep_speed=cbf_keep_speed,
                    sigma=cone_sigma,
                    deadzone_range=cbf_d_safe + cbf_deadzone,
                    side=cone_side,
                )
                diag["cbf_project_active"] = diag.get("cbf_project_active", 0) + 1

            pred_chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
            chunk_ctrl = np.stack(
                [
                    action_to_control(
                        pred_chunk[k],
                        action_mode=action_mode,
                        max_forward_speed=max_forward_speed,
                        max_lateral_speed=max_lateral_speed,
                        max_yaw_rate=max_yaw_rate,
                    )
                    for k in range(pred_chunk.shape[0])
                ],
                axis=0,
            )
            smoother.add(step, chunk_ctrl)
            action_3d = smoother.get(step)

            # Force v_lat=0 -> differential-drive (forward + yaw only).
            if zero_lateral and action_3d.shape[0] >= 2:
                action_3d = action_3d.copy()
                action_3d[1] = 0.0

            # Post-hoc CBF projection (Step 1): sample was clean (no during-sampling
            # guidance in project mode); brake forward speed near a head-on obstacle.
            # Yaw is never touched here, so the CBF cannot cause yaw thrash.
            if cbf and cbf_mode == "project" and proj_obstacle_point is not None:
                action_3d, _braked = project_forward_velocity_cbf(
                    action_3d,
                    proj_obstacle_point,
                    proj_v_o,
                    d_safe=cbf_d_safe,
                    gamma=cbf_gamma,
                    deadzone=cbf_deadzone,
                    trust=(cbf_trust if cbf_trust > 0.0 else None),
                )
                if _braked:
                    diag["cbf_project_active"] = diag.get("cbf_project_active", 0) + 1

            safety = obstacle_safety_action(
                action_3d,
                obstacle_mask,
                depth,
                enabled=obstacle_safety_override,
                min_px=obstacle_safety_min_px,
                center_frac=obstacle_safety_center_frac,
                close_depth=obstacle_safety_close_depth,
                forward_speed=obstacle_safety_forward,
                lateral_speed=obstacle_safety_lateral,
                yaw_rate=obstacle_safety_yaw,
            )
            if safety["active"]:
                action_3d = safety["action"]

            # Arrival brake: ease forward/lateral speed to ~0 as we approach the goal
            # so the robot stops AT stop_dist instead of ramming onto the goal object.
            if arrival_brake_dist > 0.0:
                cur_goal_dist = planar_distance(position, goal_pos) - goal_radius  # surface distance
                if cur_goal_dist < arrival_brake_dist:
                    span = max(arrival_brake_dist - stop_dist, 1e-3)
                    frac = float(np.clip((cur_goal_dist - stop_dist) / span, arrival_min_speed_frac, 1.0))
                    action_3d = action_3d.copy()
                    action_3d[0] *= frac
                    if action_3d.shape[0] >= 2:
                        action_3d[1] *= frac

            next_position, next_yaw = integrate_action(
                position,
                yaw,
                action_3d,
                dt,
                convention=motion_convention,
            )
            if use_navmesh_step:
                next_position = navmesh_try_step(sim, position, next_position)
            next_waypoint = next_position.astype(np.float32)

            goal_distance = planar_distance(position, goal_pos)
            min_goal_distance = min(min_goal_distance, goal_distance)

            rows["rgb"].append(rgb.copy())
            rows["depth"].append(depth.copy())
            rows["pose"].append(pose.astype(np.float32))
            rows["action_3d"].append(action_3d.astype(np.float32))
            rows["action_2d"].append(np.asarray([action_3d[0], action_3d[2]], dtype=np.float32))
            rows["action_waypoint"].append(next_waypoint)
            rows["seg_masks"].append(_combined_seg(observed_mask, obstacle_mask))
            rows["true_seg_masks"].append(_combined_seg(true_mask, obstacle_mask))
            rows["goal_visible_pixels"].append(observed_visible_px)
            rows["true_goal_visible_pixels"].append(true_visible_px)
            rows["obstacle_visible_pixels"].append(int(obstacle_mask.sum()))
            rows["pred_action_chunk"].append(pred_chunk)
            rows["belief_tensor"].append(belief.astype(np.float32))
            rows["synthetic_occlusion"].append(bool(hide_mask))
            rows["obstacle_safety_active"].append(bool(safety["active"]))
            rows["bboxes"].append(
                {
                    "goal": mask_to_bbox(observed_mask, step),
                    "obstacle": mask_to_bbox(obstacle_mask, step),
                }
            )
            rows["goal_distance"].append(goal_distance)

            # Success from the goal SURFACE, not its centre: subtract the object radius.
            if (goal_distance - goal_radius) <= stop_dist:
                success = True
                break

            position = next_position.astype(np.float32)
            yaw = wrap_angle(next_yaw)

    if not rows["rgb"]:
        return None

    n_frames = len(rows["rgb"])
    if cbf:
        mind = diag["min_obstacle_dist"]
        mind_str = f"{mind:.2f}m" if mind != float("inf") else "never"
        print(
            f"    [CBF diag] frames={n_frames} belief_init={diag['belief_init']} "
            f"cbf_active={diag['cbf_active']} tangential={diag['tangential_active']} "
            f"takeover={diag['takeover_active']} return={diag['return_active']} "
            f"ghost_active={diag.get('ghost_active', 0)} min_obst_dist={mind_str}",
            flush=True,
        )
    # Belief-input accuracy: low err while goal is lost => good input, fault is the policy.
    avg_err = diag["belief_err_sum"] / max(diag["belief_err_n"], 1)
    lost_err = diag["belief_lost_err_sum"] / max(diag["belief_lost_err_n"], 1)
    print(
        f"    [belief diag] range_err avg={avg_err:.2f}m | "
        f"goal_lost_frames={diag['belief_lost_err_n']} "
        f"range_err_while_lost={lost_err:.2f}m",
        flush=True,
    )

    final_goal_distance = float(rows["goal_distance"][-1])
    return {
        "rgb": np.stack(rows["rgb"]).astype(np.uint8),
        "depth": np.stack(rows["depth"]).astype(np.float32),
        "pose": np.stack(rows["pose"]).astype(np.float32),
        "action_2d": np.stack(rows["action_2d"]).astype(np.float32),
        "action_3d": np.stack(rows["action_3d"]).astype(np.float32),
        "action_waypoint": np.stack(rows["action_waypoint"]).astype(np.float32),
        "seg_masks": np.stack(rows["seg_masks"]).astype(np.uint8),
        "true_seg_masks": np.stack(rows["true_seg_masks"]).astype(np.uint8),
        "goal_visible_pixels": np.asarray(rows["goal_visible_pixels"], dtype=np.int32),
        "true_goal_visible_pixels": np.asarray(rows["true_goal_visible_pixels"], dtype=np.int32),
        "obstacle_visible_pixels": np.asarray(rows["obstacle_visible_pixels"], dtype=np.int32),
        "pred_action_chunk": np.stack(rows["pred_action_chunk"]).astype(np.float32),
        "belief_tensor": np.stack(rows["belief_tensor"]).astype(np.float32),
        "synthetic_occlusion": np.asarray(rows["synthetic_occlusion"], dtype=bool),
        "obstacle_safety_active": np.asarray(rows["obstacle_safety_active"], dtype=bool),
        "goal_distance": np.asarray(rows["goal_distance"], dtype=np.float32),
        "belief_range_err": np.asarray(rows["belief_range_err"], dtype=np.float32),
        "belief_goal_lost": np.asarray(rows["belief_goal_lost"], dtype=bool),
        "bboxes": rows["bboxes"],
        "success": bool(success),
        "final_goal_distance": final_goal_distance,
        "min_goal_distance": float(min_goal_distance),
        "spawn_distance": float(spawn.get("initial_distance", planar_distance(np.asarray(spawn["position"]), goal_pos))),
        "num_goals": int(len(scene_spec.get("goal_instances", []))),
        "num_obstacles": int(len(scene_spec.get("obstacles", []))),
        "obstacle_category": str(scene_spec.get("obstacles", [{}])[0].get("category", "")) if scene_spec.get("obstacles") else "",
        "obstacle_position": np.asarray(scene_spec.get("obstacles", [{}])[0].get("position", [np.nan, np.nan, np.nan]), dtype=np.float32) if scene_spec.get("obstacles") else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
    }


def frame_to_spatial(
    depth: np.ndarray,
    mask: np.ndarray,
    image_size: int,
    obstacle_mask: Optional[np.ndarray] = None,
    include_obstacle_channel: bool = False,
) -> torch.Tensor:
    depth_r, mask_r = _resize_depth_mask(depth, mask, (image_size, image_size))
    depth_norm = _normalize_depth(depth_r)
    mask_float = (mask_r > 0).astype(np.float32)
    parts = [mask_float]
    if include_obstacle_channel:
        if obstacle_mask is None:
            obstacle_r = np.zeros_like(mask_r)
        else:
            _, obstacle_r = _resize_depth_mask(depth, obstacle_mask, (image_size, image_size))
        parts.append((obstacle_r > 0).astype(np.float32))
    parts.append(depth_norm)
    spatial = np.stack(parts, axis=0).astype(np.float32)
    return torch.from_numpy(spatial[None]).float()


def make_belief_observation(
    extractor: SAMDepthTargetExtractor,
    category: str,
    mask: np.ndarray,
    depth: np.ndarray,
    visible_pixels: int,
    min_visible_pixels: int,
) -> Dict[str, Dict[str, object]]:
    if visible_pixels < min_visible_pixels or int(mask.sum()) < min_visible_pixels:
        return {category: {"visible": False, "position": None, "confidence": 0.0}}
    return extractor.extract({category: mask.astype(bool)}, depth)


def _object_mask(object_ids: np.ndarray, object_oids: Sequence[int]) -> np.ndarray:
    if not object_oids:
        return np.zeros_like(object_ids, dtype=np.uint8)
    return np.isin(object_ids, np.asarray(object_oids, dtype=object_ids.dtype)).astype(np.uint8)


def _combined_seg(goal_mask: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    seg = np.zeros_like(goal_mask, dtype=np.uint8)
    seg[np.asarray(goal_mask) > 0] = 1
    seg[np.asarray(obstacle_mask) > 0] = 2
    return seg


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    arr = (np.asarray(mask) > 0).astype(np.uint8)
    if radius <= 0 or not arr.any():
        return arr
    padded = np.pad(arr, radius, mode="constant")
    out = np.zeros_like(arr)
    k = 2 * radius + 1
    for dy in range(k):
        for dx in range(k):
            out = np.maximum(out, padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]])
    return out.astype(np.uint8)


def obstacle_safety_action(
    action: np.ndarray,
    obstacle_mask: np.ndarray,
    depth: np.ndarray,
    *,
    enabled: bool,
    min_px: int,
    center_frac: float,
    close_depth: float,
    forward_speed: float,
    lateral_speed: float,
    yaw_rate: float,
) -> Dict[str, object]:
    current = np.asarray(action, dtype=np.float32).copy()
    if not enabled:
        return {"active": False, "action": current}

    mask = np.asarray(obstacle_mask) > 0
    ys, xs = np.where(mask)
    if xs.size < int(min_px):
        return {"active": False, "action": current}

    depth_vals = np.asarray(depth, dtype=np.float32)[mask]
    depth_vals = depth_vals[np.isfinite(depth_vals) & (depth_vals > 0)]
    median_depth = float(np.median(depth_vals)) if depth_vals.size else float("inf")
    if median_depth > float(close_depth):
        return {"active": False, "action": current}

    h, w = mask.shape
    cx = float(np.median(xs))
    x_norm = (cx - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)
    if abs(x_norm) > float(center_frac):
        return {"active": False, "action": current}

    # Image-right obstacle -> dodge left (+v_lat). Image-left -> dodge right.
    dodge_sign = 1.0 if x_norm >= 0.0 else -1.0
    current[0] = min(float(current[0]), float(forward_speed))
    if current.shape[0] >= 2:
        current[1] = dodge_sign * float(lateral_speed)
    if current.shape[0] >= 3:
        current[2] = dodge_sign * float(yaw_rate)
    return {"active": True, "action": current.astype(np.float32)}


class ActionSmoother:
    """Temporally smooth executed velocity commands across control steps.

    The policy re-samples an independent H-step chunk every step and only the
    first action was executed, so consecutive commands jumped between unrelated
    diffusion samples -> jerky motion. This buffers each chunk's predictions by
    absolute time and, for the current step, returns an exponentially weighted
    average of every still-valid prediction that covers it (ACT-style temporal
    ensembling), or a simple EMA low-pass, or the raw first action.
    """

    def __init__(self, mode: str = "ensemble", ensemble_decay: float = 0.5, ema_alpha: float = 0.6):
        self.mode = str(mode)
        self.decay = float(ensemble_decay)
        self.alpha = float(ema_alpha)
        self.buffer: Dict[int, list] = {}
        self.prev: Optional[np.ndarray] = None
        self.latest: Optional[np.ndarray] = None

    def add(self, origin_step: int, chunk_ctrl: np.ndarray) -> None:
        chunk = np.asarray(chunk_ctrl, dtype=np.float32)
        self.latest = chunk[0].copy()
        if self.mode == "ensemble":
            for k in range(chunk.shape[0]):
                self.buffer.setdefault(int(origin_step) + k, []).append((int(origin_step), chunk[k].copy()))

    def get(self, step: int) -> np.ndarray:
        if self.mode == "ensemble":
            preds = self.buffer.get(int(step), [])
            if preds:
                ages = np.asarray([step - origin for origin, _ in preds], dtype=np.float32)
                weights = np.exp(-self.decay * ages)
                weights = weights / float(weights.sum())
                acts = np.stack([a for _, a in preds], axis=0)
                out = (weights[:, None] * acts).sum(axis=0).astype(np.float32)
            else:
                out = self.latest.copy()
            for past_t in [t for t in self.buffer if t < int(step)]:
                del self.buffer[past_t]
        elif self.mode == "ema":
            if self.prev is None:
                out = self.latest.copy()
            else:
                out = (self.alpha * self.prev + (1.0 - self.alpha) * self.latest).astype(np.float32)
        else:
            out = self.latest.copy()
        self.prev = out
        return out.astype(np.float32)


def action_to_control(
    action: np.ndarray,
    *,
    action_mode: str,
    max_forward_speed: float,
    max_lateral_speed: float,
    max_yaw_rate: float,
) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_mode == "action3d":
        if a.shape[0] < 3:
            raise ValueError("action3d policy output must have at least 3 values")
        v_fwd, v_lat, yaw_rate = float(a[0]), float(a[1]), float(a[2])
    elif action_mode == "action2d":
        if a.shape[0] < 2:
            raise ValueError("action2d policy output must have at least 2 values")
        v_fwd, v_lat, yaw_rate = float(a[0]), 0.0, float(a[1])
    else:
        raise ValueError(f"cannot execute action_mode={action_mode!r}")
    return np.asarray(
        [
            np.clip(v_fwd, -max_forward_speed, max_forward_speed),
            np.clip(v_lat, -max_lateral_speed, max_lateral_speed),
            np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate),
        ],
        dtype=np.float32,
    )


def integrate_action(
    position: np.ndarray,
    yaw: float,
    action_3d: np.ndarray,
    dt: float,
    convention: str,
) -> tuple[np.ndarray, float]:
    v_fwd, v_lat, yaw_rate = [float(x) for x in action_3d]
    fwd_x, fwd_z, left_x, left_z = motion_basis(yaw, convention)
    delta_x = (fwd_x * v_fwd + left_x * v_lat) * dt
    delta_z = (fwd_z * v_fwd + left_z * v_lat) * dt
    out = np.asarray(position, dtype=np.float32).copy()
    out[0] += delta_x
    out[2] += delta_z
    return out, float(yaw + yaw_rate * dt)


def motion_basis(yaw: float, convention: str) -> tuple[float, float, float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    if convention == "habitat":
        # Habitat agents commonly face local -Z at yaw=0.
        return -s, -c, -c, s
    if convention == "x-forward":
        return c, s, -s, c
    if convention == "z-forward":
        return s, c, -c, s
    raise ValueError(f"unknown motion convention: {convention}")


def navmesh_try_step(sim, position: np.ndarray, next_position: np.ndarray) -> np.ndarray:
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None:
        return next_position
    try:
        stepped = pathfinder.try_step(np.asarray(position, dtype=np.float32), np.asarray(next_position, dtype=np.float32))
        return np.asarray(stepped, dtype=np.float32)
    except Exception:
        return next_position


def pose_from_position_yaw(position: np.ndarray, yaw: float, planner2) -> np.ndarray:
    q = planner2._np_quat_from_yaw(float(yaw))
    p = np.asarray(position, dtype=np.float32)
    return np.asarray([p[0], p[1], p[2], q.x, q.y, q.z, q.w], dtype=np.float32)


def should_hide_mask(
    step: int,
    window: Optional[Tuple[int, int]],
    dropout_prob: float,
    rng: np.random.Generator,
) -> bool:
    if window is not None:
        start, end = window
        if start <= step < end:
            return True
    p = float(dropout_prob)
    if p <= 0.0:
        return False
    return bool(rng.random() < p)


def parse_window(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("--hide-mask-window must be START:END")
    start, end = int(parts[0]), int(parts[1])
    if end <= start:
        raise ValueError("--hide-mask-window END must be greater than START")
    return start, end


def mask_to_bbox(mask: np.ndarray, step: int) -> Dict[str, object]:
    rows, cols = np.where(np.asarray(mask) > 0)
    if rows.size == 0:
        return {
            "step": int(step),
            "visible": False,
            "x_min": -1,
            "y_min": -1,
            "x_max": -1,
            "y_max": -1,
            "pixel_count": 0,
        }
    return {
        "step": int(step),
        "visible": True,
        "x_min": int(cols.min()),
        "y_min": int(rows.min()),
        "x_max": int(cols.max()),
        "y_max": int(rows.max()),
        "pixel_count": int(rows.size),
    }


def save_npz(path: Path, rec: Mapping[str, object], category: str, seed: int, resolution: int, hz: float) -> None:
    np.savez_compressed(
        path,
        rgb=rec["rgb"],
        depth=rec["depth"],
        pose=rec["pose"],
        action_2d=rec["action_2d"],
        action_3d=rec["action_3d"],
        action_waypoint=rec["action_waypoint"],
        seg_masks=rec["seg_masks"],
        true_seg_masks=rec["true_seg_masks"],
        goal_visible_pixels=rec["goal_visible_pixels"],
        true_goal_visible_pixels=rec["true_goal_visible_pixels"],
        obstacle_visible_pixels=rec["obstacle_visible_pixels"],
        pred_action_chunk=rec["pred_action_chunk"],
        belief_tensor=rec["belief_tensor"],
        synthetic_occlusion=rec["synthetic_occlusion"],
        obstacle_safety_active=rec["obstacle_safety_active"],
        goal_distance=rec["goal_distance"],
        goal_category=category,
        scene_seed=seed,
        hz=float(hz),
        resolution=int(resolution),
        success=bool(rec["success"]),
        final_goal_distance=float(rec["final_goal_distance"]),
        min_goal_distance=float(rec["min_goal_distance"]),
        spawn_distance=float(rec["spawn_distance"]),
        num_goals=int(rec["num_goals"]),
        num_obstacles=int(rec["num_obstacles"]),
        obstacle_category=str(rec.get("obstacle_category", "")),
        obstacle_position=np.asarray(rec.get("obstacle_position", [np.nan, np.nan, np.nan]), dtype=np.float32),
    )


def save_bboxes(
    path: Path,
    episode: str,
    category: str,
    seed: int,
    resolution: int,
    bboxes: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w") as f:
        json.dump(
            {
                "episode": episode,
                "category": category,
                "scene_seed": seed,
                "image_wh": [resolution, resolution],
                "bboxes": list(bboxes),
            },
            f,
            indent=2,
        )


def default_intrinsics(height: int, width: int) -> Dict[str, float]:
    f = float(max(height, width))
    return {"fx": f, "fy": f, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(aa[[0, 2]] - bb[[0, 2]]))


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def slugify(category: str) -> str:
    return category.replace(" ", "_")


def normalize_category(category: str) -> str:
    return category.replace("_", " ")


def parse_category_list(value: str) -> set[str]:
    return {
        normalize_category(item.strip())
        for item in str(value).split(",")
        if item.strip()
    }


def relative_path_if_possible(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()