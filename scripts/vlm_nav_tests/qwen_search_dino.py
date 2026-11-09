"""Same manual search-and-handoff as qwen_search_rollout.py, but `found` only
flips True once GroundingDINO visually confirms a "flag"-labeled box near the
geometrically-projected goal pixel -- the policy then centers on what DINO
actually saw, not the privileged world-point projection. Also supports
--belief-adapter (ported from rollout_navdp_policy.py): once DINO has ever
confirmed the goal, a body-frame belief estimate is seeded on confirmation
ticks and dead-reckoned every tick after, so the policy steers back toward
the last confirmed sighting instead of falling back to manual search whenever
DINO loses it. Without --belief-adapter, behavior is identical to
qwen_search_rollout.py whenever DINO hasn't confirmed.

Usage:
    python scripts/vlm_nav_tests/qwen_search_dino.py \\
      --navdp-root /path/to/navdp_sam --ckpt /path/to/navdp_sam/runs/.../ckpt_last.pt \\
      --scene marsyard2022_tri.glb --terrain-obj marsyard2022.obj --scene-height-flip-z \\
      --start-x 0 --start-z 8 --start-yaw-deg 0 \\
      --goal-out-of-view --goal-bearing-deg 180 --goal-range 8 \\
      --command-file command.txt \\
      --dino-prompt "a small flag on a pole." --dino-max-range 15 \\
      --belief-adapter navdp/belief_adapter.pt \\
      --max-steps 400 --save-video --out mars_manual_search_dino

Then edit command.txt with 'left', 'right', 'straight', 'back', or 'stop' while it runs
(only honored before the goal has ever been confirmed by DINO).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse both scripts' machinery rather than duplicating it.
import rollout_navdp_policy as M
from qwen_ghost_rollout import GHOST_GOAL_HEIGHT, body_offset_to_world

# Belief-return primitives, ported verbatim from rollout_navdp_policy_mars.py /
# train_belief_adapter.py -- NOT reimplemented, imported so behavior matches exactly.
# rollout_navdp_policy_mars.py lives alongside this file (same dir as HERE, already on
# sys.path above). train_belief_adapter.py lives at ../navdp_sam/scripts/ relative to
# this file -- i.e. inside the NavDP repo, only resolvable once M.add_navdp_to_path(navdp_root)
# has run inside main() (it needs --navdp-root from argparse first). So pixel_to_body /
# propagate_body_point are safe to import here at module level, but belief_feat is deferred
# into main(), imported right next to VLAAdapter (same pattern the source script uses for
# train_vla_adapter -- both are late imports off the NavDP repo path).
from rollout_navdp_policy import pixel_to_body, propagate_body_point


def parse_search_command(text: str) -> str:
    """Map free text to one of: 'left', 'right', 'straight', 'back', 'stop', '' (unset).
    Extends M.command_intent (which only knows left/right/stop) with straight/back."""
    t = (text or "").strip().lower()
    if not t:
        return ""
    if any(k in t for k in ("stop", "halt", "brake", "wait", "hold")):
        return "stop"
    if any(k in t for k in ("back", "reverse", "retreat")):
        return "back"
    if any(k in t for k in ("straight", "forward", "ahead")):
        return "straight"
    left, right = "left" in t, "right" in t
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return ""


BEARING_SIGN = {
    "left": 1.0,
    "right": -1.0,
    "straight": 0.0,
    "back": None,
}  # None = 180 special-cased


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mars rollout: manual search, DINO-confirmed handoff to the real goal."
    )
    ap.add_argument("--navdp-root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default=str(M.DEFAULT_SCENE))
    ap.add_argument("--out", default="mars_manual_search_dino_rollout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", choices=["model", "ema"], default="model")
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--hfov-deg", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--stop-dist", type=float, default=1.2)
    ap.add_argument("--replan-every", type=int, default=1)

    # Start pose
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=8.0)
    ap.add_argument("--start-yaw-deg", type=float, default=0.0)

    # REAL goal: explicit world point, or auto-placed OUT OF the initial frustum
    ap.add_argument("--goal-x", type=float, default=None)
    ap.add_argument("--goal-z", type=float, default=None)
    ap.add_argument("--goal-y", type=float, default=None)
    ap.add_argument("--goal-height", type=float, default=1.2)
    ap.add_argument("--goal-terrain-radius", type=float, default=0.8)
    ap.add_argument("--goal-radius", type=int, default=18)
    ap.add_argument("--no-clamp-goal-to-edge", action="store_true")
    ap.add_argument(
        "--goal-out-of-view",
        action="store_true",
        help="Auto-place the goal at --goal-bearing-deg/--goal-range from the START pose.",
    )
    ap.add_argument("--goal-bearing-deg", type=float, default=180.0)
    ap.add_argument("--goal-range", type=float, default=8.0)
    ap.add_argument(
        "--lost-goal-min-px",
        type=int,
        default=10,
        help="Geometric mask pixel count gate BEFORE DINO is even asked to look (cheap "
        "pre-filter -- see --dino-max-range for the main gate).",
    )
    ap.add_argument(
        "--flag-mesh",
        type=str,
        default="assets/flag.glb",
        help="Flag asset placed at the real goal's world point. Same asset used for "
        "detection AND rendering now (previously purely cosmetic).",
    )
    ap.add_argument(
        "--no-flag",
        action="store_true",
        help="Disable spawning the flag entirely "
        "(also disables detection, since there is nothing to see).",
    )

    # --- GroundingDINO confirmation --------------------------------------------------
    ap.add_argument("--dino-model-id", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument(
        "--dino-prompt",
        type=str,
        default="a small flag on a pole.",
        help="Open-vocab text prompt. Lowercase, end with a period (GroundingDINO "
        "convention); describe the actual asset/placeholder shape, not just 'flag', "
        "for better zero-shot recall against Mars terrain clutter.",
    )
    ap.add_argument("--dino-box-threshold", type=float, default=0.35)
    ap.add_argument("--dino-text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--dino-pixel-tol",
        type=float,
        default=60.0,
        help="Max pixel distance between DINO's box centroid and the geometric "
        "projection of the known world point for the detection to COUNT as the "
        "flag (rejects DINO false positives elsewhere in frame).",
    )
    ap.add_argument(
        "--dino-max-range",
        type=float,
        default=15.0,
        help="Only run DINO when the geometric range to the goal is under this (m) AND "
        "the geometric mask is nonzero -- avoids wasting inference on frames where "
        "the flag cannot possibly be resolved (too far / behind us / occluded).",
    )
    ap.add_argument(
        "--dino-every",
        type=int,
        default=1,
        help="Run DINO every N ticks once the cheap gate passes (raise this to save "
        "compute; 'found' holds its last value between DINO calls).",
    )
    ap.add_argument(
        "--no-dino",
        action="store_true",
        help="Fall back to pure geometric found/lost (identical to "
        "qwen_search_rollout_mars.py) -- useful for A/B comparison runs.",
    )

    # Manual command interface (replaces the VLA adapter with ghost-mask execution)
    ap.add_argument("--command", type=str, default="")
    ap.add_argument("--command-file", type=str, default="")
    ap.add_argument("--search-bearing-deg", type=float, default=60.0)
    ap.add_argument("--search-distance", type=float, default=5.0)

    # QWEN-DRIVEN search (unchanged from the base script)
    ap.add_argument("--qwen-search", action="store_true")
    ap.add_argument("--instruction", type=str, default="explore and find the target")
    ap.add_argument("--qwen-search-hold-s", type=float, default=6.0)
    ap.add_argument("--qwen-search-memory", type=int, default=4)
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--qwen-no-4bit", action="store_true")

    # --- Belief-return adapter (NEW) --------------------------------------------------
    ap.add_argument(
        "--belief-adapter",
        type=str,
        default=None,
        help="Trained belief_adapter.pt (see train_belief_adapter.py). When set, a "
        "body-frame belief of the goal is seeded/re-seeded on every DINO-confirmed "
        "tick and dead-reckoned via odometry thereafter. Once the goal has been "
        "confirmed at least once, ALL subsequent found=False ticks are driven by "
        "this belief (manual/Qwen commands are ignored) instead of ghost-mask "
        "search. If omitted, behavior is identical to the pre-belief script: "
        "manual/Qwen search is used for the entire run whenever DINO hasn't "
        "confirmed the flag.",
    )
    ap.add_argument(
        "--belief-odom-noise",
        type=float,
        default=0.0,
        help="Gaussian odom noise per step for belief propagation (see "
        "propagate_body_point). 0 = perfect dead-reckoning.",
    )

    # Terrain
    ap.add_argument(
        "--terrain-height-mode",
        choices=["auto", "heightmap", "obj", "flat"],
        default="auto",
    )
    ap.add_argument("--heightmap", default=None)
    ap.add_argument("--terrain-obj", default=str(M.DEFAULT_OBJ))
    ap.add_argument("--flat-y", type=float, default=0.0)
    ap.add_argument("--clearance", type=float, default=1.4)
    ap.add_argument("--pose-terrain-radius", type=float, default=0.8)
    ap.add_argument("--size-x", type=float, default=M.SIZE_X)
    ap.add_argument("--size-z", type=float, default=M.SIZE_Z)
    ap.add_argument("--size-y", type=float, default=M.SIZE_Y)
    ap.add_argument("--flip-heightmap-x", action="store_true")
    ap.add_argument(
        "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--swap-heightmap-xz", action="store_true")
    ap.add_argument(
        "--scene-height-flip-x", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--scene-height-flip-z", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--scene-height-swap-xz", action=argparse.BooleanOptionalAction, default=False
    )

    # Policy / action modes
    ap.add_argument(
        "--habitat-proprio-mode", choices=["pose7", "planar3", "zero"], default=None
    )
    ap.add_argument(
        "--habitat-action-mode",
        choices=["action3d", "action2d", "waypoint"],
        default=None,
    )
    ap.add_argument("--habitat-yaw-axis", choices=["x", "y", "z"], default=None)
    ap.add_argument(
        "--habitat-use-obstacle-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--zero-lateral", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--max-forward-speed", type=float, default=1.0)
    ap.add_argument("--max-lateral-speed", type=float, default=1.0)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument(
        "--action-smoothing", choices=["ensemble", "ema", "none"], default="ensemble"
    )
    ap.add_argument("--ensemble-decay", type=float, default=0.5)
    ap.add_argument("--ema-alpha", type=float, default=0.6)

    # Optional real-obstacle CBF safety net
    ap.add_argument("--cbf", action="store_true")
    ap.add_argument("--cbf-mode", choices=["project", "cone"], default="cone")
    ap.add_argument("--cbf-d-safe", type=float, default=1.2)
    ap.add_argument("--cbf-gamma", type=float, default=0.3)
    ap.add_argument("--cbf-deadzone", type=float, default=0.8)
    ap.add_argument("--cbf-proj-iters", type=int, default=15)
    ap.add_argument("--cbf-proj-lr", type=float, default=0.08)
    ap.add_argument("--cbf-cone-margin", type=float, default=0.05)
    ap.add_argument("--cbf-trust", type=float, default=0.3)
    ap.add_argument("--cbf-smooth", type=float, default=0.0)
    ap.add_argument("--cbf-keep-speed", type=float, default=1.0)
    ap.add_argument(
        "--cbf-metric", choices=["euclidean", "mahalanobis"], default="mahalanobis"
    )
    ap.add_argument("--cbf-cov-base", type=float, default=1.0)
    ap.add_argument("--cbf-cov-growth", type=float, default=0.6)
    ap.add_argument(
        "--cbf-cov-mode", choices=["grow", "flat", "shrink"], default="shrink"
    )
    ap.add_argument("--robot-radius", type=float, default=0.25)
    ap.add_argument("--safety-margin", type=float, default=0.15)
    ap.add_argument(
        "--cbf-cone-project", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--cbf-hard-gate", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--obstacle-mode", choices=["none", "depth"], default="none")
    ap.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
    ap.add_argument("--obstacle-min-y-frac", type=float, default=0.45)
    ap.add_argument(
        "--cbf-commit-side", action=argparse.BooleanOptionalAction, default=True
    )

    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    navdp_root = M.resolve_navdp_root(args.navdp_root)
    M.add_navdp_to_path(navdp_root)

    from navdp.data.habitat_route_dataset import (
        _empty_belief_tensor,
        _proprio_from_pose,
    )
    from navdp.extensions import (
        DepthObstacleMap,
        horizon_growth_covariance,
        nearest_obstacle_point,
        nearest_obstacle_state,
        project_chunk_cone,
        project_forward_velocity_cbf,
    )
    from rollout_habitat_policy import (
        ActionSmoother,
        action_to_control,
        frame_to_spatial,
        load_model,
        resolve_modes,
        resolve_obstacle_channel,
    )

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    qwen_explorer = None
    if args.qwen_search:
        from navdp.extensions import QwenExplorer

        qwen_explorer = QwenExplorer(
            model_id=args.qwen_model_id,
            device=args.device,
            load_in_4bit=not args.qwen_no_4bit,
            memory_len=int(args.qwen_search_memory),
        )
        if not args.command_file:
            args.command_file = str(out_dir / "qwen_command.txt")
        print(
            f"[QWEN] search: instruction={args.instruction!r} every {args.qwen_search_hold_s:.1f}s, "
            f"writing decisions to {args.command_file}",
            flush=True,
        )
    qwen_search_interval = max(1, int(round(args.qwen_search_hold_s * args.hz)))

    # --- GroundingDINO confirmation detector --------------------------------------
    dino = None
    if not args.no_dino and not args.no_flag:
        from navdp.extensions import GroundingDINODetector

        dino = GroundingDINODetector(
            model_id=args.dino_model_id,
            device=args.device,
            box_threshold=args.dino_box_threshold,
            text_threshold=args.dino_text_threshold,
        )
        print(
            f"[DINO] confirmation enabled: prompt={args.dino_prompt!r} "
            f"max_range={args.dino_max_range:.1f}m pixel_tol={args.dino_pixel_tol:.0f}px "
            f"box_thresh={args.dino_box_threshold} every={args.dino_every}",
            flush=True,
        )
    elif args.no_dino:
        print(
            "[DINO] disabled (--no-dino): falling back to pure geometric found/lost, "
            "identical to qwen_search_rollout_mars.py",
            flush=True,
        )

    # --- Belief-return adapter (NEW): loaded exactly as in rollout_navdp_policy_mars.py ---
    # train_belief_adapter.py (for belief_feat) and train_vla_adapter.py (for VLAAdapter) both
    # live under navdp_root/scripts/, which is only on sys.path after add_navdp_to_path() above
    # -- hence these imports are deferred here rather than at module top-level.
    belief_adapter = None
    belief_feat_3d = None
    if args.belief_adapter:
        from train_vla_adapter import VLAAdapter
        from train_belief_adapter import belief_feat as belief_feat_3d

        _bck = torch.load(args.belief_adapter, map_location=args.device)
        belief_adapter = VLAAdapter(
            _bck["belief_feat_dim"], _bck["dim"], num_tokens=_bck.get("num_tokens", 4)
        ).to(args.device)
        belief_adapter.load_state_dict(_bck["adapter"])
        belief_adapter.eval()
        print(
            f"[BELIEF] belief-return enabled from {args.belief_adapter}: once DINO confirms the "
            f"goal at least once, all subsequent unconfirmed ticks are driven by dead-reckoned "
            f"belief (manual/Qwen search commands are ignored while belief is active); "
            f"tokens={belief_adapter.num_tokens}",
            flush=True,
        )
    else:
        print(
            "[BELIEF] --belief-adapter not passed: found=False ticks always use manual/Qwen "
            "ghost-mask search (identical to the pre-belief script).",
            flush=True,
        )

    raw_terrain = M.TerrainHeight(
        mode=args.terrain_height_mode,
        heightmap=(
            Path(args.heightmap).expanduser().resolve() if args.heightmap else None
        ),
        obj=Path(args.terrain_obj).expanduser().resolve() if args.terrain_obj else None,
        flat_y=args.flat_y,
        size_x=args.size_x,
        size_z=args.size_z,
        size_y=args.size_y,
        flip_x=args.flip_heightmap_x,
        flip_z=args.flip_heightmap_z,
        swap_xz=args.swap_heightmap_xz,
    )
    terrain = M.SceneMappedTerrain(
        raw_terrain,
        flip_x=bool(args.scene_height_flip_x),
        flip_z=bool(args.scene_height_flip_z),
        swap_xz=bool(args.scene_height_swap_xz),
    )

    device = args.device
    model, train_args = load_model(
        Path(args.ckpt).expanduser().resolve(), device, args.weights
    )
    modes = resolve_modes(args, train_args)
    if modes["action_mode"] == "waypoint":
        raise ValueError(
            "Mars rollout executes velocity actions; use action3d or action2d checkpoint/mode."
        )
    use_obstacle_channel = resolve_obstacle_channel(args, train_args)
    image_size = int(args.image_size or train_args.get("image_size", 224))
    intr = M.intrinsics_from_hfov(args.height, args.width, args.hfov_deg)
    obstacle_builder = DepthObstacleMap(camera_intrinsics=intr)
    smoother = ActionSmoother(
        args.action_smoothing, args.ensemble_decay, args.ema_alpha
    )

    sim = M.make_sim(
        Path(args.scene), args.height, args.width, args.hfov_deg, with_semantic=False
    )
    agent = sim.initialize_agent(0)

    x = float(args.start_x)
    z = float(args.start_z)
    yaw = math.radians(float(args.start_yaw_deg))
    dt = 1.0 / float(args.hz)

    if args.goal_out_of_view:
        bearing_rad = math.radians(args.goal_bearing_deg)
        fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
        left_x, left_z = -math.cos(yaw), math.sin(yaw)
        gx = x + args.goal_range * (
            math.cos(bearing_rad) * fwd_x + math.sin(bearing_rad) * left_x
        )
        gz = z + args.goal_range * (
            math.cos(bearing_rad) * fwd_z + math.sin(bearing_rad) * left_z
        )
        gy = terrain.local_height_max(gx, gz, float(args.goal_terrain_radius)) + float(
            args.goal_height
        )
        goal = np.asarray([gx, gy, gz], dtype=np.float32)
        half_fov = math.radians(args.hfov_deg) / 2.0
        in_view_at_spawn = (
            abs(((bearing_rad + math.pi) % (2 * math.pi)) - math.pi) <= half_fov
        )
        if in_view_at_spawn:
            print(
                f"[WARN] --goal-bearing-deg {args.goal_bearing_deg:.1f} is INSIDE the {args.hfov_deg:.0f}deg "
                f"FOV (half={math.degrees(half_fov):.1f}deg) -- the goal will be visible at spawn.",
                flush=True,
            )
    elif args.goal_x is not None and args.goal_z is not None:
        goal_y = args.goal_y
        if goal_y is None:
            goal_y = terrain.local_height_max(
                float(args.goal_x), float(args.goal_z), float(args.goal_terrain_radius)
            ) + float(args.goal_height)
        goal = np.asarray(
            [float(args.goal_x), float(goal_y), float(args.goal_z)], dtype=np.float32
        )
    else:
        raise SystemExit("Pass --goal-x/--goal-z, or --goal-out-of-view.")

    M.spawn_flag_mesh(sim, goal, args.flag_mesh, out_dir, enabled=not args.no_flag)

    rows: Dict[str, list] = {
        k: []
        for k in [
            "rgb",
            "depth",
            "goal_mask",
            "obstacle_mask",
            "seg_masks",
            "pose",
            "proprio",
            "action_3d",
            "pred_chunk",
            "goal_distance",
            "found",
            "command",
            "obstacle_visible_pixels",
            "cone_correction_step0",
            "cone_correction_last",
            "hard_gate_tick",
            "dino_ran",
            "dino_score",
            "dino_u",
            "dino_v",
            "geom_u",
            "geom_v",
            "belief_active",
            "belief_fwd",
            "belief_left",
        ]
    }
    video_frames = []
    last_pred_chunk = None
    replan_every = max(int(args.replan_every), 1)
    cbf_active = 0
    cone_side_latch = None
    hard_gate_fired = 0
    found = False
    belief_g = (
        None  # body-frame [forward, left] estimate, seeded only on DINO-confirmed ticks
    )

    print("Mars manual ghost-mask search rollout (DINO-confirmed handoff)", flush=True)
    print(f"  navdp_root : {navdp_root}", flush=True)
    print(f"  scene      : {Path(args.scene).expanduser().resolve()}", flush=True)
    print(f"  ckpt       : {Path(args.ckpt).expanduser().resolve()}", flush=True)
    print(
        f"  goal       : x={goal[0]:.2f} y={goal[1]:.2f} z={goal[2]:.2f} "
        f"(out_of_view={args.goal_out_of_view}, bearing={args.goal_bearing_deg:.1f}deg)",
        flush=True,
    )
    print(
        f"  modes      : action={modes['action_mode']} proprio={modes['proprio_mode']} obstacle_channel={use_obstacle_channel}",
        flush=True,
    )
    print(
        "  Edit --command-file with left/right/straight/back/stop while running "
        "(only used before the goal is first confirmed by DINO).",
        flush=True,
    )

    try:
        for step in range(int(args.max_steps)):
            cmd_txt = args.command
            if args.command_file:
                try:
                    cmd_txt = (
                        Path(args.command_file).read_text(encoding="utf-8").strip()
                        or args.command
                    )
                except Exception:
                    pass
            intent = parse_search_command(cmd_txt)

            y = terrain.local_height_max(x, z, float(args.pose_terrain_radius)) + float(
                args.clearance
            )
            position = np.asarray([x, y, z], dtype=np.float32)
            M.set_agent_pose(agent, x, y, z, yaw)
            obs = sim.get_sensor_observations()
            rgb, depth = M.rgb_depth(obs)

            # --- geometric projection: cheap, exact, always computed ------------------------
            real_goal_mask, real_goal_info = M.project_goal_mask(
                goal=goal,
                position=position,
                yaw=yaw,
                height=rgb.shape[0],
                width=rgb.shape[1],
                hfov_deg=args.hfov_deg,
                radius=args.goal_radius,
                clamp_to_edge=not args.no_clamp_goal_to_edge,
            )
            real_goal_px = int((real_goal_mask > 0).sum())
            geom_visible = real_goal_px >= int(args.lost_goal_min_px)

            # --- DINO confirmation gate ------------------------------------------------
            dino_ran = False
            dino_score = float("nan")
            dino_u = float("nan")
            dino_v = float("nan")
            was_found = found

            if dino is None:
                # --no-dino / --no-flag fallback: identical behaviour to the base script.
                found = geom_visible
                goal_u_render, goal_v_render = real_goal_info["u"], real_goal_info["v"]
            else:
                dino_gate = (
                    geom_visible
                    and real_goal_info["range"] <= float(args.dino_max_range)
                    and step % max(int(args.dino_every), 1) == 0
                )
                if dino_gate:
                    dino_ran = True
                    det = dino.detect_best(rgb, text_prompt=args.dino_prompt)
                    if det is not None and det.confirms(
                        real_goal_info["u"],
                        real_goal_info["v"],
                        pixel_tol=args.dino_pixel_tol,
                    ):
                        found = True
                        dino_score = det.score
                        dino_u, dino_v = det.u, det.v
                    else:
                        # geometrically visible but DINO didn't resolve/confirm it this tick
                        found = False
                elif not geom_visible:
                    # cannot possibly be found if geometry says it's not even in frustum
                    found = False
                # else: dino_gate false but geom_visible true (out of dino-max-range, or
                # held between --dino-every ticks) -> hold the previous `found` value

                if found and dino_ran and not np.isnan(dino_u):
                    goal_u_render, goal_v_render = (
                        dino_u,
                        dino_v,
                    )  # render at what was SEEN
                elif found:
                    goal_u_render, goal_v_render = (
                        real_goal_info["u"],
                        real_goal_info["v"],
                    )  # held frame
                else:
                    goal_u_render, goal_v_render = None, None

            if found and not was_found:
                src = (
                    f"DINO score={dino_score:.2f}"
                    if dino is not None and not np.isnan(dino_score)
                    else "geometry"
                )
                print(
                    f"[SEARCH] step {step}: goal FOUND ({src}) -> driving toward it",
                    flush=True,
                )
            elif was_found and not found:
                belief_txt = (
                    " -> switching to belief-return"
                    if (belief_adapter is not None and belief_g is not None)
                    else " -> back to manual search"
                )
                print(
                    f"[SEARCH] step {step}: goal lost/unconfirmed again{belief_txt}",
                    flush=True,
                )

            # --- belief seed/update (NEW): ONLY on ticks where DINO actually confirmed this
            # tick (a fresh detection, not a held frame -- dino_u/dino_v are only real numbers
            # on such ticks per the block above). Re-seeding (not just first-time seeding) keeps
            # the estimate corrected on every fresh sighting, exactly like belief_update_on_sight
            # in the source script. -----------------------------------------------------------
            if (
                belief_adapter is not None
                and found
                and dino_ran
                and not np.isnan(dino_u)
            ):
                belief_g = pixel_to_body(
                    dino_u,
                    dino_v,
                    depth,
                    rgb.shape[0],
                    rgb.shape[1],
                    args.hfov_deg,
                    fallback_range=float(real_goal_info["range"]),
                )

            # QWEN-DRIVEN search: only while not found AND belief is not already available
            # (belief fully owns steering once we have ever confirmed the goal -- see spec).
            belief_will_drive = (
                belief_adapter is not None and belief_g is not None and not found
            )
            if (
                qwen_explorer is not None
                and not found
                and not belief_will_drive
                and step % qwen_search_interval == 0
            ):
                decision = qwen_explorer.decide(rgb, extra_hint=args.instruction)
                Path(args.command_file).write_text(decision.direction, encoding="utf-8")
                print(
                    f"[QWEN] step {step}: instruction={args.instruction!r} -> wrote '{decision.direction}' "
                    f"to {args.command_file}  (raw={decision.raw!r}, mem={qwen_explorer.memory})",
                    flush=True,
                )

            belief_token = None
            if found:
                # GOAL MASK RETIRED ON FOUND: render an EMPTY goal-mask channel, not a disc at
                # the confirmed pixel and not the search ghost. Once DINO has confirmed the
                # flag, no goal-mask blob is drawn at all for the rest of the "found" state --
                # goal_info still carries the confirmed pixel/range for logging and for the
                # goal_dist stopping check below, but the policy's goal-mask input is blank.
                goal_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                goal_info = {
                    "visible": 1.0,
                    "u": float(goal_u_render) if goal_u_render is not None else -1.0,
                    "v": float(goal_v_render) if goal_v_render is not None else -1.0,
                    "range": float(real_goal_info["range"]),
                    "bearing": float(real_goal_info["bearing"]),
                }
            elif belief_will_drive:
                # BELIEF-RETURN (NEW): goal has been confirmed before but is currently
                # unconfirmed. Manual/Qwen commands are ignored; the goal-mask channel stays
                # blank (same rendering as FOUND -- there is no pixel to draw a disc at) and
                # the policy is steered via belief_adapter's conditioning token instead.
                goal_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                goal_info = {
                    "visible": 0.0,
                    "u": -1.0,
                    "v": -1.0,
                    "range": float(np.hypot(belief_g[0], belief_g[1])),
                    "bearing": float(math.atan2(belief_g[1], belief_g[0])),
                }
                with torch.no_grad():
                    feat = (
                        torch.from_numpy(belief_feat_3d(belief_g)[None])
                        .float()
                        .to(device)
                    )
                    belief_token = belief_adapter(feat)
            else:
                # UNCHANGED manual/Qwen ghost-mask search: only reached when the goal has never
                # been confirmed yet (belief_g is None), or --belief-adapter was not passed.
                direction = (
                    intent
                    if intent in ("left", "right", "straight", "back")
                    else "straight"
                )
                bearing = (
                    math.pi
                    if direction == "back"
                    else math.radians(args.search_bearing_deg) * BEARING_SIGN[direction]
                )
                fwd = args.search_distance * math.cos(bearing)
                left = args.search_distance * math.sin(bearing)
                search_ghost = body_offset_to_world(position, yaw, fwd, left, terrain)
                goal_mask, goal_info = M.project_goal_mask(
                    goal=search_ghost,
                    position=position,
                    yaw=yaw,
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    hfov_deg=args.hfov_deg,
                    radius=args.goal_radius,
                    clamp_to_edge=not args.no_clamp_goal_to_edge,
                )

            if args.obstacle_mode == "depth":
                obstacle_mask = M.depth_obstacle_mask(
                    depth, args.obstacle_depth_threshold, args.obstacle_min_y_frac
                )
            else:
                obstacle_mask = np.zeros_like(goal_mask, dtype=np.uint8)

            spatial = frame_to_spatial(
                depth,
                goal_mask,
                image_size,
                obstacle_mask,
                include_obstacle_channel=use_obstacle_channel,
            ).to(device)
            obstacle_map = (
                obstacle_builder.build(depth)
                if args.obstacle_mode == "depth"
                else np.zeros((96, 96), dtype=np.float32)
            )
            obstacle_t = torch.from_numpy(obstacle_map[None]).float().to(device)

            qx, qy, qz, qw = M.yaw_quat_xyzw(yaw)
            pose = np.asarray([x, y, z, qx, qy, qz, qw], dtype=np.float32)
            proprio = _proprio_from_pose(
                pose,
                modes["proprio_mode"],
                planar_axes=(0, 2),
                yaw_axis=modes["yaw_axis"],
            )
            proprio_t = torch.from_numpy(proprio[None]).float().to(device)
            belief_t = torch.from_numpy(_empty_belief_tensor()[None]).float().to(device)
            route_index = torch.zeros(1, dtype=torch.long, device=device)
            active_goal_index = torch.zeros(1, dtype=torch.long, device=device)

            obstacle_point = None
            obstacle_radius_perceived = None
            if args.cbf and int(obstacle_mask.sum()) > 0:
                _obs_state = nearest_obstacle_state(obstacle_mask, depth, intr)
                if _obs_state is not None:
                    obstacle_point = _obs_state["p0"]
                    obstacle_radius_perceived = _obs_state["radius"]
                else:
                    obstacle_point = nearest_obstacle_point(obstacle_mask, depth, intr)
            if obstacle_point is None:
                cone_side_latch = None

            do_replan = (step % replan_every == 0) or (last_pred_chunk is None)
            cone_correction_step0 = float("nan")
            cone_correction_last = float("nan")
            if do_replan:
                pred = model.sample(
                    spatial,
                    proprio_t,
                    steps=int(args.sample_steps),
                    belief_tensor=belief_t,
                    obstacle_map=obstacle_t,
                    route_index=route_index,
                    active_goal_index=active_goal_index,
                    extra_cond_tokens=belief_token,
                )
                if (
                    args.cbf
                    and args.cbf_mode == "cone"
                    and obstacle_point is not None
                    and args.cbf_cone_project
                ):
                    cbf_active += 1
                    v_o = np.zeros(2, dtype=np.float32)
                    if args.zero_lateral and pred.shape[-1] >= 3:
                        pred = pred.clone()
                        pred[..., 1] = 0.0
                    p_lat = float(obstacle_point[1])
                    side = -1.0 if p_lat > 0.0 else 1.0
                    if args.cbf_commit_side:
                        if cone_side_latch is None:
                            cone_side_latch = side
                        side = cone_side_latch
                    cone_sigma = None
                    if args.cbf_metric == "mahalanobis":
                        cone_sigma = horizon_growth_covariance(
                            pred.shape[1],
                            pred.shape[2],
                            base=args.cbf_cov_base,
                            growth=args.cbf_cov_growth,
                            mode=args.cbf_cov_mode,
                            device=pred.device,
                            dtype=pred.dtype,
                        )
                    r_used = (
                        obstacle_radius_perceived
                        + args.robot_radius
                        + args.safety_margin
                        if obstacle_radius_perceived is not None
                        else args.cbf_d_safe
                    )
                    pre0 = pred[0, 0, :].detach().cpu().numpy().copy()
                    preL = pred[0, -1, :].detach().cpu().numpy().copy()
                    pred = project_chunk_cone(
                        pred,
                        obstacle_point,
                        v_o,
                        r=r_used,
                        dt=dt,
                        vel_scale=1.0,
                        iters=args.cbf_proj_iters,
                        lr=args.cbf_proj_lr,
                        trust=args.cbf_trust,
                        margin=args.cbf_cone_margin,
                        smooth_weight=args.cbf_smooth,
                        keep_speed=args.cbf_keep_speed,
                        sigma=cone_sigma,
                        deadzone_range=r_used + args.cbf_deadzone,
                        side=side,
                    )
                    post0 = pred[0, 0, :].detach().cpu().numpy()
                    postL = pred[0, -1, :].detach().cpu().numpy()
                    cone_correction_step0 = float(np.linalg.norm(post0 - pre0))
                    cone_correction_last = float(np.linalg.norm(postL - preL))

                pred_chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
                chunk_ctrl = np.stack(
                    [
                        action_to_control(
                            a,
                            action_mode=modes["action_mode"],
                            max_forward_speed=args.max_forward_speed,
                            max_lateral_speed=args.max_lateral_speed,
                            max_yaw_rate=args.max_yaw_rate,
                        )
                        for a in pred_chunk
                    ]
                ).astype(np.float32)
                smoother.add(step, chunk_ctrl)
                last_pred_chunk = pred_chunk
            else:
                pred_chunk = last_pred_chunk

            action_3d = smoother.get(step)
            if args.zero_lateral and action_3d.shape[0] >= 2:
                action_3d = action_3d.copy()
                action_3d[1] = 0.0
            if args.cbf and args.cbf_mode == "project" and obstacle_point is not None:
                action_3d, _ = project_forward_velocity_cbf(
                    action_3d,
                    obstacle_point,
                    np.zeros(2, dtype=np.float32),
                    d_safe=args.cbf_d_safe,
                    gamma=args.cbf_gamma,
                    deadzone=args.cbf_deadzone,
                    trust=args.cbf_trust,
                )
            hard_gate_fired_tick = False
            if (
                args.cbf
                and args.cbf_mode == "cone"
                and args.cbf_hard_gate
                and obstacle_point is not None
            ):
                r_used2 = (
                    obstacle_radius_perceived + args.robot_radius + args.safety_margin
                    if obstacle_radius_perceived is not None
                    else args.cbf_d_safe
                )
                action_3d, _gated = project_forward_velocity_cbf(
                    action_3d,
                    obstacle_point,
                    np.zeros(2, dtype=np.float32),
                    d_safe=r_used2,
                    gamma=args.cbf_gamma,
                    deadzone=args.cbf_deadzone,
                    trust=None,
                )
                if _gated:
                    hard_gate_fired += 1
                    hard_gate_fired_tick = True

            # Manual "stop" command is only honored in the same phase manual search runs in
            # (never confirmed yet, no belief). Once belief-return or FOUND owns control, "stop"
            # is no longer read -- matches "manual/Qwen commands are ignored" while belief drives.
            if intent == "stop" and not found and not belief_will_drive:
                action_3d = np.zeros(3, dtype=np.float32)

            next_position, next_yaw = M.integrate_mars(position, yaw, action_3d, dt)
            x = float(
                np.clip(
                    next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
                )
            )
            z = float(
                np.clip(
                    next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
                )
            )
            yaw = M.wrap_angle(next_yaw)

            # Propagate belief by the EXECUTED motion every tick it exists, regardless of
            # found/not-found, so it's current whenever it's next needed (dead-reckoning).
            if belief_g is not None:
                belief_g = propagate_body_point(
                    belief_g, action_3d, dt, args.belief_odom_noise
                )

            goal_dist = float(np.hypot(goal[0] - x, goal[2] - z))

            rows["rgb"].append(rgb)
            rows["depth"].append(depth)
            rows["goal_mask"].append(goal_mask.astype(np.uint8))
            rows["obstacle_mask"].append(obstacle_mask.astype(np.uint8))
            seg = np.zeros_like(goal_mask, dtype=np.uint8)
            seg[goal_mask > 0] = 1
            seg[obstacle_mask > 0] = 2
            rows["seg_masks"].append(seg)
            rows["pose"].append(pose)
            rows["proprio"].append(proprio.astype(np.float32))
            rows["action_3d"].append(action_3d.astype(np.float32))
            rows["pred_chunk"].append(pred_chunk.astype(np.float32))
            rows["goal_distance"].append(goal_dist)
            rows["found"].append(bool(found))
            rows["command"].append(intent)
            rows["obstacle_visible_pixels"].append(int(obstacle_mask.sum()))
            rows["cone_correction_step0"].append(cone_correction_step0)
            rows["cone_correction_last"].append(cone_correction_last)
            rows["hard_gate_tick"].append(bool(hard_gate_fired_tick))
            rows["dino_ran"].append(bool(dino_ran))
            rows["dino_score"].append(dino_score)
            rows["dino_u"].append(dino_u)
            rows["dino_v"].append(dino_v)
            rows["geom_u"].append(float(real_goal_info["u"]))
            rows["geom_v"].append(float(real_goal_info["v"]))
            rows["belief_active"].append(bool(belief_will_drive))
            rows["belief_fwd"].append(
                float(belief_g[0]) if belief_g is not None else float("nan")
            )
            rows["belief_left"].append(
                float(belief_g[1]) if belief_g is not None else float("nan")
            )

            if found:
                mode_txt = "FOUND"
            elif belief_will_drive:
                mode_txt = "belief_return"
            else:
                mode_txt = f"search({intent or 'straight'})"
            if step % max(int(args.save_every), 1) == 0:
                dino_txt = f" dino={dino_score:.2f}" if not np.isnan(dino_score) else ""
                text = f"t={step} mode={mode_txt} dist={goal_dist:.2f} v={action_3d[0]:.2f} yaw={math.degrees(yaw):.1f}{dino_txt}"
                frame = M.overlay_frame(rgb, goal_mask, obstacle_mask, text)
                frame.save(frame_dir / f"frame_{step:04d}.png")
                video_frames.append(frame)
            if step % 10 == 0:
                print(
                    f"step {step:04d} | mode={mode_txt} | dist={goal_dist:.2f} | goal_px={int((goal_mask>0).sum())} "
                    f"| dino_ran={dino_ran} dino_score={dino_score:.2f} "
                    f"| action=[{action_3d[0]:.2f},{action_3d[1]:.2f},{action_3d[2]:.2f}]",
                    flush=True,
                )

            if goal_dist <= float(args.stop_dist):
                print(f"Reached goal at step {step} dist={goal_dist:.2f}m", flush=True)
                break
    finally:
        sim.close()

    print(
        f"[CBF diag] cbf_active={cbf_active} hard_gate_fired={hard_gate_fired}",
        flush=True,
    )
    n_dino_runs = int(sum(rows["dino_ran"]))
    n_dino_confirms = int(sum(rows["found"]))
    n_belief_ticks = int(sum(rows["belief_active"]))
    print(f"[DINO diag] runs={n_dino_runs} steps_found={n_dino_confirms}", flush=True)
    print(f"[BELIEF diag] belief_return_ticks={n_belief_ticks}", flush=True)

    success = bool(
        rows["goal_distance"] and rows["goal_distance"][-1] <= float(args.stop_dist)
    )
    npz_path = out_dir / "rollout.npz"
    np.savez_compressed(
        npz_path,
        rgb=np.stack(rows["rgb"]).astype(np.uint8),
        depth=np.stack(rows["depth"]).astype(np.float32),
        goal_mask=np.stack(rows["goal_mask"]).astype(np.uint8),
        obstacle_mask=np.stack(rows["obstacle_mask"]).astype(np.uint8),
        seg_masks=np.stack(rows["seg_masks"]).astype(np.uint8),
        pose=np.stack(rows["pose"]).astype(np.float32),
        proprio=np.stack(rows["proprio"]).astype(np.float32),
        action_3d=np.stack(rows["action_3d"]).astype(np.float32),
        pred_chunk=np.stack(rows["pred_chunk"]).astype(np.float32),
        goal_distance=np.asarray(rows["goal_distance"], dtype=np.float32),
        found=np.asarray(rows["found"], dtype=bool),
        command=np.array(rows["command"]),
        obstacle_visible_pixels=np.asarray(
            rows["obstacle_visible_pixels"], dtype=np.int32
        ),
        cone_correction_step0=np.asarray(
            rows["cone_correction_step0"], dtype=np.float32
        ),
        cone_correction_last=np.asarray(rows["cone_correction_last"], dtype=np.float32),
        hard_gate_tick=np.asarray(rows["hard_gate_tick"], dtype=bool),
        dino_ran=np.asarray(rows["dino_ran"], dtype=bool),
        dino_score=np.asarray(rows["dino_score"], dtype=np.float32),
        dino_u=np.asarray(rows["dino_u"], dtype=np.float32),
        dino_v=np.asarray(rows["dino_v"], dtype=np.float32),
        geom_u=np.asarray(rows["geom_u"], dtype=np.float32),
        geom_v=np.asarray(rows["geom_v"], dtype=np.float32),
        belief_active=np.asarray(rows["belief_active"], dtype=bool),
        belief_fwd=np.asarray(rows["belief_fwd"], dtype=np.float32),
        belief_left=np.asarray(rows["belief_left"], dtype=np.float32),
        goal_position=goal.astype(np.float32),
        success=np.asarray(success, dtype=bool),
        hz=np.asarray(float(args.hz), dtype=np.float32),
    )
    manifest = {
        "success": success,
        "frames": len(rows["rgb"]),
        "final_distance": (
            float(rows["goal_distance"][-1]) if rows["goal_distance"] else None
        ),
        "goal_position": goal.tolist(),
        "goal_out_of_view": args.goal_out_of_view,
        "dino_enabled": dino is not None,
        "dino_prompt": args.dino_prompt if dino is not None else None,
        "dino_max_range": args.dino_max_range if dino is not None else None,
        "dino_runs": n_dino_runs,
        "belief_adapter_enabled": belief_adapter is not None,
        "belief_adapter_path": (
            args.belief_adapter if belief_adapter is not None else None
        ),
        "belief_return_ticks": n_belief_ticks,
        "ckpt": str(Path(args.ckpt).expanduser().resolve()),
        "scene": str(Path(args.scene).expanduser().resolve()),
        "npz": str(npz_path),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if args.save_video:
        M.save_video(
            video_frames,
            out_dir / "rollout.mp4",
            fps=max(float(args.hz) / max(int(args.save_every), 1), 1.0),
        )
    print(f"Saved rollout: {npz_path}", flush=True)
    print(f"Output dir   : {out_dir}", flush=True)


if __name__ == "__main__":
    main()
