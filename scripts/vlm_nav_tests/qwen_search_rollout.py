"""Legacy manual left/right/straight/back search rollout, no VLM in the loop:
a direction typed into --command-file is rendered directly as a body-frame
"ghost" goal mask (the same channel the policy trains on, so nothing is
out-of-distribution) until the REAL goal (which may start --goal-out-of-view)
becomes visible, at which point commands are ignored and the policy drives
straight to it (SEARCHING -> FOUND -> SEARCHING again if lost).

Usage:
    python scripts/vlm_nav_tests/qwen_search_rollout.py \\
      --navdp-root /path/to/navdp_sam --ckpt /path/to/navdp_sam/runs/.../ckpt_last.pt \\
      --scene marsyard2022_tri.glb --terrain-obj marsyard2022.obj --scene-height-flip-z \\
      --start-x 0 --start-z 8 --start-yaw-deg 0 \\
      --goal-out-of-view --goal-bearing-deg 180 --goal-range 8 \\
      --command-file command.txt --max-steps 400 --save-video --out mars_manual_search

Then edit command.txt with 'left', 'right', 'straight', 'back', or 'stop' while it runs.
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
        description="Mars rollout: manual left/right/straight/back ghost-mask search, auto-handoff when the real goal is found."
    )
    ap.add_argument("--navdp-root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default=str(M.DEFAULT_SCENE))
    ap.add_argument("--out", default="mars_manual_search_rollout")
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
        help="Auto-place the goal at --goal-bearing-deg/--goal-range from the START pose "
        "instead of --goal-x/--goal-z -- a plain world point needs no depth/mesh to "
        "define, so it can start genuinely out of frame with zero setup.",
    )
    ap.add_argument(
        "--goal-bearing-deg",
        type=float,
        default=180.0,
        help="Body-frame bearing (deg, + = left) from the START pose for --goal-out-of-view. "
        "180 = directly behind; anything with |bearing| > hfov/2 is guaranteed out of view.",
    )
    ap.add_argument(
        "--goal-range",
        type=float,
        default=8.0,
        help="Distance (m) for --goal-out-of-view.",
    )
    ap.add_argument(
        "--lost-goal-min-px",
        type=int,
        default=10,
        help="Real-goal mask pixel count that counts as 'found' -> hand off from search.",
    )
    ap.add_argument(
        "--flag-mesh",
        type=str,
        default="assets/flag.glb",
        help="Cosmetic flag asset (.glb/.obj) placed at the real goal's world point. "
        "Resolved relative to this script's directory if not absolute. Purely "
        "visual -- detection still uses project_goal_mask against the known "
        "world point, unchanged. Falls back to a procedural placeholder flag "
        "if the file doesn't exist yet.",
    )
    ap.add_argument(
        "--no-flag",
        action="store_true",
        help="Disable spawning the cosmetic flag entirely.",
    )

    # Manual command interface (replaces the VLA adapter with ghost-mask execution)
    ap.add_argument(
        "--command",
        type=str,
        default="",
        help="Fixed command: 'left' / 'right' / 'straight' / 'back' / 'stop' / '' (default straight).",
    )
    ap.add_argument(
        "--command-file",
        type=str,
        default="",
        help="Path polled every tick for the current command. With --qwen-search, Qwen WRITES "
        "its decisions here (defaults to <out>/qwen_command.txt if not given) instead of "
        "you typing them -- inspect the file (or the console log) to see what it chose.",
    )
    ap.add_argument(
        "--search-bearing-deg",
        type=float,
        default=60.0,
        help="Body-frame bearing (deg) for left/right -- a diagonal arc, not a pure sideways "
        "strafe, so forward progress continues while turning.",
    )
    ap.add_argument(
        "--search-distance",
        type=float,
        default=5.0,
        help="Ghost distance (m) ahead of the rover.",
    )

    # QWEN-DRIVEN search: translate a VAGUE instruction into left/right/straight/back, written
    # into --command-file on a slow cadence -- the SAME file/parsing path a human typing into it
    # would use, so nothing else in the pipeline changes. Not asked every tick (per-tick pixel
    # grounding is high-variance/jittery); a fresh direction is only picked every
    # --qwen-search-hold-s while the goal is still not found.
    ap.add_argument(
        "--qwen-search",
        action="store_true",
        help="Let Qwen pick left/right/straight/back from --instruction + the current view, "
        "instead of you editing --command-file by hand.",
    )
    ap.add_argument(
        "--instruction",
        type=str,
        default="explore and find the target",
        help="Vague natural-language goal passed to Qwen alongside the current view.",
    )
    ap.add_argument(
        "--qwen-search-hold-s",
        type=float,
        default=6.0,
        help="Seconds to hold each Qwen-chosen direction before asking again.",
    )
    ap.add_argument(
        "--qwen-search-memory",
        type=int,
        default=4,
        help="How many recent directions Qwen remembers, so it doesn't re-try a dead end.",
    )
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument(
        "--qwen-no-4bit",
        action="store_true",
        help="Load Qwen in fp16 instead of 4-bit.",
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

    # Optional real-obstacle CBF safety net (off by default -- this script's focus is search, not avoidance)
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
        # A plain world point needs no depth/mesh to define, so it can be placed OUT OF the
        # initial frustum with zero setup -- unlike a rendered-mask goal, which would need to
        # already be visible (with valid depth) to build the mesh from.
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
                f"FOV (half={math.degrees(half_fov):.1f}deg) -- the goal will be visible at spawn. "
                f"Use a bearing with |bearing| > {math.degrees(half_fov):.1f} to guarantee out-of-view.",
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
        ]
    }
    video_frames = []
    last_pred_chunk = None
    replan_every = max(int(args.replan_every), 1)
    cbf_active = 0
    cone_side_latch = None
    hard_gate_fired = 0
    found = False

    print("Mars manual ghost-mask search rollout", flush=True)
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
        "  Edit --command-file with left/right/straight/back/stop while running.",
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

            was_found = found
            found = real_goal_px >= int(args.lost_goal_min_px)
            if found and not was_found:
                print(
                    f"[SEARCH] step {step}: goal FOUND ({real_goal_px}px) -> driving straight to it",
                    flush=True,
                )
            elif was_found and not found:
                print(
                    f"[SEARCH] step {step}: goal lost again -> back to manual search",
                    flush=True,
                )

            # QWEN-DRIVEN search: translate the vague --instruction into left/right/straight/back,
            # WRITTEN into --command-file (picked up on the next tick's read above, same as a human
            # typing into it -- one tick of lag at a several-second hold is negligible). Not asked
            # while found (nothing left to search for), and only every qwen_search_interval ticks.
            if (
                qwen_explorer is not None
                and not found
                and step % qwen_search_interval == 0
            ):
                decision = qwen_explorer.decide(rgb, extra_hint=args.instruction)
                Path(args.command_file).write_text(decision.direction, encoding="utf-8")
                print(
                    f"[QWEN] step {step}: instruction={args.instruction!r} -> wrote '{decision.direction}' "
                    f"to {args.command_file}  (raw={decision.raw!r}, mem={qwen_explorer.memory})",
                    flush=True,
                )

            if found:
                goal_mask, goal_info = real_goal_mask, real_goal_info
            else:
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

            if intent == "stop" and not found:
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

            mode_txt = "FOUND" if found else f"search({intent or 'straight'})"
            if step % max(int(args.save_every), 1) == 0:
                text = f"t={step} mode={mode_txt} dist={goal_dist:.2f} v={action_3d[0]:.2f} yaw={math.degrees(yaw):.1f}"
                frame = M.overlay_frame(rgb, goal_mask, obstacle_mask, text)
                frame.save(frame_dir / f"frame_{step:04d}.png")
                video_frames.append(frame)
            if step % 10 == 0:
                print(
                    f"step {step:04d} | mode={mode_txt} | dist={goal_dist:.2f} | goal_px={int((goal_mask>0).sum())} "
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
