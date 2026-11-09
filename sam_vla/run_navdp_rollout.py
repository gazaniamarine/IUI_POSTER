"""Modular NavDP-diffusion-policy rollout entry point (the sam_vla equivalent
of the legacy scripts/vlm_nav_tests/rollout_navdp*.py): first-frame VLM goal
detection, CBF cone-safety layer, and BeliefGoalTracker, with optional
base-station out-and-back goal sequencing.

Usage:
    conda activate habitat
    # Default single-goal run: --policy-backend navdp-upstream (the real, published
    # NavDP), needs $NAVDP_UPSTREAM_ROOT set / --navdp-upstream-root passed, and picks
    # up navdp/navdp-cross-modal.ckpt automatically if present.
    python -m sam_vla.run_navdp_rollout \
        --scene-path assets/marsyard2022.glb \
        --heightmap-path marsyard2022_terrain_hm_1025.tif \
        --navdp-upstream-root /path/to/InternRobotics/NavDP \
        --start-x 0 --start-z 8 --start-yaw 0 \
        --max-steps 300 --cbf \
        --out-dir outputs/navdp_upstream_smoke_test

    # --multi-goal/--base-station still need the legacy in-house backend
    # (navdp-upstream doesn't support them yet), selected automatically:
    python -m sam_vla.run_navdp_rollout \
        --scene-path assets/marsyard2022.glb \
        --heightmap-path marsyard2022_terrain_hm_1025.tif \
        --ckpt navdp/ckpt_last.pt \
        --navdp-root ./navdp \
        --start-x -2 --start-z 8 --start-yaw 120 \
        --base-station --dwell-seconds 3 --goal-success-radius 1.0 \
        --lost-goal-forward 0.2 \
        --max-steps 900 --cbf \
        --out-dir outputs/base_station_smoke_test
"""

import argparse
import math
import time, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.sim_utils import distance_to_goal
from sam_vla.vlm.qwen_server_manager import QwenServerManager
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.policy.navdp_policy import NavdpPolicy
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance
from sam_vla.core.belief_tracking import (
    BeliefGoalTracker,
    lost_goal_heading_assist,
    mask_to_body,
)
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
    mask_pixel_center,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.sensor_noise import apply_odom_noise
from sam_vla.core.types import Action, GoalSpec
from sam_vla.core.uncertainty_motion import (
    drive_toward_heading,
    resolve_absolute_heading_deg,
    rotate_sweep,
)
from sam_vla.logging.rollout_logger import RolloutLogger
from sam_vla.perception.semantic_overlay import overlay_semantic_masks
from vl_direction.client import get_client
from vl_direction.intervention import mode_flag
from vl_direction.intervention.mode_flag import SessionMode
from vl_direction.qwen_server_manager import QwenServerManager as VlQwenServerManager
from vl_direction.uncertainty_session import UncertaintySession


def register_goal_obstacle_masks(
    env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir
):
    """Give the chosen goal object a goal-mask mesh and every other detected
    object an obstacle-mask mesh, each a disc of `obj_mask_radius` around its
    bbox's backprojected world coords. The rest of the scene is untouched."""
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


def register_obstacle_masks_only(
    env, obs0, obstacle_bboxes_norm, obj_mask_radius, out_dir
):
    """Same obstacle-registration body as register_goal_obstacle_masks, minus the
    goal-mesh half -- used by the multi-goal path, where goals come from live
    SAM3+CLIP instead of a fixed first-frame bbox, so there's no single
    goal_position to register a goal mesh for."""
    for i, obstacle_bbox in enumerate(obstacle_bboxes_norm):
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


def _body_odom_from_poses(prev_pose, cur_pose):
    """Realized motion from prev_pose to cur_pose as [forward, left, dtheta] in
    prev_pose's body frame -- the inverse of pose_integrator.integrate_mars's
    world-frame projection, same yaw/forward/right convention (see its
    docstring). Feeds SubgoalBeliefBank.update's odom_delta with already-
    realized motion (mirroring navdp.deploy.multi_target_runner's pose-diff
    convention) rather than a forward-looking guess from the not-yet-decided
    action."""
    x0, z0, yaw0 = prev_pose
    x1, z1, yaw1 = cur_pose
    dx_world = x1 - x0
    dz_world = z1 - z0
    cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
    forward = -dx_world * sin_yaw - dz_world * cos_yaw
    right = dx_world * cos_yaw - dz_world * sin_yaw
    dtheta = (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
    return [forward, -right, dtheta]


def _goal_observation(
    mask: np.ndarray,
    depth: np.ndarray,
    min_px: int,
    hfov_deg: float,
    fallback_range: float,
) -> dict:
    """SubgoalBeliefBank observation dict for one goal_id, computed straight from its
    current MESH_GOAL_ID mask via belief_tracking.mask_to_body -- the base-station
    mode's replacement for SAMDepthTargetExtractor (see next.md §1: goals here are
    persistent registered meshes, not live SAM3 detections, so there's nothing to
    extract from a detector)."""
    if int((np.asarray(mask) > 0).sum()) < min_px:
        return {"visible": False, "position": None, "confidence": 0.0}
    height, width = np.asarray(depth).shape[:2]
    seed = mask_to_body(mask, depth, height, width, hfov_deg, fallback_range, min_px)
    if seed is None:
        return {"visible": False, "position": None, "confidence": 0.0}
    return {"visible": True, "position": seed, "confidence": 1.0}


def _multi_goal_resegment(
    sam3_tracker,
    clip_classifier,
    multi_goal_spec,
    route,
    last_known_masks,
    rgb,
    step,
    clip_goal_thresh,
    clip_reid_thresh,
    max_goals,
):
    """One periodic SAM3+CLIP cycle: push the live frame, resegment, classify
    and re-ID each returned mask -- minting a new TrackedGoal (and appending it
    to the route) the first time a physical object clears clip_goal_thresh and
    doesn't match an already-tracked goal's embedding. Returns {goal_id: mask}
    for this cycle, fed to SAMDepthTargetExtractor for the belief-bank update."""
    from sam_vla.core.types import TrackedGoal

    sam3_tracker.push_frame(rgb)
    raw_masks = sam3_tracker.resegment(step)

    goal_masks: dict[str, np.ndarray] = {}
    for mask in raw_masks.values():
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        crop = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        category, score, embedding = clip_classifier.classify(crop)
        if score < clip_goal_thresh:
            continue

        goal_id = clip_classifier.match_or_mint(
            embedding, multi_goal_spec.goals, clip_reid_thresh
        )
        if goal_id not in multi_goal_spec.goals:
            if len(multi_goal_spec.goals) >= max_goals:
                continue
            height, width = rgb.shape[:2]
            bbox_norm = (
                float(xs.min()) / width,
                float(ys.min()) / height,
                float(xs.max() + 1) / width,
                float(ys.max() + 1) / height,
            )
            multi_goal_spec.goals[goal_id] = TrackedGoal(
                goal_id=goal_id,
                category=category,
                clip_embedding=embedding,
                clip_score=score,
                first_seen_step=step,
                bbox_norm=bbox_norm,
            )
            route.append(goal_id)
            print(
                f"[multi-goal] minted goal {goal_id} category={category!r} score={score:.3f}",
                flush=True,
            )

        goal_masks[goal_id] = mask
        last_known_masks[goal_id] = mask

    return goal_masks


_UNCERTAINTY_HEADING_LAYOUT = {
    # key -> (angle_deg, grid_row, grid_col); mirrors kb_teleop_vl.py's
    # UNCERTAINTY_HEADING_KEYS numpad convention (8=front/0, 6=right/+90,
    # 4=left/-90, 2=back/180, positive=clockwise) so a human used to that
    # live demo sees the same layout here.
    "7": (-45.0, 0, 0),
    "8": (0.0, 0, 1),
    "9": (45.0, 0, 2),
    "4": (-90.0, 1, 0),
    "6": (90.0, 1, 2),
    "1": (-135.0, 2, 0),
    "2": (180.0, 2, 1),
    "3": (135.0, 2, 2),
}

# Keyword -> angle_deg, used to guess a "recommended" heading from the VLM's
# free-text sweep description so the popup can pre-highlight one button.
# Purely a cosmetic nudge (next.md Goal 2 open question) -- never trusted as
# a parsed decision, the human still has to click to actually pick.
_DIRECTION_KEYWORDS = [
    ("front-left", -45.0),
    ("forward-left", -45.0),
    ("front-right", 45.0),
    ("forward-right", 45.0),
    ("straight ahead", 0.0),
    ("straight", 0.0),
    ("forward", 0.0),
    ("front", 0.0),
    ("ahead", 0.0),
    ("left", -90.0),
    ("right", 90.0),
    ("behind", 180.0),
    ("back", 180.0),
]


def _guess_recommended_heading(sweep_description):
    text = (sweep_description or "").lower()
    for keyword, angle in _DIRECTION_KEYWORDS:
        if keyword in text:
            return angle
    return None


def _prompt_human_heading(frame_rgb, sweep_description, attempt, max_retries):
    """Study 1 (next.md) Phase 2 human-input surface: a modal CustomTkinter
    popup, blocking (mainloop) until the human picks a heading. Needs a real
    display (DISPLAY set / X available) -- the tradeoff next.md's Phase 2
    open question flagged in choosing this over a headless terminal prompt:
    the human sees the actual sweep frame, but this can't run over a bare
    SSH session. Explicit Cancel button and window-close both fall back to
    0deg (straight ahead), logged distinctly so an accidental close doesn't
    read as a considered "front" pick in the console trail."""
    import customtkinter as ctk
    from PIL import Image

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title(f"Uncertainty handoff -- attempt {attempt + 1}/{max_retries + 1}")
    root.resizable(False, False)
    chosen = {"angle_deg": None, "explicit": False}
    recommended = _guess_recommended_heading(sweep_description)

    header = ctk.CTkFrame(root, fg_color="transparent")
    header.pack(fill="x", padx=16, pady=(14, 6))
    ctk.CTkLabel(
        header,
        text="Rover is uncertain -- pick a heading",
        font=ctk.CTkFont(size=17, weight="bold"),
    ).pack(side="left")
    ctk.CTkLabel(
        header,
        text=f"attempt {attempt + 1} / {max_retries + 1}",
        font=ctk.CTkFont(size=13),
        text_color="#f59e0b" if attempt > 0 else "#94a3b8",
    ).pack(side="right")

    # Larger/clearer sweep-frame display (next.md Goal 2, point 1): upscale
    # small sim frames rather than showing them at native (often tiny) size.
    img = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8))
    display_w = max(img.width, 480)
    scale = display_w / img.width
    display_size = (display_w, int(img.height * scale))
    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=display_size)
    img_label = ctk.CTkLabel(root, image=ctk_img, text="")
    img_label.pack(padx=16, pady=4)

    ctk.CTkLabel(
        root,
        text=f"VLM sweep: {sweep_description}",
        wraplength=display_w,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=16, pady=(4, 10))

    def pick(angle, explicit=True):
        chosen["angle_deg"] = angle
        chosen["explicit"] = explicit
        root.destroy()

    grid = ctk.CTkFrame(root, fg_color="transparent")
    grid.pack(padx=16, pady=(0, 6))
    for key, (angle, r, c) in _UNCERTAINTY_HEADING_LAYOUT.items():
        is_recommended = recommended is not None and angle == recommended
        btn = ctk.CTkButton(
            grid,
            text=f"{key}: {angle:+.0f}°" + (" ★" if is_recommended else ""),
            width=96,
            height=56,
            fg_color="#15803d" if is_recommended else None,
            hover_color="#166534" if is_recommended else None,
            command=lambda a=angle: pick(a),
        )
        btn.grid(row=r, column=c, padx=5, pady=5)
    if recommended is not None:
        ctk.CTkLabel(
            root,
            text="★ = direction the VLM's sweep description implies",
            font=ctk.CTkFont(size=11),
            text_color="#4ade80",
        ).pack(pady=(0, 6))

    footer = ctk.CTkFrame(root, fg_color="transparent")
    footer.pack(fill="x", padx=16, pady=(4, 14))
    ctk.CTkButton(
        footer,
        text="Cancel (defaults to 0°/front)",
        fg_color="#7f1d1d",
        hover_color="#991b1b",
        command=lambda: pick(0.0, explicit=True),
    ).pack(side="right")

    def on_close():
        pick(0.0, explicit=False)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    if chosen["angle_deg"] is None:
        chosen["angle_deg"] = 0.0
    if not chosen["explicit"]:
        print(
            "[uncertainty] popup window closed without an explicit pick -- "
            "defaulting to 0.0deg (front)"
        )
    return chosen["angle_deg"]


def _run_uncertainty_handoff(
    env,
    pose,
    step,
    episode_id,
    client,
    current_uncertainty,
    uncertainty_threshold,
    dt,
    sweep_degrees_per_step,
    sweep_steps,
    max_units,
    max_retries,
    drive_v_fwd,
    lost_goal_min_px,
    prompt_fn=None,
):
    """Study 1 (next.md) Phase 2, Condition A: pause normal policy stepping
    and run the stop-rotate-ask-drive-retry cycle around the real
    UncertaintySession + Phase 1's rotate_sweep/drive_toward_heading. A
    fresh UncertaintySession is built per call (covariance_value = the live
    uncertainty at this trigger) rather than reused across triggers, since
    UncertaintySession.covariance_value is fixed at construction -- see
    next.md's non-goal against changing vl_direction's own contract.

    prompt_fn(frame, sweep_description, attempt, max_retries) -> angle_deg is
    the human-input surface (a Tk popup by default, see
    _prompt_human_heading) -- injectable so this orchestration is testable
    with a canned prompt_fn and no real display/VLM.

    Returns a dict: final_pose, attempt, resolved, vlm_inference_ms,
    human_decision_ms, drive_ms.
    """
    prompt_fn = prompt_fn if prompt_fn is not None else _prompt_human_heading
    session = UncertaintySession(
        episode_id=episode_id,
        covariance_threshold=uncertainty_threshold,
        covariance_value=current_uncertainty,
        client=client,
    )

    def goal_visible_fn(obs):
        return int((np.asarray(obs.semantic) == MESH_GOAL_ID).sum()) >= lost_goal_min_px

    reference_yaw = pose.yaw
    cur_pose = pose
    vlm_inference_ms = 0.0
    human_decision_ms = 0.0
    drive_ms = 0.0
    attempt = 0
    resolved = False

    mode_flag.set_mode(SessionMode.HUMAN_INTERVENED)
    try:
        while attempt <= max_retries and not resolved:
            frames = rotate_sweep(
                env,
                cur_pose,
                sweep_degrees_per_step,
                sweep_steps,
                dt=dt,
                frame_idx_start=step,
            )
            cur_pose = frames[-1][0]
            last_frame = frames[-1][1]

            vlm_result = (
                session.request_human_heading(last_frame)
                if attempt == 0
                else session.retry(last_frame)
            )
            vlm_inference_ms += vlm_result.latency_ms
            sweep_description = vlm_result.raw_response

            t_human = time.monotonic()
            angle_deg = prompt_fn(last_frame, sweep_description, attempt, max_retries)
            human_decision_ms += (time.monotonic() - t_human) * 1000.0

            submit_result = session.submit_heading(
                angle_deg=angle_deg, max_units=max_units
            )
            payload = submit_result.uncertainty_payload
            target_heading_deg = resolve_absolute_heading_deg(reference_yaw, angle_deg)

            t_drive = time.monotonic()
            cur_pose, _units, found = drive_toward_heading(
                env,
                cur_pose,
                target_heading_deg,
                payload.max_units,
                goal_visible_fn,
                v_fwd=drive_v_fwd,
                dt=dt,
                frame_idx_start=step,
            )
            drive_ms += (time.monotonic() - t_drive) * 1000.0
            resolved = found
            attempt += 1
    finally:
        mode_flag.reset()

    return {
        "final_pose": cur_pose,
        "attempt": attempt,
        "resolved": resolved,
        "vlm_inference_ms": vlm_inference_ms,
        "human_decision_ms": human_decision_ms,
        "drive_ms": drive_ms,
    }


def _default_navdp_upstream_ckpt() -> Optional[str]:
    """Repo-relative fallback for --navdp-upstream-ckpt, mirroring navdp_policy.py's
    in-repo fallback for --navdp-root: navdp-upstream is now the default policy
    backend, so a plain invocation should just work on a checkout that already has
    the (gitignored) checkpoint at navdp/navdp-cross-modal.ckpt, same path Phase 0
    of next.md's Integration project downloaded it to."""
    candidate = Path(__file__).resolve().parent.parent / "navdp" / "navdp-cross-modal.ckpt"
    return str(candidate) if candidate.exists() else None


def run(
    scene_path: str,
    heightmap_path: str,
    ckpt_path: str,
    out_dir: str,
    navdp_root: str = None,
    device: str = "cuda",
    sample_steps: int = 20,
    max_steps: int = 500,
    dt: float = 0.1,
    save_video: bool = False,
    save_frames: bool = False,
    video_fps: int = 30,
    start_x: float = 0.0,
    start_z: float = 8.0,
    start_yaw_deg: float = 0.0,
    randomise_spawn: bool = False,
    rock_field_path: str = None,
    obj_mask_radius: float = 0.5,
    cbf: bool = False,
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
    belief_odom_noise: float = 0.0,
    drive_odom_noise_std: float = 0.0,
    drive_odom_noise_seed: int = None,
    lost_goal_min_px: int = 10,
    lost_goal_ghost: bool = False,
    lost_goal_turn_kp: float = 1.4,
    lost_goal_forward: float = 0.0,
    lost_goal_bearing_deg: float = 30.0,
    multi_goal: bool = False,
    seg_interval_steps: int = 13,
    sam3_window_frames: int = 5,
    sam3_checkpoint: str = None,
    clip_goal_thresh: float = 0.24,
    clip_reid_thresh: float = 0.9,
    goal_vocab: str = None,
    stop_on_route_finished: bool = True,
    max_goals: int = 8,
    base_station: bool = False,
    dwell_seconds: float = 5.0,
    goal_success_radius: float = 1.0,
    base_marker_radius: float = None,
    uncertainty_threshold: float = None,
    uncertainty_condition: str = "autonomous",
    uncertainty_max_retries: int = 3,
    uncertainty_max_units: float = None,
    uncertainty_sweep_degrees_per_step: float = 45.0,
    uncertainty_sweep_steps: int = 8,
    uncertainty_drive_v_fwd: float = 0.5,
    uncertainty_mock_client: bool = False,
    policy_backend: str = None,
    navdp_upstream_ckpt: str = None,
    navdp_upstream_root: str = None,
    navdp_upstream_port: int = None,
    navdp_upstream_stop_threshold: float = 0.0,
    navdp_upstream_lookahead: int = 3,
    navdp_upstream_replan_every: int = 1,
    navdp_upstream_max_forward_speed: float = 1.0,
    navdp_upstream_turn_kp: float = 1.4,
    navdp_upstream_request_timeout: float = 30.0,
) -> None:
    if base_station and multi_goal:
        raise ValueError(
            "--base-station and --multi-goal are mutually exclusive -- they solve "
            "different problems, see next.md §1"
        )
    if policy_backend is None:
        # Default to the real, published NavDP model (navdp-upstream) -- navdp-s2dit
        # (this repo's own in-house model) is now the legacy fallback, kept only for
        # --multi-goal/--base-station, which navdp-upstream doesn't support yet (see
        # next.md's Integration project non-goals).
        policy_backend = "navdp-s2dit" if (multi_goal or base_station) else "navdp-upstream"
    if policy_backend not in ("navdp-s2dit", "navdp-upstream"):
        raise ValueError(
            f"--policy-backend must be 'navdp-s2dit' or 'navdp-upstream', got {policy_backend!r}"
        )
    if policy_backend == "navdp-upstream":
        if multi_goal or base_station:
            raise ValueError(
                "--policy-backend navdp-upstream only supports the single-goal path today "
                "-- pass --policy-backend navdp-s2dit (legacy) for --multi-goal/--base-station"
            )
        if not navdp_upstream_ckpt:
            navdp_upstream_ckpt = _default_navdp_upstream_ckpt()
        if not navdp_upstream_ckpt:
            raise ValueError(
                "--navdp-upstream-ckpt is required with --policy-backend navdp-upstream "
                "(no checkpoint found at the default navdp/navdp-cross-modal.ckpt either)"
            )
    elif not ckpt_path:
        raise ValueError("--ckpt is required with --policy-backend navdp-s2dit (legacy)")
    # Both --multi-goal and --base-station need `from navdp.extensions import ...`
    # before NavdpPolicy is constructed below (that's normally what puts navdp_root
    # on sys.path, via its own _add_navdp_to_path call) -- do it here first so those
    # earlier imports resolve regardless of setup order.
    from sam_vla.policy.navdp_policy import _add_navdp_to_path, _resolve_navdp_root

    _add_navdp_to_path(_resolve_navdp_root(navdp_root))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Still needed for the one-shot first-frame goal selection below
    # (first_frame_resolver -> qwen_client.select_goal_verbose); the driving
    # loop itself no longer calls the VLM per frame -- NavdpPolicy drives.
    qwen_manager = QwenServerManager()
    logger = RolloutLogger()

    # Study 1 (next.md) Phase 2, Condition A: only spin up vl_direction's own
    # Qwen server + client when a human-in-the-loop handoff can actually fire
    # (single-goal mode, threshold set, condition asked for). get_client("qwen")
    # is used rather than the InternVL backend -- InternVL's model runner is
    # still NotImplementedError stubs (see vl_direction/client.py's docstring),
    # so "qwen" is the only backend that's actually wired end-to-end today,
    # exactly what kb_teleop_vl.py's real, working integration uses.
    human_handoff_enabled = (
        uncertainty_condition == "human"
        and uncertainty_threshold is not None
        and not multi_goal
        and not base_station
    )
    vl_qwen_manager = None
    uncertainty_client = None
    if human_handoff_enabled:
        uncertainty_client = get_client("mock" if uncertainty_mock_client else "qwen")
        if not uncertainty_mock_client:
            vl_qwen_manager = VlQwenServerManager()
    uncertainty_model_load_charged = False

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        services=[qwen_manager] + ([vl_qwen_manager] if vl_qwen_manager else []),
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        randomise_spawn=randomise_spawn,
        with_semantic=True,
        rock_field_path=rock_field_path,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)

        # multi-goal state -- left None on the (default) single-goal path, which
        # takes the exact same code path as before this feature existed.
        goal_spec = None
        goal_position = None
        multi_goal_spec = bank = route = sam3_tracker = clip_classifier = (
            target_extractor
        ) = None
        last_known_masks: dict = {}
        prev_body_pose = (obs0.pose.x, obs0.pose.z, obs0.pose.yaw)

        # base-station state -- left None/unset on every other path.
        goal_1_obj = base_station_obj = base_position = None
        phase = "OUTBOUND"
        dwell_steps_total = dwell_steps_remaining = 0

        if multi_goal:
            from navdp.extensions import (
                RouteManager,
                SAMDepthTargetExtractor,
                SubgoalBeliefBank,
            )

            from sam_vla.core.types import MultiGoalSpec
            from sam_vla.goal_resolution import goal_vocabulary_resolver
            from sam_vla.goal_resolution.first_frame_resolver import resolve_obstacles
            from sam_vla.perception.clip_goal_classifier import ClipGoalClassifier
            from sam_vla.perception.sam3_goal_tracker import Sam3GoalTracker

            cli_vocab = (
                [t.strip() for t in goal_vocab.split(",") if t.strip()]
                if goal_vocab
                else None
            )
            vocab_terms, instruction_text = (
                goal_vocabulary_resolver.resolve_goal_vocabulary(
                    obs0.rgb, cli_vocab, use_qwen=cli_vocab is None
                )
            )
            obstacle_detections = resolve_obstacles(obs0.rgb)
            obstacle_bboxes = [d.bbox_norm for d in obstacle_detections]
            register_obstacle_masks_only(
                env, obs0, obstacle_bboxes, obj_mask_radius, out_dir
            )

            sam3_tracker = Sam3GoalTracker(
                vocab_terms=vocab_terms,
                window_frames=sam3_window_frames,
                checkpoint_path=sam3_checkpoint,
            )
            clip_classifier = ClipGoalClassifier(goal_vocabulary=vocab_terms)
            height0, width0 = obs0.rgb.shape[:2]
            target_extractor = SAMDepthTargetExtractor(
                intrinsics_from_hfov(height0, width0, HFOV_DEG)
            )
            multi_goal_spec = MultiGoalSpec(
                goals={},
                route=[],
                obstacle_bboxes_norm=obstacle_bboxes,
                instruction_text=instruction_text,
                goal_vocabulary=vocab_terms,
            )
            bank = SubgoalBeliefBank(goal_ids=[])
            route = RouteManager(route=[])
            print(
                f"[multi-goal] vocabulary={vocab_terms} instruction={instruction_text!r}",
                flush=True,
            )
        elif base_station:
            from navdp.extensions import RouteManager, SubgoalBeliefBank

            goal_spec, goal_vlm_result, sam_detections = (
                first_frame_resolver.resolve_verbose(obs0.rgb)
            )
            goal_position = backproject_goal_position(
                obs0, goal_spec, hfov_deg=HFOV_DEG
            )
            logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
            logger.save_sam_first_frame(obs0.rgb, sam_detections, goal_spec, out_dir)
            print(
                f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}"
            )

            goal_1_obj = None
            if goal_position is not None:
                goal_1_obj = env.register_object_mask(
                    goal_position, MESH_GOAL_ID, obj_mask_radius, out_dir, "goal"
                )
            else:
                print(
                    "[WARN] goal bbox had no valid depth; skipping goal mask",
                    flush=True,
                )
            register_obstacle_masks_only(
                env, obs0, goal_spec.obstacle_bboxes_norm, obj_mask_radius, out_dir
            )

            # Base station = the rover's own spawn pose, not a detection -- registered
            # with semantic_id=0 (neutral: invisible to both goal/obstacle channels
            # until DWELL re-tags it), per next.md §3. terrain_patch_mesh ignores
            # world_pos's own y and resamples height from self._terrain itself, so
            # there's no second height lookup to keep in sync with register_object_mask's.
            base_marker_r = (
                base_marker_radius
                if base_marker_radius is not None
                else obj_mask_radius
            )
            base_position = (start_x, 0.0, start_z)
            base_station_obj = env.register_object_mask(
                base_position, 0, base_marker_r, out_dir, "base_station"
            )

            route = RouteManager(
                route=["goal_1", "base_station"], success_radius=goal_success_radius
            )
            bank = SubgoalBeliefBank(goal_ids=["goal_1", "base_station"])
            # Seed base_station's belief as "seen at [0, 0]" (the robot's own local
            # frame right now) since the rover spawns exactly there -- without this,
            # the slot stays uninitialized (belief_bank.py's update() resets it to
            # zero/large-uncertainty every step instead of dead-reckoning) until the
            # marker happens to be glimpsed in-frame, which next.md §6 explicitly
            # says may never happen on the way back. Seeding here means ordinary
            # per-step odometry dead-reckoning (already running every OUTBOUND step
            # below) tracks the true return offset the whole way out.
            bank.update(
                {
                    "base_station": {
                        "visible": True,
                        "position": [0.0, 0.0],
                        "confidence": 1.0,
                    }
                },
                odom_delta=[0.0, 0.0, 0.0],
                step=-1,
            )
            phase = "OUTBOUND"
            dwell_steps_total = max(round(dwell_seconds / dt), 0)
            dwell_steps_remaining = 0
            print(
                f"[base-station] base_position={base_position} "
                f"dwell_steps={dwell_steps_total} success_radius={goal_success_radius}",
                flush=True,
            )
        else:
            goal_spec, goal_vlm_result, sam_detections = (
                first_frame_resolver.resolve_verbose(obs0.rgb)
            )
            goal_position = backproject_goal_position(
                obs0, goal_spec, hfov_deg=HFOV_DEG
            )
            logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
            logger.save_sam_first_frame(obs0.rgb, sam_detections, goal_spec, out_dir)
            print(
                f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}"
            )
            register_goal_obstacle_masks(
                env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir
            )

        # <-- policy plugged in here. --policy-backend selects between
        # NavdpUpstreamPolicy (the real published NavDP via its HTTP server, see
        # next.md's "Integration project" section -- default for single-goal runs)
        # and NavdpPolicy (this repo's own in-house S2DiT/VL3-DP model, now the
        # legacy backend, auto-selected for --multi-goal/--base-station, which
        # navdp-upstream doesn't support yet) -- both share the same
        # act_verbose(obs, semantic, goal_spec, step) -> (Action, dict) shape, so
        # the loop below needs no branching beyond this construction.
        if policy_backend == "navdp-upstream":
            policy = NavdpUpstreamPolicy(
                checkpoint_path=navdp_upstream_ckpt,
                navdp_upstream_root=navdp_upstream_root,
                port=navdp_upstream_port,
                stop_threshold=navdp_upstream_stop_threshold,
                image_hw=obs0.rgb.shape[:2],
                hfov_deg=HFOV_DEG,
                lookahead=navdp_upstream_lookahead,
                replan_every=navdp_upstream_replan_every,
                max_forward_speed=navdp_upstream_max_forward_speed,
                turn_kp=navdp_upstream_turn_kp,
                max_yaw_rate=max_yaw_rate,
                request_timeout=navdp_upstream_request_timeout,
            )
        else:
            policy = NavdpPolicy(
                ckpt_path=ckpt_path,
                navdp_root=navdp_root,
                device=device,
                sample_steps=sample_steps,
            )

        # Belief-goal tracking: re-seed a body-frame [forward, left] estimate of the
        # goal from the live rendered mask whenever it's visible, dead-reckon it by
        # odometry the rest of the time -- ported from rollout_navdp_policy.py's
        # mesh_tracking_mode. In multi-goal mode the real SubgoalBeliefBank (built
        # above) plays this role per-goal instead. avoidance is None unless --cbf
        # is passed (constructed after NavdpPolicy so navdp.extensions is
        # importable -- see its docstring).
        belief_tracker = (
            None
            if (multi_goal or base_station)
            else BeliefGoalTracker(
                hfov_deg=HFOV_DEG,
                goal_range=belief_goal_range,
                min_px=lost_goal_min_px,
                odom_noise=belief_odom_noise,
            )
        )
        avoidance = (
            CbfObstacleAvoidance(
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
            if cbf
            else None
        )
        cbf_active_steps = 0
        hard_gate_fired_steps = 0
        # Study 2 (next.md): one RNG for the whole episode, so --drive-odom-noise-seed
        # reproduces the exact noise sequence run-to-run. Unused (no draws) when
        # drive_odom_noise_std <= 0.0.
        drive_noise_rng = np.random.default_rng(drive_odom_noise_seed)
        # Study 1 Phase 0 (next.md): rising-edge trigger only -- logged once per
        # crossing, not every step spent above threshold, so a long occlusion
        # doesn't spam one uncertainty_trigger event per step.
        uncertainty_was_above = False

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            semantic = env.get_semantic_frame()

            active_goal_id = None
            active_slot = None
            route_finished_this_step = False
            just_entered_dwell = False
            phase_for_log = phase
            navdp_upstream_goal_unresolved = False
            if multi_goal:
                goal_masks = {}
                if step % seg_interval_steps == 0:
                    goal_masks = _multi_goal_resegment(
                        sam3_tracker,
                        clip_classifier,
                        multi_goal_spec,
                        route,
                        last_known_masks,
                        obs.rgb,
                        step,
                        clip_goal_thresh,
                        clip_reid_thresh,
                        max_goals,
                    )
                obs_extract = (
                    target_extractor.extract(goal_masks, obs.depth)
                    if goal_masks
                    else {}
                )
                observations = {
                    gid: obs_extract.get(
                        gid, {"visible": False, "position": None, "confidence": 0.0}
                    )
                    for gid in multi_goal_spec.goals
                }
                cur_body_pose = (obs.pose.x, obs.pose.z, obs.pose.yaw)
                odom_delta = _body_odom_from_poses(prev_body_pose, cur_body_pose)
                prev_body_pose = cur_body_pose
                bank.update(observations, odom_delta=odom_delta, step=step)
                route.update(robot_position=[0.0, 0.0], belief_bank=bank)

                if not route.is_finished():
                    active_goal_id = route.get_active_goal()
                    active_slot = bank.get(active_goal_id)
                goal_visible = (
                    bool(active_slot.visible) if active_slot is not None else False
                )
                goal_bearing = (
                    math.atan2(float(active_slot.mu[1]), float(active_slot.mu[0]))
                    if (active_slot is not None and active_slot.initialized)
                    else None
                )

                semantic_render = semantic.copy()
                active_mask = (
                    last_known_masks.get(active_goal_id) if active_goal_id else None
                )
                if active_mask is not None:
                    semantic_render[active_mask] = MESH_GOAL_ID
                goal_spec_for_policy = GoalSpec(
                    goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                    obstacle_bboxes_norm=[],
                    instruction_text=multi_goal_spec.instruction_text,
                )
            elif base_station:
                if phase in ("OUTBOUND", "RETURN"):
                    active_goal_id = route.get_active_goal()
                    active_mask = semantic == MESH_GOAL_ID
                    observations = {
                        gid: (
                            _goal_observation(
                                active_mask,
                                obs.depth,
                                lost_goal_min_px,
                                HFOV_DEG,
                                belief_goal_range,
                            )
                            if gid == active_goal_id
                            else {"visible": False}
                        )
                        for gid in ("goal_1", "base_station")
                    }
                    cur_body_pose = (obs.pose.x, obs.pose.z, obs.pose.yaw)
                    odom_delta = _body_odom_from_poses(prev_body_pose, cur_body_pose)
                    prev_body_pose = cur_body_pose
                    bank.update(observations, odom_delta=odom_delta, step=step)
                    route_status = route.update(
                        robot_position=[0.0, 0.0], belief_bank=bank
                    )
                    active_slot = bank.get(active_goal_id)
                    goal_visible = bool(active_slot.visible)
                    goal_bearing = (
                        math.atan2(float(active_slot.mu[1]), float(active_slot.mu[0]))
                        if active_slot.initialized
                        else None
                    )

                    if route_status["advanced"]:
                        if phase == "OUTBOUND":
                            # Re-tag now, immediately after this step's semantic frame
                            # was already fetched+consumed above (still correctly
                            # OUTBOUND-tagged), so this same step finishes out as a
                            # normal (final) OUTBOUND drive step below, and the *next*
                            # get_semantic_frame() call -- the first real DWELL step's,
                            # at the top of the next loop iteration -- is the first one
                            # to see the swap. Retagging any later would leave a frame
                            # with two goal-tagged regions at once (next.md §3/§6);
                            # retagging into this step's own semantic_render would
                            # instead make this step's already-computed belief/route
                            # update inconsistent with what it drove on.
                            if goal_1_obj is not None:
                                goal_1_obj.semantic_id = MESH_OBST_ID
                            if base_station_obj is not None:
                                base_station_obj.semantic_id = MESH_GOAL_ID
                            phase = "DWELL"
                            dwell_steps_remaining = dwell_steps_total
                            just_entered_dwell = True
                        else:  # RETURN -> DONE
                            route_finished_this_step = True

                    phase_for_log = (
                        "DONE"
                        if route_finished_this_step
                        else "OUTBOUND" if just_entered_dwell else phase
                    )

                semantic_render = semantic
                goal_spec_for_policy = goal_spec
            else:
                semantic_render = semantic
                goal_spec_for_policy = goal_spec
                if policy_backend == "navdp-upstream":
                    # NavdpUpstreamPolicy needs this step's body-frame goal point
                    # *before* act_verbose() (it's the request payload, not
                    # something derived after the fact like NavdpPolicy's mask
                    # conditioning is) -- prime belief_tracker from this step's
                    # fresh mask now; the unconditional belief_tracker.observe()
                    # call below (after act_verbose) re-runs on the same
                    # mask/depth and is a harmless no-op repeat for this backend.
                    pregoal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
                    belief_tracker.observe(pregoal_mask, obs.depth)
                    if belief_tracker.belief_g is not None:
                        goal_forward, goal_left = belief_tracker.belief_g
                        policy.set_goal_body(goal_forward, goal_left)
                    else:
                        navdp_upstream_goal_unresolved = True

            if navdp_upstream_goal_unresolved:
                # belief_tracker has never seen the goal mask with valid enough
                # pixels/depth this episode (e.g. the annotated goal marker is
                # on-screen but the mask is a thin, grazing-angle sliver below
                # lost_goal_min_px, or its depth patch is invalid), so
                # belief_g is still None. Calling act_verbose() now would send
                # NavdpUpstreamPolicy's own hidden constructor default
                # (default_goal_forward=5.0, default_goal_left=0.0 -- "go
                # straight") to the server, which drives the rover confidently
                # in an arbitrary direction while every log field *looks* like
                # real goal-seeking. That silent fallback is exactly what next.md's
                # Integration-project Phase 4 run hit. Hold instead of guessing,
                # and say so loudly -- the moment the mask/depth resolve (even
                # one frame with a valid sighting), the branch above takes over
                # and driving resumes with a real point.
                action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                new_pose = obs.pose
                env.step(new_pose)
                vis_rgb = overlay_semantic_masks(
                    obs.rgb, semantic_render, text=f"t={step} WAITING FOR GOAL SIGHTING"
                )
                vla_result = {"goal_observed": False}
                logger.log_step(
                    obs, action, new_pose, vla_result=vla_result, vis_rgb=vis_rgb
                )
                if step % 10 == 0:
                    print(
                        f"[navdp-upstream] step={step} goal never observed yet -- "
                        f"holding instead of driving on a fabricated default goal point",
                        flush=True,
                    )
                continue

            if base_station and phase == "DWELL" and not just_entered_dwell:
                # Hold position -- skip the policy/CBF call entirely (next.md §4) but
                # keep stepping/logging so the saved video shows a clean stationary
                # hold instead of a gap.
                action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                new_pose = obs.pose
                env.step(new_pose)
                dwell_steps_remaining -= 1

                dist = (
                    distance_to_goal(new_pose, base_position)
                    if base_position is not None
                    else None
                )
                dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
                overlay_text = (
                    f"t={step} DWELL dist={dist_txt} v=[0.00,0.00] yaw_rate=0.00"
                )
                vis_rgb = overlay_semantic_masks(
                    obs.rgb, semantic_render, text=overlay_text
                )
                goal_mask = (semantic_render == MESH_GOAL_ID).astype("uint8") * 255
                vla_result = {
                    "phase": "DWELL",
                    "route_index": route.get_route_index(),
                    "dwell_steps_remaining": max(dwell_steps_remaining, 0),
                }
                logger.log_step(
                    obs, action, new_pose, vla_result=vla_result, vis_rgb=vis_rgb
                )
                if step % 10 == 0:
                    print(
                        f"[traj] step={step} | phase=DWELL | "
                        f"dwell_steps_remaining={max(dwell_steps_remaining, 0)}"
                    )
                if dwell_steps_remaining <= 0:
                    phase = "RETURN"
                continue

            raw_action, vla_result = policy.act_verbose(
                obs, semantic_render, goal_spec_for_policy, step
            )
            action = safety_filter_fn(raw_action, obs)

            goal_mask = (semantic_render == MESH_GOAL_ID).astype("uint8") * 255
            obstacle_mask = (semantic_render == MESH_OBST_ID).astype("uint8") * 255
            if not multi_goal and not base_station:
                goal_visible = belief_tracker.observe(goal_mask, obs.depth)
                goal_bearing = belief_tracker.bearing()

            obstacle_point = None
            if avoidance is not None:
                height, width = obs.depth.shape[:2]
                intr = intrinsics_from_hfov(height, width, HFOV_DEG)
                obstacle_point = avoidance.nearest_obstacle(
                    obstacle_mask, obs.depth, intr
                )

            blocked = (
                avoidance.is_blocked(obstacle_point, goal_bearing)
                if avoidance is not None
                else False
            )

            if lost_goal_ghost and not blocked and goal_bearing is not None:
                action = lost_goal_heading_assist(
                    action,
                    goal_bearing,
                    goal_lost=not goal_visible,
                    turn_kp=lost_goal_turn_kp,
                    forward_floor=lost_goal_forward,
                    bearing_deg_thresh=lost_goal_bearing_deg,
                    max_yaw_rate=max_yaw_rate,
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

            # Study 2 (next.md): commanded_action is what the policy/CBF actually
            # decided (also what belief_tracker dead-reckons on below -- a real
            # robot's odometry model integrates its own commanded/measured
            # velocity, not ground truth). executed_action is what really moves
            # the rover, corrupted by --drive-odom-noise-std to simulate real
            # actuator/wheel-slip noise. Keeping these separate (rather than
            # perturbing `action` in place before belief_tracker.propagate) is
            # what makes this a real belief-vs-reality mismatch instead of a
            # noise the belief tracker is silently told about -- see next.md's
            # Study 2 injection-point note.
            commanded_action = action
            executed_action = (
                apply_odom_noise(commanded_action, drive_odom_noise_std, drive_noise_rng)
                if drive_odom_noise_std > 0.0
                else commanded_action
            )
            new_pose = integrate_mars(obs.pose, executed_action, dt)
            env.step(new_pose)
            action = executed_action
            uncertainty_event = None
            if not multi_goal and not base_station:
                belief_tracker.propagate(commanded_action, dt)
                if uncertainty_threshold is not None:
                    current_uncertainty = belief_tracker.uncertainty_value()
                    is_above = current_uncertainty >= uncertainty_threshold
                    if is_above and not uncertainty_was_above:
                        # Study 1 Phase 2: Condition A (human_handoff_enabled) pauses
                        # normal policy stepping and runs the stop-rotate-ask-drive-
                        # retry cycle; Condition B (the default) just logs and lets
                        # the policy keep driving on its own belief -- see next.md's
                        # "What Condition B means" section.
                        handoff = None
                        if human_handoff_enabled:
                            handoff = _run_uncertainty_handoff(
                                env=env,
                                pose=new_pose,
                                step=step,
                                episode_id=Path(out_dir).name,
                                client=uncertainty_client,
                                current_uncertainty=current_uncertainty,
                                uncertainty_threshold=uncertainty_threshold,
                                dt=dt,
                                sweep_degrees_per_step=uncertainty_sweep_degrees_per_step,
                                sweep_steps=uncertainty_sweep_steps,
                                max_units=uncertainty_max_units,
                                max_retries=uncertainty_max_retries,
                                drive_v_fwd=uncertainty_drive_v_fwd,
                                lost_goal_min_px=lost_goal_min_px,
                            )
                            new_pose = handoff["final_pose"]

                        model_load_ms = None
                        if human_handoff_enabled:
                            if vl_qwen_manager is None:
                                model_load_ms = 0.0  # mock client, nothing to load
                            elif not uncertainty_model_load_charged:
                                model_load_ms = vl_qwen_manager.load_ms
                                uncertainty_model_load_charged = True
                            else:
                                model_load_ms = 0.0  # already charged this episode

                        uncertainty_event = {
                            "type": "uncertainty_trigger",
                            "step": step,
                            "uncertainty": current_uncertainty,
                            "threshold": uncertainty_threshold,
                            "belief_forward": (
                                None
                                if belief_tracker.belief_g is None
                                else float(belief_tracker.belief_g[0])
                            ),
                            "belief_left": (
                                None
                                if belief_tracker.belief_g is None
                                else float(belief_tracker.belief_g[1])
                            ),
                            "bearing": belief_tracker.bearing(),
                            "distance": belief_tracker.distance(),
                            "condition": uncertainty_condition,
                            "attempt": handoff["attempt"] if handoff else 0,
                            "resolved": handoff["resolved"] if handoff else None,
                            "model_load_ms": model_load_ms,
                            "vlm_inference_ms": (
                                handoff["vlm_inference_ms"] if handoff else None
                            ),
                            "human_decision_ms": (
                                handoff["human_decision_ms"] if handoff else None
                            ),
                            "drive_ms": handoff["drive_ms"] if handoff else None,
                        }
                        print(
                            f"[uncertainty] trigger at step={step} "
                            f"uncertainty={current_uncertainty:.4f} "
                            f"(threshold={uncertainty_threshold}) "
                            f"condition={uncertainty_condition} "
                            f"resolved={uncertainty_event['resolved']}",
                            flush=True,
                        )
                    uncertainty_was_above = is_above

            if base_station:
                dist_target = (
                    goal_position if active_goal_id == "goal_1" else base_position
                )
            else:
                dist_target = goal_position
            dist = (
                distance_to_goal(new_pose, dist_target)
                if dist_target is not None
                else None
            )
            dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
            overlay_text = (
                f"t={step} dist={dist_txt} "
                f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f}"
            )
            vis_rgb = overlay_semantic_masks(
                obs.rgb, semantic_render, text=overlay_text
            )
            if multi_goal:
                vla_result = {
                    **vla_result,
                    "belief_forward": (
                        None if active_slot is None else float(active_slot.mu[0])
                    ),
                    "belief_left": (
                        None if active_slot is None else float(active_slot.mu[1])
                    ),
                    "goal_visible": goal_visible,
                    "active_goal_id": active_goal_id,
                    "route_index": route.get_route_index(),
                    "num_goals": len(multi_goal_spec.goals),
                    **cbf_info,
                }
            elif base_station:
                vla_result = {
                    **vla_result,
                    "phase": phase_for_log,
                    "belief_forward": (
                        None if active_slot is None else float(active_slot.mu[0])
                    ),
                    "belief_left": (
                        None if active_slot is None else float(active_slot.mu[1])
                    ),
                    "goal_visible": goal_visible,
                    "active_goal_id": active_goal_id,
                    "route_index": route.get_route_index(),
                    **cbf_info,
                }
            else:
                vla_result = {
                    **vla_result,
                    "belief_forward": (
                        None
                        if belief_tracker.belief_g is None
                        else float(belief_tracker.belief_g[0])
                    ),
                    "belief_left": (
                        None
                        if belief_tracker.belief_g is None
                        else float(belief_tracker.belief_g[1])
                    ),
                    "goal_visible": goal_visible,
                    # Study 2 (next.md): logged every step (not just on threshold
                    # crossing, unlike uncertainty_event) so the full trajectory is
                    # comparable against belief_exp's offline calibration metrics.
                    "uncertainty_value": belief_tracker.uncertainty_value(),
                    **cbf_info,
                }
            logger.log_step(
                obs,
                action,
                new_pose,
                vla_result=vla_result,
                vis_rgb=vis_rgb,
                uncertainty_event=uncertainty_event,
            )

            if step % 10 == 0:
                goal_pixel = mask_pixel_center(goal_mask)
                if multi_goal:
                    extra = f" active_goal={active_goal_id}"
                elif base_station:
                    extra = f" phase={phase_for_log} active_goal={active_goal_id}"
                else:
                    extra = ""
                print(
                    f"[traj] step={step} | distance_to_goal={dist} | "
                    f"goal_pixel={goal_pixel} | action={action}{extra}"
                )

            if (
                multi_goal
                and stop_on_route_finished
                and multi_goal_spec.goals
                and route.is_finished()
            ):
                print(f"[multi-goal] route finished at step={step}", flush=True)
                break

            if base_station and route_finished_this_step:
                print(f"[base-station] route finished at step={step}", flush=True)
                break

        if avoidance is not None:
            print(
                f"[CBF diag] blocked_steps={cbf_active_steps} hard_gate_fired={hard_gate_fired_steps}",
                flush=True,
            )
        if policy_backend == "navdp-upstream":
            policy.stop()
        logger.flush(out_dir)
        if save_frames:
            logger.save_frames(out_dir)
        if save_video:
            logger.save_video(out_dir, fps=video_fps)

    print("[inf] qwen_manager: stop confirmed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to trained NavDP/S2DiT checkpoint (legacy backend; required only with "
        "--policy-backend navdp-s2dit)",
    )
    parser.add_argument(
        "--navdp-root",
        default=None,
        help="Path to the navdp repo (default: ./navdp or $NAVDP_ROOT) -- only used by the "
        "legacy navdp-s2dit backend",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument(
        "--policy-backend",
        choices=["navdp-s2dit", "navdp-upstream"],
        default=None,
        help=(
            "navdp-upstream (default for single-goal runs): the real published NavDP "
            "(Cai et al., arXiv 2505.08712) via its HTTP server -- see next.md's "
            "'Integration project' section. Requires a vendored InternRobotics/NavDP "
            "checkout ($NAVDP_UPSTREAM_ROOT/--navdp-upstream-root) and a gated checkpoint "
            "(Google Form on their repo; defaults to navdp/navdp-cross-modal.ckpt if present); "
            "only supports the single-goal path. navdp-s2dit: legacy, this repo's own "
            "in-house S2DiT/VL3-DP model via NavdpPolicy -- automatically used instead "
            "when --multi-goal/--base-station is passed, since navdp-upstream doesn't "
            "support those yet."
        ),
    )
    parser.add_argument(
        "--navdp-upstream-ckpt",
        default=None,
        help="Path to the upstream NavDP .ckpt (default: navdp/navdp-cross-modal.ckpt if it "
        "exists, with --policy-backend navdp-upstream)",
    )
    parser.add_argument(
        "--navdp-upstream-root",
        default=None,
        help="Path to the vendored InternRobotics/NavDP checkout (default: $NAVDP_UPSTREAM_ROOT)",
    )
    parser.add_argument("--navdp-upstream-port", type=int, default=None)
    parser.add_argument("--navdp-upstream-stop-threshold", type=float, default=0.0)
    parser.add_argument("--navdp-upstream-lookahead", type=int, default=3)
    parser.add_argument("--navdp-upstream-replan-every", type=int, default=1)
    parser.add_argument("--navdp-upstream-max-forward-speed", type=float, default=1.0)
    parser.add_argument("--navdp-upstream-turn-kp", type=float, default=1.4)
    parser.add_argument("--navdp-upstream-request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--out-dir",
        default=f"navdp_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout.mp4 from logged RGB frames",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual PNG frames under out_dir/frames/",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--start-x", type=float, default=0.0, help="Rover spawn x coordinate"
    )
    parser.add_argument(
        "--start-z", type=float, default=8.0, help="Rover spawn z coordinate"
    )
    parser.add_argument(
        "--start-yaw", type=float, default=0.0, help="Rover spawn yaw in degrees"
    )
    parser.add_argument(
        "--randomise-spawn",
        action="store_true",
        help="Ignore --start-x/--start-z/--start-yaw and pick a random (x, z) spawn within the "
        "heightmap bounds, with height sampled from the heightmap",
    )
    parser.add_argument(
        "--rock-field",
        default=None,
        help="Path to a rock_field.json produced by generate_rock_env.py. Loads that fixed, "
        "already-placed rock layout into the scene instead of an empty terrain -- use the same "
        "path across ablation runs to keep the obstacle layout identical.",
    )
    parser.add_argument(
        "--obj-mask-radius",
        type=float,
        default=0.5,
        help="Radius (m) of the goal/obstacle mask mesh placed around each detected object's "
        "backprojected world coords",
    )
    parser.add_argument(
        "--cbf",
        action="store_true",
        help="Enable cone-mode CBF obstacle avoidance (orbit controller + hard-gate backstop)",
    )
    parser.add_argument("--cbf-d-safe", type=float, default=0.75)
    parser.add_argument("--cbf-gamma", type=float, default=0.3)
    parser.add_argument("--cbf-deadzone", type=float, default=0.6)
    parser.add_argument(
        "--cbf-orbit-kr",
        type=float,
        default=0.8,
        help="radial pull-back gain (rad/m) onto the d_safe circle",
    )
    parser.add_argument(
        "--cbf-orbit-hyst",
        type=float,
        default=0.4,
        help="extra clearance (m) required to leave the orbit once committed",
    )
    parser.add_argument(
        "--cbf-pursuit-kp",
        type=float,
        default=1.8,
        help="gain from tangent heading error to yaw-rate",
    )
    parser.add_argument(
        "--cbf-goaround-forward",
        type=float,
        default=0.5,
        help="cruise speed (m/s) while orbiting",
    )
    parser.add_argument(
        "--cbf-escape-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="orbit around a blocking obstacle instead of only braking",
    )
    parser.add_argument(
        "--cbf-hard-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-tick backstop: brake if the executed action would breach the collision radius",
    )
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--obstacle-radius", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument(
        "--zero-lateral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="zero v_lat before CBF avoidance (only applied when --cbf is set)",
    )
    parser.add_argument(
        "--belief-goal-range",
        type=float,
        default=8.0,
        help="fallback range (m) for the goal belief when depth at the mask is invalid",
    )
    parser.add_argument(
        "--belief-odom-noise",
        type=float,
        default=0.0,
        help="Gaussian odom noise per step for belief dead-reckoning (0 = perfect)",
    )
    parser.add_argument(
        "--drive-odom-noise-std",
        type=float,
        default=0.0,
        help="Study 2 (next.md): Gaussian noise std injected into the REAL commanded "
        "action (v_fwd/v_lat full-scale, yaw_rate half-scale) immediately before "
        "integrate_mars/env.step, simulating real actuator/wheel-slip noise on the "
        "rover's actual motion. Distinct from --belief-odom-noise, which is the belief "
        "tracker's own noise-belief and never perturbs real motion (0 = no noise, "
        "today's unchanged behavior). belief_tracker.propagate() still sees the clean, "
        "pre-noise action -- it dead-reckons on what was commanded, not on the real "
        "(noisy) motion, so a real belief/reality mismatch can appear.",
    )
    parser.add_argument(
        "--drive-odom-noise-seed",
        type=int,
        default=None,
        help="Seed for --drive-odom-noise-std's RNG (default: unseeded, non-reproducible)",
    )
    parser.add_argument(
        "--lost-goal-min-px",
        type=int,
        default=10,
        help="goal-mask pixels below this count means the goal is out of view",
    )
    parser.add_argument(
        "--lost-goal-ghost",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="proportional heading assist toward the tracked goal belief when it's off-centre "
        "or out of view. Default: off, except under --base-station where it defaults on "
        "(next.md §6) -- pass --no-lost-goal-ghost to force it off there too.",
    )
    parser.add_argument("--lost-goal-turn-kp", type=float, default=1.4)
    parser.add_argument(
        "--lost-goal-forward",
        type=float,
        default=0.0,
        help="forward speed floor while the goal is fully out of view (pivot recovery)",
    )
    parser.add_argument(
        "--lost-goal-bearing-deg",
        type=float,
        default=30.0,
        help="engage heading assist once |goal bearing| exceeds this angle; 0 disables the angle trigger",
    )
    parser.add_argument(
        "--multi-goal",
        action="store_true",
        help="Discover multiple goals via periodic SAM3 segmentation + per-mask CLIP instead of "
        "resolving one goal from the first frame. Additive: unset, behavior is unchanged.",
    )
    parser.add_argument(
        "--seg-interval-steps",
        type=int,
        default=13,
        help="how often (in steps) to re-run the SAM3 batched-re-window resegment cycle; default "
        "picked from bench_sam3_window.py at --sam3-window-frames=5, dt=0.1 (~1.3s cadence)",
    )
    parser.add_argument(
        "--sam3-window-frames",
        type=int,
        default=5,
        help="ring-buffer size for the SAM3 batched-re-window cycle",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        default=None,
        help="Path to a local SAM3.1 checkpoint (default: download from HF)",
    )
    parser.add_argument(
        "--clip-goal-thresh",
        type=float,
        default=0.24,
        help="min CLIP cosine similarity to accept a mask as goal-worthy; calibrate via calibrate_clip_stock_scene.py",
    )
    parser.add_argument(
        "--clip-reid-thresh",
        type=float,
        default=0.9,
        help="min CLIP cosine similarity to re-identify a mask as an already-tracked goal rather than minting a new one",
    )
    parser.add_argument(
        "--goal-vocab",
        default=None,
        help="comma-separated goal vocabulary terms; bypasses Qwen's describe_goal_vocabulary call when given",
    )
    parser.add_argument(
        "--stop-on-route-finished",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="end the episode once every discovered goal has been visited",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=8,
        help="cap on simultaneously tracked goals in the multi-goal path",
    )
    parser.add_argument(
        "--base-station",
        action="store_true",
        help="After reaching the resolved first-frame goal, dwell then drive back to the "
        "rover's own spawn pose (two fixed, known-ahead-of-time goals -- see next.md). "
        "Mutually exclusive with --multi-goal.",
    )
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=5.0,
        help="hold duration (s) at goal_1 before returning to the base station",
    )
    parser.add_argument(
        "--goal-success-radius",
        type=float,
        default=1.0,
        help="success radius (m) for both legs of the base-station route "
        "(RouteManager.success_radius)",
    )
    parser.add_argument(
        "--base-marker-radius",
        type=float,
        default=None,
        help="radius (m) of the synthetic base-station marker disc (default: --obj-mask-radius)",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=None,
        help="Study 1 (next.md) trigger: BeliefGoalTracker.uncertainty_value() crossing this "
        "logs an uncertainty_trigger event (single-goal mode only, i.e. not --multi-goal/"
        "--base-station). Default: off (no trigger, no behavior change).",
    )
    parser.add_argument(
        "--uncertainty-condition",
        choices=["autonomous", "human"],
        default="autonomous",
        help="Study 1 ablation condition at each --uncertainty-threshold crossing: "
        "'autonomous' (Condition B, default) just logs the trigger and lets the policy "
        "keep driving; 'human' (Condition A) pauses policy stepping and runs the "
        "stop-rotate-ask-drive-retry handoff via a Tk popup. No effect if "
        "--uncertainty-threshold is unset.",
    )
    parser.add_argument(
        "--uncertainty-max-retries",
        type=int,
        default=3,
        help="Condition A: cap on rotate+ask+drive retries per trigger before giving up "
        "unresolved",
    )
    parser.add_argument(
        "--uncertainty-max-units",
        type=float,
        default=None,
        help="Condition A: forward distance (m) to drive per attempt before re-sweeping; "
        "default falls through to vl_direction.config.DEFAULT_MAX_UNITS via "
        "UncertaintySession.submit_heading",
    )
    parser.add_argument(
        "--uncertainty-sweep-degrees-per-step",
        type=float,
        default=45.0,
        help="Condition A: rotation increment (deg) per rotate_sweep step",
    )
    parser.add_argument(
        "--uncertainty-sweep-steps",
        type=int,
        default=8,
        help="Condition A: number of rotate_sweep steps (default 8 * 45deg = full 360 sweep)",
    )
    parser.add_argument(
        "--uncertainty-drive-v-fwd",
        type=float,
        default=0.5,
        help="Condition A: forward speed (m/s) during drive_toward_heading",
    )
    parser.add_argument(
        "--uncertainty-mock-client",
        action="store_true",
        help="Condition A: use MockInternVLClient instead of spawning vl_direction's own "
        "Qwen server -- for exercising the handoff loop without a live qwen_vlm server",
    )
    args = parser.parse_args()

    if args.base_station and args.multi_goal:
        parser.error("--base-station and --multi-goal are mutually exclusive")
    # next.md §6: without a steering signal toward the (initially out-of-frame) base
    # marker, the return leg has nothing to steer on until it happens to enter view --
    # recommended on by default for this mode, overridable via --no-lost-goal-ghost.
    lost_goal_ghost = (
        args.lost_goal_ghost
        if args.lost_goal_ghost is not None
        else bool(args.base_station)
    )

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        navdp_root=args.navdp_root,
        device=args.device,
        sample_steps=args.sample_steps,
        max_steps=args.max_steps,
        dt=args.dt,
        save_video=args.save_video,
        save_frames=args.save_frames,
        video_fps=args.video_fps,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw,
        randomise_spawn=args.randomise_spawn,
        rock_field_path=args.rock_field,
        obj_mask_radius=args.obj_mask_radius,
        cbf=args.cbf,
        cbf_d_safe=args.cbf_d_safe,
        cbf_gamma=args.cbf_gamma,
        cbf_deadzone=args.cbf_deadzone,
        cbf_orbit_kr=args.cbf_orbit_kr,
        cbf_orbit_hyst=args.cbf_orbit_hyst,
        cbf_pursuit_kp=args.cbf_pursuit_kp,
        cbf_goaround_forward=args.cbf_goaround_forward,
        cbf_escape_yaw=args.cbf_escape_yaw,
        cbf_hard_gate=args.cbf_hard_gate,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        obstacle_radius=args.obstacle_radius,
        max_yaw_rate=args.max_yaw_rate,
        zero_lateral=args.zero_lateral,
        belief_goal_range=args.belief_goal_range,
        belief_odom_noise=args.belief_odom_noise,
        drive_odom_noise_std=args.drive_odom_noise_std,
        drive_odom_noise_seed=args.drive_odom_noise_seed,
        lost_goal_min_px=args.lost_goal_min_px,
        lost_goal_ghost=lost_goal_ghost,
        lost_goal_turn_kp=args.lost_goal_turn_kp,
        lost_goal_forward=args.lost_goal_forward,
        lost_goal_bearing_deg=args.lost_goal_bearing_deg,
        multi_goal=args.multi_goal,
        seg_interval_steps=args.seg_interval_steps,
        sam3_window_frames=args.sam3_window_frames,
        sam3_checkpoint=args.sam3_checkpoint,
        clip_goal_thresh=args.clip_goal_thresh,
        clip_reid_thresh=args.clip_reid_thresh,
        goal_vocab=args.goal_vocab,
        stop_on_route_finished=args.stop_on_route_finished,
        max_goals=args.max_goals,
        base_station=args.base_station,
        dwell_seconds=args.dwell_seconds,
        goal_success_radius=args.goal_success_radius,
        base_marker_radius=args.base_marker_radius,
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_condition=args.uncertainty_condition,
        uncertainty_max_retries=args.uncertainty_max_retries,
        uncertainty_max_units=args.uncertainty_max_units,
        uncertainty_sweep_degrees_per_step=args.uncertainty_sweep_degrees_per_step,
        uncertainty_sweep_steps=args.uncertainty_sweep_steps,
        uncertainty_drive_v_fwd=args.uncertainty_drive_v_fwd,
        uncertainty_mock_client=args.uncertainty_mock_client,
        policy_backend=args.policy_backend,
        navdp_upstream_ckpt=args.navdp_upstream_ckpt,
        navdp_upstream_root=args.navdp_upstream_root,
        navdp_upstream_port=args.navdp_upstream_port,
        navdp_upstream_stop_threshold=args.navdp_upstream_stop_threshold,
        navdp_upstream_lookahead=args.navdp_upstream_lookahead,
        navdp_upstream_replan_every=args.navdp_upstream_replan_every,
        navdp_upstream_max_forward_speed=args.navdp_upstream_max_forward_speed,
        navdp_upstream_turn_kp=args.navdp_upstream_turn_kp,
        navdp_upstream_request_timeout=args.navdp_upstream_request_timeout,
    )
