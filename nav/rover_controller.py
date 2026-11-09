"""Threaded, in-process controller wrapping MarsHabitatEnv + the real,
published NavDP model (sam_vla.policy.navdp_upstream_policy.NavdpUpstreamPolicy)
-- not this repo's own in-house S2DiT/VL3-DP model
(sam_vla.policy.navdp_policy.NavdpPolicy) -- into a single control loop a GUI
can drive interactively.

Runs the sim + policy + belief/CBF stepping on one dedicated background
thread (habitat-sim's render context is not meant to be touched from more
than one thread), exposing a small thread-safe command API
(set_manual/random_goal/go_home/request_resolve/request_reset/stop_driving)
plus snapshot() for a GUI to poll at its own refresh rate. No zenoh, no
second conda-env process for the sim itself: MarsHabitatEnv runs in-process
here (the `habitat` conda env already has everything this module imports --
see CLAUDE.md's env table), the real NavDP model and the Qwen VLM used for
salience-pick auto-resolve/nav-command-parsing/uncertainty-halt prompts are
the only subprocesses, spawned automatically by
NavdpUpstreamPolicy/QwenServerManager exactly as sam_vla.run_navdp_rollout
already does. Open-vocabulary target grounding (Ground Target / a mission's
GO_TO/FIND step) runs in-process instead, via
sam_vla.goal_resolution.dino_grounding_resolver (navdp.extensions.
GroundingDINODetector, lazily loaded onto self.dino_device on first use --
no subprocess/server of its own).

Driving modes:
  idle    -- zero action, no policy call.
  manual  -- direct (v_fwd, yaw_rate) from the caller, bypassing the policy
             and CBF entirely (a human takes full responsibility, same
             convention every manual-drive tool in this repo uses).
  point   -- drive to a world-frame (x, z) point (random-ahead / go-home /
             click-to-goal / any caller-supplied point) with no detector in
             the loop: goal_math.body_frame_goal recovers the ground-truth
             body-frame point every tick, fed straight into BeliefGoalTracker
             via observe_body_point (see its docstring -- built for exactly
             this, ground-truth callers with no mask/depth to source a
             sighting from). A click-to-goal point is anchored once by
             request_pixel_goal's depth backprojection (goal_geometry.
             bbox_to_world on a small patch around the clicked pixel), not
             re-derived from anything that could drift -- and kept visually
             marked in the live camera view every frame by reprojecting it
             back to pixel coords (goal_geometry.project_world_to_pixel)
             rather than registering a scene object, since dynamically
             registered objects render zero pixels on this machine (see
             CLAUDE.md's dynamic-object-render-bug note).
  resolve -- with no target_text: one-shot first_frame_resolver.resolve_verbose()
             on the current frame (SAM2 detections + Qwen VLM salience pick,
             no per-preset text targeting -- qwen_client.select_goal has no
             such parameter, see next.md). With a target_text (Ground
             Target / a mission GO_TO/FIND step): dino_grounding_resolver
             instead -- GroundingDINO detects the named open-vocabulary
             object directly (flags, the home-base cuboid; SAM2 still runs
             to seed obstacles). Either way, the resulting goal/obstacle
             masks are tracked every tick exactly as sam_vla.run_navdp_rollout's
             single-goal path does (BeliefGoalTracker.observe(mask, depth),
             holding position rather than driving on a fabricated default
             goal point if the mask has never been sighted -- next.md's
             Integration-project Phase 5 fix, reproduced here since this is
             an independent implementation of the same loop, not an import
             of it).

CBF cone-mode obstacle avoidance (sam_vla.safety.cbf_avoidance) wraps every
mode except manual, same as sam_vla.run_navdp_rollout.

VLM navigation (start_vlm_navigation, nav.vlm_point_navigator.VlmPointNavigator):
an InternVLA-N1-style, detector-free alternative to the click/random/DINO-grounded
point sources above -- Qwen3.5-9B looks at the raw camera frame plus its own
rolling per-instruction history and predicts a goal point directly (no
SAM/DINO), re-queried every vlm_replan_every ticks against an ordered queue
of instructions. It doesn't introduce a new mode: each query's point is fed
straight into set_vlm_point_goal, which is the exact same MODE_POINT entry
point request_forward_point/random_goal/go_home already use, so everything
above about MODE_POINT's ground-truth-point driving applies unchanged.
start_vlm_navigation_from_command is the free-text alternative: one
possibly-compound instruction (e.g. "go straight until you reach the door
and turn left and search for a water bottle") is split into that ordered
instruction queue by a separate, quick Qwen text-decomposition call
(qwen_client.split_instructions_verbose, the existing 3B model/qwen_server
-- not the 9B point-goal model) before being handed to the same
VlmPointNavigator.
"""

from __future__ import annotations

import copy
import math
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.ghost_mask import draw_ghost_mask, project_body_point_to_pixel
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
    project_world_point_with_pixel_radius,
    project_world_to_pixel,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, GoalSpec, Pose
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.terrain import SIZE_X, SIZE_Z
from sam_vla.goal_resolution import dino_grounding_resolver, first_frame_resolver
from sam_vla.perception.semantic_overlay import (
    draw_point_marker,
    overlay_semantic_masks,
)
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance
from sam_vla.vlm import qwen_client
from sam_vla.vlm.qwen_config import QWEN_SERVER_HOST, QWEN_SERVER_PORT
from sam_vla.vlm.qwen_server_manager import QwenServerManager
from sam_vla.vlm.qwen35_server_manager import Qwen35ServerManager

from nav.event_log import EventLogger
from nav.goal_math import body_frame_goal, heading_ahead_point, random_ahead_point
from nav.mission import GoalKind, Mission
from nav.object_belief import ObjectBeliefMemory
from nav.vlm_point_navigator import VlmPointNavigator, parse_simple_command

# Uncertainty-halt: reuses vl_direction's own "uncertainty" prompt/session
# machinery (its documented sole integration point, see vl_direction's module
# docstring) against the real BeliefGoalTracker.uncertainty_value() computed
# below -- not scripts/habitat_tests/kb_teleop_vl.py's synthetic covariance
# proxy, and not imported from that script, to keep nav/ decoupled from it.
# Talks to the same qwen_server this module already starts for goal
# resolution/VLA actions (sam_vla.vlm.qwen_server now answers vl_direction's
# "generate" wire mode too) instead of spawning a second qwen_vlm-env model
# copy on its own port -- see QwenSocketClient construction in _run() below.
from vl_direction import config as vl_dir_config
from vl_direction.client import QwenSocketClient as VlDirectionQwenClient
from vl_direction.uncertainty_session import UncertaintySession

MODE_IDLE = "idle"
MODE_POINT = "point"
MODE_RESOLVE = "resolve"
MODE_MANUAL = "manual"
# Drives NavdpUpstreamPolicy toward a ghost point placed along the bearing to
# self._mission_target_yaw (see TURN_GHOST_* below) rather than spinning in
# place -- only ever entered by a Mission's TURN sub-goal (see
# _start_mission_subgoal), not user-reachable directly.
MODE_TURN = "turn"
# Holding state between a completed SAM2/Qwen resolve and the user accepting
# it -- entered by _do_resolve instead of MODE_RESOLVE, left via
# request_confirm_segmentation (-> MODE_RESOLVE, starts driving),
# request_rerun_segmentation (stays here, re-invokes _do_resolve), or
# request_pick_manually (-> MODE_IDLE, falls through to click-to-goal).
MODE_REVIEW_SEGMENTATION = "review_segmentation"

# Ground-truth distance at which a point goal counts as reached
POINT_GOAL_REACHED_M = 2.0  # revert to 1.5 if near reach required
_TURN_WORDS = ("turn", "face", "rotate", "around", "backward", "degree")
VLM_LEG_REACHED_M = 0.5 # arrival radius for VLM forward-N-metres / home legs

# How far a VLM-driven point-goal creeps forward along the current heading
# (same math request_forward_point uses), used in two places in _env_loop:
#
# 1. A found=False poll -- leaving world_goal untouched freezes the rover
#    (zero action), which is a deadlock when the reason Qwen can't decide is
#    exactly that the current view is ambiguous (e.g. "multiple doorways,
#    none identifiable") -- the view then can never change to resolve that
#    ambiguity, and every subsequent poll repeats found=False against the
#    literal same frame forever (confirmed live).
# 2. Reaching a VLM-set point-goal early -- POINT_GOAL_REACHED_M's 2m
#    threshold is often satisfied well before Qwen's next ~5-7s poll (model
#    inference latency dominates the cadence, not the tick-count-based
#    replan_every), so the rover would otherwise stop and idle-wait between
#    polls -- a visible "drive, stop, drive, stop" pattern instead of
#    InternVLA-style continuous motion toward a standing egocentric goal
#    (confirmed live). Creeping instead keeps it moving in the meantime;
#    only applies while VlmPointNavigator owns the goal -- a manual click
#    or a Mission GO_TO subgoal still stops cleanly on reach, since those
#    aren't waiting on a slow re-poll to supply the next goal.
VLM_CREEP_M = 1.5

# Consecutive point-goal polls whose target stays within this radius of the
# previous poll's target, with the VLM still saying advance=false, before
# RoverController force-advances the instruction anyway (see
# self._vlm_stuck_streak) -- catches instructions the model never explicitly
# confirms even once satisfied, without touching the model itself.
VLM_STUCK_RADIUS_M = 1.0
VLM_STUCK_POLLS = 3

# Ground-truth yaw error at which a MODE_TURN sub-goal counts as reached
TURN_GOAL_REACHED_RAD = math.radians(3.0)

# MODE_TURN no longer spins in place -- it drives NavdpUpstreamPolicy toward a
# body-frame "ghost point" set via policy.set_goal_body(forward, left), same
# point-goal mechanism MODE_POINT/MODE_RESOLVE already use. The point sits
# TURN_GHOST_LOOKAHEAD_M ahead along the remaining-yaw-error bearing, clamped
# to +/- half the camera's HFOV so forward stays positive (always drivable/
# projectable) -- full remaining turn pins it to the frame edge, and it
# slides to dead center as the turn completes. Deliberately not routed
# through any depth/3D backprojection: project_body_point_to_pixel always
# resolves to the single horizontal line at the frame's vertical center, so
# the on-screen position is a deterministic function of the clamped bearing
# alone.
# Draw every path NavDP sampled in the last plan on the GUI camera frame
# (projected onto the floor; colour = NavDP critic value, green high / red low;
# the path actually being driven thick yellow). Independent of
# SHOW_FRAME_MARKERS.
SHOW_NAVDP_PATHS = False  # off on request (2026-11-09): clean camera view


def draw_navdp_paths(rgb, candidates, values, chosen, cam_h, hfov_deg):
    """Project body-frame (forward, left) floor paths into the image of a level
    camera cam_h above the floor: u = cx - fx*left/forward, v = cy + fy*cam_h/forward."""
    import cv2

    img = np.ascontiguousarray(rgb).copy()
    h, w = img.shape[:2]
    fx = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    cx, cy = w / 2.0, h / 2.0
    cam_h = max(float(cam_h), 0.05)

    def pix(path):
        p = np.asarray(path, np.float64)[:, :2]
        p = p[p[:, 0] > 0.15]                      # only points in front of the camera
        if len(p) < 2:
            return None
        u = cx - fx * p[:, 1] / p[:, 0]
        v = cy + fx * cam_h / p[:, 0]
        pts = np.stack([u, v], 1)
        pts = pts[(np.abs(pts[:, 0]) < 4 * w) & (pts[:, 1] < 4 * h)]
        return None if len(pts) < 2 else np.round(pts).astype(np.int32).reshape(-1, 1, 2)

    if candidates is not None and len(candidates):
        vals = None if values is None or len(values) != len(candidates) else np.asarray(values, np.float64)
        lo, hi = (float(vals.min()), float(vals.max())) if vals is not None else (0.0, 1.0)
        order = np.argsort(vals) if vals is not None else range(len(candidates))
        for i in order:                              # worst first, best drawn on top
            q = 0.5 if vals is None or hi - lo < 1e-6 else (vals[i] - lo) / (hi - lo)
            colour = (int(255 * (1 - q)), int(255 * q), 60)   # RGB: red (low) -> green (high)
            pts = pix(candidates[i])
            if pts is not None:
                cv2.polylines(img, [pts], False, colour, 1, cv2.LINE_AA)
    if chosen is not None and len(chosen) >= 2:
        pts = pix(chosen)
        if pts is not None:
            cv2.polylines(img, [pts], False, (255, 220, 0), 3, cv2.LINE_AA)
    return img


# Real rover: max distance of the goal handed to NavDP (see _make_policy).
REAL_GOAL_MAX_M = 2.0

# Goal-point / target-ghost / turn-ghost markers drawn on the GUI camera frame.
# Off: the camera view shows the plain frame (navigation is unaffected).
SHOW_FRAME_MARKERS = False

TURN_GHOST_LOOKAHEAD_M = 3.0
TURN_GHOST_MAX_BEARING_RAD = math.radians(HFOV_DEG / 2.0)
TURN_GHOST_RADIUS_PX = 18.0

# World-space radius of the ghost-mask circle drawn over a DINO-grounded
# open-vocabulary target (see _do_resolve/self._ground_target_world) --
# reprojected to a depth-dependent pixel radius every frame, not a fixed
# on-screen size (see project_world_point_with_pixel_radius).
GHOST_TARGET_RADIUS_M = 0.5

# Default annotations dir for seg_overlay="mesh" -- the dataset
# sam_lora_runs/exp10 (the default seg_backend="lora" checkpoint) was
# trained against. Repo-relative so this holds regardless of caller cwd,
# same convention nav/gui.py's REPO_ROOT uses.
DEFAULT_ANNOTATIONS_DIR = str(
    Path(__file__).resolve().parent.parent / "annotations" / "mesh_tight_bound2"
)


@dataclass
class DisplayState:
    """Snapshot of controller state for the GUI thread to read. Never mutated
    in place by the GUI -- RoverController.snapshot() hands out a shallow
    copy each call."""

    vis_rgb: Optional[np.ndarray] = None
    # Fixed bird's-eye camera frame (see RoverController's enable_topdown_viz)
    # -- human-viewing/recording only, never consumed by policy/CBF/belief
    # logic. None unless enable_topdown_viz=True.
    vis_rgb_topdown: Optional[np.ndarray] = None
    vis_rgb_time: float = 0.0  # time.monotonic() when vis_rgb was produced
    pose: Optional[Pose] = None
    mode: str = MODE_IDLE
    status_text: str = "starting"
    action: Action = field(default_factory=lambda: Action(0.0, 0.0, 0.0))
    distance: Optional[float] = None
    goal_world: Optional[tuple] = None
    # Human-readable name of whatever goal is currently active (GoalSpec.
    # instruction_text) -- covers both ad-hoc goals (Segment/Ground/Random)
    # and the current Mission sub-goal, for nav/gui.py's exp.md-style
    # "CURRENT GOAL" display. "" when no goal is resolved yet.
    goal_label: str = ""
    belief_g: Optional[tuple] = None  # (forward, left), body frame
    trajectory: Optional[np.ndarray] = None  # chosen NavDP waypoints, body frame
    obstacle_point: Optional[tuple] = None  # nearest CBF obstacle, body frame
    cbf_info: dict = field(default_factory=dict)
    step: int = 0
    frame_count: int = 0
    goal_reached: bool = False
    error_text: str = ""
    click_status: str = ""
    # Active Mission's status() line (nav/gui.py's Command panel), "" when no
    # mission is running -- see nav.mission.Mission.
    mission_status: str = ""
    # Structured view of the same Mission, for nav/gui.py's exp.md-style
    # Mission panel goal list (checklist of done/current/upcoming SubGoals).
    # A shallow copy of Mission.goals/.idx -- SubGoal instances are never
    # mutated after Mission.__post_init__ builds them, so sharing references
    # across the controller/GUI thread boundary here is safe. Empty/-1 when
    # no mission is running (ad-hoc Segment/Random/point-click goals aren't a
    # Mission at all -- gui.py falls back to a single synthesized entry).
    mission_goals: List = field(default_factory=list)
    mission_goal_idx: int = -1
    # A Mission whose (directions, goals) VLM split has finished parsing but
    # is awaiting operator confirmation before it starts driving (see
    # submit_nav_command's docstring / request_confirm_mission /
    # request_cancel_mission) -- nav/gui1.py's Confirm-Mission panel reads
    # .instruction/.directions/.goal_texts/.goals off this to show "command
    # interpretation" before any motion happens. None when nothing is
    # awaiting confirmation.
    mission_review: Optional[Mission] = None
    # Uncertainty-halt (vl_direction's "uncertainty" prompt against the real
    # BeliefGoalTracker, see this module's imports): uncertainty_value grows
    # while a MODE_RESOLVE goal mask stays unseen; once it reaches
    # uncertainty_threshold, driving halts and a heading is requested.
    uncertainty_enabled: bool = False
    uncertainty_value: float = 0.0
    uncertainty_threshold: float = 0.0
    uncertainty_halted: bool = False
    uncertainty_request_in_flight: bool = False
    uncertainty_searching: bool = False
    uncertainty_line: str = ""
    # Belief-tracker uncertainty for the persistent "home base" entry in
    # goal_beliefs (see _env_loop's home_base_seeded comment) -- tracked and
    # dead-reckoned for the whole mission, independent of whichever sub-goal
    # is currently active, so this keeps growing even while driving toward a
    # GO_TO/FIND leg rather than only once a RETURN leg starts. 0.0 until a
    # mission has seeded a home-base belief.
    home_base_uncertainty_value: float = 0.0
    home_base_uncertainty_threshold: float = 0.0
    # Human intervention: autonomy is paused and the operator drives by hand.
    # target_uncertainty is the VLM target's belief sigma (m) while a VLM
    # navigation is running (None otherwise).
    intervention_active: bool = False
    intervention_reason: str = ""
    target_uncertainty: Optional[float] = None
    intervention_sigma_threshold: float = 0.0
    # True after an AUTOMATIC hand-over: the human has sole control and
    # "Resume" stays locked until the target uncertainty is back under the limit.
    intervention_locked: bool = False


class RoverController:
    def __init__(
        self,
        *,
        scene_path: str,
        heightmap_path: Optional[str] = None,
        navdp_upstream_ckpt: str,
        navdp_upstream_root: Optional[str] = None,
        navdp_root: Optional[str] = None,
        scene_dataset_config: Optional[str] = None,
        navmesh_path: Optional[str] = None,
        spawn_flags: bool = True,
        gen_home_base: bool = False,
        scenery: Optional[list] = None,
        rock_field_path: Optional[str] = None,
        flag_seed: Optional[int] = None,
        num_flags: int = 6,
        flag_min_spacing: float = 1.5,
        flag_boundary_margin: float = 2.0,
        enable_topdown_viz: bool = False,
        rover_marker_scale: float = 1.0,
        start_x: float = 0.0,
        start_z: float = 8.0,
        start_yaw_deg: float = 0.0,
        dt: float = 0.1,
        hz: float = 10.0,
        cbf_enabled: bool = True,
        cbf_d_safe: float = 0.75,
        cbf_gamma: float = 0.3,
        cbf_deadzone: float = 0.6,
        cbf_orbit_kr: float = 0.8,
        cbf_orbit_hyst: float = 0.4,
        cbf_pursuit_kp: float = 1.8,
        cbf_goaround_forward: float = 0.5,
        cbf_hard_gate: bool = True,
        robot_radius: float = 0.25,
        safety_margin: float = 0.15,
        obstacle_radius: float = 0.25,
        obj_mask_radius: float = 0.5,
        belief_goal_range: float = 8.0,
        lost_goal_min_px: int = 10,
        max_forward_speed: float = 0.6,
        max_yaw_rate: float = 0.6,
        action_smoothing_alpha: float = 0.35,
        navdp_upstream_port: Optional[int] = None,
        navdp_upstream_lookahead: int = 3,
        navdp_upstream_replan_every: int = 1,
        navdp_upstream_server_variant: str = "navdp",
        navdp_upstream_planner_mode: str = "s2diff",
        tube_guided: bool = False,
        terrain_glb: Optional[str] = None,
        intervention_sigma: float = 6.0,
        real_rover_endpoint: Optional[str] = None,
        real_camera_height: float = 0.45,
        real_yaw_rate_scale: float = 1.0,
        real_linear_scale: float = 1.0,
        tube_server: str = "http://127.0.0.1:8890",
        tube_robot_radius: float = 0.45,
        qwen35_port: Optional[int] = None,
        vlm_model_id: Optional[str] = None,
        vlm_prompt_style: Optional[str] = None,
        vlm_replan_every: int = 15,
        vlm_num_history: int = 8,
        world_margin: float = 2.0,
        random_goal_bearing_deg: float = 60.0,
        random_goal_dist_range: tuple = (4.0, 8.0),
        go_direction_distance_m: float = 4.0,
        seg_backend: str = "lora",
        seg_checkpoint: Optional[str] = None,
        seg_overlay: str = "mesh",
        dino_model_id: str = dino_grounding_resolver.DEFAULT_DINO_MODEL_ID,
        dino_device: str = dino_grounding_resolver.DEFAULT_DINO_DEVICE,
        dino_box_threshold: float = dino_grounding_resolver.DEFAULT_BOX_THRESHOLD,
        dino_text_threshold: float = dino_grounding_resolver.DEFAULT_TEXT_THRESHOLD,
        use_obs_detect: bool = False,
        mission_sweep_yaws: int = 8,
        mission_belief_sweep_every: int = 15,
        mission_belief_cov_growth: float = 0.0002,
        mission_belief_cov_growth_rate: float = 0.0,
        annotations_dir: Optional[str] = DEFAULT_ANNOTATIONS_DIR,
        annotation_categories: Optional[Sequence[str]] = None,
        uncertainty_enabled: bool = True,
        uncertainty_cov_threshold: float = vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD,
        uncertainty_cov_growth: float = 0.01,
        uncertainty_cov_growth_rate: float = 0.0,
        point_unc_var_per_m: float = 0.03,
        point_unc_var_per_rad: float = 0.0005,
        uncertainty_search_dist: float = 4.0,
    ):
        self.scene_path = scene_path
        self.heightmap_path = heightmap_path
        self.navdp_upstream_ckpt = navdp_upstream_ckpt
        self.navdp_upstream_root = navdp_upstream_root
        self.navdp_root = navdp_root
        self.scene_dataset_config = scene_dataset_config
        self.navmesh_path = navmesh_path
        self.spawn_flags = bool(spawn_flags)
        self.gen_home_base = bool(gen_home_base)
        self.scenery = list(scenery or [])
        self.rock_field_path = rock_field_path
        self.flag_seed = flag_seed
        self.num_flags = int(num_flags)
        self.flag_min_spacing = float(flag_min_spacing)
        self.flag_boundary_margin = float(flag_boundary_margin)
        self.enable_topdown_viz = bool(enable_topdown_viz)
        self.rover_marker_scale = float(rover_marker_scale)
        self.start_x = float(start_x)
        self.start_z = float(start_z)
        self.start_yaw_deg = float(start_yaw_deg)
        self.dt = float(dt)
        self.hz = float(hz)
        # Cadence (in ticks) for the periodic home-base-uncertainty log line
        # below -- every ~5s of sim time, not every tick, so a long RETURN
        # leg doesn't flood nav/logs/ at self.hz.
        self._home_base_uncertainty_log_every = max(1, round(self.hz * 5.0))
        self.cbf_enabled = bool(cbf_enabled)
        self._cbf_kwargs = dict(
            d_safe=cbf_d_safe,
            gamma=cbf_gamma,
            deadzone=cbf_deadzone,
            orbit_kr=cbf_orbit_kr,
            orbit_hyst=cbf_orbit_hyst,
            pursuit_kp=cbf_pursuit_kp,
            goaround_forward=cbf_goaround_forward,
            hard_gate=cbf_hard_gate,
            robot_radius=robot_radius,
            safety_margin=safety_margin,
            obstacle_radius=obstacle_radius,
            max_yaw_rate=max_yaw_rate,
        )
        self.obj_mask_radius = float(obj_mask_radius)
        self.belief_goal_range = float(belief_goal_range)
        self.lost_goal_min_px = int(lost_goal_min_px)
        self.max_forward_speed = float(max_forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        # Exponential-moving-average factor applied to (v_fwd, v_lat,
        # yaw_rate) each tick before integration -- see _env_loop's use,
        # right before integrate_mars. Lower = smoother/laggier, higher =
        # more responsive/jerkier. 1.0 disables smoothing outright (each
        # tick's raw action passes straight through, exactly like before
        # this existed). Only applied in autonomous driving modes (not
        # MODE_MANUAL, where an operator expects direct/unfiltered control,
        # matching CbfObstacleAvoidance's own mode != MODE_MANUAL gate).
        self.action_smoothing_alpha = float(action_smoothing_alpha)
        self.navdp_upstream_port = navdp_upstream_port
        self.navdp_upstream_lookahead = int(navdp_upstream_lookahead)
        self.navdp_upstream_replan_every = int(navdp_upstream_replan_every)
        self.navdp_upstream_server_variant = navdp_upstream_server_variant
        self.navdp_upstream_planner_mode = navdp_upstream_planner_mode
        # Segmentation-BEV + tube-guided NavDP (navdp_tube, "ours"): replaces the
        # plain upstream NavDP policy when set. See nav/tube_guided_policy.py.
        self.tube_guided = bool(tube_guided)
        self.terrain_glb = terrain_glb
        self.intervention_sigma = float(intervention_sigma)
        # Real rover (nav/real_rover_env.py): zenoh endpoint, None = sim.
        self.real_rover_endpoint = real_rover_endpoint
        self.real_camera_height = float(real_camera_height)
        self.real_yaw_rate_scale = float(real_yaw_rate_scale)
        self.real_linear_scale = float(real_linear_scale)
        if real_rover_endpoint:
            # Every HFOV-dependent computation in this module reads these
            # module globals at call time: switch them to the real D435i.
            from nav.real_rover_env import REAL_HFOV_DEG

            global HFOV_DEG, TURN_GHOST_MAX_BEARING_RAD
            HFOV_DEG = REAL_HFOV_DEG
            TURN_GHOST_MAX_BEARING_RAD = math.radians(HFOV_DEG / 2.0)
        self._pending_intervention: Optional[str] = None
        # True while the human has control: manual commands must not cancel the
        # paused VLM navigation (see _cancel_mission_review_locked).
        self._intervention_paused = False
        self._intervention_locked = False
        self._intervention_reason = ""
        self.tube_server = tube_server
        self.tube_robot_radius = float(tube_robot_radius)
        # Qwen3.5-9B point-goal navigator (nav.vlm_point_navigator) config --
        # see start_vlm_navigation. Independent of navdp_upstream_* above:
        # this only decides *where* NavdpUpstreamPolicy should drive, not how
        # it drives there.
        self.qwen35_port = qwen35_port
        # None/None = original Qwen3.5-9B + original prompt (see
        # Qwen35ServerManager).
        self.vlm_model_id = vlm_model_id
        self.vlm_prompt_style = vlm_prompt_style
        self.vlm_replan_every = int(vlm_replan_every)
        self.vlm_num_history = int(vlm_num_history)
        self.random_goal_bearing_deg = float(random_goal_bearing_deg)
        self.random_goal_dist_range = tuple(random_goal_dist_range)
        self.go_direction_distance_m = float(go_direction_distance_m)
        self.world_limit = max(SIZE_X, SIZE_Z) / 2.0 - float(world_margin)

        # Segmentation backend used by the "resolve" mode's SAM2 detection
        # (see first_frame_resolver._detect): which checkpoint, and whether
        # to feed it a mesh-overlay frame (see _do_resolve /
        # MarsHabitatEnv.get_mesh_overlay_rgb) instead of the plain camera
        # frame -- the overlay is only ever handed to the segmentation
        # model, never shown in the GUI or sent to the goal-selection VLM.
        self.seg_backend = seg_backend
        self.seg_checkpoint = seg_checkpoint
        self.seg_overlay = seg_overlay
        self.annotations_dir = annotations_dir if self.seg_overlay == "mesh" else None
        self.annotation_categories = annotation_categories

        # GroundingDINO grounder config (see _do_resolve's target_text path
        # -> dino_grounding_resolver). Independent of seg_backend/seg_checkpoint
        # above, which stay SAM2/SAM2-LoRA -- DINO only replaces the Qwen
        # open-vocabulary grounder, not the SAM2 obstacle/salience path.
        self.dino_model_id = dino_model_id
        self.dino_device = dino_device
        self.dino_box_threshold = float(dino_box_threshold)
        self.dino_text_threshold = float(dino_text_threshold)
        # Gates the SAM2 obstacle-detection pass inside dino_grounding_resolver's
        # target_text path (see its use_obs_detect param) -- off by default
        # since that checkpoint currently predicts artifact slivers, not real
        # rock geometry (CLAUDE.md's "Annotation masks are thin silhouette
        # slivers" known issue), so running it just burns time for boxes that
        # get dropped or mislead the CBF.
        self.use_obs_detect = bool(use_obs_detect)

        # In-place 360deg scan a GO_TO/FIND mission sub-goal runs (see
        # _start_mission_subgoal) before the existing single-frame
        # _do_resolve -- the search/scan behaviour mission.py's GoalKind.FIND
        # docstring already promises but _start_mission_subgoal previously
        # never actually did (a target outside the current single frame
        # just failed to resolve and got skipped).
        self.mission_sweep_yaws = int(mission_sweep_yaws)

        # Periodic multi-goal DINO belief sweep while a Mission is driving (see
        # _sweep_goal_beliefs) -- checks the current frame against every remaining
        # GO_TO/FIND target text every this-many ticks, seeding/refreshing a
        # persistent per-goal-text BeliefGoalTracker (goal_beliefs in _env_loop) so
        # a goal sighted while driving toward an earlier one isn't forgotten by the
        # time the mission reaches it. <= 0 disables the periodic sweep entirely
        # (a mission step can still resolve via _do_resolve's own front-facing
        # check when it starts, same as before this feature existed).
        self.mission_belief_sweep_every = int(mission_belief_sweep_every)

        # Growth rate for beliefs seeded passively by _sweep_goal_beliefs (a goal
        # other than the one currently being driven to, possibly not seen again for
        # the rest of that leg -- tens of seconds to minutes). Deliberately much
        # slower than uncertainty_cov_growth below: that rate is tuned for the
        # uncertainty-halt's live-tracking use case (a goal briefly losing its mask
        # for a couple seconds), and reusing it here made any sweep-seeded belief
        # decay past uncertainty_cov_threshold within ~1s of the sighting -- almost
        # always before the mission actually reached that goal, defeating the sweep
        # entirely (see nav/rover_controller.py's _start_mission_subgoal "usable
        # prior belief" check). Trackers promoted to actively-driven (see
        # _activate_belief_tracker) switch back to the tight halt-tuned rate.
        self.mission_belief_cov_growth = float(mission_belief_cov_growth)
        self.mission_belief_cov_growth_rate = float(mission_belief_cov_growth_rate)

        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_cov_threshold = float(uncertainty_cov_threshold)
        # Point-goal (VLM) current uncertainty: odometry (aleatoric) noise that
        # accumulates only while the target is out of view and only when the
        # rover actually moves. Variance (m^2) added per metre driven, and per
        # radian turned times the squared range to the target (a heading error
        # displaces a far target more).
        self.point_unc_var_per_m = float(point_unc_var_per_m)
        self.point_unc_var_per_rad = float(point_unc_var_per_rad)
        self._point_unc_value = 0.0
        self.uncertainty_cov_growth = float(uncertainty_cov_growth)
        self.uncertainty_cov_growth_rate = float(uncertainty_cov_growth_rate)
        self.uncertainty_search_dist = float(uncertainty_search_dist)

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._mode = MODE_IDLE
        self._manual_action = Action(0.0, 0.0, 0.0)
        self._world_goal: Optional[tuple] = None
        self._pending_resolve = False
        # None -> first_frame_resolver's SAM2+Qwen-salience path (rocks etc,
        # unchanged). Set -> dino_grounding_resolver's direct-DINO-bbox path
        # for open-vocabulary targets (flags, the home-base cuboid) SAM2
        # wasn't trained on. Persists across a Rerun of the same resolve
        # (see request_rerun_segmentation), cleared on Reset.
        self._resolve_target_text: Optional[str] = None
        # World position of the last DINO-grounded target (see _do_resolve),
        # for the ghost-mask overlay drawn every frame in _run. Only ever
        # touched from _run's own background thread (_do_resolve/_clear_masks
        # write it, _run's render section reads it, all on that one thread) --
        # no lock needed, unlike the self._lock-guarded fields above/below
        # that cross threads.
        self._ground_target_world: Optional[tuple] = None
        self._pending_reset = False
        self._reset_pose: Optional[tuple] = None
        self._pending_clear_all = False
        self._pending_pixel_click: Optional[tuple] = None
        self._pending_confirm_segmentation = False
        self._pending_rerun_segmentation = False
        self._pending_pick_manually = False
        self._pending_confirm_mission = False
        self._pending_cancel_mission = False
        self._pending_uncertainty_heading: Optional[float] = None
        self._pending_uncertainty_retry = False
        # Queued by start_vlm_navigation/stop_vlm_navigation (GUI thread),
        # consumed next tick by _env_loop -- same pending-field handoff
        # every other potentially-slow VLM-backed command in this class uses
        # (c.f. self._pending_nav_command_text), so the (possibly
        # model-loading) Qwen3.5-9B server spawn happens on the controller
        # thread, not the GUI thread.
        self._pending_vlm_instructions: Optional[list] = None
        self._pending_stop_vlm = False
        # Free-text compound-instruction entry point (see
        # start_vlm_navigation_from_command) -- split into
        # self._pending_vlm_instructions next tick via
        # qwen_client.split_instructions_verbose using that tick's live
        # frame, then handled by the exact same block as a direct
        # start_vlm_navigation call.
        self._pending_vlm_command_text: Optional[str] = None
        # Active/queued Mission (nav/gui.py's Command panel) -- only ever
        # touched under self._lock: submit_nav_command (GUI thread) queues,
        # _env_loop's mission-advance block and every manual-override command
        # method (GUI thread, cancelling a stale mission) both read/write
        # self._mission, so unlike self._ground_target_world this can't be
        # left controller-thread-only.
        # Raw text waiting for a frame so its (directions, goals) VLM split
        # can be dispatched (see submit_nav_command/_dispatch_nav_command) --
        # distinct from self._pending_mission below, which holds the
        # *already-parsed* Mission once that background call returns.
        self._pending_nav_command_text: Optional[str] = None
        self._pending_mission: Optional[Mission] = None
        # A Mission whose (directions, goals) split has finished parsing but
        # hasn't been dispatched yet -- see submit_nav_command's docstring
        # and request_confirm_mission/request_cancel_mission. Populated from
        # self._pending_mission once parsing completes, mirrored into
        # self.display.mission_review for the GUI's Confirm-Mission panel,
        # and cleared (without ever becoming self._mission) by any of the
        # manual-override commands below via _cancel_mission_review_locked,
        # same as an active self._mission would be.
        self._mission_awaiting_confirmation: Optional[Mission] = None
        self._mission: Optional[Mission] = None
        # Absolute target world yaw (radians, Pose.yaw's convention) for an
        # active MODE_TURN sub-goal -- controller-thread-only (set and read
        # inside _env_loop/_start_mission_subgoal alone), same pattern as
        # self._ground_target_world.
        self._mission_target_yaw: Optional[float] = None

        # Active Qwen3.5-9B point-goal navigator (nav.vlm_point_navigator),
        # None when VLM navigation isn't running. The reference itself is
        # only ever touched under self._lock (same pattern as self._mission:
        # _env_loop's controller thread constructs/replaces/clears it,
        # _cancel_mission_review_locked below clears it on any manual
        # override from the GUI thread) -- but its internal state
        # (.idx/.history via maybe_step/advance) is only ever called from
        # _env_loop, so no lock is needed for those calls themselves.
        # self._qwen35_manager is the lazily-started subprocess manager for
        # its model server (see qwen35_server_manager), controller-thread-only
        # (only ever touched inside _env_loop), kept separate from
        # self._vlm_navigator's lifetime -- once loaded, the model stays
        # resident for the rest of this process rather than reloading on
        # every start_vlm_navigation call.
        self._vlm_navigator: Optional[VlmPointNavigator] = None
        # Background Qwen poll (see _poll_vlm_async): one in-flight
        # (future, navigator) at a time, so a 1-6 s Qwen call never stalls
        # the control loop / video / NavDP steering.
        from concurrent.futures import ThreadPoolExecutor

        self._vlm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-poll")
        self._vlm_job = None
        # Newest raw camera frame source for the GUI (real rover only).
        self._live_env = None
        self._raw_frame = None
        # planner-health log state (_log_planner_health)
        self._planner_gap_t0 = None
        self._planner_gap_reason = None
        self._planner_gap_last_log = 0.0
        self._qwen35_manager: Optional[Qwen35ServerManager] = None
        # Set (to "left"/"right"/"back") while a VLM-issued turn instruction
        # is physically executing via set_vlm_turn/MODE_TURN -- _env_loop
        # skips further maybe_step polling while this is set and instead
        # waits for MODE_TURN's own ground-truth goal_reached edge to
        # advance the instruction (see set_vlm_turn's docstring; this is
        # what fixes the earlier bug where advancing depended on the model's
        # own "has the scene changed" judgment and could stall indefinitely).
        self._vlm_turn_direction: Optional[str] = None
        # Geometry-based safety net for the "VLM keeps saying advance=false
        # forever" deadlock (confirmed live on the R2R-CE benchmark: stative
        # instructions like "wait at the door"/"stop in front of X" often
        # never get an explicit advance=true even once the rover is
        # physically there, see VLM_STUCK_POLLS below) -- tracks how many
        # consecutive point-goal polls landed within VLM_STUCK_RADIUS_M of
        # the previous poll's target, independent of the model's own
        # judgment. Reset whenever the instruction actually advances (natural
        # or forced) or a fresh VLM nav session starts.
        self._vlm_stuck_streak = 0
        self._vlm_last_target: Optional[tuple] = None

        # Only ever touched inside self._lock (read/written from both the
        # controller thread and the uncertainty-request background thread).
        self._uncertainty_halted = False
        self._uncertainty_request_in_flight = False
        self._uncertainty_search_goal: Optional[tuple] = None

        self.display = DisplayState()
        self.display.uncertainty_enabled = self.uncertainty_enabled
        self.display.uncertainty_threshold = self.uncertainty_cov_threshold
        self.display.home_base_uncertainty_threshold = self.uncertainty_cov_threshold
        self._rng = np.random.default_rng()
        # Goal-detected/goal-reached event log (nav/logs/, gitignored) --
        # see EventLogger's docstring. Created once per RoverController (one
        # log file per GUI process lifetime), not per-episode/reset.
        self._event_log = EventLogger()

    def _cancel_mission_review_locked(self) -> None:
        """Discard a Mission awaiting confirmation (see
        request_confirm_mission/request_cancel_mission) and stop any active
        VLM point-goal navigation (see start_vlm_navigation) -- called
        alongside every `self._mission = None` below, so a manual override
        (manual drive, Random Goal, Go Home, a click/resolve goal, Reset,
        Clear All) also drops a stale interpreted mission sitting in the
        GUI's Confirm-Mission panel, and stops a Qwen3.5-9B navigation run
        from fighting the operator's explicit command, instead of leaving
        either running out of context. Caller must already hold self._lock."""
        self._mission_awaiting_confirmation = None
        self.display.mission_review = None
        if not self._intervention_paused:
            self._vlm_navigator = None
        self._vlm_stuck_streak = 0
        self._vlm_last_target = None
        self._vlm_turn_direction = None

    # ------------------------------------------------------------------ #
    # thread-safe command API -- called from the GUI thread
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rover-controller"
        )
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def vlm_navigation_active(self) -> bool:
        """True while a VlmPointNavigator queue still has instructions left
        (start_vlm_navigation/start_vlm_navigation_from_command until the
        last instruction's advance() -- see _env_loop). Needed alongside
        display.goal_reached for an external, headless caller to detect
        "truly done": mode briefly reads MODE_IDLE with goal_reached=True
        right after a TURN instruction physically completes, one tick before
        the next queued instruction's first Qwen poll sets a new point goal
        -- during that gap this is still True, disambiguating a genuine
        finish from that transient handoff (confirmed live: an external
        poller checking only mode/goal_reached can catch that gap and
        declare an episode "finished" seconds into a multi-instruction
        chain)."""
        with self._lock:
            return self._vlm_navigator is not None

    def set_manual(self, v_fwd: float, yaw_rate: float) -> None:
        with self._lock:
            self._mode = MODE_MANUAL
            self._manual_action = Action(
                v_fwd=float(v_fwd), v_lat=0.0, yaw_rate=float(yaw_rate)
            )
            self._mission = None
            self._cancel_mission_review_locked()

    def request_intervention(self, reason: str = "operator took over") -> None:
        """Pause autonomy and let the operator drive by hand (WASD/arrows)."""
        with self._lock:
            self._pending_intervention = "start"
            self._intervention_reason = reason
            self._intervention_paused = True
            self._intervention_locked = False

    def _can_resume(self) -> bool:
        """After an automatic hand-over the human keeps sole control until the
        target uncertainty is back under the limit."""
        with self._lock:
            nav = self._vlm_navigator
            locked = self._intervention_locked
        if not locked or nav is None:
            return True
        u = getattr(nav, "last_uncertainty", None)
        if u is not None and u >= self.intervention_sigma:
            return False
        # A current-uncertainty hand-over holds until the target is seen again
        # (a sighting resets it below the threshold).
        return not (self.uncertainty_enabled
                    and self._point_unc_value >= self.uncertainty_cov_threshold)

    def resume_autonomy(self) -> None:
        """Operator hands control back; a paused VLM navigation carries on."""
        with self._lock:
            self._pending_intervention = "resume"

    def random_goal(self) -> None:
        with self._lock:
            pose = self.display.pose
        if pose is None:
            return
        goal = random_ahead_point(
            pose,
            self._rng,
            bearing_range_deg=self.random_goal_bearing_deg,
            dist_range=self.random_goal_dist_range,
            world_limit=self.world_limit,
        )
        with self._lock:
            self._world_goal = goal
            self._mode = MODE_POINT
            self.display.goal_reached = False
            self._mission = None
            self._cancel_mission_review_locked()

    def go_home(self) -> None:
        with self._lock:
            self._world_goal = (self.start_x, self.start_z)
            self._mode = MODE_POINT
            self.display.goal_reached = False
            self._mission = None
            self._cancel_mission_review_locked()

    def request_pixel_goal(self, x_norm: float, y_norm: float) -> None:
        """Queue a click-to-goal request: (x_norm, y_norm) are normalized
        [0, 1] image coords in the *displayed* camera frame (origin
        top-left). Resolved against the live depth frame on the controller
        thread next tick -- see _handle_pixel_click."""
        with self._lock:
            self._pending_pixel_click = (float(x_norm), float(y_norm))
            self._mission = None
            self._cancel_mission_review_locked()

    def request_forward_point(self, distance_m: float) -> None:
        """Set a world-frame point `distance_m` straight ahead of the
        rover's current pose/heading and switch to MODE_POINT driving toward
        it -- same math GoalKind.ADVANCE uses in _start_mission_subgoal, just
        callable directly by a script that isn't driving through a Mission
        (e.g. nav/run_find_and_turn_demo.py's "go straight" leg)."""
        with self._lock:
            pose = self.display.pose
        if pose is None:
            return
        tx = pose.x - float(distance_m) * math.sin(pose.yaw)
        tz = pose.z - float(distance_m) * math.cos(pose.yaw)
        with self._lock:
            self._world_goal = (tx, tz)
            self._mode = MODE_POINT
            self.display.goal_reached = False
            self._mission = None
            self._cancel_mission_review_locked()

    def set_vlm_turn(self, direction: str, degrees: float = 90.0) -> None:
        """Same in-place-turn driving as request_turn (MODE_TURN off
        self._mission_target_yaw), but callable from inside _env_loop by the
        active VlmPointNavigator (see set_vlm_point_goal's equivalent
        docstring note) -- deliberately does not touch self._mission or call
        _cancel_mission_review_locked, which would cancel the very VLM
        navigation run that just called this. Unlike request_turn's fixed
        90/180 headings (still used by nav.mission's closed-vocabulary
        turn_left/turn_right/turn_back tokens), degrees is an arbitrary
        magnitude in (0, 180] -- see qwen35_prompts.build_point_goal_history_
        prompt, which lets Qwen specify e.g. "turn left 45 degrees" instead
        of always snapping to a quarter or half turn."""
        heading_deg = float(degrees) if direction == "right" else -float(degrees)
        with self._lock:
            pose = self.display.pose
        if pose is None:
            return
        with self._lock:
            self._mission_target_yaw = pose.yaw - math.radians(heading_deg)
            self._mode = MODE_TURN
            self.display.goal_reached = False

    def _vlm_busy(self, nav) -> bool:
        job = self._vlm_job
        return job is not None and job[1] is nav and not job[0].done()

    def _wait_vlm_idle(self, nav, timeout: float = 15.0) -> None:
        """Block (rarely: resume) until no background poll is touching nav."""
        job = self._vlm_job
        if job is not None and job[1] is nav:
            try:
                job[0].result(timeout=timeout)
            except Exception:
                pass

    def _poll_vlm_async(self, nav, obs, step):
        """Non-blocking replacement for nav.maybe_step(obs, step): on a cadence
        tick, start the Qwen poll in the background; return its result on the
        first tick after it finishes (None otherwise). Its targets are in world
        coordinates, so they stay valid while the rover kept moving. Results
        for a navigator that has since been replaced are dropped."""
        job = self._vlm_job
        if job is not None:
            fut, jnav = job
            if not fut.done():
                return None
            self._vlm_job = None
            if jnav is not nav:
                return None
            try:
                return fut.result()
            except Exception:
                traceback.print_exc()
                return None
        if nav.finished or step % nav.replan_every != 0:
            return None
        # Real rover: find the target on a colour+depth+pose set from the SAME
        # moment (RealRoverEnv's obs.synced); skip the poll until one exists
        # rather than pairing pixels with depth from another heading.
        p_obs = getattr(obs, "synced", None)
        if p_obs is None:
            if self.real_rover_endpoint:
                return None
            p_obs = obs
        self._vlm_job = (self._vlm_pool.submit(nav.maybe_step, p_obs, step), nav)
        return None

    def _log_planner_health(self, info, forward, left, obs) -> None:
        """Log why NavDP (tube-guided) has no path while there IS a goal/belief
        to drive to: once when the reason changes, every 2 s while it lasts,
        and once when a path is back (with how long it was missing)."""
        diag = (info or {}).get("tube_diag") if isinstance(info, dict) else None
        if not diag:
            return
        now = time.monotonic()
        has_path = diag.get("path_pts_ahead", 0) > 1 and diag.get("state") == "driving"
        if has_path:
            if self._planner_gap_t0 is not None:
                self._event_log.log(
                    "navdp_path_ok", missing_for_s=round(now - self._planner_gap_t0, 1),
                    plan_ms=diag.get("plan_ms"), plans=diag.get("plans"),
                )
            self._planner_gap_t0, self._planner_gap_reason = None, None
            return
        if diag.get("error"):
            reason = "planner error: " + diag["error"][:120]
        elif diag.get("plans", 0) == 0:
            reason = "no plan yet (first plan pending)"
        elif str(diag.get("state", "")).startswith("no plan"):
            reason = (f"plan too old: {diag.get('plan_age_s')} s > {diag.get('plan_timeout_s')} s timeout "
                      f"(last plan took {diag.get('plan_ms')} ms, planning now={diag.get('planning_now')})")
        elif diag.get("navdp_path_m", 0.0) < 0.05:
            reason = "NavDP returned no path (stop / all-zero trajectory)"
        elif diag.get("path_pts_ahead", 0) <= 1:
            reason = "rover already at the end of the last path (waiting for the next plan)"
        else:
            reason = f"tube state {diag.get('state')!r}"
        if (reason != self._planner_gap_reason
                or now - self._planner_gap_last_log >= 2.0):
            if self._planner_gap_t0 is None:
                self._planner_gap_t0 = now
            self._planner_gap_reason = reason
            self._planner_gap_last_log = now
            self._event_log.log(
                "navdp_no_path", reason=reason,
                goal_body=(round(float(forward), 2), round(float(left), 2)),
                missing_for_s=round(now - self._planner_gap_t0, 1),
                pose_xz=(round(obs.pose.x, 2), round(obs.pose.z, 2)),
                **{k: diag.get(k) for k in ("plans", "plan_age_s", "plan_ms", "planning_now",
                                            "choice", "n_valid", "navdp_path_m", "occ_cells", "goal_dist")},
            )

    def latest_rgb_pose(self):
        """(raw rgb, pose) of the newest controller tick, or None."""
        return self._raw_frame

    @staticmethod
    def hfov_deg() -> float:
        return HFOV_DEG  # module global: 55.5 on the real rover, 90 in the sim

    def latest_camera_rgb(self):
        """Newest raw camera frame (real rover), for the GUI's live view."""
        env = self._live_env
        if env is None or not hasattr(env, "look_down_rgb"):
            return None
        try:
            return env.look_down_rgb()
        except Exception:
            return None

    def _start_vlm_motion(self, vlm_result, pose, vlm_navigator) -> None:
        """Dispatch a VLM 'forward' (N metres straight ahead), 'home' (back to
        the spawn point) or 'stop' action from inside _env_loop. forward/home
        reuse the turn bookkeeping: _vlm_turn_direction is set to a label so
        _env_loop stops polling Qwen and advances the instruction on the
        ground-truth goal_reached edge (same as a turn). 'stop' halts and
        advances immediately."""
        action = vlm_result.action
        if action == "stop":
            with self._lock:
                self._mode = MODE_IDLE
                self._world_goal = None
                self.display.status_text = "VLM nav: stopped"
            finished = vlm_navigator.advance()
            if finished:
                with self._lock:
                    self._vlm_navigator = None
        else:
            if action == "forward":
                d = float(vlm_result.distance_m)
                goal = (
                    pose.x - d * math.sin(pose.yaw),
                    pose.z - d * math.cos(pose.yaw),
                )
                label = f"forward {d:g} m"
            else:
                goal = (self.start_x, self.start_z)
                label = "home"
            self.set_vlm_point_goal(*goal)
            with self._lock:
                self._vlm_turn_direction = action
                self.display.status_text = (
                    f"VLM nav [{vlm_navigator.idx + 1}/"
                    f"{len(vlm_navigator.instructions)}]: "
                    f"{vlm_result.instruction_text} -- {label}"
                )
        self._event_log.log(
            f"vlm_{action}_started",
            instruction=vlm_result.instruction_text,
            distance_m=vlm_result.distance_m,
            reasoning=vlm_result.reasoning,
        )

    def set_vlm_point_goal(self, x: float, z: float) -> None:
        """Set a world-frame (x, z) point and switch to MODE_POINT driving
        toward it, same shape as request_forward_point/random_goal/go_home
        -- but called from inside _env_loop itself (by the active
        VlmPointNavigator, once per replan_every cadence tick), not the GUI
        thread, and NOT a manual override: unlike every command above, this
        deliberately does not touch self._mission or call
        _cancel_mission_review_locked (that would immediately cancel the
        very VLM navigation run that just called this)."""
        with self._lock:
            self._world_goal = (float(x), float(z))
            self._mode = MODE_POINT
            self.display.goal_reached = False

    def request_turn(self, direction: str) -> None:
        """In-place turn to a heading relative to the rover's current yaw --
        same heading_deg mapping GoalKind.TURN uses in _start_mission_subgoal
        (and the same MODE_TURN driving, off self._mission_target_yaw), just
        callable directly without a Mission."""
        heading_deg = {"right": 90.0, "left": -90.0, "back": 180.0}.get(
            direction, 180.0
        )
        with self._lock:
            pose = self.display.pose
        if pose is None:
            return
        with self._lock:
            self._mission_target_yaw = pose.yaw - math.radians(heading_deg)
            self._mode = MODE_TURN
            self.display.goal_reached = False
            self._mission = None
            self._cancel_mission_review_locked()

    def request_resolve(self, target_text: Optional[str] = None) -> None:
        """target_text=None -> the original SAM2+Qwen-salience auto-resolve
        (unchanged). A non-empty target_text (e.g. "flag", "blue cuboid")
        switches to direct GroundingDINO grounding for that object instead --
        see dino_grounding_resolver, wired in via _do_resolve. A manual
        override, so it cancels any active Mission (see submit_nav_command)
        the same way every other manual command below does."""
        with self._lock:
            self._pending_resolve = True
            self._resolve_target_text = target_text or None
            self._mission = None
            self._cancel_mission_review_locked()

    def request_confirm_segmentation(self) -> None:
        """Accept the goal/obstacle masks from the most recent resolve/rerun
        -- MODE_REVIEW_SEGMENTATION -> MODE_RESOLVE, driving starts next
        tick. No-op if not currently in review (stale click after a reset)."""
        with self._lock:
            self._pending_confirm_segmentation = True

    def request_rerun_segmentation(self) -> None:
        """Reject the current segmentation and re-invoke the resolver.
        Stays in MODE_REVIEW_SEGMENTATION for another round of review."""
        with self._lock:
            self._pending_rerun_segmentation = True

    def request_pick_manually(self) -> None:
        """Reject the auto-resolved goal entirely and fall through to the
        existing click-to-goal flow: untags the resolved masks and returns
        to MODE_IDLE so a subsequent request_pixel_goal drives normally."""
        with self._lock:
            self._pending_pick_manually = True

    def request_reset(self, pose: Optional[tuple] = None) -> None:
        """Teleport back to spawn, or to `pose` = (x, z, yaw_rad) when given
        (the user study puts every trial at a recorded start pose)."""
        with self._lock:
            self._pending_reset = True
            self._reset_pose = None if pose is None else tuple(float(v) for v in pose)
            self._mission = None
            self._cancel_mission_review_locked()

    def clear_all_goals(self) -> None:
        """Hard reset of every goal/mission/mask/uncertainty-halt state back
        to a blank idle slate -- everything request_reset does *except*
        teleporting the rover back to spawn (see _env_loop's
        `do_reset or do_clear_all` block, which shares the rest of that
        logic). Distinct from stop_driving(), which only clears the active
        world_goal/mission and leaves resolved masks and uncertainty state
        alone."""
        with self._lock:
            self._pending_clear_all = True
            self._mission = None
            self._cancel_mission_review_locked()

    def request_confirm_mission(self) -> None:
        """Accept the Mission currently awaiting confirmation (see
        submit_nav_command's docstring) -- dispatched next tick, replacing
        whatever mission/goal was previously active. No-op if nothing is
        currently awaiting confirmation (stale click after a Reset/Clear
        All or a manual override already discarded it)."""
        with self._lock:
            self._pending_confirm_mission = True

    def request_cancel_mission(self) -> None:
        """Reject the Mission currently awaiting confirmation without
        starting it -- the rover stays exactly as it was (unlike Reset).
        The operator can then edit and resubmit the Taskwork command."""
        with self._lock:
            self._pending_cancel_mission = True

    def submit_uncertainty_heading(self, angle_deg: float) -> None:
        """Answer an active uncertainty halt with a rover-front-relative
        heading (degrees, same convention as heading_ahead_point/
        nav/gui1.py's numpad panel: 0=front, positive=left/counterclockwise).
        No-op (consumed and dropped) if not currently halted -- guards a
        stale click racing a halt that's already resolved by a fresh
        sighting."""
        with self._lock:
            self._pending_uncertainty_heading = float(angle_deg)

    def retry_uncertainty_request(self) -> None:
        """Re-request the VLM sweep description for an active uncertainty
        halt. No-op if not currently halted or a request is already in
        flight."""
        with self._lock:
            self._pending_uncertainty_retry = True

    def submit_nav_command(self, text: str) -> None:
        """Queue a free-text nav command (nav/gui1.py's Taskwork panel) for
        background splitting into (directions, goals) via the Qwen VLM
        (qwen_client.parse_nav_command, see
        qwen_prompts.build_parse_nav_command_prompt) -- consumed next tick by
        _env_loop, which dispatches the VLM call on a background thread
        (_dispatch_nav_command) using that tick's live frame. The result
        becomes a new Mission (nav.mission.Mission, nav.mission.parse_parts:
        every direction first as an in-place TURN, then every goal as
        GO_TO/RETURN), which _env_loop stages into self._mission_awaiting_
        confirmation (and mirrors into self.display.mission_review) rather
        than dispatching it immediately -- the GUI's Confirm-Mission panel
        shows the interpreted step list and the operator must
        request_confirm_mission() before _start_mission_subgoal actually
        starts driving (or request_cancel_mission() to discard it). Once
        confirmed: GO_TO/FIND ground the named target via
        dino_grounding_resolver and auto-confirm per step (no per-step
        review pause -- the human already reviewed the whole instruction
        up front), RETURN drives to the spawn point, TURN rotates in place.
        Advances on each step's ground-truth goal_reached signal, same as a
        manually driven point/resolve goal."""
        with self._lock:
            self._pending_nav_command_text = text

    def stop_driving(self) -> None:
        with self._lock:
            self._mode = MODE_IDLE
            self._world_goal = None
            self._manual_action = Action(0.0, 0.0, 0.0)
            self._mission = None
            self._cancel_mission_review_locked()

    def start_vlm_navigation(self, instructions: list) -> None:
        """Queue an ordered list of navigation instructions for a Qwen3.5-9B
        point-goal navigator (nav.vlm_point_navigator.VlmPointNavigator) to
        drive through one at a time -- see this module's docstring and that
        class for the InternVLA-N1-style design (no detector, periodic
        re-query, per-instruction history). A manual override like every
        other command above, so it cancels any active Mission the same way.
        Consumed next tick by _env_loop, which does the (potentially slow,
        first-call model-loading) Qwen3.5-9B server spawn on the controller
        thread rather than blocking the caller here."""
        with self._lock:
            self._mission = None
            self._cancel_mission_review_locked()
            self._pending_vlm_instructions = list(instructions)

    def stop_vlm_navigation(self) -> None:
        with self._lock:
            self._pending_stop_vlm = True

    def start_vlm_navigation_from_command(self, command_text: str) -> None:
        """Free-text compound-instruction alternative to start_vlm_navigation
        -- e.g. "go straight until you reach the door and turn left and
        search for a water bottle" -- split into an ordered list of
        self-contained atomic instructions next tick (Qwen's
        qwen_prompts.build_split_instruction_prompt, using that tick's live
        frame for context), then handed to start_vlm_navigation's same
        VlmPointNavigator queue. A manual override like start_vlm_navigation
        itself."""
        with self._lock:
            self._mission = None
            self._cancel_mission_review_locked()
            self._pending_vlm_command_text = command_text

    def snapshot(self) -> DisplayState:
        with self._lock:
            return copy.copy(self.display)

    def _run(self) -> None:
        # CbfObstacleAvoidance needs `navdp.extensions` (this repo's own navdp/
        # package's generic CBF/obstacle math -- unrelated to which driving
        # policy is active) importable; this is the same sys.path setup
        # sam_vla.run_navdp_rollout does before constructing it, reused as
        # plain infrastructure, not as "the custom model" itself.
        from sam_vla.policy.navdp_policy import _add_navdp_to_path, _resolve_navdp_root

        _add_navdp_to_path(_resolve_navdp_root(self.navdp_root))

        qwen_manager = QwenServerManager()
        services = [qwen_manager]
        # Uncertainty-halt reuses this same qwen_server (see this module's
        # import comment) rather than spawning vl_direction's own -- no
        # second service to start here, just a client pointed at the
        # sam_vla.vlm qwen_config host/port.
        uncertainty_client = None
        if self.uncertainty_enabled:
            uncertainty_client = VlDirectionQwenClient(
                host=QWEN_SERVER_HOST, port=QWEN_SERVER_PORT
            )
        try:
            if self.real_rover_endpoint:
                from nav.real_rover_env import RealRoverEnv

                with RealRoverEnv(
                    self.real_rover_endpoint,
                    services=services,
                    dt=self.dt,
                    max_linear=self.max_forward_speed,
                    max_angular=self.max_yaw_rate,
                    camera_height=self.real_camera_height,
                    yaw_rate_scale=self.real_yaw_rate_scale,
                    linear_scale=self.real_linear_scale,
                ) as env:
                    self._env_loop(env, uncertainty_client)
                return
            with MarsHabitatEnv(
                self.scene_path,
                self.heightmap_path,
                services=services,
                scene_dataset_config=self.scene_dataset_config,
                navmesh_path=self.navmesh_path,
                start_x=self.start_x,
                start_z=self.start_z,
                start_yaw=math.radians(self.start_yaw_deg),
                with_semantic=True,
                spawn_flags=self.spawn_flags,
                gen_home_base=self.gen_home_base,
                scenery=self.scenery,
                terrain_glb=self.terrain_glb,
                rock_field_path=self.rock_field_path,
                flag_seed=self.flag_seed,
                num_flags=self.num_flags,
                flag_min_spacing=self.flag_min_spacing,
                flag_boundary_margin=self.flag_boundary_margin,
                enable_topdown_viz=self.enable_topdown_viz,
                rover_marker_scale=self.rover_marker_scale,
                annotations_dir=self.annotations_dir,
                annotation_categories=self.annotation_categories,
            ) as env:
                self._env_loop(env, uncertainty_client)
        except Exception as exc:  # pragma: no cover - surfaced to the GUI, not raised
            traceback.print_exc()
            with self._lock:
                self.display.status_text = f"FATAL: {exc}"
                self.display.error_text = str(exc)
        finally:
            # self._qwen35_manager isn't in `services` above (it's started
            # lazily, only if VLM navigation is ever actually requested --
            # see _env_loop's vlm_instructions handling), so MarsHabitatEnv's
            # own service teardown never reaches it. Stop it here instead,
            # same lifetime scope services get (whole controller run).
            if self._qwen35_manager is not None:
                self._qwen35_manager.stop()

    def _make_policy(self, env, obs0):
        """The driving policy: plain upstream NavDP (default), or NavDP guided
        by the segmentation BEV map when tube_guided is set."""
        policy = self._make_policy_inner(env, obs0)
        if self.real_rover_endpoint:
            # Real rover: NavDP only predicts a few metres reliably; a far goal
            # (live: 5-28 m to a door) degenerates its path into hard left/right
            # turns whatever the true direction (validated on this rover,
            # real_rover_qwen35_navdp.NAVDP_GOAL_MAX_DIST_M). Hand it a carrot
            # at most REAL_GOAL_MAX_M ahead along the true direction instead;
            # each replan moves the carrot as the rover closes in.
            raw_set = policy.set_goal_body

            def _carrot(forward, left, _raw=raw_set):
                n = math.hypot(forward, left)
                if n > REAL_GOAL_MAX_M:
                    forward, left = forward * REAL_GOAL_MAX_M / n, left * REAL_GOAL_MAX_M / n
                _raw(forward, left)

            policy.set_goal_body = _carrot
        return policy

    def _make_policy_inner(self, env, obs0):
        if self.tube_guided:
            from nav.tube_guided_policy import TubeGuidedPolicy

            return TubeGuidedPolicy(
                terrain=env._terrain,
                size_x=SIZE_X,
                size_z=SIZE_Z,
                start_xz=(self.start_x, self.start_z),
                image_hw=obs0.rgb.shape[:2],
                hfov_deg=HFOV_DEG,
                max_forward_speed=self.max_forward_speed,
                max_yaw_rate=self.max_yaw_rate,
                robot_radius=self.tube_robot_radius,
                server=self.tube_server,
                navdp_root=self.navdp_upstream_root,
                navdp_ckpt=self.navdp_upstream_ckpt,
                # Real rover: the segmentation model was trained on the sim's
                # Mars terrain and mislabels a real floor, so no class marks
                # ground as occupied -- obstacles come from depth only (anything
                # standing 0.15-1.0 m above the floor, SegBEVMap's object test).
                seg_classes=(
                    dict(occupied_classes="", rock_classes="",
                         free_classes="soil,bedrock,sand,bigrock")
                    if self.real_rover_endpoint else None
                ),
                # Real rover: plans take 0.3-0.5 s and start every ~0.6 s, so a
                # 1.0 s limit kept expiring and stopping the rover (live log
                # 20261109-154300: 18x "plan too old", 1.0-1.1 s).
                plan_timeout=2.5 if self.real_rover_endpoint else 1.0,
            )
        return NavdpUpstreamPolicy(
            checkpoint_path=self.navdp_upstream_ckpt,
            navdp_upstream_root=self.navdp_upstream_root,
            port=self.navdp_upstream_port,
            image_hw=obs0.rgb.shape[:2],
            hfov_deg=HFOV_DEG,
            lookahead=self.navdp_upstream_lookahead,
            replan_every=self.navdp_upstream_replan_every,
            max_forward_speed=self.max_forward_speed,
            max_yaw_rate=self.max_yaw_rate,
            server_variant=self.navdp_upstream_server_variant,
            planner_mode=self.navdp_upstream_planner_mode,
        )

    def _env_loop(self, env: MarsHabitatEnv, uncertainty_client=None) -> None:
        mask_dir = tempfile.mkdtemp(prefix="mars_nav_masks_")
        self._live_env = env
        obs0 = env.get_observation(frame_idx=0)

        def capture_vlm_look_down(pitch_deg: float = -40.0):
            # Same real-world look-down sub-flow run_internvla_n1_compare.py
            # uses for InternVLA-N1 (pitch the RGB sensor's SceneNode for one
            # extra render, then restore it) -- wired into our own
            # VlmPointNavigator too now, see its look_down_frame_fn.
            if hasattr(env, "look_down_rgb"):  # real rover: camera can't pitch
                return env.look_down_rgb()
            import magnum as mn

            rgb_sensor = env._agent._sensors["rgb"]
            forward_rotation = rgb_sensor.node.rotation
            rgb_sensor.node.rotation = mn.Quaternion.rotation(
                mn.Deg(pitch_deg), mn.Vector3.x_axis()
            )
            raw = env._sim.get_sensor_observations()
            rgb_sensor.node.rotation = forward_rotation
            rgb = np.asarray(raw["rgb"])[:, :, :3]
            return rgb
        # EMA state for action smoothing (see action_smoothing_alpha) --
        # None whenever there's no smoothed action yet to blend against
        # (start of the loop, or just reset on a MODE_MANUAL/MODE_IDLE tick
        # so an autonomous mode starting up next doesn't inherit a stale
        # velocity left over from an unrelated earlier goal).
        smoothed_action: Optional[Action] = None

        uncertainty_session: Optional[UncertaintySession] = None
        if self.uncertainty_enabled:
            uncertainty_session = UncertaintySession(
                episode_id=f"nav-gui-{int(time.time())}",
                covariance_threshold=self.uncertainty_cov_threshold,
                covariance_value=0.0,
                client=uncertainty_client,
            )

        policy = self._make_policy(env, obs0)
        with self._lock:
            self.display.status_text = "NavDP policy is being loaded"
        policy.start()

        belief_tracker = self._new_belief_tracker()
        # Point-goal current-uncertainty bookkeeping (see the block before the
        # mode dispatch in the loop).
        pt_nav_id = None
        pt_seen = None
        pt_goal_xz = None
        pt_prev_pose = None
        # Persistent per-goal-text belief store for an active Mission (nav.mission.
        # Mission) -- distinct from belief_tracker above, which stays the ad-hoc
        # single-target tracker for Ground Target / plain Segment / click-to-goal
        # (self._mission is None in all of those). While a Mission is active,
        # belief_tracker is reassigned to alias goal_beliefs[current sub-goal's
        # target] (see the _start_mission_subgoal call sites below) so the rest of
        # this loop's MODE_RESOLVE handling needs no changes -- it just always
        # operates on "whichever tracker is active". See _sweep_goal_beliefs for
        # how entries other than the active one get seeded/refreshed.
        goal_beliefs: dict[str, BeliefGoalTracker] = {}
        # Last measured VLM target (world xz): stops the between-poll creep
        # from driving on through a target the rover has already reached.
        vlm_target_xz = None
        # Object-location belief shared by every VLM navigation in this session.
        object_memory = ObjectBeliefMemory()
        intervention_active = False
        with self._lock:
            self.display.intervention_sigma_threshold = self.intervention_sigma
        avoidance = self._new_avoidance() if self.cbf_enabled else None
        # goal/obstacle mask objects registered by the last "resolve" --
        # actually removed (env.remove_object_mask) rather than merely
        # untagged whenever they stop being current: on the next
        # resolve/rerun, on reset, on abandoning an unconfirmed review (Pick
        # Manually or switching to another driving mode without Confirm), or
        # on reaching a confirmed resolve goal -- see _clear_masks. Without
        # this, every resolve within one run leaves its old mesh behind.
        goal_objects: list = []
        obstacle_objects: list = []
        goal_spec = GoalSpec(
            goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
            obstacle_bboxes_norm=[],
            instruction_text="(no goal resolved yet)",
        )

        with self._lock:
            self.display.pose = obs0.pose
            self.display.status_text = "idle -- pick a driving mode"

        step = 0
        period = 1.0 / self.hz
        was_reviewing = False
        try:
            while self._running:
                t0 = time.time()

                (
                    do_resolve,
                    do_reset,
                    do_clear_all,
                    pixel_click,
                    do_confirm_seg,
                    do_rerun_seg,
                    do_pick_manual,
                    uncertainty_heading,
                    do_uncertainty_retry,
                    nav_command_text,
                    resolve_target_text,
                    do_confirm_mission,
                    do_cancel_mission,
                    vlm_instructions,
                    do_stop_vlm,
                    vlm_command_text,
                ) = self._consume_pending()

                if do_reset or do_clear_all:
                    # Shared by request_reset (do_reset) and clear_all_goals
                    # (do_clear_all, the Escape-key hard clear) -- everything
                    # below except the env.step teleport back to spawn, which
                    # only do_reset wants (clear_all_goals leaves the rover
                    # exactly where it is).
                    goal_objects, obstacle_objects = self._clear_masks(
                        env, goal_objects, obstacle_objects
                    )
                    with self._lock:
                        self._resolve_target_text = None
                        self._mission = None
                        self._pending_mission = None
                        self._cancel_mission_review_locked()
                    self._mission_target_yaw = None
                    belief_tracker = self._new_belief_tracker()
                    goal_beliefs = {}
                    if avoidance is not None:
                        avoidance = self._new_avoidance()
                    if do_reset:
                        with self._lock:
                            rp, self._reset_pose = self._reset_pose, None
                        rx, rz, ryaw = rp if rp is not None else (
                            self.start_x, self.start_z, math.radians(self.start_yaw_deg))
                        y = env.get_height_at_xz(rx, rz)
                        env.step(Pose(x=rx, y=y, z=rz, yaw=ryaw))
                    with self._lock:
                        self._mode = MODE_IDLE
                        self._world_goal = None
                        self._manual_action = Action(0.0, 0.0, 0.0)
                        self._uncertainty_halted = False
                        self._uncertainty_search_goal = None
                        self.display.status_text = (
                            "reset to spawn" if do_reset else "cleared -- idle"
                        )
                        self.display.goal_reached = False
                        self.display.click_status = ""
                        self.display.uncertainty_halted = False
                        self.display.uncertainty_searching = False
                        self.display.uncertainty_line = ""

                if vlm_instructions is not None:
                    # start_vlm_navigation already cleared self._mission /
                    # any prior self._vlm_navigator (see
                    # _cancel_mission_review_locked) on the GUI thread --
                    # this is just the (potentially slow, first-call
                    # model-loading) part deferred to the controller thread.
                    with self._lock:
                        self.display.status_text = (
                            f"{self.vlm_model_id or 'Qwen3.5-9B'} is being loaded"
                        )
                    if self._qwen35_manager is None:
                        self._qwen35_manager = Qwen35ServerManager(
                            port=self.qwen35_port,
                            model_id=self.vlm_model_id,
                            prompt_style=self.vlm_prompt_style,
                        )
                        self._qwen35_manager.start()
                    navigator = VlmPointNavigator(
                        vlm_instructions,
                        hfov_deg=HFOV_DEG,
                        replan_every=self.vlm_replan_every,
                        num_history=self.vlm_num_history,
                        qwen35_port=self.qwen35_port,
                        look_down_frame_fn=capture_vlm_look_down,
                        memory=object_memory,
                        intervention_sigma=self.intervention_sigma,
                    )
                    with self._lock:
                        self._vlm_navigator = navigator
                        self.display.status_text = (
                            f"VLM nav [1/{len(vlm_instructions)}]: "
                            f"{navigator.current_instruction}"
                        )
                    self._event_log.log(
                        "vlm_navigation_started", instructions=vlm_instructions
                    )

                if do_stop_vlm:
                    with self._lock:
                        self._vlm_navigator = None
                        # clear leg state so a later manual goal doesn't
                        # inherit the tight forward/home arrival radius
                        self._vlm_turn_direction = None
                        self._mode = MODE_IDLE
                        self._world_goal = None
                        self.display.status_text = "VLM navigation stopped"

                if do_resolve or do_rerun_seg:
                    goal_objects, obstacle_objects, goal_spec = self._do_resolve(
                        env,
                        step,
                        mask_dir,
                        goal_objects,
                        obstacle_objects,
                        goal_spec,
                        target_text=resolve_target_text,
                    )
                    belief_tracker.belief_g = None
                    with self._lock:
                        self._uncertainty_halted = False
                        self._uncertainty_search_goal = None
                        self.display.uncertainty_halted = False
                        self.display.uncertainty_searching = False

                if do_confirm_seg:
                    with self._lock:
                        if self._mode == MODE_REVIEW_SEGMENTATION:
                            self._mode = MODE_RESOLVE
                            self.display.status_text = (
                                f"resolved: {goal_spec.instruction_text}"
                            )
                            self.display.goal_reached = False

                if do_pick_manual:
                    with self._lock:
                        pick_manual_was_reviewing = (
                            self._mode == MODE_REVIEW_SEGMENTATION
                        )
                        if pick_manual_was_reviewing:
                            self._mode = MODE_IDLE
                            self._world_goal = None
                            self.display.status_text = (
                                "pick a goal point manually -- click the camera view"
                            )
                            self.display.goal_reached = False
                    if pick_manual_was_reviewing:
                        goal_objects, obstacle_objects = self._clear_masks(
                            env, goal_objects, obstacle_objects
                        )

                # A Mission finished parsing (background thread from a prior
                # tick's _dispatch_nav_command, see submit_nav_command) --
                # stage it for operator review instead of starting it
                # immediately. nav/gui1.py's Confirm-Mission panel reads
                # d.mission_review (.instruction/.directions/.goal_texts/
                # .goals) to show the interpreted step list; the operator
                # must request_confirm_mission() before anything below
                # actually dispatches it, or request_cancel_mission() to
                # discard it and try a different instruction.
                with self._lock:
                    pending_mission = self._pending_mission
                    self._pending_mission = None
                if pending_mission is not None:
                    with self._lock:
                        self._mission_awaiting_confirmation = pending_mission
                        self.display.mission_review = pending_mission
                        self.display.status_text = (
                            f"{pending_mission.instruction!r} -- confirm to begin"
                        )

                # Operator acted on a Mission awaiting confirmation.
                if do_confirm_mission:
                    with self._lock:
                        confirmed_mission = self._mission_awaiting_confirmation
                        self._cancel_mission_review_locked()
                    if confirmed_mission is not None:
                        with self._lock:
                            self._mission = confirmed_mission
                        # Record the rover's spawn point as a "home base" entry in
                        # the persistent belief bank (goal_beliefs) the moment a
                        # mission is dispatched, keyed the same way _sweep_goal_beliefs
                        # keys everything else -- so home is remembered right from
                        # the prompt, not only once a RETURN sub-goal is reached.
                        # This seed is what RETURN's own drive (below) actually
                        # navigates off of via MODE_RESOLVE -- like every other
                        # tracked goal it's then dead-reckoned every tick (see the
                        # "for bt in goal_beliefs.values()" propagate loop below),
                        # so a RETURN leg accrues the same odometry drift/
                        # uncertainty a GO_TO/FIND leg would, rather than homing in
                        # on an exact ground-truth world position the rover has no
                        # real way of knowing.
                        home_forward, home_left = body_frame_goal(
                            self.display.pose, (self.start_x, self.start_z)
                        )
                        goal_beliefs.setdefault(
                            "home base", self._new_belief_tracker(mission_sweep=True)
                        ).observe_body_point(home_forward, home_left)
                        self._event_log.log(
                            "mission_started",
                            text=confirmed_mission.instruction,
                            world=(round(self.start_x, 2), round(self.start_z, 2)),
                        )
                        goal_objects, obstacle_objects, goal_spec, belief_tracker = (
                            self._start_mission_subgoal(
                                env,
                                step,
                                mask_dir,
                                goal_objects,
                                obstacle_objects,
                                goal_spec,
                                goal_beliefs,
                                belief_tracker,
                            )
                        )
                elif do_cancel_mission:
                    with self._lock:
                        self._cancel_mission_review_locked()
                        self.display.status_text = "mission cancelled -- idle"

                obs = env.get_observation(frame_idx=step)
                self._raw_frame = (obs.rgb, obs.pose)  # newest raw frame (BEV map display)
                semantic = env.get_semantic_frame()
                goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
                obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255

                with self._lock:
                    mission_snapshot = self._mission
                if (
                    mission_snapshot is not None
                    and self.mission_belief_sweep_every > 0
                    and step % self.mission_belief_sweep_every == 0
                ):
                    self._sweep_goal_beliefs(
                        mission_snapshot, obs.rgb, obs.depth, goal_beliefs
                    )
                    with self._lock:
                        turning_now = self._mode == MODE_TURN
                    if turning_now and self._check_turn_interrupt(
                        mission_snapshot, goal_beliefs
                    ):
                        goal_objects, obstacle_objects, goal_spec, belief_tracker = (
                            self._start_mission_subgoal(
                                env,
                                step,
                                mask_dir,
                                goal_objects,
                                obstacle_objects,
                                goal_spec,
                                goal_beliefs,
                                belief_tracker,
                            )
                        )

                if pixel_click is not None:
                    self._handle_pixel_click(obs, pixel_click)

                if nav_command_text is not None:
                    self._dispatch_nav_command(nav_command_text, obs.rgb)

                if vlm_command_text is not None:
                    # Blocking Qwen call on the controller thread, same
                    # reasoning as _do_resolve/the VLM point-goal poll below
                    # -- split_instructions_verbose is a single quick text-
                    # decomposition call (3B model), not the slower 9B
                    # point-goal reasoning. Feeds self._pending_vlm_instructions
                    # so next tick's existing vlm_instructions-handling block
                    # (Qwen3.5-9B server spawn + VlmPointNavigator construction)
                    # picks it up exactly as if start_vlm_navigation had been
                    # called directly with the split result.
                    instructions, split_result = qwen_client.split_instructions_verbose(
                        obs.rgb, vlm_command_text
                    )
                    self._event_log.log(
                        "vlm_command_split",
                        command_text=vlm_command_text,
                        instructions=instructions,
                    )
                    with self._lock:
                        self._pending_vlm_instructions = instructions

                if uncertainty_session is not None:
                    self._handle_uncertainty_commands(
                        uncertainty_session,
                        belief_tracker,
                        obs,
                        uncertainty_heading,
                        do_uncertainty_retry,
                    )

                with self._lock:
                    iv_cmd = self._pending_intervention
                    self._pending_intervention = None
                    iv_reason = self._intervention_reason
                if iv_cmd == "start" and not intervention_active:
                    intervention_active = True
                    with self._lock:
                        self._vlm_turn_direction = None
                        self._mode = MODE_IDLE
                        self._world_goal = None
                        self._manual_action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                        self.display.intervention_active = True
                        self.display.intervention_locked = self._intervention_locked
                        self.display.intervention_reason = iv_reason
                        self.display.status_text = f"HUMAN INTERVENTION: {iv_reason}"
                    vlm_target_xz = None
                    self._event_log.log("human_intervention_started", reason=iv_reason)
                elif iv_cmd == "resume" and intervention_active and not self._can_resume():
                    with self._lock:
                        self.display.status_text = (
                            "Human control only: target still too uncertain — "
                            "drive until the target is found, then resume"
                        )
                    self._event_log.log("human_resume_refused")
                elif iv_cmd == "resume" and intervention_active:
                    intervention_active = False
                    with self._lock:
                        self._intervention_paused = False
                        nav_paused = self._vlm_navigator
                        self._mode = MODE_IDLE
                        self._manual_action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                        self.display.intervention_active = False
                        self.display.intervention_locked = False
                        self._intervention_locked = False
                        self.display.intervention_reason = ""
                        self.display.status_text = "autonomy resumed"
                    if nav_paused is not None:
                        self._wait_vlm_idle(nav_paused)
                        self._vlm_job = None  # pre-hand-over result is stale
                        nav_paused.resume_after_human()
                    self._event_log.log("human_intervention_ended")
                with self._lock:
                    vlm_navigator = self._vlm_navigator
                    vlm_turn_direction = self._vlm_turn_direction
                    self.display.target_uncertainty = (
                        getattr(vlm_navigator, "last_uncertainty", None)
                        if vlm_navigator is not None
                        else None
                    )
                if intervention_active:
                    with self._lock:
                        nav_obs = self._vlm_navigator
                    _p_obs = getattr(obs, "synced", None) or (None if self.real_rover_endpoint else obs)
                    if nav_obs is not None and _p_obs is not None and not self._vlm_busy(nav_obs):
                        try:
                            nav_obs.observe_while_paused(_p_obs, step)
                        except Exception:
                            pass  # best-effort: never break manual driving
                if intervention_active:
                    # Autonomy paused: no polling, no turn/leg bookkeeping --
                    # the operator's set_manual() commands drive (CBF is off in
                    # MODE_MANUAL by design). The navigator stays queued.
                    vlm_navigator = None
                    vlm_turn_direction = None
                if vlm_navigator is not None and vlm_turn_direction is not None:
                    # A turn dispatched via set_vlm_turn is physically
                    # executing (MODE_TURN) -- no Qwen polling while it's in
                    # flight, just wait for the same ground-truth
                    # goal_reached edge MODE_TURN produces for any other
                    # caller (see request_turn/_start_mission_subgoal). This
                    # is what replaced the old "ask Qwen every tick whether
                    # the turn looks done" scheme, which could stall
                    # indefinitely if the scene ahead never visibly changed.
                    with self._lock:
                        turn_done = self.display.goal_reached
                    if turn_done:
                        self._event_log.log(
                            "vlm_turn_completed", direction=vlm_turn_direction
                        )
                        with self._lock:
                            self._vlm_turn_direction = None
                        completed_instruction = vlm_navigator.current_instruction
                        # A turn only satisfies an instruction that IS a turn
                        # ("turn left", "turn around", "walk backward"). When
                        # the model turns while looking for a target ("go to
                        # the blue flag" -> turn right to re-find it) it's a
                        # search rotation: keep the instruction active and
                        # re-query. Advancing here is what falsely reported
                        # "reached" without ever reaching the target.
                        is_search_turn = (
                            vlm_turn_direction in ("left", "right")
                            and not any(
                                w in completed_instruction.lower()
                                for w in _TURN_WORDS
                            )
                        ) or (
                            # an exploratory "forward N m" the navigator
                            # added while seeking a target (the instruction
                            # itself isn't a distance command)
                            vlm_turn_direction == "forward"
                            and parse_simple_command(completed_instruction)
                            is None
                        )
                        if is_search_turn:
                            self._event_log.log(
                                "vlm_search_turn_done",
                                instruction=completed_instruction,
                            )
                            finished = False
                        else:
                            finished = vlm_navigator.advance()
                            self._event_log.log(
                                "vlm_instruction_advanced",
                                instruction=completed_instruction,
                                finished=finished,
                            )
                        if finished:
                            with self._lock:
                                self._vlm_navigator = None
                                self.display.status_text = "VLM navigation finished"
                elif vlm_navigator is not None:
                    # Blocking Qwen3.5-9B call on the controller thread, same
                    # as _do_resolve's SAM2+Qwen call above -- maybe_step
                    # itself no-ops (returns None immediately) on every tick
                    # that isn't a replan_every cadence tick, so this is a
                    # cheap no-op the vast majority of ticks.
                    # Qwen runs in the background (_poll_vlm_async): the loop,
                    # the video and NavDP keep going at full rate meanwhile.
                    vlm_result = self._poll_vlm_async(vlm_navigator, obs, step)
                    if getattr(vlm_navigator, "needs_human", None) and not self._vlm_busy(
                        vlm_navigator
                    ):
                        # Too uncertain where the target is: hand over to the
                        # operator (handled at the top of the next tick).
                        with self._lock:
                            self._pending_intervention = "start"
                            self._intervention_reason = "human action needed (use WASD)"
                            self._intervention_paused = True
                            self._intervention_locked = True
                        vlm_navigator.needs_human = None
                        vlm_result = None
                    if vlm_result is not None and vlm_result.action == "turn":
                        vlm_target_xz = None
                        self.set_vlm_turn(vlm_result.direction, vlm_result.degrees)
                        with self._lock:
                            self._vlm_turn_direction = vlm_result.direction
                        self._event_log.log(
                            "vlm_turn_started",
                            instruction=vlm_result.instruction_text,
                            direction=vlm_result.direction,
                            degrees=vlm_result.degrees,
                            reasoning=vlm_result.reasoning,
                        )
                        with self._lock:
                            self.display.status_text = (
                                f"VLM nav [{vlm_navigator.idx + 1}/"
                                f"{len(vlm_navigator.instructions)}]: "
                                f"{vlm_result.instruction_text} -- turning "
                                f"{vlm_result.direction}"
                            )
                    elif vlm_result is not None and vlm_result.action in (
                        "forward",
                        "stop",
                        "home",
                    ):
                        vlm_target_xz = None
                        self._start_vlm_motion(vlm_result, obs.pose, vlm_navigator)
                    elif vlm_result is not None:
                        crept = False
                        if vlm_result.world_xz is not None:
                            self.set_vlm_point_goal(*vlm_result.world_xz)
                            vlm_target_xz = vlm_result.target_world_xz
                            target = vlm_result.world_xz
                        else:
                            # found=False, or the point's depth was invalid --
                            # creep instead of freezing (see VLM_CREEP_M's
                            # docstring for the deadlock this avoids). If
                            # the model did name a horizontal position,
                            # creep toward that bearing rather than blindly
                            # straight ahead (right of centre = yaw decreases,
                            # same convention as set_vlm_turn).
                            bearing = 0.0
                            if vlm_result.steer_u is not None:
                                bearing = math.radians(
                                    (vlm_result.steer_u - 0.5) * HFOV_DEG
                                )
                            creep_yaw = obs.pose.yaw - bearing
                            creep_x = obs.pose.x - VLM_CREEP_M * math.sin(creep_yaw)
                            creep_z = obs.pose.z - VLM_CREEP_M * math.cos(creep_yaw)
                            self.set_vlm_point_goal(creep_x, creep_z)
                            crept = True
                            target = (creep_x, creep_z)

                        if self._vlm_last_target is not None and math.hypot(
                            target[0] - self._vlm_last_target[0],
                            target[1] - self._vlm_last_target[1],
                        ) < VLM_STUCK_RADIUS_M:
                            self._vlm_stuck_streak += 1
                        else:
                            self._vlm_stuck_streak = 1
                        self._vlm_last_target = target
                        stuck_force_advance = (
                            not vlm_result.advance
                            and self._vlm_stuck_streak >= VLM_STUCK_POLLS
                            # Only force-finish if the model at least sees the
                            # target; being stuck while the target is unseen
                            # must not be reported as reached.
                            and vlm_result.target_visible
                            # ...and it is MEASURED to be near (depth-resolved
                            # target within 3.5 m). A creeping rover toward an
                            # unresolved point always looks "stuck"; that must
                            # never count as arrival (live: every flag test
                            # ended forced=True 10+ m from the flag).
                            and vlm_result.target_world_xz is not None
                            and math.hypot(
                                vlm_result.target_world_xz[0] - obs.pose.x,
                                vlm_result.target_world_xz[1] - obs.pose.z,
                            )
                            <= 3.5
                        )

                        self._event_log.log(
                            "vlm_point_goal",
                            crept=crept,
                            instruction=vlm_result.instruction_text,
                            found=vlm_result.found,
                            # Explicit bool, not just world_xz's presence:
                            # EventLogger.log drops any field whose value is
                            # None (see its docstring/`if v is not None`), so
                            # a found=True-but-unresolved point (Qwen named a
                            # spot but pixel_to_world couldn't ground it --
                            # e.g. invalid/background depth there) would
                            # otherwise be silently indistinguishable from
                            # found=False in this log -- confirmed live
                            # 2026-08-24, that gap is what made an actual
                            # stuck-in-place bug hard to see at first.
                            resolved=vlm_result.world_xz is not None,
                            advance=vlm_result.advance,
                            reasoning=vlm_result.reasoning,
                            world_xz=vlm_result.world_xz,
                            pose_xz=(round(obs.pose.x, 3), round(obs.pose.z, 3)),
                            pose_yaw_deg=round(math.degrees(obs.pose.yaw), 1),
                            gyro_ok=getattr(self._live_env, "vis_accepted", None),
                            gyro_rejected=getattr(self._live_env, "vis_rejected", None),
                        )
                        with self._lock:
                            self.display.status_text = (
                                f"VLM nav [{vlm_navigator.idx + 1}/"
                                f"{len(vlm_navigator.instructions)}]: "
                                f"{vlm_result.instruction_text} -- {vlm_result.reasoning}"
                            )
                        if vlm_result.advance or stuck_force_advance:
                            finished = vlm_navigator.advance()
                            self._vlm_stuck_streak = 0
                            self._vlm_last_target = None
                            self._event_log.log(
                                "vlm_instruction_advanced",
                                instruction=vlm_result.instruction_text,
                                finished=finished,
                                forced=stuck_force_advance,
                            )
                            if finished:
                                with self._lock:
                                    self._vlm_navigator = None
                                    self.display.status_text = (
                                        "VLM navigation finished"
                                    )

                with self._lock:
                    mode = self._mode
                    manual_action = self._manual_action
                    world_goal = self._world_goal

                if mode != MODE_RESOLVE:
                    with self._lock:
                        if (
                            self._uncertainty_halted
                            or self._uncertainty_search_goal is not None
                        ):
                            self._uncertainty_halted = False
                            self._uncertainty_search_goal = None
                            self.display.uncertainty_halted = False
                            self.display.uncertainty_searching = False

                if was_reviewing and mode not in (
                    MODE_REVIEW_SEGMENTATION,
                    MODE_RESOLVE,
                ):
                    # Review was abandoned by switching to another driving
                    # mode directly (random goal / go home / manual drive /
                    # click-to-goal / stop) instead of going through Confirm,
                    # Rerun, or Pick Manually -- those already clear their own
                    # masks, so this only fires for the remaining paths.
                    goal_objects, obstacle_objects = self._clear_masks(
                        env, goal_objects, obstacle_objects
                    )
                was_reviewing = mode == MODE_REVIEW_SEGMENTATION

                # ── point-goal current uncertainty ─────────────────────────
                # Reset to the floor only when the navigator actually SEES the
                # target in a check; between sightings it grows with the
                # rover's real motion (metres driven, radians turned x range^2),
                # never with time alone, so a stopped rover holds its value.
                # Runs during a human hand-over too, so the operator sees it
                # fall back once the target is in view again.
                with self._lock:
                    pt_nav = self._vlm_navigator if self._mission is None else None
                if pt_nav is None:
                    pt_nav_id = pt_seen = pt_goal_xz = pt_prev_pose = None
                else:
                    if id(pt_nav) != pt_nav_id:
                        pt_nav_id, pt_seen, pt_goal_xz, pt_prev_pose = id(pt_nav), None, None, None
                        belief_tracker = self._new_belief_tracker()
                    if mode == MODE_POINT and world_goal is not None:
                        pt_goal_xz = (float(world_goal[0]), float(world_goal[1]))
                    if pt_goal_xz is not None:
                        f_b, l_b = body_frame_goal(obs.pose, pt_goal_xz)
                        seen_n = int(getattr(pt_nav, "seen_count", 0))
                        if belief_tracker.belief_g is None or seen_n != pt_seen:
                            belief_tracker.observe_body_point(f_b, l_b)
                            pt_seen = seen_n
                        else:
                            belief_tracker.belief_g = np.asarray([f_b, l_b], dtype=np.float32)
                            if pt_prev_pose is not None and self.uncertainty_enabled:
                                px, pz, pyaw = pt_prev_pose
                                dd = math.hypot(obs.pose.x - px, obs.pose.z - pz)
                                dth = abs((obs.pose.yaw - pyaw + math.pi) % (2 * math.pi) - math.pi)
                                rng2 = f_b * f_b + l_b * l_b
                                u = belief_tracker.uncertainty
                                belief_tracker.uncertainty = math.sqrt(
                                    u * u
                                    + self.point_unc_var_per_m * dd
                                    + self.point_unc_var_per_rad * rng2 * dth
                                )
                    pt_prev_pose = (float(obs.pose.x), float(obs.pose.z), float(obs.pose.yaw))
                self._point_unc_value = (
                    belief_tracker.uncertainty_value() if pt_nav is not None and pt_goal_xz is not None else 0.0
                )
                if (
                    pt_nav is not None
                    and self.uncertainty_enabled
                    and not intervention_active
                    and mode == MODE_POINT
                    and self._point_unc_value >= self.uncertainty_cov_threshold
                ):
                    # (an in-flight Qwen check is fine: resume waits for it and
                    # drops its result)
                    with self._lock:
                        if self._pending_intervention is None:
                            self._pending_intervention = "start"
                            self._intervention_reason = "human action needed (use WASD)"
                            self._intervention_paused = True
                            self._intervention_locked = True
                    self._event_log.log(
                        "point_uncertainty_handover",
                        value=round(self._point_unc_value, 3),
                        threshold=round(self.uncertainty_cov_threshold, 3),
                    )

                action = Action(0.0, 0.0, 0.0)
                trajectory = None
                goal_bearing: Optional[float] = None
                unresolved = False
                dist_to_goal: Optional[float] = None
                turn_ghost_body: Optional[tuple[float, float]] = None

                if self._uncertainty_halted:
                    # Frozen awaiting a human heading -- same "ignore drive
                    # commands while halted" contract kb_teleop_vl.py's
                    # uncertainty halt uses, just enforced here instead of at
                    # the input layer (nav/gui.py never blocks its own D-pad).
                    goal_bearing = belief_tracker.bearing()
                elif mode == MODE_MANUAL:
                    action = manual_action
                elif mode == MODE_POINT and world_goal is not None:
                    forward, left = body_frame_goal(obs.pose, world_goal)
                    if pt_nav is None:
                        # plain click/random goal: a known point, always "seen"
                        belief_tracker.observe_body_point(forward, left)
                    goal_bearing = belief_tracker.bearing()
                    with self._lock:
                        _on_leg = self._vlm_turn_direction in ("forward", "home")
                    # Explicit-distance legs ("forward 2 m") need a tight
                    # arrival radius; the normal 2 m one would end them
                    # before they start.
                    if math.hypot(forward, left) < (
                        VLM_LEG_REACHED_M if _on_leg else POINT_GOAL_REACHED_M
                    ):
                        with self._lock:
                            vlm_driving = self._vlm_navigator is not None
                            # A forward/home leg ends here; only ordinary
                            # VLM point goals keep creeping.
                            vlm_leg = self._vlm_turn_direction in ("forward", "home")
                        _at_target = (
                            vlm_target_xz is not None
                            and math.hypot(
                                vlm_target_xz[0] - obs.pose.x,
                                vlm_target_xz[1] - obs.pose.z,
                            )
                            <= 2.5
                        )
                        if vlm_driving and not vlm_leg and not _at_target:
                            # Keep moving instead of idling until the next
                            # VLM poll -- see VLM_CREEP_M's docstring
                            # (point 2) for the stop-start pattern this
                            # avoids; only while VlmPointNavigator owns the
                            # goal, a manual/mission point-goal still stops
                            # cleanly below.
                            creep_x = obs.pose.x - VLM_CREEP_M * math.sin(obs.pose.yaw)
                            creep_z = obs.pose.z - VLM_CREEP_M * math.cos(obs.pose.yaw)
                            self.set_vlm_point_goal(creep_x, creep_z)
                            world_goal = (creep_x, creep_z)
                            self._event_log.log(
                                "goal_reached",
                                kind="point",
                                world=world_goal,
                                vlm_creep=True,
                            )
                            forward, left = body_frame_goal(obs.pose, world_goal)
                            policy.set_goal_body(forward, left)
                            action, _vla_result = policy.act_verbose(
                                obs, semantic, goal_spec, step
                            )
                            trajectory = getattr(policy, "_last_trajectory", None)
                            self._log_planner_health(_vla_result, forward, left, obs)
                        else:
                            with self._lock:
                                self._mode = MODE_IDLE
                                self._world_goal = None
                                self.display.goal_reached = True
                            mode = MODE_IDLE
                            self._event_log.log(
                                "goal_reached", kind="point", world=world_goal
                            )
                    else:
                        policy.set_goal_body(forward, left)
                        action, _vla_result = policy.act_verbose(
                            obs, semantic, goal_spec, step
                        )
                        trajectory = getattr(policy, "_last_trajectory", None)
                        self._log_planner_health(_vla_result, forward, left, obs)
                elif mode == MODE_RESOLVE:
                    seen = belief_tracker.observe(goal_mask, obs.depth)
                    if not seen and self._ground_target_world is not None:
                        # DINO-grounded targets never actually re-seed belief
                        # via observe() above: register_object_mask's MESH_GOAL_ID
                        # object renders 0px on this GPU/driver (see CLAUDE.md's
                        # dynamic-object-render-bug note), so goal_mask is always
                        # empty and uncertainty grows unbounded even while the
                        # ghost mask is visibly on-screen. Fall back to seeding
                        # belief straight from the ground-truth world position
                        # (same mechanism MODE_POINT's observe_body_point uses),
                        # gated on the exact same in-frame test project_world_to_pixel
                        # runs below to decide whether to draw the ghost mask at
                        # all -- so belief resets precisely when the ghost mask
                        # is showing, not on some independent visibility notion.
                        gx, gy, gz = self._ground_target_world
                        if (
                            project_world_to_pixel(
                                obs.pose,
                                (gx, gy, gz),
                                HFOV_DEG,
                                obs.rgb.shape[1],
                                obs.rgb.shape[0],
                            )
                            is not None
                        ):
                            forward, left = body_frame_goal(obs.pose, (gx, gz))
                            belief_tracker.observe_body_point(forward, left)
                            seen = True
                    if seen and self._uncertainty_search_goal is not None:
                        # Goal re-sighted mid-search -- observe() above
                        # already reset uncertainty_value() to the sighted
                        # floor, so abandon the human-chosen heading and
                        # resume tracking the (now freshly re-anchored)
                        # belief directly.
                        self._uncertainty_search_goal = None
                        with self._lock:
                            self.display.uncertainty_searching = False
                    if belief_tracker.belief_g is not None:
                        forward, left = (float(v) for v in belief_tracker.belief_g)
                        searching = self._uncertainty_search_goal is not None
                        if searching:
                            search_forward, search_left = body_frame_goal(
                                obs.pose, self._uncertainty_search_goal
                            )
                            if (
                                math.hypot(search_forward, search_left)
                                < POINT_GOAL_REACHED_M
                            ):
                                # Reached the searched heading without a
                                # re-sighting -- fall back to the (still
                                # uncertain) dead-reckoned belief; the halt
                                # trigger below will ask again since
                                # uncertainty_value() is still >= threshold.
                                self._uncertainty_search_goal = None
                                searching = False
                                with self._lock:
                                    self.display.uncertainty_searching = False
                            else:
                                forward, left = search_forward, search_left
                        # "Reached" detection (Qwen touch-bottom query) was
                        # removed; this is the ground-truth-adjacent
                        # replacement -- same distance threshold MODE_POINT
                        # uses (POINT_GOAL_REACHED_M), checked against
                        # belief_tracker.distance() (the belief-tracked
                        # position, since a resolved/grounded goal never has
                        # a ground-truth one) rather than the possibly
                        # search-heading-overridden local forward/left.
                        # Gated on not-searching: the search sub-flow's own
                        # heading-reached fallback above is a different
                        # mechanic -- it just resumes belief tracking, it
                        # doesn't complete the goal. Setting goal_reached
                        # here feeds the same self._mission advance-on-
                        # goal_reached edge MODE_POINT/MODE_TURN already
                        # drive, so a mission's GO_TO/FIND sub-goal (and the
                        # belief bank behind it) now advances too.
                        if (
                            not searching
                            and belief_tracker.distance() < POINT_GOAL_REACHED_M
                        ):
                            with self._lock:
                                self._mode = MODE_IDLE
                                self.display.goal_reached = True
                            mode = MODE_IDLE
                            self._event_log.log(
                                "goal_reached",
                                kind="resolve",
                                name=goal_spec.instruction_text,
                            )
                        else:
                            policy.set_goal_body(forward, left)
                            action, _vla_result = policy.act_verbose(
                                obs, semantic, goal_spec, step
                            )
                            trajectory = getattr(policy, "_last_trajectory", None)
                            self._log_planner_health(_vla_result, forward, left, obs)
                            # CBF's steering hint must track whichever target
                            # is actually being driven to -- the searched
                            # heading while searching, the dead-reckoned
                            # belief otherwise -- not always belief_g's own
                            # (possibly stale) bearing.
                            goal_bearing = math.atan2(left, forward)
                    else:
                        # Never sighted the goal mask yet this episode -- hold
                        # rather than drive on NavdpUpstreamPolicy's hidden
                        # constructor default (see this module's docstring /
                        # next.md's Integration-project Phase 5).
                        unresolved = True
                elif mode == MODE_TURN and self._mission_target_yaw is not None:
                    yaw_err = (self._mission_target_yaw - obs.pose.yaw + math.pi) % (
                        2.0 * math.pi
                    ) - math.pi
                    if abs(yaw_err) < TURN_GOAL_REACHED_RAD:
                        with self._lock:
                            self._mode = MODE_IDLE
                            self.display.goal_reached = True
                        mode = MODE_IDLE
                        self._mission_target_yaw = None
                        self._event_log.log("goal_reached", kind="turn")
                    else:
                        # Ghost point proportional to how much turn is left
                        # -- see TURN_GHOST_* docstring above. Clamping the
                        # bearing (not just the resulting pixel) keeps
                        # forward strictly positive, so this is always a
                        # normal in-front point goal for the policy, never
                        # a behind-the-camera one.
                        turn_bearing = max(
                            -TURN_GHOST_MAX_BEARING_RAD,
                            min(TURN_GHOST_MAX_BEARING_RAD, yaw_err),
                        )
                        forward = TURN_GHOST_LOOKAHEAD_M * math.cos(turn_bearing)
                        left = TURN_GHOST_LOOKAHEAD_M * math.sin(turn_bearing)
                        turn_ghost_body = (forward, left)
                        goal_bearing = turn_bearing
                        # Pure in-place rotation: no linear velocity at all
                        # (the ghost point above is only the on-screen marker).
                        yaw_cmd = max(0.15, min(self.max_yaw_rate, 1.5 * abs(yaw_err)))
                        action = Action(
                            v_fwd=0.0, v_lat=0.0,
                            yaw_rate=math.copysign(yaw_cmd, yaw_err),
                        )
                        trajectory = None

                if self._mission is not None and self.display.goal_reached:
                    # Ground-truth "reached" edge for the mission's current
                    # sub-goal (same signal a manually driven point/resolve
                    # goal already produces) -- advance and kick off whatever
                    # driving mode the next sub-goal needs. _start_mission_subgoal
                    # clears goal_reached again (or clears self._mission
                    # entirely on GoalKind.DONE), so this only fires once per
                    # sub-goal completion, not every tick after.
                    with self._lock:
                        if self._mission is not None:
                            completed_step = self._mission.current
                            self._event_log.log(
                                "mission_subgoal_reached",
                                step=completed_step.raw,
                                kind=completed_step.kind.name,
                            )
                            self._mission.advance()
                    goal_objects, obstacle_objects, goal_spec, belief_tracker = (
                        self._start_mission_subgoal(
                            env,
                            step,
                            mask_dir,
                            goal_objects,
                            obstacle_objects,
                            goal_spec,
                            goal_beliefs,
                            belief_tracker,
                        )
                    )

                obstacle_point = None
                cbf_info: dict = {}
                if (
                    avoidance is not None
                    and mode not in (MODE_MANUAL, MODE_TURN)
                    and not self._uncertainty_halted
                ):
                    height, width = obs.depth.shape[:2]
                    intr = intrinsics_from_hfov(height, width, HFOV_DEG)
                    obstacle_point = avoidance.nearest_obstacle(
                        obstacle_mask, obs.depth, intr
                    )
                    action, cbf_info = avoidance.apply(
                        action, obstacle_point, goal_bearing
                    )

                # Action smoothing (see action_smoothing_alpha's docstring):
                # blends this tick's action into a running EMA rather than
                # applying it raw, damping the tick-to-tick jitter that CBF's
                # orbit/pursuit switching and NavDP's own per-tick trajectory
                # resampling can otherwise produce as visible left-right
                # weave even while the high-level goal point stays fixed.
                # Off during MODE_MANUAL (an operator expects direct,
                # unfiltered control), MODE_IDLE, and MODE_TURN -- a turn
                # needs full yaw_rate immediately to reach TURN_GOAL_REACHED_RAD
                # promptly (confirmed live: with smoothing applied here too, a
                # plain 90-degree turn that normally completes in ~5-10s took
                # 40s+ to ramp up to enough yaw rate), and turning is already a
                # single deliberate maneuver rather than continuous driving, so
                # it doesn't have the same jitter-from-repeated-replanning
                # problem this smoothing targets. Resets the EMA state on any
                # of these three so the next smoothed leg starts from that
                # tick's actual action instead of inheriting stale momentum.
                if (
                    mode in (MODE_MANUAL, MODE_IDLE, MODE_TURN)
                    or self.action_smoothing_alpha >= 1.0
                ):
                    smoothed_action = None
                else:
                    alpha = self.action_smoothing_alpha
                    if smoothed_action is None:
                        smoothed_action = action
                    else:
                        smoothed_action = Action(
                            v_fwd=smoothed_action.v_fwd
                            + alpha * (action.v_fwd - smoothed_action.v_fwd),
                            v_lat=smoothed_action.v_lat
                            + alpha * (action.v_lat - smoothed_action.v_lat),
                            yaw_rate=smoothed_action.yaw_rate
                            + alpha * (action.yaw_rate - smoothed_action.yaw_rate),
                        )
                    action = smoothed_action

                new_pose = integrate_mars(obs.pose, action, self.dt)
                env.step(new_pose)

                if (
                    (mode == MODE_RESOLVE or (mode == MODE_POINT and pt_nav is None))
                    and self._mission is None
                    and not self._uncertainty_halted
                ):
                    # (VLM point goals grow by motion in the block above instead)
                    belief_tracker.propagate(action, self.dt)

                if self._mission is not None and not self._uncertainty_halted:
                    # Every tracked goal belief (not just the active one) dead-
                    # reckons against the rover's actual executed motion this tick,
                    # regardless of which mode that motion happened under -- a
                    # MODE_TURN/MODE_MANUAL leg still moves the rover, and a
                    # previously-sighted goal's remembered bearing must still
                    # decay/rotate with it. belief_tracker (the active goal's
                    # tracker, aliased into goal_beliefs by _start_mission_subgoal)
                    # is one of these entries, so it's propagated here instead of
                    # the ad-hoc-only line above. Frozen while uncertainty-halted,
                    # same "no growth while frozen" contract kb_teleop_vl.py's
                    # _update_uncertainty_covariance uses -- action is pinned to
                    # zero during a halt anyway, but propagate()'s uncertainty
                    # growth is time-based (uncertainty_growth_increment), not
                    # action-scaled, so without this guard it kept climbing every
                    # tick even while frozen awaiting a human heading.
                    for bt in goal_beliefs.values():
                        bt.propagate(action, self.dt)

                if (
                    uncertainty_session is not None
                    and mode == MODE_RESOLVE
                    and not self._uncertainty_halted
                    and not self._uncertainty_request_in_flight
                    and self._uncertainty_search_goal is None
                    and belief_tracker.belief_g is not None
                    and belief_tracker.uncertainty_value()
                    >= self.uncertainty_cov_threshold
                ):
                    self._uncertainty_halted = True
                    with self._lock:
                        self.display.uncertainty_halted = True
                    self._dispatch_uncertainty_request(
                        uncertainty_session, belief_tracker, obs.rgb, retry=False
                    )

                if mode == MODE_POINT and world_goal is not None:
                    dist_to_goal = math.hypot(
                        world_goal[0] - new_pose.x, world_goal[1] - new_pose.z
                    )
                elif mode == MODE_RESOLVE and belief_tracker.belief_g is not None:
                    dist_to_goal = belief_tracker.distance()

                goal_pixel = None
                if mode == MODE_POINT and world_goal is not None:
                    goal_y = env.get_height_at_xz(world_goal[0], world_goal[1])
                    goal_pixel = project_world_to_pixel(
                        obs.pose,
                        (world_goal[0], goal_y, world_goal[1]),
                        HFOV_DEG,
                        obs.rgb.shape[1],
                        obs.rgb.shape[0],
                    )

                status = self._status_text(mode, unresolved, goal_spec, dist_to_goal)
                vis_rgb = overlay_semantic_masks(obs.rgb, semantic)
                if SHOW_NAVDP_PATHS:
                    cands = getattr(policy, "_last_candidates", None)
                    if cands is not None or trajectory is not None:
                        try:
                            cam_h = float(obs.pose.y) - float(env._terrain(obs.pose.x, obs.pose.z))
                        except Exception:
                            cam_h = self.real_camera_height
                        vis_rgb = draw_navdp_paths(
                            vis_rgb, cands, getattr(policy, "_last_values", None),
                            trajectory, cam_h, HFOV_DEG,
                        )
                if SHOW_FRAME_MARKERS and goal_pixel is not None:
                    vis_rgb = draw_point_marker(vis_rgb, goal_pixel)
                if SHOW_FRAME_MARKERS and self._ground_target_world is not None:
                    # Reprojected every frame (not baked in once) so the
                    # ghost mask tracks correctly as the rover moves -- only
                    # drawn when the target's fixed world position actually
                    # reprojects into this frame (project_world_point_with_pixel_radius
                    # returns None otherwise), i.e. only ever shown over what
                    # can currently be seen, never extrapolated off-screen.
                    ground_proj = project_world_point_with_pixel_radius(
                        obs.pose,
                        self._ground_target_world,
                        GHOST_TARGET_RADIUS_M,
                        HFOV_DEG,
                        obs.rgb.shape[1],
                        obs.rgb.shape[0],
                    )
                    if ground_proj is not None:
                        gu, gv, gr = ground_proj
                        vis_rgb = draw_ghost_mask(vis_rgb, gu, gv, gr)
                if SHOW_FRAME_MARKERS and turn_ghost_body is not None:
                    # Same single-horizontal-line projection MODE_TURN's
                    # driving goal was built from -- forward > 0 is
                    # guaranteed by the bearing clamp above, so this always
                    # resolves to a real pixel, never None in practice.
                    turn_pixel = project_body_point_to_pixel(
                        turn_ghost_body[0],
                        turn_ghost_body[1],
                        HFOV_DEG,
                        obs.rgb.shape[0],
                        obs.rgb.shape[1],
                    )
                    if turn_pixel is not None:
                        tu, tv = turn_pixel
                        vis_rgb = draw_ghost_mask(vis_rgb, tu, tv, TURN_GHOST_RADIUS_PX)

                vis_rgb_topdown = (
                    env.get_topdown_rgb() if self.enable_topdown_viz else None
                )

                with self._lock:
                    d = self.display
                    d.vis_rgb = vis_rgb
                    d.vis_rgb_time = time.monotonic()
                    d.vis_rgb_topdown = vis_rgb_topdown
                    d.pose = new_pose
                    d.mode = mode
                    d.status_text = status
                    d.action = action
                    d.distance = dist_to_goal
                    d.goal_world = world_goal
                    d.goal_label = (
                        goal_spec.instruction_text if goal_spec is not None else ""
                    )
                    d.belief_g = (
                        None
                        if belief_tracker.belief_g is None
                        else tuple(float(v) for v in belief_tracker.belief_g)
                    )
                    d.trajectory = trajectory
                    d.obstacle_point = (
                        None
                        if obstacle_point is None
                        else tuple(float(v) for v in obstacle_point)
                    )
                    d.cbf_info = cbf_info
                    d.step = step
                    d.frame_count += 1
                    d.uncertainty_value = belief_tracker.uncertainty_value()
                    home_bt = goal_beliefs.get("home base")
                    d.home_base_uncertainty_value = (
                        home_bt.uncertainty_value() if home_bt is not None else 0.0
                    )
                    d.mission_status = (
                        self._mission.status() if self._mission is not None else ""
                    )
                    d.mission_goals = (
                        list(self._mission.goals) if self._mission is not None else []
                    )
                    d.mission_goal_idx = (
                        self._mission.idx if self._mission is not None else -1
                    )

                if (
                    home_bt is not None
                    and step % self._home_base_uncertainty_log_every == 0
                ):
                    self._event_log.log(
                        "home_base_uncertainty",
                        value=round(d.home_base_uncertainty_value, 3),
                        threshold=round(self.uncertainty_cov_threshold, 3),
                    )

                step += 1
                remaining = period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            policy.stop()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _consume_pending(
        self,
    ) -> tuple[
        bool,
        bool,
        bool,
        Optional[tuple],
        bool,
        bool,
        bool,
        Optional[float],
        bool,
        Optional[str],
        Optional[str],
        bool,
        bool,
        Optional[list],
        bool,
        Optional[str],
    ]:
        with self._lock:
            do_resolve = self._pending_resolve
            resolve_target_text = self._resolve_target_text
            do_reset = self._pending_reset
            do_clear_all = self._pending_clear_all
            pixel_click = self._pending_pixel_click
            do_confirm_seg = self._pending_confirm_segmentation
            do_rerun_seg = self._pending_rerun_segmentation
            do_pick_manual = self._pending_pick_manually
            uncertainty_heading = self._pending_uncertainty_heading
            do_uncertainty_retry = self._pending_uncertainty_retry
            nav_command_text = self._pending_nav_command_text
            do_confirm_mission = self._pending_confirm_mission
            do_cancel_mission = self._pending_cancel_mission
            vlm_instructions = self._pending_vlm_instructions
            do_stop_vlm = self._pending_stop_vlm
            vlm_command_text = self._pending_vlm_command_text
            self._pending_resolve = False
            self._pending_reset = False
            self._pending_clear_all = False
            self._pending_pixel_click = None
            self._pending_confirm_segmentation = False
            self._pending_rerun_segmentation = False
            self._pending_pick_manually = False
            self._pending_uncertainty_heading = None
            self._pending_uncertainty_retry = False
            self._pending_nav_command_text = None
            self._pending_confirm_mission = False
            self._pending_cancel_mission = False
            self._pending_vlm_instructions = None
            self._pending_stop_vlm = False
            self._pending_vlm_command_text = None
        return (
            do_resolve,
            do_reset,
            do_clear_all,
            pixel_click,
            do_confirm_seg,
            do_rerun_seg,
            do_pick_manual,
            uncertainty_heading,
            do_uncertainty_retry,
            nav_command_text,
            resolve_target_text,
            do_confirm_mission,
            do_cancel_mission,
            vlm_instructions,
            do_stop_vlm,
            vlm_command_text,
        )

    def _handle_pixel_click(self, obs, pixel_click: tuple) -> None:
        """Backproject a clicked (x_norm, y_norm) into a world-frame (x, z)
        point via the live depth frame and switch to MODE_POINT driving
        toward it -- same bbox_to_world median-over-patch machinery
        first_frame_resolver's goal already uses, applied to a small patch
        around the clicked pixel (not just the single pixel) so a click that
        lands exactly on a depth discontinuity (a rock's silhouette edge)
        doesn't seed a wildly wrong point. Unlike the belief-tracked RESOLVE
        goal, this point is never re-derived from odometry/dead-reckoning --
        it's a fixed world coordinate, recovered fresh from ground-truth pose
        every tick by body_frame_goal, so it can't drift."""
        x_norm, y_norm = pixel_click
        height, width = np.asarray(obs.depth).shape[:2]
        px, py = x_norm * width, y_norm * height
        margin = 4.0
        bbox_norm = (
            max(0.0, (px - margin) / width),
            max(0.0, (py - margin) / height),
            min(1.0, (px + margin) / width),
            min(1.0, (py + margin) / height),
        )
        goal_xyz = bbox_to_world(obs, bbox_norm, hfov_deg=HFOV_DEG)
        with self._lock:
            if goal_xyz is None:
                self.display.click_status = "click ignored: no valid depth there"
                return
            gx, _gy, gz = goal_xyz
            self._world_goal = (gx, gz)
            self._mode = MODE_POINT
            self.display.goal_reached = False
            self.display.click_status = f"point goal set at world ({gx:.1f}, {gz:.1f})"

    def _handle_uncertainty_commands(
        self,
        session: UncertaintySession,
        belief_tracker: BeliefGoalTracker,
        obs,
        heading_deg: Optional[float],
        do_retry: bool,
    ) -> None:
        """Applies a queued retry/heading-submit command against an active
        uncertainty halt (see _env_loop's halt-trigger block). Both are
        no-ops if not currently halted, or a request is already in flight --
        guards a stale GUI click racing a halt that's since been cleared by
        a fresh sighting or a mode switch (see the `mode != MODE_RESOLVE`
        cleanup in _env_loop)."""
        if (
            do_retry
            and self._uncertainty_halted
            and not self._uncertainty_request_in_flight
        ):
            self._dispatch_uncertainty_request(
                session, belief_tracker, obs.rgb, retry=True
            )

        if (
            heading_deg is None
            or not self._uncertainty_halted
            or self._uncertainty_request_in_flight
        ):
            return

        # Pure local packaging, no VLM call (directive_engine._query_uncertainty's
        # "submit phase") -- safe to resolve synchronously on this thread.
        result = session.submit_heading(angle_deg=heading_deg)
        payload = result.uncertainty_payload
        self._uncertainty_search_goal = heading_ahead_point(
            obs.pose, heading_deg, self.uncertainty_search_dist, self.world_limit
        )
        self._uncertainty_halted = False
        with self._lock:
            self.display.uncertainty_halted = False
            self.display.uncertainty_searching = True
            self.display.uncertainty_line = (
                f"heading {heading_deg:+.0f}deg -> traverse up to "
                f"{payload.max_units:.1f} units, or until the goal is re-sighted"
            )

    def _dispatch_uncertainty_request(
        self,
        session: UncertaintySession,
        belief_tracker: BeliefGoalTracker,
        frame: np.ndarray,
        retry: bool,
    ) -> None:
        """Kicks off the (slow) VLM sweep-description call on a background
        thread -- the control loop above must keep ticking at self.hz
        regardless of inference latency, same reasoning as
        NavdpUpstreamPolicy's own async replanning. session.covariance_value
        is refreshed here (kb_teleop_vl.py's version leaves it fixed at
        construction) so the sweep-description prompt reflects the actual
        uncertainty that crossed the threshold, not a stale value."""
        session.covariance_value = belief_tracker.uncertainty_value()
        self._uncertainty_request_in_flight = True
        with self._lock:
            self.display.uncertainty_request_in_flight = True
            self.display.uncertainty_line = (
                "retrying VLM sweep description..."
                if retry
                else "requesting VLM sweep description..."
            )
        target = session.retry if retry else session.request_human_heading
        thread = threading.Thread(
            target=self._uncertainty_worker, args=(target, frame), daemon=True
        )
        thread.start()

    def _uncertainty_worker(self, session_method, frame: np.ndarray) -> None:
        """Runs on a background thread -- the actual sweep-description VLM
        call. Never touches habitat-sim; only writes self.display under
        self._lock, same cross-thread pattern the rest of this controller
        uses for its snapshot() contract."""
        try:
            result = session_method(frame)
            payload = result.uncertainty_payload
            line = (
                f"attempt {payload.attempt}: {result.raw_response!r} -- "
                "pick a heading, or Retry"
            )
        except Exception as exc:
            line = f"uncertainty request failed: {exc}"
        with self._lock:
            self._uncertainty_request_in_flight = False
            self.display.uncertainty_request_in_flight = False
            self.display.uncertainty_line = line

    def _dispatch_nav_command(self, text: str, frame: np.ndarray) -> None:
        """Kicks off the (slow) nav-command-parsing VLM call on a background
        thread -- same reasoning as _dispatch_uncertainty_request, the
        control loop must keep ticking at self.hz regardless of inference
        latency."""
        thread = threading.Thread(
            target=self._nav_command_worker, args=(text, frame), daemon=True
        )
        thread.start()

    def _nav_command_worker(self, text: str, frame: np.ndarray) -> None:
        """Runs on a background thread -- splits a free-text nav command
        into (directions, goals) via qwen_client.parse_nav_command (see
        qwen_prompts.build_parse_nav_command_prompt), then turns that split
        into a Mission (nav.mission.Mission/parse_parts: every direction
        first as an in-place TURN, then every goal as GO_TO/RETURN). Never
        touches habitat-sim; only writes self._pending_mission under
        self._lock, picked up next tick by _env_loop and staged for operator
        confirmation (see request_confirm_mission's docstring) rather than
        dispatched directly -- the "mission_started" event log fires once
        the operator actually confirms it, not here."""
        try:
            directions, goals = qwen_client.parse_nav_command(frame, text)
            print(f"[nav command] {text!r} -> directions={directions} goals={goals}")
            mission = Mission(text, directions=directions, goal_texts=goals)
        except Exception as exc:
            print(f"[nav command] {text!r} failed: {exc}")
            return
        self._event_log.log(
            "mission_parsed", text=text, directions=directions, goals=goals
        )
        with self._lock:
            self._pending_mission = mission

    def _start_mission_subgoal(
        self,
        env,
        step,
        mask_dir,
        goal_objects,
        obstacle_objects,
        goal_spec,
        goal_beliefs,
        belief_tracker,
    ):
        """Kicks off whichever driving mode the active Mission's current
        sub-goal needs (see nav.mission.Mission) -- called once when a
        parsed Mission is confirmed (request_confirm_mission, see
        submit_nav_command's docstring) and again every time the
        mission-advance edge in _env_loop fires. Only ever called from the
        controller thread. A GO_TO/FIND resolve that fails (DINO doesn't
        ground the target in the frame this step starts on) no longer means
        the step is abandoned outright -- if a previous sighting (this
        step's own earlier attempt, or _sweep_goal_beliefs picking it up
        while an earlier leg of the mission was driving) left a confident
        belief for this exact target text in `goal_beliefs`, resume driving
        toward that remembered position instead. Only truly skipped (as
        before this feature existed) when neither a fresh resolve nor a
        usable prior belief is available.

        `goal_beliefs` is the persistent dict[goal-text -> BeliefGoalTracker]
        for the whole Mission (see _env_loop) -- entries survive across
        sub-goals so a target sighted while driving toward an earlier one
        isn't forgotten by the time the mission reaches it. `belief_tracker`
        is the CURRENTLY active tracker (whichever one _env_loop's tick loop
        is calling propagate()/observe() on); RETURN/TURN/DONE have no goal
        text of their own and just pass it through unchanged. Both are
        returned so the caller can rebind its own locals, same pattern
        goal_spec already uses."""
        while True:
            with self._lock:
                mission = self._mission
                goal = mission.current if mission is not None else None
            if goal is None:
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind == GoalKind.DONE:
                with self._lock:
                    self._mode = MODE_IDLE
                    self._world_goal = None
                    self._mission = None
                self._mission_target_yaw = None
                self._event_log.log("mission_complete")
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind in (GoalKind.GO_TO, GoalKind.FIND):
                # Single front-facing DINO resolve on whatever's already in
                # the current frame -- no 360 sweep. sweep_and_seed_beliefs
                # (dino_grounding_resolver) rotated through 8 discrete
                # headings with no cross-frame consistency check on what it
                # found, which let a single high-scoring false positive
                # (e.g. a blue cuboid grounding "flag") pick the bearing the
                # rover then turned to face and re-resolved against. Disabled
                # for now -- see this session's flag-misdetection debugging.
                # (_sweep_goal_beliefs below is a different, lower-stakes use
                # of the same detector -- it only ever seeds a passive belief,
                # never itself re-orients the rover.)
                goal_objects, obstacle_objects, goal_spec = self._do_resolve(
                    env,
                    step,
                    mask_dir,
                    goal_objects,
                    obstacle_objects,
                    goal_spec,
                    target_text=goal.target,
                )
                with self._lock:
                    # _do_resolve always leaves a *successful* resolve in
                    # MODE_REVIEW_SEGMENTATION with an empty click_status
                    # (goal_mask_warning) -- see its docstring. A mission
                    # step auto-confirms straight through that review pause
                    # instead of waiting for a human Confirm click.
                    resolved = (
                        self._mode == MODE_REVIEW_SEGMENTATION
                        and not self.display.click_status
                    )
                    if resolved:
                        self._mode = MODE_RESOLVE
                        self.display.status_text = (
                            f"[mission] resolved: {goal_spec.instruction_text}"
                        )
                if resolved:
                    # Fresh sighting -- (re)anchor this goal's persistent
                    # belief and discard whatever it held before: belief_g is
                    # reset to None so MODE_RESOLVE's own per-tick fallback
                    # (self._ground_target_world -> body_frame_goal ->
                    # observe_body_point, see the MODE_RESOLVE branch below)
                    # re-seeds it from the same ground_target_world
                    # _do_resolve just set, as soon as the target is next
                    # in-frame -- same reset _do_resolve's ad-hoc caller
                    # already relies on for its own single shared tracker.
                    belief_tracker = goal_beliefs.setdefault(
                        goal.target, self._new_belief_tracker()
                    )
                    belief_tracker.belief_g = None
                    self._activate_belief_tracker(belief_tracker)
                    return goal_objects, obstacle_objects, goal_spec, belief_tracker

                prior = goal_beliefs.get(goal.target)
                if (
                    prior is not None
                    and prior.belief_g is not None
                    and prior.uncertainty_value() < self.uncertainty_cov_threshold
                ):
                    # Not in the frame this step started on, but a previous
                    # sighting (an earlier leg's periodic _sweep_goal_beliefs
                    # hit, most likely) left a still-confident belief for this
                    # exact target -- resume driving toward it rather than
                    # abandoning the step. No mask/obstacle registration here
                    # (goal_objects/obstacle_objects are already empty --
                    # _do_resolve's failure path clears them same as always),
                    # and goal_spec content doesn't matter to the driving
                    # policy (NavdpUpstreamPolicy.act_verbose only reads the
                    # body-frame point set via policy.set_goal_body, see this
                    # module's docstring) -- a placeholder is safe, same one
                    # _env_loop starts with before any resolve has happened.
                    with self._lock:
                        self._mode = MODE_RESOLVE
                        self.display.status_text = (
                            f"[mission] resuming toward previously-sighted "
                            f"'{goal.target}'"
                        )
                        self.display.goal_reached = False
                    placeholder_spec = GoalSpec(
                        goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                        obstacle_bboxes_norm=[],
                        instruction_text=(
                            f"Navigate to the {goal.target} (from memory)."
                        ),
                    )
                    self._activate_belief_tracker(prior)
                    return goal_objects, obstacle_objects, placeholder_spec, prior

                print(
                    f"[mission] step {goal.raw!r} failed to resolve and no "
                    "usable prior belief -- skipping"
                )
                self._event_log.log(
                    "mission_subgoal_skipped", step=goal.raw, target=goal.target
                )
                with self._lock:
                    if self._mission is mission:
                        self._mission.advance()
                continue

            if goal.kind == GoalKind.RETURN:
                # _clear_masks also nulls self._ground_target_world -- without
                # this, MODE_RESOLVE's unseen-goal fallback (below, "if not
                # seen and self._ground_target_world is not None") would
                # re-seed the home belief from whatever DINO-grounded target
                # (e.g. "flag") the mission's previous GO_TO leg left behind,
                # since RETURN itself never calls _do_resolve to clear it.
                goal_objects, obstacle_objects = self._clear_masks(
                    env, goal_objects, obstacle_objects
                )
                home = goal_beliefs.get("home base")
                if home is not None and home.belief_g is not None:
                    # Drive off the dead-reckoned "home base" belief seeded at
                    # mission dispatch (see home_base_seeded above) instead of
                    # the ground-truth spawn point -- same MODE_RESOLVE
                    # mechanism/placeholder-spec pattern the "resume toward
                    # previously-sighted target" path above uses, so RETURN
                    # is subject to the same odometry drift/uncertainty halt
                    # every other belief-tracked goal is, not a magic exact
                    # fix on the world frame.
                    placeholder_spec = GoalSpec(
                        goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                        obstacle_bboxes_norm=[],
                        instruction_text="Navigate back to home base.",
                    )
                    with self._lock:
                        self._mode = MODE_RESOLVE
                        self.display.status_text = (
                            "[mission] returning to home base (belief)"
                        )
                        self.display.goal_reached = False
                    self._activate_belief_tracker(home)
                    return goal_objects, obstacle_objects, placeholder_spec, home
                # No seeded belief -- shouldn't normally happen (home_base_seeded
                # runs at mission dispatch, before any RETURN leg can be
                # reached), but fall back to the exact ground-truth spawn
                # rather than stalling the mission.
                with self._lock:
                    self._world_goal = (self.start_x, self.start_z)
                    self._mode = MODE_POINT
                    self.display.goal_reached = False
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind == GoalKind.TURN:
                pose = self.display.pose
                heading_deg = {
                    "right": 90.0,
                    "left": -90.0,
                    "back": 180.0,
                    "around": 180.0,
                }.get(goal.target, 180.0)
                self._mission_target_yaw = pose.yaw - math.radians(heading_deg)
                with self._lock:
                    self._mode = MODE_TURN
                    self.display.goal_reached = False
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

            if goal.kind == GoalKind.ADVANCE:
                # "go left"/"go right"/"go straight" -- unlike TURN (rotate
                # only), this actually drives: a world-frame point
                # go_direction_distance_m straight ahead along whatever
                # heading the rover is CURRENTLY facing (the preceding TURN
                # sub-goal, if any, has already finished by the time this
                # runs -- see nav.mission._direction_subgoals). Same
                # ground-truth MODE_POINT mechanism RETURN/random-goal
                # driving already use (CBF obstacle avoidance included), just
                # aimed at a point instead of the spawn.
                pose = self.display.pose
                dist = self.go_direction_distance_m
                tx = pose.x - dist * math.sin(pose.yaw)
                tz = pose.z - dist * math.cos(pose.yaw)
                with self._lock:
                    self._world_goal = (tx, tz)
                    self._mode = MODE_POINT
                    self.display.goal_reached = False
                return goal_objects, obstacle_objects, goal_spec, belief_tracker

    def _sweep_goal_beliefs(self, mission, rgb, depth, goal_beliefs) -> None:
        """Opportunistic multi-goal check against the current frame while a
        Mission is driving normally (no rotation, unlike
        dino_grounding_resolver.sweep_and_seed_beliefs's 360deg scan) --
        called every self.mission_belief_sweep_every ticks from _env_loop.
        Checks every GO_TO/FIND target from the mission's CURRENT step
        onward (already-completed steps aren't worth the DINO call) against
        this one frame, and seeds/refreshes goal_beliefs for whichever ones
        hit -- including the currently active goal: its own live-mask/
        ground-truth-anchor path (self._ground_target_world) already
        re-derives it every tick when in view, so a sweep hit on it is either
        redundant (overwritten again next tick) or, when that anchor is
        itself out of view, the only correction available -- both cases want
        inclusion, not exclusion. This never changes self._mode/drives the
        rover by itself; it only ever seeds a passive belief later consumed
        (or not) by _start_mission_subgoal."""
        queries = {
            g.target
            for g in mission.goals[mission.idx :]
            if g.kind in (GoalKind.GO_TO, GoalKind.FIND)
        }
        if self.gen_home_base:
            # RETURN goals are excluded above (their target text varies --
            # "home base"/"spawn point"/whatever phrase matched _RETURN_WORDS
            # -- and driving itself never calls _do_resolve), so without this
            # "home base" would never be checked here even while the rover
            # drives straight at it, leaving its belief tracker to only ever
            # dead-reckon from home_base_seeded's spawn-point seed and never
            # get uncertainty reset by a real sighting. goal_beliefs' key for
            # it is always the literal "home base" string regardless of the
            # RETURN goal's own target text -- see home_base_seeded.
            queries.add("home base")
        if not queries:
            return
        hits = dino_grounding_resolver.detect_in_frame(
            rgb,
            depth,
            list(queries),
            HFOV_DEG,
            dino_model_id=self.dino_model_id,
            dino_device=self.dino_device,
            dino_box_threshold=self.dino_box_threshold,
            dino_text_threshold=self.dino_text_threshold,
        )
        for query, (forward, left, score) in hits.items():
            goal_beliefs.setdefault(
                query, self._new_belief_tracker(mission_sweep=True)
            ).observe_body_point(forward, left)
            # Silent before this: a sweep hit only ever showed up indirectly, as
            # whatever _start_mission_subgoal later did with the belief it seeded
            # -- logged explicitly so a periodic sweep is actually observable/
            # debuggable from the event log instead of inferred after the fact.
            # score is DINO's own box confidence for this query's best box --
            # logged so a confident cross-class mixup (e.g. "green flag"
            # grounding onto the white flag once the real one leaves frame)
            # is distinguishable from a low-confidence noise hit.
            self._event_log.log(
                "mission_sweep_hit",
                name=query,
                forward=round(float(forward), 2),
                left=round(float(left), 2),
                score=round(float(score), 3),
            )

    def _check_turn_interrupt(self, mission, goal_beliefs) -> bool:
        """While a mission TURN sub-goal is spinning, don't finish the spin
        (or any ADVANCE step after it) if the goal it was turning toward is
        already confidently sighted -- called right after _sweep_goal_beliefs
        so it's reacting to the same detection, not an extra DINO call of its
        own. Only ever considers the NEAREST upcoming GO_TO/FIND sub-goal
        (the one this turn exists to face), same as a person stopping mid-
        turn the moment they spot what they were looking for. Jumps the
        mission straight to that sub-goal (skipping any ADVANCE in between --
        no point blindly driving forward first when the target is already in
        view) and lets the caller kick off _start_mission_subgoal, whose own
        GO_TO/FIND handling picks the belief seeded here right back up (fresh
        resolve, or this prior belief as fallback -- see its docstring).
        Returns whether an interrupt happened."""
        upcoming = next(
            (
                g
                for g in mission.goals[mission.idx :]
                if g.kind in (GoalKind.GO_TO, GoalKind.FIND)
            ),
            None,
        )
        if upcoming is None:
            return False
        bt = goal_beliefs.get(upcoming.target)
        if (
            bt is None
            or bt.belief_g is None
            or bt.uncertainty_value() >= self.uncertainty_cov_threshold
        ):
            return False
        with self._lock:
            if self._mission is not mission:
                return False
            while self._mission.current is not upcoming:
                self._mission.advance()
            self._mission_target_yaw = None
        self._event_log.log("turn_interrupted_goal_sighted", target=upcoming.target)
        return True

    def _new_belief_tracker(self, *, mission_sweep: bool = False) -> BeliefGoalTracker:
        # odom_noise/odom_noise_growth_rate drive uncertainty_value()'s real
        # growth while the goal mask is unseen (sam_vla.core.belief_tracking's
        # accelerating-drift formula) -- what the uncertainty halt below
        # compares against uncertainty_cov_threshold. Zero unless the halt is
        # enabled, so a disabled run doesn't silently start dead-reckoning
        # noisier than before.
        #
        # mission_sweep=True is for trackers _sweep_goal_beliefs seeds passively
        # for a goal not currently being driven to -- see mission_belief_cov_growth
        # above for why these need a much slower rate than the actively-driven
        # tracker's halt-tuned one. Callers that promote such a tracker to actively
        # driven must switch it back via _activate_belief_tracker.
        if mission_sweep:
            odom_noise = (
                self.mission_belief_cov_growth if self.uncertainty_enabled else 0.0
            )
            odom_noise_growth_rate = (
                self.mission_belief_cov_growth_rate if self.uncertainty_enabled else 0.0
            )
        else:
            odom_noise = (
                self.uncertainty_cov_growth if self.uncertainty_enabled else 0.0
            )
            odom_noise_growth_rate = (
                self.uncertainty_cov_growth_rate if self.uncertainty_enabled else 0.0
            )
        return BeliefGoalTracker(
            hfov_deg=HFOV_DEG,
            goal_range=self.belief_goal_range,
            min_px=self.lost_goal_min_px,
            odom_noise=odom_noise,
            odom_noise_growth_rate=odom_noise_growth_rate,
        )

    def _activate_belief_tracker(self, bt: BeliefGoalTracker) -> None:
        """Switch a belief tracker to the tight, halt-tuned growth rate at the
        moment it becomes the actively-driven belief_tracker (see
        _start_mission_subgoal's two return points below) -- undoes the loose
        mission_sweep=True rate a passively-seeded tracker may have started
        with, so the uncertainty halt still fires on its normal schedule once
        the rover is actually depending on this belief to drive."""
        bt.odom_noise = self.uncertainty_cov_growth if self.uncertainty_enabled else 0.0
        bt.odom_noise_growth_rate = (
            self.uncertainty_cov_growth_rate if self.uncertainty_enabled else 0.0
        )

    def _new_avoidance(self) -> CbfObstacleAvoidance:
        return CbfObstacleAvoidance(**self._cbf_kwargs)

    def _clear_masks(self, env, goal_objects: list, obstacle_objects: list) -> tuple:
        """Actually remove every registered goal/obstacle mask mesh from the
        scene (not just untag it) and return the emptied (goal, obstacle)
        lists -- see _env_loop's goal_objects/obstacle_objects docstring.
        Also drops any ground-target ghost-mask overlay (self._ground_target_world,
        see _do_resolve) -- every call site here already means "this resolve's
        state is being invalidated", the same boundary the ghost overlay
        should disappear at."""
        self._ground_target_world = None
        for obj in goal_objects + obstacle_objects:
            env.remove_object_mask(obj)
        return [], []

    def _do_resolve(
        self,
        env,
        step,
        mask_dir,
        goal_objects,
        obstacle_objects,
        current_goal_spec,
        target_text: Optional[str] = None,
    ):
        # A new detection task (DINO-grounded or plain SAM2 "Segment")
        # invalidates whatever the previous one left registered -- drop it
        # up front, before running the new detector, rather than waiting for
        # a successful result to overwrite it. This also means a FAILED new
        # resolve (below) now leaves the rover with no masks at all instead
        # of the stale previous target, which used to survive a failed
        # resolve deliberately; that tradeoff was intentionally accepted so
        # "new task requested" always reads as "old detections gone", even
        # when the new one doesn't pan out.
        goal_objects, obstacle_objects = self._clear_masks(
            env, goal_objects, obstacle_objects
        )

        obs_r = env.get_observation(frame_idx=step)
        # detect_rgb (if any) is a separate frame with the mesh_tight_bound2
        # annotation hulls composited in, fed only to the segmentation
        # model -- obs_r.rgb (the plain camera frame) is what the VLM sees
        # and what the GUI/backprojection use, unmodified.
        detect_rgb = env.get_mesh_overlay_rgb() if self.seg_overlay == "mesh" else None
        try:
            if target_text:
                # Open-vocabulary target (flag, blue cuboid, ...):
                # GroundingDINO grounds it directly, SAM2 only seeds
                # obstacles. See dino_grounding_resolver's module docstring
                # for why this is a separate path from first_frame_resolver
                # rather than a branch inside it.
                goal_spec_r, vlm_result, dets = dino_grounding_resolver.resolve_verbose(
                    obs_r.rgb,
                    target_text,
                    detect_rgb=detect_rgb,
                    backend=self.seg_backend,
                    checkpoint_path=self.seg_checkpoint,
                    dino_model_id=self.dino_model_id,
                    dino_device=self.dino_device,
                    dino_box_threshold=self.dino_box_threshold,
                    dino_text_threshold=self.dino_text_threshold,
                    use_obs_detect=self.use_obs_detect,
                )
            else:
                goal_spec_r, vlm_result, dets = first_frame_resolver.resolve_verbose(
                    obs_r.rgb,
                    detect_rgb=detect_rgb,
                    backend=self.seg_backend,
                    checkpoint_path=self.seg_checkpoint,
                )
        except Exception as exc:
            # Console print too, not just the status banner -- a
            # dino_grounding_resolver "not found" RuntimeError carries
            # GroundingDINO's own reasoning (see its message), which was
            # previously only visible by reading the (often truncated)
            # sidebar status text.
            print(f"[nav resolve] failed: {exc}")
            self._event_log.log("goal_detect_failed", name=target_text, reason=str(exc))
            with self._lock:
                self.display.status_text = f"resolve failed: {exc}"
            # goal_objects/obstacle_objects are already emptied (cleared
            # above, before this detector call) -- only goal_spec falls back
            # to whatever was previously resolved, since there's no new one
            # to replace it with.
            return goal_objects, obstacle_objects, current_goal_spec

        # resolve_verbose's raw VLM result/detections were previously discarded
        # here -- surfaced now so a resolve can actually be audited from the
        # terminal (which detection the VLM picked as goal_index, and whether
        # that specific bbox's depth backprojection is what silently dropped
        # the goal mask below, as opposed to the VLM never picking one).
        if target_text:
            # dino_grounding_resolver's result shape (found/u/v/score/reasoning),
            # not select_goal's (goal_index) -- dets here are SAM2's
            # obstacle-only detections (see dino_grounding_resolver), not
            # goal candidates, so labeled accordingly to avoid implying they
            # were candidates DINO picked among.
            print(
                f"[nav resolve] dino ground_object(target={target_text!r}): "
                f"found={vlm_result.get('found')}, "
                f"u={vlm_result.get('u')}, v={vlm_result.get('v')}, "
                f"score={vlm_result.get('score')}, "
                f"reasoning={vlm_result.get('reasoning', '')!r}, "
                f"{len(dets)} SAM2 obstacle detection(s)"
            )
        else:
            print(
                f"[nav resolve] {len(dets)} detection(s), "
                f"goal_index={vlm_result.get('goal_index')}, "
                f"reasoning={vlm_result.get('reasoning', '')!r}"
            )

        goal_position = backproject_goal_position(obs_r, goal_spec_r, hfov_deg=HFOV_DEG)
        goal_bbox_norm = goal_spec_r.goal_bbox_norm
        obstacle_bboxes_norm = list(goal_spec_r.obstacle_bboxes_norm)
        used_fallback = False
        if goal_position is None:
            # bbox_to_world only returns None when literally every pixel in
            # the *padded* interior patch is non-finite or <= 0 -- dump those
            # same patch stats so a "bad depth" print above is diagnosable
            # (void/sky hit vs. NaN vs. a degenerate 0-area bbox) instead of
            # a bare yes/no. pad_px must match bbox_to_world's own default.
            pad_px = 6
            depth = np.asarray(obs_r.depth)
            h, w = depth.shape[:2]
            x0, y0, x1, y1 = goal_bbox_norm
            ix0, ix1 = sorted(
                (min(max(int(x0 * w), 0), w - 1), min(max(int(x1 * w), 0), w - 1))
            )
            iy0, iy1 = sorted(
                (min(max(int(y0 * h), 0), h - 1), min(max(int(y1 * h), 0), h - 1))
            )
            ix0, ix1 = max(ix0 - pad_px, 0), min(ix1 + pad_px, w - 1)
            iy0, iy1 = max(iy0 - pad_px, 0), min(iy1 + pad_px, h - 1)
            patch = depth[iy0 : iy1 + 1, ix0 : ix1 + 1]
            print(
                f"[nav resolve] goal bbox_norm={goal_bbox_norm} -> "
                f"pixels x[{ix0}:{ix1}] y[{iy0}:{iy1}] of {w}x{h}, "
                f"depth patch min={np.nanmin(patch):.4f} max={np.nanmax(patch):.4f} "
                f"nonfinite={int((~np.isfinite(patch)).sum())}/{patch.size} "
                f"zero_or_neg={int((patch <= 0.0).sum())}/{patch.size}"
            )
            # A whole padded neighborhood reading uniformly invalid means
            # this specific detection isn't backed by real geometry at all
            # (see CLAUDE.md's "Annotation masks are thin silhouette
            # slivers" known issue -- these checkpoints predict artifact
            # detections, not just imprecise ones) -- no amount of local
            # padding fixes that. Rather than leaving the episode with
            # obstacles and no goal, fall through the VLM's other ranked
            # detections (in the order it returned them) and promote the
            # first one that actually backprojects. Only if none of them do
            # does this genuinely have no usable goal this resolve.
            for i, candidate_bbox in enumerate(obstacle_bboxes_norm):
                candidate_position = bbox_to_world(
                    obs_r, candidate_bbox, hfov_deg=HFOV_DEG
                )
                if candidate_position is not None:
                    print(
                        f"[nav resolve] falling back to candidate {i} "
                        "(top pick had no valid depth) as goal"
                    )
                    goal_position = candidate_position
                    goal_bbox_norm = candidate_bbox
                    del obstacle_bboxes_norm[i]
                    obstacle_bboxes_norm.append(goal_spec_r.goal_bbox_norm)
                    used_fallback = True
                    break

        # Ghost-mask overlay for a DINO-grounded target (register_object_mask
        # above is a live 3D mesh, but dynamically registered objects render
        # 0px on this GPU/driver -- see CLAUDE.md's known-issues memory --
        # so it alone gives no visual confirmation grounding worked. This is
        # read every frame in _run's render loop and reprojected via
        # project_world_point_with_pixel_radius + ghost_mask.draw_ghost_mask,
        # which (unlike register_object_mask) is a real 2D image overlay and
        # does render. The clear at the top of this function already reset
        # this to None; only set it back when this was actually a
        # DINO-grounded resolve with a valid position -- the classic SAM2
        # auto-resolve path (target_text=None) keeps relying on the
        # mesh-mask overlay as before. Excludes used_fallback: that promotes
        # an *obstacle* rock's position as the goal when the grounded
        # point's own depth was invalid, which would otherwise show the
        # ghost mask over a rock instead of the actual requested target.
        if target_text and goal_position is not None and not used_fallback:
            self._ground_target_world = goal_position
        new_goal_objects: list = []
        new_obstacle_objects: list = []
        goal_mask_warning = ""
        if goal_position is not None:
            new_goal_objects.append(
                env.register_object_mask(
                    goal_position, MESH_GOAL_ID, self.obj_mask_radius, mask_dir, "goal"
                )
            )
            # goal_spec_r.instruction_text still describes the VLM's original
            # top pick (its reasoning references that specific detection) --
            # note the substitution explicitly rather than let the sidebar
            # imply the reasoning still matches whichever bbox actually got
            # registered as the goal.
            instruction = goal_spec_r.instruction_text
            if used_fallback:
                instruction += (
                    " [top pick had no valid depth -- fell back to another "
                    "detection as the actual goal]"
                )
            status_msg = (
                f"reviewing resolved goal: '{instruction}' -- "
                "Confirm to drive, Rerun to retry, or Pick Manually"
            )
            gx, gy, gz = goal_position
            self._event_log.log(
                "goal_detected",
                name=target_text or goal_spec_r.instruction_text,
                method="dino" if target_text else "sam2+qwen",
                world=(round(gx, 2), round(gy, 2), round(gz, 2)),
                fallback=used_fallback,
                score=vlm_result.get("score"),
            )
        else:
            status_msg = (
                "resolved goal has no valid depth -- skipping mask "
                "(Rerun or Pick Manually)"
            )
            self._event_log.log(
                "goal_detect_failed",
                name=target_text or goal_spec_r.instruction_text,
                reason="no valid depth",
            )
            # This is the case the user needs to actually see: the VLM did
            # pick a goal (goal_spec_r.instruction_text above reflects it),
            # but neither its bbox nor any fallback candidate got a mask in
            # the scene, so the rover has only obstacles registered and
            # nothing to drive toward. The per-tick `d.status_text = status`
            # in _env_loop overwrites display.status_text with the generic
            # mode banner on the very next frame, which would otherwise
            # erase this warning before it's ever seen -- click_status
            # isn't touched every tick, so park it there instead.
            goal_mask_warning = (
                "resolve: no detection had valid depth, only obstacle "
                "mask(s) registered -- Rerun or Pick Manually"
            )
        for i, obstacle_bbox in enumerate(obstacle_bboxes_norm):
            obstacle_position = bbox_to_world(obs_r, obstacle_bbox, hfov_deg=HFOV_DEG)
            if obstacle_position is None:
                continue
            new_obstacle_objects.append(
                env.register_object_mask(
                    obstacle_position,
                    MESH_OBST_ID,
                    self.obj_mask_radius,
                    mask_dir,
                    f"obstacle_{i}",
                )
            )
        print(
            f"[nav resolve] goal_mask={'registered' if new_goal_objects else 'MISSING (bad depth)'}, "
            f"obstacle_masks={len(new_obstacle_objects)}/{len(goal_spec_r.obstacle_bboxes_norm)} registered"
        )

        with self._lock:
            self._mode = MODE_REVIEW_SEGMENTATION
            self._world_goal = None
            self.display.status_text = status_msg
            self.display.click_status = goal_mask_warning
            self.display.goal_reached = False
        return new_goal_objects, new_obstacle_objects, goal_spec_r

    @staticmethod
    def _status_text(mode, unresolved, goal_spec, dist) -> str:
        dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
        if mode == MODE_IDLE:
            return "IDLE"
        if mode == MODE_MANUAL:
            return "MANUAL DRIVE"
        if mode == MODE_POINT:
            return f"POINT GOAL  dist={dist_txt}"
        if mode == MODE_REVIEW_SEGMENTATION:
            label = goal_spec.instruction_text if goal_spec is not None else "?"
            return f"REVIEW SEGMENTATION '{label}' -- Confirm / Rerun / Pick Manually"
        if mode == MODE_RESOLVE:
            if unresolved:
                return "RESOLVE  waiting for goal sighting..."
            label = goal_spec.instruction_text if goal_spec is not None else "?"
            return f"RESOLVE '{label}'  dist={dist_txt}"
        if mode == MODE_TURN:
            return "TURN"
        return mode
