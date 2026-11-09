"""Same kb_teleop.py UI (reused via import, not duplicated) with vl_direction
wired in live against a synthetic obstacle field (random ground-plane
"rocks", since there's no real obstacle detector here yet). Each rendered
frame dispatches one of vl_direction's four modes on a background thread
(movement never blocks on it): "cbf" go-around guidance when an obstacle is
close and in view, "exploration" four-way prior otherwise, "uncertainty"
heading-request halt when a synthetic covariance proxy drifts too high
(numpad 1/2/3/4/6/7/8/9 to answer), and "ghost_mask" to
place a translucent ellipse toward a tracked belief anchor once it's out of
frame. An optional real ground-truth goal (--goal-x/--goal-z/--goal-radius)
feeds BeliefGoalTracker directly and, once acquired, permanently silences
the exploration direction prompt (CBF keeps firing) per vl_direction's
"if goal identified -> dormant" rule. Console prints prompt/response/latency
per VL call.

Usage:
    conda activate habitat && python scripts/habitat_tests/kb_teleop_vl.py [--vl-every-n-frames N]
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
import customtkinter as ctk

import quaternion

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kb_teleop as kb
from sam_vla.core.belief_tracking import BeliefGoalTracker, uncertainty_growth_increment
from sam_vla.core.ghost_mask import (
    belief_to_bearing_range_uncertainty,
    draw_ghost_ellipse,
    draw_ghost_mask,
    project_body_point_to_pixel,
    project_or_clamp_body_point_to_pixel,
    uncertainty_to_radius_px,
)
from sam_vla.core.types import Action
from vl_direction import config as vl_dir_config
from vl_direction.client import get_client
from vl_direction.directive_engine import query as vl_query
from vl_direction.qwen_server_manager import QwenServerManager
from vl_direction.schemas import CBFContext, ExplorationContext, GhostMaskContext
from vl_direction.uncertainty_session import UncertaintySession

NUM_OBSTACLES = 6
OBSTACLE_RADIUS_M = 0.5
OBSTACLE_SEED = 7
OBSTACLE_SPAWN_HALF_EXTENT = 18.0  # sampled uniformly in [-extent, extent]^2, comfortably inside kb.BOUNDARY_LIMIT

CBF_DISTANCE_THRESHOLD_M = 3.5
EXPLORATION_TASK_STR = "explore the terrain ahead"

# --- uncertainty halt: synthetic covariance proxy (this script has no real belief
# tracker) -- confidence drifts up while no synthetic obstacle is close enough to act
# as a visual anchor, resets when one is. ---
UNCERTAINTY_GROUNDING_RANGE_M = CBF_DISTANCE_THRESHOLD_M
DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD = vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD
DEFAULT_UNCERTAINTY_GROWTH_PER_STEP = 0.01
DEFAULT_UNCERTAINTY_GROWTH_RATE = 0.0

# --- ghost mask: translucent green circle at the stand-in "lost goal" (nearest
# obstacle's body-frame point, reused here only because this script has no
# real goal-tracking subsystem) whose radius tracks self.uncertainty_covariance,
# same value that drives the uncertainty halt above -- see sam_vla/core/ghost_mask.py
# and next.md's Phase 1. Scale is picked so the ghost saturates to
# OVERLAY_MAX_PIXEL_RADIUS right as uncertainty reaches the halt threshold. ---
GHOST_ALPHA = 0.45
UNCERTAINTY_HEADING_KEYS = {
    "8": 0.0,
    "9": 45.0,
    "6": 90.0,
    "3": 135.0,
    "2": 180.0,
    "1": -135.0,
    "4": -90.0,
    "7": -45.0,
}
# Numpad grid position for each heading key above -- drives the heads-up
# panel's clickable button layout (next.md Goal 2, point 2: "numpad layout
# drawn as actual labeled regions, not just implied by key bindings").
UNCERTAINTY_HEADING_GRID_POS = {
    "7": (0, 0),
    "8": (0, 1),
    "9": (0, 2),
    "4": (1, 0),
    "6": (1, 2),
    "1": (2, 0),
    "2": (2, 1),
    "3": (2, 2),
}

CAMERA_HFOV_DEG = 90.0
_FRAME_H, _FRAME_W = kb.RGBD_RESOLUTION
_FOCAL_PX = (_FRAME_W / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))

OVERLAY_ALPHA = 0.5
OVERLAY_MIN_PIXEL_RADIUS = 3
OVERLAY_MAX_PIXEL_RADIUS = 100

# --- real goal (--goal-x/--goal-z): a terrain patch of radius --goal-radius is
# extracted to its own OBJ (extract_disc_mesh/write_obj_mesh, same tight-clipping
# approach mesh_annotation_tool.py uses for its polygon hulls, just circular) and
# ground-truth-projected in blue whenever it's actually in frame -- distinct from
# the red synthetic obstacles and the green ghost mask. Being in frame is also
# what snaps BeliefGoalTracker onto it (BeliefGoalTracker.observe_body_point);
# once acquired, mode dispatch latches out of "exploration" for good (see
# render()'s exploration_dormant), per vl_direction/DESIGN.md's documented
# caller-side rule: "if goal identified -> dormant". ---
DEFAULT_GOAL_RADIUS_M = 0.5
GOAL_MESH_SPACING_M = 0.05
GOAL_COLOR = (0.0, 0.0, 255.0)

USE_MOCK_CLIENT = False

EPISODE_ID = f"kb-teleop-vl-{int(time.time())}"

# --- VL query cadence: fires on whichever comes first (frame count is
# manually configurable at startup; the seconds trigger is a fixed backstop
# so a stalled frame counter still keeps directives fresh) ---
DEFAULT_VL_QUERY_EVERY_N_FRAMES = 3
VL_QUERY_EVERY_N_SECONDS = 5.0


def make_obstacles(rng, count, half_extent):
    xs = rng.uniform(-half_extent, half_extent, size=count)
    zs = rng.uniform(-half_extent, half_extent, size=count)
    return list(zip(xs.tolist(), zs.tolist()))


def project_point(px_world, py_world, pz_world, rover_x, rover_y, rover_z, yaw):
    """World point -> (pixel_x, pixel_y, depth), or None if behind the camera.
    Matches kb_teleop's movement convention: forward at yaw=0 is -Z, right is
    +X (habitat_sim's camera looks down -Z in its local frame, +Y up)."""
    dx = px_world - rover_x
    dy = py_world - rover_y
    dz = pz_world - rover_z

    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    right_x, right_z = math.cos(yaw), -math.sin(yaw)

    depth = dx * forward_x + dz * forward_z
    if depth <= 1e-3:
        return None

    local_x = dx * right_x + dz * right_z
    local_y = dy

    pixel_x = _FRAME_W / 2.0 + _FOCAL_PX * local_x / depth
    pixel_y = _FRAME_H / 2.0 - _FOCAL_PX * local_y / depth
    return pixel_x, pixel_y, depth


def body_frame_forward_left(px_world, pz_world, rover_x, rover_z, yaw):
    """World-plane point -> body-frame [forward, left], the same convention
    BeliefGoalTracker/ghost_mask use. Ground-plane-only counterpart of
    project_point's forward/right math above (no vertical/pixel component)."""
    dx = px_world - rover_x
    dz = pz_world - rover_z
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    right_x, right_z = math.cos(yaw), -math.sin(yaw)
    forward = dx * forward_x + dz * forward_z
    right = dx * right_x + dz * right_z
    return forward, -right


def overlay_obstacles(rgb, projected_circles, color=(255.0, 0.0, 0.0)):
    """Alpha-blends solid `color` circles onto rgb (uint8 HWC) at each
    (pixel_x, pixel_y, pixel_radius) in projected_circles. Defaults to red
    (synthetic obstacles); the real goal marker reuses this with
    color=GOAL_COLOR (blue) so both share one blend implementation."""
    if not projected_circles:
        return rgb.copy()

    annotated = rgb.astype(np.float32)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for pixel_x, pixel_y, pixel_radius in projected_circles:
        mask |= (xx - pixel_x) ** 2 + (yy - pixel_y) ** 2 <= pixel_radius**2

    color_arr = np.array(color, dtype=np.float32)
    annotated[mask] = (
        annotated[mask] * (1.0 - OVERLAY_ALPHA) + color_arr * OVERLAY_ALPHA
    )
    return annotated.astype(np.uint8)


def extract_disc_mesh(
    center_x, center_z, radius, spacing=GOAL_MESH_SPACING_M, max_grid_res=200
):
    """Terrain patch mesh within `radius` of (center_x, center_z): a regular
    (x, z) grid sampled at `spacing`, each vertex's height read from
    kb.terrain_height_at (bilinear, matches HeightmapGrid's normalize-then-
    subtract-mean convention), keeping only quads whose 4 corners fall inside
    the circle -- the same tight-clipping approach
    mesh_annotation_tool.py's compute_tight_boundary_mesh uses for its
    polygon hulls, just with a circular inside-test instead of a polygon
    one. Returns (verts (N,3), faces (M,3) triangle indices)."""
    n = int(np.clip(round((2.0 * radius) / spacing) + 1, 2, max_grid_res))
    xs = np.linspace(center_x - radius, center_x + radius, n)
    zs = np.linspace(center_z - radius, center_z + radius, n)
    Xg, Zg = np.meshgrid(xs, zs)
    Yg = np.vectorize(kb.terrain_height_at)(Xg, Zg)
    inside = (Xg - center_x) ** 2 + (Zg - center_z) ** 2 <= radius**2

    row, col = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    quad_inside = (
        inside[row, col]
        & inside[row, col + 1]
        & inside[row + 1, col + 1]
        & inside[row + 1, col]
    ).ravel()
    i0 = (row * n + col).ravel()
    i1 = (row * n + col + 1).ravel()
    i2 = ((row + 1) * n + col + 1).ravel()
    i3 = ((row + 1) * n + col).ravel()
    faces = np.concatenate(
        [
            np.stack([i0, i1, i2], axis=-1)[quad_inside],
            np.stack([i0, i2, i3], axis=-1)[quad_inside],
        ],
        axis=0,
    )
    if len(faces) == 0:
        raise ValueError(
            f"--goal-radius {radius} is too small to extract a terrain mesh at "
            f"spacing {spacing} -- increase --goal-radius or decrease the spacing"
        )

    verts_full = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=-1)
    used = np.unique(faces)
    remap = -np.ones(len(verts_full), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts_full[used], remap[faces]


def write_obj_mesh(path, verts, faces, name="goal"):
    """Writes verts/faces (as returned by extract_disc_mesh) to a plain OBJ
    file, same minimal format mesh_annotation_tool.py's write_hull_obj uses
    (1-indexed face vertices, no normals/UVs)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


class VLTeleopApp:
    def __init__(
        self,
        vl_query_every_n_frames=DEFAULT_VL_QUERY_EVERY_N_FRAMES,
        uncertainty_covariance_threshold=DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD,
        uncertainty_growth_per_step=DEFAULT_UNCERTAINTY_GROWTH_PER_STEP,
        uncertainty_growth_rate=DEFAULT_UNCERTAINTY_GROWTH_RATE,
        ghost_mask_use_vlm=True,
        goal_x=None,
        goal_z=None,
        goal_radius=DEFAULT_GOAL_RADIUS_M,
    ):
        self.ghost_mask_use_vlm = ghost_mask_use_vlm
        self.last_ghost_mask_payload = None
        self.last_ghost_mask_line = ""

        self.vl_query_every_n_frames = vl_query_every_n_frames
        self.vl_query_every_n_seconds = VL_QUERY_EVERY_N_SECONDS
        self.last_vl_frame_idx = None
        self.last_vl_time = None
        self.vl_query_in_flight = False
        self.vl_lock = threading.Lock()

        self.uncertainty_covariance_threshold = uncertainty_covariance_threshold
        self.uncertainty_growth_per_step = uncertainty_growth_per_step
        self.uncertainty_growth_rate = uncertainty_growth_rate
        self.uncertainty_covariance = 0.0
        self.uncertainty_time_since_seen = 0.0
        self.halted_for_uncertainty = False
        self.uncertainty_request_in_flight = False
        self.last_uncertainty_line = ""
        self.last_annotated_rgb = None
        # Ghost radius saturates to OVERLAY_MAX_PIXEL_RADIUS right as
        # uncertainty_covariance reaches the halt threshold.
        self.ghost_radius_scale = OVERLAY_MAX_PIXEL_RADIUS / max(
            uncertainty_covariance_threshold, 1e-6
        )

        self.sim = kb.make_sim()
        self.agent = self.sim.initialize_agent(0)

        self.x = kb.START_X
        self.z = kb.START_Z
        self.yaw = np.deg2rad(kb.START_YAW_DEG)
        self.clearance = kb.INITIAL_CLEARANCE

        rng = np.random.default_rng(OBSTACLE_SEED)
        obstacle_xz = make_obstacles(rng, NUM_OBSTACLES, OBSTACLE_SPAWN_HALF_EXTENT)
        self.obstacles = [
            (ox, kb.terrain_height_at(ox, oz) + OBSTACLE_RADIUS_M, oz)
            for ox, oz in obstacle_xz
        ]

        print(
            f"[VLTeleopApp] {NUM_OBSTACLES} synthetic obstacles (r={OBSTACLE_RADIUS_M}m): "
            + ", ".join(f"({o[0]:.1f},{o[2]:.1f})" for o in self.obstacles)
        )

        self.goal_enabled = goal_x is not None
        self.goal_acquired = False
        self.goal_belief = None
        self._goal_belief_prev_pose = None
        if self.goal_enabled:
            self.goal_x = goal_x
            self.goal_z = goal_z
            self.goal_radius = goal_radius

            mesh_verts, mesh_faces = extract_disc_mesh(
                self.goal_x, self.goal_z, self.goal_radius
            )
            self.goal_mesh_path = Path(f"output/meshes/goal_mesh_{EPISODE_ID}.obj")
            write_obj_mesh(self.goal_mesh_path, mesh_verts, mesh_faces, name="goal")

            self.goal_belief = BeliefGoalTracker(
                hfov_deg=CAMERA_HFOV_DEG,
                odom_noise=uncertainty_growth_per_step,
                odom_noise_growth_rate=uncertainty_growth_rate,
            )

            print(
                f"[VLTeleopApp] goal at ({self.goal_x:.2f},{self.goal_z:.2f}) r={self.goal_radius}m -- "
                f"extracted {len(mesh_faces)} tris -> {self.goal_mesh_path}"
            )

        self.server_manager = QwenServerManager()
        if USE_MOCK_CLIENT:
            self.client = get_client("mock")
        else:
            print("[VLTeleopApp] starting Qwen server")
            self.server_manager.start()
            self.client = get_client("qwen")

        print(
            f"[VLTeleopApp] VL query cadence: every {self.vl_query_every_n_frames} frames "
            f"or every {self.vl_query_every_n_seconds:.1f}s, whichever comes first"
        )

        self.uncertainty_session = UncertaintySession(
            episode_id=EPISODE_ID,
            covariance_threshold=self.uncertainty_covariance_threshold,
            covariance_value=self.uncertainty_covariance,
            client=self.client,
        )
        print(
            f"[VLTeleopApp] uncertainty halt: growth={self.uncertainty_growth_per_step}/step, "
            f"threshold={self.uncertainty_covariance_threshold}"
        )

        self.frame_idx = 0
        self.closed = False
        self.last_vl_line = ""

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("kb teleop")

        video_frame = ctk.CTkFrame(self.root)
        video_frame.pack(padx=10, pady=(10, 4))
        # Plain tk.Label (not CTkLabel) for the per-frame video blit -- this
        # updates every render() call and a raw PhotoImage swap is cheaper
        # than rebuilding a CTkImage every frame; it still lives inside a
        # CTkFrame so it reads as part of the same themed window.
        self.image_label = tk.Label(video_frame, bd=0)
        self.image_label.pack()

        mono_font = ctk.CTkFont(family="Consolas", size=12)
        self.status_panel = ctk.CTkFrame(self.root)
        self.status_panel.pack(fill="x", padx=10, pady=4)
        self.status_value_label = ctk.CTkLabel(
            self.status_panel, text="", anchor="w", justify="left", font=mono_font
        )
        self.status_value_label.pack(fill="x", padx=10, pady=(8, 2))
        self.vl_line_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=mono_font,
            text_color="#e8e04a",
        )
        self.vl_line_label.pack(fill="x", padx=10)
        self.ghost_line_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=mono_font,
            text_color="#4ade80",
        )
        self.ghost_line_label.pack(fill="x", padx=10)
        self.goal_status_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=mono_font,
            text_color="#60a5fa",
        )
        if self.goal_enabled:
            self.goal_status_label.pack(fill="x", padx=10, pady=(0, 8))
        else:
            self.status_value_label.pack_configure(pady=(8, 8))

        # Uncertainty-halt heads-up panel (next.md Goal 2, point 2): a bounded
        # box with its own background/border so it's legible over any terrain
        # color, packed only while halted_for_uncertainty -- see
        # _sync_uncertainty_panel. Distinct border/title color for "VLM
        # request in flight" vs. "waiting for your keypress" (today both
        # looked identical, plain cyan text, until the text itself changed).
        self.uncertainty_panel = ctk.CTkFrame(self.root, border_width=2)
        self.uncertainty_title = ctk.CTkLabel(
            self.uncertainty_panel, text="", font=ctk.CTkFont(size=15, weight="bold")
        )
        self.uncertainty_title.pack(pady=(10, 2))
        self.uncertainty_desc = ctk.CTkLabel(
            self.uncertainty_panel,
            text="",
            wraplength=460,
            justify="left",
            font=ctk.CTkFont(size=12),
        )
        self.uncertainty_desc.pack(padx=12, pady=(0, 10), fill="x")

        heading_grid = ctk.CTkFrame(self.uncertainty_panel, fg_color="transparent")
        heading_grid.pack(pady=(0, 6))
        self.uncertainty_buttons = {}
        for key, angle in UNCERTAINTY_HEADING_KEYS.items():
            r, c = UNCERTAINTY_HEADING_GRID_POS[key]
            btn = ctk.CTkButton(
                heading_grid,
                text=f"{key}\n{angle:+.0f}°",
                width=70,
                height=50,
                command=lambda a=angle: self._submit_uncertainty_heading(a),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.uncertainty_buttons[key] = btn

        self.uncertainty_retry_button = ctk.CTkButton(
            self.uncertainty_panel,
            text="Retry (R)",
            fg_color="#b45309",
            hover_color="#92400e",
            command=self._request_uncertainty_retry,
        )
        self.uncertainty_retry_button.pack(pady=(0, 12))
        # Not packed yet -- _sync_uncertainty_panel() packs/unpacks it as
        # halted_for_uncertainty flips. Tracked via a plain flag rather than
        # winfo_ismapped(), which reflects window-manager mapping state and
        # can lag pack()/pack_forget() calls right after window creation.
        self._uncertainty_panel_visible = False

        self.info_label = ctk.CTkLabel(
            self.root,
            text=(
                "W/S move | A/D turn | Q/E height | X quit"
            ),
            font=ctk.CTkFont(size=12),
        )
        self.info_label.pack(pady=(0, 8))

        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.set_agent_pose()
        self.render()

    def set_agent_pose(self):
        self.terrain_y = kb.terrain_height_at(self.x, self.z)
        self.y = self.terrain_y + self.clearance

        state = self.agent.get_state()
        state.position = np.array([self.x, self.y, self.z], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0.0, self.yaw, 0.0])
        self.agent.set_state(state)

    def _project_obstacles(self):
        """Returns (circles_for_overlay, nearest_any, nearest_visible).
        edge_distance is ground-plane distance from the rover to the
        obstacle surface (negative if the rover is already inside it).
        nearest_any drives mode selection; nearest_visible (a subset that
        actually projects into frame) is what CBFContext gets a bbox from."""
        circles = []
        nearest_any = None
        nearest_visible = None

        for ox, oy, oz in self.obstacles:
            edge_distance = math.hypot(ox - self.x, oz - self.z) - OBSTACLE_RADIUS_M
            if nearest_any is None or edge_distance < nearest_any["edge_distance"]:
                forward, left = body_frame_forward_left(
                    ox, oz, self.x, self.z, self.yaw
                )
                nearest_any = {
                    "edge_distance": edge_distance,
                    "forward": forward,
                    "left": left,
                }

            projected = project_point(ox, oy, oz, self.x, self.y, self.z, self.yaw)
            if projected is None:
                continue
            pixel_x, pixel_y, depth = projected
            pixel_radius = float(
                np.clip(
                    _FOCAL_PX * OBSTACLE_RADIUS_M / depth,
                    OVERLAY_MIN_PIXEL_RADIUS,
                    OVERLAY_MAX_PIXEL_RADIUS,
                )
            )

            x1, y1 = pixel_x - pixel_radius, pixel_y - pixel_radius
            x2, y2 = pixel_x + pixel_radius, pixel_y + pixel_radius
            x1c, y1c = max(0.0, x1), max(0.0, y1)
            x2c, y2c = min(float(_FRAME_W), x2), min(float(_FRAME_H), y2)
            if x2c <= x1c or y2c <= y1c:
                continue  # projects fully outside the frame

            circles.append((pixel_x, pixel_y, pixel_radius))
            bbox = (int(x1c), int(y1c), int(math.ceil(x2c)), int(math.ceil(y2c)))
            if (
                nearest_visible is None
                or edge_distance < nearest_visible["edge_distance"]
            ):
                nearest_visible = {"edge_distance": edge_distance, "bbox": bbox}

        return circles, nearest_any, nearest_visible

    def _project_goal(self):
        """Ground-truth projection of the goal disc (top of a
        goal_radius-tall dome, same convention as the obstacle spheres).
        Returns (circle_for_overlay, bbox) or (None, None) if disabled or
        not currently in frame -- bbox is what drives both the blue overlay
        and the belief-snap trigger in _update_goal_belief."""
        if not self.goal_enabled:
            return None, None

        goal_y = kb.terrain_height_at(self.goal_x, self.goal_z) + self.goal_radius
        projected = project_point(
            self.goal_x, goal_y, self.goal_z, self.x, self.y, self.z, self.yaw
        )
        if projected is None:
            return None, None
        pixel_x, pixel_y, depth = projected
        pixel_radius = float(
            np.clip(
                _FOCAL_PX * self.goal_radius / depth,
                OVERLAY_MIN_PIXEL_RADIUS,
                OVERLAY_MAX_PIXEL_RADIUS,
            )
        )

        x1, y1 = pixel_x - pixel_radius, pixel_y - pixel_radius
        x2, y2 = pixel_x + pixel_radius, pixel_y + pixel_radius
        x1c, y1c = max(0.0, x1), max(0.0, y1)
        x2c, y2c = min(float(_FRAME_W), x2), min(float(_FRAME_H), y2)
        if x2c <= x1c or y2c <= y1c:
            return None, None  # projects fully outside the frame

        circle = (pixel_x, pixel_y, pixel_radius)
        bbox = (int(x1c), int(y1c), int(math.ceil(x2c)), int(math.ceil(y2c)))
        return circle, bbox

    def _update_goal_belief(self, goal_visible):
        """Dead-reckons self.goal_belief by whatever motion happened since
        the last render() call (mirrors BeliefGoalTracker.propagate()'s
        real-rollout usage, one step per render() the same way
        _update_uncertainty_covariance treats one render() call as one
        step), then re-seeds it from ground truth whenever the goal disc is
        actually in frame this call -- this script has no semantic renderer
        to source a live mask from, so BeliefGoalTracker.observe_body_point
        (ground-truth body-frame point) stands in for observe(mask, depth).
        Once seeded once, self.goal_acquired latches permanently -- render()
        uses this to stop dispatching "exploration" VL calls for the rest of
        the run (vl_direction/DESIGN.md's documented caller-side rule: "if
        goal identified -> dormant")."""
        if not self.goal_enabled:
            return

        if self._goal_belief_prev_pose is not None:
            old_x, old_z, old_yaw = self._goal_belief_prev_pose
            forward_x, forward_z = -math.sin(old_yaw), -math.cos(old_yaw)
            right_x, right_z = math.cos(old_yaw), -math.sin(old_yaw)
            dx, dz = self.x - old_x, self.z - old_z
            forward_disp = dx * forward_x + dz * forward_z
            right_disp = dx * right_x + dz * right_z
            yaw_delta = self.yaw - old_yaw
            action = Action(v_fwd=forward_disp, v_lat=right_disp, yaw_rate=yaw_delta)
            self.goal_belief.propagate(action, dt=1.0)
        self._goal_belief_prev_pose = (self.x, self.z, self.yaw)

        if goal_visible:
            forward, left = body_frame_forward_left(
                self.goal_x, self.goal_z, self.x, self.z, self.yaw
            )
            self.goal_belief.observe_body_point(forward, left)
            self.goal_acquired = True

    def _ghost_belief_anchor(self, nearest_any, goal_out_of_frame):
        """Returns (forward, left, uncertainty) to visualize as a ghost mask,
        or None if there's nothing to show right now. When a real goal is
        configured (--goal-x/--goal-z), self.goal_belief takes over as the
        anchor -- and keeps doing so *past* goal_acquired, since
        BeliefGoalTracker dead-reckons it every frame (_update_goal_belief):
        the whole point of tracking belief instead of a one-shot fact is
        that the ghost can keep pointing back toward the goal if it drops
        out of frame again after being found once. No ghost while the goal
        is plainly visible in the current frame -- no need to guess where
        something you can already see is. Falls back to the legacy stand-in
        (nearest synthetic obstacle + the synthetic uncertainty proxy) when
        no real goal is configured, unchanged from the original demo
        behavior."""
        if self.goal_enabled:
            if self.goal_belief.belief_g is None or not goal_out_of_frame:
                return None
            return (
                float(self.goal_belief.belief_g[0]),
                float(self.goal_belief.belief_g[1]),
                self.goal_belief.uncertainty_value(),
            )
        if nearest_any is None:
            return None
        return nearest_any["forward"], nearest_any["left"], self.uncertainty_covariance

    def _should_query_vl(self):
        """Gate on whichever cadence trigger fires first: every N frames
        (manually configured, default 3) or every VL_QUERY_EVERY_N_SECONDS
        wall-clock seconds since the last dispatched call. Always true on
        the first frame. Skips while a previous call is still in flight so
        slow inference can't pile up overlapping requests -- the render
        loop itself is never gated on this, only whether a *new* dispatch
        happens."""
        if self.vl_query_in_flight:
            return False
        if self.last_vl_frame_idx is None:
            return True
        if self.frame_idx - self.last_vl_frame_idx >= self.vl_query_every_n_frames:
            return True
        if time.time() - self.last_vl_time >= self.vl_query_every_n_seconds:
            return True
        return False

    def _dispatch_vl_query(
        self, annotated_rgb, ghost_frame, nearest_any, nearest_visible, ghost_anchor
    ):
        """Builds the VL request and hands the actual model call to a
        background thread so render()/movement never blocks on inference
        latency -- the sim keeps stepping on whatever self.last_vl_line
        already holds (possibly still "", if no result has landed yet).
        ghost_frame is the obstacle-only frame (no ghost mask baked in yet)
        -- handed separately to the ghost_mask query so the model isn't
        shown a circle it's being asked to place. ghost_anchor (from
        _ghost_belief_anchor, computed once in render() so both this
        dispatch and the on-screen fallback drawing agree) is threaded
        through to _maybe_query_ghost_mask below -- fires independently of
        mode, except mid-CBF, so it keeps guiding back to a since-acquired
        goal that's dropped out of frame again instead of going silent once
        self.goal_acquired latches."""
        mode = (
            "cbf"
            if nearest_any is not None
            and nearest_any["edge_distance"] <= CBF_DISTANCE_THRESHOLD_M
            else ("dormant" if self.goal_acquired else "exploration")
        )
        fallback_note = ""
        context = None

        if mode == "cbf" and nearest_visible is not None:
            context = CBFContext(
                bbox_xyxy=nearest_visible["bbox"], frame_wh=(_FRAME_W, _FRAME_H)
            )
        elif mode == "cbf":
            fallback_note = " (nearest obstacle not in view, fell back to exploration)"
            mode = "dormant" if self.goal_acquired else "exploration"

        if mode == "exploration":
            hint = (
                f"nearest obstacle is {nearest_any['edge_distance']:.1f}m away"
                if nearest_any is not None
                else None
            )
            context = ExplorationContext(task_str=EXPLORATION_TASK_STR, vague_hint=hint)
        # mode == "dormant": no direction context needed -- _vl_worker skips
        # the vl_query() call entirely for this mode (it isn't a real
        # vl_direction Mode), but ghost_mask dispatch below still runs.

        frame_idx_snapshot = self.frame_idx
        self.last_vl_frame_idx = self.frame_idx
        self.last_vl_time = time.time()
        self.vl_query_in_flight = True

        thread = threading.Thread(
            target=self._vl_worker,
            args=(
                mode,
                annotated_rgb,
                ghost_frame,
                context,
                fallback_note,
                frame_idx_snapshot,
                ghost_anchor,
            ),
            daemon=True,
        )
        thread.start()

    def _vl_worker(
        self,
        mode,
        annotated_rgb,
        ghost_frame,
        context,
        fallback_note,
        frame_idx_snapshot,
        ghost_anchor,
    ):
        """Runs on a background thread -- the actual (slow) model call.
        Never touches Tkinter directly; schedules the UI update back onto
        the main loop via root.after so this stays thread-safe. mode
        "dormant" (goal already acquired, no obstacle nearby) skips the
        direction call entirely -- "dormant" isn't a real vl_direction Mode,
        it's this file's own marker for "nothing to ask the direction
        prompt" -- but still falls through to the ghost-mask dispatch."""
        if mode == "dormant":
            line = f"[{frame_idx_snapshot:05d}] dormant -> goal already acquired, exploration suppressed"
        else:
            try:
                result = vl_query(
                    mode, [annotated_rgb], context, EPISODE_ID, client=self.client
                )
            except Exception as e:
                line = f"[{frame_idx_snapshot:05d}] {mode} -> ERROR -> n/a -> {e}"
            else:
                direction_str = (
                    result.direction.value if result.direction is not None else "NONE"
                )
                line = (
                    f"[{frame_idx_snapshot:05d}] {mode} -> {direction_str} -> "
                    f"{result.latency_ms:.1f}ms -> {result.raw_response!r}{fallback_note}"
                )

        print(line)

        ghost_payload, ghost_line = None, None
        if mode != "cbf" and ghost_anchor is not None:
            ghost_payload, ghost_line = self._maybe_query_ghost_mask(
                ghost_frame, ghost_anchor, frame_idx_snapshot
            )
            if ghost_line:
                print(ghost_line)

        with self.vl_lock:
            self.vl_query_in_flight = False

        if not self.closed:
            try:
                self.root.after(0, self._apply_vl_line, line, ghost_payload, ghost_line)
            except Exception:
                pass  # window may have been torn down between the check and this call

    def _maybe_query_ghost_mask(self, ghost_frame, ghost_anchor, frame_idx_snapshot):
        """Second VLM call, gated on --ghost-mask-vlm (checked by the
        caller alongside mode != "cbf" -- a lost-goal cue has no meaning
        mid-CBF-avoidance, see next.md's cbf/exploration binary) and on
        ghost_anchor being available at all (from _ghost_belief_anchor: the
        real goal's belief once it's ever been seen and is currently out of
        frame, or the legacy nearest-obstacle stand-in with no real goal
        configured). Belief is handed to the model as text
        (GhostMaskContext); the model answers with pixel placement instead
        of sam_vla.core.ghost_mask's trigonometry deciding it. Runs on the
        same background thread as the direction call above, so this only
        ever blocks the (already async) worker, never render()."""
        if not self.ghost_mask_use_vlm:
            return None, None

        anchor_forward, anchor_left, anchor_uncertainty = ghost_anchor
        # The model gets mu (body-frame mean) + Sigma (2x2 covariance), never
        # raw world x/z -- a Cartesian coordinate pair means nothing to a
        # VLM's spatial reasoning, whereas bearing/distance do. Neither
        # BeliefGoalTracker nor the legacy nearest-obstacle stand-in tracks a
        # real (possibly anisotropic) covariance today, only a scalar
        # uncertainty, so Sigma is built isotropic from it here
        # (sigma^2 * I) -- this is the single seam a real anisotropic
        # covariance (e.g. from navdp's SubgoalBeliefBank) would plug into
        # later without touching anything downstream of mu/Sigma.
        mu = np.array([anchor_forward, anchor_left], dtype=np.float64)
        sigma = (anchor_uncertainty**2) * np.eye(2, dtype=np.float64)
        bearing_deg, distance_m, bearing_uncertainty_deg, distance_uncertainty_m = (
            belief_to_bearing_range_uncertainty(mu, sigma)
        )
        ghost_context = GhostMaskContext(
            bearing_deg=bearing_deg,
            distance_m=distance_m,
            bearing_uncertainty_deg=bearing_uncertainty_deg,
            distance_uncertainty_m=distance_uncertainty_m,
            frame_wh=(_FRAME_W, _FRAME_H),
            min_radius_px=float(OVERLAY_MIN_PIXEL_RADIUS),
            max_radius_px=float(OVERLAY_MAX_PIXEL_RADIUS),
        )
        # "kind" here is the same VLM-ellipse-vs-fallback-circle split render()
        # makes when drawing (see the ghost_anchor block below) -- a stale
        # self.last_ghost_mask_payload from an earlier successful call still
        # draws as an ellipse even on a failed/errored extraction this frame,
        # so it's distinguished from "fallback-default" (the deterministic
        # project_body_point_to_pixel/uncertainty_to_radius_px circle, drawn
        # only once no prior VLM placement exists at all).
        stale_kind = (
            "ghost-mask-ellipse(stale)"
            if self.last_ghost_mask_payload is not None
            else "fallback-default"
        )

        try:
            ghost_result = vl_query(
                "ghost_mask",
                [ghost_frame],
                ghost_context,
                EPISODE_ID,
                client=self.client,
            )
        except Exception as e:
            return None, (
                f"[{frame_idx_snapshot:05d}] ghost_mask -> ERROR -> kind={stale_kind} "
                f"belief={ghost_context!r} -> {e}"
            )

        if not ghost_result.parse_ok:
            return None, (
                f"[{frame_idx_snapshot:05d}] ghost_mask -> PARSE FAILED, "
                f"keeping last placement -> kind={stale_kind} "
                f"belief={ghost_context!r} raw={ghost_result.raw_response!r}"
            )

        payload = ghost_result.ghost_mask_payload
        line = (
            f"[{frame_idx_snapshot:05d}] ghost_mask -> "
            f"u={payload.u:.0f} v={payload.v:.0f} "
            f"ru={payload.radius_u_px:.0f} rv={payload.radius_v_px:.0f} -> "
            f"kind=vlm-inferred belief={ghost_context!r} "
            f"{ghost_result.latency_ms:.1f}ms raw={ghost_result.raw_response!r}"
        )
        return payload, line

    def _apply_vl_line(self, line, ghost_payload=None, ghost_line=None):
        """Runs on the main/Tkinter thread via root.after. Only updates the
        cached line(s); does not force a redraw so a slow VL result never
        stalls movement -- it just appears baked into the next
        keypress-triggered render(). A failed ghost-mask parse leaves
        self.last_ghost_mask_payload untouched (keeps the last good VLM
        placement rather than snapping back to the deterministic fallback)."""
        self.last_vl_line = line
        if ghost_payload is not None:
            self.last_ghost_mask_payload = ghost_payload
        if ghost_line is not None:
            self.last_ghost_mask_line = ghost_line

    def _update_uncertainty_covariance(self, nearest_any):
        """Synthetic proxy for a real covariance signal -- this script has no
        belief tracker to source one from. Drifts up while no obstacle is
        close enough to act as a visual anchor, resets when one is, using the
        same accelerating-drift formula as BeliefGoalTracker.propagate()
        (sam_vla.core.belief_tracking.uncertainty_growth_increment) so this
        demo and the real Phase 2 wiring exercise one formula, not two
        divergent ones. Frozen while already halted so it can't re-trigger
        mid-halt."""
        if self.halted_for_uncertainty:
            return
        grounded = (
            nearest_any is not None
            and nearest_any["edge_distance"] <= UNCERTAINTY_GROUNDING_RANGE_M
        )
        if grounded:
            self.uncertainty_covariance = 0.0
            self.uncertainty_time_since_seen = 0.0
        else:
            self.uncertainty_time_since_seen += 1.0  # one render() call per step
            self.uncertainty_covariance += uncertainty_growth_increment(
                self.uncertainty_growth_per_step,
                self.uncertainty_growth_rate,
                self.uncertainty_time_since_seen,
            )

    def _dispatch_uncertainty_request(self, retry):
        """Enters (or re-enters, on retry) the uncertainty halt: dispatches
        the sweep-description call to a background thread, same pattern as
        _dispatch_vl_query, so it doesn't block rendering/input handling."""
        self.halted_for_uncertainty = True
        self.uncertainty_request_in_flight = True
        frame = self.last_annotated_rgb
        target = (
            self.uncertainty_session.retry
            if retry
            else self.uncertainty_session.request_human_heading
        )

        thread = threading.Thread(
            target=self._uncertainty_worker, args=(target, frame), daemon=True
        )
        thread.start()

    def _uncertainty_worker(self, session_method, frame):
        """Runs on a background thread -- calls UncertaintySession's request
        or retry method (a real Qwen sweep-description call). Never
        touches Tkinter directly; schedules the UI update via root.after."""
        try:
            result = session_method(frame)
        except Exception as e:
            line = f"[uncertainty] ERROR -> {e}"
        else:
            payload = result.uncertainty_payload
            line = (
                f"[uncertainty] attempt={payload.attempt} -> {payload.status.value} -> "
                f"sweep: {result.raw_response!r} -- "
                f"press 1/2/3/4/6/7/8/9 for heading, R to retry"
            )

        print(line)

        with self.vl_lock:
            self.uncertainty_request_in_flight = False

        if not self.closed:
            try:
                self.root.after(0, self._apply_uncertainty_line, line)
            except Exception:
                pass  # window may have been torn down between the check and this call

    def _apply_uncertainty_line(self, line):
        """Runs on the main/Tkinter thread via root.after. Re-renders (unlike
        _apply_vl_line) so the halted-state prompt on screen picks up the
        freshly landed sweep description right away."""
        self.last_uncertainty_line = line
        self.render()

    def _handle_halted_key(self, key):
        """Only reachable while self.halted_for_uncertainty is True -- this
        is the actual movement halt, called from on_key instead of the usual
        WASD handling. Ignores everything but retry/heading-submit while a
        sweep-description call is in flight, to avoid double-dispatching."""
        if self.uncertainty_request_in_flight:
            return

        if key == "r":
            self._request_uncertainty_retry()
            return

        angle = UNCERTAINTY_HEADING_KEYS.get(key)
        if angle is None:
            return

        self._submit_uncertainty_heading(angle)

    def _request_uncertainty_retry(self):
        """Shared by the 'R' key and the heads-up panel's Retry button."""
        if self.uncertainty_request_in_flight:
            return
        self._dispatch_uncertainty_request(retry=True)
        self._sync_uncertainty_panel()

    def _submit_uncertainty_heading(self, angle):
        """Shared by the numpad keys and the heads-up panel's heading
        buttons -- resolves the halt with the given rover-front-relative
        angle_deg."""
        if self.uncertainty_request_in_flight:
            return

        result = self.uncertainty_session.submit_heading(angle_deg=angle)
        payload = result.uncertainty_payload
        line = (
            f"[uncertainty] submit_heading({angle:.0f}deg) -> {payload.status.value} -> "
            f"max_units={payload.max_units:.1f}"
        )
        print(line)
        self.last_uncertainty_line = line
        self.uncertainty_covariance = 0.0
        self.uncertainty_time_since_seen = 0.0
        self.halted_for_uncertainty = False
        self.render()

    def _sync_uncertainty_panel(self):
        """Shows/hides the heads-up panel and updates its in-flight vs.
        waiting-for-input visual state. Called every render() plus right
        after dispatching a retry (button click doesn't otherwise trigger a
        render until the VLM result lands)."""
        if not self.halted_for_uncertainty:
            if self._uncertainty_panel_visible:
                self.uncertainty_panel.pack_forget()
                self._uncertainty_panel_visible = False
            return

        if not self._uncertainty_panel_visible:
            self.uncertainty_panel.pack(fill="x", padx=10, pady=(0, 8))
            self._uncertainty_panel_visible = True

        in_flight = self.uncertainty_request_in_flight
        button_state = "disabled" if in_flight else "normal"
        for btn in self.uncertainty_buttons.values():
            btn.configure(state=button_state)
        self.uncertainty_retry_button.configure(state=button_state)

        if in_flight:
            self.uncertainty_panel.configure(border_color="#f59e0b")
            self.uncertainty_title.configure(
                text="REQUESTING VLM SWEEP...", text_color="#f59e0b"
            )
        else:
            self.uncertainty_panel.configure(border_color="#38bdf8")
            self.uncertainty_title.configure(
                text="HALTED -- choose a heading", text_color="#38bdf8"
            )
        self.uncertainty_desc.configure(
            text=self.last_uncertainty_line or "Waiting for VLM sweep description..."
        )

    def render(self):
        obs = self.sim.get_sensor_observations()
        self.latest_obs = obs

        rgb, _, depth_rgb = kb.rgb_depth_from_obs(obs)

        circles, nearest_any, nearest_visible = self._project_obstacles()
        obstacle_rgb = overlay_obstacles(rgb, circles)

        # Real goal (--goal-x/--goal-z): ground-truth blue circle, drawn only
        # while the extracted goal disc actually projects into frame -- same
        # "in frame" bbox test also drives the belief snap in
        # _update_goal_belief, so the marker and the snap always agree. Baked
        # into obstacle_rgb (before ghost-mask logic below) so it's visible
        # in both the on-screen frame and whatever's sent to the VLM.
        goal_circle, goal_bbox = self._project_goal()
        self._update_goal_belief(goal_visible=goal_bbox is not None)
        if goal_circle is not None:
            obstacle_rgb = overlay_obstacles(
                obstacle_rgb, [goal_circle], color=GOAL_COLOR
            )

        # Real goal / legacy-obstacle CBF proximity + ghost anchor, computed
        # once and shared by both the on-screen ghost draw below and the VL
        # dispatch further down so they always agree.
        near_obstacle = (
            nearest_any is not None
            and nearest_any["edge_distance"] <= CBF_DISTANCE_THRESHOLD_M
        )
        goal_out_of_frame = self.goal_enabled and goal_bbox is None
        ghost_anchor = self._ghost_belief_anchor(nearest_any, goal_out_of_frame)

        # Ghost mask: translucent green circle marking the believed goal
        # location while it's out of frame. With a real goal configured
        # (--goal-x/--goal-z), the anchor is self.goal_belief -- which stays
        # usable *after* self.goal_acquired latches, since BeliefGoalTracker
        # dead-reckons it every frame (_update_goal_belief), so the ghost
        # keeps pointing back toward the goal if it drops out of view again
        # instead of going silent once found. Falls back to the legacy
        # nearest-synthetic-obstacle stand-in when no real goal is
        # configured (see _ghost_belief_anchor). When --ghost-mask-vlm is on
        # (default) and a VLM placement has landed, u/v/radius come from
        # self.last_ghost_mask_payload -- the model's own call, given the
        # belief as text (see _maybe_query_ghost_mask) -- and stay fixed
        # on-screen between VL query cycles rather than tracking rover
        # motion, unlike the deterministic path below. Falls back to the old
        # project_body_point_to_pixel/uncertainty_to_radius_px geometry
        # (sam_vla/core/ghost_mask.py) whenever no VLM placement exists yet,
        # parsing keeps failing, or --no-ghost-mask-vlm was passed -- that
        # deterministic path recomputes u/v from the current belief every
        # single frame (no holding between VL cycles), and when the belief
        # isn't directly projectable (behind the rover, or beyond the HFOV),
        # clamps to the frame edge it's headed toward via
        # project_or_clamp_body_point_to_pixel instead of disappearing: a
        # side edge if it's closer to one side, the bottom edge if it's
        # dead behind / effectively underneath the rover. Purely
        # visual/advisory either way: only ever touches this VLM/display copy
        # of the frame, never anything fed to a policy (there is none here).
        annotated_rgb = obstacle_rgb
        if ghost_anchor is not None:
            anchor_forward, anchor_left, anchor_uncertainty = ghost_anchor
            if self.ghost_mask_use_vlm and self.last_ghost_mask_payload is not None:
                ghost_u = self.last_ghost_mask_payload.u
                ghost_v = self.last_ghost_mask_payload.v
                ghost_radius_u = self.last_ghost_mask_payload.radius_u_px
                ghost_radius_v = self.last_ghost_mask_payload.radius_v_px
                annotated_rgb = draw_ghost_ellipse(
                    obstacle_rgb,
                    ghost_u,
                    ghost_v,
                    ghost_radius_u,
                    ghost_radius_v,
                    alpha=GHOST_ALPHA,
                )
            else:
                ghost_px = project_body_point_to_pixel(
                    anchor_forward,
                    anchor_left,
                    CAMERA_HFOV_DEG,
                    _FRAME_H,
                    _FRAME_W,
                )
                if ghost_px is None:
                    # Out of view this frame (behind the rover, or beyond the
                    # HFOV): rather than drop the ghost, clamp it to the
                    # frame edge it's headed toward -- side edge if it's
                    # closer to one side, bottom edge if it's dead behind /
                    # right underneath -- so it keeps tracking the belief
                    # every frame instead of vanishing.
                    ghost_px = project_or_clamp_body_point_to_pixel(
                        anchor_forward,
                        anchor_left,
                        CAMERA_HFOV_DEG,
                        _FRAME_H,
                        _FRAME_W,
                    )
                ghost_u, ghost_v = ghost_px
                ghost_radius = uncertainty_to_radius_px(
                    anchor_uncertainty,
                    OVERLAY_MIN_PIXEL_RADIUS,
                    OVERLAY_MAX_PIXEL_RADIUS,
                    self.ghost_radius_scale,
                )
                annotated_rgb = draw_ghost_mask(
                    obstacle_rgb, ghost_u, ghost_v, ghost_radius, alpha=GHOST_ALPHA
                )

        self.last_annotated_rgb = annotated_rgb

        self._update_uncertainty_covariance(nearest_any)
        if (
            not self.halted_for_uncertainty
            and not self.uncertainty_request_in_flight
            and self.uncertainty_covariance >= self.uncertainty_covariance_threshold
        ):
            self._dispatch_uncertainty_request(retry=False)

        # Dispatch is always attempted (subject to cadence/halt gating) even
        # once self.goal_acquired latches -- _dispatch_vl_query internally
        # turns the direction call into a no-op "dormant" mode in that case
        # (vl_direction/DESIGN.md's "if goal identified -> dormant"), but
        # still carries ghost_anchor through so the ghost-mask call keeps
        # firing. Skipping dispatch entirely here would also skip
        # ghost_mask, which must not go quiet just because the goal was
        # found once -- it's what points back to it if lost again.
        if not self.halted_for_uncertainty and self._should_query_vl():
            self._dispatch_vl_query(
                annotated_rgb, obstacle_rgb, nearest_any, nearest_visible, ghost_anchor
            )
        self.frame_idx += 1

        if kb.SHOW_DEPTH_BESIDE_RGB:
            img_arr = np.hstack([annotated_rgb, depth_rgb])
        else:
            img_arr = annotated_rgb

        img = Image.fromarray(img_arr)

        # Status/telemetry text used to be baked into the frame with
        # ImageDraw; now rendered as real CTkLabel widgets in self.status_panel
        # (next.md Goal 2 UI pass) so the video stays a clean image and text
        # is crisp/selectable regardless of terrain color underneath.
        status = f"x={self.x:.2f} y={self.y:.2f} z={self.z:.2f} yaw={np.rad2deg(self.yaw):.1f} clearance={self.clearance:.2f} cov={self.uncertainty_covariance:.2f}"
        if self.halted_for_uncertainty:
            status += "  [HALTED: awaiting heading]"
        self.status_value_label.configure(text=status)
        self.vl_line_label.configure(text=self.last_vl_line)
        self.ghost_line_label.configure(text=self.last_ghost_mask_line)

        if self.goal_enabled:
            bearing = self.goal_belief.bearing()
            distance = self.goal_belief.distance()
            if bearing is None:
                goal_status = f"goal: searching ({self.goal_x:.1f},{self.goal_z:.1f}) r={self.goal_radius}m"
            else:
                goal_status = (
                    f"goal: {'ACQUIRED' if self.goal_acquired else 'searching'} "
                    f"bearing={math.degrees(bearing):.1f}deg dist={distance:.2f}m "
                    f"unc={self.goal_belief.uncertainty_value():.2f}"
                )
            self.goal_status_label.configure(text=goal_status)

        self._sync_uncertainty_panel()

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def on_key(self, event):
        key = event.keysym.lower()

        if key == "x" or key == "escape":
            self.close()
            return

        if self.halted_for_uncertainty:
            self._handle_halted_key(key)
            return

        old_x = self.x
        old_z = self.z

        if key == "w":
            self.x += -np.sin(self.yaw) * kb.MOVE_STEP
            self.z += -np.cos(self.yaw) * kb.MOVE_STEP
        elif key == "s":
            self.x -= -np.sin(self.yaw) * kb.MOVE_STEP
            self.z -= -np.cos(self.yaw) * kb.MOVE_STEP
        elif key == "a":
            self.yaw += kb.TURN_STEP
        elif key == "d":
            self.yaw -= kb.TURN_STEP
        elif key == "q":
            self.clearance = max(kb.MIN_CLEARANCE, self.clearance - kb.CLEARANCE_STEP)
        elif key == "e":
            self.clearance = min(kb.MAX_CLEARANCE, self.clearance + kb.CLEARANCE_STEP)
        else:
            return

        self.x, self.z = kb.apply_boundary(self.x, self.z, old_x, old_z)

        self.set_agent_pose()
        self.render()

    def close(self):
        if self.closed:
            return
        self.closed = True

        try:
            self.sim.close()
        except Exception:
            pass
        try:
            self.server_manager.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

        print("Done.")

    def run(self):
        self.root.mainloop()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwery-n",
        type=int,
        default=DEFAULT_VL_QUERY_EVERY_N_FRAMES,
        help=f"query VL every N frames (default {DEFAULT_VL_QUERY_EVERY_N_FRAMES}); "
        f"also fires every {VL_QUERY_EVERY_N_SECONDS:.1f}s regardless, whichever comes first",
    )
    parser.add_argument(
        "--cov-thresh",
        type=float,
        default=DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD,
        help="synthetic covariance value at which the uncertainty halt fires "
        f"(default {DEFAULT_UNCERTAINTY_COVARIANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--cov-growth",
        type=float,
        default=DEFAULT_UNCERTAINTY_GROWTH_PER_STEP,
        help="base per-frame covariance growth while no synthetic obstacle is nearby "
        f"(default {DEFAULT_UNCERTAINTY_GROWTH_PER_STEP}); lower this to trigger the "
        "halt sooner for testing",
    )
    parser.add_argument(
        "--cov-growth-rate",
        type=float,
        default=DEFAULT_UNCERTAINTY_GROWTH_RATE,
        help="accelerating-drift factor: growth speeds up the longer the "
        f"target has been ungrounded (default {DEFAULT_UNCERTAINTY_GROWTH_RATE}, "
        "i.e. flat per-frame growth unless overridden); also drives the "
        "ghost-mask radius alongside --cov-growth",
    )
    parser.add_argument(
        "--ghost-mask-vlm",
        dest="ghost_mask_vlm",
        action="store_true",
        default=True,
        help="have the VLM place the ghost mask itself, given the belief "
        "(mu/Sigma, resolved to bearing/distance/uncertainty text) -- default on",
    )
    parser.add_argument(
        "--no-ghost-mask-vlm",
        dest="ghost_mask_vlm",
        action="store_false",
        help="revert to the deterministic bearing/uncertainty projection "
        "(sam_vla/core/ghost_mask.py), skipping the extra VLM call entirely",
    )
    parser.add_argument(
        "--goal-x",
        type=float,
        default=9,
        help="world x of a real goal point to track (must be given with --goal-z); "
        "extracts a terrain-patch OBJ around it, draws it in blue whenever it's in "
        "frame, and snaps belief to it on first sighting, latching VL exploration "
        "calls off for the rest of the run",
    )
    parser.add_argument(
        "--goal-z",
        type=float,
        default=10,
        help="world z of a real goal point to track (must be given with --goal-x)",
    )
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=DEFAULT_GOAL_RADIUS_M,
        help=f"radius (m) of the terrain patch extracted/marked around the goal "
        f"(default {DEFAULT_GOAL_RADIUS_M})",
    )
    args = parser.parse_args()
    if args.qwery_n < 1:
        parser.error("--qwery-n must be >= 1")
    if (args.goal_x is None) != (args.goal_z is None):
        parser.error("--goal-x and --goal-z must be given together")
    if args.goal_radius <= 0:
        parser.error("--goal-radius must be > 0")
    return args


if __name__ == "__main__":
    args = _parse_args()
    app = VLTeleopApp(
        vl_query_every_n_frames=args.qwery_n,
        uncertainty_covariance_threshold=args.cov_thresh,
        uncertainty_growth_per_step=args.cov_growth,
        uncertainty_growth_rate=args.cov_growth_rate,
        ghost_mask_use_vlm=args.ghost_mask_vlm,
        goal_x=args.goal_x,
        goal_z=args.goal_z,
        goal_radius=args.goal_radius,
    )
    app.run()
