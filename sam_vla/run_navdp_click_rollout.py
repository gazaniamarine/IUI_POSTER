"""Interactive click-to-goal NavDP rollout, built from scratch (not layered on
run_navdp_rollout.py's VLM/belief-tracker/CBF machinery, which next.md's
Integration project flagged as broken for the navdp-upstream backend).

Flow: load the sim, show the live RGB feed in a Tkinter window, let the user
click a pixel on the image, deproject that pixel through the current frame's
depth map into a world-frame point (goal_geometry.bbox_to_world), then
register a small (~0.5m radius) terrain-following, semantically-tagged disc
mesh there (MarsHabitatEnv.register_object_mask -- the same primitive
run_navdp_rollout.py uses to mark a VLM-detected goal, lifted ~5cm off the
terrain so it doesn't z-fight and disappear). That mesh is a real object in
the scene: it renders into the live RGB feed, and the semantic sensor tags it
MESH_GOAL_ID.

Every step, the goal's body-frame (forward, left) point is read directly off
that mesh's *live rendered mask* (core.belief_tracking.BeliefGoalTracker /
mask_to_body -- bearing from the mask centroid column, range from the
median depth over the mask), not re-derived by subtracting stored world
coordinates from the robot's own (possibly drifting) pose estimate. Driving
is gated on visibility: the rover only drives while the mesh is actually
visible in-frame this step; the moment it drops out of view the rover holds
and waits, exactly like run_navdp_rollout.py's "navdp_upstream_goal_unresolved"
branch, rather than following a silently-extrapolated guess.

On "is the goal point supposed to be fed fresh every frame, or held fixed?":
verified against the real vendored server (baselines/navdp/navdp_server.py +
policy_agent.py, InternRobotics/NavDP@master). /pointgoal_step is stateless
w.r.t. the goal -- NavDP_Agent.step_pointgoal takes whatever goal_x/goal_y
came in *this* request and does not remember it between calls (only the RGB
memory_queue persists across calls, for temporal context). So yes: any
request that *is* sent must carry a goal expressed in the egocentric frame
at that exact moment, recomputed from the current pose -- that's not
optional, it's how the endpoint is defined. What's *not* required is
sending a request every simulation tick: the response is a predict_size=24
waypoint chunk (a whole short plan, cumulative body-frame-at-request-time
positions -- see navdp_upstream_client.py's docstring), meant to be walked
open-loop for several steps before replanning. That cadence is already
NavdpUpstreamPolicy's job (--replan-every/--lookahead): calling
set_goal_body() every tick here only refreshes cheap local state so that
whichever tick actually triggers a replan sends a correct, current point;
it does not by itself cause an HTTP call.

KNOWN BLOCKER (2026-08-07): on this machine, dynamically-added habitat_sim
rigid objects -- which is what register_object_mask uses to place the goal
marker mesh above -- render 0 pixels in both the RGB and semantic sensors,
regardless of position/material/API used. Confirmed as an upstream
habitat-sim/GPU-driver issue (reproduces on habitat-sim's own official
tutorial scene, unrelated to this project's code), not something fixable
here. See memory `project_dynamic_object_render_bug` for the full
investigation. Until that's fixed upstream, `BeliefGoalTracker.observe()`
below will never see the mesh and the rover will just hold in place after
every click. The click -> deproject -> register-mesh -> gate-on-visibility
architecture is otherwise correct and should work as-is once the underlying
render bug is resolved.

Usage:
    conda activate habitat
    python -m sam_vla.run_navdp_click_rollout \
        --scene-path assets/marsyard2022.glb \
        --heightmap-path marsyard2022_terrain_hm_1025.tif \
        --navdp-upstream-root /path/to/InternRobotics/NavDP \
        --start-x 0 --start-z 8 --start-yaw 0

Controls: left-click the RGB pane to set/replace the goal, C clears the
current goal (rover holds), X/Escape quits.
"""

from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.goal_geometry import GoalPosition, MESH_GOAL_ID, bbox_to_world
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, Observation
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.perception.semantic_overlay import overlay_semantic_masks
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy

DEPTH_VIS_MAX_METERS = 10.0
CLICK_PATCH_HALF_PX = 4


def default_navdp_upstream_ckpt() -> Optional[str]:
    """Repo-relative fallback, same convention as run_navdp_rollout.py's
    _default_navdp_upstream_ckpt: navdp/navdp-cross-modal.ckpt if present."""
    candidate = Path(__file__).resolve().parent.parent / "navdp" / "navdp-cross-modal.ckpt"
    return str(candidate) if candidate.exists() else None


def pixel_to_world(
    obs: Observation, px: int, py: int, hfov_deg: float, patch_half_px: int = CLICK_PATCH_HALF_PX
) -> Optional[GoalPosition]:
    """Deproject a clicked pixel into a world-frame point, to anchor the goal
    mesh at click time. Uses a small pixel patch around the click (not just
    the single pixel) and lets goal_geometry.bbox_to_world's
    median-depth-over-patch logic absorb a click that lands right on a depth
    discontinuity, same robustness reason it exists for bbox centers. Returns
    None if no pixel in the patch has valid depth (e.g. the click landed on
    sky/void)."""
    height, width = obs.depth.shape[:2]
    bbox_norm = (
        min(max((px - patch_half_px) / width, 0.0), 1.0),
        min(max((py - patch_half_px) / height, 0.0), 1.0),
        min(max((px + patch_half_px) / width, 0.0), 1.0),
        min(max((py + patch_half_px) / height, 0.0), 1.0),
    )
    if bbox_norm[0] >= bbox_norm[2] or bbox_norm[1] >= bbox_norm[3]:
        return None
    return bbox_to_world(obs, bbox_norm, hfov_deg)


def depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, DEPTH_VIS_MAX_METERS)
    vis = (clipped / DEPTH_VIS_MAX_METERS * 255.0).astype(np.uint8)
    return np.stack([vis, vis, vis], axis=-1)


class ClickGoalRolloutApp:
    def __init__(
        self,
        env: MarsHabitatEnv,
        policy: NavdpUpstreamPolicy,
        out_dir: str,
        dt: float,
        obj_mask_radius: float,
        goal_success_radius: float,
        min_visible_px: int,
        belief_goal_range: float,
        hfov_deg: float,
        log_path: Optional[str],
    ):
        self.env = env
        self.policy = policy
        self.out_dir = out_dir
        self.dt = dt
        self.obj_mask_radius = obj_mask_radius
        self.goal_success_radius = goal_success_radius
        self.hfov_deg = hfov_deg
        self.log_file = open(log_path, "a") if log_path else None

        # Body-frame [forward, left] belief re-seeded from the goal mesh's live
        # rendered mask whenever visible; dead-reckoned by executed motion the
        # rest of the time -- this (not raw pose subtraction against a stored
        # world point) is what feeds NavdpUpstreamPolicy.set_goal_body().
        self.belief_tracker = BeliefGoalTracker(
            hfov_deg=hfov_deg, goal_range=belief_goal_range, min_px=min_visible_px
        )
        self.goal_obj = None  # currently registered goal-mesh rigid object, or None
        self._goal_click_count = 0
        self.step_idx = 0
        self.running = True
        self.last_obs: Optional[Observation] = None
        self.last_status = "no goal -- click the image to set one"

        self.root = tk.Tk()
        self.root.title("NavDP click-to-goal rollout")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.on_click)

        self.info_label = tk.Label(
            self.root,
            text="left-click = set goal | C = clear goal | X/Esc = quit",
            font=("Arial", 12),
        )
        self.info_label.pack()

        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._tick()

    def _retire_goal_obj(self) -> None:
        """Detach the current goal mesh from the goal semantic channel (rather
        than removing it from the scene -- cheap, and mirrors
        run_navdp_rollout.py's base-station code directly retagging
        obj.semantic_id in place). It stays onstage as an inert marker of
        where a past click landed."""
        if self.goal_obj is not None:
            self.goal_obj.semantic_id = 0
        self.goal_obj = None
        self.belief_tracker.belief_g = None

    def on_click(self, event) -> None:
        if self.last_obs is None:
            return
        height, width = self.last_obs.depth.shape[:2]
        if not (0 <= event.x < width and 0 <= event.y < height):
            return  # click landed on the depth pane, not the RGB pane
        world_pos = pixel_to_world(self.last_obs, event.x, event.y, self.hfov_deg)
        if world_pos is None:
            print(
                f"[goal] click at pixel=({event.x},{event.y}) has no valid depth "
                "in its neighborhood -- ignoring (try a point on solid ground/rock)",
                flush=True,
            )
            return

        self._retire_goal_obj()
        self._goal_click_count += 1
        self.goal_obj = self.env.register_object_mask(
            world_pos,
            MESH_GOAL_ID,
            self.obj_mask_radius,
            self.out_dir,
            f"goal_{self._goal_click_count}",
        )
        print(
            f"[goal] click at pixel=({event.x},{event.y}) -> world={world_pos} -- "
            f"registered {self.obj_mask_radius}m marker mesh (first drive step may take a "
            "while: NavDP server model load)",
            flush=True,
        )

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in ("x", "escape"):
            self.close()
        elif key == "c":
            self._retire_goal_obj()
            print("[goal] cleared", flush=True)

    def _tick(self) -> None:
        if not self.running:
            return

        obs = self.env.get_observation(frame_idx=self.step_idx)
        semantic = self.env.get_semantic_frame()
        self.last_obs = obs

        goal_mask = semantic == MESH_GOAL_ID
        if self.goal_obj is not None:
            visible = self.belief_tracker.observe(goal_mask, obs.depth)
            if visible:
                forward, left = (float(v) for v in self.belief_tracker.belief_g)
                self.policy.set_goal_body(forward, left)
                try:
                    action, _vla_result = self.policy.act_verbose(
                        obs, semantic=None, goal_spec=None, step=self.step_idx
                    )
                except Exception as exc:  # e.g. server not up yet / request error
                    print(f"[navdp] act_verbose failed: {exc!r} -- holding this step", flush=True)
                    action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                dist = self.belief_tracker.distance()
                if dist is not None and dist <= self.goal_success_radius:
                    print(f"[goal] reached (dist={dist:.2f}m) -- holding", flush=True)
                    self._retire_goal_obj()
                    action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                    self.last_status = f"goal reached, dist={dist:.2f}m"
                else:
                    self.last_status = (
                        f"VISIBLE dist={dist:.2f}m goal=[fwd={forward:.2f},left={left:.2f}] "
                        f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f}"
                    )
            else:
                action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
                self.last_status = "goal mesh registered but NOT VISIBLE this frame -- holding"
            self.belief_tracker.propagate(action, self.dt)
        else:
            action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
            self.last_status = "no goal -- click the image to set one"

        new_pose = integrate_mars(obs.pose, action, self.dt)
        self.env.step(new_pose)

        if self.log_file is not None:
            self.log_file.write(
                f"{self.step_idx} x={new_pose.x:.4f} z={new_pose.z:.4f} "
                f"yaw={new_pose.yaw:.4f} v_fwd={action.v_fwd:.4f} "
                f"v_lat={action.v_lat:.4f} yaw_rate={action.yaw_rate:.4f} "
                f"status={self.last_status!r}\n"
            )
            self.log_file.flush()

        self.render(obs, semantic)
        self.step_idx += 1
        if self.step_idx % 20 == 0:
            print(f"[traj] step={self.step_idx} {self.last_status}", flush=True)

        self.root.after(max(int(self.dt * 1000), 1), self._tick)

    def render(self, obs: Observation, semantic: np.ndarray) -> None:
        depth_vis = depth_to_vis(obs.depth)
        vis_rgb = overlay_semantic_masks(
            obs.rgb, semantic, text=f"t={self.step_idx} {self.last_status}"
        )
        img_arr = np.hstack([vis_rgb, depth_vis])
        img = Image.fromarray(img_arr)
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        try:
            self.policy.stop()
        except Exception:
            pass
        if self.log_file is not None:
            self.log_file.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run(
    scene_path: str,
    heightmap_path: str,
    out_dir: str,
    navdp_upstream_ckpt: Optional[str] = None,
    navdp_upstream_root: Optional[str] = None,
    navdp_upstream_port: Optional[int] = None,
    start_x: float = 0.0,
    start_z: float = 8.0,
    start_yaw_deg: float = 0.0,
    dt: float = 0.1,
    obj_mask_radius: float = 0.5,
    goal_success_radius: float = 0.75,
    min_visible_px: int = 10,
    belief_goal_range: float = 8.0,
    stop_threshold: float = 0.0,
    lookahead: int = 3,
    replan_every: int = 1,
    max_forward_speed: float = 1.0,
    turn_kp: float = 1.4,
    max_yaw_rate: float = 1.0,
    request_timeout: float = 30.0,
) -> None:
    if not navdp_upstream_ckpt:
        navdp_upstream_ckpt = default_navdp_upstream_ckpt()
    if not navdp_upstream_ckpt:
        raise ValueError(
            "--navdp-upstream-ckpt is required (no checkpoint found at the default "
            "navdp/navdp-cross-modal.ckpt either)"
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = str(Path(out_dir) / "click_rollout.log")

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        with_semantic=True,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)

        policy = NavdpUpstreamPolicy(
            checkpoint_path=navdp_upstream_ckpt,
            navdp_upstream_root=navdp_upstream_root,
            port=navdp_upstream_port,
            stop_threshold=stop_threshold,
            image_hw=obs0.rgb.shape[:2],
            hfov_deg=HFOV_DEG,
            lookahead=lookahead,
            replan_every=replan_every,
            max_forward_speed=max_forward_speed,
            turn_kp=turn_kp,
            max_yaw_rate=max_yaw_rate,
            request_timeout=request_timeout,
        )

        app = ClickGoalRolloutApp(
            env=env,
            policy=policy,
            out_dir=out_dir,
            dt=dt,
            obj_mask_radius=obj_mask_radius,
            goal_success_radius=goal_success_radius,
            min_visible_px=min_visible_px,
            belief_goal_range=belief_goal_range,
            hfov_deg=HFOV_DEG,
            log_path=log_path,
        )
        app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument(
        "--out-dir",
        default=f"navdp_click_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}",
    )
    parser.add_argument(
        "--navdp-upstream-ckpt",
        default=None,
        help="Path to the upstream NavDP checkpoint (default: navdp/navdp-cross-modal.ckpt if present)",
    )
    parser.add_argument(
        "--navdp-upstream-root",
        default=None,
        help="Path to the vendored InternRobotics/NavDP checkout (default: $NAVDP_UPSTREAM_ROOT)",
    )
    parser.add_argument("--navdp-upstream-port", type=int, default=None)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=8.0)
    parser.add_argument("--start-yaw", type=float, default=0.0, dest="start_yaw_deg")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--obj-mask-radius",
        type=float,
        default=0.5,
        help="Radius (m) of the terrain-following goal marker mesh registered at the click",
    )
    parser.add_argument("--goal-success-radius", type=float, default=0.75)
    parser.add_argument(
        "--min-visible-px",
        type=int,
        default=10,
        help="Minimum goal-mask pixel count this frame to count as 'visible' and drive",
    )
    parser.add_argument("--belief-goal-range", type=float, default=8.0)
    parser.add_argument("--stop-threshold", type=float, default=0.0)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--max-forward-speed", type=float, default=1.0)
    parser.add_argument("--turn-kp", type=float, default=1.4)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        out_dir=args.out_dir,
        navdp_upstream_ckpt=args.navdp_upstream_ckpt,
        navdp_upstream_root=args.navdp_upstream_root,
        navdp_upstream_port=args.navdp_upstream_port,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw_deg,
        dt=args.dt,
        obj_mask_radius=args.obj_mask_radius,
        goal_success_radius=args.goal_success_radius,
        min_visible_px=args.min_visible_px,
        belief_goal_range=args.belief_goal_range,
        stop_threshold=args.stop_threshold,
        lookahead=args.lookahead,
        replan_every=args.replan_every,
        max_forward_speed=args.max_forward_speed,
        turn_kp=args.turn_kp,
        max_yaw_rate=args.max_yaw_rate,
        request_timeout=args.request_timeout,
    )
