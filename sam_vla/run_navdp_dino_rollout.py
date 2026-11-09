"""Headless DINO+NavDP text-goal rollout: drive to whatever object
``--target-text`` describes (e.g. "big rock", "the boulder"), using live
open-vocabulary detection instead of a fixed first-frame bbox or a closed
goal_resolution/ vocabulary.

This is sam_vla's port of a teammate's Nav_new/MARS DINO+NavDP navigation
stack (github.com/priyan212/Nav_new/tree/master/MARS -- specifically
habitat_sim_node.py + mars_gui.py's ``MarsPipeline``) -- reimplemented
against this repo's own infrastructure, not imported from that repo:

  * Grounding DINO + SAM2 box segmentation (perception/dino_goal_detector.py,
    perception/box_prompted_sam.py, both new) resolve the text goal to a
    live mask every frame, in-process, instead of that repo's Isaac/Zenoh
    camera bridge.
  * The mask feeds core.belief_tracking.BeliefGoalTracker (already this
    repo's own mask -> body-frame-goal + dead-reckoning belief) instead of
    a new belief system -- it does the same job as their
    ``navdp.extensions.belief_bank.SubgoalBeliefBank``-driven
    ``MarsPipeline._step_inner``: while the target is out of view, its
    body-frame estimate is propagated by the rover's own executed motion
    and a scalar uncertainty grows, so SEARCH only kicks in once that
    uncertainty crosses ``--belief-max-uncertainty`` rather than after a
    fixed frame count.
  * policy.navdp_upstream_policy.NavdpUpstreamPolicy (already this repo's
    HTTP client for the real, published NavDP model -- see next.md's
    Integration project) drives TRACK, instead of their in-process
    nav_pipeline/navdp_crossmodal.py reimplementation of the same
    checkpoint's transformer. Same model weights
    (navdp/navdp-cross-modal.ckpt), different serving path -- this repo
    already had a working backend for this exact checkpoint, so there was
    no reason to duplicate ~400 lines of transformer/DDPM code to get it.
  * safety.depth_obstacle_guard (new, ported from their obstacle_guard.py)
    adds a hard, depth-only AVOID veto with anti-oscillation cooldown, on
    top of whatever NavDP itself decides.

State machine per step, mirroring MarsPipeline._step_inner (see
dino_navdp_step's docstring): SEARCH (never yet acquired, or belief
uncertainty too high -- spin in place toward the last-known side), AVOID
(obstacle guard hard-stop -- reverse/turn, skips NavDP entirely this tick),
TRACK (steer via NavDP on the belief's current body-frame estimate), STOP
(within --stop-distance of the belief's estimate).

Unlike run_navdp_click_rollout.py's click-to-goal flow, this never registers
a dynamic marker mesh into the scene (MarsHabitatEnv.register_object_mask)
-- the goal is read directly off SAM's mask of a real, already-in-scene
object -- so it isn't blocked by the dynamic-object render bug documented in
that script's docstring (memory: project_dynamic_object_render_bug).

Usage:
    conda activate habitat
    python -m sam_vla.run_navdp_dino_rollout --scene-path assets/marsyard2022.glb --heightmap-path marsyard2022_terrain_hm_1025.tif --navdp-upstream-root ../navdp_upstream/ --target-text "big rock" --start-x 8 --start-z 10 --start-yaw 0 --out-dir navdp_dino_rollout_out --save-video

"""

from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from sam_vla.core.belief_tracking import BeliefGoalTracker
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, Observation
from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.logging.rollout_logger import RolloutLogger
from sam_vla.perception.box_prompted_sam import Sam2BoxSegmenter
from sam_vla.perception.dino_goal_detector import GroundingDinoDetector
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.safety.depth_obstacle_guard import (
    GuardConfig,
    apply_avoid_cooldown,
    depth_to_obstacle_points,
    forward_guard,
)

GOAL_MASK_OVERLAY_COLOR = np.array([0, 255, 60], dtype=np.float32)


def default_navdp_upstream_ckpt() -> Optional[str]:
    """Repo-relative fallback, same convention as run_navdp_rollout.py's
    _default_navdp_upstream_ckpt / run_navdp_click_rollout.py's
    default_navdp_upstream_ckpt: navdp/navdp-cross-modal.ckpt if present."""
    candidate = (
        Path(__file__).resolve().parent.parent / "navdp" / "navdp-cross-modal.ckpt"
    )
    return str(candidate) if candidate.exists() else None


class _AvoidState:
    """Mutable across-step state apply_avoid_cooldown's contract needs
    (latched escape side + remaining cooldown ticks), plus the AVOID
    hard-stop confirmation streak -- kept in one small object rather than
    threaded through dino_navdp_step's return value so the caller doesn't
    have to unpack/repack it every step."""

    __slots__ = ("streak", "side", "cooldown")

    def __init__(self) -> None:
        self.streak = 0
        self.side = 0.0
        self.cooldown = 0


def overlay_goal_mask(
    rgb: np.ndarray, mask: Optional[np.ndarray], text: str
) -> np.ndarray:
    """Alpha-blend green over the live SAM goal mask (if any this frame) and
    draw a status banner -- mirrors perception.semantic_overlay's look, but
    over a plain bool mask rather than a semantic-id frame (this pipeline has
    no semantic sensor at all, see run()'s with_semantic=False)."""
    overlaid = np.asarray(rgb, dtype=np.float32).copy()
    if mask is not None and mask.shape[:2] == rgb.shape[:2]:
        overlaid[mask] = 0.55 * overlaid[mask] + 0.45 * GOAL_MASK_OVERLAY_COLOR
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    img = Image.fromarray(overlaid)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def dino_navdp_step(
    obs: Observation,
    target_text: str,
    detector: GroundingDinoDetector,
    segmenter: Sam2BoxSegmenter,
    belief: BeliefGoalTracker,
    navdp_policy: NavdpUpstreamPolicy,
    avoid_state: _AvoidState,
    guard_cfg: GuardConfig,
    hfov_deg: float,
    stop_distance: float = 0.75,
    search_angular: float = 0.4,
    max_forward_speed: float = 1.0,
    max_yaw_rate: float = 1.0,
    avoid_confirm_ticks: int = 2,
    avoid_cooldown_ticks: int = 8,
    avoid_bias_gain: float = 0.3,
    belief_max_uncertainty: float = 0.6,
    step: int = 0,
) -> tuple[Action, str, dict]:
    """One tick of the SEARCH/TRACK/AVOID/STOP state machine (see this
    module's docstring). Returns (action, state, info); info carries
    detection/mask/timing fields for logging plus enough to call
    belief.propagate(action, dt) afterward -- that call is the caller's
    responsibility (needs dt, which this function doesn't have), same split
    as run_navdp_click_rollout.py's ClickGoalRolloutApp._tick().

    Precedence mirrors MarsPipeline._step_inner: detection/belief update
    first; SEARCH returns immediately if there's nothing to steer toward
    (skips the obstacle guard -- it isn't driving forward regardless); only
    once TRACK is possible does the obstacle guard get a say, matching
    upstream MARS exactly.
    """
    info: dict = {
        "detection_score": None,
        "detection_label": None,
        "visible": False,
        "belief_used": False,
        "distance": None,
        "min_forward": None,
        "mask": None,
    }

    det = detector.detect_best(obs.rgb, target_text)
    mask = None
    if det is not None:
        info["detection_score"] = det.score
        info["detection_label"] = det.label
        mask = segmenter.segment_box(obs.rgb, det.box)
    visible = belief.observe(mask, obs.depth) if mask is not None else False
    info["visible"] = visible
    if visible:
        info["mask"] = mask

    if not visible:
        if (
            belief.belief_g is None
            or belief.uncertainty_value() > belief_max_uncertainty
        ):
            side = 1.0
            if belief.belief_g is not None and belief.belief_g[1] < 0:
                side = (
                    -1.0
                )  # last known position was to the right -- keep searching that way
            avoid_state.streak = 0
            return (
                Action(v_fwd=0.0, v_lat=0.0, yaw_rate=side * search_angular),
                "SEARCH",
                info,
            )
        info["belief_used"] = True

    distance = belief.distance()
    info["distance"] = distance
    if distance is not None and distance <= stop_distance:
        avoid_state.streak = 0
        return Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0), "STOP", info

    height, width = obs.depth.shape[:2]
    obstacle_pts = depth_to_obstacle_points(
        obs.depth,
        height,
        width,
        hfov_deg,
        guard_cfg,
        exclude_mask=mask if visible else None,
    )
    min_forward, escape = forward_guard(obstacle_pts, guard_cfg)
    info["min_forward"] = min_forward

    if min_forward < guard_cfg.hard_stop_dist:
        avoid_state.streak += 1
        if avoid_state.streak >= avoid_confirm_ticks:
            avoid_state.side = escape
            avoid_state.cooldown = avoid_cooldown_ticks
            v_fwd = (
                -0.5 * max_forward_speed
                if min_forward < guard_cfg.reverse_dist
                else 0.0
            )
            return (
                Action(v_fwd=v_fwd, v_lat=0.0, yaw_rate=escape * max_yaw_rate),
                "AVOID",
                info,
            )
    else:
        avoid_state.streak = 0

    forward, left = (float(v) for v in belief.belief_g)
    navdp_policy.set_goal_body(forward, left)
    action, vla_result = navdp_policy.act_verbose(
        obs, semantic=None, goal_spec=None, step=step
    )
    info["vla_result"] = vla_result
    yaw_rate, avoid_state.cooldown = apply_avoid_cooldown(
        action.yaw_rate,
        "TRACK",
        avoid_state.side,
        avoid_state.cooldown,
        avoid_bias_gain,
        max_yaw_rate,
    )
    return (
        Action(v_fwd=action.v_fwd, v_lat=action.v_lat, yaw_rate=yaw_rate),
        "TRACK",
        info,
    )


def run(
    scene_path: str,
    heightmap_path: str,
    out_dir: str,
    target_text: str,
    navdp_upstream_ckpt: Optional[str] = None,
    navdp_upstream_root: Optional[str] = None,
    navdp_upstream_port: Optional[int] = None,
    start_x: float = 0.0,
    start_z: float = 8.0,
    start_yaw_deg: float = 0.0,
    dt: float = 0.1,
    max_steps: int = 500,
    device: str = "cuda:0",
    dino_model_id: str = "IDEA-Research/grounding-dino-base",
    dino_box_threshold: float = 0.35,
    dino_text_threshold: float = 0.25,
    sam_model_id: str = "facebook/sam2.1-hiera-small",
    stop_distance: float = 0.75,
    search_angular: float = 0.4,
    belief_goal_range: float = 8.0,
    belief_min_px: int = 10,
    belief_odom_noise: float = 0.02,
    belief_odom_noise_growth_rate: float = 0.1,
    belief_max_uncertainty: float = 0.6,
    guard_hard_stop_dist: float = 0.60,
    guard_reverse_dist: float = 0.35,
    guard_slow_dist: float = 2.5,
    guard_max_climb_deg: float = 20.0,
    guard_max_range: float = 4.0,
    avoid_confirm_ticks: int = 2,
    avoid_cooldown_ticks: int = 8,
    avoid_bias_gain: float = 0.3,
    lookahead: int = 3,
    replan_every: int = 1,
    max_forward_speed: float = 1.0,
    turn_kp: float = 1.4,
    max_yaw_rate: float = 1.0,
    request_timeout: float = 30.0,
    save_video: bool = False,
    save_frames: bool = False,
    video_fps: int = 10,
) -> None:
    if not navdp_upstream_ckpt:
        navdp_upstream_ckpt = default_navdp_upstream_ckpt()
    if not navdp_upstream_ckpt:
        raise ValueError(
            "--navdp-upstream-ckpt is required (no checkpoint found at the default "
            "navdp/navdp-cross-modal.ckpt either)"
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger = RolloutLogger()

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        with_semantic=False,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)

        detector = GroundingDinoDetector(
            model_id=dino_model_id,
            device=device,
            box_threshold=dino_box_threshold,
            text_threshold=dino_text_threshold,
        )
        segmenter = Sam2BoxSegmenter(model_id=sam_model_id, device=device)
        belief = BeliefGoalTracker(
            hfov_deg=HFOV_DEG,
            goal_range=belief_goal_range,
            min_px=belief_min_px,
            odom_noise=belief_odom_noise,
            odom_noise_growth_rate=belief_odom_noise_growth_rate,
        )
        navdp_policy = NavdpUpstreamPolicy(
            checkpoint_path=navdp_upstream_ckpt,
            navdp_upstream_root=navdp_upstream_root,
            port=navdp_upstream_port,
            image_hw=obs0.rgb.shape[:2],
            hfov_deg=HFOV_DEG,
            lookahead=lookahead,
            replan_every=replan_every,
            max_forward_speed=max_forward_speed,
            turn_kp=turn_kp,
            max_yaw_rate=max_yaw_rate,
            request_timeout=request_timeout,
        )
        guard_cfg = GuardConfig(
            hard_stop_dist=guard_hard_stop_dist,
            reverse_dist=guard_reverse_dist,
            slow_dist=guard_slow_dist,
            max_climb_deg=guard_max_climb_deg,
            max_range=guard_max_range,
        )
        avoid_state = _AvoidState()

        state = "SEARCH"
        try:
            for step in range(max_steps):
                obs = env.get_observation(frame_idx=step)
                action, state, info = dino_navdp_step(
                    obs,
                    target_text,
                    detector,
                    segmenter,
                    belief,
                    navdp_policy,
                    avoid_state,
                    guard_cfg,
                    HFOV_DEG,
                    stop_distance=stop_distance,
                    search_angular=search_angular,
                    max_forward_speed=max_forward_speed,
                    max_yaw_rate=max_yaw_rate,
                    avoid_confirm_ticks=avoid_confirm_ticks,
                    avoid_cooldown_ticks=avoid_cooldown_ticks,
                    avoid_bias_gain=avoid_bias_gain,
                    belief_max_uncertainty=belief_max_uncertainty,
                    step=step,
                )
                belief.propagate(action, dt)

                dist_txt = (
                    f"{info['distance']:.2f}m" if info["distance"] is not None else "-"
                )
                min_fwd_txt = (
                    f"{info['min_forward']:.2f}m"
                    if info["min_forward"] is not None
                    and np.isfinite(info["min_forward"])
                    else "-"
                )
                status = (
                    f"t={step} [{state}] dist={dist_txt} fwd-clear={min_fwd_txt} "
                    f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f}"
                )
                vis_rgb = overlay_goal_mask(obs.rgb, info["mask"], status)
                logger.log_step(
                    obs,
                    action,
                    obs.pose,
                    vla_result={
                        "state": state,
                        "detection_score": info["detection_score"],
                        "detection_label": info["detection_label"],
                        "visible": info["visible"],
                        "belief_used": info["belief_used"],
                        "distance": info["distance"],
                        "min_forward": info["min_forward"],
                        "navdp": info.get("vla_result"),
                    },
                    vis_rgb=vis_rgb,
                )

                if step % 20 == 0 or state != "TRACK":
                    print(f"[navdp-dino] {status}", flush=True)

                new_pose = integrate_mars(obs.pose, action, dt)
                env.step(new_pose)

                if state == "STOP":
                    print(
                        f"[navdp-dino] goal reached at step {step} (dist={dist_txt})",
                        flush=True,
                    )
                    break
            else:
                print(
                    f"[navdp-dino] max_steps={max_steps} reached without STOP",
                    flush=True,
                )
        finally:
            navdp_policy.stop()

        logger.flush(out_dir)
        if save_frames:
            logger.save_frames(out_dir)
        if save_video:
            logger.save_video(out_dir, fps=video_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument(
        "--target-text", required=True, help='e.g. "big rock", "the boulder"'
    )
    parser.add_argument(
        "--out-dir",
        default=f"navdp_dino_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}",
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
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dino-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--dino-box-threshold", type=float, default=0.35)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--stop-distance", type=float, default=0.75)
    parser.add_argument("--search-angular", type=float, default=0.4)
    parser.add_argument("--belief-goal-range", type=float, default=8.0)
    parser.add_argument("--belief-min-px", type=int, default=10)
    parser.add_argument("--belief-odom-noise", type=float, default=0.02)
    parser.add_argument("--belief-odom-noise-growth-rate", type=float, default=0.1)
    parser.add_argument(
        "--belief-max-uncertainty",
        type=float,
        default=0.6,
        help="give up on the propagated goal memory and switch to SEARCH once the "
        "belief's uncertainty (grows while unseen, floored/reset on each sighting) "
        "crosses this",
    )
    parser.add_argument("--guard-hard-stop-dist", type=float, default=0.60)
    parser.add_argument("--guard-reverse-dist", type=float, default=0.35)
    parser.add_argument("--guard-slow-dist", type=float, default=2.5)
    parser.add_argument(
        "--guard-max-climb-deg",
        type=float,
        default=20.0,
        help="terrain rising up to this many degrees is treated as driveable ground, "
        "not an obstacle -- raise this if the rover balks at climbing real slopes/hills, "
        "lower it if it's driving over things it shouldn't",
    )
    parser.add_argument("--guard-max-range", type=float, default=4.0)
    parser.add_argument("--avoid-confirm-ticks", type=int, default=2)
    parser.add_argument("--avoid-cooldown-ticks", type=int, default=8)
    parser.add_argument("--avoid-bias-gain", type=float, default=0.3)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--max-forward-speed", type=float, default=1.0)
    parser.add_argument("--turn-kp", type=float, default=1.4)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        out_dir=args.out_dir,
        target_text=args.target_text,
        navdp_upstream_ckpt=args.navdp_upstream_ckpt,
        navdp_upstream_root=args.navdp_upstream_root,
        navdp_upstream_port=args.navdp_upstream_port,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw_deg,
        dt=args.dt,
        max_steps=args.max_steps,
        device=args.device,
        dino_model_id=args.dino_model_id,
        dino_box_threshold=args.dino_box_threshold,
        dino_text_threshold=args.dino_text_threshold,
        sam_model_id=args.sam_model_id,
        stop_distance=args.stop_distance,
        search_angular=args.search_angular,
        belief_goal_range=args.belief_goal_range,
        belief_min_px=args.belief_min_px,
        belief_odom_noise=args.belief_odom_noise,
        belief_odom_noise_growth_rate=args.belief_odom_noise_growth_rate,
        belief_max_uncertainty=args.belief_max_uncertainty,
        guard_hard_stop_dist=args.guard_hard_stop_dist,
        guard_reverse_dist=args.guard_reverse_dist,
        guard_slow_dist=args.guard_slow_dist,
        guard_max_climb_deg=args.guard_max_climb_deg,
        guard_max_range=args.guard_max_range,
        avoid_confirm_ticks=args.avoid_confirm_ticks,
        avoid_cooldown_ticks=args.avoid_cooldown_ticks,
        avoid_bias_gain=args.avoid_bias_gain,
        lookahead=args.lookahead,
        replan_every=args.replan_every,
        max_forward_speed=args.max_forward_speed,
        turn_kp=args.turn_kp,
        max_yaw_rate=args.max_yaw_rate,
        request_timeout=args.request_timeout,
        save_video=args.save_video,
        save_frames=args.save_frames,
        video_fps=args.video_fps,
    )
