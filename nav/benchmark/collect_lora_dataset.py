"""Phase A of the LoRA fine-tuning plan (see /home/gpu/.claude/plans/
prancy-plotting-lerdorf.md): drive ONE real R2R-CE TRAIN-split episode
(scene 17DRP5sb8fy -- the val_seen split stays untouched, reserved for
eval) through the exact same production stack run_ours_episode.py uses
(VlmPointNavigator + NavdpUpstreamPolicy), but instead of only computing
final metrics, dump every replan-tick's frame+prompt alongside a HINDSIGHT
label derived from the episode's own ground-truth goal/reference_path --
privileged information used only to construct training labels here, never
exposed to the model at inference time.

Runs the real production decisions (not an oracle-scripted trajectory) so
the frames a training example sees match the real inference-time
distribution -- teaching the model from states it will actually encounter,
not states a perfect driver would visit.

Known limitations (v1 of this dataset, see plan):
  - Single-turn examples only, no multi-turn history -- VlmPointNavigator's
    own rolling history buffer grows on every replan tick (found or not),
    while a training row is only written when the reference-path lookahead
    projects in-frame (a different, stricter condition); reconstructing an
    exactly-aligned history for each written row needs saving/labeling
    every tick regardless of whether it becomes a training example, which
    is out of scope for this first pass. This still trains real found=True
    pointing + terminal-instruction advance judgment, just without teaching
    multi-turn context use.
  - Only found=True (in-frame lookahead point) examples are ever emitted;
    found=False frames are skipped rather than labeled.

    conda activate habitat
    python -m nav.benchmark.collect_lora_dataset --episode-idx 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from nav.benchmark.r2rce_common import SCENE_DATASET_CONFIG, SCENE_PATH, quat_to_yaw
from nav.goal_math import body_frame_goal
from nav.rover_controller import (
    HFOV_DEG,
    POINT_GOAL_REACHED_M,
    TURN_GHOST_LOOKAHEAD_M,
    TURN_GOAL_REACHED_RAD,
    VLM_CREEP_M,
)
from nav.vlm_point_navigator import VlmPointNavigator
from sam_vla.core.goal_geometry import project_world_point_with_pixel_radius
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import GoalSpec
from sam_vla.env.habitat_env import MarsHabitatEnv
from sam_vla.policy.navdp_upstream_policy import NavdpUpstreamPolicy
from sam_vla.vlm import qwen_client
from sam_vla.vlm.qwen35_prompts import build_point_goal_history_prompt
from sam_vla.vlm.qwen35_server_manager import Qwen35ServerManager
from sam_vla.vlm.qwen_server_manager import QwenServerManager

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TRAIN_EPISODES_PATH = HERE / "r2rce_17DRP5sb8fy_train.json"
DEFAULT_CKPT = str(REPO_ROOT / "navdp" / "navdp-cross-modal.ckpt")
DEFAULT_NAVDP_UPSTREAM_ROOT = str(REPO_ROOT.parent / "navdp_upstream")
DT = 0.1
MAX_TICKS = 1500  # 150s, matches run_ours_episode.py's MAX_WALL_S budget
TURN_GHOST_MAX_BEARING_RAD = math.radians(HFOV_DEG / 2.0)


def load_train_episodes() -> list:
    with open(TRAIN_EPISODES_PATH) as f:
        return json.load(f)


def _closest_path_idx(reference_path: list, x: float, z: float) -> int:
    dists = [math.hypot(p[0] - x, p[2] - z) for p in reference_path]
    return min(range(len(dists)), key=lambda i: dists[i])


def _lookahead_target(reference_path: list, x: float, z: float) -> tuple:
    idx = _closest_path_idx(reference_path, x, z)
    target_idx = min(idx + 1, len(reference_path) - 1)
    return tuple(reference_path[target_idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-idx", type=int, required=True)
    ap.add_argument("--out-dir", default=str(HERE / "lora_dataset"))
    # Isolated ports for running multiple collection workers concurrently
    # (see qwen_server_manager.py/qwen35_server_manager.py/
    # navdp_upstream_server_manager.py's now-fixed --port passthrough) --
    # None on all three preserves the old single-worker defaults exactly.
    ap.add_argument("--qwen-port", type=int, default=None)
    ap.add_argument("--qwen35-port", type=int, default=None)
    ap.add_argument("--navdp-port", type=int, default=None)
    args = ap.parse_args()

    episodes = load_train_episodes()
    ep = episodes[args.episode_idx]
    start_x, _sy, start_z = ep["start_position"]
    start_yaw_deg = math.degrees(quat_to_yaw(ep["start_rotation"]))
    goal_x, _gy, goal_z = ep["goal_position"]
    goal_radius = ep["goal_radius"]
    reference_path = ep["reference_path"]
    instruction_text = ep["instruction"]

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "train.jsonl"

    print(
        f"[!] train episode {ep['episode_id']}: start=({start_x:.2f},{start_z:.2f}) "
        f"goal=({goal_x:.2f},{goal_z:.2f}) instruction={instruction_text!r}"
    )

    qwen_manager = QwenServerManager(port=args.qwen_port)
    qwen_manager.start()
    qwen35_manager = Qwen35ServerManager(port=args.qwen35_port)
    qwen35_manager.start()

    rows_written = 0
    try:
        with MarsHabitatEnv(
            SCENE_PATH,
            None,
            services=[],
            scene_dataset_config=SCENE_DATASET_CONFIG,
            start_x=start_x,
            start_z=start_z,
            start_yaw=math.radians(start_yaw_deg),
            spawn_flags=False,
            gen_home_base=False,
            annotations_dir=None,
        ) as env:
            obs0 = env.get_observation(frame_idx=0)
            instructions, _split_result = qwen_client.split_instructions_verbose(
                obs0.rgb, instruction_text, port=args.qwen_port
            )
            print(f"[!] split into {len(instructions)} instructions: {instructions}")
            navigator = VlmPointNavigator(
                instructions, hfov_deg=HFOV_DEG, qwen35_port=args.qwen35_port
            )

            policy = NavdpUpstreamPolicy(
                checkpoint_path=DEFAULT_CKPT,
                navdp_upstream_root=DEFAULT_NAVDP_UPSTREAM_ROOT,
                image_hw=obs0.rgb.shape[:2],
                hfov_deg=HFOV_DEG,
                port=args.navdp_port,
            )
            policy.start()
            goal_spec = GoalSpec(
                goal_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                obstacle_bboxes_norm=[],
                instruction_text=instruction_text,
            )

            pose = obs0.pose
            world_goal = None
            turn_target_yaw = None
            dataset_f = open(dataset_path, "a")

            tick = 0
            while tick < MAX_TICKS and not navigator.finished:
                obs = env.get_observation(frame_idx=tick)
                is_last_instruction = navigator.idx == len(navigator.instructions) - 1
                try:
                    result = navigator.maybe_step(obs, tick)
                except ValueError as exc:
                    # A malformed Qwen response (rare parse failure, e.g.
                    # missing 'advance' field) used to crash the whole
                    # episode's data collection -- confirmed live, 4/75
                    # episodes lost entirely to this on the first sweep.
                    # Skip this one poll instead: the next replan_every
                    # tick will just re-query with a fresh frame.
                    print(f"[tick {tick}] maybe_step failed, skipping poll: {exc}")
                    result = None

                if result is not None:
                    print(
                        f"[poll tick={tick} idx={navigator.idx}] action={result.action} "
                        f"found={result.found} advance={result.advance} "
                        f"world_xz={result.world_xz} pose=({obs.pose.x:.2f},{obs.pose.z:.2f})"
                    )

                if result is not None and result.action == "point":
                    height, width = obs.rgb.shape[0], obs.rgb.shape[1]
                    proj = project_world_point_with_pixel_radius(
                        obs.pose,
                        _lookahead_target(reference_path, obs.pose.x, obs.pose.z),
                        0.3,
                        HFOV_DEG,
                        width,
                        height,
                    )
                    if proj is None:
                        # Off the reference path (confirmed live: the rover
                        # can drift/overshoot well away from it) -- fall
                        # back to the true goal itself as the label target.
                        # Still an honest label (drive toward the real
                        # goal), and captures far more of the trajectory
                        # than only the strict on-path lookahead would.
                        # Still None if the goal is out-of-frame/behind the
                        # rover (e.g. after overshooting past it) -- no
                        # invented point-based label for "should have
                        # turned around", correctly skipped.
                        proj = project_world_point_with_pixel_radius(
                            obs.pose, (goal_x, obs.pose.y, goal_z), 0.3, HFOV_DEG, width, height
                        )
                    print(f"[poll tick={tick}] proj in-frame={proj is not None}")
                    if proj is not None:
                        px, py, _r = proj
                        dist_to_goal = math.hypot(obs.pose.x - goal_x, obs.pose.z - goal_z)
                        advance_label = (
                            (dist_to_goal < goal_radius) if is_last_instruction else None
                        )
                        frame_path = frames_dir / f"ep{ep['episode_id']}_tick{tick:04d}.png"
                        Image.fromarray(obs.rgb).save(frame_path)
                        prompt = build_point_goal_history_prompt(
                            navigator.instructions[navigator.idx], tick
                        )
                        row = {
                            "episode_id": ep["episode_id"],
                            "tick": tick,
                            "frame_path": str(frame_path),
                            "prompt": prompt,
                            "instruction": navigator.instructions[navigator.idx],
                            "target": {
                                "found": True,
                                "u": round(px / width, 4),
                                "v": round(py / height, 4),
                                "advance": advance_label,
                            },
                        }
                        dataset_f.write(json.dumps(row) + "\n")
                        dataset_f.flush()
                        rows_written += 1

                if result is not None and result.action == "turn":
                    sign = 1.0 if result.direction == "right" else -1.0
                    turn_target_yaw = pose.yaw + sign * math.radians(result.degrees)
                    world_goal = None
                elif result is not None and result.action == "point":
                    if result.world_xz is not None:
                        world_goal = result.world_xz
                    else:
                        world_goal = (
                            obs.pose.x - VLM_CREEP_M * math.sin(obs.pose.yaw),
                            obs.pose.z - VLM_CREEP_M * math.cos(obs.pose.yaw),
                        )
                    if result.advance:
                        if navigator.advance():
                            break

                if turn_target_yaw is not None:
                    yaw_err = (turn_target_yaw - pose.yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(yaw_err) < TURN_GOAL_REACHED_RAD:
                        turn_target_yaw = None
                        if navigator.advance():
                            break
                    else:
                        bearing = max(
                            -TURN_GHOST_MAX_BEARING_RAD, min(TURN_GHOST_MAX_BEARING_RAD, yaw_err)
                        )
                        forward = TURN_GHOST_LOOKAHEAD_M * math.cos(bearing)
                        left = TURN_GHOST_LOOKAHEAD_M * math.sin(bearing)
                        policy.set_goal_body(forward, left)
                        action, _ = policy.act_verbose(obs, None, goal_spec, tick)
                        pose = integrate_mars(pose, action, DT)
                        env.step(pose)
                elif world_goal is not None:
                    forward, left = body_frame_goal(pose, world_goal)
                    if math.hypot(forward, left) < POINT_GOAL_REACHED_M:
                        creep_x = pose.x - VLM_CREEP_M * math.sin(pose.yaw)
                        creep_z = pose.z - VLM_CREEP_M * math.cos(pose.yaw)
                        world_goal = (creep_x, creep_z)
                        forward, left = body_frame_goal(pose, world_goal)
                    policy.set_goal_body(forward, left)
                    action, _ = policy.act_verbose(obs, None, goal_spec, tick)
                    pose = integrate_mars(pose, action, DT)
                    env.step(pose)

                tick += 1

            dataset_f.close()
            policy.stop()
    finally:
        qwen_manager.stop()
        qwen35_manager.stop()

    print(f"[!] wrote {rows_written} rows to {dataset_path}")


if __name__ == "__main__":
    main()
