"""Run ONE real R2R-CE val_seen episode (scene 17DRP5sb8fy) through the real
InternVLA-N1-DualVLN model (via internvla_n1_server.py, the same cross-conda-
env TCP bridge run_internvla_n1_compare.py uses) and log a pose trace +
standard VLN-CE metrics -- the InternVLA-N1 counterpart to
run_ours_episode.py, same scene/instruction/goal per episode, same metric
definitions, so the two are directly comparable.

Reuses run_internvla_n1_compare.py's own action-decoding helpers (discrete
VLN-CE token replay, continuous-trajectory replay, look-down sub-flow) rather
than reimplementing them -- those were already verified live against the real
model's actual output shapes.

    conda activate habitat
    python -m nav.benchmark.run_internvla_episode --episode-idx 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_internvla_n1_compare as ivc  # noqa: E402

from nav.benchmark.r2rce_common import (  # noqa: E402
    SCENE_DATASET_CONFIG,
    SCENE_PATH,
    compute_metrics,
    load_episodes,
    quat_to_yaw,
)
from sam_vla.core.types import Pose  # noqa: E402
from sam_vla.env.habitat_env import MarsHabitatEnv  # noqa: E402

MAX_TICKS = 1500  # 150s of sim time at DT=0.1, matching run_ours_episode.py's MAX_WALL_S budget


def run_episode(ep: dict, out_dir: Path) -> dict:
    start_x, _start_y, start_z = ep["start_position"]
    start_yaw = quat_to_yaw(ep["start_rotation"])
    goal_x, _goal_y, goal_z = ep["goal_position"]
    instruction = ep["instruction"]

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"internvla_ep{ep['episode_id']}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log_f = open(trace_path, "w")
    path_xz = []

    def log_pose(pose: Pose, event: str = "tick", **extra) -> None:
        path_xz.append((pose.x, pose.z))
        rec = {"t": round(time.monotonic() - t0, 3), "x": pose.x, "z": pose.z, "yaw": pose.yaw, "event": event}
        rec.update(extra)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    print(f"[!] pinging internvla_n1_server...")
    ivc._send_request("ping", {})
    ivc._send_request("reset", {})

    print(
        f"[!] episode {ep['episode_id']}: start=({start_x:.2f},{start_z:.2f}) "
        f"yaw={math.degrees(start_yaw):.1f} goal=({goal_x:.2f},{goal_z:.2f}) "
        f"geo={ep['geodesic_distance']:.2f}m\n    instruction={instruction!r}"
    )

    stopped = False
    with MarsHabitatEnv(
        SCENE_PATH,
        None,
        services=[],
        scene_dataset_config=SCENE_DATASET_CONFIG,
        start_x=start_x,
        start_z=start_z,
        start_yaw=start_yaw,
        spawn_flags=False,
        gen_home_base=False,
        annotations_dir=None,
    ) as env:
        obs = env.get_observation(frame_idx=0)
        pose = obs.pose
        t0 = time.monotonic()
        log_pose(pose, event="start")

        rgb_sensor = env._agent._sensors["rgb"]
        agent_forward_rotation = rgb_sensor.node.rotation

        def capture_look_down(pitch_deg: float = -40.0):
            import magnum as mn

            rgb_sensor.node.rotation = mn.Quaternion.rotation(mn.Deg(pitch_deg), mn.Vector3.x_axis())
            raw = env._sim.get_sensor_observations()
            rgb_sensor.node.rotation = agent_forward_rotation
            rgb = ivc.np.asarray(raw["rgb"])[:, :, :3]
            depth = ivc.np.asarray(raw["depth"], dtype=ivc.np.float32)
            return rgb, depth

        def query_step(rgb, depth, look_down: bool = False) -> dict:
            return ivc._send_request(
                "step",
                {
                    "rgb_b64": ivc._encode_rgb(rgb),
                    "depth_b64": ivc._encode_depth(depth),
                    "height": int(depth.shape[0]),
                    "width": int(depth.shape[1]),
                    "pose": ivc._pose_to_4x4(pose),
                    "instruction": instruction,
                    "look_down": look_down,
                },
            )

        tick = 0
        while tick < MAX_TICKS and not stopped:
            obs = env.get_observation(frame_idx=tick)
            resp = query_step(obs.rgb, obs.depth)

            look_down_retries = 0
            while (
                resp["output_action"] is not None
                and len(resp["output_action"]) == 1
                and resp["output_action"][0] == 5
                and look_down_retries < 3
            ):
                ld_rgb, ld_depth = capture_look_down()
                resp = query_step(ld_rgb, ld_depth, look_down=True)
                look_down_retries += 1

            def step_and_log(new_pose: Pose) -> None:
                env.step(new_pose)
                log_pose(new_pose)

            if resp["output_action"] is not None:
                tokens = [t for t in resp["output_action"] if t != 5]
                print(f"[tick {tick}] discrete action(s): {[ivc.ACTIONS2IDX_INV.get(t, t) for t in tokens]}")
                if not tokens:
                    tick += ivc.DISCRETE_STEP_TICKS
                for tok in tokens:
                    if tok == 0:
                        log_pose(pose, event="stop")
                        stopped = True
                        break
                    pose = ivc._apply_discrete_action(pose, tok, step_and_log)
                    tick += ivc.DISCRETE_STEP_TICKS
                    if tick >= MAX_TICKS:
                        break
            elif resp["output_trajectory"] is not None:
                print(f"[tick {tick}] trajectory, {len(resp['output_trajectory'])} waypoints")
                pose = ivc._apply_trajectory(pose, resp["output_trajectory"], step_and_log)
                tick += ivc.PLAN_STEP_GAP
            else:
                tick += ivc.PLAN_STEP_GAP

        log_pose(pose, event="finished" if not stopped else "stopped")
        print(f"[!] {'stopped' if stopped else 'TIMED OUT'} after {tick} ticks, {len(path_xz)} samples")

    log_f.close()

    metrics = compute_metrics(
        path_xz,
        goal_xz=(goal_x, goal_z),
        goal_radius=ep["goal_radius"],
        geodesic_distance=ep["geodesic_distance"],
    )
    metrics["episode_id"] = ep["episode_id"]
    metrics["finished"] = stopped
    metrics["trace_path"] = str(trace_path)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-idx", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0, help="repeat index, for multi-seed variance runs")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    episodes = load_episodes()
    ep = episodes[args.episode_idx]
    out_dir = Path(args.out_dir)
    metrics = run_episode(ep, out_dir)
    metrics["rep"] = args.rep
    print(f"[!] metrics: {metrics}")
    result_path = out_dir / f"internvla_ep{ep['episode_id']}_rep{args.rep}_metrics.json"
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[!] wrote {result_path}")


if __name__ == "__main__":
    main()
