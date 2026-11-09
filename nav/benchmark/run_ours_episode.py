"""Run ONE real R2R-CE val_seen episode (scene 17DRP5sb8fy) through this
project's own production navigation stack (Qwen3.5+NavDP, via
nav.rover_controller.RoverController headlessly -- exact same code path
nav/gui1.py's Taskwork box drives, just without the Tkinter window) and log a
pose trace + standard VLN-CE metrics.

One episode per process invocation (matches the checkpoint/timeout semantics
NavdpUpstreamServerManager already assumes -- see its docstring: each run
either reuses an already-running navdp_server with a fresh /navigator_reset,
or spawns and later tears down its own), run under the `habitat` conda env:

    conda activate habitat
    python -m nav.benchmark.run_ours_episode --episode-idx 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from nav.benchmark.r2rce_common import (
    SCENE_DATASET_CONFIG,
    SCENE_PATH,
    compute_metrics,
    load_episodes,
    quat_to_yaw,
)
from nav.rover_controller import MODE_IDLE, RoverController

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = str(HERE.parent.parent / "navdp" / "navdp-cross-modal.ckpt")
MAX_WALL_S = 150.0  # generous vs. these episodes' 5-8m geodesic distances
POLL_HZ = 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-idx", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0, help="repeat index, for multi-seed variance runs")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    episodes = load_episodes()
    ep = episodes[args.episode_idx]
    start_x, _start_y, start_z = ep["start_position"]
    start_yaw_deg = math.degrees(quat_to_yaw(ep["start_rotation"]))
    goal_x, _goal_y, goal_z = ep["goal_position"]
    instruction = ep["instruction"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"ours_ep{ep['episode_id']}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    trace_f = open(trace_path, "w")

    print(
        f"[!] episode {ep['episode_id']}: start=({start_x:.2f},{start_z:.2f}) "
        f"yaw={start_yaw_deg:.1f} goal=({goal_x:.2f},{goal_z:.2f}) "
        f"geo={ep['geodesic_distance']:.2f}m\n    instruction={instruction!r}"
    )

    controller = RoverController(
        scene_path=SCENE_PATH,
        heightmap_path=None,
        navdp_upstream_ckpt=DEFAULT_CKPT,
        navdp_upstream_root=str(HERE.parent.parent.parent / "navdp_upstream"),
        scene_dataset_config=SCENE_DATASET_CONFIG,
        spawn_flags=False,
        gen_home_base=False,
        annotations_dir=None,
        start_x=start_x,
        start_z=start_z,
        start_yaw_deg=start_yaw_deg,
    )
    controller.start()
    try:
        # Give _run's thread time to open the env / spawn Qwen3.5+NavDP
        # servers before issuing the nav command -- start_vlm_navigation_from_
        # command just queues _pending_vlm_command_text, picked up next tick
        # regardless of when it's called, but polling snapshot() before the
        # env is open would just see the "starting" placeholder state.
        t0 = time.monotonic()
        while controller.snapshot().pose is None:
            if time.monotonic() - t0 > 60.0:
                raise RuntimeError("env never became ready within 60s")
            time.sleep(0.2)

        controller.start_vlm_navigation_from_command(instruction)

        path_xz = []
        t0 = time.monotonic()
        finished = False
        while time.monotonic() - t0 < MAX_WALL_S:
            d = controller.snapshot()
            if d.pose is not None:
                path_xz.append((d.pose.x, d.pose.z))
                trace_f.write(
                    json.dumps(
                        {
                            "t": round(time.monotonic() - t0, 3),
                            "x": d.pose.x,
                            "z": d.pose.z,
                            "yaw": d.pose.yaw,
                            "status": d.status_text,
                        }
                    )
                    + "\n"
                )
            # NOT d.status_text == "VLM navigation finished" -- that string
            # is overwritten by the generic per-tick self._status_text(...)
            # refresh (rover_controller.py's `d.status_text = status`) within
            # the same tick it's set, so it's never observable from outside
            # (confirmed live: every episode reported "TIMED OUT" even when
            # the recorded path showed it reached the goal in seconds).
            #
            # mode==MODE_IDLE and goal_reached ALONE isn't enough either: a
            # TURN instruction's physical completion sets exactly that pair
            # for one tick before the next queued instruction's first Qwen
            # poll supplies a new point goal -- confirmed live, this
            # transient handoff gap got mistaken for "episode finished"
            # seconds into a multi-instruction chain. vlm_navigation_active()
            # is False only once VlmPointNavigator.advance() has returned
            # finished=True for the LAST instruction, so combined with
            # goal_reached (the final residual point was actually reached)
            # this is the real "truly done" signal.
            if not controller.vlm_navigation_active() and d.mode == MODE_IDLE and d.goal_reached:
                finished = True
                break
            time.sleep(1.0 / POLL_HZ)
        trace_f.flush()
        trace_f.close()

        elapsed = time.monotonic() - t0
        print(f"[!] {'finished' if finished else 'TIMED OUT'} after {elapsed:.1f}s, {len(path_xz)} samples")
    finally:
        controller.shutdown(timeout=30.0)

    metrics = compute_metrics(
        path_xz,
        goal_xz=(goal_x, goal_z),
        goal_radius=ep["goal_radius"],
        geodesic_distance=ep["geodesic_distance"],
    )
    metrics["episode_id"] = ep["episode_id"]
    metrics["rep"] = args.rep
    metrics["finished"] = finished
    metrics["trace_path"] = str(trace_path)
    result_path = out_dir / f"ours_ep{ep['episode_id']}_rep{args.rep}_metrics.json"
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[!] metrics: {metrics}")
    print(f"[!] wrote {result_path}")


if __name__ == "__main__":
    main()
