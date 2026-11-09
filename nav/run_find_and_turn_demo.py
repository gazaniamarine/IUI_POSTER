#!/usr/bin/env python3
"""Headless scripted rollout: drive straight, ask Qwen2.5-VL each poll
whether a named target (e.g. "door") is visible, and once it reports a
single (u, v) point, drive to it and execute a fixed turn -- see
nav/rover_controller.py's request_forward_point/request_pixel_goal/
request_turn.

Qwen's only role here is qwen_client.ground_object_verbose's point-grounding
call (sam_vla.vlm.qwen_prompts.build_ground_object_prompt) -- no SAM2
detections, no obstacle boxes, no belief-mask tracking. All driving is
MODE_POINT/MODE_TURN, which (unlike MODE_RESOLVE's mask-tracked path) never
depends on a rendered semantic mask, so this is unaffected by the dynamic-
object-render bug on this machine (see CLAUDE.md's "Known issues").

Reuses nav.gui1's CLI/config plumbing (scene/checkpoint/CBF/etc.) for every
flag except the ones this script adds itself. Run via:

    conda activate habitat
    cd mars-habitatsim
    python -m nav.run_find_and_turn_demo --target-text flag \
        --gen-home-base --navdp-upstream-root ../navdp_upstream/

("flag" is already placed in the scene, so it's the way to sanity-check the
pipeline before pointing --target-text at an actual door-like asset.)
"""

from __future__ import annotations

import argparse
import json
import time

from nav.gui1 import build_controller, parse_args
from sam_vla.vlm import qwen_client


def parse_demo_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--target-text",
        default="door",
        help='object phrase to ground via Qwen (e.g. "door", "flag")',
    )
    ap.add_argument(
        "--forward-distance-m",
        type=float,
        default=15.0,
        help="how far the initial go-straight leg reaches before giving up",
    )
    ap.add_argument(
        "--turn-direction",
        default="left",
        choices=("left", "right", "back"),
    )
    ap.add_argument(
        "--poll-interval-s",
        type=float,
        default=1.0,
        help="how often to call Qwen (VLM inference is seconds-scale)",
    )
    ap.add_argument(
        "--find-timeout-s",
        type=float,
        default=60.0,
        help="give up looking for the target after this many seconds",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "no-VLM control run: skip the Qwen polling entirely and always go "
            "straight through the fixed forward leg then turn, regardless of "
            "whether the target would have been visible. For benchmarking the "
            "Qwen-point-goal redirect against a blind open-loop policy."
        ),
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="write a structured benchmark record (phase timestamps/poses) here",
    )
    return ap.parse_known_args(argv)


def _wait_until(predicate, poll_interval_s: float, timeout_s: float = None) -> bool:
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while not predicate():
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval_s)
    return True


def _pose_dict(pose) -> dict:
    if pose is None:
        return {}
    return {"x": pose.x, "z": pose.z, "yaw": pose.yaw}


def main(argv=None) -> None:
    demo_args, remaining_argv = parse_demo_args(argv)
    args = parse_args(remaining_argv)
    controller = build_controller(args)
    controller.start()

    t0 = time.monotonic()
    phases: list[dict] = []
    qwen_calls: list[dict] = []

    def log_phase(name: str) -> None:
        entry = {
            "phase": name,
            "t_s": round(time.monotonic() - t0, 3),
            "pose": _pose_dict(controller.display.pose),
        }
        phases.append(entry)
        print(f"[bench] {entry}")

    try:
        print("[demo] waiting for first observation...")
        _wait_until(lambda: controller.display.pose is not None, 0.1)
        log_phase("start")

        print(
            f"[demo] driving straight ({demo_args.forward_distance_m}m)"
            + (
                ""
                if demo_args.baseline
                else f", watching for {demo_args.target_text!r} every "
                f"{demo_args.poll_interval_s}s"
            )
        )
        controller.request_forward_point(demo_args.forward_distance_m)
        log_phase("forward_leg_start")

        found = False
        if demo_args.baseline:
            _wait_until(lambda: controller.display.goal_reached, demo_args.poll_interval_s)
            log_phase("forward_leg_reached")
        else:
            deadline = time.monotonic() + demo_args.find_timeout_s
            while not found and time.monotonic() < deadline:
                if controller.display.goal_reached:
                    print("[demo] reached the forward point without sighting the target")
                    log_phase("forward_leg_reached_no_target")
                    break
                time.sleep(demo_args.poll_interval_s)
                frame = controller.display.vis_rgb
                if frame is None:
                    continue
                call_t0 = time.monotonic()
                point_norm, vlm_result = qwen_client.ground_object_verbose(
                    frame, demo_args.target_text
                )
                qwen_calls.append(
                    {
                        "t_s": round(call_t0 - t0, 3),
                        "latency_s": round(time.monotonic() - call_t0, 3),
                        "found": point_norm is not None,
                    }
                )
                print(f"[demo] qwen ground_object({demo_args.target_text!r}) -> {vlm_result}")
                if point_norm is not None:
                    u, v = point_norm
                    print(f"[demo] found {demo_args.target_text!r} at ({u:.3f}, {v:.3f}), driving to it")
                    controller.request_pixel_goal(u, v)
                    found = True
                    log_phase("target_found")

            if found:
                print("[demo] waiting to reach target...")
                _wait_until(lambda: controller.display.goal_reached, demo_args.poll_interval_s)
                log_phase("target_reached")

        if demo_args.baseline or found:
            print(f"[demo] turning {demo_args.turn_direction}")
            controller.request_turn(demo_args.turn_direction)
            _wait_until(lambda: controller.display.goal_reached, demo_args.poll_interval_s)
            log_phase("turn_complete")
            print("[demo] done: completed turn" + ("" if demo_args.baseline else " after finding target"))
        else:
            print("[demo] done: target never found within forward leg / timeout")
            log_phase("done_not_found")
    finally:
        controller.shutdown()

    if demo_args.out_json:
        record = {
            "mode": "baseline" if demo_args.baseline else "target",
            "target_text": None if demo_args.baseline else demo_args.target_text,
            "forward_distance_m": demo_args.forward_distance_m,
            "turn_direction": demo_args.turn_direction,
            "found": found if not demo_args.baseline else None,
            "phases": phases,
            "qwen_calls": qwen_calls,
        }
        with open(demo_args.out_json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[demo] wrote benchmark record to {demo_args.out_json}")


if __name__ == "__main__":
    main()
