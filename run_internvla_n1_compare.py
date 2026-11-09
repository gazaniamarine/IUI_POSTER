"""
One-off trajectory-quality comparison: drive the SAME HM3D scene/start-pose/
instruction through the real InternVLA-N1-DualVLN model instead of this
project's own Qwen3.5+NavDP pipeline, logging a comparable pose-over-time
trace. InternVLA-N1 runs in a separate `internvla` conda env (python3.9,
torch2.8+cu128 for this machine's Blackwell GPU) via internvla_n1_server.py
(/home/gpu/Desktop/pineapple/InternNav/internvla_n1_server.py) -- this script
is the client + sim driver, run under the `habitat` env:

    conda activate habitat
    cd mars-habitatsim
    python run_internvla_n1_compare.py

Action interpretation:
  - output_action (discrete VLN-CE tokens: STOP/up/left/right/down): applied
    one at a time using the SAME 0.25m forward-step / 15deg turn-angle
    convention InternNav's own discrete-action evaluation uses (see
    internnav/model/utils/vln_utils.py's trajectory_to_discrete_actions_
    close_to_goal default args) -- the model was trained against this action
    space, so these units aren't a guess.
  - output_trajectory (continuous body-frame waypoints from System-1): the
    last waypoint's (x, y, theta) is converted to (linear_vel, angular_vel)
    exactly as internvla_n1_agent_realworld.py's own trajectory_tovw does
    (kp=1.0, linear clipped to [0, 0.5], angular to [-0.5, 0.5]), then held
    for PLAN_STEP_GAP ticks before the next server call -- matching the
    model's own re-plan cadence rather than an arbitrary duration.

Both action types are integrated into world pose via sam_vla.core.
pose_integrator.integrate_mars, the exact same kinematics
nav/rover_controller.py's own tick loop uses -- so any difference between
this run's trajectory and this project's own logged runs reflects the
driving *decisions* the two systems made, not a different motion model.
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action, Pose
from sam_vla.env.habitat_env import MarsHabitatEnv

HERE = Path(__file__).resolve().parent

SCENE_PATH = str(
    HERE
    / "assets/hm3d/versioned_data/hm3d-0.2/hm3d/minival/00802-wcojb4TFT35/wcojb4TFT35.basis.glb"
)
SCENE_DATASET_CONFIG = str(
    HERE / "assets/hm3d/versioned_data/hm3d-0.2/hm3d/minival/hm3d_minival_basis.scene_dataset_config.json"
)
START_X, START_Z, START_YAW_DEG = -0.71, -6.09, 0.0
INSTRUCTION = sys.argv[1] if len(sys.argv) > 1 else "go straight until you find the door and turn left"
LOG_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "default"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8791
_HEADER_SIZE = 4

DT = 0.1  # matches nav/rover_controller.py's own default --dt
PLAN_STEP_GAP = 4  # matches internvla_n1_server.py's Args.plan_step_gap
DISCRETE_FORWARD_M = 0.25
DISCRETE_TURN_DEG = 15.0
DISCRETE_STEP_TICKS = 5  # ticks to spend integrating one discrete action, for a smooth (not teleported) trace
MAX_TICKS = 900  # 90s of sim time at DT=0.1 -- matches a typical R2R-CE-style episode time budget
OUT_LOG = HERE / "nav" / "logs" / f"internvla_n1_compare_{LOG_SUFFIX}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def _recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed before expected bytes were received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_request(mode: str, payload: dict, timeout: float = 60.0) -> dict:
    message = json.dumps({"mode": mode, "payload": payload}).encode("utf-8")
    with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=timeout) as conn:
        conn.sendall(struct.pack(">I", len(message)) + message)
        header = _recv_exact(conn, _HEADER_SIZE)
        (body_len,) = struct.unpack(">I", header)
        body = _recv_exact(conn, body_len)
    response = json.loads(body.decode("utf-8"))
    if "error" in response:
        raise RuntimeError(f"internvla_n1_server error: {response['error']}")
    return response


def _encode_rgb(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_depth(depth: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(depth, dtype=np.float32).tobytes()).decode("ascii")


def _pose_to_4x4(pose: Pose) -> list:
    # Habitat-sim's own yaw convention (matches sam_vla.env.sim_utils.
    # set_agent_pose): rotation about +Y by `yaw` via quaternion.
    # from_rotation_vector([0, yaw, 0]). Row-major 4x4, translation in the
    # last column -- this agent mostly uses pose for bookkeeping/logging
    # rather than fine-grained control, so exact convention match matters
    # less than internal consistency across calls, which this gives.
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return [
        [c, 0.0, s, pose.x],
        [0.0, 1.0, 0.0, pose.y],
        [-s, 0.0, c, pose.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


ACTIONS2IDX_INV = {0: "STOP", 1: "forward", 2: "left", 3: "right", 5: "down"}


def _apply_discrete_action(pose: Pose, token: int, log_fn) -> Pose:
    name = ACTIONS2IDX_INV.get(token, f"unknown({token})")
    if name == "STOP":
        return pose
    if name == "forward":
        action = Action(v_fwd=DISCRETE_FORWARD_M / (DISCRETE_STEP_TICKS * DT), v_lat=0.0, yaw_rate=0.0)
    elif name == "left":
        action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=math.radians(DISCRETE_TURN_DEG) / (DISCRETE_STEP_TICKS * DT))
    elif name == "right":
        action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=-math.radians(DISCRETE_TURN_DEG) / (DISCRETE_STEP_TICKS * DT))
    else:
        # "down" (look-down, real-world-only camera tilt) -- no sim analogue, no-op.
        return pose
    for _ in range(DISCRETE_STEP_TICKS):
        pose = integrate_mars(pose, action, DT)
        log_fn(pose)
    return pose


def _apply_trajectory(pose: Pose, trajectory, log_fn) -> Pose:
    # traj_to_actions(..., use_discrate_action=False) -> vln_utils.py's
    # reconstruct_xy_from_delta returns [T+1, 2] CUMULATIVE (x, y) body-frame
    # positions only (see its own docstring) -- no separate heading
    # component, unlike internvla_n1_agent_realworld.py's own (unused by
    # step()) trajectory_tovw helper, which assumes a 3rd theta element and
    # was the wrong reference to copy (confirmed live: IndexError on
    # subgoal[2], size-2 waypoints). Derive heading from the goal point's
    # own bearing instead -- same reasoning our own MODE_POINT driving uses
    # (steer toward wherever the point is), not a value the model gives
    # directly for this output shape.
    if not trajectory or len(trajectory) < 2:
        return pose
    subgoal = np.asarray(trajectory[-1], dtype=np.float32)  # skip index 0, the fixed (0,0) start
    linear_vel = float(np.clip(np.linalg.norm(subgoal), 0.0, 0.5))
    bearing = math.atan2(subgoal[1], subgoal[0]) if linear_vel > 1e-6 else 0.0
    angular_vel = float(np.clip(bearing, -0.5, 0.5))
    action = Action(v_fwd=linear_vel, v_lat=0.0, yaw_rate=angular_vel)
    for _ in range(PLAN_STEP_GAP):
        pose = integrate_mars(pose, action, DT)
        log_fn(pose)
    return pose


def main() -> None:
    print("[!] pinging internvla_n1_server...")
    _send_request("ping", {})
    print("[!] server reachable, resetting agent")
    _send_request("reset", {})

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_LOG, "w")

    def log_pose(pose: Pose, event: str = "tick", **extra) -> None:
        rec = {"t": round(time.monotonic() - t0, 3), "x": pose.x, "z": pose.z, "yaw": pose.yaw, "event": event}
        rec.update(extra)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    with MarsHabitatEnv(
        SCENE_PATH,
        None,
        services=[],
        scene_dataset_config=SCENE_DATASET_CONFIG,
        start_x=START_X,
        start_z=START_Z,
        start_yaw=math.radians(START_YAW_DEG),
        spawn_flags=False,
        gen_home_base=False,
        annotations_dir=None,
    ) as env:
        obs = env.get_observation(frame_idx=0)
        pose = obs.pose
        t0 = time.monotonic()
        log_pose(pose, event="start")
        print(f"[!] scene ready, start pose=({pose.x:.3f},{pose.z:.3f},yaw={pose.yaw:.3f}), instruction={INSTRUCTION!r}")

        rgb_sensor = env._agent._sensors["rgb"]
        agent_forward_rotation = rgb_sensor.node.rotation

        def capture_look_down(pitch_deg: float = -40.0):
            # Real InternVLA-N1 deployments physically tilt the camera down
            # on a "down" discrete action and feed the resulting floor-level
            # frame back via agent.step(..., look_down=True) -- see
            # internvla_zenoh_node.py's module docstring. This sim has a
            # fixed-orientation RGB-D sensor, so approximate the same thing
            # by pitching the sensor's SceneNode for one extra render, then
            # restoring it -- confirmed live this produces a genuine
            # floor-level view, not a black/degenerate frame.
            import magnum as mn

            rgb_sensor.node.rotation = mn.Quaternion.rotation(
                mn.Deg(pitch_deg), mn.Vector3.x_axis()
            )
            raw = env._sim.get_sensor_observations()
            rgb_sensor.node.rotation = agent_forward_rotation
            rgb = np.asarray(raw["rgb"])[:, :, :3]
            depth = np.asarray(raw["depth"], dtype=np.float32)
            return rgb, depth

        def query_step(rgb: np.ndarray, depth: np.ndarray, look_down: bool = False) -> dict:
            return _send_request(
                "step",
                {
                    "rgb_b64": _encode_rgb(rgb),
                    "depth_b64": _encode_depth(depth),
                    "height": int(depth.shape[0]),
                    "width": int(depth.shape[1]),
                    "pose": _pose_to_4x4(pose),
                    "instruction": INSTRUCTION,
                    "look_down": look_down,
                },
            )

        tick = 0
        stopped = False
        while tick < MAX_TICKS and not stopped:
            obs = env.get_observation(frame_idx=tick)
            resp = query_step(obs.rgb, obs.depth)

            # A "down" (look-down) request is answered in its own real
            # sub-flow, not by driving -- capture a floor-pitched frame and
            # re-query with look_down=True, per agent.step's own contract
            # (appends to conversation rather than resetting it). Capped at
            # a few retries: a model that keeps requesting look-down even
            # with a real floor view each time is a genuine stuck pattern
            # worth surfacing, not something to loop on forever.
            look_down_retries = 0
            while (
                resp["output_action"] is not None
                and len(resp["output_action"]) == 1
                and resp["output_action"][0] == 5
                and look_down_retries < 3
            ):
                print(f"[tick {tick}] look-down requested (retry {look_down_retries + 1}/3)")
                ld_rgb, ld_depth = capture_look_down()
                resp = query_step(ld_rgb, ld_depth, look_down=True)
                look_down_retries += 1

            def step_and_log(new_pose: Pose) -> None:
                env.step(new_pose)
                log_pose(new_pose)

            if resp["output_action"] is not None:
                tokens = [t for t in resp["output_action"] if t != 5]
                print(f"[tick {tick}] discrete action(s): {[ACTIONS2IDX_INV.get(t, t) for t in tokens]}")
                if not tokens:
                    # Still only look-down after retries -- advance the tick
                    # budget anyway so this can't spin forever without ever
                    # making progress or hitting MAX_TICKS.
                    print(f"[tick {tick}] still only look-down after retries -- holding")
                    tick += DISCRETE_STEP_TICKS
                for tok in tokens:
                    if tok == 0:
                        log_pose(pose, event="stop")
                        stopped = True
                        break
                    pose = _apply_discrete_action(pose, tok, step_and_log)
                    tick += DISCRETE_STEP_TICKS
                    if tick >= MAX_TICKS:
                        break
            elif resp["output_trajectory"] is not None:
                print(f"[tick {tick}] trajectory, {len(resp['output_trajectory'])} waypoints")
                pose = _apply_trajectory(pose, resp["output_trajectory"], step_and_log)
                tick += PLAN_STEP_GAP
            else:
                print(f"[tick {tick}] no actionable output (pixel-goal-only latent stage), holding")
                tick += PLAN_STEP_GAP

        log_pose(pose, event="finished" if not stopped else "stopped")
        print(f"[!] done: {tick} ticks, final pose=({pose.x:.3f},{pose.z:.3f}), log={OUT_LOG}")

    log_f.close()


if __name__ == "__main__":
    main()
