"""Earliest raw-habitat_sim walk/RGBD scratch script: registers built-in
move/turn ActionSpecs and steps through a fixed action sequence, saving RGBD
frames. Superseded by rgbd_test.py/rgbd_drive.py; kept for reference only.

Usage:
    python scripts/vlm_nav_tests/round1_walk_rgbd.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import habitat_sim
from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "marsyard2022_tri.glb")
OUT_DIR = "round1_outputs"


def make_sensor(uuid, sensor_type, resolution=(480, 640), position=(0.0, 1.5, 0.0)):
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = uuid
    sensor.sensor_type = sensor_type
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = list(resolution)
    sensor.position = list(position)
    return sensor


def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.enable_physics = False

    rgb_sensor = make_sensor("rgb", habitat_sim.SensorType.COLOR)
    depth_sensor = make_sensor("depth", habitat_sim.SensorType.DEPTH)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.35)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=20.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=20.0)),
    }

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def save_obs(obs, idx):
    rgb = obs["rgb"]
    depth = obs["depth"]

    # RGB comes as RGBA sometimes, keep first 3 channels
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    Image.fromarray(rgb.astype(np.uint8)).save(f"{OUT_DIR}/rgb_{idx:03d}.png")

    # Depth is in meters. Clip to 10m and visualize as grayscale.
    depth_clip = np.clip(depth, 0.0, 10.0)
    depth_vis = (depth_clip / 10.0 * 255.0).astype(np.uint8)
    Image.fromarray(depth_vis).save(f"{OUT_DIR}/depth_{idx:03d}.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sim = make_sim()
    sim.initialize_agent(0)

    actions = [
        "move_forward",
        "move_forward",
        "turn_left",
        "move_forward",
        "move_forward",
        "turn_right",
        "move_forward",
        "move_forward",
    ]

    obs = sim.get_sensor_observations()
    save_obs(obs, 0)

    for i, action in enumerate(actions, start=1):
        obs = sim.step(action)
        save_obs(obs, i)
        print(f"Saved frame {i}: action={action}")

    sim.close()
    print(f"Done. Check output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
