"""Minimal raw habitat_sim RGBD sensor smoke test: no rover/teleop scaffolding,
just spawns RGB+depth sensors on the scene and saves one frame to check the
sim/scene loads and sensors render.

Usage:
    python scripts/habitat_tests/rgbd_test.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import habitat_sim
from habitat_sim.agent import AgentConfiguration

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "marsyard2022_tri.glb")
OUT_DIR = "mars_rgbd_test_out"


def make_sensor(uuid, sensor_type):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [480, 640]
    spec.position = [0.0, 0.0, 0.0]
    return spec


def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.enable_physics = False

    rgb = make_sensor("rgb", habitat_sim.SensorType.COLOR)
    depth = make_sensor("depth", habitat_sim.SensorType.DEPTH)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def save_obs(obs, name):
    os.makedirs(OUT_DIR, exist_ok=True)

    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    Image.fromarray(rgb.astype(np.uint8)).save(f"{OUT_DIR}/{name}_rgb.png")

    depth = obs["depth"]
    depth_clip = np.clip(depth, 0.0, 20.0)
    depth_vis = (depth_clip / 20.0 * 255.0).astype(np.uint8)
    Image.fromarray(depth_vis).save(f"{OUT_DIR}/{name}_depth.png")

    print("RGB shape:", rgb.shape)
    print("Depth min/max:", float(np.min(depth)), float(np.max(depth)))


def main():
    sim = make_sim()
    agent = sim.initialize_agent(0)

    # Habitat uses Y-up.
    # Position format: [X, Y, Z]
    # Start above/away from terrain, looking toward -Z by default.
    test_positions = [
        np.array([0.0, 3.0, 15.0]),
        np.array([0.0, 6.0, 20.0]),
        np.array([-10.0, 4.0, 10.0]),
        np.array([10.0, 4.0, 10.0]),
    ]

    for i, pos in enumerate(test_positions):
        state = agent.get_state()
        state.position = pos
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        save_obs(obs, f"pose_{i:02d}")

        print("Saved pose", i, "at", pos)

    sim.close()
    print("Done. Check:", OUT_DIR)


if __name__ == "__main__":
    main()
