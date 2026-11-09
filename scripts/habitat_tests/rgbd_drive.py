"""Raw habitat_sim smoke test: drives a fixed forward-move sequence while
holding the camera at CAMERA_CLEARANCE above the heightmap-sampled terrain
height, saving RGBD frames per step -- checks terrain-following math without
any teleop/policy scaffolding.

Usage:
    python scripts/habitat_tests/rgbd_drive.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import habitat_sim
from habitat_sim.agent import AgentConfiguration

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "assets" / "marsyard2022.glb")

HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm.png")

OUT_DIR = "mars_follow_terrain_out"

SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566

# Tune this.
# 0.7 to 1.2 = rover camera close to ground.
# 1.5 to 2.0 = safer first pass.
CAMERA_CLEARANCE = 0.5


def load_heightmap(path):
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)

    # Same height math as hm2obj.py
    y = arr * SIZE_Y
    y = y - np.mean(y)

    return y


HEIGHT = load_heightmap(HEIGHTMAP)
HM_H, HM_W = HEIGHT.shape


def terrain_height_at(x, z):
    """
    Habitat world:
      X/Z = ground plane
      Y   = height/up

    Our terrain covers:
      X in [-SIZE_X/2, SIZE_X/2]
      Z in [-SIZE_Z/2, SIZE_Z/2]

    Returns interpolated terrain Y height.
    """

    # Map world x,z to image coordinates.
    u = (x + SIZE_X / 2.0) / SIZE_X
    v = (z + SIZE_Z / 2.0) / SIZE_Z

    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    px = u * (HM_W - 1)
    py = v * (HM_H - 1)

    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, HM_W - 1)
    y1 = min(y0 + 1, HM_H - 1)

    dx = px - x0
    dy = py - y0

    h00 = HEIGHT[y0, x0]
    h10 = HEIGHT[y0, x1]
    h01 = HEIGHT[y1, x0]
    h11 = HEIGHT[y1, x1]

    h0 = h00 * (1.0 - dx) + h10 * dx
    h1 = h01 * (1.0 - dx) + h11 * dx

    return float(h0 * (1.0 - dy) + h1 * dy)


def make_sensor(uuid, sensor_type):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [480, 640]

    # Sensor is at agent origin. We directly control agent height.
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

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def save_obs(obs, idx, pose):
    os.makedirs(OUT_DIR, exist_ok=True)

    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    Image.fromarray(rgb.astype(np.uint8)).save(f"{OUT_DIR}/rgb_{idx:04d}.png")

    depth = obs["depth"]
    depth_clip = np.clip(depth, 0.0, 10.0)
    depth_vis = (depth_clip / 10.0 * 255.0).astype(np.uint8)

    Image.fromarray(depth_vis).save(f"{OUT_DIR}/depth_{idx:04d}.png")

    with open(f"{OUT_DIR}/poses.txt", "a") as f:
        f.write(
            f"{idx:04d} "
            f"x={pose[0]:.4f} y={pose[1]:.4f} z={pose[2]:.4f} "
            f"terrain_y={pose[3]:.4f}\n"
        )

    print(
        f"frame {idx:04d} | "
        f"x={pose[0]:.2f} y={pose[1]:.2f} z={pose[2]:.2f} "
        f"terrain={pose[3]:.2f} "
        f"rgb_mean={rgb.mean():.2f} "
        f"depth_mean={depth.mean():.2f}"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    poses_file = f"{OUT_DIR}/poses.txt"
    if os.path.exists(poses_file):
        os.remove(poses_file)

    sim = make_sim()
    agent = sim.initialize_agent(0)

    # First terrain-following path.
    # Keep inside [-25, 25].
    start_x = -20.0
    end_x = 20.0
    z = 8.0

    num_frames = 120

    for i in range(num_frames):
        t = i / (num_frames - 1)

        x = start_x * (1.0 - t) + end_x * t

        terrain_y = terrain_height_at(x, z)
        y = terrain_y + CAMERA_CLEARANCE

        state = agent.get_state()
        state.position = np.array([x, y, z], dtype=np.float32)
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        save_obs(obs, i, [x, y, z, terrain_y])

    sim.close()

    print("Done. Output:", OUT_DIR)
    print("Pose log:", poses_file)


if __name__ == "__main__":
    main()
