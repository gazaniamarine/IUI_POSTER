"""Base Tkinter WASD teleop UI for driving the rover around marsyard2022 and
recording frames (Space=record, P=snapshot, Q/E=camera clearance). Runs in
the `habitat` conda env.

Usage:
    python scripts/habitat_tests/kb_teleop.py
"""

import os
import time
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk

import habitat_sim
from habitat_sim.agent import AgentConfiguration
import quaternion

HERE = Path(__file__).resolve().parent.parent.parent

SCENE = str(HERE / "assets" / "marsyard2022.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm_1025.tif")

OUT_DIR = f"mars_teleop_out{int(time.time())}"

# Terrain scale
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566

# Start pose
START_X = 0.0
START_Z = 8.0
START_YAW_DEG = 0.0

# Movement
MOVE_STEP = 0.35
TURN_STEP_DEG = 10.0

# Camera height above terrain
INITIAL_CLEARANCE = 0.9
CLEARANCE_STEP = 0.1
MIN_CLEARANCE = 0.25
MAX_CLEARANCE = 3.0

# Bounds
BOUNDARY_LIMIT = 24.0
AUTOSTOP_AT_BOUNDARY = True

# Recording
START_RECORDING = False
SAVE_ON_RECORDING_MOVEMENT_ONLY = True
SAVE_FRAME_ON_RECORDING_START = True

# Display
SHOW_DEPTH_BESIDE_RGB = True
DEPTH_VIS_MAX_METERS = 10.0
RGBD_RESOLUTION = [480, 640]

# Heightmap correction
FLIP_HEIGHTMAP_X = False
FLIP_HEIGHTMAP_Z = True
SWAP_HEIGHTMAP_XZ = False

# ============================================================


TURN_STEP = np.deg2rad(TURN_STEP_DEG)


def _load_heightmap_array(path):
    """Loads raw heightmap pixels. Most heightmaps (PNG, 8/16/32-bit TIFF)
    decode fine via PIL; GeoTIFF DEMs with 64-bit float samples aren't
    supported by Pillow's TIFF codec, so those fall back to rasterio."""
    try:
        return np.array(Image.open(path))
    except Exception:
        if Path(path).suffix.lower() not in (".tif", ".tiff"):
            raise
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                f"PIL could not decode {path} and rasterio is not "
                "installed; install rasterio to load this GeoTIFF heightmap"
            ) from exc
        with rasterio.open(path) as src:
            return src.read(1)


def load_heightmap(path):
    arr = _load_heightmap_array(path)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)

    y = arr * SIZE_Y
    y = y - np.mean(y)

    return y


HEIGHT = load_heightmap(HEIGHTMAP)
HM_H, HM_W = HEIGHT.shape


def terrain_height_at(x, z):
    # Optional swap if GLB export swapped terrain axes
    if SWAP_HEIGHTMAP_XZ:
        x, z = z, x
    u = (x + SIZE_X / 2.0) / SIZE_X
    v = (z + SIZE_Z / 2.0) / SIZE_Z
    if FLIP_HEIGHTMAP_X:
        u = 1.0 - u
    if FLIP_HEIGHTMAP_Z:
        v = 1.0 - v
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
    spec.resolution = RGBD_RESOLUTION
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


def rgb_depth_from_obs(obs):
    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb.astype(np.uint8)

    depth = obs["depth"]
    depth_clip = np.clip(depth, 0.0, DEPTH_VIS_MAX_METERS)
    depth_vis = (depth_clip / DEPTH_VIS_MAX_METERS * 255.0).astype(np.uint8)

    depth_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)

    return rgb, depth_vis, depth_rgb


def save_obs(obs, idx, x, y, z, yaw, clearance, recording):
    os.makedirs(OUT_DIR, exist_ok=True)

    rgb, depth_vis, _ = rgb_depth_from_obs(obs)

    Image.fromarray(rgb).save(f"{OUT_DIR}/rgb_{idx:04d}.png")
    Image.fromarray(depth_vis).save(f"{OUT_DIR}/depth_{idx:04d}.png")

    with open(f"{OUT_DIR}/poses.txt", "a") as f:
        f.write(
            f"{idx:04d} "
            f"x={x:.4f} y={y:.4f} z={z:.4f} "
            f"yaw_rad={yaw:.4f} yaw_deg={np.rad2deg(yaw):.2f} "
            f"clearance={clearance:.4f} "
            f"recording={int(recording)}\n"
        )


def apply_boundary(x, z, old_x, old_z):
    inside = (
        -BOUNDARY_LIMIT <= x <= BOUNDARY_LIMIT
        and -BOUNDARY_LIMIT <= z <= BOUNDARY_LIMIT
    )

    if inside:
        return x, z

    if AUTOSTOP_AT_BOUNDARY:
        return old_x, old_z

    return (
        float(np.clip(x, -BOUNDARY_LIMIT, BOUNDARY_LIMIT)),
        float(np.clip(z, -BOUNDARY_LIMIT, BOUNDARY_LIMIT)),
    )


class MarsTeleopApp:
    def __init__(self):
        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)

        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)
        self.clearance = INITIAL_CLEARANCE

        self.recording = START_RECORDING
        self.frame_idx = 0
        self.recorded = False
        self.closed = False

        os.makedirs(OUT_DIR, exist_ok=True)
        poses_path = f"{OUT_DIR}/poses.txt"
        if os.path.exists(poses_path):
            os.remove(poses_path)

        self.root = tk.Tk()
        self.root.title("Kb Teleop")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="W/S move | A/D turn | Q/E height | SPACE record | P save | X quit",
            font=("Arial", 12),
        )
        self.info_label.pack()

        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.set_agent_pose()
        self.render()

    def set_agent_pose(self):
        self.terrain_y = terrain_height_at(self.x, self.z)
        self.y = self.terrain_y + self.clearance

        state = self.agent.get_state()
        state.position = np.array([self.x, self.y, self.z], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0.0, self.yaw, 0.0])
        self.agent.set_state(state)

    def render(self):
        obs = self.sim.get_sensor_observations()
        self.latest_obs = obs

        rgb, _, depth_rgb = rgb_depth_from_obs(obs)

        if SHOW_DEPTH_BESIDE_RGB:
            img_arr = np.hstack([rgb, depth_rgb])
        else:
            img_arr = rgb

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        status = (
            f"x={self.x:.2f} y={self.y:.2f} z={self.z:.2f} "
            f"terrain={self.terrain_y:.2f} "
            f"yaw={np.rad2deg(self.yaw):.1f} "
            f"clearance={self.clearance:.2f} "
            f"REC={'ON' if self.recording else 'OFF'}"
        )

        draw.rectangle([0, 0, img.width, 55], fill=(0, 0, 0))
        draw.text((10, 8), status, fill=(255, 255, 255))
        draw.text(
            (10, 30),
            "W/S move | A/D turn | Q/E height | SPACE record | P save | X quit",
            fill=(255, 255, 255),
        )

        if self.recording:
            draw.ellipse([10, 65, 30, 85], fill=(255, 0, 0))
            draw.text((38, 66), "RECORDING", fill=(255, 0, 0))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def save_current_frame(self):
        save_obs(
            self.latest_obs,
            self.frame_idx,
            self.x,
            self.y,
            self.z,
            self.yaw,
            self.clearance,
            self.recording,
        )

        self.recorded = True

        print(f"saved frame {self.frame_idx:04d}")
        self.frame_idx += 1

    def on_key(self, event):
        key = event.keysym.lower()

        old_x = self.x
        old_z = self.z
        moved = False

        if key == "x" or key == "escape":
            self.close()
            return

        elif key == "space":
            self.recording = not self.recording
            print(f"Recording {'ON' if self.recording else 'OFF'}")

            if self.recording and SAVE_FRAME_ON_RECORDING_START:
                self.save_current_frame()

        elif key == "w":
            self.x += -np.sin(self.yaw) * MOVE_STEP
            self.z += -np.cos(self.yaw) * MOVE_STEP
            moved = True

        elif key == "s":
            self.x -= -np.sin(self.yaw) * MOVE_STEP
            self.z -= -np.cos(self.yaw) * MOVE_STEP
            moved = True

        elif key == "a":
            self.yaw += TURN_STEP
            moved = True

        elif key == "d":
            self.yaw -= TURN_STEP
            moved = True

        elif key == "q":
            self.clearance = max(MIN_CLEARANCE, self.clearance - CLEARANCE_STEP)
            moved = True

        elif key == "e":
            self.clearance = min(MAX_CLEARANCE, self.clearance + CLEARANCE_STEP)
            moved = True

        elif key == "p":
            self.save_current_frame()

        self.x, self.z = apply_boundary(self.x, self.z, old_x, old_z)

        self.set_agent_pose()
        self.render()

        if self.recording:
            if SAVE_ON_RECORDING_MOVEMENT_ONLY:
                if moved:
                    self.save_current_frame()
            else:
                self.save_current_frame()

    def close(self):
        if self.closed:
            return

        self.closed = True

        try:
            self.sim.close()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

        if self.recorded:
            print(f"Done. Output: {OUT_DIR}")
        else:
            print("Done. No frames recorded.")
            shutil.rmtree(OUT_DIR, ignore_errors=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MarsTeleopApp()
    app.run()
