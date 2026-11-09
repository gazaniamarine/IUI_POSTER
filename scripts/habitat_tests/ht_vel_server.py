"""UDP velocity-command server variant of kb_teleop.py: renders the same
teleop window but drives the rover from (linear_x, angular_y) commands
received over UDP instead of keyboard input, so another process (e.g.
ht_vel_client.py) can drive it.

Usage:
    python scripts/habitat_tests/ht_vel_server.py    # listens on 127.0.0.1:5055
"""

import os
import shutil
import socket
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk

import habitat_sim
from habitat_sim.agent import AgentConfiguration
import quaternion

HERE = Path(__file__).resolve().parent

SCENE = str(HERE / "marsyard2022_tri.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm.png")

OUT_DIR = f"mars_ht_vel_out{int(time.time())}"

# UDP server
UDP_HOST = "127.0.0.1"
UDP_PORT = 5055

# Terrain scale
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566

# Start pose
START_X = 0.0
START_Z = 8.0
START_YAW_DEG = 0.0

# Simulation update
UPDATE_HZ = 30.0

# Velocity limits
MAX_LINEAR_X = 1.0  # m/s
MAX_ANGULAR_Y = 1.5  # rad/s

# Deadband
LINEAR_DEADBAND = 0.01
ANGULAR_DEADBAND = 0.01

# Autostop
AUTOSTOP_ON_TIMEOUT = True
COMMAND_TIMEOUT_SEC = 0.5

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
SAVE_FRAME_WHILE_RECORDING = True
SAVE_FRAME_ON_RECORDING_START = True

# Display
SHOW_DEPTH_BESIDE_RGB = True
DEPTH_VIS_MAX_METERS = 10.0
RGBD_RESOLUTION = [480, 640]

# Heightmap correction
FLIP_HEIGHTMAP_X = False
FLIP_HEIGHTMAP_Z = True
SWAP_HEIGHTMAP_XZ = False

DT_MS = int(1000.0 / UPDATE_HZ)


def load_heightmap(path):
    img = Image.open(path)
    arr = np.array(img)

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


class HabitatVelocityServer:
    def __init__(self):
        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)
        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)
        self.clearance = INITIAL_CLEARANCE
        self.linear_x = 0.0
        self.angular_y = 0.0
        self.last_command_time = 0.0
        self.recording = START_RECORDING
        self.frame_idx = 0
        self.recorded = False
        self.closed = False
        os.makedirs(OUT_DIR, exist_ok=True)
        poses_path = f"{OUT_DIR}/poses.txt"
        if os.path.exists(poses_path):
            os.remove(poses_path)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)
        self.root = tk.Tk()
        self.root.title("bambulab")
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text=(
                "UDP ht_vel server | SPACE record | P save | " "Q/E height | X/ESC quit"
            ),
            font=("Arial", 12),
        )
        self.info_label.pack()
        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.set_agent_pose()
        self.render()
        print("\nserver is up and listening")
        print(f"Listening on UDP {UDP_HOST}:{UDP_PORT}")
        print("Commands:")
        print("  ht_vel <lin_x> <ang_y>")
        print("  ht_stop")
        print("  ht_rec on")
        print("  ht_rec off")
        print("  ht_rec toggle")
        print("  ht_save")
        print("  ht_clearance <meters>")
        print("  ht_quit \n")
        self.root.after(DT_MS, self.update_loop)

    def parse_command(self, text):
        parts = text.strip().split()
        if not parts:
            return
        cmd = parts[0].lower()
        if cmd == "ht_vel":
            if len(parts) != 3:
                print(
                    "[e] bad ht_vel command. suggested usage: ht_vel <linear_x> <angular_y>"
                )
                return
            lin = float(parts[1])
            ang = float(parts[2])
            lin = float(np.clip(lin, -MAX_LINEAR_X, MAX_LINEAR_X))
            ang = float(np.clip(ang, -MAX_ANGULAR_Y, MAX_ANGULAR_Y))
            if abs(lin) < LINEAR_DEADBAND:
                lin = 0.0
            if abs(ang) < ANGULAR_DEADBAND:
                ang = 0.0
            self.linear_x = lin
            self.angular_y = ang
            self.last_command_time = time.time()

        elif cmd == "ht_stop":
            self.linear_x = 0.0
            self.angular_y = 0.0
            self.last_command_time = time.time()
            print("[!] Stopped")

        elif cmd == "ht_rec":
            if len(parts) != 2:
                print("[e] bad ht_rec command. suggested usage: ht_rec on/off/toggle")
                return

            mode = parts[1].lower()

            if mode == "on":
                if not self.recording:
                    self.recording = True
                    print("[!] Recording ON")
                    if SAVE_FRAME_ON_RECORDING_START:
                        self.save_current_frame()

            elif mode == "off":
                self.recording = False
                print("[!] Recording OFF")

            elif mode == "toggle":
                self.recording = not self.recording
                print(f"[!] recording: {'ON' if self.recording else 'OFF'}")
                if self.recording and SAVE_FRAME_ON_RECORDING_START:
                    self.save_current_frame()

        elif cmd == "ht_save":
            self.save_current_frame()

        elif cmd == "ht_clearance":
            if len(parts) != 2:
                print(
                    "[e] bad ht_clearance command. suggested usage: ht_clearance <meters>"
                )
                return

            self.clearance = float(parts[1])
            self.clearance = float(
                np.clip(self.clearance, MIN_CLEARANCE, MAX_CLEARANCE)
            )
            self.set_agent_pose()
            self.render()
            print(f"[!] clearance={self.clearance:.2f}")

        elif cmd == "ht_quit":
            self.close()

        else:
            print(f"[e] Unknown command: {text}")

    def poll_udp(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break

            try:
                text = data.decode("utf-8").strip()
                self.parse_command(text)
            except Exception as e:
                print(f"[e] Bad UDP command: {e}")

    def apply_velocity(self, dt):
        now = time.time()

        if AUTOSTOP_ON_TIMEOUT and self.last_command_time > 0:
            if now - self.last_command_time > COMMAND_TIMEOUT_SEC:
                self.linear_x = 0.0
                self.angular_y = 0.0

        old_x = self.x
        old_z = self.z

        # Habitat is Y-up:
        # +linear_x  = forward
        # +angular_y = turn left
        self.yaw += self.angular_y * dt

        self.x += -np.sin(self.yaw) * self.linear_x * dt
        self.z += -np.cos(self.yaw) * self.linear_x * dt

        self.x, self.z = apply_boundary(self.x, self.z, old_x, old_z)

        moved = (
            abs(self.x - old_x) > 1e-6
            or abs(self.z - old_z) > 1e-6
            or abs(self.angular_y) > 1e-6
        )

        return moved

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

        cmd_age = (
            time.time() - self.last_command_time
            if self.last_command_time > 0
            else 999.0
        )

        status = (
            f"x={self.x:.2f} y={self.y:.2f} z={self.z:.2f} "
            f"terrain={self.terrain_y:.2f} "
            f"yaw={np.rad2deg(self.yaw):.1f} "
            f"lin_x={self.linear_x:.2f} ang_y={self.angular_y:.2f} "
            f"cmd_age={cmd_age:.2f}s "
            f"REC={'ON' if self.recording else 'OFF'}"
        )

        controls = (
            "UDP: ht_vel lin_x ang_y | SPACE rec | " "P save | Q/E height | X quit"
        )

        draw.rectangle([0, 0, img.width, 60], fill=(0, 0, 0))
        draw.text((10, 8), status, fill=(255, 255, 255))
        draw.text((10, 32), controls, fill=(255, 255, 255))

        if self.recording:
            draw.ellipse([10, 70, 30, 90], fill=(255, 0, 0))
            draw.text((38, 71), "RECORDING", fill=(255, 0, 0))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def save_current_frame(self):
        os.makedirs(OUT_DIR, exist_ok=True)

        rgb, depth_vis, _ = rgb_depth_from_obs(self.latest_obs)

        Image.fromarray(rgb).save(f"{OUT_DIR}/rgb_{self.frame_idx:04d}.png")
        Image.fromarray(depth_vis).save(f"{OUT_DIR}/depth_{self.frame_idx:04d}.png")

        with open(f"{OUT_DIR}/poses.txt", "a") as f:
            f.write(
                f"{self.frame_idx:04d} "
                f"x={self.x:.4f} y={self.y:.4f} z={self.z:.4f} "
                f"yaw_rad={self.yaw:.4f} yaw_deg={np.rad2deg(self.yaw):.2f} "
                f"linear_x={self.linear_x:.4f} "
                f"angular_y={self.angular_y:.4f} "
                f"clearance={self.clearance:.4f} "
                f"recording={int(self.recording)}\n"
            )

        self.recorded = True
        print(f"[!] saved frame {self.frame_idx:04d}")
        self.frame_idx += 1

    def update_loop(self):
        if self.closed:
            return

        self.poll_udp()

        dt = 1.0 / UPDATE_HZ
        moved = self.apply_velocity(dt)

        self.set_agent_pose()
        self.render()

        if self.recording and SAVE_FRAME_WHILE_RECORDING and moved:
            self.save_current_frame()

        self.root.after(DT_MS, self.update_loop)

    def on_key(self, event):
        key = event.keysym.lower()

        if key == "x" or key == "escape":
            self.close()
            return

        elif key == "space":
            self.recording = not self.recording
            print(f"[!] recording: {'ON' if self.recording else 'OFF'}")

            if self.recording and SAVE_FRAME_ON_RECORDING_START:
                self.save_current_frame()

        elif key == "p":
            self.save_current_frame()

        elif key == "q":
            self.clearance = max(MIN_CLEARANCE, self.clearance - CLEARANCE_STEP)
            self.set_agent_pose()
            self.render()

        elif key == "e":
            self.clearance = min(MAX_CLEARANCE, self.clearance + CLEARANCE_STEP)
            self.set_agent_pose()
            self.render()

    def close(self):
        if self.closed:
            return

        self.closed = True

        try:
            self.sock.close()
        except Exception:
            pass

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
    app = HabitatVelocityServer()
    app.run()
