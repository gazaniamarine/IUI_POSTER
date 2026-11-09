"""SAM interactive visualizer for the Mars rover scene: fly a free camera
(WASD + turn), press SPACE to capture the current frame and run it through
the trained SAM2 segmentation model (sam/sam/inference.py); the detected
segmentation (class overlay + bedrock/big_rock boxes) opens in a second
window with an option to save.

Usage:
    python scripts/vlm_nav_tests/vlm_nav_demo.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk
from tkinter import messagebox

import cv2
import torch
import habitat_sim
from habitat_sim.agent import AgentConfiguration
import quaternion

HERE = Path(__file__).resolve().parent

# sam/sam/*.py use unqualified imports of each other (e.g. inference.py
# does `from evaluate_sam2_simple_fast import ...`), so that directory
# must be on sys.path rather than imported as a package.
SAM_DIR = HERE / "sam" / "sam"
if str(SAM_DIR) not in sys.path:
    sys.path.insert(0, str(SAM_DIR))

from inference import (  # noqa: E402
    load_best_model,
    create_filtered_mask,
    extract_bounding_boxes,
    BEDROCK_CLASS,
    BIGROCK_CLASS,
)
from evaluate_sam2_simple_fast import (  # noqa: E402
    preprocess_image_for_model,
    colorize_mask,
    overlay_masks,
    CLASS_COLORS,
    DEVICE,
)

SCENE = str(HERE / "marsyard2022_tri.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm.png")

OUT_DIR = f"sam_visualizer_out_{int(time.time())}"

# Terrain scale
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566

# Start pose
START_X = 12.2
START_Z = 10.0
START_YAW_DEG = 10.0

# Free-camera movement
MOVE_STEP = 0.35
TURN_STEP_DEG = 10.0

# Camera height above terrain
INITIAL_CLEARANCE = 0.9
CLEARANCE_STEP = 0.1
MIN_CLEARANCE = 0.25
MAX_CLEARANCE = 3.0

# Bounds
BOUNDARY_LIMIT = 24.0

# Display
RGBD_RESOLUTION = [480, 640]

# Heightmap correction
FLIP_HEIGHTMAP_X = False
FLIP_HEIGHTMAP_Z = True
SWAP_HEIGHTMAP_XZ = False

TURN_STEP = np.deg2rad(TURN_STEP_DEG)

# --- Go-to-goal controller (retained for vlm_nav_interactive.py) ---------
# vlm_nav_demo.py used to be the standalone nav demo; it's now the SAM
# visualizer, but vlm_nav_interactive.py still imports GoToGoalController,
# DT, and MAX_STEPS from here, so those stay defined in this module.

# Controller parameters
HEADING_ERROR_THRESHOLD = np.deg2rad(15.0)  # rotate in place if error > 15°
MAX_LINEAR_VELOCITY = 0.8  # m/s
MAX_ANGULAR_VELOCITY = np.deg2rad(45.0)  # rad/s
PROPORTIONAL_GAIN_HEADING = 1.0  # heading correction gain while driving
PROPORTIONAL_GAIN_ANGULAR = 2.0  # rotation-only gain when heading error large
STANDOFF_DISTANCE = 0.5  # Stop this distance from target

# Simulation
DT = 0.1  # time step (seconds)
MAX_STEPS = 1000  # Max simulation steps


class GoToGoalController:
    """Proportional go-to-goal controller for rover navigation."""

    def __init__(self, target_x, target_y, target_z):
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.at_target = False

    def update(self, rover_x, rover_y, rover_z, rover_yaw, debug=False):
        """
        Compute velocity command to reach target.

        Returns:
            (linear_x, angular_y): forward velocity (m/s) and rotation rate (rad/s)
            (distance, bearing_error): diagnostic info
        """
        # Compute target in rover's local frame
        dx = self.target_x - rover_x
        dz = self.target_z - rover_z
        distance = np.sqrt(dx**2 + dz**2)

        # Bearing to target in global frame
        # Note: negated to match velocity integration convention (positive yaw = left turn)
        target_bearing = -np.arctan2(dx, -dz)

        if debug:
            print(
                f"[DEBUG] rover: ({rover_x:.2f}, {rover_z:.2f}) yaw={np.rad2deg(rover_yaw):.1f}°"
            )
            print(f"[DEBUG] target: ({self.target_x:.2f}, {self.target_z:.2f})")
            print(f"[DEBUG] delta: dx={dx:.2f}, dz={dz:.2f}")
            print(
                f"[DEBUG] bearing={np.rad2deg(target_bearing):.1f}°, distance={distance:.2f}m"
            )

        # Heading error: wrap to [-pi, pi]
        heading_error = target_bearing - rover_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        linear_x = 0.0
        angular_y = 0.0

        # Stop if at target
        if distance <= STANDOFF_DISTANCE:
            self.at_target = True
            linear_x = 0.0
            angular_y = 0.0
        # Rotate in place if heading error is large
        elif abs(heading_error) > HEADING_ERROR_THRESHOLD:
            linear_x = 0.0
            angular_y = (
                np.sign(heading_error)
                * MAX_ANGULAR_VELOCITY
                * PROPORTIONAL_GAIN_ANGULAR
            )
        # Drive forward with heading correction
        else:
            linear_x = MAX_LINEAR_VELOCITY
            angular_y = heading_error * MAX_ANGULAR_VELOCITY * PROPORTIONAL_GAIN_HEADING

        return linear_x, angular_y, distance, heading_error

    def obstacle_avoidance_hook(self, linear_x, angular_y, depth_frame):
        """
        Placeholder for obstacle avoidance logic.

        This will be implemented in a follow-up task. For now, returns velocities unchanged.

        Args:
            linear_x, angular_y: commanded velocities from go-to-goal
            depth_frame: depth image from current frame

        Returns:
            (linear_x, angular_y): potentially modified velocities
        """
        # TODO: Implement obstacle avoidance here
        # - Compute obstacle map from depth frame (e.g., distance-to-obstacle in forward direction)
        # - If obstacle ahead, reduce linear_x or adjust angular_y to steer around
        # - Log obstacle detections for debugging
        return linear_x, angular_y


def load_heightmap(path):
    """Load and normalize heightmap from PNG."""
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
    """Bilinear interpolation of terrain height at (x, z)."""
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
    """Create a camera sensor spec (RGB only - this rig is a camera, not a rover)."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = RGBD_RESOLUTION
    spec.position = [0.0, 0.0, 0.0]
    return spec


def make_sim():
    """Initialize Habitat simulator with scene and an RGB sensor."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.enable_physics = False

    rgb = make_sensor("rgb", habitat_sim.SensorType.COLOR)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def rgb_from_obs(obs):
    """Extract RGB frame from sensor observation."""
    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def apply_boundary(x, z, old_x, old_z):
    inside = (
        -BOUNDARY_LIMIT <= x <= BOUNDARY_LIMIT
        and -BOUNDARY_LIMIT <= z <= BOUNDARY_LIMIT
    )
    if inside:
        return x, z
    return old_x, old_z


def run_segmentation(rgb_frame, model):
    """
    Run the SAM2 model on an RGB frame.

    Returns a dict with the full-resolution class overlay (RGB uint8)
    and bedrock/big_rock bounding boxes, mirroring the per-frame
    pipeline in sam/sam/inference.py::process_video.
    """
    height, width = rgb_frame.shape[:2]
    bgr_frame = rgb_frame[..., ::-1]
    img_tensor = preprocess_image_for_model(bgr_frame)

    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(DEVICE)
        logits = model(inp)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    pred_resized = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

    color_mask = colorize_mask(pred_resized)
    overlay = overlay_masks(rgb_frame, color_mask, alpha=0.5)

    filtered_mask = create_filtered_mask(pred_resized, rgb_frame.shape)
    bedrock_boxes = extract_bounding_boxes(filtered_mask, BEDROCK_CLASS)
    bigrock_boxes = extract_bounding_boxes(filtered_mask, BIGROCK_CLASS)

    return {
        "overlay": overlay,
        "bedrock_boxes": bedrock_boxes,
        "bigrock_boxes": bigrock_boxes,
    }


class SegmentationResultWindow:
    """Shows a captured frame's segmentation result, with a save option."""

    def __init__(self, parent, rgb_frame, result, pose):
        self.rgb_frame = rgb_frame
        self.result = result
        self.pose = pose

        overlay_img = Image.fromarray(result["overlay"])
        draw = ImageDraw.Draw(overlay_img)

        for box in result["bedrock_boxes"]:
            self._draw_box(draw, box, CLASS_COLORS[BEDROCK_CLASS])
        for box in result["bigrock_boxes"]:
            self._draw_box(draw, box, CLASS_COLORS[BIGROCK_CLASS])

        self.overlay_img = overlay_img

        self.win = tk.Toplevel(parent)
        self.win.title("SAM Segmentation Result")

        self.tk_img = ImageTk.PhotoImage(overlay_img)
        tk.Label(self.win, image=self.tk_img).pack()

        n_bedrock = len(result["bedrock_boxes"])
        n_bigrock = len(result["bigrock_boxes"])
        tk.Label(
            self.win,
            text=f"bedrock: {n_bedrock}  |  big_rock: {n_bigrock}",
            font=("Arial", 11),
        ).pack(pady=(4, 0))

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Close", command=self.win.destroy).pack(
            side=tk.LEFT, padx=5
        )

        self.win.bind("<KeyPress-s>", lambda e: self.save())

    @staticmethod
    def _draw_box(draw, box, color):
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text(
            (x, max(0, y - 12)), f"{box['class_name']} ({box['area']})", fill=color
        )

    def save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        ts = int(time.time() * 1000)

        Image.fromarray(self.rgb_frame).save(f"{OUT_DIR}/rgb_{ts}.png")
        self.overlay_img.save(f"{OUT_DIR}/seg_{ts}.png")

        with open(f"{OUT_DIR}/boxes_{ts}.json", "w") as f:
            json.dump(
                {
                    "pose": self.pose,
                    "bedrock": self.result["bedrock_boxes"],
                    "bigrock": self.result["bigrock_boxes"],
                },
                f,
                indent=2,
            )

        print(f"[inf] Saved capture to {OUT_DIR}/ (ts={ts})")
        messagebox.showinfo("Saved", f"Saved to {OUT_DIR}/ (ts={ts})")


class MarsCameraApp:
    """Free-fly camera over the Mars scene, with click-to-segment via SAM."""

    def __init__(self):
        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)

        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)
        self.clearance = INITIAL_CLEARANCE

        self.sam_model = None
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Marsyard Habitat - SAM Interactive Visualizer")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="W/S move | A/D turn | Q/E height | SPACE segment | X quit",
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

    def render(self, status_override=None):
        obs = self.sim.get_sensor_observations()
        self.latest_rgb = rgb_from_obs(obs)

        img = Image.fromarray(self.latest_rgb)
        draw = ImageDraw.Draw(img)

        status = status_override or (
            f"x={self.x:.2f}  z={self.z:.2f}  yaw={np.rad2deg(self.yaw):.1f}"
        )

        draw.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
        draw.text((6, 4), status, fill=(255, 255, 255))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def ensure_model_loaded(self):
        if self.sam_model is None:
            self.render(status_override="Loading SAM model (first run)...")
            self.root.update()
            print("[inf] Loading SAM2 segmentation model...")
            self.sam_model = load_best_model()
            print("[inf] Model loaded.")

    def capture_and_segment(self):
        rgb = self.latest_rgb.copy()
        pose = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw_deg": float(np.rad2deg(self.yaw)),
        }

        try:
            self.ensure_model_loaded()
            self.render(status_override="Running SAM segmentation...")
            self.root.update()
            result = run_segmentation(rgb, self.sam_model)
        except Exception as e:
            print(f"[err] Segmentation failed: {e}")
            messagebox.showerror("Segmentation failed", str(e))
            self.render()
            return

        self.render()
        SegmentationResultWindow(self.root, rgb, result, pose)

    def on_key(self, event):
        key = event.keysym.lower()

        old_x = self.x
        old_z = self.z

        if key == "x" or key == "escape":
            self.close()
            return

        elif key == "space":
            self.capture_and_segment()
            return

        elif key == "w":
            self.x += -np.sin(self.yaw) * MOVE_STEP
            self.z += -np.cos(self.yaw) * MOVE_STEP

        elif key == "s":
            self.x -= -np.sin(self.yaw) * MOVE_STEP
            self.z -= -np.cos(self.yaw) * MOVE_STEP

        elif key == "a":
            self.yaw += TURN_STEP

        elif key == "d":
            self.yaw -= TURN_STEP

        elif key == "q":
            self.clearance = max(MIN_CLEARANCE, self.clearance - CLEARANCE_STEP)

        elif key == "e":
            self.clearance = min(MAX_CLEARANCE, self.clearance + CLEARANCE_STEP)

        else:
            return

        self.x, self.z = apply_boundary(self.x, self.z, old_x, old_z)

        self.set_agent_pose()
        self.render()

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

        print("Done.")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MarsCameraApp()
    app.run()
