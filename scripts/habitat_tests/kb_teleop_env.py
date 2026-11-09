"""Same WASD/space/x teleop UI as kb_teleop.py, but driven through
sam_vla.env.habitat_env.MarsHabitatEnv with the annotation meshes registered
-- the exact sim setup run_segmentation_sweep.py uses, so saved frames
(rgb/, masks_instance/, masks_category/, segmentation_frames.jsonl) are a
drop-in --run-dir for finetune_sam2_lora.py / spot_check_segmentation.

Usage:
    python scripts/habitat_tests/kb_teleop_env.py
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sam_vla.core.types import Pose
from sam_vla.env.habitat_env import MarsHabitatEnv
from sam_vla.env.terrain import SIZE_X, SIZE_Z
from sam_vla.logging.episode_logger import EpisodeLogger, make_run_id
from sam_vla.perception.segmentation_capture import (
    build_category_lut,
    capture_frame_record,
    write_segmentation_assets,
)
from sam_vla.perception.spot_check_segmentation import (
    build_palette,
    overlay_category_mask,
)

SCENE = str(HERE / "assets" / "marsyard2022.glb")
HEIGHTMAP = str(HERE / "marsyard2022_terrain_hm_1025.tif")

OUT_ROOT = "output"  # same root run_segmentation_sweep.py's --out-dir defaults to
ANNOTATIONS_DIR = str(HERE / "annotations" / "mesh_tight_bound2")
CATEGORIES = ["small_rock", "big_rock", "bedrock", "hole_in_ground"]
BACKGROUND_CATEGORY = "background"

# Start pose
START_X = 0.0
START_Z = 8.0
START_YAW_DEG = 0.0

# Movement
MOVE_STEP = 0.35
TURN_STEP_DEG = 10.0

# Bounds -- matches run_segmentation_sweep.py's --boundary-margin default
BOUNDARY_MARGIN = 2.0
AUTOSTOP_AT_BOUNDARY = True

# Recording
START_RECORDING = False
SAVE_ON_RECORDING_MOVEMENT_ONLY = True
SAVE_FRAME_ON_RECORDING_START = True

# Display: RGB | depth | category-mask overlay, side by side
SHOW_DEPTH_BESIDE_RGB = True
SHOW_MASK_OVERLAY = True
DEPTH_VIS_MAX_METERS = 10.0
MASK_OVERLAY_ALPHA = 0.45

# ============================================================

TURN_STEP = np.deg2rad(TURN_STEP_DEG)


def depth_vis_from(depth: np.ndarray) -> np.ndarray:
    depth_clip = np.clip(depth, 0.0, DEPTH_VIS_MAX_METERS)
    depth_vis = (depth_clip / DEPTH_VIS_MAX_METERS * 255.0).astype(np.uint8)
    return np.stack([depth_vis, depth_vis, depth_vis], axis=-1)


def apply_boundary(x, z, old_x, old_z):
    half_x = SIZE_X / 2.0 - BOUNDARY_MARGIN
    half_z = SIZE_Z / 2.0 - BOUNDARY_MARGIN

    inside = -half_x <= x <= half_x and -half_z <= z <= half_z

    if inside:
        return x, z

    if AUTOSTOP_AT_BOUNDARY:
        return old_x, old_z

    return float(np.clip(x, -half_x, half_x)), float(np.clip(z, -half_z, half_z))


class MarsTeleopApp:
    def __init__(self):
        self.env = MarsHabitatEnv(
            SCENE,
            HEIGHTMAP,
            start_x=START_X,
            start_z=START_Z,
            start_yaw=np.deg2rad(START_YAW_DEG),
            with_semantic=True,
            annotations_dir=ANNOTATIONS_DIR,
            annotation_categories=CATEGORIES,
        )
        self.env.__enter__()

        self.lut, self.class_names = build_category_lut(
            self.env.annotation_mesh_id_map, CATEGORIES, BACKGROUND_CATEGORY
        )
        self.palette = build_palette(self.class_names)

        config = {
            "goal_mode": "kb-teleop",
            "steering_mode": "manual",
            "obstacle_count": "na",
            "obstacle_seed": "na",
            "categories": CATEGORIES,
            "background_category": BACKGROUND_CATEGORY,
            "annotations_dir": ANNOTATIONS_DIR,
        }
        run_id = make_run_id(config)
        self.logger = EpisodeLogger(run_id, config, log_root=OUT_ROOT)

        self.x = START_X
        self.z = START_Z
        self.yaw = np.deg2rad(START_YAW_DEG)

        self.recording = START_RECORDING
        self.frame_idx = 0
        self.recorded = False
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Kb Teleop (sweep-matched env + masks)")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="W/S move | A/D turn | SPACE record | P save | X quit",
            font=("Arial", 12),
        )
        self.info_label.pack()

        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.env.step(Pose(x=self.x, y=0.0, z=self.z, yaw=self.yaw))
        self.render()

    def render(self):
        obs = self.env.get_full_observation(self.frame_idx)
        self.latest_obs = obs
        self.pose_y = obs.pose.y

        self.category_mask, self.objects = capture_frame_record(
            obs.rgb,
            obs.semantic,
            self.env.annotation_mesh_id_map,
            self.lut,
            self.class_names,
        )

        panels = [obs.rgb]
        if SHOW_DEPTH_BESIDE_RGB:
            panels.append(depth_vis_from(obs.depth))
        if SHOW_MASK_OVERLAY:
            panels.append(
                overlay_category_mask(
                    obs.rgb,
                    self.category_mask,
                    self.class_names,
                    self.palette,
                    MASK_OVERLAY_ALPHA,
                )
            )
        img_arr = np.hstack(panels)

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        n_objects = len(self.objects)
        status = (
            f"x={self.x:.2f} y={self.pose_y:.2f} z={self.z:.2f} "
            f"yaw={np.rad2deg(self.yaw):.1f} "
            f"objects={n_objects} "
            f"REC={'ON' if self.recording else 'OFF'}"
        )

        draw.rectangle([0, 0, img.width, 55], fill=(0, 0, 0))
        draw.text((10, 8), status, fill=(255, 255, 255))
        draw.text(
            (10, 30),
            "W/S move | A/D turn | SPACE record | P save | X quit",
            fill=(255, 255, 255),
        )

        if self.recording:
            draw.ellipse([10, 65, 30, 85], fill=(255, 0, 0))
            draw.text((38, 66), "RECORDING", fill=(255, 0, 0))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def save_current_frame(self):
        obs = self.latest_obs
        frame_id = f"teleop_{self.frame_idx:06d}"

        paths = write_segmentation_assets(
            self.logger.run_dir,
            frame_id,
            obs.rgb,
            obs.semantic.astype(np.uint16),
            self.category_mask,
        )
        self.logger.log_segmentation_frame(
            frame_id=frame_id,
            objects=[
                {
                    "mesh_id": o.mesh_id,
                    "category": o.category,
                    "pixel_count": o.pixel_count,
                    "bbox": list(o.bbox),
                }
                for o in self.objects
            ],
            camera_pose={
                "x": obs.pose.x,
                "y": obs.pose.y,
                "z": obs.pose.z,
                "yaw": obs.pose.yaw,
            },
            **paths,
        )

        self.recorded = True

        print(f"saved frame {frame_id} ({len(self.objects)} objects)")
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

        elif key == "p":
            self.save_current_frame()

        self.x, self.z = apply_boundary(self.x, self.z, old_x, old_z)

        self.env.step(Pose(x=self.x, y=0.0, z=self.z, yaw=self.yaw))
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

        run_dir = self.logger.run_dir
        self.logger.finalize(
            {
                "total_frames": self.frame_idx,
                "class_names": self.class_names,
                "source": "kb_teleop_env",
            }
        )

        try:
            self.env.__exit__(None, None, None)
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

        if self.recorded:
            print(f"Done. Output: {run_dir}")
        else:
            print("Done. No frames recorded.")
            shutil.rmtree(run_dir, ignore_errors=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MarsTeleopApp()
    app.run()
