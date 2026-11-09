"""Offline top-down replay for a saved rollout.npz (e.g.
runs/navdp_ablation/*/seed_*/rollout.npz) -- re-renders the recorded
episode's trajectory through the real marsyard2022 terrain from the same
fixed bird's-eye camera nav/gui.py's --top-down-viz uses live, with the
Perseverance model marker (MarsHabitatEnv._register_rover_marker) tracking
pose[i] each frame, plus a goal marker, obstacle marker(s), and a
trajectory trail drawn on top. rollout.npz itself is treated as read-only
data (pose, goal_position, obstacle_position(s), goal_distance, hz,
success) -- this never re-runs the policy, only replays where the rover
already was.

Usage:
    python -m nav.rollout_topdown_viewer --rollout runs/.../rollout.npz
        opens an interactive scrubber window (Play/Pause + a frame slider)
    python -m nav.rollout_topdown_viewer --rollout runs/.../rollout.npz --out topdown.mp4
        renders the whole episode straight to an MP4 and exits -- no
        display needed, safe to run headless
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from sam_vla.core.types import Pose
from sam_vla.env.habitat_env import MarsHabitatEnv
from sam_vla.perception.semantic_overlay import draw_point_marker

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = REPO_ROOT / "assets" / "marsyard2022.glb"
DEFAULT_HEIGHTMAP = REPO_ROOT / "marsyard2022_terrain_hm_1025.tif"

GOAL_COLOR = (0, 215, 0)  # gold
OBSTACLE_COLOR = (220, 30, 30)  # red
TRAIL_COLOR = (0, 210, 255)  # cyan


class RolloutData:
    """Parses the handful of rollout.npz fields this viewer needs -- pose
    (N, 7) as world (x, y, z) + quaternion (qx, qy, qz, qw), a single fixed
    goal_position, one or more obstacle_positions, and per-step
    goal_distance/hz/success. Ignores the recorded rgb/depth/goal_mask/
    obstacle_mask/trajectory-candidate arrays entirely -- those are the
    rover's own forward camera and NavDP planning internals, unrelated to a
    top-down replay."""

    def __init__(self, path: Path):
        d = np.load(str(path), allow_pickle=True)
        pose = np.asarray(d["pose"], dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] < 7:
            raise ValueError(
                f"{path}: expected pose array shaped (N, 7) [x,y,z,qx,qy,qz,qw], "
                f"got {pose.shape}"
            )
        self.n = int(pose.shape[0])
        self.x = pose[:, 0]
        self.z = pose[:, 2]
        # Pure-yaw (about world Y) quaternion, scalar-last (qx,qy,qz,qw) --
        # same convention sam_vla.env.sim_utils.set_agent_pose writes via
        # quaternion.from_rotation_vector([0, yaw, 0]), so yaw recovers
        # exactly via 2*atan2(qy, qw).
        self.yaw = 2.0 * np.arctan2(pose[:, 4], pose[:, 6])

        self.goal: Optional[Tuple[float, float, float]] = None
        if "goal_position" in d.files:
            gx, gy, gz = np.asarray(d["goal_position"], dtype=np.float64)
            self.goal = (float(gx), float(gy), float(gz))

        self.obstacles: List[Tuple[float, float, float]] = []
        if "obstacle_positions" in d.files:
            for row in np.asarray(d["obstacle_positions"], dtype=np.float64):
                self.obstacles.append((float(row[0]), float(row[1]), float(row[2])))
        elif "obstacle_position" in d.files:
            ox, oy, oz = np.asarray(d["obstacle_position"], dtype=np.float64)
            self.obstacles.append((float(ox), float(oy), float(oz)))

        self.goal_distance = (
            np.asarray(d["goal_distance"], dtype=np.float64)
            if "goal_distance" in d.files
            else None
        )
        self.hz = float(d["hz"]) if "hz" in d.files else 10.0
        self.success = bool(d["success"]) if "success" in d.files else None


class TopdownRolloutRenderer:
    """Drives MarsHabitatEnv's driving agent through a RolloutData's
    recorded poses one at a time and captures+annotates the fixed top-down
    camera's frame -- the trajectory/goal/obstacle pixel projections are
    computed once up front since the camera never moves."""

    def __init__(self, env: MarsHabitatEnv, data: RolloutData):
        self.env = env
        self.data = data
        self._trail_px = [
            env.project_world_to_topdown_pixel(float(x), float(z))
            for x, z in zip(data.x, data.z)
        ]
        self._goal_px = (
            env.project_world_to_topdown_pixel(data.goal[0], data.goal[2])
            if data.goal is not None
            else None
        )
        self._obstacle_px = [
            env.project_world_to_topdown_pixel(ox, oz) for ox, _oy, oz in data.obstacles
        ]

    def render(self, i: int) -> np.ndarray:
        d = self.data
        self.env.step(
            Pose(x=float(d.x[i]), y=0.0, z=float(d.z[i]), yaw=float(d.yaw[i]))
        )
        frame = self.env.get_topdown_rgb()

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        trail = [p for p in self._trail_px[: i + 1] if p is not None]
        if len(trail) > 1:
            draw.line(trail, fill=TRAIL_COLOR, width=3)
        frame = np.asarray(img, dtype=np.uint8)

        if self._goal_px is not None:
            frame = draw_point_marker(frame, self._goal_px, radius=16, color=GOAL_COLOR)
        for opx in self._obstacle_px:
            if opx is not None:
                frame = draw_point_marker(frame, opx, radius=14, color=OBSTACLE_COLOR)

        status = f"step {i + 1}/{d.n}  t={i / d.hz:.1f}s"
        if d.goal_distance is not None:
            status += f"  dist={d.goal_distance[i]:.2f}m"
        if d.success is not None and i == d.n - 1:
            status += "  SUCCESS" if d.success else "  FAILED"
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 26], fill=(0, 0, 0))
        draw.text((6, 6), status, fill=(255, 255, 255))
        return np.asarray(img, dtype=np.uint8)


def export_video(renderer: TopdownRolloutRenderer, out_path: Path, fps: float) -> None:
    import imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps)
    try:
        for i in range(renderer.data.n):
            writer.append_data(renderer.render(i))
    finally:
        writer.close()


def run_viewer(renderer: TopdownRolloutRenderer) -> None:
    import tkinter as tk

    from PIL import ImageTk

    data = renderer.data
    root = tk.Tk()
    root.title(f"Rollout Viewer -- {data.n} steps @ {data.hz:g}Hz")

    canvas = tk.Canvas(root, bg="#111111", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    controls = tk.Frame(root)
    controls.pack(fill="x")
    play_btn = tk.Button(controls, text="Play")
    play_btn.pack(side="top", padx=6, pady=6)
    step_label = tk.Label(controls, text=f"1/{data.n}")
    step_label.pack(side="top", padx=6)

    slider = tk.Scale(root, from_=0, to=max(0, data.n - 1), orient="horizontal")
    slider.pack(fill="x")

    state = {"i": 0, "playing": False, "photo": None, "suppress": False}

    def draw(i: int) -> None:
        i = max(0, min(data.n - 1, i))
        state["i"] = i
        frame = renderer.render(i)
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = frame.shape[1], frame.shape[0]
        img = Image.fromarray(frame)
        scale = min(cw / img.width, ch / img.height)
        dw, dh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        img = img.resize((dw, dh))
        state["photo"] = ImageTk.PhotoImage(img)
        canvas.delete("frame")
        x0, y0 = (cw - dw) // 2, (ch - dh) // 2
        canvas.create_image(x0, y0, anchor="nw", image=state["photo"], tags="frame")
        step_label.config(text=f"{i + 1}/{data.n}")

    def on_slide(val) -> None:
        if state["suppress"]:
            return
        draw(int(float(val)))

    slider.config(command=on_slide)

    def tick() -> None:
        if not state["playing"]:
            return
        i = state["i"]
        if i >= data.n - 1:
            state["playing"] = False
            play_btn.config(text="Play")
            return
        i += 1
        state["suppress"] = True
        slider.set(i)
        state["suppress"] = False
        draw(i)
        root.after(max(15, int(1000.0 / data.hz)), tick)

    def toggle_play() -> None:
        state["playing"] = not state["playing"]
        play_btn.config(text="Pause" if state["playing"] else "Play")
        if state["playing"]:
            tick()

    play_btn.config(command=toggle_play)

    root.geometry("1000x1040")
    root.after(50, lambda: draw(0))
    root.mainloop()


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rollout", required=True, help="path to a rollout.npz")
    ap.add_argument("--scene-path", default=str(DEFAULT_SCENE))
    ap.add_argument("--heightmap-path", default=str(DEFAULT_HEIGHTMAP))
    ap.add_argument(
        "--out",
        default=None,
        help="write an MP4 here instead of opening the interactive scrubber window",
    )
    ap.add_argument("--topdown-height", type=int, default=1080)
    ap.add_argument("--topdown-width", type=int, default=1080)
    ap.add_argument(
        "--margin",
        type=float,
        default=4.0,
        help="extra clearance (m) kept around the trajectory/goal/obstacle bounding "
        "box when framing the fixed camera (default 4.0)",
    )
    ap.add_argument(
        "--rover-marker-scale",
        type=float,
        default=0.5,
        help="scale multiplier for the perseverance_mars_rover.glb marker (1.0 = "
        "native real-world meter scale)",
    )
    ap.add_argument(
        "--show-flags",
        action="store_true",
        help="also spawn the scene's decorative flag diamond around the rover's "
        "first recorded pose. Off by default -- this rollout's own goal/obstacle "
        "markers are what actually matters, flags would just be clutter unrelated "
        "to the recorded episode",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = RolloutData(Path(args.rollout))

    env = MarsHabitatEnv(
        args.scene_path,
        args.heightmap_path,
        services=[],
        start_x=float(data.x[0]),
        start_z=float(data.z[0]),
        start_yaw=float(data.yaw[0]),
        spawn_flags=args.show_flags,
        enable_topdown_viz=True,
        topdown_resolution=(args.topdown_height, args.topdown_width),
        topdown_margin_m=args.margin,
        rover_marker_scale=args.rover_marker_scale,
    )
    with env:
        extra_xz = list(zip(data.x.tolist(), data.z.tolist()))
        if data.goal is not None:
            extra_xz.append((data.goal[0], data.goal[2]))
        for ox, _oy, oz in data.obstacles:
            extra_xz.append((ox, oz))
        env.refit_topdown_camera(extra_xz)

        renderer = TopdownRolloutRenderer(env, data)
        if args.out:
            export_video(renderer, Path(args.out), fps=data.hz)
            print(f"wrote {args.out} ({data.n} frames @ {data.hz:g}fps)")
        else:
            run_viewer(renderer)


if __name__ == "__main__":
    main()
