"""Legacy self-contained NavDP/S2DiT rollout adapter, an iteration of
rollout_navdp.py that adds loading a fixed procedural rock field
(--rock-field, from generate_rock_env.py) into the scene before the
goal/obstacle setup. Superseded by sam_vla/run_navdp_rollout.py; kept for
reference and for flags not yet ported over.

Usage:
    python scripts/vlm_nav_tests/rollout_navdp2.py \
      --navdp-root /path/to/navdp_sam --ckpt /path/to/ckpt_last.pt \
      --rock-field rock_envs/run1/rock_field.json \
      --cbf --cbf-mode cone --cbf-hard-gate --out mars_rocks_rollout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import habitat_sim
from habitat_sim.agent import AgentConfiguration
import quaternion

import qwen_vlm_client
from sam_vla.env.rock_generation import load_rock_field, register_rocks
from sam_vla.logging.episode_logger import EpisodeLogger, make_run_id

HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "assets/marsyard2022.glb"
DEFAULT_OBJ = HERE / "assets/marsyard2022.obj"


def qwen_steer_prompt(frame_fraction_pct: float) -> str:
    """
    Build the steering prompt using the measured percentage of image pixels
    occupied by the green goal mask.

    frame_fraction_pct must be in percentage units:
        0.50 means 0.50%
        1.25 means 1.25%
    """
    stop_threshold_pct = 1.0
    threshold_exceeded = frame_fraction_pct > stop_threshold_pct
    goal_not_visible = frame_fraction_pct <= 0.0

    return f"""
You are controlling a rover moving toward a navigation goal.

Scene definition:
- The bright neon-green highlighted region is the navigation goal.
- Any bright red highlighted region is an obstacle to drive around.
- Large beige rocks with no red highlight may still be obstacles; treat them
  as such even if unhighlighted.
- The black region is background and must be ignored.

A separate image-processing function has already measured the goal-mask
coverage. Treat this measurement as exact. Do not estimate it from the image.

MEASURED GOAL COVERAGE: {frame_fraction_pct:.4f}%
STOP THRESHOLD: greater than {stop_threshold_pct:.2f}%
IS THE STOP THRESHOLD EXCEEDED: {"YES" if threshold_exceeded else "NO"}
IS THE GOAL VISIBLE AT ALL: {"NO" if goal_not_visible else "YES"}

Follow this decision process internally and in the exact order shown.

STEP 0 — GOAL NOT VISIBLE

If IS THE GOAL VISIBLE AT ALL is NO (measured coverage is exactly 0.0000%,
i.e. no green pixels anywhere in the image), the stop rule in Step 1 CANNOT
apply -- there is no goal coverage to have exceeded any threshold. In this
case:
- The action MUST NOT be stop, no matter what else is in the frame.
- This holds even if a large obstacle fills most or all of the image --
  a full frame of obstacle is not a reason to stop; it only means the goal
  is currently hidden behind or beside it.
- Skip Step 1 entirely and go straight to Step 2, using the obstacle's
  position (not the goal, which you cannot see) to pick left or right:
  choose the side with more clear, obstacle-free terrain visible at the
  edges of the frame.

If IS THE GOAL VISIBLE AT ALL is YES, continue to Step 1 as normal.

STEP 1 — CHECK THE STOP RULE

If IS THE STOP THRESHOLD EXCEEDED is YES:
- The action MUST be stop.
- This rule overrides every visual steering consideration.
- Do not select left or right.
- Do not reconsider the decision based on the goal's position.
- Do not estimate the goal coverage yourself.

If IS THE STOP THRESHOLD EXCEEDED is NO:
- The action must not be stop.
- Continue to Step 2 and select either left or right.

STEP 2 — SELECT A STEERING DIRECTION

Only perform this step when the stop threshold is not exceeded.

1. Locate the bright neon-green goal, if any is visible.
2. Determine whether the safest route toward it is to the left or right.
   If no goal is visible (Step 0 applied), determine the safest route
   AROUND the obstacle instead.
3. Treat beige rocks as obstacles.
4. Ignore the black background.
5. If the goal is approximately centered, or no goal is visible, choose the
   side with more clear, obstacle-free terrain.

STEP 3 — VERIFY THE DECISION

Before responding, silently verify:
- If the measured coverage is exactly 0.0000%, the action is left or right,
  never stop -- regardless of how much of the frame an obstacle fills.
- If the measured coverage is greater than 1.00%, the action is stop.
- If the measured coverage is not greater than 1.00%, the action is left or right.
- The measured numerical value has priority over visual appearance.

Do not provide a detailed chain of thought. Provide only the selected action
and one brief reason.

Output exactly one word in this format:

ACTION: left, right, or stop

Examples:

MEASURED GOAL COVERAGE: 1.24%
ACTION: stop

MEASURED GOAL COVERAGE: 0.72%
ACTION: left

MEASURED GOAL COVERAGE: 0.0000% (goal not visible, obstacle fills the frame)
ACTION: right

STOP has more priority over STEER. If the measured coverage is GREATER than 1.00%, the action is stop, even if the goal appears to be centered or to the left or right. The ONE exception is exactly 0.0000% coverage (goal not visible at all) -- that case is NEVER stop, even if an obstacle fills the whole frame.
"""


SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566


class TerrainHeight:
    def __init__(
        self,
        *,
        mode: str,
        heightmap: Optional[Path],
        obj: Optional[Path],
        flat_y: float,
        size_x: float,
        size_z: float,
        size_y: float,
        flip_x: bool,
        flip_z: bool,
        swap_xz: bool,
    ):
        self.mode = mode
        self.flat_y = float(flat_y)
        self.size_x = float(size_x)
        self.size_z = float(size_z)
        self.size_y = float(size_y)
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)
        self.height = None
        self.hm_h = 0
        self.hm_w = 0
        self.obj_xs = None
        self.obj_zs = None
        self.obj_h = None

        if mode == "auto":
            if heightmap is not None and heightmap.exists():
                mode = "heightmap"
            elif obj is not None and obj.exists():
                mode = "obj"
            else:
                mode = "flat"
        self.mode = mode

        if self.mode == "heightmap":
            if heightmap is None or not heightmap.exists():
                raise FileNotFoundError(f"heightmap not found: {heightmap}")
            self._load_heightmap(heightmap)
        elif self.mode == "obj":
            if obj is None or not obj.exists():
                raise FileNotFoundError(f"OBJ terrain not found: {obj}")
            self._load_obj_grid(obj)
        elif self.mode == "flat":
            pass
        else:
            raise ValueError(f"unknown terrain height mode: {self.mode}")

    def _load_heightmap(self, path: Path) -> None:
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
        y = arr * self.size_y
        y = y - float(np.mean(y))
        self.height = y.astype(np.float32)
        self.hm_h, self.hm_w = self.height.shape

    def _load_obj_grid(self, path: Path) -> None:
        verts = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    # hm2obj.py wrote OBJ as v x row_axis height.  Blender/Habitat
                    # turns this into x/z ground plane with y-up height.
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
        if not verts:
            raise RuntimeError(f"no OBJ vertices found in {path}")
        arr = np.asarray(verts, dtype=np.float32)
        xs = np.unique(arr[:, 0])
        zs = np.unique(arr[:, 1])
        xs.sort()
        zs.sort()
        grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
        x_to_i = {float(x): i for i, x in enumerate(xs.tolist())}
        z_to_i = {float(z): i for i, z in enumerate(zs.tolist())}
        for x, z, h in arr:
            grid[z_to_i[float(z)], x_to_i[float(x)]] = h
        if np.isnan(grid).any():
            fill = float(np.nanmean(grid))
            grid = np.nan_to_num(grid, nan=fill)
        self.obj_xs = xs.astype(np.float32)
        self.obj_zs = zs.astype(np.float32)
        self.obj_h = grid.astype(np.float32)

    def __call__(self, x: float, z: float) -> float:
        if self.mode == "flat":
            return self.flat_y
        if self.mode == "heightmap":
            return self._sample_heightmap(x, z)
        return self._sample_obj(x, z)

    def _map_xz(self, x: float, z: float) -> Tuple[float, float]:
        if self.swap_xz:
            x, z = z, x
        u = (x + self.size_x / 2.0) / self.size_x
        v = (z + self.size_z / 2.0) / self.size_z
        if self.flip_x:
            u = 1.0 - u
        if self.flip_z:
            v = 1.0 - v
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    def _sample_heightmap(self, x: float, z: float) -> float:
        assert self.height is not None
        u, v = self._map_xz(x, z)
        px = u * (self.hm_w - 1)
        py = v * (self.hm_h - 1)
        return bilinear_grid(self.height, px, py)

    def _sample_obj(self, x: float, z: float) -> float:
        assert (
            self.obj_xs is not None
            and self.obj_zs is not None
            and self.obj_h is not None
        )
        xx = float(np.clip(x, float(self.obj_xs[0]), float(self.obj_xs[-1])))
        zz = float(np.clip(z, float(self.obj_zs[0]), float(self.obj_zs[-1])))
        col = np.searchsorted(self.obj_xs, xx) - 1
        row = np.searchsorted(self.obj_zs, zz) - 1
        col = int(np.clip(col, 0, len(self.obj_xs) - 2))
        row = int(np.clip(row, 0, len(self.obj_zs) - 2))
        x0, x1 = float(self.obj_xs[col]), float(self.obj_xs[col + 1])
        z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
        tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
        tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
        h00 = float(self.obj_h[row, col])
        h10 = float(self.obj_h[row, col + 1])
        h01 = float(self.obj_h[row + 1, col])
        h11 = float(self.obj_h[row + 1, col + 1])
        h0 = h00 * (1.0 - tx) + h10 * tx
        h1 = h01 * (1.0 - tx) + h11 * tx
        return float(h0 * (1.0 - tz) + h1 * tz)


class SceneMappedTerrain:
    def __init__(self, base, *, flip_x: bool, flip_z: bool, swap_xz: bool):
        self.base = base
        self.mode = getattr(base, "mode", "unknown")
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)

    def _map(self, x: float, z: float) -> Tuple[float, float]:
        xx = float(x)
        zz = float(z)
        if self.swap_xz:
            xx, zz = zz, xx
        if self.flip_x:
            xx = -xx
        if self.flip_z:
            zz = -zz
        return xx, zz

    def __call__(self, x: float, z: float) -> float:
        xx, zz = self._map(x, z)
        return float(self.base(xx, zz))

    def local_height_max(
        self, x: float, z: float, radius: float, samples: int = 5
    ) -> float:
        radius = max(float(radius), 0.0)
        samples = max(int(samples), 1)
        if radius <= 1e-6 or samples == 1:
            return float(self(x, z))
        vals = []
        for dx in np.linspace(-radius, radius, samples):
            for dz in np.linspace(-radius, radius, samples):
                if dx * dx + dz * dz <= radius * radius + 1e-8:
                    vals.append(float(self(float(x) + float(dx), float(z) + float(dz))))
        return float(max(vals)) if vals else float(self(x, z))


def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
    h, w = grid.shape
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = float(px - x0)
    dy = float(py - y0)
    h00 = float(grid[y0, x0])
    h10 = float(grid[y0, x1])
    h01 = float(grid[y1, x0])
    h11 = float(grid[y1, x1])
    h0 = h00 * (1.0 - dx) + h10 * dx
    h1 = h01 * (1.0 - dx) + h11 * dx
    return float(h0 * (1.0 - dy) + h1 * dy)


def add_navdp_to_path(navdp_root: Path) -> None:
    root = navdp_root.expanduser().resolve()
    scripts = root / "scripts"
    for p in (root, scripts):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def resolve_navdp_root(raw: Optional[str]) -> Path:
    candidates = []
    if raw:
        candidates.append(Path(raw))
    env = os.environ.get("NAVDP_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            HERE.parent / "navdp_sam",
            HERE.parent / "New code",
            HERE.parent / "ICRA2027" / "New code",
        ]
    )
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "model_s2_dit.py").exists() and (
            c / "scripts" / "rollout_habitat_policy.py"
        ).exists():
            return c
    raise FileNotFoundError(
        "Could not find NavDP repo. Pass --navdp-root /path/to/navdp_sam "
        "or set NAVDP_ROOT."
    )


def make_sensor(uuid: str, sensor_type, height: int, width: int, hfov_deg: float):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [int(height), int(width)]
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = float(hfov_deg)
    return spec


def make_sim(
    scene: Path, height: int, width: int, hfov_deg: float, with_semantic: bool = False
):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene.expanduser().resolve())
    sim_cfg.enable_physics = False
    specs = [
        make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
        make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
    ]
    if with_semantic:  # only added for --goal-mesh-uv; non-mesh runs are unchanged
        specs.append(
            make_sensor(
                "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
            )
        )
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def yaw_quat_xyzw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.asarray([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float32)


def set_agent_pose(agent, x: float, y: float, z: float, yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray([x, y, z], dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    agent.set_state(state)


def rgb_depth(obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(obs["rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb.astype(np.uint8), depth.astype(np.float32)


def camera_coords(
    point: np.ndarray, position: np.ndarray, yaw: float
) -> Tuple[float, float, float]:
    d = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    forward = float(fwd_x * d[0] + fwd_z * d[2])
    left = float(left_x * d[0] + left_z * d[2])
    right = -left
    up = float(d[1])
    return right, up, forward


def intrinsics_from_hfov(height: int, width: int, hfov_deg: float) -> Dict[str, float]:
    hfov = math.radians(float(hfov_deg))
    fx = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    fy = fx
    return {"fx": fx, "fy": fy, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


# --- rendered-mask goal: place a semantic mesh from a pixel, render its mask, belief from mask ---
MESH_GOAL_ID = 1
MESH_OBST_ID = 2


def semantic_from_obs(obs) -> np.ndarray:
    s = np.asarray(obs["semantic"])
    if s.ndim == 3:
        s = s[..., 0]
    return s.astype(np.int32)


def pixel_to_world(u, v, d, position, yaw, intr):
    """Unproject pixel (u=col, v=row) + planar depth d -> world point (mars camera conventions)."""
    right = (u - intr["cx"]) * d / intr["fx"]
    up = -(v - intr["cy"]) * d / intr["fy"]
    fx_vec = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    rt_vec = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    return (
        np.asarray(position, np.float64)
        + d * fx_vec
        + right * rt_vec
        + up * np.array([0.0, 1.0, 0.0])
    )


def mask_to_body(mask, depth_img, height, width, hfov_deg, fallback_range, min_px=1):
    """Body-frame goal point [forward, left] from a rendered mask: bearing from the mask's centroid
    column, range from the MEDIAN depth over all mask pixels. Mirrors bbox_to_body's robustness --
    a single centroid pixel (the old behavior here) can land on a depth discontinuity (a rock's
    silhouette edge, a gap between the mesh and the background behind it) and seed a badly wrong
    range that then dead-reckons, uncorrected, for the rest of the episode (mesh_goal_mode seeds
    belief_g ONCE and never re-corrects it -- see the caller)."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size < min_px:
        return None
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    patch = np.asarray(depth_img)[ys, xs]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    rng = float(np.median(valid)) if valid.size > 0 else float(fallback_range)
    u = float(xs.mean())
    right = (u - intr["cx"]) * rng / max(intr["fx"], 1e-6)
    return np.asarray([rng, -right], dtype=np.float32)  # [forward, left]


def belief_feat(belief, r_scale=10.0):
    """[forward,left] -> [cos(bearing), sin(bearing), range/scale]  (matches train_belief_adapter)."""
    f, l = float(belief[0]), float(belief[1])
    bearing = math.atan2(l, f)
    return np.array(
        [math.cos(bearing), math.sin(bearing), min(math.hypot(f, l) / r_scale, 1.0)],
        np.float32,
    )


def depth_patch_mesh(
    u0, v0, half, stride, depth, position, yaw, intr, lift=0.03, jump=0.4
):
    """A surface-following patch: back-project a pixel window through depth so verts sit on the
    surface the camera sees (no floating). Skips cells that bridge a depth discontinuity.
    """
    H, W = depth.shape
    us = list(range(max(0, int(u0 - half)), min(W, int(u0 + half) + 1), stride))
    vs = list(range(max(0, int(v0 - half)), min(H, int(v0 + half) + 1), stride))
    idx = -np.ones((len(vs), len(us)), int)
    dep = np.full((len(vs), len(us)), np.nan)
    verts = []
    for j, vv in enumerate(vs):
        for i, uu in enumerate(us):
            dd = float(depth[vv, uu])
            if not np.isfinite(dd) or dd <= 0.1:
                continue
            idx[j, i] = len(verts)
            dep[j, i] = dd
            verts.append(
                tuple(
                    pixel_to_world(uu, vv, dd, position, yaw, intr)
                    + lift * np.array([0.0, 1.0, 0.0])
                )
            )
    faces = []
    for j in range(len(vs) - 1):
        for i in range(len(us) - 1):
            a, b, c, e = idx[j, i], idx[j, i + 1], idx[j + 1, i], idx[j + 1, i + 1]
            if min(a, b, c, e) < 0:
                continue
            q = dep[j, i], dep[j, i + 1], dep[j + 1, i], dep[j + 1, i + 1]
            if max(q) - min(q) > jump:
                continue
            faces.append((a, c, e))
            faces.append((a, e, b))
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def _save_obj(path, verts, faces):
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def register_semantic_mesh(sim, mesh_path, semantic_id):
    """Add a render-only (kinematic, non-collidable) mesh carrying a semantic id."""
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    t = otm.create_new_template(mesh_path)
    t.render_asset_handle = mesh_path
    t.collision_asset_handle = mesh_path
    t.is_collidable = False
    tid = otm.register_template(t, f"sem_{semantic_id}_{os.path.basename(mesh_path)}")
    obj = rom.add_object_by_template_handle(otm.get_template_handle_by_id(tid))
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.collidable = False
    obj.semantic_id = int(semantic_id)
    return obj


def _parse_uv(s, W, H):
    fu, fv = (float(t) for t in str(s).split(","))
    return fu * W, fv * H


def place_mesh_goal_obstacle(sim, depth, position, yaw, intr, args, out_dir):
    """Place a goal (and optional raised obstacle) mesh from first-frame pixels; return world centroids."""
    md = Path(out_dir) / "meshes"
    md.mkdir(parents=True, exist_ok=True)
    H, W = depth.shape[:2]
    gu, gv = _parse_uv(args.goal_mesh_uv, W, H)
    gvv, gff = depth_patch_mesh(
        gu, gv, int(args.mesh_half_px), 2, depth, position, yaw, intr
    )
    goal_world = None
    if len(gvv):
        gp = str(md / "goal.obj")
        _save_obj(gp, gvv, gff)
        register_semantic_mesh(sim, gp, MESH_GOAL_ID)
        goal_world = gvv.mean(axis=0)
        print(
            f"[MASK] goal mesh: {len(gvv)} verts at pixel ({gu:.0f},{gv:.0f}) -> "
            f"world=({goal_world[0]:.2f},{goal_world[1]:.2f},{goal_world[2]:.2f})",
            flush=True,
        )
    else:
        print(
            f"[MASK] WARN goal pixel ({gu:.0f},{gv:.0f}) had no valid depth (sky?)",
            flush=True,
        )
    obst_world = None
    if args.obstacle_mesh_uv:
        ou, ov = _parse_uv(args.obstacle_mesh_uv, W, H)
        ovv, off = depth_patch_mesh(
            ou,
            ov,
            int(args.mesh_half_px),
            2,
            depth,
            position,
            yaw,
            intr,
            lift=float(args.mesh_obstacle_lift),
        )
        if len(ovv):
            op = str(md / "obstacle.obj")
            _save_obj(op, ovv, off)
            register_semantic_mesh(sim, op, MESH_OBST_ID)
            obst_world = ovv.mean(axis=0)
            print(
                f"[MASK] obstacle mesh: {len(ovv)} verts at pixel ({ou:.0f},{ov:.0f})",
                flush=True,
            )
    return goal_world, obst_world


def draw_circle_mask(
    height: int, width: int, u: float, v: float, radius: int
) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - float(u)) ** 2 + (yy - float(v)) ** 2 <= float(radius) ** 2
    return mask.astype(np.uint8)


def goal_pixel_ratio(goal_mask: np.ndarray) -> Dict[str, float]:
    """Fraction of the current frame occupied by goal-object pixels vs. the rest of the image.

    Feeds qwen_steer_prompt() the measured "goal obstacle % of the entire image" value so the
    VLM is told the ratio instead of eyeballing it; callers can also threshold `frame_fraction`
    directly instead of (or alongside) the VLM's answer.
    """
    total_px = int(goal_mask.shape[0] * goal_mask.shape[1])
    goal_px = int(np.count_nonzero(goal_mask))
    rest_px = total_px - goal_px
    return {
        "goal_px": goal_px,
        "rest_px": rest_px,
        "frame_fraction": goal_px / total_px if total_px > 0 else 0.0,
        "goal_to_rest_ratio": goal_px / rest_px if rest_px > 0 else float("inf"),
    }


def project_goal_mask(
    *,
    goal: np.ndarray,
    position: np.ndarray,
    yaw: float,
    height: int,
    width: int,
    hfov_deg: float,
    radius: int,
    clamp_to_edge: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    right, up, forward = camera_coords(goal, position, yaw)
    visible = forward > 0.05
    if not visible:
        return np.zeros((height, width), dtype=np.uint8), {
            "visible": 0.0,
            "u": -1.0,
            "v": -1.0,
            "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
            "bearing": float(
                math.atan2(right, forward if abs(forward) > 1e-6 else 1e-6)
            ),
        }
    u = intr["cx"] + intr["fx"] * right / max(forward, 1e-6)
    v = intr["cy"] - intr["fy"] * up / max(forward, 1e-6)
    in_frame = radius <= u < width - radius and radius <= v < height - radius
    if not in_frame and clamp_to_edge:
        u = float(np.clip(u, radius, width - radius - 1))
        v = float(np.clip(v, radius, height - radius - 1))
        in_frame = True
    if not in_frame:
        return np.zeros((height, width), dtype=np.uint8), {
            "visible": 0.0,
            "u": float(u),
            "v": float(v),
            "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
            "bearing": float(math.atan2(right, forward)),
        }
    mask = draw_circle_mask(height, width, u, v, radius)
    return mask, {
        "visible": 1.0,
        "u": float(u),
        "v": float(v),
        "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
        "bearing": float(math.atan2(right, forward)),
    }


def obstacle_point_from_world(
    obstacle: np.ndarray, position: np.ndarray, yaw: float
) -> Optional[np.ndarray]:
    right, _up, forward = camera_coords(obstacle, position, yaw)
    if forward <= 0.05:
        return None
    # CBF helpers use robot-frame [x_forward, y_left].
    return np.asarray([forward, -right], dtype=np.float32)


def project_body_point_mask(bg, height, width, hfov_deg, radius, clamp_to_edge):
    """Render a filled-circle goal mask from a BODY-frame point bg=[forward, left]. Used to draw
    the belief-tracked goal (bg is a propagated estimate, not the known goal). Mirrors
    project_goal_mask's projection."""
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    forward, left = float(bg[0]), float(bg[1])
    right = -left
    info = {
        "visible": 0.0,
        "u": -1.0,
        "v": -1.0,
        "range": float(math.hypot(forward, left)),
        "bearing": float(math.atan2(left, forward)),
    }
    if forward <= 0.05:
        return np.zeros((height, width), dtype=np.uint8), info
    u = intr["cx"] + intr["fx"] * right / max(forward, 1e-6)
    v = intr["cy"]
    in_frame = radius <= u < width - radius and radius <= v < height - radius
    if not in_frame and clamp_to_edge:
        u = float(np.clip(u, radius, width - radius - 1))
        v = float(np.clip(v, radius, height - radius - 1))
        in_frame = True
    if not in_frame:
        info.update({"u": float(u), "v": float(v)})
        return np.zeros((height, width), dtype=np.uint8), info
    info.update({"visible": 1.0, "u": float(u), "v": float(v)})
    return draw_circle_mask(height, width, u, v, radius), info


def pixel_to_body(u, v, depth_img, height, width, hfov_deg, fallback_range):
    """Unproject an image pixel (u=col, v=row) to a body-frame point [forward, left] using the
    depth at that pixel (or a fallback range if depth is missing). This is how a VLM-grounded
    goal pixel becomes the belief seed -- language -> where the goal is, in metres."""
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    iu = int(np.clip(u, 0, width - 1))
    iv = int(np.clip(v, 0, height - 1))
    d = float(depth_img[iv, iu]) if depth_img is not None else 0.0
    rng = d if (np.isfinite(d) and d > 0.1) else float(fallback_range)
    right = (float(u) - intr["cx"]) * rng / max(intr["fx"], 1e-6)
    return np.asarray([rng, -right], dtype=np.float32)  # [forward, left]


def bbox_to_body(bbox_xyxy, depth_img, height, width, hfov_deg, fallback_range):
    """Body-frame point [forward, left] from a VLM bbox: bearing from the bbox's center column,
    range from the MEDIAN depth over the bbox's interior. Mirrors bbox_to_world_seed's
    median-over-samples robustness (vlm_nav_interactive.py) but stays image-only -- no world
    pose needed, so it doesn't break the belief-only "language decides WHERE" design.
    pixel_to_body's single center pixel is fragile: it can land on a depth discontinuity (a
    rock's silhouette edge, or a gap between the rock and the background) and seed a badly
    wrong range that then dead-reckons, uncorrected, for the rest of the episode -- the ghost
    stays glued to that wrong point and drifts off-screen as the rover moves past it."""
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    x1, y1, x2, y2 = bbox_xyxy
    iu1, iu2 = int(np.clip(min(x1, x2), 0, width - 1)), int(
        np.clip(max(x1, x2), 0, width - 1)
    )
    iv1, iv2 = int(np.clip(min(y1, y2), 0, height - 1)), int(
        np.clip(max(y1, y2), 0, height - 1)
    )
    patch = np.asarray(depth_img)[iv1 : iv2 + 1, iu1 : iu2 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    rng = float(np.median(valid)) if valid.size > 0 else float(fallback_range)
    u = 0.5 * (x1 + x2)
    right = (u - intr["cx"]) * rng / max(intr["fx"], 1e-6)
    return np.asarray([rng, -right], dtype=np.float32)  # [forward, left]


class VlmSelectionPixelGoal:
    """Adapts a one-shot VLM object selection (resolve_vlm_selection, run once on an
    already-captured+annotated frame) to the .ground(rgb, instruction) grounder
    interface used by --grounder stub/qwen, so --goal-from-vlm seeds the belief via
    the same image-pixel path -- never a world coordinate. The bbox center is stored
    as a FRACTION of the frame it was resolved on so it reprojects correctly onto the
    live rollout frame, whose resolution can differ from the capture resolution."""

    # A one-shot capture-time selection, not a live re-detector: the belief should be
    # seeded from it ONCE (dead-reckoned by odometry after), never re-queried on a
    # cadence like --grounder stub/qwen (see the main loop's grounder-call gate).
    one_shot = True

    def __init__(self, bbox_xyxy, capture_hw):
        cap_h, cap_w = capture_hw
        x1, y1, x2, y2 = bbox_xyxy
        self._u_frac = 0.5 * (x1 + x2) / cap_w
        self._v_frac = 0.5 * (y1 + y2) / cap_h
        self._x1_frac, self._y1_frac = x1 / cap_w, y1 / cap_h
        self._x2_frac, self._y2_frac = x2 / cap_w, y2 / cap_h

    def ground(self, rgb, instruction):
        h, w = rgb.shape[0], rgb.shape[1]
        bbox = (
            self._x1_frac * w,
            self._y1_frac * h,
            self._x2_frac * w,
            self._y2_frac * h,
        )
        return SimpleNamespace(
            u=self._u_frac * w, v=self._v_frac * h, in_view=True, bbox=bbox
        )


def propagate_body_point(bg, action, dt, odom_noise=0.0, rng=None):
    """Move a body-frame point [forward, left] under the robot's own SE(2) motion (dead-reckoning):
    translate back by v*dt and rotate by -yaw*dt -- the same propagation the cone uses. Optional
    Gaussian odom noise makes the belief DRIFT, so it must be corrected by sightings to stay good.
    """
    v_fwd, v_lat, yaw_rate = float(action[0]), float(action[1]), float(action[2])
    if odom_noise > 0.0 and rng is not None:
        v_fwd += float(rng.normal(0.0, odom_noise))
        yaw_rate += float(rng.normal(0.0, odom_noise))
    th = -yaw_rate * dt
    c, s = math.cos(th), math.sin(th)
    qx = float(bg[0]) - v_fwd * dt
    qy = float(bg[1]) - v_lat * dt
    return np.asarray([c * qx - s * qy, s * qx + c * qy], dtype=np.float32)


def paint_obstacle_map_point(
    obstacle_map: np.ndarray,
    builder,
    point_forward_left: Optional[np.ndarray],
    radius_cells: int,
) -> np.ndarray:
    out = np.asarray(obstacle_map, dtype=np.float32).copy()
    if point_forward_left is None:
        return out
    p = np.asarray(point_forward_left, dtype=np.float32).reshape(-1)
    if p.size < 2 or not np.isfinite(p[:2]).all():
        return out
    rows, cols = builder.world_to_grid(
        np.asarray([p[0]], dtype=np.float32), np.asarray([p[1]], dtype=np.float32)
    )
    r = int(rows[0])
    c = int(cols[0])
    rad = max(int(radius_cells), 0)
    h, w = out.shape
    for rr in range(max(0, r - rad), min(h, r + rad + 1)):
        for cc in range(max(0, c - rad), min(w, c + rad + 1)):
            if (rr - r) ** 2 + (cc - c) ** 2 <= rad**2:
                out[rr, cc] = 1.0
    return out


def depth_obstacle_mask(
    depth: np.ndarray, threshold: float, min_y_frac: float
) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    h, _ = arr.shape
    yy = np.arange(h)[:, None]
    mask = (
        np.isfinite(arr)
        & (arr > 0.0)
        & (arr < float(threshold))
        & (yy >= h * float(min_y_frac))
    )
    return mask.astype(np.uint8)


def overlay_frame(
    rgb: np.ndarray, goal_mask: np.ndarray, obstacle_mask: np.ndarray, text: str
) -> Image.Image:
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pix = np.asarray(overlay).copy()
    gm = np.asarray(goal_mask) > 0
    om = np.asarray(obstacle_mask) > 0
    pix[gm] = [0, 255, 0, 120]
    pix[om] = [255, 0, 0, 100]
    overlay = Image.fromarray(pix, mode="RGBA")
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 46], fill=(0, 0, 0, 170))
    draw.text((8, 6), text, fill=(255, 255, 255, 255))
    return img.convert("RGB")


def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[WARN] imageio unavailable; skipping video: {exc}", flush=True)
        return
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=float(fps))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a trained NavDP/S2DiT policy inside the Mars HabitatSim terrain."
    )
    ap.add_argument(
        "--navdp-root",
        default=None,
        help="Path to the navdp_sam repo containing model_s2_dit.py",
    )
    ap.add_argument(
        "--ckpt", required=True, help="Path to trained NavDP/S2DiT checkpoint"
    )
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument(
        "--rock-field",
        default=None,
        help="Path to a rock_field.json produced by generate_rock_env.py. Loads that fixed, "
        "already-placed rock layout into the scene (visible in RGB before any goal/obstacle "
        "resolution runs) instead of an empty terrain -- use the same path across ablation "
        "runs to keep the obstacle layout identical.",
    )
    ap.add_argument(
        "--out",
        default=f"navdp_rollout{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for the rollout results.",
    )
    ap.add_argument(
        "--log-root",
        default="logs",
        help="Root dir for structured per-episode ablation logs (config/frames/obstacles/"
        "qwen_queries/cbf_events/summary); a timestamped run_id subdir is created "
        "under it every rollout, separate from --out.",
    )
    ap.add_argument(
        "--log-save-frames",
        action="store_true",
        help="Also dump each step's raw RGB frame under the log run dir's frames/ folder.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", choices=["model", "ema"], default="model")
    # Compatibility knobs matching scripts/rollout_habitat_policy.py.
    ap.add_argument(
        "--scene-mode",
        default="mars",
        help="Accepted for NavDP command compatibility; ignored by Mars adapter.",
    )
    ap.add_argument(
        "--obstacle-pool",
        default="none",
        help="Accepted for NavDP command compatibility; ignored unless ghost/depth obstacles are provided.",
    )
    ap.add_argument(
        "--categories",
        nargs="*",
        default=["chair"],
        help="Accepted for command compatibility; the Mars target is set by --goal-x/--goal-z.",
    )
    ap.add_argument(
        "--episodes-per-category",
        type=int,
        default=1,
        help="Accepted for command compatibility; Mars adapter runs one rollout.",
    )
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--hfov-deg", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--stop-dist", type=float, default=1.0)
    ap.add_argument(
        "--qwen-stop-frame-fraction",
        type=float,
        default=0.01,
        help="If the goal's measured frame_fraction (from goal_pixel_ratio()) is at "
        "least this, treat it as proximity for a Qwen 'stop' verdict, same as "
        "near_obstacle/near_goal -- matches the 1%% threshold already stated in "
        "qwen_steer_prompt().",
    )
    ap.add_argument(
        "--vla-dump",
        type=str,
        default="",
        help="If set, dump paired left/right counterfactual VLA training samples "
        "(neutral-goal obs + orbit-generated left & right chunks) to this dir at blocked steps.",
    )
    ap.add_argument(
        "--vla-dump-every",
        type=int,
        default=3,
        help="Dump one sample per N blocked steps.",
    )
    ap.add_argument(
        "--vla-horizon",
        type=int,
        default=8,
        help="Orbit target chunk length (match the policy horizon).",
    )
    ap.add_argument(
        "--command",
        type=str,
        default="",
        help="Real-time language command: 'pass left' / 'pass right' / 'stop' / '' (default). "
        "Overrides the geometric side choice while an obstacle blocks the path.",
    )
    ap.add_argument(
        "--command-file",
        type=str,
        default="",
        help="Path polled every tick for the current command (a human or a VLM writes to it). "
        "Overrides --command. This is the LIVE inference interface.",
    )
    ap.add_argument(
        "--qwen-steer",
        action="store_true",
        help="Once near the goal (same stop-dist+deadzone gate used for goal-reached), poll "
        "the persistent Qwen VLM server (qwen_vlm_server.py) for a live left/right/stop "
        "steering command instead of --command/--command-file. Feeds into the same "
        "command_intent path -- overrides --command/--command-file once active.",
    )
    ap.add_argument(
        "--qwen-steer-hz",
        type=float,
        default=3.0,
        help="Poll rate (Hz) for --qwen-steer once near-goal; independent of --hz.",
    )
    ap.add_argument("--qwen-host", default=qwen_vlm_client.DEFAULT_HOST)
    ap.add_argument("--qwen-port", type=int, default=qwen_vlm_client.DEFAULT_PORT)
    ap.add_argument(
        "--vla-adapter",
        type=str,
        default="",
        help="Path to a trained vla_adapter.pt. REGIME B: the language-conditioned POLICY "
        "produces the maneuver (orbit override + soft cone projection off; hard gate keeps it "
        "safe). Without it, the command drives the orbit controller (Regime A).",
    )
    ap.add_argument(
        "--vla-alpha-scale",
        type=float,
        default=1.25,
        help="Scale the adapter's language token at inference (ablation showed ~1.25 gives the "
        "cleanest instruction-following).",
    )
    ap.add_argument(
        "--belief-goal",
        action="store_true",
        help="Track the goal via BELIEF: seed a body-frame estimate from the goal ONCE, then "
        "propagate it by odometry and draw the ghost from the estimate. The known goal touches "
        "the system only at t=0 (and, if enabled, to correct on sight) -- no per-frame geometry.",
    )
    ap.add_argument(
        "--belief-odom-noise",
        type=float,
        default=0.0,
        help="Gaussian odom noise per step for the belief propagation. 0 = perfect dead-reckoning "
        "(numerically equals geometry); >0 makes the belief drift (its value shows under sightings).",
    )
    ap.add_argument(
        "--belief-update-on-sight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-seed the belief from the goal whenever the goal is actually in view (corrects "
        "drift). --no-belief-update-on-sight = pure dead-reckoning from the initial mask only.",
    )
    ap.add_argument(
        "--goal-bearing-deg",
        type=float,
        default=None,
        help="IMAGE goal (no world xyz): seed the belief from a bearing (+ = right of forward) and "
        "--goal-range in the FIRST view, then dead-reckon by odometry. --goal-x/z become only a "
        "reference for the success metric, never used by control.",
    )
    ap.add_argument(
        "--goal-range",
        type=float,
        default=8.0,
        help="Initial/fallback range (m) for image-grounded goals.",
    )
    ap.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Language instruction for VLM goal grounding (with --grounder).",
    )
    ap.add_argument(
        "--grounder",
        choices=["none", "stub", "qwen"],
        default="none",
        help="GROUNDED GOAL: point the goal from RGB+instruction. stub=fixed pixel (test the wiring), "
        "qwen=Qwen2.5-VL zero-shot. Implies --belief-goal; the pixel seeds the belief, odometry tracks it.",
    )
    ap.add_argument(
        "--grounder-every",
        type=int,
        default=15,
        help="Re-ground every N steps (odometry tracks between).",
    )
    ap.add_argument(
        "--grounder-uv",
        type=str,
        default="0.5,0.5",
        help="stub grounder pixel as fraction 'fx,fy'.",
    )
    ap.add_argument("--grounder-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=8.0)
    ap.add_argument("--start-yaw-deg", type=float, default=0.0)
    ap.add_argument(
        "--goal-x",
        type=float,
        default=None,
        help="World goal X (required unless --goal-mesh-uv/--goal-from-vlm).",
    )
    ap.add_argument(
        "--goal-z",
        type=float,
        default=None,
        help="World goal Z (required unless --goal-mesh-uv/--goal-from-vlm).",
    )
    ap.add_argument(
        "--goal-from-vlm",
        action="store_true",
        help="BELIEF-tracked goal (mode b): resolve a VLM object selection and seed the "
        "belief via the same grounder pixel path as --grounder stub/qwen (implies "
        "--belief-goal). DEFAULT: the frame is captured LIVE from this rollout's own "
        "start pose and auto-annotated with SAM -- no pre-existing files needed. Pass "
        "--manual-annotate to instead use a pre-existing, already-annotated frame from "
        "vlm_nav_interactive's capture session. The resolved world position is kept "
        "only as a logging/success-metric reference, like --goal-bearing-deg -- never "
        "fed to control.",
    )
    ap.add_argument(
        "--manual-annotate",
        action="store_true",
        help="With --goal-from-vlm: use a pre-existing, manually labelme-annotated frame "
        "(vlm_nav_interactive's OUT_DIR/ANNOTATIONS_DIR, keyed by --vlm-frame-idx) "
        "instead of the default live-capture+SAM path. Warns if --start-x/z/yaw-deg "
        "differ from the pose that frame was captured at, since the annotated bbox is "
        "a fixed pixel fraction only valid for that pose.",
    )
    ap.add_argument(
        "--vlm-frame-idx",
        type=int,
        default=0,
        help="Frame index for --goal-from-vlm. With --manual-annotate, selects the "
        "pre-captured/annotated frame to load; by default (SAM live-capture), it "
        "just names the live-captured frame's output/annotation files.",
    )
    ap.add_argument(
        "--goal-y",
        type=float,
        default=None,
        help="World Y of goal marker; default terrain height + goal-height",
    )
    ap.add_argument(
        "--goal-height",
        type=float,
        default=1.2,
        help="Goal marker height above terrain when --goal-y is omitted",
    )
    ap.add_argument(
        "--goal-terrain-radius",
        type=float,
        default=0.8,
        help="Raise ghost goal from local max terrain height in this radius",
    )
    ap.add_argument("--goal-radius", type=int, default=18)
    ap.add_argument("--no-clamp-goal-to-edge", action="store_true")
    ap.add_argument(
        "--goal-mesh-uv",
        type=str,
        default=None,
        help="RENDERED-MASK goal: place a semantic mesh at this first-frame pixel fraction "
        "'fu,fv'; each step the mask is rendered and the belief is built from it "
        "(the policy's goal channel IS the mask). Enables mesh mode; no --goal-x needed.",
    )
    ap.add_argument(
        "--obstacle-mesh-uv",
        type=str,
        default=None,
        help="Optional: place a raised obstacle mesh at this pixel fraction; auto-enables "
        "--obstacle-mode depth so the cone avoids it.",
    )
    ap.add_argument(
        "--mesh-half-px",
        type=int,
        default=26,
        help="half-size (px) of the pixel window per patch mesh",
    )
    ap.add_argument(
        "--mesh-obstacle-lift",
        type=float,
        default=0.5,
        help="raise the obstacle mesh so depth sees it",
    )
    ap.add_argument(
        "--belief-adapter",
        type=str,
        default=None,
        help="trained belief-return adapter (belief_adapter.pt). When the goal is OFF-SCREEN "
        "the belief token drives the POLICY back to it -- replaces the P-controller.",
    )
    ap.add_argument(
        "--lang-turn-hyst",
        type=float,
        default=0.6,
        help="extra distance beyond cbf-d-safe+cbf-deadzone before the near-obstacle "
        "maneuver gate releases (hysteresis; stops it flicking on/off at the boundary).",
    )
    ap.add_argument(
        "--belief-reacquire-px",
        type=int,
        default=None,
        help="goal-pixel count above which the belief-return gate releases (hysteresis); "
        "default 3x --lost-goal-min-px.",
    )
    ap.add_argument(
        "--terrain-height-mode",
        choices=["auto", "heightmap", "obj", "flat"],
        default="auto",
    )
    ap.add_argument("--heightmap", default=None)
    ap.add_argument("--terrain-obj", default=str(DEFAULT_OBJ))
    ap.add_argument("--flat-y", type=float, default=0.0)
    ap.add_argument("--clearance", type=float, default=1.4)
    ap.add_argument(
        "--pose-terrain-radius",
        type=float,
        default=0.8,
        help="Use local max terrain height around rover footprint before adding clearance",
    )
    ap.add_argument("--size-x", type=float, default=SIZE_X)
    ap.add_argument("--size-z", type=float, default=SIZE_Z)
    ap.add_argument("--size-y", type=float, default=SIZE_Y)
    ap.add_argument("--flip-heightmap-x", action="store_true")
    ap.add_argument(
        "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--swap-heightmap-xz", action="store_true")
    ap.add_argument(
        "--scene-height-flip-x", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--scene-height-flip-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror Habitat scene Z before terrain-height lookup; matches the Mars GLB export",
    )
    ap.add_argument(
        "--scene-height-swap-xz", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--habitat-proprio-mode", choices=["pose7", "planar3", "zero"], default=None
    )
    ap.add_argument(
        "--habitat-action-mode",
        choices=["action3d", "action2d", "waypoint"],
        default=None,
    )
    ap.add_argument("--habitat-yaw-axis", choices=["x", "y", "z"], default=None)
    ap.add_argument(
        "--habitat-use-obstacle-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument("--obstacle-mode", choices=["none", "depth"], default="none")
    ap.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
    ap.add_argument("--obstacle-min-y-frac", type=float, default=0.45)
    ap.add_argument(
        "--ghost-obstacle-x",
        type=float,
        default=None,
        help="Optional world X for a synthetic/ghost obstacle mask.",
    )
    ap.add_argument(
        "--ghost-obstacle-z",
        type=float,
        default=None,
        help="Optional world Z for a synthetic/ghost obstacle mask.",
    )
    ap.add_argument(
        "--ghost-obstacle-y",
        type=float,
        default=None,
        help="World Y of ghost obstacle marker; default terrain height + ghost-obstacle-height.",
    )
    ap.add_argument("--ghost-obstacle-height", type=float, default=0.45)
    ap.add_argument(
        "--ghost-obstacle-radius",
        type=int,
        default=24,
        help="Pixel radius for the synthetic obstacle mask.",
    )
    ap.add_argument(
        "--ghost-obstacle-map-radius",
        type=int,
        default=4,
        help="Radius in 96x96 obstacle-map cells for the ghost obstacle.",
    )
    ap.add_argument("--no-clamp-obstacle-to-edge", action="store_true")
    ap.add_argument(
        "--zero-lateral", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--max-forward-speed", type=float, default=1.0)
    ap.add_argument("--max-lateral-speed", type=float, default=1.0)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument(
        "--action-smoothing", choices=["ensemble", "ema", "none"], default="none"
    )
    ap.add_argument("--ensemble-decay", type=float, default=0.5)
    ap.add_argument("--ema-alpha", type=float, default=0.6)
    ap.add_argument("--cbf", action="store_true")
    ap.add_argument("--cbf-mode", choices=["project", "cone"], default="cone")
    ap.add_argument("--cbf-d-safe", type=float, default=0.75)
    ap.add_argument("--cbf-gamma", type=float, default=0.3)
    ap.add_argument("--cbf-deadzone", type=float, default=0.6)
    ap.add_argument("--cbf-proj-iters", type=int, default=15)
    ap.add_argument("--cbf-proj-lr", type=float, default=0.08)
    ap.add_argument("--cbf-cone-margin", type=float, default=0.05)
    ap.add_argument("--cbf-trust", type=float, default=0.3)
    ap.add_argument("--cbf-smooth", type=float, default=0.0)
    ap.add_argument("--cbf-keep-speed", type=float, default=1.0)
    ap.add_argument(
        "--cbf-metric", choices=["euclidean", "mahalanobis"], default="euclidean"
    )
    ap.add_argument("--cbf-cov-base", type=float, default=1.0)
    ap.add_argument("--cbf-cov-growth", type=float, default=0.6)
    ap.add_argument(
        "--cbf-cov-mode", choices=["grow", "flat", "shrink"], default="shrink"
    )
    ap.add_argument(
        "--cbf-radius-mode", choices=["fixed", "perceived"], default="fixed"
    )
    ap.add_argument("--robot-radius", type=float, default=0.25)
    ap.add_argument("--safety-margin", type=float, default=0.15)
    ap.add_argument("--ghost-obstacle-world-radius", type=float, default=0.25)
    # --- ported safety layer (per-tick hard gate + escape yaw + committed side) ---
    ap.add_argument(
        "--cbf-hard-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cone mode: re-check the FINAL executed action every tick against the obstacle "
        "and brake forward if it would breach. The soft chunk projection alone is diluted by "
        "the smoother / skipped between replans -> not safe without this.",
    )
    ap.add_argument(
        "--cbf-escape-yaw",
        type=float,
        default=0.6,
        help="cone mode: ENABLE tangent-point pursuit around the obstacle (any value >0 turns it "
        "on; 0=off, fall back to the plain distance brake). The turn rate itself is computed by "
        "the pursuit law and capped at --max-yaw-rate, so the magnitude here is just the switch.",
    )
    ap.add_argument(
        "--cbf-pursuit-kp",
        type=float,
        default=1.8,
        help="cone mode: proportional gain from tangent heading error to yaw-rate for the smooth "
        "pursuit. Higher = turns onto the tangent sooner (crisper); too high can overshoot.",
    )
    ap.add_argument(
        "--cbf-orbit-kr",
        type=float,
        default=0.8,
        help="cone mode: radial pull-back gain (rad/m) onto the d_safe circle. The orbit law is "
        "tangential heading + this*(dist - d_safe); it settles ON the circle instead of the "
        "asin-tangent's bounce. 0 = pure tangential (twitchy when hugging tight).",
    )
    ap.add_argument(
        "--cbf-orbit-hyst",
        type=float,
        default=0.4,
        help="cone mode: extra clearance (m) required to LEAVE the orbit once committed. Hysteresis "
        "on the orbit<->goal switch so it cannot rapid-toggle at the boundary (chatter).",
    )
    ap.add_argument(
        "--cbf-goaround-forward",
        type=float,
        default=0.5,
        help="cone mode: constant cruise speed (m/s) while skirting the obstacle with tangent "
        "pursuit. Keep <= max-yaw-rate * d_safe so the circle stays trackable (1.0*1.2=1.2 here).",
    )
    ap.add_argument(
        "--cbf-commit-side",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cone mode: hold the go-around side while the obstacle stays in view instead of "
        "recomputing sign(p_lat) every replan (which dithers -> yaw stutter).",
    )
    ap.add_argument(
        "--lost-goal-ghost",
        action="store_true",
        help="Steer toward the known ghost goal (proportional heading assist) when it drifts to/past the frame edge, where the mask-conditioned policy only steers weakly.",
    )
    ap.add_argument(
        "--lost-goal-min-px",
        type=int,
        default=10,
        help="Goal-mask pixels below this count means the goal is behind us (mask empty) -> pivot recovery.",
    )
    ap.add_argument("--lost-goal-turn-kp", type=float, default=1.4)
    ap.add_argument(
        "--lost-goal-forward",
        type=float,
        default=0.0,
        help="Forward speed floor during recovery. When the goal is merely off to the side we keep the policy's forward and only override yaw; this floor applies when the goal is fully behind (pivot).",
    )
    ap.add_argument(
        "--lost-goal-bearing-deg",
        type=float,
        default=30.0,
        help="Engage the proportional heading assist once |goal bearing| exceeds this angle. The ghost is clamped to the frame edge beyond ~hfov/2 (=45deg at hfov 90), where the policy's yaw response saturates weakly; a value below hfov/2 kicks the strong turn in just before the edge. 0 disables the angle trigger (mask-empty only).",
    )
    ap.add_argument(
        "--replan-every",
        type=int,
        default=1,
        help="Sample a fresh diffusion chunk every N control ticks.",
    )
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    navdp_root = resolve_navdp_root(args.navdp_root)
    add_navdp_to_path(navdp_root)

    from navdp.data.habitat_route_dataset import (
        _empty_belief_tensor,
        _proprio_from_pose,
    )
    from navdp.extensions import (
        DepthObstacleMap,
        horizon_growth_covariance,
        nearest_obstacle_point,
        project_chunk_cone,
        project_forward_velocity_cbf,
    )
    from rollout_habitat_policy import (
        ActionSmoother,
        action_to_control,
        frame_to_spatial,
        load_model,
        resolve_modes,
        resolve_obstacle_channel,
    )

    if args.goal_from_vlm:
        from vlm_nav_interactive import (
            OUT_DIR as VLM_OUT_DIR,
            ANNOTATIONS_DIR as VLM_ANNOTATIONS_DIR,
            RGBD_RESOLUTION as VLM_CAPTURE_HW,
            START_X as VLM_START_X,
            START_Z as VLM_START_Z,
            START_YAW_DEG as VLM_START_YAW_DEG,
            draw_annotation_overlay,
            resolve_vlm_selection,
            save_mission_metadata,
            save_pose as vlm_save_pose,
        )

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    raw_terrain = TerrainHeight(
        mode=args.terrain_height_mode,
        heightmap=(
            Path(args.heightmap).expanduser().resolve() if args.heightmap else None
        ),
        obj=Path(args.terrain_obj).expanduser().resolve() if args.terrain_obj else None,
        flat_y=args.flat_y,
        size_x=args.size_x,
        size_z=args.size_z,
        size_y=args.size_y,
        flip_x=args.flip_heightmap_x,
        flip_z=args.flip_heightmap_z,
        swap_xz=args.swap_heightmap_xz,
    )
    terrain = SceneMappedTerrain(
        raw_terrain,
        flip_x=bool(args.scene_height_flip_x),
        flip_z=bool(args.scene_height_flip_z),
        swap_xz=bool(args.scene_height_swap_xz),
    )

    device = args.device
    model, train_args = load_model(
        Path(args.ckpt).expanduser().resolve(), device, args.weights
    )
    modes = resolve_modes(args, train_args)
    if modes["action_mode"] == "waypoint":
        raise ValueError(
            "Mars rollout executes velocity actions; use action3d or action2d checkpoint/mode."
        )
    use_obstacle_channel = resolve_obstacle_channel(args, train_args)
    image_size = int(args.image_size or train_args.get("image_size", 224))
    intr = intrinsics_from_hfov(args.height, args.width, args.hfov_deg)
    obstacle_builder = DepthObstacleMap(camera_intrinsics=intr)
    smoother = ActionSmoother(
        args.action_smoothing, args.ensemble_decay, args.ema_alpha
    )

    # REGIME B: load the trained language adapter + text encoder (frozen). The command's text
    # token is appended to the policy cond set so the POLICY produces the maneuver.
    vla_adapter = None
    vla_text_enc = None
    vla_tok_cache = {}
    belief_adapter = None  # trained belief-return adapter (--belief-adapter); replaces the P-controller
    if args.vla_adapter:
        from train_vla_adapter import VLAAdapter
        from sentence_transformers import SentenceTransformer

        _vck = torch.load(args.vla_adapter, map_location=device)
        vla_text_enc = SentenceTransformer(_vck["text_encoder"], device=device)
        vla_adapter = VLAAdapter(
            _vck["text_dim"], _vck["dim"], num_tokens=_vck.get("num_tokens", 1)
        ).to(device)
        vla_adapter.load_state_dict(_vck["adapter"])
        vla_adapter.eval()
        print(
            f"[VLA] Regime B: policy driven by language adapter {args.vla_adapter} "
            f"(alpha_scale={args.vla_alpha_scale}, tokens={vla_adapter.num_tokens})",
            flush=True,
        )

    if args.belief_adapter:
        from train_vla_adapter import VLAAdapter

        _bck = torch.load(args.belief_adapter, map_location=device)
        belief_adapter = VLAAdapter(
            _bck["belief_feat_dim"], _bck["dim"], num_tokens=_bck.get("num_tokens", 4)
        ).to(device)
        belief_adapter.load_state_dict(_bck["adapter"])
        belief_adapter.eval()
        print(
            f"[BELIEF] learned return: the belief token drives the policy when the goal is off-screen "
            f"(replaces the P-controller); tokens={belief_adapter.num_tokens}",
            flush=True,
        )

    # GROUNDED GOAL: a VLM (or stub) points the goal from RGB + instruction; the pixel seeds the
    # belief and odometry tracks it -> language decides WHERE, geometry never sees the world goal.
    grounder = None
    if args.grounder == "stub":
        from navdp.extensions import StubPixelGoal

        _uv = tuple(float(x) for x in args.grounder_uv.split(","))
        grounder = StubPixelGoal(uv=_uv, as_fraction=True)
    elif args.grounder == "qwen":
        from navdp.extensions import QwenVLPixelGoal

        grounder = QwenVLPixelGoal(model_id=args.grounder_model, device=device)
    if grounder is not None:
        args.belief_goal = True
        print(
            f"[VLA] grounded goal: {args.grounder} every {args.grounder_every} steps, "
            f"instruction={args.instruction!r}",
            flush=True,
        )

    mesh_goal_mode = bool(args.goal_mesh_uv)
    if mesh_goal_mode:
        args.belief_goal = (
            True  # reuse belief propagation + ghost recovery, but seed from the mask
        )
        # NOTE: obstacle comes from the rendered obstacle-MESH mask (below), NOT depth thresholding,
        # which would flag the whole near ground as an obstacle. Leave --obstacle-mode as-is.
        print(
            f"[MASK] rendered-mask goal at pixel {args.goal_mesh_uv}"
            + (
                f" + obstacle mesh at {args.obstacle_mesh_uv}"
                if args.obstacle_mesh_uv
                else ""
            ),
            flush=True,
        )

    sim = make_sim(
        Path(args.scene),
        args.height,
        args.width,
        args.hfov_deg,
        with_semantic=(mesh_goal_mode or args.goal_from_vlm),
    )
    agent = sim.initialize_agent(0)

    rocks, _rock_config = [], None
    if args.rock_field:
        rocks, _rock_config = load_rock_field(Path(args.rock_field))
        register_rocks(sim, rocks)
        print(f"[ROCKS] loaded {len(rocks)} rocks from {args.rock_field}", flush=True)

    x = float(args.start_x)
    z = float(args.start_z)
    yaw = math.radians(float(args.start_yaw_deg))
    dt = 1.0 / float(args.hz)

    cbf_obstacle_id = (
        "obstacle"  # identifies the single obstacle the live CBF math tracks, for
    )
    # cbf_events.jsonl -- refined below once VLM/ghost obstacle resolve
    vlm_goal_mesh = None
    vlm_mesh_tracking = (
        False  # True once the VLM's resolved goal mesh is registered with the
    )
    # semantic sensor below, so the main loop tracks it from the live
    # per-frame mask instead of one-shot pixel + odometry dead-reckoning.
    if args.goal_from_vlm:
        # SEMANTIC-MASK-tracked goal: resolve the VLM's object selection ONCE here, register its
        # already-saved mesh (selected_bbox_to_object_mesh's on-disk .obj) with the semantic sensor,
        # and re-render that mesh's mask every step (see MESH_GOAL_ID handling in the main loop) --
        # mirrors --goal-mesh-uv's rendered-mask tracking instead of dead-reckoning by odometry alone.
        frame_idx = args.vlm_frame_idx
        rgb_path = f"{VLM_OUT_DIR}/rgb_{frame_idx:04d}.png"
        overlay_path = f"{VLM_OUT_DIR}/rgb_{frame_idx:04d}_at.png"
        annotation_path = f"{VLM_ANNOTATIONS_DIR}/rgb_{frame_idx:04d}.json"

        if not args.manual_annotate:
            # DEFAULT: capture the actual live first frame (RGB + depth + pose) at THIS rollout's
            # own start pose -- run after sim/agent exist so it's a real render, not an assumption
            # that vlm_nav_interactive.py already produced these files on disk. SAM then annotates
            # it in place of a human labelme session; resolve_vlm_selection below is unchanged and
            # can't tell the difference between the two annotation sources.
            y0 = terrain.local_height_max(
                x, z, float(args.pose_terrain_radius)
            ) + float(args.clearance)
            set_agent_pose(agent, x, y0, z, yaw)
            obs0 = sim.get_sensor_observations()
            rgb0, depth0 = rgb_depth(obs0)
            os.makedirs(VLM_OUT_DIR, exist_ok=True)
            Image.fromarray(rgb0).save(rgb_path)
            np.save(
                f"{VLM_OUT_DIR}/depth_{frame_idx:04d}.npy", depth0.astype(np.float32)
            )
            depth_vis = (np.clip(depth0, 0.0, 10.0) / 10.0 * 255.0).astype(np.uint8)
            Image.fromarray(depth_vis).save(f"{VLM_OUT_DIR}/depth_{frame_idx:04d}.png")
            vlm_save_pose(frame_idx, x, y0, z, yaw)

            from sam_annotation_adapter import sam_frame_to_annotation

            annotation_path, sam_valid, sam_status = sam_frame_to_annotation(
                rgb_path, annotation_path
            )
            if not sam_valid:
                raise SystemExit(
                    f"--goal-from-vlm: SAM annotation invalid: {sam_status}"
                )
            print(
                f"[SAM] live-captured frame {frame_idx} at this rollout's start pose "
                f"({x:.2f},{z:.2f},{math.degrees(yaw):.1f}deg) -> {annotation_path}",
                flush=True,
            )

        # ANNOTATED FRAME FOR REFERENCE: draw the (SAM- or labelme-) annotation's
        # labeled boxes onto the raw frame and save it to overlay_path. resolve_vlm_selection()
        # doesn't do this itself (it only forwards overlay_path to query_vlm, which never
        # generates it -- see query_vlm's commented-out --overlay arg); the interactive
        # run_vlm_on_frame() draws it before calling resolve_vlm_selection, so mirror that here.
        draw_annotation_overlay(rgb_path, annotation_path, overlay_path)

        vlm_success, vlm_result, vlm_status = resolve_vlm_selection(
            rgb_path, overlay_path, annotation_path, frame_idx
        )
        if not vlm_success:
            raise SystemExit(f"--goal-from-vlm: VLM selection failed: {vlm_status}")
        vlm_response, vlm_goal_mesh, vlm_obstacle_meshes = vlm_result
        register_semantic_mesh(sim, vlm_goal_mesh["mesh_path"], MESH_GOAL_ID)
        vlm_mesh_tracking = True
        # Kept only as an inert fallback (unused once vlm_mesh_tracking routes through the
        # MESH_GOAL_ID branch below) in case the goal mesh ever fails to register.
        grounder = VlmSelectionPixelGoal(vlm_goal_mesh["bbox"], VLM_CAPTURE_HW)
        args.belief_goal = True
        print(
            f"[VLM] goal '{vlm_goal_mesh['label']}' mesh={vlm_goal_mesh['mesh_path']} registered as "
            f"MESH_GOAL_ID; belief re-derived from the live rendered mask every step (no dead-reckoning "
            f"while in view)",
            flush=True,
        )

        # OBSTACLE: the VLM's prompt (VLM_PROMPT in vlm_nav_interactive.py) already restricts
        # it to exactly one "rock" obstacle, chosen only from SAM's bigrock detections (bedrock
        # is dropped upstream in sam_annotation_adapter.py, so it's never a candidate). Register
        # its mesh too (MESH_OBST_ID) so the obstacle mask the policy sees, and the mask-based CBF
        # fallback, are also ground-truth per frame. ALSO wire its resolved world seed into
        # --ghost-obstacle-x/y/z so the CBF/orbit avoidance's cone-mode math (which prefers a
        # stable world point over the mask to avoid abeam-pass flicker -- see ctrl_op below) still
        # has one, unless the caller passed an explicit ghost obstacle of their own.
        if vlm_obstacle_meshes:
            cbf_obstacle_id = "vlm_obstacle"
            vlm_obstacle_mesh = vlm_obstacle_meshes[0]
            register_semantic_mesh(sim, vlm_obstacle_mesh["mesh_path"], MESH_OBST_ID)
            if args.ghost_obstacle_x is None and args.ghost_obstacle_z is None:
                obs_vx, obs_vy, obs_vz = vlm_obstacle_mesh["seed_world"]
                args.ghost_obstacle_x = obs_vx
                args.ghost_obstacle_z = obs_vz
                # y is left to the existing terrain-height + --ghost-obstacle-height computation
                # below (unless the caller passed --ghost-obstacle-y explicitly), same pattern as
                # the goal's y a few lines up -- seed_world's y sits at the rock's own surface, not
                # the elevated marker height the rest of this script expects for a ghost obstacle.
                print(
                    f"[VLM] obstacle '{vlm_obstacle_mesh['label']}' bbox={vlm_obstacle_mesh['bbox']} "
                    f"mesh={vlm_obstacle_mesh['mesh_path']} registered as MESH_OBST_ID; ghost world="
                    f"({obs_vx:.2f},{obs_vy:.2f},{obs_vz:.2f}) -> driving around it",
                    flush=True,
                )
        else:
            print(
                "[VLM] no obstacle resolved from the VLM's selection; proceeding without one",
                flush=True,
            )

        if args.manual_annotate:
            # The bbox is a FIXED pixel fraction from an OFFLINE labelme session (not a live
            # re-detection), so it's only valid if the rollout starts from ~the pose that frame
            # was captured at. Doesn't apply to the default SAM path above: that frame IS this
            # rollout's own start pose, by construction, so it can't be stale.
            if (
                abs(args.start_x - VLM_START_X) > 0.5
                or abs(args.start_z - VLM_START_Z) > 0.5
                or abs(args.start_yaw_deg - VLM_START_YAW_DEG) > 5.0
            ):
                print(
                    f"[WARN] --start-x/z/yaw-deg ({args.start_x},{args.start_z},{args.start_yaw_deg}) "
                    f"differ from the capture pose ({VLM_START_X},{VLM_START_Z},{VLM_START_YAW_DEG}); "
                    "the seeded pixel may not land on the object in the first live frame.",
                    flush=True,
                )

    if args.goal_from_vlm:
        # World position of the VLM selection, kept ONLY as the success-metric/logging
        # reference (mirrors --goal-bearing-deg) -- control is driven by the pixel-seeded
        # belief wired above, this world point is never read by the control path.
        goal_vx, goal_vy, goal_vz = vlm_goal_mesh["seed_world"]
        print(
            f"[VLM] goal '{vlm_goal_mesh['label']}' reference world=({goal_vx:.2f},{goal_vy:.2f},{goal_vz:.2f})",
            flush=True,
        )
        goal_y = args.goal_y
        if goal_y is None:
            goal_y = terrain.local_height_max(
                goal_vx, goal_vz, float(args.goal_terrain_radius)
            ) + float(args.goal_height)
        goal = np.asarray([goal_vx, goal_y, goal_vz], dtype=np.float32)

        # MISSION RECORD: first frame (rgb_path), its SAM-annotated overlay (overlay_path),
        # the annotation JSON (annotation_path), and the raw VLM prompt/response
        # (rgb_{idx}_vlm_prompt.txt / rgb_{idx}_vlm.txt, written by vlm_query.py) are already
        # on disk under VLM_OUT_DIR/VLM_ANNOTATIONS_DIR; this adds one consolidated JSON tying
        # the VLM's parsed goal+obstacle choice to their resolved world positions/meshes.
        save_mission_metadata(
            frame_idx,
            vlm_response,
            vlm_goal_mesh,
            vlm_obstacle_meshes,
            goal_target_world=goal,
        )
    elif args.goal_x is None or args.goal_z is None:
        if not mesh_goal_mode:
            raise SystemExit(
                "Pass --goal-x and --goal-z, --goal-from-vlm, or use --goal-mesh-uv for a rendered-mask goal."
            )
        goal = np.zeros(
            3, dtype=np.float32
        )  # placeholder; set from the mesh centroid at step 0
    else:
        goal_y = args.goal_y
        if goal_y is None:
            goal_y = terrain.local_height_max(
                float(args.goal_x), float(args.goal_z), float(args.goal_terrain_radius)
            ) + float(args.goal_height)
        goal = np.asarray(
            [float(args.goal_x), float(goal_y), float(args.goal_z)], dtype=np.float32
        )

    ghost_obstacle = None
    if (args.ghost_obstacle_x is None) != (args.ghost_obstacle_z is None):
        raise ValueError(
            "pass both --ghost-obstacle-x and --ghost-obstacle-z, or neither"
        )
    if args.ghost_obstacle_x is not None and args.ghost_obstacle_z is not None:
        obstacle_y = args.ghost_obstacle_y
        if obstacle_y is None:
            obstacle_y = terrain.local_height_max(
                float(args.ghost_obstacle_x),
                float(args.ghost_obstacle_z),
                float(args.pose_terrain_radius),
            ) + float(args.ghost_obstacle_height)
        ghost_obstacle = np.asarray(
            [
                float(args.ghost_obstacle_x),
                float(obstacle_y),
                float(args.ghost_obstacle_z),
            ],
            dtype=np.float32,
        )
        if cbf_obstacle_id == "obstacle":
            cbf_obstacle_id = "ghost"

    # Structured per-episode ablation logs (separate from the --out npz/manifest dump above):
    # config.json / obstacles.json up front, frames.jsonl / qwen_queries.jsonl / cbf_events.jsonl
    # appended to every step, summary.json once at the end. See sam_vla/logging/episode_logger.py.
    goal_mode = (
        "vlm" if args.goal_from_vlm else ("uv" if args.goal_mesh_uv else "coord")
    )
    steering_mode = "nudge" if args.qwen_steer else "none"
    log_config = {
        "scene_glb": str(Path(args.scene).expanduser().resolve()),
        "goal_mode": goal_mode,
        "goal_coord": (
            [float(args.goal_x), float(args.goal_z)]
            if (args.goal_x is not None and args.goal_z is not None)
            else None
        ),
        "steering_mode": steering_mode,
        "cbf_enabled": bool(args.cbf),
        "obstacle_count": (
            len(rocks) if rocks else (1 if ghost_obstacle is not None else 0)
        ),
        "obstacle_seed": _rock_config.seed if _rock_config is not None else None,
        "obstacle_distance_threshold_X": float(args.cbf_d_safe),
        "goal_distance_threshold_Y": float(args.stop_dist),
        "max_steps": int(args.max_steps),
        "agent_height_offset": float(args.clearance),
    }
    run_id = make_run_id(log_config)
    episode_logger = EpisodeLogger(
        run_id, log_config, log_root=args.log_root, save_frames=args.log_save_frames
    )
    if rocks:
        obstacle_records = [
            {
                "id": f"rock_{r.id:03d}",
                "position": [float(r.x), float(r.y), float(r.z)],
                "orientation": [float(v) for v in yaw_quat_xyzw(float(r.yaw))],
                "radius": float(r.radius),
                "is_goal": False,
            }
            for r in rocks
        ]
        episode_logger.write_obstacles(
            obstacle_records, goal_id=("vlm_goal" if args.goal_from_vlm else None)
        )
    elif ghost_obstacle is not None:
        episode_logger.write_obstacles(
            [
                {
                    "id": "ghost",
                    "position": [
                        float(ghost_obstacle[0]),
                        float(ghost_obstacle[1]),
                        float(ghost_obstacle[2]),
                    ],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "radius": float(args.ghost_obstacle_world_radius),
                    "is_goal": False,
                }
            ],
            goal_id=("vlm_goal" if args.goal_from_vlm else None),
        )
    else:
        episode_logger.write_obstacles(
            [], goal_id=("vlm_goal" if args.goal_from_vlm else None)
        )
    termination_reason = "timeout"

    rows = {
        k: []
        for k in [
            "rgb",
            "depth",
            "goal_mask",
            "obstacle_mask",
            "seg_masks",
            "pose",
            "proprio",
            "action_3d",
            "pred_chunk",
            "goal_visible_pixels",
            "goal_u",
            "goal_v",
            "goal_distance",
            "obstacle_visible_pixels",
            "obstacle_u",
            "obstacle_v",
            "obstacle_distance",
            "belief_fwd",
            "belief_left",  # body-frame belief_g each tick (nan if not tracking) -> lets
            "goal_frame_fraction",  # goal_px / total_px this tick, from goal_pixel_ratio()
        ]
    }  # a post-hoc script measure belief ACCURACY vs the true goal
    video_frames = []
    prev_obstacle_point = None
    last_pred_chunk = None
    chunk_len = 0
    replan_every = max(int(args.replan_every), 1)
    cbf_active = 0
    cone_side_latch = (
        None  # committed cone-projection side while the obstacle is in view
    )
    around_side = None  # committed tangent-pursuit side (+1 = pass on obstacle's left)
    hard_gate_fired = 0
    escape_active = 0
    vla_count = 0  # counter for --vla-dump paired-sample writing
    mesh_tracking_mode = (
        mesh_goal_mode or vlm_mesh_tracking
    )  # goal (+ obstacle, if resolved) tracked
    # from the live rendered semantic mask each step, not dead-reckoned
    belief_g = (
        None  # body-frame [forward, left] belief estimate of the goal (--belief-goal)
    )
    belief_rng = np.random.default_rng(0)
    near_obstacle_state = (
        False  # hysteresis-latched maneuver gate (avoids flicker at the boundary)
    )
    belief_state = False  # hysteresis-latched belief-return gate
    qwen_cmd_txt = (
        ""  # last command text received from --qwen-steer (persists between polls)
    )
    last_qwen_poll_t = 0.0
    qwen_steer_active = False  # logs once, the tick --qwen-steer polling starts

    print("Mars NavDP rollout", flush=True)
    print(f"  navdp_root : {navdp_root}", flush=True)
    print(f"  scene      : {Path(args.scene).expanduser().resolve()}", flush=True)
    print(f"  ckpt       : {Path(args.ckpt).expanduser().resolve()}", flush=True)
    print(
        f"  terrain    : {terrain.mode} scene_flip_x={args.scene_height_flip_x} "
        f"scene_flip_z={args.scene_height_flip_z} scene_swap_xz={args.scene_height_swap_xz}",
        flush=True,
    )
    print(f"  goal       : x={goal[0]:.2f} y={goal[1]:.2f} z={goal[2]:.2f}", flush=True)
    if (
        args.scene_mode != "mars"
        or args.obstacle_pool != "none"
        or args.episodes_per_category != 1
    ):
        print(
            "  compat    : scene/category generator flags were accepted but Mars runs one explicit scene",
            flush=True,
        )
    if args.lost_goal_ghost:
        print(
            f"  ghost     : lost-goal recovery enabled min_px={args.lost_goal_min_px} "
            f"turn_kp={args.lost_goal_turn_kp:g} forward={args.lost_goal_forward:g}",
            flush=True,
        )
    if ghost_obstacle is not None:
        print(
            f"  obstacle   : ghost x={ghost_obstacle[0]:.2f} "
            f"y={ghost_obstacle[1]:.2f} z={ghost_obstacle[2]:.2f}",
            flush=True,
        )
    print(
        f"  modes      : action={modes['action_mode']} proprio={modes['proprio_mode']} obstacle_channel={use_obstacle_channel}",
        flush=True,
    )

    try:
        for step in range(int(args.max_steps)):
            y = terrain.local_height_max(x, z, float(args.pose_terrain_radius)) + float(
                args.clearance
            )
            position = np.asarray([x, y, z], dtype=np.float32)
            set_agent_pose(agent, x, y, z, yaw)
            obs = sim.get_sensor_observations()
            rgb, depth = rgb_depth(obs)

            # LIVE language command (real-time inference). Read once per tick so it can drive the
            # sample below. With --vla-adapter the intent becomes the policy's text token (Regime
            # B: the POLICY executes the maneuver); without it, intent drives the orbit (Regime A).
            goal_dist_now = float(
                np.linalg.norm(goal[[0, 2]] - np.asarray([x, z], dtype=np.float32))
            )
            near_goal = goal_dist_now <= float(args.stop_dist) + float(
                args.cbf_deadzone
            )

            cmd_txt = args.command
            if args.command_file:
                try:
                    cmd_txt = (
                        Path(args.command_file).read_text(encoding="utf-8").strip()
                        or args.command
                    )
                except Exception:
                    pass

            if mesh_tracking_mode:
                # RENDERED-MASK goal: a semantic mesh (placed at t=0 from a pixel, or registered
                # up-front from the VLM's resolved selection) is rendered each step; the belief is
                # RE-DERIVED from that live mask every step it's visible, so tracking follows the
                # mask's ground truth instead of dead-reckoning by odometry alone (which drifts) --
                # dead-reckoning only bridges the gap while the mask briefly drops out of view.
                if mesh_goal_mode and step == 0:
                    _gw, _ow = place_mesh_goal_obstacle(
                        sim, depth, position, yaw, intr, args, out_dir
                    )
                    if _gw is not None:
                        goal[:] = np.asarray(_gw, dtype=np.float32)
                    obs = (
                        sim.get_sensor_observations()
                    )  # re-render now that the meshes exist
                    rgb, depth = rgb_depth(obs)
                _sem = semantic_from_obs(obs)
                _gm = np.where(_sem == MESH_GOAL_ID, 255, 0).astype(np.uint8)
                if int(_gm.sum()) >= int(args.lost_goal_min_px):
                    belief_g = mask_to_body(
                        _gm,
                        depth,
                        rgb.shape[0],
                        rgb.shape[1],
                        args.hfov_deg,
                        float(args.goal_range),
                    )
                    _ys, _xs = np.where(_gm > 0)
                    goal_mask = _gm
                    goal_info = {
                        "u": float(_xs.mean()),
                        "v": float(_ys.mean()),
                        "distance": (
                            float(np.hypot(belief_g[0], belief_g[1]))
                            if belief_g is not None
                            else float("nan")
                        ),
                        "visible": 1.0,
                    }
                elif (
                    belief_g is not None and belief_adapter is None
                ):  # ghost recovery = the P-controller path
                    goal_mask, goal_info = project_body_point_mask(
                        belief_g,
                        rgb.shape[0],
                        rgb.shape[1],
                        args.hfov_deg,
                        args.goal_radius,
                        clamp_to_edge=not args.no_clamp_goal_to_edge,
                    )
                else:
                    goal_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                    goal_info = {
                        "u": -1.0,
                        "v": -1.0,
                        "distance": float("nan"),
                        "visible": 0.0,
                    }
            elif args.belief_goal:
                # BELIEF-tracked goal: the ghost comes from a body-frame estimate propagated by
                # odometry. It is seeded either from an IMAGE bearing+range (no world xyz) or from
                # the world goal at t=0; with a world goal it can also correct on sight.
                if grounder is not None and (
                    belief_g is None
                    or (
                        not getattr(grounder, "one_shot", False)
                        and step % max(1, int(args.grounder_every)) == 0
                    )
                ):
                    # LANGUAGE grounds the goal: RGB + instruction -> pixel -> body point (belief)
                    pg = grounder.ground(rgb, args.instruction)
                    if pg.in_view:
                        bbox = getattr(pg, "bbox", None)
                        if bbox is not None:
                            # Robust: median depth over the whole VLM bbox, not one pixel that
                            # can land on a discontinuity (see bbox_to_body's docstring).
                            belief_g = bbox_to_body(
                                bbox,
                                depth,
                                rgb.shape[0],
                                rgb.shape[1],
                                args.hfov_deg,
                                args.goal_range,
                            )
                        else:
                            belief_g = pixel_to_body(
                                pg.u,
                                pg.v,
                                depth,
                                rgb.shape[0],
                                rgb.shape[1],
                                args.hfov_deg,
                                args.goal_range,
                            )
                elif belief_g is None:
                    if args.goal_bearing_deg is not None:
                        _b = math.radians(
                            float(args.goal_bearing_deg)
                        )  # + = right of forward
                        belief_g = np.asarray(
                            [
                                float(args.goal_range) * math.cos(_b),
                                -float(args.goal_range) * math.sin(_b),
                            ],
                            dtype=np.float32,
                        )
                    else:
                        _gr, _gu, _gf = camera_coords(goal, position, yaw)
                        belief_g = np.asarray([_gf, -_gr], dtype=np.float32)
                elif (
                    grounder is None
                    and args.goal_bearing_deg is None
                    and args.belief_update_on_sight
                ):
                    _gr, _gu, _gf = camera_coords(goal, position, yaw)
                    if _gf > 0.05:
                        belief_g = np.asarray(
                            [_gf, -_gr], dtype=np.float32
                        )  # correct drift on sight
                goal_mask, goal_info = project_body_point_mask(
                    belief_g,
                    rgb.shape[0],
                    rgb.shape[1],
                    args.hfov_deg,
                    args.goal_radius,
                    clamp_to_edge=not args.no_clamp_goal_to_edge,
                )
            else:
                goal_mask, goal_info = project_goal_mask(
                    goal=goal,
                    position=position,
                    yaw=yaw,
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    hfov_deg=args.hfov_deg,
                    radius=args.goal_radius,
                    clamp_to_edge=not args.no_clamp_goal_to_edge,
                )
            if args.obstacle_mode == "depth":
                obstacle_mask = depth_obstacle_mask(
                    depth, args.obstacle_depth_threshold, args.obstacle_min_y_frac
                )
            else:
                obstacle_mask = np.zeros_like(goal_mask, dtype=np.uint8)

            ghost_obstacle_mask = np.zeros_like(goal_mask, dtype=np.uint8)
            obstacle_info = {
                "u": -1.0,
                "v": -1.0,
                "range": float("nan"),
                "visible": 0.0,
            }
            ghost_obstacle_point = None
            if ghost_obstacle is not None:
                ghost_obstacle_mask, obstacle_info = project_goal_mask(
                    goal=ghost_obstacle,
                    position=position,
                    yaw=yaw,
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    hfov_deg=args.hfov_deg,
                    radius=args.ghost_obstacle_radius,
                    clamp_to_edge=not args.no_clamp_obstacle_to_edge,
                )
                ghost_obstacle_point = obstacle_point_from_world(
                    ghost_obstacle, position, yaw
                )
                obstacle_mask = np.maximum(obstacle_mask, ghost_obstacle_mask).astype(
                    np.uint8
                )

            if mesh_tracking_mode:
                # obstacle = ONLY the rendered obstacle-mesh pixels (semantic id), never the ground.
                # All-zero if no obstacle mesh was registered (no --obstacle-mesh-uv / no VLM obstacle
                # resolved), same as before -- this only ever ADDS ground-truth precision, never
                # removes the "no obstacle" case.
                obstacle_mask = np.where(_sem == MESH_OBST_ID, 255, 0).astype(np.uint8)

            goal_ratio = goal_pixel_ratio(goal_mask)

            if args.qwen_steer:
                # Near the goal, hand steering over to the persistent Qwen VLM server, polled at
                # qwen_steer_hz (NOT every control tick -- --hz is typically much higher). Its
                # reply is fed straight into command_intent below, same as any live command text.
                # Runs AFTER goal_mask/obstacle_mask are finalized for THIS step so the frame sent
                # to the VLM is this tick's rgb (never a stale/reused frame) with the current goal
                # mask composited on top -- without that overlay the model has no way to tell which
                # object in frame the prompt's "goal obstacle" refers to.
                if not qwen_steer_active:
                    print(
                        f"[qwen-steer] near goal (dist={goal_dist_now:.2f}m) -- polling Qwen at {args.qwen_steer_hz:g}Hz",
                        flush=True,
                    )
                    qwen_steer_active = True
                now_t = time.time()
                if now_t - last_qwen_poll_t >= 1.0 / float(args.qwen_steer_hz):
                    last_qwen_poll_t = now_t
                    frame_path = str(out_dir / "qwen_steer_frame.jpg")
                    steer_img = rgb.astype(np.float32).copy()
                    # The raw goal_mask (esp. the rendered semantic-mesh mask in --goal-from-vlm
                    # mode) comes out as a sparse crosshatch -- likely z-fighting between the goal
                    # mesh and the terrain it rests on -- rather than a solid blob. Reading as
                    # scattered noise instead of "a close, large green region" made Qwen ignore the
                    # measured frame_fraction and never choose stop. Close the mask ONLY for this
                    # composited display frame (goal_mask itself, used for belief/physics/frame_
                    # fraction/logging, is untouched) so what the VLM sees matches what it's told.
                    gm_display = (
                        cv2.morphologyEx(
                            (goal_mask > 0).astype(np.uint8) * 255,
                            cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                            iterations=2,
                        )
                        > 0
                    )
                    steer_img[gm_display] = (
                        steer_img[gm_display] * 0.5
                        + np.asarray([0.0, 255.0, 0.0]) * 0.5
                    )
                    # obstacle_mask suffers the same rendered-mesh z-fighting sparseness as goal_mask
                    # in --goal-from-vlm mode (both come from MESH_GOAL_ID/MESH_OBST_ID semantic
                    # renders), so close it the same way before compositing. Applied after the goal
                    # tint so an obstacle wins any pixel where the two masks overlap, matching
                    # overlay_frame's ordering.
                    om_display = (
                        cv2.morphologyEx(
                            (obstacle_mask > 0).astype(np.uint8) * 255,
                            cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                            iterations=2,
                        )
                        > 0
                    )
                    steer_img[om_display] = (
                        steer_img[om_display] * 0.5
                        + np.asarray([255.0, 0.0, 0.0]) * 0.5
                    )
                    Image.fromarray(
                        np.clip(steer_img, 0, 255).astype(np.uint8)
                    ).convert("RGB").save(frame_path)
                    steer_prompt = qwen_steer_prompt(
                        goal_ratio["frame_fraction"] * 100.0
                    )
                    query_t0 = time.perf_counter()
                    try:
                        qwen_cmd_txt = qwen_vlm_client.query_vlm_persistent(
                            frame_path,
                            prompt=steer_prompt,
                            max_new_tokens=512,
                            host=args.qwen_host,
                            port=args.qwen_port,
                        )
                        print(
                            f"[qwen-steer] t={now_t:.2f} step={step} goal_frame_pct={goal_ratio['frame_fraction'] * 100.0:.2f} -> {qwen_cmd_txt!r}",
                            flush=True,
                        )
                        episode_logger.log_qwen_query(
                            step=step,
                            query_type="action_suggestion",
                            trigger="goal_proximity",
                            input_data={
                                "goal_belief": {"range": goal_dist_now},
                                "goal_frame_fraction": goal_ratio["frame_fraction"],
                                "frame": frame_path,
                            },
                            output_data={"action": qwen_cmd_txt},
                            latency_ms=(time.perf_counter() - query_t0) * 1000.0,
                        )
                    except Exception as e:
                        print(f"[qwen-steer] query failed: {e}", flush=True)
                cmd_txt = qwen_cmd_txt
            intent = command_intent(cmd_txt)
            force_side = (
                1.0 if intent == "left" else (-1.0 if intent == "right" else None)
            )
            vla_token = None  # set below, once the obstacle distance is known
            stop_cmd = False  # set below (gated on obstacle proximity)

            spatial = frame_to_spatial(
                depth,
                goal_mask,
                image_size,
                obstacle_mask,
                include_obstacle_channel=use_obstacle_channel,
            ).to(device)
            obstacle_map = (
                obstacle_builder.build(depth)
                if args.obstacle_mode == "depth"
                else np.zeros((96, 96), dtype=np.float32)
            )
            obstacle_map = paint_obstacle_map_point(
                obstacle_map,
                obstacle_builder,
                ghost_obstacle_point,
                args.ghost_obstacle_map_radius,
            )
            obstacle_t = torch.from_numpy(obstacle_map[None]).float().to(device)

            qx, qy, qz, qw = yaw_quat_xyzw(yaw)
            pose = np.asarray([x, y, z, qx, qy, qz, qw], dtype=np.float32)
            proprio = _proprio_from_pose(
                pose,
                modes["proprio_mode"],
                planar_axes=(0, 2),
                yaw_axis=modes["yaw_axis"],
            )
            proprio_t = torch.from_numpy(proprio[None]).float().to(device)
            belief_t = torch.from_numpy(_empty_belief_tensor()[None]).float().to(device)
            route_index = torch.zeros(1, dtype=torch.long, device=device)
            active_goal_index = torch.zeros(1, dtype=torch.long, device=device)

            obstacle_point = None
            if args.cbf and int(obstacle_mask.sum()) > 0:
                obstacle_point = ghost_obstacle_point
                if obstacle_point is None:
                    obstacle_point = nearest_obstacle_point(obstacle_mask, depth, intr)
            if obstacle_point is None:
                cone_side_latch = (
                    None  # obstacle gone -> release the committed cone-proj side
                )
                # (around_side is released below when the obstacle stops blocking the goal ray)

            # Proximity gate: a maneuver command applies WHEN YOU REACH the obstacle. Drive
            # straight toward the goal until the obstacle is close (within d_safe+deadzone AND
            # ahead), then turn (left/right) or stop; after passing, continue to the goal. Without
            # this a persistent command would turn/stop the whole way.
            # Hysteresis: ENTER near-obstacle at d_safe+deadzone, only EXIT past an extra margin, so
            # small distance jitter right at the boundary can't flip the gate back and forth every
            # tick (that flicker -- not the action smoother -- was the source of the jerkiness).
            near_obstacle_enter = (
                obstacle_point is not None
                and float(obstacle_point[0]) > 0.0
                and float(np.hypot(obstacle_point[0], obstacle_point[1]))
                < args.cbf_d_safe + args.cbf_deadzone
            )
            near_obstacle_exit = (
                obstacle_point is None
                or float(obstacle_point[0]) <= 0.0
                or float(np.hypot(obstacle_point[0], obstacle_point[1]))
                > args.cbf_d_safe + args.cbf_deadzone + float(args.lang_turn_hyst)
            )
            if near_obstacle_enter:
                near_obstacle_state = True
            elif near_obstacle_exit:
                near_obstacle_state = False
            near_obstacle = near_obstacle_state
            # "stop" is a language TRIGGER, but the HALT itself is proximity-gated: engage once close
            # to the obstacle OR the goal (whichever comes first), not the instant "stop" is said.
            # Without the goal term, "stop" never fired on a goal-only run (no obstacle to be near).
            # (goal_dist_now / near_goal already computed above, at the top of the loop.)
            near_goal_visual = goal_ratio["frame_fraction"] >= float(
                args.qwen_stop_frame_fraction
            )
            stop_cmd = (intent == "stop") and (
                near_obstacle or near_goal or near_goal_visual
            )
            if vla_adapter is not None:
                # Hard, full-strength switch (NOT a blend): interpolating between two different
                # instruction tokens landed off the trained manifold -- the adapter only ever
                # produces alpha*adapter(ONE instruction), never a weighted sum of two -- and since
                # diffusion sampling is nonlinear in its conditioning, every replan along a blend
                # sampled an uncorrelated, effectively random chunk (far worse than one clean cut).
                # The hysteresis above still does its job: it debounces near_obstacle so this switch
                # doesn't fire repeatedly from boundary jitter.
                if intent in ("left", "right") and near_obstacle:
                    _phrase = cmd_txt
                elif intent in ("left", "right", "straight", "stop"):
                    _phrase = "navigate to the goal"
                else:
                    _phrase = None
                if _phrase is not None:
                    if _phrase not in vla_tok_cache:
                        with torch.no_grad():
                            _e = (
                                torch.from_numpy(
                                    vla_text_enc.encode(
                                        [_phrase], normalize_embeddings=True
                                    )
                                )
                                .float()
                                .to(device)
                            )
                            vla_tok_cache[_phrase] = float(
                                args.vla_alpha_scale
                            ) * vla_adapter(_e)
                    vla_token = vla_tok_cache[_phrase]

            # BELIEF-RETURN: when the goal is OFF-SCREEN (mask gone) inject the belief token so the
            # POLICY turns back toward it -- the learned return that replaces the P-controller.
            # Hysteresis only (enter on empty mask, exit only once well re-acquired); full-strength,
            # same reasoning as above -- no magnitude ramp, so every replan sees a token identical
            # to the one the adapter was trained/ablated on, not an in-between value.
            goal_px = int((goal_mask > 0).sum())
            exit_px = (
                int(args.belief_reacquire_px)
                if args.belief_reacquire_px is not None
                else 3 * int(args.lost_goal_min_px)
            )
            if belief_adapter is not None and belief_g is not None:
                if goal_px < int(args.lost_goal_min_px):
                    belief_state = True
                elif goal_px > exit_px:
                    belief_state = False
            else:
                belief_state = False
            belief_token = None
            if belief_adapter is not None and belief_g is not None and belief_state:
                with torch.no_grad():
                    belief_token = belief_adapter(
                        torch.from_numpy(belief_feat(belief_g)[None]).float().to(device)
                    )
            _toks = [t for t in (belief_token, vla_token) if t is not None]
            extra_cond = torch.cat(_toks, dim=1) if _toks else None

            do_replan = (step % replan_every == 0) or (last_pred_chunk is None)
            if do_replan:
                pred = model.sample(
                    spatial,
                    proprio_t,
                    steps=int(args.sample_steps),
                    belief_tensor=belief_t,
                    obstacle_map=obstacle_t,
                    route_index=route_index,
                    active_goal_index=active_goal_index,
                    extra_cond_tokens=extra_cond,  # belief token (off-screen) and/or language token
                )

                if (
                    args.cbf
                    and args.cbf_mode == "cone"
                    and obstacle_point is not None
                    and not args.vla_adapter
                ):
                    cbf_active += 1
                    v_o = np.zeros(2, dtype=np.float32)
                    if args.zero_lateral and pred.shape[-1] >= 3:
                        pred = pred.clone()
                        pred[..., 1] = 0.0
                    p_lat = float(obstacle_point[1])
                    side = -1.0 if p_lat > 0.0 else 1.0
                    if args.cbf_commit_side:
                        if cone_side_latch is None:
                            cone_side_latch = side
                        side = cone_side_latch
                    cone_sigma = None
                    if args.cbf_metric == "mahalanobis":
                        cone_sigma = horizon_growth_covariance(
                            pred.shape[1],
                            pred.shape[2],
                            base=args.cbf_cov_base,
                            growth=args.cbf_cov_growth,
                            mode=args.cbf_cov_mode,
                            device=pred.device,
                            dtype=pred.dtype,
                        )
                    if (
                        args.cbf_radius_mode == "perceived"
                        and ghost_obstacle is not None
                    ):
                        r_used = (
                            args.ghost_obstacle_world_radius
                            + args.robot_radius
                            + args.safety_margin
                        )
                    else:
                        r_used = args.cbf_d_safe
                    pred = project_chunk_cone(
                        pred,
                        obstacle_point,
                        v_o,
                        r=r_used,
                        dt=dt,
                        vel_scale=1.0,
                        iters=args.cbf_proj_iters,
                        lr=args.cbf_proj_lr,
                        trust=args.cbf_trust,
                        margin=args.cbf_cone_margin,
                        smooth_weight=args.cbf_smooth,
                        keep_speed=args.cbf_keep_speed,
                        sigma=cone_sigma,
                        deadzone_range=r_used + args.cbf_deadzone,
                        side=side,
                    )
                    prev_obstacle_point = obstacle_point

                pred_chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
                chunk_ctrl = np.stack(
                    [
                        action_to_control(
                            a,
                            action_mode=modes["action_mode"],
                            max_forward_speed=args.max_forward_speed,
                            max_lateral_speed=args.max_lateral_speed,
                            max_yaw_rate=args.max_yaw_rate,
                        )
                        for a in pred_chunk
                    ]
                ).astype(np.float32)
                smoother.add(step, chunk_ctrl)
                last_pred_chunk = pred_chunk
                chunk_len = int(pred_chunk.shape[0])
                if step == 0 and replan_every > chunk_len:
                    print(
                        f"[WARN] --replan-every {replan_every} > chunk length {chunk_len}; "
                        "actions will repeat after the buffer runs dry.",
                        flush=True,
                    )
            else:
                pred_chunk = last_pred_chunk
            _ = prev_obstacle_point

            action_3d = smoother.get(step)
            goal_lost = int(goal_mask.sum()) < int(args.lost_goal_min_px)

            # Perceived safety radius (obstacle + rover + margin) or a fixed d_safe.
            if args.cbf_radius_mode == "perceived" and ghost_obstacle is not None:
                r_gate = (
                    args.ghost_obstacle_world_radius
                    + args.robot_radius
                    + args.safety_margin
                )
            else:
                r_gate = args.cbf_d_safe

            # Physical collision radius (obstacle + rover + margin). The orbit hugs the larger
            # r_gate circle; this smaller radius is only the hard-breach backstop.
            r_cone = (
                args.ghost_obstacle_world_radius
                + args.robot_radius
                + args.safety_margin
            )

            # Control obstacle point [forward, left]. For a GHOST obstacle use the RAW geometry
            # (no forward>0.05 cutoff) so distance/bearing never flicker as it passes abeam --
            # that flicker toggles the avoid state and chatters the commands. Perceived
            # obstacles keep the mask-based obstacle_point.
            ctrl_op = None
            if args.cbf and args.cbf_mode == "cone":
                if ghost_obstacle is not None:
                    _r, _u, _f = camera_coords(ghost_obstacle, position, yaw)
                    ctrl_op = np.asarray([_f, -_r], dtype=np.float32)  # [forward, left]
                elif obstacle_point is not None:
                    ctrl_op = np.asarray(obstacle_point, dtype=np.float32)

            # Within the cone+deadzone shell, and is the obstacle actually BETWEEN us and the
            # goal? Hysteresis on the perpendicular clearance (wider to leave than to enter) so
            # the orbit<->goal decision cannot rapid-toggle at the boundary.
            avoiding = (
                ctrl_op is not None
                and float(np.hypot(ctrl_op[0], ctrl_op[1])) < r_gate + args.cbf_deadzone
            )
            blocked = False
            if ctrl_op is not None:
                ox, oy = float(ctrl_op[0]), float(ctrl_op[1])
                L = math.hypot(ox, oy)
                phi = math.atan2(oy, ox)
                beta = (
                    math.atan2(float(belief_g[1]), float(belief_g[0]))
                    if (args.belief_goal and belief_g is not None)
                    else planar_goal_bearing(position, yaw, goal)
                )
                proj = ox * math.cos(beta) + oy * math.sin(beta)
                perp = math.sqrt(max(L * L - proj * proj, 0.0))
                thresh = r_gate + (
                    args.cbf_orbit_hyst if around_side is not None else 0.0
                )
                blocked = avoiding and (proj > 0.0) and (perp < thresh)

            # (language command already read at the top of the loop -> intent / force_side /
            # stop_cmd / vla_token)

            # Ghost heading assist. The goal ghost is a binary mask; once the goal drifts
            # past ~hfov/2 it clamps to the SAME border pixel no matter how far off-axis it
            # is, so the mask-conditioned policy only steers weakly and can't tell "just
            # off-screen" from "behind me". Override the yaw with a proportional turn toward
            # the true goal bearing whenever the goal is off-centre OR fully behind:
            #   - off to the side (still ahead): KEEP the policy's forward and only steer
            #     hard, so we arc toward the goal without stopping (no stop-and-pivot judder);
            #   - fully behind (mask empty): pivot with the forward floor.
            # Suppressed only while the rock BLOCKS the goal ray (the orbit owns steering then),
            # so the goal pull can't drag us back through it; a nearby-but-clear rock still lets
            # the assist run.
            if (
                args.lost_goal_ghost and not blocked and belief_adapter is None
            ):  # belief adapter replaces this P-controller
                bearing = (
                    math.atan2(float(belief_g[1]), float(belief_g[0]))
                    if (args.belief_goal and belief_g is not None)
                    else planar_goal_bearing(position, yaw, goal)
                )
                goal_behind = goal_lost
                goal_offcentre = args.lost_goal_bearing_deg > 0.0 and abs(
                    bearing
                ) > math.radians(float(args.lost_goal_bearing_deg))
                if goal_behind or goal_offcentre:
                    yaw_cmd = float(
                        np.clip(
                            float(args.lost_goal_turn_kp) * bearing,
                            -float(args.max_yaw_rate),
                            float(args.max_yaw_rate),
                        )
                    )
                    fwd = (
                        float(args.lost_goal_forward)
                        if goal_behind
                        else float(action_3d[0])
                    )
                    fwd = max(fwd, float(args.lost_goal_forward))
                    action_3d = np.asarray([fwd, 0.0, yaw_cmd], dtype=np.float32)
            if args.zero_lateral and action_3d.shape[0] >= 2:
                action_3d = action_3d.copy()
                action_3d[1] = 0.0
            if args.cbf and args.cbf_mode == "project" and obstacle_point is not None:
                action_3d, _ = project_forward_velocity_cbf(
                    action_3d,
                    obstacle_point,
                    np.zeros(2, dtype=np.float32),
                    d_safe=args.cbf_d_safe,
                    gamma=args.cbf_gamma,
                    deadzone=args.cbf_deadzone,
                    trust=args.cbf_trust,
                )

            # Smooth ORBIT controller around the obstacle's safety circle. While the rock blocks
            # the goal ray, steer along the tangent (phi + side*90deg) plus a linear radial
            # pull-back toward the r_gate circle: it settles ON the circle and traces a smooth
            # line-arc-line detour at constant cruise -> no stop/rotate/go judder, and no
            # asin-tangent bounce when hugging tight. Body frame is [forward, left]; +heading
            # turns left. Released back to goal-seeking once the rock clears the ray (+hyst).
            if blocked and args.cbf_escape_yaw > 0.0 and not args.vla_adapter:
                # Commit which way around: the tangent heading closest to the goal bearing
                # (least detour, natural return). Latched until the rock stops blocking.
                if force_side is not None:
                    around_side = (
                        force_side  # language command overrides the geometric side
                    )
                elif around_side is None:
                    a = math.asin(min(1.0, r_gate / max(L, 1e-6)))
                    dl = abs(wrap_angle(phi + a - beta))
                    dr = abs(wrap_angle(phi - a - beta))
                    around_side = 1.0 if dl <= dr else -1.0
                corr = max(-1.2, min(1.2, float(args.cbf_orbit_kr) * (L - r_gate)))
                psi = wrap_angle(phi + around_side * (0.5 * math.pi - corr))
                yaw_cmd = float(
                    np.clip(
                        float(args.cbf_pursuit_kp) * psi,
                        -float(args.max_yaw_rate),
                        float(args.max_yaw_rate),
                    )
                )
                nominal_action_orbit = [float(v) for v in action_3d]
                action_3d = np.asarray(
                    [float(args.cbf_goaround_forward), 0.0, yaw_cmd], dtype=np.float32
                )
                escape_active += 1
                episode_logger.log_cbf_event(
                    step=step,
                    obstacle_id=cbf_obstacle_id,
                    distance=float(L),
                    nominal_action=nominal_action_orbit,
                    overridden_action=[float(v) for v in action_3d],
                    mode="orbit",
                )
            elif around_side is not None:
                around_side = (
                    None  # rock no longer blocks the goal ray -> release the side
                )

            # HARD per-tick safety backstop for cone mode. Tangent pursuit steers along the
            # r_gate circle so it never approaches closer than r_gate (> the collision radius
            # r_cone); the distance brake's approach-rate term would otherwise fight that by
            # braking the cruise during the turn-in. So WHILE pursuit is active and we are
            # still outside the collision radius, trust the steering and stay smooth; only if
            # we somehow penetrate r_cone (a genuine breach) do we fall back to the brake.
            # When pursuit is off (escape-yaw 0), the plain distance brake applies as before.
            if (
                args.cbf
                and args.cbf_mode == "cone"
                and args.cbf_hard_gate
                and ctrl_op is not None
            ):
                p_fwd, p_lat = float(ctrl_op[0]), float(ctrl_op[1])
                # Release on LATERAL clearance, not distance: driving straight forward MISSES the
                # obstacle once its lateral offset exceeds the collision radius (i.e. we have
                # turned enough). Releasing on distance never holds once the policy drives up
                # close, so it brakes forever and pins the rover in front of the rock. Committing
                # the pass the moment the heading clears it lets the (turning) policy drive around.
                cone_clears = (p_fwd <= 0.0) or (abs(p_lat) >= r_cone)
                if cone_clears and float(action_3d[0]) > 0.0:
                    pass  # turned enough that forward motion misses the obstacle -> let it drive
                else:
                    nominal_action_gate = [float(v) for v in action_3d]
                    action_3d, _gated = project_forward_velocity_cbf(
                        action_3d,
                        ctrl_op,
                        np.zeros(2, dtype=np.float32),
                        d_safe=r_gate,
                        gamma=args.cbf_gamma,
                        deadzone=args.cbf_deadzone,
                        trust=None,
                    )
                    if _gated:
                        hard_gate_fired += 1
                        episode_logger.log_cbf_event(
                            step=step,
                            obstacle_id=cbf_obstacle_id,
                            distance=float(math.hypot(p_fwd, p_lat)),
                            nominal_action=nominal_action_gate,
                            overridden_action=[float(v) for v in action_3d],
                            mode="brake",
                        )

            # VLA counterfactual data: at blocked steps, save the NEUTRAL-goal observation the
            # policy sees, plus ALL FOUR instruction targets on that SAME observation:
            #   left / right = orbit around each side (homotopy classes),
            #   stop         = decelerate-to-stop before the obstacle,
            #   straight     = navigate to the goal (default / prior-preservation).
            # Four targets, one observation -> the ONLY thing that can explain the difference is
            # the instruction, so the language adapter is forced to use the text.
            if args.vla_dump and blocked and ghost_obstacle is not None:
                if vla_count % max(1, int(args.vla_dump_every)) == 0:
                    dump_dir = Path(args.vla_dump)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    Hc = int(args.vla_horizon)
                    kr, cr, kp, mw = (
                        args.cbf_orbit_kr,
                        args.cbf_goaround_forward,
                        args.cbf_pursuit_kp,
                        args.max_yaw_rate,
                    )
                    ck_left = orbit_chunk(
                        position,
                        yaw,
                        ghost_obstacle,
                        1.0,
                        Hc,
                        dt,
                        r_gate,
                        kr,
                        cr,
                        kp,
                        mw,
                    )
                    ck_right = orbit_chunk(
                        position,
                        yaw,
                        ghost_obstacle,
                        -1.0,
                        Hc,
                        dt,
                        r_gate,
                        kr,
                        cr,
                        kp,
                        mw,
                    )
                    ck_stop = brake_chunk(cr, Hc)
                    ck_straight = goal_chunk(
                        position, yaw, goal, Hc, dt, cr, args.lost_goal_turn_kp, mw
                    )
                    np.savez_compressed(
                        str(dump_dir / f"{Path(args.out).name}_{vla_count:06d}.npz"),
                        spatial=spatial.detach().cpu().numpy()[0].astype(np.float32),
                        proprio=proprio.astype(np.float32),
                        obstacle_map=obstacle_map.astype(np.float32),
                        chunk_left=ck_left.astype(np.float32),
                        chunk_right=ck_right.astype(np.float32),
                        chunk_stop=ck_stop.astype(np.float32),
                        chunk_straight=ck_straight.astype(np.float32),
                        classes=np.array("left,right,stop,straight"),
                    )
                vla_count += 1

            if stop_cmd:
                action_3d = np.zeros(
                    3, dtype=np.float32
                )  # real-time STOP command halts the rover

            next_position, next_yaw = integrate_mars(position, yaw, action_3d, dt)
            x = float(
                np.clip(
                    next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
                )
            )
            z = float(
                np.clip(
                    next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
                )
            )
            yaw = wrap_angle(next_yaw)

            # Log belief_g BEFORE propagation, so it's relative to the SAME (pre-move) pose already
            # saved in rows["pose"] -- logging it after propagate_body_point (post-move pose) but
            # pairing it with the pre-move saved pose applies the WRONG yaw to reconstruct the world
            # estimate: a one-tick rotation mismatch, which at real belief range (~10m) becomes a
            # few-metre positional error -- flat across the episode (same mismatch every tick), not
            # growing drift. That's what produced a suspiciously constant ~3.6m "error" before.
            if belief_g is not None:
                bf, bl = float(belief_g[0]), float(belief_g[1])
            else:
                bf, bl = float("nan"), float("nan")

            # Propagate the goal belief by the executed motion (dead-reckoning; drifts if noisy).
            if args.belief_goal and belief_g is not None:
                belief_g = propagate_body_point(
                    belief_g, action_3d, dt, args.belief_odom_noise, belief_rng
                )

            goal_dist = float(
                np.linalg.norm(goal[[0, 2]] - np.asarray([x, z], dtype=np.float32))
            )
            seg = np.zeros_like(goal_mask, dtype=np.uint8)
            seg[goal_mask > 0] = 1
            seg[obstacle_mask > 0] = 2

            rows["rgb"].append(rgb)
            rows["depth"].append(depth)
            rows["goal_mask"].append(goal_mask.astype(np.uint8))
            rows["obstacle_mask"].append(obstacle_mask.astype(np.uint8))
            rows["seg_masks"].append(seg.astype(np.uint8))
            rows["pose"].append(pose)
            rows["proprio"].append(proprio.astype(np.float32))
            rows["action_3d"].append(action_3d.astype(np.float32))
            rows["pred_chunk"].append(pred_chunk.astype(np.float32))
            rows["goal_visible_pixels"].append(int(goal_mask.sum()))
            rows["goal_u"].append(float(goal_info["u"]))
            rows["goal_v"].append(float(goal_info["v"]))
            rows["goal_distance"].append(goal_dist)
            rows["obstacle_visible_pixels"].append(int(obstacle_mask.sum()))
            rows["obstacle_u"].append(float(obstacle_info["u"]))
            rows["obstacle_v"].append(float(obstacle_info["v"]))
            rows["obstacle_distance"].append(float(obstacle_info["range"]))
            rows["belief_fwd"].append(bf)
            rows["belief_left"].append(bl)
            rows["goal_frame_fraction"].append(goal_ratio["frame_fraction"])

            if rocks:
                distances_to_obstacles = {
                    f"rock_{r.id:03d}": float(math.hypot(pose[0] - r.x, pose[2] - r.z))
                    for r in rocks
                }
            elif ghost_obstacle is not None:
                distances_to_obstacles = {
                    cbf_obstacle_id: float(obstacle_info["range"])
                }
            else:
                distances_to_obstacles = {}
            episode_logger.log_frame(
                step=step,
                position=[float(pose[0]), float(pose[1]), float(pose[2])],
                orientation=[
                    float(pose[3]),
                    float(pose[4]),
                    float(pose[5]),
                    float(pose[6]),
                ],
                action={
                    "v_fwd": float(action_3d[0]),
                    "v_lat": float(action_3d[1]),
                    "yaw_rate": float(action_3d[2]),
                },
                distances_to_obstacles=distances_to_obstacles,
                cbf_active=bool(avoiding),
                goal_belief={
                    "range": goal_dist,
                    "bearing": math.degrees(planar_goal_bearing(position, yaw, goal)),
                    "source": (
                        "observed"
                        if int(goal_mask.sum()) >= int(args.lost_goal_min_px)
                        else "dead_reckoned"
                    ),
                },
            )
            episode_logger.log_rendered_frame(step, rgb)

            if step % max(int(args.save_every), 1) == 0:
                lost_txt = (
                    " LOST" if int(goal_mask.sum()) < int(args.lost_goal_min_px) else ""
                )
                text = f"t={step} dist={goal_dist:.2f} obs={int(obstacle_mask.sum())} v={action_3d[0]:.2f} yaw={math.degrees(yaw):.1f}{lost_txt}"
                frame = overlay_frame(rgb, goal_mask, obstacle_mask, text)
                frame.save(frame_dir / f"frame_{step:04d}.png")
                video_frames.append(frame)
                # binary mask: goal=white, obstacle=red, background=black
                mimg = np.zeros(
                    (goal_mask.shape[0], goal_mask.shape[1], 3), dtype=np.uint8
                )
                mimg[goal_mask > 0] = (255, 255, 255)
                mimg[obstacle_mask > 0] = (255, 0, 0)
                Image.fromarray(mimg).save(frame_dir / f"mask_{step:04d}.png")

            if step % 10 == 0:
                print(
                    f"step {step:04d} | dist={goal_dist:.2f} | goal_px={int(goal_mask.sum())} "
                    f"({goal_ratio['frame_fraction'] * 100.0:.2f}% of frame) "
                    f"| obs_px={int(obstacle_mask.sum())} "
                    f"| action=[{action_3d[0]:.2f},{action_3d[1]:.2f},{action_3d[2]:.2f}]",
                    flush=True,
                )
            # Stop on the TRUE world distance, not the belief estimate: belief_g can collapse much
            # faster than the rover actually moves (its centroid drifts onto near ground as the mask
            # grows), which stopped the run ~4-5m short while belief read <1.2m. goal_dist is real
            # (computed from the actual world goal position each tick) -- always use it to arrive.
            belief_dist = (
                float(np.hypot(belief_g[0], belief_g[1]))
                if belief_g is not None
                else float("nan")
            )
            if goal_dist <= float(args.stop_dist):
                print(
                    f"Reached goal at step {step} dist={goal_dist:.2f}m (belief={belief_dist:.2f})",
                    flush=True,
                )
                termination_reason = "goal_reached"
                break
            if (
                stop_cmd
            ):  # language "stop" already halted (action zeroed above) -- end the rollout,
                print(
                    f"Stopped by language command at step {step} dist={goal_dist:.2f}m",
                    flush=True,
                )
                termination_reason = "qwen_stop"
                break
    except BaseException:
        # Make sure the structured logs are flushed and closed even if the rollout crashes
        # mid-episode -- the happy-path finalize() call below is skipped once this re-raises.
        episode_logger.finalize({"success": False, "termination_reason": "exception"})
        raise
    finally:
        sim.close()

    print(
        f"[CBF diag] cbf_active={cbf_active} hard_gate_fired={hard_gate_fired} "
        f"escape_active={escape_active}",
        flush=True,
    )
    success = bool(
        rows["goal_distance"] and rows["goal_distance"][-1] <= float(args.stop_dist)
    )
    episode_logger.finalize(
        {
            "success": success,
            "termination_reason": termination_reason,
        }
    )
    npz_path = out_dir / "rollout.npz"
    np.savez_compressed(
        npz_path,
        rgb=np.stack(rows["rgb"]).astype(np.uint8),
        depth=np.stack(rows["depth"]).astype(np.float32),
        goal_mask=np.stack(rows["goal_mask"]).astype(np.uint8),
        obstacle_mask=np.stack(rows["obstacle_mask"]).astype(np.uint8),
        seg_masks=np.stack(rows["seg_masks"]).astype(np.uint8),
        pose=np.stack(rows["pose"]).astype(np.float32),
        proprio=np.stack(rows["proprio"]).astype(np.float32),
        action_3d=np.stack(rows["action_3d"]).astype(np.float32),
        pred_chunk=np.stack(rows["pred_chunk"]).astype(np.float32),
        goal_visible_pixels=np.asarray(rows["goal_visible_pixels"], dtype=np.int32),
        goal_u=np.asarray(rows["goal_u"], dtype=np.float32),
        goal_v=np.asarray(rows["goal_v"], dtype=np.float32),
        goal_distance=np.asarray(rows["goal_distance"], dtype=np.float32),
        obstacle_visible_pixels=np.asarray(
            rows["obstacle_visible_pixels"], dtype=np.int32
        ),
        obstacle_u=np.asarray(rows["obstacle_u"], dtype=np.float32),
        obstacle_v=np.asarray(rows["obstacle_v"], dtype=np.float32),
        obstacle_distance=np.asarray(rows["obstacle_distance"], dtype=np.float32),
        belief_fwd=np.asarray(rows["belief_fwd"], dtype=np.float32),
        belief_left=np.asarray(rows["belief_left"], dtype=np.float32),
        goal_frame_fraction=np.asarray(rows["goal_frame_fraction"], dtype=np.float32),
        goal_position=goal.astype(np.float32),
        obstacle_position=(
            ghost_obstacle.astype(np.float32)
            if ghost_obstacle is not None
            else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
        ),
        success=np.asarray(success, dtype=bool),
        hz=np.asarray(float(args.hz), dtype=np.float32),
    )
    manifest = {
        "success": success,
        "frames": len(rows["rgb"]),
        "final_distance": (
            float(rows["goal_distance"][-1]) if rows["goal_distance"] else None
        ),
        "goal_position": goal.tolist(),
        "ghost_obstacle_position": (
            ghost_obstacle.tolist() if ghost_obstacle is not None else None
        ),
        "ckpt": str(Path(args.ckpt).expanduser().resolve()),
        "scene": str(Path(args.scene).expanduser().resolve()),
        "terrain_mode": terrain.mode,
        "scene_height_flip_x": bool(args.scene_height_flip_x),
        "scene_height_flip_z": bool(args.scene_height_flip_z),
        "scene_height_swap_xz": bool(args.scene_height_swap_xz),
        "clearance": float(args.clearance),
        "pose_terrain_radius": float(args.pose_terrain_radius),
        "goal_height": float(args.goal_height),
        "goal_terrain_radius": float(args.goal_terrain_radius),
        "replan_every": replan_every,
        "cbf_active": cbf_active,
        "hard_gate_fired": hard_gate_fired,
        "escape_active": escape_active,
        "cbf_metric": args.cbf_metric,
        "cbf_cov_mode": args.cbf_cov_mode,
        "cbf_radius_mode": args.cbf_radius_mode,
        "npz": str(npz_path),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if args.save_video:
        save_video(
            video_frames,
            out_dir / "rollout.mp4",
            fps=max(float(args.hz) / max(int(args.save_every), 1), 1.0),
        )
    print(f"Saved rollout: {npz_path}", flush=True)
    print(f"Output dir   : {out_dir}", flush=True)


def planar_goal_bearing(position: np.ndarray, yaw: float, goal: np.ndarray) -> float:
    dx = float(goal[0] - position[0])
    dz = float(goal[2] - position[2])
    desired_yaw = math.atan2(-dx, -dz)
    return wrap_angle(desired_yaw - float(yaw))


def integrate_mars(
    position: np.ndarray, yaw: float, action_3d: np.ndarray, dt: float
) -> Tuple[np.ndarray, float]:
    v_fwd, v_lat, yaw_rate = [
        float(x) for x in np.asarray(action_3d, dtype=np.float32).reshape(-1)[:3]
    ]
    fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    out = np.asarray(position, dtype=np.float32).copy()
    out[0] += (fwd_x * v_fwd + left_x * v_lat) * float(dt)
    out[2] += (fwd_z * v_fwd + left_z * v_lat) * float(dt)
    return out, float(yaw + yaw_rate * float(dt))


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def orbit_chunk(
    position, yaw, ghost_obstacle, side, H, dt, r_gate, kr, cruise, pursuit_kp, max_yaw
):
    """Roll the ORBIT controller forced to `side` for H steps -> an [H, 3] action chunk
    ([v_fwd, v_lat, yaw], already in the policy's action units) that goes around the obstacle
    on that side. side=+1 passes on the obstacle's left, -1 on its right. This is the
    counterfactual training TARGET for the language adapter: same observation, the two chunks
    differ only by the requested homotopy class -> the text must carry the difference.
    """
    p = np.asarray(position, np.float32).copy()
    y = float(yaw)
    out = []
    for _ in range(int(H)):
        right, _u, fwd = camera_coords(ghost_obstacle, p, y)
        ox, oy = float(fwd), float(-right)  # [forward, left]
        L = math.hypot(ox, oy)
        phi = math.atan2(oy, ox)
        corr = max(-1.2, min(1.2, float(kr) * (L - float(r_gate))))
        psi = wrap_angle(phi + float(side) * (0.5 * math.pi - corr))
        yaw_cmd = float(
            np.clip(float(pursuit_kp) * psi, -float(max_yaw), float(max_yaw))
        )
        a = np.asarray([float(cruise), 0.0, yaw_cmd], np.float32)
        out.append(a)
        p, y = integrate_mars(p, y, a, dt)
    return np.stack(out, 0)


def command_intent(text):
    """Map a real-time language command to an intent: 'left' / 'right' / 'stop' / '' (default).
    Keyword now; swap for the embedding grounder or a VLM call. Same interface either way.
    """
    t = (text or "").strip().lower()
    if not t:
        return ""
    # qwen_steer_prompt asks for "ACTION: left/right/stop" but the model often appends a
    # REASON line (e.g. "...less than the 1.00% stop threshold...") that contains the
    # substring "stop" even when the actual action is left/right. Parse only the ACTION
    # field when present so that reasoning text can't false-trigger a stop; fall back to
    # the first line for plain one-word replies ("stop" / "left" / "right").
    m = re.search(r"action\s*:\s*([a-z]+)", t)
    field = m.group(1) if m else t.splitlines()[0]
    words = re.findall(r"[a-z]+", field)
    if any(w in ("stop", "halt", "brake", "wait", "hold") for w in words):
        return "stop"
    left, right = "left" in words, "right" in words
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return ""  # navigate normally / unrecognised -> default geometric behaviour


def brake_chunk(v0, H):
    """Decelerate-to-stop chunk: forward ramps v0 -> 0, no turn. Target for 'stop before the
    obstacle' -- a physically feasible braking horizon, distinct from all the moving chunks.
    """
    out = []
    for k in range(int(H)):
        v = float(v0) * max(0.0, 1.0 - k / max(int(H) - 1, 1))
        out.append(np.asarray([v, 0.0, 0.0], np.float32))
    return np.stack(out, 0)


def goal_chunk(position, yaw, goal, H, dt, cruise, kp, max_yaw):
    """Pursuit-to-goal chunk: steer toward the goal bearing at cruise, ignoring the obstacle.
    Target for 'navigate normally / steer to the goal mask' -- the policy's default goal-seeking
    behaviour (also the prior-preservation class: the adapter should ~reproduce the default).
    """
    p = np.asarray(position, np.float32).copy()
    y = float(yaw)
    out = []
    for _ in range(int(H)):
        bearing = planar_goal_bearing(p, y, goal)
        yaw_cmd = float(np.clip(float(kp) * bearing, -float(max_yaw), float(max_yaw)))
        a = np.asarray([float(cruise), 0.0, yaw_cmd], np.float32)
        out.append(a)
        p, y = integrate_mars(p, y, a, dt)
    return np.stack(out, 0)


if __name__ == "__main__":
    main()
