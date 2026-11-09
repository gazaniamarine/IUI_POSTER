"""Shared helpers for the R2R-CE (val_seen, single free MP3D scene
17DRP5sb8fy) benchmark scripts -- episode loading, R2R-CE's own start_rotation
quaternion -> yaw convention, and the standard VLN-CE metrics (NE, SR, OSR,
SPL) computed exactly as habitat-lab's own NDTW/SPL measures define them
(success = final distance to goal < goal radius; SPL = success * geodesic /
max(geodesic, path_length); OSR = oracle success anywhere along the path).

episodes are pre-filtered (see nav/benchmark/r2rce_17DRP5sb8fy_val_seen.json,
extracted from the real, freely-downloadable R2R_VLNCE_v1-3 val_seen split)
to the one MP3D scene this repo has without the gated full-dataset download --
6 of the split's 778 episodes reference it. Real instructions/start poses/
goals/geodesic distances, not synthesized.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
EPISODES_PATH = HERE / "r2rce_17DRP5sb8fy_val_seen.json"
# mp3d_data/ lives beside mars-habitatsim (sibling of the repo root), not
# inside it -- HERE is nav/benchmark/, so parent.parent.parent is the repo
# root's own parent directory. MarsHabitatEnv.__enter__ always calls
# .expanduser().resolve() on whatever scene_path it's given, so this must be
# the correct absolute path (a config-relative form like
# "17DRP5sb8fy/17DRP5sb8fy.glb" gets resolved against the *process* cwd, not
# the dataset config's directory, and breaks -- confirmed live).
_MP3D_ROOT = (
    HERE.parent.parent.parent
    / "mp3d_data"
    / "versioned_data"
    / "mp3d_example_scene_1.1"
)
SCENE_PATH = str(_MP3D_ROOT / "17DRP5sb8fy" / "17DRP5sb8fy.glb")
SCENE_DATASET_CONFIG = str(_MP3D_ROOT / "mp3d.scene_dataset_config.json")


def load_episodes() -> list:
    with open(EPISODES_PATH) as f:
        return json.load(f)


def quat_to_yaw(rotation: list) -> float:
    """R2R-CE episodes store start_rotation as [x, y, z, w] with x=z=0 (yaw-
    only, no pitch/roll). This project's own set_agent_pose builds its yaw
    quaternion via quaternion.from_rotation_vector([0, yaw, 0]), which is the
    numpy-quaternion library's exp(v/2) form -- for a pure Y-axis rotation
    that gives w=cos(yaw/2), y=sin(yaw/2), the exact same two components the
    dataset already stores in the same [x,y,z,w] order and the same
    right-handed-about-+Y convention. So yaw = 2*atan2(y, w) recovers this
    project's yaw directly, no coordinate-frame conversion needed -- both are
    raw habitat-sim world coordinates for the same scene."""
    _x, y, _z, w = rotation
    return 2.0 * math.atan2(float(y), float(w))


def path_length(points_xz: list) -> float:
    total = 0.0
    for (x0, z0), (x1, z1) in zip(points_xz, points_xz[1:]):
        total += math.hypot(x1 - x0, z1 - z0)
    return total


def compute_metrics(
    path_xz: list,
    goal_xz: tuple,
    goal_radius: float,
    geodesic_distance: float,
) -> dict:
    """Standard VLN-CE episode metrics (Anderson et al. 2018 / Krantz et al.
    2020 conventions):
      NE  = navigation error, final agent-to-goal distance (meters)
      SR  = success rate, 1.0 if NE < goal_radius else 0.0
      OSR = oracle success rate, 1.0 if ANY point on the path came within
            goal_radius of the goal (best-case stopping)
      SPL = success weighted by path length,
            SR * geodesic_distance / max(geodesic_distance, path_length_taken)
            (0 if the episode produced no path at all)
    """
    if not path_xz:
        return {
            "ne": float("inf"),
            "sr": 0.0,
            "osr": 0.0,
            "spl": 0.0,
            "path_length": 0.0,
            "geodesic_distance": geodesic_distance,
        }
    gx, gz = goal_xz
    dists = [math.hypot(x - gx, z - gz) for x, z in path_xz]
    ne = dists[-1]
    sr = 1.0 if ne < goal_radius else 0.0
    osr = 1.0 if min(dists) < goal_radius else 0.0
    taken = path_length(path_xz)
    spl = sr * geodesic_distance / max(geodesic_distance, taken, 1e-6)
    return {
        "ne": ne,
        "sr": sr,
        "osr": osr,
        "spl": spl,
        "path_length": taken,
        "geodesic_distance": geodesic_distance,
    }


def summarize(per_episode: list) -> dict:
    n = len(per_episode)
    if n == 0:
        return {"n": 0}
    keys = ("ne", "sr", "osr", "spl")
    out = {"n": n}
    for k in keys:
        vals = [e[k] for e in per_episode if math.isfinite(e[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out
