from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np


class DepthObstacleMap:
    """Build a local egocentric occupancy map from a depth image.

    Map convention:
        shape: [grid_size, grid_size]
        rows decrease with forward x
        columns increase with left y
        robot origin is at (row=grid_size-1, col=grid_size/2)
    """

    def __init__(
        self,
        grid_size: int = 96,
        resolution: float = 0.05,
        height_min: float = 0.05,
        height_max: float = 1.5,
        camera_intrinsics: Optional[Mapping[str, float]] = None,
        depth_scale: float = 1.0,
        dilate_radius: int = 1,
    ):
        self.grid_size = int(grid_size)
        self.resolution = float(resolution)
        self.height_min = float(height_min)
        self.height_max = float(height_max)
        self.camera_intrinsics = dict(camera_intrinsics or {})
        self.depth_scale = float(depth_scale)
        self.dilate_radius = int(dilate_radius)

    def build(self, depth: np.ndarray) -> np.ndarray:
        depth_m = _as_depth_hw(depth).astype(np.float32) * self.depth_scale
        h, w = depth_m.shape
        fx = float(self.camera_intrinsics.get("fx", max(h, w)))
        fy = float(self.camera_intrinsics.get("fy", max(h, w)))
        cx = float(self.camera_intrinsics.get("cx", (w - 1) * 0.5))
        cy = float(self.camera_intrinsics.get("cy", (h - 1) * 0.5))

        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not valid.any():
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        ys, xs = np.nonzero(valid)
        z = depth_m[ys, xs]
        x_cam = (xs.astype(np.float32) - cx) * z / fx
        y_cam = (ys.astype(np.float32) - cy) * z / fy

        # Camera optical [right, down, forward] -> robot [forward, left, up].
        x_forward = z
        y_left = -x_cam
        z_up = -y_cam
        height_mask = (z_up >= self.height_min) & (z_up <= self.height_max)
        if height_mask.sum() < 8:
            height_mask = np.ones_like(x_forward, dtype=bool)

        rows, cols = self.world_to_grid(x_forward[height_mask], y_left[height_mask])
        in_bounds = (rows >= 0) & (rows < self.grid_size) & (cols >= 0) & (cols < self.grid_size)
        occ = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        occ[rows[in_bounds], cols[in_bounds]] = 1.0
        if self.dilate_radius > 0:
            occ = _dilate_binary(occ, self.dilate_radius)
        return occ.astype(np.float32)

    def world_to_grid(self, x_forward: np.ndarray, y_left: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_forward, dtype=np.float32)
        y = np.asarray(y_left, dtype=np.float32)
        rows = self.grid_size - 1 - np.floor(x / self.resolution).astype(np.int64)
        cols = np.floor(y / self.resolution + self.grid_size * 0.5).astype(np.int64)
        return rows, cols

    def compute_clearance(self, obstacle_map: np.ndarray) -> np.ndarray:
        return compute_clearance(obstacle_map, resolution=self.resolution)


def compute_clearance(obstacle_map: np.ndarray, resolution: float = 0.05) -> np.ndarray:
    occ = np.asarray(obstacle_map) > 0.5
    if not occ.any():
        max_dist = float(max(occ.shape) * resolution)
        return np.full(occ.shape, max_dist, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        return (distance_transform_edt(~occ) * float(resolution)).astype(np.float32)
    except Exception:
        return _brute_force_clearance(occ, float(resolution))


def _brute_force_clearance(occ: np.ndarray, resolution: float) -> np.ndarray:
    obstacle_rc = np.argwhere(occ)
    out = np.zeros(occ.shape, dtype=np.float32)
    for r in range(occ.shape[0]):
        for c in range(occ.shape[1]):
            d2 = (obstacle_rc[:, 0] - r) ** 2 + (obstacle_rc[:, 1] - c) ** 2
            out[r, c] = float(np.sqrt(d2.min()) * resolution)
    return out


def _dilate_binary(occ: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(occ, radius, mode="constant")
    out = np.zeros_like(occ)
    k = 2 * radius + 1
    for dr in range(k):
        for dc in range(k):
            out = np.maximum(out, padded[dr : dr + occ.shape[0], dc : dc + occ.shape[1]])
    return out


def _as_depth_hw(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("depth must have shape [H,W], [H,W,1], or [1,H,W]")
    return arr

