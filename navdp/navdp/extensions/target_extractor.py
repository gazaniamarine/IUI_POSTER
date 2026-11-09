from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np


class SAMDepthTargetExtractor:
    """Estimate local target positions from goal-id SAM masks and depth.

    Output positions are in robot coordinates. By default, camera optical
    coordinates [right, down, forward] are mapped to robot [forward, left, up].
    """

    def __init__(
        self,
        camera_intrinsics: Mapping[str, float],
        min_mask_area: int = 50,
        depth_scale: float = 1.0,
        max_depth: Optional[float] = None,
        position_dim: int = 2,
        camera_to_robot: Optional[np.ndarray] = None,
    ):
        self.fx = float(camera_intrinsics["fx"])
        self.fy = float(camera_intrinsics["fy"])
        self.cx = float(camera_intrinsics["cx"])
        self.cy = float(camera_intrinsics["cy"])
        self.min_mask_area = int(min_mask_area)
        self.depth_scale = float(depth_scale)
        self.max_depth = None if max_depth is None else float(max_depth)
        if position_dim not in (2, 3):
            raise ValueError("position_dim must be 2 or 3")
        self.position_dim = position_dim
        self.camera_to_robot = None if camera_to_robot is None else np.asarray(camera_to_robot, dtype=np.float32)
        if self.camera_to_robot is not None and self.camera_to_robot.shape != (4, 4):
            raise ValueError("camera_to_robot must have shape [4,4]")

    def extract(
        self,
        sam_masks: Mapping[str, np.ndarray],
        depth: np.ndarray,
    ) -> Dict[str, Dict[str, object]]:
        depth_m = _as_depth_hw(depth).astype(np.float32) * self.depth_scale
        out: Dict[str, Dict[str, object]] = {}
        h, w = depth_m.shape
        for goal_id, raw_mask in sam_masks.items():
            mask = np.asarray(raw_mask).astype(bool)
            if mask.shape != depth_m.shape:
                out[goal_id] = _not_visible()
                continue

            area = int(mask.sum())
            if area < self.min_mask_area:
                out[goal_id] = _not_visible()
                continue

            valid = mask & np.isfinite(depth_m) & (depth_m > 0)
            if self.max_depth is not None:
                valid &= depth_m <= self.max_depth
            if int(valid.sum()) == 0:
                out[goal_id] = _not_visible()
                continue

            ys, xs = np.nonzero(valid)
            z = float(np.median(depth_m[valid]))
            u = float(np.median(xs))
            v = float(np.median(ys))
            xyz_cam = self._backproject(u, v, z)
            xyz_robot = self._to_robot_frame(xyz_cam)
            if not np.isfinite(xyz_robot).all():
                out[goal_id] = _not_visible()
                continue

            valid_fraction = float(valid.sum() / max(area, 1))
            area_fraction = float(area / max(h * w, 1))
            confidence = float(np.clip(valid_fraction * min(1.0, area_fraction * 20.0), 0.0, 1.0))
            position = xyz_robot[: self.position_dim].astype(np.float32)
            out[goal_id] = {
                "visible": True,
                "position": position,
                "confidence": confidence,
            }
        return out

    def _backproject(self, u: float, v: float, z: float) -> np.ndarray:
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.asarray([x, y, z], dtype=np.float32)

    def _to_robot_frame(self, xyz_cam: np.ndarray) -> np.ndarray:
        if self.camera_to_robot is not None:
            p = np.ones(4, dtype=np.float32)
            p[:3] = xyz_cam
            return (self.camera_to_robot @ p)[:3].astype(np.float32)
        x_cam, y_cam, z_cam = xyz_cam
        return np.asarray([z_cam, -x_cam, -y_cam], dtype=np.float32)


def _as_depth_hw(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("depth must have shape [H,W], [H,W,1], or [1,H,W]")
    return arr


def _not_visible() -> Dict[str, object]:
    return {"visible": False, "position": None, "confidence": 0.0}

