# #!/usr/bin/env python3
# """Gaussian body-frame goal belief and ghost PixelGoal projection.

# The live detector supplies a binary goal mask while the target is visible.
# During occlusion, the belief is propagated using the rover's executed SE(2)
# motion.  NavDP receives the projected belief mean as an in-frame PixelGoal;
# the projected covariance is rendered only for diagnostics because released
# NavDP internally constructs its own fixed-size PixelGoal panel.
# """

# from __future__ import annotations

# import math
# from dataclasses import dataclass
# from typing import Optional

# import numpy as np


# @dataclass(frozen=True)
# class GhostPixelGoal:
#     pixel_uv: np.ndarray
#     mask: np.ndarray
#     bearing_rad: float
#     pixel_sigma: float
#     clipped_to_edge: bool


# def body_measurement_from_mask(
#     mask: np.ndarray,
#     depth: np.ndarray,
#     intrinsic: np.ndarray,
#     *,
#     minimum_pixels: int = 10,
#     minimum_depth: float = 0.10,
# ) -> Optional[tuple[np.ndarray, float]]:
#     """Return body-frame ``[forward, left]`` and centroid row from a mask."""

#     mask = np.asarray(mask) > 0
#     depth = np.asarray(depth, dtype=np.float32)
#     intrinsic = np.asarray(intrinsic, dtype=np.float32)
#     if mask.shape != depth.shape:
#         raise ValueError("goal mask and depth must have the same [H,W] shape")
#     if intrinsic.shape != (3, 3):
#         raise ValueError("intrinsic must have shape [3,3]")
#     rows, columns = np.nonzero(mask)
#     if columns.size < int(minimum_pixels):
#         return None
#     mask_depth = depth[rows, columns]
#     valid = mask_depth[np.isfinite(mask_depth) & (mask_depth > minimum_depth)]
#     if valid.size == 0:
#         return None
#     forward = float(np.median(valid))
#     centroid_u = float(columns.mean())
#     centroid_v = float(rows.mean())
#     fx = max(float(intrinsic[0, 0]), 1.0e-6)
#     cx = float(intrinsic[0, 2])
#     left = -(centroid_u - cx) * forward / fx
#     return np.asarray([forward, left], dtype=np.float32), centroid_v


# class GaussianGoalBelief:
#     """Track a stationary goal in rover body coordinates ``[forward,left]``."""

#     def __init__(
#         self,
#         intrinsic: np.ndarray,
#         image_shape: tuple[int, int],
#         *,
#         minimum_visible_pixels: int = 10,
#         measurement_std: float = 0.05,
#         translation_process_std: float = 0.03,
#         yaw_process_std: float = math.radians(1.0),
#         initial_vertical_fraction: float = 0.62,
#     ) -> None:
#         self.intrinsic = np.asarray(intrinsic, dtype=np.float32)
#         if self.intrinsic.shape != (3, 3):
#             raise ValueError("intrinsic must have shape [3,3]")
#         self.height, self.width = [int(value) for value in image_shape]
#         if self.height < 2 or self.width < 2:
#             raise ValueError("image dimensions must be at least two")
#         if minimum_visible_pixels < 1:
#             raise ValueError("minimum_visible_pixels must be positive")
#         if min(measurement_std, translation_process_std, yaw_process_std) < 0.0:
#             raise ValueError("belief noise standard deviations must be non-negative")
#         self.minimum_visible_pixels = int(minimum_visible_pixels)
#         self.measurement_std = float(measurement_std)
#         self.translation_process_std = float(translation_process_std)
#         self.yaw_process_std = float(yaw_process_std)
#         self.default_v = float(initial_vertical_fraction) * (self.height - 1)
#         self.mu: Optional[np.ndarray] = None
#         self.Sigma: Optional[np.ndarray] = None
#         self.visible = False
#         self.time_since_seen = 0.0
#         self.last_seen_v = self.default_v
#         self._last_nonzero_bearing_sign = 1.0

#     @property
#     def initialized(self) -> bool:
#         return self.mu is not None and self.Sigma is not None

#     def initialize(
#         self, body_point: np.ndarray, covariance_std: float, *, visible: bool = False
#     ) -> None:
#         point = np.asarray(body_point, dtype=np.float32).reshape(-1)
#         if point.shape != (2,) or not np.all(np.isfinite(point)):
#             raise ValueError("body_point must be finite [forward,left]")
#         if covariance_std < 0.0:
#             raise ValueError("covariance_std must be non-negative")
#         self.mu = point.copy()
#         self.Sigma = np.eye(2, dtype=np.float32) * float(covariance_std) ** 2
#         self.visible = bool(visible)
#         self.time_since_seen = 0.0
#         self._remember_bearing_sign()

#     def predict(self, executed_action: np.ndarray, dt: float) -> None:
#         """Propagate through executed ``[forward_velocity,left_velocity,yaw_rate]``."""

#         if not self.initialized:
#             return
#         action = np.asarray(executed_action, dtype=np.float32).reshape(-1)
#         if action.shape != (3,) or not np.all(np.isfinite(action)):
#             raise ValueError("executed_action must be finite [v_forward,v_left,yaw_rate]")
#         if dt <= 0.0:
#             raise ValueError("dt must be positive")
#         assert self.mu is not None and self.Sigma is not None
#         translation = action[:2] * float(dt)
#         angle = -float(action[2]) * float(dt)
#         cosine, sine = math.cos(angle), math.sin(angle)
#         rotation = np.asarray(
#             [[cosine, -sine], [sine, cosine]], dtype=np.float32
#         )
#         translated = self.mu - translation
#         self.mu = rotation @ translated

#         translation_variance = self.translation_process_std**2 * float(dt)
#         yaw_variance = self.yaw_process_std**2 * float(dt)
#         rotational_jacobian = np.asarray(
#             [-float(self.mu[1]), float(self.mu[0])], dtype=np.float32
#         )
#         process_noise = (
#             np.eye(2, dtype=np.float32) * translation_variance
#             + yaw_variance * np.outer(rotational_jacobian, rotational_jacobian)
#         )
#         self.Sigma = rotation @ self.Sigma @ rotation.T + process_noise
#         self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
#         self.visible = False
#         self.time_since_seen += float(dt)
#         self._remember_bearing_sign()

#     def observe(self, goal_mask: np.ndarray, depth: np.ndarray) -> bool:
#         measurement = body_measurement_from_mask(
#             goal_mask,
#             depth,
#             self.intrinsic,
#             minimum_pixels=self.minimum_visible_pixels,
#         )
#         if measurement is None:
#             self.visible = False
#             return False
#         body_point, centroid_v = measurement
#         measurement_covariance = (
#             np.eye(2, dtype=np.float32) * self.measurement_std**2
#         )
#         if not self.initialized:
#             self.mu = body_point
#             self.Sigma = measurement_covariance
#         else:
#             assert self.mu is not None and self.Sigma is not None
#             innovation_covariance = self.Sigma + measurement_covariance
#             gain = np.linalg.solve(innovation_covariance.T, self.Sigma.T).T
#             self.mu = self.mu + gain @ (body_point - self.mu)
#             identity = np.eye(2, dtype=np.float32)
#             posterior_factor = identity - gain
#             self.Sigma = (
#                 posterior_factor @ self.Sigma @ posterior_factor.T
#                 + gain @ measurement_covariance @ gain.T
#             )
#             self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
#         self.visible = True
#         self.time_since_seen = 0.0
#         self.last_seen_v = centroid_v
#         self._remember_bearing_sign()
#         return True

#     def project(
#         self,
#         *,
#         margin: int = 11,
#         base_radius: int = 10,
#         covariance_scale: float = 2.0,
#         maximum_radius: int = 80,
#     ) -> GhostPixelGoal:
#         if not self.initialized:
#             raise RuntimeError("goal belief has not been initialized")
#         assert self.mu is not None and self.Sigma is not None
#         forward, left = [float(value) for value in self.mu]
#         bearing = math.atan2(left, forward)
#         if forward <= 0.0 and abs(left) < 1.0e-5:
#             bearing = self._last_nonzero_bearing_sign * math.pi
#         if abs(bearing) > 1.0e-5:
#             self._last_nonzero_bearing_sign = math.copysign(1.0, bearing)

#         fx = float(self.intrinsic[0, 0])
#         cx = float(self.intrinsic[0, 2])
#         usable_half_width = max(min(cx - margin, self.width - 1 - margin - cx), 1.0)
#         maximum_bearing = math.atan2(usable_half_width, max(fx, 1.0e-6))
#         projected_bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
#         clipped = not math.isclose(projected_bearing, bearing, abs_tol=1.0e-7)
#         u = cx - fx * math.tan(projected_bearing)
#         v = float(np.clip(self.last_seen_v, margin, self.height - margin - 1))
#         pixel = np.asarray(
#             [
#                 int(np.clip(round(u), margin, self.width - margin - 1)),
#                 int(np.clip(round(v), margin, self.height - margin - 1)),
#             ],
#             dtype=np.int32,
#         )

#         squared_range = max(forward * forward + left * left, 1.0e-6)
#         bearing_jacobian = np.asarray(
#             [-left / squared_range, forward / squared_range], dtype=np.float64
#         )
#         bearing_variance = float(bearing_jacobian @ self.Sigma @ bearing_jacobian.T)
#         bearing_sigma = math.sqrt(max(bearing_variance, 0.0))
#         pixel_sigma = abs(fx / max(math.cos(projected_bearing) ** 2, 1.0e-6)) * bearing_sigma
#         horizontal_radius = int(
#             np.clip(
#                 round(base_radius + covariance_scale * pixel_sigma),
#                 base_radius,
#                 maximum_radius,
#             )
#         )
#         vertical_radius = max(int(base_radius), 1)
#         mask = ellipse_mask(
#             self.height,
#             self.width,
#             int(pixel[0]),
#             int(pixel[1]),
#             horizontal_radius,
#             vertical_radius,
#         )
#         return GhostPixelGoal(pixel, mask, bearing, pixel_sigma, clipped)

#     def covariance_trace(self) -> float:
#         if self.Sigma is None:
#             return float("inf")
#         return float(np.trace(self.Sigma))

#     def _remember_bearing_sign(self) -> None:
#         if self.mu is None:
#             return
#         bearing = math.atan2(float(self.mu[1]), float(self.mu[0]))
#         if abs(bearing) > 1.0e-5:
#             self._last_nonzero_bearing_sign = math.copysign(1.0, bearing)


# def ellipse_mask(
#     height: int,
#     width: int,
#     centre_u: int,
#     centre_v: int,
#     radius_u: int,
#     radius_v: int,
# ) -> np.ndarray:
#     rows, columns = np.ogrid[:height, :width]
#     normalized = (
#         ((columns - centre_u) / max(radius_u, 1)) ** 2
#         + ((rows - centre_v) / max(radius_v, 1)) ** 2
#     )
#     return (normalized <= 1.0).astype(np.uint8)
#!/usr/bin/env python3
"""Gaussian body-frame goal belief and ghost PixelGoal projection.

The live detector supplies a binary goal mask while the target is visible.
During occlusion, the belief is propagated using the rover's executed SE(2)
motion.  NavDP receives the projected belief mean as an in-frame PixelGoal;
the projected covariance is rendered only for diagnostics because released
NavDP internally constructs its own fixed-size PixelGoal panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GhostPixelGoal:
    pixel_uv: np.ndarray
    mask: np.ndarray
    bearing_rad: float
    pixel_sigma: float
    clipped_to_edge: bool


def body_measurement_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    minimum_pixels: int = 10,
    minimum_depth: float = 0.10,
) -> Optional[tuple[np.ndarray, float]]:
    """Return body-frame ``[forward, left]`` and centroid row from a mask."""

    mask = np.asarray(mask) > 0
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if mask.shape != depth.shape:
        raise ValueError("goal mask and depth must have the same [H,W] shape")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape [3,3]")
    rows, columns = np.nonzero(mask)
    if columns.size < int(minimum_pixels):
        return None
    mask_depth = depth[rows, columns]
    valid = mask_depth[np.isfinite(mask_depth) & (mask_depth > minimum_depth)]
    if valid.size == 0:
        return None
    forward = float(np.median(valid))
    centroid_u = float(columns.mean())
    centroid_v = float(rows.mean())
    fx = max(float(intrinsic[0, 0]), 1.0e-6)
    cx = float(intrinsic[0, 2])
    left = -(centroid_u - cx) * forward / fx
    return np.asarray([forward, left], dtype=np.float32), centroid_v


class GaussianGoalBelief:
    """Track a stationary goal in rover body coordinates ``[forward,left]``."""

    def __init__(
        self,
        intrinsic: np.ndarray,
        image_shape: tuple[int, int],
        *,
        minimum_visible_pixels: int = 10,
        measurement_std: float = 0.05,
        translation_process_std: float = 0.03,
        yaw_process_std: float = math.radians(1.0),
        initial_vertical_fraction: float = 0.62,
    ) -> None:
        self.intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if self.intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape [3,3]")
        self.height, self.width = [int(value) for value in image_shape]
        if self.height < 2 or self.width < 2:
            raise ValueError("image dimensions must be at least two")
        if minimum_visible_pixels < 1:
            raise ValueError("minimum_visible_pixels must be positive")
        if min(measurement_std, translation_process_std, yaw_process_std) < 0.0:
            raise ValueError("belief noise standard deviations must be non-negative")
        self.minimum_visible_pixels = int(minimum_visible_pixels)
        self.measurement_std = float(measurement_std)
        self.translation_process_std = float(translation_process_std)
        self.yaw_process_std = float(yaw_process_std)
        self.default_v = float(initial_vertical_fraction) * (self.height - 1)
        self.mu: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self.visible = False
        self.time_since_seen = 0.0
        self.last_seen_v = self.default_v
        self._last_nonzero_bearing_sign = 1.0

    @property
    def initialized(self) -> bool:
        return self.mu is not None and self.Sigma is not None

    def initialize(
        self, body_point: np.ndarray, covariance_std: float, *, visible: bool = False
    ) -> None:
        point = np.asarray(body_point, dtype=np.float32).reshape(-1)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("body_point must be finite [forward,left]")
        if covariance_std < 0.0:
            raise ValueError("covariance_std must be non-negative")
        self.mu = point.copy()
        self.Sigma = np.eye(2, dtype=np.float32) * float(covariance_std) ** 2
        self.visible = bool(visible)
        self.time_since_seen = 0.0
        self._remember_bearing_sign()

    def predict(self, executed_action: np.ndarray, dt: float) -> None:
        """Propagate through executed ``[forward_velocity,left_velocity,yaw_rate]``."""

        if not self.initialized:
            return
        action = np.asarray(executed_action, dtype=np.float32).reshape(-1)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError(
                "executed_action must be finite [v_forward,v_left,yaw_rate]"
            )
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        assert self.mu is not None and self.Sigma is not None
        translation = action[:2] * float(dt)
        angle = -float(action[2]) * float(dt)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        translated = self.mu - translation
        self.mu = rotation @ translated

        translation_variance = self.translation_process_std**2 * float(dt)
        yaw_variance = self.yaw_process_std**2 * float(dt)
        rotational_jacobian = np.asarray(
            [-float(self.mu[1]), float(self.mu[0])], dtype=np.float32
        )
        process_noise = np.eye(
            2, dtype=np.float32
        ) * translation_variance + yaw_variance * np.outer(
            rotational_jacobian, rotational_jacobian
        )
        self.Sigma = rotation @ self.Sigma @ rotation.T + process_noise
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        self.visible = False
        self.time_since_seen += float(dt)
        self._remember_bearing_sign()

    def observe(self, goal_mask: np.ndarray, depth: np.ndarray) -> bool:
        measurement = body_measurement_from_mask(
            goal_mask,
            depth,
            self.intrinsic,
            minimum_pixels=self.minimum_visible_pixels,
        )
        if measurement is None:
            self.visible = False
            return False
        body_point, centroid_v = measurement
        measurement_covariance = np.eye(2, dtype=np.float32) * self.measurement_std**2
        if not self.initialized:
            self.mu = body_point
            self.Sigma = measurement_covariance
        else:
            assert self.mu is not None and self.Sigma is not None
            innovation_covariance = self.Sigma + measurement_covariance
            gain = np.linalg.solve(innovation_covariance.T, self.Sigma.T).T
            self.mu = self.mu + gain @ (body_point - self.mu)
            identity = np.eye(2, dtype=np.float32)
            posterior_factor = identity - gain
            self.Sigma = (
                posterior_factor @ self.Sigma @ posterior_factor.T
                + gain @ measurement_covariance @ gain.T
            )
            self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        self.visible = True
        self.time_since_seen = 0.0
        self.last_seen_v = centroid_v
        self._remember_bearing_sign()
        return True

    def project(
        self,
        *,
        margin: int = 11,
        base_radius: int = 10,
        covariance_scale: float = 2.0,
        maximum_radius: int = 80,
    ) -> GhostPixelGoal:
        if not self.initialized:
            raise RuntimeError("goal belief has not been initialized")
        assert self.mu is not None and self.Sigma is not None
        forward, left = [float(value) for value in self.mu]
        bearing = math.atan2(left, forward)
        if forward <= 0.0 and abs(left) < 1.0e-5:
            bearing = self._last_nonzero_bearing_sign * math.pi
        if abs(bearing) > 1.0e-5:
            self._last_nonzero_bearing_sign = math.copysign(1.0, bearing)

        fx = float(self.intrinsic[0, 0])
        cx = float(self.intrinsic[0, 2])
        usable_half_width = max(min(cx - margin, self.width - 1 - margin - cx), 1.0)
        maximum_bearing = math.atan2(usable_half_width, max(fx, 1.0e-6))
        projected_bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
        clipped = not math.isclose(projected_bearing, bearing, abs_tol=1.0e-7)
        u = cx - fx * math.tan(projected_bearing)
        v = float(np.clip(self.last_seen_v, margin, self.height - margin - 1))
        pixel = np.asarray(
            [
                int(np.clip(round(u), margin, self.width - margin - 1)),
                int(np.clip(round(v), margin, self.height - margin - 1)),
            ],
            dtype=np.int32,
        )

        squared_range = max(forward * forward + left * left, 1.0e-6)
        bearing_jacobian = np.asarray(
            [-left / squared_range, forward / squared_range], dtype=np.float64
        )
        bearing_variance = float(bearing_jacobian @ self.Sigma @ bearing_jacobian.T)
        bearing_sigma = math.sqrt(max(bearing_variance, 0.0))
        pixel_sigma = (
            abs(fx / max(math.cos(projected_bearing) ** 2, 1.0e-6)) * bearing_sigma
        )
        horizontal_radius = int(
            np.clip(
                round(base_radius + covariance_scale * pixel_sigma),
                base_radius,
                maximum_radius,
            )
        )
        vertical_radius = max(int(base_radius), 1)
        mask = ellipse_mask(
            self.height,
            self.width,
            int(pixel[0]),
            int(pixel[1]),
            horizontal_radius,
            vertical_radius,
        )
        return GhostPixelGoal(pixel, mask, bearing, pixel_sigma, clipped)

    def covariance_trace(self) -> float:
        if self.Sigma is None:
            return float("inf")
        return float(np.trace(self.Sigma))

    def _remember_bearing_sign(self) -> None:
        if self.mu is None:
            return
        bearing = math.atan2(float(self.mu[1]), float(self.mu[0]))
        if abs(bearing) > 1.0e-5:
            self._last_nonzero_bearing_sign = math.copysign(1.0, bearing)


def ellipse_mask(
    height: int,
    width: int,
    centre_u: int,
    centre_v: int,
    radius_u: int,
    radius_v: int,
) -> np.ndarray:
    rows, columns = np.ogrid[:height, :width]
    normalized = ((columns - centre_u) / max(radius_u, 1)) ** 2 + (
        (rows - centre_v) / max(radius_v, 1)
    ) ** 2
    return (normalized <= 1.0).astype(np.uint8)
