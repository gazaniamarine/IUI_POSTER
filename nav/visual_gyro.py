"""Visual gyro: the rover's yaw change measured from consecutive camera frames.

This rover publishes no usable heading (the IMU field on /rover/rpm is NaN and
there is no camera IMU topic), so the GUI's heading is otherwise only
dead-reckoned from the commanded turn rate -- wrong whenever the real rate
differs (ramp-up, deadband, slip). During a turn IN PLACE the scene moves by
a pure camera rotation, so matched ORB features between two frames are
related by a homography H = K R K^-1; R = K^-1 H K, and the rotation about the
camera's vertical axis is the yaw. Measured, not assumed.

Only trustworthy while the rover is not translating (forward motion makes
near features violate the pure-rotation model) -- RealRoverEnv uses it only
when the commanded forward speed is ~0. Returns None when there is too little
texture / too few inliers, and the caller falls back to dead reckoning.

Sign: + = the rover turned LEFT (this project's +yaw), i.e. the scene moved
right in the image.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

WORK_W = 320  # frames are downscaled to this width before feature matching
MIN_INLIERS = 25


class VisualGyro:
    def __init__(self, hfov_deg: float, n_features: int = 600):
        self.hfov_deg = float(hfov_deg)
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev = None  # (keypoints, descriptors, K)
        self.last_inliers = 0

    def reset(self) -> None:
        self._prev = None

    def _prep(self, rgb):
        h, w = rgb.shape[:2]
        s = WORK_W / float(w)
        gray = cv2.cvtColor(cv2.resize(rgb, (WORK_W, int(round(h * s)))), cv2.COLOR_RGB2GRAY)
        gh, gw = gray.shape
        f = gw / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))
        K = np.array([[f, 0, gw / 2.0], [0, f, gh / 2.0], [0, 0, 1.0]])
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des, K

    def update(self, rgb) -> Optional[float]:
        """Feed the next frame; returns the yaw change (rad, +left) since the
        previous frame, or None if it can't be measured."""
        kp, des, K = self._prep(rgb)
        prev, self._prev = self._prev, (kp, des, K)
        self.last_inliers = 0
        if prev is None or des is None or prev[1] is None or len(kp) < MIN_INLIERS:
            return None
        matches = self.matcher.match(prev[1], des)
        if len(matches) < MIN_INLIERS:
            return None
        p1 = np.float32([prev[0][m.queryIdx].pt for m in matches])
        p2 = np.float32([kp[m.trainIdx].pt for m in matches])
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
        if H is None or inl is None or int(inl.sum()) < MIN_INLIERS:
            return None
        self.last_inliers = int(inl.sum())
        R = np.linalg.inv(K) @ H @ K
        R /= np.cbrt(np.linalg.det(R)) if np.linalg.det(R) > 0 else 1.0
        # camera x right, y down, z forward; scene moving right <=> camera yawed left
        yaw = math.atan2(R[0, 2], R[2, 2])
        if abs(yaw) > math.radians(20.0):  # implausible for one frame at <=0.6 rad/s
            return None
        return yaw
