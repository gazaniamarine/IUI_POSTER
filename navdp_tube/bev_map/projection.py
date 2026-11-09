"""Depth -> 3D points, depth filtering, and floor-based correction of Depth Anything depth.

Camera convention (MuJoCo): camera x = right, y = up, looks along -z.
cam_rot columns are those axes expressed in world; world z is up and the floor is z = 0.
"""

from collections import deque

import cv2
import numpy as np


def pixel_rays(K, height, width, cam_rot, stride=1):
    """World-frame ray per pixel, scaled so that z-depth 1 along the optical axis = 1 unit of t."""
    v, u = np.mgrid[0:height:stride, 0:width:stride].astype(np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0]
    y = -(v + 0.5 - K[1, 2]) / K[1, 1]
    dirs_cam = np.stack([x, y, -np.ones_like(x)], axis=-1)       # (h, w, 3)
    return dirs_cam @ cam_rot.T, v.astype(int), u.astype(int)


def floor_depth(K, height, width, cam_pos, cam_rot):
    """Expected z-depth of the floor plane (z=0) at each pixel; inf above the horizon."""
    rays, _, _ = pixel_rays(K, height, width, cam_rot)
    rz = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(rz < -1e-6, cam_pos[2] / -rz, np.inf)
    return d.astype(np.float32)


def segmentation_depth(labels, conf, floor_d, obstacle_ids, free_ids, min_conf=0.5):
    """Depth from a terrain segmentation and the known ground: no depth model.

    free pixels (free_ids)     -> the ground's depth there (floor_d: ray hits the ground plane / terrain)
    obstacle pixels            -> standing upright on the ground: every pixel of an unbroken run of obstacle
                                  pixels in an image column gets the ground depth at the run's lowest pixel
                                  (where the obstacle meets the ground), so a rock becomes a vertical surface
                                  with a height. Backprojected, these are points like a depth camera gives.
    everything else / conf < min_conf / no ground hit (sky, above the horizon) -> NaN (not used).
    Returns depth (HxW float32, NaN = no depth), obstacle mask, free mask.
    """
    sure = conf >= min_conf
    obst = np.isin(labels, list(obstacle_ids)) & sure
    free = np.isin(labels, list(free_ids)) & sure
    h, w = labels.shape
    bottom = np.empty((h, w), np.int64)          # lowest row of the obstacle run each pixel is in
    cur = np.full(w, -1, np.int64)
    rows = np.arange(h)
    for v in rows[::-1]:
        cur = np.where(obst[v], np.where(cur >= 0, cur, v), -1)
        bottom[v] = cur
    depth = np.full((h, w), np.nan, np.float32)
    depth[free] = floor_d[free]
    vv, uu = np.nonzero(obst)
    depth[vv, uu] = floor_d[bottom[vv, uu], uu]
    depth[~np.isfinite(depth)] = np.nan
    return depth, obst, free


def seg_cell_evidence(cell_x, cell_y, cell_z, labels, conf, floor_d, rock_d, K, cam_pos, cam_rot, max_range,
                      free_ids, flat_occ_ids, rock_ids, min_conf=0.0, mask=None, rock_depth=0.5, vis_tol=0.05,
                      hidden_free=False):
    """Map cells -> camera: each cell's ground point is projected into the image and takes the class of that pixel.

    Exactly what the segmentation shows, laid on the ground: no gaps between pixel rows far out, no smear.
      visible : the cell's depth matches where that pixel's ray meets the ground (floor_d), i.e. not behind a hill
      free_ids      -> free;  flat_occ_ids -> occupied (ground surfaces: sand, bedrock, ...)
      rock_ids      -> occupied from where the rock meets the ground (rock_d, projection.segmentation_depth) to
                       rock_depth behind it; ground hidden farther behind the rock -> no evidence (unknown)
    hidden_free: ground hidden by the terrain itself (behind a rise / in a dip) -> free instead of unknown
                 (never behind a rock)
    Returns occupied, free (bool, grid shape).
    """
    h, w = labels.shape
    d = np.stack([cell_x - cam_pos[0], cell_y - cam_pos[1], cell_z - cam_pos[2]], -1)
    xc, yc, zc = d @ cam_rot[:, 0], d @ cam_rot[:, 1], -(d @ cam_rot[:, 2])      # right, up, forward
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(K[0, 0] * xc / zc + K[0, 2] - 0.5)
        v = np.round(-K[1, 1] * yc / zc + K[1, 2] - 0.5)
    ok = (zc > 0.05) & (np.hypot(d[..., 0], d[..., 1]) < max_range) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    occ = np.zeros(cell_x.shape, bool)
    free = np.zeros(cell_x.shape, bool)
    iy, ix = np.nonzero(ok)
    uu, vv, z = u[iy, ix].astype(int), v[iy, ix].astype(int), zc[iy, ix]
    lab = labels[vv, uu]
    sure = conf[vv, uu] >= min_conf
    if mask is not None:
        sure &= mask[vv, uu]
    fd = floor_d[vv, uu]
    seen = sure & np.isfinite(fd) & (np.abs(z - fd) <= vis_tol * fd + 0.05)        # the pixel sees this cell
    is_rock = np.isin(lab, list(rock_ids)) & sure
    rd = rock_d[vv, uu]
    rock_here = is_rock & np.isfinite(rd) & (z >= rd - 0.05) & (z <= rd + rock_depth)
    occ[iy, ix] = (seen & np.isin(lab, list(flat_occ_ids))) | rock_here
    fr = seen & np.isin(lab, list(free_ids)) & ~is_rock
    if hidden_free:
        scene = np.ones(len(z), bool) if mask is None else mask[vv, uu]
        any_rock = np.isin(lab, list(rock_ids))
        fr |= scene & np.isfinite(fd) & ~seen & ~any_rock & ~occ[iy, ix]
    free[iy, ix] = fr
    return occ, free


def edge_mask(depth, rel_grad, dilate):
    """True where depth jumps sharply (object boundaries -> 'flying pixels')."""
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    m = cv2.dilate(d, k) - cv2.erode(d, k)
    bad = m > rel_grad * np.maximum(d, 1e-3)
    if dilate > 0:
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((2 * dilate + 1,) * 2, np.uint8)) > 0
    return bad


def backproject(depth, valid, K, cam_pos, cam_rot, stride):
    """World points (N,3) and their pixel coords for valid pixels on a strided grid."""
    h, w = depth.shape
    rays, v, u = pixel_rays(K, h, w, cam_rot, stride)
    d = depth[v, u]
    keep = valid[v, u]
    pts = cam_pos + rays[keep] * d[keep, None]
    return pts, v[keep], u[keep]


class FloorCorrector:
    """Fits depth_true ~ a * depth_pred + b on pixels that look like floor.

    The floor's true depth at every pixel is known from the camera height and pose, so floor
    pixels act as a free, per-frame calibration target. Obstacles sit *in front of* the floor
    and are rejected as RANSAC outliers.
    """

    def __init__(self, mode="affine", ema=0.7, max_range=3.5, sample=4000, iters=200,
                 rel_tol=0.04, seed=0):
        self.mode, self.ema, self.max_range = mode, ema, max_range
        self.sample, self.iters, self.rel_tol = sample, iters, rel_tol
        self.rng = np.random.default_rng(seed)
        self.a, self.b = None, 0.0
        self.last_inlier_frac = 0.0

    def reset(self):
        self.a, self.b = None, 0.0

    def __call__(self, pred, d_floor):
        if self.mode == "none":
            return pred
        a, b = self._fit(pred, d_floor)
        if a is not None:
            if self.a is None or self.ema <= 0:
                self.a, self.b = a, b
            else:
                self.a = self.ema * self.a + (1 - self.ema) * a
                self.b = self.ema * self.b + (1 - self.ema) * b
        if self.a is None:
            return pred
        return self.a * pred + self.b

    def _fit(self, pred, d_floor):
        cand = np.isfinite(d_floor) & (d_floor < self.max_range) & (pred > 0) & np.isfinite(pred)
        idx = np.flatnonzero(cand)
        if idx.size < 200:
            return None, None
        sub = self.rng.choice(idx, min(self.sample, idx.size), replace=False)
        x, y = pred.ravel()[sub].astype(np.float64), d_floor.ravel()[sub].astype(np.float64)

        best, best_n = None, 0
        for _ in range(self.iters):
            if self.mode == "scale":
                i = self.rng.integers(len(x))
                a, b = y[i] / x[i], 0.0
            else:
                i, j = self.rng.choice(len(x), 2, replace=False)
                if abs(x[i] - x[j]) < 0.05 * max(x[i], x[j]):
                    continue
                a = (y[i] - y[j]) / (x[i] - x[j]); b = y[i] - a * x[i]
            if not (0.02 < a < 50):
                continue
            n = np.count_nonzero(np.abs(a * x + b - y) < self.rel_tol * y)
            if n > best_n:
                best, best_n = (a, b), n
        if best is None or best_n < 0.25 * len(x):
            return None, None

        a, b = best
        inl = np.abs(a * x + b - y) < self.rel_tol * y
        # least-squares refinement on inliers, weighted by 1/y so near and far count equally
        w = 1.0 / y[inl]
        if self.mode == "scale":
            a, b = float(np.sum(w * x[inl] * y[inl]) / np.sum(w * x[inl] ** 2)), 0.0
        else:
            A = np.stack([x[inl], np.ones(inl.sum())], 1) * w[:, None]
            a, b = np.linalg.lstsq(A, y[inl] * w, rcond=None)[0]
        self.last_inlier_frac = best_n / len(x)
        return float(a), float(b)


def first_hit_filter(pts, obst, cam_xy, bin_deg=0.5, quantile=0.05, min_rank=2,
                     margin=0.08, margin_per_m=0.04):
    """Drop obstacle points lying well behind the nearest obstacle in their viewing direction.

    Monocular depth smears a surface into a band behind it. From one viewpoint only the
    first surface along a bearing can be seen, so anything farther than (robust nearest
    obstacle range + margin) in the same bearing bin is smear. Returns a new obstacle mask.
    """
    idx = np.flatnonzero(obst)
    if idx.size == 0:
        return obst
    d = pts[idx, :2] - cam_xy
    r = np.hypot(d[:, 0], d[:, 1])
    b = np.floor(np.degrees(np.arctan2(d[:, 1], d[:, 0])) / bin_deg).astype(np.int64)
    order = np.lexsort((r, b))
    b_s, r_s = b[order], r[order]
    starts = np.flatnonzero(np.r_[True, b_s[1:] != b_s[:-1]])
    counts = np.diff(np.r_[starts, len(b_s)])
    k = np.minimum(np.maximum(min_rank, (quantile * counts).astype(int)), counts - 1)
    near = np.repeat(r_s[starts + k], counts)
    keep_sorted = r_s <= near + margin + margin_per_m * near
    keep = np.empty_like(keep_sorted)
    keep[order] = keep_sorted
    out = np.zeros_like(obst)
    out[idx[keep]] = True
    return out


def _rot_between(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


class FloorPoseRefiner:
    """Measures the camera's real pitch/roll each frame from the floor visible in the depth image.

    The assumed camera rotation (from the mount / odometry) is only a prior. Floor pixels are fitted
    with a RANSAC plane in the camera frame; the plane normal is the true 'up' direction as the camera
    sees it. The assumed rotation is then tilted so that 'up' matches, keeping the heading (yaw).
    Works on unscaled Depth Anything depth: a plane's normal does not depend on depth scale.
    """

    def __init__(self, max_tilt_deg=8.0, rel_tol=0.04, min_inlier_frac=0.3, stride=8,
                 iters=150, max_range=3.5, seed=0):
        self.max_tilt = np.radians(max_tilt_deg)
        self.rel_tol, self.min_frac = rel_tol, min_inlier_frac
        self.stride, self.iters, self.max_range = stride, iters, max_range
        self.rng = np.random.default_rng(seed)
        self.last = {"ok": False, "tilt_deg": 0.0, "inlier_frac": 0.0}

    def __call__(self, depth, K, cam_rot):
        """Return (corrected cam_rot, info). Falls back to cam_rot if the floor is not found."""
        h, w = depth.shape
        rays_w, v, u = pixel_rays(K, h, w, cam_rot, self.stride)
        below = rays_w[..., 2] < -np.sin(np.radians(2.0))           # prior: below the horizon
        d = depth[v, u]
        ok = below & np.isfinite(d) & (d > 0)
        if ok.sum() < 100:
            return cam_rot, self._fail()
        x = (u[ok] + 0.5 - K[0, 2]) / K[0, 0]
        y = -(v[ok] + 0.5 - K[1, 2]) / K[1, 1]
        P = np.stack([x, y, -np.ones_like(x)], 1) * d[ok, None]         # camera-frame points
        P = P[np.linalg.norm(P, axis=1) < np.percentile(np.linalg.norm(P, axis=1), 90)]
        up0 = cam_rot.T @ np.array([0.0, 0.0, 1.0])                     # prior 'up' in camera frame

        best, best_n = None, 0
        for _ in range(self.iters):
            a, b, c = P[self.rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            if n @ up0 < 0:
                n = -n
            if np.arccos(np.clip(n @ up0, -1, 1)) > self.max_tilt:
                continue
            hgt = -(a @ n)                                              # camera height, depth units
            if hgt <= 0:
                continue
            cnt = np.count_nonzero(np.abs(P @ n + hgt) < self.rel_tol * hgt)
            if cnt > best_n:
                best, best_n = (n, hgt), cnt
        if best is None or best_n < self.min_frac * len(P):
            return cam_rot, self._fail()

        n, hgt = best
        inl = P[np.abs(P @ n + hgt) < self.rel_tol * hgt]
        cen = inl.mean(0)
        n = np.linalg.svd(inl - cen)[2][2]
        n = n if n @ up0 > 0 else -n
        up_world_meas = cam_rot @ n                                     # measured up, in prior world
        corrected = _rot_between(up_world_meas, np.array([0.0, 0.0, 1.0])) @ cam_rot
        self.last = {"ok": True, "tilt_deg": float(np.degrees(np.arccos(np.clip(n @ up0, -1, 1)))),
                     "inlier_frac": best_n / len(P), "height_units": float(-(cen @ n))}
        return corrected, self.last

    def _fail(self):
        self.last = {"ok": False, "tilt_deg": 0.0, "inlier_frac": 0.0}
        return self.last


def tilt_rotation(cam_rot, pitch_deg, roll_deg):
    """cam_rot tilted by pitch (about the camera x axis, +ve = look further down) and roll
    (about the optical axis). Heading (yaw) is unchanged for small angles."""
    def rot(axis, a):
        axis = axis / np.linalg.norm(axis)
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    R = rot(cam_rot[:, 0], -np.radians(pitch_deg)) @ cam_rot
    return rot(R[:, 2], np.radians(roll_deg)) @ R


class JointFloorFit:
    """Per-frame joint fit of depth correction (a, b) AND camera tilt (roll; pitch optional).

    Floor pixels must satisfy  a * depth_pred + b == floor depth predicted by the camera pose
    at the known camera height. Solving for all four together handles both Depth Anything's
    scale/offset error and the camera tilting on the real rover (bumps, braking, mount error),
    which a fixed-pose floor model cannot. Robust least squares on RANSAC floor inliers.
    """

    def __init__(self, max_tilt_deg=8.0, max_range=3.5, stride=6, ema=0.7, min_inlier_frac=0.3,
                 fit_pitch=False, gate_rel=0.0, gate_window=7, seed=0):
        self.max_tilt, self.max_range, self.stride = max_tilt_deg, max_range, stride
        # Pitch is NOT observable from monocular depth: Depth Anything's own depth error looks
        # like a pitch change (measured in sim: 1.4 deg median, up to 10 deg error). Roll is
        # observable (0.16 deg). So by default pitch comes from outside (mount calibration / IMU).
        self.fit_pitch = fit_pitch
        self.ema, self.min_frac = ema, min_inlier_frac
        # Scale-jump gate: a fit whose scale differs from the median of the recent fits by more
        # than gate_rel is treated as failed (bad floor detection, glare, a wall filling the view).
        # Every fit still enters the history, so a lasting change of scene is accepted after a
        # few frames. 0 = off.
        self.gate_rel, self.hist = gate_rel, deque(maxlen=max(3, gate_window))
        self.init = FloorCorrector("affine", ema=0, max_range=max_range, rel_tol=0.08, seed=seed)
        self.a, self.b = None, 0.0
        self.last = {"ok": False, "pitch_deg": 0.0, "roll_deg": 0.0, "inlier_frac": 0.0}
        self._dirs = None

    def reset(self):
        self.a, self.b = None, 0.0
        self.hist.clear()

    def _cam_dirs(self, K, h, w):
        key = (h, w, *K.ravel())
        if self._dirs is None or self._dirs[0] != key:
            v, u = np.mgrid[0:h:self.stride, 0:w:self.stride].astype(np.float64)
            d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], -(v + 0.5 - K[1, 2]) / K[1, 1],
                          -np.ones_like(u)], -1).reshape(-1, 3)
            self._dirs = (key, d, v.astype(int).ravel(), u.astype(int).ravel())
        return self._dirs[1:]

    def __call__(self, pred, K, cam_pos, cam_rot):
        """Return (corrected depth, corrected cam_rot)."""
        from scipy.optimize import least_squares
        h, w = pred.shape
        dirs, v, u = self._cam_dirs(K, h, w)
        x = pred[v, u].astype(np.float64)
        height = float(cam_pos[2])

        def floor_d(p, r):
            rz = dirs @ tilt_rotation(cam_rot, p, r)[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(rz < -1e-3, height / -rz, np.inf)

        d0 = floor_d(0.0, 0.0)
        cand = np.isfinite(x) & (x > 0) & (d0 < self.max_range)
        if cand.sum() < 200:
            return self._fallback(pred, cam_rot)
        # initial floor pixels + (a, b) with the assumed tilt and a loose tolerance
        a0, b0 = self.init._fit(x[cand][None], d0[cand][None])
        if a0 is None:
            return self._fallback(pred, cam_rot)
        inl = cand & (np.abs(a0 * x + b0 - d0) < 0.10 * d0)

        params = np.array([0.0, 0.0, a0, b0])
        for tol in (0.06, 0.03):
            idx = np.flatnonzero(inl)
            if idx.size < 100:
                return self._fallback(pred, cam_rot)

            def resid(q):
                df = floor_d(q[0], q[1])[idx]
                df = np.where(np.isfinite(df), df, 1e3)
                return (q[2] * x[idx] + q[3] - df) / np.minimum(df, self.max_range)

            pmax = self.max_tilt if self.fit_pitch else 1e-6
            sol = least_squares(resid, params, loss="soft_l1", f_scale=0.02, x_scale=[1, 1, 0.1, 0.1],
                                bounds=([-pmax, -self.max_tilt, 0.02, -2.0],
                                        [pmax, self.max_tilt, 50.0, 2.0]))
            params = sol.x
            df = floor_d(params[0], params[1])
            inl = cand & np.isfinite(df) & (np.abs(params[2] * x + params[3] - df) < tol * df)

        frac = inl.sum() / cand.sum()
        if frac < self.min_frac:
            return self._fallback(pred, cam_rot, "few_floor_px")
        p, r, a, b = params
        if self.gate_rel > 0:
            prev = list(self.hist)
            self.hist.append(a)
            if len(prev) >= 3:
                med = float(np.median(prev))
                if abs(a - med) > self.gate_rel * med:
                    return self._fallback(pred, cam_rot, "scale_jump")
        if self.a is None or self.ema <= 0:
            self.a, self.b = a, b
        else:
            self.a = self.ema * self.a + (1 - self.ema) * a
            self.b = self.ema * self.b + (1 - self.ema) * b
        self.last = {"ok": True, "pitch_deg": float(p), "roll_deg": float(r), "inlier_frac": float(frac),
                     "a": float(a), "b": float(b), "reason": ""}
        return self.a * pred + self.b, tilt_rotation(cam_rot, p, r)

    def _fallback(self, pred, cam_rot, reason="no_fit"):
        self.last = {"ok": False, "pitch_deg": 0.0, "roll_deg": 0.0, "inlier_frac": 0.0, "reason": reason}
        if self.a is None:
            return pred, cam_rot
        return self.a * pred + self.b, cam_rot
