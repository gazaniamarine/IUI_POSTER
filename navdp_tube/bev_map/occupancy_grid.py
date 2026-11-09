"""Incremental log-odds occupancy grid in the world (odom) frame.

Cell (row=iy, col=ix) covers x in [x_min + ix*res, ...), y in [y_min + iy*res, ...).
States: FREE = 0, OCCUPIED = 100, UNKNOWN = -1 (same convention as ROS OccupancyGrid).
"""

import cv2
import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


class OccupancyGrid:
    def __init__(self, gcfg, ccfg):
        self.res = gcfg["resolution"]
        self.x_min, self.x_max = gcfg["x_min"], gcfg["x_max"]
        self.y_min, self.y_max = gcfg["y_min"], gcfg["y_max"]
        self.nx = int(round((self.x_max - self.x_min) / self.res))
        self.ny = int(round((self.y_max - self.y_min) / self.res))
        self.g, self.c = gcfg, ccfg
        self.L = np.zeros((self.ny, self.nx), np.float32)
        cx = self.x_min + (np.arange(self.nx) + 0.5) * self.res
        cy = self.y_min + (np.arange(self.ny) + 0.5) * self.res
        self.cell_x, self.cell_y = np.meshgrid(cx, cy)

    def reset(self):
        self.L[:] = 0

    # ------------------------------------------------------------------ helpers
    def world_to_cell(self, x, y):
        ix = np.floor((x - self.x_min) / self.res).astype(np.int64)
        iy = np.floor((y - self.y_min) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def _flat(self, pts):
        ix, iy, ok = self.world_to_cell(pts[:, 0], pts[:, 1])
        return iy[ok] * self.nx + ix[ok], ok

    def _counts(self, pts):
        flat, _ = self._flat(pts)
        return np.bincount(flat, minlength=self.nx * self.ny).reshape(self.ny, self.nx)

    # ------------------------------------------------------------------ update
    def slack(self, dist):
        """Extra height tolerance at a given distance from the camera.

        An uncorrected camera pitch error of e degrees moves the floor by dist * tan(e) at that
        distance, so floor/obstacle thresholds widen with range by that amount.
        """
        return dist * np.tan(np.radians(self.c.get("pitch_uncertainty_deg", 0.0)))

    def classify(self, pts, cam_xy=None):
        z = pts[:, 2]
        s = 0.0 if cam_xy is None else self.slack(np.hypot(pts[:, 0] - cam_xy[0], pts[:, 1] - cam_xy[1]))
        floor = np.abs(z) < self.c["floor_tol"] + s
        obst = (z > self.c["obstacle_min_z"] + s) & (z < self.c["obstacle_max_z"])
        return floor, obst

    def update(self, pts, cam_xy, max_range, floor=None, obst=None):
        """Fuse one frame of world points (N,3). Returns per-frame evidence masks."""
        g, c = self.g, self.c
        z = pts[:, 2]
        if floor is None or obst is None:
            floor, obst = self.classify(pts, cam_xy)

        n_obst = self._counts(pts[obst])
        n_floor = self._counts(pts[floor])
        # tallest obstacle point per cell: a real obstacle has vertical extent, a sheet of
        # floor noise lifted just above floor_tol does not
        tallest = np.zeros(self.nx * self.ny, np.float32)
        flat, ok = self._flat(pts[obst])
        np.maximum.at(tallest, flat, z[obst][ok].astype(np.float32))
        tallest = tallest.reshape(self.ny, self.nx)

        # required obstacle points per cell falls off with distance (fewer pixels land in far cells)
        dist = np.hypot(self.cell_x - cam_xy[0], self.cell_y - cam_xy[1])
        need = np.clip(g["min_points_near"] * (g["near_ref"] / np.maximum(dist, 1e-3)) ** 2,
                       g["min_points_far"], g["min_points_near"])
        in_range = dist < max_range
        occ_range = dist < g.get("obstacle_max_range", max_range)

        occ_ev = (n_obst >= need) & (tallest >= c["min_obstacle_top"] + self.slack(dist)) & occ_range
        free_ev = (n_floor > 0) & ~occ_ev & (n_obst == 0) & in_range

        self.L[occ_ev] += g["l_occ"]
        self.L[free_ev] += g["l_free"]
        np.clip(self.L, g["l_min"], g["l_max"], out=self.L)
        return occ_ev, free_ev

    def fuse(self, occ_ev, free_ev):
        """Add per-cell evidence masks (e.g. projection.seg_cell_evidence) to the log-odds."""
        self.L[occ_ev] += self.g["l_occ"]
        self.L[free_ev & ~occ_ev] += self.g["l_free"]
        np.clip(self.L, self.g["l_min"], self.g["l_max"], out=self.L)
        return occ_ev, free_ev & ~occ_ev

    # ------------------------------------------------------------------ output
    def state(self, clean=True):
        g = self.g
        s = np.full(self.L.shape, UNKNOWN, np.int8)
        occ = self.L > g["occ_threshold"]
        free = self.L < g["free_threshold"]
        if clean:
            occ = remove_small_blobs(occ, g["min_blob_cells"])
            if g["close_free_iters"] > 0:
                k = np.ones((3, 3), np.uint8)
                closed = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, k,
                                          iterations=g["close_free_iters"]) > 0
                free = free | (closed & ~occ)
            free &= ~occ
        s[free] = FREE
        s[occ] = OCCUPIED
        return s

    def to_ros_prob(self):
        """int8 grid in ROS convention (-1 unknown, 0..100 occupancy %), row 0 = y_min."""
        prob = np.full(self.L.shape, -1, np.int8)
        touched = self.L != 0
        prob[touched] = np.round(100.0 / (1.0 + np.exp(-self.L[touched]))).astype(np.int8)
        return prob

    def save(self, stem, state=None):
        """Write <stem>.npy (ROS int8 probabilities), <stem>.pgm + <stem>.yaml (ROS map_server)."""
        np.save(f"{stem}.npy", self.to_ros_prob())
        state = self.state() if state is None else state
        img = np.full(state.shape, 205, np.uint8)
        img[state == FREE] = 254
        img[state == OCCUPIED] = 0
        img = np.flipud(img)                       # PGM row 0 = top; ROS origin = bottom-left
        with open(f"{stem}.pgm", "wb") as f:
            f.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
            f.write(img.tobytes())
        name = stem.rsplit("/", 1)[-1]
        with open(f"{stem}.yaml", "w") as f:
            f.write(f"image: {name}.pgm\nresolution: {self.res}\n"
                    f"origin: [{self.x_min}, {self.y_min}, 0.0]\nnegate: 0\n"
                    f"occupied_thresh: 0.65\nfree_thresh: 0.25\n")

    def inflate(self, state, radius):
        """Occupied cells grown by the robot radius (for planning)."""
        r = max(1, int(np.ceil(radius / self.res)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate((state == OCCUPIED).astype(np.uint8), k) > 0


def remove_small_blobs(mask, min_cells):
    if min_cells <= 1:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_cells
    return keep[lab]
