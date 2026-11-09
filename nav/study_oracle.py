"""Oracle (shortest collision-free) path for the user study.

Ground truth comes from the yard GLB itself (navdp_tube's MarsScene, via
sam_vla.env.glb_terrain.GlbTerrain): every boulder's footprint, inflated by
the rover radius, is an obstacle; terrain and bedrock are free; flags are not
obstacles (they are the goals). The oracle is the geodesic from the start to
the goal's success disc (the first point within `radius` of the goal), found
with Dijkstra on a 0.1 m, 16-connected grid (scipy.sparse.csgraph, C speed)
and then string-pulled with line-of-sight checks, so its length is close to
the true continuous shortest path (a raw 8-connected grid path overestimates
diagonal-ish paths by up to ~8 %).

World frame here is this project's (X, Z); the scene's planner frame is
(x, y) = (X, -Z).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# 16-connected moves (dy, dx): 8 neighbours + 8 knight moves
_MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
          (1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]


class OraclePlanner:
    def __init__(self, scene, res: float = 0.10, robot_radius: float = 0.35):
        """scene: navdp_tube MarsScene (GlbTerrain(...).scene)."""
        self.res = float(res)
        self.robot_radius = float(robot_radius)
        g = scene.grid
        k = max(1, int(round(self.res / g.res)))
        self.res = g.res * k
        ids = [i for i, n in enumerate(scene.names) if not n.startswith("flag")]
        occ = np.isin(scene.labels, ids)
        ny, nx = occ.shape[0] // k, occ.shape[1] // k
        occ = occ[: ny * k, : nx * k].reshape(ny, k, nx, k).any(axis=(1, 3))
        clearance = distance_transform_edt(~occ) * self.res
        self.free = clearance >= self.robot_radius
        self.x_min, self.y_min = g.x_min, g.y_min
        self.ny, self.nx = ny, nx
        self.obstacle_names = [scene.names[i] for i in ids]
        self._build_graph()

    # -- grid <-> world ---------------------------------------------------------
    def _cell(self, X: float, Z: float) -> Tuple[int, int]:
        x, y = X, -Z
        ix = int(np.clip((x - self.x_min) / self.res, 0, self.nx - 1))
        iy = int(np.clip((y - self.y_min) / self.res, 0, self.ny - 1))
        return iy, ix

    def _world(self, iy: int, ix: int) -> Tuple[float, float]:
        x = self.x_min + (ix + 0.5) * self.res
        y = self.y_min + (iy + 0.5) * self.res
        return float(x), float(-y)

    def _build_graph(self) -> None:
        n = self.ny * self.nx
        idx = np.arange(n).reshape(self.ny, self.nx)
        rows, cols, w = [], [], []
        for dy, dx in _MOVES:
            ys = slice(max(0, -dy), self.ny - max(0, dy))
            xs = slice(max(0, -dx), self.nx - max(0, dx))
            yd = slice(max(0, dy), self.ny - max(0, -dy))
            xd = slice(max(0, dx), self.nx - max(0, -dx))
            ok = self.free[ys, xs] & self.free[yd, xd]
            if abs(dy) + abs(dx) == 3:  # knight move: the two cells it cuts through must be free
                sy, sx = int(math.copysign(1, dy)), int(math.copysign(1, dx))
                if abs(dx) == 2:
                    via = [(0, sx), (dy, sx)]
                else:
                    via = [(sy, 0), (sy, dx)]
                for vy, vx in via:
                    ok &= self._shift(self.free, vy, vx)[ys, xs]
            rows.append(idx[ys, xs][ok])
            cols.append(idx[yd, xd][ok])
            w.append(np.full(int(ok.sum()), math.hypot(dy, dx) * self.res))
        self.graph = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n, n)).tocsr()

    def _shift(self, a: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """out[y, x] = a[y + dy, x + dx] (False outside)."""
        out = np.zeros_like(a)
        ys = slice(max(0, -dy), a.shape[0] - max(0, dy))
        xs = slice(max(0, -dx), a.shape[1] - max(0, dx))
        yd = slice(max(0, dy), a.shape[0] - max(0, -dy))
        xd = slice(max(0, dx), a.shape[1] - max(0, -dx))
        out[ys, xs] = a[yd, xd]
        return out

    def _nearest_free(self, iy: int, ix: int) -> Tuple[int, int]:
        if self.free[iy, ix]:
            return iy, ix
        fy, fx = np.nonzero(self.free)
        j = int(np.argmin((fy - iy) ** 2 + (fx - ix) ** 2))
        return int(fy[j]), int(fx[j])

    def _los(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        ys = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        xs = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return bool(self.free[ys, xs].all())

    # -- planning -----------------------------------------------------------------
    def plan(self, start_xz: Sequence[float], goal_xz: Sequence[float], radius: float):
        """Shortest path from start to within `radius` of goal.
        -> dict(length_m, path_xz [list of (X, Z)], end_xz, reachable)."""
        s = self._nearest_free(*self._cell(*start_xz))
        straight = math.hypot(goal_xz[0] - start_xz[0], goal_xz[1] - start_xz[1])
        if straight <= radius:
            return {"length_m": 0.0, "path_xz": [tuple(start_xz)], "end_xz": tuple(start_xz), "reachable": True}
        src = s[0] * self.nx + s[1]
        dist, pred = dijkstra(self.graph, directed=False, indices=src, return_predecessors=True)
        dist = dist.reshape(self.ny, self.nx)
        gy, gx = self._cell(*goal_xz)
        r = int(math.ceil(radius / self.res)) + 1
        y0, y1 = max(0, gy - r), min(self.ny, gy + r + 1)
        x0, x1 = max(0, gx - r), min(self.nx, gx + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx, cz = np.vectorize(self._world)(yy, xx)
        disc = np.hypot(cx - goal_xz[0], cz - goal_xz[1]) <= radius
        sub = np.where(disc, dist[y0:y1, x0:x1], np.inf)
        if not np.isfinite(sub).any():
            return {"length_m": None, "path_xz": [], "end_xz": None, "reachable": False}
        j = np.unravel_index(int(np.argmin(sub)), sub.shape)
        end = (y0 + int(j[0]), x0 + int(j[1]))
        cells = [end]
        cur = end[0] * self.nx + end[1]
        while cur != src and cur >= 0:
            cur = int(pred[cur])
            if cur < 0:
                break
            cells.append(divmod(cur, self.nx))
        cells.reverse()
        # string-pull: keep only the corners needed for line of sight
        pulled = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j2 = len(cells) - 1
            while j2 > i + 1 and not self._los(cells[i], cells[j2]):
                j2 -= 1
            pulled.append(cells[j2])
            i = j2
        pts = [tuple(start_xz)] + [self._world(*c) for c in pulled[1:]]
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        return {"length_m": float(length), "path_xz": pts, "end_xz": pts[-1], "reachable": True}

    def plan_sequence(self, start_xz, goals: List[Tuple[float, float]], radius: float):
        """Oracle over an ordered list of goals (each leg starts where the
        previous reached its success disc)."""
        legs, cur, total, path = [], tuple(start_xz), 0.0, [tuple(start_xz)]
        for g in goals:
            leg = self.plan(cur, g, radius)
            legs.append(leg)
            if not leg["reachable"]:
                return {"length_m": None, "legs": legs, "path_xz": path, "reachable": False}
            total += leg["length_m"]
            path += leg["path_xz"][1:]
            cur = leg["end_xz"]
        return {"length_m": total, "legs": legs, "path_xz": path, "reachable": True}


def load_yard_oracle(yard_glb: str, robot_radius: float = 0.35) -> Tuple[OraclePlanner, dict]:
    """-> (planner, {object name: (X, Z)}) for a yard GLB (flags + boulders)."""
    from sam_vla.env.glb_terrain import GlbTerrain

    scene = GlbTerrain(yard_glb).scene
    objects = {n: (float(p[0]), -float(p[1])) for n, p in scene.origin.items()}
    return OraclePlanner(scene, robot_radius=robot_radius), objects


class StraightLineOracle:
    """Fallback when no obstacle ground truth exists (heightmap scenes)."""

    obstacle_names: list = []

    def plan_sequence(self, start_xz, goals, radius):
        cur, total, path, legs = tuple(start_xz), 0.0, [tuple(start_xz)], []
        for g in goals:
            d = math.hypot(g[0] - cur[0], g[1] - cur[1])
            step = max(0.0, d - radius)
            end = cur if d <= 1e-9 else (cur[0] + (g[0] - cur[0]) * step / d, cur[1] + (g[1] - cur[1]) * step / d)
            legs.append({"length_m": step, "path_xz": [cur, end], "end_xz": end, "reachable": True})
            total += step
            path.append(end)
            cur = end
        return {"length_m": total, "legs": legs, "path_xz": path, "reachable": True}
