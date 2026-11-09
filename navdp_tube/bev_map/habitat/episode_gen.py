"""Episodes for data collection on the Mars yard: the 6 boulders moved near the 5 flags, a rover spawn, the oracle.

Frames as mars_scene.py (planner: x, y, height up; glTF / Habitat: X, Y up, Z = -y).

  prepare_assets   the yard without its boulders (the Habitat stage) + one .glb per boulder (an object to move)
  Yard             the terrain + flag poles, the boulders wherever an episode puts them, the hazard map
  make_episode     seed -> boulder placements, goal flag, start pose (spawn mode), oracle route
  Oracle           A* (8-connected, 0.1 m grid) on the true footprints grown by the robot radius; slopes and craters
                   do not block it (the rover can drive over them). Route = A* path shortened by line of sight.
  Route            distance to go along the route, signed distance off it, the route's turn ahead
"""
from __future__ import annotations

import heapq
import json
import math
import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mars_scene import MarsScene, read_glb, to_planner, write_glb_subset

FLAGS = ("flag_red", "flag_blue", "flag_yellow", "flag_green", "flag_purple")
BOULDERS = tuple(f"boulder_{i}" for i in range(6))
KINDS = ("rock", "flag", "slope", "crater")          # nearest-obstacle kinds (codes = index)
SPAWN_MODES = ("random", "facing_goal", "facing_away", "behind_rock", "near_rock")
# the yard floor north of the V-shaped ditch (scatter_rocks.py ALLOWED_AREA): starts are drawn in here
YARD = [(-19, -10.3), (-16, -10.5), (-12, -11.2), (-8, -13.5), (-5, -17), (-3, -19), (-1, -17.5), (1.5, -15.5),
        (1.5, -13), (2, -10), (3.5, -7.5), (5.5, -6.5), (6.5, -5), (7.5, -3.5), (10, -2.5), (12, -2), (14, 0),
        (16, 2.5), (17.5, 3.5), (17.5, 20), (14, 22.5), (-11, 22), (-19, 17)]


def prepare_assets(glb, out_dir):
    """-> (stage .glb: terrain + flags, {boulder: .glb with that boulder centred at its origin}). Cached in out_dir
    (rewritten when glb is newer)."""
    os.makedirs(out_dir, exist_ok=True)
    stage = os.path.join(out_dir, "marsyard_no_boulders.glb")
    rocks = {b: os.path.join(out_dir, f"{b}.glb") for b in BOULDERS}
    fresh = lambda p: os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(glb)  # noqa: E731
    if not fresh(stage):
        verts, _ = read_glb(glb)
        write_glb_subset(glb, set(verts) - set(BOULDERS) | {"marsyard_terrain"}, stage)
    for b, p in rocks.items():
        if not fresh(p):
            write_glb_subset(glb, {b}, p, bake=True)
    return stage, rocks


def inside_poly(x, y, poly=YARD):
    inside = False
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def rot_y(yaw):
    """glTF rotation about +Y (up) by yaw: in the planner frame a ccw turn by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class Yard:
    """The scene for every episode. scene: MarsScene of the stage (terrain, flag poles); the boulders are added with
    place_boulders. Hazards (terrain only, they never block): slope = smoothed terrain steeper than max_slope_deg,
    crater = steep ground inside a hollow, or a hollow deeper than crater_depth (relative to 2 m around)."""

    def __init__(self, stage_glb, rock_glbs, max_slope_deg=25.0, crater_depth=0.3, oracle_res=0.10):
        self.scene = MarsScene(stage_glb)
        self.rock_local = {}                 # boulder -> [N, 3] glTF vertices about its centre (subsampled)
        for b, p in rock_glbs.items():
            v = np.vstack(list(read_glb(p)[0].values()))
            self.rock_local[b] = v[:: max(1, len(v) // 6000)]
        s = self.scene
        r = s.hgrid.res
        h = gaussian_filter(s.height, 0.25 / r)
        gy, gx = np.gradient(h, r)
        self.slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        hollow = h - gaussian_filter(h, 3.0 / r)
        depth = h - gaussian_filter(h, 2.0 / r)
        steep = self.slope_deg > max_slope_deg
        crater = (steep & (hollow < -0.05)) | (depth < -crater_depth)
        self.hazard_kind = np.full(h.shape, -1, np.int8)          # on the height grid (0.1 m)
        self.hazard_kind[steep] = KINDS.index("slope")
        self.hazard_kind[crater] = KINDS.index("crater")
        d, (iy, ix) = distance_transform_edt(self.hazard_kind < 0, return_indices=True)
        self.hazard_dist = d * r
        self.hazard_near = self.hazard_kind[iy, ix]
        self.oracle_res = oracle_res
        self.placements = {}

    # ---------------------------------------------------------------- boulders
    def boulder_pose(self, b, x, y, yaw, sink=0.08):
        """Seat boulder b at (x, y) turned by yaw: lowered until its whole base is in the ground, plus sink x its
        height (as scatter_rocks.py). -> (glTF translation, planner vertices [N, 3], burial fraction)."""
        q = to_planner(self.rock_local[b] @ rot_y(yaw).T)        # about the centre, planner axes
        zmin, zmax = q[:, 2].min(), q[:, 2].max()
        height = zmax - zmin
        bottom = q[q[:, 2] < zmin + 0.2 * height]
        ground = self.scene.height_at(x + bottom[:, 0], y + bottom[:, 1])
        h0 = -float(np.max(bottom[:, 2] - ground)) - sink * height
        burial = (float(self.scene.height_at(x, y)) - (h0 + zmin)) / height
        pts = q + np.array([x, y, h0])
        return (float(x), float(h0), float(-y)), pts, burial

    def place_boulders(self, placements):
        """placements: {boulder: (x, y, yaw)} -> footprints in the scene; returns {boulder: glTF translation}."""
        out = {}
        for b, (x, y, yaw) in placements.items():
            t, pts, _ = self.boulder_pose(b, x, y, yaw)
            self.scene.set_footprint(b, pts)
            self.scene.origin[b] = (float(x), float(y))
            out[b] = t
        self.placements = dict(placements)
        self.refresh()
        return out

    def refresh(self, exclude=()):
        """The clearance field (solid_at) over every footprint but those named in exclude (the goal's pole)."""
        s = self.scene
        ids = [s.names.index(n) for n in exclude if n in s.names]
        occ = s.occ & ~np.isin(s.labels, ids)
        d, (iy, ix) = distance_transform_edt(~occ, return_indices=True)
        self.solid_dist = d * s.grid.res
        self.solid_near = s.labels[iy, ix]

    def kind_of(self, label):
        if label < 0:
            return -1
        return KINDS.index("flag") if self.scene.names[label].startswith("flag") else KINDS.index("rock")

    # ---------------------------------------------------------------- queries (planner x, y)
    def solid_at(self, x, y):
        """-> (centre distance to the nearest rock / pole footprint, its label)."""
        iy, ix = self.scene.grid.index(x, y)
        return float(self.solid_dist[iy, ix]), int(self.solid_near[iy, ix])

    def hazard_at(self, x, y):
        """-> (distance to the nearest slope / crater cell (0 = on one), its kind code)."""
        iy, ix = self.scene.hgrid.index(x, y)
        return float(self.hazard_dist[iy, ix]), int(self.hazard_near[iy, ix])

    def ray(self, x, y, yaw, max_d=5.0, step=0.05):
        """First rock / pole footprint along the heading -> (label or -1, distance or max_d)."""
        lab = self.scene.labels
        for d in np.arange(step, max_d + 1e-9, step):
            px, py = x + d * math.cos(yaw), y + d * math.sin(yaw)
            if not self.scene.grid.inside(px, py):
                break
            iy, ix = self.scene.grid.index(px, py)
            if lab[iy, ix] >= 0:
                return int(lab[iy, ix]), float(d)
        return -1, float(max_d)


# -------------------------------------------------------------------- oracle
class Oracle:
    """Shortest safe route to the goal pole on a res grid: a cell is free when its centre is at least `radius` from
    every rock / pole footprint (the goal pole excepted); the goal = the free cells whose rover circle is within
    reach_edge of the goal pole. Slopes and craters do not block."""

    def __init__(self, yard: Yard, goal_name, radius, reach_edge):
        s, res = yard.scene, yard.oracle_res
        self.yard, self.res, self.radius = yard, res, radius
        x0, x1, y0, y1 = s.bounds
        self.x0, self.y0 = x0, y0
        self.nx, self.ny = int((x1 - x0) / res), int((y1 - y0) / res)
        cx = x0 + (np.arange(self.nx) + 0.5) * res
        cy = y0 + (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(cx, cy)
        iy, ix = s.grid.index(X, Y)
        clear = s.distance_field(exclude=(goal_name,))[iy, ix]
        goal_d = s.distance_field(names=(goal_name,))[iy, ix] - radius
        self.free = clear >= radius
        self.goal = self.free & (goal_d <= reach_edge)
        self.goal_xy = s.origin[goal_name]
        self.reach = radius + reach_edge
        self._togo = None

    def cell(self, x, y):
        return (min(max(int((y - self.y0) / self.res), 0), self.ny - 1),
                min(max(int((x - self.x0) / self.res), 0), self.nx - 1))

    def xy(self, iy, ix):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def astar(self, start):
        """A* from start (x, y) -> [K, 2] cell-centre path ending in the goal set, or None. A start in a blocked cell
        starts from the nearest free one."""
        sy, sx = self.cell(*start)
        if not self.free[sy, sx]:
            d, (iy, ix) = distance_transform_edt(~self.free, return_indices=True)
            if d[sy, sx] * self.res > 0.5:
                return None
            sy, sx = int(iy[sy, sx]), int(ix[sy, sx])
        nx, free, goal, res = self.nx, self.free.ravel(), self.goal.ravel(), self.res
        gx, gy = self.goal_xy

        def h(i):
            x, y = self.x0 + (i % nx + 0.5) * res, self.y0 + (i // nx + 0.5) * res
            return max(0.0, math.hypot(x - gx, y - gy) - self.reach)

        s = sy * nx + sx
        g = {s: 0.0}
        parent = {s: -1}
        heap = [(h(s), s)]
        done = set()
        steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while heap:
            _, i = heapq.heappop(heap)
            if i in done:
                continue
            if goal[i]:
                path = []
                while i >= 0:
                    path.append(self.xy(i // nx, i % nx))
                    i = parent[i]
                return np.array(path[::-1])
            done.add(i)
            iy, ix = divmod(i, nx)
            for dy, dx in steps:
                jy, jx = iy + dy, ix + dx
                if not (0 <= jy < self.ny and 0 <= jx < nx):
                    continue
                j = jy * nx + jx
                if not free[j] or j in done:
                    continue
                if dy and dx and not (free[iy * nx + jx] and free[jy * nx + ix]):
                    continue                          # no corner cutting
                ng = g[i] + (res * math.sqrt(2) if dy and dx else res)
                if ng < g.get(j, math.inf):
                    g[j], parent[j] = ng, i
                    heapq.heappush(heap, (ng + h(j), j))
        return None

    def visible(self, a, b, step=0.05):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        for t in np.linspace(0, 1, n):
            iy, ix = self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not self.free[iy, ix]:
                return False
        return True

    def shorten(self, path):
        """Line-of-sight shortcut of the grid path (string pulling): the route the reference length is taken on."""
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.visible(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return np.array(out)

    def togo_field(self):
        """Geodesic distance (m) from every free cell to the goal set (Dijkstra, same graph as A*); inf elsewhere."""
        if self._togo is not None:
            return self._togo
        ny, nx, res = self.ny, self.nx, self.res
        free = self.free
        idx = np.arange(ny * nx).reshape(ny, nx)
        rows, cols, w = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):       # each undirected neighbour pair once
            ya, yb = slice(0, ny - dy), slice(dy, ny)
            xa, xb = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
            ok = free[ya, xa] & free[yb, xb]
            if dy and dx:                                       # no corner cutting, as in A*
                ok &= free[ya, xb] & free[yb, xa]
            rows.append(idx[ya, xa][ok])
            cols.append(idx[yb, xb][ok])
            w.append(np.full(int(ok.sum()), res * (math.sqrt(2) if dy and dx else 1.0)))
        r, c, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        g = csr_matrix((np.r_[ww, ww], (np.r_[r, c], np.r_[c, r])), shape=(ny * nx, ny * nx))
        d = dijkstra(g, directed=False, indices=np.flatnonzero(self.goal.ravel()), min_only=True)
        self._togo = d.reshape(ny, nx)
        return self._togo

    def togo_at(self, x, y):
        f = self.togo_field()
        iy, ix = self.cell(x, y)
        return float(f[iy, ix])


class Route:
    """The oracle route as a polyline: arclength, projection, signed offset (+ = left of the route's direction)."""

    def __init__(self, pts):
        self.p = np.asarray(pts, float)
        seg = np.diff(self.p, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        keep = self.seg_len > 1e-9
        self.a, self.d, self.seg_len = self.p[:-1][keep], seg[keep], self.seg_len[keep]
        self.s0 = np.r_[0.0, np.cumsum(self.seg_len)[:-1]]
        self.length = float(self.seg_len.sum())

    def project(self, x, y):
        """-> (arclength s of the nearest route point, signed offset (+ left), unit tangent there)."""
        q = np.array([x, y])
        t = np.clip(((q - self.a) * self.d).sum(1) / self.seg_len ** 2, 0, 1)
        foot = self.a + t[:, None] * self.d
        dist = np.linalg.norm(q - foot, axis=1)
        k = int(np.argmin(dist))
        tan = self.d[k] / self.seg_len[k]
        rel = q - foot[k]
        side = tan[0] * rel[1] - tan[1] * rel[0]
        return float(self.s0[k] + t[k] * self.seg_len[k]), float(math.copysign(dist[k], side) if dist[k] > 1e-9 else 0.0), tan

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        k = int(np.clip(np.searchsorted(self.s0, s, side="right") - 1, 0, len(self.a) - 1))
        return self.a[k] + (s - self.s0[k]) / self.seg_len[k] * self.d[k]

    def side_of(self, x, y):
        """Which side of the route the point (x, y) is on, travelling toward the goal: +1 left, -1 right."""
        _, off, _ = self.project(x, y)
        return 1 if off >= 0 else -1


# -------------------------------------------------------------------- episodes
def make_episode(yard: Yard, seed, radius=0.3, reach_edge=0.5, spawn_modes=SPAWN_MODES, min_len=4.0, max_len=30.0,
                 near_flag=(1.0, 6.0), goal_rocks=(1, 3), tries=200):
    """seed -> dict(seed, goal, spawn, start, yaw, boulders {name: (x, y, yaw)}, route, astar_path, oracle_length_m)
    and the Oracle (with the boulders already placed in yard). None if nothing valid was found."""
    rng = np.random.default_rng(seed)
    s = yard.scene
    for _ in range(20):
        goal = FLAGS[int(rng.integers(len(FLAGS)))]
        # boulders: k of them round the goal flag, the rest round random flags, each 1..6 m from its flag's pole
        k = int(rng.integers(goal_rocks[0], goal_rocks[1] + 1))
        hosts = [goal] * k + [FLAGS[int(rng.integers(len(FLAGS)))] for _ in range(len(BOULDERS) - k)]
        order = rng.permutation(len(BOULDERS))
        placed, footprint_r = {}, {}
        for b_i, host in zip(order, hosts):
            b = BOULDERS[b_i]
            fx, fy = s.origin[host]
            for _ in range(tries):
                yaw = float(rng.uniform(-math.pi, math.pi))
                q = to_planner(yard.rock_local[b] @ rot_y(yaw).T)
                rad = float(np.max(np.hypot(q[:, 0], q[:, 1])))
                d = float(rng.uniform(near_flag[0] + rad, near_flag[1] + rad))
                a = float(rng.uniform(-math.pi, math.pi))
                x, y = fx + d * math.cos(a), fy + d * math.sin(a)
                if not inside_poly(x, y) or not s.grid.inside(x, y):
                    continue
                if any(math.hypot(x - px, y - py) < rad + footprint_r[o] + 0.8 for o, (px, py, _) in placed.items()):
                    continue
                if any(math.hypot(x - s.origin[f][0], y - s.origin[f][1]) < rad + near_flag[0] for f in FLAGS):
                    continue
                iy, ix = s.hgrid.index(x, y)
                if yard.slope_deg[iy, ix] > 20.0:
                    continue
                _, _, burial = yard.boulder_pose(b, x, y, yaw)
                if burial > 0.45:
                    continue
                placed[b], footprint_r[b] = (x, y, yaw), rad
                break
        if len(placed) < len(BOULDERS):
            continue
        yard.place_boulders(placed)
        oracle = Oracle(yard, goal, radius, reach_edge)
        if not oracle.goal.any():
            continue
        gx, gy = s.origin[goal]
        for _ in range(tries):
            mode = spawn_modes[int(rng.integers(len(spawn_modes)))]
            start = spawn(yard, rng, mode, goal, radius)
            if start is None:
                continue
            x, y, yaw = start
            if not inside_poly(x, y) or yard.solid_at(x, y)[0] < radius + 0.15:
                continue
            if math.hypot(gx - x, gy - y) < min_len:
                continue
            path = oracle.astar((x, y))
            if path is None:
                continue
            route = oracle.shorten(np.vstack([[x, y], path]))
            length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
            if not min_len <= length <= max_len:
                continue
            return {"seed": int(seed), "goal": goal, "goal_xy": [gx, gy], "spawn": mode,
                    "start": [float(x), float(y)], "yaw": float(yaw),
                    "boulders": {b: [round(v, 4) for v in p] for b, p in placed.items()},
                    "route": route.round(3).tolist(), "astar_path": path.round(3).tolist(),
                    "astar_length_m": float(np.sum(np.linalg.norm(np.diff(np.vstack([[x, y], path]), axis=0), axis=1))),
                    "oracle_length_m": length, "radius_m": radius, "reach_edge_m": reach_edge}, oracle
    return None, None


def spawn(yard: Yard, rng, mode, goal, radius):
    """A start pose (x, y, yaw) for the spawn mode, or None:
      random       anywhere on the yard floor, any heading
      facing_goal  anywhere, heading toward the goal +- 20 deg
      facing_away  anywhere, heading away from the goal +- 30 deg
      behind_rock  0.4..2 m behind a boulder on the boulder-goal line (the rock between rover and goal), facing the
                   goal +- 15 deg; the boulder is one round the goal flag when there is one
      near_rock    beside a boulder (rover's edge 0.1..0.6 m from it), any heading"""
    s = yard.scene
    gx, gy = s.origin[goal]
    xs, ys = [p[0] for p in YARD], [p[1] for p in YARD]
    if mode in ("random", "facing_goal", "facing_away"):
        x, y = float(rng.uniform(min(xs), max(xs))), float(rng.uniform(min(ys), max(ys)))
        to_goal = math.atan2(gy - y, gx - x)
        yaw = {"random": float(rng.uniform(-math.pi, math.pi)),
               "facing_goal": to_goal + math.radians(float(rng.uniform(-20, 20))),
               "facing_away": to_goal + math.pi + math.radians(float(rng.uniform(-30, 30)))}[mode]
        return x, y, (yaw + math.pi) % (2 * math.pi) - math.pi
    rocks = list(yard.placements)
    near_goal = [b for b in rocks if math.hypot(s.origin[b][0] - gx, s.origin[b][1] - gy) < 8.0]
    b = (near_goal if near_goal and mode == "behind_rock" else rocks)[int(rng.integers(len(near_goal) if near_goal
                                                                        and mode == "behind_rock" else len(rocks)))]
    own = s.distance_field(names=(b,))
    bx, by = s.origin[b]
    if mode == "behind_rock":
        gap = float(rng.uniform(0.4, 2.0))
        d = math.hypot(bx - gx, by - gy)
        ux, uy = (bx - gx) / d, (by - gy) / d
        t = 0.0
        while s.lookup(own, bx + ux * t, by + uy * t) - radius < gap:
            t += 0.02
            if t > 10:
                return None
        x, y = bx + ux * t, by + uy * t
        yaw = math.atan2(gy - y, gx - x) + math.radians(float(rng.uniform(-15, 15)))
        return x, y, (yaw + math.pi) % (2 * math.pi) - math.pi
    gap = float(rng.uniform(0.1, 0.6))                    # near_rock
    a = float(rng.uniform(-math.pi, math.pi))
    t = 0.0
    while s.lookup(own, bx + math.cos(a) * t, by + math.sin(a) * t) - radius < gap:
        t += 0.02
        if t > 10:
            return None
    return bx + math.cos(a) * t, by + math.sin(a) * t, float(rng.uniform(-math.pi, math.pi))


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def draw_episode(yard: Yard, ep, trails=(), events=(), px_per_m=20, margin=4.0):
    """Top view: terrain, hazards (slope orange, crater blue), footprints, the oracle route (white), trails
    [(name, [K, 2], BGR)], event markers [(x, y, label, BGR)]."""
    s = yard.scene
    pts = np.vstack([np.array(ep["route"])] + [np.asarray(t[1]) for t in trails if len(t[1])])
    x0, x1 = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    y0, y1 = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    w, h = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    X, Y = np.meshgrid(x0 + (np.arange(w) + 0.5) / px_per_m, y1 - (np.arange(h) + 0.5) / px_per_m)
    H = s.height_at(X, Y).astype(np.float32)
    gy, gx = np.gradient(H)
    shade = np.clip(0.6 + 4.0 * (gx - gy) * px_per_m / 24, 0.15, 1.0)
    img = np.stack([shade * 150, shade * 175, shade * 215], -1).astype(np.uint8)
    hy, hx = s.hgrid.index(X, Y)
    hk = yard.hazard_kind[hy, hx]
    img[hk == KINDS.index("slope")] = (img[hk == KINDS.index("slope")] * 0.5 + np.array([0, 110, 220]) * 0.5)
    img[hk == KINDS.index("crater")] = (img[hk == KINDS.index("crater")] * 0.5 + np.array([200, 80, 20]) * 0.5)
    iy, ix = s.grid.index(X, Y)
    img[s.occ[iy, ix]] = (60, 60, 60)
    P = lambda p: (int((p[0] - x0) * px_per_m), int((y1 - p[1]) * px_per_m))  # noqa: E731
    cv2.polylines(img, [np.array([P(p) for p in ep["route"]], np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
    for name, trail, col in trails:
        if len(trail) > 1:
            cv2.polylines(img, [np.array([P(p) for p in trail], np.int32)], False, col, 2, cv2.LINE_AA)
    sx, sy = ep["start"]
    cv2.circle(img, P((sx, sy)), max(2, int(ep["radius_m"] * px_per_m)), (200, 60, 20), 2)
    cv2.line(img, P((sx, sy)), P((sx + 0.8 * math.cos(ep["yaw"]), sy + 0.8 * math.sin(ep["yaw"]))), (200, 60, 20), 2)
    cv2.circle(img, P(ep["goal_xy"]), 8, (0, 200, 0), 2)
    for x, y, label, col in events:
        cv2.drawMarker(img, P((x, y)), col, cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(img, label, (P((x, y))[0] + 6, P((x, y))[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                    cv2.LINE_AA)
    y_txt = 18
    for name, _, col in trails:
        cv2.putText(img, name, (8, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y_txt += 18
    cv2.putText(img, f"seed {ep['seed']}  {ep['spawn']} -> {ep['goal']}  oracle {ep['oracle_length_m']:.1f} m",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
