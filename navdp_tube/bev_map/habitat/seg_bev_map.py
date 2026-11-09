"""The BEV occupancy map of the Mars yard from the SAM2 terrain segmentation (best.pth), shared by
teleop_seg_bev.py (keyboard driving) and run_mars_habitat.py --map seg (the planners).

Built with the original BEVPipeline (depth.source = sam2): every map cell near the rover is projected into the
camera image and takes the class of the pixel that sees it (projection.seg_cell_evidence):
  free classes (soil)             -> free          occupied classes (sand, bedrock) -> occupied where they lie
  rock classes (bigrock)          -> occupied from where the rock meets the ground to rock_depth behind it
  boulders / flags (object mask)  -> standing obstacles, whatever their class; the ground behind them stays unknown
  ground hidden by the terrain    -> free (hidden_free)          sky / black backdrop -> ignored
The ground's shape is the yard's known terrain (mars_scene). Habitat's depth is used only for the sky (depth 0) and
the boulders / flags: a pixel whose point is > object_height above the terrain is on one, and it is placed at that
depth (a bulging boulder hides where it meets the ground, so its column would put it up to ~1 m too far).
The ground classes come from the segmentation alone.
"""

import math
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bev_pipeline import BEVPipeline, load_config  # noqa: E402
from bev_render import COLORS  # noqa: E402
from occupancy_grid import OCCUPIED, UNKNOWN  # noqa: E402
from tube_nav import WorldGrid  # noqa: E402

SEG_COLORS = np.array([[60, 160, 60], [150, 150, 150], [60, 200, 230], [40, 40, 230]], np.uint8)  # BGR per class
COLOR_NAMES = {"soil": "green", "bedrock": "grey", "sand": "yellow", "bigrock": "red"}


def terrain_depth(scene, K, width, height, x, y, yaw, cam_h, max_range=6.0, step=0.05, stride=2):
    """z-depth at which each pixel's ray (level camera at x, y, cam_h, looking along yaw) meets the terrain; inf if
    not within max_range. Marched every `step` m on a pixel grid of `stride`, then refined linearly."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    u = (np.arange(0, width, stride) + 0.5 - cx) / f
    v = (np.arange(0, height, stride) + 0.5 - cy) / f
    xr, yd = np.meshgrid(u, v)                       # right / down per unit of depth
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = c + xr * s, s - xr * c                  # ground-plane direction per unit of depth
    out = np.full(xr.shape, np.inf, np.float32)
    todo = np.ones(xr.shape, bool)
    prev = np.full(xr.shape, cam_h - float(scene.height_at(x, y)))   # height above terrain at the camera
    t_prev = 0.0
    for t in np.arange(step, max_range + step, step):
        idx = np.nonzero(todo)
        if idx[0].size == 0:
            break
        above = cam_h - t * yd[idx] - scene.height_at(x + t * dx[idx], y + t * dy[idx])
        hit = above <= 0
        if hit.any():
            a0 = prev[idx][hit]
            out[idx[0][hit], idx[1][hit]] = t_prev + step * a0 / np.maximum(a0 - above[hit], 1e-9)
            todo[idx[0][hit], idx[1][hit]] = False
        prev[idx] = above
        t_prev = t
    if stride > 1:              # back to full size: bilinear where the 4 neighbours hit the terrain, else nearest
        hit = np.isfinite(out)
        lin = cv2.resize(np.where(hit, out, 0).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        ok = cv2.resize(hit.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.999
        near = cv2.resize(out, (width, height), interpolation=cv2.INTER_NEAREST)
        out = np.where(ok, lin, near).astype(np.float32)
    return out



def class_list(s):
    from sam2_seg import CLASS_NAMES
    names = [n.strip() for n in (s.split(",") if isinstance(s, str) else s) if n.strip()]
    bad = [n for n in names if n not in CLASS_NAMES]
    if bad:
        raise SystemExit(f"unknown class {bad}: use {', '.join(CLASS_NAMES)}")
    return names


def add_args(ap):
    """The map's command-line options (teleop_seg_bev.py, run_mars_habitat.py)."""
    g = ap.add_argument_group("segmentation BEV map")
    g.add_argument("--occupied", default="sand,bedrock", help="classes whose ground is occupied (comma list)")
    g.add_argument("--free", default="soil", help="free-ground classes")
    g.add_argument("--rocks", default="bigrock", help="standing-obstacle classes")
    g.add_argument("--rock-depth", type=float, default=0.5, help="m behind a rock's front that is occupied")
    g.add_argument("--object-height", type=float, default=0.15,
                   help="m above the terrain from which a pixel is on a boulder / flag (not ground)")
    g.add_argument("--object-top", type=float, default=1.0,
                   help="boulder / flag points up to this high above the terrain mark their cell occupied (rover height)")
    g.add_argument("--hidden-unknown", action="store_true",
                   help="ground hidden by the terrain (behind rises, in dips) stays unknown (default: free)")
    g.add_argument("--views", type=int, default=1,
                   help="views of a class before a cell shows it (1 = the map follows each view at once)")
    g.add_argument("--range", type=float, default=3.5, help="map cells up to this far from the rover (m)")
    g.add_argument("--min-conf", type=float, default=0.0, help="ignore pixels whose class probability is below this")
    return g


class SegBEVMap:
    """Whole-yard BEV map (5 cm cells). Map frame = yard frame shifted so the start is (0, 0)."""

    def __init__(self, scene, K, width, height, start, args, overrides=()):
        self.scene, self.K, self.W, self.H, self.args = scene, K, width, height, args
        self.sx, self.sy = start
        self.occ_c, self.free_c = class_list(args.occupied), class_list(args.free)
        self.rock_c = class_list(args.rocks)
        yg = scene.grid
        sx, sy = self.sx, self.sy
        cfg = load_config(overrides=[
            "depth.source=sam2", f"depth.max_range={args.range}", f"grid.obstacle_max_range={args.range}",
            f"grid.x_min={yg.x_min - sx:.3f}", f"grid.x_max={yg.x_min + yg.nx * yg.res - sx:.3f}",
            f"grid.y_min={yg.y_min - sy:.3f}", f"grid.y_max={yg.y_min + yg.ny * yg.res - sy:.3f}",
            f"seg.min_conf={args.min_conf}",
            "grid.resolution=0.05", "grid.min_blob_cells=1", "grid.close_free_iters=0",
            f"grid.l_occ={2.05 / args.views:.3f}", f"grid.l_free={-2.05 / args.views:.3f}",   # threshold 2.0
            "grid.free_threshold=-2.0", "grid.l_max=3.0", "grid.l_min=-3.0", f"seg.rock_depth={args.rock_depth}",
            f"seg.terrain_hidden_free={str(not args.hidden_unknown).lower()}"] + list(overrides))
        cfg["seg"].update(obstacle_classes=self.rock_c, free_classes=self.free_c,
                          ground_occupied_classes=self.occ_c)
        self.cfg = cfg
        self.pipe = BEVPipeline(cfg)
        self.z0 = float(scene.height_at(sx, sy))     # map frame: z = 0 at the terrain under the start
        self.fov = 2 * math.atan(width / 2 / K[0, 0])
        self.last = None                              # (frame, result, ms) of the last update
        f = K[0, 0]
        self.xr = (np.arange(width)[None, :] + 0.5 - K[0, 2]) / f      # right / down per metre of depth
        self.yd = (np.arange(height)[:, None] + 0.5 - K[1, 2]) / K[1, 1]

    @property
    def grid(self):
        return self.pipe.grid

    def reset(self):
        self.pipe.reset()

    def frame(self, rgb, hd, x, y, yaw, cam_h):
        """Habitat RGB + depth (masks only) at yard pose x, y, yaw, camera height cam_h -> a BEVPipeline frame."""
        a, scene = self.args, self.scene
        scene_px = hd > 0                              # False = sky / black backdrop (not part of the yard)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[s, 0, -c], [-c, 0, -s], [0, 1, 0]], float)      # columns: camera right, up, back
        far = a.range + 0.5
        ground = terrain_depth(scene, self.K, self.W, self.H, x, y, yaw, cam_h, far + 3.0, stride=4)
        px, py = x + hd * (c + self.xr * s), y + hd * (s - self.xr * c)
        above = cam_h - hd * self.yd - scene.height_at(px, py)
        objects = scene_px & (hd < far) & (above > a.object_height)
        objects = cv2.morphologyEx(objects.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        sx, sy, z0 = self.sx, self.sy, self.z0
        body = objects & (above <= a.object_top)       # the part the rover can hit: its footprint is occupied
        object_xy = np.stack([px[body] - sx, py[body] - sy], -1)
        return SimpleNamespace(rgb=rgb, K=self.K, cam_pos=np.array([x - sx, y - sy, cam_h - z0]), cam_rot=rot,
                               ground_z=lambda qx, qy: scene.height_at(qx + sx, qy + sy) - z0,
                               mask=scene_px, objects=objects, object_depth=hd, object_xy=object_xy,
                               floor_depth=ground)

    def update(self, rgb, hd, x, y, yaw, cam_h):
        fr = self.frame(rgb, hd, x, y, yaw, cam_h)
        t0 = time.time()
        r = self.pipe.process(fr)
        self.last = (fr, r, (time.time() - t0) * 1000)
        return self.last

    # ------------------------------------------------------------------ for the planners
    def occupied(self):
        """Occupied cells as a WorldGrid in the yard frame."""
        g = self.grid
        return WorldGrid(self.grid.state() == OCCUPIED, g.res, (g.x_min + self.sx, g.y_min + self.sy))

    def world_map(self, x, y, goal, nc):
        """What the planner gets: occupied cells within nav.obstacle_range of the rover, the goal disk erased
        (the goal flag itself), no robot size (the driver adds the inflation)."""
        wm = self.occupied()
        g = self.grid
        cx = g.cell_x + self.sx
        cy = g.cell_y + self.sy
        occ = wm.occupied & (np.hypot(cx - x, cy - y) <= nc.get("obstacle_range", 4.0))
        occ &= np.hypot(cx - goal[0], cy - goal[1]) > nc.get("goal_clear_radius", 0.4)
        return WorldGrid(occ, wm.resolution, wm.origin)

    # ------------------------------------------------------------------ drawing
    def ego_view(self, x, y, yaw, size, robot_radius, side_m=None, extra=None):
        """Egocentric map image (size x size): robot low in the view facing up, FOV lines, range ring, robot.
        extra(img, to_px): optional drawing callback, to_px(world xy array) -> pixel coords."""
        a, g = self.args, self.grid
        side_m = side_m or 2 * a.range + 0.5
        k = size / side_m
        eu, ev = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
        right, fwd = (eu - size / 2) / k, (size * 0.8 - ev) / k
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = x - self.sx, y - self.sy
        ix, iy, ok = g.world_to_cell(px + fwd * c + right * s, py + fwd * s - right * c)
        st = g.state()
        cell = st[np.clip(iy, 0, g.ny - 1), np.clip(ix, 0, g.nx - 1)]
        img = np.full((size, size, 3), COLORS[UNKNOWN], np.uint8)
        for key, col in COLORS.items():
            img[ok & (cell == key)] = col
        rc = (size // 2, int(size * 0.8))

        def to_px(xy):
            d = np.asarray(xy, float) - (x, y)
            f_, r_ = d[..., 0] * c + d[..., 1] * s, d[..., 0] * s - d[..., 1] * c
            return np.stack([rc[0] + r_ * k, rc[1] - f_ * k], -1).astype(np.int32)

        if extra is not None:
            extra(img, to_px)
        for sd in (-1, 1):
            ang = sd * self.fov / 2
            cv2.line(img, rc, (int(rc[0] + math.sin(ang) * a.range * k), int(rc[1] - math.cos(ang) * a.range * k)),
                     (60, 170, 60), 1, cv2.LINE_AA)
        cv2.circle(img, rc, int(a.range * k), (170, 170, 170), 1, cv2.LINE_AA)
        cv2.circle(img, rc, max(2, int(robot_radius * k)), (200, 120, 0), 2, cv2.LINE_AA)
        cv2.arrowedLine(img, rc, (rc[0], int(rc[1] - 1.6 * robot_radius * k)), (200, 120, 0), 2, cv2.LINE_AA,
                        tipLength=0.4)
        return img

    def seg_view(self, fr=None, r=None):
        """Camera image with the used classes coloured and the boulders / flags outlined (cyan), BGR."""
        from sam2_seg import CLASS_NAMES
        if fr is None:
            fr, r, _ = self.last
        lab = r["labels"]
        used = np.isin(lab, [CLASS_NAMES.index(n) for n in self.occ_c + self.free_c + self.rock_c]) & fr.mask
        cam = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR)
        mix = cv2.addWeighted(cam, 0.4, SEG_COLORS[lab], 0.6, 0)
        cam[used] = mix[used]
        cnt, _ = cv2.findContours(fr.objects.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cam, cnt, -1, (255, 255, 0), 2)
        return cam

    def legend(self):
        occ = ", ".join(self.occ_c + self.rock_c) or "-"
        cols = ", ".join(f"{COLOR_NAMES[n]} {n}" for n in self.occ_c + self.free_c + self.rock_c)
        return (f"occupied (black): {occ}   free (white): {', '.join(self.free_c) or '-'}   grey = unknown",
                f"camera: {cols}, cyan outline = boulder / flag")
