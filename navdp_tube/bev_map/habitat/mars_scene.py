"""The Mars yard GLB as the Habitat runner needs it: terrain height, obstacle footprints, goal flags.

Reads the .glb itself (no trimesh): every node's vertices in the world. Frames:
  glTF / Habitat (Y up):  (X, Y, Z)
  planner (z up, as in the MuJoCo sim and rock_layout.txt):  x = X, y = -Z, height = Y
The scene is loaded in Habitat through a stage config with up = +Y (stage_config), else Habitat treats the file as
Z-up and stands the yard on its side.
"""

import json
import math
import os
import struct

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

TERRAIN_NODE = "marsyard_terrain"
_DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def stage_config(glb, out_dir):
    """Write <name>.stage_config.json (up +Y, front -Z) next to the runs and return its path (Habitat's scene_id)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.splitext(os.path.basename(glb))[0] + ".stage_config.json")
    cfg = {"render_asset": os.path.abspath(glb), "up": [0, 1, 0], "front": [0, 0, -1], "origin": [0, 0, 0],
           "requires_lighting": False}
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    m[:3, :3] = _quat_mat(n.get("rotation", [0, 0, 0, 1])) * np.array(n.get("scale", [1, 1, 1]))
    m[:3, 3] = n.get("translation", [0, 0, 0])
    return m


def _read_accessor(j, buf, i):
    a = j["accessors"][i]
    bv = j["bufferViews"][a["bufferView"]]
    k = _NCOMP[a["type"]]
    dt = np.dtype(_DTYPES[a["componentType"]])
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or k * dt.itemsize
    if stride == k * dt.itemsize:
        return np.frombuffer(buf, dt, a["count"] * k, start).reshape(-1, k).astype(np.float64)
    rows = np.lib.stride_tricks.as_strided(np.frombuffer(buf, np.uint8, offset=start),
                                           (a["count"], k * dt.itemsize), (stride, 1))
    return rows.copy().view(dt).reshape(-1, k).astype(np.float64)


def read_glb(path):
    """-> {node name: (N, 3) world vertices in glTF axes}, {node name: world translation}."""
    with open(path, "rb") as f:
        magic, _, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a binary glTF")
        n, _ = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(n))
        n, _ = struct.unpack("<II", f.read(8))
        buf = f.read(n)

    def accessor(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        k = _NCOMP[a["type"]]
        dt = np.dtype(_DTYPES[a["componentType"]])
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or k * dt.itemsize
        if stride == k * dt.itemsize:
            return np.frombuffer(buf, dt, a["count"] * k, start).reshape(-1, k)
        rows = np.lib.stride_tricks.as_strided(np.frombuffer(buf, np.uint8, offset=start),
                                               (a["count"], k * dt.itemsize), (stride, 1))
        return rows.copy().view(dt).reshape(-1, k)

    verts, origin = {}, {}

    def walk(idx, parent):
        node = j["nodes"][idx]
        m = parent @ _node_matrix(node)
        name = node.get("name", f"node{idx}")
        origin[name] = m[:3, 3].copy()
        if "mesh" in node:
            pts = [accessor(p["attributes"]["POSITION"]).astype(np.float64)
                   for p in j["meshes"][node["mesh"]]["primitives"]]
            v = np.vstack(pts)
            verts[name] = v @ m[:3, :3].T + m[:3, 3]
        for c in node.get("children", []):
            walk(c, m)

    for root in j["scenes"][j.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return verts, origin


def write_glb_subset(src, keep, out, bake=False):
    """Write a .glb with only the top-level nodes named in `keep` (and their children), and only the meshes,
    materials, textures and buffer data they use. bake: one rock as an object to place anywhere: each kept mesh node
    (no children) gets its rotation and scale baked into its vertices, centred on their bounding box, and an
    identity transform (so the object's frame is the glTF frame at the rock's centre, whatever Habitat does)."""
    with open(src, "rb") as f:
        f.read(12)
        n, _ = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(n))
        n, _ = struct.unpack("<II", f.read(8))
        buf = f.read(n)
    new = {k: [] for k in ("nodes", "meshes", "accessors", "bufferViews", "materials", "textures", "images",
                           "samplers")}
    maps = {k: {} for k in new}
    blob = bytearray()

    def take(kind, i, fix=None):
        if i not in maps[kind]:
            item = json.loads(json.dumps(j[kind][i]))
            maps[kind][i] = len(new[kind])
            new[kind].append(item)
            if fix:
                fix(item)
        return maps[kind][i]

    def fix_view(v):
        data = buf[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
        blob.extend(b"\0" * (-len(blob) % 4))
        v["buffer"], v["byteOffset"] = 0, len(blob)
        blob.extend(data)

    def fix_accessor(a):
        if "bufferView" in a:
            a["bufferView"] = take("bufferViews", a["bufferView"], fix_view)

    def fix_texinfo(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                if "index" in v and k.lower().endswith("texture"):
                    v["index"] = take("textures", v["index"], fix_texture)
                fix_texinfo(v)

    def fix_texture(t):
        if "source" in t:
            t["source"] = take("images", t["source"], fix_image)
        if "sampler" in t:
            t["sampler"] = take("samplers", t["sampler"])

    def fix_image(im):
        if "bufferView" in im:
            im["bufferView"] = take("bufferViews", im["bufferView"], fix_view)

    def fix_mesh(m):
        for p in m["primitives"]:
            p["attributes"] = {k: take("accessors", v, fix_accessor) for k, v in p["attributes"].items()}
            if "indices" in p:
                p["indices"] = take("accessors", p["indices"], fix_accessor)
            if "material" in p:
                p["material"] = take("materials", p["material"], fix_texinfo)

    def fix_node(nd):
        if "mesh" in nd:
            nd["mesh"] = take("meshes", nd["mesh"], fix_mesh)
        nd["children"] = [take("nodes", c, fix_node) for c in nd.get("children", [])]
        if not nd["children"]:
            nd.pop("children")

    def baked(i):
        """Copy of node i (a mesh, no children) with its rotation and scale in its POSITION / NORMAL data."""
        nd = j["nodes"][i]
        m = _node_matrix({k: v for k, v in nd.items() if k != "translation"})[:3, :3]
        rot = m / np.linalg.norm(m, axis=0)
        mesh = json.loads(json.dumps(j["meshes"][nd["mesh"]]))
        pos = [_read_accessor(j, buf, p["attributes"]["POSITION"]) @ m.T for p in mesh["primitives"]]
        allp = np.vstack(pos)
        centre = (allp.min(0) + allp.max(0)) / 2
        for p, P in zip(mesh["primitives"], pos):
            attrs = {}
            for k, a in p["attributes"].items():
                if k in ("POSITION", "NORMAL"):
                    data = (P - centre) if k == "POSITION" else _read_accessor(j, buf, a) @ rot.T
                    data = np.ascontiguousarray(data, np.float32)
                    blob.extend(b"\0" * (-len(blob) % 4))
                    new["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes})
                    blob.extend(data.tobytes())
                    acc = {"bufferView": len(new["bufferViews"]) - 1, "componentType": 5126, "count": len(data),
                           "type": "VEC3"}
                    if k == "POSITION":
                        acc.update(min=data.min(0).tolist(), max=data.max(0).tolist())
                    new["accessors"].append(acc)
                    attrs[k] = len(new["accessors"]) - 1
                else:
                    attrs[k] = take("accessors", a, fix_accessor)
            p["attributes"] = attrs
            if "indices" in p:
                p["indices"] = take("accessors", p["indices"], fix_accessor)
            if "material" in p:
                p["material"] = take("materials", p["material"], fix_texinfo)
        new["meshes"].append(mesh)
        new["nodes"].append({"name": nd.get("name", f"node{i}"), "mesh": len(new["meshes"]) - 1})
        return len(new["nodes"]) - 1

    roots = []
    for i in j["scenes"][j.get("scene", 0)]["nodes"]:
        if j["nodes"][i].get("name") in keep:
            roots.append(baked(i) if bake else take("nodes", i, fix_node))
    blob.extend(b"\0" * (-len(blob) % 4))
    out_j = {"asset": j["asset"], "scene": 0, "scenes": [{"nodes": roots}], "buffers": [{"byteLength": len(blob)}]}
    out_j.update({k: v for k, v in new.items() if v})
    for k in ("extensionsUsed", "extensionsRequired"):
        if k in j:
            out_j[k] = j[k]
    js = json.dumps(out_j).encode()
    js += b" " * (-len(js) % 4)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))
    return out


def to_planner(v):
    """glTF (X, Y up, Z) -> planner (x, y, height)."""
    v = np.asarray(v, float)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


class Grid:
    """Square grid over the planner's x, y. Cell (iy, ix) covers x_min + ix*res .. +res."""

    def __init__(self, x_min, x_max, y_min, y_max, res):
        self.x_min, self.y_min, self.res = x_min, y_min, res
        self.nx = int(math.ceil((x_max - x_min) / res))
        self.ny = int(math.ceil((y_max - y_min) / res))

    def index(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x_min) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y_min) / self.res).astype(int), 0, self.ny - 1)
        return iy, ix

    def inside(self, x, y):
        return (self.x_min <= x < self.x_min + self.nx * self.res) and (self.y_min <= y < self.y_min + self.ny * self.res)


class MarsScene:
    """Terrain height, and the footprint of every other node (rocks, flags) between min_h and max_h above the
    terrain (a flag's cloth, high up, is not an obstacle; its pole is)."""

    def __init__(self, glb, height_res=0.10, obst_res=0.05, min_h=0.05, max_h=1.0):
        verts, origin = read_glb(glb)
        if TERRAIN_NODE not in verts:
            raise ValueError(f"no node named {TERRAIN_NODE!r} in {glb}")
        terrain = to_planner(verts.pop(TERRAIN_NODE))
        lo, hi = terrain[:, :2].min(0), terrain[:, :2].max(0)
        self.bounds = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

        # terrain height: mean of the vertices in each cell, empty cells from the nearest filled one
        self.hgrid = Grid(*self.bounds, height_res)
        iy, ix = self.hgrid.index(terrain[:, 0], terrain[:, 1])
        s = np.zeros((self.hgrid.ny, self.hgrid.nx))
        c = np.zeros_like(s)
        np.add.at(s, (iy, ix), terrain[:, 2])
        np.add.at(c, (iy, ix), 1)
        h = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        _, (ny_, nx_) = distance_transform_edt(np.isnan(h), return_indices=True)
        self.height = h[ny_, nx_]

        # obstacle footprints: convex hull of each node's vertices in the height band
        self.grid = Grid(*self.bounds, obst_res)
        self.min_h, self.max_h = min_h, max_h
        self.labels = np.full((self.grid.ny, self.grid.nx), -1, np.int32)
        self.names, self.origin = [], {}
        for name, v in verts.items():
            self.origin[name] = tuple(float(t) for t in to_planner(origin[name])[:2])
            self.set_footprint(name, to_planner(v))
        self.occ = self.labels >= 0

    def set_footprint(self, name, p):
        """(Re)place the footprint of `name` from its vertices p [N, 3] in the planner frame: the convex hull of those
        min_h..max_h above the terrain. A name seen before keeps its label id (its old cells are cleared). -> bool."""
        if name in self.names:
            self.labels[self.labels == self.names.index(name)] = -1
        above = p[:, 2] - self.height_at(p[:, 0], p[:, 1])
        ok = above.max() >= self.min_h                   # else all of it under the ground
        band = (above >= self.min_h) & (above <= self.max_h)
        # a flag pole is a cylinder with vertices only at its buried foot and its top: take the low ones
        p = p[band] if band.sum() >= 3 else p[above <= self.max_h]
        if ok and len(p) >= 3:
            res = self.grid.res
            px = np.stack([(p[:, 0] - self.grid.x_min) / res, (p[:, 1] - self.grid.y_min) / res], -1)
            hull = cv2.convexHull(np.round(px).astype(np.int32))
            mask = np.zeros(self.labels.shape, np.uint8)
            cv2.fillPoly(mask, [hull], 1)
            if name not in self.names:
                self.names.append(name)
            self.labels[mask > 0] = self.names.index(name)
        self.occ = self.labels >= 0
        return bool(ok and len(p) >= 3)

    def height_at(self, x, y):
        """Bilinear terrain height (m) at planner x, y (arrays fine)."""
        g = self.hgrid
        fx = np.clip((np.asarray(x, float) - g.x_min) / g.res - 0.5, 0, g.nx - 1.001)
        fy = np.clip((np.asarray(y, float) - g.y_min) / g.res - 0.5, 0, g.ny - 1.001)
        x0, y0 = fx.astype(int), fy.astype(int)
        dx, dy = fx - x0, fy - y0
        H = self.height
        return ((H[y0, x0] * (1 - dx) + H[y0, x0 + 1] * dx) * (1 - dy)
                + (H[y0 + 1, x0] * (1 - dx) + H[y0 + 1, x0 + 1] * dx) * dy)

    def distance_field(self, names=None, exclude=()):
        """Metres from each cell to the nearest footprint of `names` (default: all but `exclude`)."""
        ids = [i for i, n in enumerate(self.names) if (names is None or n in names) and n not in exclude]
        occ = np.isin(self.labels, ids)
        return distance_transform_edt(~occ) * self.grid.res

    def lookup(self, field, x, y):
        iy, ix = self.grid.index(x, y)
        return float(field[iy, ix])

    def start_behind(self, rock, goal_xy, gap, radius):
        """Start (x, y) on the far side of `rock` from the goal, on the goal-rock line: the rover's circle `gap` m
        from the rock's surface, so the rock stands right between the rover and the goal."""
        own = self.distance_field(names=(rock,))
        bx, by = self.origin[rock]
        d = math.hypot(bx - goal_xy[0], by - goal_xy[1])
        ux, uy = (bx - goal_xy[0]) / d, (by - goal_xy[1]) / d
        t = 0.0
        while self.lookup(own, bx + ux * t, by + uy * t) - radius < gap:
            t += 0.01
            if t > 50:
                raise ValueError(f"no free spot behind {rock}")
        return bx + ux * t, by + uy * t

    def nearest_name(self, x, y, radius):
        """Name of the footprint nearest to (x, y) within radius, else None."""
        iy, ix = self.grid.index(x, y)
        r = int(math.ceil(radius / self.grid.res))
        win = self.labels[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1]
        hit = win[win >= 0]
        return self.names[int(np.bincount(hit).argmax())] if hit.size else None
