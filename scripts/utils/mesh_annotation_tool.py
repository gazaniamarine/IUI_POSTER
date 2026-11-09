"""Interactive tkinter tool (next.md Pipeline Step 1) for tracing object
outlines on the top-down terrain texture: click around a rock/crater's
boundary, "Finish object" builds a mesh from the real terrain clipped to that
polygon (each vertex sampled from the actual heightmap, so it always sits on
the ground -- unlike the legacy `--mesh-mode convex_hull` fallback). Writes
each object as its own OBJ plus a mesh_id -> category JSON registry, and
autosaves in-progress work to <out-dir>/session.json.

Usage:
    /home/gpu/miniconda3/envs/habitat/bin/python scripts/utils/mesh_annotation_tool.py \\
        --out-dir annotations/mesh_segmentation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError

import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sam_vla.env.terrain import HeightmapGrid  # noqa: E402

DEFAULT_HEIGHTMAP = HERE / "marsyard2022_terrain_hm.png"
DEFAULT_TEXTURE = HERE / "marsyard2022_terrain_texture.png"

CATEGORIES = ["small_rock", "big_rock", "bedrock", "hole_in_ground"]
CATEGORY_COLORS = {
    "small_rock": "#e0813e",
    "big_rock": "#b23a2f",
    "bedrock": "#8a8a70",
    "hole_in_ground": "#2f5c8a",
}
FALLBACK_COLORS = ["#7a4fa3", "#3f9e6e", "#c2a83e", "#4f7fa3", "#a34f7f"]


# --------------------------------------------------------------------------
# Terrain / geometry helpers
# --------------------------------------------------------------------------


def color_for_category(category: str) -> str:
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    idx = sum(map(ord, category)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def _sample_grid_bilinear(
    arr: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    size_x: float,
    size_z: float,
    flip_x: bool,
    flip_z: bool,
    swap_xz: bool,
) -> np.ndarray:
    """Vectorized bilinear sample of a (h, w) source array at world (xs, zs)
    (arrays of matching shape), using the exact same world->uv convention as
    HeightmapGrid._to_uv/bilinear_sample."""
    xx, zz = (zs, xs) if swap_xz else (xs, zs)
    u = (xx + size_x / 2.0) / size_x
    v = (zz + size_z / 2.0) / size_z
    if flip_x:
        u = 1.0 - u
    if flip_z:
        v = 1.0 - v
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    h, w = arr.shape[0], arr.shape[1]
    px = u * (w - 1)
    py = v * (h - 1)
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    dx = px - x0
    dy = py - y0
    v00, v10 = arr[y0, x0], arr[y0, x1]
    v01, v11 = arr[y1, x0], arr[y1, x1]
    top = v00 * (1 - dx) + v10 * dx
    bot = v01 * (1 - dx) + v11 * dx
    return top * (1 - dy) + bot * dy


def world_to_uv(
    x: float,
    z: float,
    size_x: float,
    size_z: float,
    flip_x: bool,
    flip_z: bool,
    swap_xz: bool,
):
    """Mirrors HeightmapGrid._to_uv exactly -- world (x, z) -> texture (u, v)
    in [0, 1], for placing markers on the texture panel."""
    if swap_xz:
        x, z = z, x
    u = (x + size_x / 2.0) / size_x
    v = (z + size_z / 2.0) / size_z
    if flip_x:
        u = 1.0 - u
    if flip_z:
        v = 1.0 - v
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def uv_to_world(
    u: float,
    v: float,
    size_x: float,
    size_z: float,
    flip_x: bool,
    flip_z: bool,
    swap_xz: bool,
):
    """Inverse of world_to_uv -- a click on the texture panel back to world
    (x, z) (height is then looked up separately from the heightmap)."""
    if flip_x:
        u = 1.0 - u
    if flip_z:
        v = 1.0 - v
    x = u * size_x - size_x / 2.0
    z = v * size_z - size_z / 2.0
    if swap_xz:
        x, z = z, x
    return float(x), float(z)


def points_in_polygon(
    px: np.ndarray, pz: np.ndarray, poly_xz: np.ndarray
) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test. `px`/`pz` are arrays of
    matching shape, `poly_xz` is (K, 2). Standard crossing-number rule,
    looped over the (small, K <= a few dozen) polygon edges."""
    poly_xz = np.asarray(poly_xz, dtype=np.float64)
    x = np.asarray(px, dtype=np.float64)
    z = np.asarray(pz, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    x1, z1 = poly_xz[-1]
    for x2, z2 in poly_xz:
        crosses = (z1 > z) != (z2 > z)
        x_at_z = (x2 - x1) * (z - z1) / (z2 - z1 + 1e-300) + x1
        inside ^= crosses & (x < x_at_z)
        x1, z1 = x2, z2
    return inside


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip())
    return name or "object"


# --------------------------------------------------------------------------
# Mesh construction
# --------------------------------------------------------------------------


def compute_tight_boundary_mesh(
    points: np.ndarray,
    grid: HeightmapGrid,
    uv_params: dict,
    spacing: float,
    max_grid_res: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Connects ALL of `points` (in click order) into a closed polygon in
    the xz plane -- the "tight boundary" -- then builds a mesh from the
    real terrain clipped to that polygon: a regular (x, z) grid is sampled
    at `spacing` over the polygon's bounding box, each vertex's height comes
    from the heightmap at that exact (x, z), and only quads whose 4 corners
    fall inside the polygon are kept (split into 2 triangles each).

    Unlike a convex hull of a few 3D points, this can never poke above or
    sink below the actual ground, since every vertex height is read from
    the terrain itself rather than interpolated between a handful of
    marked points.
    """
    poly_xz = points[:, [0, 2]]
    xmin, zmin = poly_xz.min(axis=0)
    xmax, zmax = poly_xz.max(axis=0)
    pad = max(float(spacing), 1e-3)
    xmin, xmax = xmin - pad, xmax + pad
    zmin, zmax = zmin - pad, zmax + pad

    nx = int(np.clip(round((xmax - xmin) / spacing) + 1, 2, max_grid_res))
    nz = int(np.clip(round((zmax - zmin) / spacing) + 1, 2, max_grid_res))
    xs = np.linspace(xmin, xmax, nx)
    zs = np.linspace(zmin, zmax, nz)
    Xg, Zg = np.meshgrid(xs, zs)  # each (nz, nx)
    Yg = _sample_grid_bilinear(grid._height, Xg, Zg, **uv_params)
    inside = points_in_polygon(Xg, Zg, poly_xz)

    row, col = np.meshgrid(np.arange(nz - 1), np.arange(nx - 1), indexing="ij")
    quad_inside = (
        inside[row, col]
        & inside[row, col + 1]
        & inside[row + 1, col + 1]
        & inside[row + 1, col]
    ).ravel()
    i0 = (row * nx + col).ravel()
    i1 = (row * nx + col + 1).ravel()
    i2 = ((row + 1) * nx + col + 1).ravel()
    i3 = ((row + 1) * nx + col).ravel()
    faces = np.concatenate(
        [
            np.stack([i0, i1, i2], axis=-1)[quad_inside],
            np.stack([i0, i2, i3], axis=-1)[quad_inside],
        ],
        axis=0,
    )
    if len(faces) == 0:
        raise ValueError(
            "Boundary too small to build a terrain mesh at this resolution -- "
            "mark a larger area or lower --mesh-spacing."
        )

    verts_full = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=-1)
    used = np.unique(faces)
    remap = -np.ones(len(verts_full), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts_full[used], remap[faces]


def compute_convex_hull_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy `--mesh-mode convex_hull` behavior: convex hull of `points`.
    Falls back to a thin extrusion along the point cloud's own normal (found
    via SVD, not assumed to be world-up) if the points are (near-)coplanar
    or collinear, since a valid 3D hull needs actual volume. Kept for
    comparison with the default tight_boundary mode -- prefer that one, this
    hull can dip below or float above the real terrain on anything but flat
    ground."""
    try:
        hull = ConvexHull(points)
        return points, hull.simplices
    except QhullError:
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]  # least-variance direction
        eps = 0.01
        thick = np.vstack([points + eps * normal, points - eps * normal])
        hull = ConvexHull(thick)
        return thick, hull.simplices


# --------------------------------------------------------------------------
# Session / registry persistence
# --------------------------------------------------------------------------


@dataclass
class AnnotatedObject:
    mesh_id: int
    name: str
    category: str
    points: list  # raw clicked points [[x,y,z], ...], in click order
    hull_points: list  # final mesh vertices (terrain patch, or legacy hull)
    hull_faces: list  # [[i,j,k], ...] indices into hull_points
    obj_path: str

    def to_json(self):
        return {
            "mesh_id": self.mesh_id,
            "name": self.name,
            "category": self.category,
            "points": self.points,
            "hull_points": self.hull_points,
            "hull_faces": self.hull_faces,
            "obj_path": self.obj_path,
        }

    @staticmethod
    def from_json(d):
        return AnnotatedObject(
            mesh_id=d["mesh_id"],
            name=d["name"],
            category=d["category"],
            points=d["points"],
            hull_points=d["hull_points"],
            hull_faces=d["hull_faces"],
            obj_path=d["obj_path"],
        )


class SessionStore:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.subobj_dir = out_dir / "subobjects"
        self.session_path = out_dir / "session.json"
        self.registry_path = out_dir / "mesh_id_map.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.subobj_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[list[AnnotatedObject], int, list, str]:
        if not self.session_path.exists():
            return [], 1000, [], ""
        data = json.loads(self.session_path.read_text())
        objects = [AnnotatedObject.from_json(d) for d in data.get("objects", [])]
        next_id = data.get("next_mesh_id", 1000)
        pending_points = data.get("pending_points", [])
        last_category = data.get("last_category", "")
        return objects, next_id, pending_points, last_category

    def save(
        self,
        objects: list[AnnotatedObject],
        next_mesh_id: int,
        pending_points: list,
        last_category: str = "",
    ):
        self.session_path.write_text(
            json.dumps(
                {
                    "next_mesh_id": next_mesh_id,
                    "pending_points": pending_points,
                    "objects": [o.to_json() for o in objects],
                    "last_category": last_category,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        self.registry_path.write_text(
            json.dumps(
                {
                    "mesh_id_map": {
                        str(o.mesh_id): {"category": o.category, "name": o.name}
                        for o in objects
                    }
                },
                indent=2,
            )
        )

    def write_hull_obj(self, obj: AnnotatedObject):
        path = self.subobj_dir / Path(obj.obj_path).name
        with open(path, "w") as f:
            f.write(f"o mesh_{obj.mesh_id}_{obj.name}\n")
            for p in obj.hull_points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for tri in obj.hull_faces:
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

    def delete_hull_obj(self, obj: AnnotatedObject):
        path = self.subobj_dir / Path(obj.obj_path).name
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


class MeshAnnotatorApp:
    def __init__(
        self,
        root: tk.Tk,
        heightmap_path: Path,
        out_dir: Path,
        texture_path: Path,
        size_x: float = 50.0,
        size_z: float = 50.0,
        size_y: float = 4.820803273566,
        flip_x: bool = False,
        flip_z: bool = True,
        swap_xz: bool = False,
        mesh_mode: str = "tight_boundary",
        mesh_spacing: float = 0.03,
        mesh_max_res: int = 300,
    ):
        self.root = root
        self.store = SessionStore(out_dir)
        self.uv_params = dict(
            size_x=size_x, size_z=size_z, flip_x=flip_x, flip_z=flip_z, swap_xz=swap_xz
        )
        self.terrain_grid = HeightmapGrid(
            heightmap_path,
            size_x=size_x,
            size_z=size_z,
            size_y=size_y,
            flip_x=flip_x,
            flip_z=flip_z,
            swap_xz=swap_xz,
        )
        self.mesh_mode = mesh_mode
        self.mesh_spacing = mesh_spacing
        self.mesh_max_res = mesh_max_res

        self.objects, self.next_mesh_id, pending, last_category = self.store.load()
        self.current_points: list[list[float]] = [list(p) for p in pending]
        self.last_category = last_category or CATEGORIES[0]

        # Editing an already-finished object's points: the object is pulled
        # out of self.objects (so it renders as the live in-progress outline
        # instead of the finished one) and its original index/data are kept
        # so "Cancel" can restore it unchanged.
        self._editing_orig: Optional[AnnotatedObject] = None
        self._editing_orig_idx: Optional[int] = None
        self._dragging_idx: Optional[int] = None
        self._highlighted_mesh_id: Optional[int] = None

        # Full-resolution texture image is the only picking surface -- no 3D
        # view. Clicking marks a point via uv_to_world + a direct heightmap
        # height lookup at that (x, z).
        self.tex_img = Image.open(texture_path).convert("RGB")
        self.tex_center = None  # (px, py) in source-image pixels; set on first layout
        self.tex_zoom = None  # source pixels per canvas pixel
        self.tex_crop_box = (
            None  # last-rendered (left, top, right, bottom) in source pixels
        )
        self._tex_photo = None  # keep a reference so Tk doesn't GC the PhotoImage
        self._tex_pan_xy = None

        self.pick_mode = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_listbox()
        self._redraw()
        if self.objects or self.current_points:
            self._status(
                f"Resumed session: {len(self.objects)} object(s), "
                f"{len(self.current_points)} pending point(s)."
            )

    def _build_ui(self):
        self.root.title("Freaky Annotations")
        self.root.geometry("1400x900")

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        texture_frame = ttk.Frame(paned)
        control_frame = ttk.Frame(paned, width=320)
        paned.add(texture_frame, weight=3)
        paned.add(control_frame, weight=2)

        ttk.Label(
            texture_frame, text="Texture map - click to mark boundary points"
        ).pack(anchor="w", padx=4, pady=(4, 0))
        self.tex_canvas = tk.Canvas(texture_frame, bg="#222", highlightthickness=0)
        self.tex_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.tex_canvas.bind("<Configure>", self._on_tex_configure)
        self.tex_canvas.bind("<Button-1>", self._on_tex_press)
        self.tex_canvas.bind("<B1-Motion>", self._on_tex_drag)
        self.tex_canvas.bind("<ButtonRelease-1>", self._on_tex_release)
        self.tex_canvas.bind("<Double-Button-1>", self._on_tex_double_click)
        self.tex_canvas.bind("<ButtonPress-3>", self._on_tex_pan_press)
        self.tex_canvas.bind("<B3-Motion>", self._on_tex_pan_move)
        self.tex_canvas.bind("<MouseWheel>", self._on_tex_scroll)  # Windows/Mac
        self.tex_canvas.bind("<Button-4>", self._on_tex_scroll)  # Linux wheel up
        self.tex_canvas.bind("<Button-5>", self._on_tex_scroll)  # Linux wheel down

        # -- controls --
        mode_desc = f"Mesh mode: {self.mesh_mode}" + (
            f" (spacing={self.mesh_spacing}m)"
            if self.mesh_mode == "tight_boundary"
            else ""
        )
        ttk.Label(control_frame, text=mode_desc, foreground="#345").pack(
            anchor="w", padx=8, pady=(10, 2)
        )
        ttk.Checkbutton(
            control_frame,
            text="Pick points (click texture panel to add)",
            variable=self.pick_mode,
        ).pack(anchor="w", padx=8, pady=(4, 2))
        ttk.Label(
            control_frame,
            text="Scroll = zoom, right-drag = pan. Click points in order around "
            "an object's outline (drag a point to move it, double-click a "
            "point to delete it), then Finish.",
            foreground="#666",
            wraplength=300,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        self.point_count_label = ttk.Label(
            control_frame, text="Current object: 0 points"
        )
        self.point_count_label.pack(anchor="w", padx=8, pady=(0, 4))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(btn_frame, text="Undo last point", command=self.undo_point).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            btn_frame, text="Cancel current object", command=self.cancel_current
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            btn_frame, text="Finish object -> mesh + label", command=self.finish_object
        ).pack(fill=tk.X, pady=(2, 10))

        ttk.Separator(control_frame).pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(
            control_frame, text="Finished objects (double-click to highlight):"
        ).pack(anchor="w", padx=8)
        list_frame = ttk.Frame(control_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.listbox = tk.Listbox(list_frame)
        self.listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.listbox.bind("<Double-Button-1>", self._on_listbox_double_click)
        scrollbar = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        ttk.Button(
            control_frame,
            text="Edit selected object's points",
            command=self.edit_selected,
        ).pack(fill=tk.X, padx=8, pady=(2, 2))
        ttk.Button(
            control_frame, text="Delete selected object", command=self.delete_selected
        ).pack(fill=tk.X, padx=8, pady=(2, 10))

        ttk.Separator(control_frame).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(
            control_frame,
            text="Recompute all meshes (current mode)",
            command=self.recompute_all,
        ).pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(control_frame, text="Save session now", command=self._save).pack(
            fill=tk.X, padx=8, pady=2
        )

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            control_frame,
            textvariable=self.status_var,
            wraplength=300,
            foreground="#345",
        ).pack(anchor="w", padx=8, pady=(10, 8))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _status(self, msg: str):
        self.status_var.set(msg)

    # -- rendering ---------------------------------------------------------

    def _redraw(self):
        self._redraw_texture_panel()
        self.point_count_label.config(
            text=f"Current object: {len(self.current_points)} points"
        )

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for obj in self.objects:
            self.listbox.insert(tk.END, f"[{obj.mesh_id}] {obj.name}  ({obj.category})")

    # -- 2D texture panel ---------------------------------------------------

    def _on_tex_configure(self, _event):
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        if self.tex_center is None:
            # First layout: fit the whole image to the canvas.
            tw, th = self.tex_img.size
            self.tex_center = [tw / 2.0, th / 2.0]
            self.tex_zoom = max(tw / cw, th / ch)
        self._redraw_texture_panel()

    def _tex_view_box(self):
        """Current (left, top, right, bottom) crop box in source pixels,
        clamped to stay inside the image."""
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        tw, th = self.tex_img.size
        half_w, half_h = cw * self.tex_zoom / 2.0, ch * self.tex_zoom / 2.0
        cx = (
            min(max(self.tex_center[0], half_w), tw - half_w)
            if tw > 2 * half_w
            else tw / 2.0
        )
        cy = (
            min(max(self.tex_center[1], half_h), th - half_h)
            if th > 2 * half_h
            else th / 2.0
        )
        self.tex_center = [cx, cy]
        left, top = cx - half_w, cy - half_h
        right, bottom = cx + half_w, cy + half_h
        return left, top, right, bottom

    def _redraw_texture_panel(self):
        if self.tex_center is None:
            return
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        left, top, right, bottom = self._tex_view_box()
        crop = self.tex_img.crop((int(left), int(top), int(right), int(bottom)))
        crop = crop.resize((cw, ch))
        self.tex_crop_box = (left, top, right, bottom)
        self._tex_photo = ImageTk.PhotoImage(crop)
        self.tex_canvas.delete("all")
        self.tex_canvas.create_image(0, 0, anchor="nw", image=self._tex_photo)

        for obj in self.objects:
            highlighted = obj.mesh_id == self._highlighted_mesh_id
            color = "#ffff00" if highlighted else color_for_category(obj.category)
            pts2d = [self._world_to_canvas_xy(x, z) for x, _y, z in obj.points]
            if len(pts2d) >= 2:
                flat = [c for xy in pts2d for c in xy]
                self.tex_canvas.create_polygon(
                    flat, outline=color, fill="", width=4 if highlighted else 2
                )
            for cx, cy in pts2d:
                self.tex_canvas.create_oval(
                    cx - 3, cy - 3, cx + 3, cy + 3, fill=color, outline=""
                )

        pts2d = [self._world_to_canvas_xy(x, z) for x, _y, z in self.current_points]
        if len(pts2d) >= 2:
            flat = [c for xy in pts2d for c in xy]
            self.tex_canvas.create_polygon(
                flat, outline="red", fill="", width=2, dash=(4, 2)
            )
        for cx, cy in pts2d:
            self.tex_canvas.create_oval(
                cx - 4, cy - 4, cx + 4, cy + 4, fill="red", outline="white"
            )

    def _on_tex_scroll(self, event):
        factor = (
            0.9
            if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4)
            else 1.1
        )
        tw, th = self.tex_img.size
        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        min_zoom = 0.02
        max_zoom = max(tw / cw, th / ch)
        self.tex_zoom = float(np.clip(self.tex_zoom * factor, min_zoom, max_zoom))
        self._redraw_texture_panel()

    def _on_tex_pan_press(self, event):
        self._tex_pan_xy = (event.x, event.y)

    def _on_tex_pan_move(self, event):
        if self._tex_pan_xy is None:
            return
        dx = (event.x - self._tex_pan_xy[0]) * self.tex_zoom
        dy = (event.y - self._tex_pan_xy[1]) * self.tex_zoom
        self.tex_center[0] -= dx
        self.tex_center[1] -= dy
        self._tex_pan_xy = (event.x, event.y)
        self._redraw_texture_panel()

    def _world_to_canvas_xy(self, x: float, z: float) -> tuple[float, float]:
        left, top, _right, _bottom = self.tex_crop_box
        u, v = world_to_uv(x, z, **self.uv_params)
        tw, th = self.tex_img.size
        px, py = u * (tw - 1), v * (th - 1)
        return (px - left) / self.tex_zoom, (py - top) / self.tex_zoom

    def _canvas_to_world(self, event) -> Optional[tuple[float, float]]:
        left, top, _right, _bottom = self.tex_crop_box
        tw, th = self.tex_img.size
        px = left + event.x * self.tex_zoom
        py = top + event.y * self.tex_zoom
        u, v = px / (tw - 1), py / (th - 1)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        return uv_to_world(u, v, **self.uv_params)

    def _hit_test_current_point(
        self, event, threshold_px: float = 10.0
    ) -> Optional[int]:
        """Index of the current-object point whose on-screen marker is
        closest to the click, if within `threshold_px` canvas pixels."""
        best_idx, best_dist = None, threshold_px
        for i, (x, _y, z) in enumerate(self.current_points):
            cx, cy = self._world_to_canvas_xy(x, z)
            dist = ((cx - event.x) ** 2 + (cy - event.y) ** 2) ** 0.5
            if dist < best_dist:
                best_idx, best_dist = i, dist
        return best_idx

    def _on_tex_press(self, event):
        if not self.pick_mode.get() or self.tex_crop_box is None:
            return
        hit = self._hit_test_current_point(event)
        if hit is not None:
            # Starting a drag on an existing marker -- don't also add a
            # new point.
            self._dragging_idx = hit
            return
        self._dragging_idx = None
        world = self._canvas_to_world(event)
        if world is None:
            return
        x, z = world
        y = float(self.terrain_grid(x, z))
        self.current_points.append([x, y, z])
        self._redraw()
        self._status(f"Added point at ({x:.2f}, {y:.2f}, {z:.2f}).")

    def _on_tex_drag(self, event):
        if self._dragging_idx is None or not self.pick_mode.get():
            return
        world = self._canvas_to_world(event)
        if world is None:
            return
        x, z = world
        y = float(self.terrain_grid(x, z))
        self.current_points[self._dragging_idx] = [x, y, z]
        self._redraw()

    def _on_tex_release(self, _event):
        if self._dragging_idx is not None:
            x, y, z = self.current_points[self._dragging_idx]
            self._status(f"Moved point to ({x:.2f}, {y:.2f}, {z:.2f}).")
        self._dragging_idx = None

    def _on_tex_double_click(self, event):
        if not self.pick_mode.get() or self.tex_crop_box is None:
            return
        hit = self._hit_test_current_point(event)
        if hit is None:
            return
        self.current_points.pop(hit)
        self._dragging_idx = None
        self._redraw()
        self._status("Deleted point.")

    def _on_listbox_double_click(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        self._highlight_object(self.objects[sel[0]])

    def _highlight_object(self, obj: AnnotatedObject):
        pts = np.array(obj.points)
        xs, zs = pts[:, 0], pts[:, 2]
        cx_world = (xs.min() + xs.max()) / 2.0
        cz_world = (zs.min() + zs.max()) / 2.0
        span = max(xs.max() - xs.min(), zs.max() - zs.min(), 1e-3) * 3.0

        tw, th = self.tex_img.size
        u, v = world_to_uv(cx_world, cz_world, **self.uv_params)
        self.tex_center = [u * (tw - 1), v * (th - 1)]

        cw, ch = self.tex_canvas.winfo_width(), self.tex_canvas.winfo_height()
        u0, v0 = world_to_uv(cx_world - span / 2, cz_world - span / 2, **self.uv_params)
        u1, v1 = world_to_uv(cx_world + span / 2, cz_world + span / 2, **self.uv_params)
        span_px = max(abs(u1 - u0) * tw, abs(v1 - v0) * th)
        min_zoom = 0.02
        max_zoom = max(tw / max(cw, 1), th / max(ch, 1))
        self.tex_zoom = float(
            np.clip(span_px / max(min(cw, ch), 1), min_zoom, max_zoom)
        )

        self._highlighted_mesh_id = obj.mesh_id
        self._redraw_texture_panel()
        self._status(f"Highlighting mesh_id {obj.mesh_id} '{obj.name}'.")
        self.root.after(1500, self._clear_highlight)

    def _clear_highlight(self):
        self._highlighted_mesh_id = None
        self._redraw_texture_panel()

    # -- mesh building -------------------------------------------------------

    def _build_mesh(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mesh_mode == "tight_boundary":
            return compute_tight_boundary_mesh(
                points,
                self.terrain_grid,
                self.uv_params,
                spacing=self.mesh_spacing,
                max_grid_res=self.mesh_max_res,
            )
        return compute_convex_hull_mesh(points)

    # -- object lifecycle --------------------------------------------------

    def undo_point(self):
        if self.current_points:
            self.current_points.pop()
            self._redraw()

    def _restore_editing_if_any(self):
        """If an existing object was pulled out for editing, put it back
        into self.objects unchanged (used by cancel and by starting a new
        edit/close while one is already in progress)."""
        if self._editing_orig is None:
            return
        idx = (
            self._editing_orig_idx
            if self._editing_orig_idx is not None
            else len(self.objects)
        )
        idx = min(idx, len(self.objects))
        self.objects.insert(idx, self._editing_orig)
        self._editing_orig = None
        self._editing_orig_idx = None
        self._refresh_listbox()

    def edit_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        obj = self.objects[sel[0]]
        if self.current_points and not messagebox.askyesno(
            "Discard current points?",
            f"You have {len(self.current_points)} unfinished point(s). Editing "
            "another object will discard them. Continue?",
        ):
            return
        self._restore_editing_if_any()
        idx = next(i for i, o in enumerate(self.objects) if o is obj)
        del self.objects[idx]
        self._editing_orig = obj
        self._editing_orig_idx = idx
        self.current_points = [list(p) for p in obj.points]
        self.pick_mode.set(True)
        self._refresh_listbox()
        self._redraw()
        self._status(
            f"Editing mesh_id {obj.mesh_id} '{obj.name}': drag a point to move it, "
            "double-click a point to delete it, click empty space to add one, "
            "then Finish object to save."
        )

    def cancel_current(self):
        if self.current_points and not messagebox.askyesno(
            "Cancel object", f"Discard {len(self.current_points)} marked point(s)?"
        ):
            return
        self.current_points = []
        was_editing = self._editing_orig is not None
        self._restore_editing_if_any()
        self._redraw()
        if was_editing:
            self._status("Cancelled edit; object restored unchanged.")

    def finish_object(self):
        if len(self.current_points) < 3:
            messagebox.showwarning(
                "Not enough points",
                "Mark at least 3 points before finishing an object.",
            )
            return

        points = np.array(self.current_points)
        try:
            mesh_points, mesh_faces = self._build_mesh(points)
        except (QhullError, ValueError) as exc:
            messagebox.showerror("Mesh failed", f"Could not build mesh: {exc}")
            return

        editing = self._editing_orig
        default_category = editing.category if editing else self.last_category
        default_name = editing.name if editing else f"object_{self.next_mesh_id}"
        dialog = LabelDialog(
            self.root, default_name=default_name, default_category=default_category
        )
        self.root.wait_window(dialog.top)
        if dialog.result is None:
            return  # cancelled, keep points as-is
        category, name = dialog.result
        name = sanitize_name(name)
        self.last_category = category

        if editing:
            mesh_id = editing.mesh_id
            obj_path = f"subobjects/mesh_{mesh_id}_{name}.obj"
            if obj_path != editing.obj_path:
                self.store.delete_hull_obj(editing)
        else:
            mesh_id = self.next_mesh_id
            self.next_mesh_id += 1
            obj_path = f"subobjects/mesh_{mesh_id}_{name}.obj"

        obj = AnnotatedObject(
            mesh_id=mesh_id,
            name=name,
            category=category,
            points=[list(p) for p in self.current_points],
            hull_points=[list(p) for p in mesh_points],
            hull_faces=[list(map(int, f)) for f in mesh_faces],
            obj_path=obj_path,
        )
        self.store.write_hull_obj(obj)
        insert_idx = (
            self._editing_orig_idx
            if editing and self._editing_orig_idx is not None
            else len(self.objects)
        )
        insert_idx = min(insert_idx, len(self.objects))
        self.objects.insert(insert_idx, obj)
        self.current_points = []
        self._editing_orig = None
        self._editing_orig_idx = None
        self._refresh_listbox()
        self._redraw()
        self._save()
        verb = "Updated" if editing else "Saved"
        self._status(f"{verb} mesh_id {mesh_id} '{name}' ({category}) -> {obj_path}")

    def recompute_all(self):
        """Rebuilds every object's mesh from its already-recorded raw click
        points using the current mesh mode -- lets you fix objects that were
        annotated back when this tool still used convex hulls, without
        re-clicking anything."""
        if not self.objects:
            return
        if not messagebox.askyesno(
            "Recompute all",
            f"Recompute mesh geometry for all {len(self.objects)} object(s) "
            f"using mesh mode '{self.mesh_mode}'? Raw marked points are kept; "
            "only the exported mesh is regenerated.",
        ):
            return
        failed = []
        for obj in self.objects:
            points = np.array(obj.points)
            try:
                mesh_points, mesh_faces = self._build_mesh(points)
            except (QhullError, ValueError) as exc:
                failed.append((obj.name, str(exc)))
                continue
            obj.hull_points = [list(p) for p in mesh_points]
            obj.hull_faces = [list(map(int, f)) for f in mesh_faces]
            self.store.write_hull_obj(obj)
        self._redraw()
        self._save()
        n_ok = len(self.objects) - len(failed)
        msg = f"Recomputed {n_ok}/{len(self.objects)} object(s)."
        if failed:
            msg += " Failed: " + ", ".join(name for name, _ in failed)
        self._status(msg)

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        obj = self.objects[sel[0]]
        if not messagebox.askyesno(
            "Delete object", f"Delete '{obj.name}' (mesh_id {obj.mesh_id})?"
        ):
            return
        self.store.delete_hull_obj(obj)
        del self.objects[sel[0]]
        self._refresh_listbox()
        self._redraw()
        self._save()
        self._status(f"Deleted mesh_id {obj.mesh_id}.")

    def _save(self):
        self.store.save(
            self.objects, self.next_mesh_id, self.current_points, self.last_category
        )

    def _on_close(self):
        if self._editing_orig is not None:
            # An in-progress edit has no persisted identity of its own (only
            # finished objects + raw pending points are saved) -- discard the
            # edit rather than silently losing the original object on resume.
            self.current_points = []
            self._restore_editing_if_any()
        self._save()
        self.root.destroy()


class LabelDialog:
    """Modal category + name prompt used when finishing an object."""

    def __init__(
        self, parent, default_name: str, default_category: str = CATEGORIES[0]
    ):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("Label object")
        self.top.transient(parent)
        self.top.grab_set()

        ttk.Label(self.top, text="Category:").grid(
            row=0, column=0, sticky="w", padx=8, pady=(10, 2)
        )
        self.category_var = tk.StringVar(value=default_category)
        combo = ttk.Combobox(
            self.top, textvariable=self.category_var, values=CATEGORIES
        )
        combo.grid(row=1, column=0, sticky="ew", padx=8)

        ttk.Label(self.top, text="Name:").grid(
            row=2, column=0, sticky="w", padx=8, pady=(10, 2)
        )
        self.name_var = tk.StringVar(value=default_name)
        name_entry = ttk.Entry(self.top, textvariable=self.name_var)
        name_entry.grid(row=3, column=0, sticky="ew", padx=8)
        name_entry.select_range(0, tk.END)

        btn_frame = ttk.Frame(self.top)
        btn_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=12)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(
            side=tk.RIGHT, padx=4
        )
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side=tk.RIGHT, padx=4)

        self.top.columnconfigure(0, weight=1)
        combo.focus_set()
        self.top.bind("<Return>", lambda _e: self._ok())
        self.top.bind("<Escape>", lambda _e: self._cancel())

    def _ok(self):
        category = self.category_var.get().strip()
        name = self.name_var.get().strip()
        if not category or not name:
            messagebox.showwarning(
                "Missing info", "Both category and name are required.", parent=self.top
            )
            return
        self.result = (category, name)
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--heightmap", default=str(DEFAULT_HEIGHTMAP))
    ap.add_argument("--texture", default=str(DEFAULT_TEXTURE))
    ap.add_argument("--size-x", type=float, default=50.0)
    ap.add_argument("--size-z", type=float, default=50.0)
    ap.add_argument("--size-y", type=float, default=4.820803273566)
    ap.add_argument("--flip-x", action="store_true", default=False)
    ap.add_argument("--flip-z", action="store_true", default=True)
    ap.add_argument("--no-flip-z", dest="flip_z", action="store_false")
    ap.add_argument("--swap-xz", action="store_true", default=False)
    ap.add_argument(
        "--mesh-mode",
        choices=["tight_boundary", "convex_hull"],
        default="tight_boundary",
        help="tight_boundary (default): connect ALL marked points, in click "
        "order, into a polygon in the xz plane, then clip the real terrain "
        "to that polygon to build the mesh -- it always sits on the actual "
        "ground. convex_hull: legacy behavior, a 3D convex hull of the "
        "marked points (can dip below or float above the terrain).",
    )
    ap.add_argument(
        "--mesh-spacing",
        type=float,
        default=0.03,
        help="tight_boundary only: terrain sampling spacing (world units) "
        "for each object's mesh grid. Smaller = more detail, more triangles.",
    )
    ap.add_argument(
        "--mesh-max-res",
        type=int,
        default=300,
        help="tight_boundary only: safety cap on the sampling grid's "
        "resolution per axis.",
    )
    ap.add_argument("--out-dir", default="annotations/mesh_segmentation")
    args = ap.parse_args()

    heightmap = Path(args.heightmap)
    if not heightmap.exists():
        raise FileNotFoundError(f"heightmap not found: {heightmap}")
    texture = Path(args.texture)
    if not texture.exists():
        raise FileNotFoundError(
            f"texture not found: {texture} -- picking now happens on the "
            "texture panel only, so a texture image is required."
        )

    root = tk.Tk()
    MeshAnnotatorApp(
        root,
        heightmap,
        Path(args.out_dir),
        texture,
        size_x=args.size_x,
        size_z=args.size_z,
        size_y=args.size_y,
        flip_x=args.flip_x,
        flip_z=args.flip_z,
        swap_xz=args.swap_xz,
        mesh_mode=args.mesh_mode,
        mesh_spacing=args.mesh_spacing,
        mesh_max_res=args.mesh_max_res,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
