"""Heightmap-based terrain sampling for the Mars habitat scene.

The Habitat scene's collision/render geometry doesn't expose a queryable
ground height, so the rover's y-coordinate (and spawn clearance) is instead
sampled from a grayscale heightmap PNG baked out alongside the scene: pixel
intensity -> world height, image (u, v) <-> world (x, z) via a fixed extent
(SIZE_X x SIZE_Z) and an axis mapping (flip/swap).

Two mapping stages, kept separate on purpose: `HeightmapGrid` encodes how
the *image* axes line up with world (x, z) (its own flip/swap), and `Terrain`
wraps it with a second, independent flip/swap for how the *scene* frame
relates to world (x, z). Callers that need the scene-space rover pose to land
on the right texel configure both stages independently rather than folding
them into one set of flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Mars scene ground-plane extent (metres) and the heightmap's encoded height range.
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566


def bilinear_sample(grid: np.ndarray, px: float, py: float) -> float:
    """Bilinearly sample a 2D grid at fractional pixel coords (px, py)."""
    h, w = grid.shape
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x0 = min(max(x0, 0), w - 1)
    y0 = min(max(y0, 0), h - 1)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = float(px - x0)
    dy = float(py - y0)
    h00 = float(grid[y0, x0])
    h10 = float(grid[y0, x1])
    h01 = float(grid[y1, x0])
    h11 = float(grid[y1, x1])
    h0 = h00 * (1.0 - dx) + h10 * dx
    h1 = h01 * (1.0 - dx) + h11 * dx
    return float(h0 * (1.0 - dy) + h1 * dy)


class HeightmapGrid:
    """Loads a heightmap PNG and samples it at a world (x, z), mapped to the
    image's own (u, v) axes via flip/swap flags."""

    def __init__(
        self,
        heightmap_path: Path,
        *,
        size_x: float = SIZE_X,
        size_z: float = SIZE_Z,
        size_y: float = SIZE_Y,
        flip_x: bool = False,
        flip_z: bool = True,
        swap_xz: bool = False,
    ):
        heightmap_path = Path(heightmap_path)
        if not heightmap_path.exists():
            raise FileNotFoundError(f"heightmap not found: {heightmap_path}")

        self.size_x = float(size_x)
        self.size_z = float(size_z)
        self.size_y = float(size_y)
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)

        arr = self._load_array(heightmap_path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
        y = arr * self.size_y
        y = y - float(np.mean(y))
        self._height = y
        self._h, self._w = self._height.shape

    @staticmethod
    def _load_array(heightmap_path: Path) -> np.ndarray:
        """Loads the raw heightmap pixels. Most heightmaps (PNG, 8/16/32-bit
        TIFF) decode fine via PIL; GeoTIFF DEMs with 64-bit float samples
        aren't supported by Pillow's TIFF codec, so those fall back to
        rasterio."""
        try:
            return np.asarray(Image.open(heightmap_path))
        except Exception:
            if heightmap_path.suffix.lower() not in (".tif", ".tiff"):
                raise
            try:
                import rasterio
            except ImportError as exc:
                raise ImportError(
                    f"PIL could not decode {heightmap_path} and rasterio is not "
                    "installed; install rasterio to load this GeoTIFF heightmap"
                ) from exc
            with rasterio.open(heightmap_path) as src:
                return src.read(1)

    def _to_uv(self, x: float, z: float) -> Tuple[float, float]:
        if self.swap_xz:
            x, z = z, x
        u = (x + self.size_x / 2.0) / self.size_x
        v = (z + self.size_z / 2.0) / self.size_z
        if self.flip_x:
            u = 1.0 - u
        if self.flip_z:
            v = 1.0 - v
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    def __call__(self, x: float, z: float) -> float:
        u, v = self._to_uv(x, z)
        px = u * (self._w - 1)
        py = v * (self._h - 1)
        return bilinear_sample(self._height, px, py)


class Terrain:
    """Wraps a `HeightmapGrid` with a second, scene-level axis mapping
    (its own flip/swap) between the rover's world (x, z) and the grid's."""

    def __init__(
        self,
        grid: HeightmapGrid,
        *,
        flip_x: bool = False,
        flip_z: bool = False,
        swap_xz: bool = False,
    ):
        self._grid = grid
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)

    def _map(self, x: float, z: float) -> Tuple[float, float]:
        xx, zz = float(x), float(z)
        if self.swap_xz:
            xx, zz = zz, xx
        if self.flip_x:
            xx = -xx
        if self.flip_z:
            zz = -zz
        return xx, zz

    def __call__(self, x: float, z: float) -> float:
        xx, zz = self._map(x, z)
        return float(self._grid(xx, zz))

    def local_height_max(
        self, x: float, z: float, radius: float, samples: int = 5
    ) -> float:
        """Max terrain height within `radius` of (x, z); used to keep a spawn
        point (or object marker) clear of nearby bumps rather than sinking
        into them when sampled at a single point."""
        radius = max(float(radius), 0.0)
        samples = max(int(samples), 1)
        if radius <= 1e-6 or samples == 1:
            return float(self(x, z))
        vals = []
        for dx in np.linspace(-radius, radius, samples):
            for dz in np.linspace(-radius, radius, samples):
                if dx * dx + dz * dz <= radius * radius + 1e-8:
                    vals.append(float(self(float(x) + float(dx), float(z) + float(dz))))
        return float(max(vals)) if vals else float(self(x, z))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m sam_vla.env.terrain <heightmap.png|.tif> [x z]")
        raise SystemExit(1)
    grid = HeightmapGrid(Path(sys.argv[1]))
    terrain = Terrain(grid)
    x = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    print(f"height at ({x}, {z}) = {terrain(x, z):.4f}")
    print(f"local max within 0.8m = {terrain.local_height_max(x, z, 0.8):.4f}")
