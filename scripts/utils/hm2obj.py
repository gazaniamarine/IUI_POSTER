"""Converts a heightmap PNG + texture PNG into a textured .obj terrain mesh
(first step of the DEM/heightmap -> Habitat-loadable GLB pipeline).

Usage:
    python scripts/utils/hm2obj.py --heightmap <hm.png> --texture <tex.png> \\
        --size-x <m> --size-y <m> --size-z <m> [--out out.obj] [--stride 4]
"""

import argparse
import os
import shutil
import numpy as np
from PIL import Image


def load_heightmap(path):
    img = Image.open(path)

    # Keep original bit depth if possible.
    arr = np.array(img)

    # If RGB/RGBA, use first channel.
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    arr = arr.astype(np.float32)

    # Normalize to 0..1.
    arr_min = float(arr.min())
    arr_max = float(arr.max())

    if arr_max - arr_min < 1e-8:
        raise RuntimeError("Heightmap has no height variation.")

    arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def write_obj(height, texture_path, out_obj, size_x, size_y, size_z, stride):
    h, w = height.shape

    # Downsample for sanity.
    height = height[::stride, ::stride]
    h, w = height.shape

    out_dir = os.path.dirname(os.path.abspath(out_obj))
    base_name = os.path.splitext(os.path.basename(out_obj))[0]

    mtl_name = base_name + ".mtl"
    tex_name = os.path.basename(texture_path)

    out_mtl = os.path.join(out_dir, mtl_name)
    out_tex = os.path.join(out_dir, tex_name)

    os.makedirs(out_dir, exist_ok=True)

    # Copy texture beside OBJ.
    shutil.copy2(texture_path, out_tex)

    # Center terrain around origin.
    xs = np.linspace(-size_x / 2.0, size_x / 2.0, w)
    ys = np.linspace(-size_y / 2.0, size_y / 2.0, h)

    # Center height around zero.
    z = height * size_z
    z = z - np.mean(z)

    with open(out_mtl, "w") as mtl:
        mtl.write("newmtl marsyard_mat\n")
        mtl.write("Ka 1.000 1.000 1.000\n")
        mtl.write("Kd 1.000 1.000 1.000\n")
        mtl.write("Ks 0.000 0.000 0.000\n")
        mtl.write("d 1.0\n")
        mtl.write("illum 2\n")
        mtl.write(f"map_Kd {tex_name}\n")

    with open(out_obj, "w") as obj:
        obj.write(f"mtllib {mtl_name}\n")
        obj.write("o marsyard_terrain\n")
        obj.write("usemtl marsyard_mat\n")

        # Vertices.
        for row in range(h):
            y = ys[row]
            for col in range(w):
                x = xs[col]
                obj.write(f"v {x:.6f} {y:.6f} {z[row, col]:.6f}\n")

        # UVs.
        # OBJ UV origin is bottom-left-ish, so flip V.
        for row in range(h):
            v = 1.0 - row / max(h - 1, 1)
            for col in range(w):
                u = col / max(w - 1, 1)
                obj.write(f"vt {u:.6f} {v:.6f}\n")

        # Faces.
        # Same index for vertex and UV.
        for row in range(h - 1):
            for col in range(w - 1):
                i0 = row * w + col + 1
                i1 = row * w + col + 2
                i2 = (row + 1) * w + col + 2
                i3 = (row + 1) * w + col + 1

                # Two triangles per quad.
                obj.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
                obj.write(f"f {i0}/{i0} {i2}/{i2} {i3}/{i3}\n")

    print("Wrote:")
    print(" OBJ:", out_obj)
    print(" MTL:", out_mtl)
    print(" TEX:", out_tex)
    print("Grid:", w, "x", h)
    print("Vertices:", w * h)
    print("Triangles:", (w - 1) * (h - 1) * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heightmap", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--out", default="marsyard2022.obj")
    parser.add_argument("--size-x", type=float, required=True)
    parser.add_argument("--size-y", type=float, required=True)
    parser.add_argument("--size-z", type=float, required=True)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    height = load_heightmap(args.heightmap)

    write_obj(
        height=height,
        texture_path=args.texture,
        out_obj=args.out,
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
