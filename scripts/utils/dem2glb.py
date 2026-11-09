#!/usr/bin/env python3
"""DEM .tif -> Habitat-loadable .glb terrain mesh, direct from a Blender grid
build (one vertex per DEM pixel) rather than subsurf/displace modifiers, so
the mesh exactly matches the upsampled height field.

Usage:
    blender --background --python scripts/utils/dem2glb.py -- \
        --input marsyard2022_terrain_hm_1025.tif \
        --output marsyard2022_terrain.glb \
        --size-x 50 --size-y 50 --size-z 4.820803273566 \
        --texture marsyard2022_terrain_texture.png
"""

import sys
import argparse

import bpy
import bmesh
import numpy as np
import rasterio


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to DEM .tif (upsampled)")
    p.add_argument("--output", required=True, help="path to output .glb")
    p.add_argument("--size-x", type=float, required=True)
    p.add_argument("--size-y", type=float, required=True)
    p.add_argument(
        "--size-z",
        type=float,
        required=True,
        help="elevation range, from <heightmap><size> in the SDF",
    )
    p.add_argument(
        "--texture",
        default=None,
        help="optional diffuse texture (e.g. marsyard2022_terrain_texture.png)",
    )
    p.add_argument(
        "--decimate-ratio",
        type=float,
        default=0.25,
        help="fraction of triangles to keep, e.g. 0.25 = keep 25%% "
        "(default 0.25; set to 1.0 to skip decimation)",
    )
    return p.parse_args(argv)


def load_dem(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
    assert data.shape[0] == data.shape[1], "expected square DEM"
    return data


def build_mesh(data, size_x, size_y, size_z):
    res = data.shape[0]

    # Gazebo/SDF heightmap convention: raw pixel values are normalized
    # (min..max of the data) then scaled to size_z, mid-level at the mean.
    # Must match sam_vla/env/terrain.py's HeightmapGrid normalization exactly
    # (scale then subtract the mean) -- that module is the sole source of the
    # rover/camera's world-Y at every pose, computed independently from this
    # mesh. Any deviation shifts this mesh vertically relative to where the
    # camera thinks the ground is, at every point on the map.
    z_min, z_max = data.min(), data.max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    heights = (data - z_min) / z_range * size_z
    heights = heights - heights.mean()

    xs = np.linspace(-size_x / 2.0, size_x / 2.0, res)
    ys = np.linspace(-size_y / 2.0, size_y / 2.0, res)

    bm = bmesh.new()
    verts = np.empty((res, res), dtype=object)

    for j in range(res):
        for i in range(res):
            verts[j, i] = bm.verts.new((xs[i], ys[j], heights[j, i]))

    bm.verts.ensure_lookup_table()

    # build quad faces (as triangles) across the grid
    for j in range(res - 1):
        for i in range(res - 1):
            v00 = verts[j, i]
            v10 = verts[j, i + 1]
            v01 = verts[j + 1, i]
            v11 = verts[j + 1, i + 1]
            bm.faces.new((v00, v10, v11))
            bm.faces.new((v00, v11, v01))

    bm.normal_update()

    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        for loop in face.loops:
            co = loop.vert.co
            u = (co.x + size_x / 2.0) / size_x
            v = (co.y + size_y / 2.0) / size_y
            loop[uv_layer].uv = (u, v)

    mesh = bpy.data.meshes.new("marsyard_terrain")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("marsyard_terrain", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def decimate_mesh(obj, ratio):
    """Reduce triangle count via the Decimate modifier (collapse type),
    applied before export. Preserves overall terrain silhouette while
    cutting redundant flat-area density."""
    if ratio >= 1.0:
        print("decimate ratio >= 1.0, skipping decimation")
        return

    mod = obj.modifiers.new(name="decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    print(
        f"decimated to ratio={ratio}: "
        f"{len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces"
    )


def apply_texture(obj, texture_path):
    mat = bpy.data.materials.new(name="marsyard_terrain_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(texture_path)
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])
    obj.data.materials.append(mat)


def main():
    args = parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    print(f"loading DEM: {args.input}")
    data = load_dem(args.input)
    print(f"DEM shape: {data.shape}, min={data.min():.4f}, max={data.max():.4f}")

    print("building mesh...")
    obj = build_mesh(data, args.size_x, args.size_y, args.size_z)
    print(f"mesh built: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")

    print(f"decimating (ratio={args.decimate_ratio})...")
    decimate_mesh(obj, args.decimate_ratio)

    if args.texture:
        print(f"applying texture: {args.texture}")
        apply_texture(obj, args.texture)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    print(f"exporting GLB -> {args.output}")
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        # Habitat loads this stage with no .stage_config.json, so it falls back
        # to its default orient_up=[0,0,1] and rotates the asset as if it were
        # raw Z-up. Exporting with the glTF-spec Y-up conversion (the default)
        # double-rotates it -- ground plane ends up on X/Y, height on Z, and
        # collision raycasts miss the mesh entirely. export_yup=False keeps the
        # raw Blender axes (height on Z, ground plane on X/Y) so Habitat's
        # rotation lands the mesh correctly, matching the previously-working
        # marsyard2022_bak.glb export.
        export_yup=False,
    )
    print("done.")


if __name__ == "__main__":
    main()
