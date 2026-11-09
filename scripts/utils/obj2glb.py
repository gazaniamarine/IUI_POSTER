"""Blender-run converter: .obj terrain mesh -> Habitat-loadable .glb. Must pass
export_yup=False (see CLAUDE.md "Known issues") or Habitat's implicit Z-up
stage rotation double-flips the mesh, producing black-void renders.

Usage:
    blender --background --python scripts/utils/obj2glb.py -- input.obj output.glb
"""

import bpy
import os
import sys


def parse_arguments():
    if "--" not in sys.argv:
        raise RuntimeError(
            "Missing arguments.\n"
            "Usage:\n"
            "  blender --background --python obj2glb.py -- input.obj output.glb"
        )

    args = sys.argv[sys.argv.index("--") + 1 :]

    if len(args) != 2:
        raise RuntimeError(
            "Expected exactly two arguments: input OBJ and output GLB.\n"
            "Usage:\n"
            "  blender --background --python obj2glb.py -- input.obj output.glb"
        )

    return os.path.abspath(args[0]), os.path.abspath(args[1])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove unused mesh and material data.
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def import_obj(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input OBJ not found: {filepath}")

    # Blender 4.x importer.
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(
            filepath=filepath,
            forward_axis="NEGATIVE_Y",
            up_axis="Z",
        )

    # Blender 3.x importer.
    elif hasattr(bpy.ops.import_scene, "obj"):
        try:
            bpy.ops.preferences.addon_enable(module="io_scene_obj")
        except Exception as exc:
            print(f"OBJ addon warning: {exc}")

        bpy.ops.import_scene.obj(
            filepath=filepath,
            axis_forward="-Y",
            axis_up="Z",
            use_image_search=True,
        )

    else:
        raise RuntimeError("No OBJ importer is available in this Blender installation.")


def prepare_meshes():
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not mesh_objects:
        raise RuntimeError("The OBJ import produced no mesh objects.")

    # Apply scale transforms.
    bpy.ops.object.select_all(action="DESELECT")

    for obj in mesh_objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True,
    )

    # Enable smooth shading across the terrain.
    for obj in mesh_objects:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

        obj.data.update()

    return mesh_objects


def export_glb(filepath):
    output_dir = os.path.dirname(filepath)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format="GLB",
        use_selection=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_apply=True,
        # See dem2glb.py: Habitat has no .stage_config.json for this asset, so
        # it falls back to orient_up=[0,0,1] and rotates as raw Z-up. The
        # glTF-spec Y-up conversion (export_yup default True) double-rotates
        # it, putting the ground plane on X/Z and height on Y -- the camera
        # ends up looking at empty space (black-void renders, zero objects
        # detected). export_yup=False keeps Blender's native Z-up axes so
        # Habitat's rotation lands the mesh correctly.
        export_yup=False,
    )


def main():
    input_obj, output_glb = parse_arguments()

    print(f"Blender version: {bpy.app.version_string}")
    print(f"Input OBJ: {input_obj}")
    print(f"Output GLB: {output_glb}")

    clear_scene()
    import_obj(input_obj)
    meshes = prepare_meshes()

    vertices = sum(len(obj.data.vertices) for obj in meshes)
    polygons = sum(len(obj.data.polygons) for obj in meshes)

    print(f"Mesh objects: {len(meshes)}")
    print(f"Vertices: {vertices}")
    print(f"Polygons: {polygons}")

    export_glb(output_glb)

    if not os.path.isfile(output_glb):
        raise RuntimeError("GLB export completed without creating the output file.")

    size_mb = os.path.getsize(output_glb) / (1024 * 1024)

    print(f"Exported successfully: {output_glb}")
    print(f"GLB size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
