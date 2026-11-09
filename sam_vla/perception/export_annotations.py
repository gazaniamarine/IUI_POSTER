"""Converts a segmentation-sweep run (rgb/ + masks_instance/ + masks_category/
+ segmentation_frames.jsonl) into standard training-annotation formats --
`coco` (one instances.json), `yolo` (bbox labels/data.yaml + an images/
symlink to rgb/), and `yolo-seg` (like yolo but with polygon labels traced
from masks_instance/ via cv2.findContours, imported lazily). Reads only
what's already on disk -- no habitat_sim/env dependency.

Usage:
    python -m sam_vla.perception.export_annotations --run-dir <run_dir> [--formats coco,yolo] [--out-dir <out_dir>]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ALL_FORMATS = ("coco", "yolo", "yolo-seg")


def load_run(run_dir: Path) -> Tuple[List[dict], List[str]]:
    """Reads run_dir/segmentation_frames.jsonl + summary.json's class_names
    (class_names[0] is always the background category, see
    segmentation_capture.build_category_lut -- never emitted as an
    annotation category)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    class_names = summary["class_names"]
    lines = (run_dir / "segmentation_frames.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    return records, class_names


def _polygons_from_instance_mask(
    instance_mask: np.ndarray, mesh_id: int
) -> List[List[float]]:
    """Traces the exact silhouette of one object (mesh_id == pixel value) into
    COCO-style flattened polygons [x1, y1, x2, y2, ...], one per external
    contour -- an object split into disjoint regions (e.g. occluded by
    another hull) yields multiple polygons, matching COCO's segmentation
    list-of-polygons convention. Contours with fewer than 3 points (can't
    form a polygon) are dropped."""
    import cv2

    mask = (np.asarray(instance_mask) == mesh_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygons.append(points.astype(np.float64).flatten().tolist())
    return polygons


def export_coco(
    run_dir: Path,
    out_path: Path,
    records: Sequence[dict],
    class_names: Sequence[str],
    with_segmentation: bool = True,
) -> dict:
    """Writes a COCO instances.json to out_path. Category ids are each
    category's index into class_names (so id 1 == class_names[1], etc.) --
    background (index 0) never appears in `categories`, matching the LUT
    convention used everywhere else in this package (spot_check_segmentation's
    palette, run_segmentation_sweep's class filtering)."""
    import imageio.v3 as iio

    run_dir = Path(run_dir)
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
        if i > 0
    ]

    images = []
    annotations = []
    ann_id = 1
    fallback_polygon_count = 0
    for image_id, rec in enumerate(records, start=1):
        instance_mask = (
            iio.imread(run_dir / rec["instance_mask_path"])
            if with_segmentation
            else None
        )
        if instance_mask is not None:
            height, width = instance_mask.shape[:2]
        else:
            height, width = iio.improps(run_dir / rec["rgb_path"]).shape[:2]

        images.append(
            {
                "id": image_id,
                "file_name": rec["rgb_path"],
                "width": int(width),
                "height": int(height),
            }
        )

        for obj in rec["objects"]:
            x, y, w, h = obj["bbox"]
            segmentation = []
            if with_segmentation:
                segmentation = _polygons_from_instance_mask(
                    instance_mask, obj["mesh_id"]
                )
            if not segmentation:
                # no traceable contour (e.g. a 1px sliver) -- fall back to the
                # bbox itself so every object still gets a valid segmentation
                # rather than silently losing its polygon.
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                fallback_polygon_count += 1

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_names.index(obj["category"]),
                    "bbox": [x, y, w, h],
                    "area": obj["pixel_count"],
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "info": {"description": f"exported from {run_dir}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco))

    if fallback_polygon_count:
        print(
            f"[export_annotations] {fallback_polygon_count} object(s) had no traceable "
            f"contour, used bbox-rectangle segmentation instead"
        )
    return coco


def _symlink_images_dir(run_dir: Path, yolo_dir: Path) -> None:
    images_link = yolo_dir / "images"
    if images_link.exists() or images_link.is_symlink():
        return
    target = os.path.relpath(run_dir / "rgb", start=yolo_dir)
    try:
        images_link.symlink_to(target, target_is_directory=True)
    except OSError as e:
        print(
            f"[export_annotations] could not symlink {images_link} -> {target} ({e}); "
            f"point your data.yaml's image path at {run_dir / 'rgb'} manually"
        )


def export_yolo(
    run_dir: Path,
    out_dir: Path,
    records: Sequence[dict],
    class_names: Sequence[str],
    segmentation: bool = False,
) -> Path:
    """Writes Ultralytics-style YOLO labels to out_dir/labels/<frame_id>.txt
    (one line per object: "class cx cy w h" normalized, or "class x1 y1 ..."
    normalized polygon points if segmentation=True), plus classes.txt and
    data.yaml. Class ids are 0-indexed over class_names[1:] (background
    excluded, since YOLO has no background class)."""
    import imageio.v3 as iio

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    names = list(class_names[1:])
    class_id_by_name = {name: i for i, name in enumerate(names)}

    empty_polygon_count = 0
    for rec in records:
        height = width = None
        instance_mask = None
        if segmentation:
            instance_mask = iio.imread(run_dir / rec["instance_mask_path"])
            height, width = instance_mask.shape[:2]
        elif rec["objects"]:
            height, width = iio.improps(run_dir / rec["rgb_path"]).shape[:2]

        lines = []
        for obj in rec["objects"]:
            class_id = class_id_by_name[obj["category"]]
            if segmentation:
                polygons = _polygons_from_instance_mask(instance_mask, obj["mesh_id"])
                if not polygons:
                    empty_polygon_count += 1
                    continue
                # YOLO-seg allows one polygon per line; emit the largest
                # contour by point count as the object's shape.
                points = max(polygons, key=len)
                normalized = [
                    (v / width if i % 2 == 0 else v / height)
                    for i, v in enumerate(points)
                ]
                lines.append(
                    " ".join([str(class_id)] + [f"{v:.6f}" for v in normalized])
                )
            else:
                x, y, w, h = obj["bbox"]
                cx, cy = (x + w / 2) / width, (y + h / 2) / height
                nw, nh = w / width, h / height
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (labels_dir / f"{rec['frame_id']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    (out_dir / "classes.txt").write_text("\n".join(names) + "\n")
    _symlink_images_dir(run_dir, out_dir)
    (out_dir / "data.yaml").write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(names)}\n"
        f"names: {json.dumps(names)}\n"
    )

    if segmentation and empty_polygon_count:
        print(
            f"[export_annotations] {empty_polygon_count} object(s) had no traceable "
            f"contour and were dropped from YOLO-seg labels"
        )
    return out_dir


def export_annotations(
    run_dir: Path, out_dir: Path, formats: Sequence[str]
) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    records, class_names = load_run(run_dir)

    written: Dict[str, Path] = {}
    if "coco" in formats:
        coco_path = out_dir / "coco" / "instances.json"
        export_coco(run_dir, coco_path, records, class_names)
        written["coco"] = coco_path
    if "yolo" in formats:
        yolo_dir = out_dir / "yolo"
        export_yolo(run_dir, yolo_dir, records, class_names, segmentation=False)
        written["yolo"] = yolo_dir
    if "yolo-seg" in formats:
        yolo_seg_dir = out_dir / "yolo-seg"
        export_yolo(run_dir, yolo_seg_dir, records, class_names, segmentation=True)
        written["yolo-seg"] = yolo_seg_dir
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="a run directory produced by run_segmentation_sweep.py",
    )
    ap.add_argument(
        "--formats",
        default="coco,yolo",
        help=f"comma-separated subset of {ALL_FORMATS}",
    )
    ap.add_argument(
        "--out-dir", default=None, help="default: <run-dir>/annotations_export/"
    )
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - set(ALL_FORMATS)
    if unknown:
        raise SystemExit(
            f"unknown format(s) {sorted(unknown)}, choose from {ALL_FORMATS}"
        )

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "annotations_export"

    written = export_annotations(run_dir, out_dir, formats)
    for fmt, path in written.items():
        print(f"[export_annotations] wrote {fmt} -> {path}")
