"""Orchestrates the full segmentation-dataset pipeline in one command, chaining
the stages documented in next.md and `usage`:

    run_segmentation_sweep -> filter_empty_segmentation_frames ->
    spot_check_segmentation -> export_annotations

Does NOT cover next.md Steps 1-2 (manual mesh division + convex hull
authoring via mesh_annotation_tool.py) -- that's a human-in-the-loop step
that has to happen once, beforehand, against --annotations-dir.

run_segmentation_sweep needs habitat_sim (run with --python pointing at that
env's interpreter, e.g. the `habitat` conda env, if this script itself isn't
running there); it's run as a subprocess for that reason. The other three
stages are pure numpy/PIL/cv2/imageio and are called in-process. The sweep
names its own timestamped run directory, so it's recovered by diffing
--out-dir's contents before/after the subprocess rather than parsing stdout.

Usage:
    python -m sam_vla.perception.run_dataset_pipeline --out-dir output/ \
        [--python /path/to/habitat/env/bin/python] \
        [--formats coco,yolo] [--spot-check-n 20] [--dry-run-filter] \
        [-- <extra run_segmentation_sweep args, e.g. --categories small_rock --mode random>]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from sam_vla.perception.export_annotations import ALL_FORMATS, export_annotations
from sam_vla.perception.filter_empty_segmentation_frames import filter_run
from sam_vla.perception.spot_check_segmentation import spot_check_run


def run_sweep_subprocess(out_dir: Path, python_bin: str, sweep_args: List[str]) -> Path:
    """Runs run_segmentation_sweep --out-dir out_dir [sweep_args] as a
    subprocess and returns the run directory it created, found by diffing
    out_dir's subdirectories before/after (the sweep picks its own
    timestamped run_id, so there's no other way to learn it from outside)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out_dir.iterdir() if p.is_dir()}

    cmd = [
        python_bin,
        "-m",
        "sam_vla.run_segmentation_sweep",
        "--out-dir",
        str(out_dir),
        *sweep_args,
    ]
    print(f"[pipeline] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    after = {p.name for p in out_dir.iterdir() if p.is_dir()}
    new_dirs = after - before
    if len(new_dirs) != 1:
        raise RuntimeError(
            f"expected exactly one new run directory under {out_dir}, found {sorted(new_dirs)} "
            "-- can't tell which run to process next"
        )
    return out_dir / new_dirs.pop()


def run_pipeline(
    out_dir: Path,
    python_bin: str,
    sweep_args: List[str],
    formats: List[str],
    spot_check_n: int,
    export_dir: Optional[Path],
    dry_run_filter: bool,
) -> Path:
    run_dir = run_sweep_subprocess(out_dir, python_bin, sweep_args)

    num_kept, num_dropped = filter_run(run_dir, dry_run=dry_run_filter)
    verb = "would drop" if dry_run_filter else "dropped"
    print(f"[pipeline] {verb} {num_dropped} empty frames, kept {num_kept}")

    if dry_run_filter:
        print("[pipeline] dry run -- stopping before spot-check/export")
        return run_dir

    if num_kept == 0:
        raise RuntimeError(
            f"all frames in {run_dir} were empty -- nothing left to export"
        )

    if spot_check_n > 0:
        written = spot_check_run(run_dir, n=spot_check_n)
        print(
            f"[pipeline] wrote {len(written)} spot-check overlays -> {run_dir / 'spot_check'}"
        )

    export_out = export_dir or (run_dir / "annotations_export")
    written = export_annotations(run_dir, export_out, formats)
    for fmt, path in written.items():
        print(f"[pipeline] wrote {fmt} -> {path}")

    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="passed through as run_segmentation_sweep's --out-dir",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter to run run_segmentation_sweep with (needs habitat_sim installed)",
    )
    ap.add_argument(
        "--formats",
        default="coco,yolo",
        help=f"comma-separated subset of {ALL_FORMATS}",
    )
    ap.add_argument(
        "--export-dir", default=None, help="default: <run_dir>/annotations_export/"
    )
    ap.add_argument(
        "--spot-check-n",
        type=int,
        default=20,
        help="spot-check overlays to render after filtering, 0 to skip",
    )
    ap.add_argument(
        "--dry-run-filter",
        action="store_true",
        help="report empty-frame drop counts without deleting anything; stops before spot-check/export "
        "since the kept-frame set isn't final",
    )
    ap.add_argument(
        "sweep_args",
        nargs=argparse.REMAINDER,
        help="everything after -- is forwarded to run_segmentation_sweep",
    )
    args = ap.parse_args()

    sweep_args = args.sweep_args
    if sweep_args and sweep_args[0] == "--":
        sweep_args = sweep_args[1:]

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - set(ALL_FORMATS)
    if unknown:
        raise SystemExit(
            f"unknown format(s) {sorted(unknown)}, choose from {ALL_FORMATS}"
        )

    run_pipeline(
        Path(args.out_dir),
        args.python,
        sweep_args,
        formats,
        args.spot_check_n,
        Path(args.export_dir) if args.export_dir else None,
        args.dry_run_filter,
    )
