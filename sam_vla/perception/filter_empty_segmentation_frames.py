"""Drops frames with no detected objects from a segmentation-sweep run
(sam_vla.run_segmentation_sweep): frames whose segmentation_frames.jsonl
record has an empty "objects" list carry no supervision signal, so this
prunes them and their rgb/masks_instance/masks_category files (and any
spot_check overlay, since spot_check_segmentation.py names those by
frame_id) together, keeping every per-frame directory in sync.

Usage:
    python -m sam_vla.perception.filter_empty_segmentation_frames --run-dir <run_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def filter_run(run_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Removes frames with empty "objects" from run_dir/segmentation_frames.jsonl
    and deletes their rgb/instance-mask/category-mask/spot-check files.
    Returns (num_kept, num_dropped)."""
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "segmentation_frames.jsonl"
    lines = jsonl_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]

    kept = [rec for rec in records if rec["objects"]]
    dropped = [rec for rec in records if not rec["objects"]]

    for rec in dropped:
        for key in ("rgb_path", "instance_mask_path", "category_mask_path"):
            path = run_dir / rec[key]
            if not dry_run and path.exists():
                path.unlink()
        spot_check_path = run_dir / "spot_check" / f"{rec['frame_id']}.png"
        if not dry_run and spot_check_path.exists():
            spot_check_path.unlink()

    if not dry_run:
        jsonl_path.write_text("".join(json.dumps(rec) + "\n" for rec in kept))
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["total_frames"] = len(kept)
            summary_path.write_text(json.dumps(summary, indent=2))

    return len(kept), len(dropped)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="a run directory produced by run_segmentation_sweep.py",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report counts without deleting files or rewriting jsonl/summary",
    )
    args = ap.parse_args()

    num_kept, num_dropped = filter_run(Path(args.run_dir), args.dry_run)
    verb = "would drop" if args.dry_run else "dropped"
    print(f"{verb} {num_dropped} empty frames, kept {num_kept}")
