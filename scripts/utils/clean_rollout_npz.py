"""Strip specified array keys out of rollout .npz files under a directory tree.

Walks --input-dir, finds every file matching --pattern (default "rollout.npz")
at any depth, and rewrites each one without the keys listed in --keys (e.g. the
bulky rgb_frames/rgb/depth/goal_mask/obstacle_mask arrays that dominate npz size).
Defaults to a dry run that only reports what would be freed; pass --apply to
actually rewrite files.

Usage:
    python clean_rollout_npz.py --input-dir output/study2_batch2 --keys rgb_frames
    python clean_rollout_npz.py --input-dir output/study2_batch2 --keys rgb_frames --apply
    python clean_rollout_npz.py --input-dir runs --keys rgb,depth,goal_mask,obstacle_mask --apply
"""

import argparse
from pathlib import Path

import numpy as np


def clean_one(path: Path, keys_to_drop: set[str], apply: bool) -> tuple[int, int, list[str]]:
    original_size = path.stat().st_size
    with np.load(path, allow_pickle=True) as data:
        dropped = sorted(set(data.files) & keys_to_drop)
        if not dropped:
            return original_size, original_size, []
        kept = {k: data[k] for k in data.files if k not in keys_to_drop}
        freed_estimate = sum(data[k].nbytes for k in dropped)

    if not apply:
        return original_size, original_size - freed_estimate, dropped

    tmp_path = path.with_suffix(".npz.tmp")
    np.savez(tmp_path, **kept)
    tmp_path.replace(path)
    new_size = path.stat().st_size
    return original_size, new_size, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Root directory to search under")
    parser.add_argument("--keys", required=True, help="Comma-separated npz array keys to strip out (e.g. rgb_frames,depth)")
    parser.add_argument("--pattern", default="rollout.npz", help="Filename to match, searched recursively (default: rollout.npz)")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite files. Without this, only reports what would happen.")
    args = parser.parse_args()

    keys_to_drop = {k.strip() for k in args.keys.split(",") if k.strip()}
    files = sorted(args.input_dir.rglob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' found under {args.input_dir}")
        return

    total_before = 0
    total_after = 0
    for path in files:
        before, after, dropped = clean_one(path, keys_to_drop, args.apply)
        total_before += before
        total_after += after
        rel = path.relative_to(args.input_dir)
        if dropped:
            verb = "cleaned" if args.apply else "would clean"
            print(f"{verb} {rel}: dropping [{', '.join(dropped)}]  {before/1e6:.1f}MB -> {after/1e6:.1f}MB")
        else:
            print(f"skip {rel}: none of the requested keys present")

    mode = "" if args.apply else " (dry run, pass --apply to actually rewrite)"
    print(f"\nTotal: {total_before/1e6:.1f}MB -> {total_after/1e6:.1f}MB, freed {(total_before-total_after)/1e6:.1f}MB{mode}")


if __name__ == "__main__":
    main()
