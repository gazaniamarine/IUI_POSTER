"""Study 1 (next.md) Phase 3: runs Condition B (--uncertainty-condition
autonomous) and Condition A (--uncertainty-condition human) over the same set
of episodes (start poses), each pair sharing an identical scene/heightmap/
checkpoint/CBF/etc config, so their uncertainty_trigger instances line up 1:1
for Phase 4's timing-subtraction analysis (next.md's "Timing decomposition"
section) -- see "Run both conditions over the same seed/episode set" in
next.md's Phase 3 description.

Each (episode, condition) pair is run as its own sam_vla.run_navdp_rollout
subprocess, the same shape as sam_vla/perception/run_dataset_pipeline.py uses
for run_segmentation_sweep, and the same thing a human would do by hand
running the CLI twice (as next.md's Integration-project Phase 4 comparison
run literally did) -- this script just automates that over a list of
episodes. Subprocess isolation, not in-process reuse, so each episode gets
its own Habitat-sim env + Qwen server subprocess lifecycle, matching every
other rollout invocation in this repo.

Usage:
    python -m sam_vla.study1_paired_runs \
        --episodes-file study1_episodes.json --out-dir study1_run1 \
        -- --scene-path assets/marsyard2022.glb --heightmap-path <hm.png> \
           --ckpt <ckpt.pt> --cbf --max-steps 300 --uncertainty-threshold 0.5

episodes.json: a JSON list of {"id": "ep00", "start_x": 0.0, "start_z": 8.0,
"start_yaw": 0.0} objects -- one per episode, reused unchanged for both
conditions.

Everything after -- is forwarded verbatim to run_navdp_rollout for both
conditions of every episode (scene/heightmap/ckpt/CBF/--uncertainty-threshold/
etc). It must NOT include --start-x/--start-z/--start-yaw/--uncertainty-
condition/--out-dir -- this driver sets those itself per (episode, condition)
pair, and argparse's plain store action would silently let the driver's own
values win over any duplicate in the forwarded list, which would hide a
config mistake rather than flag it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

CONDITIONS = ["autonomous", "human"]

# Flags this driver owns per (episode, condition) pair -- forwarding any of
# these would be ambiguous about which value actually takes effect.
_DRIVER_OWNED_FLAGS = {
    "--start-x",
    "--start-z",
    "--start-yaw",
    "--uncertainty-condition",
    "--out-dir",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_episodes(episodes_file: Path) -> List[dict]:
    episodes = json.loads(episodes_file.read_text())
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{episodes_file} must contain a non-empty JSON list")
    seen_ids = set()
    for ep in episodes:
        missing = {"id", "start_x", "start_z", "start_yaw"} - set(ep)
        if missing:
            raise ValueError(f"episode {ep} missing required key(s) {missing}")
        if ep["id"] in seen_ids:
            raise ValueError(f"duplicate episode id {ep['id']!r} in {episodes_file}")
        seen_ids.add(ep["id"])
    return episodes


def check_forwarded_args(forwarded_args: List[str]) -> None:
    owned_present = [a for a in forwarded_args if a in _DRIVER_OWNED_FLAGS]
    if owned_present:
        raise ValueError(
            f"forwarded args must not set {sorted(_DRIVER_OWNED_FLAGS)} -- "
            f"study1_paired_runs owns these per (episode, condition) pair, "
            f"but found {owned_present} in the forwarded args"
        )


def build_command(
    python_bin: str,
    episode: dict,
    condition: str,
    pair_out_dir: Path,
    forwarded_args: List[str],
) -> List[str]:
    return [
        python_bin,
        "-m",
        "sam_vla.run_navdp_rollout",
        *forwarded_args,
        "--start-x",
        str(episode["start_x"]),
        "--start-z",
        str(episode["start_z"]),
        "--start-yaw",
        str(episode["start_yaw"]),
        "--uncertainty-condition",
        condition,
        "--out-dir",
        str(pair_out_dir),
    ]


def run_paired_episodes(
    episodes: List[dict],
    out_dir: Path,
    python_bin: str,
    forwarded_args: List[str],
    conditions: List[str] = CONDITIONS,
    keep_going: bool = False,
    dry_run: bool = False,
) -> dict:
    check_forwarded_args(forwarded_args)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": _now_iso(),
        "forwarded_args": forwarded_args,
        "conditions": conditions,
        "episodes": [],
    }

    for episode in episodes:
        ep_record = {
            "id": episode["id"],
            "start_x": episode["start_x"],
            "start_z": episode["start_z"],
            "start_yaw": episode["start_yaw"],
            "runs": {},
        }
        for condition in conditions:
            pair_out_dir = out_dir / f"{episode['id']}_{condition}"
            cmd = build_command(
                python_bin, episode, condition, pair_out_dir, forwarded_args
            )
            print(f"[study1] {episode['id']}/{condition}: {' '.join(cmd)}", flush=True)

            run_record = {"out_dir": str(pair_out_dir), "cmd": cmd}
            if dry_run:
                run_record["returncode"] = None
            else:
                result = subprocess.run(cmd)
                run_record["returncode"] = result.returncode
                if result.returncode != 0:
                    print(
                        f"[study1] {episode['id']}/{condition} FAILED "
                        f"(returncode={result.returncode})",
                        flush=True,
                    )
                    if not keep_going:
                        ep_record["runs"][condition] = run_record
                        manifest["episodes"].append(ep_record)
                        _write_manifest(out_dir, manifest)
                        raise RuntimeError(
                            f"{episode['id']}/{condition} failed with "
                            f"returncode={result.returncode} -- stopping "
                            f"(pass --keep-going to continue past failures)"
                        )
            ep_record["runs"][condition] = run_record
        manifest["episodes"].append(ep_record)
        _write_manifest(out_dir, manifest)

    return manifest


def _write_manifest(out_dir: Path, manifest: dict) -> Path:
    manifest_path = out_dir / "pairs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episodes-file", required=True, help="JSON episode list, see module docstring")
    ap.add_argument("--out-dir", required=True, help="base directory for per-pair out_dirs + pairs_manifest.json")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter to run run_navdp_rollout with (needs habitat_sim installed)",
    )
    ap.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help=f"comma-separated subset of {CONDITIONS} to run per episode",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="continue to the next (episode, condition) pair after a failure instead of stopping",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run without running them",
    )
    ap.add_argument(
        "forwarded_args",
        nargs=argparse.REMAINDER,
        help="everything after -- is forwarded to run_navdp_rollout for both conditions of every episode",
    )
    args = ap.parse_args()

    forwarded_args = args.forwarded_args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown condition(s) {sorted(unknown)}, choose from {CONDITIONS}")

    episodes = load_episodes(Path(args.episodes_file))

    run_paired_episodes(
        episodes,
        Path(args.out_dir),
        args.python,
        forwarded_args,
        conditions=conditions,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )
