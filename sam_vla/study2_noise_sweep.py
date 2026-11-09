"""Study 2 (next.md) sweep driver: runs sam_vla.run_navdp_rollout once per
(episode, noise_level) pair, sweeping --drive-odom-noise-std over the range
belief_exp's own sigma_min sweeps already tested (0.0 to ~0.15,
env_odom_noise_std -- see next.md's "Important caveat" table, recovered from
belief_exp/results/*_summary.csv) so this study's real-sim results line up
against belief_exp's offline numpy findings at the same nominal noise levels.

Same subprocess-per-run shape as sam_vla/study1_paired_runs.py (itself modeled
on sam_vla/perception/run_dataset_pipeline.py) -- each (episode, noise_level)
gets its own Habitat-sim env + Qwen server subprocess lifecycle, not in-process
reuse.

Usage:
    python -m sam_vla.study2_noise_sweep \\
        --episodes-file study2_episodes.json --out-dir study2_run1 \\
        --noise-levels 0.0,0.041,0.068,0.075,0.109,0.15 \\
        -- --scene-path assets/marsyard2022.glb --heightmap-path <hm.png> \\
           --ckpt <ckpt.pt> --cbf --max-steps 300

episodes.json: a JSON list of {"id": "ep00", "start_x": 0.0, "start_z": 8.0,
"start_yaw": 0.0} objects -- one per episode, reused unchanged across every
noise level swept.

Everything after -- is forwarded verbatim to run_navdp_rollout for every
(episode, noise_level) run. It must NOT include --start-x/--start-z/
--start-yaw/--drive-odom-noise-std/--drive-odom-noise-seed/--out-dir -- this
driver sets those itself per run, and argparse's plain store action would
silently let the driver's own values win over any duplicate in the forwarded
list, which would hide a config mistake rather than flag it.

Each episode gets its own --drive-odom-noise-seed (derived from the episode's
index, so it's stable/reproducible across a run and the same seed is reused
for every noise level that episode is swept at -- the point being to compare
the *magnitude* of noise, not have both the seed and the magnitude vary
together confoundingly).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# From next.md's recovered belief_exp/results/21_max_confidence_summary.csv
# table (200 episodes/config) plus the ~0.075 non-viable stress point flagged
# by the stricter run -- a starting point, not a standing spec; re-derive from
# belief_exp/results/*.csv (present locally, see next.md's caveat) if the
# grid needs finer resolution.
DEFAULT_NOISE_LEVELS = [0.0, 0.041, 0.068, 0.075, 0.109, 0.15]

_DRIVER_OWNED_FLAGS = {
    "--start-x",
    "--start-z",
    "--start-yaw",
    "--drive-odom-noise-std",
    "--drive-odom-noise-seed",
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
            f"study2_noise_sweep owns these per (episode, noise_level) run, "
            f"but found {owned_present} in the forwarded args"
        )


def _episode_seed(episode_index: int, base_seed: int = 20260807) -> int:
    """Stable per-episode seed, independent of which noise level is being run
    (so the same seed is reused across an episode's whole noise sweep --
    only the magnitude varies, not both magnitude and the noise draw
    sequence)."""
    return base_seed + episode_index


def build_command(
    python_bin: str,
    episode: dict,
    noise_level: float,
    seed: int,
    run_out_dir: Path,
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
        "--drive-odom-noise-std",
        str(noise_level),
        "--drive-odom-noise-seed",
        str(seed),
        "--out-dir",
        str(run_out_dir),
    ]


def run_sweep(
    episodes: List[dict],
    noise_levels: List[float],
    out_dir: Path,
    python_bin: str,
    forwarded_args: List[str],
    keep_going: bool = False,
    dry_run: bool = False,
) -> dict:
    check_forwarded_args(forwarded_args)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": _now_iso(),
        "forwarded_args": forwarded_args,
        "noise_levels": noise_levels,
        "episodes": [],
    }

    for ep_index, episode in enumerate(episodes):
        seed = _episode_seed(ep_index)
        ep_record = {
            "id": episode["id"],
            "start_x": episode["start_x"],
            "start_z": episode["start_z"],
            "start_yaw": episode["start_yaw"],
            "seed": seed,
            "runs": {},
        }
        for noise_level in noise_levels:
            run_out_dir = out_dir / f"{episode['id']}_noise{noise_level}"
            cmd = build_command(
                python_bin, episode, noise_level, seed, run_out_dir, forwarded_args
            )
            print(
                f"[study2] {episode['id']}/noise={noise_level}: {' '.join(cmd)}",
                flush=True,
            )

            run_record = {"out_dir": str(run_out_dir), "cmd": cmd}
            if dry_run:
                run_record["returncode"] = None
            else:
                result = subprocess.run(cmd)
                run_record["returncode"] = result.returncode
                if result.returncode != 0:
                    print(
                        f"[study2] {episode['id']}/noise={noise_level} FAILED "
                        f"(returncode={result.returncode})",
                        flush=True,
                    )
                    if not keep_going:
                        ep_record["runs"][str(noise_level)] = run_record
                        manifest["episodes"].append(ep_record)
                        _write_manifest(out_dir, manifest)
                        raise RuntimeError(
                            f"{episode['id']}/noise={noise_level} failed with "
                            f"returncode={result.returncode} -- stopping "
                            f"(pass --keep-going to continue past failures)"
                        )
            ep_record["runs"][str(noise_level)] = run_record
        manifest["episodes"].append(ep_record)
        _write_manifest(out_dir, manifest)

    return manifest


def _write_manifest(out_dir: Path, manifest: dict) -> Path:
    manifest_path = out_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episodes-file", required=True, help="JSON episode list, see module docstring")
    ap.add_argument("--out-dir", required=True, help="base directory for per-run out_dirs + sweep_manifest.json")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter to run run_navdp_rollout with (needs habitat_sim installed)",
    )
    ap.add_argument(
        "--noise-levels",
        default=",".join(str(v) for v in DEFAULT_NOISE_LEVELS),
        help="comma-separated env_odom_noise_std values to sweep (default: next.md's recovered belief_exp range)",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="continue to the next (episode, noise_level) run after a failure instead of stopping",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run without running them",
    )
    ap.add_argument(
        "forwarded_args",
        nargs=argparse.REMAINDER,
        help="everything after -- is forwarded to run_navdp_rollout for every (episode, noise_level) run",
    )
    args = ap.parse_args()

    forwarded_args = args.forwarded_args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    noise_levels = [float(v.strip()) for v in args.noise_levels.split(",") if v.strip()]
    episodes = load_episodes(Path(args.episodes_file))

    run_sweep(
        episodes,
        noise_levels,
        Path(args.out_dir),
        args.python,
        forwarded_args,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )
