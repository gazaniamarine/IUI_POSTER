#!/usr/bin/env python3
"""Parse rollout log directories under logs/ into summary tables.

Single-episode mode:
    python scripts/utils/log_reader.py --log-dir logs/2026-07-12_230309_vlm-goal_nudge_obs100_seed0 --k 5

Batch mode (walks every episode dir under a root):
    python scripts/utils/log_reader.py --root-dir logs --k 5 --format csv --output out.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_episode_dir(path: Path) -> bool:
    return (path / "summary.json").is_file()


def find_episode_dirs(root: Path):
    if is_episode_dir(root):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and is_episode_dir(p))


def closest_obstacles(summary: dict, k: int):
    dists = summary.get("closest_approach_per_obstacle", {})
    return sorted(dists.items(), key=lambda kv: kv[1])[:k]


def parse_episode(episode_dir: Path, k: int):
    summary = load_json(episode_dir / "summary.json")
    config = (
        load_json(episode_dir / "config.json")
        if (episode_dir / "config.json").is_file()
        else {}
    )
    qwen_queries = load_jsonl(episode_dir / "qwen_queries.jsonl")
    cbf_events = load_jsonl(episode_dir / "cbf_events.jsonl")

    top_k = closest_obstacles(summary, k)
    overall_closest_id, overall_closest_dist = top_k[0] if top_k else (None, None)

    return {
        "run_id": summary.get("run_id", episode_dir.name),
        "dir": str(episode_dir),
        "success": summary.get("success"),
        "termination_reason": summary.get("termination_reason"),
        "total_steps": summary.get("total_steps"),
        "final_distance_to_goal": summary.get("final_distance_to_goal"),
        "num_cbf_interventions": summary.get("num_cbf_interventions", len(cbf_events)),
        "num_qwen_queries": summary.get("num_qwen_queries", len(qwen_queries)),
        "num_goal_proximity_events": summary.get("num_goal_proximity_events"),
        "goal_mode": config.get("goal_mode"),
        "steering_mode": config.get("steering_mode"),
        "obstacle_count": config.get("obstacle_count"),
        "obstacle_seed": config.get("obstacle_seed"),
        "closest_obstacle_id": overall_closest_id,
        "closest_obstacle_dist": overall_closest_dist,
        "top_k_closest": top_k,
    }


def print_episode_table(rows, k):
    for row in rows:
        print(f"run_id:                 {row['run_id']}")
        print(f"success:                {row['success']}")
        print(f"termination_reason:     {row['termination_reason']}")
        print(f"total_steps:            {row['total_steps']}")
        print(f"final_distance_to_goal: {row['final_distance_to_goal']}")
        print(f"num_cbf_interventions:  {row['num_cbf_interventions']}")
        print(f"num_qwen_queries:       {row['num_qwen_queries']}")
        print(
            f"goal_mode/steering:     {row['goal_mode']}/{row['steering_mode']}  "
            f"(obstacles={row['obstacle_count']}, seed={row['obstacle_seed']})"
        )
        print(f"{k} closest obstacles:")
        for obs_id, dist in row["top_k_closest"]:
            print(f"    {obs_id:20s} {dist:.3f}")
        print("-" * 60)


def print_aggregate_table(rows):
    n = len(rows)
    n_success = sum(1 for r in rows if r["success"])
    steps = [r["total_steps"] for r in rows if r["total_steps"] is not None]
    closest = [
        r["closest_obstacle_dist"]
        for r in rows
        if r["closest_obstacle_dist"] is not None
    ]
    qwen = [r["num_qwen_queries"] for r in rows if r["num_qwen_queries"] is not None]

    print(f"episodes:          {n}")
    print(
        f"successes:         {n_success} ({100.0 * n_success / n:.1f}%)"
        if n
        else "successes:         0"
    )
    if steps:
        print(
            f"steps  min/avg/max: {min(steps)} / {sum(steps) / len(steps):.1f} / {max(steps)}"
        )
    if closest:
        print(
            f"closest-obstacle min/avg/max: {min(closest):.3f} / {sum(closest) / len(closest):.3f} / {max(closest):.3f}"
        )
    if qwen:
        print(
            f"qwen queries min/avg/max: {min(qwen)} / {sum(qwen) / len(qwen):.1f} / {max(qwen)}"
        )
    print("-" * 60)
    print(
        f"{'run_id':<45} {'success':<8} {'steps':<6} {'reason':<20} {'closest_id':<15} {'closest_dist':<12} {'qwen':<5}"
    )
    for r in rows:
        cd = (
            f"{r['closest_obstacle_dist']:.3f}"
            if r["closest_obstacle_dist"] is not None
            else "-"
        )
        print(
            f"{r['run_id']:<45} {str(r['success']):<8} {str(r['total_steps']):<6} "
            f"{str(r['termination_reason']):<20} {str(r['closest_obstacle_id']):<15} {cd:<12} {str(r['num_qwen_queries']):<5}"
        )


def write_json(rows, path):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = [k for k in rows[0].keys() if k != "top_k_closest"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["top_k_closest"])
        writer.writeheader()
        for row in rows:
            out = {k: row[k] for k in fieldnames}
            out["top_k_closest"] = ";".join(
                f"{oid}:{dist:.3f}" for oid, dist in row["top_k_closest"]
            )
            writer.writerow(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--log-dir", type=Path, help="Path to a single episode log directory"
    )
    src.add_argument(
        "--root-dir",
        type=Path,
        help="Path to logs/ root; parses every episode dir inside it",
    )
    ap.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of closest obstacle distances to report per episode (default: 5)",
    )
    ap.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="Write output to this file instead of stdout (required for csv/json unless --format table)",
    )
    args = ap.parse_args()

    root = args.log_dir if args.log_dir else args.root_dir
    if not root.is_dir():
        sys.exit(f"error: {root} is not a directory")

    episode_dirs = find_episode_dirs(root)
    if not episode_dirs:
        sys.exit(f"error: no episode logs (summary.json) found under {root}")

    rows = [parse_episode(d, args.k) for d in episode_dirs]

    if args.format == "json":
        if args.output:
            write_json(rows, args.output)
            print(f"wrote {len(rows)} episode(s) to {args.output}")
        else:
            print(json.dumps(rows, indent=2))
        return

    if args.format == "csv":
        out_path = args.output or Path("log_summary.csv")
        write_csv(rows, out_path)
        print(f"wrote {len(rows)} episode(s) to {out_path}")
        return

    # table format
    if args.output:
        orig_stdout = sys.stdout
        sys.stdout = open(args.output, "w")
    try:
        if args.log_dir:
            print_episode_table(rows, args.k)
        else:
            print_aggregate_table(rows)
    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = orig_stdout
            print(f"wrote table output to {args.output}")


if __name__ == "__main__":
    main()
