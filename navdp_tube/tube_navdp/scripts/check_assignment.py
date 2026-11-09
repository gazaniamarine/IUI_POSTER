#!/usr/bin/env python3
"""Check the first assignment of the grid tubes (navdp/tube/stt.py, steps 1-2): push NavDP's waypoints clear of the
occupied cells, then group the 16 waypoints of each index j into clusters. No NavDP, no server, no LP.

Scenario: a rock (occupied disc, the robot's size already included) on the way, and 16 stand-in paths that go
straight through it with a sideways spread, as NavDP's clean-path estimate can at the split step.

Checks, per the maths:
  A  after the push, no waypoint is in an occupied cell (and how many are closer than the clearance c after the
     step-limited sweep pulled them back);
  B  every link (k, l) of a cluster: |y_k - y_l|_inf <= link_m and the box between them holds no occupied cell;
  C  every cluster with more than one point: its whole box holds no occupied cell;
  D  next to the rock, the clusters split into a left and a right one (no cluster has points on both sides).

    python scripts/check_assignment.py                   # the first push (nearest), writes check_assignment_nearest.png
    python scripts/check_assignment.py --push smooth     # the nearest push, smoothed inside free boxes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdp.tube.stt import PUSH_MODES, Bev, TubeConfig, clusters, push_clear, push_clearance  # noqa: E402

ROCK = (3.0, 0.0, 1.0)            # forward, left, radius (m), robot's size included


def disc_bev(discs, resolution=0.05, half_m=6.5):
    n = int(round(2 * half_m / resolution))
    centres = -half_m + (np.arange(n) + 0.5) * resolution
    forward, left = np.meshgrid(centres, centres, indexing="ij")
    occupied = np.zeros((n, n), bool)
    for of, ol, r in discs:
        occupied |= (forward - of) ** 2 + (left - ol) ** 2 <= (r + resolution / np.sqrt(2)) ** 2
    return Bev(occupied, resolution, (-half_m, -half_m))


def stand_in_paths(n=16, horizon=24, length=5.5, spread=0.6, seed=0):
    """Straight to (length, l_k), l_k evenly in [-spread, spread], with a little noise: they cross the rock."""
    rng = np.random.default_rng(seed)
    t = (np.arange(horizon) + 1) / horizon
    ends = np.linspace(-spread, spread, n)
    paths = np.stack([np.stack([length * t, end * t], axis=1) for end in ends])
    return paths + rng.normal(0, 0.03, paths.shape)


def turns(paths):
    """[N, H, 2] -> per path: largest turn between consecutive steps (deg) and the number of steps under 2 cm."""
    full = np.concatenate([np.zeros((len(paths), 1, 2)), paths], axis=1)
    d = np.diff(full, axis=1)
    heading = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    turn = np.abs((np.diff(heading, axis=1) + 180) % 360 - 180)
    return turn.max(axis=1), (np.linalg.norm(d, axis=2) < 0.02).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", choices=PUSH_MODES, default="nearest")
    args = parser.parse_args()
    cfg, bev = TubeConfig(push_mode=args.push), disc_bev([ROCK])
    x = stand_in_paths()
    y = push_clear(x, bev, cfg)
    clearance = push_clearance(bev, cfg)
    near = bev.inflated(clearance)

    in_cell = sum(bev.box_blocked(p, p) for p in y.reshape(-1, 2))
    too_close = sum(near.box_blocked(p, p) for p in y.reshape(-1, 2))
    moved = int((np.abs(y - x).max(axis=2) > 1e-9).sum())
    turn, stalls = turns(y)
    print(f"push mode: {args.push}")
    print(f"push: {moved} of {x.shape[0] * x.shape[1]} waypoints moved; clearance c = {clearance:.2f} m")
    print(f"  A  in an occupied cell after the push: {in_cell}   closer than c (after the sweep): {too_close}")
    print(f"  K  largest turn per path (deg): max {turn.max():.0f}, median {np.median(turn):.0f};"
          f"  steps under 2 cm: {int(stalls.sum())}")

    bad_link = bad_box = mixed = 0
    rows = []
    for j in range(y.shape[1]):
        pts = y[:, j]
        groups = clusters(pts, bev, cfg.link_m)
        for g in groups:
            if len(g) > 1 and bev.box_blocked(pts[g].min(axis=0), pts[g].max(axis=0)):
                bad_box += 1
            # links: every member must reach another member through allowed links (connected), checked pairwise
            for a in g:
                if len(g) == 1:
                    break
                ok = any(np.abs(pts[a] - pts[b]).max() <= cfg.link_m
                         and not bev.box_blocked(np.minimum(pts[a], pts[b]), np.maximum(pts[a], pts[b]))
                         for b in g if b != a)
                bad_link += not ok
            beside = abs(pts[g, 0].mean() - ROCK[0]) < ROCK[2]
            if beside and len(g) > 1 and (pts[g, 1] > ROCK[1]).any() and (pts[g, 1] < ROCK[1]).any():
                mixed += 1
        sizes = sorted((len(g) for g in groups), reverse=True)
        rows.append((j, len(groups), sizes, float(pts[:, 0].mean())))
    print(f"  B  members with no allowed link to their cluster: {bad_link}")
    print(f"  C  clusters whose box holds an occupied cell: {bad_box}")
    print(f"  D  clusters beside the rock with points on both sides: {mixed}")
    print("\n  j  forward  clusters  sizes")
    for j, m, sizes, f in rows:
        print(f"{j + 1:3d}  {f:6.2f}   {m:4d}     {sizes}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.add_patch(Circle((ROCK[0], ROCK[1]), ROCK[2], color="0.6"))
    for k in range(len(x)):
        ax.plot(x[k, :, 0], x[k, :, 1], color="0.8", lw=0.8)
        ax.plot(y[k, :, 0], y[k, :, 1], color="tab:blue", lw=0.8)
    colours = plt.cm.tab10.colors
    for j in (3, 9, 13, 17, 23):
        pts = y[:, j]
        for c, g in enumerate(clusters(pts, bev, cfg.link_m)):
            lo, hi = pts[g].min(axis=0), pts[g].max(axis=0)
            ax.add_patch(Rectangle(lo - 0.03, *(hi - lo + 0.06), fill=False, color=colours[c % 10], lw=1.2))
            ax.scatter(pts[g, 0], pts[g, 1], s=8, color=colours[c % 10])
    ax.set_aspect("equal")
    ax.set_xlabel("forward (m)")
    ax.set_ylabel("left (m)")
    ax.set_title(f"push {args.push} - grey: paths at the split, blue: pushed; boxes: clusters at j = 4, 10, 14, 18, 24")
    out = Path.cwd() / f"check_assignment_{args.push}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
