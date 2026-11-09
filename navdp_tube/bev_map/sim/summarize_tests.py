#!/usr/bin/env python3
"""Table of finished sim test runs (metrics.json written by run_tube_sim.py).

    python summarize_tests.py                      # every run in runs/tests/
    python summarize_tests.py runs/tests/a runs/tests/b   # only these run folders
    python summarize_tests.py --runs               # also one line per run

Grouped by test and mode: how many runs reached the object without hitting anything, and the averages.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(dirs):
    paths = [os.path.join(d, "metrics.json") for d in dirs] if dirs else \
        glob.glob(os.path.join(HERE, "runs", "tests", "*", "metrics.json"))
    rows = []
    for p in sorted(paths, key=os.path.getmtime):
        try:
            m = json.load(open(p))
        except (OSError, ValueError):
            continue
        m["_run"] = os.path.basename(os.path.dirname(p))
        m.setdefault("test", m.get("scenario", "?"))
        rows.append(m)
    return rows


def fmt(x, spec="5.2f"):
    return format(x, spec) if isinstance(x, (int, float)) else "  -  "


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def ok(m):
    """Success = reached the object with no physical collision."""
    return bool(m.get("reached")) and m.get("collision_s", 0.0) == 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--runs", action="store_true", help="also one line per run")
    args = ap.parse_args()
    rows = load(args.dirs)
    if not rows:
        print("no finished runs")
        return

    if args.runs:
        print(f"\n{'run':34s} {'mode':24s} {'end':8s} {'time':>6s} {'path':>5s} {'edge':>5s} {'clear':>6s} "
              f"{'coll_s':>6s} {'cert':>7s}  collided with")
        for m in rows:
            print(f"{m['_run'][:34]:34s} {m['mode']:24s} {m.get('end', str(m['reached'])):8s} {m['time_s']:6.1f} "
                  f"{fmt(m.get('path_length_m'), '5.1f')} {fmt(m.get('final_edge_dist_m'))} "
                  f"{fmt(m.get('min_clearance_excl_goal_object_m'), '6.3f')} {fmt(m.get('collision_s'), '6.1f')} "
                  f"{(str(m['plans_certified']) + '/' + str(m['plans'])) if 'plans_certified' in m else '-':>7s}  {','.join(m.get('collided_with', []))}")

    groups = defaultdict(list)
    for m in rows:
        groups[(m["test"], m["mode"], m.get("controller", ""))].append(m)
    print(f"\n{'test':12s} {'mode':24s} {'ctrl':5s} {'success':>7s} {'stuck':>5s} {'tmout':>5s} {'coll':>4s} "
          f"{'time':>6s} {'path/line':>9s} {'clear':>6s} {'cert%':>5s} {'in%':>4s} {'V':>7s} {'map P/R':>9s}")
    for (test, mode, ctrl), ms in groups.items():
        n = len(ms)
        succ = sum(ok(m) for m in ms)
        stuck = sum(m.get("end") == "stuck" for m in ms)
        tmout = sum(m.get("end") == "timeout" for m in ms)
        coll = sum(m.get("collision_s", 0) > 0 for m in ms)
        t_ok = mean([m["time_s"] for m in ms if ok(m)])
        ratio = mean([m["path_length_m"] / m["straight_line_m"] for m in ms
                      if ok(m) and m.get("straight_line_m") and m.get("path_length_m") is not None])
        clear = min((m.get("min_clearance_excl_goal_object_m") for m in ms
                     if isinstance(m.get("min_clearance_excl_goal_object_m"), (int, float))), default=None)
        cert = mean([100.0 * m["plans_certified"] / m["plans"] for m in ms if m.get("plans") and "plans_certified" in m])
        inc = mean([100.0 * m["chosen_in_cells_share"] for m in ms if m.get("chosen_in_cells_share") is not None])
        cv = mean([m.get("chosen_v_mean") for m in ms])
        mp, mr = mean([m.get("map_precision") for m in ms]), mean([m.get("map_recall") for m in ms])
        print(f"{test:12s} {mode:24s} {ctrl:5s} {f'{succ}/{n}':>7s} {stuck:5d} {tmout:5d} {coll:4d} "
              f"{fmt(t_ok, '6.1f')} {fmt(ratio, '9.2f')} {fmt(clear, '6.3f')} {fmt(cert, '5.0f')} {fmt(inc, '4.0f')} "
              f"{fmt(cv, '7.4f')} "
              f"{fmt(mp, '4.2f')}/{fmt(mr, '4.2f')}")
    print("\nsuccess = reached the object (rover within reach_edge of its surface) with no physical collision.\n"
          "stuck / tmout = runs that ended without reaching it; coll = runs that touched something.\n"
          "time, path/line = averages over successful runs (path/line = driven path / straight line; 1.0 = straight).\n"
          "clear = closest the rover body came to any obstacle other than the goal object, over all runs (> ~0.05 m"
          " is good).\ncert% = share of plans whose path the tube certified safe. map P/R = BEV map precision /"
          " recall against the true map at the end.\nin% = share of plans whose chosen path had a waypoint in an"
          " occupied (grown) cell; V = its mean depth in the obstacles (1/2 sum dist^2, 0 = clear). Both show whether"
          " NavDP keeps the paths clear after --guide-until.")


if __name__ == "__main__":
    main()
