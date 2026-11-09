"""Automatic event marks from one planner run's step log (collect_episodes.py: steps.npz + episode.json).

Works on the logged arrays only, so a finished run can be relabelled with other thresholds:
    python episode_events.py runs/collect/ep_000003/navdp --stuck-steps 50

  stuck         moved < stuck_dist (0.1 m) over stuck_steps plan steps (100 = 20 s at 0.2 s per step)
  off_route     more than off_route_m (1 m) from the oracle route
  collision     clearance (rover centre to the nearest rock / pole) under the robot radius, or a blocked move
  brush         clearance under radius + brush_margin (0.10 m), no collision
  stopped_short the stop rule fired while the rover's edge was more than stop_short_m (0.2 m) from the goal
  wrong_way     distance to go along the route rising for wrong_way_steps (25) steps in a row
Each event: kind, first / last step, time, where (x, y, yaw) and the details asked for per kind.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

KINDS = ("rock", "flag", "slope", "crater")
DEFAULTS = dict(stuck_steps=100, stuck_dist=0.1, off_route_m=1.0, brush_margin=0.10, stop_short_m=0.2,
                wrong_way_steps=25, wrong_way_eps=0.005)


def _runs(mask):
    """Boolean [N] -> [(first, last)] of the True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _wrap_deg(a):
    return float((a + 180.0) % 360.0 - 180.0)


def _side(v):
    return "left" if v >= 0 else "right"


class _Route:
    def __init__(self, pts):
        p = np.asarray(pts, float)
        self.a, self.d = p[:-1], np.diff(p, axis=0)
        self.l = np.maximum(np.linalg.norm(self.d, axis=1), 1e-9)

    def nearest(self, x, y):
        """-> (unit tangent, foot point) of the route point nearest (x, y)."""
        q = np.array([x, y])
        t = np.clip(((q - self.a) * self.d).sum(1) / self.l ** 2, 0, 1)
        foot = self.a + t[:, None] * self.d
        k = int(np.argmin(np.linalg.norm(q - foot, axis=1)))
        return self.d[k] / self.l[k], foot[k]


def _obstacle(st, ep, i):
    lab = int(st["nearest_solid"][i])
    names = ep["obstacles"]
    return {"name": names[lab] if lab >= 0 else None,
            "kind": KINDS[int(st["nearest_solid_kind"][i])] if lab >= 0 else None,
            "clearance_m": round(float(st["clear_solid"][i]), 3),
            "edge_m": round(float(st["clear_solid"][i]) - ep["radius_m"], 3),
            "nearest_hazard": KINDS[int(st["hazard_kind"][i])] if st["hazard_kind"][i] >= 0 else None,
            "hazard_m": round(float(st["clear_hazard"][i]), 3)}


def _where(st, i):
    x, y, yaw = (float(v) for v in st["pose"][i])
    return {"step": int(i), "t": round(float(st["t"][i]), 2), "x": round(x, 3), "y": round(y, 3),
            "yaw_deg": round(math.degrees(yaw), 1)}


def detect(st, ep, **kw):
    """st: the step arrays (dict of np arrays), ep: episode.json. -> list of event dicts."""
    p = {**DEFAULTS, **kw}
    route = _Route(ep["route"])
    centres = ep.get("obstacle_xy", {})
    r = ep["radius_m"]
    n = len(st["t"])
    xy = st["pose"][:, :2]
    events = []

    # stuck: the window [i - N, i] stays within stuck_dist of where it began
    N = int(p["stuck_steps"])
    stuck = np.zeros(n, bool)
    for i in range(N, n):
        stuck[i] = np.max(np.linalg.norm(xy[i - N:i + 1] - xy[i - N], axis=1)) < p["stuck_dist"]
    for a, b in _runs(stuck):
        i = a                                   # detected here; it began N steps before
        x, y, yaw = st["pose"][i]
        ob = _obstacle(st, ep, i)
        tan, foot = route.nearest(x, y)
        e = {"kind": "stuck", "first": int(max(a - N, 0)), "detected": int(a), "last": int(b),
             "duration_s": round(float(st["t"][b] - st["t"][max(a - N, 0)]), 2), **_where(st, i), "obstacle": ob,
             "oracle_turn": _side(st["route_turn_deg"][i]), "oracle_turn_deg": round(float(st["route_turn_deg"][i]), 1),
             "rover_side_of_route": _side(st["cross_track"][i]), "off_route_m": round(float(st["cross_track"][i]), 3),
             "togo_route_m": round(float(st["togo_route"][i]), 3)}
        if ob["name"] in centres:              # the side of the obstacle the route goes round, toward the goal
            ox, oy = centres[ob["name"]]
            tan_o, foot_o = route.nearest(ox, oy)
            rel = np.array([ox, oy]) - foot_o
            on_right = tan_o[0] * rel[1] - tan_o[1] * rel[0] < 0
            e["oracle_passes_obstacle_on"] = "left" if on_right else "right"
        events.append(e)

    # off the route
    for a, b in _runs(np.abs(st["cross_track"]) > p["off_route_m"]):
        i = a
        x, y, yaw = st["pose"][i]
        tan, _ = route.nearest(x, y)
        gx, gy = ep["goal_xy"]
        hit = int(st["facing_obstacle"][i])
        k = int(np.argmax(np.abs(st["cross_track"][a:b + 1]))) + a
        events.append({"kind": "off_route", "first": int(a), "last": int(b),
                       "duration_s": round(float(st["t"][b] - st["t"][a]), 2), **_where(st, i),
                       "side": _side(st["cross_track"][i]),
                       "heading_vs_route_deg": round(_wrap_deg(math.degrees(yaw - math.atan2(tan[1], tan[0]))), 1),
                       "heading_vs_goal_deg": round(_wrap_deg(math.degrees(yaw - math.atan2(gy - y, gx - x))), 1),
                       "facing": {"name": ep["obstacles"][hit] if hit >= 0 else None,
                                  "distance_m": round(float(st["facing_dist"][i]), 2)},
                       "max_off_m": round(float(abs(st["cross_track"][k])), 3), "obstacle": _obstacle(st, ep, i)})

    # collisions and brushes
    coll = (st["clear_solid"] < r) | (st["blocked_ticks"] > 0)
    brush = (st["clear_solid"] < r + p["brush_margin"]) & ~coll
    for kind, mask in (("collision", coll), ("brush", brush)):
        for a, b in _runs(mask):
            k = int(np.argmin(st["clear_solid"][a:b + 1])) + a
            events.append({"kind": kind, "first": int(a), "last": int(b),
                           "duration_s": round(float(st["t"][b] - st["t"][a]), 2), **_where(st, k),
                           "obstacle": _obstacle(st, ep, k), "min_clearance_m": round(float(st["clear_solid"][k]), 3),
                           "blocked_ticks": int(st["blocked_ticks"][a:b + 1].sum())})

    # the stop rule far from the goal
    for a, b in _runs(st["stop_fired"].astype(bool) & (st["goal_edge"] > p["stop_short_m"])):
        events.append({"kind": "stopped_short", "first": int(a), "last": int(b),
                       "duration_s": round(float(st["t"][b] - st["t"][a]), 2), **_where(st, a),
                       "goal_edge_m": round(float(st["goal_edge"][a]), 3),
                       "best_value": round(float(np.nanmax(st["values"][a])), 3),
                       "togo_route_m": round(float(st["togo_route"][a]), 3), "obstacle": _obstacle(st, ep, a)})

    # wrong way: rises by more than eps, step after step (flat steps neither count nor break the run)
    K, eps = int(p["wrong_way_steps"]), p["wrong_way_eps"]
    d = np.diff(st["togo_route"])
    run, first = 0, None
    for i, dd in enumerate(d, start=1):
        if dd > eps:
            run, first = run + 1, (i - 1 if run == 0 else first)
        elif dd < -eps:
            if run >= K:
                events.append(_wrong(st, ep, first, i - 1, run))
            run = 0
    if run >= K:
        events.append(_wrong(st, ep, first, n - 1, run))
    events.sort(key=lambda e: e["first"])
    return events


def _wrong(st, ep, a, b, rises):
    return {"kind": "wrong_way", "first": int(a), "last": int(b), "rising_steps": int(rises),
            "duration_s": round(float(st["t"][b] - st["t"][a]), 2), **_where(st, a),
            "togo_gain_m": round(float(st["togo_route"][b] - st["togo_route"][a]), 3),
            "heading_vs_goal_deg": round(_wrap_deg(math.degrees(
                st["pose"][a][2] - math.atan2(ep["goal_xy"][1] - st["pose"][a][1], ep["goal_xy"][0] - st["pose"][a][0]))), 1),
            "obstacle": _obstacle(st, ep, a)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a planner folder of an episode (has steps.npz; episode.json one up)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    st = dict(np.load(os.path.join(a.run, "steps.npz")))
    with open(os.path.join(a.run, "..", "episode.json")) as f:
        ep = json.load(f)
    ev = detect(st, ep, **{k: getattr(a, k) for k in DEFAULTS})
    with open(os.path.join(a.run, "events.json")) as f:
        old = json.load(f)
    old.update(events=ev, event_params={k: getattr(a, k) for k in DEFAULTS})
    with open(os.path.join(a.run, "events.json"), "w") as f:
        json.dump(old, f, indent=1)
    from collections import Counter
    print(dict(Counter(e["kind"] for e in ev)))


if __name__ == "__main__":
    main()
