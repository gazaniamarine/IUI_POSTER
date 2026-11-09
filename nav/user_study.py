"""User study: language-guided vs human-adjusted navigation, per participant.

Protocol (shown to every participant at the start, saved in the report):
  1. Language task  -- the participant types an instruction ("go to the
     yellow flag") and the uncertainty-aware VLN drives itself. They take
     over (WASD) only when the rover asks for help.
  2. Human-adjusted -- same kind of instruction, but the participant
     collaborates: they may take over and adjust the rover whenever they
     think it helps, then hand back with "Resume autonomy".
  3. For both: every prompt is logged; every human intervention (who
     triggered it, when, how long, how far it drove); path length and time to
     reach the goal, split autonomy vs human; and the oracle (shortest
     collision-free) path is computed automatically for comparison.

Everything is recorded from the outside: StudyRecorder samples
RoverController.snapshot() at 10 Hz and gets each typed prompt from the GUI's
command hook. Nothing here steers the rover (the optional "reset at trial
start" uses the GUI's own Reset).

Output: nav/study_logs/<participant>_<time>/
  events.jsonl      every prompt / hand-over / trial event as it happens
  prompts.csv       every prompt the participant typed (+ system hand-over prompts)
  trial_<n>.json    full per-trial record (metrics, interventions, goals)
  trial_<n>_traj.csv, trial_<n>.png   trajectory (autonomy/human) vs oracle
  report.md / report.json / trials.csv   the final report
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional

STUDY_DIR = Path(__file__).resolve().parent / "study_logs"
SAMPLE_MS = 100
CONDITIONS = [
    ("language", "Language task (autonomy drives)"),
    ("human_guided", "Human-guided (participant drives)"),
    ("uncertainty_0.25", "Uncertainty 0.25, go to the yellow flag"),
]
N_LANGUAGE = 2
YELLOW_THRESHOLD = 0.25
# The fixed protocol, in order: (label, condition, index)
PROTOCOL = ([(f"L{i}", "language", i) for i in range(1, N_LANGUAGE + 1)]
            + [(f"H{i}", "human_guided", i) for i in range(1, N_LANGUAGE + 1)]
            + [("U1", "uncertainty_0.25", 1)])
BRIEFING = """Welcome, and thank you for taking part.

This rover runs an uncertainty-aware, obstacle-avoiding vision-language
navigation model. You give it a task in plain language, and you can work
together with it. There are three parts.

PART 1 - LANGUAGE TASK (2 trials)
   Each trial, type ONE instruction of your choice in the Taskwork box
   (e.g. "go to the red flag", "go to the blue flag then return to home")
   and press Send. The rover drives itself. Only take over (WASD / arrow
   keys) when the screen says HUMAN ACTION NEEDED, then press "Resume
   autonomy" once the goal is in view again.

PART 2 - HUMAN-GUIDED (2 trials)
   The rover is put back at the same start positions as in Part 1. This
   time drive it YOURSELF (WASD / arrow keys) to the same goal as the
   matching Part 1 trial.

PART 3 - UNCERTAINTY TASK (1 trial)
   The rover starts facing the yellow flag with a stricter uncertainty
   limit (0.25). Type "go to the yellow flag" and press Send. Help the rover
   whenever it asks (HUMAN ACTION NEEDED), then resume autonomy.

For every trial we record your prompts, when and how long you take over,
the path length and the time to reach the goal, and compare them with the
shortest possible (oracle) path."""

STEP_INSTRUCTIONS = {
    "language": "Type ONE instruction of your choice in Taskwork and press Send. "
                "Take over only when HUMAN ACTION NEEDED appears.",
    "human_guided": "Drive the rover YOURSELF (WASD / arrows) to: {goals}",
    "uncertainty_0.25": "Type \"go to the yellow flag\" in Taskwork and press Send. "
                       "Help when HUMAN ACTION NEEDED appears, then Resume autonomy.",
}

_FLAG_COLORS = ("red", "green", "blue", "yellow", "purple")


def _now() -> float:
    return time.time()


def _fmt(v, nd=2, unit=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


class StudyRecorder:
    """Participant session: trials, prompts, metrics, report."""

    def __init__(self, controller, *, split_fn: Callable[[str], List[str]], yard_glb: Optional[str],
                 success_radius: float = 3.0, out_root: Path = STUDY_DIR, robot_radius: float = 0.35):
        self.controller = controller
        self.split_fn = split_fn
        self.success_radius = float(success_radius)
        self.out_root = Path(out_root)
        self.participant: Optional[str] = None
        self.dir: Optional[Path] = None
        self.trials: List[dict] = []
        self.trial: Optional[dict] = None
        self.prompts: List[dict] = []
        self._last = None  # previous sample (t, X, Z, yaw, human)
        self._iv_open: Optional[dict] = None
        self.looking = False  # a 360-degree look-around is running (not counted as driving)
        self._oracle = None
        self.objects: dict = {}
        self.home_xz = (float(getattr(controller, "start_x", 0.0)), float(getattr(controller, "start_z", 0.0)))
        self.max_speed = float(getattr(controller, "max_forward_speed", 0.5) or 0.5)
        try:
            if yard_glb:
                from nav.study_oracle import load_yard_oracle

                self._oracle, self.objects = load_yard_oracle(yard_glb, robot_radius=robot_radius)
        except Exception as e:  # never break the GUI over the oracle
            print(f"[study] oracle unavailable: {type(e).__name__}: {e}")
        if self._oracle is None:
            from nav.study_oracle import StraightLineOracle

            self._oracle = StraightLineOracle()
        self.oracle_kind = "geodesic (boulders inflated)" if self.objects else "straight line (no obstacle ground truth)"

    # ── session ────────────────────────────────────────────────────────────
    def start_participant(self, pid: str) -> Path:
        pid = re.sub(r"[^A-Za-z0-9_-]+", "_", pid.strip()) or "anon"
        self.participant = pid
        self.dir = self.out_root / f"{pid}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trials, self.prompts, self.trial = [], [], None
        (self.dir / "briefing.txt").write_text(BRIEFING + "\n")
        self._event("participant_started", participant=pid, oracle=self.oracle_kind,
                     success_radius_m=self.success_radius)
        self._event("system_prompt", text="briefing shown", briefing=BRIEFING)
        return self.dir

    def _event(self, event_name: str, **fields) -> None:
        if self.dir is None:
            return
        rec = {"t": round(_now(), 3), "time": time.strftime("%H:%M:%S"), "event": event_name,
               "trial": self.trial["n"] if self.trial else None, **fields}
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ── prompts ────────────────────────────────────────────────────────────
    def log_prompt(self, text: str, source: str = "participant") -> None:
        """Every instruction typed in Taskwork (and every hand-over message
        the system showed the participant)."""
        if self.dir is None:
            return
        rec = {"time": time.strftime("%H:%M:%S"), "t": _now(), "source": source, "text": text,
               "trial": self.trial["n"] if self.trial else None,
               "condition": self.trial["condition"] if self.trial else None}
        self.prompts.append(rec)
        self._event("prompt", source=source, text=text)
        tr = self.trial
        if tr is not None and source == "participant":
            if tr["t_first_prompt"] is None:
                tr["t_first_prompt"] = rec["t"]
            tr["prompts"].append({"t_rel": round(rec["t"] - tr["t_start"], 2), "text": text})
            if tr["preset_goals"]:
                return  # goals fixed by the protocol (human-guided)
            for ins in self.split_fn(text):
                g = self._resolve_goal(ins)
                if g is not None:
                    tr["goals"].append(g)
            self._event("goals", goals=[{k: g[k] for k in ("label", "xz")} for g in tr["goals"]])

    def _resolve_goal(self, ins: str) -> Optional[dict]:
        t = ins.lower()
        try:
            from nav.vlm_point_navigator import parse_simple_command

            simple = parse_simple_command(ins)
        except Exception:
            simple = None
        if simple is not None:
            if simple.get("action") == "home":
                return {"label": "home", "xz": self.home_xz, "instruction": ins, "t_reached": None}
            return None  # "go forward 2 m", "turn left": no destination
        if "flag" in t:
            for c in _FLAG_COLORS:
                if re.search(rf"\b{c}\b", t):
                    xz = self.objects.get(f"flag_{c}") or self._env_flag(c)
                    return {"label": f"{c} flag", "xz": xz, "instruction": ins, "t_reached": None}
            # "a flag": whichever flag the rover actually goes to (resolved on arrival)
            return {"label": "any flag", "xz": None, "any_flag": True, "instruction": ins, "t_reached": None}
        if re.search(r"\b(home|base|start)\b", t):
            return {"label": "home", "xz": self.home_xz, "instruction": ins, "t_reached": None}
        for name, xz in self.objects.items():
            if name.split("_")[0] in t and not name.startswith("flag"):
                return {"label": name, "xz": xz, "instruction": ins, "t_reached": None}
        return {"label": ins[:40], "xz": None, "instruction": ins, "t_reached": None}  # unknown target

    def _env_flag(self, color):
        env = getattr(self.controller, "_live_env", None)
        for f in getattr(env, "flags", []) or []:
            if getattr(f, "color", "") == color:
                return (float(f.x), float(f.z))
        return None

    def _flags(self):
        out = {n.replace("flag_", "") + " flag": xz for n, xz in self.objects.items() if n.startswith("flag")}
        env = getattr(self.controller, "_live_env", None)
        for f in getattr(env, "flags", []) or []:
            out.setdefault(f"{f.color} flag", (float(f.x), float(f.z)))
        return out

    # ── trials ─────────────────────────────────────────────────────────────
    def start_trial(self, condition: str, label: Optional[str] = None, goals: Optional[list] = None,
                    note: str = "") -> dict:
        """goals: preset goals (human-guided trials copy their language
        trial's); otherwise they come from the participant's prompts."""
        if self.trial is not None:
            self.end_trial("superseded by a new trial")
        d = self.controller.snapshot()
        p = d.pose
        n = len(self.trials) + 1
        self.trial = {
            "n": n, "label": label or f"T{n}", "participant": self.participant, "condition": condition,
            "note": note, "preset_goals": goals is not None,
            "uncertainty_threshold": float(getattr(self.controller, "uncertainty_cov_threshold", float("nan"))),
            "t_start": _now(), "t_first_prompt": None, "t_first_move": None, "t_end": None,
            "start_xz": (float(p.x), float(p.z)) if p is not None else None,
            "start_pose": (float(p.x), float(p.z), float(p.yaw)) if p is not None else None,
            "prompts": [], "goals": [], "interventions": [], "samples": [],
            "path_m": 0.0, "path_auto_m": 0.0, "path_human_m": 0.0,
            "time_auto_s": 0.0, "time_human_s": 0.0, "turn_auto_rad": 0.0, "turn_human_rad": 0.0,
            "end_reason": None,
        }
        if goals is not None:
            self.trial["goals"] = [dict(g, t_reached=None) for g in goals]
        self._last = None
        self._iv_open = None
        self._event("trial_started", condition=condition, start_xz=self.trial["start_xz"])
        return self.trial

    def sample(self) -> Optional[str]:
        """Called every SAMPLE_MS by the window. Returns a message when the
        trial auto-ends."""
        tr = self.trial
        if tr is None:
            return None
        d = self.controller.snapshot()
        p = d.pose
        if p is None:
            return None
        t = _now()
        if tr["start_xz"] is None:
            tr["start_xz"] = (float(p.x), float(p.z))
            tr["start_pose"] = (float(p.x), float(p.z), float(p.yaw))
        if tr["t_first_move"] is None and math.hypot(p.x - tr["start_xz"][0], p.z - tr["start_xz"][1]) > 0.05:
            tr["t_first_move"] = t
        human = bool(d.intervention_active) or d.mode == "manual" or tr["condition"] == "human_guided"
        tr["samples"].append((round(t - tr["t_start"], 2), round(float(p.x), 3), round(float(p.z), 3),
                              round(float(p.yaw), 4), int(human), int(bool(d.intervention_active)),
                              int(self.looking)))
        if self.looking:
            tr["look_s"] = tr.get("look_s", 0.0) + ((t - self._last[0]) if self._last else 0.0)
            self._last = (t, float(p.x), float(p.z), float(p.yaw), human)
            return None
        if self._last is not None:
            lt, lx, lz, lyaw, _ = self._last
            dt = t - lt
            ds = math.hypot(p.x - lx, p.z - lz)
            if ds > 3.0:  # reset/teleport, not driving
                ds = 0.0
            dth = abs((p.yaw - lyaw + math.pi) % (2 * math.pi) - math.pi)
            moving = ds > 0.002 or dth > 0.002
            tr["path_m"] += ds
            key = "human" if human else "auto"
            tr[f"path_{key}_m"] += ds
            tr[f"turn_{key}_rad"] += dth
            if moving or human:
                tr[f"time_{key}_s"] += dt
        self._last = (t, float(p.x), float(p.z), float(p.yaw), human)

        # interventions (hand-over to the human): open / close segments
        if d.intervention_active and self._iv_open is None:
            reason = d.intervention_reason or ""
            kind = "system" if "human action needed" in reason.lower() else "operator"
            self._iv_open = {"t_start_rel": round(t - tr["t_start"], 2), "reason": reason, "kind": kind,
                             "start_xz": (float(p.x), float(p.z)), "path_m": 0.0, "_last": (p.x, p.z),
                             "uncertainty_at_start": round(float(d.uncertainty_value), 3),
                             "target_uncertainty_at_start": d.target_uncertainty}
            self._event("intervention_started", kind=kind, reason=reason)
            if kind == "system":
                self.log_prompt("HUMAN ACTION NEEDED - drive with WASD (" + reason + ")", source="system")
        if self._iv_open is not None:
            lx, lz = self._iv_open["_last"]
            ds = math.hypot(p.x - lx, p.z - lz)
            self._iv_open["path_m"] += ds if ds <= 3.0 else 0.0
            self._iv_open["_last"] = (p.x, p.z)
            if not d.intervention_active:
                self._close_intervention(t)

        # goal arrival (in order)
        for g in tr["goals"]:
            if g["t_reached"] is not None:
                continue
            if g.get("any_flag"):
                for label, xz in self._flags().items():
                    if math.hypot(p.x - xz[0], p.z - xz[1]) <= self.success_radius:
                        g.update(label=label, xz=xz, t_reached=t)
                        self._event("goal_reached", goal=label)
                        break
            elif g["xz"] is not None and math.hypot(p.x - g["xz"][0], p.z - g["xz"][1]) <= self.success_radius:
                g["t_reached"] = t
                self._event("goal_reached", goal=g["label"])
            break  # only the current goal
        goals = tr["goals"]
        if goals and all(g["t_reached"] is not None for g in goals):
            settle = t - max(g["t_reached"] for g in goals)
            if (not self.controller.vlm_navigation_active() and not d.intervention_active) or settle > 5.0:
                self.end_trial("all goals reached")
                return f"trial {tr['n']} done: all goals reached"
        return None

    def _close_intervention(self, t):
        iv = self._iv_open
        tr = self.trial
        iv["t_end_rel"] = round(t - tr["t_start"], 2)
        iv["duration_s"] = round(iv["t_end_rel"] - iv["t_start_rel"], 2)
        iv["path_m"] = round(iv["path_m"], 2)
        iv.pop("_last", None)
        tr["interventions"].append(iv)
        self._event("intervention_ended", kind=iv["kind"], duration_s=iv["duration_s"], path_m=iv["path_m"])
        self._iv_open = None

    def end_trial(self, reason: str = "ended by experimenter") -> Optional[dict]:
        tr = self.trial
        if tr is None:
            return None
        t = _now()
        if self._iv_open is not None:
            self._close_intervention(t)
        tr["t_end"] = t
        tr["end_reason"] = reason
        self.trial = None
        self._finalize(tr)
        self.trials.append(tr)
        self._write_trial(tr)
        self._event("trial_ended", reason=reason, success=tr["metrics"]["success"])
        self.write_report()
        return tr

    # ── metrics ────────────────────────────────────────────────────────────
    def _finalize(self, tr: dict) -> None:
        # clock starts at the first prompt (language / uncertainty trials) or,
        # with no prompt (human-guided), when the rover first moves
        t0 = tr["t_first_prompt"] or tr["t_first_move"] or tr["t_start"]
        tr["clock_start"] = "first prompt" if tr["t_first_prompt"] else (
            "first movement" if tr["t_first_move"] else "trial start")
        goals = tr["goals"]
        resolved = [g for g in goals if g["xz"] is not None]
        success = bool(goals) and all(g["t_reached"] is not None for g in goals)
        t_goal = (max(g["t_reached"] for g in goals) - t0) if success else None
        oracle = None
        if tr["start_xz"] is not None and resolved and len(resolved) == len(goals):
            try:
                oracle = self._oracle.plan_sequence(tr["start_xz"], [tuple(g["xz"]) for g in goals],
                                                    self.success_radius)
            except Exception as e:
                print(f"[study] oracle failed: {e}")
        oracle_len = oracle["length_m"] if oracle and oracle.get("reachable") else None
        # path / time up to reaching the (last) goal: cut the samples there
        if success:
            cut = max(g["t_reached"] for g in goals) - tr["t_start"]
        else:
            cut = None
        # time split within the task window [first prompt, goal reached or end]
        t_end_rel = (cut if cut is not None else tr["t_end"] - tr["t_start"])
        t0_rel = t0 - tr["t_start"]
        auto_t = human_t = 0.0
        path_task = 0.0
        s = tr["samples"]
        for a, b in zip(s, s[1:]):
            if b[0] <= t0_rel or a[0] >= t_end_rel:
                continue
            dt = b[0] - a[0]
            if len(b) > 6 and b[6]:
                continue  # 360-degree look-around: not driving
            if b[4]:
                human_t += dt
            else:
                auto_t += dt
            ds = math.hypot(b[1] - a[1], b[2] - a[2])
            path_task += ds if ds <= 3.0 else 0.0
        total_t = auto_t + human_t
        ivs = tr["interventions"]
        per_goal = []
        for g in goals:
            per_goal.append({"goal": g["label"], "instruction": g["instruction"],
                             "xz": None if g["xz"] is None else [round(v, 2) for v in g["xz"]],
                             "reached": g["t_reached"] is not None,
                             "time_to_reach_s": None if g["t_reached"] is None else round(g["t_reached"] - t0, 2)})
        d = self.controller.snapshot()
        final_dist = None
        if d.pose is not None and goals and goals[-1]["xz"] is not None:
            final_dist = math.hypot(d.pose.x - goals[-1]["xz"][0], d.pose.z - goals[-1]["xz"][1])
        m = {
            "success": success,
            "time_to_goal_s": None if t_goal is None else round(t_goal, 2),
            "task_time_s": round(total_t, 2),
            "time_autonomy_s": round(auto_t, 2),
            "time_human_s": round(human_t, 2),
            "human_time_share": round(human_t / total_t, 3) if total_t > 0 else None,
            "path_length_m": round(path_task, 2),
            "path_total_incl_after_goal_m": round(tr["path_m"], 2),
            "path_autonomy_m": round(tr["path_auto_m"], 2),
            "path_human_m": round(tr["path_human_m"], 2),
            "human_path_share": round(tr["path_human_m"] / tr["path_m"], 3) if tr["path_m"] > 0 else None,
            "oracle_path_m": None if oracle_len is None else round(oracle_len, 2),
            "oracle_time_s": None if oracle_len is None else round(oracle_len / self.max_speed, 2),
            "path_ratio": (round(path_task / oracle_len, 3) if oracle_len else None),
            "spl": (round(float(success) * oracle_len / max(path_task, oracle_len), 3) if oracle_len else None),
            "time_ratio": (round(t_goal / (oracle_len / self.max_speed), 3) if (oracle_len and t_goal) else None),
            "n_prompts": len(tr["prompts"]),
            "n_interventions": len(ivs),
            "n_system_handovers": sum(iv["kind"] == "system" for iv in ivs),
            "n_operator_takeovers": sum(iv["kind"] == "operator" for iv in ivs),
            "intervention_time_s": round(sum(iv["duration_s"] for iv in ivs), 2),
            "intervention_path_m": round(sum(iv["path_m"] for iv in ivs), 2),
            "first_intervention_s": (round(ivs[0]["t_start_rel"] - t0_rel, 2) if ivs else None),
            "final_distance_to_goal_m": None if final_dist is None else round(final_dist, 2),
            "n_look_arounds": tr.get("n_look_arounds", 0),
            "look_around_s": round(tr.get("look_s", 0.0), 2),
        }
        tr["metrics"] = m
        tr["per_goal"] = per_goal
        tr["oracle"] = None if oracle is None else {
            "kind": self.oracle_kind, "length_m": oracle_len,
            "path_xz": [[round(a, 2), round(b, 2)] for a, b in oracle["path_xz"]]}

    # ── output ─────────────────────────────────────────────────────────────
    def _write_trial(self, tr: dict) -> None:
        n = tr["n"]
        rec = {k: v for k, v in tr.items() if k != "samples"}
        rec["goals"] = [{k: v for k, v in g.items()} for g in tr["goals"]]
        with open(self.dir / f"trial_{n}.json", "w") as f:
            json.dump(rec, f, indent=2, default=str)
        with open(self.dir / f"trial_{n}_traj.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "x", "z", "yaw_rad", "human_driving", "intervention", "look_around"])
            w.writerows(tr["samples"])
        try:
            self._plot(tr, self.dir / f"trial_{n}.png")
        except Exception as e:
            print(f"[study] plot failed: {e}")

    def _plot(self, tr: dict, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        for name, (x, z) in self.objects.items():
            if name.startswith("flag"):
                c = name.split("_")[1]
                ax.plot(x, z, marker="^", color=c if c != "yellow" else "gold", ms=10, mec="k")
            else:
                ax.add_patch(plt.Circle((x, z), 1.0, color="0.45", alpha=0.6))
        s = tr["samples"]
        for a, b in zip(s, s[1:]):
            if math.hypot(b[1] - a[1], b[2] - a[2]) > 3.0:
                continue
            ax.plot([a[1], b[1]], [a[2], b[2]], color="#f97316" if b[4] else "#2563eb", lw=2)
        if tr.get("oracle"):
            o = tr["oracle"]["path_xz"]
            ax.plot([q[0] for q in o], [q[1] for q in o], "k--", lw=1.3, label=f"oracle {tr['oracle']['length_m']:.1f} m")
        for iv in tr["interventions"]:
            ax.plot(*iv["start_xz"], "rx", ms=9, mew=2)
        for g in tr["goals"]:
            if g["xz"] is not None:
                ax.add_patch(plt.Circle(g["xz"], self.success_radius, fill=False, ls=":", color="green"))
        if tr["start_xz"]:
            ax.plot(*tr["start_xz"], "ko", ms=6)
        ax.plot([], [], color="#2563eb", lw=2, label=f"autonomy {tr['metrics']['path_autonomy_m']:.1f} m")
        ax.plot([], [], color="#f97316", lw=2, label=f"human {tr['metrics']['path_human_m']:.1f} m")
        ax.plot([], [], "rx", label="intervention")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        m = tr["metrics"]
        ax.set_title(f"{tr['participant']} {tr.get('label', tr['n'])} [{tr['condition']}] "
                     f"{'SUCCESS' if m['success'] else 'FAIL'}  t={_fmt(m['time_to_goal_s'], 1, 's')}", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def write_report(self) -> Optional[Path]:
        if self.dir is None:
            return None
        with open(self.dir / "prompts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "trial", "condition", "source", "text"])
            for p in self.prompts:
                w.writerow([p["time"], p["trial"], p["condition"], p["source"], p["text"]])
        cols = ["n", "condition", "success", "time_to_goal_s", "task_time_s", "time_autonomy_s", "time_human_s",
                "human_time_share", "path_length_m", "path_autonomy_m", "path_human_m", "human_path_share",
                "oracle_path_m", "oracle_time_s", "path_ratio", "spl", "time_ratio", "n_prompts",
                "n_interventions", "n_system_handovers", "n_operator_takeovers", "intervention_time_s",
                "intervention_path_m", "first_intervention_s", "final_distance_to_goal_m"]
        with open(self.dir / "trials.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["participant", "label", "start_pose", "uncertainty_threshold", "goals", "prompts"] + cols)
            for tr in self.trials:
                m = tr["metrics"]
                w.writerow([tr["participant"], tr.get("label"),
                            None if tr.get("start_pose") is None else " ".join(f"{v:.2f}" for v in tr["start_pose"]),
                            tr.get("uncertainty_threshold"), " -> ".join(g["label"] for g in tr["goals"]),
                            " | ".join(p["text"] for p in tr["prompts"])]
                           + [tr["n"], tr["condition"]] + [m.get(c) for c in cols[2:]])
        summary = self._summary()
        with open(self.dir / "report.json", "w") as f:
            json.dump({"participant": self.participant, "oracle": self.oracle_kind,
                       "success_radius_m": self.success_radius, "summary_by_condition": summary,
                       "trials": [{k: v for k, v in tr.items() if k != "samples"} for tr in self.trials],
                       "prompts": self.prompts}, f, indent=2, default=str)
        (self.dir / "report.md").write_text(self._markdown(summary))
        return self.dir / "report.md"

    def _summary(self) -> dict:
        keys = ["time_to_goal_s", "path_length_m", "oracle_path_m", "path_ratio", "spl", "time_ratio",
                "human_time_share", "human_path_share", "n_interventions", "n_system_handovers",
                "n_operator_takeovers", "intervention_time_s", "n_prompts"]
        out = {}
        for cond, _ in CONDITIONS:
            trs = [t for t in self.trials if t["condition"] == cond]
            if not trs:
                continue
            s = {"trials": len(trs), "success_rate": round(sum(t["metrics"]["success"] for t in trs) / len(trs), 3)}
            for k in keys:
                vals = [t["metrics"][k] for t in trs if t["metrics"].get(k) is not None]
                if vals:
                    s[k] = {"mean": round(statistics.mean(vals), 3),
                            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0}
            out[cond] = s
        return out

    def _markdown(self, summary: dict) -> str:
        L = [f"# User study report: participant {self.participant}", "",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- Trials: {len(self.trials)}",
             f"- Oracle: {self.oracle_kind}; success = within {self.success_radius:.1f} m of each goal, in order",
             f"- Oracle time = oracle length / max speed ({self.max_speed:.2f} m/s)",
             "- Time to goal is measured from the first prompt of the trial.", ""]
        if summary:
            L += ["## Summary by condition", "",
                  "| metric | " + " | ".join(summary) + " |",
                  "|---|" + "---|" * len(summary)]
            rows = ["trials", "success_rate", "time_to_goal_s", "path_length_m", "oracle_path_m", "path_ratio",
                    "spl", "time_ratio", "human_time_share", "human_path_share", "n_interventions",
                    "n_system_handovers", "n_operator_takeovers", "intervention_time_s", "n_prompts"]
            for r in rows:
                cells = []
                for c in summary:
                    v = summary[c].get(r)
                    if isinstance(v, dict):
                        cells.append(f"{v['mean']:.2f} ± {v['std']:.2f}")
                    else:
                        cells.append(_fmt(v))
                L.append(f"| {r} | " + " | ".join(cells) + " |")
            L.append("")
        pairs = []
        for i in range(1, N_LANGUAGE + 1):
            lt = next((t for t in self.trials if t.get("label") == f"L{i}"), None)
            ht = next((t for t in self.trials if t.get("label") == f"H{i}"), None)
            if lt and ht:
                pairs.append((i, lt, ht))
        if pairs:
            L += ["## Language (autonomy) vs human-guided, same start and goal", "",
                  "| pair | goal | oracle (m) | L time (s) | H time (s) | L path (m) | H path (m) "
                  "| L SPL | H SPL | L interventions | L human share of time |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for i, lt, ht in pairs:
                a, b = lt["metrics"], ht["metrics"]
                L.append(f"| {i} | {' → '.join(g['label'] for g in lt['goals']) or '-'} | {_fmt(a['oracle_path_m'], 1)} "
                         f"| {_fmt(a['time_to_goal_s'], 1)} | {_fmt(b['time_to_goal_s'], 1)} "
                         f"| {_fmt(a['path_length_m'], 1)} | {_fmt(b['path_length_m'], 1)} "
                         f"| {_fmt(a['spl'])} | {_fmt(b['spl'])} | {a['n_interventions']} | {_fmt(a['human_time_share'])} |")
            L.append("")
        L += ["## Trials", "",
              "| # | condition | goals | success | time to goal (s) | path (m) | oracle (m) | path ratio | SPL "
              "| auto / human time (s) | auto / human path (m) | interventions (system / operator) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for tr in self.trials:
            m = tr["metrics"]
            L.append(f"| {tr.get('label', tr['n'])} | {tr['condition']} | {' → '.join(g['label'] for g in tr['goals']) or '-'} "
                     f"| {'yes' if m['success'] else 'no'} | {_fmt(m['time_to_goal_s'], 1)} | {_fmt(m['path_length_m'], 1)} "
                     f"| {_fmt(m['oracle_path_m'], 1)} | {_fmt(m['path_ratio'])} | {_fmt(m['spl'])} "
                     f"| {m['time_autonomy_s']:.1f} / {m['time_human_s']:.1f} "
                     f"| {m['path_autonomy_m']:.1f} / {m['path_human_m']:.1f} "
                     f"| {m['n_interventions']} ({m['n_system_handovers']} / {m['n_operator_takeovers']}) |")
        L.append("")
        for tr in self.trials:
            m = tr["metrics"]
            sp = tr.get("start_pose")
            L += [f"### {tr.get('label', tr['n'])} ({tr['condition']}) - ended: {tr['end_reason']}", "",
                  f"Start pose: {'n/a' if sp is None else f'x={sp[0]:.2f}, z={sp[1]:.2f}, yaw={math.degrees(sp[2]):.0f} deg'}; "
                  f"uncertainty threshold {tr.get('uncertainty_threshold')}; clock from {tr.get('clock_start', '-')}"
                  + (f"; {tr['note']}" if tr.get("note") else ""), ""]
            L.append("Prompts:")
            for p in tr["prompts"]:
                L.append(f"- t+{p['t_rel']:.1f}s: \"{p['text']}\"")
            if not tr["prompts"]:
                L.append("- (none)")
            L.append("")
            for g in tr["per_goal"]:
                L.append(f"- Goal **{g['goal']}** at {g['xz']}: "
                         + (f"reached in {g['time_to_reach_s']:.1f} s" if g["reached"] else "NOT reached"))
            if tr["interventions"]:
                L.append("")
                L.append("Interventions:")
                for i, iv in enumerate(tr["interventions"], 1):
                    L.append(f"- {i}. {iv['kind']} at t+{iv['t_start_rel']:.1f}s for {iv['duration_s']:.1f} s, "
                             f"drove {iv['path_m']:.1f} m (reason: {iv['reason'] or '-'}; current uncertainty "
                             f"{iv['uncertainty_at_start']})")
            L += ["", f"Final distance to goal: {_fmt(m['final_distance_to_goal_m'], 2, ' m')}; "
                      f"oracle time {_fmt(m['oracle_time_s'], 1, ' s')}, time ratio {_fmt(m['time_ratio'])}",
                  "", f"![trial {tr['n']}](trial_{tr['n']}.png)", ""]
        L += ["## All prompts", "", "| time | trial | condition | source | text |", "|---|---|---|---|---|"]
        for p in self.prompts:
            L.append(f"| {p['time']} | {p['trial'] or '-'} | {p['condition'] or '-'} | {p['source']} | "
                     f"{p['text'].replace('|', '/')} |")
        L.append("")
        return "\n".join(L)


class StudyProtocol:
    """Plans each protocol step: where the rover starts, the uncertainty
    threshold, preset goals and the instruction for the participant.

      L1..L2  language: the participant's own prompt. L1 starts at spawn; with
              chain_language (default) L2 starts where L1
              ended, giving two different start positions.
      H1..H2  human-guided: autonomy off, rover put back at L_i's exact start
              pose, participant drives to L_i's goal(s) themselves.
      U1      threshold 0.25, rover at spawn facing the yellow flag, prompt
              "go to the yellow flag"."""

    def __init__(self, rec: StudyRecorder, chain_language: bool = True):
        self.rec = rec
        self.chain_language = chain_language
        self.idx = 0

    def current(self):
        return PROTOCOL[self.idx] if self.idx < len(PROTOCOL) else None

    def spawn_pose(self):
        c = self.rec.controller
        return (float(c.start_x), float(c.start_z), math.radians(float(getattr(c, "start_yaw_deg", 0.0))))

    def _trial(self, label):
        for t in reversed(self.rec.trials):
            if t.get("label") == label:
                return t
        return None

    def plan(self, step) -> dict:
        label, cond, i = step
        out = {"label": label, "condition": cond, "pose": None, "threshold": None, "goals": None, "note": ""}
        if cond == "language":
            if i == 1 or not self.chain_language:
                out["pose"] = self.spawn_pose()
                out["note"] = "start: spawn"
            else:
                out["note"] = f"start: where L{i - 1} ended"
            out["instruction"] = STEP_INSTRUCTIONS["language"]
        elif cond == "human_guided":
            lt = self._trial(f"L{i}")
            if lt is None:
                raise RuntimeError(f"run L{i} before H{i}")
            out["pose"] = tuple(lt["start_pose"])
            goals = [{"label": g["label"], "xz": g["xz"], "instruction": g["instruction"]}
                     for g in lt["goals"] if g["xz"] is not None]
            out["goals"] = goals
            out["note"] = f"same start and goal(s) as L{i}"
            out["instruction"] = STEP_INSTRUCTIONS["human_guided"].format(
                goals=" then ".join(g["label"] for g in goals) or "(L%d had no goal - end manually)" % i)
        else:
            x, z, _ = self.spawn_pose()
            fx, fz = self.rec.objects.get("flag_yellow") or self.rec._env_flag("yellow") or (x, z - 1.0)
            yaw = math.atan2(-(fx - x), -(fz - z))  # forward = (-sin yaw, -cos yaw)
            out["pose"] = (x, z, yaw)
            out["threshold"] = YELLOW_THRESHOLD
            out["note"] = "start: spawn facing the yellow flag, uncertainty threshold 0.25"
            out["instruction"] = STEP_INSTRUCTIONS["uncertainty_0.25"]
        return out


def set_uncertainty_threshold(controller, value: float) -> float:
    """Change the live current-uncertainty threshold; returns the old one."""
    old = float(controller.uncertainty_cov_threshold)
    with controller._lock:
        controller.uncertainty_cov_threshold = float(value)
        controller.display.uncertainty_threshold = float(value)
        controller.display.home_base_uncertainty_threshold = float(value)
    return old


class StudyWindow:
    """Experimenter panel (separate window): participant, the fixed protocol
    (L1-L2, H1-H2, U1), live metrics, final report."""

    def __init__(self, root, recorder: StudyRecorder, participant: Optional[str] = None):
        import tkinter as tk

        self.tk = tk
        self.root, self.rec = root, recorder
        self.proto = StudyProtocol(recorder)
        self._restore_threshold: Optional[float] = None
        self._busy = False
        bg, fg = "#111827", "#e5e7eb"
        self.win = tk.Toplevel(root)
        self.win.title("User study")
        self.win.configure(bg=bg)
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        pad = dict(padx=8, pady=3)

        row = tk.Frame(self.win, bg=bg)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Participant ID", fg=fg, bg=bg).pack(side="left")
        self.pid = tk.Entry(row, width=14)
        self.pid.pack(side="left", padx=6)
        if participant:
            self.pid.insert(0, participant)
        tk.Button(row, text="Start participant", command=self.start_participant).pack(side="left")
        self.chain_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row, text="L2 starts where L1 ended", variable=self.chain_var, fg=fg, bg=bg,
                       selectcolor="#374151", activebackground=bg).pack(side="left", padx=8)

        self.brief = tk.Text(self.win, width=78, height=12, wrap="word", bg="#1f2937", fg=fg, relief="flat")
        self.brief.insert("1.0", BRIEFING)
        self.brief.configure(state="disabled")
        self.brief.pack(fill="both", expand=True, **pad)

        self.steps = tk.Label(self.win, text="", fg="#cbd5e1", bg=bg, anchor="w", font=("TkFixedFont", 11))
        self.steps.pack(fill="x", **pad)
        self.instr = tk.Label(self.win, text="", fg="#fde68a", bg=bg, anchor="w", justify="left",
                              wraplength=640, font=("TkDefaultFont", 13, "bold"))
        self.instr.pack(fill="x", **pad)

        row = tk.Frame(self.win, bg=bg)
        row.pack(fill="x", **pad)
        self.start_btn = tk.Button(row, text="Start next step", command=self.start_step, state="disabled")
        self.start_btn.pack(side="left", padx=4)
        self.end_btn = tk.Button(row, text="End step", command=lambda: self.end_step("ended by experimenter"),
                                 state="disabled")
        self.end_btn.pack(side="left", padx=4)
        self.redo_btn = tk.Button(row, text="Redo previous step", command=self.redo, state="disabled")
        self.redo_btn.pack(side="left", padx=4)
        self.report_btn = tk.Button(row, text="Finish participant → report", command=self.finish, state="disabled")
        self.report_btn.pack(side="left", padx=4)
        self.look_btn = tk.Button(row, text="⟳ Look around 360°", command=self.look_around, state="disabled")
        self.look_btn.pack(side="left", padx=(16, 4))
        self._look = None  # {"acc": rad turned, "yaw": last yaw}

        self.live = tk.Label(self.win, text="enter a participant ID and press Start participant", fg="#93c5fd",
                             bg=bg, justify="left", anchor="w", font=("TkFixedFont", 10))
        self.live.pack(fill="x", **pad)
        self._render_steps()
        self._tick()

    # -- helpers --------------------------------------------------------------
    def _set_live(self, text):
        self.live.configure(text=text)

    def _render_steps(self):
        done = {t.get("label") for t in self.rec.trials}
        parts = []
        for k, (label, _, _) in enumerate(PROTOCOL):
            if self.rec.trial is not None and self.rec.trial.get("label") == label:
                parts.append(f"[▶{label}]")
            elif k == self.proto.idx and self.rec.trial is None:
                parts.append(f"[{label}]")
            else:
                parts.append(("✓" if label in done else " ") + label)
        self.steps.configure(text="Protocol: " + "  ".join(parts))

    def _buttons(self):
        running = self.rec.trial is not None
        have_p = self.rec.dir is not None
        nxt = self.proto.current()
        self.start_btn.configure(state="normal" if have_p and not running and nxt and not self._busy else "disabled",
                                 text=f"Start {nxt[0]}" if nxt else "Protocol complete")
        self.end_btn.configure(state="normal" if running else "disabled")
        self.redo_btn.configure(state="normal" if have_p and not running and self.proto.idx > 0 and not self._busy
                                else "disabled")
        self.report_btn.configure(state="normal" if have_p else "disabled")
        c = self.rec.controller
        autonomy = c.vlm_navigation_active() and not c.snapshot().intervention_active
        self.look_btn.configure(state="normal" if have_p and not self._busy and self._look is None and not autonomy
                                else "disabled",
                                text="⟳ looking around..." if self._look is not None else "⟳ Look around 360°")
        self._render_steps()

    # -- actions --------------------------------------------------------------
    def start_participant(self):
        if self.rec.trial is not None:
            self.end_step("participant changed")
        d = self.rec.start_participant(self.pid.get() or "anon")
        self.proto = StudyProtocol(self.rec, chain_language=self.chain_var.get())
        self.instr.configure(text="Read the briefing above with the participant, then Start L1.")
        self._set_live(f"participant {self.rec.participant}\nlogs: {d}")
        self._buttons()

    def start_step(self):
        step = self.proto.current()
        if step is None:
            return
        self.proto.chain_language = self.chain_var.get()
        try:
            plan = self.proto.plan(step)
        except Exception as e:
            self._set_live(f"cannot start {step[0]}: {e}")
            return
        self._busy = True
        self._buttons()
        c = self.rec.controller
        self.instr.configure(text=f"{plan['label']}: preparing...")
        c.stop_vlm_navigation()  # autonomy off between trials

        def release():
            if c.snapshot().intervention_active:
                c.resume_autonomy()  # no navigator any more, so this just ends the hand-over
            self.root.after(300, place)

        def place():
            if plan["threshold"] is not None:
                old = set_uncertainty_threshold(c, plan["threshold"])
                if self._restore_threshold is None:
                    self._restore_threshold = old
            if plan["pose"] is not None:
                c.request_reset(pose=plan["pose"])
            self.root.after(1500 if plan["pose"] is not None else 300, go)

        def go():
            tr = self.rec.start_trial(plan["condition"], label=plan["label"], goals=plan["goals"], note=plan["note"])
            self._busy = False
            self.instr.configure(text=f"{plan['label']}  ({plan['note']})\n→ {plan['instruction']}"
                                      "\n(Optional: press ⟳ Look around 360° first to see the surroundings.)")
            self.rec._event("step_instruction", label=plan["label"], instruction=plan["instruction"])
            self.rec.log_prompt(f"[{plan['label']}] {plan['instruction']}", source="study")
            self._set_live(f"{tr['label']} started")
            self._buttons()

        self.root.after(300, release)

    LOOK_YAW_RATE = 0.6  # rad/s, about 10.5 s for a full turn

    def look_around(self):
        """Spin in place once (360 degrees) so the participant can see the
        surroundings. Not counted as driving in the trial metrics."""
        c = self.rec.controller
        d = c.snapshot()
        if d.pose is None or self._look is not None:
            return
        if c.vlm_navigation_active() and not d.intervention_active:
            self._set_live("look-around is only available while the rover is not driving itself")
            return
        self._look = {"acc": 0.0, "yaw": float(d.pose.yaw), "t": time.time()}
        self.rec.looking = True
        tr = self.rec.trial
        if tr is not None:
            tr["n_look_arounds"] = tr.get("n_look_arounds", 0) + 1
        self.rec._event("look_around_started")
        c.set_manual(0.0, self.LOOK_YAW_RATE)
        self._buttons()

    def _look_tick(self):
        lk = self._look
        if lk is None:
            return
        c = self.rec.controller
        d = c.snapshot()
        if d.pose is not None:
            dy = (float(d.pose.yaw) - lk["yaw"] + math.pi) % (2 * math.pi) - math.pi
            lk["acc"] += abs(dy)
            lk["yaw"] = float(d.pose.yaw)
        if lk["acc"] >= 2 * math.pi - 0.03 or time.time() - lk["t"] > 40.0:
            if d.intervention_active:
                c.set_manual(0.0, 0.0)  # the human keeps driving (hand-over)
            else:
                c.stop_driving()  # back to idle, so later idle time is not "human"
            self._look = None
            self.rec.looking = False
            self.rec._event("look_around_done", turned_deg=round(math.degrees(lk["acc"]), 1))
            self._buttons()
        else:
            c.set_manual(0.0, self.LOOK_YAW_RATE)  # keep commanding (a key release would zero it)

    def _after_trial(self):
        if self._restore_threshold is not None:
            set_uncertainty_threshold(self.rec.controller, self._restore_threshold)
            self._restore_threshold = None
        self.rec.controller.stop_vlm_navigation()
        self.proto.idx += 1
        nxt = self.proto.current()
        self.instr.configure(text=f"Next: {nxt[0]}" if nxt else "Protocol complete - press Finish participant → report")
        self._buttons()

    def end_step(self, reason):
        tr = self.rec.end_trial(reason)
        if tr:
            self._show_result(tr, reason)
        self._after_trial()

    def redo(self):
        self.proto.idx = max(0, self.proto.idx - 1)
        self.instr.configure(text=f"Redo {self.proto.current()[0]} (the new attempt replaces it for pairing)")
        self._buttons()

    def _show_result(self, tr, reason):
        m = tr["metrics"]
        self._set_live(f"{tr['label']} ended ({reason}): {'SUCCESS' if m['success'] else 'not reached'}  "
                       f"time {_fmt(m['time_to_goal_s'], 1, ' s')}  path {m['path_length_m']:.1f} m  "
                       f"oracle {_fmt(m['oracle_path_m'], 1, ' m')}  SPL {_fmt(m['spl'])}  "
                       f"interventions {m['n_interventions']}")

    def finish(self):
        if self.rec.trial is not None:
            self.end_step("participant finished")
        path = self.rec.write_report()
        s = self.rec._summary()
        lines = [f"REPORT: {path}"]
        for c, v in s.items():
            tg = v.get("time_to_goal_s", {}).get("mean")
            pr = v.get("path_ratio", {}).get("mean")
            iv = v.get("n_interventions", {}).get("mean")
            lines.append(f"{c}: {v['trials']} trials, success {v['success_rate']:.0%}, "
                         f"time {_fmt(tg, 1, ' s')}, path/oracle {_fmt(pr)}, interventions {_fmt(iv, 1)}")
        self._set_live("\n".join(lines))
        print("[study] " + " | ".join(lines), flush=True)
        try:
            os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")
        except Exception:
            pass

    def _tick(self):
        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return
        try:
            self._look_tick()
            if self._look is None and self.rec.dir is not None and int(time.time() * 2) % 2 == 0:
                self._buttons()  # keep the look-around button in step with autonomy on/off
            msg = self.rec.sample()
            tr = self.rec.trial
            if msg:
                done = self.rec.trials[-1]
                self._show_result(done, "all goals reached")
                self._after_trial()
            elif tr is not None:
                t0 = tr["t_first_prompt"] or tr["t_first_move"]
                el = (time.time() - t0) if t0 else 0.0
                goals = " → ".join(("✓ " if g["t_reached"] else "") + g["label"] for g in tr["goals"]) or "(waiting for prompt)"
                ivs = len(tr["interventions"]) + (1 if self.rec._iv_open else 0)
                self._set_live(
                    f"{tr['label']} [{tr['condition']}]  {el:5.1f} s   goals: {goals}\n"
                    f"path {tr['path_m']:.1f} m (auto {tr['path_auto_m']:.1f} / human {tr['path_human_m']:.1f})   "
                    f"interventions {ivs}   prompts {len(tr['prompts'])}   "
                    f"threshold {self.rec.controller.uncertainty_cov_threshold:.2f}")
        except Exception as e:
            print(f"[study] sample error: {type(e).__name__}: {e}")
        self.root.after(SAMPLE_MS, self._tick)
