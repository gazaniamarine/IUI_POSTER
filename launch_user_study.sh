#!/usr/bin/env bash
# ============================================================
#  User study (simulation): nav/gui1.py + the "User study" window
#  (nav/user_study.py). Per participant: briefing -> L1-L2 language prompts,
#  H1-H2 participant drives from the same starts, U1 threshold 0.25 facing the yellow flag -> final report.
#
#    ./launch_user_study.sh [PARTICIPANT_ID] [extra gui1 flags...]
#
#  Logs + report: nav/study_logs/<participant>_<time>/  (report.md, trials.csv,
#  prompts.csv, events.jsonl, trial_<n>.png / .json / _traj.csv)
#  Env knobs: SUCCESS_RADIUS=3.0  YARD=assets/marsyard_with_rocks.glb  HABITAT_PYTHON=...
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
YARD="${YARD:-assets/marsyard_with_rocks.glb}"
SUCCESS_RADIUS="${SUCCESS_RADIUS:-3.0}"
PID_ARG=()
if [[ $# -gt 0 && "$1" != -* ]]; then PID_ARG=(--study-participant "$1"); shift; fi
[[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]] || { echo "  ✗ no DISPLAY" >&2; exit 1; }
pkill -f "^$HABITAT_PYTHON -u -m nav.gui1" 2>/dev/null && sleep 1
exec "$HABITAT_PYTHON" -u -m nav.gui1 --tube-guided --yard-glb "$YARD" --gen-home-base \
    --navdp-upstream-root ../navdp_upstream/ --study --study-success-radius "$SUCCESS_RADIUS" \
    "${PID_ARG[@]}" "$@"
