#!/usr/bin/env bash
# ============================================================
#  ONE launcher for the whole project (nav/gui1.py: uncertainty-aware,
#  obstacle-avoiding vision-language navigation with belief memory,
#  Qwen3-VL-8B point goals, NavDP + tube, human hand-over).
#
#    ./launch.sh                      interactive menu
#    ./launch.sh sim   [gui1 flags]   simulation GUI (marsyard_with_rocks yard)
#    ./launch.sh study [ID] [flags]   user study (L1-L2, H1-H2, U1 -> report)
#    ./launch.sh rover [PI_IP] [...]  real rover GUI (zenoh, cmd_vel)
#    ./launch.sh rover-bev [PI_IP]    real rover GUI + LemNAV BEV map window
#    ./launch.sh stop                 stop any running nav.gui1
#
#  Real rover modes take the same env knobs as before, e.g.
#    SKIP_BRINGUP=1 ./launch.sh rover 10.72.244.162
#    PI_PASS=... ./launch.sh rover-bev 10.72.244.162
#  (see launch_real_rover_gui.sh / launch_real_rover_bev_gui.sh).
#
#  Env knobs (sim / study): YARD=assets/marsyard_with_rocks.glb  TUBE_GUIDED=1
#    COV_THRESHOLD=0.75 (current-uncertainty hand-over)  SUCCESS_RADIUS=3.0 (study)
#    HABITAT_PYTHON=/home/gpu/miniconda3/envs/habitat/bin/python
#
#  Logs: nav/logs/<time>.log (events), nav/study_logs/<ID>_<time>/ (study reports)
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
YARD="${YARD:-assets/marsyard_with_rocks.glb}"
TUBE_GUIDED="${TUBE_GUIDED:-1}"
COV_THRESHOLD="${COV_THRESHOLD:-0.75}"
SUCCESS_RADIUS="${SUCCESS_RADIUS:-3.0}"

if [[ -t 1 ]]; then _G=$'\033[0;32m'; _Y=$'\033[1;33m'; _C=$'\033[0;36m'; _R=$'\033[0;31m'; _N=$'\033[0m'
else _G=""; _Y=""; _C=""; _R=""; _N=""; fi
ok()   { echo "${_G}  ✓ $*${_N}"; }
info() { echo "${_C}  → $*${_N}"; }
warn() { echo "${_Y}  ⚠ $*${_N}" >&2; }
die()  { echo "${_R}  ✗ $*${_N}" >&2; exit 1; }

stop_gui() {
    # anchored to the python binary so it never matches this shell
    if pkill -f "^$HABITAT_PYTHON -u -m nav.gui1" 2>/dev/null; then
        sleep 1; ok "stopped the running nav.gui1"
    fi
    true
}

preflight_sim() {
    [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]] || die "no DISPLAY -- the GUI needs a desktop session"
    [[ -x "$HABITAT_PYTHON" ]] || die "habitat python not found: $HABITAT_PYTHON"
    "$HABITAT_PYTHON" -c "import habitat_sim, tkinter, customtkinter" 2>/dev/null \
        || die "$HABITAT_PYTHON lacks habitat_sim/tkinter/customtkinter"
    [[ -f "$YARD" ]] || die "yard not found: $YARD"
    [[ -e navdp/navdp-cross-modal.ckpt ]] || warn "navdp/navdp-cross-modal.ckpt missing -- NavDP will fail to load"
    [[ -d ../navdp_upstream ]] || warn "../navdp_upstream missing -- NavDP upstream server will fail"
    ok "python: $HABITAT_PYTHON   yard: $YARD"
}

sim_args() {
    local a=(--yard-glb "$YARD" --gen-home-base --navdp-upstream-root ../navdp_upstream/
             --cov-threshold "$COV_THRESHOLD")
    [[ "$TUBE_GUIDED" == 1 ]] && a=(--tube-guided "${a[@]}")
    printf '%s\n' "${a[@]}"
}

run_sim() {
    preflight_sim; stop_gui
    mapfile -t A < <(sim_args)
    info "Starting simulation GUI (tube-guided=$TUBE_GUIDED, uncertainty threshold $COV_THRESHOLD)"
    exec "$HABITAT_PYTHON" -u -m nav.gui1 "${A[@]}" "$@"
}

run_study() {
    preflight_sim; stop_gui
    mapfile -t A < <(sim_args)
    local pid=()
    if [[ $# -gt 0 && "$1" != -* ]]; then pid=(--study-participant "$1"); shift; fi
    info "Starting USER STUDY (protocol L1-L2, H1-H2, U1; success radius ${SUCCESS_RADIUS} m)"
    info "reports -> $ROOT/nav/study_logs/"
    exec "$HABITAT_PYTHON" -u -m nav.gui1 "${A[@]}" --study --study-success-radius "$SUCCESS_RADIUS" \
        "${pid[@]}" "$@"
}

menu() {
    echo
    echo "  Mars rover VLN -- what do you want to launch?"
    echo "    1) Simulation GUI"
    echo "    2) User study (simulation)"
    echo "    3) Real rover GUI"
    echo "    4) Real rover GUI + BEV map"
    echo "    5) Stop a running GUI"
    echo "    q) Quit"
    read -r -p "  choice: " c
    case "$c" in
        1) run_sim ;;
        2) read -r -p "  participant ID (blank = type it in the window): " p
           if [[ -n "$p" ]]; then run_study "$p"; else run_study; fi ;;
        3|4) read -r -p "  Pi IP [10.72.244.162]: " ip; ip=${ip:-10.72.244.162}
           read -r -p "  rover services already running on the Pi? (Y/n): " up
           [[ "${up,,}" != n ]] && export SKIP_BRINGUP=1
           if [[ "$c" == 3 ]]; then exec "$ROOT/launch_real_rover_gui.sh" "$ip"
           else exec "$ROOT/launch_real_rover_bev_gui.sh" "$ip"; fi ;;
        5) stop_gui ;;
        *) exit 0 ;;
    esac
}

mode="${1:-}"
[[ $# -gt 0 ]] && shift
case "$mode" in
    "")              menu ;;
    sim)             run_sim "$@" ;;
    study)           run_study "$@" ;;
    rover|real)      exec "$ROOT/launch_real_rover_gui.sh" "$@" ;;
    rover-bev|bev)   exec "$ROOT/launch_real_rover_bev_gui.sh" "$@" ;;
    stop)            stop_gui ;;
    -h|--help|help)  sed -n '2,24p' "$0" | sed 's/^#  \{0,1\}//' ;;
    *)               die "unknown mode '$mode' (sim | study | rover | rover-bev | stop | help)" ;;
esac
