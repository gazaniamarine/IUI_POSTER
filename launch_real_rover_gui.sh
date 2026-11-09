#!/usr/bin/env bash
# ============================================================
#  REAL ROVER launcher for nav/gui1.py (belief memory + human hand-over +
#  Qwen point-goal + NavDP), modelled on Nihil/LemNAV/LAUNCH/launch_rover1.sh:
#    preflight -> Pi bring-up (ssh: restart rover-camera/rover-agent/rover-zenoh,
#    wait for camera + ESP32 heartbeat) -> python env -> exec nav.gui1 --real-rover
#
#  The GUI reads camera + depth from the rover's zenoh bridge and sends cmd_vel
#  back (nav/real_rover_env.py). Speeds are capped at 0.15 m/s / 0.3 rad/s and
#  forward motion is zeroed whenever real depth is closer than 0.5 m.
#
#    PI_PASS=... ./launch_real_rover_gui.sh [PI_IP] [extra gui1 flags...]
#    SKIP_BRINGUP=1 ./launch_real_rover_gui.sh 10.86.180.125     # services already up
#    TUBE_GUIDED=0  ./launch_real_rover_gui.sh 10.72.244.162     # plain NavDP instead of NavDP + tube (default 1)
#
#  Env knobs: SKIP_BRINGUP=0|1  TUBE_GUIDED=0|1  CAMERA_HEIGHT=0.45  YAW_RATE_SCALE=2.0  LINEAR_SCALE=1.0  MAX_LINEAR=0.10  MAX_ANGULAR=0.20  HABITAT_PYTHON=...
#
#  REAL HARDWARE -- confirm supervision and a clear area before every run.
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── log helpers (as LemNAV LAUNCH/lib/log.sh) ───────────────────────────────
if [[ -t 1 ]]; then _G=$'\033[0;32m'; _Y=$'\033[1;33m'; _C=$'\033[0;36m'; _R=$'\033[0;31m'; _N=$'\033[0m'
else _G=""; _Y=""; _C=""; _R=""; _N=""; fi
ok()   { echo "${_G}  ✓ $*${_N}"; }
info() { echo "${_C}  → $*${_N}"; }
warn() { echo "${_Y}  ⚠ $*${_N}" >&2; }
die()  { echo "${_R}  ✗ $*${_N}" >&2; exit 1; }

# ── args: first non-flag = Pi IP, every flag goes to gui1.py ─────────────────
PI_IP=""; GUI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --rover) ;;                                   # only backend supported here
        -*) GUI_ARGS+=("$arg") ;;
        *) if [[ -z "$PI_IP" ]]; then PI_IP=$arg; else GUI_ARGS+=("$arg"); fi ;;
    esac
done
if [[ -z "$PI_IP" ]]; then PI_IP=10.86.180.125; warn "no Pi IP given -- using default $PI_IP"; fi
PI_USER=${PI_USER:-pi}
SKIP_BRINGUP=${SKIP_BRINGUP:-0}
TUBE_GUIDED=${TUBE_GUIDED:-1}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-0.45}
# Calibration: real turn rate / commanded (2.0: "turn right" ended at ~180 deg
# and "turn around" at ~360 with 1.0, live 2026-11-09). Re-check with "turn right".
YAW_RATE_SCALE=${YAW_RATE_SCALE:-2.0}
LINEAR_SCALE=${LINEAR_SCALE:-1.0}
# Speed caps (lowered 2026-11-09 on request; the GUI never exceeds 0.15 / 0.3)
MAX_LINEAR=${MAX_LINEAR:-0.10}
MAX_ANGULAR=${MAX_ANGULAR:-0.20}
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"

# ── preflight (as LemNAV LAUNCH/lib/preflight.sh) ───────────────────────────
[[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]] || die "no DISPLAY -- the GUI needs a desktop session"
[[ -x "$HABITAT_PYTHON" ]] || die "habitat python not found: $HABITAT_PYTHON"
"$HABITAT_PYTHON" -c "import zenoh, tkinter, cv2" 2>/dev/null || die "$HABITAT_PYTHON lacks zenoh/tkinter/cv2"
[[ -e "$ROOT/navdp/navdp-cross-modal.ckpt" ]] || warn "navdp/navdp-cross-modal.ckpt missing -- NavDP will fail to load"
[[ -f /home/gpu/Desktop/pineapple/Nav_new/nav_pipeline/zenoh_node.py ]] || die "Nav_new/nav_pipeline/zenoh_node.py missing (zenoh CDR helpers)"
ok "python: $HABITAT_PYTHON"

# ── Pi bring-up (as LemNAV LAUNCH/lib/backend.sh, rover backend, zenoh camera) ─
ssh_pi() { sshpass -p "$PI_PASS" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_IP" "$@"; }

info "Pinging Pi at $PI_IP ..."
ping -c5 -i0.5 -W2 "$PI_IP" >/dev/null 2>&1 || die "Pi unreachable -- is it ON and on this network?"
ok "Pi reachable"

if [[ "$SKIP_BRINGUP" == 1 ]]; then
    warn "SKIP_BRINGUP=1 -- not restarting rover services (they must already be streaming)"
else
    command -v sshpass >/dev/null || die "sshpass not installed (sudo apt install sshpass), or use SKIP_BRINGUP=1"
    if [[ -z "${PI_PASS:-}" && -t 0 ]]; then read -r -s -p "  Pi password for $PI_USER@$PI_IP: " PI_PASS; echo; fi
    [[ -n "${PI_PASS:-}" ]] || die "PI_PASS not set (PI_PASS=... $0 <PI_IP>), or use SKIP_BRINGUP=1"
    export PI_PASS
    ssh_pi 'echo ssh_ok' 2>/dev/null | grep -q ssh_ok || die "SSH failed (user=$PI_USER, check PI_PASS)"

    services="rover-camera rover-agent rover-zenoh"; n=3
    # a cold restart of an already-streaming D435i can crash-loop for 20-30 s: leave it alone
    if ssh_pi "journalctl -u rover-camera -n 6 --no-pager 2>/dev/null | grep -q '\[STATUS\]'" 2>/dev/null; then
        ok "rover-camera already streaming -- not restarting it"
        services="rover-agent rover-zenoh"; n=2
    fi
    info "Restarting rover services on Pi ($services)..."
    if ssh_pi "echo $PI_PASS | sudo -S -p '' systemctl restart $services 2>/dev/null; sleep 4; systemctl is-active $services" \
        | grep -c active | grep -q "$n"; then
        ok "services active: $services"
    else
        warn "services not all active -- check the Pi (scripts/pi_install_services.sh)"
    fi

    info "Waiting for camera to stream (up to 40 s)..."
    cam_ok=false
    for i in $(seq 1 20); do
        sleep 2
        if ssh_pi "journalctl -u rover-camera -n 4 --no-pager 2>/dev/null | grep -q 'color_published='" 2>/dev/null; then
            cam_ok=true; ok "camera streaming [$((i*2))s]"; break
        fi
    done
    $cam_ok || warn "camera not streaming -- check: ssh $PI_USER@$PI_IP 'journalctl -u rover-camera -n 20'"

    info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
    if ssh_pi 'bash -lc "source /opt/ros/humble/setup.bash; export ROS_LOCALHOST_ONLY=1; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
        | grep -q "layout\|data"; then
        ok "ESP32 alive (/rover/rpm publishing)"
    else
        warn "No /rover/rpm -- the wheels may not respond (journalctl -u rover-agent)"
    fi
fi

# ── stale GUIs from a previous launch keep old flags and keep the rover's cmd_vel ─
# anchored to the python binary so it can't match a shell that merely mentions the module
pkill -f "^$HABITAT_PYTHON -u -m nav.gui1 --real-rover" 2>/dev/null && sleep 1
true

EXTRA=()
[[ "$TUBE_GUIDED" == 1 ]] && EXTRA+=(--tube-guided)
info "Starting nav.gui1 on the REAL rover (zenoh tcp/$PI_IP:7447, camera height ${CAMERA_HEIGHT} m, tube-guided=$TUBE_GUIDED)"
exec "$HABITAT_PYTHON" -u -m nav.gui1 --real-rover --pi-ip "$PI_IP" --camera-height "$CAMERA_HEIGHT" \
    --yaw-rate-scale "$YAW_RATE_SCALE" --linear-scale "$LINEAR_SCALE" \
    --max-linear "$MAX_LINEAR" --max-angular "$MAX_ANGULAR" \
    --navdp-upstream-root ../navdp_upstream/ "${EXTRA[@]}" "${GUI_ARGS[@]}"
