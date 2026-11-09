#!/usr/bin/env bash
# ============================================================
#  Nav_new -- REAL ROVER / LanderPi launcher, Qwen3.5+NavDP GUI variant.
#
#  Literally the same bring-up as Nihil/LAUNCH/launch_rover_s2diff.sh (SSH
#  into the Pi, restart rover-camera/rover-agent/rover-zenoh, wait for the
#  camera topic + ESP32 heartbeat -- see Nihil/LAUNCH/_backend.sh) -- only
#  the model/GUI after bring-up changes: instead of Nav_new's
#  DINO+SAM+S2Diff-guided-NavDP stack (nav_pipeline.isaac_gui, needs the
#  `internnav` env), this execs THIS repo's own Qwen3.5-9B point-goal +
#  real-NavDP driver (nav.real_rover_gui / nav.real_rover_qwen35_navdp),
#  whose GUI (nav/real_rover_gui.py) is deliberately laid out to match
#  isaac_gui's camera+top-down-plot+status-bar shape -- see that file's own
#  docstring for why it doesn't reuse isaac_gui.py directly (no `internnav`
#  env / GroundingDINO stack on this machine).
#
#  Needs, already running before this script starts (all already up as of
#  2026-08-27 -- start them if a fresh boot doesn't have them):
#    conda run -n navdp    python navdp_server.py --port 8766 --checkpoint navdp/navdp-cross-modal.ckpt &
#    conda run -n qwen_vlm python -m sam_vla.vlm.qwen_server &            # only used to auto-split ONE compound --instruction
#    conda run -n qwen_vlm python -m sam_vla.vlm.qwen35_server &         # VlmPointNavigator's point-goal model
#
#  Examples:
#    ./launch_rover_s2diff_qwennavdp_gui.sh                                     # default rover, default Pi IP, default instruction
#    ./launch_rover_s2diff_qwennavdp_gui.sh --hiwonder 10.86.180.125 --instruction "Go toward the door."
#    ./launch_rover_s2diff_qwennavdp_gui.sh --hiwonder --instruction "Go toward the door." --autostart
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

source Nihil/LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# ── Launch OUR Qwen3.5+NavDP GUI (real-rover caps), same bring-up as S2Diff ─
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"

pkill -f "nav.real_rover_gui" 2>/dev/null && sleep 1
pkill -f "nav.real_rover_qwen35_navdp" 2>/dev/null && sleep 1
true  # pkills above are allowed to find nothing

MODULE="nav.real_rover_gui"
ARGS=(--endpoint "tcp/$PI_IP:7447")
have_instruction=0
for a in "$@"; do
    [[ "$a" == "--instruction" ]] && have_instruction=1
    if [[ "$a" == "--headless" ]]; then
        MODULE="nav.real_rover_qwen35_navdp"
    else
        ARGS+=("$a")
    fi
done
if [[ "$have_instruction" != "1" ]]; then
    ARGS+=(--instruction "Drive forward a short distance and stop.")
fi

info "Starting Qwen3.5+NavDP GUI [$BACKEND] (pi-ip=$PI_IP, endpoint tcp/$PI_IP:7447, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s)..."
exec "$HABITAT_PYTHON" -u -m "$MODULE" "${ARGS[@]}"
