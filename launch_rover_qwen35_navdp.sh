#!/usr/bin/env bash
# ============================================================
#  REAL ROVER launcher, Qwen3.5-9B point-goal + real NavDP (InternVLA-N1-
#  style, detector-free) -- sim-only sibling is launch_qwen35_navdp.sh.
#
#  Structure mirrors Nihil/LAUNCH/launch_rover_s2diff.sh (header/example
#  block, stale-process pkill cleanup before exec, set -uo pipefail) but
#  wires our own module (nav.real_rover_qwen35_navdp) instead of Nav_new's
#  own s2diff_runner/pipeline.py -- this is a different System-2 (Qwen3.5-9B
#  point-goal + rolling history, this repo's own qwen35_prompts.py) and a
#  different System-1 hookup (real InternRobotics/NavDP via HTTP
#  /pointgoal_step, not Nav_new's in-process DDPM/S2Diff sampler), talking to
#  the SAME real rover zenoh bridge (nav_pipeline.zenoh_node CDR helpers,
#  imported from Nav_new -- see nav/real_rover_qwen35_navdp.py's own
#  docstring for why an explicit endpoint, not multicast).
#
#  Deliberately skips Nihil/LAUNCH/_backend.sh's SSH bring-up (ping + sudo
#  systemctl restart rover-camera/rover-agent/rover-zenoh on the Pi): this
#  robot's camera/depth/zenoh streams were already confirmed live at
#  172.22.217.125 with no bring-up step (2026-08-24), and this repo has no
#  verified SSH credentials for that Pi. If a run ever finds the bridge down,
#  bring it up manually first rather than adding a blind SSH+sudo call here.
#
#  NavDP (port 8766) and the Qwen3.5-9B server (port 8769) are NOT spawned by
#  this script -- unlike nav/rover_controller.py's sim path, which owns
#  Qwen35ServerManager/NavdpUpstreamServerManager lazily, this standalone
#  real-hardware driver talks to already-running servers on fixed ports (see
#  CLAUDE.md's env table: navdp env for navdp_server.py, qwen_vlm env for
#  qwen35_server). Start them first if they aren't already up:
#    conda run -n navdp python navdp_server.py --port 8766 --checkpoint navdp/navdp-cross-modal.ckpt &
#    conda run -n qwen_vlm python -m sam_vla.vlm.qwen35_server &
#
#  REAL HARDWARE -- confirm supervision + a clear area before every run, not
#  just the first time. Conservative caps baked into the driver itself
#  (max_linear=0.15, bounded --max-steps, safety zero-twist on every exit
#  path) -- override only with a specific reason.
#
#  --gui switches to nav/real_rover_gui.py: a live camera feed + status panel
#  (current instruction, model reasoning, commanded velocities, the
#  DEPTH_STOP_M safety banner) with a Start/STOP button, instead of the
#  headless CLI's log-lines-only output. Same drive_loop function underneath
#  either way -- see nav/real_rover_gui.py's module docstring.
#
#  Examples:
#    ./launch_rover_qwen35_navdp.sh --instruction "Drive forward a short distance and stop."
#    ./launch_rover_qwen35_navdp.sh --gui --instruction "Go near the door." --instruction "Turn left."
#    ./launch_rover_qwen35_navdp.sh \
#        --instruction "Go toward the hallway." --instruction "Stop near the chairs." \
#        --replan-every 8 --max-steps 90
#    ./launch_rover_qwen35_navdp.sh --endpoint tcp/172.22.217.125:7447 --instruction "..."
# ============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pkill -f "nav.real_rover_qwen35_navdp" 2>/dev/null && sleep 1
pkill -f "nav.real_rover_gui" 2>/dev/null && sleep 1
true  # pkills above are allowed to find nothing; don't let that trip anything after

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"

MODULE="nav.real_rover_qwen35_navdp"
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--gui" ]]; then
        MODULE="nav.real_rover_gui"
    else
        ARGS+=("$a")
    fi
done

echo "[launch_rover_qwen35_navdp] python=$HABITAT_PYTHON module=$MODULE"
exec "$HABITAT_PYTHON" -u -m "$MODULE" "${ARGS[@]}"
