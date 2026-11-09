#!/usr/bin/env bash
# ============================================================
#  Qwen3.5-9B point-goal + real NavDP navigation (InternVLA-N1-style,
#  detector-free), sim only.
#
#  Runs nav/gui1.py (this repo's interactive rover console, driving the
#  real InternRobotics/NavDP model via NavdpUpstreamPolicy) with a Qwen3.5-9B
#  point-goal navigator (nav.vlm_point_navigator.VlmPointNavigator) wired in:
#  Qwen looks at the raw camera frame plus its own rolling per-instruction
#  history and predicts a goal point directly -- no SAM/DINO detector in the
#  loop, same shape as InternVLA-N1's System-2 -- re-queried every
#  --vlm-replan-every ticks against the ordered instructions given below.
#  See nav/rover_controller.py's module docstring ("VLM navigation" section)
#  and nav/vlm_point_navigator.py for the design.
#
#  Unlike Nihil/LAUNCH/launch_hiwonder_dualvln.sh (its style/structure this
#  script otherwise mirrors): that one drives a REAL 6WD rover over Zenoh,
#  using InternVLA-N1's OWN native System-2+System-1 dual-system end to end
#  (QwenVL-2.5-7B + a Diffusion Transformer, both from the same checkpoint).
#  This one is sim-only (Habitat-Sim, this repo's marsyard2022 scene) and
#  swaps in a different System-1 (real NavDP, not that checkpoint's own
#  Diffusion Transformer) and a different System-2 model/prompt (Qwen3.5-9B,
#  this repo's own qwen35_prompts.py) -- no Zenoh, no Pi, no camera/robot
#  bringup at all; MarsHabitatEnv runs in-process (see rover_controller.py's
#  docstring).
#
#  Examples:
#    ./launch_qwen35_navdp.sh --vlm-instruction "Drive to the large rock cluster ahead."
#    ./launch_qwen35_navdp.sh \
#        --vlm-instruction "Drive to the large rock cluster ahead." \
#        --vlm-instruction "Return to the home base marker." \
#        --vlm-replan-every 20
#    ./launch_qwen35_navdp.sh --navdp-upstream-ckpt navdp/navdp-cross-modal.ckpt \
#        --vlm-instruction "..."
#
#  --vlm-command takes one free-text, possibly-compound instruction instead
#  and has Qwen split it into --vlm-instruction's ordered queue itself (see
#  qwen_prompts.build_split_instruction_prompt / RoverController.
#  start_vlm_navigation_from_command) -- use this when the steps aren't
#  already segmented:
#    ./launch_qwen35_navdp.sh --vlm-command "go straight until you reach the \
#        door and turn left and search for a water bottle"
#
#  Every flag nav/gui1.py itself takes (--scene-path, --navdp-upstream-ckpt,
#  --cbf/--no-cbf, ...) can be passed straight through -- this launcher adds
#  no flags of its own, it only fixes the interpreter/env. See
#  `python -m nav.gui1 --help` for the full list, or CLAUDE.md's Common
#  commands section.
#
#  Stale-process cleanup below mirrors Nihil/LAUNCH/launch_rover_s2diff.sh's
#  own pkill block (kill any previous run's GUI/server processes before
#  starting a fresh one) -- same reasoning: a leftover navdp_server.py or
#  qwen35_server holding its port from a prior run makes the next launch's
#  spawn-or-adopt health check silently adopt a stale process instead of
#  starting clean. set -uo pipefail (not -e), also matching that script: a
#  pkill that matches nothing exits nonzero, and "pkill ... && sleep 1"
#  would abort the whole launcher under -e.
# ============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pkill -f "nav.gui1" 2>/dev/null && sleep 1
pkill -f "navdp_server.py" 2>/dev/null && sleep 1
pkill -f "sam_vla.vlm.qwen35_server" 2>/dev/null && sleep 1
true  # the pkills above are allowed to find nothing; don't let that trip anything after

# habitat env has MarsHabitatEnv/tkinter/RoverController's own deps (see
# CLAUDE.md's env table) -- NOT transformers/Qwen3.5-9B itself, which runs in
# a second, independent qwen_vlm-env subprocess spawned automatically by
# Qwen35ServerManager the first time --vlm-instruction is actually used (see
# rover_controller.py's _env_loop), same lazy-spawn pattern
# QwenServerManager/NavdpUpstreamPolicy already use for their own
# subprocesses. Nothing here needs to conda-activate qwen_vlm directly.
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"

echo "[launch_qwen35_navdp] python=$HABITAT_PYTHON"
exec "$HABITAT_PYTHON" -u -m nav.gui1 "$@"
