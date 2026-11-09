#!/usr/bin/env bash
set -euo pipefail
export PYTHONFAULTHANDLER=1

: "${NAVDP_ROOT:?Set NAVDP_ROOT to the custom NavDP repository}"
: "${NAVDP_CHECKPOINT:?Set NAVDP_CHECKPOINT to the NavDP checkpoint}"
: "${MARS_SCENE:?Set MARS_SCENE to the Habitat .glb scene}"
: "${MARS_TERRAIN_OBJ:?Set MARS_TERRAIN_OBJ to the terrain .obj}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_FILE="${SCRIPT_DIR}/rollout_navdp_policy.py"
echo "[freeform-launcher] rollout=${ROLLOUT_FILE}"
REQUIRED_LOCAL_FILES=(
  rollout_navdp_policy.py
  qwen_navdp_homotopy.py
  qwen_homotopy_server.py
  belief_pixel_goal.py
  belief_heading_recovery.py
)
for required_file in "${REQUIRED_LOCAL_FILES[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${required_file}" ]]; then
    echo "ERROR: missing ${SCRIPT_DIR}/${required_file}" >&2
    echo "Copy the complete free-form PixelGoal file set, not only this launcher." >&2
    exit 2
  fi
done
if ! grep -q -- '"--qwen-freeform-mission"' "${ROLLOUT_FILE}"; then
  echo "ERROR: rollout_navdp_policy.py is an older incompatible version." >&2
  echo "Replace it with the copy shipped beside run_navdp_qwen_freeform_mission.sh." >&2
  exit 2
fi


HABITAT_PYTHON_BIN="${HABITAT_PYTHON:-python}"
if ! PYTHON_EXECUTABLE="$("${HABITAT_PYTHON_BIN}" -c 'import sys; print(sys.executable)')"; then
  echo "ERROR: HABITAT_PYTHON is not a working Python interpreter: ${HABITAT_PYTHON_BIN}" >&2
  exit 2
fi
if [[ -z "${PYTHON_EXECUTABLE}" ]]; then
  echo "ERROR: HABITAT_PYTHON returned no Python executable: ${HABITAT_PYTHON_BIN}" >&2
  exit 2
fi
echo "[freeform-launcher] habitat_python=${PYTHON_EXECUTABLE}"
RUN_MARKER="$(mktemp)"
trap 'rm -f "${RUN_MARKER}"' EXIT

OUTPUT="${OUTPUT:-${SCRIPT_DIR}/runs/qwen_freeform_mission}"

START_X="${START_X:-0}"
START_Z="${START_Z:-8}"
START_YAW_DEG="${START_YAW_DEG:-0}"
OBSTACLE_X="${OBSTACLE_X:-0}"
OBSTACLE_Z="${OBSTACLE_Z:-0}"
GOAL_X="${GOAL_X:-0}"
GOAL_Z="${GOAL_Z:--8}"

BELIEF_BOOTSTRAP_ARGS=(--belief-bootstrap-world-goal)
if [[ "${BELIEF_BOOTSTRAP_WORLD_GOAL:-1}" == "0" ]]; then
  BELIEF_BOOTSTRAP_ARGS=(--no-belief-bootstrap-world-goal)
fi

# Leave MISSION_COMMAND unset to type any vague instruction at startup.
# Unattended example:
#   MISSION_COMMAND="visit the marker and report back" ./run_navdp_qwen_freeform_mission.sh
MISSION_COMMAND_ARGS=()
if [[ -n "${MISSION_COMMAND:-}" ]]; then
  MISSION_COMMAND_ARGS=(--mission-command "${MISSION_COMMAND}")
fi

"${HABITAT_PYTHON_BIN}" "${ROLLOUT_FILE}" \
  --navdp-root "${NAVDP_ROOT}" \
  --navdp-checkpoint "${NAVDP_CHECKPOINT}" \
  --navdp-python "${NAVDP_PYTHON:-${HABITAT_PYTHON_BIN}}" \
  --navdp-device "${NAVDP_DEVICE:-cuda:0}" \
  --planner-mode s2diff \
  --goal-mode pixel \
  --belief-pixel-goal \
  --qwen-freeform-mission \
  "${MISSION_COMMAND_ARGS[@]}" \
  "${BELIEF_BOOTSTRAP_ARGS[@]}" \
  --belief-minimum-goal-pixels "${BELIEF_MINIMUM_GOAL_PIXELS:-10}" \
  --belief-measurement-std "${BELIEF_MEASUREMENT_STD:-0.05}" \
  --belief-translation-process-std "${BELIEF_TRANSLATION_PROCESS_STD:-0.03}" \
  --belief-yaw-process-std-deg "${BELIEF_YAW_PROCESS_STD_DEG:-1.0}" \
  --belief-bootstrap-std "${BELIEF_BOOTSTRAP_STD:-0.50}" \
  --belief-ghost-base-radius "${BELIEF_GHOST_BASE_RADIUS:-10}" \
  --belief-ghost-covariance-scale "${BELIEF_GHOST_COVARIANCE_SCALE:-2.0}" \
  --belief-ghost-maximum-radius "${BELIEF_GHOST_MAXIMUM_RADIUS:-80}" \
  --belief-heading-recovery \
  --belief-recovery-bearing-deg "${BELIEF_RECOVERY_BEARING_DEG:-35}" \
  --belief-recovery-yaw-gain "${BELIEF_RECOVERY_YAW_GAIN:-1.5}" \
  --belief-recovery-maximum-yaw-rate "${BELIEF_RECOVERY_MAXIMUM_YAW_RATE:-0.70}" \
  --belief-recovery-maximum-forward-speed "${BELIEF_RECOVERY_MAXIMUM_FORWARD_SPEED:-0.12}" \
  --return-goal-obstacle-activation-distance "${RETURN_GOAL_OBSTACLE_ACTIVATION_DISTANCE:-1.35}" \
  --return-goal-obstacle-dilation-pixels "${RETURN_GOAL_OBSTACLE_DILATION_PIXELS:-30}" \
  --scene "${MARS_SCENE}" \
  --terrain-obj "${MARS_TERRAIN_OBJ}" \
  --terrain-height-mode obj \
  --start-x "${START_X}" \
  --start-z "${START_Z}" \
  --start-yaw-deg "${START_YAW_DEG}" \
  --goal-x "${GOAL_X}" \
  --goal-z "${GOAL_Z}" \
  --goal-mesh \
  --obstacle-mode mesh \
  "--obstacle-world-xz-item=${OBSTACLE_X},${OBSTACLE_Z}" \
  --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT:-0.75}" \
  --world-obstacle-height "${OBSTACLE_HEIGHT:-1.40}" \
  --robot-radius "${ROBOT_RADIUS:-0.24}" \
  --candidates "${CANDIDATES:-8}" \
  --particles "${PARTICLES:-2}" \
  --particle-std "${PARTICLE_STD:-0.22}" \
  --guidance-strength "${GUIDANCE_STRENGTH:-0.95}" \
  --safe-distance "${SAFE_DISTANCE:-0.70}" \
  --hard-collision-distance "${HARD_COLLISION_DISTANCE:-0.45}" \
  --safety-weight "${SAFETY_WEIGHT:-70.0}" \
  --barrier-weight "${BARRIER_WEIGHT:-50.0}" \
  --circulation-activation-distance "${CIRCULATION_ACTIVATION_DISTANCE:-2.20}" \
  --maximum-obstacle-depth "${MAXIMUM_OBSTACLE_DEPTH:-6.0}" \
  --qwen-homotopy \
  --qwen-device "${QWEN_DEVICE:-auto}" \
  --qwen-homotopy-python "${QWEN_PYTHON:-${HABITAT_PYTHON_BIN}}" \
  --qwen-homotopy-port "${QWEN_HOMOTOPY_PORT:-8890}" \
  --qwen-homotopy-timeout "${QWEN_HOMOTOPY_TIMEOUT:-600}" \
  --homotopy-minimum-obstacle-pixels "${HOMOTOPY_MINIMUM_OBSTACLE_PIXELS:-30}" \
  --homotopy-release-clear-frames "${HOMOTOPY_RELEASE_CLEAR_FRAMES:-8}" \
  --homotopy-consistency-repeats "${HOMOTOPY_CONSISTENCY_REPEATS:-5}" \
  --max-steps "${MAX_STEPS:-1000}" \
  --output "${OUTPUT}" \
  --save-frames \
  --save-video \
  --overlay-masks

if [[ ! -f "${OUTPUT}/manifest.json" || ! -f "${OUTPUT}/rollout.npz" ]]; then
  echo "ERROR: rollout process returned without creating manifest.json and rollout.npz" >&2
  exit 3
fi
if [[ ! "${OUTPUT}/manifest.json" -nt "${RUN_MARKER}" ]]; then
  echo "ERROR: rollout process did not update ${OUTPUT}/manifest.json" >&2
  exit 3
fi
rm -f "${RUN_MARKER}"
trap - EXIT
echo "Saved Qwen free-form mission rollout to ${OUTPUT}"

