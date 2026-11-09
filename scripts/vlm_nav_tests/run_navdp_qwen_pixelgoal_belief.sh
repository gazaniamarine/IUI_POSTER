#!/usr/bin/env bash
set -euo pipefail
export PYTHONFAULTHANDLER=1

: "${NAVDP_ROOT:?Set NAVDP_ROOT to the custom NavDP repository}"
: "${NAVDP_CHECKPOINT:?Set NAVDP_CHECKPOINT to the NavDP checkpoint}"
: "${MARS_SCENE:?Set MARS_SCENE to the Habitat .glb scene}"
: "${MARS_TERRAIN_OBJ:?Set MARS_TERRAIN_OBJ to the terrain .obj}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/runs/qwen_pixelgoal_belief}"

START_X="${START_X:-0}"
START_Z="${START_Z:-8}"
START_YAW_DEG="${START_YAW_DEG:-0}"
OBSTACLE_X="${OBSTACLE_X:-0}"
OBSTACLE_Z="${OBSTACLE_Z:-0}"
GOAL_X="${GOAL_X:-0}"
GOAL_Z="${GOAL_Z:--8}"

# In this collinear simulation the obstacle can hide the goal at the first
# frame. World bootstrap is an explicit simulation convenience. Set
# BELIEF_BOOTSTRAP_WORLD_GOAL=0 for a strict live-mask-only experiment; then
# the goal must be visible at least once before it can be tracked.
BELIEF_BOOTSTRAP_ARGS=(--belief-bootstrap-world-goal)
if [[ "${BELIEF_BOOTSTRAP_WORLD_GOAL:-1}" == "0" ]]; then
  BELIEF_BOOTSTRAP_ARGS=(--no-belief-bootstrap-world-goal)
fi

"${HABITAT_PYTHON:-python}" "${SCRIPT_DIR}/rollout_s2dn_policy.py" \
  --navdp-root "${NAVDP_ROOT}" \
  --navdp-checkpoint "${NAVDP_CHECKPOINT}" \
  --navdp-python "${NAVDP_PYTHON:-${HABITAT_PYTHON:-python}}" \
  --navdp-device "${NAVDP_DEVICE:-cuda:0}" \
  --planner-mode s2diff \
  --goal-mode pixel \
  --belief-pixel-goal \
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
  --candidates "${CANDIDATES:-16}" \
  --particles "${PARTICLES:-4}" \
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
  --qwen-homotopy-python "${QWEN_PYTHON:-${HABITAT_PYTHON:-python}}" \
  --qwen-homotopy-port "${QWEN_HOMOTOPY_PORT:-8890}" \
  --qwen-homotopy-timeout "${QWEN_HOMOTOPY_TIMEOUT:-600}" \
  --homotopy-minimum-obstacle-pixels "${HOMOTOPY_MINIMUM_OBSTACLE_PIXELS:-30}" \
  --homotopy-release-clear-frames "${HOMOTOPY_RELEASE_CLEAR_FRAMES:-8}" \
  --homotopy-consistency-repeats "${HOMOTOPY_CONSISTENCY_REPEATS:-5}" \
  --max-steps "${MAX_STEPS:-500}" \
  --output "${OUTPUT}" \
  --save-frames \
  --save-video \
  --overlay-masks

echo "Saved belief PixelGoal rollout to ${OUTPUT}"
