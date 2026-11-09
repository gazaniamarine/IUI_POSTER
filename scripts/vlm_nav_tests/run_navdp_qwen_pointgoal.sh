#!/usr/bin/env bash
set -euo pipefail
export PYTHONFAULTHANDLER=1

: "${NAVDP_ROOT:?Set NAVDP_ROOT to the custom NavDP repository}"
: "${NAVDP_CHECKPOINT:?Set NAVDP_CHECKPOINT to the NavDP checkpoint}"
: "${MARS_SCENE:?Set MARS_SCENE to the Habitat .glb scene}"
: "${MARS_TERRAIN_OBJ:?Set MARS_TERRAIN_OBJ to the terrain .obj}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/runs/qwen_homotopy_straight}"

# Collinear default layout, facing world -Z:
#   start (0, 8)  ->  obstacle (0, 0)  ->  fixed PointGoal (0, -8)
START_X="${START_X:-0}"
START_Z="${START_Z:-8}"
START_YAW_DEG="${START_YAW_DEG:-0}"
OBSTACLE_X="${OBSTACLE_X:-0}"
OBSTACLE_Z="${OBSTACLE_Z:-0}"
GOAL_X="${GOAL_X:-0}"
GOAL_Z="${GOAL_Z:--8}"

# The numeric PointGoal and green overlay do not require a rendered goal mesh.
# Keep it off by default to reduce Habitat native-object complexity while
# diagnosing the previous segmentation fault. Set GOAL_MESH=1 to enable it.
GOAL_MESH_ARGS=(--no-goal-mesh)
if [[ "${GOAL_MESH:-0}" == "1" ]]; then
  GOAL_MESH_ARGS=(--goal-mesh)
fi

"${HABITAT_PYTHON:-python}" "${SCRIPT_DIR}/rollout_s2dn_policy.py" \
  --navdp-root "${NAVDP_ROOT}" \
  --navdp-checkpoint "${NAVDP_CHECKPOINT}" \
  --navdp-python "${NAVDP_PYTHON:-${HABITAT_PYTHON:-python}}" \
  --navdp-device "${NAVDP_DEVICE:-cuda:0}" \
  --planner-mode s2diff \
  --goal-mode point \
  --scene "${MARS_SCENE}" \
  --terrain-obj "${MARS_TERRAIN_OBJ}" \
  --terrain-height-mode obj \
  --start-x "${START_X}" \
  --start-z "${START_Z}" \
  --start-yaw-deg "${START_YAW_DEG}" \
  --goal-x "${GOAL_X}" \
  --goal-z "${GOAL_Z}" \
  "${GOAL_MESH_ARGS[@]}" \
  --obstacle-mode mesh \
  "--obstacle-world-xz-item=${OBSTACLE_X},${OBSTACLE_Z}" \
  --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT:-0.75}" \
  --world-obstacle-height "${OBSTACLE_HEIGHT:-1.40}" \
  --robot-radius "${ROBOT_RADIUS:-1.8}" \
  --candidates "${CANDIDATES:-16}" \
  --particles "${PARTICLES:-4}" \
  --particle-std "${PARTICLE_STD:-0.22}" \
  --guidance-strength "${GUIDANCE_STRENGTH:-0.85}" \
  --safe-distance "${SAFE_DISTANCE:-0.42}" \
  --hard-collision-distance "${HARD_COLLISION_DISTANCE:-0.24}" \
  --maximum-obstacle-depth "${MAXIMUM_OBSTACLE_DEPTH:-5.0}" \
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

"${HABITAT_PYTHON:-python}" "${SCRIPT_DIR}/analyze_qwen_homotopy.py" "${OUTPUT}"
