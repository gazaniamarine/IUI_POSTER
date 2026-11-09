# #!/usr/bin/env bash
# set -euo pipefail

# # Direct energy-gradient guidance for frozen PointGoal NavDP.
# # This static-obstacle launcher uses no S2Diff particles. It differentiates the
# # soft HLC trajectory energy with respect to each clean action chunk at every
# # reverse-diffusion step, while retaining terminal hard rejection.

# HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
# MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
# NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
# CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
# SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
# TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
# ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/rollout_s2dn_policy.py}"
# OUTPUT="${OUTPUT:-${MARS_ROOT}/runs/navdp_gradient_static}"

# START_X="${START_X:-0}"
# START_Z="${START_Z:-8}"
# START_YAW_DEG="${START_YAW_DEG:-0}"
# GOAL_X="${GOAL_X:-8}"
# GOAL_Z="${GOAL_Z:--8}"
# OBSTACLE_UV="${OBSTACLE_UV:-0.50,0.68}"
# MESH_HALF_PIXELS="${MESH_HALF_PIXELS:-32}"

# CANDIDATES="${CANDIDATES:-4}"
# GRADIENT_STEPS="${GRADIENT_STEPS:-3}"
# GRADIENT_STEP_SIZE="${GRADIENT_STEP_SIZE:-0.04}"
# GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-1.0}"
# ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"
# SAFE_DISTANCE="${SAFE_DISTANCE:-1.20}"
# HARD_DISTANCE="${HARD_DISTANCE:-0.35}"
# SEED="${SEED:-7}"
# MAX_STEPS="${MAX_STEPS:-800}"

# exec "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}" \
#   --navdp-root "${NAVDP_ROOT}" \
#   --navdp-checkpoint "${CHECKPOINT}" \
#   --navdp-python "${HABITAT_PYTHON}" \
#   --navdp-device cuda:0 \
#   --planner-mode gradient \
#   --remove-critic \
#   --scene "${SCENE}" \
#   --terrain-obj "${TERRAIN_OBJ}" \
#   --start-x "${START_X}" \
#   --start-z "${START_Z}" \
#   --start-yaw-deg "${START_YAW_DEG}" \
#   --goal-x "${GOAL_X}" \
#   --goal-z "${GOAL_Z}" \
#   --obstacle-mode mesh \
#   --obstacle-mesh-uv "${OBSTACLE_UV}" \
#   --mesh-half-pixels "${MESH_HALF_PIXELS}" \
#   --maximum-obstacle-depth 12.0 \
#   --candidates "${CANDIDATES}" \
#   --particles 1 \
#   --gradient-steps "${GRADIENT_STEPS}" \
#   --gradient-step-size "${GRADIENT_STEP_SIZE}" \
#   --guidance-strength "${GUIDANCE_STRENGTH}" \
#   --robot-radius "${ROBOT_RADIUS}" \
#   --safe-distance "${SAFE_DISTANCE}" \
#   --hard-collision-distance "${HARD_DISTANCE}" \
#   --safety-weight 80 \
#   --barrier-weight 30 \
#   --barrier-rate 0.15 \
#   --circulation-weight 25 \
#   --circulation-activation-distance 2.0 \
#   --circulation-activation-sharpness 0.25 \
#   --minimum-circulation-progress 0.035 \
#   --blocking-alignment-threshold 0.15 \
#   --circulation-switch-weight 3.0 \
#   --escape-lateral-target 0.40 \
#   --seed "${SEED}" \
#   --max-steps "${MAX_STEPS}" \
#   --output "${OUTPUT}" \
#   "$@"
#!/usr/bin/env bash
set -euo pipefail

# Direct energy-gradient guidance for frozen PointGoal NavDP.
# This static-obstacle launcher uses no S2Diff particles. It differentiates the
# soft HLC trajectory energy with respect to each clean action chunk at every
# reverse-diffusion step, while retaining terminal hard rejection.

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
# SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022_tri.glb}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/rollout_s2dn_policy.py}"
OUTPUT="${OUTPUT:-${MARS_ROOT}/runs/navdp_gradient_static_head_on}"

START_X="${START_X:-0}"
START_Z="${START_Z:-8}"
START_YAW_DEG="${START_YAW_DEG:-0}"
GOAL_X="${GOAL_X:-0}"
GOAL_Z="${GOAL_Z:--8}"
OBSTACLE_WORLD_XZ="${OBSTACLE_WORLD_XZ:-0,0}"
OBSTACLE_HALF_EXTENT="${OBSTACLE_HALF_EXTENT:-0.75}"
OBSTACLE_HEIGHT="${OBSTACLE_HEIGHT:-1.40}"
GOAL_MESH_HALF_EXTENT="${GOAL_MESH_HALF_EXTENT:-0.25}"
GOAL_MESH_HEIGHT="${GOAL_MESH_HEIGHT:-1.50}"

CANDIDATES="${CANDIDATES:-4}"
GRADIENT_STEPS="${GRADIENT_STEPS:-3}"
GRADIENT_STEP_SIZE="${GRADIENT_STEP_SIZE:-0.04}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-1.0}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"
SAFE_DISTANCE="${SAFE_DISTANCE:-1.20}"
HARD_DISTANCE="${HARD_DISTANCE:-0.35}"
SEED="${SEED:-7}"
MAX_STEPS="${MAX_STEPS:-800}"

exec "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}" \
  --navdp-root "${NAVDP_ROOT}" \
  --navdp-checkpoint "${CHECKPOINT}" \
  --navdp-python "${HABITAT_PYTHON}" \
  --navdp-device cuda:0 \
  --planner-mode gradient \
  --remove-critic \
  --scene "${SCENE}" \
  --terrain-obj "${TERRAIN_OBJ}" \
  --start-x "${START_X}" \
  --start-z "${START_Z}" \
  --start-yaw-deg "${START_YAW_DEG}" \
  --goal-x "${GOAL_X}" \
  --goal-z "${GOAL_Z}" \
  --obstacle-mode mesh \
  --evaluation-layout gradient_head_on \
  --obstacle-world-xz "${OBSTACLE_WORLD_XZ}" \
  --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT}" \
  --world-obstacle-height "${OBSTACLE_HEIGHT}" \
  --goal-mesh \
  --goal-mesh-half-extent "${GOAL_MESH_HALF_EXTENT}" \
  --goal-mesh-height "${GOAL_MESH_HEIGHT}" \
  --no-overlay-masks \
  --maximum-obstacle-depth 12.0 \
  --candidates "${CANDIDATES}" \
  --particles 1 \
  --gradient-steps "${GRADIENT_STEPS}" \
  --gradient-step-size "${GRADIENT_STEP_SIZE}" \
  --guidance-strength "${GUIDANCE_STRENGTH}" \
  --robot-radius "${ROBOT_RADIUS}" \
  --safe-distance "${SAFE_DISTANCE}" \
  --hard-collision-distance "${HARD_DISTANCE}" \
  --safety-weight 80 \
  --barrier-weight 30 \
  --barrier-rate 0.15 \
  --circulation-weight 25 \
  --circulation-activation-distance 2.0 \
  --circulation-activation-sharpness 0.25 \
  --minimum-circulation-progress 0.035 \
  --blocking-alignment-threshold 0.15 \
  --circulation-switch-weight 3.0 \
  --escape-lateral-target 0.40 \
  --seed "${SEED}" \
  --max-steps "${MAX_STEPS}" \
  --output "${OUTPUT}" \
  "$@"

