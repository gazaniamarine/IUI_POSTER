#!/usr/bin/env bash
set -euo pipefail
export PYTHONFAULTHANDLER=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${SCRIPT_DIR}/rollout_navdp_policy.py}"
ANALYZER="${ANALYZER:-${SCRIPT_DIR}/compare_soft_cbf_lyap_rollout.py}"
SEED="${SEED:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/cbf_only_strong/seed_${SEED}}"

for required in "${HABITAT_PYTHON}" "${CHECKPOINT}" "${SCENE}"     "${TERRAIN_OBJ}" "${ROLLOUT_SCRIPT}" "${ANALYZER}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
[[ -d "${NAVDP_ROOT}" ]] || { echo "Missing NavDP root: ${NAVDP_ROOT}" >&2; exit 1; }

common=(
    --navdp-root "${NAVDP_ROOT}"
    --navdp-checkpoint "${CHECKPOINT}"
    --navdp-python "${HABITAT_PYTHON}"
    --navdp-device "${NAVDP_DEVICE:-cuda:0}"
    --no-remove-critic
    --seed "${SEED}"
    --goal-mode pixel
    --belief-pixel-goal
    --belief-bootstrap-world-goal
    --no-belief-heading-recovery
    --no-interactive-return-home
    --no-qwen-freeform-mission
    --no-qwen-homotopy
    --scene "${SCENE}"
    --terrain-obj "${TERRAIN_OBJ}"
    --terrain-height-mode obj
    --start-x 0
    --start-z 8
    --start-yaw-deg 0
    --goal-x 0
    --goal-z -8
    --goal-mesh
    --obstacle-mode mesh
    --obstacle-world-xz-item=0,0
    --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT:-0.75}"
    --world-obstacle-height "${OBSTACLE_HEIGHT:-1.80}"
    --robot-radius "${ROBOT_RADIUS:-0.24}"
    --candidates "${CANDIDATES:-8}"
    --particles "${PARTICLES:-2}"
    --particle-std "${PARTICLE_STD:-0.22}"
    --maximum-obstacle-depth "${MAXIMUM_OBSTACLE_DEPTH:-10.0}"
    --safe-distance "${SAFE_DISTANCE:-0.70}"
    --hard-collision-distance "${HARD_COLLISION_DISTANCE:-0.45}"
    --max-steps "${MAX_STEPS:-500}"
    --stop-distance "${STOP_DISTANCE:-1.0}"
    --evaluation-layout H01_head_on_soft_only
    --save-frames
    --save-video
    --archive-observations
    --overlay-masks
)

PURE_OUTPUT="${OUTPUT_ROOT}/pure_navdp"
GUIDED_OUTPUT="${OUTPUT_ROOT}/cbf_only"
mkdir -p "${PURE_OUTPUT}" "${GUIDED_OUTPUT}"

echo "=== Pure NavDP: same scene, seed, goal, and obstacle ==="
"${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"     "${common[@]}"     --planner-mode pure-navdp     --output "${PURE_OUTPUT}"

echo "=== CBF-only guidance: Lyapunov weight zero, no hard loss or deterministic rescue ==="
"${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"     "${common[@]}"     --planner-mode s2diff     --particle-anchor     --particle-energy-reweighting     --no-particle-collision-mask     --particle-noise-schedule     --progressive-guidance     --guidance-strength "${GUIDANCE_STRENGTH:-0.95}"     --temperature "${TEMPERATURE:-0.35}"     --hard-collision-penalty 0     --no-hard-collision-rejection     --no-deterministic-escape     --safety-weight 0     --terminal-goal-weight 0     --nominal-weight 0     --smoothness-weight 0     --step-weight 0     --circulation-weight 0     --circulation-switch-weight 0     --barrier-weight "${BARRIER_WEIGHT:-500.0}"     --barrier-rate "${BARRIER_RATE:-0.15}"     --lyapunov-weight "${LYAPUNOV_WEIGHT:-0.0}"     --output "${GUIDED_OUTPUT}"

"${HABITAT_PYTHON}" "${ANALYZER}"     --pure "${PURE_OUTPUT}/rollout.npz"     --guided "${GUIDED_OUTPUT}/rollout.npz"     --output "${OUTPUT_ROOT}/comparison.json"

echo
echo "Comparison complete:"
echo "  pure video     ${PURE_OUTPUT}/rollout.mp4"
echo "  guided video   ${GUIDED_OUTPUT}/rollout.mp4"
echo "  metrics        ${OUTPUT_ROOT}/comparison.json"
