#!/usr/bin/env bash
set -euo pipefail

# Verify five paper claims with matched static, escape-stress, and moving-obstacle
# experiments. Defaults are a one-seed functional check. For reported results:
#   SEEDS="1 2 3 4 5" ./run_navdp_claims_eval.sh

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/rollout_s2dn_policy.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/analyze_navdp_ablation.py}"
CLAIM_ANALYZER="${CLAIM_ANALYZER:-${MARS_ROOT}/scripts/vlm_nav_tests/analyze_navdp_claims.py}"
BARRIER_AUDITOR="${BARRIER_AUDITOR:-${MARS_ROOT}/scripts/vlm_nav_tests/audit_navdp_barrier.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/navdp_claims_eval}"

SEEDS="${SEEDS:-7}"
MAX_STEPS="${MAX_STEPS:-800}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"
PARTICLES="${PARTICLES:-8}"

for required_file in \
    "${HABITAT_PYTHON}" \
    "${CHECKPOINT}" \
    "${SCENE}" \
    "${TERRAIN_OBJ}" \
    "${ROLLOUT_SCRIPT}" \
    "${ANALYZER_SCRIPT}" \
    "${CLAIM_ANALYZER}" \
    "${BARRIER_AUDITOR}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -d "${NAVDP_ROOT}" ]]; then
    echo "Missing NavDP root: ${NAVDP_ROOT}" >&2
    exit 1
fi

# name|sx|sz|yaw|gx|gz|obstacle_uv|mesh_half_pixels
STATIC_LAYOUTS=(
    "head_on|0|8|0|0|-8|0.50,0.68|32"
    "diagonal|0|8|-26.565|8|-8|0.50,0.68|32"
    "offset_left|0|8|0|0|-8|0.42,0.68|32"
    "offset_right|0|8|0|0|-8|0.58,0.68|32"
    "double_block|0|8|0|0|-8|0.42,0.68 0.58,0.68|32"
    "near_head_on|0|6|0|0|-8|0.50,0.74|36"
)

# These deliberately use larger obstacle patches and K=1 so the all-candidates-
# collide branch is exercised. Verify escape_count > 0 before making a claim.
ESCAPE_LAYOUTS=(
    "escape_head_on|0|8|0|0|-8|0.50,0.72|64"
    "escape_near|0|6|0|0|-8|0.50,0.78|68"
    "escape_double|0|8|0|0|-8|0.43,0.70 0.57,0.70|56"
    "escape_diagonal|0|8|-26.565|8|-8|0.50,0.72|64"
)

# name|sx|sz|yaw|gx|gz|obstacle_uv|mesh_half_pixels|velocity_vx_vz
# Velocities are world-frame metres per second. The rollout moves both the
# rendered semantic mesh and the independent collision geometry each frame.
MOVING_LAYOUTS=(
    "cross_left_to_right|0|8|0|0|-8|0.34,0.68|36|0.35,0.0"
    "cross_right_to_left|0|8|0|0|-8|0.66,0.68|36|-0.35,0.0"
    "oncoming_slow|0|8|0|0|-8|0.50,0.60|36|0.0,0.20"
    "oncoming_fast|0|8|0|0|-8|0.50,0.58|36|0.0,0.40"
    "diagonal_cross|0|8|-26.565|8|-8|0.35,0.65|36|0.25,0.15"
    "receding|0|8|0|0|-8|0.50,0.72|36|0.0,-0.20"
)

COMMON=(
    --navdp-root "${NAVDP_ROOT}"
    --navdp-checkpoint "${CHECKPOINT}"
    --navdp-python "${HABITAT_PYTHON}"
    --navdp-device cuda:0
    --no-remove-critic
    --scene "${SCENE}"
    --terrain-obj "${TERRAIN_OBJ}"
    --obstacle-mode mesh
    --maximum-obstacle-depth 12.0
    --safe-distance 1.20
    --hard-collision-distance 0.35
    --safety-weight 80
    --guidance-strength 1.0
    --temperature 0.25
    --particle-std 0.28
    --particles "${PARTICLES}"
    --robot-radius "${ROBOT_RADIUS}"
    --max-steps "${MAX_STEPS}"
    --no-archive-observations
    --no-save-frames
    --no-save-video
)

HLC=(
    --barrier-weight 30
    --barrier-rate 0.15
    --circulation-weight 25
    --circulation-activation-distance 2.0
    --circulation-activation-sharpness 0.25
    --minimum-circulation-progress 0.035
    --blocking-alignment-threshold 0.15
    --circulation-switch-weight 3.0
    --escape-lateral-target 0.40
)
BASE_S2DIFF=(
    --barrier-weight 0
    --circulation-weight 0
    --circulation-switch-weight 0
    --escape-lateral-target 0
)

STATIC_ARGS=()
ESCAPE_ARGS=()
MOVING_ARGS=()

run_episode() {
    local suite="$1"
    local label="$2"
    local seed="$3"
    shift 3
    local output="${OUTPUT_ROOT}/${suite}/${label}/${CURRENT_LAYOUT}/seed_${seed}"
    local archive="${output}/rollout.npz"
    local velocity_arguments=()
    if [[ -n "${CURRENT_VELOCITIES}" ]]; then
        local velocity_values=()
        read -r -a velocity_values <<< "${CURRENT_VELOCITIES}"
        velocity_arguments=(--obstacle-velocity-xz "${velocity_values[@]}")
    fi

    if [[ "${SKIP_COMPLETED}" != "1" || ! -f "${archive}" ]]; then
        mkdir -p "${output}"
        local command=(
            "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"
            "${COMMON[@]}"
            --evaluation-layout "${CURRENT_LAYOUT}"
            --seed "${seed}"
            --start-x "${CURRENT_START_X}"
            --start-z "${CURRENT_START_Z}"
            --start-yaw-deg "${CURRENT_YAW}"
            --goal-x "${CURRENT_GOAL_X}"
            --goal-z "${CURRENT_GOAL_Z}"
            --obstacle-mesh-uv "${CURRENT_OBSTACLE_UV[@]}"
            --mesh-half-pixels "${CURRENT_MESH_HALF_PIXELS}"
            "${velocity_arguments[@]}"
            --output "${output}"
            "$@"
        )
        printf '%q ' "${command[@]}" > "${output}/command.txt"
        printf '\n' >> "${output}/command.txt"
        echo
        echo "=== ${suite}: ${label}, ${CURRENT_LAYOUT}, seed=${seed} ==="
        "${command[@]}"
    else
        echo "=== SKIP completed: ${suite}, ${label}, ${CURRENT_LAYOUT}, seed=${seed} ==="
    fi

    if [[ ! -f "${archive}" ]]; then
        echo "Missing rollout archive: ${archive}" >&2
        exit 1
    fi
    case "${suite}" in
        static) STATIC_ARGS+=(--run "${label}=${archive}") ;;
        escape) ESCAPE_ARGS+=(--run "${label}=${archive}") ;;
        moving) MOVING_ARGS+=(--run "${label}=${archive}") ;;
        *) echo "Unknown suite: ${suite}" >&2; exit 1 ;;
    esac
}

load_static_layout() {
    IFS='|' read -r \
        CURRENT_LAYOUT CURRENT_START_X CURRENT_START_Z CURRENT_YAW \
        CURRENT_GOAL_X CURRENT_GOAL_Z obstacle_spec CURRENT_MESH_HALF_PIXELS \
        <<< "$1"
    read -r -a CURRENT_OBSTACLE_UV <<< "${obstacle_spec}"
    CURRENT_VELOCITIES=""
}

load_moving_layout() {
    IFS='|' read -r \
        CURRENT_LAYOUT CURRENT_START_X CURRENT_START_Z CURRENT_YAW \
        CURRENT_GOAL_X CURRENT_GOAL_Z obstacle_spec CURRENT_MESH_HALF_PIXELS \
        CURRENT_VELOCITIES <<< "$1"
    read -r -a CURRENT_OBSTACLE_UV <<< "${obstacle_spec}"
}

for specification in "${STATIC_LAYOUTS[@]}"; do
    load_static_layout "${specification}"
    for seed in ${SEEDS}; do
        run_episode static pure_navdp "${seed}" \
            --planner-mode pure-navdp --candidates 4
        run_episode static s2diff_base_k4 "${seed}" \
            --planner-mode s2diff --candidates 4 "${BASE_S2DIFF[@]}"
        run_episode static hlc_full_k4 "${seed}" \
            --planner-mode s2diff --candidates 4 "${HLC[@]}"
    done
done

for specification in "${ESCAPE_LAYOUTS[@]}"; do
    load_static_layout "${specification}"
    for seed in ${SEEDS}; do
        run_episode escape hlc_escape_on "${seed}" \
            --planner-mode s2diff --candidates 1 \
            --safe-distance 1.50 --hard-collision-distance 0.55 \
            "${HLC[@]}"
        run_episode escape hlc_escape_off "${seed}" \
            --planner-mode s2diff --candidates 1 \
            --safe-distance 1.50 --hard-collision-distance 0.55 \
            "${HLC[@]}" --escape-lateral-target 0
    done
done

for specification in "${MOVING_LAYOUTS[@]}"; do
    load_moving_layout "${specification}"
    for seed in ${SEEDS}; do
        run_episode moving pure_navdp "${seed}" \
            --planner-mode pure-navdp --candidates 4
        run_episode moving s2diff_base_k4 "${seed}" \
            --planner-mode s2diff --candidates 4 "${BASE_S2DIFF[@]}"
        run_episode moving hlc_full_k4 "${seed}" \
            --planner-mode s2diff --candidates 4 "${HLC[@]}"
    done
done

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${STATIC_ARGS[@]}" \
    --reference hlc_full_k4 \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --output "${OUTPUT_ROOT}/static_results"

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${ESCAPE_ARGS[@]}" \
    --reference hlc_escape_on \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --output "${OUTPUT_ROOT}/escape_results"

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${MOVING_ARGS[@]}" \
    --reference hlc_full_k4 \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --output "${OUTPUT_ROOT}/moving_results"

"${HABITAT_PYTHON}" "${BARRIER_AUDITOR}" \
    "${STATIC_ARGS[@]}" \
    --output "${OUTPUT_ROOT}/static_barrier_audit.json"

"${HABITAT_PYTHON}" "${BARRIER_AUDITOR}" \
    "${MOVING_ARGS[@]}" \
    --output "${OUTPUT_ROOT}/moving_barrier_audit.json"

"${HABITAT_PYTHON}" "${CLAIM_ANALYZER}" \
    --static-results "${OUTPUT_ROOT}/static_results.json" \
    --escape-results "${OUTPUT_ROOT}/escape_results.json" \
    --moving-results "${OUTPUT_ROOT}/moving_results.json" \
    --output "${OUTPUT_ROOT}/claim_verification"

echo
echo "Claim verification complete:"
echo "  ${OUTPUT_ROOT}/claim_verification.md"
echo "  ${OUTPUT_ROOT}/claim_verification.json"
echo "  ${OUTPUT_ROOT}/static_results.csv"
echo "  ${OUTPUT_ROOT}/escape_results.csv"
echo "  ${OUTPUT_ROOT}/moving_results.csv"
echo "  ${OUTPUT_ROOT}/static_barrier_audit.json"
echo "  ${OUTPUT_ROOT}/moving_barrier_audit.json"