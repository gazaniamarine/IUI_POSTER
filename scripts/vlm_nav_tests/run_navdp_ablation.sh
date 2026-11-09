#!/usr/bin/env bash
set -euo pipefail

# Complete NavDP/HLC-S2Diff ablation for the Mars Habitat rollout.
# Default: pure NavDP, base S2Diff, full HLC, component removals, and a
# K={1,2,4,8,16} candidate-count sweep.
# The critic remains loaded in every default run. HLC-S2Diff does not call it.

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/rollout_s2dn_policy.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/analyze_navdp_ablation.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/navdp_ablation_complete}"

# Three seeds are a smoke test. Use at least 30 for paper results and repeat
# over multiple start/goal/obstacle layouts.
SEEDS="${SEEDS:-7 8 9}"
MAX_STEPS="${MAX_STEPS:-800}"
RUN_COMPONENTS="${RUN_COMPONENTS:-1}"
RUN_CANDIDATE_SWEEP="${RUN_CANDIDATE_SWEEP:-1}"
RUN_CRITIC_CONTROL="${RUN_CRITIC_CONTROL:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

START_X="${START_X:-0}"
START_Z="${START_Z:-8}"
START_YAW_DEG="${START_YAW_DEG:-0}"
GOAL_X="${GOAL_X:-8}"
GOAL_Z="${GOAL_Z:--8}"
OBSTACLE_UV="${OBSTACLE_UV:-0.50,0.68}"

# K=4 preserves both circulation directions while being cheaper than K=16/24.
MAIN_CANDIDATES="${MAIN_CANDIDATES:-4}"
PARTICLES="${PARTICLES:-8}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"

for required_file in \
    "${HABITAT_PYTHON}" \
    "${CHECKPOINT}" \
    "${SCENE}" \
    "${TERRAIN_OBJ}" \
    "${ROLLOUT_SCRIPT}" \
    "${ANALYZER_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${NAVDP_ROOT}" ]]; then
    echo "Missing NavDP root: ${NAVDP_ROOT}" >&2
    exit 1
fi

COMMON=(
    --navdp-root "${NAVDP_ROOT}"
    --navdp-checkpoint "${CHECKPOINT}"
    --navdp-python "${HABITAT_PYTHON}"
    --navdp-device cuda:0
    --no-remove-critic
    --scene "${SCENE}"
    --terrain-obj "${TERRAIN_OBJ}"
    --start-x "${START_X}"
    --start-z "${START_Z}"
    --start-yaw-deg "${START_YAW_DEG}"
    --goal-x "${GOAL_X}"
    --goal-z "${GOAL_Z}"
    --obstacle-mode mesh
    --obstacle-mesh-uv "${OBSTACLE_UV}"
    --mesh-half-pixels 32
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

LABELS=()

remember_label() {
    local requested="$1"
    local existing
    for existing in "${LABELS[@]-}"; do
        [[ "${existing}" == "${requested}" ]] && return
    done
    LABELS+=("${requested}")
}

run_episode() {
    local label="$1"
    local seed="$2"
    shift 2
    local output="${OUTPUT_ROOT}/${label}/seed_${seed}"
    local archive="${output}/rollout.npz"
    remember_label "${label}"

    if [[ "${SKIP_COMPLETED}" == "1" && -f "${archive}" ]]; then
        echo "=== SKIP completed: ${label}, seed=${seed} ==="
        return
    fi

    mkdir -p "${output}"
    local command=(
        "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"
        "${COMMON[@]}"
        --seed "${seed}"
        --output "${output}"
        "$@"
    )
    printf '%q ' "${command[@]}" > "${output}/command.txt"
    printf '\n' >> "${output}/command.txt"

    echo
    echo "=== ${label}, seed=${seed} ==="
    "${command[@]}"

    if [[ ! -f "${archive}" ]]; then
        echo "Rollout completed without producing ${archive}" >&2
        exit 1
    fi
}

for seed in ${SEEDS}; do
    # A: released NavDP denoising plus learned-critic selection.
    run_episode pure_navdp "${seed}" \
        --planner-mode pure-navdp \
        --candidates "${MAIN_CANDIDATES}"

    # B: in-denoising S2Diff, without the proposed HLC package.
    run_episode s2diff_base_k4 "${seed}" \
        --planner-mode s2diff \
        --candidates "${MAIN_CANDIDATES}" \
        "${BASE_S2DIFF[@]}"

    # C: complete proposed method. Critic is loaded but never called here.
    run_episode hlc_full_k4 "${seed}" \
        --planner-mode s2diff \
        --candidates "${MAIN_CANDIDATES}" \
        "${HLC[@]}"

    if [[ "${RUN_COMPONENTS}" == "1" ]]; then
        run_episode hlc_no_barrier "${seed}" \
            --planner-mode s2diff --candidates "${MAIN_CANDIDATES}" \
            "${HLC[@]}" --barrier-weight 0

        # Emergency escape remains enabled; this isolates planned circulation.
        run_episode hlc_no_circulation "${seed}" \
            --planner-mode s2diff --candidates "${MAIN_CANDIDATES}" \
            "${HLC[@]}" --circulation-weight 0 --circulation-switch-weight 0

        run_episode hlc_no_latch "${seed}" \
            --planner-mode s2diff --candidates "${MAIN_CANDIDATES}" \
            "${HLC[@]}" --circulation-switch-weight 0

        run_episode hlc_no_escape "${seed}" \
            --planner-mode s2diff --candidates "${MAIN_CANDIDATES}" \
            "${HLC[@]}" --escape-lateral-target 0
    fi

    if [[ "${RUN_CANDIDATE_SWEEP}" == "1" ]]; then
        # hlc_full_k4 above supplies K=4 without running it twice.
        for candidates in 1 2 8 16; do
            run_episode "hlc_full_k${candidates}" "${seed}" \
                --planner-mode s2diff \
                --candidates "${candidates}" \
                "${HLC[@]}"
        done
    fi

    if [[ "${RUN_CRITIC_CONTROL}" == "1" ]]; then
        # Optional memory-only control; this should not change HLC actions.
        run_episode hlc_full_k4_critic_removed "${seed}" \
            --planner-mode s2diff \
            --remove-critic \
            --candidates "${MAIN_CANDIDATES}" \
            "${HLC[@]}"
    fi
done

ANALYZER_ARGS=()
for label in "${LABELS[@]}"; do
    for seed in ${SEEDS}; do
        archive="${OUTPUT_ROOT}/${label}/seed_${seed}/rollout.npz"
        if [[ ! -f "${archive}" ]]; then
            echo "Missing expected archive: ${archive}" >&2
            exit 1
        fi
        ANALYZER_ARGS+=(--run "${label}=${archive}")
    done
done

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${ANALYZER_ARGS[@]}" \
    --output "${OUTPUT_ROOT}/comparison"

echo
echo "Finished. Summary:"
echo "  ${OUTPUT_ROOT}/comparison.csv"
echo "  ${OUTPUT_ROOT}/comparison.json"
echo
echo "Main comparisons:"
echo "  pure_navdp      vs hlc_full_k4 : complete method vs released policy"
echo "  s2diff_base_k4  vs hlc_full_k4 : total HLC novelty contribution"
echo "  hlc_no_*        vs hlc_full_k4 : individual component contribution"
echo "  hlc_full_k1/2/4/8/16           : candidate-count/latency tradeoff"