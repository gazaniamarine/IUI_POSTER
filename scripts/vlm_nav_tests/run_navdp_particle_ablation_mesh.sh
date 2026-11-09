#!/usr/bin/env bash
set -euo pipefail

# Particle-only ablation for original S2Diff-guided NavDP.
# Same frozen policy, HLC energy, controller, and static world meshes in every
# run. Each core variant changes exactly one particle mechanism.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="${MARS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${SCRIPT_DIR}/rollout_s2dn_policy.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${SCRIPT_DIR}/analyze_navdp_ablation.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/navdp_particle_ablation_mesh}"

# Three seeds/layouts are an ablation smoke test. Use >=30 matched seeds for
# paper claims. Set RUN_SWEEPS=1 for particle-count/std/temperature sweeps.
SEEDS="${SEEDS:-7 8 9}"
LAYOUTS="${LAYOUTS:-head_on offset_left offset_right}"
MAX_STEPS="${MAX_STEPS:-500}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_SWEEPS="${RUN_SWEEPS:-0}"
SAVE_MEDIA="${SAVE_MEDIA:-0}"

CANDIDATES="${CANDIDATES:-4}"
PARTICLES="${PARTICLES:-8}"
PARTICLE_STD="${PARTICLE_STD:-0.28}"
TEMPERATURE="${TEMPERATURE:-0.25}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"
OBSTACLE_HALF_EXTENT="${OBSTACLE_HALF_EXTENT:-0.75}"
OBSTACLE_HEIGHT="${OBSTACLE_HEIGHT:-1.40}"

for required_file in \
    "${HABITAT_PYTHON}" "${CHECKPOINT}" "${SCENE}" \
    "${TERRAIN_OBJ}" "${ROLLOUT_SCRIPT}" "${ANALYZER_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        echo "Override that path with an environment variable." >&2
        exit 1
    fi
done
[[ -d "${NAVDP_ROOT}" ]] || {
    echo "Missing NavDP root: ${NAVDP_ROOT}" >&2
    exit 1
}

MEDIA_ARGS=(--no-save-frames --no-save-video --no-archive-observations)
if [[ "${SAVE_MEDIA}" == "1" ]]; then
    MEDIA_ARGS=(--save-frames --save-video --archive-observations)
fi

COMMON=(
    --navdp-root "${NAVDP_ROOT}"
    --navdp-checkpoint "${CHECKPOINT}"
    --navdp-python "${HABITAT_PYTHON}"
    --navdp-device cuda:0
    --planner-mode s2diff
    --remove-critic
    --scene "${SCENE}"
    --terrain-obj "${TERRAIN_OBJ}"
    --obstacle-mode mesh
    --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT}"
    --world-obstacle-height "${OBSTACLE_HEIGHT}"
    --goal-mesh
    --goal-mesh-half-extent 0.25
    --goal-mesh-height 1.50
    --no-overlay-masks
    --maximum-obstacle-depth 12.0
    --candidates "${CANDIDATES}"
    --particles "${PARTICLES}"
    --particle-std "${PARTICLE_STD}"
    --temperature "${TEMPERATURE}"
    --guidance-strength 1.0
    --particle-anchor
    --particle-energy-reweighting
    --particle-collision-mask
    --particle-noise-schedule
    --progressive-guidance
    --robot-radius "${ROBOT_RADIUS}"
    --safe-distance 1.20
    --hard-collision-distance 0.35
    --safety-weight 80
    --barrier-weight 30
    --barrier-rate 0.15
    --circulation-weight 25
    --circulation-activation-distance 2.0
    --circulation-activation-sharpness 0.25
    --minimum-circulation-progress 0.035
    --blocking-alignment-threshold 0.15
    --circulation-switch-weight 3.0
    --escape-lateral-target 0.40
    --max-steps "${MAX_STEPS}"
    "${MEDIA_ARGS[@]}"
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

set_layout() {
    case "$1" in
        head_on)
            START_X=0; START_Z=8; START_YAW_DEG=0
            GOAL_X=0; GOAL_Z=-8; OBSTACLE_WORLD_XZ=0,0
            ;;
        offset_left)
            START_X=0; START_Z=8; START_YAW_DEG=0
            GOAL_X=6; GOAL_Z=-8; OBSTACLE_WORLD_XZ=3,0
            ;;
        offset_right)
            START_X=0; START_Z=8; START_YAW_DEG=0
            GOAL_X=-6; GOAL_Z=-8; OBSTACLE_WORLD_XZ=-3,0
            ;;
        *)
            echo "Unknown layout: $1" >&2
            exit 1
            ;;
    esac
}

run_episode() {
    local label="$1"
    local layout="$2"
    local seed="$3"
    shift 3
    set_layout "${layout}"
    remember_label "${label}"

    local output="${OUTPUT_ROOT}/${label}/${layout}/seed_${seed}"
    local archive="${output}/rollout.npz"
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${archive}" ]]; then
        echo "=== SKIP ${label} / ${layout} / seed=${seed} ==="
        return
    fi

    mkdir -p "${output}"
    local command=(
        "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"
        "${COMMON[@]}"
        --start-x "${START_X}" --start-z "${START_Z}"
        --start-yaw-deg "${START_YAW_DEG}"
        --goal-x "${GOAL_X}" --goal-z "${GOAL_Z}"
        "--obstacle-world-xz=${OBSTACLE_WORLD_XZ}"
        --evaluation-layout "${layout}"
        --seed "${seed}" --output "${output}"
        "$@"
    )
    printf '%q ' "${command[@]}" > "${output}/command.txt"
    printf '\n' >> "${output}/command.txt"
    echo
    echo "=== ${label} / ${layout} / seed=${seed} ==="
    "${command[@]}"
    [[ -f "${archive}" ]] || {
        echo "Rollout did not produce ${archive}" >&2
        exit 1
    }
}

for layout in ${LAYOUTS}; do
    for seed in ${SEEDS}; do
        run_episode particle_full "${layout}" "${seed}"
        run_episode particle_p1 "${layout}" "${seed}" --particles 1
        run_episode particle_sigma0 "${layout}" "${seed}" --particle-std 0
        run_episode particle_uniform_weights "${layout}" "${seed}" \
            --no-particle-energy-reweighting
        run_episode particle_no_collision_mask "${layout}" "${seed}" \
            --no-particle-collision-mask
        run_episode particle_no_anchor "${layout}" "${seed}" \
            --no-particle-anchor
        run_episode particle_fixed_noise_scale "${layout}" "${seed}" \
            --no-particle-noise-schedule
        run_episode particle_constant_guidance "${layout}" "${seed}" \
            --no-progressive-guidance
        run_episode particle_no_feedback "${layout}" "${seed}" \
            --guidance-strength 0

        if [[ "${RUN_SWEEPS}" == "1" ]]; then
            for particles in 2 4 16; do
                run_episode "particle_p${particles}" "${layout}" "${seed}" \
                    --particles "${particles}"
            done
            for sigma in 0.10 0.40; do
                label="particle_sigma_${sigma/./p}"
                run_episode "${label}" "${layout}" "${seed}" \
                    --particle-std "${sigma}"
            done
            for temperature in 0.10 0.75; do
                label="particle_temp_${temperature/./p}"
                run_episode "${label}" "${layout}" "${seed}" \
                    --temperature "${temperature}"
            done
        fi
    done
done

ANALYZER_ARGS=()
for label in "${LABELS[@]}"; do
    for layout in ${LAYOUTS}; do
        for seed in ${SEEDS}; do
            archive="${OUTPUT_ROOT}/${label}/${layout}/seed_${seed}/rollout.npz"
            [[ -f "${archive}" ]] || {
                echo "Missing expected archive: ${archive}" >&2
                exit 1
            }
            ANALYZER_ARGS+=(--run "${label}=${archive}")
        done
    done
done

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${ANALYZER_ARGS[@]}" \
    --reference particle_full \
    --output "${OUTPUT_ROOT}/comparison"

echo
echo "Particle-only mesh ablation complete."
echo "Summary: ${OUTPUT_ROOT}/comparison.csv"
echo "Paired effects: ${OUTPUT_ROOT}/comparison_paired.csv"
echo "Episodes: ${OUTPUT_ROOT}/comparison_episodes.csv"
