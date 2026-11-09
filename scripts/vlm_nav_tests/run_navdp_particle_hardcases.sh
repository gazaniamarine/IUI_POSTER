#!/usr/bin/env bash
set -euo pipefail

# Matched particle-guidance ablation for the current outbound-only system:
# PixelGoal + Gaussian goal belief + Qwen obstacle homotopy + static meshes.
# Default: 8 variants x 4 cases x 1 seed = exactly 32 episodes.
# Return-home mission parsing is explicitly disabled in every run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${SCRIPT_DIR}/rollout_navdp_policy.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${SCRIPT_DIR}/analyze_navdp_ablation.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${SCRIPT_DIR}/plot_navdp_particle_ablation.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/navdp_particle_pixelbelief_core}"

PROFILE="${PROFILE:-smoke}"
EXPERIMENT_SET="${EXPERIMENT_SET:-core}"
# Zero means every requested episode runs and overwrites its old archive.
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
SAVE_MEDIA="${SAVE_MEDIA:-1}"
MAX_STEPS="${MAX_STEPS:-800}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

# Deployed setting: 8 independent candidates and 2 local particles each.
CANDIDATES="${CANDIDATES:-8}"
PARTICLES="${PARTICLES:-2}"
PARTICLE_STD="${PARTICLE_STD:-0.22}"
TEMPERATURE="${TEMPERATURE:-0.35}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-0.95}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"
OBSTACLE_HALF_EXTENT="${OBSTACLE_HALF_EXTENT:-0.75}"
OBSTACLE_HEIGHT="${OBSTACLE_HEIGHT:-1.40}"
SAFE_DISTANCE="${SAFE_DISTANCE:-1.00}"
HARD_COLLISION_DISTANCE="${HARD_COLLISION_DISTANCE:-0.45}"
LYAPUNOV_WEIGHT="${LYAPUNOV_WEIGHT:-2.0}"
BARRIER_WEIGHT="${BARRIER_WEIGHT:-50}"
CIRCULATION_ACTIVATION_DISTANCE="${CIRCULATION_ACTIVATION_DISTANCE:-3.00}"
MAXIMUM_OBSTACLE_DEPTH="${MAXIMUM_OBSTACLE_DEPTH:-10.0}"

case "${PROFILE}" in
    smoke)
        SEEDS="${SEEDS:-7}"
        LAYOUT_FILTER="${LAYOUT_FILTER:-H01_head_on H03_near_start H05_near_goal H07_narrow_gate}"
        ;;
    paper)
        SEEDS="${SEEDS:-1 2 3 4 5}"
        LAYOUT_FILTER="${LAYOUT_FILTER:-all}"
        ;;
    *)
        echo "PROFILE must be smoke or paper" >&2
        exit 2
        ;;
esac
if [[ "${EXPERIMENT_SET}" != "core" && "${EXPERIMENT_SET}" != "mechanisms" && "${EXPERIMENT_SET}" != "full" ]]; then
    echo "EXPERIMENT_SET must be core, mechanisms, or full" >&2
    exit 2
fi

for required_file in \
    "${HABITAT_PYTHON}" "${CHECKPOINT}" "${SCENE}" "${TERRAIN_OBJ}" \
    "${ROLLOUT_SCRIPT}" "${ANALYZER_SCRIPT}"; do
    [[ -f "${required_file}" ]] || {
        echo "Missing required file: ${required_file}" >&2
        exit 1
    }
done
[[ -d "${NAVDP_ROOT}" ]] || {
    echo "Missing NavDP root: ${NAVDP_ROOT}" >&2
    exit 1
}

MEDIA_ARGS=(--no-save-frames --no-save-video --no-archive-observations)
if [[ "${SAVE_MEDIA}" == "1" ]]; then
    MEDIA_ARGS=(--save-frames --save-video --archive-observations)
fi

# name|family|start_x|start_z|yaw|goal_x|goal_z|world obstacle centers
# H06 and H11 put goals near obstacles. H07 is physically traversable but
# narrower than twice the desired clearance and exposes conservative stopping.
LAYOUT_SPECS=(
    "H01_head_on|single|0|8|0|0|-8|0,0"
    "H02_offset_block|single|0|8|0|0|-8|1.10,0"
    "H03_near_start|near_start|0|8|0|0|-8|0,5.0"
    "H04_near_start_offset|near_start|0|8|0|0|-8|0.70,5.2"
    "H05_near_goal|near_goal|0|8|0|0|-8|0,-5.4"
    "H06_goal_beside_rock|near_goal|0|7|0|1.8|-5|0.40,-5"
    "H07_narrow_gate|passage|0|8|0|0|-8|-1.55,0 1.55,0"
    "H08_asymmetric_gate|passage|0|8|0|0|-8|-1.25,0 1.85,0"
    "H09_staggered_s|multi|0|8|0|0|-8|-0.80,1.5 0.80,-1.5"
    "H10_double_center|multi|0|8|0|0|-8|0,1.3 0,-1.3"
    "H11_goal_behind_gate|near_goal|0|8|0|0|-6|-1.55,-4.2 1.55,-4.2"
    "H12_short_head_on|short|0|4|0|0|-4|0,0"
)

layout_enabled() {
    local name="$1"
    [[ "${LAYOUT_FILTER}" == "all" ]] && return 0
    local requested
    for requested in ${LAYOUT_FILTER}; do
        [[ "${requested}" == "${name}" ]] && return 0
    done
    return 1
}

COMMON=(
    --navdp-root "${NAVDP_ROOT}"
    --navdp-checkpoint "${CHECKPOINT}"
    --navdp-python "${HABITAT_PYTHON}"
    --navdp-device "${NAVDP_DEVICE:-cuda:0}"
    --planner-mode s2diff
    --no-remove-critic
    --goal-mode pixel
    --belief-pixel-goal
    --belief-bootstrap-world-goal
    --belief-minimum-goal-pixels "${BELIEF_MINIMUM_GOAL_PIXELS:-10}"
    --belief-measurement-std "${BELIEF_MEASUREMENT_STD:-0.05}"
    --belief-translation-process-std "${BELIEF_TRANSLATION_PROCESS_STD:-0.03}"
    --belief-yaw-process-std-deg "${BELIEF_YAW_PROCESS_STD_DEG:-1.0}"
    --belief-bootstrap-std "${BELIEF_BOOTSTRAP_STD:-0.50}"
    --belief-ghost-base-radius "${BELIEF_GHOST_BASE_RADIUS:-10}"
    --belief-ghost-covariance-scale "${BELIEF_GHOST_COVARIANCE_SCALE:-2.0}"
    --belief-ghost-maximum-radius "${BELIEF_GHOST_MAXIMUM_RADIUS:-80}"
    --belief-heading-recovery
    --belief-recovery-bearing-deg "${BELIEF_RECOVERY_BEARING_DEG:-35}"
    --belief-recovery-yaw-gain "${BELIEF_RECOVERY_YAW_GAIN:-1.5}"
    --belief-recovery-maximum-yaw-rate "${BELIEF_RECOVERY_MAXIMUM_YAW_RATE:-0.70}"
    --belief-recovery-maximum-forward-speed "${BELIEF_RECOVERY_MAXIMUM_FORWARD_SPEED:-0.12}"
    --no-interactive-return-home
    --no-qwen-freeform-mission
    --qwen-homotopy
    --qwen-device "${QWEN_DEVICE:-auto}"
    --qwen-homotopy-python "${QWEN_PYTHON:-${HABITAT_PYTHON}}"
    --qwen-homotopy-port "${QWEN_HOMOTOPY_PORT:-8890}"
    --qwen-homotopy-timeout "${QWEN_HOMOTOPY_TIMEOUT:-600}"
    --homotopy-minimum-obstacle-pixels "${HOMOTOPY_MINIMUM_OBSTACLE_PIXELS:-30}"
    --homotopy-release-clear-frames "${HOMOTOPY_RELEASE_CLEAR_FRAMES:-8}"
    --homotopy-consistency-repeats "${HOMOTOPY_CONSISTENCY_REPEATS:-5}"
    --scene "${SCENE}"
    --terrain-obj "${TERRAIN_OBJ}"
    --terrain-height-mode obj
    --goal-mesh
    --goal-mesh-half-extent 0.25
    --goal-mesh-height 1.50
    --obstacle-mode mesh
    --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT}"
    --world-obstacle-height "${OBSTACLE_HEIGHT}"
    --robot-radius "${ROBOT_RADIUS}"
    --candidates "${CANDIDATES}"
    --particles "${PARTICLES}"
    --particle-std "${PARTICLE_STD}"
    --temperature "${TEMPERATURE}"
    --guidance-strength "${GUIDANCE_STRENGTH}"
    --particle-anchor
    --particle-energy-reweighting
    --particle-collision-mask
    --particle-noise-schedule
    --progressive-guidance
    --safe-distance "${SAFE_DISTANCE}"
    --hard-collision-distance "${HARD_COLLISION_DISTANCE}"
    --safety-weight 70
    --barrier-weight "${BARRIER_WEIGHT}"
    --barrier-rate 0.15
    --circulation-weight 25
    --circulation-activation-distance "${CIRCULATION_ACTIVATION_DISTANCE}"
    --circulation-activation-sharpness 0.25
    --minimum-circulation-progress 0.035
    --blocking-alignment-threshold 0.15
    --circulation-switch-weight 3.0
    --escape-lateral-target 0.40
    --maximum-obstacle-depth "${MAXIMUM_OBSTACLE_DEPTH}"
    --max-steps "${MAX_STEPS}"
    --overlay-masks
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

run_episode() {
    local label="$1" layout="$2" family="$3" seed="$4"
    local sx="$5" sz="$6" yaw="$7" gx="$8" gz="$9"
    local obstacle_spec="${10}"
    shift 10
    remember_label "${label}"

    local output="${OUTPUT_ROOT}/${label}/${layout}/seed_${seed}"
    local archive="${output}/rollout.npz"
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${archive}" ]]; then
        echo "=== SKIP ${label} / ${layout} / seed=${seed} ==="
        return
    fi

    mkdir -p "${output}"
    read -r -a obstacle_centers <<< "${obstacle_spec}"
    local obstacle_args=()
    local center
    for center in "${obstacle_centers[@]}"; do
        obstacle_args+=("--obstacle-world-xz-item=${center}")
    done
    local command=(
        "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"
        "${COMMON[@]}"
        --start-x "${sx}" --start-z "${sz}" --start-yaw-deg "${yaw}"
        --goal-x "${gx}" --goal-z "${gz}"
        "${obstacle_args[@]}"
        --evaluation-layout "${layout}"
        --seed "${seed}" --output "${output}"
        "$@"
    )
    printf '%q ' "${command[@]}" > "${output}/command.txt"
    printf '\n' >> "${output}/command.txt"
    printf '%s\n' "${family}" > "${output}/layout_family.txt"
    echo
    echo "=== ${label} / ${layout} / seed=${seed} ==="
    "${command[@]}"
    [[ -f "${archive}" ]] || {
        echo "Rollout did not produce ${archive}" >&2
        exit 1
    }
}

run_core() {
    local layout="$1" family="$2" seed="$3" sx="$4" sz="$5"
    local yaw="$6" gx="$7" gz="$8" obstacles="$9"
    local base=("${layout}" "${family}" "${seed}" "${sx}" "${sz}" "${yaw}" "${gx}" "${gz}" "${obstacles}")

    # Eight matched variants isolate CBF, Lyapunov, their interaction,
    # in-denoising feedback, local exploration, and energy weighting.
    run_episode pure_navdp "${base[@]}" --planner-mode pure-navdp --no-qwen-homotopy
    run_episode particle_full_p2 "${base[@]}" --lyapunov-weight "${LYAPUNOV_WEIGHT}"
    run_episode particle_no_cbf_barrier "${base[@]}" --lyapunov-weight "${LYAPUNOV_WEIGHT}" --barrier-weight 0
    run_episode particle_no_lyapunov "${base[@]}" --lyapunov-weight 0
    run_episode particle_no_cbf_no_lyapunov "${base[@]}" --lyapunov-weight 0 --barrier-weight 0
    run_episode particle_no_denoise_feedback "${base[@]}" --lyapunov-weight "${LYAPUNOV_WEIGHT}" --guidance-strength 0
    run_episode particle_p1 "${base[@]}" --lyapunov-weight "${LYAPUNOV_WEIGHT}" --particles 1
    run_episode particle_uniform_weights "${base[@]}" --lyapunov-weight "${LYAPUNOV_WEIGHT}" --no-particle-energy-reweighting
}

run_mechanisms() {
    local layout="$1" family="$2" seed="$3" sx="$4" sz="$5"
    local yaw="$6" gx="$7" gz="$8" obstacles="$9"
    local base=("${layout}" "${family}" "${seed}" "${sx}" "${sz}" "${yaw}" "${gx}" "${gz}" "${obstacles}")

    run_core "${base[@]}"
    run_episode particle_sigma0 "${base[@]}" --particle-std 0
    run_episode particle_no_collision_mask "${base[@]}" --no-particle-collision-mask
    run_episode particle_no_anchor "${base[@]}" --no-particle-anchor
    run_episode particle_fixed_noise "${base[@]}" --no-particle-noise-schedule
    run_episode particle_constant_guidance "${base[@]}" --no-progressive-guidance
}

run_sweeps() {
    local layout="$1" family="$2" seed="$3" sx="$4" sz="$5"
    local yaw="$6" gx="$7" gz="$8" obstacles="$9"
    local base=("${layout}" "${family}" "${seed}" "${sx}" "${sz}" "${yaw}" "${gx}" "${gz}" "${obstacles}")

    for particles in 4 8 16; do
        run_episode "particle_p${particles}" "${base[@]}" --particles "${particles}"
    done
    for sigma in 0.10 0.40; do
        run_episode "particle_sigma_${sigma/./p}" "${base[@]}" --particle-std "${sigma}"
    done
    for temperature in 0.10 0.20 0.70 1.00; do
        run_episode "particle_temp_${temperature/./p}" "${base[@]}" --temperature "${temperature}"
    done
    for strength in 0.25 0.50 0.75; do
        run_episode "particle_strength_${strength/./p}" "${base[@]}" --guidance-strength "${strength}"
    done
}

SELECTED_LAYOUTS=()
for specification in "${LAYOUT_SPECS[@]}"; do
    IFS='|' read -r layout family sx sz yaw gx gz obstacles <<< "${specification}"
    layout_enabled "${layout}" || continue
    SELECTED_LAYOUTS+=("${layout}")
    for seed in ${SEEDS}; do
        if [[ "${EXPERIMENT_SET}" == "core" ]]; then
            run_core "${layout}" "${family}" "${seed}" "${sx}" "${sz}" "${yaw}" "${gx}" "${gz}" "${obstacles}"
        else
            run_mechanisms "${layout}" "${family}" "${seed}" "${sx}" "${sz}" "${yaw}" "${gx}" "${gz}" "${obstacles}"
        fi
        if [[ "${EXPERIMENT_SET}" == "full" ]]; then
            run_sweeps "${layout}" "${family}" "${seed}" "${sx}" "${sz}" "${yaw}" "${gx}" "${gz}" "${obstacles}"
        fi
    done
done
[[ "${#SELECTED_LAYOUTS[@]}" -gt 0 ]] || {
    echo "LAYOUT_FILTER selected no known layouts" >&2
    exit 2
}

ANALYZER_ARGS=()
for label in "${LABELS[@]}"; do
    for layout in "${SELECTED_LAYOUTS[@]}"; do
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
    --reference particle_full_p2 \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --output "${OUTPUT_ROOT}/particle_results"

if [[ -f "${PLOT_SCRIPT}" ]] && "${HABITAT_PYTHON}" -c "import matplotlib" 2>/dev/null; then
    "${HABITAT_PYTHON}" "${PLOT_SCRIPT}" \
        --results "${OUTPUT_ROOT}/particle_results.json" \
        --output-dir "${OUTPUT_ROOT}/figures"
else
    echo "Plot script or matplotlib unavailable; CSV/JSON results are complete." >&2
fi

echo
echo "Particle hard-case ablation complete:"
echo "  overall       ${OUTPUT_ROOT}/particle_results.csv"
echo "  per episode   ${OUTPUT_ROOT}/particle_results_episodes.csv"
echo "  per hard case ${OUTPUT_ROOT}/particle_results_layouts.csv"
echo "  paired tests  ${OUTPUT_ROOT}/particle_results_paired.csv"
echo "  figures       ${OUTPUT_ROOT}/figures/"
