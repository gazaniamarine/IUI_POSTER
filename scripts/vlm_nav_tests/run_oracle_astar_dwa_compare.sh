#!/usr/bin/env bash
set -euo pipefail

# Four new oracle A*+DWA episodes are run.  The four existing final-model
# episodes are reused; NavDP/Qwen are not launched by this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${SCRIPT_DIR}/rollout_oracle_astar_dwa.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${SCRIPT_DIR}/analyze_navdp_ablation.py}"
MODEL_ROOT="${MODEL_ROOT:-${MARS_ROOT}/runs/navdp_particle_pixelbelief_core/particle_full_p2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/oracle_astar_dwa_comparison}"

SEEDS="${SEEDS:-7}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_STEPS="${MAX_STEPS:-800}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.24}"
OBSTACLE_HALF_EXTENT="${OBSTACLE_HALF_EXTENT:-0.75}"
OBSTACLE_HEIGHT="${OBSTACLE_HEIGHT:-1.40}"
PLANNING_CLEARANCE="${PLANNING_CLEARANCE:-0.18}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

for required in "${HABITAT_PYTHON}" "${SCENE}" "${TERRAIN_OBJ}" \
    "${ROLLOUT_SCRIPT}" "${ANALYZER_SCRIPT}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

# Exactly the same four smoke-test layouts used by the completed particle run.
LAYOUT_SPECS=(
    "H01_head_on|0|8|0|0|-8|0,0"
    "H03_near_start|0|8|0|0|-8|0,5.0"
    "H05_near_goal|0|8|0|0|-8|0,-5.4"
    "H07_narrow_gate|0|8|0|0|-8|-1.55,0 1.55,0"
)

ORACLE_ARCHIVES=()
MODEL_ARCHIVES=()
for specification in "${LAYOUT_SPECS[@]}"; do
    IFS='|' read -r layout sx sz yaw gx gz obstacle_spec <<< "${specification}"
    for seed in ${SEEDS}; do
        output="${OUTPUT_ROOT}/oracle_astar_dwa/${layout}/seed_${seed}"
        archive="${output}/rollout.npz"
        model_archive="${MODEL_ROOT}/${layout}/seed_${seed}/rollout.npz"
        [[ -f "${model_archive}" ]] || {
            echo "Missing completed final-model archive: ${model_archive}" >&2
            echo "Set MODEL_ROOT to the particle_full_p2 directory you already ran." >&2
            exit 1
        }
        if [[ "${SKIP_COMPLETED}" != "1" || ! -f "${archive}" ]]; then
            mkdir -p "${output}"
            read -r -a obstacle_centers <<< "${obstacle_spec}"
            obstacle_args=()
            for center in "${obstacle_centers[@]}"; do
                obstacle_args+=("--obstacle-world-xz-item=${center}")
            done
            command=(
                "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}"
                --scene "${SCENE}"
                --terrain-obj "${TERRAIN_OBJ}"
                --terrain-height-mode obj
                --start-x "${sx}" --start-z "${sz}" --start-yaw-deg "${yaw}"
                --goal-x "${gx}" --goal-z "${gz}"
                "${obstacle_args[@]}"
                --world-obstacle-half-extent "${OBSTACLE_HALF_EXTENT}"
                --world-obstacle-height "${OBSTACLE_HEIGHT}"
                --goal-mesh-half-extent 0.25
                --goal-mesh-height 1.50
                --robot-radius "${ROBOT_RADIUS}"
                --planning-clearance "${PLANNING_CLEARANCE}"
                --max-steps "${MAX_STEPS}"
                --evaluation-layout "${layout}"
                --seed "${seed}"
                --output "${output}"
                --save-frames --save-video
            )
            printf '%q ' "${command[@]}" > "${output}/command.txt"
            printf '\n' >> "${output}/command.txt"
            echo
            echo "=== oracle_astar_dwa / ${layout} / seed=${seed} ==="
            "${command[@]}"
        else
            echo "=== SKIP oracle_astar_dwa / ${layout} / seed=${seed} ==="
        fi
        [[ -f "${archive}" ]] || { echo "Missing oracle archive: ${archive}" >&2; exit 1; }
        ORACLE_ARCHIVES+=("${archive}")
        MODEL_ARCHIVES+=("${model_archive}")
    done
done

analysis_args=()
for archive in "${MODEL_ARCHIVES[@]}"; do
    analysis_args+=(--run "particle_full_p2=${archive}")
done
for archive in "${ORACLE_ARCHIVES[@]}"; do
    analysis_args+=(--run "oracle_astar_dwa=${archive}")
done

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${analysis_args[@]}" \
    --reference particle_full_p2 \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --output "${OUTPUT_ROOT}/oracle_comparison"

echo
echo "Oracle comparison complete (4 new classical episodes; model archives reused):"
echo "  overall       ${OUTPUT_ROOT}/oracle_comparison.csv"
echo "  per episode   ${OUTPUT_ROOT}/oracle_comparison_episodes.csv"
echo "  per hard case ${OUTPUT_ROOT}/oracle_comparison_layouts.csv"
echo "  paired gaps   ${OUTPUT_ROOT}/oracle_comparison_paired.csv"
echo "  videos        ${OUTPUT_ROOT}/oracle_astar_dwa/<layout>/seed_<n>/rollout.mp4"
echo
echo "Paper label: privileged/oracle A*+DWA reference (exact map, goal, and obstacle geometry)."
