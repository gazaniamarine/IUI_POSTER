#!/usr/bin/env bash
# Study 2 (next.md): actually runs sam_vla.study2_noise_sweep against the real
# sim, driving sam_vla.run_navdp_rollout (the same "modular" mechanism the
# repo's other real rollouts use, not the legacy scripts/vlm_nav_tests/
# rollout_navdp*.py entry points). Deliberately small-scale by default -- a
# first real validation batch, not the full statistical-power sweep next.md's
# Study 2 "Open questions" leaves for a human episode-count/compute-budget
# decision. Override any variable below via env, e.g.:
#   NOISE_LEVELS=0.0,0.041,0.068,0.075,0.109,0.15 EPISODES_FILE=my_eps.json \
#     ./sam_vla/study2_run_real_sweep.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
SCENE_PATH="${SCENE_PATH:-assets/marsyard2022.glb}"
HEIGHTMAP_PATH="${HEIGHTMAP_PATH:-marsyard2022_terrain_hm.png}"
CKPT="${CKPT:-navdp/ckpt_last.pt}"
EPISODES_FILE="${EPISODES_FILE:-study2_episodes.json}"
OUT_DIR="${OUT_DIR:-study2_real_run1}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0,0.075,0.15}"
MAX_STEPS="${MAX_STEPS:-150}"

echo "[study2] python=$HABITAT_PYTHON scene=$SCENE_PATH heightmap=$HEIGHTMAP_PATH ckpt=$CKPT"
echo "[study2] episodes=$EPISODES_FILE noise_levels=$NOISE_LEVELS max_steps=$MAX_STEPS out_dir=$OUT_DIR"

"$HABITAT_PYTHON" -m sam_vla.study2_noise_sweep \
  --episodes-file "$EPISODES_FILE" \
  --out-dir "$OUT_DIR" \
  --python "$HABITAT_PYTHON" \
  --noise-levels "$NOISE_LEVELS" \
  --keep-going \
  -- \
  --scene-path "$SCENE_PATH" \
  --heightmap-path "$HEIGHTMAP_PATH" \
  --ckpt "$CKPT" \
  --cbf --zero-lateral \
  --max-steps "$MAX_STEPS" \
  --save-video

echo "[study2] sweep done -- running analysis"
"$HABITAT_PYTHON" -m sam_vla.study2_analysis \
  --sweep-manifest "$OUT_DIR/sweep_manifest.json" \
  --out-csv "$OUT_DIR/analysis.csv"
