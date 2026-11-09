#!/usr/bin/env bash
# Orchestrates calibrate_clip_stock_scene.py across the two conda envs it
# needs (no single env here has habitat_sim + torch/sam3/open_clip together):
#   1. capture   -- `habitat` env, habitat_sim only, dumps raw RGB frames
#   2. calibrate -- `sam3` env, torch/sam3/open_clip, runs SAM3+CLIP and
#                   writes the annotated GIF (+ optionally PNG frames)
#
# Captured/annotated frames live under a throwaway /tmp dir and are deleted
# when the script exits, unless --save-frames <dir> is given.
#
# Usage:
#   run_calibrate_clip_stock_scene.sh --scene-glb <path/to/scene.glb> --gif-out <path/to/out.gif>
#
# Re-run against already-captured frames (skips the habitat/capture phase
# entirely, e.g. while iterating on --vocab-terms or --clip-reid-thresh):
#   run_calibrate_clip_stock_scene.sh --frames-dir <dir> --gif-out <path/to/out.gif>

set -euo pipefail

SCENE_GLB=""
FRAMES_DIR=""
GIF_OUT=""
VOCAB_TERMS="chair,door,picture frame,window,table"
SCRIPT_LENGTH=12
WINDOW_FRAMES=5
SEG_INTERVAL=3
CLIP_REID_THRESH=0.9
CHECKPOINT=""
WIDTH=640
HEIGHT=480
GIF_FPS=2
SAVE_FRAMES_DIR=""
HABITAT_ENV="habitat"
SAM3_ENV="sam3"

usage() {
  cat <<'EOF'
Usage: run_calibrate_clip_stock_scene.sh --gif-out PATH (--scene-glb PATH | --frames-dir DIR) [options]

Required (one of):
  --scene-glb PATH        habitat_test_scenes .glb to walk and capture frames from
  --frames-dir DIR        skip capture; classify frames already captured (e.g. from a
                           previous --save-frames run)
  --gif-out PATH          where to write the final annotated GIF

Options:
  --vocab-terms STR       comma-separated terms (default: "chair,door,picture frame,window,table")
  --script-length N       scripted walk steps (default: 12)
  --window-frames N       SAM3 batched-re-window size (default: 5)
  --seg-interval N        resegment every N steps (default: 3)
  --clip-reid-thresh F    re-ID cosine similarity threshold (default: 0.9)
  --checkpoint PATH       local SAM3.1 checkpoint (default: download from HF)
  --width N               capture width (default: 640)
  --height N              capture height (default: 480)
  --gif-fps F             GIF playback fps (default: 2)
  --save-frames DIR       copy raw + annotated frames here instead of deleting them
  --habitat-env NAME      conda env with habitat_sim (default: habitat)
  --sam3-env NAME         conda env with torch/sam3/open_clip (default: sam3)
  -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-glb) SCENE_GLB="$2"; shift 2 ;;
    --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
    --gif-out) GIF_OUT="$2"; shift 2 ;;
    --vocab-terms) VOCAB_TERMS="$2"; shift 2 ;;
    --script-length) SCRIPT_LENGTH="$2"; shift 2 ;;
    --window-frames) WINDOW_FRAMES="$2"; shift 2 ;;
    --seg-interval) SEG_INTERVAL="$2"; shift 2 ;;
    --clip-reid-thresh) CLIP_REID_THRESH="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --gif-fps) GIF_FPS="$2"; shift 2 ;;
    --save-frames) SAVE_FRAMES_DIR="$2"; shift 2 ;;
    --habitat-env) HABITAT_ENV="$2"; shift 2 ;;
    --sam3-env) SAM3_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$GIF_OUT" ]]; then
  echo "error: --gif-out is required" >&2; usage; exit 1
fi
if [[ -z "$SCENE_GLB" && -z "$FRAMES_DIR" ]]; then
  echo "error: one of --scene-glb or --frames-dir is required" >&2; usage; exit 1
fi
if [[ -n "$SCENE_GLB" && -n "$FRAMES_DIR" ]]; then
  echo "error: pass only one of --scene-glb or --frames-dir, not both" >&2; usage; exit 1
fi

WORK_DIR="$(mktemp -d /tmp/calib_clip_stock_scene.XXXXXX)"
ANNOTATED_DIR="$WORK_DIR/annotated"

cleanup() {
  # WORK_DIR is always scratch: when --save-frames is given, the copy into
  # SAVE_FRAMES_DIR already happened earlier in the script body, before this
  # EXIT trap fires -- so the tmp copy is redundant and always removed.
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

if [[ -n "$FRAMES_DIR" ]]; then
  RAW_DIR="$FRAMES_DIR"
  echo "[1/2] using already-captured frames: $RAW_DIR (skipping habitat capture)"
else
  RAW_DIR="$WORK_DIR/raw"
  echo "[1/2] capturing frames in '$HABITAT_ENV' env -> $RAW_DIR"
  conda run -n "$HABITAT_ENV" --no-capture-output python -m sam_vla.perception.calibrate_clip_stock_scene \
    --scene-glb "$SCENE_GLB" \
    --dump-frames-dir "$RAW_DIR" \
    --script-length "$SCRIPT_LENGTH" \
    --width "$WIDTH" --height "$HEIGHT" \
    --capture-only
fi

echo "[2/2] running SAM3+CLIP in '$SAM3_ENV' env -> $GIF_OUT"
CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

mkdir -p "$(dirname "$GIF_OUT")"
conda run -n "$SAM3_ENV" --no-capture-output python -m sam_vla.perception.calibrate_clip_stock_scene \
  --load-frames-dir "$RAW_DIR" \
  --vocab-terms "$VOCAB_TERMS" \
  --window-frames "$WINDOW_FRAMES" \
  --seg-interval "$SEG_INTERVAL" \
  --clip-reid-thresh "$CLIP_REID_THRESH" \
  --gif-path "$GIF_OUT" \
  --gif-fps "$GIF_FPS" \
  --annotate-dir "$ANNOTATED_DIR" \
  "${CHECKPOINT_ARGS[@]}"

echo
echo "GIF written to: $GIF_OUT"
if [[ -n "$SAVE_FRAMES_DIR" ]]; then
  mkdir -p "$SAVE_FRAMES_DIR"
  if [[ -z "$FRAMES_DIR" ]]; then
    cp -r "$RAW_DIR" "$SAVE_FRAMES_DIR/raw"
  fi
  cp -r "$ANNOTATED_DIR" "$SAVE_FRAMES_DIR/annotated"
  echo "Frames saved to: $SAVE_FRAMES_DIR"
else
  echo "Captured/annotated frames were in a tmp dir and have been deleted (pass --save-frames <dir> to keep them)."
fi
