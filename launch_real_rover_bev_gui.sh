#!/usr/bin/env bash
# ============================================================
#  REAL ROVER launcher: launch_real_rover_gui.sh + LemNAV's monocular BEV map.
#
#  Takes ONLY the BEV map from Nihil/LemNAV/LAUNCH/launch_rover1_graph.sh
#  (its BEV_DISPLAY=1, OCCUPANCY_SOURCE=depth): a camera-only occupancy map --
#  Depth Anything V2 metric depth, corrected every frame against the known
#  floor (camera height, level mount), fused into a log-odds grid -- shown in
#  its own window. DISPLAY ONLY: it never feeds NavDP, the 0.5 m depth stop or
#  the belief memory. Nothing else from that launcher (graph-constraint legs,
#  fact3r memory, stagewise planner, ...) is used.
#
#  The map runs LemNAV's own code (nav_pipeline.bev_occupancy / bev_display /
#  depth_estimator, imported from $LEMNAV_ROOT, not copied) in a separate
#  `navdp`-env process: nav/bev_worker.py, driven by nav/bev_display.py.
#
#    PI_PASS=... ./launch_real_rover_bev_gui.sh [PI_IP] [extra gui1 flags...]
#    SKIP_BRINGUP=1 ./launch_real_rover_bev_gui.sh 10.72.244.162
#
#  Env knobs: everything launch_real_rover_gui.sh takes, plus
#    BEV_CAM_HEIGHT   camera height for the BEV floor model (m), default $CAMERA_HEIGHT (0.45)
#    BEV_CAM_FORWARD  camera offset ahead of the rover centre (m), default 0.07
#    LEMNAV_ROOT      default /home/gpu/Desktop/Nihil/LemNAV
#    BEV_PYTHON       python for the BEV worker, default the `navdp` conda env
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export LEMNAV_ROOT="${LEMNAV_ROOT:-/home/gpu/Desktop/Nihil/LemNAV}"
export BEV_PYTHON="${BEV_PYTHON:-/home/gpu/miniconda3/envs/navdp/bin/python}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0.45}"
BEV_CAM_HEIGHT="${BEV_CAM_HEIGHT:-$CAMERA_HEIGHT}"
BEV_CAM_FORWARD="${BEV_CAM_FORWARD:-0.07}"

die() { echo "  ✗ $*" >&2; exit 1; }
[[ -f "$LEMNAV_ROOT/nav_pipeline/bev_occupancy.py" ]] || die "LemNAV BEV map not found under $LEMNAV_ROOT (set LEMNAV_ROOT)"
[[ -e "$LEMNAV_ROOT/checkpoints/depth_anything_v2_metric_hypersim_vits.pth" ]] \
    || die "Depth Anything checkpoint missing: $LEMNAV_ROOT/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
[[ -x "$BEV_PYTHON" ]] || die "BEV worker python not found: $BEV_PYTHON (set BEV_PYTHON)"
"$BEV_PYTHON" -c "import torch, yaml, cv2" 2>/dev/null || die "$BEV_PYTHON lacks torch/yaml/cv2"
echo "  ✓ BEV map: LemNAV=$LEMNAV_ROOT  python=$BEV_PYTHON  cam height=${BEV_CAM_HEIGHT} m"

export CAMERA_HEIGHT
exec "$ROOT/launch_real_rover_gui.sh" "$@" --bev-display \
    "--bev-cam-height=$BEV_CAM_HEIGHT" "--bev-cam-forward=$BEV_CAM_FORWARD"
