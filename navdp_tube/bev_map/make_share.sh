#!/usr/bin/env bash
# Package only the files needed on a real robot (no simulation, recordings or calibration results).
#   ./make_share.sh            -> bev_map_share.zip next to this script
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-bev_map_share.zip}"
FILES=(
  readme.md config.yaml requirements.txt make_share.sh .gitignore
  bev_pipeline.py projection.py occupancy_grid.py depth_estimator.py
  run_bev_real.py real_source.py telemetry.py gstream.py rover_kinematics.py calibrate_camera.py
  bev_render.py web_view.py
  rover_bridge/rover_state_bridge.py
)
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done
rm -f "$OUT"
python3 - "$OUT" "${FILES[@]}" <<'PY'
import sys, zipfile
out, files = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, "bev_map/" + f)
    z.writestr("bev_map/calib/README.txt", "Calibration results go here (python calibrate_camera.py ...).\n")
print(f"wrote {out} ({len(files)} files)")
PY
