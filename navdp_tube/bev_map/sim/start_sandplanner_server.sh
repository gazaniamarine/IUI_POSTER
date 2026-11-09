#!/usr/bin/env bash
# Start the SanD-Planner server (port 8892). Keep this terminal open while testing; Ctrl+C stops it.
#   ./start_sandplanner_server.sh                          # GPU, checkpoint NoMax.pth
#   ./start_sandplanner_server.sh --checkpoint ~/Desktop/sandplanner/checkpoints/Max_2.1m.pth
#   other options: python ../../tube_navdp/scripts/sandplanner_server.py -h
# Needs the clone ~/Desktop/sandplanner (github.com/WangJinCheng1998/sandplanner) with checkpoints/NoMax.pth, and
# timm in ~/Desktop/sandplanner/.deps (pip install --no-deps --target .deps timm==1.0.15, in the navdp env).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # repo root
SAND="${SAND_ROOT:-$(cd "$ROOT/.." && pwd)/sandplanner}"
PORT=8892

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  PID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  echo "Port $PORT is already in use (pid ${PID:-?}): the server is probably running already."
  [ -n "${PID:-}" ] && ps -o pid,etime,cmd -p "$PID" | tail -1
  echo "You can run the tests now (./run_sandplanner_tests.sh), or stop it with:  kill ${PID:-<pid>}"
  exit 1
fi
[ -d "$SAND/sand_planner" ] || { echo "SanD-Planner clone not found at $SAND (set SAND_ROOT=...)"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate navdp

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2   # keep CPU use low
export PYTHONPATH="$SAND/.deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$SAND"                                           # anything SanD-Planner writes lands in its own folder
exec python "$ROOT/tube_navdp/scripts/sandplanner_server.py" --sand-root "$SAND" --port "$PORT" "$@"
