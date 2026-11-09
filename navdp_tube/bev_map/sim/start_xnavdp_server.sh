#!/usr/bin/env bash
# Start the X-NavDP server (port 8891). Keep this terminal open while testing; Ctrl+C stops it.
#   ./start_xnavdp_server.sh                  # GPU, wheeled embodiment
#   ./start_xnavdp_server.sh --stuck-xy 0.08  # other options: python ../../tube_navdp/scripts/xnavdp_server.py -h
# Checkpoint: NavDP/baselines/x-navdp/pretrain_model/x-navdp_posttrain.ckpt (not in git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # repo root
PORT=8891

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  PID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  echo "Port $PORT is already in use (pid ${PID:-?}): the server is probably running already."
  [ -n "${PID:-}" ] && ps -o pid,etime,cmd -p "$PID" | tail -1
  if [ -n "${PID:-}" ] && ps -o stat= -p "$PID" | grep -q T; then
    echo "It is PAUSED (Ctrl+Z): it holds the port but answers nothing. Remove it and start again:"
    echo "  kill -9 $PID && $0"
    exit 1
  fi
  echo "You can run the tests now (./run_xnavdp_tests.sh), or stop it with:  kill ${PID:-<pid>}"
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate navdp

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2   # keep CPU use low
exec python "$ROOT/tube_navdp/scripts/xnavdp_server.py" \
  --checkpoint "$ROOT/NavDP/baselines/x-navdp/pretrain_model/x-navdp_posttrain.ckpt" \
  --port "$PORT" "$@"
