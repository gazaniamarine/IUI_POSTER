#!/usr/bin/env bash
# NavDP, X-NavDP, SanD-Planner and ours (NavDP score-guided by the map, no tubes) on the Mars yard in Habitat.
# Start the servers you need first (each in its own terminal, from ../sim):
#   ./start_tube_server.sh (NavDP, 8890)   ./start_xnavdp_server.sh (8891)   ./start_sandplanner_server.sh (8892)
#
#   ./run_mars_tests.sh                                  # all 4 planners x all 5 flags
#   ./run_mars_tests.sh --planners ours behind           # only ours (uses the tube server, 8890)
#   ./run_mars_tests.sh --planners ours behind           # ours plans on the SAM2 segmentation BEV map (--map seg, default)
#   ./run_mars_tests.sh --planners ours --map truth all  # ours on the true rock map (--map depth: Habitat depth map)
#   ./run_mars_tests.sh flag_red flag_blue               # some flags
#   ./run_mars_tests.sh behind                           # rock right in front: start behind boulder 0-5, facing a flag
#   ./run_mars_tests.sh behind_1 behind_4 --gap 0.5      # some of them, rock 0.5 m ahead (default 1.0)
#   ./run_mars_tests.sh --planners navdp,xnavdp --repeat 3 all
#   ./run_mars_tests.sh --summary                        # the table over runs/tests/
#   ./run_mars_tests.sh --summary behind                 # only runs whose name starts with "behind"
# Any other --option is passed on to run_mars_habitat.py (e.g. --max-speed 0.3 --camera-height 0.4 --no-window).
set -uo pipefail
cd "$(dirname "$0")"
FLAGS=(flag_red flag_blue flag_yellow flag_green flag_purple)
BEHIND=(behind_0 behind_1 behind_2 behind_3 behind_4 behind_5)
PLANNERS="navdp,xnavdp,sandplanner,ours"
REPEAT=1
TESTS=()
PASS=()
finished() { for d in runs/tests/${1:-}*/; do [ -f "$d/metrics.json" ] && echo "$d"; done; }   # runs with results
while [ $# -gt 0 ]; do
  case "$1" in
    --planners) PLANNERS="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --summary) shift; exec /usr/bin/env python3 ../sim/summarize_tests.py $(finished "${1:-}") ;;
    all) TESTS+=("${FLAGS[@]}"); shift ;;
    behind) TESTS+=("${BEHIND[@]}"); shift ;;
    flag_*|behind_*) TESTS+=("$1"); shift ;;
    --*) PASS+=("$1"); if [ $# -gt 1 ] && [[ "$2" != --* ]] && [[ "$2" != flag_* ]] && [[ "$2" != behind* ]] && [ "$2" != all ]; then
           PASS+=("$2"); shift; fi; shift ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done
[ ${#TESTS[@]} -eq 0 ] && TESTS=("${FLAGS[@]}")

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate habitat

port() { case "$1" in navdp|ours) echo 8890 ;; xnavdp) echo 8891 ;; sandplanner) echo 8892 ;; esac; }
IFS=, read -ra PL <<< "$PLANNERS"
for p in "${PL[@]}"; do
  if ! ss -ltn 2>/dev/null | grep -q ":$(port "$p") "; then
    echo "[error] the $p server (port $(port "$p")) is not running: start it from ../sim (see the top of this file)"
    exit 1
  fi
done

for n in $(seq 1 "$REPEAT"); do
  for t in "${TESTS[@]}"; do
    for p in "${PL[@]}"; do
      echo "=== $t  $p  run $n/$REPEAT"
      case "$t" in behind_*) T=(--test "$t") ;; *) T=(--goal "$t") ;; esac
      PYTHONUNBUFFERED=1 python run_mars_habitat.py --planner "$p" "${T[@]}" "${PASS[@]}"
    done
  done
done
python ../sim/summarize_tests.py $(finished)
