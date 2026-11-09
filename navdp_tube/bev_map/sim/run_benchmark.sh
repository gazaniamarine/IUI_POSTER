#!/usr/bin/env bash
# The benchmark: every test with the 4 methods of the study, then one comparison table.
#   1. astar      A* on the BEV map + path tracker                 (no server)
#   2. astar_dwa  A* on the BEV map + DWA                          (no server)
#   3. navdp_dwa  plain NavDP + DWA on the BEV map                 (tube server)
#   4. smooth     ours: NavDP + tubes on the BEV map, smooth push, split 1, rebuild, Lyapunov min  (tube server)
# Start ./start_tube_server.sh first (methods 3 and 4).
#
#   ./run_benchmark.sh box                       # one test, 4 runs
#   ./run_benchmark.sh --repeat 5 obstacles      # NavDP samples randomly: repeat for success rates
#   ./run_benchmark.sh --methods astar,smooth box  # a subset
#   any other option of run_sim_tests.sh works too (--web, --max-time, --consistency / --beta for ours, ...)
set -uo pipefail
cd "$(dirname "$0")"
METHODS="astar,astar_dwa,navdp_dwa,smooth"   # ours = the smooth push
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --methods) METHODS="$2"; shift 2 ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec ./run_sim_tests.sh "${ARGS[@]}" --modes "$METHODS"
