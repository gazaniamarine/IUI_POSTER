#!/usr/bin/env bash
# Same sim tests, A* + DWA: A* on the BEV map as the global path, the Dynamic Window Approach follows it. No server needed.
#
#   ./run_astar_dwa_tests.sh box                  # one test
#   ./run_astar_dwa_tests.sh --repeat 5 obstacles # repeat for a success rate; any test / group / option of run_sim_tests.sh works
#
# Runs go to runs/tests/<test>_astar_dwa_<n>_<time>/; python summarize_tests.py puts them next to the other methods.
set -uo pipefail
cd "$(dirname "$0")"
exec ./run_sim_tests.sh "$@" --modes astar_dwa
