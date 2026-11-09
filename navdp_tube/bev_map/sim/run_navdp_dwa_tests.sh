#!/usr/bin/env bash
# Same sim tests, NavDP + DWA: plain NavDP's path, the Dynamic Window Approach follows it on the BEV map. Start ./start_tube_server.sh first.
#
#   ./run_navdp_dwa_tests.sh box                  # one test
#   ./run_navdp_dwa_tests.sh --repeat 5 obstacles # repeat for a success rate; any test / group / option of run_sim_tests.sh works
#
# Runs go to runs/tests/<test>_navdp_dwa_<n>_<time>/; python summarize_tests.py puts them next to the other methods.
set -uo pipefail
cd "$(dirname "$0")"
exec ./run_sim_tests.sh "$@" --modes navdp_dwa
