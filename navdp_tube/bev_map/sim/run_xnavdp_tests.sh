#!/usr/bin/env bash
# Same sim tests with X-NavDP (no tubes, no map: X-NavDP's own path). Start ./start_xnavdp_server.sh first.
#
#   ./run_xnavdp_tests.sh duck                # one test
#   ./run_xnavdp_tests.sh --repeat 5 box      # repeat for a success rate
#   ./run_xnavdp_tests.sh objects             # any test / group / option of run_sim_tests.sh works here
#
# Runs go to runs/tests/<test>_xnavdp_<n>_<time>/, next to the tube (_tube_) and NavDP (_pure_) runs, so
#   python summarize_tests.py                 # compares all three: one row per test and planner
set -uo pipefail
cd "$(dirname "$0")"
exec ./run_sim_tests.sh "$@" --modes xnavdp
