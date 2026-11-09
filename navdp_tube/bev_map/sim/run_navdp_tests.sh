#!/usr/bin/env bash
# Same sim tests, plain NavDP only (no tubes, no stt, no map: NavDP's own path). Start ./start_tube_server.sh first.
#
#   ./run_navdp_tests.sh duck                 # one test
#   ./run_navdp_tests.sh --repeat 5 duck      # repeat for a success rate
#   ./run_navdp_tests.sh objects              # any test / group / option of run_sim_tests.sh works here
#
# Runs go to runs/tests/<test>_pure_<n>_<time>/. Tube runs (./run_sim_tests.sh) go to <test>_tube_..., so
#   python summarize_tests.py                 # compares both: one tube row and one pure row per test
set -uo pipefail
cd "$(dirname "$0")"
exec ./run_sim_tests.sh "$@" --modes pure
