#!/usr/bin/env bash
# Same sim tests with SanD-Planner (no tubes, no map: SanD-Planner's own path). Start ./start_sandplanner_server.sh
# first.
#
#   ./run_sandplanner_tests.sh box             # one test
#   ./run_sandplanner_tests.sh --repeat 3 box  # repeat for a success rate
#   ./run_sandplanner_tests.sh obstacles       # any test / group / option of run_sim_tests.sh works here
#
# Runs go to runs/tests/<test>_sandplanner_<n>_<time>/, so  python summarize_tests.py  compares it with the others
# (row "sandplanner").
set -uo pipefail
cd "$(dirname "$0")"
exec ./run_sim_tests.sh "$@" --modes sandplanner
