#!/usr/bin/env bash
# Same sim tests, OUR METHOD: tube-guided NavDP with the SMOOTH push (the nearest push, then smoothed inside free
# boxes round the pushed waypoints), guided from denoising step 1 (--split 1), tubes rebuilt at every guided step
# (--rebuild) and the Lyapunov pull (--lyapunov min --delta 0.3). Any of these can be overridden, e.g. --split 2.
# run_sim_tests.sh uses the plain nearest push. Both use the same server: start ./start_tube_server.sh first.
#
#   ./run_smooth_tests.sh duck                # one test
#   ./run_smooth_tests.sh --repeat 5 box      # repeat for a success rate
#   ./run_smooth_tests.sh obstacles           # any test / group / option of run_sim_tests.sh works here
#   ./run_smooth_tests.sh --consistency 0.3 --beta 2 box   # plus consistency (mu) and smooth projection (beta)
#                                             -> runs <test>_smooth_*, row "smooth+c0.3+b2+s1+r+lmin0.3"
#   ./run_smooth_tests.sh --guide-until 5 box # pull only at steps 1-4, then NavDP alone (must be in [split+1, 10])
#                                             -> row "smooth+s1+r+u5+lmin0.3"; the summary's "in%" / "V" columns
#                                                show whether NavDP keeps the paths clear once the pull stops
#   ./run_smooth_tests.sh --with-obstacle box # also the score-guided baseline with no tubes at all, the same as
#                                             ./run_sim_tests.sh --modes obstacle --split 1 --lyapunov min --beta 2
#                                             -> row "obstacle+b2+s1+lmin" (fast: no tubes are built)
#   ./run_smooth_tests.sh --with-obstacle --no-smooth box   # only that baseline (skip ours)
#   (other options, e.g. --guide-until 5 or --delta 0.5, apply to every run)
#
# Ctrl+C (or q in the window) stops the current run and the script: no next run starts.
# Runs go to runs/tests/<test>_smooth_<n>_<time>/ (tube runs with the nearest push: <test>_tube_...), so
#   python summarize_tests.py                 # compares them: a "tube" row and a "smooth+s1+r+lmin0.3" row per test
set -uo pipefail
cd "$(dirname "$0")"

SMOOTH=1; OBST=0; SPLIT=1; UNTIL=0; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --with-obstacle) OBST=1; shift ;;
    --no-smooth) SMOOTH=0; shift ;;
    --split) SPLIT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --guide-until) UNTIL="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help) sed -n 2,23p "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
MODES=()
[ "$SMOOTH" = 1 ] && MODES+=(smooth)
[ "$OBST" = 1 ] && MODES+=(obstacle_lya)
[ ${#MODES[@]} -eq 0 ] && { echo "--no-smooth leaves nothing to run: add --with-obstacle"; exit 1; }
# the server refuses every plan with a bad guide_until: catch it before any run starts
if [ "$UNTIL" != 0 ] && { [ "$UNTIL" -le "$SPLIT" ] || [ "$UNTIL" -gt 10 ]; }; then
  echo "--guide-until must be 0 (off) or in [split + 1, 10] (split is $SPLIT)"; exit 1
fi
exec ./run_sim_tests.sh "${ARGS[@]}" --modes "$(IFS=,; echo "${MODES[*]}")"
