#!/usr/bin/env bash
# Sim tests of BEV map + tube-guided NavDP (no Qwen). Start ./start_tube_server.sh in another terminal first.
#
#   ./run_sim_tests.sh duck                     # one test (tube)
#   ./run_navdp_tests.sh duck                   # plain NavDP only: separate script (same options)
#   ./run_xnavdp_tests.sh duck                  # X-NavDP only (needs ./start_xnavdp_server.sh)
#   ./run_sandplanner_tests.sh duck             # SanD-Planner only (needs ./start_sandplanner_server.sh)
#   ./run_benchmark.sh box                      # the 4 methods of the study: A*, A*+DWA, NavDP+DWA, ours
#   ./run_sim_tests.sh --modes astar,astar_dwa,navdp_dwa box   # any of them (A* needs no server)
#   ./run_smooth_tests.sh duck                  # tubes with the smooth push (this script: the nearest push)
#   ./run_sim_tests.sh --consistency 0.3 --beta 2 box   # tube smoothing: mu (previous plan), beta (projection)
#   ./run_sim_tests.sh --split 1 --rebuild --lyapunov min box   # guide early, rebuild tubes, Lyapunov pull
#                                               (--delta D sets its rate; rows like tube+s1+r+lmin)
#   ./run_sim_tests.sh --modes obstacle --split 0 --guide-until 5 box   # guide steps 0-4 only, then NavDP alone
#                                               (row obstacle+s0+u5; metrics: chosen_in_cells_share, model_v_mean)
#   ./run_sim_tests.sh --modes obstacle --split 1 --lyapunov min --beta 2 box   # no tubes at all (score-guided)
#   ... --push-step 0.5 --push-margin 0.05     # obstacle push: reach per waypoint, m past the obstacles (+p0.5+m0.05)
#   ./run_sim_tests.sh objects                  # group: duck tv thermos books
#   ./run_sim_tests.sh study                    # group: duck tv thermos books box box_tv box_duck
#   ./run_sim_tests.sh obstacles                # group: every test with added boxes
#   ./run_sim_tests.sh all                      # both groups
#   ./run_sim_tests.sh --modes tube,navdp box_tv      # navdp = NavDP + the map safety layer (override, stop)
#   ./run_sim_tests.sh --modes pure duck              # NavDP by itself: no map, no override, no stop, no stt
#   ./run_sim_tests.sh --repeat 3 gap_tight           # NavDP samples randomly: repeat for a success rate
#   ./run_sim_tests.sh --reach-edge 0.30 thermos      # how close to the object's surface counts as reached (0.35)
#   ./run_sim_tests.sh --web duck               # watch live at http://localhost:8090 (needs an ssh tunnel, see below)
#   ./run_sim_tests.sh --web-lan duck           # page open to the network: http://<gpu1 IP>:8090
#   ./run_sim_tests.sh --threads 2 duck         # fewer CPU threads (default 4)
#   ./run_sim_tests.sh --max-time 200 slalom    # sim seconds before 'timeout' (default 150)
#   ./run_sim_tests.sh --summary                # only print the tables of finished runs (all of runs/tests/)
#   ./run_sim_tests.sh --viewer box_tv          # on a machine with a screen: also the MuJoCo 3D viewer
#   ./run_sim_tests.sh --no-window duck         # no dashboard window (it opens by itself when there is a screen)
#   ./run_sim_tests.sh --no-top duck            # no top-down camera panel (shown beside the map by default)
#
# Tests (goal = an object; the objects are solid; reached = rover within 0.35 m of the object's surface):
#   A. objects:   duck  tv  thermos  books
#   B1. box on the straight line:        box (goal point 0.6,0)  box_thermos  box_tv  box_duck
#   B2. box a little to one side:        side_left (pass it on the right)  side_right (pass it on the left)
#   B3. gap between two boxes:           gap_wide (0.80 m)  gap_tight (0.55 m; rover is 0.20 m wide)
#   B4. three staggered boxes:           slalom
#   B7. box in front of the object:      near_duck  near_tv
#   duck_stt   duck with the tube's own speed law (controller stt) instead of the path follower
#   custom     your own goal/boxes: GOAL=thermos BOX="-0.6,-0.2,0.1 0.3,0.1,0.08" ./run_sim_tests.sh custom
#              (GOAL = an object name or x,y)
#   ./run_sim_tests.sh --list   prints the tests with a description
#
# Seeing the page from your own computer over plain ssh: on YOUR computer run
#   ssh -N -L 8090:localhost:8090 nahar@<gpu1>      (leave it open), then open http://localhost:8090
#
# Each run writes runs/tests/<test>_<mode>_<n>_<time>/ (video.mp4, final.png, metrics.json).
set -uo pipefail
cd "$(dirname "$0")"

OBJECTS=(duck tv thermos books)
STUDY=(duck tv thermos books box box_tv box_duck)   # the 4 objects + a box on the line (point, TV, duck)
OBSTACLES=(box box_thermos box_tv box_duck side_left side_right gap_wide gap_tight slalom near_duck near_tv)

REPEAT=1; WEB=""; EXTRA=""; THREADS=4; MAXT=150; SUMMARY_ONLY=0; MODES="tube"; TESTS=(); TUBE_SETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repeat) REPEAT="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --reach-edge) EXTRA="$EXTRA --reach-edge $2"; shift 2 ;;
    --web) WEB="--web"; shift ;;
    --web-lan) WEB="--web --host 0.0.0.0"; shift ;;
    --viewer) EXTRA="$EXTRA --viewer"; shift ;;
    --no-window) EXTRA="$EXTRA --no-window"; shift ;;
    --no-top) EXTRA="$EXTRA --no-top"; shift ;;
    --threads) THREADS="$2"; shift 2 ;;
    --max-time) MAXT="$2"; shift 2 ;;
    --consistency) TUBE_SETS+=(nav.tube_consistency="$2"); shift 2 ;;
    --beta) TUBE_SETS+=(nav.tube_smooth="$2"); shift 2 ;;
    --split) TUBE_SETS+=(nav.tube_split="$2"); shift 2 ;;
    --rebuild) TUBE_SETS+=(nav.tube_rebuild=true); shift ;;
    --lyapunov) TUBE_SETS+=(nav.tube_lyapunov="$2"); shift 2 ;;
    --delta) TUBE_SETS+=(nav.tube_lyapunov_delta="$2"); shift 2 ;;
    --guide-until) TUBE_SETS+=(nav.tube_guide_until="$2"); shift 2 ;;
    --push-step) TUBE_SETS+=(nav.tube_push_step="$2"); shift 2 ;;
    --push-margin) TUBE_SETS+=(nav.tube_push_margin="$2"); shift 2 ;;
    --summary) SUMMARY_ONLY=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n 2,39p "$0"; exit 0 ;;
    objects) TESTS+=("${OBJECTS[@]}"); shift ;;
    study) TESTS+=("${STUDY[@]}"); shift ;;
    obstacles) TESTS+=("${OBSTACLES[@]}"); shift ;;
    all) TESTS+=("${OBJECTS[@]}" "${OBSTACLES[@]}"); shift ;;
    *) TESTS+=("$1"); shift ;;
  esac
done

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolo
mkdir -p runs/tests

[ "${LIST:-0}" = 1 ] && { python run_tube_sim.py --list; exit 0; }
[ "$SUMMARY_ONLY" = 1 ] && { python summarize_tests.py --runs; exit 0; }
[ ${#TESTS[@]} -eq 0 ] && { echo "Name tests or a group (objects, obstacles, all). See ./run_sim_tests.sh --help"; exit 1; }

server_for(){
  case "$1" in
    astar|astar_dwa) echo "none" ;;                                   # no NavDP: no server needed
    xnavdp) echo "8891 X-NavDP ./start_xnavdp_server.sh" ;;
    sandplanner) echo "8892 SanD-Planner ./start_sandplanner_server.sh" ;;
    *) echo "8890 tube-NavDP ./start_tube_server.sh" ;;
  esac
}
IFS=',' read -ra MODE_LIST <<< "$MODES"
for mode in "${MODE_LIST[@]}"; do
  read -r PORT NAME SCRIPT <<< "$(server_for "$mode")"
  [ "$PORT" = none ] && continue
  if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "The $NAME server (port $PORT) is not running. Start it in another terminal:  $SCRIPT"
    exit 1
  fi
  SPID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -n "$SPID" ] && ps -o stat= -p "$SPID" | grep -q T; then
    echo "The $NAME server (port $PORT, pid $SPID) is PAUSED (Ctrl+Z) and answers nothing."
    echo "In its terminal run 'fg' then Ctrl+C, or: kill -9 $SPID; then start it again: $SCRIPT"
    exit 1
  fi
  # a server keeps the code it started with: refuse to test on one older than the code
  if [ -n "$SPID" ]; then
    STARTED=$(date -d "$(ps -o lstart= -p "$SPID")" +%s)
    CODE=(../../tube_navdp/navdp ../../tube_navdp/scripts)
    [ "$PORT" = 8891 ] && CODE+=(../../NavDP/baselines/x-navdp/eval/src)
    [ "$PORT" = 8892 ] && CODE+=(../../../sandplanner/sand_planner)
    NEWEST=$(find "${CODE[@]}" -name '*.py' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1)
    if [ -n "$NEWEST" ] && [ "${NEWEST%%.*}" -gt "$STARTED" ]; then
      echo "The $NAME server (pid $SPID) started before the code changed (${NEWEST#* }):"
      echo "it still runs the old code. Restart it: Ctrl+C in its terminal (or kill $SPID), then $SCRIPT"
      exit 1
    fi
  fi
done

DONE=(); STOP=0
trap 'STOP=1' INT                     # Ctrl+C: finish this run's files, then stop (no next run)
for t in "${TESTS[@]}"; do
  SETS=()
  case "$t" in
    duck_stt) ARGS=(--scenario duck); SETS=(nav.controller=stt) ;;
    custom)
      [ -z "${GOAL:-}" ] && { echo "custom needs GOAL=<object or x,y> (and optional BOX=\"x,y,half ...\")"; exit 1; }
      ARGS=(--goal "$GOAL"); [ -n "${BOX:-}" ] && ARGS+=(--box $BOX) || ARGS+=(--box) ;;
    *)
      if ! python run_tube_sim.py --list | awk '{print $1}' | grep -qx "$t"; then
        echo "unknown test '$t' (see ./run_sim_tests.sh --list)"; exit 1
      fi
      ARGS=(--scenario "$t") ;;
  esac
  for mode in "${MODE_LIST[@]}"; do
    for i in $(seq 1 "$REPEAT"); do
      OUT="runs/tests/${t}_${mode}_${i}_$(date +%H%M%S)"
      echo "=== $t, mode $mode (run $i/$REPEAT) -> $OUT"
      case "$mode" in
        tube) MSET=(nav.mode=tube nav.tube_push=nearest) ;;
        smooth) MSET=(nav.mode=tube nav.tube_push=smooth nav.tube_split=1 nav.tube_rebuild=true
                      nav.tube_lyapunov=min nav.tube_lyapunov_delta=0.3) ;;   # ours: guide from step 1, rebuild, Lyapunov
        obstacle_lya) MSET=(nav.mode=obstacle nav.tube_split=1 nav.tube_lyapunov=min nav.tube_smooth=2) ;;
                      # = --modes obstacle --split 1 --lyapunov min --beta 2: no tubes (run_smooth_tests.sh --with-obstacle)
        pure) MSET=(nav.mode=navdp nav.pure_navdp=true nav.controller=track) ;;
        xnavdp) MSET=(nav.planner=xnavdp nav.mode=navdp nav.pure_navdp=true nav.controller=track) ;;
        sandplanner) MSET=(nav.planner=sandplanner nav.mode=navdp nav.pure_navdp=true nav.controller=track) ;;
        astar) MSET=(nav.planner=astar) ;;
        astar_dwa) MSET=(nav.planner=astar_dwa) ;;
        navdp_dwa) MSET=(nav.planner=navdp_dwa nav.mode=navdp nav.pure_navdp=true nav.controller=track) ;;
        *) MSET=(nav.mode="$mode") ;;
      esac
      PYTHONUNBUFFERED=1 python run_tube_sim.py "${ARGS[@]}" --name "$t" --set "${MSET[@]}" "${TUBE_SETS[@]}" "${SETS[@]}" \
        --max-time "$MAXT" --threads "$THREADS" $WEB $EXTRA --out "$OUT" 2>&1 \
        | grep --line-buffered -v -i "warning" \
        | grep --line-buffered -E "^t=|arrived|stuck|^scenario|Open http|=====|\[nav\]|Error|error"
      [ "${PIPESTATUS[0]}" = 130 ] && STOP=1   # the run was quit (Ctrl+C or q in its window)
      [ -f "$OUT/metrics.json" ] && DONE+=("$OUT")
      [ "$STOP" = 1 ] && { echo "stopped by hand: no more runs"; break 3; }
      read -r PORT NAME SCRIPT <<< "$(server_for "$mode")"
      if [ "$PORT" != none ] && ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "The $NAME server stopped during $t: restart $SCRIPT and run again."; break 3
      fi
    done
  done
done
[ ${#DONE[@]} -gt 0 ] && python summarize_tests.py --runs "${DONE[@]}" || echo "no finished runs"
