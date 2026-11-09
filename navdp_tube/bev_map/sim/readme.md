# BEV map – simulation of this rover (MuJoCo)

Simulation-only tools for the UGV rover in `../../simulation`. Not needed on a real robot and not part
of the shared package. They use the pipeline files one folder up (`../`), plus `config_sim.yaml`
(merged on top of `../config.yaml`). Recordings and saved maps go to `sim/runs/`.

| file | purpose |
|---|---|
| `run_bev_sim.py` | live map while driving the simulated rover |
| `sim_source.py` | renders camera frames + exact poses from MuJoCo |
| `record_sim.py` | records a scripted drive (RGB, true depth, poses) |
| `evaluate_sim.py` | scores maps against the true obstacle layout |
| `ground_truth.py` | true occupancy from the scene + precision/recall |
| `run_tube_sim.py` | one closed-loop navigation run (any method) in the arena; the `*.sh` scripts below call it |
| `start_*_server.sh`, `run_*.sh` | planner servers and test scripts: see **Test scripts** below |
| `summarize_tests.py` | the comparison table of finished test runs |

## Test scripts: what to run, when, how
All of them run from this folder (`cd bev_map/sim` from the repo root). A test needs up to two terminals:
**terminal 1** keeps a planner server running, **terminal 2** runs the tests. The scripts activate the conda
envs themselves (`navdp` for the servers, `yolo` for the sim).

### 1. Start the server (terminal 1, leave it open)
| script | port | needed by | when to (re)start |
|---|---|---|---|
| `./start_tube_server.sh` | 8890 | ours (tube, smooth), NavDP alone, NavDP + DWA | before testing those; **again after every `git pull` or change in `tube_navdp/`** |
| `./start_xnavdp_server.sh` | 8891 | X-NavDP alone | before X-NavDP tests; again after a change in `tube_navdp/` or `x-navdp` |
| none | – | A*, A* + DWA | – |

Both servers can run at once. Stop a server with **Ctrl+C** (never Ctrl+Z: that only pauses it, it keeps the port
and answers nothing). The test scripts refuse to start when the server they need is missing, paused or older than
the code, and print the command that fixes it.

### 2. Run the tests (terminal 2)
| script | method | when to use it |
|---|---|---|
| `./run_benchmark.sh` | the 4 methods of the study: A*, A* + DWA, NavDP + DWA, ours (smooth push) | the study / paper numbers: one command, one comparison table |
| `./run_sim_tests.sh` | ours: NavDP + tubes on the BEV map (nearest push) | developing / checking the tube method |
| `./run_smooth_tests.sh` | **ours**: smooth push + `--split 1 --rebuild --lyapunov min --delta 0.3` (as in the benchmark) | checking our method alone |
| `./run_astar_tests.sh` | A* on the BEV map + path tracker (no server) | baseline 1 alone |
| `./run_astar_dwa_tests.sh` | A* + Dynamic Window Approach (no server) | baseline 2 alone |
| `./run_navdp_dwa_tests.sh` | plain NavDP's path + DWA | baseline 3 alone |
| `./run_navdp_tests.sh` | NavDP alone, no map, no tubes | how NavDP does by itself |
| `./run_xnavdp_tests.sh` | X-NavDP alone, no map, no tubes | how X-NavDP does by itself |

Every script takes test names and the same options:
```bash
./run_benchmark.sh box                       # one test
./run_benchmark.sh box duck gap_tight        # several
./run_benchmark.sh objects                   # a group: objects (duck tv thermos books), obstacles, all
./run_benchmark.sh study                     # group: duck tv thermos books + box, box_tv, box_duck
./run_benchmark.sh --repeat 5 all            # NavDP samples randomly: repeat for success rates
./run_benchmark.sh --methods astar,smooth box  # (run_benchmark.sh only) a subset of the 4 methods
./run_sim_tests.sh --list                    # every test with a description
```

| option | what it does |
|---|---|
| `--repeat N` | N runs of each test and method |
| `--web` | watch live at http://localhost:8090 (VS Code forwards the port; plain ssh: `ssh -N -L 8090:localhost:8090 nahar@gpu1`) |
| `--web-lan` | the same page, open to the network (http://<gpu1 IP>:8090) |
| `--no-window` | no dashboard window (it opens by itself when there is a screen) |
| `--viewer` | also the MuJoCo 3D viewer (needs a screen) |
| `--max-time S` | sim seconds before a run ends as `timeout` (default 150) |
| `--reach-edge M` | object goals: reached when the rover is this close to the object's surface (default 0.35 m) |
| `--threads N` | CPU threads (default 4) |
| `--consistency MU`, `--beta B` | ours only: tube smoothing, pull toward the previous plan (try 0.3) and smooth projection (try 2) |
| `--modes a,b` | (`run_sim_tests.sh`) any methods: `tube smooth astar astar_dwa navdp_dwa pure xnavdp` |

Tests (group `study` = the 4 objects + `box`, `box_tv`, `box_duck`): `duck tv thermos books` (drive to an object), `box box_thermos box_tv box_duck` (a box on the way),
`side_left side_right` (a box just to one side), `gap_wide gap_tight` (two boxes, 0.80 / 0.55 m gap), `slalom`
(three boxes), `near_duck near_tv` (a box in front of the goal), `duck_stt` (ours with the tube's own speed law),
`custom` (`GOAL=thermos BOX="-0.6,-0.2,0.1" ./run_sim_tests.sh custom`).

### 3. Read the results
```bash
./run_sim_tests.sh --summary                 # the table over every finished run in runs/tests/
python summarize_tests.py runs/tests/box_*   # only some runs
```
One row per test and method: `success` (reached without touching anything), `stuck` / `tmout` (runs that ended
without reaching), `coll` (runs that touched something), `time` and `path/line` (successful runs), `clear`
(closest the rover came to anything but the goal; > ~0.05 m is good), `cert%` (ours: plans with a certified
tube), `map P/R` (BEV map precision / recall). Each run's folder `runs/tests/<test>_<method>_<n>_<time>/` has
`video.mp4`, `final.png` and `metrics.json`.

While a run goes, the terminal prints a status line every 2 s, e.g.
`driving  goal 1.35 m  v=0.09 w=-0.35  tubes 0/1  tight  r0.20 ...` (ours) or `... A*+DWA  2 ms  90/90 ok`.
`[arrived]`, `[stuck]` (moved < 0.10 m in 20 s) or `[error]` (the server failed, with the reason) ends it.

### Typical sessions
```bash
# the benchmark (paper numbers)
./start_tube_server.sh                                   # terminal 1
./run_benchmark.sh --repeat 5 study                      # terminal 2: the 7 study tests (4 methods x 7 x 5 runs)
./run_benchmark.sh --repeat 5 all                        # or every test (long: 4 methods x 15 tests x 5 runs)
./run_sim_tests.sh --summary

# after changing the tube code
#   terminal 1: Ctrl+C, then ./start_tube_server.sh
./run_sim_tests.sh box duck                              # terminal 2: quick check of ours
./run_benchmark.sh --methods navdp_dwa,smooth box        # compare with a baseline

# a quick check of the baselines alone (no server needed)
./run_benchmark.sh --methods astar,astar_dwa obstacles
```

## Run it

### Over SSH / VS Code Remote (no display) → in the browser
```bash
conda activate yolo
cd bev_map/sim                     # from the repo root (here ~/Desktop/navdp_tube)
python run_bev_sim.py            # or: python run_bev_sim.py --auto
```
With no `DISPLAY` the script switches to browser mode by itself and prints
`Open http://localhost:8090`. VS Code forwards the port (a pop-up appears, or see the
**Ports** tab); open the link in your local browser. Without VS Code:
`ssh -L 8090:localhost:8090 nahar@gpu1` and open the same link.
Click the page once, then drive with the keyboard (hold an arrow to move, release to stop).
The MuJoCo 3D viewer cannot open without a display, so the page shows a chase camera instead.
Use `--port 8091` if 8090 is taken; Ctrl+C in the terminal or Q in the page quits (and saves).

### On a machine with a screen (sitting at gpu1, or ssh -X)
```bash
python run_bev_sim.py --viewer   # MuJoCo 3D view + "BEV" window, you drive
python run_bev_sim.py            # only the "BEV" window, you drive
python run_bev_sim.py --auto     # scripted drive through the arena (good for a demo)
python run_bev_sim.py --depth sim               # reference map from perfect MuJoCo depth
python run_bev_sim.py --set grid.resolution=0.05  # override any config.yaml value
```
The first start takes ~10 s while Depth Anything loads. Then the **BEV** window shows:
camera | corrected Depth Anything depth (with the per-frame fit `a`, `b`) | global map | robot view.

Keys (click the BEV window / browser page first; with `--viewer` the arrows also work in the 3D window):

| key | action |
|---|---|
| ↑ ↓ ← → or W S A D | drive forward / back / turn (in the browser: hold to drive) |
| SPACE | stop |
| J / L, I / K | pan / tilt the camera |
| G | show the true obstacle outline (red) over the map |
| R | clear the map |
| P | save map + screenshot |
| Q / ESC | quit (also saves) |

Map colours: white = free, dark = obstacle, grey = not seen yet, orange = robot-radius safety
margin, green lines = camera view, orange line = path driven.
Saved to `sim/runs/live_<time>/`: `_dashboard.png`, `_bev.png`, `.npy` (ROS int8: -1 unknown,
0-100 occupancy %), `.pgm` + `.yaml` (ROS map_server).

Tip for a demo: drive slowly, stop and turn in place to look around; obstacles turn dark after
a few frames of agreement, single bad frames do not show up.

## Check accuracy against the simulator
```bash
MUJOCO_GL=egl python record_sim.py --out runs/sim_route     # record a scripted drive (~1.5 GB)
MUJOCO_GL=egl python evaluate_sim.py --run runs/sim_route --every 2
```
Writes `sim/runs/sim_route/eval/compare.png` and `metrics.json`. Current results:

| depth | precision | recall | noise cells |
|---|---|---|---|
| MuJoCo ground truth | 1.00 | 0.87 | 1 |
| Depth Anything, raw | 0.01 | – | 538 (map ~3x too large) |
| Depth Anything + floor fit (old default, smoothing 0.7) | 0.72 | 0.83 | 380 |
| Depth Anything + floor fit (default: smoothing 0.3, scale gate 10 %) | 0.78 | 0.83 | 275 |

Precision = share of mapped obstacle cells that are real (1-cell tolerance); recall = share of
visible real obstacles that were mapped. The remaining DA error is mostly walls drawn a few cm
too thick, not random specks.

