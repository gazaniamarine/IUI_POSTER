# Mars yard in Habitat: NavDP, X-NavDP, SanD-Planner, ours

Point-goal drives on `~/Desktop/mars_environment/marsyard_with_rocks.glb` (50 x 50 m yard, 6 boulders, 5 flags),
rendered by habitat-sim (conda env `habitat`). The planners are the same servers as in `../sim`.

| file | purpose |
|---|---|
| `run_mars_habitat.py` | one run: a planner drives from (0, 0) to a flag (or any x,y) |
| `run_mars_tests.sh` | every planner x every flag, then the comparison table |
| `pick_start.py` | click a start (and goal) on the yard map, see the rover's view, ENTER runs a planner from there |
| `mars_scene.py` | reads the GLB: terrain height, rock / flag footprints, Habitat stage config |

## Run
```bash
# terminal 1-3 (from ../sim): the servers you need
./start_tube_server.sh          # NavDP        (8890)
./start_xnavdp_server.sh        # X-NavDP      (8891)
./start_sandplanner_server.sh   # SanD-Planner (8892)

# terminal 4 (from here)
./run_mars_tests.sh                                   # 4 planners x 5 flags
./run_mars_tests.sh --planners ours behind            # ours only (tube server, 8890)
./run_mars_tests.sh --planners navdp,sandplanner --repeat 3 flag_red flag_blue
./run_mars_tests.sh --summary                         # table over runs/tests/
conda activate habitat
python run_mars_habitat.py --planner xnavdp --goal flag_green     # one run (live window if there is a screen)
python run_mars_habitat.py --list                     # flags, behind-rock starts
./run_mars_tests.sh behind                            # rock right in front of the rover (behind_0..5)
python pick_start.py                                  # pick the start yourself (needs a screen)
```
`pick_start.py`: left click = start (left drag = start + heading), right click = goal (snaps to a flag),
A/D turn, F face goal, 1/2/3/4 planner (4 = ours), +/- zoom, ENTER run, P print the command, Q quit.
Each run writes `runs/tests/<flag>_<planner>_<time>/`: `video.mp4` (camera | depth | top-down map with the
planner's candidate paths in grey and its chosen one in green), `final.png`, `metrics.json`.

## How it works
- **Scene**: Habitat would treat the GLB as Z-up and stand the yard on its side, so it is loaded through a stage
  config with up = +Y (written to `runs/`). Planner frame (z up, as `rock_layout.txt`): x = X, y = -Z, height = Y.
- **Rover**: kinematic unicycle (moves exactly as commanded), camera `--camera-height` (0.5 m) above the mean
  terrain under it, kept level (no roll / pitch from the slope). 640x480, hfov 79 deg (SanD-Planner's training
  camera). `--max-speed 0.5`, `--max-yaw-rate 0.6`.
- **Planning**: one blocking plan per camera frame (`--frame-dt 0.2` s of sim time), Habitat's true depth; the same
  `TubeNavigator` + `PathFollower` as the MuJoCo sim, with `nav.pure_navdp=true` (no map, no tubes).
- **Ours** (`--planner ours`): NavDP score-guided away from the map's obstacles, no tubes: the same settings as
  `../sim/run_smooth_tests.sh --with-obstacle --no-smooth` (mode obstacle, split 1, Lyapunov min, beta 2), with the
  driver's farthest-path choice. Its map (`--map depth`, default): Habitat depth pixels 10 cm to 1 m above the known
  terrain, a cell occupied after 2 hits, kept for the run; cells within `--obstacle-range` (4 m) go to the planner,
  grown by the inflation radius (robot radius + nav.inflation_margin). `--map truth` = the true rock footprints.
- **Collisions**: rocks and flag poles are solid footprints (vertices 5 cm to 1 m above the terrain; the flag cloth
  is higher). A step into one (rover circle `--robot-radius` 0.3 m) is not taken and counts as collision time.
  Terrain is never an obstacle: hills and the ditch south of the yard can be driven over.
- **Reached**: the rover circle within `--reach-edge` (0.5 m) of the goal pole. **Stuck**: < 0.25 m in 20 s.
