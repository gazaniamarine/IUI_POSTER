# Tube-guided NavDP

Point-goal navigation with [NavDP](https://github.com/InternRobotics/NavDP) where spatiotemporal tubes
(Das, Basu, Jagtap, [arXiv 2411.13834](https://arxiv.org/abs/2411.13834)) built on a BEV occupancy grid steer the
diffusion samples (score guidance), and an optional tube controller turns the result into velocity commands.

```
camera + goal + BEV map ──► NavDP denoises 16 paths in one pass
                             ├─ step 4: push waypoints clear of occupied cells; at every waypoint, cluster the
                             │          16 waypoints and grow one free box per cluster; paths with the same chain
                             │          of boxes share one tube (no side of any obstacle is chosen)
                             └─ steps 4-9: score guidance pulls each path into its tube (exactly inside at the end)
                            critic picks the best path whose tube is valid (Theorem 3.1 checked exactly)
                            controller ──► forward speed, yaw rate
```

You supply four things: RGB-D images, the robot's pose, a goal, and an occupancy grid. No
retraining; NavDP's official weights are used unchanged. Nothing below depends on Habitat. The maths is in the
docstrings of `stt.py` (tubes, certificate) and `guided_sampler.py` (score guidance).

## Use it in your own environment

### 1. Install

You need this repository, the [NavDP repository](https://github.com/InternRobotics/NavDP) (for `baselines/navdp`) and
its checkpoint. Python 3.9 or newer, a GPU for real-time use:

```bash
pip install torch torchvision numpy scipy diffusers flask requests pillow opencv-python matplotlib einops
```

### 2. Start the server

Run from the root of this repository. Use `--device cpu` if you have no GPU (about 1.3 s per plan on a laptop rather than about
0.13 s).

```bash
python scripts/tube_navdp_server.py --navdp-root /path/to/NavDP --checkpoint /path/to/navdp-cross-modal.ckpt --port 8890
```

Check that it works with the bundled example, which uses a stand-in camera and a perfect robot:

```bash
python scripts/tube_navdp_example.py --seconds 2
```

### 3. Write your loop

`TubeDriver` ([driver.py](driver.py)) is the only class you need. Plan a few times per second, command at your control
rate:

```python
import numpy as np
from navdp.tube.driver import DriverConfig, TubeDriver, WorldGrid

intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])        # your camera
driver = TubeDriver("http://127.0.0.1:8890", intrinsic, DriverConfig(mode="tube", controller="stt", robot_radius=0.6))

while not at_goal:
    pose = get_pose()                                              # (x, y, yaw)
    if time_to_plan:                                               # e.g. 5 Hz
        rgb, depth = get_rgbd()
        driver.plan(rgb, depth, goal_xy, pose, WorldGrid(occupied, res, (x_min, y_min)), now)
    v, omega, info = driver.command(pose, now)                     # e.g. 30 Hz
    send_velocity(v, omega)

driver.close()
```

Call `driver.reset()` at the start of each new run: it clears NavDP's memory of the last 8 frames.

### What you must provide

| Input | Format |
|---|---|
| `rgb` | `uint8` array `[H, W, 3]`, RGB order |
| `depth` | metres, `[H, W]`; 0, NaN or inf = invalid. NavDP ignores depth under 0.1 m and over 5 m |
| `intrinsic` | 3x3 camera matrix |
| `pose` | `(x, y, yaw)` in any fixed world frame: metres, radians, yaw counter-clockwise from +x |
| `goal_xy` | `(x, y)`, same frame |
| `bev` | `WorldGrid(occupied [ny, nx] bool, resolution, (x_min, y_min))`, same frame (row = y, like ROS), **without** the robot's size; unknown cells count as free. `None` = no obstacles |
| `now` | seconds, from any clock that only moves forward; use the same clock for `plan` and `command` |

### What you get back

`driver.command` returns `(v, omega, info)`:

| Value | Meaning |
|---|---|
| `v` | forward speed in m/s, between 0 and `max_speed` |
| `omega` | yaw rate in rad/s, counter-clockwise positive |
| `info["inside"]` | the controlled point is inside the tube (`None` if there is no tube) |
| `info["saturated"]` | the command was clipped to the robot's limits. The tube law's guarantee does not hold on these steps |
| `info["fallback"]` | no tube was solved, so the waypoint tracker's command was used |

`driver.plan` returns the planner's result:

| Field | Meaning |
|---|---|
| `.candidates` | all paths, `[16, 24, 3]`, forward and left in metres in the robot's frame (ignore the third channel) |
| `.values` | NavDP's critic score per path |
| `.trajectory` | the path chosen to drive |
| `.tube["certified"]` | per path: its tube is valid and every waypoint is inside |
| `.tube["chosen_certified"]` | whether the chosen path is certified |
| `.tube["tubes"]` | each tube: `feasible`, `theorem` (`slack >= lipschitz * gap`), `eta`, its boxes (`lo`, `hi`) and coefficients |

Log `chosen_certified`, `saturated` and `fallback`. They are how you tell whether the safety argument applies.

## Settings

Set your robot's limits; the defaults are for a slow rover:

```python
from navdp.tube.controller import FunnelConfig
from navdp.tube.driver import DriverConfig

config = DriverConfig(
    mode="tube",            # navdp | obstacle | tube
    controller="stt",       # stt: tube law | track: chase a waypoint of the chosen path
    robot_radius=0.6,       # metres, the map's occupied cells are grown by this
    funnel=FunnelConfig(max_speed=0.4, max_yaw_rate=0.8, gain=0.035, lookahead_m=0.2),
)
```

| Mode | During denoising | Choice |
|---|---|---|
| `navdp` | unchanged | best critic value |
| `obstacle` | pushes waypoints out of occupied cells | best critic value |
| `tube` | pulls samples into tubes | best certified path, else best value |

| Controller | Behaviour |
|---|---|
| `track` | chases one waypoint about 1 m ahead; faster, passes closer to obstacles |
| `stt` | the paper's law on a point `lookahead_m` ahead of the robot; wider berth, slower (the tube moves at 0.8 x `max_speed`) |

`lookahead_m` must be at most 0.25 m. Server options are in `python scripts/tube_navdp_server.py --help`, including
`--split` (the denoising step where tubes are built, default 4), `--tube-degree`, `--tube-margin` and
`--tube-samples-per-step` (together they set how steep a tube may be and still pass Theorem 3.1).
`--tube-push nearest|smooth` picks how waypoints are pushed out of occupied cells: `nearest` (default, the first
version: each waypoint to its own nearest free cell) or `smooth` (the nearest push, then the smoothest path closest to
NavDP's own that stays in a free box round each pushed waypoint: ADMM on 1/2 |p - x|^2 + beta/2 |D2 p|^2, boxes and
the 0.25 m rate as constraints).

Earlier guidance and Lyapunov guidance (all off by default; each also settable per request, and from bev_map as
`nav.tube_split`, `nav.tube_rebuild`, `nav.tube_lyapunov`, `nav.tube_lyapunov_delta`, or
`run_sim_tests.sh --split 1 --rebuild --lyapunov min --delta 0.3`):

- `--split 1` (or 2): guide from an earlier denoising step, while the sample can still change its way round.
- `--rebuild-tubes`: build the tubes again at every guided step from that step's estimate (not only at the split);
  the finished paths are certified against the last step's tubes. Costs one tube build per guided step.
- `--lyapunov min|ramp` with `--lyapunov-delta` (0.3): LyaGuide (arXiv 2607.14272) in discrete form. V = 1/2
  dist^2(x0, boxes). The pull a is the smallest with V(x0') <= (1 - delta) V(previous guided x0), i.e.
  a_req = max(0, 1 - sqrt(2 (1 - delta) V_prev / |x0 - P|^2)); `min` uses a = a_req (no more pull than needed),
  `ramp` uses max(alpha ramp, a_req). The last `--project-last` steps still project exactly. The tube report gives
  `lyapunov_v` (mean V per guided step) and `tube_builds`; run_tube_sim.py's metrics give `lyapunov_decrease_ok`.
  It also works in `obstacle` mode (no tubes, no linear programs): there V = 1/2 sum_j dist^2(waypoint j, free cells
  of the grown grid), P is the obstacle pull's target (nearest free cell, swept, smoothed with `--smooth`), and the
  pull is raised only where V would not fall by delta. E.g. `run_sim_tests.sh --modes obstacle --lyapunov min box`.
- `--guide-until u` (0 = off; `nav.tube_guide_until`, `run_sim_tests.sh --guide-until 5`): the tube / obstacle pull
  runs only at denoising steps split <= i < u (the alpha ramp and `--project-last` within that window), then NavDP
  denoises by itself; the consistency pull, if on, still runs. To see whether the model keeps the guided paths clear,
  the tube report always gives (with obstacles in the grid; V = 1/2 sum_j dist^2(waypoint j, free cells)):
  `model_v` (mean V of the model's own clean-path estimate at every step, before guidance), `final_v` (mean V of the
  finished paths), `final_in_cells` (how many have a waypoint in an occupied cell), `chosen_v` and `chosen_in_cells`.
  run_tube_sim.py's metrics: `chosen_in_cells_share`, `chosen_v_mean`, `final_v_mean`, `final_in_cells_mean`,
  `model_v_mean` (per step), and rows like `obstacle+s0+u5`.
- `--push-margin m` and `--push-step s` (obstacle mode; `nav.tube_push_margin`, `nav.tube_push_step`,
  `run_sim_tests.sh --push-margin 0.05 --push-step 0.5`; 0 and 0.25 = as before): the push lands m past the grown
  obstacles instead of on their edge (the driver checks whole segments, so edge paths fail it), and a pushed path may
  move up to s per waypoint and axis instead of NavDP's 0.25, so the sweep no longer drags a far push back into the
  obstacle. Above 0.25 the moves leave NavDP's [-1, 1]: the scheduler's clip is turned off for the call and done in
  the sampler instead ([-1, 1] at unguided steps, as before; +-4 s at guided ones). Rows like `obstacle+...+p0.5+m0.05`. `model_v` is computed only with guide_until > 0.
- No tubes on the finished paths: navdp / obstacle mode is pure score-guided denoising (no `stt.tubes_for` at all,
  nothing certified; the bev_map driver checks the chosen path against its map: clear / tight / turn). Only tube
  mode builds tubes, for its guidance, and certifies with those. (The `reject` mode and `--no-certify` are gone.)

Smoothness (both off by default, so the server runs as before):

- `--consistency mu` (try 0.3) with `--consistency-decay` (0.9) and `--consistency-from` (0): the driver sends the
  previous plan, moved into the new body frame and shifted by the waypoints already driven (`driver.carry_over`); the
  sampler adds E_c = 1/2 sum_j mu gamma^j |x0_j - x_prev_j|^2 to the guidance, before the tube pull (the tube still
  has the last word). Consecutive plans then agree, as X-NavDP's temporal-consistency guidance does.
- `--smooth beta` (try 2) with `--smooth-iters` (100): the projection into the boxes is the smoothest path in them
  (min 1/2 |p - x|^2 + beta/2 |D2 p|^2, boxes and the 0.25 m rate as constraints, ADMM, then the exact sweep) instead
  of a per-waypoint clamp; a sample with no tube is no longer pinned to its pushed path but gets that path smoothed.

NavDP's waypoints carry no timing, so your speed limits matter only in the controllers. A robot much faster than
0.4 m/s has not been tried.

### Without HTTP, or with another planner

- `navdp.tube.agent.make_agent` runs the same agent in your own process; call `agent.step_pointgoal_tube(...)`.
- `navdp.tube.guided_sampler.guided_denoise` works with any epsilon-predicting trajectory diffusion model: pass your
  `predict_noise(x, k)` and scheduler. It assumes NavDP's scale (moves in [-1, 1], one unit = 4 m).
- `navdp.tube.stt.tubes_for(paths, Bev(...))` builds tubes from any paths `[N, H, 2]` on a body-frame grid. It
  needs only numpy and scipy.

## What has been tested

Everything below was run on one simulated environment: a Mars yard in Habitat, with kinematic motion (the robot
moves exactly as commanded), the obstacle's position taken from the simulator, one static obstacle.

| Result | Setting | Number |
|---|---|---|
| `tube` vs `obstacle`, passed a bar set in advance | 20 new scenarios x 2 repeats, `track` controller | 0.91 s less contact per drive (95% interval 0.54 to 1.28), same goal rate and drive time |
| `tube` vs plain `navdp` (reported, no bar) | same | 3.42 s less contact per drive |
| `stt` vs `track` controller (development only, no bar) | 3 scenarios | no contact in either; closest approach 1.78 m vs 1.55 m; `stt` takes 1.5 x as long |

`TubeDriver` itself is new. It was checked with CPU tests and against the real NavDP weights with a stand-in camera,
not in a full drive; the Habitat runs used `scripts/rollout_tube_navdp_mars.py`, which has the same logic in Habitat's
coordinates.

## Limits

- **Your map is trusted.** Tubes only avoid occupied cells; unknown cells count as free. NavDP's own depth-based
  avoidance still runs underneath.
- **Static obstacles only.** Tubes are indexed by waypoint, not time. A moving obstacle must be re-sent every plan and
  gets no prediction.
- **Free boxes are conservative.** Beside a diagonal wall or in clutter they are thin, the 0.3 m width can fail, and
  that path gets no tube (it is never given one that overlaps an occupied cell).
- **Going round, not waiting.** A tube can only swing sideways past an obstacle; it can't slow down for one.
- **The tube guarantee is conditional.** It needs a valid tube (`feasible`), an unclipped command and exact pose. Real
  slip, delay and pose error are untested. Treat it as a planner, not as a safety system.
- **Close range.** When the robot's start box is inside the grown obstacles, no tube exists; the path is steered
  clear but not certified.
- **The grid tubes are new**: written for this repo, not yet run in the sim or on the rover.
- **Parameters** other than NavDP's own scale were set by reasoning, not tuned on your setup.

## Files

| File | Contents |
|---|---|
| `navdp/tube/driver.py` | `TubeDriver`: the interface for your own environment |
| `navdp/tube/stt.py` | tubes on the grid: push, per-waypoint clusters, free boxes, linear programs, Theorem 3.1 |
| `navdp/tube/guided_sampler.py` | one-pass guided denoising, modes, choice |
| `navdp/tube/controller.py` | tube law and timing |
| `navdp/tube/agent.py`, `client.py` | NavDP with the loop swapped; HTTP client |
| `scripts/tube_navdp_server.py` | the server |
| `scripts/tube_navdp_example.py` | smallest working loop |
| `scripts/rollout_tube_navdp_mars.py` | the Mars experiments |
| `scripts/test_tube_navdp.py` | CPU tests: `python scripts/test_tube_navdp.py` |
