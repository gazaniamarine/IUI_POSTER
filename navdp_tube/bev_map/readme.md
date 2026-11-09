# BEV occupancy map from a monocular camera (Depth Anything)

Builds a top-down occupancy map (free / obstacle / unknown) for a ground robot from one RGB camera,
using Depth Anything depth corrected every frame against the known floor geometry.

## Folder layout
| path | shared? | contents |
|---|---|---|
| `bev_pipeline.py`, `projection.py`, `occupancy_grid.py`, `depth_estimator.py` | yes | the mapping pipeline (robot independent) |
| `run_bev_real.py`, `real_source.py`, `telemetry.py`, `gstream.py` | yes | real robot: camera + pose in, map out |
| `rover_kinematics.py` | yes, **edit per robot** | camera mount geometry -> camera pose |
| `calibrate_camera.py` | yes | checkerboard calibration (intrinsics, camera height/tilt) |
| `bev_render.py`, `web_view.py` | yes | drawing, browser dashboard |
| `rover_bridge/` | yes | sends odom/IMU from a ROS 2 robot to the server |
| `run_bev_tube.py`, `tube_nav.py`, `nav_obstacles.py` | no (needs `../tube_navdp`) | tube-guided NavDP navigation on the BEV map |
| `config.yaml` | yes, **edit per robot** | all settings (`real:` section for the robot) |
| `calib/` | no (per robot) | calibration results, created by `calibrate_camera.py` |
| `runs/` | no | real recordings and saved maps |
| `sim/` | no | MuJoCo simulation of this rover (see `sim/readme.md`) |

**To share:** `./make_share.sh` writes `bev_map_share.zip` with only the shared files.

**Adapting to another robot:** edit the geometry constants in `rover_kinematics.py` (or replace
`camera_in_base()` with a fixed transform for a fixed camera), the robot sizes in `config.yaml`
(`render.robot_radius`, `classify.obstacle_max_z` = robot height, `classify.obstacle_min_z`, `real.*`),
the topic names of `rover_bridge/rover_state_bridge.py`, then calibrate. Assumes a level floor and an
indoor scene (`depth.model_id`; use the Metric-Outdoor model outside).

## How it works
1. Frame source – `real_source.py` (camera + telemetry, real robot) or `sim/sim_source.py` (MuJoCo).
   Either one gives: RGB, intrinsics `K`, camera position/rotation in a z-up world with the floor at z = 0.
2. `depth_estimator.py` – Depth-Anything-V2-Metric-Indoor-Large (fp16).
3. `projection.py` – **floor fit**: the floor's true depth at every pixel is known from camera
   height + pose, so each frame fits `depth = a*DA + b` on floor pixels (RANSAC). In the sim DA is
   ~3x too far; after the fit the median depth error is ~3.5%. Also removes flying pixels at
   depth edges and smear behind walls (keeps only points near the first obstacle per bearing).
4. `occupancy_grid.py` – points with z near 0 → free, z in [4 cm, 60 cm] → obstacle (needs enough
   points and some height); log-odds fusion over frames; speck removal. Obstacles only within
   2 m (`grid.obstacle_max_range`), free space up to 3.5 m.
5. `bev_render.py` – global map, robot-centred view, robot-radius inflation (orange).
6. `web_view.py` – browser dashboard (MJPEG stream + keys) for headless/SSH use.

All tunables are in `config.yaml`.

## Floor depth and camera tilt
**How floor depth is computed:** for every pixel, the camera ray (from `K` and the camera rotation)
is intersected with the floor plane z = 0: `floor_depth = camera_height / -ray_z`. This is redone
every frame from that frame's camera pose, so when the robot moves or pitches, the floor model moves
with it. In the sim the pose is exact (MuJoCo), so the floor model is exact.

**Why tilt matters:** the camera is only 21 cm high, so a pitch error of e degrees moves the floor by
`distance * tan(e)`: 1 deg = 3.5 cm at 2 m, which is already the obstacle threshold.
Measured on the recorded sim drive (Depth Anything, precision of obstacle cells):

| pitch error given to the pipeline | precision | recall |
|---|---|---|
| none | 0.72 | 0.83 |
| random 1 deg per frame (bouncing) | 0.58 | 0.79 |
| random 2 deg per frame | 0.49 | 0.72 |

**What the pipeline does about it** (`pose_refine` in config.yaml):
- Each frame it fits the depth correction (scale `a`, offset `b`) **and the camera roll** to the
  visible floor (`JointFloorFit` in `projection.py`).
- It does **not** fit pitch: tested in sim, pitch is not observable from Depth Anything because its
  depth error looks the same as a pitch change (1.4 deg median error, up to 10 deg; roll: 0.16 deg).
  With exact depth (`--depth sim`) a plain plane fit recovers pitch and roll exactly.
- `classify.pitch_uncertainty_deg` widens the floor/obstacle height thresholds with distance, so
  a leftover pitch error does not turn far floor into obstacles (costs some far recall).

**For the real rover pitch must come from outside the image:**
1. Calibrate the mount once (height + pitch, e.g. checkerboard flat on the floor).
2. Use the rover's IMU (the URDF has `base_imu_link`) for per-frame pitch/roll while driving.
3. Set `pitch_uncertainty_deg` to ~0.5 with IMU, ~1.5 without.
4. Optionally skip frames while the robot accelerates or turns hard (most bounce happens then).
In this repo the sim can inject tilt errors to test settings: `sim/evaluate_sim.py --pitch-noise 1 --roll-noise 1`.

## Real rover

### Files
| file | where it runs | what it does |
|---|---|---|
| `run_bev_real.py` | GPU server | live BEV map (same pipeline as the sim), optional keyboard driving, record / replay |
| `real_source.py` | GPU server | camera stream (`v2/gstream.py`) -> undistorted, resized frame; record + replay |
| `telemetry.py` | GPU server | receives odom/IMU/gimbal from the rover, looks up the pose at a frame's time |
| `rover_kinematics.py` | GPU server | odom + IMU + pan/tilt -> camera pose (rover geometry from the URDF/MuJoCo model) |
| `calibrate_camera.py` | GPU server | checkerboard: intrinsics + lens distortion, then camera height/tilt on the floor |
| `rover_bridge/rover_state_bridge.py` | rover (ROS 2 container) | sends `/odom` + `/imu/data` (+ gimbal) to the server, ZMQ port 5557 |

Ports: camera RTP/JPEG UDP 5000 (rover -> server, existing), cmd_vel ZMQ 5556 (server -> rover,
existing; only one script may publish it), telemetry ZMQ 5557 (rover -> server, new).

### Start
```bash
# rover, terminal 1 (existing)
./start_camera_stream.sh <SERVER_IP>
# rover, terminal 2: copy rover_bridge/rover_state_bridge.py into the container once, then
docker exec -it ugv_rpi_ros_humble bash
ros2 topic list                                   # check the odom / imu topic names
python3 rover_state_bridge.py --server <SERVER_IP> # --odom-topic /odom --imu-topic /imu/data
# rover, terminal 3 (only for --drive, existing): python3 navdp_cmdvel_bridge.py

# server
conda activate yolo && cd <repo>/bev_map
python run_bev_real.py --drive --record runs/real_001     # browser at http://localhost:8090 over SSH
```
The bridge prints message counts every 5 s; 0 odom or 0 imu means a wrong topic name.
Driving keys are hold-to-move; the rover stops 0.6 s after the last key (dead-man), also if the
browser tab loses focus or the connection drops.

### Calibrate once (and after touching the camera mount)
Print a checkerboard (default 9x6 inner corners, 25 mm squares; pass `--board`/`--square` otherwise).
```bash
python calibrate_camera.py capture --out calib/images   # wave the board around, 25 views saved automatically
python calibrate_camera.py intrinsics                    # K + distortion -> calib/real_camera.yaml (want < 0.5 px)
# board FLAT on the floor 0.4-0.9 m in front, rover still, state bridge running:
python calibrate_camera.py mount                         # camera height + tilt, IMU level offset
python calibrate_camera.py check                         # tilt the gimbal: measured and predicted should agree
```
`check` is also how to confirm `real.tilt_sign` / `real.pan_sign` (gimbal angle convention): if the
predicted pitch moves the wrong way when the gimbal tilts, flip the sign.

### How the height errors are corrected on the real rover
| error source | correction | where |
|---|---|---|
| Depth Anything scale/offset (sim: ~3x too far, varies per image) | per-frame fit `depth = a*DA + b` against the known floor | `JointFloorFit` |
| wrong camera height (1 cm ≈ 5 % scale) | measured with the board; then recomputed every frame from the gimbal tilt (camera is 5 cm in front of the tilt axis: 1.4 cm lower at 17 deg down) | `calibrate_camera.py mount`, `rover_kinematics.py` |
| mount pitch/roll error | measured with the board, stored as a small mount rotation | `calibrate_camera.py mount` |
| rover pitch/roll while driving (bumps, braking) | IMU roll/pitch every frame (level-floor offset removed); roll also refined from the floor | `rover_state_bridge.py`, `JointFloorFit` |
| leftover pitch error (not observable from the image) | thresholds widen by `range*tan(0.5 deg)` | `real.pitch_uncertainty_deg` |
| one bad depth frame (glare, wall fills the view) | frame skipped if its scale jumps >10 % from the recent median, or the floor fit fails | `depth.scale_gate_rel`, `depth.skip_unfitted` |
| camera frame older than its arrival | pose looked up at `arrival - camera_latency` | `real.camera_latency` |
| fast turns / bouncing (pose + timing errors largest) | frame not added to the map | `real.max_yaw_rate`, `real.max_tilt_rate` |
| lens distortion | frames undistorted before anything else | `real_source.Undistorter` |

Not done on purpose: a RANSAC plane fit on the Depth Anything cloud to "level" the points, and the
scale `s = h_known / h_est` from that plane. Tested on the sim drive: the floor stays flat but obstacle
heights get 6x noisier (3.7 cm frame-to-frame), because DA bends the floor and the fitted plane is
tilted 1-2 deg.

Each run writes `runs/real_<time>/frames_log.csv` (per frame: mapped or why skipped, `a`, `b`,
roll fix, floor inliers, IMU roll/pitch, camera height/pitch, gimbal, yaw rate, pose). What to look at:
- `skip_reason` mostly empty; many `fit:scale_jump` -> the floor is hard to see (raise `scale_gate_rel`).
- `roll_fix_deg` averaging clearly away from 0 -> redo the mount calibration.
- walls smeared or doubled after turning -> tune `real.camera_latency` on a recording:
  `python run_bev_real.py --replay runs/real_001 --set real.camera_latency=0.15`.

### Checked in sim (no rover available yet)
- `rover_kinematics.py` matches MuJoCo's camera pose exactly for any pan/tilt/roll/pitch/yaw.
- Mount calibration on a synthetic board recovers the camera tilt to 0.02 deg and height to 0.1 mm.
- The sim drive converted to a "real" recording (images + odometry-style telemetry) and replayed
  through `run_bev_real.py`: precision 0.65, recall 0.80 (evaluate_sim without the sim-only
  background mask: 0.69 / 0.83; the gap is the 0.14 deg tilt the sim rover really has but the
  "IMU" reported as 0). A camera delay of 0.15 s left uncorrected costs 0.02 precision and 0.035
  recall; setting `camera_latency` to it recovers both fully.
- Not testable here: the ROS bridge (needs the rover's ROS 2), real topic names and gimbal conventions.

## Tube-guided navigation (NavDP + BEV map)
`run_bev_tube.py` drives the rover to a goal with tube-guided NavDP (`../tube_navdp`, copied from navdp_sam):
NavDP proposes 16 paths from the camera image + corrected depth, the tube planner pulls them around the obstacles
of this map and certifies the ones that stay in a valid tube, and the chosen path becomes cmd_vel.

```bash
# server, terminal 1 (conda env navdp)
cd <repo>/tube_navdp && python scripts/tube_navdp_server.py --navdp-root ../NavDP \
    --checkpoint ../NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt --port 8890
# rover: start_camera_stream.sh, rover_state_bridge.py, navdp_cmdvel_bridge.py (as for --drive above)
# server, terminal 2 (conda env yolo)
cd <repo>/bev_map && python run_bev_tube.py          # browser: click a goal
python run_bev_tube.py --dry-run                                      # plan + draw only, no cmd_vel
```
Goal: click the object in the camera view (3D point from the corrected depth), click the global map, use
`--goal x,y` (odom), or send JSON to `nav.goal_bind` (ZMQ, port 5558): `{"x": .., "y": ..}` or
`{"forward": .., "left": ..}` relative to the rover. X / SPACE cancel, arrow keys take over by hand.
Each run writes `runs/tube_<time>/nav_log.csv`.

| step | file | what |
|---|---|---|
| map | `nav_obstacles.py`, `tube_nav.nav_map` | occupied cells within `nav.obstacle_range`; cells within `nav.goal_clear_radius` of the goal **erased**; specks under `nav.min_blob_cells` dropped. The cells themselves go to the planner (no circles) |
| footprint | `tube_navdp/navdp/tube/driver.py` | the map is sent **without** the robot; the driver grows its occupied cells by `nav.robot_radius` (0.20 m: rover ~0.25 x 0.20 m, enclosing circle 0.16 m, + margin) and cuts a body-frame grid. The map is not inflated as well |
| plan | `tube_nav.py` | in a background thread at `nav.plan_hz`, with the robot pose at the image time |
| choice | `tube_nav.py` | certified path if any; else the best-valued path clear of the map's cells grown by the robot; else turn in place toward the goal |
| control | `tube_nav.py` | `nav.controller: track` (waypoint tracker) at `nav.control_hz`; stop at `nav.stop_distance` from the goal, when the plan is older than `nav.plan_timeout`, or telemetry is stale; no forward motion when a map obstacle is within `robot_radius + safety_margin` ahead |

Why the goal is erased: with the target object in the map, the paths' last waypoints sit in occupied cells, so no
tube reaches the goal and the rover is steered away from it. Why `stop_distance` is 0.55 m: NavDP turns any path shorter
than 0.5 m into "stop", so the rover would never get closer. A clicked camera goal lies on the object's front
surface, so the rover stops ~0.55 m from that surface.

Test in the MuJoCo arena first: `sim/run_tube_sim.py` (see `sim/readme.md`).
