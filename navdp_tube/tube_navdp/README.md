# Tube-guided NavDP (copied)

Copied from `github.com/yashaswisinha05/navdp_sam` (private), commit `0b0257e`, then changed:
- only `navdp/official_client.py`, `navdp/tube/` and three scripts were copied;
- `scripts/test_tube_navdp.py`: the Mars-experiment tests were removed (their scripts are not here);
- the tubes no longer pick a side of each obstacle and take no circles: they are built on the BEV occupancy
  grid, from per-waypoint clusters of NavDP's paths and free boxes grown round them (`navdp/tube/stt.py`), with the
  STT paper's Theorem 3.1 checked exactly; the controller's time map is smooth (PCHIP).

How it works and its settings: [navdp/tube/README.md](navdp/tube/README.md).

## Start the server (uses this project's NavDP and checkpoint)
```bash
conda activate navdp
cd <repo>/tube_navdp
python scripts/tube_navdp_server.py --navdp-root ../NavDP \
    --checkpoint ../NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt --port 8890
```
Port 8890, so it can run next to the plain NavDP server (8888).

## Check it
```bash
python scripts/test_tube_navdp.py              # CPU tests, no server needed
python scripts/tube_navdp_example.py --seconds 2   # needs the server; stand-in camera and robot
```

## Used by the BEV map
The integration lives in `../bev_map` (`nav_obstacles.py`, `tube_nav.py`, `run_bev_tube.py`,
`sim/run_tube_sim.py`); settings in the `nav:` section of `../bev_map/config.yaml`. See the
"Tube-guided navigation" section of `../bev_map/readme.md`. In short:
- `robot_radius` 0.16 m: the rover is about 0.25 x 0.20 m with wheels, enclosing circle 0.16 m (no extra pad: the
  BEV map already draws obstacles a few cm too thick). The
  driver grows the map by it (plus the 0.2 m look-ahead with `stt`), so the BEV map is NOT inflated as well.
- Input: the BEV map itself, `driver.plan(rgb, depth, goal, pose, WorldGrid(occupied, res, (x_min, y_min)), now)`
  with `occupied` the cleaned mask from `nav_obstacles.obstacle_cells` (goal disk erased: a goal on an object would
  sit in occupied cells and get no tube; cells beyond `obstacle_range` dropped; specks removed). No circles.
- `controller: track`, not `stt`: `stt` grows obstacles by its 0.2 m look-ahead as well, so with the 0.3 m
  minimum tube width a gap needs ~1.1 m; in the sim arena (0.9 m gaps) it found almost no valid tubes.
- When no path is certified, `bev_map/tube_nav.py` drives the best-valued path that stays clear of the
  map's occupied cells grown by the robot, or turns in place, instead of NavDP's best-valued path regardless of
  obstacles.
- Speeds 0.10 m/s, 0.35 rad/s; stop 0.55 m from the goal (NavDP turns paths shorter than 0.5 m into "stop").
