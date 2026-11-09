# nav/ — interactive rover control, driven by the actual NavDP

One-command bringup for the Mars habitat sim plus a live control GUI, in the
spirit of `Nav_new/MARS/launch_mars.sh` -- but built entirely in-house under
`mars-habitatsim/nav/` (no imports from `Nav_new` or its `scripts`/
`nav_pipeline`), and driving with the **actual, published NavDP model**
(`sam_vla.policy.navdp_upstream_policy.NavdpUpstreamPolicy`, Cai et al.,
arXiv 2505.08712) rather than this repo's own in-house S2DiT+NavDP model
(`sam_vla.policy.navdp_policy.NavdpPolicy`) -- see `next.md`'s "Integration
project" section for how that backend itself was built and validated.

## Run it

```bash
./nav/launch_nav.sh
# or directly:
conda activate habitat
python -m nav.gui
```

Needs, same as `sam_vla.run_navdp_rollout --policy-backend navdp-upstream`:
- the `habitat` conda env (has habitat-sim, tkinter, and everything else
  `nav/` imports),
- a vendored `InternRobotics/NavDP` checkout (`$NAVDP_UPSTREAM_ROOT` or
  `--navdp-upstream-root`), and a `navdp` conda env to run it in (spawned
  automatically as a subprocess -- you never `conda activate` it yourself),
- a checkpoint (`--navdp-upstream-ckpt`, defaults to
  `navdp/navdp-cross-modal.ckpt` if present).

Useful flags: `--rock-field rock_envs/run1/rock_field.json`,
`--start-x/--start-z/--start-yaw`, `--no-cbf`. Run `python -m nav.gui --help`
for the full list.

## Architecture

Unlike `launch_mars.sh`'s two-process Zenoh bridge (a separate `habitat_sim_node.py`
sim process + a `mars_gui.py` control process, needed there because the GUI ran
in a different conda env from habitat-sim itself), everything here runs
**in one process**: `MarsHabitatEnv` (`sam_vla.env.habitat_env`) is driven
directly, in-process, from a dedicated background thread
(`rover_controller.RoverController`). The only subprocesses are the ones
`NavdpUpstreamPolicy`/`QwenServerManager` already spawn automatically (the
real NavDP HTTP server in the `navdp` env, and a Qwen VLM server used only
for one-shot goal resolution) -- the same subprocesses
`sam_vla.run_navdp_rollout` uses.

- **`goal_math.py`** -- pure SE(2) helpers (world point -> body-frame
  `[forward, left]`, random-ahead-point sampling). No sim/torch dependency.
- **`rover_controller.py`** -- `RoverController`: owns the env, the NavDP-upstream
  policy, `BeliefGoalTracker`, and (optionally) `CbfObstacleAvoidance`, and
  runs the actual step loop on its own thread. Exposes a small thread-safe
  command API (`set_manual`, `random_goal`, `go_home`, `request_resolve`,
  `request_reset`, `stop_driving`, `snapshot`) for a GUI (or a test) to drive.
- **`gui.py`** -- a Tkinter control panel (camera view with live goal/obstacle
  mask overlay, a body-frame trajectory/goal plot, manual-drive buttons +
  arrow keys, and the mode buttons below) polling `RoverController.snapshot()`
  at its own refresh rate.

## Driving modes

- **Resolve Goal (auto)** -- one-shot `first_frame_resolver.resolve_verbose()`
  on the current frame (SAM2 detections + Qwen VLM salience pick -- there is
  no per-preset text targeting; `qwen_client.select_goal` has no such
  parameter today, see `next.md`). The chosen goal (and every other detected
  object, as an obstacle) gets a small terrain-following mask mesh registered
  in the scene, tracked every tick via `BeliefGoalTracker`, same as
  `sam_vla.run_navdp_rollout`'s single-goal path. If the goal mask is never
  sighted (bad depth, thin sliver, etc.), the rover holds position instead of
  driving on `NavdpUpstreamPolicy`'s hidden default goal point -- the same
  fix `next.md`'s Integration-project Phase 5 made.
- **Random Goal** -- a random world point 4-8 m ahead, within +/-60 degrees
  of the current heading. No detector involved: the body-frame goal is
  recomputed from ground truth every tick via `goal_math.body_frame_goal`
  and fed straight into the belief tracker's `observe_body_point`.
- **Go Home** -- point-goal back to the spawn pose.
- **Reset Rover** -- teleports back to spawn, neutralises any registered
  goal/obstacle masks, and gives the belief tracker/CBF a clean slate.
- **Manual drive** -- arrow keys or the on-screen buttons; bypasses the
  policy and CBF entirely, same convention every manual-drive tool in this
  repo uses.
- **STOP** -- zero velocity, back to idle.

CBF cone-mode obstacle avoidance (`sam_vla.safety.cbf_avoidance`) wraps every
mode except manual drive, on by default (`--no-cbf` to disable).

## Non-goals

- No `--multi-goal`/base-station goal sequencing -- `NavdpUpstreamPolicy`
  only supports the single-goal point-goal path today (see `next.md`); this
  GUI is scoped the same way.
- No offline logging/video export -- this is a live-only tool, like
  `launch_mars.sh`'s own GUI. Use `sam_vla.run_navdp_rollout` for recorded
  rollouts.
- Nothing under `navdp/`, `sam_vla/`, or `Nav_new/` is modified by anything
  in this directory.
