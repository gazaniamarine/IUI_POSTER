# nav/gui.py — REACH console UI spec

Supervisory rover-control-console layout for `nav/gui.py` (`NavGuiApp`), built on
`nav/rover_controller.py`'s `RoverController`/`DisplayState`. This is the reference
`nav/gui.py` itself points back to ("exp.md's ASCII layout and per-region field lists") —
read this before touching root-grid structure or adding/removing a sidebar stat cell.

The interaction design (contextual panels, glove-safe sizing, three-stage feedback, etc.)
follows the REACH principles (Ma et al., "Human-Robot Interaction through REACH",
DLR/NASA), numbered P1–P10 in `nav/gui.py`'s module docstring. Each control below is
tagged `[Pn]` where it embodies one of those principles.

## Root layout

Four persistent regions stacked top to bottom, plus a working area split left/right.
Rows 0/2/3 span the full window width; row 1 splits 3:1 (camera : sidebar, sidebar
floor `SIDEBAR_MIN_W = 360px`).

```
┌──────────────────────────────────────────────────────────────────────┐
│ Row 0  TOP STATUS BAR                                     (full width)│
├─────────────────────────────────────────┬────────────────────────────┤
│ Row 1                                    │ Sec 3  MISSION             │
│ Sec 2  CAMERA (point & select)           │ Sec 4  ROVER STATUS        │
│                                           │        TASKWORK            │
│                                           │  [contextual] Review /     │
│                                           │   Confirm-point / Uncert.  │
│         weight 3                         │  footer: click status      │
│                                           │        weight 1, min 360px │
├─────────────────────────────────────────┴────────────────────────────┤
│ Row 2  Sec 5  TRAJECTORY / BELIEF                          (full width)│
├──────────────────────────────────────────────────────────────────────┤
│ Row 3  ■ EMERGENCY STOP                        │  Reset Rover          │
└──────────────────────────────────────────────────────────────────────┘
```

The sidebar (`ctk.CTkScrollableFrame`) scrolls independently so a contextual panel
appearing near the bottom never gets clipped on a short window; the emergency bar and
top status bar sit _outside_ it on their own root rows so they're always visible
regardless of scroll position `[P8]`.

---

## Sec 1 — Top Status bar

Full-width strip above everything else, the one line meant to be readable without
inspecting any other panel `[P9/P10]`.

- **State label** (large, bold, colored) — one word from a fixed vocabulary: `IDLE`,
  `PARSING MISSION`, `SEARCHING`, `NAVIGATING`, `AWAITING HUMAN`, `RECOVERING`,
  `GOAL REACHED`, `MISSION COMPLETE`, `MANUAL CONTROL`. Color is keyed per-state
  (`TOP_STATUS_COLORS`) so color alone tells you the rover's situation at a glance.
  Derived in `_derive_top_status` from the controller snapshot: uncertainty-halt beats
  manual beats mission-complete beats goal-reached beats searching/navigating, falling
  back to `IDLE`.
- **Detail line** (small, grey) — the _why_: goal name ("Navigating to ANTENNA"), a halt
  reason ("Goal localization uncertainty exceeded threshold"), or the controller's raw
  `status_text`. Whenever a control was just pressed, this line shows that action's
  acknowledgment text instead, for exactly one refresh tick — so a press never feels
  unregistered even before the controller thread's own state has caught up `[P9]`.
- **"controller thread died — see console"** — hidden by default; only gridded in if
  `RoverController.is_alive()` goes false, in red. Not a normal state color because it's
  not something the controller reported, it's the GUI noticing the controller stopped
  reporting at all.

## Sec 2 — Camera (point & select)

The left column's hero element, filling all available space in row 1. The live
Habitat-sim RGB frame (`d.vis_rgb`) is letterboxed into a `tk.Canvas` every refresh,
tracking window resizes.

- **Live view canvas** — click anywhere on the frame to designate that pixel as a goal
  `[P7]`, the direct desktop analog of REACH's "point and select an object in the
  environment". A click drops a cyan crosshair+ring marker at the clicked point (not yet
  committed) and opens the **Confirm Point Goal** contextual panel in the sidebar. A
  confirmed goal instead draws a gold marker, redrawn by the controller itself every
  frame via `draw_point_marker`, distinguishing "pending" from "committed" by color.
- **"POINT & SELECT — click any point in the live view to send the rover there"** — a
  static caption below the canvas, permanently telling the operator this affordance
  exists (no separate button, so the caption is the only cue).

## Sec 3 — Mission panel

Sidebar, first section. "What is the rover trying to accomplish, how far away is it,
how confident is the rover about it?"

- **"MISSION"** — section title label.
- **Goal list** — the mission's ordered sub-goals, one per line, prefixed `✓` (done),
  `→` (current), or `○` (upcoming) — e.g. `✓ 1. FLAG_A`, `→ 2. HOME`. Built from
  `d.mission_goals`/`d.mission_goal_idx` (`nav.mission.Mission`). If no VLM-parsed
  mission is running, degrades to a single synthesized `→ 1. <goal>` line around
  whatever ad-hoc goal (Segment/Ground/Random/point-click) is active, or
  `(no active mission)` if nothing is. This is REACH's "chunked, linear taskwork" list
  `[P3]` — the operator can self-check progress without asking ground control what's
  next.
- **"CURRENT GOAL"** label + value — the active sub-goal's display name in large bold
  text (landmark name for GO_TO/FIND, `HOME` for RETURN, `TURN LEFT/RIGHT` for TURN,
  `ADVANCE` for ADVANCE legs), or `--` if nothing is active.
- **Three stat cells**, side by side (shared `_stat_cell` widget — small grey title over
  a bold value, used again in Sec 4):
  - **DISTANCE** — straight-line meters to the goal (`d.distance`), or `--`.
  - **GOAL BELIEF** — `HIGH CONFIDENCE` (green) / `LOW CONFIDENCE` (amber, then red as
    it worsens) derived from how close `d.uncertainty_value` is to
    `d.uncertainty_threshold`; `--` when uncertainty tracking is off or no belief exists
    yet. A human-readable read of the same covariance the Rover Status panel shows
    numerically and the Trajectory/Belief plot shows spatially as a ring.
  - **STATUS** — the sub-goal's own execution state: `SEARCHING` → `DETECTED` →
    `NAVIGATING` → `REACHED`, or `UNCERTAIN`/`REACQUIRING` during an uncertainty halt.
    Folded from existing controller signals in `_derive_goal_status` (no dedicated
    controller field) — this _is_ REACH's three-stage command feedback (received →
    executing → complete), just expressed in exp.md's own goal-state vocabulary instead
    of a generic pill `[P2]`.

## Sec 4 — Rover Status panel

Sidebar, second section. "How is the rover operating right now, independent of any
particular goal?"

- **"ROVER STATUS"** — section title label.
- **Mode label** — one of `IDLE` / `AUTONOMOUS` / `HUMAN GUIDANCE` / `MANUAL`, colored
  per `ROVER_MODE_COLORS`. `HUMAN GUIDANCE` overrides everything else whenever an
  uncertainty halt is actively waiting on the operator, regardless of the controller's
  own finer-grained internal mode string.
- **Two stat cells**:
  - **CURRENT UNCERTAINTY** — `value/threshold` for the active goal's belief covariance
    (e.g. `0.31/0.80`), colored green/amber/red by how close to threshold; `off` if
    uncertainty-halt is disabled.
  - **HOME UNCERTAINTY** — same `value/threshold` readout, but for the persistent
    home-base belief that's tracked for the whole mission (keeps growing even while
    driving toward an unrelated GO_TO/FIND leg, not just during a RETURN leg).
- **Telemetry block** (monospace) — three lines updated every frame:
  - `pos (x, z)  heading=NNdeg` — world position and heading in degrees.
  - `v=N.NN m/s  yaw_rate=+N.NN rad/s` — current commanded linear/angular velocity.
  - `nav state: MODE_NAME [(GOAL REACHED)]` — the controller's raw mode string
    (uppercased), with a suffix when the active goal has been reached.
    Deliberately only these five operational values — CBF/model-internal debug detail is
    left out of this panel and lives in the Trajectory/Belief HUD instead `[P6]`.

## Taskwork panel

Sidebar, third section — free-text mission dispatch. This is REACH's "chunked, linear
taskwork" input side (Sec 3's goal list is the output/progress side) `[P3]`.

- **Command entry** (text field) — placeholder `"go to a flag then return to home"`.
  Free text typed here is sent to `RoverController.submit_nav_command`, which splits it
  via the Qwen VLM into an ordered list of directions/goals (`nav.mission.parse_parts`)
  and drives through them one at a time. Pressing **Enter** while focused submits, same
  as clicking Send.
- **Send button** — submits the entry's text, echoes an immediate "taskwork queued: …"
  acknowledgment, and clears the field.
- **Mission status label** — the active `Mission.status()` line (e.g. progress through
  the parsed sub-goals), blank when no mission is running.

Typing in this entry temporarily disables the numpad/arrow-key shortcuts below
(`_guarded` checks focus) so a digit or arrow key typed as part of a command doesn't
also fire a manual-drive or uncertainty-heading shortcut.

### Manual drive (no dedicated section)

Not a rendered panel — arrow keys (`↑↓←→`) hold-to-drive directly via
`controller.set_manual`, for as long as they're held, blended additively (e.g. holding
both `↑` and `←` drives forward-left). Releasing all held keys stops driving. Exists as
a keyboard-only affordance so the sidebar doesn't need a dedicated d-pad section `[P6]`.

## Contextual panel A — Review Resolved Goal

Hidden by default; only gridded in while the controller is in
`MODE_REVIEW_SEGMENTATION` (a Segment/Ground-Target resolution just produced a
candidate goal awaiting operator sign-off). Amber border reads as "action needed." This
_is_ a REACH contextual menu — these three actions exist only because a resolved
selection exists to act on `[P1]`.

- **Description label** — the controller's `status_text` describing what was resolved.
- **Confirm** (green) — accept the resolved selection as the goal.
- **Rerun** (amber) — reject it and re-resolve with the same intent (new segmentation/
  grounding attempt).
- **Pick Manually** (neutral) — abandon auto-resolution and fall back to point-and-select
  in the camera view `[P1+P7]`.

## Contextual panel B — Confirm Point Goal

Hidden by default; gridded in the instant a camera click is pending confirmation. Cyan
border, matching the pending-click marker's color.

- **"Set the goal at the point you clicked?"** — description label.
- **Confirm Point Goal** (green) — commits the pending click as the actual goal
  (`controller.request_pixel_goal`) `[P7]`.
- **Cancel** (grey) — discards the pending click, closing the panel without setting a
  goal — REACH's contextual "back button" `[P1]`.

## Contextual panel C — Uncertainty Halt

Hidden by default; gridded in whenever `d.uncertainty_halted` is true — the rover has
stopped itself because its belief about the goal's location grew past
`uncertainty_threshold` without a re-sighting, and is asking the operator for a
heading. Sky-blue border; turns amber with title `REQUESTING VLM SWEEP...` while a
retry request is in flight (all buttons disabled during that window). This is the
clearest instance of shared-autonomy in the console: the rover enforces its own safety
stop `[P8]`, then hands control back to the human with an explicit request `[P10]`.

- **Title** — `UNCERTAINTY HALT -- choose a heading` (or the in-flight message above).
- **Description label** — `d.uncertainty_line`, the VLM sweep's own description of what
  it currently sees in each direction.
- **8-button compass numpad** — one button per heading (`8`=front/0°, `9`=45°, `6`=90°,
  `3`=135°, `2`=180°/behind, `1`=-135°, `4`=-90°, `7`=-45°), arranged in the same 3×3
  keypad-minus-center grid as a phone numpad, each showing its key digit and signed
  angle (e.g. `9\n+45°`). Clicking — or pressing the matching number key — submits that
  heading (`controller.submit_uncertainty_heading`); the rover drives
  `--uncertainty-search-dist` meters that way before falling back to dead-reckoning
  `[P1+P5]`. Large buttons with generous spacing so a heading can't be mis-selected
  under stress `[P5]`.
- **Retry (R)** button (amber) — re-requests a fresh VLM sweep instead of committing to
  a heading, for when none of the eight look right `[P1]`. Same action as pressing `r`/
  `R` on the keyboard.

## Footer — click status

Bottom of the sidebar (absorbs any leftover vertical space so the panels above stay
top-anchored rather than floating mid-window). One grey line echoing the outcome of the
last point-goal click attempt, e.g. "point goal set at …" or "click ignored: no valid
depth" `[P9]`.

## Sec 5 — Trajectory / Belief bar

Full-width strip below the camera+sidebar body, fixed height (`PLOT_BOTTOM_H = 200px`),
outside the scrollable sidebar so it's never affected by scroll position. "Where has the
rover been, and where does it currently believe the goal is?" — a persistent
body-frame map `[P4]`.

- **"TRAJECTORY / BELIEF"** — section title label.
- **Plot canvas** — body-frame (rover-centered, rover always at bottom-center facing
  up) view spanning `±PLOT_RANGE` (6 m) drawn fresh every frame:
  - **Rover marker** — small light dot, fixed at bottom-center.
  - **Trajectory line** — the NavDP policy's chosen waypoints (`d.trajectory`), drawn as
    a connected coral line.
  - **Obstacle marker** — red ring at the nearest CBF-detected obstacle
    (`d.obstacle_point`), when one exists.
  - **Goal marker** (`*`, gold) — the current belief estimate of the goal's position
    (`d.belief_g`), body-frame. Surrounded by a **dispersion ring** whose radius grows
    with `uncertainty_value` (clamped at `uncertainty_threshold`) and is colored
    green/amber/red the same way the stat cells are — so confidence reads spatially
    here, not just as text in Sec 3's GOAL BELIEF cell.

## Pinned Emergency bar (row 3)

Its own full-width root row, outside the scrollable sidebar, always visible regardless
of scroll or mode — the paper's emergency-takeover affordance gets the largest,
reddest, most reachable real estate in the whole window `[P8]`.

- **■ EMERGENCY STOP** (red, large, `height=56`) — immediately halts driving
  (`controller.stop_driving`), clears any held manual-drive keys, and cancels a pending
  camera click. Does _not_ clear goals/missions or teleport the rover — it only stops
  motion.
- **Reset Rover** (grey, narrower, visually subordinate to STOP) — requests a full
  reset (`controller.request_reset`), teleporting the rover back to its spawn pose.
  Placed right next to STOP for reachability but styled down so it isn't mistaken for
  the actual emergency control `[P8]`.

## Global keyboard shortcuts (not rendered as buttons)

- **Escape** — hard clear (`clear_all`): stops driving, cancels a pending click, and
  also strips the resolved goal/obstacle masks, active mission, and any uncertainty
  halt — leaving the rover exactly where it is (unlike Reset Rover, which also
  teleports it back to spawn).
- **Arrow keys** — hold-to-drive manual control (see Taskwork panel above).
- **Digits 1–4, 6–9** — uncertainty-halt heading shortcuts, mirroring the numpad panel's
  buttons; a no-op whenever the controller isn't actually halted.
- **R / r** — retry the uncertainty VLM sweep, mirroring the Retry button.

All of the above are suppressed while the Taskwork command entry has keyboard focus, so
typing a mission description never doubles as a manual-drive or heading command.

## Secondary window — Top-Down View (optional)

A completely separate `CTkToplevel` window, only created when the controller was
started with `--top-down-viz`. Shows a fixed bird's-eye habitat-sim camera framing the
course from directly above, for human viewing/video recording only — it has no buttons,
click handlers, or wiring into navigation/CBF/belief logic, and never touches the main
console's layout.
