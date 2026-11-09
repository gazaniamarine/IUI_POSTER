"""Real-rover driver: Qwen3.5-9B point-goal navigator + real NavDP, talking to
the ACTUAL rover's zenoh bridge over an explicit tcp connect endpoint (not
multicast -- the rover and this machine are on different subnets) instead of
an in-process Habitat-Sim node.

Promoted from a scratchpad prototype after a live validated run against the
real rover at 172.22.217.125 (2026-08-24): drove 90 ticks / 18s at
max_linear=0.15 toward a glass-door hallway, never collided, safety-stopped
cleanly at the step bound. See module-level constants below for why each
cap/format-handling choice is what it is.

Key design point: real hardware has no ground-truth world pose (no "mars/pose"
topic), so there's no persistent world frame to round-trip a point goal
through -- every query is fresh anyway (no cross-tick drift to track), so a
fixed, always-zero Pose(0,0,0,yaw=0) is passed into VlmPointNavigator (which
internally calls goal_geometry.pixel_to_world) and then immediately back out
through body_frame_goal with that SAME zero pose. With both calls anchored at
the identical zero pose, the round trip algebraically collapses to a direct
camera-frame -> body-frame conversion (world_x/world_z ARE x_cam/-z_cam at
zero pose, and body_frame_goal's own formula recovers exactly
(forward=z_cam, left=-x_cam) from that) -- so this reuses
VlmPointNavigator/body_frame_goal unmodified rather than reimplementing the
same pinhole math a third time. This also matches InternVLA-N1's own
"System-1 re-derives a trajectory from the current frame every tick using the
last (possibly stale) latent goal" design this whole system is modeled on.

Does NOT do Nihil/LAUNCH/_backend.sh-style SSH bring-up (ping + sudo
systemctl restart rover-camera/rover-agent/rover-zenoh on the Pi): this
robot's camera/depth/zenoh services were already confirmed live over a direct
zenoh connect with no bring-up step needed, and this repo doesn't have
verified SSH credentials for 172.22.217.125's Pi -- restarting services we
don't need to touch, with credentials we haven't validated against this
specific host, is not a step to take by default. If a future run finds the
bridge down, bring it up manually first rather than adding blind SSH+sudo
here.
"""

import argparse
import math
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/gpu/Desktop/pineapple/Nav_new")
sys.path.insert(0, "/home/gpu/Desktop/pineapple/mars-habitatsim")

import cv2  # noqa: E402
import requests  # noqa: E402
import zenoh  # noqa: E402
from nav_pipeline.zenoh_node import (  # noqa: E402
    CAMERA_COMPRESSED_KEYS,
    CAMERA_KEYS,
    DEPTH_KEYS,
    parse_compressed_image,
    parse_image,
    serialize_twist,
)

from nav.goal_math import body_frame_goal  # noqa: E402
from nav.vlm_point_navigator import VlmPointNavigator  # noqa: E402
from sam_vla.core.types import Observation, Pose  # noqa: E402
from sam_vla.policy.navdp_upstream_client import (  # noqa: E402
    pointgoal_step,
    select_action_from_trajectory,
)
from sam_vla.vlm import qwen_client  # noqa: E402
from sam_vla.vlm.navdp_upstream_server_manager import _reset_payload  # noqa: E402

DEFAULT_ENDPOINT = "tcp/10.86.180.125:7447"  # this robot's confirmed-reachable IP, 2026-08-25
# (was tcp/172.22.217.125:7447 on 2026-08-24 -- the rover moved to a different network/AP)
NAVDP_BASE_URL = "http://127.0.0.1:8766"
HFOV_DEG = 55.5  # measured, this D435i, 640x480 -- see Nihil/LAUNCH/_backend.sh
MAX_LINEAR = 0.15  # m/s, matches every real-rover launcher's cap
MAX_ANGULAR = 0.3  # rad/s -- tightened 2026-08-25 (was 0.5). An isolated actuation probe (raw
# lin=0.15, ang=0.0 for 5s, no navigation logic) confirmed the rover's wheels and heartbeat/
# watchdog handling both work fine -- it translates cleanly on a clean command. But every
# navigation run showed continuous saturated angular commands (raw +-0.5 rad/s) alternating
# almost every tick with barely any net displacement across 7 live runs: on this differential-
# drive rover, alternating hard turns while also commanding forward speed cancels into a wobble
# instead of translation, rather than any actuation/firmware issue. The smoothing that actually
# addresses this is MOTION_RAMP_TIME_S below (replaced a fixed-alpha EMA that lived here) plus
# the lookahead-averaging fix to WHERE on the trajectory the steering target comes from -- see
# LOOKAHEAD_FRAC/LOOKAHEAD_WINDOW below, ported from InternVLA-N1's own real-rover integration
# in this codebase.
TICK_HZ = 5.0
POINT_GOAL_REACHED_M = 0.6  # tighter than sim's 1.0 -- real space, real consequences
NAVDP_GOAL_MAX_DIST_M = 2.0  # cap the (forward, left) distance handed to pointgoal_step -- see
# its use in drive_loop. Live-confirmed 2026-08-27: the VLM's own estimated target distance can
# be 3-7m+ (or tens of meters for a genuinely distant landmark), but NavDP's diffusion policy only
# reliably predicts a few meters per call -- feeding it a goal near/beyond its real prediction
# horizon made the lookahead point (LOOKAHEAD_FRAC=0.75 along the trajectory) degenerate to a
# saturated (0, +-1) turn-in-place with zero forward component, regardless of the true direction,
# on EVERY test past the first (short-range) one. Clamping to a nearby carrot point and relying on
# the replan cadence (every replan_every ticks) to re-issue a fresh, still-nearby goal as the rover
# closes in is the standard way a short-range point-goal policy is meant to be driven -- it was
# never meant to solve the whole remaining distance in one call.
REALIGN_DIST_M = 1.2  # body_goal magnitude used by the last-known-bearing realignment
# fallback (see drive_loop) -- a direction carrier only, not a real distance estimate,
# comfortably under NAVDP_GOAL_MAX_DIST_M so it's never itself rescaled by that clamp.
LOCK_MAX_COAST_TICKS = 40  # ~8s at TICK_HZ=5 -- see drive_loop's Qwen-cutoff comment.
# Bounds how long NavDP is allowed to drive on a dead-reckoned goal alone before a
# fresh Qwen check-in is forced, since dead-reckoning error (no persistent world
# pose, purely relative deltas) grows unbounded without ever re-grounding against
# a real frame. ~8s is generously more than one Qwen call's own ~4.7s latency, so
# the rover is never worse off waiting for a fresh check-in than it would be
# calling Qwen on every replan tick anyway.
ZERO_POSE = Pose(x=0.0, y=0.0, z=0.0, yaw=0.0)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
DEPTH_INVALID_MM = 65535  # RealSense sentinel for "no return" -- not a real 65m reading
DEPTH_STOP_M = 0.5  # hard forward-collision guard, independent of Qwen's "advance"/NavDP's own
# output. Added 2026-08-25 after Qwen's advance fired on a self-reported "~0.1m from the glass
# doors" that had no basis in the real scene (physical check showed the rover hadn't materially
# moved) -- the model's distance claims in free-text `reasoning` aren't grounded in the actual
# depth data, so they can't be trusted to prevent a collision. This checks REAL measured depth
# every tick and force-zeros the command if anything is too close, no matter what the VLM/NavDP
# decided.
DEPTH_STOP_COL_FRAC = (0.35, 0.65)  # center band of the depth frame's columns
DEPTH_STOP_PERCENTILE = 5.0  # low percentile, not bare min, of the forward-band depth --
# matches Nihil/reference/internvla_dualvln_zenoh_node.py's own navdp_cone_percentile: a real
# rover run on this exact codebase found a bare min() over the forward cone got tripped by noisy
# outlier pixels; a robust low percentile keeps the same "something is close" sensitivity without
# one bad pixel deciding it.

# --- Architecture ported from Nihil/reference/internvla_dualvln_zenoh_node.py -----------------
# That file is InternVLA-N1's own real-rover integration in this codebase, independently tuned
# against live hardware over many iterations. Several of its documented, dated findings are
# exactly the symptom class hit here (continuous non-zero commands, ~no net displacement) --
# ported the concrete fixes rather than re-deriving them from scratch:
NAVDP_PREDICT_SIZE = 24  # this server's trajectory length (sam_vla/run_navdp_click_rollout.py's
# own comment: "predict_size=24") -- needed to scale N1's lookahead numbers, which were tuned
# against ITS checkpoint's 32-step trajectory, proportionally onto this one.
LOOKAHEAD_FRAC = 0.75  # N1's navdp_lookahead_idx=24 on a 32-step trajectory is 75% along --
# "Index 8/32 under-represented the model's actual overall direction... where diffusion-
# averaging noise is worst (smallest displacement -> atan2 most noise-sensitive)". Our own
# waypoint_index=3 on a 24-step trajectory (12.5% along) was in exactly that noise-sensitive
# regime -- this targets the same 75%-along point instead, scaled to this trajectory's length.
LOOKAHEAD_WINDOW = 2  # average +/- this many points around the lookahead index (N1 used +/-4 on
# a 32-step trajectory, i.e. ~12.5% of the horizon each side; scaled to 24 steps that's ~+/-3, but
# kept slightly tighter since our replan cadence is shorter -- tune alongside LOOKAHEAD_FRAC if
# steering still looks noisy).
MOTION_RAMP_TIME_S = 0.35  # seconds to ramp (lin, ang) from 0 to the commanded value, scaled by
# ACTUAL elapsed wall-clock time since the last tick rather than a fixed per-tick alpha --
# replaces the old fixed-alpha YAW_SMOOTH_ALPHA EMA. Matches N1's own motion_ramp_time_s: real
# per-tick latency varies a lot here too (a NavDP HTTP call is usually fast, but a Qwen replan
# tick can take much longer), and a fixed-alpha EMA implicitly assumes a constant tick period --
# under variable latency a slow tick either barely moves the smoothed value (if alpha was tuned
# for fast ticks) or overshoots it (if tuned for slow ones). Ramping by actual dt/ramp_time
# auto-adapts either way. 0.35s is N1's own already-tuned value for "make direction changes
# still land within a couple of real ticks, without snapping instantly" -- start here, retune
# only this constant if it needs adjusting (this project's own one-constant-at-a-time practice).
STALL_FRAME_DIFF_THRESH = 1.5  # mean abs grayscale delta (0-255, 32x24 downsample) below this
# while commanding real motion = "not actually moving" -- N1's own threshold and downsample size,
# verified live there against a real stall (screenshot-diff confirmed ~0.7/255 during a genuine
# wedge). Directly targets this project's own repeated symptom: continuous non-zero cmd_vel in
# the log with near-identical camera frames across an entire run.
STALL_TICKS_CONFIRM = 4  # consecutive stalled ticks (while commanding real motion) before
# triggering recovery -- avoids reacting to one noisy frame-diff sample.
STALL_RECOVER_TICKS = 3  # ticks to hold the reverse+turn recovery maneuver once triggered.
STALL_BACKUP_LIN = -0.12  # reverse speed during recovery (negative = backward).
STALL_BACKUP_ANG = 0.4  # turn speed during recovery, signed toward whichever way the rover was
# already trying to turn (continues the model's last intent rather than picking a new direction).
REPLAN_FRESH_FRAME_TIMEOUT_S = 3.0  # see the fresh-frame wait in drive_loop's replan branch --
# generous enough for even a slow (~1Hz) camera to produce at least one new frame, bounded so a
# genuinely dead camera can't hang the loop forever.


def parse_compressed_depth(cdr_data: bytes):
    """This rover's depth_raw isn't a raw sensor_msgs/Image like a Habitat
    sim node publishes (that's parse_image's job) -- it's PNG16-compressed
    (confirmed live: uint16 mm, 320x240 -- HALF the RGB stream's 640x480,
    which is fine: VlmPointNavigator/pixel_to_world index off depth's OWN
    shape using normalized (u,v), so a resolution mismatch with RGB is
    geometrically harmless).

    Hand-parsing this CDR message's header field-by-field (the way
    parse_compressed_image does for the sim's simpler, self-authored wire
    format) does NOT work here -- confirmed live: reading
    int32+uint32+string (stamp/frame_id) then a format string produced
    garbage offsets, because this robot's bridge is a real ROS2/DDS node
    whose actual Header/CompressedImage field layout doesn't line up with
    that hand-rolled reader. Locating the PNG magic bytes directly in the
    raw payload sidesteps needing to get every preceding field exactly
    right -- confirmed empirically to decode cleanly and repeatedly on live
    frames. Returns float32 meters, or None."""
    idx = cdr_data.find(PNG_MAGIC)
    if idx < 0:
        return None
    arr = np.frombuffer(cdr_data[idx:], dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2:
        return None
    depth_m = img.astype(np.float32) / 1000.0
    depth_m[img >= DEPTH_INVALID_MM] = 0.0  # sentinel -> invalid, not "65m away"
    return depth_m


def min_forward_depth(depth):
    """Real measured range (meters) in the center forward band of the depth
    frame, ignoring invalid (0) pixels -- a low percentile (DEPTH_STOP_PERCENTILE),
    not the bare minimum, so one noisy outlier pixel can't trip the guard.
    None if no valid depth."""
    if depth is None:
        return None
    h, w = depth.shape
    c0, c1 = int(w * DEPTH_STOP_COL_FRAC[0]), int(w * DEPTH_STOP_COL_FRAC[1])
    region = depth[:, c0:c1]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, DEPTH_STOP_PERCENTILE))


def _lookahead_point(traj_xy):
    """Average body-frame point around a far-along, low-noise index into the
    predicted trajectory -- see LOOKAHEAD_FRAC/LOOKAHEAD_WINDOW's comments.
    Returns (forward, left)."""
    n = traj_xy.shape[0]
    idx = int(round(LOOKAHEAD_FRAC * (n - 1)))
    lo = max(0, idx - LOOKAHEAD_WINDOW)
    hi = min(n, idx + LOOKAHEAD_WINDOW + 1)
    window = traj_xy[lo:hi]
    return float(window[:, 0].mean()), float(window[:, 1].mean())


def _dead_reckon_goal(goal_forward, goal_left, v, yaw_rate, dt):
    """Update a body-frame (forward, left) goal point for one tick's worth of
    the rover's OWN commanded motion, with no persistent world pose -- same
    relative-delta philosophy as last_bearing/yaw_since_bearing (see
    drive_loop), generalized from angle-only to full 2D position. Used to let
    NavDP keep driving toward a fixed real-world target across many ticks
    after Qwen is cut off (see LOCK_MAX_COAST_TICKS), by tracking where that
    target must now be relative to the CURRENT body frame given how far
    forward and how much the rover has turned since it was last actually
    seen. Assumes forward-only translation each tick (a differential-drive
    rover doesn't slip sideways) then a rotation by yaw_rate*dt -- standard
    body-frame dead reckoning, not a claim about absolute position."""
    # Translate: the goal is now v*dt closer in the OLD frame's forward axis.
    fx = goal_forward - v * dt
    fy = goal_left
    # Rotate into the NEW (post-turn) frame -- inverse of the frame's own
    # rotation, same sign convention as realign_yaw = last_bearing -
    # yaw_since_bearing above.
    dtheta = yaw_rate * dt
    cos_t, sin_t = math.cos(dtheta), math.sin(dtheta)
    new_forward = fx * cos_t + fy * sin_t
    new_left = -fx * sin_t + fy * cos_t
    return new_forward, new_left


def _ramp_toward(current, target, max_step):
    """Move `current` toward `target` by at most max_step -- the wall-clock-
    scaled replacement for a fixed-alpha EMA (see MOTION_RAMP_TIME_S)."""
    delta = target - current
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return current + delta


def _frame_stall_score(rgb, prev_gray):
    """Downsampled grayscale frame vs. the previous tick's, for the stall
    detector (see STALL_FRAME_DIFF_THRESH). Returns (score_or_None, new_gray)."""
    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 24)).astype(np.float32)
    if prev_gray is None:
        return None, small
    return float(np.abs(small - prev_gray).mean()), small


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rgb = None
        self.depth = None
        # arrival times + a short colour history, so a depth frame (1.8 Hz
        # live) can be paired with the colour frame of the same moment
        self.rgb_t = 0.0
        self.depth_t = 0.0
        from collections import deque

        self.rgb_hist = deque(maxlen=40)  # (t, rgb), ~2.5 s at 15 Hz


def connect(endpoint):
    """Open the zenoh session, subscribe camera+depth, return
    (session, pub_cmd, st). Shared by the CLI entry point and the GUI so
    there's exactly one place that does this wiring."""
    st = SharedState()

    def on_image(sample):
        img = parse_compressed_image(bytes(sample.payload.to_bytes()))
        if img is None:
            img = parse_image(bytes(sample.payload.to_bytes()))
        if img is not None and img.ndim == 3:
            t = time.monotonic()
            with st.lock:
                st.rgb = img
                st.rgb_t = t
                st.rgb_hist.append((t, img))

    def on_depth(sample):
        raw = bytes(sample.payload.to_bytes())
        d = parse_compressed_depth(raw)
        if d is None:
            d = parse_image(raw)  # fall back to raw sensor_msgs/Image, in case
        if d is not None and d.ndim == 2:
            t = time.monotonic()
            with st.lock:
                st.depth = d
                st.depth_t = t

    cfg = zenoh.Config()
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    session = zenoh.open(cfg)
    # Keep every Subscriber handle alive in `subs` -- an unreferenced handle
    # from declare_subscriber() gets garbage-collected almost immediately
    # over an explicit-endpoint (non-multicast) connection, silently tearing
    # the subscription down before any sample ever arrives. Confirmed live.
    subs = []
    for k in CAMERA_COMPRESSED_KEYS + CAMERA_KEYS:
        subs.append(session.declare_subscriber(k, on_image))
    for k in DEPTH_KEYS:
        subs.append(session.declare_subscriber(k, on_depth))
    pub_cmd = session.declare_publisher("cmd_vel")
    st.subs = subs  # SharedState has a real __dict__ -- keeps the Subscriber handles alive
    print(f"[real-rover] zenoh connected to {endpoint}", flush=True)
    return session, pub_cmd, st


def wait_for_frames(st, timeout_s=30.0):
    # Was 10.0 -- too short. journalctl -u rover-camera on the Pi shows its
    # RealSense driver crash-loops and self-recovers every so often ("frame
    # capture/align failed" x5 -> exits -> systemd restarts it), and a fresh
    # pipe.start() takes ~15-20s to report "Started OK" again. Every abort
    # we hit (both this driver and Nav_new's isaac_gui, independently) landed
    # inside that recovery window, not a real absence of camera frames --
    # confirmed live 2026-08-27. 30s gives margin above the observed ~20s.
    print("[real-rover] waiting for first camera+depth frame...", flush=True)
    t_wait = time.time()
    while time.time() - t_wait < timeout_s:
        with st.lock:
            ok = st.rgb is not None and st.depth is not None
        if ok:
            print("[real-rover] frames flowing", flush=True)
            return True
        time.sleep(0.2)
    print(f"[real-rover] no camera/depth within {timeout_s:.0f}s, aborting", flush=True)
    return False


HEARTBEAT_PERIOD_S = 0.15  # matches Nihil/nav_pipeline's DinoNavDPZenohNode.HEARTBEAT_PERIOD_S /
# isaac_gui.py's heartbeat_loop -- the rover firmware zeroes velocity if no cmd_vel arrives
# within ~500ms (see Nav_new/nav_pipeline/zenoh_node.py's own comment on this). Every validated
# real-hardware script in this codebase re-publishes the last command on its own timer,
# decoupled from the (variable-latency, sometimes >500ms) inference loop -- added 2026-08-25
# after multiple live runs showed continuous non-zero commands in the log but almost no net
# rover displacement, which this watchdog interaction explains: a single NavDP HTTP call or
# Qwen replan call slower than 500ms let the firmware zero the rover out between our publishes.


def _heartbeat(pub_cmd, get_cmd, stop_flag):
    while not stop_flag.is_set():
        lin, ang = get_cmd()
        pub_cmd.put(serialize_twist(lin, ang))
        time.sleep(HEARTBEAT_PERIOD_S)


def resolve_instructions(rgb, instruction_args: list) -> list:
    """A single compound instruction (the common case -- one `--instruction`
    flag, or the GUI's Send box) gets auto-split into atomic steps via the
    same qwen_client.split_instructions_verbose call nav/gui1.py's sim path
    already uses -- confirmed live (2026-08-27) that feeding a compound
    instruction like "drive towards the door and turn left" straight into
    VlmPointNavigator as ONE instruction makes the model try to satisfy both
    clauses at once and never confidently report advance=true, so it just
    runs out of step budget instead of completing. Multiple EXPLICIT
    `--instruction` flags are left exactly as given (already atomic by the
    caller's own construction) -- only a single instruction ever gets
    split."""
    if len(instruction_args) != 1:
        return instruction_args
    instructions, _split_result = qwen_client.split_instructions_verbose(rgb, instruction_args[0])
    return instructions


def navdp_reset():
    reset_resp = requests.post(
        f"{NAVDP_BASE_URL}/navigator_reset",
        json=_reset_payload(480, 640, HFOV_DEG, 0.0, 1),
        timeout=30.0,
    )
    reset_resp.raise_for_status()
    print(f"[real-rover] navdp navigator_reset -> {reset_resp.json()}", flush=True)


def drive_loop(pub_cmd, st, navigator, max_steps, max_linear, max_angular, on_tick=None, stop_event=None):
    """The actual per-tick control loop -- SAFETY-CRITICAL, single source of
    truth: both the CLI entry point and nav/real_rover_gui.py call this same
    function rather than each keeping their own copy, so a fix (like the
    DEPTH_STOP_M guard) can't accidentally exist in only one of them.

    `on_tick(info: dict)` is called once per tick (if given) with everything
    a caller might want to display -- step, rgb, depth, fwd_depth, the
    navigator's result fields, body_goal, and the published (lin, ang).
    `stop_event` (threading.Event), if given and set, halts the loop within
    one tick and still runs the same zero-twist finally block.
    """
    period = 1.0 / TICK_HZ
    step = 0
    body_goal = None  # (forward, left), re-derived fresh from camera frame each query
    smoothed_yaw = 0.0
    smoothed_lin = 0.0
    prev_gray = None
    stall_counter = 0
    recovery_ticks_left = 0
    prev_t0 = time.time()
    last_rgb_obj = None  # identity (not value) tracking -- see the stall-check comment below
    # Last-known-target-bearing memory (2026-08-27): no persistent world pose
    # exists here by design (see module docstring) -- this tracks only a
    # RELATIVE heading delta, not an absolute pose, so it doesn't need one.
    # last_bearing is atan2(left, forward) at the last tick Qwen actually
    # reported found=True with a real point; yaw_since_bearing accumulates
    # commanded yaw_rate*dt every tick since then (dead-reckoning of how much
    # the rover itself has turned), reset to 0 whenever a fresh sighting
    # lands. When Qwen loses the target (found=False -> body_goal is None),
    # rotating last_bearing by -yaw_since_bearing gives where that target
    # should now be relative to the CURRENT body frame -- used to keep
    # steering toward it instead of stopping dead. None = never seen it yet.
    last_bearing = None
    yaw_since_bearing = 0.0
    last_replan_rgb_obj = None  # separate identity tracker for the fresh-frame-before-replan gate

    # Qwen-cutoff / NavDP-only coasting (2026-08-28): a single Qwen3.5-9B call
    # measured live at ~4.7s -- by far the dominant source of perceived lag,
    # not camera frame delay (measured ~2.4fps/bursty but averaging ~420ms
    # gaps, an order of magnitude faster than a Qwen call). Once Qwen gives a
    # genuine target sighting (target_visible), stop re-querying it on the
    # normal replan cadence and let NavDP alone (each call ~150-170ms, see
    # earlier measurements) drive every tick toward that fixed target,
    # DEAD-RECKONING the goal's (forward, left) each tick by the commanded
    # (lin, ang) actually in effect -- same no-persistent-world-pose,
    # relative-delta approach as last_bearing/yaw_since_bearing above, just
    # extended from angle-only to full 2D position via _dead_reckon_goal.
    # locked=True while coasting; locked_goal is the running dead-reckoned
    # estimate; locked_ticks bounds how long we coast before forcing a fresh
    # Qwen check-in (LOCK_MAX_COAST_TICKS), since dead-reckoning error grows
    # unbounded without ever re-grounding against a real frame.
    locked = False
    locked_goal = None
    locked_ticks = 0

    last_cmd_lock = threading.Lock()
    last_cmd = [0.0, 0.0]

    def set_cmd(lin, ang):
        # Only the heartbeat thread ever calls pub_cmd.put() -- matches
        # isaac_gui.py/point_goal_gui.py's own split (inference_loop only
        # updates st.last_cmd; heartbeat_loop is the sole publisher) rather
        # than having two threads racing to publish on the same handle.
        # The heartbeat's 150ms period is well under one TICK_HZ tick
        # (200ms), so a state change here reaches the wire promptly anyway.
        with last_cmd_lock:
            last_cmd[0], last_cmd[1] = lin, ang

    def get_cmd():
        with last_cmd_lock:
            return tuple(last_cmd)

    hb_stop = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat, args=(pub_cmd, get_cmd, hb_stop), daemon=True)
    hb_thread.start()

    try:
        while step < max_steps:
            if stop_event is not None and stop_event.is_set():
                print("[real-rover] stop requested, halting", flush=True)
                break

            t0 = time.time()
            dt_actual = min(max(t0 - prev_t0, 1e-3), 1.0)  # clamp: avoid a huge single-tick
            # ramp jump after an unusually slow tick, and avoid div-by-zero on a near-instant one
            prev_t0 = t0
            # Dead-reckon heading change since the last confident sighting, using the
            # command that was actually in effect for the dt_actual interval that just
            # elapsed (smoothed_yaw, not yet updated for this tick) -- see last_bearing's
            # comment. A no-op (last_bearing stays None) until the first real sighting.
            if last_bearing is not None:
                yaw_since_bearing += smoothed_yaw * dt_actual
            # Same idea, extended to full position -- see _dead_reckon_goal's
            # comment. Uses the same "command actually in effect during the
            # elapsed interval" values as the bearing update above.
            if locked and locked_goal is not None:
                locked_goal = _dead_reckon_goal(locked_goal[0], locked_goal[1], smoothed_lin, smoothed_yaw, dt_actual)
                locked_ticks += 1
            with st.lock:
                rgb, depth = st.rgb, st.depth

            obs = Observation(rgb=rgb, depth=depth, pose=ZERO_POSE, frame_idx=step)
            info = {"step": step, "rgb": rgb, "depth": depth}

            # Stall detector (ported from internvla_dualvln_zenoh_node.py -- see
            # STALL_FRAME_DIFF_THRESH's comment): if the last tick commanded real
            # motion but the camera frame barely changed, something is physically
            # blocking the rover in a way neither the depth guard nor the model
            # noticed. Independent of both of those.
            #
            # STALE-FRAME GUARD (2026-08-27, live-confirmed root cause of a real
            # rover backing up almost continuously with zero forward progress):
            # this rover's actual camera publish rate can be slower than
            # TICK_HZ, so `rgb` is often the exact same already-processed frame
            # object as last tick, not a new one that happens to look similar.
            # `_frame_stall_score` compares by VALUE and can't tell those apart
            # -- a same-object re-read scores a bit-identical 0.00 diff every
            # time, which is indistinguishable from a genuine wedge to it.
            # Track frame IDENTITY (`is`, not equality -- each real decode in
            # connect()'s on_image callback produces a fresh array object) so
            # "no new frame arrived since last tick" is treated as "no data to
            # judge yet" and skipped entirely, never counted as stall evidence
            # -- only a genuinely NEW frame that still looks unchanged is real
            # stall evidence. The backward-recovery maneuver has no rear
            # obstacle awareness at all, so false positives here are a real
            # collision risk, not just wasted time.
            is_new_frame = rgb is not last_rgb_obj
            last_rgb_obj = rgb
            was_commanding_motion = abs(smoothed_lin) > 0.02 or abs(smoothed_yaw) > 0.02
            if is_new_frame:
                stall_score, prev_gray = _frame_stall_score(rgb, prev_gray)
            else:
                stall_score = None
            if is_new_frame and recovery_ticks_left == 0 and was_commanding_motion \
                    and stall_score is not None and stall_score < STALL_FRAME_DIFF_THRESH:
                stall_counter += 1
                if stall_counter >= STALL_TICKS_CONFIRM:
                    recovery_ticks_left = STALL_RECOVER_TICKS
                    stall_counter = 0
                    # A stall means the dead-reckoned locked_goal (if any) is no longer
                    # trustworthy -- whatever's actually happening (stuck, deflected,
                    # etc.) isn't captured by "we drove forward v*dt", so force a fresh
                    # Qwen check-in once recovery finishes instead of continuing to
                    # coast on a now-suspect estimate.
                    locked = False
                    locked_goal = None
                    print(
                        f"[real-rover]   STALL DETECTED (frame diff {stall_score:.2f} < "
                        f"{STALL_FRAME_DIFF_THRESH}): starting {STALL_RECOVER_TICKS}-tick "
                        "reverse+turn recovery",
                        flush=True,
                    )
            elif is_new_frame:
                stall_counter = 0
            info["stall_score"] = stall_score
            info["stale_frame"] = not is_new_frame

            if recovery_ticks_left > 0:
                recover_ang = math.copysign(STALL_BACKUP_ANG, smoothed_yaw) if smoothed_yaw else STALL_BACKUP_ANG
                smoothed_lin, smoothed_yaw = STALL_BACKUP_LIN, recover_ang
                set_cmd(smoothed_lin, smoothed_yaw)
                info["cmd"] = (smoothed_lin, smoothed_yaw)
                info["recovering"] = True
                recovery_ticks_left -= 1
                print(f"[real-rover]   RECOVERY: lin={smoothed_lin:.3f} ang={smoothed_yaw:+.3f}", flush=True)
                if on_tick:
                    on_tick(info)
                step += 1
                dt = period - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
                continue

            if navigator.finished:
                set_cmd(0.0, 0.0)
                print("[real-rover] all instructions finished, stopping", flush=True)
                info["nav_finished"] = True
                info["cmd"] = (0.0, 0.0)
                if on_tick:
                    on_tick(info)
                break

            # Force a genuinely fresh frame before spending a replan on it --
            # same underlying staleness as the stall-detector fix above, but
            # here it starves the actual navigation DECISION instead of
            # falsely tripping a safety guard. Live-confirmed 2026-08-27:
            # body_goal stayed bit-identical across 16+ consecutive replans
            # while the rover was continuously commanded to drive forward --
            # the model was reasoning over the same cached frame every time
            # and could never perceive the real progress being made, so it
            # never confidently reported advance=true. Only gates replan
            # ticks (maybe_step no-ops on every other tick anyway, so this
            # can't add latency to ticks that wouldn't call the model).
            if locked and locked_ticks >= LOCK_MAX_COAST_TICKS:
                print(
                    f"[real-rover]   locked-mode coast budget ({LOCK_MAX_COAST_TICKS} ticks) "
                    "exceeded -- unlocking, next replan tick will re-query Qwen",
                    flush=True,
                )
                locked = False

            if step % navigator.replan_every == 0 and not navigator.finished and not locked:
                wait_t0 = time.time()
                while rgb is last_replan_rgb_obj and time.time() - wait_t0 < REPLAN_FRESH_FRAME_TIMEOUT_S:
                    # Depth guard stays live even during this bounded wait --
                    # this loop can hold the tick for up to
                    # REPLAN_FRESH_FRAME_TIMEOUT_S, and DEPTH_STOP_M's own
                    # check below only runs once the wait exits, which would
                    # otherwise leave a real (if short) blind window right
                    # when the model is about to make its next decision.
                    fwd_depth_wait = min_forward_depth(depth)
                    if fwd_depth_wait is not None and fwd_depth_wait < DEPTH_STOP_M:
                        set_cmd(0.0, 0.0)
                        smoothed_lin, smoothed_yaw = 0.0, 0.0
                        print(
                            f"[real-rover]   SAFETY STOP during fresh-frame wait: forward depth "
                            f"{fwd_depth_wait:.2f}m < {DEPTH_STOP_M}m",
                            flush=True,
                        )
                        break
                    time.sleep(0.05)
                    with st.lock:
                        rgb, depth = st.rgb, st.depth
                if rgb is last_replan_rgb_obj:
                    print(
                        f"[real-rover]   WARNING: no fresh frame within {REPLAN_FRESH_FRAME_TIMEOUT_S}s "
                        "of replan -- proceeding with a possibly-stale frame anyway",
                        flush=True,
                    )
                last_replan_rgb_obj = rgb
                obs = Observation(rgb=rgb, depth=depth, pose=ZERO_POSE, frame_idx=step)
                info["rgb"], info["depth"] = rgb, depth

            # Suppressed entirely while locked (2026-08-28, Qwen-cutoff): navigator.
            # maybe_step would otherwise still fire its own internal replan_every
            # check and call Qwen regardless of our lock -- see LOCK_MAX_COAST_TICKS'
            # comment for why this is safe to skip for a bounded number of ticks.
            result = navigator.maybe_step(obs, step) if not locked else None
            if result is not None:
                print(
                    f"[real-rover] step={step} instr={result.instruction_text!r} "
                    f"found={result.found} advance={result.advance} "
                    f"reasoning={result.reasoning!r}",
                    flush=True,
                )
                info.update(
                    instruction_text=result.instruction_text,
                    found=result.found,
                    advance=result.advance,
                    reasoning=result.reasoning,
                )
                if result.world_xz is not None:
                    body_goal = body_frame_goal(ZERO_POSE, result.world_xz)
                    print(f"[real-rover]   body_goal (forward, left) = {body_goal}", flush=True)
                else:
                    body_goal = None
                # Bearing memory keyed off result.target_visible/target_world_xz,
                # NOT result.world_xz/body_goal (2026-08-27): world_xz comes from
                # the "affordance" point -- a reasonable NEXT driving step that the
                # model attempts almost every tick by design (see
                # qwen35_prompts.build_point_goal_history_prompt's docstring on the
                # affordance/target split), true even on a plain "keep going
                # forward, I don't actually see it" guess. Using that as "last
                # confident sighting" meant the realignment fallback was chasing
                # wherever the model last pointed for ANY reason, not where the
                # named target actually was. target_visible is the honest signal --
                # only true when the model would genuinely point a finger at the
                # instruction's own target on screen.
                if result.target_visible and result.target_world_xz is not None:
                    target_body_goal = body_frame_goal(ZERO_POSE, result.target_world_xz)
                    last_bearing = math.atan2(target_body_goal[1], target_body_goal[0])
                    yaw_since_bearing = 0.0
                    # Enter/refresh Qwen-cutoff coast mode -- see LOCK_MAX_COAST_TICKS'
                    # comment. A genuine sighting is exactly the "re-grounded against a
                    # real frame" event that resets the coast budget.
                    locked = True
                    locked_goal = target_body_goal
                    locked_ticks = 0
                    print(
                        f"[real-rover]   target genuinely visible at (forward, left) = "
                        f"{target_body_goal} -- bearing memory refreshed, Qwen cut off "
                        f"for up to {LOCK_MAX_COAST_TICKS} ticks (NavDP-only)",
                        flush=True,
                    )
                if result.advance:
                    finished = navigator.advance()
                    print(f"[real-rover] advanced -- finished={finished}", flush=True)
                    body_goal = None
                    # The old bearing/lock belonged to the instruction we just finished --
                    # meaningless for whatever comes next, and would otherwise make the
                    # realignment fallback / locked-mode override below steer back toward
                    # it. Matches VlmPointNavigator.advance()'s own per-instruction history
                    # reset.
                    last_bearing = None
                    yaw_since_bearing = 0.0
                    locked = False
                    locked_goal = None
                    info["nav_finished"] = finished

            # Locked-mode override (2026-08-28): while coasting on Qwen-cutoff, the
            # dead-reckoned locked_goal IS this tick's body_goal -- takes priority over
            # (and preempts) the realignment fallback below, since a dead-reckoned real
            # sighting is strictly better information than "direction we last pointed".
            if locked and locked_goal is not None:
                body_goal = locked_goal

            # Target realignment fallback (2026-08-27): Qwen reported found=False
            # this replan (no fresh sighting -- typically the target left frame
            # mid-avoidance-turn) and we have no body_goal for this tick. Instead
            # of stopping dead (the old behavior, which threw away the direction
            # entirely), steer toward where the target should now be given how
            # much we've turned since we last actually saw it -- REALIGN_DIST_M is
            # just a direction carrier, not a real known distance, so it's kept
            # short enough to stay under NAVDP_GOAL_MAX_DIST_M's clamp untouched.
            if body_goal is None and last_bearing is not None:
                realign_yaw = last_bearing - yaw_since_bearing
                body_goal = (REALIGN_DIST_M * math.cos(realign_yaw), REALIGN_DIST_M * math.sin(realign_yaw))
                info["realigning"] = True
                print(
                    f"[real-rover]   target out of frame -- realigning toward last known "
                    f"bearing {math.degrees(realign_yaw):+.0f} deg (turned "
                    f"{math.degrees(yaw_since_bearing):+.0f} deg since last sighting)",
                    flush=True,
                )
            info["body_goal"] = body_goal

            fwd_depth = min_forward_depth(depth)
            too_close = fwd_depth is not None and fwd_depth < DEPTH_STOP_M
            info["fwd_depth"] = fwd_depth
            info["too_close"] = too_close

            if too_close:
                set_cmd(0.0, 0.0)
                smoothed_yaw = 0.0
                smoothed_lin = 0.0
                info["cmd"] = (0.0, 0.0)
                print(
                    f"[real-rover]   SAFETY STOP: forward depth {fwd_depth:.2f}m < {DEPTH_STOP_M}m, "
                    "overriding to zero-twist",
                    flush=True,
                )
            elif body_goal is None:
                set_cmd(0.0, 0.0)
                smoothed_lin = 0.0
                smoothed_yaw = 0.0
                info["cmd"] = (0.0, 0.0)
            else:
                forward, left = body_goal
                if math.hypot(forward, left) < POINT_GOAL_REACHED_M:
                    set_cmd(0.0, 0.0)
                    smoothed_lin = 0.0
                    smoothed_yaw = 0.0
                    info["cmd"] = (0.0, 0.0)
                    if locked:
                        # Arrived while coasting on Qwen-cutoff -- there's no next
                        # Qwen replan queued to supply an `advance` judgment (that's
                        # the whole point of cutting it off), so the dead-reckoned
                        # distance itself is what decides arrival here. Advance and
                        # unlock so the NEXT instruction (if any) gets a fresh Qwen
                        # sighting rather than incorrectly reusing this goal.
                        finished = navigator.advance()
                        print(
                            f"[real-rover]   locked-mode goal reached (dist="
                            f"{math.hypot(forward, left):.2f}m) -- advancing, finished={finished}",
                            flush=True,
                        )
                        locked = False
                        locked_goal = None
                        last_bearing = None
                        yaw_since_bearing = 0.0
                        info["nav_finished"] = finished
                else:
                    dist = math.hypot(forward, left)
                    if dist > NAVDP_GOAL_MAX_DIST_M:
                        scale = NAVDP_GOAL_MAX_DIST_M / dist
                        navdp_forward, navdp_left = forward * scale, left * scale
                    else:
                        navdp_forward, navdp_left = forward, left
                    try:
                        # REVERTED 2026-08-27 (same day as the change): feeding NavDP
                        # a uniform constant "clear" depth instead of the real sensor
                        # depth was meant to stop it independently re-deciding
                        # avoidance on top of Qwen's chosen point, but live-confirmed
                        # it instead pushed the checkpoint out of its training
                        # distribution -- depth was part of what it was trained on,
                        # and a flat, information-free depth map made it emit the
                        # SAME trajectory tick after tick regardless of the actual
                        # goal (watched body_goal swing from a hard-left realignment
                        # bearing to a nearly-straight-ahead (7.9, 0.0) sighting, and
                        # the commanded action stayed bit-identical for 5+ ticks
                        # after). Back to real depth -- NavDP's own obstacle steering
                        # on top of Qwen's point is a real behavior, not the bug; if
                        # it needs suppressing again, do it some other way than
                        # feeding the model out-of-distribution input.
                        traj_xy, _values = pointgoal_step(
                            NAVDP_BASE_URL, rgb, depth, navdp_forward, navdp_left
                        )
                        info["traj_xy"] = traj_xy
                        # Steer toward a far-along, window-averaged point on the
                        # predicted trajectory instead of an early single sample --
                        # see LOOKAHEAD_FRAC/LOOKAHEAD_WINDOW's comments (ported from
                        # internvla_dualvln_zenoh_node.py: an early index is where
                        # per-tick diffusion-sampling noise is worst).
                        look_fwd, look_left = _lookahead_point(traj_xy)
                        action = select_action_from_trajectory(
                            np.array([[look_fwd, look_left]], dtype=np.float32),
                            waypoint_index=0,
                            max_forward_speed=max_linear,
                            max_yaw_rate=max_angular,
                        )
                        raw_lin = max(0.0, min(action.v_fwd, max_linear))
                        raw_yaw = max(-max_angular, min(action.yaw_rate, max_angular))
                        # Decouple forward speed from a saturated turn -- confirmed live
                        # 2026-08-27 as the actual mechanism behind "moving randomly, not
                        # toward the target": select_action_from_trajectory's v_fwd is
                        # clip(hypot(forward,left), 0, max_forward_speed), so a lookahead
                        # point that's nearly PURELY LATERAL (forward~=0, e.g. (0.00,
                        # +-1.00), which the logs show constantly -- the model deciding
                        # "turn", not "go") still reports a large hypot distance and gets
                        # clipped straight to max forward speed. The rover then drives at
                        # full speed while simultaneously turning at max_angular -- a wide
                        # skidding arc, not a deliberate turn-in-place, which is what reads
                        # as "random" motion. Scale raw_lin down as the commanded turn
                        # saturates, same intent as an Ackermann/skid-steer "slow down in a
                        # sharp turn" rule; at raw_yaw == max_angular this now actually turns
                        # in place instead of arcing through it at full speed.
                        turn_frac = abs(raw_yaw) / max_angular if max_angular > 0 else 0.0
                        raw_lin *= max(0.0, 1.0 - turn_frac)
                        # Wall-clock-scaled ramp limiter, replacing the old fixed-alpha
                        # EMA -- see MOTION_RAMP_TIME_S's comment.
                        max_lin_step = max_linear * (dt_actual / MOTION_RAMP_TIME_S)
                        max_ang_step = max_angular * (dt_actual / MOTION_RAMP_TIME_S)
                        smoothed_lin = _ramp_toward(smoothed_lin, raw_lin, max_lin_step)
                        smoothed_yaw = _ramp_toward(smoothed_yaw, raw_yaw, max_ang_step)
                        set_cmd(smoothed_lin, smoothed_yaw)
                        info["cmd"] = (smoothed_lin, smoothed_yaw)
                        info["raw_yaw"] = raw_yaw
                        print(
                            f"[real-rover]   cmd: lin={smoothed_lin:.3f} ang={smoothed_yaw:+.3f} "
                            f"(raw lin={raw_lin:.3f} ang={raw_yaw:+.3f}, lookahead=({look_fwd:.2f},{look_left:.2f}))",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[real-rover] NavDP step failed: {e}", flush=True)
                        set_cmd(0.0, 0.0)
                        smoothed_yaw = 0.0
                        smoothed_lin = 0.0
                        info["cmd"] = (0.0, 0.0)
                        info["error"] = str(e)

            if on_tick:
                on_tick(info)

            step += 1
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    finally:
        set_cmd(0.0, 0.0)
        time.sleep(0.1)
        set_cmd(0.0, 0.0)
        hb_stop.set()
        hb_thread.join(timeout=1.0)
        print("[real-rover] stopped, zero velocity sent", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--instruction", action="append", required=True)
    ap.add_argument("--max-steps", type=int, default=90)  # ~18s at 5Hz
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--max-linear", type=float, default=MAX_LINEAR)
    ap.add_argument("--max-angular", type=float, default=MAX_ANGULAR)
    args = ap.parse_args()

    session, pub_cmd, st = connect(args.endpoint)
    if not wait_for_frames(st):
        pub_cmd.put(serialize_twist(0.0, 0.0))
        # No session.close() -- see the finally block below's comment.
        return

    with st.lock:
        first_rgb = st.rgb
    instructions = resolve_instructions(first_rgb, args.instruction)
    # Scale the step budget by how many atomic steps the instruction turned
    # into -- args.max_steps was always sized for ONE instruction (~18s at
    # the default 90), so a 2-3 step split needs proportionally more room to
    # actually finish rather than running out mid-instruction.
    max_steps = args.max_steps * max(len(instructions), 1)

    navdp_reset()
    navigator = VlmPointNavigator(
        instructions,
        hfov_deg=HFOV_DEG,
        replan_every=args.replan_every,
        num_history=8,
    )
    print(f"[real-rover] instructions: {instructions} (max_steps={max_steps})", flush=True)
    print(
        f"[real-rover] STARTING TO DRIVE -- max_linear={args.max_linear} max_angular={args.max_angular}",
        flush=True,
    )
    try:
        drive_loop(pub_cmd, st, navigator, max_steps, args.max_linear, args.max_angular)
    finally:
        # NOT calling session.close() (2026-08-27): live-confirmed root cause of
        # a repeated SIGABRT (native "terminate called without an active
        # exception" / "FATAL: exception not rethrown", core-dumped) right after
        # every successful run in nav/real_rover_gui.py's own copy of this same
        # pattern -- zenoh-python's native session teardown appears to race its
        # still-alive subscriber callback threads (kept referenced in st.subs by
        # design, see connect()). A native abort isn't a Python exception, so no
        # try/except can catch it -- only skipping the call does. main() returns
        # right after this either way, so the OS reclaims the socket/threads on
        # process exit regardless of an explicit close().
        pass


if __name__ == "__main__":
    main()
