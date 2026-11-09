#!/usr/bin/env python3
"""Interactive control panel for nav.rover_controller.RoverController --
the in-house, single-process equivalent of Nav_new/MARS/launch_mars.sh's
DINO+NavDP GUI, driving the actual published NavDP model instead of this
repo's own custom S2DiT+NavDP model (see rover_controller.py's docstring).

Run via nav/launch_nav.sh, or directly:
    conda activate habitat
    cd mars-habitatsim
    python -m nav.gui [--scene-path ...] [--heightmap-path ...] [--cbf/--no-cbf] ...
    # this command works as is
    python -m nav.gui1 --gen-home-base --navdp-upstream-root ../navdp_upstream/

Open-vocabulary target grounding ("Ground Target" / a Command-panel GO_TO/FIND
step) uses GroundingDINO (navdp.extensions.GroundingDINODetector, see
sam_vla.goal_resolution.dino_grounding_resolver) instead of Qwen VLM
point-grounding -- ported from scripts/vlm_nav_tests/qwen_search_dino.py's
own DINO usage, --dino-* flags below configure it. The SAM2+Qwen-salience
"Segment" path and the Qwen-based free-text Command-splitting/uncertainty-halt
prompts are unrelated and still use Qwen.

Top-level region layout (Top Status bar / Camera / Mission+Rover-Status
sidebar / Trajectory+Belief bar / pinned Emergency Stop) follows exp.md's
supervisory rover-control-console spec verbatim -- see that file's ASCII
layout and per-region field lists before touching root-grid structure or
adding/removing a sidebar stat cell. The REACH principles below shape the
*interaction design within* that layout (contextual panels, glove-safe
sizing, three-stage-style feedback, etc.), not the region layout itself.

========================================================================
REACH UI/UX PRINCIPLES  (Ma et al., "Human-Robot Interaction through
REACH", DLR/NASA) -- this layout is a deliberate re-implementation of the
REACH Handheld-Device interface principles for a desktop sim console.
Every control below carries a `# REACH Pn:` tag naming the principle it
serves; the same numbering is used in reach_gui/REACH_UI_justification.md.

  P1  CONTEXTUAL / ADAPTIVE MENUS. Actions are shown *because of* the
      current selection or rover state, not all at once. (REACH: "a
      contextual menu that adapts based on user selection" -- object vs
      module vs rover give different actions.)
  P2  THREE-STAGE COMMAND FEEDBACK. Every command surfaces an explicit
      lifecycle: RECEIVED -> EXECUTING -> COMPLETE. (REACH rover-status
      bar: "1) acknowledgment of command receipt, 2) task execution
      status, 3) task completion confirmation" -- the rover gives no
      other signal.)
  P3  CHUNKED, LINEAR TASKWORK. Multi-step work is shown as an ordered
      list with N/total progress so the operator can self-check without
      line-by-line direction. (REACH taskwork pages, "feasible chunks".)
  P4  MAP / SITUATIONAL AWARENESS. A persistent frame of reference:
      rover pose, goal, planned path and hazards, available at any
      moment. (REACH map: rover + user position, terrain/hazards,
      suggested path.)
  P5  SIMPLIFIED, GLOVE-SAFE CONTROLS. Few, large, well-separated
      buttons to prevent accidental activation. (REACH: buttons "spaced
      ... further apart to prevent accidental activation, when wearing
      thick gloves"; "controls were simplified".)
  P6  ONLY ESSENTIAL INFORMATION. Minimize clutter / cognitive load;
      advanced detail is de-emphasized or revealed on demand. (REACH:
      "only essential information is exchanged"; reduced cognitive load.)
  P7  INTUITIVE POINT-AND-SELECT. Direct the rover by pointing at a
      target in the live view. (REACH: "pointing and selecting ... an
      object in environment".)
  P8  ALWAYS-AVAILABLE SAFETY / EMERGENCY TAKEOVER. Stop is reachable at
      all times regardless of scroll or mode. (REACH Rover-Driver
      Console: "emergency takeover"; proximity safety.)
  P9  CONTINUOUS FEEDBACK, LOW FRUSTRATION. Immediate acknowledgment of
      every action to "curb frustration/confusion". (REACH results:
      -40% frustration driven by continuous feedback.)
  P10 AUTONOMY & INTENT COMMUNICATION. Show what the rover is doing and
      why (mode, goal, progress), supporting perceived autonomy. (REACH:
      "communicate intents and progress for better HRC".)
========================================================================
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Optional

import tkinter as tk
import tkinter.font as tkfont
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from sam_vla.goal_resolution import dino_grounding_resolver
from vl_direction import config as vl_dir_config

from nav.mission import GoalKind
from nav.rover_controller import MODE_REVIEW_SEGMENTATION, MODE_TURN, RoverController

REPO_ROOT = Path(__file__).resolve().parent.parent
REFRESH_MS = 66  # ~15 Hz GUI repaint, independent of the controller's own hz
LIVE_STALE_S = 0.25  # controller frame older than this -> draw the raw camera frame

# NASA-STD-3001 Vol. 2 / Appendix F display typography baseline.
#
# The standard defines legibility in angular character height (0.25 deg
# minimum, >=0.4 deg preferred), not in pixels.  These tokens are therefore
# the console's program desktop baseline; final verification still has to be
# performed at the program's worst-case viewing distance, display scaling,
# ambient light, glare, vibration, and operator position.
#
# Appendix F calls for sans-serif text, mixed case for non-acronyms, and a
# fixed-width face for numerical/tabular data.  Keep widget fonts mapped to
# these semantic roles instead of introducing one-off CTkFont instances.
TYPE_SIZE = {
    "caption": 16,  # minimum operational label baseline
    "body": 16,  # normal readable text
    "control": 17,  # buttons and editable controls
    "section": 18,  # panel/region headings
    "value": 20,  # operationally important state/value
    "status": 26,  # persistent current-status heading
    "emergency": 24,  # safety-critical emergency control
    "plot_marker": 28,  # map goal marker
}

FONT_FAMILY = {
    # Exact program font standard, selected from NASA's Horizon digital
    # typography system.  These fonts are deployment dependencies; allowing
    # silent platform substitution would invalidate visual verification.
    "display": "Inter",
    "body": "Public Sans",
    "data": "DM Mono",
}


def _require_standard_fonts(root: tk.Misc) -> None:
    """Fail clearly instead of silently rendering an unverified substitute."""
    installed = {name.casefold() for name in tkfont.families(root)}
    missing = [
        name for name in FONT_FAMILY.values() if name.casefold() not in installed
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required UI font families: {names}. "
            "Install the Inter, Public Sans, and DM Mono font families on "
            "the target system before launching this verified display."
        )


PLOT_BG = "#242424"
PLOT_AXIS = "#555555"
PLOT_ROVER = "#e5e7eb"
PLOT_OBSTACLE = "#f87171"
PLOT_TRAJECTORY = "#f97373"
PLOT_GOAL = "#fbbf24"
# Explicit fill against the plot canvas's own fixed PLOT_BG, unlike a themed
# CTkLabel whose text/background colors can each independently follow the OS
# appearance mode and end up low-contrast -- see _draw_plot_telemetry.
PLOT_TELEMETRY_TEXT = "#e5e7eb"

# REACH P5: glove-safe control sizing. One place to tune the whole
# console's button footprint so every primary action stays large and
# every gap stays wide enough to miss with a thick glove -- the paper's
# core hardware fix ("spaced the buttons further apart").
BTN_H = 48  # primary action button height (px; 4 px layout grid)
BTN_GAP = 8  # inter-button spacing (px)
SECTION_GAP = 12  # spacing between labeled sections (px)

# exp.md Sec 1 (Top Status / Current Status) -- state word -> accent color
# for the persistent full-width header. REACH P10: keeps color the one
# consistent cue for "what is the rover doing right now" across the window.
TOP_STATUS_COLORS = {
    "IDLE": "#9ca3af",
    "PARSING MISSION": "#c084fc",
    "SEARCHING": "#38bdf8",
    "NAVIGATING": "#4ade80",
    "AWAITING HUMAN": "#f59e0b",
    "RECOVERING": "#f87171",
    "GOAL REACHED": "#22d3ee",
    "MISSION COMPLETE": "#4ade80",
    "MANUAL CONTROL": "#60a5fa",
    "USE MANUAL CONTROL": "#ef4444",
    "HUMAN ACTION NEEDED": "#ef4444",
    # REACH P8: the emergency-stop latch outranks every other state --
    # _derive_top_status checks it first, so this color always wins the
    # header the instant STOP is engaged.
    "EMERGENCY STOP": "#dc2626",
}

# Sidebar "Status & alerts" panel (free space below the contextual-panel
# stack, see the panel built in __init__) -- one shared severity vocabulary
# for every folded-in signal (CBF avoidance, low belief confidence, the
# controller thread dying, the emergency-stop latch, the last click
# outcome), most severe first. REACH P9/P10: a single place that always
# tells the operator what's currently wrong, instead of scattered one-off
# labels.
STATUS_LEVEL_COLORS = {
    "critical": "#f87171",
    "warning": "#fb923c",
    "caution": "#fbbf24",
    "info": "#9ca3af",
    "nominal": "#4ade80",
}
STATUS_LEVEL_ORDER = {"critical": 0, "warning": 1, "caution": 2, "info": 3}

# exp.md Sec 3 (Mission panel) "Goal Status" -- current sub-goal's
# execution state -> accent color, shown as one of the Mission panel's
# stat cells (REACH P2: the same continuous three-stage-style feedback,
# expressed in exp.md's own vocabulary instead of a generic RECEIVED/
# EXECUTING/COMPLETE pill).
GOAL_STATUS_COLORS = {
    "SEARCHING": "#38bdf8",
    "DETECTED": "#c084fc",
    "NAVIGATING": "#4ade80",
    "UNCERTAIN": "#f59e0b",
    "REACQUIRING": "#fb923c",
    "REACHED": "#22d3ee",
}

# exp.md Sec 4 (Rover Status panel) "Mode" -> accent color.
ROVER_MODE_COLORS = {
    "IDLE": "#9ca3af",
    "AUTONOMOUS": "#4ade80",
    "HUMAN GUIDANCE": "#f59e0b",
    "MANUAL": "#60a5fa",
}

# Uncertainty-halt numpad panel: rover-front-relative headings (degrees,
# same convention as nav/goal_math.heading_ahead_point/body_frame_goal --
# 0=front, positive=left/counterclockwise, matching the belief tracker's
# (forward, left) body frame (see _draw_plot's to_px and the sigma ellipse
# it draws) -- NOT scripts/habitat_tests/kb_teleop_vl.py's numpad, which
# feeds a different consumer (uncertainty_motion.resolve_absolute_heading_deg)
# with the opposite sign (positive=right) and is intentionally left alone.
# Bound to the same physical key layout as that script's panel (same keys,
# same grid positions) purely so the on-screen layout looks familiar, redrawn
# independently here rather than imported (nav/ stays decoupled from that
# script). Grid position drives the heads-up panel's clickable button
# layout, keys drive the numpad keyboard shortcuts.
UNCERTAINTY_HEADING_KEYS = {
    "8": 0.0,
    "9": -45.0,
    "6": -90.0,
    "3": -135.0,
    "2": 180.0,
    "1": 135.0,
    "4": 90.0,
    "7": 45.0,
}
UNCERTAINTY_HEADING_GRID_POS = {
    "7": (0, 0),
    "8": (0, 1),
    "9": (0, 2),
    "4": (1, 0),
    "6": (1, 2),
    "1": (2, 0),
    "2": (2, 1),
    "3": (2, 2),
}


def split_sequence(text: str) -> list[str]:
    """"turn left; go forward 3 m then go home" -> 3 ordered instructions.
    Splits on ';', newlines, and the word "then" (also ", then" / "and then");
    no model call. Always returns at least one item."""
    # "and" splits only before a motion verb, so "find the red and blue
    # flag" stays one instruction.
    verbs = r"(?:turn|go|walk|move|stop|drive|find|come|return|head|face|wait)\b"
    parts = re.split(
        rf"\s*(?:;|\n|,?\s+and\s+then\s+|,?\s+then\s+|,?\s+and\s+(?={verbs})|,\s+(?={verbs}))\s*",
        text.strip(),
        flags=re.I,
    )
    parts = [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]
    return parts or [text.strip()]


def _default_navdp_upstream_ckpt() -> Optional[str]:
    """Same repo-relative fallback sam_vla.run_navdp_rollout uses -- a plain
    invocation just works on a checkout that already has Phase 0's
    (next.md's Integration project) gitignored checkpoint in place."""
    candidate = REPO_ROOT / "navdp" / "navdp-cross-modal.ckpt"
    return str(candidate) if candidate.exists() else None


class NavGuiApp:
    SIDEBAR_MIN_W = 360
    # exp.md Sec 5: TRAJECTORY / BELIEF is a fixed-height bar below the
    # camera, not a square panel sized off the sidebar's own width. Its
    # column width now matches the camera's own (see plot_frame.grid in
    # __init__) rather than spanning the full window, so the sidebar can
    # rowspan into the freed column-1 space at that row.
    PLOT_BOTTOM_H = 200
    PLOT_RANGE = 6.0  # meters shown top-to-bottom of the body-frame plot

    def __init__(
        self,
        root: ctk.CTk,
        controller: RoverController,
    ):
        self.root = root
        self.controller = controller
        self.closed = False
        # callbacks(text) for every Taskwork prompt sent (nav/user_study.py logs them)
        self.command_listeners: list = []
        self._pending_click_norm: Optional[tuple[float, float]] = None
        self._seg_panel_visible = False
        self._click_panel_visible = False
        self._cam_img_box: Optional[tuple[int, int, int, int]] = None
        self._cam_photo = None
        # Manual drive: no dedicated dpad/section in this layout, but the
        # arrow keys still hold-to-drive via the same controller.set_manual
        # path the old dpad buttons used (see manual_press/_manual_update).
        self._manual_held: set[str] = set()
        # REACH P9: last action acknowledged, echoed in the status header
        # the instant a button is pressed (before the controller's own
        # state catches up on the next tick) so no press ever feels lost.
        # Expires after _ACK_HOLD_S so it can't permanently mask a later
        # derived subtext (e.g. a stale "taskwork queued: ..." sitting under
        # a since-reached "GOAL REACHED" label with nothing else queued).
        self._ack_text = ""
        self._ack_until = 0.0

        # REACH P8: the emergency stop is a latch, not a fire-and-forget
        # button -- once engaged it stays visibly ACTIVE (button + top
        # status + the Status & alerts panel all reflect it) and locks out
        # every drive-initiating control (manual_press/submit_command/
        # confirm_segmentation/confirm_pixel_goal/submit_uncertainty_heading)
        # until explicitly released via the same button or Reset Rover --
        # see stop()/_release_estop()/_update_estop_button().
        self._estop_active = False

        # Most recent alerts list (see _derive_alerts), refreshed every
        # tick by _sync_alerts_panel -- the hover-detail popup's Enter
        # handler reads this directly for an instant, un-stale first paint
        # instead of waiting on the next refresh() tick (see
        # _show_alerts_tooltip / _ALERTS_TOOLTIP_MAX_ROWS section below).
        self._last_alerts: list[tuple[str, str]] = []

        # Purely additive, human-viz-only window (see RoverController.
        # enable_topdown_viz) -- never touches the main console's layout
        # below; only exists when the controller was started with the flag.
        self._topdown_window = None
        self._topdown_canvas = None
        self._topdown_photo = None
        if getattr(self.controller, "enable_topdown_viz", False):
            self._build_topdown_window()

        root.title("mars-habitatsim/nav · REACH console")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.minsize(900, 640)

        # Start maximized so the layout below (sized entirely off grid
        # weights, not fixed pixel panels) actually gets the whole screen to
        # lay out into. Always set an explicit near-full-screen geometry
        # first -- window managers that ignore "zoomed"/"-zoomed" (or a
        # WM-less display) would otherwise silently leave the window at its
        # small natural size -- then best-effort request a true maximize on
        # top of that for the window managers that do support it.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{int(sw * 0.95)}x{int(sh * 0.9)}")
        try:
            root.state("zoomed")
        except tk.TclError:
            try:
                root.attributes("-zoomed", True)
            except tk.TclError:
                pass

        # Root grid, following exp.md's four persistent regions top to
        # bottom: row 0 is the Top Status bar, row 1 is the working area
        # (camera hero + sidebar), row 2 is the Trajectory/Belief bar, row 3
        # is the pinned emergency bar. REACH P8: the STOP control lives
        # OUTSIDE the scrollable sidebar, on its own always-visible root
        # row, so an emergency takeover is one click away no matter how far
        # the sidebar is scrolled or what mode is active.
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=0)
        root.grid_rowconfigure(3, weight=0)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=1, minsize=self.SIDEBAR_MIN_W)

        # One program-wide display vocabulary, as required by NASA-STD-3001.
        # Fonts are exact deployment dependencies so tested character shape,
        # metrics, wrapping, and control geometry cannot silently change.
        _require_standard_fonts(root)
        self.display_font_family = FONT_FAMILY["display"]
        self.body_font_family = FONT_FAMILY["body"]
        self.data_font_family = FONT_FAMILY["data"]
        self.fonts = {
            "caption": ctk.CTkFont(
                family=self.body_font_family, size=TYPE_SIZE["caption"]
            ),
            "caption_bold": ctk.CTkFont(
                family=self.body_font_family,
                size=TYPE_SIZE["caption"],
                weight="bold",
            ),
            "data_label": ctk.CTkFont(
                family=self.data_font_family,
                size=TYPE_SIZE["caption"],
                weight="bold",
            ),
            "body": ctk.CTkFont(family=self.body_font_family, size=TYPE_SIZE["body"]),
            "body_bold": ctk.CTkFont(
                family=self.body_font_family,
                size=TYPE_SIZE["body"],
                weight="bold",
            ),
            "control": ctk.CTkFont(
                family=self.body_font_family,
                size=TYPE_SIZE["control"],
                weight="bold",
            ),
            "section": ctk.CTkFont(
                family=self.display_font_family,
                size=TYPE_SIZE["section"],
                weight="bold",
            ),
            "value": ctk.CTkFont(
                family=self.data_font_family,
                size=TYPE_SIZE["value"],
                weight="bold",
            ),
            "state": ctk.CTkFont(
                family=self.display_font_family,
                size=TYPE_SIZE["value"],
                weight="bold",
            ),
            "status": ctk.CTkFont(
                family=self.display_font_family,
                size=TYPE_SIZE["status"],
                weight="bold",
            ),
            "emergency": ctk.CTkFont(
                family=self.display_font_family,
                size=TYPE_SIZE["emergency"],
                weight="bold",
            ),
        }

        # ===== TOP STATUS / CURRENT STATUS (exp.md Sec. 1) =============== #
        # Full-width, persistent, above everything else -- the one line the
        # operator should be able to read without inspecting the camera,
        # mission, or telemetry panels (REACH P9/P10: continuous, unmissable
        # feedback on what the rover is doing right now and why).
        top_status_bar = ctk.CTkFrame(root, border_width=2, border_color="#3f3f46")
        self.top_status_bar = top_status_bar
        top_status_bar.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4)
        )
        # REACH P6/P9/P10: "Status & alerts" shares this row with the
        # primary status readout instead of living buried at the bottom of
        # the sidebar -- an active alert is often *why* the state above it
        # just changed, so both need to land in the same glance. Mirrors the
        # camera:sidebar split directly below it so the two horizontal
        # divisions read as the same visual system instead of two different
        # ratios stacked on top of each other. That split is nominally 3:1
        # (root.grid_columnconfigure weight=3/weight=1) but ISN'T always
        # exactly that in practice -- SIDEBAR_MIN_W can win over the
        # weight=1 share on a narrow enough window -- so the ratio set here
        # is only the construction-time fallback; _sync_top_bar_ratio()
        # (bound to cam_frame/sidebar's own <Configure>, called once real
        # widths exist) overwrites it with the two rows' actual live pixel
        # ratio, every time the window resizes. The alerts column is
        # consequently narrow -- see _sync_alerts_panel, which always shows
        # the single most useful (most severe) alert and bolds it so it
        # still reads clearly even when truncated. Row 0 has weight=0 (see
        # root.grid_rowconfigure below) so this bar's height is fixed by
        # its own content -- both
        # columns are built to stay at a constant 2-line height (see
        # _sync_alerts_panel's single-line, ellipsis-truncated message) so
        # neither column's content can grow the bar and steal height from
        # the camera/sidebar row underneath it.
        # uniform="topbar": plain weight=N/weight=M only splits the space
        # left over *after* each column's own content minimum, and the
        # alerts column's minimum (driven by its "Status & alerts" header
        # text) is proportionally much bigger than the status column's
        # (empty labels until refresh() fills them) -- that skewed a plain
        # weight=3/weight=1 split to ~2:1 in practice. A shared uniform
        # group forces widths strictly proportional to weight regardless of
        # each column's own minimum; _sync_top_bar_ratio() below keeps the
        # weight values themselves equal to cam_frame/sidebar's live pixel
        # widths, so "proportional to weight" means "proportional to the
        # row underneath it" rather than a fixed guess.
        top_status_bar.grid_columnconfigure(0, weight=3, uniform="topbar")
        top_status_bar.grid_columnconfigure(1, weight=0)
        top_status_bar.grid_columnconfigure(2, weight=1, uniform="topbar")
        top_status_bar.grid_rowconfigure(0, weight=1)

        # width=1/height=1: without an explicit override each defaults to
        # CTkFrame's own 200x200, which (like the divider above) can act as
        # a floor that fights the 3:1 weight split -- pinning both down to
        # their content's actual size is what makes the ratio deterministic
        # instead of only approximately 3:1 depending on current text.
        status_col = ctk.CTkFrame(top_status_bar, fg_color="transparent", width=1, height=1)
        status_col.grid(row=0, column=0, sticky="nsew")
        status_col.grid_columnconfigure(0, weight=1)
        self.top_status_label = ctk.CTkLabel(
            status_col,
            text="",
            anchor="w",
            justify="left",
            font=self.fonts["status"],
        )
        self.top_status_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 0))
        self.top_status_detail_label = ctk.CTkLabel(
            status_col,
            text="",
            anchor="w",
            justify="left",
            font=self.fonts["body"],
            text_color="#e5e7eb",
        )
        self.top_status_detail_label.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 8)
        )
        status_col.bind(
            "<Configure>",
            lambda e: self.top_status_detail_label.configure(
                wraplength=max(200, e.width - 28)
            ),
        )
        self.alive_label = ctk.CTkLabel(
            status_col,
            text="controller thread died -- see console",
            anchor="w",
            justify="left",
            font=self.fonts["caption_bold"],
            text_color="#f87171",
        )
        self._alive_label_visible = False

        # height=1: CTkFrame defaults to height=200 and this divider has no
        # children of its own to shrink-wrap around, so without an explicit
        # override it silently forced the whole bar to ~220px tall (the
        # actual bug behind the too-thick top bar) -- sticky="ns" still
        # stretches it to match status_col/alerts_col's real content height.
        divider = ctk.CTkFrame(top_status_bar, fg_color="#3f3f46", width=2, height=1)
        divider.grid(row=0, column=1, sticky="ns", pady=10)

        # ===== STATUS & ALERTS (shares the top bar -- REACH P6+P9+P10) === #
        # Compact by design, not a truncated version of a bigger list: only
        # the single most severe active condition is shown (STATUS_LEVEL_
        # ORDER), with a "(+N more)" suffix for anything else active --
        # matches REACH P6 (only essential information) now that this lives
        # in permanently-visible real estate instead of a scrollable
        # sidebar. Overflow is handled by fitting to the column's actual
        # measured pixel width (see _fit_single_line / _on_alerts_col_
        # configure), truncating with an ellipsis rather than wrapping, so
        # a long CBF/error message can never grow this row's fixed height.
        alerts_col = ctk.CTkFrame(top_status_bar, fg_color="transparent", width=1, height=1)
        self.alerts_col = alerts_col
        alerts_col.grid(row=0, column=2, sticky="nsew")
        alerts_col.grid_columnconfigure(0, weight=1)
        self.alerts_header_label = ctk.CTkLabel(
            alerts_col,
            text="Status & alerts",
            anchor="w",
            justify="left",
            font=self.fonts["caption_bold"],
            text_color="#9ca3af",
        )
        self.alerts_header_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 0))
        # Bold: the 3:1 ratio leaves little room, so whatever survives
        # truncation (see _sync_alerts_panel) still has to read as the
        # single most-important thing on the row, not just fit in it.
        self.alerts_message_label = ctk.CTkLabel(
            alerts_col,
            text="",
            anchor="w",
            justify="left",
            font=self.fonts["body_bold"],
        )
        self.alerts_message_label.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 8)
        )
        self._alerts_col_width = 0

        def _on_alerts_col_configure(e) -> None:
            self._alerts_col_width = e.width

        alerts_col.bind("<Configure>", _on_alerts_col_configure)

        # ---- hover-to-expand detail overlay (REACH P6+P9+P10) ----------- #
        # The bar itself only ever shows the single worst alert (space is
        # fixed-height, see above) -- hovering reveals every active alert in
        # full, un-truncated, in a borderless popup anchored just below the
        # column so nothing about the compact/detailed pair has to be
        # learned separately from the rest of this console's bordered
        # panels. Built once (hidden) rather than per-hover so showing it is
        # just a reposition + deiconify, not a widget rebuild every time the
        # mouse passes over.
        self._ALERTS_TOOLTIP_MAX_ROWS = 8
        self._alerts_tooltip_visible = False
        self._alerts_hide_job: Optional[str] = None
        self._alerts_tooltip = ctk.CTkToplevel(self.root)
        self._alerts_tooltip.withdraw()
        self._alerts_tooltip.overrideredirect(True)
        self._alerts_tooltip.attributes("-topmost", True)
        tooltip_frame = ctk.CTkFrame(
            self._alerts_tooltip, border_width=2, border_color="#3f3f46", width=1
        )
        tooltip_frame.pack(fill="both", expand=True)
        tooltip_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            tooltip_frame,
            text="Status & alerts",
            font=self.fonts["section"],
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))
        self._alerts_tooltip_rows: list[ctk.CTkLabel] = []
        for i in range(self._ALERTS_TOOLTIP_MAX_ROWS):
            lbl = ctk.CTkLabel(
                tooltip_frame,
                text="",
                anchor="w",
                justify="left",
                font=self.fonts["body"],
                wraplength=420,
            )
            lbl.grid(row=1 + i, column=0, sticky="ew", padx=14, pady=3)
            lbl.grid_remove()
            self._alerts_tooltip_rows.append(lbl)
        self._alerts_tooltip_nominal_label = ctk.CTkLabel(
            tooltip_frame,
            text="●  nominal — no active warnings",
            anchor="w",
            justify="left",
            font=self.fonts["body"],
            text_color=STATUS_LEVEL_COLORS["nominal"],
        )
        self._alerts_tooltip_nominal_label.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(3, 10)
        )
        # Bound on the trigger area AND on the popup itself: without the
        # latter, moving the mouse off alerts_col and down into the popup
        # to actually read a wrapped line would itself count as a Leave and
        # start the hide timer before the pointer ever lands inside it.
        for w in (
            alerts_col,
            self.alerts_header_label,
            self.alerts_message_label,
            tooltip_frame,
        ):
            w.bind("<Enter>", self._on_alerts_hover_enter)
            w.bind("<Leave>", self._on_alerts_hover_leave)

        # ---------------- camera hero (REACH P7: point-and-select) ------- #
        # Fills all left-column space; the frame is letterboxed into it on
        # every refresh so it tracks window size. Clicking the live view is
        # the primary, most intuitive way to designate a goal -- the direct
        # analog of REACH "point and select an object in the environment".
        cam_frame = ctk.CTkFrame(root)
        self.cam_frame = cam_frame
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(4, 4))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_canvas = tk.Canvas(cam_frame, bg="#111111", highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 2))
        self.cam_canvas.bind("<Button-1>", self.on_cam_click)
        ctk.CTkLabel(
            cam_frame,
            text="",
            font=self.fonts["caption_bold"],
            text_color="#22d3ee",
        ).grid(row=1, column=0, pady=(0, 6))

        # ---------------- sidebar ---------------------------------------- #
        # REACH P6 (only essential info) + P5 (spacing): one vertical
        # column of clearly-labeled sections, top-anchored, generous gaps.
        # Scrollable so no control is ever clipped off a short monitor
        # (contextual review panels can push the stack past screen height).
        sidebar = ctk.CTkScrollableFrame(root, fg_color="transparent")
        # Rowspan down into row 2 (the trajectory bar's row): the trajectory
        # bar is now narrowed to the camera's own 3-of-4 column width (see
        # plot_frame.grid below), which frees column 1 at row 2 for the
        # sidebar to occupy instead of leaving it empty -- taller sidebar,
        # fewer controls pushed below the fold / behind a scroll.
        sidebar.grid(
            row=1, column=1, rowspan=2, sticky="nsew", padx=(5, 10), pady=(4, 4)
        )
        sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar = sidebar

        row = 0

        def _stat_cell(parent, col, title) -> ctk.CTkLabel:
            # Small titled stat card shared by the Mission and Rover Status
            # panels below (exp.md's DISTANCE / GOAL BELIEF / GOAL STATUS /
            # CURRENT UNCERTAINTY / HOME UNCERTAINTY cells) -- one visual
            # vocabulary for "labeled single value" across both panels.
            cell = ctk.CTkFrame(parent, fg_color="#1a1a1e", corner_radius=6)
            cell.grid(row=0, column=col, sticky="nsew", padx=3)
            ctk.CTkLabel(
                cell,
                text=title,
                font=self.fonts["data_label"],
                text_color="#9ca3af",
            ).pack(pady=(6, 0))
            value_label = ctk.CTkLabel(cell, text="--", font=self.fonts["value"])
            value_label.pack(pady=(0, 6))
            return value_label

        # ===== SECTION 1 — MISSION (exp.md Sec. 3) ======================= #
        # "What is the rover trying to accomplish, how far away is the
        # goal, and how confident is it about that goal?" -- the ordered
        # goal list plus the four at-a-glance stats about the active goal.
        self.mission_panel = ctk.CTkFrame(
            sidebar, border_width=2, border_color="#3f3f46"
        )
        self.mission_panel.grid(row=row, column=0, sticky="ew")
        row += 1
        self.mission_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.mission_panel,
            text="Mission",
            font=self.fonts["section"],
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        self.goal_list_label = ctk.CTkLabel(
            self.mission_panel,
            text="",
            anchor="w",
            justify="left",
            font=self.fonts["body"],
        )
        self.goal_list_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        ctk.CTkLabel(
            self.mission_panel,
            text="Current goal",
            font=self.fonts["data_label"],
            text_color="#9ca3af",
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 0))
        self.current_goal_label = ctk.CTkLabel(
            self.mission_panel,
            text="--",
            anchor="w",
            justify="left",
            font=self.fonts["state"],
        )
        self.current_goal_label.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))

        mission_stat_row = ctk.CTkFrame(self.mission_panel, fg_color="transparent")
        mission_stat_row.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 10))
        for i in range(3):
            mission_stat_row.grid_columnconfigure(i, weight=1)
        self.goal_distance_label = _stat_cell(mission_stat_row, 0, "Distance")
        self.goal_belief_label = _stat_cell(mission_stat_row, 1, "Goal belief")
        self.goal_status_label = _stat_cell(mission_stat_row, 2, "Status")

        self.mission_panel.bind(
            "<Configure>",
            lambda e: self.goal_list_label.configure(wraplength=max(200, e.width - 20)),
        )

        # ===== SECTION 2 — ROVER STATUS (exp.md Sec. 4) =================== #
        # "How is the rover operating, what mode is it in, and how certain
        # is its localization?" -- mode plus the two uncertainty readouts
        # plus low-level telemetry, kept separate from Mission/goal state.
        self.status_panel = ctk.CTkFrame(
            sidebar, border_width=2, border_color="#3f3f46"
        )
        self.status_panel.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        self.status_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.status_panel,
            text="Rover status",
            font=self.fonts["section"],
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        self.mode_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=self.fonts["state"],
        )
        self.mode_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        rover_stat_row = ctk.CTkFrame(self.status_panel, fg_color="transparent")
        rover_stat_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        rover_stat_row.grid_columnconfigure(0, weight=1)
        rover_stat_row.grid_columnconfigure(1, weight=1)
        self.current_uncertainty_label = _stat_cell(
            rover_stat_row, 0, "Current uncertainty"
        )
        self.target_uncertainty_label = _stat_cell(rover_stat_row, 1, "Target uncertainty")

        # Telemetry: only the operational values exp.md calls out (position,
        # heading, velocity, yaw rate, nav state) -- model-internal/CBF
        # debugging detail stays out of this panel (see the trajectory/
        # belief bar's own HUD for that). Each value gets its own labeled
        # stat cell (same title-over-bold-value vocabulary as the
        # uncertainty cells above) instead of one crammed wrapped text
        # blob -- lets the eye jump straight to a single field instead of
        # re-parsing "pos (..)  heading=..deg" strings on every glance.
        telemetry_row1 = ctk.CTkFrame(self.status_panel, fg_color="transparent")
        telemetry_row1.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 4))
        telemetry_row1.grid_columnconfigure(0, weight=1)
        telemetry_row1.grid_columnconfigure(1, weight=1)
        self.telemetry_position_label = _stat_cell(telemetry_row1, 0, "Position (x, z)")
        self.telemetry_heading_label = _stat_cell(telemetry_row1, 1, "Heading")

        telemetry_row2 = ctk.CTkFrame(self.status_panel, fg_color="transparent")
        telemetry_row2.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        telemetry_row2.grid_columnconfigure(0, weight=1)
        telemetry_row2.grid_columnconfigure(1, weight=1)
        self.telemetry_velocity_label = _stat_cell(telemetry_row2, 0, "Velocity")
        self.telemetry_yaw_rate_label = _stat_cell(telemetry_row2, 1, "Yaw rate")

        self.telemetry_nav_state_label = ctk.CTkLabel(
            self.status_panel,
            text="--",
            anchor="w",
            justify="left",
            font=self.fonts["body_bold"],
            text_color="#e5e7eb",
        )
        self.telemetry_nav_state_label.grid(
            row=5, column=0, sticky="ew", padx=10, pady=(0, 10)
        )
        self.status_panel.bind(
            "<Configure>",
            lambda e: self.telemetry_nav_state_label.configure(
                wraplength=max(200, e.width - 20)
            ),
        )

        # ===== SECTION 4 — TASKWORK (REACH P3 + P10) ===================== #
        # Free-text command -> ordered, chunked mission the operator can
        # self-check step by step (progress synced from d.mission_status,
        # and shown structured in the MISSION panel above via d.mission_goals
        # once dispatched), exactly REACH's linear taskwork pages that
        # "reduce the need for constant, line-by-line directions from
        # ground control".
        command_panel = ctk.CTkFrame(sidebar)
        command_panel.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        command_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(command_panel, text="Taskwork", font=self.fonts["section"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )
        self.command_entry = ctk.CTkEntry(
            command_panel,
            placeholder_text='"go to a flag then return to home"',
            height=BTN_H,
            font=self.fonts["body"],
        )
        self.command_entry.grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 4))
        self.command_entry.bind("<Return>", lambda e: self.submit_command())
        # REACH P3: dispatch the ordered mission.
        ctk.CTkButton(
            command_panel,
            text="Send",
            width=72,
            height=BTN_H,
            font=self.fonts["control"],
            command=self.submit_command,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(0, 4))
        # Plain (non-latching) Stop: cancels the running instruction queue
        # and halts the rover; unlike Emergency Stop it doesn't lock the
        # controls, so a new command can be typed straight away.
        ctk.CTkButton(
            command_panel,
            text="Stop",
            height=BTN_H,
            font=self.fonts["control"],
            fg_color="#b45309",
            hover_color="#92400e",
            command=self.stop_task,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        # Human intervention: pause autonomy and drive by hand (WASD/arrows);
        # press again to hand control back. The rover also offers this by
        # itself when the target's position is too uncertain.
        self._iv_active = False
        self.intervention_button = ctk.CTkButton(
            command_panel,
            text="Take over (human drives)",
            height=BTN_H,
            font=self.fonts["control"],
            fg_color="#1d4ed8",
            hover_color="#1e40af",
            command=self.toggle_intervention,
        )
        self.intervention_button.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8)
        )
        self.mission_status_label = ctk.CTkLabel(
            command_panel,
            text="",
            anchor="w",
            justify="left",
            font=self.fonts["body"],
            text_color="#e5e7eb",
            wraplength=self.SIDEBAR_MIN_W - 20,
        )
        self.mission_status_label.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, BTN_GAP)
        )

        # ===== CONTEXTUAL PANEL — CONFIRM MISSION (REACH P1+P2+P10) ===== #
        # Only gridded while a Taskwork command has finished VLM-parsing and
        # is awaiting operator sign-off (RoverController._mission_awaiting_
        # confirmation / DisplayState.mission_review) -- shows the
        # interpreted direction/goal split *before* any driving starts, so
        # the operator can catch a misparse from the words alone instead of
        # discovering it from the rover already moving (REACH P2:
        # interpreted-command feedback shown explicitly, ahead of
        # EXECUTING; REACH P10: communicate intent before acting on it).
        self._mission_review_row = row
        row += 1
        self.mission_review_panel = ctk.CTkFrame(
            sidebar, border_width=2, border_color="#a78bfa"
        )
        ctk.CTkLabel(
            self.mission_review_panel,
            text="Confirm mission",
            font=self.fonts["section"],
            text_color="#a78bfa",
        ).pack(pady=(10, 2))
        self.mission_review_desc_label = ctk.CTkLabel(
            self.mission_review_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
            font=self.fonts["body_bold"],
        )
        self.mission_review_desc_label.pack(padx=12, pady=(0, 4), fill="x")
        self.mission_review_steps_label = ctk.CTkLabel(
            self.mission_review_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
            font=self.fonts["body"],
        )
        self.mission_review_steps_label.pack(padx=12, pady=(0, 8), fill="x")
        self.mission_review_panel.bind(
            "<Configure>",
            lambda e: [
                self.mission_review_desc_label.configure(
                    wraplength=max(200, e.width - 24)
                ),
                self.mission_review_steps_label.configure(
                    wraplength=max(200, e.width - 24)
                ),
            ],
        )
        mission_review_btn_col = ctk.CTkFrame(
            self.mission_review_panel, fg_color="transparent"
        )
        mission_review_btn_col.pack(pady=(0, 12), fill="x", padx=12)
        mission_review_btn_col.grid_columnconfigure(0, weight=1)
        # REACH P1+P2: accept the interpreted mission -- driving starts next tick.
        ctk.CTkButton(
            mission_review_btn_col,
            text="Confirm & Start",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.confirm_mission,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: reject -- rover stays exactly where it is (no motion
        # ever started), operator can edit and resubmit the Taskwork entry.
        ctk.CTkButton(
            mission_review_btn_col,
            text="Cancel",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.cancel_mission,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=1, column=0, sticky="ew", pady=3)
        self._mission_review_panel_visible = False

        # Manual drive: arrow keys hold-to-drive (no dpad/section rendered --
        # see class docstring notes near _manual_held); guarded so typing in
        # the taskwork entry doesn't also drive the rover.
        for key, direction in (
            ("Up", "fwd"),
            ("Down", "back"),
            ("Left", "left"),
            ("Right", "right"),
        ):
            root.bind(
                f"<KeyPress-{key}>",
                lambda e, d=direction: self._guarded(self.manual_press, d),
            )
            root.bind(
                f"<KeyRelease-{key}>",
                lambda e, d=direction: self._guarded(self.manual_release, d),
            )

        # Escape is a global keyboard shortcut bound on root (fires on every
        # keypress regardless of focus) -- cancels a pending camera click /
        # hard-clears goals, independent of any particular sidebar section.
        root.bind("<Escape>", lambda e: self.clear_all())

        # Numpad heading shortcuts for the uncertainty-halt panel below --
        # bound unconditionally (these digits aren't used elsewhere in this
        # GUI); the controller itself drops a submission that isn't
        # currently halted, so this is a no-op the rest of the time.
        for key, angle in UNCERTAINTY_HEADING_KEYS.items():
            root.bind(
                f"<KeyPress-{key}>",
                lambda e, a=angle: self._guarded(self.submit_uncertainty_heading, a),
            )
        root.bind("<KeyPress-r>", lambda e: self._guarded(self.retry_uncertainty))
        root.bind("<KeyPress-R>", lambda e: self._guarded(self.retry_uncertainty))

        # ===== SEGMENT GOAL (REACH P1) ==================================== #
        # Ad-hoc SAM2+Qwen-salience "Segment" goal resolution
        # (RoverController.request_resolve() with no target text): the
        # custom SAM2-LoRA head proposes rock candidates in the current
        # frame, and Qwen VLM salience picks one as the goal, registering
        # the rest as obstacle masks. This is a manual override with no
        # other trigger in this console (mission GO_TO/FIND steps use DINO
        # grounding and auto-confirm instead, see submit_nav_command's
        # docstring). Firing this is the only thing that can ever put the
        # controller into MODE_REVIEW_SEGMENTATION here, so the "REVIEW
        # RESOLVED GOAL" contextual panel right below only ever becomes
        # reachable once this section runs.
        seg_trigger_panel = ctk.CTkFrame(sidebar)
        seg_trigger_panel.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        seg_trigger_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            seg_trigger_panel,
            text="Segment Goal",
            font=self.fonts["section"],
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
        ctk.CTkButton(
            seg_trigger_panel,
            text="Run Segmentation",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.run_segmentation,
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # ===== CONTEXTUAL PANEL A — REVIEW RESOLVED GOAL (REACH P1) ===== #
        # Only gridded while actually reviewing a resolved goal. This IS a
        # REACH contextual menu: the Confirm / Rerun / Pick-manually
        # actions exist only because a segmentation/grounding selection
        # exists to act on. Bordered so it reads as "action needed".
        self._seg_row = row
        row += 1
        self.seg_panel = ctk.CTkFrame(sidebar, border_width=2, border_color="#f59e0b")
        ctk.CTkLabel(
            self.seg_panel,
            text="Review resolved goal",
            font=self.fonts["section"],
            text_color="#f59e0b",
        ).pack(pady=(10, 2))
        self.seg_desc_label = ctk.CTkLabel(
            self.seg_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
            font=self.fonts["body"],
        )
        self.seg_desc_label.pack(padx=12, pady=(0, 8), fill="x")
        self.seg_panel.bind(
            "<Configure>",
            lambda e: self.seg_desc_label.configure(wraplength=max(200, e.width - 24)),
        )
        seg_btn_col = ctk.CTkFrame(self.seg_panel, fg_color="transparent")
        seg_btn_col.pack(pady=(0, 12), fill="x", padx=12)
        seg_btn_col.grid_columnconfigure(0, weight=1)
        # REACH P1: accept this selection as the goal.
        ctk.CTkButton(
            seg_btn_col,
            text="Confirm",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.confirm_segmentation,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: reject and re-resolve (same intent, new result).
        ctk.CTkButton(
            seg_btn_col,
            text="Rerun",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.rerun_segmentation,
            fg_color="#b45309",
            hover_color="#92400e",
        ).grid(row=1, column=0, sticky="ew", pady=3)
        # REACH P1 + P7: fall back to point-and-select if the auto result
        # is wrong.
        ctk.CTkButton(
            seg_btn_col,
            text="Pick Manually",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.pick_manually,
        ).grid(row=2, column=0, sticky="ew", pady=3)

        # ===== CONTEXTUAL PANEL B — CONFIRM POINT GOAL (REACH P1 + P7) == #
        # Only gridded while a click is pending confirmation -- the
        # contextual action menu for a point-and-select target.
        self._click_row = row
        row += 1
        self.click_panel = ctk.CTkFrame(sidebar, border_width=2, border_color="#00b8d4")
        self.click_desc_label = ctk.CTkLabel(
            self.click_panel,
            text="Set the goal at the point you clicked?",
            font=self.fonts["body_bold"],
            text_color="#22d3ee",
            wraplength=self.SIDEBAR_MIN_W - 40,
        )
        self.click_desc_label.pack(pady=(10, 6), padx=12)
        click_btn_col = ctk.CTkFrame(self.click_panel, fg_color="transparent")
        click_btn_col.pack(pady=(0, 10), fill="x", padx=12)
        click_btn_col.grid_columnconfigure(0, weight=1)
        # REACH P7: commit the pointed-at goal.
        ctk.CTkButton(
            click_btn_col,
            text="Confirm Point Goal",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.confirm_pixel_goal,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: back out of the selection (REACH menu "back button").
        ctk.CTkButton(
            click_btn_col,
            text="Cancel",
            height=BTN_H,
            font=self.fonts["control"],
            command=self.cancel_pixel_click,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=1, column=0, sticky="ew", pady=3)

        # ===== CONTEXTUAL PANEL C — UNCERTAINTY HALT (REACH P1+P8+P10) == #
        # Only gridded while the controller is halted on belief
        # uncertainty. A contextual heading-request menu: the rover has
        # stopped itself (P8 safety) and asks the human for a direction
        # (P10 shared autonomy). Same bordered numpad shape as
        # scripts/habitat_tests/kb_teleop_vl.py's panel, redrawn here.
        self._uncertainty_row = row
        row += 1
        self.uncertainty_panel = ctk.CTkFrame(
            sidebar, border_width=2, border_color="#38bdf8"
        )
        self.uncertainty_title = ctk.CTkLabel(
            self.uncertainty_panel, text="", font=self.fonts["section"]
        )
        self.uncertainty_title.pack(pady=(10, 2))
        self.uncertainty_desc_label = ctk.CTkLabel(
            self.uncertainty_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
            font=self.fonts["body"],
        )
        self.uncertainty_desc_label.pack(padx=12, pady=(0, 8), fill="x")
        self.uncertainty_panel.bind(
            "<Configure>",
            lambda e: self.uncertainty_desc_label.configure(
                wraplength=max(200, e.width - 24)
            ),
        )
        uncertainty_grid = ctk.CTkFrame(self.uncertainty_panel, fg_color="transparent")
        uncertainty_grid.pack(pady=(0, 6))
        self.uncertainty_buttons: dict[str, ctk.CTkButton] = {}
        # REACH P1 + P5: one large heading key per compass direction, in
        # the numpad layout, spaced to prevent mis-selection.
        for key, angle in UNCERTAINTY_HEADING_KEYS.items():
            r, c = UNCERTAINTY_HEADING_GRID_POS[key]
            btn = ctk.CTkButton(
                uncertainty_grid,
                text=f"{key}\n{angle:+.0f}°",
                width=68,
                height=50,
                font=self.fonts["control"],
                command=lambda a=angle: self.submit_uncertainty_heading(a),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.uncertainty_buttons[key] = btn
        # REACH P1: re-request a VLM heading sweep instead of committing.
        self.uncertainty_retry_button = ctk.CTkButton(
            self.uncertainty_panel,
            text="Retry (R)",
            height=BTN_H,
            font=self.fonts["control"],
            fg_color="#b45309",
            hover_color="#92400e",
            command=self.retry_uncertainty,
        )
        self.uncertainty_retry_button.pack(pady=(0, 12))
        self._uncertainty_panel_visible = False

        # ===== TRAJECTORY / BELIEF (exp.md Sec. 5) ======================= #
        # "Where has the rover been, and where does it currently believe
        # the goal is?" -- persistent, below the camera (REACH P4 map/
        # situational-awareness), pulled out of the scrollable sidebar so
        # it's never affected by sidebar scroll position. Width matches the
        # camera's own column (the same 3-of-4-parts split as row 1's
        # camera:sidebar division) rather than spanning the full window --
        # narrowing it this way frees column 1 at this row for the sidebar
        # to occupy instead (see sidebar's rowspan=2 above), so the sidebar
        # gets more height and needs less scrolling to show its controls.
        plot_frame = ctk.CTkFrame(root)
        plot_frame.grid(row=2, column=0, sticky="ew", padx=(10, 5), pady=(0, 4))
        plot_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            plot_frame,
            text="Trajectory / belief",
            font=self.fonts["section"],
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))
        self.plot = tk.Canvas(
            plot_frame, height=self.PLOT_BOTTOM_H, bg=PLOT_BG, highlightthickness=0
        )
        self.plot.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 6))

        # ===== PINNED EMERGENCY BAR (REACH P8) ========================== #
        # On its own root row, spanning the full width, OUTSIDE the
        # scrollable sidebar -- always on screen. This is the paper's
        # emergency-takeover affordance: the single most safety-critical
        # control gets the largest, reddest, most reachable real estate.
        #
        # The button is a LATCH, not a fire-and-forget action: engaging it
        # visibly changes shape/color/label (outlined "ready" -> solid
        # filled "ACTIVE", taller, brighter, high-contrast text) and stays
        # that way -- an unmissable, continuously-true readout of whether
        # driving is currently locked out, not just a click that happened
        # and is over (REACH P9: continuous feedback, not one-shot).
        # _update_estop_button() is the single place that redraws it; every
        # entry point that changes _estop_active calls it (stop(),
        # reset_rover()).
        stop_bar = ctk.CTkFrame(root, fg_color="transparent")
        stop_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        stop_bar.grid_columnconfigure(0, weight=3)
        stop_bar.grid_columnconfigure(1, weight=1)
        self.stop_button = ctk.CTkButton(
            stop_bar,
            text="",
            height=56,
            font=self.fonts["emergency"],
            command=self.stop,
            border_width=3,
        )
        self.stop_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        # REACH P8: reset/recover is destructive-adjacent, so it sits next
        # to STOP but is visually subordinate (grey, narrower). Also the
        # one other control that releases the estop latch (a full reset
        # already puts the rover back in a known-safe idle state).
        ctk.CTkButton(
            stop_bar,
            text="Reset Rover",
            height=56,
            font=self.fonts["control"],
            command=self.reset_rover,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._update_estop_button()

        # Keep the top bar's status:alerts split live-matched to
        # cam_frame/sidebar's actual widths (see _sync_top_bar_ratio) --
        # bound to both rather than just cam_frame since either one can be
        # the one that changes width first (e.g. SIDEBAR_MIN_W clamping the
        # sidebar on a resize toward a narrower window). winfo_width() is
        # still 1x1 at construction time (nothing has been realized on
        # screen yet), so the static uniform="topbar" weights set above are
        # what's visible until the first real <Configure> event corrects
        # them. add=True is required on the sidebar: CTkScrollableFrame's
        # own __init__ already does `self.bind("<Configure>", ...)` to keep
        # its scrollregion in sync with its content (that's what makes
        # scrolling and _scroll_sidebar_to_bottom/_scroll_sidebar_to_top
        # work at all) -- binding without add=True replaces that handler
        # instead of adding to it and silently breaks scrolling entirely.
        self.cam_frame.bind("<Configure>", self._sync_top_bar_ratio, add=True)
        self.sidebar.bind("<Configure>", self._sync_top_bar_ratio, add=True)

        self.root.after(REFRESH_MS, self.refresh)

    # ---------------- dynamic sizing ---------------- #
    def _scroll_sidebar_to_bottom(self) -> None:
        # The contextual review/confirm panels grid in near the bottom of
        # the sidebar stack; on a short window the sidebar scrolls rather
        # than clipping, so jump the scroll down as soon as one appears
        # instead of leaving an action-required panel to be found by
        # accident (REACH P9: never make the operator hunt for the next
        # step). Deferred one tick since the panel's own .grid() hasn't been
        # laid out (scrollregion hasn't grown to include it) yet this frame.
        self.root.after_idle(lambda: self.sidebar._parent_canvas.yview_moveto(1.0))

    def _scroll_sidebar_to_top(self) -> None:
        # Mirror of _scroll_sidebar_to_bottom for when a contextual panel
        # closes -- return to the always-on status/map sections above.
        self.root.after_idle(lambda: self.sidebar._parent_canvas.yview_moveto(0.0))

    def _sync_top_bar_ratio(self, _event=None) -> None:
        # Keeps the top bar's status:alerts split matched to cam_frame's and
        # sidebar's own live pixel widths -- NOT a hardcoded 3:1, because
        # that's only the nominal ratio; SIDEBAR_MIN_W can win over the
        # weight=1 share and widen the sidebar's real fraction on a narrow
        # enough window (see root.grid_columnconfigure in __init__), and a
        # hardcoded ratio would silently drift out of sync with the actual
        # camera/sidebar boundary when that happens. Grid `weight` is just a
        # relative ratio, so handing it these two frames' current
        # winfo_width() directly (combined with the uniform="topbar" group
        # set at construction, which is what makes weight ratio -> visual
        # ratio exact regardless of each column's own content minimum)
        # reproduces whatever that real ratio currently is.
        cam_w = self.cam_frame.winfo_width()
        side_w = self.sidebar.winfo_width()
        if cam_w <= 1 or side_w <= 1:
            return  # not realized on screen yet -- keep the built-in 3:1 fallback
        self.top_status_bar.grid_columnconfigure(0, weight=cam_w, uniform="topbar")
        self.top_status_bar.grid_columnconfigure(2, weight=side_w, uniform="topbar")

    # ---------------- commands ---------------- #
    def _guarded(self, fn, *fn_args) -> None:
        # Skip root-level keyboard shortcuts (the uncertainty-halt numpad
        # heading/retry bindings below) while the taskwork entry has focus,
        # so typing a command there doesn't also submit a heading.
        if self.root.focus_get() is self.command_entry:
            return
        fn(*fn_args)

    _ACK_HOLD_S = 3.0

    def _ack(self, text: str) -> None:
        # REACH P9: record an immediate human-readable acknowledgment of
        # the just-pressed control; refresh() echoes it in the status
        # header so no press ever feels unregistered. Held for
        # _ACK_HOLD_S then falls back to the derived subtext (see
        # _sync_top_status) rather than sticking around forever.
        self._ack_text = text
        self._ack_until = time.time() + self._ACK_HOLD_S

    def _estop_block(self, what: str) -> bool:
        # REACH P8: the latch actually gates every drive-initiating control,
        # not just the initial stop_driving() call -- otherwise an
        # incidental arrow-key tap or a stale Confirm click could silently
        # resume motion while the button still reads ACTIVE. Called from
        # each such entry point below; a no-op (returns False) whenever the
        # estop isn't currently engaged.
        if not self._estop_active:
            return False
        self._ack(f"blocked — release emergency stop before {what}")
        return True

    def submit_command(self) -> None:
        # Handed to RoverController.start_vlm_navigation_from_command, the
        # same InternVLA-N1-style pipeline --vlm-command uses: Qwen splits
        # the free text into self-contained instruction steps
        # (qwen_prompts.build_split_instruction_prompt) and
        # nav.vlm_point_navigator.VlmPointNavigator drives them one at a
        # time, starting immediately (no separate confirm step). Previously
        # this went through RoverController.submit_nav_command's
        # closed-vocabulary Mission parser (nav.mission.parse_parts, staged
        # into the Confirm-Mission panel) -- that parser's fixed
        # direction-token vocabulary (turn_left/turn_right/go_left/go_right/
        # go_front only) silently drops any landmark-relative movement
        # phrase that doesn't literally match one of those tokens (e.g. "go
        # through the doorway on right"), so a command like "go through the
        # doorway on right and find a chair" collapsed to just "find a
        # chair" with the doorway leg discarded entirely. VlmPointNavigator
        # keeps each split step as its own full sentence, so no such lossy
        # discretization happens. submit_nav_command/Mission still exist
        # for any other caller, just no longer wired to this button.
        text = self.command_entry.get().strip()
        if not text:
            return
        if self._estop_block("dispatching taskwork"):
            return
        print(f"[nav command] submitted: {text}")
        for cb in self.command_listeners:
            try:
                cb(text)
            except Exception as exc:  # logging must never block driving
                print(f"[nav command] listener failed: {exc}")
        self._ack(f"VLM navigation started: {text}")
        # No Qwen splitting: a plain ";" / "then" split into an ordered queue
        # (a single instruction stays a one-item queue).
        self.controller.start_vlm_navigation(split_sequence(text))
        self.command_entry.delete(0, "end")

    def confirm_mission(self) -> None:
        # REACH P2: the operator has reviewed the Confirm-Mission panel's
        # interpreted step list -- this is the one action that actually
        # starts driving for a Taskwork command (see submit_command above,
        # which only queues parsing).
        if self._estop_block("starting a mission"):
            return
        self._ack("mission confirmed — starting")
        self.controller.request_confirm_mission()

    def cancel_mission(self) -> None:
        self._ack("mission cancelled")
        self.controller.request_cancel_mission()

    def reset_rover(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        # A full reset already puts the rover back in a known-safe idle
        # state at spawn, so it's the other control (besides the stop
        # button itself) that releases the latch.
        self._estop_active = False
        self._update_estop_button()
        self._ack("rover reset")
        self.controller.request_reset()

    def toggle_intervention(self) -> None:
        if self._estop_block("driving"):
            return
        self._manual_held.clear()
        if self._iv_active:
            self.controller.resume_autonomy()
            self._ack("autonomy resumed")
        else:
            self.controller.request_intervention("operator took over")
            self._ack("HUMAN INTERVENTION — drive with WASD/arrows, press again to resume")

    def _sync_intervention(self, d) -> None:
        active = bool(getattr(d, "intervention_active", False))
        if active != self._iv_active:
            self._iv_active = active
            if active:
                reason = getattr(d, "intervention_reason", "") or "uncertain"
                self._ack(f"HUMAN INTERVENTION — {reason}. Drive with WASD/arrows, then press Resume.")
                if reason != "operator took over":
                    self._show_manual_control_alert(reason)
            else:
                self._close_manual_control_alert()
            self.intervention_button.configure(
                text="Resume autonomy" if active else "Take over (human drives)",
                fg_color="#15803d" if active else "#1d4ed8",
                hover_color="#166534" if active else "#1e40af",
            )
        # Automatic hand-over = human control ONLY: no resume until the target
        # uncertainty has dropped back under the limit.
        if active and getattr(d, "intervention_locked", False):
            tu = getattr(d, "target_uncertainty", None)
            thr = getattr(d, "intervention_sigma_threshold", 0.0)
            blocked = tu is not None and thr > 0 and tu >= thr
            self.intervention_button.configure(
                state="disabled" if blocked else "normal",
                text="Human control only — find the target to resume" if blocked else "Resume autonomy",
            )
        else:
            self.intervention_button.configure(state="normal")

    def _show_manual_control_alert(self, reason: str) -> None:
        """The rover has stopped (target uncertainty over the limit): a simple,
        centered, always-on-top message plus a bell, until acknowledged."""
        self._close_manual_control_alert()
        try:
            self.root.bell()
            win = ctk.CTkToplevel(self.root)
            win.title("Human action needed")
            win.attributes("-topmost", True)
            win.configure(fg_color="#7f1d1d")
            ctk.CTkLabel(
                win, text="HUMAN ACTION NEEDED",
                font=ctk.CTkFont(size=30, weight="bold"), text_color="#ffffff",
            ).pack(padx=60, pady=(40, 24))
            ctk.CTkButton(
                win, text="OK", width=140, height=40, command=self._close_manual_control_alert,
                fg_color="#dc2626", hover_color="#b91c1c",
            ).pack(pady=(0, 36))
            win.protocol("WM_DELETE_WINDOW", self._close_manual_control_alert)
            win.update_idletasks()
            w, h = win.winfo_reqwidth(), win.winfo_reqheight()
            rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
            rw, rh = self.root.winfo_width(), self.root.winfo_height()
            win.geometry(f"{w}x{h}+{rx + (rw - w) // 2}+{ry + (rh - h) // 2}")
            self._manual_alert_win = win
        except Exception:
            self._manual_alert_win = None

    def _close_manual_control_alert(self) -> None:
        win = getattr(self, "_manual_alert_win", None)
        self._manual_alert_win = None
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def stop_task(self) -> None:
        """Stop button: cancel the VLM instruction queue + halt driving."""
        self._manual_held.clear()
        self.controller.stop_vlm_navigation()
        self.controller.stop_driving()
        self._ack("stopped — task cancelled")

    def stop(self) -> None:
        # REACH P8: a latch, not a one-shot action -- first press engages
        # (halts driving, locks out every drive-initiating control, button
        # goes solid/ACTIVE); the SAME button's second press releases the
        # lock (does not itself resume driving -- a new command still has
        # to be issued deliberately). _update_estop_button keeps the
        # button's shape/color/label in sync as the one always-visible
        # readout of this state.
        self._manual_held.clear()
        self.cancel_pixel_click()
        if self._estop_active:
            self._estop_active = False
            self._update_estop_button()
            self._ack("emergency stop released — driving unlocked")
            return
        self._estop_active = True
        self._update_estop_button()
        self._ack("EMERGENCY STOP ENGAGED — driving locked")
        # Also cancel the VLM queue, or the navigator keeps issuing goals.
        self.controller.stop_vlm_navigation()
        self.controller.stop_driving()

    def _update_estop_button(self) -> None:
        if self._estop_active:
            self.stop_button.configure(
                text="■  STOPPED — driving locked (click to release)",
                fg_color="#dc2626",
                hover_color="#b91c1c",
                border_color="#fecaca",
                text_color="#ffffff",
            )
        else:
            self.stop_button.configure(
                text="■  Emergency stop",
                fg_color="#7f1d1d",
                hover_color="#991b1b",
                border_color="#b91c1c",
                text_color="#fca5a5",
            )

    def clear_all(self) -> None:
        # Escape key: a hard clear, not just stop() -- also strips resolved
        # goal/obstacle masks, the active mission, and any uncertainty halt,
        # leaving the rover exactly where it is (unlike reset_rover, which
        # also teleports it back to spawn). See RoverController.clear_all_goals.
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("cleared — all goals, missions, and masks removed")
        self.controller.clear_all_goals()

    # ---------------- manual drive ---------------- #
    def manual_press(self, direction: str) -> None:
        if self._estop_block("driving"):
            return
        self.cancel_pixel_click()
        self._manual_held.add(direction)
        self._manual_update()

    def manual_release(self, direction: str) -> None:
        self._manual_held.discard(direction)
        self._manual_update()

    def _manual_update(self) -> None:
        if not self._manual_held:
            self.controller.stop_driving()
            return
        lin = ang = 0.0
        if "fwd" in self._manual_held:
            lin += self.controller.max_forward_speed
        if "back" in self._manual_held:
            lin -= 0.5 * self.controller.max_forward_speed
        if "left" in self._manual_held:
            ang += self.controller.max_yaw_rate
        if "right" in self._manual_held:
            ang -= self.controller.max_yaw_rate
        self.controller.set_manual(lin, ang)

    # ---------------- segmentation trigger + review ---------------- #
    def run_segmentation(self) -> None:
        self._manual_held.clear()
        self._ack("running segmentation…")
        self.controller.request_resolve()

    def confirm_segmentation(self) -> None:
        if self._estop_block("confirming a goal"):
            return
        self._ack("goal confirmed")
        self.controller.request_confirm_segmentation()

    def rerun_segmentation(self) -> None:
        self._ack("re-resolving goal…")
        self.controller.request_rerun_segmentation()

    def pick_manually(self) -> None:
        self._manual_held.clear()
        self._ack("pick a point in the live view")
        self.controller.request_pick_manually()

    # ---------------- uncertainty halt ---------------- #
    def submit_uncertainty_heading(self, angle_deg: float) -> None:
        if self._estop_block("sending a heading"):
            return
        self._ack(f"heading sent: {angle_deg:+.0f}°")
        self.controller.submit_uncertainty_heading(angle_deg)

    def retry_uncertainty(self) -> None:
        self._ack("requesting VLM sweep…")
        self.controller.retry_uncertainty_request()

    # ---------------- click-to-goal ---------------- #
    def on_cam_click(self, event) -> None:
        # The camera frame is letterboxed into cam_canvas (see _draw_camera),
        # so a click has to be mapped through the last-drawn image's actual
        # on-canvas box, not the canvas's own (possibly wider/taller) bounds.
        if self._cam_img_box is None:
            return
        x0, y0, x1, y1 = self._cam_img_box
        if not (x0 <= event.x <= x1 and y0 <= event.y <= y1):
            return
        nx = min(max((event.x - x0) / (x1 - x0), 0.0), 1.0)
        ny = min(max((event.y - y0) / (y1 - y0), 0.0), 1.0)
        self._pending_click_norm = (nx, ny)
        self._ack("point selected — confirm below")

    def confirm_pixel_goal(self) -> None:
        if self._pending_click_norm is None:
            return
        if self._estop_block("committing a goal"):
            return
        nx, ny = self._pending_click_norm
        self._ack("point goal committed")
        self.controller.request_pixel_goal(nx, ny)
        self._pending_click_norm = None

    def cancel_pixel_click(self) -> None:
        self._pending_click_norm = None

    def _build_topdown_window(self) -> None:
        """Second, independent window showing the fixed bird's-eye camera
        (see RoverController.enable_topdown_viz / MarsHabitatEnv.
        get_topdown_rgb) for video-recording/human-viz purposes only -- not
        wired into any click/goal/manual-drive handler, and never touches
        the main console window built by the rest of __init__."""
        win = ctk.CTkToplevel(self.root)
        win.title("mars-habitatsim/nav · Top-Down View")
        win.geometry("900x900")
        win.minsize(360, 360)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(win, bg="#111111", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        self._topdown_window = win
        self._topdown_canvas = canvas

    def _draw_topdown(self, vis_rgb) -> None:
        cw = self._topdown_canvas.winfo_width()
        ch = self._topdown_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        img = Image.fromarray(vis_rgb).convert("RGB")
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((dw, dh))
        self._topdown_photo = ImageTk.PhotoImage(img)
        x0, y0 = (cw - dw) // 2, (ch - dh) // 2
        self._topdown_canvas.delete("frame")
        self._topdown_canvas.create_image(
            x0, y0, anchor="nw", image=self._topdown_photo, tags="frame"
        )

    def on_close(self) -> None:
        self.closed = True
        if self._topdown_window is not None:
            self._topdown_window.destroy()
        self.root.destroy()

    # ---------------- refresh loop ---------------- #
    def _derive_goal_status(self, d) -> Optional[str]:
        # exp.md Sec 3 "Goal Status" -- the active sub-goal's execution
        # state, folded from the controller's existing signals (no new
        # controller field needed, same trick the old REACH 3-stage pills
        # used). Precedence: REACHED wins, then the uncertainty-halt
        # states, then SEARCHING/DETECTED/NAVIGATING based on whether a
        # goal has been sighted and whether the rover is currently moving.
        # None when no goal is active at all (idle / manual drive) -- the
        # Mission panel shows a dash then.
        if d.goal_reached:
            return "REACHED"
        if d.uncertainty_halted:
            return "UNCERTAIN"
        if d.uncertainty_searching:
            return "REACQUIRING"
        active = d.mode in ("point", "resolve", MODE_TURN, MODE_REVIEW_SEGMENTATION)
        if not active:
            return None
        if d.mode == MODE_REVIEW_SEGMENTATION:
            return "DETECTED"
        if d.mode == "resolve" and d.belief_g is None:
            return "SEARCHING"
        moving = abs(d.action.v_fwd) > 0.01 or abs(d.action.yaw_rate) > 0.01
        return "NAVIGATING" if moving else "DETECTED"

    @staticmethod
    def _subgoal_label(g) -> str:
        # nav.mission.SubGoal -> exp.md Goal-List display text. GO_TO/FIND
        # show the landmark name; RETURN/TURN/ADVANCE (direction legs, not
        # landmarks) get a short verb label instead.
        if g.kind in (GoalKind.GO_TO, GoalKind.FIND):
            return g.target.upper()
        if g.kind == GoalKind.RETURN:
            return "HOME"
        if g.kind == GoalKind.TURN:
            return f"TURN {g.target.upper()}"
        if g.kind == GoalKind.ADVANCE:
            return "ADVANCE"
        return g.raw.upper() or "?"

    def _current_goal_text(self, d) -> str:
        if d.mission_goals and 0 <= d.mission_goal_idx < len(d.mission_goals):
            g = d.mission_goals[d.mission_goal_idx]
            if g.kind != GoalKind.DONE:
                return self._subgoal_label(g)
        # Ad-hoc (non-mission) goal -- Segment/Ground/Random/point-click --
        # d.goal_label is GoalSpec.instruction_text, still its startup
        # placeholder until a goal is actually resolved.
        if d.goal_label and d.goal_label != "(no goal resolved yet)":
            return d.goal_label.upper()
        return ""

    def _goal_list_text(self, d) -> str:
        # exp.md Sec 3 "Goal List" -- ✓ done / → current / ○ upcoming.
        if not d.mission_goals:
            # No VLM-parsed Mission running -- degrade to a single-entry
            # list around whatever ad-hoc goal is active, rather than
            # leaving the panel blank.
            label = self._current_goal_text(d)
            return f"→ 1. {label}" if label else "(no active mission)"
        lines = []
        n = 0
        for i, g in enumerate(d.mission_goals):
            if g.kind == GoalKind.DONE:
                continue
            n += 1
            if i < d.mission_goal_idx:
                marker = "✓"
            elif i == d.mission_goal_idx:
                marker = "→"
            else:
                marker = "○"
            lines.append(f"{marker} {n}. {self._subgoal_label(g)}")
        return "\n".join(lines) if lines else "(no active mission)"

    def _mission_review_steps_text(self, mission) -> str:
        # Confirm-Mission panel's step list -- same _subgoal_label
        # vocabulary as _goal_list_text, but with no done/current markers
        # since nothing has started yet (the whole point of this panel is
        # that it's shown *before* the mission is dispatched).
        lines = [
            f"{n}. {self._subgoal_label(g)}"
            for n, g in enumerate(
                (g for g in mission.goals if g.kind != GoalKind.DONE), start=1
            )
        ]
        return "\n".join(lines) if lines else "(no steps parsed)"

    def _derive_goal_belief(self, d) -> tuple[str, str]:
        # exp.md Sec 3 "Goal Belief" -- a human-readable confidence label,
        # not the raw covariance value (that stays numeric in the Rover
        # Status panel / trajectory-belief HUD). Ratio thresholds mirror
        # _draw_plot_telemetry's own 0.8-of-threshold warning color.
        if (
            d.belief_g is None
            or not d.uncertainty_enabled
            or d.uncertainty_threshold <= 0
        ):
            return "--", PLOT_TELEMETRY_TEXT
        ratio = d.uncertainty_value / d.uncertainty_threshold
        if ratio >= 0.8:
            return "LOW CONFIDENCE", "#f87171"
        if ratio >= 0.5:
            return "LOW CONFIDENCE", "#fbbf24"
        return "HIGH CONFIDENCE", "#4ade80"

    @staticmethod
    def _derive_rover_mode(d) -> str:
        # exp.md Sec 4 "Mode" -- collapses the controller's finer-grained
        # d.mode into the four states the operator actually needs to tell
        # apart; HUMAN GUIDANCE wins whenever an uncertainty halt is
        # actively asking the operator for a heading, regardless of d.mode.
        if getattr(d, "intervention_active", False):
            return "HUMAN DRIVING"
        if d.mode == "manual":
            return "MANUAL"
        if d.uncertainty_halted:
            return "HUMAN GUIDANCE"
        if d.mode == "idle":
            return "IDLE"
        return "AUTONOMOUS"

    def _derive_top_status(
        self, d, goal_status: Optional[str], alive: bool
    ) -> tuple[str, str, str]:
        # exp.md Sec 1 -- (state, subtext, color). `state` is drawn from
        # exp.md's own "typical states" vocabulary; `subtext` carries the
        # specific why/what (goal name, halt reason), matching that
        # section's own "Navigating to ANTENNA" / "Goal localization
        # uncertainty exceeded threshold" examples.
        if self._estop_active:
            # REACH P8: outranks everything else -- the header is the first
            # place an operator looks, and a locked-out console must never
            # read as "navigating" or "idle" while the latch is engaged.
            return (
                "EMERGENCY STOP",
                "Driving locked — release the emergency stop to resume",
                TOP_STATUS_COLORS["EMERGENCY STOP"],
            )
        if not alive:
            return (
                "RECOVERING",
                "controller thread died -- see console",
                TOP_STATUS_COLORS["RECOVERING"],
            )
        if d.mission_review is not None:
            # REACH P2/P10: a Taskwork command has finished VLM-parsing but
            # hasn't been dispatched -- outranks whatever an already-active
            # mission is doing, since this is the freshest thing needing the
            # operator's attention (see Confirm-Mission panel below).
            return (
                "AWAITING HUMAN",
                f'Confirm mission "{d.mission_review.instruction}" below to begin',
                TOP_STATUS_COLORS["AWAITING HUMAN"],
            )
        if getattr(d, "intervention_active", False):
            return (
                "HUMAN ACTION NEEDED",
                "Drive with WASD / arrow keys",
                TOP_STATUS_COLORS["HUMAN ACTION NEEDED"],
            )
        if d.uncertainty_halted:
            reason = (
                d.uncertainty_line or "Goal localization uncertainty exceeded threshold"
            )
            return "AWAITING HUMAN", reason, TOP_STATUS_COLORS["AWAITING HUMAN"]
        if d.mode == "manual":
            return (
                "MANUAL CONTROL",
                "Operator driving",
                TOP_STATUS_COLORS["MANUAL CONTROL"],
            )
        if (
            d.mission_goals
            and d.mission_goal_idx >= 0
            and d.mission_goals[d.mission_goal_idx].kind == GoalKind.DONE
        ):
            return (
                "MISSION COMPLETE",
                d.mission_status or "All goals reached",
                TOP_STATUS_COLORS["MISSION COMPLETE"],
            )
        if d.goal_reached:
            target = self._current_goal_text(d)
            # d.mission_goals/mission_goal_idx are only populated while
            # RoverController has an active multi-goal mission stepping
            # itself (rover_controller.py's _env_loop) -- empty/-1 here
            # means there is no queued next sub-goal to auto-advance to,
            # so reaching this one just leaves the rover sitting idle
            # unless the operator picks a new mission.
            if not d.mission_goals or d.mission_goal_idx < 0:
                subtext = (
                    f"Reached {target} -- no further goals queued. Pick the next mission."
                    if target
                    else "No further goals queued -- pick the next mission."
                )
            else:
                subtext = f"Reached {target}" if target else d.status_text
            return "GOAL REACHED", subtext, TOP_STATUS_COLORS["GOAL REACHED"]
        if d.mode == MODE_REVIEW_SEGMENTATION:
            return (
                "PARSING MISSION",
                d.status_text,
                TOP_STATUS_COLORS["PARSING MISSION"],
            )
        if goal_status == "SEARCHING":
            target = self._current_goal_text(d)
            subtext = f"Searching for {target}" if target else d.status_text
            return "SEARCHING", subtext, TOP_STATUS_COLORS["SEARCHING"]
        if goal_status in ("NAVIGATING", "DETECTED", "REACQUIRING"):
            target = self._current_goal_text(d)
            subtext = f"Navigating to {target}" if target else d.status_text
            return "NAVIGATING", subtext, TOP_STATUS_COLORS["NAVIGATING"]
        # d.status_text defaults to "idle -- pick a driving mode" here,
        # which would just repeat "idle" back under a header that already
        # says it -- show one unambiguous operating-mode indicator instead.
        return (
            "IDLE",
            "Ready for mission -- pick a driving mode to begin",
            TOP_STATUS_COLORS["IDLE"],
        )

    def _sync_mission_panel(self, d, goal_status: Optional[str]) -> None:
        self.goal_list_label.configure(text=self._goal_list_text(d))
        current = self._current_goal_text(d)
        self.current_goal_label.configure(text=current or "--")
        self.goal_distance_label.configure(
            text=f"{d.distance:.1f} m" if d.distance is not None else "--"
        )
        belief_text, belief_color = self._derive_goal_belief(d)
        self.goal_belief_label.configure(text=belief_text, text_color=belief_color)
        self.goal_status_label.configure(
            text=goal_status.title() if goal_status else "--",
            text_color=GOAL_STATUS_COLORS.get(goal_status, "#e5e7eb"),
        )

    @staticmethod
    def _unc_color(value: float, threshold: float) -> str:
        if threshold <= 0:
            return PLOT_TELEMETRY_TEXT
        ratio = value / threshold
        if ratio >= 0.8:
            return "#f87171"
        if ratio >= 0.5:
            return "#fbbf24"
        return "#4ade80"

    def _sync_rover_status_panel(self, d) -> None:
        rover_mode = self._derive_rover_mode(d)
        self.mode_label.configure(
            text=rover_mode.title(),
            text_color=ROVER_MODE_COLORS.get(rover_mode, "#e5e7eb"),
        )

        if d.uncertainty_enabled:
            cur_text = f"{d.uncertainty_value:.2f}/{d.uncertainty_threshold:.2f}"
            cur_color = self._unc_color(d.uncertainty_value, d.uncertainty_threshold)
        else:
            cur_text, cur_color = "off", PLOT_TELEMETRY_TEXT
        self.current_uncertainty_label.configure(text=cur_text, text_color=cur_color)
        self._sync_intervention(d)

        # Target uncertainty: how unsure the rover is where the VLM target is
        # (belief sigma + staleness, metres). At the threshold the rover hands
        # control to the human (see RoverController / the Take-over button).
        tu = getattr(d, "target_uncertainty", None)
        thr = getattr(d, "intervention_sigma_threshold", 0.0)
        if tu is None or thr <= 0:
            tgt_text, tgt_color = "--", PLOT_TELEMETRY_TEXT
        elif tu >= 90.0:
            tgt_text, tgt_color = "unknown", "#f87171"
        else:
            tgt_text, tgt_color = f"{tu:.1f}/{thr:.1f} m", self._unc_color(tu, thr)
        if getattr(d, "intervention_active", False):
            tgt_text, tgt_color = f"HUMAN ({tgt_text})", "#f87171"
        self.target_uncertainty_label.configure(text=tgt_text, text_color=tgt_color)

        # exp.md Sec 4 "Telemetry" -- only the operational values it calls
        # out (position, heading, velocity, yaw rate, nav state);
        # model-internal/CBF debugging detail stays in the trajectory/
        # belief bar's own HUD instead of this panel.
        if d.pose is not None:
            self.telemetry_position_label.configure(
                text=f"{d.pose.x:.1f}, {d.pose.z:.1f} m"
            )
            self.telemetry_heading_label.configure(
                text=f"{math.degrees(d.pose.yaw):.0f}°"
            )
        else:
            self.telemetry_position_label.configure(text="--")
            self.telemetry_heading_label.configure(text="--")

        act = d.action
        self.telemetry_velocity_label.configure(text=f"{act.v_fwd:.2f} m/s")
        self.telemetry_yaw_rate_label.configure(text=f"{act.yaw_rate:+.2f} rad/s")

        nav_state = d.mode.upper().replace("_", " ")
        if d.goal_reached:
            nav_state += "  ·  GOAL REACHED"
        self.telemetry_nav_state_label.configure(
            text=f"Nav state: {nav_state}",
            text_color=TOP_STATUS_COLORS["GOAL REACHED"]
            if d.goal_reached
            else "#e5e7eb",
        )

    def _sync_top_status(self, d, goal_status: Optional[str], alive: bool) -> None:
        state, subtext, color = self._derive_top_status(d, goal_status, alive)
        self.top_status_label.configure(text=state.title(), text_color=color)
        # REACH P9: prefer the freshest human ack over the derived subtext
        # so a just-pressed control's acknowledgment always wins on the
        # same frame it fires, before the controller's own state catches up.
        ack_live = self._ack_text if time.time() < self._ack_until else ""
        self.top_status_detail_label.configure(text=ack_live or subtext)

    def _derive_alerts(self, d, alive: bool) -> list[tuple[str, str]]:
        # "Status & alerts" feed (top bar, next to the primary status
        # readout) -- every signal folded in here already exists on the
        # controller snapshot or in this GUI's own state, it just had no
        # single place to surface consistently before (CBF avoidance/
        # hard-brake only ever showed up as a jitter in the trajectory
        # plot; low belief confidence only as a color on a small stat cell;
        # the emergency latch only on the button itself). Most severe first
        # (STATUS_LEVEL_ORDER); the handful of conditions checked below
        # bound the list's length naturally, no cap needed. _sync_alerts_
        # panel only ever displays alerts[0] plus a count of the rest --
        # the fixed-height top bar has no room for a full list.
        alerts: list[tuple[str, str]] = []
        if getattr(d, "intervention_active", False):
            alerts.append(
                ("critical", "HUMAN ACTION NEEDED")
            )
        if self._estop_active:
            alerts.append(("critical", "emergency stop active — driving locked"))
        if not alive:
            alerts.append(("critical", "controller thread died — see console"))
        if d.error_text:
            alerts.append(("critical", d.error_text))
        cbf = d.cbf_info or {}
        if cbf.get("hard_gate_fired"):
            alerts.append(("warning", "CBF hard brake — collision-radius breach"))
        elif cbf.get("orbiting") or cbf.get("blocked"):
            alerts.append(("caution", "avoiding obstacle"))
        if (
            d.uncertainty_enabled
            and not d.uncertainty_halted
            and d.uncertainty_threshold > 0
            and d.uncertainty_value / d.uncertainty_threshold >= 0.8
        ):
            alerts.append(("caution", "goal belief confidence low"))
        _tu = getattr(d, "target_uncertainty", None)
        _thr = getattr(d, "intervention_sigma_threshold", 0.0)
        if (
            _tu is not None
            and _thr > 0
            and _tu / _thr >= 0.7
            and not getattr(d, "intervention_active", False)
        ):
            alerts.append(("caution", "target position uncertain — human help may be needed"))
        if d.click_status:
            alerts.append(("info", d.click_status))
        alerts.sort(key=lambda a: STATUS_LEVEL_ORDER[a[0]])
        return alerts

    @staticmethod
    def _fit_single_line(font: tkfont.Font, text: str, max_width: int) -> str:
        # Binary-search the longest prefix (+ ellipsis) that still measures
        # within max_width in actual rendered pixels -- auto-fits to
        # whatever width the column currently has (window resizes, ratio
        # changes) rather than a fixed character-count guess, which would
        # either clip a fixed-width font's text early or overflow a
        # proportional one.
        if max_width <= 0:
            return ""
        if font.measure(text) <= max_width:
            return text
        ellipsis = "…"
        if font.measure(ellipsis) > max_width:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.measure(text[:mid] + ellipsis) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return (text[:lo] + ellipsis) if lo > 0 else ellipsis

    def _sync_alerts_panel(self, d, alive: bool) -> None:
        alerts = self._derive_alerts(d, alive)
        self._last_alerts = alerts
        if self._alerts_tooltip_visible:
            self._populate_alerts_tooltip(alerts)
        if alerts:
            level, text = alerts[0]
            color = STATUS_LEVEL_COLORS[level]
            extra = len(alerts) - 1
            suffix = f"  (+{extra} more)" if extra > 0 else ""
        else:
            color = STATUS_LEVEL_COLORS["nominal"]
            text, suffix = "nominal — no active warnings", ""

        # Reserve the bullet + "(+N more)" suffix's actual measured width
        # first -- those never truncate -- then fit only the message text
        # into whatever's left, so the count is never the part that gets
        # cut off. Measured against body_bold (matching the label's actual
        # font, see its construction) -- bold glyphs measure wider than
        # regular, so measuring against the wrong weight would under-fit
        # and clip a character or two early.
        font = self.fonts["body_bold"]
        bullet = "●  "
        avail = max(0, self._alerts_col_width - 28)  # padx=14 each side
        reserve = font.measure(bullet) + font.measure(suffix)
        message = self._fit_single_line(font, text, max(0, avail - reserve))
        self.alerts_message_label.configure(
            text=f"{bullet}{message}{suffix}", text_color=color
        )

    _ALERTS_HOVER_HIDE_DELAY_MS = 150

    def _on_alerts_hover_enter(self, _event=None) -> None:
        # Cancels a pending hide from a moment ago -- without this, moving
        # the mouse from alerts_col onto one of its own child labels (or
        # from the trigger area down into the popup itself) registers as a
        # Leave immediately followed by an Enter, and an un-cancelled hide
        # timer would close the popup out from under the pointer mid-hover.
        if self._alerts_hide_job is not None:
            self.root.after_cancel(self._alerts_hide_job)
            self._alerts_hide_job = None
        self._show_alerts_tooltip()

    def _on_alerts_hover_leave(self, _event=None) -> None:
        if self._alerts_hide_job is not None:
            self.root.after_cancel(self._alerts_hide_job)
        self._alerts_hide_job = self.root.after(
            self._ALERTS_HOVER_HIDE_DELAY_MS, self._hide_alerts_tooltip
        )

    def _show_alerts_tooltip(self) -> None:
        x = self.alerts_col.winfo_rootx()
        y = self.alerts_col.winfo_rooty() + self.alerts_col.winfo_height()
        self._alerts_tooltip.geometry(f"+{x}+{y}")
        self._alerts_tooltip.deiconify()
        self._alerts_tooltip.lift()
        self._alerts_tooltip_visible = True
        # Paint immediately from the last tick's alerts instead of waiting
        # up to REFRESH_MS for the next refresh() -- no blank-then-populate
        # flash on hover.
        self._populate_alerts_tooltip(self._last_alerts)

    def _hide_alerts_tooltip(self) -> None:
        self._alerts_hide_job = None
        self._alerts_tooltip.withdraw()
        self._alerts_tooltip_visible = False

    def _populate_alerts_tooltip(self, alerts: list[tuple[str, str]]) -> None:
        # Unlike the compact bar (worst alert + a count), the hover popup
        # is the "full detail" view -- every active alert gets its own row,
        # full text, no truncation (wraplength wraps instead of cutting).
        if alerts:
            self._alerts_tooltip_nominal_label.grid_remove()
        else:
            self._alerts_tooltip_nominal_label.grid()
        for i, lbl in enumerate(self._alerts_tooltip_rows):
            if i < len(alerts):
                level, text = alerts[i]
                lbl.configure(text=f"●  {text}", text_color=STATUS_LEVEL_COLORS[level])
                lbl.grid()
            else:
                lbl.grid_remove()

    def refresh(self) -> None:
        if self.closed:
            return
        d = self.controller.snapshot()

        # Live view: the controller's annotated frame when it is fresh, else
        # (controller busy) the newest raw camera frame, so the video never
        # freezes behind a slow tick. Real rover only; the sim has no
        # separate camera stream.
        live = None
        if time.monotonic() - getattr(d, "vis_rgb_time", 0.0) > LIVE_STALE_S:
            live = self.controller.latest_camera_rgb()
        if live is not None:
            self._draw_camera(live)
        elif d.vis_rgb is not None:
            self._draw_camera(d.vis_rgb)

        if (
            d.vis_rgb_topdown is not None
            and self._topdown_canvas is not None
            and self._topdown_window.winfo_viewable()
        ):
            self._draw_topdown(d.vis_rgb_topdown)

        self._draw_plot(d)
        self._sync_mission_review_panel(d)
        self._sync_seg_panel(d)
        self._sync_click_panel()
        self._sync_uncertainty_panel(d)
        self.mission_status_label.configure(text=d.mission_status)

        goal_status = self._derive_goal_status(d)
        self._sync_mission_panel(d, goal_status)
        self._sync_rover_status_panel(d)

        alive = self.controller.is_alive()
        self._sync_top_status(d, goal_status, alive)
        self._sync_alerts_panel(d, alive)
        if not alive and not self._alive_label_visible:
            self.alive_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
            self._alive_label_visible = True
        elif alive and self._alive_label_visible:
            self.alive_label.grid_forget()
            self._alive_label_visible = False

        self.root.after(REFRESH_MS, self.refresh)

    def _sync_mission_review_panel(self, d) -> None:
        mission = d.mission_review
        pending = mission is not None
        if pending and not self._mission_review_panel_visible:
            self.mission_review_panel.grid(
                row=self._mission_review_row,
                column=0,
                sticky="ew",
                pady=(SECTION_GAP, 0),
            )
            self._mission_review_panel_visible = True
            self._scroll_sidebar_to_bottom()
        elif not pending and self._mission_review_panel_visible:
            self.mission_review_panel.grid_forget()
            self._mission_review_panel_visible = False
            self._scroll_sidebar_to_top()
        if pending:
            self.mission_review_desc_label.configure(text=f'"{mission.instruction}"')
            self.mission_review_steps_label.configure(
                text=self._mission_review_steps_text(mission)
            )

    def _sync_seg_panel(self, d) -> None:
        in_review = d.mode == MODE_REVIEW_SEGMENTATION
        if in_review and not self._seg_panel_visible:
            self.seg_panel.grid(
                row=self._seg_row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            self._seg_panel_visible = True
            self._scroll_sidebar_to_bottom()
        elif not in_review and self._seg_panel_visible:
            self.seg_panel.grid_forget()
            self._seg_panel_visible = False
            self._scroll_sidebar_to_top()
        if in_review:
            self.seg_desc_label.configure(text=d.status_text)

    def _sync_click_panel(self) -> None:
        pending = self._pending_click_norm is not None
        if pending and not self._click_panel_visible:
            self.click_panel.grid(
                row=self._click_row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            self._click_panel_visible = True
            self._scroll_sidebar_to_bottom()
        elif not pending and self._click_panel_visible:
            self.click_panel.grid_forget()
            self._click_panel_visible = False
            self._scroll_sidebar_to_top()

    def _sync_uncertainty_panel(self, d) -> None:
        if not d.uncertainty_halted:
            if self._uncertainty_panel_visible:
                self.uncertainty_panel.grid_forget()
                self._uncertainty_panel_visible = False
                self._scroll_sidebar_to_top()
            return

        if not self._uncertainty_panel_visible:
            self.uncertainty_panel.grid(
                row=self._uncertainty_row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            self._uncertainty_panel_visible = True
            self._scroll_sidebar_to_bottom()

        in_flight = d.uncertainty_request_in_flight
        button_state = "disabled" if in_flight else "normal"
        for btn in self.uncertainty_buttons.values():
            btn.configure(state=button_state)
        self.uncertainty_retry_button.configure(state=button_state)

        if in_flight:
            self.uncertainty_panel.configure(border_color="#f59e0b")
            self.uncertainty_title.configure(
                text="Requesting VLM sweep", text_color="#f59e0b"
            )
        else:
            self.uncertainty_panel.configure(border_color="#38bdf8")
            self.uncertainty_title.configure(
                text="Uncertainty halt — choose a heading", text_color="#38bdf8"
            )
        self.uncertainty_desc_label.configure(
            text=d.uncertainty_line or "Waiting for VLM sweep description"
        )

    def _draw_camera(self, vis_rgb) -> None:
        # Letterbox the (possibly non-square) frame into whatever size the
        # hero canvas currently is, so the view actually scales with the
        # window instead of sitting at a fixed pixel size.
        cw = self.cam_canvas.winfo_width()
        ch = self.cam_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        img = Image.fromarray(vis_rgb).convert("RGB")
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((dw, dh))
        if self._pending_click_norm is not None:
            img = self._draw_pending_click(img, dw, dh)
        self._cam_photo = ImageTk.PhotoImage(img)
        x0, y0 = (cw - dw) // 2, (ch - dh) // 2
        self.cam_canvas.delete("frame")
        self.cam_canvas.create_image(
            x0, y0, anchor="nw", image=self._cam_photo, tags="frame"
        )
        self._cam_img_box = (x0, y0, x0 + dw, y0 + dh)

    def _draw_pending_click(self, img: Image.Image, dw: int, dh: int) -> Image.Image:
        # Not-yet-confirmed marker at the last click, cyan to read as distinct
        # from the gold confirmed-goal marker the controller reprojects into
        # vis_rgb every frame once a click is confirmed (rover_controller's
        # draw_point_marker).
        nx, ny = self._pending_click_norm
        x, y = nx * dw, ny * dh
        r = max(6, min(dw, dh) // 48)
        draw = ImageDraw.Draw(img)
        draw.line([(x - r, y), (x + r, y)], fill="#00e5ff", width=2)
        draw.line([(x, y - r), (x, y + r)], fill="#00e5ff", width=2)
        draw.ellipse([x - r, y - r, x + r, y + r], outline="#00e5ff", width=2)
        return img

    def _draw_plot(self, d) -> None:
        # exp.md Sec 5: a camera-width, fixed-height bottom bar now (not a
        # square panel sized off the sidebar) -- scale is derived from the
        # canvas's actual current size every frame instead of a cached
        # square _plot_size, and the drawing is centered horizontally so
        # extra width just widens the visible left/right margin.
        self.plot.delete("all")
        W = max(2, self.plot.winfo_width())
        H = max(2, self.plot.winfo_height())
        R = self.PLOT_RANGE
        scale = min(W / (2 * R), 0.92 * H / R)
        cx = W / 2
        baseline_y = H - 20

        def to_px(
            forward, left
        ):  # body frame (fwd, left) -> canvas (rover at bottom-center, facing up)
            return cx - left * scale, baseline_y - forward * scale

        self.plot.create_line(0, baseline_y, W, baseline_y, fill=PLOT_AXIS)
        self.plot.create_oval(
            cx - 5, baseline_y - 5, cx + 5, baseline_y + 5, fill=PLOT_ROVER, outline=""
        )  # rover

        if d.obstacle_point is not None:
            ox, oy = to_px(d.obstacle_point[0], d.obstacle_point[1])
            self.plot.create_oval(
                ox - 5, oy - 5, ox + 5, oy + 5, outline=PLOT_OBSTACLE, width=2
            )

        if d.trajectory is not None and len(d.trajectory) > 1:
            pts = [to_px(float(p[0]), float(p[1])) for p in d.trajectory]
            self.plot.create_line(
                *[c for xy in pts for c in xy], fill=PLOT_TRAJECTORY, width=3
            )

        if d.belief_g is not None:
            gx, gy = to_px(d.belief_g[0], d.belief_g[1])
            # exp.md Sec 5 "Belief": the goal marker's own dispersion ring
            # grows with uncertainty_value (clamped at uncertainty_threshold)
            # so high-confidence vs. increasing vs. high uncertainty reads
            # spatially here, not only as the Mission panel's HIGH/LOW
            # CONFIDENCE text.
            if d.uncertainty_enabled and d.uncertainty_threshold > 0:
                ratio = min(1.0, d.uncertainty_value / d.uncertainty_threshold)
            else:
                ratio = 0.0
            ring_r_px = max(6.0, (0.25 + 1.5 * ratio) * scale)
            ring_color = (
                self._unc_color(d.uncertainty_value, d.uncertainty_threshold)
                if d.uncertainty_enabled
                else PLOT_GOAL
            )
            self.plot.create_oval(
                gx - ring_r_px,
                gy - ring_r_px,
                gx + ring_r_px,
                gy + ring_r_px,
                outline=ring_color,
                width=2,
            )
            self.plot.create_text(
                gx,
                gy,
                text="*",
                fill=PLOT_GOAL,
                font=(self.display_font_family, TYPE_SIZE["plot_marker"], "bold"),
            )

    def tick_forever(self) -> None:
        self.root.mainloop()


def build_controller(args: argparse.Namespace) -> RoverController:
    navdp_upstream_ckpt = args.navdp_upstream_ckpt or _default_navdp_upstream_ckpt()
    if not navdp_upstream_ckpt:
        raise ValueError(
            "--navdp-upstream-ckpt is required (no checkpoint found at the default "
            "navdp/navdp-cross-modal.ckpt either)"
        )
    scene_path, heightmap_path, spawn_flags, terrain_glb = (
        args.scene_path, args.heightmap_path or None, not args.no_spawn_flags, None
    )
    if args.yard_glb:
        # A yard GLB with its own ground mesh, boulders and flags
        # (e.g. marsyard_with_rocks.glb): Habitat needs a Y-up stage config,
        # heights come from the GLB, and its baked-in flags replace ours.
        from sam_vla.env.glb_terrain import write_stage_config

        terrain_glb = str(Path(args.yard_glb).expanduser().resolve())
        scene_path = write_stage_config(terrain_glb, str(REPO_ROOT / "runs"))
        heightmap_path, spawn_flags = None, False
    real_endpoint = None
    if args.real_rover:
        if args.zenoh_endpoint:
            real_endpoint = args.zenoh_endpoint
        elif args.pi_ip:
            real_endpoint = f"tcp/{args.pi_ip}:7447"
        else:
            raise ValueError("--real-rover needs --pi-ip or --zenoh-endpoint")
        # real-hardware speed caps (same as every validated real-rover driver)
        args.max_linear = min(args.max_linear, 0.15)
        args.max_angular = min(args.max_angular, 0.3)
    return RoverController(
        real_rover_endpoint=real_endpoint,
        real_camera_height=args.camera_height,
        real_yaw_rate_scale=args.yaw_rate_scale,
        real_linear_scale=args.linear_scale,
        scene_path=scene_path,
        heightmap_path=heightmap_path,
        terrain_glb=terrain_glb,
        navdp_upstream_ckpt=navdp_upstream_ckpt,
        navdp_upstream_root=args.navdp_upstream_root,
        navdp_root=args.navdp_root,
        scene_dataset_config=args.scene_dataset_config,
        navmesh_path=args.navmesh_path,
        spawn_flags=spawn_flags,
        gen_home_base=args.gen_home_base,
        scenery=(
            []
            if args.no_scenery
            else [
                dict(
                    glb=args.scenery_glb,
                    x=args.scenery_x,
                    z=args.scenery_z,
                    yaw=math.radians(args.scenery_yaw),
                    scale=args.scenery_scale,
                )
            ]
        ),
        rock_field_path=args.rock_field,
        flag_seed=args.flag_seed,
        num_flags=args.num_flags,
        flag_min_spacing=args.flag_min_spacing,
        flag_boundary_margin=args.flag_boundary_margin,
        enable_topdown_viz=args.tdviz,
        rover_marker_scale=args.rover_marker_scale,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw,
        dt=args.dt,
        hz=args.hz,
        cbf_enabled=not args.no_cbf,
        cbf_d_safe=args.cbf_d_safe,
        cbf_gamma=args.cbf_gamma,
        cbf_deadzone=args.cbf_deadzone,
        max_forward_speed=args.max_linear,
        action_smoothing_alpha=args.action_smoothing_alpha,
        max_yaw_rate=args.max_angular,
        navdp_upstream_port=args.navdp_upstream_port,
        navdp_upstream_lookahead=args.navdp_upstream_lookahead,
        navdp_upstream_replan_every=args.navdp_upstream_replan_every,
        navdp_upstream_server_variant=args.navdp_upstream_server_variant,
        navdp_upstream_planner_mode=args.navdp_upstream_planner_mode,
        tube_guided=args.tube_guided,
        intervention_sigma=args.intervention_sigma,
        tube_server=args.tube_server,
        tube_robot_radius=args.tube_robot_radius,
        qwen35_port=args.qwen35_port,
        vlm_model_id=args.vlm_model_id,
        vlm_prompt_style=args.vlm_prompt_style,
        vlm_replan_every=args.vlm_replan_every,
        vlm_num_history=args.vlm_num_history,
        random_goal_bearing_deg=args.random_goal_bearing_deg,
        random_goal_dist_range=(args.random_goal_min_dist, args.random_goal_max_dist),
        go_direction_distance_m=args.go_direction_distance,
        seg_backend=args.seg_backend,
        seg_checkpoint=args.seg_checkpoint,
        seg_overlay=args.seg_overlay,
        annotations_dir=args.annotations_dir,
        annotation_categories=args.annotation_categories,
        dino_model_id=args.dino_model_id,
        dino_device=args.dino_device,
        dino_box_threshold=args.dino_box_threshold,
        dino_text_threshold=args.dino_text_threshold,
        use_obs_detect=args.use_obs_detect,
        mission_belief_sweep_every=args.mission_belief_sweep_every,
        mission_belief_cov_growth=args.mission_belief_cov_growth,
        mission_belief_cov_growth_rate=args.mission_belief_cov_growth_rate,
        uncertainty_enabled=not args.no_uncertainty_halt,
        uncertainty_cov_threshold=args.cov_threshold,
        uncertainty_cov_growth=args.cov_growth,
        uncertainty_cov_growth_rate=args.cov_growth_rate,
        point_unc_var_per_m=args.point_unc_var_per_m,
        point_unc_var_per_rad=args.point_unc_var_per_rad,
        uncertainty_search_dist=args.uncertainty_search_dist,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scene-path", default=str(REPO_ROOT / "assets" / "marsyard2022.glb")
    )
    ap.add_argument(
        "--yard-glb",
        default=None,
        help="use a yard GLB that carries its own ground mesh, boulders and flags "
        "(e.g. assets/marsyard_with_rocks.glb) instead of --scene-path/--heightmap-path",
    )
    ap.add_argument(
        "--heightmap-path",
        default=str(REPO_ROOT / "marsyard2022_terrain_hm_1025.tif"),
        help="DEM heightmap for ground-height sampling. Pass \"\" to opt out and use "
        "navmesh-based height sampling instead (see --scene-dataset-config/"
        "--navmesh-path) -- for scenes with no heightmap, e.g. HM3D",
    )
    ap.add_argument(
        "--scene-dataset-config",
        default=None,
        help="scene_dataset_config.json for --scene-path (needed for HM3D-style "
        "scenes to get their up/front stage rotation right; marsyard2022.glb "
        "doesn't need one, leave unset)",
    )
    ap.add_argument(
        "--navmesh-path",
        default=None,
        help="explicit .navmesh path, only needed with --heightmap-path \"\" if the "
        "scene's own sibling .navmesh doesn't auto-load",
    )
    ap.add_argument(
        "--no-spawn-flags",
        action="store_true",
        help="skip placing the decorative flag markers -- for non-outdoor-course "
        "scenes (e.g. HM3D) where they don't make sense",
    )
    ap.add_argument(
        "--rock-field", default=None, help="rock_field.json manifest (optional)"
    )
    ap.add_argument(
        "--gen-home-base",
        action="store_true",
        help="spawn the ghost home-base marker (a fixed, non-collidable blue "
        "cuboid 5m from the rover's start pose). Off by default",
    )
    ap.add_argument(
        "--scenery-glb",
        default=str(REPO_ROOT / "assets" / "25042_Perseverance.glb"),
        help="static decoration model placed in the scene (default: the "
        "Perseverance rover, true 1:1 scale, 2.7 x 2.3 x 4.5 m)",
    )
    ap.add_argument("--scenery-x", type=float, default=-9.5, help="world x (m)")
    ap.add_argument("--scenery-z", type=float, default=-1.0, help="world z (m)")
    ap.add_argument("--scenery-yaw", type=float, default=35.0, help="degrees")
    ap.add_argument("--scenery-scale", type=float, default=1.0)
    ap.add_argument(
        "--no-scenery", action="store_true", help="don't place the scenery model"
    )
    ap.add_argument(
        "--flag-seed",
        type=int,
        default=None,
        help="randomize flag-marker placement (assets/flags/*.glb) seeded with this "
        "value -- same seed gives the same layout every run; unset (default) places "
        "a fixed 4-flag diamond around the rover's spawn point instead",
    )
    ap.add_argument(
        "--num-flags",
        type=int,
        default=6,
        help="flags to place when --flag-seed is set (unseeded default is always 4)",
    )
    ap.add_argument(
        "--flag-min-spacing",
        type=float,
        default=1.5,
        help="minimum meters between placed flags",
    )
    ap.add_argument(
        "--flag-boundary-margin",
        type=float,
        default=2.0, 
        help="keep placed flags this many meters clear of the scene bounds",
    )
    ap.add_argument(
        "--tdviz",
        action="store_true",
        help="open a second, fixed bird's-eye-view window (a separate high-res "
        "habitat-sim camera framing the four flags from directly above) for "
        "video recording -- human viz only, plays no part in navigation/CBF/VLM. "
        "Off by default",
    )
    ap.add_argument(
        "--rover-marker-scale",
        type=float,
        default=0.5,
        help="scale multiplier for the assets/perseverance_mars_rover.glb marker "
        "shown at the rover's live position in --top-down-viz (1.0 = the asset's "
        "native, real-world meter scale). Only affects the top-down window -- "
        "the marker never appears in the main camera view. Turn down if it reads "
        "too large against the course",
    )
    ap.add_argument(
        "--navdp-upstream-ckpt",
        default=None,
        help="Path to the upstream NavDP .ckpt (default: navdp/navdp-cross-modal.ckpt if present)",
    )
    ap.add_argument(
        "--navdp-upstream-root",
        default=None,
        help="Path to the vendored InternRobotics/NavDP checkout (default: $NAVDP_UPSTREAM_ROOT)",
    )
    ap.add_argument("--navdp-upstream-port", type=int, default=None)
    ap.add_argument("--navdp-upstream-lookahead", type=int, default=3)
    ap.add_argument("--navdp-upstream-replan-every", type=int, default=1)
    ap.add_argument(
        "--navdp-upstream-server-variant",
        choices=["navdp", "s2diff"],
        default="navdp",
        help="'navdp' spawns InternRobotics/NavDP's own baselines/navdp/navdp_server.py; "
        "'s2diff' spawns this project's obstacle-guided navdp_s2diff_server.py fork instead "
        "(same checkpoint, see navdp_upstream_server_manager.py's docstring)",
    )
    ap.add_argument(
        "--navdp-upstream-planner-mode",
        choices=["pure-navdp", "s2diff", "gradient"],
        default="s2diff",
        help="server_variant=s2diff only: which guidance mode navdp_s2diff_server.py runs",
    )
    ap.add_argument(
        "--tube-guided",
        action="store_true",
        help="drive with NavDP score-guided by the SAM2-segmentation BEV obstacle map "
        "(navdp_tube 'ours', best.pth) instead of plain upstream NavDP",
    )
    ap.add_argument(
        "--intervention-sigma",
        type=float,
        default=6.0,
        help="hand control to the human when the VLM target is out of view and its "
        "belief sigma (m) reaches this (also after repeated empty scans)",
    )
    ap.add_argument("--tube-server", default="http://127.0.0.1:8890")
    ap.add_argument("--tube-robot-radius", type=float, default=0.45)
    ap.add_argument(
        "--navdp-root",
        default=None,
        help="Path to this repo's own navdp/ package (default: ./navdp or $NAVDP_ROOT) -- only "
        "needed for the CBF safety layer's generic obstacle/avoidance math, unrelated to which "
        "driving policy is active",
    )
    ap.add_argument(
        "--vlm-instruction",
        action="append",
        default=None,
        dest="vlm_instructions",
        help="queue a navigation instruction for the Qwen3.5-9B point-goal navigator "
        "(nav.vlm_point_navigator.VlmPointNavigator, see rover_controller.py's "
        "'VLM navigation' docstring section) -- repeat for multiple, driven through in "
        "order. Starts VLM navigation automatically once the controller is running; "
        "unset (default) leaves VLM navigation off, same as every other run of this GUI. "
        "Mutually exclusive with --vlm-command -- if both are given, --vlm-command wins",
    )
    ap.add_argument(
        "--vlm-command",
        default=None,
        help='one free-text, possibly-compound instruction (e.g. "go straight until '
        'you reach the door and turn left and search for a water bottle"), split '
        "into --vlm-instruction's ordered queue automatically by Qwen "
        "(qwen_prompts.build_split_instruction_prompt) using the live first frame -- "
        "see RoverController.start_vlm_navigation_from_command. Alternative to passing "
        "--vlm-instruction yourself when the steps aren't already segmented",
    )
    ap.add_argument(
        "--vlm-replan-every",
        type=int,
        default=15,
        help="controller ticks between Qwen3.5-9B point-goal re-queries (default 15, "
        "~1.5s at --hz 10) -- NavdpUpstreamPolicy keeps driving toward the last point "
        "every tick in between, same two-speed design as InternVLA-N1's System-2/System-1",
    )
    ap.add_argument(
        "--vlm-num-history",
        type=int,
        default=8,
        help="rolling per-instruction history turns re-sent to Qwen3.5-9B each query "
        "(default 8, matching InternVLA-N1's num_history) -- cleared whenever the "
        "active instruction advances",
    )
    ap.add_argument(
        "--qwen35-port",
        type=int,
        default=None,
        help="port for the Qwen3.5-9B point-goal server (default: "
        "sam_vla.vlm.qwen35_config.QWEN35_SERVER_PORT), distinct from the existing "
        "Qwen2.5-VL-3B server's port -- only spawned if --vlm-instruction is used",
    )
    ap.add_argument(
        "--vlm-model-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF id of the point-goal VLM served on --qwen35-port (default: "
        "Qwen3-VL-8B-Instruct; pass Qwen/Qwen3.5-9B for the original model)",
    )
    ap.add_argument(
        "--vlm-prompt-style",
        choices=["reasoning", "default"],
        default="reasoning",
        help="'reasoning' = compact reasoning-first prompt (qwen35_prompts."
        "build_reasoning_goal_prompt); 'default' = original long prompt",
    )
    ap.add_argument(
        "--start-x", type=float, default=-5.5
    )  # 7.1, 7.6, 2.2, 7.5, # 1.7, # -5.5, 15.7
    ap.add_argument(
        "--start-z", type=float, default=2.2
    )  # 7.7, 7.1, -1.9, 6.9, # 0.7 # 2.2, 1.5
    ap.add_argument(
        "--start-yaw", type=float, default=86.0, help="degrees"
    )  # 34, 41, 161, 34, # 58 # 86, 86
    ap.add_argument(
        "--real-rover",
        action="store_true",
        help="drive the PHYSICAL rover over its zenoh bridge (camera+depth in, "
        "cmd_vel out; see nav/real_rover_env.py) instead of the Habitat sim",
    )
    ap.add_argument(
        "--pi-ip", default=None,
        help="--real-rover: the rover Pi's IP (zenoh endpoint tcp/<ip>:7447)",
    )
    ap.add_argument(
        "--zenoh-endpoint", default=None,
        help="--real-rover: full zenoh endpoint, overrides --pi-ip",
    )
    ap.add_argument(
        "--camera-height", type=float, default=0.45,
        help="--real-rover: camera height above the floor (m)",
    )
    ap.add_argument(
        "--yaw-rate-scale", type=float, default=1.0,
        help="--real-rover: the rover's real turn rate / commanded turn rate "
        "(calibration for the command-based heading; 2.0 if a 90 deg turn ends at 180)",
    )
    ap.add_argument(
        "--linear-scale", type=float, default=1.0,
        help="--real-rover: the rover's real forward speed / commanded speed",
    )
    ap.add_argument(
        "--bev-display",
        action="store_true",
        help="show LemNAV's monocular BEV map (Depth Anything + floor fit) in its own "
        "window -- display only, never used for planning (nav/bev_display.py)",
    )
    ap.add_argument(
        "--bev-cam-height", type=float, default=None,
        help="--bev-display: camera height above the floor (m); default --camera-height",
    )
    ap.add_argument(
        "--bev-cam-forward", type=float, default=0.07,
        help="--bev-display: camera offset ahead of the rover centre (m)",
    )
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--hz", type=float, default=10.0, help="controller tick rate")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument(
        "--action-smoothing-alpha",
        type=float,
        default=0.35,
        help="EMA factor blending each tick's (v_fwd, v_lat, yaw_rate) action into "
        "a running average before driving -- lower is smoother/laggier, higher is "
        "more responsive/jerkier, 1.0 disables smoothing entirely. Off during manual "
        "driving regardless of this value",
    )
    ap.add_argument(
        "--no-cbf", action="store_true", help="disable CBF cone-mode obstacle avoidance"
    )
    ap.add_argument("--cbf-d-safe", type=float, default=0.75)
    ap.add_argument("--cbf-gamma", type=float, default=0.3)
    ap.add_argument("--cbf-deadzone", type=float, default=0.6)
    ap.add_argument("--random-goal-bearing-deg", type=float, default=60.0)
    ap.add_argument("--random-goal-min-dist", type=float, default=4.0)
    ap.add_argument("--random-goal-max-dist", type=float, default=8.0)
    ap.add_argument(
        "--seg-backend",
        choices=["lora", "legacy"],
        default="lora",
        help="segmentation checkpoint used by 'Segment' goal resolution: 'lora' (default) is "
        "sam_lora_runs/exp10/best, LoRA-finetuned on mesh_tight_bound2-overlay frames; "
        "'legacy' is the original single-checkpoint model (best_model.pth)",
    )
    ap.add_argument(
        "--seg-checkpoint",
        default=None,
        help="override the checkpoint path (legacy: a .pth file) or dir (lora: a "
        "finetune_sam2_lora.py out-dir, e.g. sam_lora_runs/exp10/best) for --seg-backend",
    )
    ap.add_argument(
        "--seg-overlay",
        choices=["mesh", "none"],
        default="mesh",
        help="'mesh' (default) composites --annotations-dir's hull meshes into a separate "
        "frame fed only to the segmentation model, matching what the default lora checkpoint "
        "was trained on -- never shown in the GUI or sent to the goal-selection VLM. "
        "'none' segments the plain camera frame",
    )
    ap.add_argument(
        "--annotations-dir",
        default=str(REPO_ROOT / "annotations" / "mesh_tight_bound2"),
        help="hull-mesh annotation dir used by --seg-overlay=mesh (default: "
        "annotations/mesh_tight_bound2, the dataset sam_lora_runs/exp10 was trained on). "
        "Pass \"\" to disable -- these are marsyard-specific, e.g. not applicable to HM3D",
    )
    ap.add_argument(
        "--annotation-categories",
        nargs="+",
        default=None,
        help="restrict --seg-overlay=mesh to these hull categories (default: all)",
    )
    ap.add_argument(
        "--dino-model-id",
        default=dino_grounding_resolver.DEFAULT_DINO_MODEL_ID,
        help="GroundingDINO checkpoint for open-vocabulary target grounding "
        "('Ground Target' / a Command-panel GO_TO/FIND step) -- HF "
        "AutoModelForZeroShotObjectDetection id, lazily loaded on first use",
    )
    ap.add_argument(
        "--dino-device",
        default=dino_grounding_resolver.DEFAULT_DINO_DEVICE,
        help="device GroundingDINO loads onto (default cuda)",
    )
    ap.add_argument(
        "--dino-box-threshold",
        type=float,
        default=dino_grounding_resolver.DEFAULT_BOX_THRESHOLD,
    )
    ap.add_argument(
        "--dino-text-threshold",
        type=float,
        default=dino_grounding_resolver.DEFAULT_TEXT_THRESHOLD,
    )
    ap.add_argument(
        "--use-obs-detect",
        action="store_true",
        help="run SAM2 obstacle detection alongside GroundingDINO target grounding "
        "('Ground Target' / a Command-panel GO_TO/FIND step). Off by default -- that "
        "checkpoint currently predicts artifact slivers, not real rock geometry (see "
        "CLAUDE.md's 'Annotation masks are thin silhouette slivers' known issue), so "
        "it burns time for obstacle boxes that get dropped or mislead the CBF anyway",
    )
    ap.add_argument(
        "--mission-belief-sweep-every",
        type=int,
        default=15,
        help="while a Command-panel Mission is driving, re-check the current frame "
        "against every remaining GO_TO/FIND target text every this-many ticks, "
        "seeding/refreshing a persistent per-goal belief -- so a goal sighted while "
        "driving toward an earlier one in the same instruction isn't forgotten by "
        "the time the mission reaches it (default 15; <= 0 disables the periodic "
        "sweep)",
    )
    ap.add_argument(
        "--mission-belief-cov-growth",
        type=float,
        default=0.0002,  # 0.00002, tested to reach back home without losing its grip over the reality
        help="per-tick belief-uncertainty growth for a goal sighted by the "
        "periodic mission sweep above but not currently being driven to -- kept "
        "far below --cov-growth (that rate is tuned for a goal briefly losing its "
        "mask for a couple seconds while actively tracked; reusing it here decayed "
        "a sweep-seeded belief past --cov-threshold within ~1s, defeating the "
        "sweep) so a sighting can survive the rest of the leg until the mission "
        "actually reaches that goal (default 0.0002)",
    )
    ap.add_argument(
        "--mission-belief-cov-growth-rate",
        type=float,
        default=0.0,
        help="accelerating-drift factor for --mission-belief-cov-growth (default "
        "0.0, i.e. flat per-tick growth unless overridden)",
    )
    ap.add_argument(
        "--no-uncertainty-halt",
        action="store_true",
        help="disable the uncertainty-halt heading-request prompt (vl_direction's "
        "'uncertainty' mode against the resolved goal's real BeliefGoalTracker -- see "
        "--cov-threshold/--cov-growth). Default on; only spins up its own Qwen VLM "
        "subprocess (distinct from --seg-backend's) once this halt actually fires",
    )
    ap.add_argument(
        "--cov-threshold",
        type=float,
        default=0.75,
        help="belief-uncertainty value (BeliefGoalTracker.uncertainty_value(), grows "
        "while a resolved goal mask stays unseen) at which MODE_RESOLVE driving halts "
        f"and a heading is requested (default {vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD})",
    )
    ap.add_argument(
        "--cov-growth",
        type=float,
        default=0.001,
        help="base per-tick belief-uncertainty growth while the goal mask is unseen "
        "(default 0.001); lower this to trigger the halt sooner for testing",
    )
    ap.add_argument(
        "--study",
        action="store_true",
        help="open the user-study window (nav/user_study.py): per-participant trials "
        "(language task / human-adjusted), prompt log, interventions, path length, time "
        "to goal, oracle path and a final report under nav/study_logs/",
    )
    ap.add_argument("--study-participant", default=None, help="pre-fill the participant ID")
    ap.add_argument(
        "--study-success-radius",
        type=float,
        default=3.0,
        help="a goal counts as reached within this distance (m) of it (default 3.0, "
        "the usual VLN success radius; the navigator itself stops at ~2.5 m)",
    )
    ap.add_argument(
        "--point-unc-var-per-m",
        type=float,
        default=0.03,
        help="point-goal current uncertainty: variance (m^2) added per metre driven "
        "while the target is out of view (default 0.03: ~0.75 after ~19 m straight)",
    )
    ap.add_argument(
        "--point-unc-var-per-rad",
        type=float,
        default=0.0005,
        help="point-goal current uncertainty: variance added per radian turned, "
        "times range^2 to the target, while it is out of view (default 0.0005)",
    )
    ap.add_argument(
        "--cov-growth-rate",
        type=float,
        default=0.0,
        help="accelerating-drift factor: growth speeds up the longer the goal has been "
        "unseen (default 0.0, i.e. flat per-tick growth unless overridden)",
    )
    ap.add_argument(
        "--uncertainty-search-dist",
        type=float,
        default=4.0,
        help="meters driven along a human-submitted heading before falling back to the "
        "(still-uncertain) dead-reckoned belief and, if still unseen, halting again "
        "(default 4.0)",
    )
    ap.add_argument(
        "--go-direction-distance",
        type=float,
        default=4.0,
        help="meters driven forward by a mission's ADVANCE step -- 'go left'/'go right'/"
        "'go straight' style commands (turn if needed, then drive this far), as opposed "
        "to 'turn left'/'turn right' which only rotate in place (default 4.0)",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    controller = build_controller(args)
    controller.start()
    if args.vlm_command:
        controller.start_vlm_navigation(split_sequence(args.vlm_command))
    elif args.vlm_instructions:
        controller.start_vlm_navigation(args.vlm_instructions)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    app = NavGuiApp(root, controller)
    bev = None
    if args.bev_display:
        # LemNAV's monocular BEV map, display only (nav/bev_display.py)
        from nav.bev_display import BevMapDisplay, BevMapWindow

        bev = BevMapDisplay(
            controller.latest_rgb_pose, controller.hfov_deg,
            cam_height=args.bev_cam_height if args.bev_cam_height is not None else args.camera_height,
            cam_forward=args.bev_cam_forward,
        )
        bev.start()
        BevMapWindow(root, bev)
    study = None
    if args.study:
        # User-study recorder + experimenter window (nav/user_study.py)
        from nav.user_study import StudyRecorder, StudyWindow

        study = StudyRecorder(
            controller, split_fn=split_sequence,
            yard_glb=str(Path(args.yard_glb).expanduser().resolve()) if args.yard_glb else None,
            success_radius=args.study_success_radius,
        )
        app.command_listeners.append(study.log_prompt)
        StudyWindow(root, study, participant=args.study_participant)
    try:
        app.tick_forever()
    finally:
        if study is not None and study.dir is not None:
            if study.trial is not None:
                study.end_trial("GUI closed")
            study.write_report()
            print(f"[study] report: {study.dir / 'report.md'}", flush=True)
        if bev is not None:
            bev.stop()
        controller.shutdown()


if __name__ == "__main__":
    main()
