#!/usr/bin/env python3
"""
LEGACY BACKUP
Interactive control panel for nav.rover_controller.RoverController --
the in-house, single-process equivalent of Nav_new/MARS/launch_mars.sh's
DINO+NavDP GUI, driving the actual published NavDP model instead of this
repo's own custom S2DiT+NavDP model (see rover_controller.py's docstring).

Run via nav/launch_nav.sh, or directly:
    conda activate habitat
    cd mars-habitatsim
    python -m nav.gui [--scene-path ...] [--heightmap-path ...] [--cbf/--no-cbf] ...

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
from pathlib import Path
from typing import Optional

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from sam_vla.goal_resolution import dino_grounding_resolver
from vl_direction import config as vl_dir_config

from nav.mission import GoalKind
from nav.rover_controller import MODE_REVIEW_SEGMENTATION, MODE_TURN, RoverController

REPO_ROOT = Path(__file__).resolve().parent.parent
REFRESH_MS = 66  # ~15 Hz GUI repaint, independent of the controller's own hz

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
BTN_H = 46  # primary action button height (px)
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
}

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
# same convention as nav/goal_math.heading_ahead_point -- 0=front,
# positive=clockwise) bound to the same physical key layout
# scripts/habitat_tests/kb_teleop_vl.py's uncertainty panel uses, redrawn
# independently here rather than imported (nav/ stays decoupled from that
# script). Grid position drives the heads-up panel's clickable button
# layout, keys drive the numpad keyboard shortcuts.
UNCERTAINTY_HEADING_KEYS = {
    "8": 0.0,
    "9": 45.0,
    "6": 90.0,
    "3": 135.0,
    "2": 180.0,
    "1": -135.0,
    "4": -90.0,
    "7": -45.0,
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


def _default_navdp_upstream_ckpt() -> Optional[str]:
    """Same repo-relative fallback sam_vla.run_navdp_rollout uses -- a plain
    invocation just works on a checkout that already has Phase 0's
    (next.md's Integration project) gitignored checkpoint in place."""
    candidate = REPO_ROOT / "navdp" / "navdp-cross-modal.ckpt"
    return str(candidate) if candidate.exists() else None


class NavGuiApp:
    SIDEBAR_MIN_W = 360
    # exp.md Sec 5: TRAJECTORY / BELIEF is a full-width bar below the
    # camera+sidebar body, not a square panel sized off the sidebar's own
    # width -- a fixed height, full window width.
    PLOT_BOTTOM_H = 200
    PLOT_RANGE = 6.0  # meters shown top-to-bottom of the body-frame plot

    def __init__(
        self,
        root: ctk.CTk,
        controller: RoverController,
        use_segmentation: bool = False,
    ):
        self.root = root
        self.controller = controller
        self.use_segmentation = use_segmentation
        self.closed = False
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
        self._ack_text = ""

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

        mode_font = ctk.CTkFont(size=17, weight="bold")

        # ===== TOP STATUS / CURRENT STATUS (exp.md Sec. 1) =============== #
        # Full-width, persistent, above everything else -- the one line the
        # operator should be able to read without inspecting the camera,
        # mission, or telemetry panels (REACH P9/P10: continuous, unmissable
        # feedback on what the rover is doing right now and why).
        top_status_bar = ctk.CTkFrame(root, border_width=2, border_color="#3f3f46")
        top_status_bar.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4)
        )
        top_status_bar.grid_columnconfigure(0, weight=1)
        self.top_status_label = ctk.CTkLabel(
            top_status_bar,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.top_status_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 0))
        self.top_status_detail_label = ctk.CTkLabel(
            top_status_bar,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color="#e5e7eb",
        )
        self.top_status_detail_label.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 8)
        )
        top_status_bar.bind(
            "<Configure>",
            lambda e: self.top_status_detail_label.configure(
                wraplength=max(200, e.width - 28)
            ),
        )
        self.alive_label = ctk.CTkLabel(
            top_status_bar,
            text="controller thread died -- see console",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
        )
        self._alive_label_visible = False

        # ---------------- camera hero (REACH P7: point-and-select) ------- #
        # Fills all left-column space; the frame is letterboxed into it on
        # every refresh so it tracks window size. Clicking the live view is
        # the primary, most intuitive way to designate a goal -- the direct
        # analog of REACH "point and select an object in the environment".
        cam_frame = ctk.CTkFrame(root)
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(4, 4))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_canvas = tk.Canvas(cam_frame, bg="#111111", highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 2))
        self.cam_canvas.bind("<Button-1>", self.on_cam_click)
        ctk.CTkLabel(
            cam_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#22d3ee",
        ).grid(row=1, column=0, pady=(0, 6))

        # ---------------- sidebar ---------------------------------------- #
        # REACH P6 (only essential info) + P5 (spacing): one vertical
        # column of clearly-labeled sections, top-anchored, generous gaps.
        # Scrollable so no control is ever clipped off a short monitor
        # (contextual review panels can push the stack past screen height).
        sidebar = ctk.CTkScrollableFrame(root, fg_color="transparent")
        sidebar.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(4, 4))
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
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="#9ca3af",
            ).pack(pady=(6, 0))
            value_label = ctk.CTkLabel(
                cell, text="--", font=ctk.CTkFont(size=13, weight="bold")
            )
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
            text="MISSION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        self.goal_list_label = ctk.CTkLabel(
            self.mission_panel,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=13),
        )
        self.goal_list_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        ctk.CTkLabel(
            self.mission_panel,
            text="CURRENT GOAL",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 0))
        self.current_goal_label = ctk.CTkLabel(
            self.mission_panel, text="--", anchor="w", justify="left", font=mode_font
        )
        self.current_goal_label.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))

        mission_stat_row = ctk.CTkFrame(self.mission_panel, fg_color="transparent")
        mission_stat_row.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 10))
        for i in range(3):
            mission_stat_row.grid_columnconfigure(i, weight=1)
        self.goal_distance_label = _stat_cell(mission_stat_row, 0, "DISTANCE")
        self.goal_belief_label = _stat_cell(mission_stat_row, 1, "GOAL BELIEF")
        self.goal_status_label = _stat_cell(mission_stat_row, 2, "STATUS")

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
            text="ROVER STATUS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        self.mode_label = ctk.CTkLabel(
            self.status_panel, text="", anchor="w", justify="left", font=mode_font
        )
        self.mode_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        rover_stat_row = ctk.CTkFrame(self.status_panel, fg_color="transparent")
        rover_stat_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        rover_stat_row.grid_columnconfigure(0, weight=1)
        rover_stat_row.grid_columnconfigure(1, weight=1)
        self.current_uncertainty_label = _stat_cell(
            rover_stat_row, 0, "CURRENT UNCERTAINTY"
        )
        self.home_uncertainty_label = _stat_cell(rover_stat_row, 1, "HOME UNCERTAINTY")

        # Telemetry: only the operational values exp.md calls out (position,
        # heading, velocity, yaw rate, nav state) -- model-internal/CBF
        # debugging detail stays out of this panel (see the trajectory/
        # belief bar's own HUD for that).
        self.telemetry_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12, family="Consolas"),
            text_color="#e5e7eb",
        )
        self.telemetry_label.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.status_panel.bind(
            "<Configure>",
            lambda e: self.telemetry_label.configure(wraplength=max(200, e.width - 20)),
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
        ctk.CTkLabel(
            command_panel, text="TASKWORK", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        self.command_entry = ctk.CTkEntry(
            command_panel,
            placeholder_text='"go to a flag then return to home"',
            height=38,
        )
        self.command_entry.grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 4))
        self.command_entry.bind("<Return>", lambda e: self.submit_command())
        # REACH P3: dispatch the ordered mission.
        ctk.CTkButton(
            command_panel, text="Send", width=72, height=38, command=self.submit_command
        ).grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(0, 4))
        self.mission_status_label = ctk.CTkLabel(
            command_panel,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#e5e7eb",
            wraplength=self.SIDEBAR_MIN_W - 20,
        )
        self.mission_status_label.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, BTN_GAP)
        )

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

        # ===== SEGMENTATION TRIGGER (REACH P1, --use-segmentation) ====== #
        # Ad-hoc SAM2+Qwen-salience "Segment" goal resolution
        # (RoverController.request_resolve() with no target text) -- gated
        # behind --use-segmentation since it's a manual override with no
        # other trigger in this console (mission GO_TO/FIND steps use
        # DINO grounding and auto-confirm instead, see submit_nav_command's
        # docstring). Firing this is the only thing that can ever put the
        # controller into MODE_REVIEW_SEGMENTATION here, so the "REVIEW
        # RESOLVED GOAL" contextual panel right below only ever becomes
        # reachable once this section is enabled.
        if self.use_segmentation:
            seg_trigger_panel = ctk.CTkFrame(sidebar)
            seg_trigger_panel.grid(
                row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            row += 1
            seg_trigger_panel.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                seg_trigger_panel,
                text="SEGMENTATION",
                font=ctk.CTkFont(size=12, weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
            ctk.CTkButton(
                seg_trigger_panel,
                text="Run Segmentation",
                height=BTN_H,
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
            text="REVIEW RESOLVED GOAL",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f59e0b",
        ).pack(pady=(10, 2))
        self.seg_desc_label = ctk.CTkLabel(
            self.seg_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
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
            command=self.confirm_segmentation,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: reject and re-resolve (same intent, new result).
        ctk.CTkButton(
            seg_btn_col,
            text="Rerun",
            height=BTN_H,
            command=self.rerun_segmentation,
            fg_color="#b45309",
            hover_color="#92400e",
        ).grid(row=1, column=0, sticky="ew", pady=3)
        # REACH P1 + P7: fall back to point-and-select if the auto result
        # is wrong.
        ctk.CTkButton(
            seg_btn_col, text="Pick Manually", height=BTN_H, command=self.pick_manually
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
            font=ctk.CTkFont(size=13, weight="bold"),
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
            command=self.confirm_pixel_goal,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: back out of the selection (REACH menu "back button").
        ctk.CTkButton(
            click_btn_col,
            text="Cancel",
            height=BTN_H,
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
            self.uncertainty_panel, text="", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.uncertainty_title.pack(pady=(10, 2))
        self.uncertainty_desc_label = ctk.CTkLabel(
            self.uncertainty_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
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
                command=lambda a=angle: self.submit_uncertainty_heading(a),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.uncertainty_buttons[key] = btn
        # REACH P1: re-request a VLM heading sweep instead of committing.
        self.uncertainty_retry_button = ctk.CTkButton(
            self.uncertainty_panel,
            text="Retry (R)",
            height=BTN_H,
            fg_color="#b45309",
            hover_color="#92400e",
            command=self.retry_uncertainty,
        )
        self.uncertainty_retry_button.pack(pady=(0, 12))
        self._uncertainty_panel_visible = False

        # ===== FOOTER — last click outcome (REACH P9) =================== #
        # Persistent one-line acknowledgment of the last point-goal click
        # ("point goal set at ..." / "click ignored: no valid depth").
        # This row absorbs leftover sidebar height so the stack stays
        # top-anchored instead of floating mid-window.
        footer_row = row
        sidebar.grid_rowconfigure(footer_row, weight=1)
        self.click_status_label = ctk.CTkLabel(
            sidebar,
            text="",
            anchor="n",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
            wraplength=self.SIDEBAR_MIN_W - 20,
        )
        self.click_status_label.grid(
            row=footer_row, column=0, sticky="new", padx=4, pady=(BTN_GAP, 0)
        )

        # ===== TRAJECTORY / BELIEF (exp.md Sec. 5) ======================= #
        # "Where has the rover been, and where does it currently believe
        # the goal is?" -- full-width, persistent, below the camera+sidebar
        # body (REACH P4 map/situational-awareness), pulled out of the
        # scrollable sidebar so it's never affected by sidebar scroll
        # position, matching exp.md's dedicated bottom region.
        plot_frame = ctk.CTkFrame(root)
        plot_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4)
        )
        plot_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            plot_frame,
            text="TRAJECTORY / BELIEF",
            font=ctk.CTkFont(size=12, weight="bold"),
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
        stop_bar = ctk.CTkFrame(root, fg_color="transparent")
        stop_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        stop_bar.grid_columnconfigure(0, weight=3)
        stop_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            stop_bar,
            text="■  EMERGENCY STOP",
            height=56,
            font=ctk.CTkFont(size=18, weight="bold"),
            command=self.stop,
            fg_color="#b91c1c",
            hover_color="#991b1b",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        # REACH P8: reset/recover is destructive-adjacent, so it sits next
        # to STOP but is visually subordinate (grey, narrower).
        ctk.CTkButton(
            stop_bar,
            text="Reset Rover",
            height=56,
            command=self.reset_rover,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

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

    # ---------------- commands ---------------- #
    def _guarded(self, fn, *fn_args) -> None:
        # Skip root-level keyboard shortcuts (the uncertainty-halt numpad
        # heading/retry bindings below) while the taskwork entry has focus,
        # so typing a command there doesn't also submit a heading.
        if self.root.focus_get() is self.command_entry:
            return
        fn(*fn_args)

    def _ack(self, text: str) -> None:
        # REACH P9: record an immediate human-readable acknowledgment of
        # the just-pressed control; refresh() echoes it in the status
        # header so no press ever feels unregistered.
        self._ack_text = text

    def submit_command(self) -> None:
        # Handed to RoverController.submit_nav_command, which splits it via
        # the Qwen VLM (qwen_client.parse_nav_command) into (directions,
        # goals) on a background thread, turns that into an ordered Mission
        # (nav.mission.parse_parts), and drives through its sub-goals one at
        # a time. Progress shows in mission_status_label.
        text = self.command_entry.get().strip()
        if not text:
            return
        print(f"[nav command] submitted: {text}")
        self._ack(f"taskwork queued: {text}")
        self.controller.submit_nav_command(text)
        self.command_entry.delete(0, "end")

    def reset_rover(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("rover reset")
        self.controller.request_reset()

    def stop(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("STOP — driving halted")
        self.controller.stop_driving()

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
        if not alive:
            return (
                "RECOVERING",
                "controller thread died -- see console",
                TOP_STATUS_COLORS["RECOVERING"],
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
        return "IDLE", d.status_text, TOP_STATUS_COLORS["IDLE"]

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
            text=goal_status or "--",
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
            text=rover_mode, text_color=ROVER_MODE_COLORS.get(rover_mode, "#e5e7eb")
        )

        if d.uncertainty_enabled:
            cur_text = f"{d.uncertainty_value:.2f}/{d.uncertainty_threshold:.2f}"
            cur_color = self._unc_color(d.uncertainty_value, d.uncertainty_threshold)
        else:
            cur_text, cur_color = "off", PLOT_TELEMETRY_TEXT
        self.current_uncertainty_label.configure(text=cur_text, text_color=cur_color)

        home_text = f"{d.home_base_uncertainty_value:.2f}/{d.home_base_uncertainty_threshold:.2f}"
        self.home_uncertainty_label.configure(
            text=home_text,
            text_color=self._unc_color(
                d.home_base_uncertainty_value, d.home_base_uncertainty_threshold
            ),
        )

        # exp.md Sec 4 "Telemetry" -- only the operational values it calls
        # out (position, heading, velocity, yaw rate, nav state);
        # model-internal/CBF debugging detail stays in the trajectory/
        # belief bar's own HUD instead of this panel.
        pose_txt = (
            f"pos ({d.pose.x:.1f}, {d.pose.z:.1f})  heading={math.degrees(d.pose.yaw):.0f}deg"
            if d.pose is not None
            else "pos: -"
        )
        act = d.action
        nav_state = d.mode.upper().replace("_", " ")
        if d.goal_reached:
            nav_state += " (GOAL REACHED)"
        lines = [
            pose_txt,
            f"v={act.v_fwd:.2f} m/s  yaw_rate={act.yaw_rate:+.2f} rad/s",
            f"nav state: {nav_state}",
        ]
        self.telemetry_label.configure(text="\n".join(lines))

    def _sync_top_status(self, d, goal_status: Optional[str], alive: bool) -> None:
        state, subtext, color = self._derive_top_status(d, goal_status, alive)
        self.top_status_label.configure(text=state, text_color=color)
        # REACH P9: prefer the freshest human ack over the derived subtext
        # so a just-pressed control's acknowledgment always wins on the
        # same frame it fires, before the controller's own state catches up.
        self.top_status_detail_label.configure(text=self._ack_text or subtext)

    def refresh(self) -> None:
        if self.closed:
            return
        d = self.controller.snapshot()

        if d.vis_rgb is not None:
            self._draw_camera(d.vis_rgb)

        if (
            d.vis_rgb_topdown is not None
            and self._topdown_canvas is not None
            and self._topdown_window.winfo_viewable()
        ):
            self._draw_topdown(d.vis_rgb_topdown)

        self._draw_plot(d)
        self._sync_seg_panel(d)
        self._sync_click_panel()
        self._sync_uncertainty_panel(d)
        self.click_status_label.configure(text=d.click_status)
        self.mission_status_label.configure(text=d.mission_status)

        goal_status = self._derive_goal_status(d)
        self._sync_mission_panel(d, goal_status)
        self._sync_rover_status_panel(d)

        alive = self.controller.is_alive()
        self._sync_top_status(d, goal_status, alive)
        if not alive and not self._alive_label_visible:
            self.alive_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
            self._alive_label_visible = True
        elif alive and self._alive_label_visible:
            self.alive_label.grid_forget()
            self._alive_label_visible = False

        self.root.after(REFRESH_MS, self.refresh)

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
                text="REQUESTING VLM SWEEP...", text_color="#f59e0b"
            )
        else:
            self.uncertainty_panel.configure(border_color="#38bdf8")
            self.uncertainty_title.configure(
                text="UNCERTAINTY HALT -- choose a heading", text_color="#38bdf8"
            )
        self.uncertainty_desc_label.configure(
            text=d.uncertainty_line or "Waiting for VLM sweep description..."
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
        # exp.md Sec 5: a full-width, fixed-height bottom bar now (not a
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
                gx, gy, text="*", fill=PLOT_GOAL, font=("TkDefaultFont", 26)
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
    return RoverController(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        navdp_upstream_ckpt=navdp_upstream_ckpt,
        navdp_upstream_root=args.navdp_upstream_root,
        navdp_root=args.navdp_root,
        gen_home_base=args.gen_home_base,
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
        max_yaw_rate=args.max_angular,
        navdp_upstream_port=args.navdp_upstream_port,
        navdp_upstream_lookahead=args.navdp_upstream_lookahead,
        navdp_upstream_replan_every=args.navdp_upstream_replan_every,
        navdp_upstream_server_variant=args.navdp_upstream_server_variant,
        navdp_upstream_planner_mode=args.navdp_upstream_planner_mode,
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
        "--heightmap-path", default=str(REPO_ROOT / "marsyard2022_terrain_hm_1025.tif")
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
        "--navdp-root",
        default=None,
        help="Path to this repo's own navdp/ package (default: ./navdp or $NAVDP_ROOT) -- only "
        "needed for the CBF safety layer's generic obstacle/avoidance math, unrelated to which "
        "driving policy is active",
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
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--hz", type=float, default=10.0, help="controller tick rate")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
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
        "annotations/mesh_tight_bound2, the dataset sam_lora_runs/exp10 was trained on)",
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
        default=vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD,
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
        "--use-segmentation",
        default=False,
        action="store_true",
        help="show the sidebar's 'Run Segmentation' button (RoverController."
        "request_resolve() with no target text -- the original SAM2+Qwen-"
        "salience auto-resolve, see --seg-backend/--seg-checkpoint/--seg-overlay) "
        "and the 'REVIEW RESOLVED GOAL' Confirm/Rerun/Pick-Manually panel it "
        "leads into. Off by default: with this flag unset, nothing in the "
        "console can enter MODE_REVIEW_SEGMENTATION at all (mission GO_TO/FIND "
        "steps use DINO grounding and auto-confirm instead)",
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

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    app = NavGuiApp(root, controller, use_segmentation=args.use_segmentation)
    try:
        app.tick_forever()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
