"""Live GUI for the real-rover Qwen3.5-9B + NavDP driver.

Laid out to match Nihil/nav_pipeline/point_goal_gui.py (the GUI
launch_rover_s2diff.sh actually runs, via isaac_gui.py): a live camera panel
next to a top-down trajectory plot (chosen NavDP path in red, goal as a
star, robot at bottom-center, same forward-up/left-right convention as that
plot's own `to_px`), with a status bar and STOP button below. This repo adds
a REASONING panel underneath since our System-2 (Qwen3.5-9B) produces
free-text reasoning per replan that point_goal_gui.py's DINO-based System-2
has no equivalent of.

Deliberately does NOT run point_goal_gui.py/isaac_gui.py's own
DinoNavDPPipeline: that requires a conda env (`internnav`) and a
GroundingDINO/SAM2/CLIP/DINOv2 dependency stack that aren't installed on
this machine (checked 2026-08-25 -- no `internnav` env exists, and
`import groundingdino` fails in every existing env). This GUI is visually
the same shape but drives through our own already-validated stack instead:
nav.real_rover_qwen35_navdp.drive_loop (Qwen3.5 point-goal + HTTP NavDP +
the DEPTH_STOP_M hard collision guard) -- the exact same function the
headless CLI uses, so there is exactly one copy of the safety-critical
per-tick logic, not two. Only an on_tick callback is added, publishing into
a GUI-owned state dict under a lock. The GUI's own STOP button sets a
threading.Event that drive_loop checks every tick and reacts to within one
tick (~0.2s at TICK_HZ=5), on top of (not instead of) the existing
zero-twist-on-exit safety net.

Run: ./launch_rover_qwen35_navdp.sh --gui --instruction "..." [--instruction "..." ...]
(launch_rover_qwen35_navdp.sh passes --gui straight through, which dispatches
here instead of the headless main()).
"""

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, "/home/gpu/Desktop/pineapple/mars-habitatsim")

from nav.real_rover_qwen35_navdp import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEPTH_STOP_M,
    HFOV_DEG,
    MAX_ANGULAR,
    MAX_LINEAR,
    connect,
    drive_loop,
    navdp_reset,
    resolve_instructions,
    wait_for_frames,
)
from nav.vlm_point_navigator import VlmPointNavigator  # noqa: E402
from nav_pipeline.zenoh_node import serialize_twist  # noqa: E402

CAM_W, CAM_H = 448, 336
PLOT_SIZE = 336  # matches point_goal_gui.py's PLOT_SIZE feel, scaled to this window
PLOT_RANGE = 3.5  # meters shown ahead -- same as point_goal_gui.py's PLOT_RANGE
BG = "#111318"
PANEL_BG = "#1b1e26"
FG = "#e8e8ec"
MUTED = "#8a8f9c"
ACCENT = "#4fb6ff"
DANGER = "#ff5a5a"
OK = "#39d17c"


def _plot_to_px(x, y, size=PLOT_SIZE, rng=PLOT_RANGE):
    """Robot frame (x forward, y left) -> canvas pixel. Identical convention
    to point_goal_gui.py's own to_px() so the plot reads the same way."""
    return size / 2 - (y / rng) * (size / 2), size - (x / rng) * size * 0.92 - 20


class RoverGuiApp:
    def __init__(self, root, instructions, endpoint, max_steps, replan_every, max_linear, max_angular,
                 autostart=False):
        self.root = root
        self.instructions = instructions
        self.endpoint = endpoint
        self.max_steps = max_steps
        self.replan_every = replan_every
        self.max_linear = max_linear
        self.max_angular = max_angular

        self.stop_event = threading.Event()
        self.info_q = queue.Queue()
        self.drive_thread = None
        self.running = False
        self.last_photo = None  # keep a reference so Tkinter doesn't GC the PhotoImage
        self.st = None  # SharedState, set once connect() succeeds -- see _poll_camera
        self._last_rendered_rgb = None  # identity check, not value -- see _poll_camera
        self._last_traj_xy = None  # persists across ticks with no fresh trajectory -- see _apply

        root.title("Real Rover -- Qwen3.5 + NavDP Point-Goal")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(outer, bg=BG)
        left.pack(side="left", fill="both", expand=False)

        cam_row = tk.Frame(left, bg=BG)
        cam_row.pack()
        self.canvas = tk.Canvas(cam_row, width=CAM_W, height=CAM_H, bg="#000000", highlightthickness=1,
                                 highlightbackground="#333844")
        self.canvas.pack(side="left")
        self.plot = tk.Canvas(cam_row, width=PLOT_SIZE, height=PLOT_SIZE, bg="#ffffff", highlightthickness=1,
                               highlightbackground="#333844")
        self.plot.pack(side="left", padx=(6, 0))

        self.safety_banner = tk.Label(left, text="", font=("Sans", 12, "bold"), bg=BG, fg=DANGER)
        self.safety_banner.pack(fill="x", pady=(6, 0))

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(8, 0))
        self.start_btn = tk.Button(btn_row, text="Start", command=self._on_start, bg=OK, fg="#0a0a0a",
                                    activebackground=OK, relief="flat", padx=14, pady=6)
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(btn_row, text="STOP", command=self._on_stop, bg=DANGER, fg="#0a0a0a",
                                   activebackground=DANGER, relief="flat", padx=14, pady=6, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        right = tk.Frame(outer, bg=PANEL_BG, padx=14, pady=12)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        def row(label):
            f = tk.Frame(right, bg=PANEL_BG)
            f.pack(fill="x", pady=3)
            tk.Label(f, text=label, font=("Sans", 9), bg=PANEL_BG, fg=MUTED, width=14, anchor="w").pack(side="left")
            val = tk.Label(f, text="--", font=("Sans", 10, "bold"), bg=PANEL_BG, fg=FG, anchor="w", justify="left")
            val.pack(side="left", fill="x", expand=True)
            return val

        tk.Label(right, text="INSTRUCTION", font=("Sans", 10, "bold"), bg=PANEL_BG, fg=ACCENT).pack(anchor="w")
        instr_row = tk.Frame(right, bg=PANEL_BG)
        instr_row.pack(fill="x", pady=(2, 8))
        self.instr_entry = tk.Entry(instr_row, bg="#12141a", fg=FG, insertbackground=FG,
                                     relief="flat", font=("Sans", 10))
        self.instr_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.instr_entry.insert(0, " -> ".join(self.instructions))
        self.instr_entry.bind("<Return>", lambda _e: self._on_send())
        self.send_btn = tk.Button(instr_row, text="Send", command=self._on_send, bg=ACCENT, fg="#0a0a0a",
                                   activebackground=ACCENT, relief="flat", padx=10)
        self.send_btn.pack(side="left", padx=(6, 0))

        tk.Label(right, text="STATUS", font=("Sans", 10, "bold"), bg=PANEL_BG, fg=ACCENT).pack(anchor="w")
        self.lbl_endpoint = row("Endpoint")
        self.lbl_step = row("Step")
        self.lbl_instr = row("Current sub-step")
        self.lbl_found = row("Found / Advance")
        self.lbl_body_goal = row("Body goal (f, l)")
        self.lbl_cmd = row("Command (lin, ang)")
        self.lbl_depth = row("Min fwd depth")

        tk.Label(right, text="REASONING", font=("Sans", 10, "bold"), bg=PANEL_BG, fg=ACCENT).pack(
            anchor="w", pady=(12, 2))
        self.txt_reasoning = tk.Text(right, height=8, wrap="word", bg="#12141a", fg=FG, relief="flat",
                                      insertbackground=FG, font=("Sans", 9))
        self.txt_reasoning.pack(fill="both", expand=True)
        self.txt_reasoning.configure(state="disabled")

        self.lbl_endpoint.config(text=self.endpoint)
        self.lbl_instr.config(text=" -> ".join(self.instructions))

        self.root.after(120, self._poll)
        self.root.after(50, self._poll_camera)
        if autostart:
            self.root.after(200, self._on_start)

    # -- thread control --------------------------------------------------

    def _on_send(self):
        # Same guard as Start -- text typed here is auto-split into atomic
        # steps by _drive_worker (see resolve_instructions), same pattern
        # nav/gui1.py's sim counterpart already uses; the resolved list gets
        # written back into this entry once the drive starts (see _apply's
        # "resolved_instructions" handling) so you can see what it actually
        # decomposed into.
        if self.running:
            self.safety_banner.config(text="Stop the current run before sending a new instruction")
            return
        text = self.instr_entry.get().strip()
        if not text:
            return
        self.instructions = [text]
        self.lbl_instr.config(text="--")
        self._on_start()

    def _on_start(self):
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.drive_thread = threading.Thread(target=self._drive_worker, daemon=True)
        self.drive_thread.start()

    def _on_stop(self):
        self.stop_event.set()
        self.stop_btn.config(state="disabled")
        self.safety_banner.config(text="STOPPING...")

    def _on_close(self):
        self.stop_event.set()
        self.root.after(300, self.root.destroy)

    def _drive_worker(self):
        try:
            session, pub_cmd, st = connect(self.endpoint)
            self.st = st  # let _poll_camera start rendering immediately, independent of drive_loop's tick cadence
        except Exception as e:
            self.info_q.put({"error": f"zenoh connect failed: {e}", "fatal": True})
            self._finish()
            return
        try:
            if not wait_for_frames(st):
                pub_cmd.put(serialize_twist(0.0, 0.0))
                self.info_q.put({"error": "no camera/depth within 10s", "fatal": True})
                return
            with st.lock:
                first_rgb = st.rgb
            instructions = resolve_instructions(first_rgb, self.instructions)
            # See resolve_instructions' docstring -- a compound instruction
            # typed as one Send needs proportionally more step budget once
            # split into multiple atomic legs, same reasoning as the
            # headless CLI path.
            max_steps = self.max_steps * max(len(instructions), 1)
            self.info_q.put({"resolved_instructions": instructions})
            navdp_reset()
            navigator = VlmPointNavigator(
                instructions, hfov_deg=HFOV_DEG, replan_every=self.replan_every, num_history=8,
            )
            drive_loop(
                pub_cmd, st, navigator, max_steps, self.max_linear, self.max_angular,
                on_tick=lambda info: self.info_q.put(info),
                stop_event=self.stop_event,
            )
        except Exception as e:
            self.info_q.put({"error": str(e), "fatal": True})
        finally:
            # NOT calling session.close() here (2026-08-27): live-confirmed root
            # cause of a real, repeated SIGABRT ("terminate called without an
            # active exception" / "FATAL: exception not rethrown", core-dumped)
            # that killed this process after every single successful stop --
            # zenoh-python's native session teardown appears to race its own
            # still-alive subscriber callback threads (the same ones connect()
            # deliberately keeps referenced in st.subs). This is a native abort,
            # not a raised Python exception -- no try/except here can catch it,
            # only avoiding the call does. The process exits shortly after this
            # function returns anyway (root.after(300, root.destroy) on window
            # close, or natural mainloop exit), so the OS reclaims the socket
            # and threads regardless of whether Python explicitly closed them.
            self._finish()

    def _finish(self):
        self.info_q.put({"done": True})

    # -- GUI-thread polling ------------------------------------------------

    def _poll(self):
        # try/finally around the drain loop (2026-08-27, same reasoning as
        # _poll_camera's own comment): _apply() raising anything other than
        # queue.Empty used to propagate straight out of this Tk .after
        # callback -- Tk's default handler swallows the exception, but the
        # reschedule call below it never runs, permanently freezing EVERY
        # info-driven panel (step, reasoning, body_goal, cmd, depth, safety
        # banner) for the rest of the run with no error in the driver's own
        # log. drive_loop keeps running for real underneath a frozen display
        # like that -- reads exactly as "not updating" / "not moving toward
        # the target" even when the robot itself may be doing something
        # else entirely. One bad info dict is now skipped, not fatal.
        try:
            while True:
                info = self.info_q.get_nowait()
                try:
                    self._apply(info)
                except Exception as e:
                    print(f"[real-rover-gui] _apply error on info={info!r}: {e}", flush=True)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll)

    def _poll_camera(self):
        # Renders straight from the live zenoh camera buffer (SharedState.rgb,
        # updated the instant a new frame decodes -- see connect()'s on_image
        # callback) on its own fast timer, INDEPENDENT of drive_loop's control
        # ticks. _apply()'s rgb draw (pushed once per tick via on_tick) is the
        # only path before this fix -- fine on a fast (NavDP-only) tick, but a
        # replan tick blocks on a Qwen3.5 HTTP call that can take 1-3+ seconds,
        # during which no new tick info ever reaches info_q, so the displayed
        # frame visibly freezes for the whole call and then jumps -- read as
        # "camera latency" even though real frames kept arriving the entire
        # time. This poll can't stall on that: it only ever touches the
        # shared, lock-protected rgb buffer, never the network/model calls.
        #
        # try/finally around the actual work (2026-08-27): a Tk .after callback
        # that raises has its exception swallowed by Tk's default handler, but
        # any code AFTER the raise point -- including the reschedule call --
        # never runs. Without this, one transient bad frame (odd shape, a
        # decode racing a resize, etc.) would silently kill camera updates for
        # the rest of the run with no error visible in the driver's own log --
        # exactly the "frame not updating" symptom live-observed 2026-08-27.
        # Always reschedule regardless of what happened above.
        try:
            if self.st is not None:
                with self.st.lock:
                    rgb = self.st.rgb
                if rgb is not None and rgb is not self._last_rendered_rgb:
                    self._last_rendered_rgb = rgb
                    self._draw_frame(rgb)
        except Exception as e:
            print(f"[real-rover-gui] _poll_camera error (non-fatal, still polling): {e}", flush=True)
        finally:
            self.root.after(50, self._poll_camera)

    def _apply(self, info):
        if info.get("done"):
            self.running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            if not info.get("error"):
                self.safety_banner.config(text="")
            return
        if info.get("error"):
            self.safety_banner.config(text=f"ERROR: {info['error']}")
        if "resolved_instructions" in info:
            instructions = info["resolved_instructions"]
            self._resolved_max_steps = self.max_steps * max(len(instructions), 1)
            if len(instructions) > 1:
                self.instr_entry.delete(0, "end")
                self.instr_entry.insert(0, " -> ".join(instructions))

        if "step" in info:
            self.lbl_step.config(text=f"{info['step']} / {getattr(self, '_resolved_max_steps', self.max_steps)}")
        if info.get("rgb") is not None:
            self._draw_frame(info["rgb"])
        if "instruction_text" in info:
            self.lbl_instr.config(text=info["instruction_text"])
        if "found" in info:
            self.lbl_found.config(text=f"found={info['found']}  advance={info.get('advance')}")
        if "reasoning" in info:
            self.txt_reasoning.configure(state="normal")
            self.txt_reasoning.delete("1.0", "end")
            self.txt_reasoning.insert("1.0", info["reasoning"])
            self.txt_reasoning.configure(state="disabled")
        if "body_goal" in info:
            # traj_xy is only populated on ticks where drive_loop actually called
            # NavDP -- absent from info entirely on safety-stop/goal-reached/
            # stall-recovery/no-goal ticks (2026-08-28, live-confirmed: the plot
            # was blanking the red trajectory line on every one of those, which
            # given how often they occur made it look like NavDP was rarely
            # producing a trajectory at all). Persist the last REAL one across
            # ticks that don't recompute one, instead of wiping it to nothing;
            # only replace it when a genuinely new traj_xy arrives.
            traj_xy = info.get("traj_xy")
            if traj_xy is not None:
                self._last_traj_xy = traj_xy
            self._draw_plot(info.get("body_goal"), self._last_traj_xy)
        if "body_goal" in info and info["body_goal"] is not None:
            f, l = info["body_goal"]
            self.lbl_body_goal.config(text=f"({f:.2f}, {l:.2f}) m")
        if "fwd_depth" in info:
            d = info["fwd_depth"]
            self.lbl_depth.config(text=("n/a" if d is None else f"{d:.2f} m"))
        if "cmd" in info:
            lin, ang = info["cmd"]
            self.lbl_cmd.config(text=f"lin={lin:.3f} m/s  ang={ang:+.3f} rad/s")
        if info.get("too_close"):
            d = info.get("fwd_depth")
            self.safety_banner.config(
                text=f"SAFETY STOP: obstacle at {d:.2f}m < {DEPTH_STOP_M}m -- forcing zero-twist"
            )
        elif not info.get("error"):
            self.safety_banner.config(text="")
        if info.get("nav_finished"):
            self.safety_banner.config(text="ALL INSTRUCTIONS FINISHED")

    def _draw_frame(self, rgb: np.ndarray):
        img = Image.fromarray(rgb).resize((CAM_W, CAM_H))
        self.last_photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.last_photo)

    def _draw_plot(self, body_goal, traj_xy):
        """Same visual convention as point_goal_gui.py's plot: robot at
        bottom-center, forward is up, left is left. Draws the chosen NavDP
        trajectory (red) and the current point goal (gold star)."""
        self.plot.delete("all")
        S = PLOT_SIZE
        self.plot.create_line(0, S - 20, S, S - 20, fill="#ddd")
        self.plot.create_oval(S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill="black")
        if traj_xy is not None and len(traj_xy):
            pts = [_plot_to_px(p[0], p[1]) for p in traj_xy]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="red", width=3)
        if body_goal is not None:
            gx, gy = _plot_to_px(body_goal[0], body_goal[1])
            gx = min(max(gx, 8), S - 8)
            gy = min(max(gy, 8), S - 8)
            self.plot.create_text(gx, gy, text="★", fill="#d4a017", font=("TkDefaultFont", 20))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--instruction", action="append", required=True)
    ap.add_argument("--max-steps", type=int, default=90)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--max-linear", type=float, default=MAX_LINEAR)
    ap.add_argument("--max-angular", type=float, default=MAX_ANGULAR)
    ap.add_argument("--autostart", action="store_true",
                     help="begin driving immediately on launch instead of waiting for the Start button")
    args = ap.parse_args()

    root = tk.Tk()
    RoverGuiApp(
        root, args.instruction, args.endpoint, args.max_steps, args.replan_every,
        args.max_linear, args.max_angular, autostart=args.autostart,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
