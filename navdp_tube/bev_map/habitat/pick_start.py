"""Pick the rover's start (and goal) on a top-down map of the Mars yard, see its camera view, run a planner from there.

    python pick_start.py                    # any env: switches to the habitat env by itself
    python pick_start.py -- --rocks ""      # options after -- go to run_mars_habitat.py (e.g. the seg map's)

Mouse (map, left of the window):
  left click         start position (the rover faces the goal, if one is set)
  left drag          start position, heading = drag direction
  right click        goal: snaps to a flag within 1.5 m, else that x, y
Keys:
  A / D  (or , .)    turn the start heading 15 deg left / right
  F                  face the goal
  1 / 2 / 3 / 4      planner: NavDP / X-NavDP / SanD-Planner / ours (guidance, no tubes; the default)
  M                  ours' map: seg (SAM2 segmentation BEV map, default) / depth (Habitat depth) / truth (true rocks)
  + / -              zoom the map in / out (around the start)
  ENTER or R         run run_mars_habitat.py from this start (its own window; back here when it ends)
  P                  print the command line instead
  Q / ESC            quit
The camera panel (right) is the rover's view from the start, as the planner will see it.
Needs a screen (DISPLAY); over plain ssh use ssh -X, or copy a command printed with P.
"""

import math
import os
import shlex
import subprocess
import sys

try:                                     # not in the habitat env: run again with its python
    import habitat_sim  # noqa: F401
except ImportError:
    _py = os.path.join(os.environ.get("CONDA_EXE", os.path.expanduser("~/miniconda3/bin/conda")).rsplit("/bin/", 1)[0],
                       "envs", "habitat", "bin", "python")
    if os.path.exists(_py) and os.path.realpath(sys.executable) != os.path.realpath(_py):
        os.execv(_py, [_py] + sys.argv)
    sys.exit("habitat_sim not found: conda activate habitat")

import cv2
import numpy as np

import run_mars_habitat as R
from mars_scene import MarsScene, stage_config

PLANNER_KEYS = {ord("1"): "navdp", ord("2"): "xnavdp", ord("3"): "sandplanner", ord("4"): "ours"}
WIN = "Mars yard: pick a start  (left click = start, right click = goal, ENTER = run, Q = quit)"
MAP_PX = 760


class Picker:
    def __init__(self, glb, radius, cam_h, width, height, hfov, extra):
        self.glb, self.radius, self.cam_h, self.extra = glb, radius, cam_h, extra
        self.size = (width, height, hfov)
        print("[pick] loading the scene ...", flush=True)
        self.s = MarsScene(glb)
        self.clear = self.s.distance_field()
        self.sim, self.agent = R.make_sim(stage_config(glb, os.path.join(R.HERE, "runs")), width, height, hfov)
        self.start, self.yaw = (0.0, 0.0), math.pi / 2
        self.goal_name, self.goal = "flag_red", self.s.origin["flag_red"]
        self.planner = "ours"
        self.map = "seg"
        self.zoom = 1.0                  # 1 = the whole yard
        self.drag = None
        self.msg = ""
        self.face_goal()
        self.render_map_base()

    # ----------------------------------------------------------- map
    def render_map_base(self):
        x0, x1, y0, y1 = self.s.bounds
        half = 0.5 * (x1 - x0) / self.zoom
        cx = min(max(self.start[0], x0 + half), x1 - half)
        cy = min(max(self.start[1], y0 + half), y1 - half)
        self.view = R.TopView(self.s, (cx - half, cy - half), (cx + half, cy + half), margin=0.0,
                              px_per_m=MAP_PX / (2 * half))
        self.k = MAP_PX / (2 * half)

    def to_world(self, u, v):
        return self.view.x0 + u / self.k, self.view.y1 - v / self.k

    def draw_map(self):
        v = self.view
        img = cv2.resize(v.base, (MAP_PX, MAP_PX)) if v.base.shape[:2] != (MAP_PX, MAP_PX) else v.base.copy()
        for name in self.s.names:                      # labels
            p = v.px(*self.s.origin[name])
            col = (0, 140, 0) if name.startswith("flag") else (40, 40, 40)
            if name.startswith("flag"):
                cv2.circle(img, p, 5, (0, 170, 255) if name == self.goal_name else (0, 200, 0), -1)
            cv2.putText(img, name.replace("boulder_", "rock "), (p[0] + 7, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        col, 1, cv2.LINE_AA)
        g = v.px(*self.goal)
        cv2.drawMarker(img, g, (0, 170, 255), cv2.MARKER_CROSS, 18, 2)
        s = v.px(*self.start)
        ok = self.start_clear() > 0
        col = (200, 60, 20) if ok else (0, 0, 255)
        cv2.circle(img, s, max(3, int(self.radius * self.k)), col, 2)
        cv2.arrowedLine(img, s, (int(s[0] + 28 * math.cos(self.yaw)), int(s[1] - 28 * math.sin(self.yaw))), col, 2,
                        tipLength=0.35)
        cv2.line(img, s, g, (0, 170, 255), 1, cv2.LINE_AA)
        # field of view
        half = math.radians(self.size[2]) / 2
        for a in (self.yaw - half, self.yaw + half):
            cv2.line(img, s, (int(s[0] + 60 * math.cos(a)), int(s[1] - 60 * math.sin(a))), (200, 60, 20), 1)
        return img

    def start_clear(self):
        if not self.s.grid.inside(*self.start):
            return -1.0
        return self.s.lookup(self.clear, *self.start) - self.radius

    # ----------------------------------------------------------- camera
    def camera(self):
        x, y = self.start
        a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        h = float(np.mean(self.s.height_at(np.r_[x, x + self.radius * np.cos(a)],
                                           np.r_[y, y + self.radius * np.sin(a)]))) + self.cam_h
        R.place_camera(self.agent, x, y, self.yaw, h)
        o = self.sim.get_sensor_observations()
        rgb = cv2.cvtColor(np.ascontiguousarray(o["rgb"][..., :3]), cv2.COLOR_RGB2BGR)
        d = np.asarray(o["depth"], np.float32)
        dv = cv2.applyColorMap((np.clip(d / 10.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dv[d <= 0] = 0
        w = 400                          # two panels + 6 text lines fit in MAP_PX
        hh = int(rgb.shape[0] * w / rgb.shape[1])
        return np.vstack([cv2.resize(rgb, (w, hh)), cv2.resize(dv, (w, hh))])

    def frame(self):
        m = self.draw_map()
        cam = self.camera()
        panel = np.zeros((MAP_PX, 560, 3), np.uint8)
        x0 = (560 - cam.shape[1]) // 2
        panel[:cam.shape[0], x0:x0 + cam.shape[1]] = cam
        dist = math.hypot(self.goal[0] - self.start[0], self.goal[1] - self.start[1])
        clear = self.start_clear()
        lines = [f"planner  {self.planner}   (1 NavDP 2 X-NavDP 3 SanD 4 ours)"
                 + (f"   map {self.map} (M)" if self.planner == "ours" else ""),
                 f"start    ({self.start[0]:+.2f}, {self.start[1]:+.2f})  heading {math.degrees(self.yaw):+.0f} deg",
                 f"goal     {self.goal_name or 'point'} ({self.goal[0]:+.2f}, {self.goal[1]:+.2f})  {dist:.1f} m away",
                 f"clear    {clear:.2f} m to the nearest rock / pole" if clear > 0 else
                 "clear    INSIDE A ROCK / OFF THE MAP: pick another start",
                 "ENTER run  P print cmd  A/D turn  F face goal  +/- zoom  Q quit",
                 self.msg]
        y = cam.shape[0] + 22
        for i, t in enumerate(lines):
            col = (0, 0, 255) if (i == 3 and clear <= 0) else (230, 230, 230)
            cv2.putText(panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += 22
        return np.hstack([m, panel])

    # ----------------------------------------------------------- input
    def face_goal(self):
        self.yaw = math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])

    def on_mouse(self, event, u, v, flags, _):
        if u >= MAP_PX:
            return
        p = self.to_world(u, v)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = p
            self.start = p
            self.face_goal()
            self.msg = ""
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            if math.hypot(p[0] - self.drag[0], p[1] - self.drag[1]) > 0.3:
                self.yaw = math.atan2(p[1] - self.drag[1], p[0] - self.drag[0])
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            near = min(R.FLAGS, key=lambda f: math.hypot(self.s.origin[f][0] - p[0], self.s.origin[f][1] - p[1]))
            if math.hypot(self.s.origin[near][0] - p[0], self.s.origin[near][1] - p[1]) < 1.5:
                self.goal_name, self.goal = near, self.s.origin[near]
            else:
                self.goal_name, self.goal = None, (round(p[0], 2), round(p[1], 2))
            self.face_goal()

    def command(self):
        goal = self.goal_name or f"{self.goal[0]:.2f},{self.goal[1]:.2f}"
        name = f"pick_{self.goal_name or 'point'}"
        return [sys.executable, os.path.join(R.HERE, "run_mars_habitat.py"), "--planner", self.planner,
                "--goal", goal, f"--start={self.start[0]:.2f},{self.start[1]:.2f}",
                f"--start-yaw={math.degrees(self.yaw):.1f}", "--name", name,
                "--robot-radius", str(self.radius), "--camera-height", str(self.cam_h)] + (
                    ["--map", self.map] if self.planner == "ours" else []) + self.extra

    def run(self):
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.frame())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in PLANNER_KEYS:
                self.planner = PLANNER_KEYS[key]
            elif key == ord("m"):
                maps = ["seg", "depth", "truth"]
                self.map = maps[(maps.index(self.map) + 1) % len(maps)]
            elif key in (ord("a"), ord(",")):
                self.yaw += math.radians(15)
            elif key in (ord("d"), ord(".")):
                self.yaw -= math.radians(15)
            elif key == ord("f"):
                self.face_goal()
            elif key in (ord("+"), ord("=")):
                self.zoom = min(self.zoom * 1.5, 8.0)
                self.render_map_base()
            elif key == ord("-"):
                self.zoom = max(self.zoom / 1.5, 1.0)
                self.render_map_base()
            elif key == ord("p"):
                print(" ".join(shlex.quote(c) for c in self.command()[1:]).replace(R.HERE + "/", ""), flush=True)
                self.msg = "command printed in the terminal"
            elif key in (13, 10, ord("r")):
                if self.start_clear() <= 0:
                    self.msg = "start is inside a rock / off the map"
                    continue
                cmd = self.command()
                print("[pick] " + " ".join(shlex.quote(c) for c in cmd[1:]), flush=True)
                self.msg = f"running {self.planner} ... (its window; Q there stops it)"
                cv2.imshow(WIN, self.frame())
                cv2.waitKey(1)
                code = subprocess.call(cmd, cwd=R.HERE)
                self.msg = f"last run ended (exit {code}): see runs/tests/ and the terminal"
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        cv2.destroyAllWindows()
        self.sim.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", default=R.DEFAULT_GLB)
    ap.add_argument("--robot-radius", type=float, default=0.3)
    ap.add_argument("--camera-height", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=79.0)
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after --: more options for run_mars_habitat.py, e.g. -- --max-speed 0.3 --gap 0.5")
    args = ap.parse_args()
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        sys.exit("pick_start.py needs a screen (DISPLAY). Over ssh: ssh -X, or run it on the machine itself.")
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    Picker(args.glb, args.robot_radius, args.camera_height, args.width, args.height, args.hfov, extra).run()


if __name__ == "__main__":
    main()
