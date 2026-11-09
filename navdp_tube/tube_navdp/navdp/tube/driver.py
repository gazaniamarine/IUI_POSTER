"""Tube-guided NavDP for any robot or simulator: camera + pose + occupancy map in, velocity command out.

``plan`` (a few times per second) asks scripts/tube_navdp_server.py for a path and its tube; ``command`` (at the
control rate) turns them into forward speed and yaw rate. Nothing here depends on Habitat.

Conventions: poses are (x, y, yaw) in any fixed world frame in metres and radians, yaw counter-clockwise from +x.
The map is a ``WorldGrid`` in that frame (occupied cells, without the robot's size); ``plan`` grows it by the robot
and cuts a body-frame grid out of it for the tube. RGB is uint8 [H, W, 3]; depth is metres [H, W].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from navdp.official_client import track_trajectory
from navdp.tube.client import TubeNavDPClient
from navdp.tube.controller import FunnelConfig, TubeClock, funnel_command
from navdp.tube.stt import Bev


@dataclass
class WorldGrid:
    """An occupancy map in the world frame, the ROS / bev_map OccupancyGrid layout: occupied[iy, ix] covers
    x in [origin_x + ix res, ...), y in [origin_y + iy res, ...). Unknown cells count as free; only occupied ones
    block the tube."""
    occupied: np.ndarray               # bool [ny, nx]
    resolution: float                  # metres per cell
    origin: tuple                      # (x_min, y_min)


def body_bev(grid: WorldGrid, pose, grow_m: float, half_m: float = 6.5) -> Bev:
    """World map -> the tube's body-frame grid round a robot at pose (x, y, yaw), +-half_m, same cell size.
    Occupied cells are first grown by grow_m (the robot's size) plus one cell, which covers the nearest-cell
    resampling into the rotated frame."""
    occupied = np.asarray(grid.occupied, bool)
    res = float(grid.resolution)
    if occupied.any():
        from scipy.ndimage import distance_transform_edt
        occupied = distance_transform_edt(~occupied) * res <= grow_m + res
    n = int(round(2 * half_m / res))
    centres = -half_m + (np.arange(n) + 0.5) * res
    forward, left = np.meshgrid(centres, centres, indexing="ij")
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x = pose[0] + c * forward - s * left
    y = pose[1] + s * forward + c * left
    ix = np.floor((x - grid.origin[0]) / res).astype(np.int64)
    iy = np.floor((y - grid.origin[1]) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < occupied.shape[1]) & (iy >= 0) & (iy < occupied.shape[0])
    body = np.zeros((n, n), bool)
    body[ok] = occupied[iy[ok], ix[ok]]
    return Bev(body, res, (-half_m, -half_m))


def to_body(point_xy, pose) -> np.ndarray:
    """World (x, y) -> (forward, left) of a robot at pose (x, y, yaw)."""
    dx, dy = float(point_xy[0]) - pose[0], float(point_xy[1]) - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def carry_over(trajectory, old_pose, new_pose):
    """The previous plan (body frame of old_pose, [H, 2+]) -> its part still ahead in the body frame of new_pose:
    the waypoints after the one nearest the rover now, so waypoint j of the result is where the rover was meant to be
    j + 1 waypoints from now. None when the plan was no path (a turn in place or NavDP's stop: no forward motion)."""
    pts = np.asarray(trajectory, np.float64)[:, :2]
    if len(pts) == 0 or np.abs(pts[:, 0]).max() < 1e-6 or np.linalg.norm(pts[-1]) < 0.05:
        return None
    c, s = math.cos(old_pose[2]), math.sin(old_pose[2])
    world = np.stack([old_pose[0] + c * pts[:, 0] - s * pts[:, 1], old_pose[1] + s * pts[:, 0] + c * pts[:, 1]], 1)
    body = np.array([to_body(p, new_pose) for p in world])
    ahead = body[int(np.argmin(np.linalg.norm(body, axis=1))) + 1:]
    return ahead if len(ahead) else None


def executed_tube(report):
    """The solved tube of the executed candidate's group, or None."""
    tubes, group, choice = report.get("tubes") or [], report.get("group"), report.get("choice")
    if choice is None or group is None or not tubes:
        return None
    tube = tubes[group[choice]]
    return tube if tube.get("solved") and tube.get("lo_coef") is not None else None


@dataclass
class DriverConfig:
    mode: str = "tube"             # navdp | obstacle | tube
    controller: str = "stt"        # stt: the tube law; track: chase a waypoint of the chosen path
    robot_radius: float = 0.6      # the map's occupied cells are grown by this (the tube sees the robot as a point)
    lookahead: int = 3             # waypoint the tracker chases
    stop_threshold: float = -4.0   # NavDP's critic stop rule
    send_previous: bool = True     # send the previous plan (carry_over) for the server's --consistency
    funnel: FunnelConfig = field(default_factory=FunnelConfig)


class TubeDriver:
    def __init__(self, server="http://127.0.0.1:8890", intrinsic=None, config: DriverConfig = None, client=None):
        self.config = config or DriverConfig()
        if self.config.controller not in ("stt", "track"):
            raise ValueError("controller must be stt or track")
        self.client = client or TubeNavDPClient(server)
        self.start_forward = self.config.funnel.lookahead_m if self.config.controller == "stt" else 0.0
        self.intrinsic = intrinsic
        self.reset()

    def reset(self):
        """Call at the start of every run: clears NavDP's image memory on the server."""
        self.client.reset(self.intrinsic, self.config.stop_threshold)
        self.plan_pose, self.plan_time, self.clock, self.track_command, self.result = None, 0.0, None, None, None

    def plan(self, rgb, depth, goal_xy, pose, bev: WorldGrid = None, now=0.0):
        """Ask for a new path from the current view. bev: the world-frame map, without the robot's size (None = no
        obstacles). -> NavDPResult (.trajectory, .candidates, .values, .tube)."""
        cfg = self.config
        grid = None if bev is None else body_bev(bev, pose, cfg.robot_radius + self.start_forward)
        result = self.client.step(rgb, depth, to_body(goal_xy, pose), grid, cfg.mode, self.start_forward,
                                  prev=self.previous_plan(pose))
        funnel = replace(cfg.funnel)
        tube = executed_tube(result.tube or {})
        self.clock = (None if tube is None else
                      TubeClock(tube["lo_coef"], tube["hi_coef"], len(result.trajectory), funnel))
        self.track_command = track_trajectory(result.trajectory, lookahead=cfg.lookahead, max_speed=funnel.max_speed,
                                              max_yaw_rate=funnel.max_yaw_rate)
        self.plan_pose, self.plan_time, self.result = tuple(float(v) for v in pose), float(now), result
        return result

    def previous_plan(self, pose):
        """The executed path of the last plan, still ahead, in the body frame of pose (carry_over), or None."""
        if not self.config.send_previous or self.result is None or self.plan_pose is None:
            return None
        if getattr(self.result, "choice_kind", "") == "turn":
            return None
        return carry_over(self.result.trajectory, self.plan_pose, pose)

    def command(self, pose, now) -> tuple:
        """-> (forward m/s, yaw rad/s, info). Call at the control rate, after at least one ``plan``. info: ``inside``
        (the controlled point is in the tube; None without a tube), ``saturated`` (the command was clipped, where the
        tube law's guarantee stops holding), ``fallback`` (no tube, so the tracker's command was used)."""
        if self.plan_pose is None:
            raise RuntimeError("call plan before command")
        funnel = self.config.funnel
        if self.clock is None or self.config.controller == "track":
            inside = None
            if self.clock is not None:
                lo, hi, _ = self.clock.at(now - self.plan_time)
                point = self._point(pose)
                inside = bool(np.all((point > lo) & (point < hi)))
            return (float(self.track_command[0]), float(self.track_command[2]),
                    {"inside": inside, "saturated": None, "fallback": self.config.controller == "stt"})
        lo, hi, velocity = self.clock.at(now - self.plan_time)
        v, omega, info = funnel_command(self._point(pose), pose[2] - self.plan_pose[2], lo, hi, velocity, funnel)
        return v, omega, {"inside": info["inside"], "saturated": info["saturated"], "fallback": False}

    def _point(self, pose):
        d = self.config.funnel.lookahead_m
        ahead = (pose[0] + d * math.cos(pose[2]), pose[1] + d * math.sin(pose[2]))
        return to_body(ahead, self.plan_pose)

    def close(self):
        self.client.close()
