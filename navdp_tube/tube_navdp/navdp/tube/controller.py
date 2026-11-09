"""The STT paper's closed-form tube controller (Das, Basu, Jagtap, arXiv 2411.13834, first-order case), driving a
wheeled rover through a look-ahead point.

For an output y inside a tube [lo(t), hi(t)] per axis, the paper's law is

    e   = (2 y - (hi + lo)) / (hi - lo)               normalised error, in (-1, 1) inside the tube
    eps = ln((1 + e) / (1 - e))                       transformed error, unbounded at the walls
    xi  = 4 / ((hi - lo) (1 - e^2))
    u   = - k xi eps                                  velocity command per axis

and keeps y inside for any k > 0 when the input is not bounded. Here (deviation, recorded): u also gets the tube
centre's own velocity as feedforward (``feedforward``), so the rover does not wait for the back wall to push it;
it is the centre's mean velocity over the next ``feedforward_window_s`` (the box narrows from the start box in the
first instants, which moves its centre without the tube going anywhere), capped at the top speed.

The rover cannot move sideways, so the output is a point d ahead of its centre, p = c + d h(theta); p can move in any
direction: p_dot = v h + omega d n, with n the left normal. Hence v = u . h and omega = (u . n) / d. The commands
are then clipped to the rover's limits (forward only, |omega| <= max); clipping is where the paper's guarantee stops
holding, and it is logged.

Time: the tube is indexed by waypoint (s = (j + 1) / H). It is given seconds here, waypoint by waypoint: each lasts
as long as the tube's centre needs to cover its step at ``speed_fraction`` of the rover's top speed (at least
``min_waypoint_s``). s(t) is a monotone cubic (PCHIP) through those knots, so the bounds are continuously
differentiable in t, as the paper's Definition 2.5 asks (a piecewise-linear s(t) would put kinks at every waypoint).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navdp.tube.stt import bernstein


@dataclass
class FunnelConfig:
    gain: float = 0.035            # k in m^2/s: in a 0.5 m wide box at e = 0.5 the correction is about 0.4 m/s
    lookahead_m: float = 0.2       # d
    feedforward: bool = True
    feedforward_window_s: float = 0.5
    speed_fraction: float = 0.8
    min_waypoint_s: float = 0.05
    max_speed: float = 0.4
    max_yaw_rate: float = 0.8
    e_limit: float = 0.999         # |e| is clipped here when the point is outside (then it is logged as outside)


class TubeClock:
    """A tube (Bernstein coefficients in the planning frame) with seconds attached."""

    def __init__(self, lo_coef, hi_coef, horizon: int, cfg: FunnelConfig):
        self.lo_coef, self.hi_coef = np.asarray(lo_coef, np.float64), np.asarray(hi_coef, np.float64)
        self.horizon, self.degree = horizon, self.lo_coef.shape[1] - 1
        self.knots_s = np.arange(horizon + 1) / horizon
        lo, hi = self.at_s(self.knots_s)
        step = np.linalg.norm(np.diff((lo + hi) / 2, axis=0), axis=1)
        durations = np.maximum(step / (cfg.speed_fraction * cfg.max_speed), cfg.min_waypoint_s)
        self.knots_t = np.concatenate([[0.0], np.cumsum(durations)])
        self.duration = float(self.knots_t[-1])
        from scipy.interpolate import PchipInterpolator
        self._s_of_t = PchipInterpolator(self.knots_t, self.knots_s)  # monotone, C1: no kinks at the waypoints
        self.window, self.max_speed = cfg.feedforward_window_s, cfg.max_speed

    def at_s(self, s):
        b = bernstein(np.atleast_1d(np.asarray(s, np.float64)), self.degree)
        return b @ self.lo_coef.T, b @ self.hi_coef.T

    def s_at(self, tau: float) -> float:
        """Seconds after planning -> the tube's s, held at 0 before and 1 after."""
        return float(np.clip(self._s_of_t(min(max(float(tau), 0.0), self.duration)), 0.0, 1.0))

    def at(self, tau: float):
        """-> lo [2], hi [2], centre velocity [2] at tau seconds after planning (held at the end)."""
        tau = min(max(float(tau), 0.0), self.duration)
        lo, hi = self.at_s([self.s_at(tau)])
        if tau >= self.duration:
            return lo[0], hi[0], np.zeros(2)
        dt = min(self.window, self.duration - tau)
        lo2, hi2 = self.at_s([self.s_at(tau + dt)])
        velocity = ((lo2 + hi2) / 2 - (lo + hi) / 2)[0] / max(dt, 1e-6)
        speed = np.linalg.norm(velocity)
        return lo[0], hi[0], velocity * min(1.0, self.max_speed / max(speed, 1e-9))


def funnel_command(point, heading, lo, hi, centre_velocity, cfg: FunnelConfig):
    """point [2] and heading (rad) in the planning frame; lo / hi / centre_velocity [2]. -> (forward m/s, yaw rad/s,
    info: e, inside, saturated)."""
    point, lo, hi = np.asarray(point, np.float64), np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    width = np.maximum(hi - lo, 1e-6)
    e_raw = (2 * point - (hi + lo)) / width
    inside = bool(np.all(np.abs(e_raw) < 1.0))
    e = np.clip(e_raw, -cfg.e_limit, cfg.e_limit)
    eps = np.log((1 + e) / (1 - e))
    xi = 4.0 / (width * (1 - e * e))
    u = -cfg.gain * xi * eps
    if cfg.feedforward:
        u = u + np.asarray(centre_velocity, np.float64)
    h = np.array([math.cos(heading), math.sin(heading)])
    n = np.array([-math.sin(heading), math.cos(heading)])
    v, omega = float(u @ h), float(u @ n) / cfg.lookahead_m
    saturated = v < 0.0 or v > cfg.max_speed or abs(omega) > cfg.max_yaw_rate
    v, omega = min(max(v, 0.0), cfg.max_speed), float(np.clip(omega, -cfg.max_yaw_rate, cfg.max_yaw_rate))
    return v, omega, {"e": e_raw.tolist(), "inside": inside, "saturated": saturated}
