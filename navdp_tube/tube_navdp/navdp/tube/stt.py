"""Spatiotemporal tubes on a BEV occupancy grid, with no sides of any obstacle chosen.

A tube is, at each normalised time s in [0, 1], a box [lo_i(s), hi_i(s)] per axis (i = forward, left) in the rover's
body frame. Each bound is a Bernstein polynomial. Time is NavDP's waypoint index: waypoint j (of H) sits at
s_j = (j + 1) / H, and s = 0 is the rover.

How the tubes are built (``tubes_for``), from NavDP's paths at the split step:

  1. push (``push_mode``): a waypoint closer than half the tube's width to an occupied cell is moved clear of it,
     then the path is made drivable again (0.25 m per step and axis).
       nearest: each waypoint goes to its own nearest free cell (the first version; neighbouring waypoints can pick
                different sides or be pushed backwards, which leaves kinks and stalls in front of an obstacle).
       smooth:  the nearest push, then the smoothest path that stays in a free box round each pushed waypoint and
                close to NavDP's own path (push_smooth): no waypoint leaves the free space the nearest push found.
  2. cluster per waypoint: at each j the paths' j-th waypoints are linked when they are within ``link_m`` of each
     other and the box spanning the two holds no occupied cell; each connected group is a cluster (split into
     single points when the box round the whole cluster is not free).
  3. one free box per cluster: grown in the grid from the cluster's points at j and j - 1 (the start box for j = 0),
     one cell at a time on each side in turn, until an occupied cell or ``max_grow_m``. It holds no occupied cell.
  4. chain: path i's chain is the box of its cluster at every j. Paths with the same chain share one tube.
  5. tube: per axis, a linear program keeps the tube inside the chain's boxes (``margin`` inside them) at the
     sampled times, close to the members' own spread of paths (L1). Containment of a box in a box is per axis, so
     the two axes are solved independently and avoidance needs no disjunction.

The paper's conditions (Das, Basu, Jagtap, arXiv 2411.13834, eq. 4 / 6) as used here:

    box(0) = S, box(1) = T                                  (4a, the end coefficients)
    Y_lo <= lo_i(s),  hi_i(s) <= Y_hi                       (4b, workspace)
    hi_i(s) - lo_i(s) >= w(s)                               (4c, width; ramps from the start box's width to
                                                             min_width over ``width_ramp`` waypoints)
    box(s) inside the chain's free box of step j(s)         (4d, avoidance: a free box holds no occupied cell)
    lo_i(s) <= c + R(s),  hi_i(s) >= c - R(s)                (added: reachable at NavDP's 0.25 m per step)
    |bound(s_j) - bound(s_j-1)| <= 0.25 m                    (added: between waypoints, so a path of NavDP moves fits)

Certificate (Theorem 3.1 of the paper, made exact): (4d) is enforced on the sampled times only. A bound's slope is at
most L = n max_k |c_k+1 - c_k| (a Bernstein property), and every s is within ``gap`` = 1 / (2 m H) of a sampled time
whose box is constrained to the same free box (waypoint times are constrained to both neighbouring boxes, and s = 0
is the start box, checked against the first box). So the tube stays inside its free boxes for every s in [0, 1] when

    slack >= L * gap          (the paper's eta_S* + L eps <= 0, with eta_S* = -slack)

where slack is the smallest realised distance of a bound to its box on the sampled times. (4b) and (4c) are checked
on a dense grid of times (``eta``). A tube is ``feasible`` when both programs solve, eta <= eta_tol and the bound
above holds. A path with no chain (a waypoint left in an occupied cell) gets no tube and is never certified.

Deviation from the paper: its objective (minimise eta) maximises robustness; here the tube is the closest valid tube
to what NavDP proposed. The same construction (Bezier curves inside free boxes) is known as a safe flight corridor
(Liu et al. 2017; Tordesillas et al., FASTER; Gao et al.); what is used here is that it is built inside one
denoising pass, from the model's own clean-path estimate (navdp/tube/guided_sampler.py).

Grid: ``Bev.occupied[i, j]`` is the cell covering forward [origin_f + i res, origin_f + (i + 1) res) and left
[origin_l + j res, ...) in the body frame. Cells outside the grid count as occupied. The grid must already hold the
rover's size (navdp/tube/driver.py grows it by the robot radius before sending it).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from math import comb

import numpy as np

AXES = ("forward", "left")


def bernstein(s, degree: int) -> np.ndarray:
    """[len(s), degree + 1] Bernstein basis on s in [0, 1]."""
    s = np.asarray(s, np.float64)[:, None]
    k = np.arange(degree + 1)[None]
    return np.array([comb(degree, int(i)) for i in range(degree + 1)])[None] * s ** k * (1 - s) ** (degree - k)


@dataclass
class TubeConfig:
    degree: int = 8                    # Bernstein degree of each bound
    min_width: float = 0.3             # w_min, metres (gamma_d in the paper)
    width_ramp: int = 4                # 4c's width grows from the start box's to min_width over this many waypoints
    start_half: float = 0.05           # S = a box of this half-width ...
    start_forward: float = 0.0         # ... centred this far ahead of the rover's centre (the tube controller's
                                       # look-ahead point)
    target_pad: float = 0.15           # T = the members' endpoints' box grown by this (and at least min_width wide)
    workspace: tuple = ((-6.25, 6.25), (-6.25, 6.25))  # Y: (forward lo, hi), (left lo, hi); NavDP's reach
    step_m: float = 0.25               # NavDP's largest move per waypoint and axis
    horizon: int = 24                  # NavDP's waypoints
    rate: float = 1.0                  # bounds move at most rate * step_m between waypoints
    margin: float = 0.05               # bounds stay this far inside their free boxes on the sampled times
    samples_per_step: int = 6          # sampled times per waypoint step (m); the certificate's gap is 1 / (2 m H):
                                       # with H = 24 it covers bound slopes up to L = 2 m H margin = 14.4 per unit s
    link_m: float = 0.3                # waypoints this close (per axis) with free space between them share a cluster
    max_grow_m: float = 1.5            # a free box grows at most this far beyond its seed, per side
    dense: int = 400                   # times on which 4b and 4c are checked
    eta_tol: float = 1e-3              # 4b / 4c may be missed by this much between the sampled times
    push_mode: str = "nearest"         # step 1: nearest (each waypoint to its nearest free cell, the first version)
                                       # or smooth (the nearest push, then smoothed inside free boxes; push_smooth)
    push_smooth_beta: float = 2.0      # smooth: weight of the bending term beta |D2 p|^2
    push_smooth_grow_m: float = 0.4    # smooth: each pushed waypoint's free box grows at most this far per side
    push_smooth_iters: int = 200       # smooth: ADMM iterations


PUSH_MODES = ("nearest", "smooth")


@dataclass
class Tube:
    lo_coef: np.ndarray                # [2, degree + 1]
    hi_coef: np.ndarray                # [2, degree + 1]
    solved: bool                       # both linear programs feasible
    eta: float                         # largest violation of 4b-4c on the dense times (valid when <= eta_tol)
    target: np.ndarray                 # [2, 2] (axis, lo/hi)
    members: list = field(default_factory=list)  # sample indices given this tube
    slack: float = -np.inf             # smallest distance of a bound to its free box on the sampled times
    lipschitz: float = np.inf          # L: largest slope of any bound in s
    gap: float = np.inf                # every s is within this of a sampled time
    eta_tol: float = 1e-3

    @property
    def theorem(self) -> bool:
        """Theorem 3.1: the tube is inside its free boxes for every s, not only on the sampled times."""
        return bool(self.solved and self.slack - self.lipschitz * self.gap >= 0.0)

    @property
    def feasible(self) -> bool:
        return bool(self.solved and self.eta <= self.eta_tol and self.theorem)

    def bounds(self, s) -> tuple:
        """-> lo [len(s), 2], hi [len(s), 2]."""
        basis = bernstein(s, self.lo_coef.shape[1] - 1)
        return basis @ self.lo_coef.T, basis @ self.hi_coef.T

    def to_dict(self, s=None) -> dict:
        out = {"eta": self.eta, "solved": self.solved, "feasible": self.feasible, "soft": False,
               "target": self.target.tolist(), "members": list(self.members), "slack": self.slack,
               "lipschitz": self.lipschitz, "gap": self.gap, "theorem": self.theorem}
        if s is not None and self.solved:
            lo, hi = self.bounds(s)
            out["lo"], out["hi"] = lo.tolist(), hi.tolist()
            out["lo_coef"], out["hi_coef"] = self.lo_coef.tolist(), self.hi_coef.tolist()
        return out


@dataclass
class Bev:
    occupied: np.ndarray               # bool [n_forward, n_left]
    resolution: float                  # metres per cell
    origin: tuple                      # (forward, left) of cell (0, 0)'s corner, body frame

    def __post_init__(self):
        self.occupied = np.asarray(self.occupied, bool)
        self.origin = (float(self.origin[0]), float(self.origin[1]))
        self._sum = np.pad(self.occupied.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        self._nearest = None           # nearest free cell of every cell, computed once (nearest_free)

    @classmethod
    def empty(cls, resolution=0.05, half_m=6.5) -> "Bev":
        n = int(round(2 * half_m / resolution))
        return cls(np.zeros((n, n), bool), resolution, (-half_m, -half_m))

    def inflated(self, radius_m: float) -> "Bev":
        """Every occupied cell grown by radius_m."""
        if radius_m <= 0 or not self.occupied.any():
            return self
        from scipy.ndimage import distance_transform_edt
        return Bev(distance_transform_edt(~self.occupied) * self.resolution <= radius_m, self.resolution, self.origin)

    def cells(self, lo, hi) -> tuple:
        """Box [lo, hi] (forward, left) in metres -> the index ranges of the cells it touches, (i0, i1, j0, j1)."""
        i0, j0 = np.floor((np.asarray(lo) - self.origin) / self.resolution + 1e-9).astype(int)
        i1, j1 = np.ceil((np.asarray(hi) - self.origin) / self.resolution - 1e-9).astype(int)
        return int(i0), int(max(i1, i0 + 1)), int(j0), int(max(j1, j0 + 1))

    def blocked(self, i0, i1, j0, j1) -> bool:
        """Any occupied cell in [i0, i1) x [j0, j1); outside the grid counts as occupied."""
        n_f, n_l = self.occupied.shape
        if i0 < 0 or j0 < 0 or i1 > n_f or j1 > n_l:
            return True
        s = self._sum
        return bool(s[i1, j1] - s[i0, j1] - s[i1, j0] + s[i0, j0] > 0)

    def box_blocked(self, lo, hi) -> bool:
        return self.blocked(*self.cells(lo, hi))

    def occupied_at(self, points, outside: bool = True) -> np.ndarray:
        """[..., 2] points -> bool [...]: the cell under each point is occupied (a point outside the grid: ``outside``)."""
        p = np.asarray(points, np.float64)
        idx = np.floor((p - self.origin) / self.resolution).astype(int)
        n_f, n_l = self.occupied.shape
        ok = (idx[..., 0] >= 0) & (idx[..., 0] < n_f) & (idx[..., 1] >= 0) & (idx[..., 1] < n_l)
        i, j = np.clip(idx[..., 0], 0, n_f - 1), np.clip(idx[..., 1], 0, n_l - 1)
        return np.where(ok, self.occupied[i, j], outside)

    def nearest_free(self, points) -> np.ndarray:
        """[..., 2] points -> the same, every point in an occupied cell moved to the centre of the nearest free cell
        (points outside the grid are left alone)."""
        out = np.array(points, np.float64)
        if not self.occupied.any() or self.occupied.all():
            return out
        if self._nearest is None:
            from scipy.ndimage import distance_transform_edt
            self._nearest = distance_transform_edt(self.occupied, return_indices=True)[1]
        fi, fj = self._nearest
        idx = np.floor((out - self.origin) / self.resolution).astype(int)
        n_f, n_l = self.occupied.shape
        ok = (idx[..., 0] >= 0) & (idx[..., 0] < n_f) & (idx[..., 1] >= 0) & (idx[..., 1] < n_l)
        i, j = np.clip(idx[..., 0], 0, n_f - 1), np.clip(idx[..., 1], 0, n_l - 1)
        move = ok & self.occupied[i, j]
        free = np.stack([fi[i, j], fj[i, j]], axis=-1)
        out[move] = np.asarray(self.origin) + (free[move] + 0.5) * self.resolution
        return out

    def encode(self) -> dict:
        return {"shape": list(self.occupied.shape), "resolution": self.resolution, "origin": list(self.origin),
                "bits": base64.b64encode(np.packbits(self.occupied, axis=None).tobytes()).decode("ascii")}

    @classmethod
    def decode(cls, data: dict) -> "Bev":
        shape = tuple(int(v) for v in data["shape"])
        bits = np.unpackbits(np.frombuffer(base64.b64decode(data["bits"]), np.uint8))[:shape[0] * shape[1]]
        return cls(bits.reshape(shape).astype(bool), float(data["resolution"]), tuple(data["origin"]))


def sample_times(horizon: int, per_step: int = 1) -> np.ndarray:
    """per_step times in each waypoint step, the step's waypoint last (s = 1 included, s = 0 excluded)."""
    return np.arange(1, per_step * horizon + 1) / (per_step * horizon)


def waypoint_times(horizon: int) -> np.ndarray:
    return (np.arange(horizon) + 1.0) / horizon


def _ramp(s, cfg: TubeConfig) -> np.ndarray:
    return np.clip(np.asarray(s, np.float64) * cfg.horizon / max(cfg.width_ramp, 1), 0.0, 1.0)


def min_width_at(s, cfg: TubeConfig) -> np.ndarray:
    """4c's width at times s: the start box's width at s = 0, growing linearly to min_width at waypoint width_ramp."""
    w0 = min(2 * cfg.start_half, cfg.min_width)
    return w0 + (cfg.min_width - w0) * _ramp(s, cfg)


def start_box(cfg: TubeConfig) -> tuple:
    centre = np.array([cfg.start_forward, 0.0])
    return centre - cfg.start_half, centre + cfg.start_half


def _solve_axis(start, target, envelope, workspace, reach, cfg: TubeConfig, s, floor, ceiling, centre=0.0,
                contain=None):
    """One axis. envelope: (lo [n], hi [n]) at times s to fit; floor / ceiling [n]: lo(s) must stay above floor and
    hi(s) below ceiling. contain: None, or (pmin [H], pmax [H]) the member paths' waypoints, which the tube must hold
    at the waypoint times (hard). -> (lo_coef, hi_coef) or None when infeasible."""
    from scipy.optimize import linprog
    z, n = cfg.degree + 1, len(s)
    b = bernstein(s, cfg.degree)
    zb, zn, eye = np.zeros_like(b), np.zeros((n, n)), np.eye(n)
    # variables: lo coefficients, hi coefficients, |lo - envelope lo|, |hi - envelope hi|
    row = lambda lo, hi, ulo=zn, uhi=zn: np.hstack([lo, hi, ulo, uhi])
    rows = [row(b, zb, -eye), row(-b, zb, -eye), row(zb, b, zn, -eye), row(zb, -b, zn, -eye),
            row(-b, zb), row(zb, b),                 # workspace
            row(b, -b),                              # width (its margin ramps up with it from the start box)
            row(b, zb), row(zb, -b),                 # reach
            row(-b, zb), row(zb, b)]                 # avoidance: inside the free boxes
    rhs = [envelope[0], -envelope[0], envelope[1], -envelope[1],
           np.full(n, -workspace[0]), np.full(n, workspace[1]), -(min_width_at(s, cfg) + cfg.margin * _ramp(s, cfg)),
           centre + reach, reach - centre,
           -np.asarray(floor), np.asarray(ceiling)]
    wp = bernstein(np.concatenate([[0.0], waypoint_times(cfg.horizon)]), cfg.degree)
    d, zd, znd = np.diff(wp, axis=0), np.zeros((cfg.horizon, z)), np.zeros((cfg.horizon, n))
    for lo_part, hi_part in ((d, zd), (-d, zd), (zd, d), (zd, -d)):  # rate limit between waypoints
        rows.append(row(lo_part, hi_part, znd, znd))
        rhs.append(np.full(cfg.horizon, cfg.rate * cfg.step_m))
    if contain is not None:                          # lo(s_j) <= pmin_j and hi(s_j) >= pmax_j at the waypoints
        bw = bernstein(waypoint_times(cfg.horizon), cfg.degree)
        zw, zwn = np.zeros_like(bw), np.zeros((cfg.horizon, n))
        rows += [np.hstack([bw, zw, zwn, zwn]), np.hstack([zw, -bw, zwn, zwn])]
        rhs += [np.asarray(contain[0], np.float64), -np.asarray(contain[1], np.float64)]
    a_eq = np.zeros((4, 2 * z + 2 * n))
    a_eq[0, 0] = a_eq[1, z] = a_eq[2, z - 1] = a_eq[3, 2 * z - 1] = 1.0  # Bernstein ends are the end coefficients
    b_eq = np.array([start[0], start[1], target[0], target[1]])
    cost = np.concatenate([np.zeros(2 * z), np.ones(2 * n)])
    bounds = [(None, None)] * (2 * z) + [(0, None)] * (2 * n)
    result = linprog(cost, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                     method="highs")
    if result.status != 0:
        return None
    return result.x[:z], result.x[z:2 * z]


def paper_eta(tube_lo, tube_hi, cfg: TubeConfig, s=None) -> float:
    """The paper's eta for 4b-4c (largest violation) on bounds [n, 2] at times s (for 4c's ramp)."""
    ws = np.asarray(cfg.workspace, np.float64)
    width = cfg.min_width if s is None else min_width_at(s, cfg)[:, None]
    return float(max(np.max(ws[:, 0] - tube_lo), np.max(tube_hi - ws[:, 1]), np.max(tube_lo - tube_hi + width)))


def envelope_of(paths, cfg: TubeConfig, s) -> tuple:
    """[n, H, 2] paths -> their per-time min / max at times s (the rover at s = 0), padded to the minimum width."""
    paths = np.asarray(paths, np.float64)
    t = np.concatenate([[0.0], waypoint_times(paths.shape[1])])
    full = np.concatenate([np.zeros((len(paths), 1, 2)), paths], axis=1)
    lo = np.stack([np.interp(s, t, full[..., a].min(axis=0)) for a in range(2)], axis=1)
    hi = np.stack([np.interp(s, t, full[..., a].max(axis=0)) for a in range(2)], axis=1)
    mid, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, cfg.min_width / 2)
    return mid - half, mid + half


def sweep(points, step: float) -> np.ndarray:
    """[..., H, 2] -> the same, each waypoint at most ``step`` per axis from the previous one (the rover at 0)."""
    out = np.array(points, np.float64)
    prev = np.zeros(out.shape[:-2] + (2,))
    for j in range(out.shape[-2]):
        out[..., j, :] = np.clip(out[..., j, :], prev - step, prev + step)
        prev = out[..., j, :]
    return out


def push_clearance(bev: Bev, cfg: TubeConfig) -> float:
    """c = w_min / 2 + margin + one cell: a waypoint this far from every occupied cell leaves room for the tube."""
    return cfg.min_width / 2 + cfg.margin + bev.resolution


def push_clear(paths, bev: Bev, cfg: TubeConfig) -> np.ndarray:
    """[N, H, 2] -> the same, waypoints closer than the clearance to an occupied cell moved clear, then the paths made
    drivable again. How (``cfg.push_mode``): nearest = each waypoint to its own nearest free cell; smooth = that,
    then smoothed inside free boxes round the pushed waypoints (push_smooth)."""
    if cfg.push_mode == "nearest":
        return sweep(bev.inflated(push_clearance(bev, cfg)).nearest_free(paths), cfg.step_m)
    if cfg.push_mode == "smooth":
        return push_smooth(paths, bev, cfg)
    raise ValueError(f"push_mode must be one of {PUSH_MODES}, not {cfg.push_mode!r}")


def sweep_boxes(points, lo, hi, step: float) -> np.ndarray:
    """[N, H, 2] -> the same, each waypoint clipped into its box [lo, hi] and to at most ``step`` per axis from the
    previous one (the rover at 0). Where the two disagree, the box side nearest the rate limit wins."""
    out, prev = np.array(points, np.float64), np.zeros(np.shape(points)[:-2] + (2,))
    for j in range(out.shape[-2]):
        low = np.maximum(lo[..., j, :], prev - step)
        high = np.maximum(np.minimum(hi[..., j, :], prev + step), low)
        prev = np.minimum(np.maximum(out[..., j, :], low), high)
        out[..., j, :] = prev
    return out


def smooth_in_boxes(points, lo, hi, beta: float, step: float, iters: int = 200, rho: float = 1.0) -> np.ndarray:
    """[N, H, 2] -> the same: per path and axis,

        argmin_p  1/2 |p - x|^2 + beta/2 |D2 p|^2    s.t.  lo_j <= p_j <= hi_j,   |p_j - p_{j-1}| <= step,  p_{-1} = 0

    (D1, D2 first and second differences through the rover at 0). Scaled ADMM with z1 = p (boxes) and z2 = D1 p
    (rate); the p-step solves (I + beta D2^T D2 + rho (I + D1^T D1)) p = x + rho (z1 - u1) + rho D1^T (z2 - u2) with
    one precomputed inverse. The result goes through sweep_boxes, so it is in the boxes and drivable exactly."""
    x = np.asarray(points, np.float64)
    h = x.shape[-2]
    eye = np.eye(h)
    d1 = eye - np.eye(h, k=-1)
    d2 = d1[1:] - d1[:-1]
    m_inv = np.linalg.inv(eye + beta * d2.T @ d2 + rho * (eye + d1.T @ d1))
    along = lambda a, v: np.einsum("ij,...jk->...ik", a, v)
    z1, z2 = np.clip(x, lo, hi), np.clip(along(d1, x), -step, step)
    u1, u2 = np.zeros_like(x), np.zeros_like(x)
    p = z1
    for _ in range(iters):
        p = along(m_inv, x + rho * (z1 - u1) + rho * along(d1.T, z2 - u2))
        a1, a2 = p + u1, along(d1, p) + u2
        z1, z2 = np.clip(a1, lo, hi), np.clip(a2, -step, step)
        u1, u2 = a1 - z1, a2 - z2
    return sweep_boxes(p, lo, hi, step)


def push_smooth(paths, bev: Bev, cfg: TubeConfig) -> np.ndarray:
    """[..., H, 2] -> the same. The nearest push, then smoothed without leaving the free space it found:

        y_j  = the nearest push of x_j (then swept),   seed_j = y_j if clear, else the nearest free cell of x_j
        F_j  = grow_box(seed_j) in the grid grown by the clearance c, at most push_smooth_grow_m per side
        p    = argmin 1/2 |p - x|^2 + beta/2 |D2 p|^2   s.t.  p_j in F_j,  |p_j - p_{j-1}| <= 0.25 m   (smooth_in_boxes)

    x is NavDP's own path, so p is the smooth path closest to it whose waypoints stay clear. A waypoint is only
    boxed when the path comes near an obstacle; elsewhere its box is the workspace."""
    near = bev.inflated(push_clearance(bev, cfg))
    shape = np.shape(paths)
    x = np.array(paths, np.float64).reshape(-1, shape[-2], 2)
    free_x = near.nearest_free(x)
    y = sweep(free_x, cfg.step_m)
    (wf0, wf1), (wl0, wl1) = cfg.workspace
    lo = np.broadcast_to(np.array([wf0, wl0]), x.shape).copy()
    hi = np.broadcast_to(np.array([wf1, wl1]), x.shape).copy()
    close = near.occupied_at(x, outside=False) | np.any(np.abs(free_x - x) > 1e-9, axis=-1)
    clear_y = ~near.occupied_at(y, outside=True)
    for k in range(len(x)):
        if not close[k].any():
            continue
        for j in range(x.shape[1]):
            seed = y[k, j] if clear_y[k, j] else free_x[k, j]
            box = grow_box(near, seed, seed, cfg.push_smooth_grow_m)
            lo[k, j], hi[k, j] = (seed, seed) if box is None else box
    out = smooth_in_boxes(x, lo, hi, cfg.push_smooth_beta, cfg.step_m, cfg.push_smooth_iters)
    return out.reshape(shape)


def grow_box(bev: Bev, lo, hi, max_grow_m: float):
    """The free box grown from the seed box [lo, hi] one cell at a time on each side in turn, until blocked or
    max_grow_m -> (lo, hi) in metres; None when the seed itself touches an occupied cell."""
    i0, i1, j0, j1 = bev.cells(lo, hi)
    if bev.blocked(i0, i1, j0, j1):
        return None
    limit = int(max_grow_m / bev.resolution)
    grown, open_ = [0, 0, 0, 0], [True, True, True, True]  # forward-, forward+, left-, left+
    while any(open_):
        for side in range(4):
            if not open_[side]:
                continue
            if grown[side] >= limit:
                open_[side] = False
                continue
            a0, a1, b0, b1 = i0, i1, j0, j1
            if side == 0:
                a0, a1 = i0 - 1, i0
            elif side == 1:
                a0, a1 = i1, i1 + 1
            elif side == 2:
                b0, b1 = j0 - 1, j0
            else:
                b0, b1 = j1, j1 + 1
            if bev.blocked(a0, a1, b0, b1):
                open_[side] = False
                continue
            grown[side] += 1
            i0, i1, j0, j1 = min(i0, a0), max(i1, a1), min(j0, b0), max(j1, b1)
    r, o = bev.resolution, np.asarray(bev.origin)
    return o + np.array([i0, j0]) * r, o + np.array([i1, j1]) * r


def clusters(points, bev: Bev, link_m: float) -> list:
    """[N, 2] waypoints (one per path, same index j) -> clusters (index arrays): points linked when within link_m
    per axis and the box spanning them holds no occupied cell. A cluster whose whole box is not free (it wraps round
    an obstacle) is split into single points."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for k in range(i + 1, n):
            near = np.abs(points[i] - points[k]).max() <= link_m
            if near and not bev.box_blocked(np.minimum(points[i], points[k]), np.maximum(points[i], points[k])):
                parent[find(i)] = find(k)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        members = np.array(members)
        if len(members) > 1 and bev.box_blocked(points[members].min(axis=0), points[members].max(axis=0)):
            out.extend(np.array([i]) for i in members)
        else:
            out.append(members)
    return out


def waypoint_boxes(paths, bev: Bev, cfg: TubeConfig) -> tuple:
    """[N, H, 2] -> (box_id [N, H], F_lo [B, 2], F_hi [B, 2]): the free box of each path's cluster at each waypoint
    (-1 when its waypoint is in an occupied cell). A cluster's box is grown from its points at j and j - 1 (the
    start box for j = 0), or from its points at j alone when that seed is not free."""
    n, horizon = paths.shape[:2]
    s_lo, s_hi = start_box(cfg)
    box_id = np.full((n, horizon), -1, np.int64)
    lows, highs = [], []
    for j in range(horizon):
        cur = paths[:, j]
        for members in clusters(cur, bev, cfg.link_m):
            seed_lo, seed_hi = cur[members].min(axis=0), cur[members].max(axis=0)
            if j == 0:
                wide_lo, wide_hi = np.minimum(seed_lo, s_lo), np.maximum(seed_hi, s_hi)
            else:
                prev = paths[members, j - 1]
                wide_lo, wide_hi = np.minimum(seed_lo, prev.min(axis=0)), np.maximum(seed_hi, prev.max(axis=0))
            box = grow_box(bev, wide_lo, wide_hi, cfg.max_grow_m)
            if box is None:
                box = grow_box(bev, seed_lo, seed_hi, cfg.max_grow_m)
            if box is None:
                continue
            box_id[members, j] = len(lows)
            lows.append(box[0])
            highs.append(box[1])
    return box_id, np.array(lows).reshape(-1, 2), np.array(highs).reshape(-1, 2)


def chain_target(endpoints, box_lo, box_hi, cfg: TubeConfig):
    """T: the endpoints' box grown by target_pad, at least min_width (+ margin) wide, slid inside the last free box
    (``margin`` in). None when that box is too thin."""
    need = cfg.min_width + cfg.margin
    room_lo, room_hi = box_lo + cfg.margin, box_hi - cfg.margin
    if np.any(room_hi - room_lo < need):
        return None
    lo, hi = endpoints.min(axis=0) - cfg.target_pad, endpoints.max(axis=0) + cfg.target_pad
    half = np.minimum(np.maximum((hi - lo) / 2, need / 2), (room_hi - room_lo) / 2)
    mid = np.clip((lo + hi) / 2, room_lo + half, room_hi - half)
    return np.stack([mid - half, mid + half], axis=1)


def _failed(cfg: TubeConfig, target=None) -> Tube:
    nan = np.full((2, cfg.degree + 1), np.nan)
    return Tube(nan, nan, False, float("inf"), np.full((2, 2), np.nan) if target is None else target,
                eta_tol=cfg.eta_tol)


def solve_chain(f_lo, f_hi, paths, cfg: TubeConfig, contain: bool = False) -> Tube:
    """One chain of free boxes ([H, 2] each) and the paths that follow it [n, H, 2] -> their tube. contain: the tube
    must hold the paths' waypoints (for certifying finished paths; the guidance projects its paths into the boxes
    afterwards, so there it only follows them)."""
    horizon, m = cfg.horizon, cfg.samples_per_step
    target = chain_target(paths[:, -1], f_lo[-1], f_hi[-1], cfg)
    if target is None:
        return _failed(cfg)
    s = sample_times(horizon, m)
    k = np.arange(1, m * horizon + 1)
    step = (k - 1) // m                                   # sampled time k lies in step j: (s_j-1, s_j]
    room_lo, room_hi = f_lo[step].copy(), f_hi[step].copy()
    at = (k % m == 0) & (step < horizon - 1)              # a waypoint time also lies in the next step's box
    room_lo[at] = np.maximum(room_lo[at], f_lo[step[at] + 1])
    room_hi[at] = np.minimum(room_hi[at], f_hi[step[at] + 1])
    s_lo, s_hi = start_box(cfg)
    env_lo, env_hi = envelope_of(paths, cfg, s)
    reach = cfg.start_half + cfg.step_m * horizon * s
    coef = []
    for a in range(2):
        solved = _solve_axis((s_lo[a], s_hi[a]), target[a], (env_lo[:, a], env_hi[:, a]), cfg.workspace[a], reach,
                             cfg, s, room_lo[:, a] + cfg.margin, room_hi[:, a] - cfg.margin,
                             centre=(s_lo[a] + s_hi[a]) / 2,
                             contain=(paths[..., a].min(axis=0), paths[..., a].max(axis=0)) if contain else None)
        if solved is None:
            return _failed(cfg, target)
        coef.append(solved)
    tube = Tube(np.stack([coef[0][0], coef[1][0]]), np.stack([coef[0][1], coef[1][1]]), True, 0.0, target,
                eta_tol=cfg.eta_tol)
    # Theorem 3.1: the realised slack on the sampled times (and at s = 0, the start box against the first box) must
    # cover the bounds' largest slope times the sampling gap.
    lo, hi = tube.bounds(s)
    tube.slack = float(min((lo - room_lo).min(), (room_hi - hi).min(), (s_lo - f_lo[0]).min(), (f_hi[0] - s_hi).min()))
    tube.lipschitz = float(cfg.degree * max(np.abs(np.diff(c, axis=1)).max() for c in (tube.lo_coef, tube.hi_coef)))
    tube.gap = 1.0 / (2 * m * horizon)
    dense = np.linspace(0, 1, cfg.dense + 1)[1:]
    tube.eta = paper_eta(*tube.bounds(dense), cfg, dense)
    return tube


def tubes_for(paths, bev: Bev, cfg: TubeConfig = TubeConfig(), push: bool = True) -> tuple:
    """[N, H, 2] paths -> (tubes, group [N], used paths [N, H, 2]): paths with the same chain of free boxes share a
    tube, group[i] its index. ``push`` first moves the paths clear of the obstacles (the guidance does; the
    certificate of finished paths does not)."""
    paths = np.asarray(paths, np.float64)[..., :2]
    if paths.shape[1] != cfg.horizon:
        raise ValueError(f"paths have {paths.shape[1]} waypoints, the tube config {cfg.horizon}")
    n = len(paths)
    used = push_clear(paths, bev, cfg) if push else paths
    if bev.box_blocked(*start_box(cfg)):  # the rover is already inside the grown obstacles: no tube can exist
        tube = _failed(cfg)
        tube.members = list(range(n))
        return [tube], np.zeros(n, np.int64), used
    box_id, f_lo, f_hi = waypoint_boxes(used, bev, cfg)
    keys = [tuple(int(v) for v in row) if (row >= 0).all() else None for row in box_id]
    tubes, group = [], np.zeros(n, np.int64)
    for key in sorted({k for k in keys if k is not None}):
        members = np.flatnonzero([k == key for k in keys])
        idx = np.array(key)
        tube = solve_chain(f_lo[idx], f_hi[idx], used[members], cfg, contain=not push)
        tube.members = members.tolist()
        group[members] = len(tubes)
        tubes.append(tube)
    lost = [i for i, k in enumerate(keys) if k is None]
    if lost:  # a waypoint in an occupied cell: no tube
        tube = _failed(cfg)
        tube.members = lost
        group[lost] = len(tubes)
        tubes.append(tube)
    return tubes, group, used


def inside(paths, tube: Tube, tol: float = 1e-4) -> np.ndarray:
    """[N, H, 2] -> [N] every waypoint inside the tube's box at its time."""
    paths = np.asarray(paths, np.float64)[..., :2]
    if not tube.feasible:
        return np.zeros(len(paths), bool)
    lo, hi = tube.bounds(waypoint_times(paths.shape[1]))
    return ((paths >= lo - tol) & (paths <= hi + tol)).all(axis=(1, 2))
