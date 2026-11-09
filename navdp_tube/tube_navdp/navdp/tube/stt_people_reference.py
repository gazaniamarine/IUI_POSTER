"""Reference implementation of the homotopy-fixed, probabilistic STT.

Companion to STT_HOMOTOPY_SPEC.md (section numbers below refer to it).
Pure numpy + scipy (HiGHS through scipy.optimize.linprog).

Frame: every quantity is in the cycle's local frame -- origin at y(t_k), x-axis
along NavDP's heading at t_k -- and time tau = t - t_k in [0, t_c].

Run:  python stt_reference.py      (runs the acceptance checks of section 11)
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace
from math import comb

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Parameters (spec section 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class STTParams:
    t_c: float = 4.8            # horizon [s]
    R: int = 24                 # time cells; eps_t = t_c / (2R)
    degree: int = 8             # Bernstein degree of each tube bound
    m: float = 0.10             # start margin [m]
    w_min: float = 0.20         # minimum tube width per axis [m]
    v_bar: float = 1.0          # bound on |d gamma / d tau| [m/s]
    r_T: float = 0.60           # target half-width [m]
    use_target: bool = True
    Y_lo: tuple = (-1.0, -3.0)  # workspace box, local frame [m]
    Y_hi: tuple = (5.5, 3.0)
    w_track: float = 1.0        # objective: weight on following NavDP
    lam_L: float = 0.05         # objective: weight on L_bar
    mu: float = 2.0             # objective: weight on extra margin eta
    eta_max: float = 0.5

    @property
    def eps(self) -> float:
        return self.t_c / (2 * self.R)

    def taus(self) -> np.ndarray:
        """Cell centres tau_r = (2r+1) eps. Cell r = [2r eps, 2(r+1) eps]."""
        return self.eps * (2 * np.arange(self.R) + 1)


# ---------------------------------------------------------------------------
# Bernstein tube (spec section 5.1)
# ---------------------------------------------------------------------------


def bernstein(tau, P: STTParams, degree=None):
    """B[t, n] = C(d,n) s^n (1-s)^(d-n), s = tau / t_c."""
    d = P.degree if degree is None else degree
    s = np.clip(np.atleast_1d(tau) / P.t_c, 0.0, 1.0)
    return np.stack([comb(d, n) * s ** n * (1 - s) ** (d - n) for n in range(d + 1)], -1)


def tube_eval(b, tau, P: STTParams, deriv=False):
    """b: control points (2 axes, 2 bounds, d+1). Returns (T, 2, 2) values or slopes."""
    if not deriv:
        return np.einsum("tn,ibn->tib", bernstein(tau, P), b)
    d = P.degree
    db = d * np.diff(b, axis=-1) / P.t_c                       # (2, 2, d)
    return np.einsum("tn,ibn->tib", bernstein(tau, P, d - 1), db)


# ---------------------------------------------------------------------------
# Obstacles (spec section 4)
# ---------------------------------------------------------------------------


@dataclass
class Ball:
    """Ball with piecewise-linear centre c(tau) and radius rho(tau) (already inflated)."""
    node_t: np.ndarray          # (K,) increasing, covers [0, t_c]
    node_c: np.ndarray          # (K, 2)
    node_r: np.ndarray          # (K,)
    group: int = -1             # person id; all modes of a person share a class
    is_person: bool = True

    def c(self, tau):
        tau = np.atleast_1d(tau)
        return np.stack([np.interp(tau, self.node_t, self.node_c[:, i]) for i in range(2)], -1)

    def r(self, tau):
        return np.interp(np.atleast_1d(tau), self.node_t, self.node_r)


def person_balls(p0, modes, sigma, q_hat, protected, *, r_det=0.10, a_max=3.0,
                 dt=0.4, r_inflate=0.55, group=0):
    """One Ball per protected mode of one person (spec 4.2).

    p0 (2,) current position; modes (K, 12, 2) centres at tau = 0.4..4.8;
    sigma (K, 12) per-step scale; protected = K_j. Node 0 = p0 with radius r_det.
    hat_rho_s = max_{|s'-s|<=1} rho_s' + a_max dt^2 / 8  (Lemma 2), then + r_inflate.
    """
    t_nodes = dt * np.arange(13)
    out = []
    for k in protected:
        c = np.vstack([p0[None], modes[k]])
        rho = np.concatenate([[r_det], q_hat * sigma[k]])
        nb = np.maximum(rho, np.maximum(np.r_[rho[1:], rho[-1]], np.r_[rho[0], rho[:-1]]))
        out.append(Ball(t_nodes, c, nb + a_max * dt ** 2 / 8.0 + r_inflate, group=group))
    return out


def protected_set(pi, delta_m):
    """Smallest set of most likely modes whose dropped chance is <= delta_m (spec 3.1)."""
    keep, dropped = [], 1.0
    for k in np.argsort(-pi):
        keep.append(int(k)); dropped -= pi[k]
        if dropped <= delta_m:
            break
    return sorted(keep)


@dataclass
class ObstacleArrays:
    """Per-cycle obstacle data at the cell centres (computed once, shared by all candidates)."""
    C: np.ndarray       # (O, R, 2) centres at tau_r
    Rad: np.ndarray     # (O, R)    radii at tau_r
    Vel: np.ndarray     # (O, R, 2) max |dc_i/dtau| over segments meeting cell r
    Rdot: np.ndarray    # (O, R)    max |drho/dtau| over segments meeting cell r
    group: np.ndarray   # (O,)
    is_person: np.ndarray


def obstacle_arrays(balls, P: STTParams) -> ObstacleArrays:
    taus, e = P.taus(), P.eps
    O, R = len(balls), P.R
    C = np.zeros((O, R, 2)); Rad = np.zeros((O, R)); Vel = np.zeros((O, R, 2)); Rd = np.zeros((O, R))
    for o, b in enumerate(balls):
        C[o], Rad[o] = b.c(taus), b.r(taus)
        dtn = np.diff(b.node_t)
        vel = np.abs(np.diff(b.node_c, axis=0)) / dtn[:, None]
        rdot = np.abs(np.diff(b.node_r)) / dtn
        lo, hi = b.node_t[:-1], b.node_t[1:]
        hit = (hi[None] >= (taus - e)[:, None]) & (lo[None] <= (taus + e)[:, None])   # (R, segs)
        Vel[o] = np.stack([np.where(hit, vel[None, :, i], 0).max(1) for i in range(2)], -1)
        Rd[o] = np.where(hit, rdot[None], 0).max(1)
    return ObstacleArrays(C, Rad, Vel, Rd,
                          np.array([b.group for b in balls]), np.array([b.is_person for b in balls]))


# ---------------------------------------------------------------------------
# Side assignment / homotopy (spec section 6)
# ---------------------------------------------------------------------------
# Option index -> (axis, side). side +1 "above": tube on the +axis side of the ball.
# Heading frame: 0 behind (yield), 1 ahead, 2 right, 3 left.
OPT = np.array([(0, -1), (0, +1), (1, -1), (1, +1)])
CLASS_ALLOWED = {"free": [0, 1, 2, 3], "left": [3, 0, 1], "right": [2, 0, 1], "yield": [0]}


def option_margins(arr: ObstacleArrays, y_hat):
    """M[o, r, q]: signed clearance of the nominal point for option q (positive = satisfied)."""
    yh = y_hat[None]                                   # (1, R, 2)
    M = np.empty(arr.Rad.shape + (4,))
    for q, (i, s) in enumerate(OPT):
        M[..., q] = (yh[..., i] - arr.C[..., i] - arr.Rad) if s > 0 else (arr.C[..., i] - arr.Rad - yh[..., i])
    return M


def assign_sides(arr: ObstacleArrays, y_hat, classes=None):
    """Option index per (obstacle, cell). classes: dict group -> class name.

    'free': argmax margin over all 4 options (the nominal homotopy).
    'left'/'right': use an x-option (behind / ahead) where the nominal path already
    clears it (margin > 0, best one), else the class's lateral option.
    'yield': always behind. Static balls are always 'free'.
    """
    classes = classes or {}
    M = option_margins(arr, y_hat)
    Q = np.argmax(M, axis=-1)                          # free by default
    for o in range(len(arr.group)):
        cls = classes.get(int(arr.group[o]), "free") if arr.is_person[o] else "free"
        if cls == "free":
            continue
        if cls == "yield":
            Q[o] = 0
            continue
        lateral = CLASS_ALLOWED[cls][0]
        mx = M[o][:, :2]                               # behind, ahead
        best_x = np.argmax(mx, axis=-1)
        Q[o] = np.where(mx.max(-1) > 0, best_x, lateral)
    return Q


# ---------------------------------------------------------------------------
# The LP (spec section 5)
# ---------------------------------------------------------------------------


class Layout:
    """x = [b (2 axes x 2 bounds x (d+1)) | L_bar | eta | z (2 x R)]."""

    def __init__(self, P: STTParams):
        self.nb = P.degree + 1
        self.L = 4 * self.nb
        self.eta = self.L + 1
        self.z0 = self.eta + 1
        self.n = self.z0 + 2 * P.R

    def b(self, i, k):          # k: 0 lower, 1 upper
        return (2 * i + k) * self.nb

    def z(self, i, r, R):
        return self.z0 + i * R + r


TOL = 1e-6   # row tightening [m] so the solver's ~1e-9 feasibility error cannot break certify()


class _Rows:
    """Sparse row builder: add(cols, vals, rhs)."""

    def __init__(self):
        self.r, self.c, self.v, self.rhs = [], [], [], []

    def add(self, cols, vals, rhs):
        k = len(self.rhs)
        self.r.extend([k] * len(cols)); self.c.extend(cols); self.v.extend(vals); self.rhs.append(rhs)

    def matrix(self, n):
        A = sparse.csr_matrix((self.v, (self.r, self.c)), shape=(len(self.rhs), n))
        return A, np.array(self.rhs)


def solve_tube_lp(P: STTParams, arr: ObstacleArrays, Q, y_hat, y0=np.zeros(2),
                  use_certificate=True):
    """Build and solve the homotopy-fixed LP for option assignment Q (O, R).

    Returns dict(ok, b (2,2,d+1), L_bar, eta, cost, solve_ms, n_vars, n_rows).
    """
    lay, R, nb, d = Layout(P), P.R, P.degree + 1, P.degree
    eps = P.eps if use_certificate else 0.0
    tol = TOL if use_certificate else 0.0      # rows tightened so the exact re-check passes
    B = bernstein(P.taus(), P)                          # (R, nb)
    rows = _Rows()
    cols = lambda i, k: list(range(lay.b(i, k), lay.b(i, k) + nb))

    for i in range(2):
        # L1 start: b_{i,L,0} <= y0 - m ; b_{i,U,0} >= y0 + m      (gamma(0) = b_0)
        rows.add([lay.b(i, 0)], [1.0], y0[i] - P.m - tol)
        rows.add([lay.b(i, 1)], [-1.0], -(y0[i] + P.m) - tol)
        # L2 target: gamma(t_c) = b_d inside [y_hat(t_c) +- r_T]
        if P.use_target:
            rows.add([lay.b(i, 0) + d], [-1.0], -(y_hat[-1, i] - P.r_T) - tol)
            rows.add([lay.b(i, 1) + d], [1.0], y_hat[-1, i] + P.r_T - tol)
        for r in range(R):
            ph = list(B[r])
            # L3 workspace, L4 width (each with certificate margin, eta >= 0)
            rows.add(cols(i, 0) + [lay.L, lay.eta], [-v for v in ph] + [eps, 1.0], -P.Y_lo[i] - tol)
            rows.add(cols(i, 1) + [lay.L, lay.eta], ph + [eps, 1.0], P.Y_hi[i] - tol)
            rows.add(cols(i, 0) + cols(i, 1) + [lay.L, lay.eta],
                     ph + [-v for v in ph] + [2 * eps, 1.0], -P.w_min - tol)
            # objective helpers z_{i,r} >= |mid_i(tau_r) - y_hat_i(tau_r)|
            half = [0.5 * v for v in ph]
            rows.add(cols(i, 0) + cols(i, 1) + [lay.z(i, r, R)], half + half + [-1.0], y_hat[r, i])
            rows.add(cols(i, 0) + cols(i, 1) + [lay.z(i, r, R)],
                     [-v for v in half + half] + [-1.0], -y_hat[r, i])
        # L7 Lipschitz: (d / t_c) |b_{n+1} - b_n| <= L_bar  (exact Bernstein derivative bound)
        for k in range(2):
            base = lay.b(i, k)
            for n in range(d):
                g = d / P.t_c
                rows.add([base + n + 1, base + n, lay.L], [g, -g, -1.0], 0.0)
                rows.add([base + n + 1, base + n, lay.L], [-g, g, -1.0], 0.0)

    # L5 avoid, one row per (obstacle, cell), per-row certificate margin
    for o in range(Q.shape[0]):
        for r in range(R):
            i, s = OPT[Q[o, r]]
            obs = eps * (arr.Vel[o, r, i] + arr.Rdot[o, r])
            if s < 0:       # below: gamma_{i,U}(tau_r) <= c_i - rho
                rows.add(cols(i, 1) + [lay.L, lay.eta], list(B[r]) + [eps, 1.0],
                         arr.C[o, r, i] - arr.Rad[o, r] - obs - tol)
            else:           # above: gamma_{i,L}(tau_r) >= c_i + rho
                rows.add(cols(i, 0) + [lay.L, lay.eta], list(-B[r]) + [eps, 1.0],
                         -(arr.C[o, r, i] + arr.Rad[o, r]) - obs - tol)

    A, bvec = rows.matrix(lay.n)
    cost = np.zeros(lay.n)
    cost[lay.z0:] = P.w_track; cost[lay.L] = P.lam_L; cost[lay.eta] = -P.mu
    bounds = [(None, None)] * (4 * nb) + [(0, P.v_bar), (0, P.eta_max)] + [(0, None)] * (2 * R)
    t0 = time.perf_counter()
    res = linprog(cost, A_ub=A, b_ub=bvec, bounds=bounds, method="highs")
    out = dict(ok=res.status == 0, status=res.status, solve_ms=1e3 * (time.perf_counter() - t0),
               n_vars=lay.n, n_rows=A.shape[0])
    if out["ok"]:
        x = res.x
        out["b"] = np.array([[x[lay.b(i, k):lay.b(i, k) + nb] for k in range(2)] for i in range(2)])
        out.update(L_bar=x[lay.L], eta=x[lay.eta], cost=res.fun)
        if use_certificate:
            out["cert_margin"], out["L_true"] = certify(P, arr, Q, out["b"], y_hat, y0)
            out["ok"] = out["cert_margin"] >= 0.0
    return out


def certify(P: STTParams, arr: ObstacleArrays, Q, b, y_hat, y0=np.zeros(2)):
    """Exact float64 re-check of Theorem 2's hypotheses (spec 5.5).

    Recomputes L_true = (d / t_c) max |b_{n+1} - b_n| from the control points and
    evaluates every certificate row with it (eta = 0). Returns (min margin, L_true);
    the tube is certified iff min margin >= 0.
    """
    d, e = P.degree, P.eps
    L = d * np.abs(np.diff(b, axis=-1)).max() / P.t_c
    g = tube_eval(b, P.taus(), P)                      # (R, 2, 2)
    lo, hi = g[:, :, 0], g[:, :, 1]
    m = [np.min(y0 - P.m - b[:, 0, 0]), np.min(b[:, 1, 0] - y0 - P.m)]   # start
    if P.use_target:
        m += [np.min(b[:, 0, d] - (y_hat[-1] - P.r_T)), np.min(y_hat[-1] + P.r_T - b[:, 1, d])]
    m += [np.min(lo - np.array(P.Y_lo) - e * L), np.min(np.array(P.Y_hi) - hi - e * L),
          np.min(hi - lo - P.w_min - 2 * e * L)]
    ax, sd = OPT[Q][..., 0], OPT[Q][..., 1]           # (O, R)
    rr = np.arange(P.R)[None].repeat(Q.shape[0], 0)
    c = np.take_along_axis(arr.C, ax[..., None], -1)[..., 0]
    v = np.take_along_axis(arr.Vel, ax[..., None], -1)[..., 0]
    gU, gL = hi[rr, ax], lo[rr, ax]
    marg = np.where(sd < 0, c - arr.Rad - gU, gL - c - arr.Rad) - e * (L + v + arr.Rdot)
    if marg.size:
        m.append(marg.min())
    return float(np.min(m)), float(L)


# ---------------------------------------------------------------------------
# Parallel homotopies (spec section 7)
# ---------------------------------------------------------------------------


def critical_groups(arr: ObstacleArrays, y_hat, n_h=2):
    """The n_h people with the smallest clearance to the nominal path."""
    clr = np.linalg.norm(arr.C - y_hat[None], axis=-1) - arr.Rad          # (O, R)
    per = {}
    for o in np.where(arr.is_person)[0]:
        g = int(arr.group[o]); per[g] = min(per.get(g, np.inf), clr[o].min())
    return [g for g, _ in sorted(per.items(), key=lambda kv: kv[1])[:n_h]]


def candidate_classes(arr: ObstacleArrays, y_hat, n_h=2, proposals=()):
    """Nominal, then {left, right, yield}^n_h over critical people, then learned proposals."""
    groups = critical_groups(arr, y_hat, n_h)
    cands = [dict()] + [dict(zip(groups, c)) for c in itertools.product(["left", "right", "yield"],
                                                                        repeat=len(groups))]
    cands += [dict(p) for p in proposals]
    out, seen = [], set()
    for c in cands:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def solve_parallel(P: STTParams, arr: ObstacleArrays, y_hat, candidates, prev=None,
                   switch_cost=0.5, relax_target=True):
    """One LP per candidate; return the cheapest certified tube (with hysteresis).

    If no candidate is feasible and relax_target, retry all without the target.
    On the robot, run the LPs in a process pool (they are independent).
    """
    for PP in ([P, replace(P, use_target=False)] if relax_target else [P]):
        best = None
        for cls in candidates:
            out = solve_tube_lp(PP, arr, assign_sides(arr, y_hat, cls), y_hat)
            if out["ok"]:
                J = out["cost"] + (switch_cost if (prev is not None and cls != prev) else 0.0)
                if best is None or J < best[0]:
                    best = (J, cls, out)
        if best is not None:
            J, cls, out = best
            out.update(classes=cls, J=J, used_target=PP.use_target)
            return out
    return None


# ---------------------------------------------------------------------------
# Dense verifier (spec section 11)
# ---------------------------------------------------------------------------


def verify_dense(b, P: STTParams, balls, dt=1e-3):
    """Check every STT condition on a 1 ms grid with exact box-ball distance."""
    tau = np.arange(0.0, P.t_c + 1e-12, dt)
    g = tube_eval(b, tau, P)
    lo, hi = g[:, :, 0], g[:, :, 1]
    v = dict(width=int(np.sum(hi - lo <= 0)),
             workspace=int(np.sum((lo < np.array(P.Y_lo)) | (hi > np.array(P.Y_hi)))),
             avoid=0, min_clearance=np.inf)
    for ball in balls:
        c, r = ball.c(tau), ball.r(tau)
        clr = np.linalg.norm(np.maximum(0.0, np.maximum(lo - c, c - hi)), axis=1) - r
        v["avoid"] += int(np.sum(clr < 0)); v["min_clearance"] = min(v["min_clearance"], float(clr.min()))
    v["max_slope"] = float(np.abs(tube_eval(b, tau, P, deriv=True)).max())
    return v


# ---------------------------------------------------------------------------
# Controller (spec section 8)
# ---------------------------------------------------------------------------


def stt_control(z, b, tau, P: STTParams, kappa=0.5):
    """Das-Basu-Jagtap funnel law, N = 1, with PER-COMPONENT xi (spec 8.2):

        e_i = (2 z_i - gU_i - gL_i) / (gU_i - gL_i)
        eps_i = ln((1 + e_i) / (1 - e_i))
        xi_i = 4 / ((gU_i - gL_i) (1 - e_i^2))
        u_i = -kappa xi_i eps_i
    """
    g = tube_eval(b, tau, P)[0]
    gL, gU = g[:, 0], g[:, 1]
    gd = gU - gL
    e = (2 * z - (gU + gL)) / gd
    if np.any(np.abs(e) >= 1):
        raise RuntimeError(f"left tube: e={e}")
    return -kappa * (4.0 / (gd * (1 - e ** 2))) * np.log((1 + e) / (1 - e)), e


def simulate_closed_loop(b, P, z0=np.zeros(2), w_bar=0.1, dt=1e-3, kappa=0.5, seed=0):
    rng = np.random.default_rng(seed)
    z, w = z0.copy(), np.zeros(2)
    max_e = max_u = 0.0
    for tau in np.arange(0.0, P.t_c, dt):
        u, e = stt_control(z, b, tau, P, kappa)
        w = np.clip(0.98 * w + 0.1 * w_bar * rng.uniform(-1, 1, 2), -w_bar, w_bar)
        z = z + dt * (u + w)
        max_e, max_u = max(max_e, np.abs(e).max()), max(max_u, np.abs(u).max())
    return max_e, max_u


# ---------------------------------------------------------------------------
# Acceptance checks (spec section 11)
# ---------------------------------------------------------------------------


def make_scene(rng, n_people, P: STTParams):
    """Synthetic stress scene: NavDP path ahead, 1-5 people, 5 protected modes each."""
    taus = P.taus()
    y_hat = np.stack([0.8 * taus, 0.15 * np.sin(taus)], -1)
    balls = []
    for j in range(n_people):
        p0 = np.array([rng.uniform(2.0, 5.0), rng.uniform(-2.5, 2.5)])
        v = rng.normal(0, 0.6, 2)
        modes = np.stack([p0 + np.outer(0.4 * np.arange(1, 13), v + rng.normal(0, 0.25, 2))
                          for _ in range(20)])
        pi = rng.dirichlet(np.ones(20) * 0.5)
        sigma = np.tile(0.05 + 0.03 * np.arange(1, 13), (20, 1))
        balls += person_balls(p0, modes, sigma, 1.5, protected_set(pi, 0.2)[:5], group=j)
    return y_hat, balls


def head_on(P):
    taus = P.taus()
    y_hat = np.stack([0.8 * taus, 0 * taus], -1)
    p0, v = np.array([3.5, 0.05]), np.array([-0.6, 0.0])
    modes = np.stack([p0 + np.outer(0.4 * np.arange(1, 13), v)] * 20)
    sigma = np.tile(0.05 + 0.02 * np.arange(1, 13), (20, 1))
    return y_hat, person_balls(p0, modes, sigma, 1.5, [0], group=0)


if __name__ == "__main__":
    P = STTParams()

    # T1 certificate soundness + T2 Lipschitz bound, nominal homotopy
    rng = np.random.default_rng(1)
    feas = viol = 0; ms = []
    scenes = [make_scene(rng, int(rng.integers(1, 6)), P) for _ in range(60)]
    for y_hat, balls in scenes:
        arr = obstacle_arrays(balls, P)
        out = solve_tube_lp(P, arr, assign_sides(arr, y_hat), y_hat)
        if out["ok"]:
            feas += 1; ms.append(out["solve_ms"])
            v = verify_dense(out["b"], P, balls)
            viol += v["avoid"] + v["width"] + v["workspace"]
            assert v["max_slope"] <= out["L_true"] + 1e-12, "T2 Lipschitz bound violated"
    print(f"T1 nominal homotopy: {feas}/60 feasible, dense violations = {viol}; T2 slope bound holds")
    print(f"   LP solve: median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms "
          f"({out['n_vars']} vars, ~{out['n_rows']} rows)")

    # T1b certificate is necessary: coarse sampling without margins violates in between
    Pc = replace(P, R=6); bad = tried = 0
    for y_hat, balls in scenes:
        arr = obstacle_arrays(balls, Pc)
        out = solve_tube_lp(Pc, arr, assign_sides(arr, y_hat[::4]), y_hat[::4], use_certificate=False)
        if out["ok"]:
            tried += 1
            v = verify_dense(out["b"], Pc, balls)
            bad += (v["avoid"] + v["width"] + v["workspace"]) > 0
    print(f"T1b no certificate, R=6: {bad}/{tried} feasible tubes violate between samples")

    # T7 parallel homotopies
    par = pviol = 0; walls = []; ncs = []
    for y_hat, balls in scenes:
        t0 = time.perf_counter()
        arr = obstacle_arrays(balls, P)
        cands = candidate_classes(arr, y_hat, n_h=2); ncs.append(len(cands))
        best = solve_parallel(P, arr, y_hat, cands)
        walls.append(1e3 * (time.perf_counter() - t0))
        if best is not None:
            par += 1; pviol += verify_dense(best["b"], P, balls)["avoid"]
    print(f"T7 parallel homotopies: {feas}/60 -> {par}/60 feasible (<= {max(ncs)} candidates, "
          f"target relaxed if needed), dense violations = {pviol}")
    print(f"   sequential wall time per cycle: median {np.median(walls):.0f} ms, "
          f"p95 {np.percentile(walls, 95):.0f} ms (divide by workers when parallel)")

    # T7b head-on person on the nominal path
    y_hat, balls = head_on(P)
    arr = obstacle_arrays(balls, P)
    for cls in ["free", "left", "right", "yield"]:
        o = solve_tube_lp(P, arr, assign_sides(arr, y_hat, {0: cls}), y_hat)
        msg = "infeasible"
        if o["ok"]:
            vv = verify_dense(o["b"], P, balls)
            msg = f"cost {o['cost']:.2f}, eta {o['eta']:.3f}, min clearance {vv['min_clearance']:.3f} m, violations {vv['avoid']}"
        print(f"T7b head-on, class={cls:5s}: {msg}")
    best = solve_parallel(P, arr, y_hat, candidate_classes(arr, y_hat))
    print(f"T7b head-on, solve_parallel picks {best['classes']} (target kept: {best['used_target']})")

    # T3 closed loop inside the tube
    for y_hat, balls in scenes:
        arr = obstacle_arrays(balls, P)
        out = solve_tube_lp(P, arr, assign_sides(arr, y_hat), y_hat)
        if out["ok"]:
            break
    runs = [simulate_closed_loop(out["b"], P, seed=s) for s in range(20)]
    print(f"T3 controller: max |e| over 20 noisy runs = {max(r[0] for r in runs):.3f} (< 1 = inside), "
          f"max |u| = {max(r[1] for r in runs):.2f} m/s")

    # T4 bounded switch (Lemma 4): e at tau = 0 of a fresh tube
    gd0 = out["b"][:, 1, 0] - out["b"][:, 0, 0]
    e0 = (2 * 0 - (out["b"][:, 1, 0] + out["b"][:, 0, 0])) / gd0
    print(f"T4 switch: |e(0)| = {np.abs(e0).round(3)} <= 1 - 2m/gamma_d = {(1 - 2 * P.m / gd0).round(3)}")
