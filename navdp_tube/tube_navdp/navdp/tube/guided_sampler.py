"""One-pass, score-guided denoising of NavDP's paths into spatiotemporal tubes built on a BEV occupancy grid.

NavDP is an epsilon-predicting DDPM. At step k, with abar = alphas_cumprod[k], its clean-path estimate is
x0 = (x - sqrt(1 - abar) eps) / sqrt(abar). From the split step on, x0 is pulled toward the tube,
x0' = x0 + alpha (P(x0) - x0) with P the step-limited projection into the tube's boxes (``pull_into_boxes``), and the
noise is recomputed so the scheduler lands on it:

    eps' = (x - sqrt(abar) x0') / sqrt(1 - abar)  =  eps - sqrt(abar) / sqrt(1 - abar) (x0' - x0)

This is score guidance: the score is -eps / sqrt(1 - abar), so

    score' = score + sqrt(abar) / (1 - abar) (x0' - x0)  =  score - lambda grad_x E,
    E(x0) = 1/2 dist(x0, tube)^2,   grad_x0 E = x0 - P(x0),   dx0/dx ~ 1 / sqrt(abar),   lambda = alpha abar / (1 - abar)

(exact when x0 is not clipped to [-1, 1]; the code uses the first form, which is exact in any case). alpha rises from
alpha0 at the split step to 1, and is 1 for the last ``project_last`` steps, where the last step's output is x0'
itself, so the finished paths are the projection. The tubes (navdp/tube/stt.py) are built once, at the split step,
from x0 pushed clear of the occupied cells; no side of any obstacle is chosen.
"""
from __future__ import annotations #Ensures class annotations are not evaluated at runtime and just saved as a string

from dataclasses import dataclass, field, replace #Decorators with __init__, __repr__, __eq__, etc. for classes; field() for default values; replace() to create a new instance with some fields changed

import numpy as np
import torch

from navdp.tube.stt import Bev, TubeConfig, inside, sweep, tubes_for, waypoint_times

# Two options on top (both off by default, so the loop above is unchanged until they are set):
#
#  consistency (mu > 0): the previous plan, moved into the new body frame and shifted by the waypoints already
#    driven (x_prev, m <= H waypoints, from the driver), is a second energy, as X-NavDP's temporal-consistency term:
#        E_c(x0) = 1/2 sum_j w_j |x0_j - x_prev_j|^2,   w_j = mu gamma^j for j < m, else 0   (mu <= 1)
#    one gradient step of size 1 on E_c is x0_j <- x0_j + w_j (x_prev_j - x0_j), applied from step
#    ``consistency_from`` on, before the tube pull (so the tube, i.e. safety, has the last word).
#
#  smooth (beta > 0): the projection P into the boxes becomes the smoothest path in them, per sample and axis
#        S(x) = argmin_p  1/2 |p - x|^2 + beta/2 |D2 p|^2   s.t.  lo_j <= p_j <= hi_j,  |p_j - p_{j-1}| <= 0.25 m
#    (D2 = second differences, the rover at p_{-1} = 0), solved by ADMM and finished with the exact sweep, so the
#    result is in the boxes as before. A sample with no tube (lo = hi = its pushed path) is not pinned to that path:
#    it gets the same problem with no boxes and its pushed path as x, i.e. the pushed path smoothed.
#
#  rebuild: the tubes are built again at every guided step from that step's clean-path estimate (not only once at
#    the split). With an early split (1 or 2) the sample can still change its way round an obstacle, and the tubes
#    follow it. The tubes of the last step are the ones the finished paths are certified against.
#
#  lyapunov (LyaGuide, Zhang et al., arXiv 2607.14272, discrete form): V(x0) = 1/2 dist^2(x0, B), B the product of
#    the sample's boxes. With P the projection used for the pull (the clamp, or S when smooth > 0) and D = |x0 - P|,
#    the pull x0' = x0 + a (P - x0) has  V(x0') <= 1/2 (1 - a)^2 D^2.  The Lyapunov condition between guided steps,
#        V(x0'_k) <= (1 - delta) V(x0'_{k-1})                      (V of the last step's guided estimate, current boxes)
#    holds for every  a >= a_req = max(0, 1 - sqrt(2 (1 - delta) V_prev / D^2)).  The pull is a = max(a_cand, a_req):
#    the candidate guidance a_cand is corrected only when it would not decrease V fast enough (the pseudo-projection
#    of their Theorem 3, along -grad V = P - x0). a_cand = 0 ("min": no more pull than the condition needs) or the
#    alpha ramp ("ramp"). The last ``project_last`` steps still project exactly (V = 0).
#    In obstacle mode (no tubes) the set is the free cells of the grown grid: V(x0) = 1/2 sum_j dist^2(x0_j, free),
#    P the obstacle pull's target (nearest free cell per waypoint, swept to 0.25 m per step, smoothed when smooth > 0),
#    and the pull is raised to a_req only where V(x0) > (1 - delta) V_prev. The bound V(x0') <= 1/2 (1 - a)^2 D^2
#    holds when P lies in free cells (the sweep or the smoothing can put a waypoint back in one; then it is
#    approximate, and the reported lyapunov_v shows it). The grid does not change during one plan, so V_prev is
#    simply V after the last guided step.
#
#  guide_until (u > 0): the tube / obstacle pull runs only at loop indices split <= i < u; the steps after are NavDP's
#    own (the consistency pull, if on, still runs). The alpha ramp and ``project_last`` apply to that window, so its
#    last steps project exactly and the model then denoises on by itself. To see whether the model keeps the guided
#    paths clear, the result always has (when the grid has obstacles), with V = 1/2 sum_j dist^2(x_j, free cells):
#        model_v [steps]: mean V of the model's own clean-path estimate at every step, before any guidance
#        final_v [N]:     V of every finished path;  final_in_cells [N]: it has a waypoint in an occupied cell
#    (model_v only when guide_until > 0 or with ``record``: it costs a copy off the GPU at every step.)
#
#  push_step / push_margin (obstacle mode; 0.25 and 0 = as before):
#    push_margin m: the push (and the Lyapunov V) uses the grid grown by m more, so waypoints land m past the
#      obstacles' edge instead of on it (the driver checks whole segments every 5 cm, so paths on the edge fail it).
#    push_step m > 0.25: a pushed path may move up to m per waypoint and axis instead of NavDP's 0.25, so a waypoint
#      can be pushed sideways further than the sweep allowed. NavDP's moves are in [-1, 1] (= 0.25 m) and its
#      scheduler clips its clean estimate to that range at every step; with push_step > 0.25 the clip is done here
#      instead ([-1, 1] at unguided steps, exactly as the scheduler did; [-4 push_step, 4 push_step] at guided ones)
#      and the scheduler's own clip is off for the call. Moves past [-1, 1] are outside what NavDP was trained on:
#      the model and the critic see them in the next steps.
#
#  No tubes on the finished paths: navdp / obstacle mode is pure score-guided denoising (stt.tubes_for is never
#    called there) and nothing is certified; the bev_map driver checks the chosen path against its map. Only tube
#    mode builds tubes (for its guidance), and certifies the finished paths with those same tubes.

MODES = ("navdp", "obstacle", "tube")
# I have had idea of 1,3,4 ye saala 2 apne dimaag se banaya hai claude
# 1. navdp:    NavDP's own loop and choice (best critic value); no tubes, nothing certified.
# 3. obstacle: from step 4, push waypoints out of the occupied cells (no tubes); NavDP's own choice; nothing
#              certified.
# 4. tube:     at step 4 build the tubes (one per chain of free boxes, no sides); pull samples into them (exact in
#              the last 2 steps); choose the best-valued certified candidate, else NavDP's own best.
@dataclass
class GuideConfig:
    mode: str = "tube"
    split: int = 4                 # loop index (of the scheduler's steps) at which the tubes are built
    alpha0: float = 0.3            # pull strength at the split step; rises linearly to 1
    project_last: int = 2          # the last steps use alpha = 1
    tube: TubeConfig = field(default_factory=TubeConfig)
    consistency: float = 0.0       # mu: pull toward the previous plan (0 = off, the first version); at most 1
    consistency_decay: float = 0.9 # gamma: w_j = mu gamma^j, the near waypoints are held most
    consistency_from: int = 0      # loop index from which the previous plan pulls
    smooth: float = 0.0            # beta: smooth projection into the boxes (0 = the per-waypoint clamp, as before)
    smooth_iters: int = 100        # ADMM iterations of the smooth projection
    rebuild: bool = False          # build the tubes again at every guided step (False = once, at the split)
    lyapunov: str = "off"          # off (the alpha ramp) | min (a = a_req) | ramp (a = max(alpha ramp, a_req))
    lyapunov_delta: float = 0.3    # delta: V must fall by at least this fraction per guided step
    guide_until: int = 0           # pull (tube / obstacle) only before this loop index; 0 = to the last step (as before)
    push_step: float = 0.25        # obstacle mode: most a pushed path moves per waypoint and axis (0.25 = NavDP's own)
    push_margin: float = 0.0       # obstacle mode: push this far past the grown obstacles (0 = to their edge, as before)


LYAPUNOV_MODES = ("off", "min", "ramp")


def positions(moves: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(moves[..., :2] / 4.0, dim=1)
# Only forward and left concerns us and then we dont care abouut anything 
# The torch.cumulative sum because output is [-1,1] so it ends up with in ms. I still dont understnd but directly copied from NavDP

def moves_from(points: torch.Tensor, moves: torch.Tensor, step: float = 0.25) -> torch.Tensor:
    """step: the most a waypoint moves per axis (0.25 m = NavDP's moves in [-1, 1])."""
    start = torch.zeros_like(points[:, :1])
    out = moves.clone()
    out[..., :2] = (4.0 * torch.diff(points, dim=1, prepend=start)).clamp(-4.0 * step, 4.0 * step)
    return out
#The opposite of positions, we get the moves from the points

def sweep_into_boxes(points, lo, hi, step: float = 0.25):
    out, prev = points.clone(), torch.zeros_like(points[:, 0]) #Taking into account whatevever was going on previously
    for j in range(points.shape[1]):
        low = torch.maximum(lo[:, j], prev - step)
        high = torch.minimum(hi[:, j], prev + step)
        high = torch.maximum(high, low)  # empty only if the boxes break the rate limit; then stay at the box side
        prev = torch.minimum(torch.maximum(points[:, j], low), high)
        out[:, j] = prev
    return out


def smoothing_matrices(horizon: int, beta: float, rho: float, device, dtype):
    """D1 (first differences, the rover at 0 before waypoint 0), and the inverse of the ADMM p-update's matrix
    I + beta D2^T D2 + rho (I + D1^T D1), D2 the second differences (also through the rover at 0)."""
    eye = torch.eye(horizon, device=device, dtype=dtype)
    d1 = eye - torch.diag(torch.ones(horizon - 1, device=device, dtype=dtype), -1)
    d2 = (d1[1:] - d1[:-1]) if horizon > 1 else torch.zeros((0, horizon), device=device, dtype=dtype)
    m = eye + beta * d2.T @ d2 + rho * (eye + d1.T @ d1)
    return d1, torch.linalg.inv(m)


def smooth_into_boxes(points, lo, hi, beta: float, iters: int = 100, rho: float = 1.0, step: float = 0.25):
    """[N, H, 2] -> the same: per sample and axis, argmin 1/2 |p - x|^2 + beta/2 |D2 p|^2 with lo <= p <= hi and
    |D1 p| <= step (scaled ADMM, splitting z1 = p for the boxes and z2 = D1 p for the rate), then the exact sweep.
    A sample whose boxes are a single path (lo = hi everywhere) gets that path as x and no boxes."""
    n, h, _ = points.shape
    pinned = (hi - lo).abs().amax(dim=(1, 2)) < 1e-9
    x = torch.where(pinned[:, None, None], lo, points)
    lo = torch.where(pinned[:, None, None], torch.full_like(lo, -1e9), lo)
    hi = torch.where(pinned[:, None, None], torch.full_like(hi, 1e9), hi)
    d1, m_inv = smoothing_matrices(h, beta, rho, points.device, points.dtype)
    along = lambda a, v: torch.einsum("ij,njk->nik", a, v)  # a matrix over the waypoints, each sample and axis
    z1 = torch.minimum(torch.maximum(x, lo), hi)
    z2 = along(d1, x).clamp(-step, step)
    u1, u2 = torch.zeros_like(x), torch.zeros_like(x)
    p = z1
    for _ in range(iters):
        p = along(m_inv, x + rho * (z1 - u1) + rho * along(d1.T, z2 - u2))
        a1, a2 = p + u1, along(d1, p) + u2
        z1, z2 = torch.minimum(torch.maximum(a1, lo), hi), a2.clamp(-step, step)
        u1, u2 = a1 - z1, a2 - z2
    return sweep_into_boxes(p, lo, hi, step)


def pull_into_boxes(moves, lo, hi, alpha: float, iters: int = 1, smooth: float = 0.0, smooth_iters: int = 100,
                    step: float = 0.25):
    """Pull the samples alpha of the way to their projection into the boxes (exactly onto it at alpha = 1). The
    projection: smooth = 0, the per-waypoint clamp (the first version); smooth = beta > 0, smooth_into_boxes."""
    free = torch.isnan(lo).any(dim=(1, 2)) #If any of the lo values are nan, then that sample is free to move anywhere
    lo, hi = torch.nan_to_num(lo, nan=-1e9), torch.nan_to_num(hi, nan=1e9)
    p = positions(moves)
    if smooth > 0:
        target = smooth_into_boxes(p, lo, hi, smooth, smooth_iters, step=step)
        p = target if alpha >= 1.0 else p + alpha * (target - p)
    elif alpha >= 1.0:    
        p = sweep_into_boxes(p, lo, hi, step)
    else:
        p = p + alpha * (torch.minimum(torch.maximum(p, lo), hi) - p)
    out = moves_from(p, moves, step)
    out[free] = moves[free]
    return out


def box_distance2(points, lo, hi) -> torch.Tensor:
    """[N, H, 2] -> [N]: dist^2 to the product of the boxes (per-waypoint clamp, the exact projection onto B)."""
    return ((points - torch.minimum(torch.maximum(points, lo), hi)) ** 2).sum(dim=(1, 2))


def lyapunov_pull(moves, lo, hi, cand_alpha: float, prev_points, cfg: GuideConfig) -> tuple:
    """One guided step with the Lyapunov condition (module notes). prev_points [N, H, 2]: the last guided step's
    estimate (None at the first one: then a = a_cand). -> (moves', V after [N], a [N])."""
    free = torch.isnan(lo).any(dim=(1, 2))
    lo, hi = torch.nan_to_num(lo, nan=-1e9), torch.nan_to_num(hi, nan=1e9)
    p = positions(moves)
    target = (smooth_into_boxes(p, lo, hi, cfg.smooth, cfg.smooth_iters) if cfg.smooth > 0
              else torch.minimum(torch.maximum(p, lo), hi))
    d2 = ((p - target) ** 2).sum(dim=(1, 2))
    a = torch.full_like(d2, cand_alpha if cfg.lyapunov == "ramp" else 0.0)
    if prev_points is not None:
        v_prev = 0.5 * box_distance2(prev_points, lo, hi)
        ratio = (2 * (1 - cfg.lyapunov_delta) * v_prev / d2.clamp_min(1e-12)).clamp(max=1.0)
        a = torch.maximum(a, 1 - ratio.sqrt())
    a = torch.where(d2 > 1e-12, a, torch.zeros_like(a)).clamp(0.0, 1.0)
    out = moves_from(p + a[:, None, None] * (target - p), moves)
    out[free] = moves[free]
    return out, 0.5 * box_distance2(positions(out), lo, hi), a


def pull_out_of_cells(moves, bev: Bev, alpha: float, smooth: float = 0.0, smooth_iters: int = 100,
                      step: float = 0.25):
    """The no-tube baseline: move every waypoint in an occupied cell to the nearest free cell, make the path drivable
    again (step m per waypoint and axis, 0.25 = NavDP's), and pull the sample alpha of the way there (smoothed when
    smooth > 0). bev: the grid the push uses (grown by push_margin in guided_denoise)."""
    p = positions(moves)
    goal = torch.as_tensor(sweep(bev.nearest_free(p.detach().cpu().numpy()), step), dtype=p.dtype, device=p.device)
    return pull_into_boxes(moves, goal, goal, alpha, smooth=smooth, smooth_iters=smooth_iters, step=step)


def cell_target(points, bev: Bev, smooth: float = 0.0, smooth_iters: int = 100, step: float = 0.25):
    """[N, H, 2] -> the obstacle pull's target, as pull_out_of_cells: every waypoint in an occupied cell moved to the
    nearest free cell, the path made drivable (step m per waypoint and axis), and smoothed when smooth > 0."""
    goal = torch.as_tensor(sweep(bev.nearest_free(points.detach().cpu().numpy()), step), dtype=points.dtype,
                           device=points.device)
    return smooth_into_boxes(goal, goal, goal, smooth, smooth_iters, step=step) if smooth > 0 else goal


def cell_v(points, bev: Bev) -> torch.Tensor:
    """[N, H, 2] -> [N]: V = 1/2 sum_j dist^2(waypoint j, free cells), how deep the path sits in the (grown)
    obstacles; 0 when every waypoint is in a free cell."""
    p = points.detach().cpu().numpy()
    v = 0.5 * ((p - bev.nearest_free(p)) ** 2).sum(axis=(1, 2))
    return torch.as_tensor(v, dtype=points.dtype, device=points.device)


def lyapunov_out_of_cells(moves, bev: Bev, cand_alpha: float, v_prev, cfg: GuideConfig) -> tuple:
    """The obstacle mode's guided step with the Lyapunov condition (module notes). v_prev [N]: V of the last guided
    step's estimate (None at the first one: then a = a_cand). -> (moves', V after [N], a [N])."""
    p = positions(moves)
    target = cell_target(p, bev, cfg.smooth, cfg.smooth_iters, cfg.push_step)
    d2 = ((p - target) ** 2).sum(dim=(1, 2))
    a = torch.full_like(d2, cand_alpha if cfg.lyapunov == "ramp" else 0.0)
    if v_prev is not None:
        ratio = (2 * (1 - cfg.lyapunov_delta) * v_prev / d2.clamp_min(1e-12)).clamp(max=1.0)
        short = cell_v(p, bev) > (1 - cfg.lyapunov_delta) * v_prev    # pull only where V would not fall enough
        a = torch.where(short, torch.maximum(a, 1 - ratio.sqrt()), a)
    a = torch.where(d2 > 1e-12, a, torch.zeros_like(a)).clamp(0.0, 1.0)
    out = moves_from(p + a[:, None, None] * (target - p), moves, cfg.push_step)
    return out, cell_v(positions(out), bev), a


def previous_weights(prev, horizon: int, cfg: GuideConfig, device) -> tuple:
    """The previous plan [m, 2] (body frame, already shifted) -> target [H, 2] and weights [H] (w_j = mu gamma^j for
    j < m, else 0), or (None, None) when there is nothing to hold to."""
    if prev is None or cfg.consistency <= 0:
        return None, None
    prev = np.asarray(prev, np.float64).reshape(-1, 2)[:horizon]
    if len(prev) == 0:
        return None, None
    target = np.zeros((horizon, 2))
    target[:len(prev)] = prev
    w = np.zeros(horizon)
    w[:len(prev)] = min(cfg.consistency, 1.0) * cfg.consistency_decay ** np.arange(len(prev))
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return as_t(target), as_t(w)


def pull_toward(moves, target, w):
    """One gradient step on E_c: x0_j <- x0_j + w_j (x_prev_j - x0_j), on the positions of every sample."""
    p = positions(moves)
    p = p + w[None, :, None] * (target[None] - p)
    return moves_from(p, moves)


def tube_boxes(tubes, group, used, horizon, device) -> tuple:
    """Each sample's boxes at the waypoint times: its tube's when the tube was solved, else its pushed path (it is
    still steered clear of the obstacles, but never certified)."""
    s = waypoint_times(horizon)
    lo, hi = np.array(used, np.float64), np.array(used, np.float64)
    for g, tube in enumerate(tubes):
        if tube.solved:
            tlo, thi = tube.bounds(s)
            lo[group == g], hi[group == g] = tlo, thi
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return as_t(lo), as_t(hi)


def certify(points: np.ndarray, tubes, group) -> np.ndarray:
    ok = np.zeros(len(points), bool)
    for g, tube in enumerate(tubes):
        members = group == g
        ok[members] = inside(points[members], tube)
    return ok


def guided_denoise(predict_noise, scheduler, shape, device, bev, cfg: GuideConfig, generator=None,
                   record: bool = False, prev=None) -> dict:
    """bev: a stt.Bev in the body frame, already grown by the rover's size (None or empty = no obstacles).
    prev: the previous plan [m, 2] in this body frame, from the waypoint the rover has reached on (None = none);
    used only when cfg.consistency > 0."""
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if cfg.lyapunov not in LYAPUNOV_MODES:
        raise ValueError(f"lyapunov must be one of {LYAPUNOV_MODES}")
    x = torch.randn(shape, device=device, generator=generator) # creating a random tensor of shape [N, H, 3] where N is the number of samples, H is the horizon (number of timesteps), and 3 represents the 3D coordinates (x, y, z) in the body frame. This tensor is initialized with random values drawn from a standard normal distribution.
    scheduler.set_timesteps(scheduler.config.num_train_timesteps) # setting the timesteps for the denoising process. The scheduler is responsible for controlling the denoising process over a series of timesteps. Here, it is configured to use the number of training timesteps specified in its configuration.
    steps = list(scheduler.timesteps) #I am listing the timesteps
    if not 0 <= cfg.split < len(steps):
        raise ValueError(f"split must be in [0, {len(steps) - 1}]")
    if cfg.guide_until and not cfg.split < cfg.guide_until <= len(steps):
        raise ValueError(f"guide_until must be 0 (off) or in [split + 1, {len(steps)}]")
    until = cfg.guide_until or len(steps)  # the pull runs at split <= i < until
    if bev is None:
        bev = Bev.empty()
    guided = cfg.mode in ("tube", "obstacle") and bool(bev.occupied.any())
    #Basically if we have obstacles and mode is tube or obstacle, then we are guided. Otherwise we are not guided.
    #Tube is the idea i am proposing and obstacle is the baseline where we just push the samples out of the occupied cells.
    has_cells = bool(bev.occupied.any()) and (cfg.guide_until > 0 or record)   # model_v diagnostic
    if not 0.25 <= cfg.push_step <= 1.0 or not 0.0 <= cfg.push_margin <= 0.5:
        raise ValueError("push_step must be in [0.25, 1] and push_margin in [0, 0.5]")
    push_bev = bev.inflated(cfg.push_margin) if cfg.mode == "obstacle" else bev   # the grid the obstacle push uses
    wide = cfg.mode == "obstacle" and cfg.push_step > 0.25   # moves past [-1, 1]: clip here, not in the scheduler
    clip_was = bool(scheduler.config.get("clip_sample", False))
    if wide:
        scheduler.register_to_config(clip_sample=False)
    try:
        return _denoise_loop(x, predict_noise, scheduler, shape, device, bev, push_bev, cfg, record, prev, steps, until,
                             guided, wide, has_cells)
    finally:
        if wide:
            scheduler.register_to_config(clip_sample=clip_was)


def _denoise_loop(x, predict_noise, scheduler, shape, device, bev, push_bev, cfg, record, prev, steps, until, guided,
                  wide, has_cells):
    """guided_denoise's loop (split out so the scheduler's clip setting is restored however it ends). n_guided: the
    guided steps (the split was a study: from it on the estimate is good enough to guide)."""
    tube_cfg = replace(cfg.tube, horizon=shape[1])
    prev_target, prev_w = previous_weights(prev, shape[1], cfg, device)
    tubes, group, split_paths, lo, hi = None, None, None, None, None
    n_guided = until - cfg.split
    x0_steps = []  # with ``record``: the model's clean-path estimate at every step, before any guidance
    guided_points, lyapunov_v, n_builds = None, [], 0  # lyapunov: last guided estimate, mean V per guided step
    v_cells = None  # lyapunov in obstacle mode: V of the last guided estimate [N]
    model_v = []    # mean V (depth in the grown obstacles) of the model's own clean-path estimate at every step
    for i, k in enumerate(steps):
        eps = predict_noise(x, k.unsqueeze(0)) #
        if record:
            ab = scheduler.alphas_cumprod[int(k)].to(x.device, x.dtype)
            x0_steps.append(positions(((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)).detach().cpu().numpy())
        if has_cells:
            ab = scheduler.alphas_cumprod[int(k)].to(x.device, x.dtype)
            raw = positions(((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0))
            model_v.append(float(cell_v(raw, bev).mean()))
        pull_tube = guided and cfg.split <= i < until
        pull_prev = prev_w is not None and i >= cfg.consistency_from
        if pull_tube or pull_prev:
            ab = scheduler.alphas_cumprod[int(k)].to(x.device, x.dtype)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0) #this we dont care about much
            if pull_prev:  # E_c first: the tube pull below has the last word
                x0 = pull_toward(x0, prev_target, prev_w)
        if pull_tube:
            j = i - cfg.split
            ramp = n_guided - cfg.project_last
            alpha = 1.0 if j >= ramp else cfg.alpha0 + (1.0 - cfg.alpha0) * j / max(ramp, 1) # if its far enough then we wont concern ourselves with it
            if cfg.mode == "tube":
                if tubes is None or cfg.rebuild:
                    estimate = positions(x0).detach().cpu().numpy()
                    if split_paths is None:
                        split_paths = estimate
                    tubes, group, used = tubes_for(estimate, bev, tube_cfg, push=True)
                    lo, hi = tube_boxes(tubes, group, used, shape[1], x.device)
                    n_builds += 1
                if cfg.lyapunov != "off" and alpha < 1.0:
                    x0, v_after, _ = lyapunov_pull(x0, lo, hi, alpha, guided_points, cfg)
                    lyapunov_v.append(float(v_after.mean()))
                else:
                    x0 = pull_into_boxes(x0, lo, hi, alpha, smooth=cfg.smooth, smooth_iters=cfg.smooth_iters)
                    if cfg.lyapunov != "off":
                        lyapunov_v.append(float(0.5 * box_distance2(positions(x0), torch.nan_to_num(lo, nan=-1e9),
                                                                    torch.nan_to_num(hi, nan=1e9)).mean()))
                guided_points = positions(x0)
            elif cfg.lyapunov != "off" and alpha < 1.0:
                x0, v_cells, _ = lyapunov_out_of_cells(x0, push_bev, alpha, v_cells, cfg)
                lyapunov_v.append(float(v_cells.mean()))
            else:
                x0 = pull_out_of_cells(x0, push_bev, alpha, smooth=cfg.smooth, smooth_iters=cfg.smooth_iters,
                                       step=cfg.push_step)
                if cfg.lyapunov != "off":
                    v_cells = cell_v(positions(x0), push_bev)
                    lyapunov_v.append(float(v_cells.mean()))
        if wide and not (pull_tube or pull_prev):  # the scheduler's clip is off: clip to [-1, 1] here, as it did
            ab = scheduler.alphas_cumprod[int(k)].to(x.device, x.dtype)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)
        if pull_tube or pull_prev or wide:
            eps = (x - ab.sqrt() * x0) / (1 - ab).clamp_min(1e-12).sqrt()  # the score correction (module docstring)
        x = scheduler.step(model_output=eps, timestep=k, sample=x).prev_sample
    points = positions(x).detach().cpu().numpy()
    if tubes is None:  # navdp / obstacle mode (or no obstacles): no tubes on the finished paths, nothing certified
        tubes, group = [], np.zeros(len(points), np.int64)
    final_v = 0.5 * ((points[..., :2] - bev.nearest_free(points[..., :2])) ** 2).sum(axis=(1, 2))
    return {"moves": x, "tubes": tubes, "group": group, "certified": certify(points, tubes, group),
            "split_paths": split_paths, "x0_steps": np.stack(x0_steps) if record else None,
            "lyapunov_v": lyapunov_v, "tube_builds": n_builds, "model_v": model_v, "final_v": final_v,
            "final_in_cells": bev.occupied_at(points[..., :2], outside=False).any(axis=1)}


def choose(values: np.ndarray, certified: np.ndarray, mode: str) -> tuple:
    """-> (index, certified?). ``navdp`` and ``obstacle`` keep NavDP's rule (best value) and only report the
    certificate; ``tube`` picks the best-valued certified sample, else the best-valued one."""
    best = int(np.argmax(values))
    if mode in ("navdp", "obstacle") or not certified.any():
        return best, bool(certified[best])
    masked = np.where(certified, values, -np.inf)
    return int(np.argmax(masked)), True
