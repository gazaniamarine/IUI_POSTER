"""Horizon-CBF safety guidance for the diffusion policy.

Everything is from the policy's own output trajectory plus the obstacle the robot
already segments -- no occupancy grid, no extra learned head.

  * Obstacle relative position p comes from the segmented obstacle mask + depth
    (unprojection -> nearest point in the robot body frame).
  * Obstacle velocity v_o comes from frame-differencing that point and subtracting
    the robot's own ego-motion ("both moving": robot motion via ego-motion, the
    obstacle's own motion via the residual).
  * The barrier h(p) = ||p||^2 - d_safe^2 is rolled out over the WHOLE predicted
    action chunk with the same SE(2) ego-motion propagation the belief bank uses,
    enforcing the discrete CBF condition h(p_{k+1}) >= (1-gamma) h(p_k) at every
    horizon step. The differentiable rollout yields a guidance gradient that is
    injected into each diffusion denoising step (see S2DiTPolicy.sample's
    guidance_fn hook), so the sampled trajectory is safe by construction.

Action chunk layout: [B, H, A] with (v_fwd, v_lat, yaw_rate); A==2 is treated as
(v_fwd, yaw_rate).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Post-hoc velocity CBF projection (Step 1: cheap, no during-sampling fighting)
# ----------------------------------------------------------------------------
def project_forward_velocity_cbf(
    action: Sequence[float],
    obstacle_point: Optional[Sequence[float]],
    v_o: Optional[Sequence[float]] = None,
    d_safe: float = 0.5,
    gamma: float = 0.15,
    deadzone: float = 0.5,
    trust: Optional[float] = None,
    min_forward: float = 0.0,
) -> tuple:
    """Project the EXECUTED velocity command onto the collision-safe set, ONCE, after
    sampling -- no gradient nudging during denoising (that is what fights the policy).

    For a differential-drive / v_lat=0 robot the only safe correction is BRAKING the
    forward speed; yaw is left ENTIRELY to the policy (so the CBF can never cause yaw
    thrash). A distance barrier h = d - d_safe with the discrete CBF condition
    d_dot + gamma*h >= 0 gives an allowed closing speed a_max; the forward speed is
    capped so the robot's approach along its heading does not exceed it.

    action     : [v_fwd, v_lat, yaw] in m/s (body frame [fwd, lat]).
    obstacle_point / v_o : nearest obstacle position / velocity in the same body frame.
    deadzone   : extra margin beyond d_safe within which braking activates (m).
    trust      : max forward-speed reduction per step (smoothness); None/<=0 = full brake.
                 A breach (d < d_safe) always full-brakes regardless of trust.
    Returns (action, braked_bool). yaw and v_lat are never modified here.
    """
    a = np.asarray(action, dtype=np.float32).copy()
    if obstacle_point is None:
        return a, False
    p = np.asarray(obstacle_point, dtype=np.float32).reshape(-1)[:2]
    d = float(np.linalg.norm(p))
    if d < 1e-3 or d >= d_safe + deadzone:
        return a, False  # deadzone: obstacle far -> CBF is a no-op (no free-space thrash)
    n = p / d
    n_fwd = float(n[0])  # how much "toward the obstacle" lies along body-forward
    if n_fwd <= 1e-3:
        return a, False  # obstacle beside/behind: driving forward does not close on it
    vo = np.asarray(v_o, dtype=np.float32).reshape(-1)[:2] if v_o is not None else np.zeros(2, np.float32)
    v_o_n = float(vo @ n)
    a_max = v_o_n + float(gamma) * (d - float(d_safe))  # allowed closing speed (m/s); <0 if inside d_safe
    v = float(a[0])
    approach = v * n_fwd
    if approach <= a_max:
        return a, False  # already safe -> untouched
    v_cap = a_max / n_fwd
    v_new = float(np.clip(v_cap, min_forward, v))  # brake only: never speed up, never reverse
    if trust is not None and trust > 0.0 and d >= float(d_safe):
        v_new = max(v_new, v - float(trust))  # limit per-step cut for smoothness (not when breaching)
        v_new = min(v_new, v)
    a[0] = v_new
    return a, True


# ----------------------------------------------------------------------------
# Collision-cone (C3BF) horizon projection: push the 8-step chunk out of the cone
# ----------------------------------------------------------------------------
def cone_barrier_horizon(
    actions: torch.Tensor,
    p0: torch.Tensor,
    v_o: torch.Tensor,
    r: float,
    dt: float,
    vel_scale: float = 1.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Sum of collision-cone violations along the rolled-out trajectory.

    At each horizon step the C3BF barrier is
        h = p . v_rel + |v_rel| * sqrt(|p|^2 - r^2),   v_rel = v_o - v_robot,
    which is >= 0 exactly when the (relative) velocity points OUTSIDE the collision
    cone (the robot will miss the obstacle) -- independent of speed. Violation is
    relu(margin - h)^2. p is propagated in the body frame with the same SE(2)
    ego-motion (including yaw) the distance-CBF uses, so the gradient reaches
    v_fwd/v_lat AND yaw_rate -> the projection can STEER, not just brake."""
    b, h_steps, a_dim = actions.shape
    device, dtype = actions.device, actions.dtype
    p = (p0 if p0.dim() == 2 else p0[None].expand(b, 2)).to(device=device, dtype=dtype).clone()
    vo = (v_o if v_o.dim() == 2 else v_o[None].expand(b, 2)).to(device=device, dtype=dtype)
    r2 = float(r) * float(r)
    cost = actions.new_zeros(())
    for k in range(h_steps):
        v_fwd = actions[:, k, 0] * vel_scale
        if a_dim >= 3:
            v_lat = actions[:, k, 1] * vel_scale
            omega = actions[:, k, 2]
        else:
            v_lat = actions.new_zeros(b)
            omega = actions[:, k, 1]
        vrx = vo[:, 0] - v_fwd
        vry = vo[:, 1] - v_lat
        vrn = torch.sqrt(vrx * vrx + vry * vry + 1e-9)
        root = torch.sqrt(torch.relu((p * p).sum(-1) - r2) + 1e-9)
        h = (p[:, 0] * vrx + p[:, 1] * vry) + vrn * root
        cost = cost + (torch.relu(margin - h) ** 2).sum()
        th = -omega * dt
        c, s = torch.cos(th), torch.sin(th)
        qx = p[:, 0] - v_fwd * dt
        qy = p[:, 1] - v_lat * dt
        px = c * qx - s * qy + vo[:, 0] * dt
        py = s * qx + c * qy + vo[:, 1] * dt
        p = torch.stack([px, py], dim=-1)
    return cost


def horizon_growth_covariance(
    h_steps: int,
    a_dim: int,
    base: float = 1.0,
    growth: float = 0.5,
    mode: str = "grow",
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Diagonal action-chunk covariance Sigma for the Mahalanobis cone projection.

    The metric decides which horizon steps ABSORB the collision correction (higher
    variance -> cheaper to change -> corrected more). Shape [H*A, H*A].
        mode="grow"   : var(k) = base*(1 + growth*k)          -> far steps corrected more.
                        Matches the diffusion "future is more uncertain" prior, BUT it
                        PROTECTS chunk[0] -- and under receding-horizon control (only
                        chunk[0] is executed) that under-corrects the executed action
                        and can COLLIDE.
        mode="shrink" : var(k) = base*(1 + growth*(H-1-k))    -> NEAR steps corrected more.
                        Emphasizes the executed action -> safe under receding horizon,
                        while still being a covariance-weighted (Mahalanobis) projection.
        mode="flat"   : var(k) = base (uniform, ~ Euclidean)."""
    n = int(h_steps) * int(a_dim)
    diag = torch.empty(n, dtype=dtype)
    H = int(h_steps)
    for k in range(H):
        if mode == "shrink":
            scale = 1.0 + float(growth) * (H - 1 - k)
        elif mode == "flat":
            scale = 1.0
        else:  # grow
            scale = 1.0 + float(growth) * k
        diag[k * a_dim:(k + 1) * a_dim] = float(base) * scale
    m = torch.diag(diag)
    return m.to(device=device) if device is not None else m


def project_chunk_cone(
    chunk: torch.Tensor,
    p0: Optional[Sequence[float]],
    v_o: Optional[Sequence[float]] = None,
    r: float = 0.75,
    dt: float = 1.0 / 30.0,
    vel_scale: float = 1.0,
    iters: int = 10,
    lr: float = 0.05,
    trust: float = 0.3,
    margin: float = 0.05,
    smooth_weight: float = 0.0,
    keep_speed: float = 0.0,
    deadzone_range: Optional[float] = None,
    sigma: Optional[torch.Tensor] = None,
    side: float = 0.0,
    side_seed: float = 0.05,
) -> torch.Tensor:
    """POST-HOC projection of a sampled action chunk out of the collision cone.

    chunk : [1, H, A] raw diffusion actions (before action_to_control).
    Runs a few gradient steps on the cone-violation cost, staying inside a trust
    region around the sampled chunk. Metric:
      * sigma is None  -> Euclidean gradient step  (dU = -lr * grad).
      * sigma given    -> covariance-preconditioned (Mahalanobis) natural-gradient
                          step dU = -lr * (Sigma @ grad_flat): pushes preferentially
                          along directions the diffusion policy is UNCERTAIN about
                          (e.g. the far-horizon steps via horizon_growth_covariance).
    smooth_weight adds a temporal-smoothness penalty on the CORRECTION so the
    projected chunk stays jerk-free (sum ||u_{k+1}-u_k||^2).

    keep_speed penalizes REDUCING the forward speed below the sampled value:
        keep_speed * sum relu(v0_fwd_k - v_fwd_k)^2.
    This is essential -- the collision cone is trivially satisfied by v=0 (a stopped
    robot never collides), so without this the projection just BRAKES TO A STOP
    instead of steering. keep_speed makes braking costly, forcing the correction into
    YAW (go around) while maintaining forward progress.

    Returns the projected chunk (detached). No-op (deadzone) beyond deadzone_range."""
    if p0 is None:
        return chunk
    device, dtype = chunk.device, chunk.dtype
    p0t = torch.as_tensor(np.asarray(p0, dtype=np.float32), device=device, dtype=dtype)
    if deadzone_range is not None and float(torch.linalg.norm(p0t)) > float(deadzone_range):
        return chunk
    vot = torch.as_tensor(np.asarray(v_o if v_o is not None else (0.0, 0.0), dtype=np.float32),
                          device=device, dtype=dtype)
    U0 = chunk.detach().clone()
    U = U0.clone()
    shape = U.shape
    # Symmetry tie-breaker: a dead-centre head-on obstacle is a saddle (yaw gradient
    # is 0, so pure descent only brakes). Seed a small yaw bias toward `side`
    # (+1 = turn left / CCW, -1 = right) so the projection commits to going AROUND.
    if side != 0.0 and shape[-1] >= 3:
        U = U.clone()
        U[..., 2] = U[..., 2] + float(side) * float(side_seed)
    # The rollout runs under torch.no_grad(); re-enable grad so the cone-cost graph
    # is built and autograd.grad works (mirrors build_cbf_guidance's guidance_fn).
    with torch.enable_grad():
        for _ in range(int(iters)):
            Uv = U.detach().requires_grad_(True)
            cost = cone_barrier_horizon(Uv, p0t, vot, r, dt, vel_scale, margin)
            if smooth_weight > 0.0 and Uv.shape[1] >= 2:
                du = Uv[:, 1:, :] - Uv[:, :-1, :]
                cost = cost + float(smooth_weight) * (du * du).sum()
            if keep_speed > 0.0:
                # penalize dropping forward speed -> steer, don't brake to a stop
                dv = torch.relu(U0[..., 0] - Uv[..., 0])
                cost = cost + float(keep_speed) * (dv * dv).sum()
            if float(cost.detach()) <= 1e-9:
                break  # already outside the cone over the whole horizon
            (g,) = torch.autograd.grad(cost, Uv)
            if sigma is not None:
                gflat = g.reshape(-1)
                step = -(lr) * (sigma.to(device=device, dtype=dtype) @ gflat).reshape(shape)
            else:
                step = -(lr) * g
            U = U + step
            U = U0 + torch.clamp(U - U0, -float(trust), float(trust))  # trust region
    return U.detach()


# ----------------------------------------------------------------------------
# Obstacle relative state from segmentation + depth (no grid)
# ----------------------------------------------------------------------------
def _unproject_mask_points(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    min_depth: float = 1e-3,
) -> Optional[np.ndarray]:
    """Unproject every valid masked pixel to body-frame (x_fwd, y_lat). Shape [N, 2], or
    None if the mask is empty / has no valid depth. Shared by nearest_obstacle_point and
    nearest_obstacle_state so the unprojection only happens once per call site.

    Camera optical [right, down, forward] -> robot body [forward, left]:
        x_fwd = z,  y_lat = -x_cam.
    """
    m = np.asarray(mask) > 0
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    z = np.asarray(depth, dtype=np.float32)[ys, xs]
    valid = np.isfinite(z) & (z > float(min_depth))
    if not valid.any():
        return None
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx = float(intrinsics.get("fx", max(depth.shape)))
    cx = float(intrinsics.get("cx", (depth.shape[1] - 1) * 0.5))
    x_cam = (xs.astype(np.float32) - cx) * z / fx
    x_fwd = z
    y_lat = -x_cam
    return np.stack([x_fwd, y_lat], axis=1).astype(np.float32)


def nearest_obstacle_point(
    obstacle_mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    min_depth: float = 1e-3,
) -> Optional[np.ndarray]:
    """Unproject the segmented obstacle and return the NEAREST point (x_fwd, y_lat).

    The nearest point is the binding constraint for the barrier. Unchanged public
    contract (used by policy_runner.py / rollout_scene_dataset_policy.py too) --
    see nearest_obstacle_state() for the version that also estimates a per-obstacle
    radius from the mask's spatial extent.
    """
    pts = _unproject_mask_points(obstacle_mask, depth, intrinsics, min_depth)
    if pts is None:
        return None
    i = int(np.argmin((pts * pts).sum(axis=1)))
    return pts[i].copy()


def nearest_obstacle_state(
    obstacle_mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    min_depth: float = 1e-3,
    radius_percentile: float = 90.0,
    min_radius: float = 0.05,
    max_radius: float = 2.0,
) -> Optional[Dict[str, object]]:
    """Like nearest_obstacle_point, but ALSO estimates a per-obstacle RADIUS from the
    segmented mask's own spatial extent -- instead of the caller hand-tuning a single
    constant clearance for every obstacle regardless of size. A small rock and a big
    boulder unproject to point clouds of different spread; a big/close obstacle fills
    more pixels at more varied depth, so its unprojected extent is larger.

    radius = robust (percentile, default 90th) distance from the mask's unprojected
    CENTROID to its own points, clamped to [min_radius, max_radius] to reject
    segmentation-noise outliers (a single stray pixel at the mask edge doesn't blow
    the estimate up the way a plain max() would).

    Returns {"p0": (2,) nearest point, "radius": float, "centroid": (2,), "n_points": int}
    or None if the mask is empty / has no valid depth.
    """
    pts = _unproject_mask_points(obstacle_mask, depth, intrinsics, min_depth)
    if pts is None:
        return None
    i_near = int(np.argmin((pts * pts).sum(axis=1)))
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)
    radius = float(np.clip(np.percentile(dists, radius_percentile), min_radius, max_radius))
    return {
        "p0": pts[i_near].copy(),
        "radius": radius,
        "centroid": centroid.astype(np.float32),
        "n_points": int(pts.shape[0]),
    }


def ego_motion_point(p_prev: np.ndarray, delta_pose: Sequence[float]) -> np.ndarray:
    """Move a body-frame point into the new frame (same SE(2) as the belief bank)."""
    dx, dy, dth = float(delta_pose[0]), float(delta_pose[1]), float(delta_pose[2])
    c, s = math.cos(-dth), math.sin(-dth)
    qx, qy = p_prev[0] - dx, p_prev[1] - dy
    return np.asarray([c * qx - s * qy, s * qx + c * qy], dtype=np.float32)


def estimate_obstacle_velocity(
    p_prev: Optional[np.ndarray],
    p_curr: Optional[np.ndarray],
    delta_pose: Sequence[float],
    dt: float,
) -> np.ndarray:
    """v_o = (p_curr - egomotion(p_prev)) / dt : the obstacle's OWN motion.

    Subtracting the ego-motion-predicted (static) position leaves only the
    obstacle's independent velocity. Static obstacle -> ~0.
    """
    if p_prev is None or p_curr is None:
        return np.zeros(2, dtype=np.float32)
    warped = ego_motion_point(np.asarray(p_prev, dtype=np.float32), delta_pose)
    return ((np.asarray(p_curr, dtype=np.float32) - warped) / max(float(dt), 1e-6)).astype(np.float32)


# ----------------------------------------------------------------------------
# Tangential circulation: which way to go AROUND a blocking obstacle
# ----------------------------------------------------------------------------
def tangential_around_obstacle(
    p_obstacle: Sequence[float],
    mu_goal: Sequence[float],
    block_range: float = 1.5,
    align_cos: float = 0.3,
) -> Optional[np.ndarray]:
    """Body-frame unit direction to slide AROUND the obstacle toward the goal.

    Returns None unless the obstacle is actually *blocking* the goal, i.e. it is
    close (||p|| < block_range), nearer than the goal (||p|| < ||mu||), and roughly
    in the goal direction (cos angle > align_cos). Otherwise the robot is not in a
    head-on deadlock and circulation should stay off so it can't hurt easy cases.

    When blocking, pick the perpendicular to the robot->obstacle ray that points
    toward the goal's side, so the robot consistently rounds ONE side of a wide
    obstacle instead of stalling/reversing head-on. Frame: [forward, left].
    """
    p = np.asarray(p_obstacle, dtype=np.float32)
    g = np.asarray(mu_goal, dtype=np.float32)
    dp = float(np.linalg.norm(p))
    dg = float(np.linalg.norm(g))
    if dp < 1e-3 or dg < 1e-3:
        return None
    if dp > float(block_range) or dp > dg:
        return None
    p_hat = p / dp
    g_hat = g / dg
    if float(p_hat @ g_hat) < float(align_cos):
        return None
    # Perpendicular to the obstacle ray, chosen toward the goal side.
    t = np.asarray([-p_hat[1], p_hat[0]], dtype=np.float32)
    if float(t @ g_hat) < 0.0:
        t = -t
    return t.astype(np.float32)


# ----------------------------------------------------------------------------
# Horizon-CBF cost over the predicted action chunk (differentiable)
# ----------------------------------------------------------------------------
def cbf_horizon_cost(
    actions: torch.Tensor,
    p0: Optional[torch.Tensor],
    v_o: torch.Tensor,
    d_safe: float,
    gamma: float,
    dt: float,
    vel_scale: float = 1.0,
    hard_weight: float = 2.0,
    mu_goal: Optional[torch.Tensor] = None,
    goal_attract_weight: float = 0.0,
    apply_cbf: bool = True,
    tangential: Optional[torch.Tensor] = None,
    tangential_weight: float = 0.0,
    tangential_target: float = 0.3,
    heading_weight: float = 0.0,
) -> torch.Tensor:
    """Sum of discrete-CBF violations + optional goal-attraction along the rolled-out trajectory.

    CBF avoidance (active when apply_cbf and p0 is not None). Rollout of the
    obstacle's relative position in the body frame, both robot and obstacle moving:
        p_{k+1} = R(-w_k dt) (p_k - u_k^xy dt) + v_o dt
    CBF condition per step:  h(p_{k+1}) >= (1-gamma) h(p_k),  h = ||p||^2 - d^2.
    A hard penalty on h<0 enforces the safe set directly.

    Goal-attraction (active when mu_goal is given and goal_attract_weight > 0):
        cost += goal_attract_weight * sum_k ||robot_k - mu_goal||^2
    where robot_k is the ROBOT'S pose (x, y) accumulated by INTEGRATING THE FULL
    SE(2) motion -- including heading from yaw_rate -- so the gradient flows to
    v_fwd, v_lat AND yaw_rate. That lets the pull actually *steer* the robot back
    toward the belief-predicted goal mu_goal (body frame of the current step),
    instead of only changing speed. This term is INDEPENDENT of the obstacle, so
    it keeps acting after the obstacle has been passed and left the mask -- which
    is exactly when the bare policy tends to forget the belief. Returns a scalar.
    """
    if actions.dim() != 3:
        raise ValueError("actions must be [B, H, A]")
    b, h_steps, a_dim = actions.shape
    device, dtype = actions.device, actions.dtype

    use_cbf = bool(apply_cbf) and (p0 is not None)
    if use_cbf:
        p = (p0 if p0.dim() == 2 else p0[None].expand(b, 2)).to(device=device, dtype=dtype).clone()
        vo = (v_o if v_o.dim() == 2 else v_o[None].expand(b, 2)).to(device=device, dtype=dtype)
        d2 = float(d_safe) * float(d_safe)
        h_prev = (p * p).sum(-1) - d2

    cost = actions.new_zeros(())

    # Robot pose integrated in the current body frame (origin, heading 0 = forward).
    use_attract = (goal_attract_weight > 0.0) and (mu_goal is not None)
    if use_attract:
        mg = (mu_goal if mu_goal.dim() == 2 else mu_goal[None].expand(b, 2)).to(device=device, dtype=dtype)
        rx = actions.new_zeros(b)
        ry = actions.new_zeros(b)
        rth = actions.new_zeros(b)

    # Tangential circulation: encourage body velocity along t_hat to round the obstacle.
    use_tan = (tangential is not None) and (tangential_weight > 0.0)
    if use_tan:
        tn = (tangential if tangential.dim() == 2 else tangential[None].expand(b, 2)).to(device=device, dtype=dtype)

    for k in range(h_steps):
        v_fwd = actions[:, k, 0] * vel_scale
        if a_dim >= 3:
            v_lat = actions[:, k, 1] * vel_scale
            omega = actions[:, k, 2]
        else:
            v_lat = actions.new_zeros(b)
            omega = actions[:, k, 1]

        if use_tan:
            # Reward velocity projected onto the tangential (around-obstacle) direction
            # up to a target speed; this slides the robot along a wide obstacle toward
            # the goal side instead of stalling/reversing head-on.
            proj = v_fwd * tn[:, 0] + v_lat * tn[:, 1]
            short = torch.relu(tangential_target - proj)
            cost = cost + tangential_weight * (short * short).sum()

        if use_cbf:
            th = -omega * dt
            c, s = torch.cos(th), torch.sin(th)
            qx = p[:, 0] - v_fwd * dt
            qy = p[:, 1] - v_lat * dt
            px = c * qx - s * qy + vo[:, 0] * dt
            py = s * qx + c * qy + vo[:, 1] * dt
            p = torch.stack([px, py], dim=-1)
            h = (p * p).sum(-1) - d2
            viol = torch.relu((1.0 - gamma) * h_prev - h)
            cost = cost + (viol * viol).sum() + hard_weight * (torch.relu(-h) ** 2).sum()
            h_prev = h

        if use_attract:
            # Forward-integrate the robot's own pose: heading first, then translate.
            rth = rth + omega * dt
            cth, sth = torch.cos(rth), torch.sin(rth)
            rx = rx + (v_fwd * cth - v_lat * sth) * dt
            ry = ry + (v_fwd * sth + v_lat * cth) * dt
            dx = rx - mg[:, 0]
            dy = ry - mg[:, 1]
            cost = cost + goal_attract_weight * (dx * dx + dy * dy).sum()

            if heading_weight > 0.0:
                # Drive yaw to point the robot's heading AT the goal (centering).
                # 1 - cos(bearing_to_goal - heading) is smooth, min when aligned, and
                # gives a strong yaw gradient the position term alone lacks.
                bearing = torch.atan2(mg[:, 1] - ry, mg[:, 0] - rx)
                cost = cost + heading_weight * (1.0 - torch.cos(bearing - rth)).sum()

    return cost


def build_cbf_guidance(
    p0: Optional[Sequence[float]] = None,
    v_o: Sequence[float] = (0.0, 0.0),
    d_safe: float = 0.5,
    gamma: float = 0.3,
    dt: float = 1.0 / 30.0,
    vel_scale: float = 1.0,
    guidance_scale: float = 0.5,
    n_steps: int = 1,
    hard_weight: float = 2.0,
    max_grad_norm: float = 1.0,
    mu_goal: Optional[Sequence[float]] = None,
    goal_attract_weight: float = 0.0,
    tangential: Optional[Sequence[float]] = None,
    tangential_weight: float = 0.0,
    tangential_target: float = 0.3,
    heading_weight: float = 0.0,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Return a guidance_fn(pred_x0)->pred_x0 for S2DiTPolicy.sample (or None if no-op).

    Each call takes a gradient step on the horizon cost, pushing the predicted
    action chunk toward a safe + goal-seeking trajectory. p0/v_o/mu_goal are in the
    robot body frame (metres, m/s).

    Three terms, each independently switchable:
      * CBF avoidance  -- active when p0 is not None. Keeps the obstacle outside
        d_safe over the whole horizon.
      * Goal-attraction -- active when mu_goal is given and goal_attract_weight>0.
        Pulls (and STEERS) the trajectory toward the belief-predicted goal. This is
        independent of the obstacle, so passing p0=None gives a pure "return to the
        belief path" guidance that keeps acting after the obstacle is gone.
      * Tangential circulation -- active when `tangential` (a body-frame unit dir,
        e.g. from tangential_around_obstacle) is given and tangential_weight>0. It
        slides the robot along a wide obstacle toward the goal side, breaking the
        head-on deadlock where avoidance and attraction cancel and the robot stalls.

    Returns None when no term is active (caller can skip the sample hook).
    """
    apply_cbf = p0 is not None
    use_attract = (mu_goal is not None) and (goal_attract_weight > 0.0)
    use_tan = (tangential is not None) and (tangential_weight > 0.0)
    if not apply_cbf and not use_attract and not use_tan:
        return None

    p0_t = torch.as_tensor(np.asarray(p0, dtype=np.float32)) if apply_cbf else None
    vo_t = torch.as_tensor(np.asarray(v_o, dtype=np.float32))
    mg_t = torch.as_tensor(np.asarray(mu_goal, dtype=np.float32)) if use_attract else None
    tn_t = torch.as_tensor(np.asarray(tangential, dtype=np.float32)) if use_tan else None

    def guidance_fn(pred_x0: torch.Tensor) -> torch.Tensor:
        out = pred_x0
        p0_d = p0_t.to(device=pred_x0.device, dtype=pred_x0.dtype) if p0_t is not None else None
        vo_d = vo_t.to(device=pred_x0.device, dtype=pred_x0.dtype)
        mg_d = mg_t.to(device=pred_x0.device, dtype=pred_x0.dtype) if mg_t is not None else None
        tn_d = tn_t.to(device=pred_x0.device, dtype=pred_x0.dtype) if tn_t is not None else None
        for _ in range(int(n_steps)):
            with torch.enable_grad():
                x = out.detach().requires_grad_(True)
                cost = cbf_horizon_cost(
                    x, p0_d, vo_d, d_safe, gamma, dt, vel_scale, hard_weight,
                    mu_goal=mg_d, goal_attract_weight=goal_attract_weight,
                    apply_cbf=apply_cbf,
                    tangential=tn_d, tangential_weight=tangential_weight,
                    tangential_target=tangential_target,
                    heading_weight=heading_weight,
                )
                (grad,) = torch.autograd.grad(cost, x)
            # Per-sample gradient clipping so a near violation can't blow up the step.
            gnorm = grad.flatten(1).norm(dim=1).clamp_min(1e-8)
            scale = torch.clamp(max_grad_norm / gnorm, max=1.0).view(-1, 1, 1)
            out = (out - guidance_scale * grad * scale).detach()
        return out

    return guidance_fn
