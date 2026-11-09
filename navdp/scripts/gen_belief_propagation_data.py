#!/usr/bin/env python3
"""gen_belief_propagation_data.py -- synthesize belief-propagation episodes for training
a belief-ONLY diffusion policy (train_belief_only_policy.py), with zero dependence on
NavDP's image-conditioned S2DiT backbone or Habitat rendering.

Simulates, in pure belief-space, the same body-frame [forward,left] goal-belief dynamics
as sam_vla/core/belief_tracking.py (BeliefGoalTracker): at each step the goal is either
"seen" (belief re-seeds to the true relative goal position, plus observation noise) or
occluded (belief dead-reckons via the robot's own executed motion, drifting under
odometry noise -- propagate_body_point, ported verbatim below). A P-controller (same law
as lost_goal_heading_assist: yaw proportional to bearing) steers off the CURRENT belief,
so the executed motion -- and hence the propagation -- is closed-loop.

Each episode logs `context_len` already-elapsed belief steps (the history the policy will
condition on) plus an `horizon`-step target action chunk: the P-controller's open-loop
continuation from the context-end belief, used as the diffusion regression target
(mirrors the [H,3] action-chunk convention in train_belief_adapter.py/
train_vla_adapter.py; column index 2 = yaw rate, v_lat is always 0 to match
run_navdp_rollout.py's zero_lateral default).

    python scripts/gen_belief_propagation_data.py --episodes 20000 --out belief_only_data/samples
"""
import argparse
import math
from pathlib import Path

import numpy as np

R_SCALE = 10.0  # matches train_belief_adapter.py's R_SCALE
FEAT_DIM = 4  # [cos(bearing), sin(bearing), range/R_SCALE, visible]


def belief_feat(belief: np.ndarray, visible: bool) -> np.ndarray:
    """body-frame [forward,left] -> rotation-aware feature (avoids the +-pi wrap),
    same recipe as train_belief_adapter.py's belief_feat() plus a visibility flag so the
    encoder can tell a freshly-observed belief from one that's only been dead-reckoned."""
    f, l = float(belief[0]), float(belief[1])
    bearing = math.atan2(l, f)
    rng = math.hypot(f, l)
    return np.array(
        [math.cos(bearing), math.sin(bearing), min(rng / R_SCALE, 1.0), float(visible)],
        np.float32,
    )


def propagate_body_point(
    bg: np.ndarray, v_fwd: float, v_left: float, yaw_rate: float, dt: float,
    odom_noise: float, rng: np.random.Generator,
) -> np.ndarray:
    """Body-frame dead-reckoning -- ported verbatim from
    sam_vla/core/belief_tracking.py:propagate_body_point."""
    if odom_noise > 0.0:
        v_fwd = v_fwd + float(rng.normal(0.0, odom_noise))
        yaw_rate = yaw_rate + float(rng.normal(0.0, odom_noise))
    th = -float(yaw_rate) * float(dt)
    c, s = math.cos(th), math.sin(th)
    qx = float(bg[0]) - float(v_fwd) * float(dt)
    qy = float(bg[1]) - float(v_left) * float(dt)
    return np.array([c * qx - s * qy, s * qx + c * qy], np.float32)


def p_controller(belief: np.ndarray, turn_kp: float, base_forward: float, max_yaw_rate: float):
    """Steering law inspired by sam_vla/core/belief_tracking.py:lost_goal_heading_assist --
    yaw proportional to bearing, forward speed tapering off as the goal swings abeam so the
    reference trajectory doesn't drive straight past a goal that's off to the side."""
    bearing = math.atan2(float(belief[1]), float(belief[0]))
    yaw = float(np.clip(turn_kp * bearing, -max_yaw_rate, max_yaw_rate))
    fwd = base_forward * max(0.0, math.cos(bearing))
    return fwd, 0.0, yaw  # v_fwd, v_lat(=0), yaw_rate


def simulate_episode(
    rng: np.random.Generator, context_len: int, horizon: int, dt: float, turn_kp: float,
    base_forward: float, max_yaw_rate: float, p_visible: float, odom_noise: float,
    obs_noise: float, bearing0_deg: float, range0: tuple,
):
    bearing0 = math.radians(rng.uniform(-bearing0_deg, bearing0_deg))
    range0_s = rng.uniform(*range0)
    true_goal = np.array(
        [range0_s * math.cos(bearing0), range0_s * math.sin(bearing0)], np.float32
    )
    belief = true_goal.copy()

    seq = [belief_feat(belief, visible=True)]  # the initial-frame sighting establishes belief
    for _ in range(context_len - 1):
        v_fwd, v_left, yaw = p_controller(belief, turn_kp, base_forward, max_yaw_rate)
        true_goal = propagate_body_point(true_goal, v_fwd, v_left, yaw, dt, 0.0, rng)
        visible = bool(rng.uniform() < p_visible)
        if visible:
            noise = rng.normal(0.0, obs_noise, size=2).astype(np.float32) if obs_noise > 0 else 0.0
            belief = (true_goal + noise).astype(np.float32)
        else:
            belief = propagate_body_point(belief, v_fwd, v_left, yaw, dt, odom_noise, rng)
        seq.append(belief_feat(belief, visible))

    # open-loop P-controller continuation from the context-end belief == the target chunk
    # (between sightings the real BeliefGoalTracker has nothing but dead-reckoning to go on).
    chunk_belief = belief.copy()
    target = np.zeros((horizon, 3), np.float32)
    for h in range(horizon):
        v_fwd, v_left, yaw = p_controller(chunk_belief, turn_kp, base_forward, max_yaw_rate)
        target[h] = [v_fwd, v_left, yaw]
        chunk_belief = propagate_body_point(chunk_belief, v_fwd, v_left, yaw, dt, 0.0, rng)

    return np.stack(seq).astype(np.float32), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context-len", type=int, default=8, help="number of already-elapsed belief steps logged per episode")
    ap.add_argument("--horizon", type=int, default=8, help="length H of the target action chunk")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--turn-kp", type=float, default=1.4)
    ap.add_argument("--base-forward", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument("--bearing0-deg", type=float, default=60.0, help="initial goal bearing sampled in +-this range (matches a typical camera HFOV)")
    ap.add_argument("--range0", type=float, nargs=2, default=[2.0, 10.0])
    ap.add_argument("--p-visible-range", type=float, nargs=2, default=[0.1, 0.9], help="fraction of context steps where the goal is re-observed, sampled per-episode so the dataset covers many occlusion regimes")
    ap.add_argument("--odom-noise-range", type=float, nargs=2, default=[0.0, 0.15], help="dead-reckoning odometry noise std, sampled per-episode")
    ap.add_argument("--obs-noise", type=float, default=0.1, help="observation noise std (m) applied when the goal is re-seen")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.episodes):
        p_visible = float(rng.uniform(*args.p_visible_range))
        odom_noise = float(rng.uniform(*args.odom_noise_range))
        belief_seq, target = simulate_episode(
            rng, args.context_len, args.horizon, args.dt, args.turn_kp, args.base_forward,
            args.max_yaw_rate, p_visible, odom_noise, args.obs_noise, args.bearing0_deg,
            tuple(args.range0),
        )
        np.savez(
            out / f"ep_{i:06d}.npz",
            belief_seq=belief_seq,
            target=target,
            odom_noise=np.float32(odom_noise),
            p_visible=np.float32(p_visible),
        )
        if i % 2000 == 0:
            print(f"[{i}/{args.episodes}] generated", flush=True)
    print(f"saved {args.episodes} episodes to {out}", flush=True)


if __name__ == "__main__":
    main()
