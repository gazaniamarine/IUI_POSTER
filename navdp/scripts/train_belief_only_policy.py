#!/usr/bin/env python3
"""train_belief_only_policy.py -- train a small diffusion policy conditioned PURELY on a
propagated goal-belief sequence, with ZERO dependence on NavDP's frozen, image-conditioned
S2DiT backbone or its pretrained waypoint-generation weights.

Unlike train_vla_adapter.py / train_belief_adapter.py (which append a small zero-init
token onto a FROZEN policy's real-image conditioning -- S2DiTPolicy.encode()/sample()
require a real spatial_semantic/proprio tensor and unconditionally run the CNN image
encoder first, so there is no path to drive the frozen policy on belief alone), this
script trains a fresh, small BeliefOnlyDiffusionPolicy end-to-end: a small transformer
encodes the FULL propagated belief sequence (as it drifts under dead-reckoning / snaps
back on re-observation -- see gen_belief_propagation_data.py) into conditioning tokens for
a freshly-initialized FlowMatchingDiT (reused from navdp/model.py -- a generic conditional
denoiser with no built-in image dependency). This isolates belief-propagation dynamics as
the ONLY signal driving the action chunk, for experimenting with how belief propagates.

    python scripts/gen_belief_propagation_data.py --episodes 20000 --out belief_only_data/samples
    python scripts/train_belief_only_policy.py --samples belief_only_data/samples \
        --out runs/belief_only_policy --epochs 60
"""
import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import FlowMatchingDiT, FrequencyDomainLoss  # generic denoiser/aux-loss, no image coupling

FEAT_DIM = 4  # [cos(bearing), sin(bearing), range/R_SCALE, visible] -- see gen_belief_propagation_data.py


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Same recipe as model_s2_dit.py's _cosine_beta_schedule, inlined so this script has
    no import path through the S2DiT policy file -- literal independence from NavDP's
    waypoint generation, not just weight independence."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return betas.clamp(1e-4, 0.999)


class BeliefSequenceEncoder(nn.Module):
    """Propagated belief sequence [B,C,feat_dim] -> per-step condition tokens [B,C,dim].
    Kept as one token per context step (not pooled) so the DiT cross-attends to the belief's
    full temporal trajectory -- how/when it was re-observed vs. dead-reckoned -- not just a
    single summary vector."""

    def __init__(self, feat_dim: int, dim: int, num_layers: int = 2, heads: int = 4, max_len: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, belief_seq: torch.Tensor) -> torch.Tensor:
        b, c, _ = belief_seq.shape
        x = self.in_proj(belief_seq) + self.pos[:, :c]
        return self.norm(self.encoder(x))


class BeliefOnlyDiffusionPolicy(nn.Module):
    def __init__(
        self, feat_dim: int, horizon: int, action_dim: int = 3, dim: int = 192,
        depth: int = 4, heads: int = 4, num_train_timesteps: int = 100,
        num_inference_steps: int = 20,
    ):
        super().__init__()
        self.belief_encoder = BeliefSequenceEncoder(feat_dim, dim, heads=heads)
        self.dit = FlowMatchingDiT(action_dim, horizon, dim=dim, depth=depth, heads=heads, cond_dim=dim)
        self.freq_loss = FrequencyDomainLoss(horizon)
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        betas = _cosine_beta_schedule(num_train_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def add_noise(self, clean_actions: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        return sqrt_alpha * clean_actions + sqrt_one_minus * noise

    def loss(self, belief_seq: torch.Tensor, actions: torch.Tensor, freq_weight: float = 0.5) -> torch.Tensor:
        cond = self.belief_encoder(belief_seq)
        b = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.num_train_timesteps, (b,), device=actions.device, dtype=torch.long)
        noisy = self.add_noise(actions, noise, t)
        t_norm = t.float() / self.num_train_timesteps
        pred_x0 = self.dit(noisy, t_norm, cond)
        return F.mse_loss(pred_x0, actions) + freq_weight * self.freq_loss(pred_x0, actions)

    def _ddim_prev(self, x_t: torch.Tensor, pred_x0: torch.Tensor, t: int, prev_t: int) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t].view(1, 1, 1)
        alpha_prev = self.alphas_cumprod[prev_t].view(1, 1, 1) if prev_t >= 0 else torch.ones_like(alpha_t)
        eps = (x_t - alpha_t.sqrt() * pred_x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-6)
        return alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * eps

    @torch.no_grad()
    def sample(self, belief_seq: torch.Tensor, steps: int = None) -> torch.Tensor:
        cond = self.belief_encoder(belief_seq)
        b = belief_seq.shape[0]
        x = torch.randn(b, self.horizon, self.action_dim, device=belief_seq.device)
        n = steps or self.num_inference_steps
        timesteps = torch.linspace(self.num_train_timesteps - 1, 0, n, device=belief_seq.device).long()
        for i, t_tensor in enumerate(timesteps):
            t = int(t_tensor.item())
            tb = torch.full((b,), t, device=x.device, dtype=torch.float32)
            pred_x0 = self.dit(x, tb / self.num_train_timesteps, cond)
            prev_t = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            x = self._ddim_prev(x, pred_x0, t, prev_t)
        return x


# --------------------------------------------------------------------------------------
# belief-space simulation helpers, shared with gen_belief_propagation_data.py's recipe --
# duplicated (not imported) so the closed-loop eval below can run against an arbitrary
# odom_noise/p_visible sweep without re-invoking the generator script.
# --------------------------------------------------------------------------------------
R_SCALE = 10.0


def belief_feat(belief: np.ndarray, visible: bool) -> np.ndarray:
    f, l = float(belief[0]), float(belief[1])
    bearing = math.atan2(l, f)
    rng = math.hypot(f, l)
    return np.array(
        [math.cos(bearing), math.sin(bearing), min(rng / R_SCALE, 1.0), float(visible)], np.float32
    )


def propagate_body_point(
    bg: np.ndarray, v_fwd: float, v_left: float, yaw_rate: float, dt: float,
    odom_noise: float, rng: np.random.Generator,
) -> np.ndarray:
    if odom_noise > 0.0:
        v_fwd = v_fwd + float(rng.normal(0.0, odom_noise))
        yaw_rate = yaw_rate + float(rng.normal(0.0, odom_noise))
    th = -float(yaw_rate) * float(dt)
    c, s = math.cos(th), math.sin(th)
    qx = float(bg[0]) - float(v_fwd) * float(dt)
    qy = float(bg[1]) - float(v_left) * float(dt)
    return np.array([c * qx - s * qy, s * qx + c * qy], np.float32)


def p_controller(belief: np.ndarray, turn_kp: float, base_forward: float, max_yaw_rate: float):
    bearing = math.atan2(float(belief[1]), float(belief[0]))
    yaw = float(np.clip(turn_kp * bearing, -max_yaw_rate, max_yaw_rate))
    fwd = base_forward * max(0.0, math.cos(bearing))
    return fwd, 0.0, yaw


def load_dataset(samples_dir: str):
    files = sorted(glob.glob(str(Path(samples_dir) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no *.npz in {samples_dir}")
    belief_seqs, targets, odom_noises, p_visibles = [], [], [], []
    for f in files:
        d = np.load(f)
        belief_seqs.append(d["belief_seq"].astype(np.float32))
        targets.append(d["target"].astype(np.float32))
        odom_noises.append(float(d["odom_noise"]))
        p_visibles.append(float(d["p_visible"]))
    return (
        np.stack(belief_seqs), np.stack(targets),
        np.array(odom_noises, np.float32), np.array(p_visibles, np.float32),
    )


@torch.no_grad()
def eval_turn_toward_goal(policy: BeliefOnlyDiffusionPolicy, context_len: int, device: str, rng: np.random.Generator):
    """Synthetic dead-reckoned-only contexts at fixed test bearings: does the predicted
    chunk's yaw sign match the bearing sign? (mirrors train_belief_adapter.py's eval)."""
    test_bearings = np.deg2rad([-150, -120, -90, 90, 120, 150])
    hits, tot = 0, 0
    lyaw, ryaw = [], []
    for beta in test_bearings:
        belief = np.array([R_SCALE * 0.6 * math.cos(beta), R_SCALE * 0.6 * math.sin(beta)], np.float32)
        seq = [belief_feat(belief, visible=True)]
        for _ in range(context_len - 1):
            belief = propagate_body_point(belief, 0.0, 0.0, 0.0, 0.5, 0.0, rng)
            seq.append(belief_feat(belief, visible=False))
        seq_t = torch.from_numpy(np.stack(seq)[None]).float().to(device)
        chunk = policy.sample(seq_t)[0]
        yaw = float(chunk[:, 2].mean())
        hits += int(np.sign(yaw) == np.sign(beta))
        tot += 1
        (lyaw if beta > 0 else ryaw).append(yaw)
    print(f"\nturn-toward-goal accuracy: {hits/max(tot,1)*100:.1f}%   (chance = 50%)", flush=True)
    print(
        f"mean yaw:  goal-left(beta>0)={np.mean(lyaw):+.3f}  goal-right(beta<0)={np.mean(ryaw):+.3f}  "
        f"(want left>0>right)", flush=True,
    )


@torch.no_grad()
def eval_propagation_rollout(
    policy: BeliefOnlyDiffusionPolicy, context_len: int, rollout_steps: int, device: str,
    odom_noise_sweep, p_visible: float, dt: float, turn_kp: float, base_forward: float,
    max_yaw_rate: float, trials: int, seed: int,
):
    """Closed-loop belief-space rollout: at each step sample a chunk, execute its first
    action (receding horizon), propagate the true goal (noise-free) and belief (re-seed if
    visible else dead-reckon with odom_noise), slide the context window. Compares the
    trained policy's final distance-to-goal against the scripted P-controller baseline
    under the SAME noise realization -- this is the actual belief-propagation experiment."""
    print("\npropagation-robustness sweep (closed-loop belief-space rollout):", flush=True)
    print(f"{'odom_noise':>10}  {'policy_dist':>12}  {'pctrl_dist':>11}", flush=True)
    for odom_noise in odom_noise_sweep:
        policy_finals, pctrl_finals = [], []
        for trial in range(trials):
            rng = np.random.default_rng(seed * 10_000 + trial)
            bearing0 = math.radians(rng.uniform(-60, 60))
            range0 = rng.uniform(2.0, 10.0)
            true_goal = np.array([range0 * math.cos(bearing0), range0 * math.sin(bearing0)], np.float32)
            belief_policy = true_goal.copy()
            belief_pctrl = true_goal.copy()
            true_goal_pctrl = true_goal.copy()
            window = [belief_feat(belief_policy, visible=True)]

            for step in range(rollout_steps):
                seq_t = torch.from_numpy(np.stack(window[-context_len:])[None]).float().to(device)
                chunk = policy.sample(seq_t)[0].cpu().numpy()
                v_fwd, v_lat, yaw = float(chunk[0, 0]), float(chunk[0, 1]), float(chunk[0, 2])
                true_goal = propagate_body_point(true_goal, v_fwd, v_lat, yaw, dt, 0.0, rng)
                visible = bool(rng.uniform() < p_visible)
                if visible:
                    belief_policy = true_goal.copy()
                else:
                    belief_policy = propagate_body_point(belief_policy, v_fwd, v_lat, yaw, dt, odom_noise, rng)
                window.append(belief_feat(belief_policy, visible))

                pv_fwd, pv_lat, pyaw = p_controller(belief_pctrl, turn_kp, base_forward, max_yaw_rate)
                true_goal_pctrl = propagate_body_point(true_goal_pctrl, pv_fwd, pv_lat, pyaw, dt, 0.0, rng)
                if visible:
                    belief_pctrl = true_goal_pctrl.copy()
                else:
                    belief_pctrl = propagate_body_point(belief_pctrl, pv_fwd, pv_lat, pyaw, dt, odom_noise, rng)

            policy_finals.append(math.hypot(*true_goal))
            pctrl_finals.append(math.hypot(*true_goal_pctrl))
        print(
            f"{odom_noise:10.3f}  {np.mean(policy_finals):12.3f}  {np.mean(pctrl_finals):11.3f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="dir of ep_*.npz from gen_belief_propagation_data.py")
    ap.add_argument("--out", default="runs/belief_only_policy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num-train-timesteps", type=int, default=100)
    ap.add_argument("--num-inference-steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--turn-kp", type=float, default=1.4)
    ap.add_argument("--base-forward", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument("--rollout-steps", type=int, default=20)
    ap.add_argument("--rollout-trials", type=int, default=30)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    belief_seqs, targets, odom_noises, p_visibles = load_dataset(args.samples)
    N, context_len, feat_dim = belief_seqs.shape
    horizon = targets.shape[1]
    print(
        f"loaded {N} episodes  context_len={context_len}  horizon={horizon}  feat_dim={feat_dim}",
        flush=True,
    )
    assert feat_dim == FEAT_DIM, f"expected feat_dim={FEAT_DIM}, got {feat_dim}"

    policy = BeliefOnlyDiffusionPolicy(
        feat_dim=FEAT_DIM, horizon=horizon, dim=args.dim, depth=args.depth, heads=args.heads,
        num_train_timesteps=args.num_train_timesteps, num_inference_steps=args.num_inference_steps,
    ).to(device)

    belief_seqs_t = torch.from_numpy(belief_seqs).float()
    targets_t = torch.from_numpy(targets).float()

    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        idx = rng.permutation(N)
        tot, n = 0.0, 0
        for i in range(0, N, args.batch_size):
            bi = idx[i : i + args.batch_size]
            bs = belief_seqs_t[bi].to(device)
            tg = targets_t[bi].to(device)
            loss = policy.loss(bs, tg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
            n += len(bi)
        print(f"epoch {ep:3d}  loss={tot/max(n,1):.4f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "belief_encoder": policy.belief_encoder.state_dict(),
            "dit": policy.dit.state_dict(),
            "feat_dim": FEAT_DIM,
            "horizon": horizon,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "num_train_timesteps": args.num_train_timesteps,
            "num_inference_steps": args.num_inference_steps,
        },
        out / "belief_only_policy.pt",
    )
    print(f"saved {out/'belief_only_policy.pt'}", flush=True)

    policy.eval()
    eval_turn_toward_goal(policy, context_len, device, rng)
    eval_propagation_rollout(
        policy, context_len, args.rollout_steps, device,
        odom_noise_sweep=[0.0, 0.05, 0.1, 0.2, 0.3], p_visible=float(np.median(p_visibles)),
        dt=args.dt, turn_kp=args.turn_kp, base_forward=args.base_forward,
        max_yaw_rate=args.max_yaw_rate, trials=args.rollout_trials, seed=0,
    )


if __name__ == "__main__":
    main()
