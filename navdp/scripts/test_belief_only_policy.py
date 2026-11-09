#!/usr/bin/env python3
"""test_belief_only_policy.py -- load a checkpoint saved by train_belief_only_policy.py and
score it, with NO retraining:

  1. (optional, if --samples is given) held-out reconstruction quality: diffusion loss and
     sampled-chunk action MSE against a dataset of ep_*.npz episodes (e.g. a split generated
     separately from the training set via gen_belief_propagation_data.py --seed <other>).
  2. turn-toward-goal sign accuracy on synthetic dead-reckoned-only contexts.
  3. closed-loop belief-space propagation-robustness sweep vs. the scripted P-controller.

(2) and (3) are the same evals train_belief_only_policy.py runs right after training --
reused here (not duplicated) so this script always tests exactly what training reported.

    python scripts/test_belief_only_policy.py --checkpoint runs/belief_only_policy/belief_only_policy.pt
    python scripts/test_belief_only_policy.py --checkpoint runs/belief_only_policy/belief_only_policy.pt \
        --samples belief_only_data/held_out
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from train_belief_only_policy import (  # noqa: E402
    FEAT_DIM,
    BeliefOnlyDiffusionPolicy,
    eval_propagation_rollout,
    eval_turn_toward_goal,
    load_dataset,
)


def load_policy(checkpoint: str, device: str) -> BeliefOnlyDiffusionPolicy:
    ckpt = torch.load(checkpoint, map_location=device)
    assert ckpt["feat_dim"] == FEAT_DIM, f"checkpoint feat_dim={ckpt['feat_dim']} != expected {FEAT_DIM}"
    policy = BeliefOnlyDiffusionPolicy(
        feat_dim=ckpt["feat_dim"],
        horizon=ckpt["horizon"],
        dim=ckpt["dim"],
        depth=ckpt["depth"],
        heads=ckpt["heads"],
        num_train_timesteps=ckpt["num_train_timesteps"],
        num_inference_steps=ckpt["num_inference_steps"],
    ).to(device)
    policy.belief_encoder.load_state_dict(ckpt["belief_encoder"])
    policy.dit.load_state_dict(ckpt["dit"])
    policy.eval()
    return policy


@torch.no_grad()
def eval_held_out(policy: BeliefOnlyDiffusionPolicy, samples_dir: str, device: str, batch_size: int):
    belief_seqs, targets, odom_noises, p_visibles = load_dataset(samples_dir)
    N, context_len, feat_dim = belief_seqs.shape
    assert feat_dim == FEAT_DIM, f"expected feat_dim={FEAT_DIM}, got {feat_dim}"
    assert targets.shape[1] == policy.horizon, (
        f"dataset horizon={targets.shape[1]} != checkpoint horizon={policy.horizon}"
    )
    belief_seqs_t = torch.from_numpy(belief_seqs).float()
    targets_t = torch.from_numpy(targets).float()

    tot_loss, tot_mse, n = 0.0, 0.0, 0
    for i in range(0, N, batch_size):
        bs = belief_seqs_t[i : i + batch_size].to(device)
        tg = targets_t[i : i + batch_size].to(device)
        loss = policy.loss(bs, tg)
        pred = policy.sample(bs)
        mse = torch.mean((pred - tg) ** 2)
        tot_loss += loss.item() * len(bs)
        tot_mse += mse.item() * len(bs)
        n += len(bs)
    print(
        f"\nheld-out ({n} episodes from {samples_dir}):  diffusion_loss={tot_loss/max(n,1):.4f}  "
        f"sampled_action_mse={tot_mse/max(n,1):.4f}",
        flush=True,
    )
    return context_len, float(np.median(p_visibles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="belief_only_policy.pt from train_belief_only_policy.py")
    ap.add_argument("--samples", default=None, help="optional dir of held-out ep_*.npz to score reconstruction quality")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--context-len", type=int, default=8, help="used for the synthetic evals when --samples is not given")
    ap.add_argument("--p-visible", type=float, default=0.5, help="used for the rollout sweep when --samples is not given")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--turn-kp", type=float, default=1.4)
    ap.add_argument("--base-forward", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument("--rollout-steps", type=int, default=20)
    ap.add_argument("--rollout-trials", type=int, default=30)
    ap.add_argument("--odom-noise-sweep", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.checkpoint, device)
    print(
        f"loaded checkpoint {args.checkpoint}  horizon={policy.horizon}  dim={policy.dit.dim if hasattr(policy.dit, 'dim') else '?'}",
        flush=True,
    )

    context_len, p_visible = args.context_len, args.p_visible
    if args.samples:
        context_len, p_visible = eval_held_out(policy, args.samples, device, args.batch_size)

    rng = np.random.default_rng(args.seed)
    eval_turn_toward_goal(policy, context_len, device, rng)
    eval_propagation_rollout(
        policy, context_len, args.rollout_steps, device,
        odom_noise_sweep=args.odom_noise_sweep, p_visible=p_visible,
        dt=args.dt, turn_kp=args.turn_kp, base_forward=args.base_forward,
        max_yaw_rate=args.max_yaw_rate, trials=args.rollout_trials, seed=args.seed,
    )


if __name__ == "__main__":
    main()
