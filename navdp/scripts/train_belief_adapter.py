#!/usr/bin/env python3
"""train_belief_adapter.py -- teach a FROZEN S2DiT policy the belief-driven RETURN.

Same recipe as train_vla_adapter.py, but the conditioning signal is the goal BEARING (from the
propagated belief) instead of a sentence embedding, and the counterfactual targets are the
P-controller return chunks. On OUT-OF-VIEW observations (empty goal mask) the belief is the ONLY
cue, so the adapter is forced to make the policy turn back to the off-screen goal -- the behavior
the field-of-view-limited mask cannot express.

    python scripts/train_belief_adapter.py --samples belief_return_data/samples \
        --ckpt ../navdp_sam/runs/mars_dp_finetune_action3d/ckpt_last.pt \
        --out runs/belief_adapter --epochs 60
"""

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_habitat_policy import load_model
from navdp.data.habitat_route_dataset import _empty_belief_tensor
from train_vla_adapter import (
    VLAAdapter,
    diffusion_loss,
    precompute_cond,
)  # reuse the exact recipe

BELIEF_FEAT_DIM = 3  # [cos(bearing), sin(bearing), range/scale]
R_SCALE = 10.0


def belief_feat(belief):
    """body-frame [forward,left] -> rotation-aware feature (avoids the +-pi wrap)."""
    f, l = float(belief[0]), float(belief[1])
    bearing = math.atan2(l, f)
    rng = math.hypot(f, l)
    return np.array(
        [math.cos(bearing), math.sin(bearing), min(rng / R_SCALE, 1.0)], np.float32
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples",
        required=True,
        help="dir of scene_*.npz from dump_belief_return_mars.py",
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/belief_adapter")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num-tokens", type=int, default=4)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    policy, train_args = load_model(
        Path(args.ckpt), device=device, weights=args.weights
    )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    dim = int(train_args.get("dim", 512))
    adapter = VLAAdapter(BELIEF_FEAT_DIM, dim, num_tokens=args.num_tokens).to(device)

    files = sorted(glob.glob(str(Path(args.samples) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no *.npz in {args.samples}")
    spatials, proprios, obs_maps, beliefs, targets = [], [], [], [], []
    for f in files:
        d = np.load(f)
        spatials.append(d["spatial"].astype(np.float32))
        proprios.append(d["proprio"].astype(np.float32))
        obs_maps.append(d["obstacle_map"].astype(np.float32))
        beliefs.append(d["beliefs"].astype(np.float32))  # [K,2]
        targets.append(d["targets"].astype(np.float32))  # [K,H,3]
    N = len(files)
    K = beliefs[0].shape[0]
    print(
        f"loaded {N} obs x {K} beliefs = {N*K} counterfactual examples  dim={dim}",
        flush=True,
    )

    belief0 = torch.from_numpy(np.asarray(_empty_belief_tensor(), dtype=np.float32))
    cond_all = precompute_cond(
        policy, spatials, proprios, obs_maps, belief0, device
    )  # [N,T,dim] cpu
    targets_t = [torch.from_numpy(t).float() for t in targets]  # list of [K,H,3]

    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        idx = rng.permutation(N)
        tot, n = 0.0, 0
        for i in range(0, N, args.batch_size):
            bi = idx[i : i + args.batch_size]
            ks = rng.integers(0, K, size=len(bi))
            feats = np.stack([belief_feat(beliefs[j][k]) for j, k in zip(bi, ks)])
            e = torch.from_numpy(feats).float().to(device)
            tok = adapter(e)
            cond = torch.cat([cond_all[bi].to(device), tok], dim=1)
            target = torch.stack([targets_t[j][k] for j, k in zip(bi, ks)]).to(device)
            loss = diffusion_loss(policy, cond, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
            n += len(bi)
        print(
            f"epoch {ep:3d}  loss={tot/max(n,1):.4f}  alpha={float(adapter.alpha):+.3f}",
            flush=True,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "belief_feat_dim": BELIEF_FEAT_DIM,
            "dim": dim,
            "num_tokens": args.num_tokens,
            "r_scale": R_SCALE,
        },
        out / "belief_adapter.pt",
    )
    print(f"saved {out/'belief_adapter.pt'}", flush=True)

    # --- eval: does the policy turn TOWARD the off-screen goal? (sign of yaw == sign of bearing) ---
    policy.eval()
    adapter.eval()
    test_bearings = np.deg2rad([-150, -120, -90, 90, 120, 150])
    eval_idx = rng.choice(N, min(64, N), replace=False)
    hits = tot_n = 0
    lyaw, ryaw = [], []
    with torch.no_grad():
        z = torch.zeros(1, dtype=torch.long, device=device)
        bel = belief0[None].to(device)
        for j in eval_idx:
            sp = torch.from_numpy(spatials[j][None]).float().to(device)
            pr = torch.from_numpy(proprios[j][None]).float().to(device)
            om = torch.from_numpy(obs_maps[j][None]).float().to(device)
            for beta in test_bearings:
                feat = torch.from_numpy(
                    np.array([[math.cos(beta), math.sin(beta), 0.6]], np.float32)
                ).to(device)
                ch = policy.sample(
                    sp,
                    pr,
                    belief_tensor=bel,
                    obstacle_map=om,
                    route_index=z,
                    active_goal_index=z,
                    extra_cond_tokens=adapter(feat),
                )[0]
                yaw = float(ch[:, 2].mean())
                hits += int(np.sign(yaw) == np.sign(beta))
                tot_n += 1
                (lyaw if beta > 0 else ryaw).append(yaw)
    print(
        f"\nturn-toward-goal accuracy: {hits/max(tot_n,1)*100:.1f}%   (chance = 50%)",
        flush=True,
    )
    print(
        f"mean yaw:  goal-left(beta>0)={np.mean(lyaw):+.3f}  goal-right(beta<0)={np.mean(ryaw):+.3f}  "
        f"(want left>0>right)",
        flush=True,
    )


if __name__ == "__main__":
    main()
