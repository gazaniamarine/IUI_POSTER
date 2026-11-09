#!/usr/bin/env python3
"""train_vla_adapter.py - train a zero-init language adapter on top of a FROZEN S2DiT policy.

The backbone and the text encoder are frozen. The only trainable module is a small MLP
(text_proj) + scalar alpha that turns a sentence embedding into ONE extra conditioning token
appended to the policy's cond set (via the new extra_cond_tokens hook). Zero-init => at start
the token is 0 => the policy is byte-for-byte the one you have. Training moves it only as much
as the counterfactual data (left/right/stop/straight targets on the SAME observation) requires,
which forces the text to be used.

    python scripts/train_vla_adapter.py --samples vla_data_mars/samples \
        --ckpt ../navdp_sam/runs/mars_turn_no_obstacle_action3d/ckpt_last.pt \
        --out runs/vla_adapter --epochs 40
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_habitat_policy import load_model
from navdp.data.habitat_route_dataset import _empty_belief_tensor

PARAPHRASES = {
    "left":  ["pass on the left", "go around the left", "pass it on your left", "take the left side",
              "keep the obstacle on your right", "veer left around it", "go left", "turn left",
              "bear left", "steer to the left", "move to the left", "head left", "leave it on your right",
              "get by on the left", "go past it on the left", "go to the left of it", "avoid it on the left"],
    "right": ["pass on the right", "go around the right", "pass it on your right", "take the right side",
              "keep the obstacle on your left", "veer right around it", "go right", "turn right",
              "bear right", "steer to the right", "move to the right", "head right", "leave it on your left",
              "get by on the right", "go past it on the right", "go to the right of it", "avoid it on the right"],
    "stop":  ["stop", "stop before the obstacle", "halt", "stop and wait", "brake now", "do not move",
              "come to a stop", "hold", "wait here", "stop moving", "stand still", "pull up", "do not proceed"],
    "straight": ["navigate to the goal", "go straight to the goal", "continue to the target", "head to the goal",
                 "keep going to the goal", "drive to the target", "proceed to the goal", "go to the goal",
                 "carry on to the goal", "advance to the target", "reach the goal", "go for the goal"],
}
CLASSES = ["left", "right", "stop", "straight"]
CHUNK_KEY = {"left": "chunk_left", "right": "chunk_right", "stop": "chunk_stop", "straight": "chunk_straight"}


class VLAAdapter(nn.Module):
    """Sentence embedding -> K conditioning tokens (zero-init => identity at start). More tokens
    give the adapter more leverage to override the (dominant) goal mask on the yaw channel."""
    def __init__(self, text_dim: int, dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.dim = int(dim)
        self.proj = nn.Sequential(nn.Linear(text_dim, dim), nn.SiLU(), nn.Linear(dim, dim * self.num_tokens))
        nn.init.zeros_(self.proj[-1].weight)   # zero the FINAL proj -> tokens=0 at init (identity)
        nn.init.zeros_(self.proj[-1].bias)
        # alpha MUST be non-zero: token = alpha*proj, and if BOTH are zero neither ever gets a
        # gradient (each is scaled by the other's zero) -> the adapter is frozen at 0. Init 1 so
        # the token is still 0 at start (proj is zero) but proj receives gradient and learns.
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, e_l: torch.Tensor) -> torch.Tensor:      # [B, text_dim] -> [B, K, dim]
        return (self.alpha * self.proj(e_l)).view(e_l.shape[0], self.num_tokens, self.dim)


def diffusion_loss(policy, cond, actions, freq_weight=0.5):
    """Replicates S2DiTPolicy.forward's loss but on a pre-built cond (so the frozen visual
    encoder runs only once, in precompute)."""
    b = actions.shape[0]
    noise = torch.randn_like(actions)
    t = torch.randint(0, policy.num_train_timesteps, (b,), device=actions.device, dtype=torch.long)
    noisy = policy.add_noise(actions, noise, t)
    t_norm = t.float() / policy.num_train_timesteps
    pred_x0 = policy.dit(noisy, t_norm, cond)
    return F.mse_loss(pred_x0, actions) + freq_weight * policy.freq_loss(pred_x0, actions)


@torch.no_grad()
def precompute_cond(policy, spatials, proprios, obs_maps, belief0, device, bs=16):
    """Frozen policy -> cond tokens per sample, computed ONCE (no text token yet)."""
    conds = []
    for i in range(0, len(spatials), bs):
        sp = torch.from_numpy(np.stack(spatials[i:i + bs])).float().to(device)
        pr = torch.from_numpy(np.stack(proprios[i:i + bs])).float().to(device)
        om = torch.from_numpy(np.stack(obs_maps[i:i + bs])).float().to(device)
        B = sp.shape[0]
        bel = belief0[None].expand(B, *belief0.shape).contiguous().to(device)
        z = torch.zeros(B, dtype=torch.long, device=device)
        c = policy.encode(sp, pr, belief_tensor=bel, obstacle_map=om, route_index=z, active_goal_index=z)
        conds.append(c.cpu())
    return torch.cat(conds, 0)   # [N, T, dim] on cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="dir of *.npz from --vla-dump")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/vla_adapter")
    ap.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num-text-tokens", type=int, default=4, help="conditioning tokens the adapter injects; more = more leverage on yaw.")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    policy, train_args = load_model(Path(args.ckpt), device=device, weights=args.weights)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    dim = int(train_args.get("dim", 512))

    from sentence_transformers import SentenceTransformer
    text_enc = SentenceTransformer(args.text_encoder, device=device)
    text_dim = int(text_enc.get_sentence_embedding_dimension())
    adapter = VLAAdapter(text_dim, dim, num_tokens=args.num_text_tokens).to(device)

    # load all samples into memory (dataset is small)
    files = sorted(glob.glob(str(Path(args.samples) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no *.npz in {args.samples}")
    spatials, proprios, obs_maps, chunks = [], [], [], {c: [] for c in CLASSES}
    for f in files:
        d = np.load(f)
        spatials.append(d["spatial"].astype(np.float32))
        proprios.append(d["proprio"].astype(np.float32))
        obs_maps.append(d["obstacle_map"].astype(np.float32))
        for c in CLASSES:
            chunks[c].append(d[CHUNK_KEY[c]].astype(np.float32))
    N = len(files)
    print(f"loaded {N} samples  (x4 classes = {N*4} examples)  text_dim={text_dim} dim={dim}", flush=True)

    belief0 = torch.from_numpy(np.asarray(_empty_belief_tensor(), dtype=np.float32))
    cond_all = precompute_cond(policy, spatials, proprios, obs_maps, belief0, device)  # [N,T,dim] cpu
    chunks_t = {c: torch.from_numpy(np.stack(chunks[c])).float() for c in CLASSES}      # [N,H,3] cpu

    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        idx = np.random.permutation(N)
        tot, n = 0.0, 0
        for i in range(0, N, args.batch_size):
            bi = idx[i:i + args.batch_size]
            cls = [random.choice(CLASSES) for _ in bi]
            instr = [random.choice(PARAPHRASES[c]) for c in cls]
            with torch.no_grad():
                e_l = torch.from_numpy(text_enc.encode(instr, normalize_embeddings=True)).float().to(device)
            tok = adapter(e_l)                                                 # [B,1,dim] (grad)
            cond = cond_all[bi].to(device)
            cond = torch.cat([cond, tok], dim=1)
            target = torch.stack([chunks_t[c][j] for c, j in zip(cls, bi)]).to(device)
            loss = diffusion_loss(policy, cond, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi); n += len(bi)
        print(f"epoch {ep:3d}  loss={tot/max(n,1):.4f}  alpha={float(adapter.alpha):+.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "text_dim": text_dim, "dim": dim,
                "num_tokens": args.num_text_tokens, "text_encoder": args.text_encoder}, out / "vla_adapter.pt")
    print(f"saved {out/'vla_adapter.pt'}", flush=True)

    # --- eval: instruction-following accuracy across MANY samples (not one) --------------
    # For each held-out obs, sample the chunk under each instruction and see which of the four
    # geometric targets it lands closest to. Diagonal = correct maneuver for the instruction.
    policy.eval(); adapter.eval()
    rng = np.random.default_rng(0)
    eval_idx = rng.choice(N, min(64, N), replace=False)
    conf = np.zeros((4, 4), dtype=np.int64)   # [instructed, nearest-target]
    lyaw, ryaw = [], []
    with torch.no_grad():
        z = torch.zeros(1, dtype=torch.long, device=device)
        bel = belief0[None].to(device)
        emb = {c: torch.from_numpy(text_enc.encode([PARAPHRASES[c][0]], normalize_embeddings=True)).float().to(device)
               for c in CLASSES}
        for j in eval_idx:
            sp = torch.from_numpy(spatials[j][None]).float().to(device)
            pr = torch.from_numpy(proprios[j][None]).float().to(device)
            om = torch.from_numpy(obs_maps[j][None]).float().to(device)
            tgt = {c: chunks_t[c][j].to(device) for c in CLASSES}
            for ci, c in enumerate(CLASSES):
                ch = policy.sample(sp, pr, belief_tensor=bel, obstacle_map=om,
                                   route_index=z, active_goal_index=z, extra_cond_tokens=adapter(emb[c]))[0]
                d = [float(F.mse_loss(ch, tgt[cc])) for cc in CLASSES]
                conf[ci, int(np.argmin(d))] += 1
                if c == "left":
                    lyaw.append(float(ch[:, 2].mean()))
                if c == "right":
                    ryaw.append(float(ch[:, 2].mean()))
    acc = float(np.trace(conf)) / max(int(conf.sum()), 1)
    print(f"\ninstruction -> maneuver accuracy: {acc*100:.1f}%   (chance = 25%)", flush=True)
    print("confusion  rows=instructed [left,right,stop,straight]  cols=nearest target:", flush=True)
    for ci, c in enumerate(CLASSES):
        print(f"  {c:9s} {conf[ci].tolist()}", flush=True)
    print(f"mean yaw:  left={np.mean(lyaw):+.3f}  right={np.mean(ryaw):+.3f}  "
          f"(left-right={np.mean(lyaw)-np.mean(ryaw):+.3f}; want left>right)", flush=True)


if __name__ == "__main__":
    main()
