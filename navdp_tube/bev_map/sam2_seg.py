"""Fine-tuned SAM2.1 (Hiera-B+) terrain segmentation: RGB -> per-pixel class (soil, bedrock, sand, bigrock).

Loads ../best.pth: the SAM2 image encoder (`enc.*`: Hiera-B+ trunk + FPN neck) and a small head (`proj.*`, `head.*`).
The SAM2 trunk / neck code below is a trimmed copy of facebookresearch/sam2 (Apache-2.0,
sam2/modeling/backbones/{hieradet,image_encoder,utils}.py), so the `sam2` package is not needed.

Head (rebuilt from the checkpoint's shapes): the neck's stride 4, 8 and 16 maps (SAM2's scalp=1 drops stride 32)
each go through a 1x1 conv 256 -> 128 (proj.i), are upsampled to stride 4 and summed, then
conv3x3 128 -> GroupNorm -> GELU -> conv3x3 64 -> GroupNorm -> GELU -> conv1x1 -> 4 classes, upsampled to the input.
"""

import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(HERE), "best.pth")
CLASS_NAMES = ("soil", "bedrock", "sand", "bigrock")       # AI4Mars labels, model output index order
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- SAM2 Hiera trunk (trimmed copy)
def _window_partition(x, ws):
    B, H, W, C = x.shape
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, (Hp, Wp)


def _window_unpartition(x, ws, pad_hw, hw):
    (Hp, Wp), (H, W) = pad_hw, hw
    B = x.shape[0] // (Hp * Wp // ws // ws)
    x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous() if (Hp > H or Wp > W) else x


def _do_pool(x, pool):
    return x if pool is None else pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class _MLP(nn.Module):
    def __init__(self, dim, hidden, out):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, hidden), nn.Linear(hidden, out)])

    def forward(self, x):
        return self.layers[1](F.gelu(self.layers[0](x)))


class _Attention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super().__init__()
        self.num_heads, self.q_pool = num_heads, q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W, _ = x.shape
        q, k, v = torch.unbind(self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1), 2)
        if self.q_pool is not None:
            q = _do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(B, H, W, -1))


class _Block(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_stride=None, window_size=0):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.dim, self.dim_out, self.window_size, self.q_stride = dim, dim_out, window_size, q_stride
        self.norm1 = norm(dim)
        self.pool = nn.MaxPool2d(q_stride, q_stride) if q_stride else None
        self.attn = _Attention(dim, dim_out, num_heads, self.pool)
        self.norm2 = norm(dim_out)
        self.mlp = _MLP(dim_out, dim_out * 4, dim_out)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = _do_pool(self.proj(x), self.pool)
        ws = self.window_size
        if ws > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, ws)
        x = self.attn(x)
        if self.q_stride:
            ws = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_hw = (H + (ws - H % ws) % ws, W + (ws - W % ws) % ws)
        if self.window_size > 0:
            x = _window_unpartition(x, ws, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, 7, 4, 3)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class Hiera(nn.Module):
    """SAM2.1 hiera_b+ settings by default."""

    def __init__(self, embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), global_att_blocks=(12, 16, 20),
                 window_spec=(8, 4, 14, 7), bkg_size=(14, 14)):
        super().__init__()
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]]
        self.patch_embed = _PatchEmbed(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *bkg_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))
        self.blocks = nn.ModuleList()
        cur = 1
        for i in range(sum(stages)):
            dim_out = embed_dim
            ws = 0 if i in global_att_blocks else window_spec[cur - 1]
            if i - 1 in self.stage_ends:
                dim_out, num_heads, cur = embed_dim * 2, num_heads * 2, cur + 1
            self.blocks.append(_Block(embed_dim, dim_out, num_heads,
                                      (2, 2) if i in q_pool_blocks else None, ws))
            embed_dim = dim_out

    def forward(self, x):
        x = self.patch_embed(x)
        h, w = x.shape[1:3]
        pe = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pe = pe + self.pos_embed_window.tile([a // b for a, b in zip(pe.shape, self.pos_embed_window.shape)])
        x = x + pe.permute(0, 2, 3, 1)
        out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:
                out.append(x.permute(0, 3, 1, 2))
        return out                                   # strides 4, 8, 16, 32


class FpnNeck(nn.Module):
    """SAM2's FPN neck (no position encoding: the head does not use it)."""

    def __init__(self, d_model=256, channels=(896, 448, 224, 112), top_down_levels=(2, 3), interp="nearest"):
        super().__init__()
        self.convs = nn.ModuleList()
        for c in channels:
            m = nn.Sequential()
            m.add_module("conv", nn.Conv2d(c, d_model, 1))
            self.convs.append(m)
        self.top_down, self.interp = set(top_down_levels), interp

    def forward(self, xs):
        n = len(self.convs) - 1
        out, prev = [None] * len(xs), None
        for i in range(n, -1, -1):
            lat = self.convs[n - i](xs[i])
            if i in self.top_down and prev is not None:
                prev = lat + F.interpolate(prev.float(), scale_factor=2.0, mode=self.interp).to(lat.dtype)
            else:
                prev = lat
            out[i] = prev
        return out                                   # strides 4, 8, 16, 32


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk, self.neck = Hiera(), FpnNeck()

    def forward(self, x):
        return self.neck(self.trunk(x))[:-1]         # scalp=1: stride 4, 8, 16


# ---------------------------------------------------------------- the fine-tuned model
class SAM2Seg(nn.Module):
    def __init__(self, num_classes=4, groups=8, act="gelu", up="bilinear"):
        super().__init__()
        A = nn.GELU if act == "gelu" else partial(nn.ReLU, inplace=True)
        self.enc = ImageEncoder()
        self.proj = nn.ModuleList([nn.Conv2d(256, 128, 1) for _ in range(3)])
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(groups, 128), A(),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(groups, 64), A(),
                                  nn.Conv2d(64, num_classes, 1))
        self.up = up

    def forward(self, x):
        feats = self.enc(x)
        size = feats[0].shape[-2:]
        y = 0
        for p, f in zip(self.proj, feats):
            f = p(f)
            if f.shape[-2:] != size:
                f = F.interpolate(f, size=size, mode=self.up,
                                  align_corners=False if self.up == "bilinear" else None)
            y = y + f
        y = self.head(y)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


class SAM2Segmenter:
    """List of HxWx3 uint8 RGB frames -> list of (labels HxW uint8, probs CxHxW float32) at input resolution."""

    def __init__(self, checkpoint=None, image_size=1024, device=None, fp16=True, **head_kw):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        self.size = image_size
        path = checkpoint or DEFAULT_CHECKPOINT
        print(f"[Seg] loading SAM2 segmentation {path} on {self.device} ({self.dtype})")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        self.model = SAM2Seg(**head_kw)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device, self.dtype).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, rgb_frames):
        out = []
        for rgb in rgb_frames:
            h, w = rgb.shape[:2]
            x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
            x = ((x - self.mean) / self.std).to(self.dtype)
            logits = self.model(x).float()
            probs = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).softmax(1)[0]
            out.append((probs.argmax(0).to(torch.uint8).cpu().numpy(), probs.cpu().numpy()))
        return out
