"""Activation recorder for NavDP and X-NavDP (for training sparse autoencoders).

Hooks the policy network once; a request that asks for activations arms it, and after the step ``save`` writes one
.npz (float16) with:
  scene           [T, D]     the RGB-D encoder's scene tokens (memory_size x 16 tokens), for the current frame
  decoder         [S, P, D]  the output of decoder layer ``layer`` at the last denoising step, every sample
  decoder_name    str        which decoder ran that step (X-NavDP: decoder_ft for its last steps)
  denoise_calls   int        denoising calls seen in this step
  critic_tokens   [S, P, D]  the critic's last normalised token features (NavDP: layernorm, X-NavDP: layernorm_q)
  critic          [S, D]     the critic feature right before the value head (NavDP: mean of critic_tokens,
                             X-NavDP: q_pool_mlp output)
S = samples, P = waypoints (24), D = 384. Works with both models: each hook is only attached where the module exists.
"""
from __future__ import annotations

import os

import numpy as np


def _np(t):
    return t.detach().float().cpu().numpy().astype(np.float16)


class ActRecorder:
    def __init__(self, model, layer=None):
        self.model = model
        n = len(model.decoder.layers)
        self.layer = n // 2 if layer is None else (int(layer) % n)   # default: the middle decoder layer
        self.armed, self.phase, self.buf = False, None, {}
        self._wrap("predict_noise", "noise:decoder")
        self._wrap("predict_noise_ft", "noise:decoder_ft")
        self._wrap("predict_critic", "critic")
        self._wrap("predict_pointgoal_q", "critic")
        model.rgbd_encoder.register_forward_hook(self._scene)
        for name in ("decoder", "decoder_ft"):
            dec = getattr(model, name, None)
            if dec is not None:
                dec.layers[self.layer].register_forward_hook(self._decoder(name))
        model.layernorm.register_forward_hook(self._critic_tokens)          # NavDP's critic (and its noise path)
        if hasattr(model, "critic_head"):
            model.critic_head.register_forward_pre_hook(self._critic_pooled)
        if hasattr(model, "layernorm_q"):                                    # X-NavDP's Q critic
            model.layernorm_q.register_forward_hook(self._critic_tokens)
            model.q_pool_mlp.register_forward_hook(lambda m, i, o: self._put("critic", o))

    def _wrap(self, name, phase):
        fn = getattr(self.model, name, None)
        if fn is None:
            return

        def wrapped(*a, **kw):
            old, self.phase = self.phase, phase
            if self.armed and phase.startswith("noise"):
                self.buf["denoise_calls"] = self.buf.get("denoise_calls", 0) + 1
            try:
                return fn(*a, **kw)
            finally:
                self.phase = old
        setattr(self.model, name, wrapped)     # instance attribute: callers look it up at call time

    def _put(self, key, tensor):
        if self.armed:
            self.buf[key] = _np(tensor)

    def _scene(self, module, inputs, output):
        if self.armed:
            self.buf["scene"] = _np(output[0])           # before the per-sample repeat: batch row 0

    def _decoder(self, name):
        def hook(module, inputs, output):
            if self.armed and self.phase == "noise:" + name:
                self.buf["decoder"] = _np(output)          # overwritten each call: the last denoising step stays
                self.buf["decoder_name"] = name
        return hook

    def _critic_tokens(self, module, inputs, output):
        if self.phase == "critic":
            self._put("critic_tokens", output)

    def _critic_pooled(self, module, inputs):
        if self.phase == "critic":
            self._put("critic", inputs[0])

    def arm(self):
        self.armed, self.buf = True, {}

    def save(self, path, **extra):
        """Write the recorded step to path (.npz) and disarm. -> True if written."""
        self.armed = False
        if not path or "scene" not in self.buf:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = {k: np.asarray(v) for k, v in self.buf.items()}
        out["layer"] = np.int32(self.layer)
        for k, v in extra.items():
            out[k] = np.asarray(v)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, path)
        self.buf = {}
        return True
