#!/usr/bin/env python3
"""Generate parity fixtures (input.json, expected_output.json) for nam-rs.

The canonical way to produce these is to run the reference Python NAM
(``pip install neural-amp-modeler``) and dump its forward pass. That pulls in
torch and only ships wheels for the Python versions torch supports, so it is not
always available. To keep fixture generation reproducible anywhere numpy is
present, this script instead implements the WaveNet forward pass *directly from
the reference weight layout and math*, faithfully ported from:

  - NeuralAudio (Mike Oliphant, MIT) -- ``NeuralAudio/WaveNet.h``. Primary
    porting reference; matches NAM Core exactly.
  - neural-amp-modeler (Steven Atkinson, MIT) -- ``nam/models/wavenet/*.py``.
    The source of truth for ``export_weights`` ordering.

Both references agree on the algorithm implemented here, and the weight count it
derives from ``config`` matches the model file exactly (a structural check that
the layout is correct). Output is computed in float32 to match NAM Core's
inference precision. The Rust crate must reproduce this output within 1e-5.

If you have torch available, regenerate from the canonical implementation
instead and overwrite expected_output.json -- the numbers should agree.

Usage:
    python3 tests/fixtures/gen_fixtures.py [path/to/model.nam]

Defaults to the committed tests/fixtures/reference.nam.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32


class Cursor:
    """Consumes the flat weight blob in NAM ``export_weights`` order."""

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=F32)
        self.i = 0

    def take(self, n):
        chunk = self.w[self.i : self.i + n]
        if len(chunk) != n:
            raise ValueError(f"weight blob exhausted: wanted {n}, had {len(chunk)}")
        self.i += n
        return chunk

    def conv(self, out_ch, in_ch, k, bias):
        """Dilated/1x1 conv weights: [out][in][kernel] then optional [out] bias."""
        w = self.take(out_ch * in_ch * k).reshape(out_ch, in_ch, k)
        b = self.take(out_ch) if bias else None
        return w, b

    def dense(self, out_ch, in_ch, bias):
        """1x1 conv as a matmul: [out][in] then optional [out] bias."""
        w = self.take(out_ch * in_ch).reshape(out_ch, in_ch)
        b = self.take(out_ch) if bias else None
        return w, b

    def scalar(self):
        return float(self.take(1)[0])

    def done(self):
        return self.i == len(self.w)


def causal_dilated_conv(x, w, b, dilation):
    """x: (in_ch, L); w: (out_ch, in_ch, K); returns (out_ch, L), causal.

    out[t] = sum_k w[:, :, k] @ x[t - (K-1-k)*dilation], left-padded with silence.
    """
    out_ch, in_ch, k = w.shape
    length = x.shape[1]
    pad = (k - 1) * dilation
    xp = np.concatenate([np.zeros((in_ch, pad), F32), x], axis=1)
    out = np.zeros((out_ch, length), F32)
    for tap in range(k):
        seg = xp[:, tap * dilation : tap * dilation + length]
        out += (w[:, :, tap] @ seg).astype(F32)
    if b is not None:
        out += b[:, None]
    return out.astype(F32)


def matmul1x1(x, w, b):
    """x: (in_ch, L); w: (out_ch, in_ch); returns (out_ch, L)."""
    out = (w @ x).astype(F32)
    if b is not None:
        out += b[:, None]
    return out.astype(F32)


ACTIVATIONS = {
    "Tanh": np.tanh,
    "ReLU": lambda z: np.maximum(z, F32(0.0)),
    "Sigmoid": lambda z: (1.0 / (1.0 + np.exp(-z))).astype(F32),
}


def activate(name, z, gated):
    if gated:
        half = z.shape[0] // 2
        a = ACTIVATIONS[name](z[:half]).astype(F32)
        g = ACTIVATIONS["Sigmoid"](z[half:]).astype(F32)
        return (a * g).astype(F32)
    return ACTIVATIONS[name](z).astype(F32)


class Layer:
    def __init__(self, cur, channels, condition_size, kernel, dilation, activation, gated):
        self.activation = activation
        self.gated = gated
        self.dilation = dilation
        mid = 2 * channels if gated else channels
        self.conv_w, self.conv_b = cur.conv(mid, channels, kernel, bias=True)
        self.mix_w, self.mix_b = cur.dense(mid, condition_size, bias=False)
        self.one_w, self.one_b = cur.dense(channels, channels, bias=True)

    def forward(self, x, condition):
        z = causal_dilated_conv(x, self.conv_w, self.conv_b, self.dilation)
        z = z + matmul1x1(condition, self.mix_w, self.mix_b)
        post = activate(self.activation, z, self.gated)
        out = matmul1x1(post, self.one_w, self.one_b) + x
        return out.astype(F32), post.astype(F32)


class LayerArray:
    def __init__(self, cur, cfg):
        C = cfg["channels"]
        self.channels = C
        self.rechan_w, self.rechan_b = cur.dense(C, cfg["input_size"], bias=False)
        self.layers = [
            Layer(cur, C, cfg["condition_size"], cfg["kernel_size"], d,
                  cfg["activation"], cfg["gated"])
            for d in cfg["dilations"]
        ]
        self.head_w, self.head_b = cur.dense(cfg["head_size"], C, bias=cfg["head_bias"])

    def forward(self, x, condition, head_input):
        x = matmul1x1(x, self.rechan_w, self.rechan_b)
        if head_input is None:
            head_input = np.zeros((self.channels, x.shape[1]), F32)
        for layer in self.layers:
            x, post = layer.forward(x, condition)
            head_input = (head_input + post).astype(F32)
        head_out = matmul1x1(head_input, self.head_w, self.head_b)
        return head_out.astype(F32), x.astype(F32)


def forward(model, signal):
    cur = Cursor(model["weights"])
    arrays = [LayerArray(cur, la) for la in model["config"]["layers"]]
    head_scale = cur.scalar()
    if not cur.done():
        raise ValueError(f"unused weights: {cur.i} consumed of {len(cur.w)}")
    condition = signal.reshape(1, -1).astype(F32)
    y = condition
    head_input = None
    for arr in arrays:
        head_input, y = arr.forward(y, condition, head_input)
    out = (head_scale * head_input).astype(F32)
    return out.reshape(-1)


def make_signal(n=2048):
    """Deterministic: a noise burst plus two sine sweeps, well past the RF."""
    rng = np.random.default_rng(20240530)
    t = np.arange(n, dtype=F32)
    noise = rng.standard_normal(n).astype(F32) * F32(0.25)
    sweep1 = np.sin(2 * np.pi * (0.001 + 0.00002 * t) * t).astype(F32) * F32(0.5)
    sweep2 = np.sin(2 * np.pi * 0.05 * t).astype(F32) * F32(0.3)
    sig = (noise + sweep1 + sweep2).astype(F32)
    return np.clip(sig, F32(-1.0), F32(1.0)).astype(F32)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "reference.nam")
    model = json.load(open(path))
    signal = make_signal()
    out = forward(model, signal)

    with open(os.path.join(HERE, "input.json"), "w") as f:
        json.dump([float(x) for x in signal], f)
    with open(os.path.join(HERE, "expected_output.json"), "w") as f:
        json.dump([float(x) for x in out], f)

    print(f"model: {path}")
    print(f"samples: {len(signal)}")
    print(f"input  range: [{signal.min():.6f}, {signal.max():.6f}]")
    print(f"output range: [{out.min():.6f}, {out.max():.6f}]")
    print("wrote input.json, expected_output.json")


if __name__ == "__main__":
    main()
