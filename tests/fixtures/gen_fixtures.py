#!/usr/bin/env python3
"""Generate parity fixtures (input.json, expected_output.json) for nam-rs.

The committed ``expected_output.json`` is produced by the **canonical** reference
``neural-amp-modeler`` package (torch) -- see ``canonical_forward``. Because torch
only ships wheels for the Python versions it supports (and may be absent), this
script also carries a dependency-light **numpy** reimplementation of the same
forward pass (``numpy_forward``), ported from:

  - NeuralAudio (Mike Oliphant, MIT) -- ``NeuralAudio/WaveNet.h``.
  - neural-amp-modeler (Steven Atkinson, MIT) -- ``nam/models/wavenet/*.py``.

``forward`` prefers the canonical path and falls back to numpy. The two agree to
~3e-7 in the steady state but differ over the receptive-field warmup (torch's
training forward pre-pads + propagates bias; a streaming engine zero-inits each
layer's history). ``tests/parity.rs`` skips that warmup and asserts ~1.5e-7 over
the steady state. float32 throughout, matching NAM Core's inference precision.

Usage:
    # canonical (recommended): a torch-capable interpreter with `nam` installed
    python -m venv venv && venv/bin/pip install neural-amp-modeler
    venv/bin/python tests/fixtures/gen_fixtures.py [path/to/model.nam] [out_prefix] [samples]
    # or torch-free numpy fallback (any python with numpy)
    python3 tests/fixtures/gen_fixtures.py [path/to/model.nam] [out_prefix] [samples]

Defaults to the committed tests/fixtures/reference.nam, the "" (empty) out_prefix
-> input.json / expected_output.json, and 2048 samples. The standard model uses
``reference_standard`` / ``_standard`` / a longer signal (its receptive field is
larger than 2048); see ``tests/fixtures/README.md``.
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


def numpy_forward(model, signal):
    """Torch-free reference forward pass (numpy). Proven equivalent to canonical
    torch NAM to ~3e-7 on the committed model."""
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


def _parse_lstm_initial_states(model):
    """Pull exported (h0, c0) per layer out of the flat blob (for the oracle)."""
    cfg = model["config"]
    H, L, in0 = cfg["hidden_size"], cfg["num_layers"], cfg["input_size"]
    cur = Cursor(model["weights"])
    h0s, c0s = [], []
    for li in range(L):
        in_dim = in0 if li == 0 else H
        cur.take(4 * H * (in_dim + H))  # W
        cur.take(4 * H)                 # bias
        h0s.append(np.array(cur.take(H), F32))
        c0s.append(np.array(cur.take(H), F32))
    return np.stack(h0s), np.stack(c0s)


def numpy_lstm_forward(model, signal):
    """Torch-free LSTM forward, streaming from the exported h0/c0. Matches the
    canonical core seeded the same way to ~1e-7 (verified H=8/L=1, H=16/L=2)."""
    cfg = model["config"]
    H, L, in0 = cfg["hidden_size"], cfg["num_layers"], cfg["input_size"]
    cur = Cursor(model["weights"])
    layers = []
    for li in range(L):
        in_dim = in0 if li == 0 else H
        W = cur.take(4 * H * (in_dim + H)).reshape(4 * H, in_dim + H)
        b = cur.take(4 * H)
        h0 = np.array(cur.take(H), F32)
        c0 = np.array(cur.take(H), F32)
        layers.append([W, b, h0, c0, in_dim])
    head_w = cur.take(H)
    head_b = float(cur.take(1)[0])
    if not cur.done():
        raise ValueError(f"unused weights: {cur.i} of {len(cur.w)}")

    def sg(z):
        return (1.0 / (1.0 + np.exp(-z))).astype(F32)

    state = [[l[2].copy(), l[3].copy()] for l in layers]
    out = np.zeros(len(signal), F32)
    for t, xt in enumerate(signal):
        x = np.array([xt], F32)
        for li, (W, b, _, _, _) in enumerate(layers):
            hp, cp = state[li]
            v = np.concatenate([x, hp]).astype(F32)
            g = (W @ v + b).astype(F32)
            i = sg(g[0:H]); f = sg(g[H:2 * H])
            gg = np.tanh(g[2 * H:3 * H]).astype(F32); o = sg(g[3 * H:4 * H])
            c = (f * cp + i * gg).astype(F32)
            h = (o * np.tanh(c)).astype(F32)
            state[li] = [h, c]
            x = h
        out[t] = float(np.dot(head_w, x) + head_b)
    return out.astype(F32)


def canonical_lstm_forward(model, signal):
    """Canonical LSTM via neural-amp-modeler. `import_weights` sets the net's
    initial hidden/cell from the blob's h0/c0, so `_forward` streams from the
    exported state (matching nam-rs). Raises ImportError if torch/nam absent."""
    import torch  # noqa: F401
    from nam.models.recurrent import LSTM

    cfg = model["config"]
    net = LSTM(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
               input_size=cfg["input_size"])
    net.eval()
    net.import_weights(np.asarray(model["weights"], dtype=F32))
    with torch.no_grad():
        y = net._forward(torch.from_numpy(signal.astype(F32)).reshape(1, -1))
    return y.numpy().reshape(-1).astype(F32)


def canonical_forward(model, signal):
    """Canonical forward via the real `neural-amp-modeler` package (torch).

    Returns full-length output matching nam-rs' zero-warmup convention by
    left-padding the input by the receptive field. Raises ImportError if `nam`
    is not installed, so the caller can fall back to numpy_forward.
    """
    import torch  # noqa: F401
    from nam.models.wavenet import _WaveNet

    cfg = model["config"]
    net = _WaveNet(
        layers_configs=cfg["layers"],
        head_config=cfg.get("head"),
        head_scale=cfg["head_scale"],
    )
    net.eval()
    net.import_weights(torch.tensor(model["weights"], dtype=torch.float32))

    x = signal.astype(F32)
    # Probe the receptive field (NAM does valid convolutions, trimming the output).
    with torch.no_grad():
        valid_len = net(torch.from_numpy(x).reshape(1, 1, -1)).shape[-1]
    rf = len(x) - valid_len + 1
    xp = np.concatenate([np.zeros(rf - 1, F32), x]).reshape(1, 1, -1)
    with torch.no_grad():
        return net(torch.from_numpy(xp)).numpy().reshape(-1).astype(F32)


def forward(model, signal):
    """Canonical NAM if available, else the numpy reference. Dispatches on arch."""
    arch = model.get("architecture", "WaveNet")
    if arch == "LSTM":
        try:
            out = canonical_lstm_forward(model, signal)
            print("forward: canonical neural-amp-modeler LSTM (torch)")
            return out
        except ImportError:
            print("forward: numpy LSTM reference (torch/`nam` unavailable)")
            return numpy_lstm_forward(model, signal)
    # WaveNet (existing behaviour)
    try:
        out = canonical_forward(model, signal)
        print("forward: canonical neural-amp-modeler (torch)")
        return out
    except ImportError:
        print("forward: numpy reference (torch/`nam` unavailable)")
        return numpy_forward(model, signal)


def make_signal(n=2048):
    """Deterministic: a noise burst plus two sine sweeps, well past the RF.

    ``n`` must exceed the model's receptive field so the parity test has steady
    state to compare; standard models (RF ~4093) need a longer signal than the
    minimal one.
    """
    rng = np.random.default_rng(20240530)
    t = np.arange(n, dtype=F32)
    noise = rng.standard_normal(n).astype(F32) * F32(0.25)
    sweep1 = np.sin(2 * np.pi * (0.001 + 0.00002 * t) * t).astype(F32) * F32(0.5)
    sweep2 = np.sin(2 * np.pi * 0.05 * t).astype(F32) * F32(0.3)
    sig = (noise + sweep1 + sweep2).astype(F32)
    return np.clip(sig, F32(-1.0), F32(1.0)).astype(F32)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "reference.nam")
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    model = json.load(open(path))
    signal = make_signal(n)
    out = forward(model, signal)

    in_path = os.path.join(HERE, f"input{prefix}.json")
    out_path = os.path.join(HERE, f"expected_output{prefix}.json")
    with open(in_path, "w") as f:
        json.dump([float(x) for x in signal], f)
    with open(out_path, "w") as f:
        json.dump([float(x) for x in out], f)

    print(f"model: {path}")
    print(f"samples: {len(signal)}")
    print(f"input  range: [{signal.min():.6f}, {signal.max():.6f}]")
    print(f"output range: [{out.min():.6f}, {out.max():.6f}]")
    print(f"wrote {os.path.basename(in_path)}, {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
