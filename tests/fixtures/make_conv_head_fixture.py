#!/usr/bin/env python3
"""Generate a small WaveNet-with-conv-head parity fixture, oracled by NAMCore.

Builds tests/fixtures/conv_head.nam (a standalone WaveNet whose layer-array head
is a multi-tap Conv1D, kernel 4), a deterministic input, and the NAMCore expected
output. Run from the repo root after building the oracle (tests/oracle/build.sh):
    python3 tests/fixtures/make_conv_head_fixture.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "oracle")
sys.path.insert(0, ORACLE_DIR)
from oracle_gen import run_oracle  # noqa: E402

CHANNELS = 2
DILATIONS = [1, 2]
KERNELS = [3, 3]
HEAD_KERNEL = 4
HEAD_SIZE = 1
N = 2048


def weight_count():
    # rechannel (channels*input, no bias) + per-layer(conv w+b, mixin, 1x1 w+b)
    # + head rechannel (head_size*channels*head_kernel + head_size bias) + head_scale.
    total = CHANNELS * 1
    for k in KERNELS:
        total += CHANNELS * CHANNELS * k + CHANNELS  # conv w + b (mid==channels)
        total += CHANNELS * 1                        # input mixer (no bias)
        total += CHANNELS * CHANNELS + CHANNELS      # 1x1 w + b
    total += HEAD_SIZE * CHANNELS * HEAD_KERNEL + HEAD_SIZE  # head rechannel w + b
    total += 1                                       # head_scale
    return total


def main():
    n_w = weight_count()
    # Small deterministic weights to avoid saturating / keep values sane.
    weights = [math.sin(i * 0.137) * 0.15 for i in range(n_w)]
    weights[-1] = 0.5  # head_scale
    config = {
        "layers": [{
            "input_size": 1, "condition_size": 1, "channels": CHANNELS, "bottleneck": CHANNELS,
            "dilations": DILATIONS, "kernel_sizes": KERNELS,
            "activation": [{"type": "LeakyReLU", "negative_slope": 0.01} for _ in KERNELS],
            "head": {"out_channels": HEAD_SIZE, "kernel_size": HEAD_KERNEL, "bias": True},
            "layer1x1": {"active": True, "groups": 1},
            "gating_mode": ["none" for _ in KERNELS],
        }],
        "head": None, "head_scale": 0.5,
    }
    model = {"version": "0.7.0", "architecture": "WaveNet", "config": config,
             "weights": weights, "sample_rate": 48000}
    nam_path = os.path.join(HERE, "conv_head.nam")
    with open(nam_path, "w") as f:
        json.dump(model, f)

    signal = [math.sin(i * 0.021) * 0.5 + math.sin(i * 0.0037) * 0.2 for i in range(N)]
    with open(os.path.join(HERE, "input_conv_head.json"), "w") as f:
        json.dump(signal, f)

    expected = run_oracle(nam_path, signal)
    with open(os.path.join(HERE, "expected_conv_head.json"), "w") as f:
        json.dump(expected, f)
    print(f"conv_head.nam: {n_w} weights; oracled {len(expected)} samples")


if __name__ == "__main__":
    main()
