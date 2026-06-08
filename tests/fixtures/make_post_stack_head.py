#!/usr/bin/env python3
"""Generate a WaveNet-with-post-stack-head parity fixture, oracled by NAMCore.

Builds tests/fixtures/post_stack_head.nam (a standalone WaveNet with a single
layer-array followed by an optional post-stack `detail::Head`: an
`activation -> Conv1d` chain run after the arrays, with `head_scale` scaling the
head's input), a deterministic input, and the NAMCore expected output.

Run from the repo root after building the oracle (tests/oracle/build.sh):
    python3 tests/fixtures/make_post_stack_head.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "oracle")
sys.path.insert(0, ORACLE_DIR)
from oracle_gen import run_oracle  # noqa: E402

CHANNELS = 4
HEAD_SIZE = 2
DILATIONS = [1, 2, 4]
KERNELS = [3, 3, 3]
HEAD_KERNEL = 1
# Post-stack head: in_channels = last array head_size (2) -> channels (4) -> out 1.
HEAD_CHANNELS = 4
HEAD_OUT = 1
HEAD_KERNEL_SIZES = [8, 1]
HEAD_SCALE = 0.7
N = 4096


def array_weight_count():
    # rechannel (channels*input, no bias)
    total = CHANNELS * 1
    for k in KERNELS:
        total += CHANNELS * CHANNELS * k + CHANNELS  # conv w + b (mid==channels)
        total += CHANNELS * 1                        # input mixer (no bias)
        total += CHANNELS * CHANNELS + CHANNELS      # layer1x1 w + b (bottleneck==channels)
    # head rechannel: head_size * channels * head_kernel (+ no head_bias)
    total += HEAD_SIZE * CHANNELS * HEAD_KERNEL
    return total


def post_stack_head_weight_count():
    total = 0
    cin = HEAD_SIZE  # in_channels = last array's head_size
    n = len(HEAD_KERNEL_SIZES)
    for i, k in enumerate(HEAD_KERNEL_SIZES):
        cout = HEAD_OUT if i + 1 == n else HEAD_CHANNELS
        total += cout * cin * k + cout  # weights + bias
        cin = cout
    return total


def weight_count():
    return array_weight_count() + post_stack_head_weight_count() + 1  # +head_scale


def main():
    n_w = weight_count()
    weights = [((i % 13) - 6) * 0.02 for i in range(n_w)]
    weights[-1] = HEAD_SCALE  # head_scale is the trailing weight
    config = {
        "layers": [{
            "input_size": 1, "condition_size": 1, "channels": CHANNELS,
            "bottleneck": CHANNELS,
            "dilations": DILATIONS, "kernel_sizes": KERNELS,
            "activation": [{"type": "Tanh"} for _ in KERNELS],
            "head": {"out_channels": HEAD_SIZE, "kernel_size": HEAD_KERNEL,
                     "bias": False},
            "layer1x1": {"active": True, "groups": 1},
            "gating_mode": ["none" for _ in KERNELS],
        }],
        "head": {
            "channels": HEAD_CHANNELS, "out_channels": HEAD_OUT,
            "kernel_sizes": HEAD_KERNEL_SIZES, "activation": "ReLU",
        },
        "head_scale": HEAD_SCALE,
    }
    model = {"version": "0.7.0", "architecture": "WaveNet", "config": config,
             "weights": weights, "sample_rate": 48000}
    nam_path = os.path.join(HERE, "post_stack_head.nam")
    with open(nam_path, "w") as f:
        json.dump(model, f)

    import math
    signal = [math.sin(i * 0.021) * 0.5 + math.sin(i * 0.0034) * 0.2 for i in range(N)]
    with open(os.path.join(HERE, "input_post_stack_head.json"), "w") as f:
        json.dump(signal, f)

    expected = run_oracle(nam_path, signal)
    with open(os.path.join(HERE, "expected_post_stack_head.json"), "w") as f:
        json.dump(expected, f)
    print(f"post_stack_head.nam: {n_w} weights; oracled {len(expected)} samples")


if __name__ == "__main__":
    main()
