#!/usr/bin/env python3
"""Generate a WaveNet-with-(mono)-condition_dsp parity fixture, oracled by NAMCore.

Builds tests/fixtures/condition_dsp_mono.nam: a standalone WaveNet whose
`condition_dsp` is a nested mono-in/mono-out WaveNet. The condition_dsp output
replaces the raw mono input as the conditioning fed to every array; the raw input
still drives the first array's layer input (NAMCore semantics).

Run from the repo root after building the oracle (tests/oracle/build.sh):
    python3 tests/fixtures/make_condition_dsp.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "oracle")
sys.path.insert(0, ORACLE_DIR)
from oracle_gen import run_oracle  # noqa: E402

N = 8192


def array_weight_count(channels, kernels, head_size, input_size, condition_size,
                       layer1x1_active):
    # rechannel (channels*input, no bias)
    total = channels * input_size
    for k in kernels:
        total += channels * channels * k + channels   # conv w + b (mid==channels)
        total += channels * condition_size            # input mixer (no bias)
        if layer1x1_active:
            total += channels * channels + channels   # layer1x1 w + b
    # head rechannel: head_size * channels * head_kernel(=1) (+ no head_bias)
    total += head_size * channels * 1
    return total


def make_mono_wavenet_config(channels, kernels, dilations, head_size,
                             input_size, condition_size, head_scale, seed):
    layers = [{
        "input_size": input_size, "condition_size": condition_size,
        "channels": channels, "bottleneck": channels,
        "dilations": dilations, "kernel_sizes": kernels,
        "activation": [{"type": "Tanh"} for _ in kernels],
        "head": {"out_channels": head_size, "kernel_size": 1, "bias": False},
        "layer1x1": {"active": True, "groups": 1},
        "gating_mode": ["none" for _ in kernels],
    }]
    n_w = array_weight_count(channels, kernels, head_size, input_size,
                             condition_size, True) + 1  # + head_scale
    weights = [math.sin((i + seed) * 0.13) * 0.1 for i in range(n_w)]
    weights[-1] = head_scale
    return {"layers": layers, "head": None, "head_scale": head_scale}, weights


def main():
    # condition_dsp: mono in/out nested WaveNet.
    cdsp_config, cdsp_weights = make_mono_wavenet_config(
        channels=2, kernels=[3, 3], dilations=[1, 2], head_size=1,
        input_size=1, condition_size=1, head_scale=0.5, seed=7)
    condition_dsp = {
        "version": "0.7.0", "architecture": "WaveNet",
        "config": cdsp_config, "weights": cdsp_weights, "sample_rate": 48000,
    }

    # Parent: condition_size 1 (mono condition from condition_dsp).
    parent_config, parent_weights = make_mono_wavenet_config(
        channels=3, kernels=[3, 3], dilations=[1, 4], head_size=1,
        input_size=1, condition_size=1, head_scale=0.8, seed=21)
    parent_config["condition_dsp"] = condition_dsp

    model = {
        "version": "0.7.0", "architecture": "WaveNet",
        "config": parent_config, "weights": parent_weights, "sample_rate": 48000,
    }
    nam_path = os.path.join(HERE, "condition_dsp_mono.nam")
    with open(nam_path, "w") as f:
        json.dump(model, f)

    signal = [math.sin(i * 0.019) * 0.5 + math.sin(i * 0.0031) * 0.2 for i in range(N)]
    with open(os.path.join(HERE, "input_condition_dsp.json"), "w") as f:
        json.dump(signal, f)

    expected = run_oracle(nam_path, signal)
    with open(os.path.join(HERE, "expected_condition_dsp.json"), "w") as f:
        json.dump(expected, f)
    print(f"condition_dsp_mono.nam: parent {len(parent_weights)} + nested "
          f"{len(cdsp_weights)} weights; oracled {len(expected)} samples")


if __name__ == "__main__":
    main()
