#!/usr/bin/env python3
"""Generate synthetic A2 WaveNet parity fixtures, oracled by NAMCore v0.5.3.

One small standalone WaveNet per exotic A2 path (grouped conv, FiLM, GATED,
BLENDED, head1x1, bottleneck<channels). For each case we author a `.nam` with
deterministic small weights whose count matches the Rust `array_weight_count`
formula exactly, generate a deterministic input, run it through the NAMCore oracle
binary, and commit `<name>.nam` + `input_<name>.json` + `expected_<name>.json`.

Run from the repo root after building the oracle (tests/oracle/build.sh):
    python3 tests/fixtures/make_a2_synthetic_fixtures.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "oracle")
sys.path.insert(0, ORACLE_DIR)
from oracle_gen import run_oracle  # noqa: E402

N = 4096


def film_default():
    return {"active": False, "shift": False, "groups": 1}


def array_weight_count(la):
    """Mirror of the Rust `array_weight_count`: NAMCore per-layer weight order."""
    channels = la["channels"]
    bottleneck = la.get("bottleneck", channels)
    cond = la["condition_size"]
    input_size = la["input_size"]
    groups_input = la.get("groups_input", 1)
    groups_input_mixin = la.get("groups_input_mixin", 1)
    kernel_sizes = la["kernel_sizes"]

    modes = la["gating_mode"]
    gated = modes[0] != "none"
    mid = 2 * bottleneck if gated else bottleneck

    layer1x1 = la.get("layer1x1", {"active": True, "groups": 1})
    head1x1 = la.get("head1x1", {"active": False})
    head1x1_active = head1x1.get("active", False)
    head1x1_out = head1x1.get("out_channels", channels)

    head = la["head"]
    head_size = head["out_channels"]
    head_kernel = head["kernel_size"]
    head_bias = head["bias"]

    film_sites = [
        ("conv_pre_film", channels),
        ("conv_post_film", mid),
        ("input_mixin_pre_film", cond),
        ("input_mixin_post_film", mid),
        ("activation_pre_film", mid),
        ("activation_post_film", bottleneck),
        ("layer1x1_post_film", channels),
        ("head1x1_post_film", head1x1_out),
    ]

    def film_count(name, input_dim):
        f = la.get(name, {"active": False})
        if not f.get("active", False):
            return 0
        out_rows = 2 * input_dim if f.get("shift", False) else input_dim
        groups = f.get("groups", 1)
        return out_rows * cond // groups + out_rows

    total = channels * input_size  # rechannel, no bias
    for k in kernel_sizes:
        layer = mid * channels * k // groups_input + mid  # conv w + b
        layer += mid * cond // groups_input_mixin  # mixin, no bias
        if layer1x1.get("active", True):
            layer += channels * bottleneck // layer1x1.get("groups", 1) + channels
        if head1x1_active:
            layer += head1x1_out * bottleneck // head1x1.get("groups", 1) + head1x1_out
        for name, input_dim in film_sites:
            layer += film_count(name, input_dim)
        total += layer

    head_in = head1x1_out if head1x1_active else bottleneck
    total += head_size * head_in * head_kernel
    if head_bias:
        total += head_size
    return total


def make_case(name, layer):
    """Build the model JSON, fill weights, oracle it, and write all three files."""
    n_layer = array_weight_count(layer)
    n_w = n_layer + 1  # + head_scale
    weights = [math.sin(i * 0.137) * 0.15 for i in range(n_w)]
    weights[-1] = 0.5  # head_scale
    config = {"layers": [layer], "head": None, "head_scale": 0.5}
    model = {
        "version": "0.7.0",
        "architecture": "WaveNet",
        "config": config,
        "weights": weights,
        "sample_rate": 48000,
    }
    nam_path = os.path.join(HERE, f"a2_{name}.nam")
    with open(nam_path, "w") as f:
        json.dump(model, f)

    signal = [math.sin(i * 0.021) * 0.5 + math.sin(i * 0.0037) * 0.2 for i in range(N)]
    with open(os.path.join(HERE, f"input_a2_{name}.json"), "w") as f:
        json.dump(signal, f)

    expected = run_oracle(nam_path, signal)
    assert len(expected) == N, f"{name}: oracle returned {len(expected)} != {N}"
    with open(os.path.join(HERE, f"expected_a2_{name}.json"), "w") as f:
        json.dump(expected, f)
    print(f"a2_{name}.nam: {n_w} weights; oracled {len(expected)} samples")


def main():
    head = {"out_channels": 1, "kernel_size": 1, "bias": False}

    cases = {}

    # grouped: groups_input=2 + grouped layer1x1 (groups=2), channels=4, bottleneck=4,
    # NONE. (groups_input_mixin stays 1: the mixin in-channels = condition_size = 1.)
    cases["grouped"] = {
        "input_size": 1, "condition_size": 1, "channels": 4, "bottleneck": 4,
        "dilations": [1, 2], "kernel_sizes": [3, 3],
        "activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "gating_mode": ["none", "none"],
        "groups_input": 2, "groups_input_mixin": 1,
        "layer1x1": {"active": True, "groups": 2},
        "head": head,
    }

    # film: conv_post_film (shift) + activation_pre_film (scale), channels=2, NONE.
    cases["film"] = {
        "input_size": 1, "condition_size": 1, "channels": 2, "bottleneck": 2,
        "dilations": [1, 2], "kernel_sizes": [3, 3],
        "activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "gating_mode": ["none", "none"],
        "layer1x1": {"active": True, "groups": 1},
        "conv_post_film": {"active": True, "shift": True, "groups": 1},
        "activation_pre_film": {"active": True, "shift": False, "groups": 1},
        "head": head,
    }

    # gated: GATED, secondary Sigmoid (A1-shape exercised end-to-end).
    cases["gated"] = {
        "input_size": 1, "condition_size": 1, "channels": 4, "bottleneck": 4,
        "dilations": [1, 2], "kernel_sizes": [3, 3],
        "activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "gating_mode": ["gated", "gated"],
        "secondary_activation": [{"type": "Sigmoid"}, {"type": "Sigmoid"}],
        "layer1x1": {"active": True, "groups": 1},
        "head": head,
    }

    # blended: BLENDED, secondary Tanh, with layer1x1_post_film active.
    cases["blended"] = {
        "input_size": 1, "condition_size": 1, "channels": 4, "bottleneck": 4,
        "dilations": [1, 2], "kernel_sizes": [3, 3],
        "activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "gating_mode": ["blended", "blended"],
        "secondary_activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "layer1x1": {"active": True, "groups": 1},
        "layer1x1_post_film": {"active": True, "shift": False, "groups": 1},
        "head": head,
    }

    # head1x1: head1x1 active out_channels=3, channels=2, bottleneck=2.
    cases["head1x1"] = {
        "input_size": 1, "condition_size": 1, "channels": 2, "bottleneck": 2,
        "dilations": [1, 2], "kernel_sizes": [3, 3],
        "activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "gating_mode": ["none", "none"],
        "layer1x1": {"active": True, "groups": 1},
        "head1x1": {"active": True, "out_channels": 3, "groups": 1},
        "head": head,
    }

    # bottleneck: bottleneck=2 < channels=4, layer1x1 maps 2->4.
    cases["bottleneck"] = {
        "input_size": 1, "condition_size": 1, "channels": 4, "bottleneck": 2,
        "dilations": [1, 2], "kernel_sizes": [3, 3],
        "activation": [{"type": "Tanh"}, {"type": "Tanh"}],
        "gating_mode": ["none", "none"],
        "layer1x1": {"active": True, "groups": 1},
        "head": head,
    }

    for name, layer in cases.items():
        make_case(name, layer)


if __name__ == "__main__":
    main()
