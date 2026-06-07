#!/usr/bin/env python3
"""Emit a small LeakyReLU WaveNet .nam (no shipped A2 example uses LeakyReLU, so we
self-generate one to parity-test the new activation). Deterministic (fixed seed).

Run from the repo root with the torch-capable venv:
    venv/bin/python tests/fixtures/make_leaky_wavenet.py
"""
import json
import os

import numpy as np
import torch
from nam.models.wavenet import _WaveNet

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(20240607)
    layers = [{
        "input_size": 1, "condition_size": 1, "channels": 3, "head_size": 1,
        "kernel_size": 2, "dilations": [1, 5, 29], "activation": "LeakyReLU",
        "gated": False, "head_bias": False,
    }]
    net = _WaveNet(layers_configs=layers, head_config=None, head_scale=0.02)
    net.eval()
    cfg = net.export_config()
    weights = np.asarray(net.export_weights(), dtype=np.float32).reshape(-1)
    model = {
        "version": "0.7.0", "architecture": "WaveNet", "config": cfg,
        "weights": [float(x) for x in weights], "sample_rate": 48000.0,
    }
    with open(os.path.join(HERE, "leaky_wavenet.nam"), "w") as f:
        json.dump(model, f)
    assert cfg["layers"][0]["activation"] == "LeakyReLU", cfg["layers"][0]["activation"]
    print(f"wrote leaky_wavenet.nam: activation={cfg['layers'][0]['activation']} "
          f"weights={len(weights)}")


if __name__ == "__main__":
    main()
