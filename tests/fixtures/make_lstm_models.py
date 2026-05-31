#!/usr/bin/env python3
"""Emit the committed LSTM .nam fixtures from canonical neural-amp-modeler.

Deterministic (fixed seed). Run from the repo root with the torch-capable venv:
    venv/bin/python tests/fixtures/make_lstm_models.py
"""
import json
import os

import torch
from nam.models.recurrent import LSTM

HERE = os.path.dirname(os.path.abspath(__file__))


def emit(path, hidden_size, num_layers, seed):
    torch.manual_seed(seed)
    net = LSTM(hidden_size=hidden_size, num_layers=num_layers, input_size=1)
    net.eval()
    d = net._get_export_dict()
    # Normalise: ensure plain JSON floats and a concrete sample_rate.
    d["weights"] = [float(w) for w in d["weights"]]
    if d.get("sample_rate") is None:
        d["sample_rate"] = 48000.0
    with open(path, "w") as f:
        json.dump(d, f)
    print(f"wrote {os.path.basename(path)}: H={hidden_size} L={num_layers} "
          f"weights={len(d['weights'])}")


if __name__ == "__main__":
    emit(os.path.join(HERE, "reference_lstm.nam"), 8, 1, 20240530)
    emit(os.path.join(HERE, "reference_lstm_standard.nam"), 16, 2, 20240531)
