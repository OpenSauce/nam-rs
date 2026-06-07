#!/usr/bin/env python3
"""Generate parity fixtures for slimmable_container.nam by extracting each submodel
and oracling it through the canonical reference (see gen_fixtures.forward).

Each container submodel is a complete standalone model, so the container at a given
slim size must reproduce that submodel's reference output. We write one shared input
plus one expected-output per submodel index.

Run from the repo root with the torch-capable venv:
    venv/bin/python tests/fixtures/make_slim_fixtures.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_fixtures as gf  # noqa: E402

# Long enough to exceed the largest submodel's receptive field (~4093, like standard).
N = 8192


def main():
    with open(os.path.join(HERE, "slimmable_container.nam")) as f:
        container = json.load(f)
    submodels = container["config"]["submodels"]
    signal = gf.make_signal(N)
    with open(os.path.join(HERE, "input_slim.json"), "w") as f:
        json.dump([float(x) for x in signal], f)
    for i, sm in enumerate(submodels):
        out = gf.forward(sm["model"], signal)
        with open(os.path.join(HERE, f"expected_slim_{i}.json"), "w") as f:
            json.dump([float(x) for x in out], f)
        print(f"submodel[{i}] arch={sm['model']['architecture']} "
              f"max_value={sm['max_value']} -> expected_slim_{i}.json ({len(out)} samples)")


if __name__ == "__main__":
    main()
