#!/usr/bin/env python3
"""Generate a multi-channel-output condition_dsp parity fixture, oracled by NAMCore.

Uses the turnkey example model shipped with NeuralAmpModelerCore,
`example_models/wavenet_condition_dsp.nam`: an outer WaveNet whose arrays use
`condition_size == 3`, fed by a nested `condition_dsp` WaveNet whose last layer-array
has `head_size == 3` (so it emits 3 output channels). Those 3 rows become the N-wide
conditioning fed to every array; the raw input still drives the first array's layer
input (NAMCore semantics).

Copies the example model to `tests/fixtures/wavenet_condition_dsp.nam` (verifying it
matches if already present), generates a sine input (N=8192), and oracles it into
`expected_condition_dsp_multi.json` + `input_condition_dsp_multi.json`.

Run from the repo root after building the oracle (tests/oracle/build.sh):
    python3 tests/fixtures/make_condition_dsp_multi.py
"""
import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "oracle")
sys.path.insert(0, ORACLE_DIR)
from oracle_gen import run_oracle  # noqa: E402

N = 8192
EXAMPLE = os.path.join(
    ORACLE_DIR, "NeuralAmpModelerCore", "example_models", "wavenet_condition_dsp.nam"
)


def main():
    nam_path = os.path.join(HERE, "wavenet_condition_dsp.nam")
    src = json.load(open(EXAMPLE))
    # Sanity: outer arrays condition_size == nested condition_dsp output channels.
    cfg = src["config"]
    cond_sizes = [la["condition_size"] for la in cfg["layers"]]
    cdsp_out = cfg["condition_dsp"]["config"]["layers"][-1]["head_size"]
    assert all(cs == cdsp_out for cs in cond_sizes), (
        f"example model condition_size {cond_sizes} != condition_dsp out {cdsp_out}"
    )
    if os.path.exists(nam_path):
        assert json.load(open(nam_path)) == src, (
            "existing wavenet_condition_dsp.nam differs from the example model"
        )
    else:
        shutil.copyfile(EXAMPLE, nam_path)

    signal = [math.sin(i * 0.019) * 0.5 + math.sin(i * 0.0031) * 0.2 for i in range(N)]
    with open(os.path.join(HERE, "input_condition_dsp_multi.json"), "w") as f:
        json.dump(signal, f)

    expected = run_oracle(nam_path, signal)
    with open(os.path.join(HERE, "expected_condition_dsp_multi.json"), "w") as f:
        json.dump(expected, f)
    print(f"wavenet_condition_dsp.nam: condition_size {cond_sizes}, "
          f"condition_dsp out {cdsp_out}; oracled {len(expected)} samples")


if __name__ == "__main__":
    main()
