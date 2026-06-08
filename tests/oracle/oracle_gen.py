#!/usr/bin/env python3
"""Run the NAMCore oracle: (model.nam, input.json) -> output list of floats.

Builds nothing; expects tests/oracle/build/oracle to exist (run build.sh first).
Reusable by later phases' fixture generators.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "build", "oracle")


def run_oracle(model_path: str, input_values) -> list:
    if not os.path.exists(ORACLE):
        raise FileNotFoundError(f"oracle binary not found at {ORACLE}; run tests/oracle/build.sh")
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.json")
        out_path = os.path.join(td, "out.json")
        with open(in_path, "w") as f:
            json.dump([float(x) for x in input_values], f)
        subprocess.run([ORACLE, model_path, in_path, out_path], check=True)
        with open(out_path) as f:
            return json.load(f)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: oracle_gen.py <model.nam> <input.json> <output.json>", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[2]) as f:
        xs = json.load(f)
    ys = run_oracle(sys.argv[1], xs)
    with open(sys.argv[3], "w") as f:
        json.dump(ys, f)
    print(f"wrote {len(ys)} samples to {sys.argv[3]}")
