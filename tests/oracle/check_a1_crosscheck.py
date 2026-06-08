#!/usr/bin/env python3
"""Validate the NAMCore oracle against the committed torch-generated A1 fixture.

Both engines are zero-init streaming, so we compare steady-state (skip the
receptive field) within 1e-5 — the same convention as tests/parity.rs.
Exits non-zero (with a max-error report) if they disagree.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle_gen import run_oracle  # noqa: E402

FIX = os.path.join(HERE, "..", "fixtures")
TOL = 1e-5
# reference_standard's receptive field (kernel 3; standard dilations). Skipping a
# generous prefix is safe: we only assert steady-state agreement.
SKIP = 4096


def main():
    model = os.path.join(FIX, "reference_standard.nam")
    with open(os.path.join(FIX, "input_standard.json")) as f:
        xs = json.load(f)
    with open(os.path.join(FIX, "expected_output_standard.json")) as f:
        want = json.load(f)
    got = run_oracle(model, xs)
    assert len(got) == len(want) == len(xs), (len(got), len(want), len(xs))
    errs = [abs(g - w) for g, w in zip(got[SKIP:], want[SKIP:])]
    max_err = max(errs) if errs else 0.0
    print(f"steady-state samples compared: {len(errs)}, max_err={max_err:.2e}, tol={TOL:.0e}")
    if max_err > TOL:
        print("ORACLE CROSS-CHECK FAILED", file=sys.stderr)
        sys.exit(1)
    print("ORACLE CROSS-CHECK PASSED")


if __name__ == "__main__":
    main()
