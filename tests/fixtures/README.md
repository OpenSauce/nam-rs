# Parity & RT-safety fixtures

These files pin nam-rs to the reference NAM WaveNet forward pass.

| File                   | What it is                                                  |
| ---------------------- | ----------------------------------------------------------- |
| `reference.nam`        | A real exported WaveNet model (NAM Core `example_models/wavenet.nam`, MIT). |
| `input.json`           | JSON array of input samples (mono).                         |
| `expected_output.json` | The **canonical** `neural-amp-modeler` (torch) output for `input.json`. |

`tests/parity.rs` asserts nam-rs reproduces `expected_output.json` from
`input.json` within `1e-5`. `tests/rt_safety.rs` uses `reference.nam` only.

## Regenerating

```bash
# canonical (recommended): needs a torch-capable Python with `nam` installed
python -m venv venv && venv/bin/pip install neural-amp-modeler
venv/bin/python tests/fixtures/gen_fixtures.py

# torch-free fallback (any python3 with numpy)
python3 tests/fixtures/gen_fixtures.py
```

`gen_fixtures.py` generates a deterministic test signal (fixed seed: noise burst +
two sweeps, 2048 samples, past the model's receptive field), runs the forward pass,
and writes `input.json` / `expected_output.json` (float32, matching NAM Core's
inference precision).

`forward()` prefers the **canonical** path — the real `neural-amp-modeler` package
(`nam.models.wavenet._WaveNet`) — and falls back to a dependency-light **numpy**
reimplementation when torch/`nam` isn't importable. The two were verified
equivalent to ~`3e-7` on the committed model, so the fixture is identical either
way. The committed `expected_output.json` was produced by the canonical torch path.

## Layering of validation

- End-to-end: nam-rs ↔ canonical torch NAM, ≤`1e-5` (`tests/parity.rs`).
- Independent unit oracle: nam-rs `Conv1d` ↔ NAM Core's hand-derived
  `test_conv1d.cpp` values (`src/wavenet/conv.rs`).
- Structural: the weight count `WaveNet::new` derives from `config` matches every
  real model checked (incl. 13,802-weight standard models).
