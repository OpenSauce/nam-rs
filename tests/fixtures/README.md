# Parity & RT-safety fixtures

These files pin nam-rs to the reference NAM WaveNet forward pass.

| File                   | What it is                                                  |
| ---------------------- | ----------------------------------------------------------- |
| `reference.nam`        | A real exported WaveNet model (NAM Core `example_models/wavenet.nam`, MIT). |
| `input.json`           | JSON array of input samples (mono).                         |
| `expected_output.json` | JSON array: the reference output for `input.json`.          |

`tests/parity.rs` asserts nam-rs reproduces `expected_output.json` from
`input.json` within `1e-5`. `tests/rt_safety.rs` uses `reference.nam` only.

## Regenerating

```bash
python3 tests/fixtures/gen_fixtures.py        # uses the committed reference.nam
```

`gen_fixtures.py` needs only **numpy**. It implements the WaveNet forward pass
directly from the reference weight layout + math (a faithful port of NeuralAudio
`WaveNet.h` and NAM's `nam/models/wavenet/*.py`), then:

1. Generates a deterministic test signal (fixed seed: noise burst + two sweeps),
   2048 samples — well past the model's receptive field — and writes `input.json`.
2. Runs the forward pass and writes `expected_output.json` (computed in `float32`
   to match NAM Core's inference precision).

## Why a numpy reference instead of `pip install neural-amp-modeler`

The canonical generator runs the Python NAM package, which pulls in torch. Torch
only ships wheels for the Python versions it supports and was not available in the
environment these fixtures were generated in, so `gen_fixtures.py` reimplements the
forward pass from the published references instead. Two things keep this honest:

- The weight count it derives from `config` matches the model file exactly — a
  structural check that the `export_weights` layout is right.
- The same algorithm is verified independently against hand-computed values in the
  per-component Rust unit tests (`src/wavenet/{conv,layer,array}.rs`).

If you have torch, regenerate from the canonical implementation and overwrite
`expected_output.json`; the numbers should agree to `1e-5`. Either way the expected
numbers come from *running* a reference, never from hand-authoring.
