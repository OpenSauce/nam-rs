# Parity & RT-safety fixtures

These files pin nam-rs to the reference NAM WaveNet forward pass.

| File                            | What it is                                                  |
| ------------------------------- | ----------------------------------------------------------- |
| `reference.nam`                 | A real exported WaveNet model (NAM Core `example_models/wavenet.nam`, MIT, 131 weights). |
| `input.json`                    | JSON array of input samples (mono), 2048 samples.           |
| `expected_output.json`          | The **canonical** `neural-amp-modeler` (torch) output for `input.json`. |
| `reference_standard.nam`        | The realistic standard model (NAM Core `example_models/wavenet_a1_standard.nam`, MIT, 13,802 weights — 16+8 channels, dilations 1..512). |
| `input_standard.json`           | Input for the standard model, 8192 samples (its receptive field ~4093 exceeds 2048). |
| `expected_output_standard.json` | Canonical torch output for `input_standard.json`.           |

`tests/parity.rs` asserts nam-rs reproduces each `expected_output*.json` from the
matching `input*.json` within `1e-5`, **skipping the first `receptive_field()`
samples** (the warmup transient — see "Warmup convention" below). The two cases
guard both the minimal model and production channel sizes. `tests/rt_safety.rs`
uses `reference.nam` only.

## Regenerating

```bash
# canonical (recommended): needs a torch-capable Python with `nam` installed.
# The default python3 may lack torch wheels; python3.10 here does support torch.
python3.10 -m venv venv && venv/bin/pip install neural-amp-modeler
venv/bin/python tests/fixtures/gen_fixtures.py                                                   # minimal
venv/bin/python tests/fixtures/gen_fixtures.py tests/fixtures/reference_standard.nam _standard 8192

# torch-free fallback (any python3 with numpy) — same arguments
python3 tests/fixtures/gen_fixtures.py
```

`gen_fixtures.py [model] [out_prefix] [samples]` generates a deterministic test
signal (fixed seed: noise burst + two sweeps, past the model's receptive field),
runs the forward pass, and writes `input{prefix}.json` / `expected_output{prefix}.json`
(float32, matching NAM Core's inference precision). It defaults to `reference.nam`,
an empty prefix, and 2048 samples.

`forward()` prefers the **canonical** path — the real `neural-amp-modeler` package
(`nam.models.wavenet._WaveNet`) — and falls back to a dependency-light **numpy**
reimplementation when torch/`nam` isn't importable. The committed
`expected_output.json` was produced by the canonical torch path. In the **steady
state** the two agree to ~`3e-7`; they diverge only over the warmup (see below), so
either generator yields a fixture the trimmed parity test accepts.

## Warmup convention

The two reference implementations disagree over the first `receptive_field` samples,
by construction:

- torch's `_WaveNet.forward` (a training graph) pre-pads the whole input with zeros
  and propagates each layer's bias/activation through the stack.
- A streaming engine — this crate, and NAM Core / NeuralAudio — starts every layer
  from a zero-filled history buffer instead.

These agree once the receptive field fills, but differ over the startup transient
(on the committed model, ~`0.023` max over the first ~22 samples, ~0.5 ms at 48 kHz).
`tests/parity.rs` therefore compares only the steady state (`signal[rf..]`), where
nam-rs matches canonical torch NAM to ~`1.5e-7`. nam-rs deliberately follows the
streaming convention because a real-time `process_buffer` cannot pre-pad an unbounded
stream; the numpy fallback follows it too.

## Layering of validation

- End-to-end: nam-rs ↔ canonical torch NAM, ≤`1e-5` (`tests/parity.rs`).
- Independent unit oracle: nam-rs `Conv1d` ↔ NAM Core's hand-derived
  `test_conv1d.cpp` values (`src/wavenet/conv.rs`).
- Structural: the weight count `WaveNet::new` derives from `config` matches every
  real model checked (incl. 13,802-weight standard models).
