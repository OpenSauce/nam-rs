# Parity & RT-safety fixtures

These files are the ground truth that pins nam-rs to the reference NAM
implementation. They are **generated from the canonical Python NAM**, not
hand-written, so that `tests/parity.rs` proves bit-level agreement.

Expected files (not yet committed):

| File                   | What it is                                              |
| ---------------------- | ------------------------------------------------------- |
| `reference.nam`        | A real exported WaveNet model file.                     |
| `input.json`           | JSON array of input samples (mono, model sample rate).  |
| `expected_output.json` | JSON array: Python NAM's output for `input.json`.       |

## Regenerating

```bash
pip install neural-amp-modeler numpy
python tests/fixtures/gen_fixtures.py path/to/model.nam
```

`gen_fixtures.py` should:

1. Load `model.nam` with `nam` and copy it to `reference.nam`.
2. Generate a deterministic test signal (e.g. a fixed-seed noise burst + a few
   sweeps), long enough to cover the model's full receptive field, and write it to
   `input.json`.
3. Run the Python model on that signal and write the result to
   `expected_output.json`.

Keep the signal short (a few thousand samples) so the fixtures stay small but long
enough to exceed the longest dilation's receptive field.

## Why generated, not authored

The whole point of the parity test is that we did **not** invent the expected
numbers — they come from the implementation we are claiming equivalence to. See the
crate-level attribution in `src/lib.rs` and `NOTICE`.
