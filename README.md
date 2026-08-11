# nam-rs

[![crates.io](https://img.shields.io/crates/v/nam-rs.svg)](https://crates.io/crates/nam-rs)
[![docs.rs](https://docs.rs/nam-rs/badge.svg)](https://docs.rs/nam-rs)
[![CI](https://github.com/OpenSauce/nam-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSauce/nam-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/nam-rs.svg)](LICENSE)

Pure-Rust, real-time-safe inference for [Neural Amp Modeler](https://www.neuralampmodeler.com/) (NAM) `.nam` models.

Loads a `.nam` file and runs its neural-network forward pass with **no heap allocation
on the audio thread** — suitable for a JACK callback, a VST3/CLAP `process()`, or any
real-time audio graph. Output matches the reference Python and C++ implementations
within `1e-5` per sample, once past the model's warmup.

## Install

```bash
cargo add nam-rs
```

MSRV: Rust 1.71.

## Usage

```rust,no_run
use nam_rs::{Model, NamModel};

fn main() -> Result<(), nam_rs::Error> {
    // Off the audio thread: parsing, construction and buffers all allocate here.
    let file = NamModel::from_file("twin_reverb.nam")?;
    let mut model = Model::from_nam(&file)?;
    let mut block = vec![0.0_f32; 512];

    // Settle the model against silence so the first real block is clean.
    // `receptive_field()` is 0 for LSTM, so this is a no-op there.
    let mut warmup = vec![0.0_f32; model.receptive_field()];
    model.process_buffer(&mut warmup);

    // On the audio thread, once per block: in place, no allocation. State
    // carries across calls, so block-wise output matches one whole-buffer call.
    model.process_buffer(&mut block);
    Ok(())
}
```

See [`examples/`](examples/) — `run_model.rs` checks that a file loads and produces
sane output, `streaming.rs` is the full block-wise host loop.

## What you need to know

**Sample rate.** A `.nam` expects audio at the rate it was captured
(`Model::expected_sample_rate()`, 48 kHz if the file omits it). `nam-rs` does not
resample — feed it audio at that rate, or resample in your host first. A mismatch
produces silently wrong output, because dilations and recurrence are defined in
samples, not seconds.

**Warmup.** A WaveNet model's first `Model::receptive_field()` samples are a startup
transient as the dilated stack fills against zero history — hence the settling step in
the example above. LSTM models load an already-settled state and need none.
`Model::reset()` clears a model's state back to that starting point.

**Levels.** `Model::loudness()`, `input_level_dbu()` and `output_level_dbu()` give the
calibration numbers for gain-staging. `nam-rs` runs the forward pass only: the DC
blocker and optional loudness normalization that the reference NAM plugin applies
belong to your audio graph, not to the model.

## Supported architectures

- **WaveNet** (A1 and A2 single models) — dilated-conv forward pass.
- **LSTM** — recurrent forward pass.
- **SlimmableContainer** (A2) — a set of complete standalone submodels with a
  real-time-safe width dial as a CPU/quality trade-off.

The A2 feature set is covered, with a few exceptions — see the
[crate docs](https://docs.rs/nam-rs) for what is and isn't supported, the selection
semantics, and the `metadata` block a model browser needs.

## Performance

From `cargo bench`, standard-size fixture models, one x86-64 desktop core, release +
LTO:

- WaveNet ≈1.9 µs/sample via `process_buffer` (≈11× real-time at 48 kHz) — around
  3.5× faster than the per-sample path, so prefer whole blocks
- LSTM ≈1.2 µs/sample (≈17× real-time). It is recurrent, so the block path is no
  faster; `process_buffer` is a loop over `process_sample`

Numbers vary with CPU and model size — run `cargo bench` on your own target.

## Development

```bash
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Parity fixtures are committed under `tests/fixtures/`; regenerate them from Python NAM
with `tests/fixtures/gen_fixtures.py` (see `tests/fixtures/README.md`).

## Attribution & license

`nam-rs` is MIT-licensed (see [`LICENSE`](LICENSE)). It is a **derivative work**: the
algorithm and `.nam` weight layout are ported from the projects below. Their license
texts are reproduced in [`NOTICE`](NOTICE).

| Project | Role | License |
| --- | --- | --- |
| [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) | Reference trainer + `.nam` exporter (source of truth for weight/config layout) | MIT |
| [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) | Canonical C++ inference library | MIT |
| [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) | High-performance C++ NAM runtime; primary porting reference | MIT |

`.nam` model files are licensed separately by whoever captured them; `nam-rs` ships
no model files.
