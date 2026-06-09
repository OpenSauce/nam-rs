# nam-rs

[![crates.io](https://img.shields.io/crates/v/nam-rs.svg)](https://crates.io/crates/nam-rs)
[![docs.rs](https://docs.rs/nam-rs/badge.svg)](https://docs.rs/nam-rs)

Pure-Rust, real-time-safe inference for [Neural Amp Modeler](https://www.neuralampmodeler.com/) (NAM) `.nam` models.

`nam-rs` loads a `.nam` model file and runs its neural-network forward pass — a whole
buffer at a time (WaveNet uses a cache-friendly block kernel) or one sample at a time —
with **no heap allocation on the audio thread**, suitable for use inside a JACK
callback, a VST3/CLAP `process()`, or any real-time audio graph.

> Status: **WaveNet, LSTM, and SlimmableContainer (NAM "A2") inference are implemented
> and tested** — parser, forward pass, parity, and RT-safety harnesses all green. Build
> any `.nam` with the architecture-agnostic `Model::from_nam`, which dispatches on the
> model's architecture.

## Design contract

1. **Parity with the reference.** Output must equal the canonical Python/C++ NAM
   implementations within float tolerance for the same model and input. Enforced by
   `tests/parity.rs` against fixtures generated from Python NAM.
2. **Real-time safety.** The runtime's `process_buffer` (for both WaveNet and LSTM,
   reached via `Model`) performs zero heap allocation, locks, or syscalls; all scratch
   buffers are pre-allocated at construction. Enforced by `tests/rt_safety.rs` via
   `assert_no_alloc`.

## Install

```bash
cargo add nam-rs
```

## Usage

```rust
use nam_rs::{Model, NamModel};

// Off the audio thread: load + allocate. `Model::from_nam` dispatches on the
// model's architecture, so the same code runs WaveNet and LSTM `.nam` files.
let model = NamModel::from_file("twin_reverb.nam")?;
let mut amp = Model::from_nam(&model)?;

// On the audio thread: in-place, allocation-free. Call once per audio block;
// state carries across calls, so block-wise output matches one whole-buffer call.
let mut audio_buffer = vec![0.0_f32; 512]; // your host's block, filled with input
amp.process_buffer(&mut audio_buffer);
```

For WaveNet models, the first `WaveNet::receptive_field()` output samples are a startup
transient (the dilated stack filling against zero-history) — the model's inherent
latency, the same convention NAM Core / NeuralAudio use. LSTM models have no such
warmup. Call `Model::reset` to return to silence.

**Sample rate.** A `.nam` expects audio at the rate it was captured
(`NamModel::expected_sample_rate()`, 48 kHz if the file omits it). `nam-rs` does not
resample: feed the model audio at that rate, or resample in your host first. A
mismatched rate produces silently wrong output, since the model's dilations and
recurrence are defined in samples, not seconds.

**Processing boundary.** `nam-rs` runs only the model's forward pass. The reference
NAM plugin additionally applies a DC blocker (high-pass) and, optionally, loudness
normalization on the output — those belong to the host's audio graph, not the model.
The calibration accessors (`NamModel::loudness()` etc.) give you the numbers for that
gain-staging.

## Supported architectures

- **WaveNet** (A1 and A2 single models) — dilated-conv forward pass, parity-tested.
- **LSTM** — recurrent forward pass, parity-tested.
- **SlimmableContainer** (NAM "A2") — a width-selectable set of complete standalone
  submodels (any mix of WaveNet/LSTM). The container holds no weights; it delegates to
  the active submodel. Select the width with `model.as_slimmable_mut()` →
  `set_slim_size(value)` (NAM Core semantics: the first submodel whose `max_value`
  exceeds `value`, else the full model) or `select(index)`. Switching is real-time-safe.

The A2 feature set is supported: FiLM, gating, bottleneck, grouped convs, multi-tap conv
heads, the optional post-stack head (an `activation → Conv1d` chain after the arrays, with
`head_scale` scaling its input), and a `condition_dsp` (a nested model whose output
replaces the conditioning fed to every array, including a multi-channel-output one whose N
rows become the N-wide conditioning). The remaining restrictions — multi-channel
input, a post-stack head with `out_channels != 1`, mixed gating modes within
one array, and exotic activations — are rejected with a clear `Error::UnsupportedFeature`
(or `Error::UnsupportedActivation`) rather than silently mis-run.

## Development

```bash
cargo test                                  # parser, parity, and RT-safety tests
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
| [waveny](https://github.com/nlpodyssey/waveny) | Go port; conceptual cross-check only | Apache-2.0 |

`.nam` model files are licensed separately by whoever captured them; `nam-rs` ships
no model files.
