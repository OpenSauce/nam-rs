# nam-rs

Pure-Rust, real-time-safe inference for [Neural Amp Modeler](https://www.neuralampmodeler.com/) (NAM) `.nam` models.

`nam-rs` loads a `.nam` model file and runs its neural-network forward pass
sample-by-sample with **no heap allocation on the audio thread** — suitable for use
inside a JACK callback, a VST3/CLAP `process()`, or any real-time audio graph.

> Status: **WaveNet inference is implemented and tested** — parser, forward pass,
> parity, and RT-safety harnesses all green. LSTM support is future work.

## Design contract

1. **Parity with the reference.** Output must equal the canonical Python/C++ NAM
   implementations within float tolerance for the same model and input. Enforced by
   `tests/parity.rs` against fixtures generated from Python NAM.
2. **Real-time safety.** `WaveNet::process_buffer` performs zero heap allocation,
   locks, or syscalls; all scratch buffers are pre-allocated in `WaveNet::new`.
   Enforced by `tests/rt_safety.rs` via `assert_no_alloc`.

## Install

```bash
cargo add nam-rs
```

## Usage

```rust
use nam_rs::{NamModel, WaveNet};

// Off the audio thread: load + allocate.
let model = NamModel::from_file("twin_reverb.nam")?;
let mut amp = WaveNet::new(&model)?;

// On the audio thread: in-place, allocation-free. Call once per audio block;
// state carries across calls, so block-wise output matches one whole-buffer call.
amp.process_buffer(&mut audio_buffer);
```

The first `amp.receptive_field()` output samples are a startup transient (the dilated
stack filling against zero-history) — the model's inherent latency, the same
convention NAM Core / NeuralAudio use. Call `WaveNet::reset` to return to silence.

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
