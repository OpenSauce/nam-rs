# nam-rs

Pure-Rust, real-time-safe inference for [Neural Amp Modeler](https://www.neuralampmodeler.com/) (NAM) `.nam` models.

`nam-rs` loads a `.nam` model file and runs its neural-network forward pass
sample-by-sample with **no heap allocation on the audio thread** — suitable for use
inside a JACK callback, a VST3/CLAP `process()`, or any real-time audio graph.

> Status: **early scaffolding.** The `.nam` parser is implemented and tested. The
> WaveNet forward pass is being built test-first against the parity and RT-safety
> harnesses (see below). LSTM support is future work.

## Design contract

1. **Parity with the reference.** Output must equal the canonical Python/C++ NAM
   implementations within float tolerance for the same model and input. Enforced by
   `tests/parity.rs` against fixtures generated from Python NAM.
2. **Real-time safety.** `WaveNet::process_buffer` performs zero heap allocation,
   locks, or syscalls; all scratch buffers are pre-allocated in `WaveNet::new`.
   Enforced by `tests/rt_safety.rs` via `assert_no_alloc`.

## Usage

```rust
use nam_rs::{NamModel, WaveNet};

// Off the audio thread: parse + allocate.
let json = std::fs::read_to_string("twin_reverb.nam")?;
let model = NamModel::from_json_str(&json)?;
let mut amp = WaveNet::new(&model)?;

// On the audio thread: in-place, allocation-free.
amp.process_buffer(&mut audio_buffer);
```

## Development

```bash
cargo test                # parser tests (parity/rt-safety are #[ignore] pending fixtures)
cargo test -- --ignored   # once tests/fixtures are generated — see tests/fixtures/README.md
```

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
