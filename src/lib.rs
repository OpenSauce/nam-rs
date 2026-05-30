//! # nam-rs
//!
//! Pure-Rust, real-time-safe inference for [Neural Amp Modeler] (NAM) `.nam` models.
//!
//! This crate loads a `.nam` model file and runs its neural network forward pass
//! sample-by-sample, with no heap allocation on the audio (hot) path. It is a
//! from-scratch Rust port of NAM's inference, written against the reference
//! implementations below and validated for **bit-level parity** against them.
//!
//! ## Design contract
//!
//! 1. **Parity with the reference.** The forward pass must produce output equal
//!    (within float tolerance) to the canonical Python/C++ NAM implementations for
//!    the same `.nam` file and input. This is enforced by `tests/parity.rs`.
//! 2. **Real-time safety.** [`WaveNet::process_buffer`] performs zero heap
//!    allocations, locks, or system calls. All scratch buffers are pre-allocated at
//!    construction. This is enforced by `tests/rt_safety.rs`.
//!
//! ## Attribution
//!
//! This is a derivative work. The algorithm and weight layout are ported from the
//! following projects (see `NOTICE` for license texts):
//!
//! - [neural-amp-modeler] — Steven Atkinson's reference trainer + `.nam` exporter
//!   (Python, MIT). The source of truth for `export_weights` / `export_config`.
//! - [NeuralAmpModelerCore] — the canonical C++ inference library (MIT).
//! - [NeuralAudio] — Mike Oliphant's high-performance C++ NAM/RTNeural runtime,
//!   designed to match NAM Core exactly (MIT). Primary porting reference.
//! - [waveny] — a Go port of NAM (Apache-2.0). Used as a conceptual cross-check only.
//!
//! [Neural Amp Modeler]: https://www.neuralampmodeler.com/
//! [neural-amp-modeler]: https://github.com/sdatkinson/neural-amp-modeler
//! [NeuralAmpModelerCore]: https://github.com/sdatkinson/NeuralAmpModelerCore
//! [NeuralAudio]: https://github.com/mikeoliphant/NeuralAudio
//! [waveny]: https://github.com/nlpodyssey/waveny

mod error;
mod model;
mod wavenet;

pub use error::Error;
pub use model::{LayerArrayConfig, NamModel, WaveNetConfig, DEFAULT_SAMPLE_RATE};
pub use wavenet::WaveNet;
