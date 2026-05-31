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
//! ## Example
//!
//! ```
//! use nam_rs::{NamModel, WaveNet};
//!
//! // From disk you'd use `NamModel::from_file("model.nam")?`.
//! // Here we use a tiny in-line model for illustration.
//! let json = r#"{
//!     "version": "0.5.4", "architecture": "WaveNet",
//!     "config": {
//!         "layers": [{
//!             "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
//!             "kernel_size": 1, "dilations": [1], "activation": "ReLU",
//!             "gated": false, "head_bias": false
//!         }],
//!         "head": null, "head_scale": 1.0
//!     },
//!     "weights": [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
//! }"#;
//!
//! let model = NamModel::from_json_str(json)?;
//! let mut wavenet = WaveNet::new(&model)?;          // builds + allocates here
//!
//! // On the audio thread: process in place, no allocation.
//! let mut buffer = [0.1_f32, 0.2, 0.3, 0.4];
//! wavenet.process_buffer(&mut buffer);
//! # Ok::<(), nam_rs::Error>(())
//! ```
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
mod lstm;
mod model;
mod reader;
mod wavenet;

pub use error::Error;
pub use lstm::Lstm;
pub use model::{
    LayerArrayConfig, LstmConfig, Metadata, ModelConfig, NamModel, WaveNetConfig,
    DEFAULT_SAMPLE_RATE,
};
pub use wavenet::WaveNet;
