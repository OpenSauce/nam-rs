#![deny(missing_docs)]
//! # nam-rs
//!
//! Pure-Rust, real-time-safe inference for [Neural Amp Modeler] (NAM) `.nam` models.
//!
//! This crate loads a `.nam` file and runs its neural-network forward pass — a whole
//! buffer at a time (WaveNet uses a cache-friendly block kernel), or one sample at a
//! time — with no heap allocation on the audio thread. It is a from-scratch Rust port
//! of NAM's inference, written against the reference implementations under
//! [Attribution](#attribution) and validated numerically against them.
//!
//! ## Quickstart
//!
//! ```no_run
//! use nam_rs::{Model, NamModel};
//!
//! // Off the audio thread: parsing, construction and buffers all allocate here.
//! let file = NamModel::from_file("twin_reverb.nam")?;
//! let mut model = Model::from_nam(&file)?;
//! let mut block = vec![0.0_f32; 512];
//!
//! // Settle the state against silence so the first real block is clean.
//! let mut warmup = vec![0.0_f32; model.receptive_field()];
//! model.process_buffer(&mut warmup);
//!
//! // On the audio thread, once per block: in place, no allocation. State carries
//! // across calls, so block-wise output matches one whole-buffer call.
//! model.process_buffer(&mut block);
//! # Ok::<(), nam_rs::Error>(())
//! ```
//!
//! ## Design contract
//!
//! 1. **Parity.** Once past the model's warmup, the forward pass produces output
//!    within `1e-5` per sample of the canonical Python and C++ implementations, for
//!    the same `.nam` file and input. Enforced by `tests/parity.rs` against
//!    reference-generated fixtures.
//! 2. **Real-time safety.** [`Model::process_buffer`] performs no heap allocation, no
//!    locking and no system calls, on every architecture. All scratch buffers are
//!    allocated in [`Model::from_nam`]. Enforced by `tests/rt_safety.rs`.
//!
//! ## Architectures
//!
//! [`Model`] dispatches on the architecture the file declares, so you never branch on
//! it yourself:
//!
//! - **WaveNet** — a dilated-convolution stack.
//! - **LSTM** — recurrent, evaluated one sample at a time.
//! - **SlimmableContainer** — a multiplexer over complete standalone submodels, each
//!   itself a full model.
//!
//! A `SlimmableContainer` carries a width dial as a CPU/quality trade-off. Each
//! submodel keeps its own state, so switching mid-stream leaves the newly-selected
//! submodel to warm up from wherever it last left off. Switching allocates nothing and
//! takes no locks, so it is safe to call from the audio thread.
//!
//! ```no_run
//! # use nam_rs::{Model, NamModel};
//! # let file = NamModel::from_file("slimmable.nam")?;
//! let mut model = Model::from_nam(&file)?;
//!
//! if let Some(slim) = model.as_slimmable_mut() {
//!     // Activates the first submodel whose `max_value` exceeds 0.5 — the
//!     // smallest one that covers the requested width, not the largest below it.
//!     slim.set_slim_size(0.5);
//!     assert!(slim.active_index() < slim.len());
//!
//!     // Or address a submodel by index; out of range clamps to the last one.
//!     slim.select(0);
//! }
//! # Ok::<(), nam_rs::Error>(())
//! ```
//!
//! ## A2 feature support
//!
//! "A2" is NAM's second-generation architecture family. An A2 file may be a single
//! WaveNet using the newer feature set, or a `SlimmableContainer` holding several
//! submodels; the file declares which.
//!
//! Supported:
//!
//! - FiLM conditioning
//! - Gating
//! - Bottleneck and grouped convolutions
//! - Per-array heads, including multi-tap conv heads
//! - The optional post-stack head
//! - `condition_dsp` — a nested model whose output replaces the conditioning fed to
//!   every array, multi-channel output included
//!
//! Rejected at load, with [`Error::UnsupportedFeature`] or
//! [`Error::UnsupportedActivation`]:
//!
//! - Multi-channel input (`in_channels != 1`)
//! - A post-stack head with `out_channels != 1`
//! - Mixed gating modes within one layer array
//! - Activations outside the implemented set
//!
//! Structurally inconsistent files are rejected at load too, rather than surfacing
//! later: a conditioning width that disagrees with the `condition_dsp` output, grouped
//! convolutions whose channel counts are not divisible by the group count, a container
//! with no submodels, and similar.
//!
//! ## Warmup and state
//!
//! A WaveNet's first [`Model::receptive_field`] samples are a startup transient, while
//! the dilated stack fills against zero history. This crate does not settle that for
//! you at load time — NAM Core does, inside its factory — so feed the model that many
//! silent samples first if you need the first real block to be clean, as the
//! [Quickstart](#quickstart) does.
//!
//! LSTM models begin from the initial hidden and cell state stored in the file, which
//! the trainer exports already settled against silence. They have no transient, and
//! [`Model::receptive_field`] is `0` for them.
//!
//! [`Model::reset`] clears a model's state back to that starting point. For a
//! `SlimmableContainer` it resets every submodel's state but leaves the active
//! selection where it is.
//!
//! ## Sample rate
//!
//! A `.nam` is captured at a specific rate ([`Model::expected_sample_rate`], 48 kHz
//! when the file does not say). `nam-rs` does *not* resample — feed the model audio at
//! that rate, or resample in your host first. A mismatched rate produces silently
//! wrong output, because the model's dilations and recurrence are defined in samples,
//! not in seconds.
//!
//! ## Processing boundary
//!
//! `nam-rs` runs the model's forward pass and nothing else. The reference NAM plugin
//! additionally applies a DC blocker (high-pass) and, optionally, loudness
//! normalization on the output; those belong to your audio graph rather than to the
//! model. [`Model::loudness`], [`Model::input_level_dbu`] and
//! [`Model::output_level_dbu`] give you the calibration numbers to do that
//! gain-staging yourself.
//!
//! ## Metadata
//!
//! [`NamModel::metadata_typed`] parses the file's `metadata` block into a [`Metadata`]
//! struct — name, who captured it, gear make/model/type, tone type, trainer, export
//! date, loudness and calibration levels. It is descriptive only: everything a model
//! browser shows, and nothing the forward pass uses. Every field is optional and
//! parsed leniently, so one malformed entry never costs you the others, and the
//! unparsed block remains on [`NamModel::metadata`].
//!
//! One derived helper, [`Metadata::includes_cab`], answers the question that changes a
//! signal chain: whether the capture already contains a speaker cabinet, and so
//! whether an impulse response after it would be a second one.
//!
//! ## Loading without a file
//!
//! [`NamModel::from_json_str`] parses a model you already hold in memory — one just
//! downloaded, or embedded in your binary — with no filesystem access:
//!
//! ```
//! use nam_rs::{NamModel, Model};
//!
//! // A minimal single-layer WaveNet, inline for illustration.
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
//! let file = NamModel::from_json_str(json)?;
//! let mut model = Model::from_nam(&file)?;
//!
//! let mut buffer = [0.1_f32, 0.2, 0.3, 0.4];
//! model.process_buffer(&mut buffer);
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
//!
//! [Neural Amp Modeler]: https://www.neuralampmodeler.com/
//! [neural-amp-modeler]: https://github.com/sdatkinson/neural-amp-modeler
//! [NeuralAmpModelerCore]: https://github.com/sdatkinson/NeuralAmpModelerCore
//! [NeuralAudio]: https://github.com/mikeoliphant/NeuralAudio

mod error;
mod lstm;
mod model;
mod model_runtime;
mod reader;
mod wavenet;

pub use error::Error;
pub use lstm::Lstm;
pub use model::{
    ActivationSpec, Date, FilmConfig, GatingMode, Head1x1Config, Layer1x1Config, LayerArrayConfig,
    LstmConfig, Metadata, ModelConfig, NamModel, PostStackHeadConfig, SlimmableConfig,
    SlimmableSubmodel, WaveNetConfig, DEFAULT_SAMPLE_RATE,
};
pub use model_runtime::{Model, Slimmable};
pub use wavenet::WaveNet;

/// Compiles the code blocks in `README.md` as doctests so the README cannot drift
/// out of sync with the API. `cfg(doctest)` keeps it out of the rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
