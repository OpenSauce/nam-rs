//! Real-time LSTM inference (NAM recurrent architecture).
//!
//! Mirrors the WaveNet runtime's contract: built once from a parsed [`NamModel`]
//! (allocating), then run on the audio thread with zero allocation. State (`h`,`c`)
//! is initialised from the model's **exported** initial hidden/cell vectors — the
//! core burned in over silence — not zeros, matching NAM Core / NeuralAudio.

mod cell;

// used by the Lstm runtime added in a later task
#[allow(unused_imports)]
use cell::LstmCell;
