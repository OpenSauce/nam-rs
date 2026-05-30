//! Real-time WaveNet inference.
//!
//! [`WaveNet`] is built once from a parsed [`NamModel`] (which may allocate), then
//! run on the audio thread via [`WaveNet::process_buffer`], which must never
//! allocate. The forward pass itself is implemented test-first against the parity
//! and RT-safety harnesses — see `tests/parity.rs` and `tests/rt_safety.rs`.

use crate::error::Error;
use crate::model::NamModel;

/// A ready-to-run WaveNet, with all scratch buffers pre-allocated.
#[derive(Debug)]
pub struct WaveNet {
    // Layer weights and pre-allocated ring/scratch buffers will live here.
    // Intentionally empty until the forward pass is implemented test-first.
}

impl WaveNet {
    /// Build a runnable model from a parsed `.nam` file.
    ///
    /// All allocation happens here. May fail if the architecture is unsupported or
    /// the weight blob does not match the config.
    pub fn new(_model: &NamModel) -> Result<Self, Error> {
        todo!("build WaveNet from config + weights")
    }

    /// Process a buffer of samples in place.
    ///
    /// **Real-time contract:** no heap allocation, locks, or syscalls. Enforced by
    /// `tests/rt_safety.rs`.
    pub fn process_buffer(&mut self, _io: &mut [f32]) {
        todo!("WaveNet forward pass")
    }

    /// Reset all internal state (ring buffers) to silence.
    pub fn reset(&mut self) {
        todo!("clear internal buffers")
    }
}
