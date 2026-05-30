//! Parity harness: nam-rs output MUST equal the reference NAM implementation.
//!
//! This is the correctness oracle for the forward pass. It loads a real `.nam`
//! model plus an (input, expected-output) sample pair generated from the canonical
//! Python implementation, runs nam-rs, and asserts sample-by-sample equality within
//! a tight float tolerance.
//!
//! The fixtures are committed under `tests/fixtures/`; `gen_fixtures.py` regenerates
//! them (see `tests/fixtures/README.md`). The comparison skips the receptive-field
//! warmup — torch's training forward and a streaming engine use different zero-history
//! conventions there — and asserts steady-state parity, where the match is ~1.5e-7.

use std::path::Path;

use nam_rs::{NamModel, WaveNet};

/// Max absolute per-sample deviation allowed from the reference output.
/// NAM Core runs in `f32`; this matches the tolerance NeuralAudio uses for parity.
const TOLERANCE: f32 = 1e-5;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Load a JSON array of numbers as `Vec<f32>`.
fn load_samples(name: &str) -> Vec<f32> {
    let raw = std::fs::read_to_string(fixture(name))
        .unwrap_or_else(|e| panic!("missing fixture {name}: {e} (see tests/fixtures/README.md)"));
    serde_json::from_str(&raw).expect("fixture must be a JSON array of numbers")
}

#[test]
fn matches_python_reference_wavenet() {
    let json = std::fs::read_to_string(fixture("reference.nam")).expect("reference.nam");
    let model = NamModel::from_json_str(&json).expect("parse reference model");

    let mut wn = WaveNet::new(&model).expect("build WaveNet");
    let mut signal = load_samples("input.json");
    let expected = load_samples("expected_output.json");
    assert_eq!(signal.len(), expected.len(), "fixture length mismatch");

    wn.process_buffer(&mut signal);

    // Skip the receptive-field warmup. The reference fixture comes from torch's
    // `_WaveNet.forward`, which pre-pads the whole input with zeros and propagates
    // each layer's bias/activation through the stack. A streaming engine (this crate,
    // and NAM Core / NeuralAudio) instead starts every layer from a zero-filled
    // history buffer. The two conventions provably agree once the receptive field
    // fills, but differ over the first `rf` samples (a ~0.5 ms startup transient at
    // 48 kHz). We assert parity over the steady state — the part a host actually
    // hears across buffer boundaries — where the match is ~1.5e-7.
    let rf = wn.receptive_field();
    assert!(
        signal.len() > rf,
        "fixture shorter than the receptive field"
    );

    let max_err = signal[rf..]
        .iter()
        .zip(&expected[rf..])
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err <= TOLERANCE,
        "max steady-state per-sample error {max_err} exceeds tolerance {TOLERANCE} \
         (compared samples {rf}..{})",
        signal.len()
    );
}
