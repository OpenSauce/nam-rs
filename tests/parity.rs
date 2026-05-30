//! Parity harness: nam-rs output MUST equal the reference NAM implementation.
//!
//! This is the correctness oracle for the forward pass. It loads a real `.nam`
//! model plus an (input, expected-output) sample pair generated from the canonical
//! Python implementation, runs nam-rs, and asserts sample-by-sample equality within
//! a tight float tolerance.
//!
//! The fixtures are NOT checked in yet — see `tests/fixtures/README.md` for the
//! script that regenerates them from `pip install neural-amp-modeler`. Until the
//! forward pass is implemented and fixtures exist, these tests are `#[ignore]`d;
//! removing the ignore is the RED step that drives the forward-pass implementation.

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
#[ignore = "needs reference fixtures + implemented forward pass; see tests/fixtures/README.md"]
fn matches_python_reference_wavenet() {
    let json = std::fs::read_to_string(fixture("reference.nam")).expect("reference.nam");
    let model = NamModel::from_json_str(&json).expect("parse reference model");

    let mut wn = WaveNet::new(&model).expect("build WaveNet");
    let mut signal = load_samples("input.json");
    let expected = load_samples("expected_output.json");
    assert_eq!(signal.len(), expected.len(), "fixture length mismatch");

    wn.process_buffer(&mut signal);

    let max_err = signal
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err <= TOLERANCE,
        "max per-sample error {max_err} exceeds tolerance {TOLERANCE}"
    );
}
