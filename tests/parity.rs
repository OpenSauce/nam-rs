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

use nam_rs::{Model, NamModel, WaveNet};

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
    assert_parity("reference.nam", "input.json", "expected_output.json");
}

/// Realistic standard-architecture model (NAM Core `wavenet_a1_standard.nam`, MIT):
/// two layer-arrays, 16+8 channels, dilations 1..512, 13,802 weights — the same
/// shape real-world amp captures use. Guards the forward pass at production channel
/// sizes, not just the 131-weight minimal model.
#[test]
fn matches_python_reference_wavenet_standard() {
    assert_parity(
        "reference_standard.nam",
        "input_standard.json",
        "expected_output_standard.json",
    );
}

#[test]
fn matches_python_reference_lstm() {
    assert_parity_full_lstm(
        "reference_lstm.nam",
        "input_lstm.json",
        "expected_output_lstm.json",
    );
}

#[test]
fn matches_python_reference_lstm_standard() {
    assert_parity_full_lstm(
        "reference_lstm_standard.nam",
        "input_lstm_standard.json",
        "expected_output_lstm_standard.json",
    );
}

/// LSTM parity: full-length compare (receptive field is 1, no warmup transient).
fn assert_parity_full_lstm(model_file: &str, input_file: &str, expected_file: &str) {
    let json = std::fs::read_to_string(fixture(model_file))
        .unwrap_or_else(|e| panic!("missing fixture {model_file}: {e}"));
    let model = NamModel::from_json_str(&json).expect("parse LSTM model");
    let mut net = Model::from_nam(&model).expect("build Model");

    let mut signal = load_samples(input_file);
    let expected = load_samples(expected_file);
    assert_eq!(signal.len(), expected.len(), "fixture length mismatch");

    net.process_buffer(&mut signal);

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

/// SlimmableContainer parity: at each slim selection, the container must reproduce
/// the corresponding submodel's reference output. Each submodel is a standalone model
/// oracled independently (make_slim_fixtures.py); the WaveNet submodels skip their
/// receptive-field warmup, the LSTM submodel (rf 0) compares full-length.
#[test]
fn matches_reference_slimmable_container() {
    let json = std::fs::read_to_string(fixture("slimmable_container.nam")).expect("read container");
    let base = NamModel::from_json_str(&json).expect("parse container");
    let input = load_samples("input_slim.json");

    // slimmable_container.nam has exactly 3 submodels (indices 0..=2); the oracle
    // fixtures are committed per index, so this range is fixed, not runtime-derived.
    for i in 0..3 {
        let expected = load_samples(&format!("expected_slim_{i}.json"));
        assert_eq!(
            input.len(),
            expected.len(),
            "fixture length mismatch (submodel {i})"
        );

        let mut model = Model::from_nam(&base).expect("build container");
        model.as_slimmable_mut().expect("is slimmable").select(i);
        let rf = model.receptive_field();

        let mut signal = input.clone();
        model.process_buffer(&mut signal);

        let max_err = signal[rf..]
            .iter()
            .zip(&expected[rf..])
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err <= TOLERANCE,
            "submodel {i}: max steady-state error {max_err} exceeds {TOLERANCE} (skipped rf={rf})"
        );
    }
}

/// LeakyReLU WaveNet parity. No shipped A2 example uses LeakyReLU, so this model is
/// self-generated (make_leaky_wavenet.py) and oracled through canonical torch.
#[test]
fn matches_reference_leaky_wavenet() {
    assert_parity(
        "leaky_wavenet.nam",
        "input_leaky.json",
        "expected_output_leaky.json",
    );
}

/// Load `model_file`, run `input_file` through it, and assert steady-state parity
/// against `expected_file` within [`TOLERANCE`].
fn assert_parity(model_file: &str, input_file: &str, expected_file: &str) {
    let json = std::fs::read_to_string(fixture(model_file))
        .unwrap_or_else(|e| panic!("missing fixture {model_file}: {e}"));
    let model = NamModel::from_json_str(&json).expect("parse reference model");

    let mut wn = WaveNet::new(&model).expect("build WaveNet");
    let mut signal = load_samples(input_file);
    let expected = load_samples(expected_file);
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
