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

/// The multi-tap conv head matches NAMCore: a committed, oracle-generated fixture
/// run through nam-rs must equal NAMCore's output in steady state within TOLERANCE.
/// CI-safe (no oracle/real files needed at test time).
#[test]
fn conv_head_matches_namcore_oracle() {
    let json = std::fs::read_to_string(fixture("conv_head.nam"))
        .unwrap_or_else(|e| panic!("missing fixture conv_head.nam: {e}"));
    let model = NamModel::from_json_str(&json).expect("parse conv-head model");
    let mut net = Model::from_nam(&model).expect("conv-head model builds");
    let input = load_samples("input_conv_head.json");
    let expected = load_samples("expected_conv_head.json");
    assert_eq!(input.len(), expected.len());
    let rf = net.receptive_field();
    let mut signal = input.clone();
    net.process_buffer(&mut signal);
    let max_err = signal[rf..]
        .iter()
        .zip(&expected[rf..])
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= TOLERANCE,
        "conv-head max steady-state error {max_err} > {TOLERANCE}"
    );
}

/// End-to-end parity for the committed synthetic A2 fixtures: each `a2_*.nam` run
/// through nam-rs must equal the NAMCore oracle output in steady state within
/// TOLERANCE. CI-safe (committed fixtures; skips cleanly if any are absent).
#[test]
fn a2_synthetic_fixtures_match_namcore_oracle() {
    for name in [
        "grouped",
        "film",
        "gated",
        "blended",
        "head1x1",
        "bottleneck",
    ] {
        let model_file = format!("a2_{name}.nam");
        let in_file = format!("input_a2_{name}.json");
        let exp_file = format!("expected_a2_{name}.json");
        if !fixture(&model_file).exists() || !fixture(&exp_file).exists() {
            eprintln!("skip a2_{name}: fixture missing");
            continue;
        }
        let json = std::fs::read_to_string(fixture(&model_file)).unwrap();
        let model = NamModel::from_json_str(&json).expect("parse a2 synthetic");
        let mut net = Model::from_nam(&model).expect("a2 synthetic builds");
        let input = load_samples(&in_file);
        let expected = load_samples(&exp_file);
        assert_eq!(input.len(), expected.len());
        let rf = net.receptive_field();
        let mut signal = input.clone();
        net.process_buffer(&mut signal);
        let max_err = signal[rf..]
            .iter()
            .zip(&expected[rf..])
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err <= TOLERANCE,
            "a2_{name}: max steady-state err {max_err} > {TOLERANCE}"
        );
    }
}

/// End-to-end parity for the real A2 captures (gitignored). For each file, compare
/// nam-rs against the NAMCore oracle binary on the same input, in steady state.
/// Skips when the oracle binary or the capture files are absent (e.g. CI).
#[test]
fn real_a2_captures_match_namcore_oracle() {
    use std::process::Command;

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let oracle = root.join("tests/oracle/build/oracle");
    let examples = root.join("tests/examples");
    if !oracle.exists() {
        eprintln!("skip: oracle binary not built ({oracle:?})");
        return;
    }
    let Ok(entries) = std::fs::read_dir(&examples) else {
        eprintln!("skip: no tests/examples dir");
        return;
    };

    let n = 8192usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.019).sin() * 0.5 + (i as f32 * 0.0031).sin() * 0.2)
        .collect();

    let mut checked = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("nam") {
            continue;
        }
        let json = std::fs::read_to_string(&path).unwrap();
        let model = NamModel::from_json_str(&json).unwrap();
        let mut net = Model::from_nam(&model)
            .unwrap_or_else(|e| panic!("{:?}: build failed {e:?}", path.file_name().unwrap()));
        let mut got = signal.clone();
        net.process_buffer(&mut got);

        // Unique per-iteration scratch paths: keyed by pid + file stem so this test
        // can never collide with a concurrently-running test (cargo runs test
        // functions on parallel threads in one process) writing to temp_dir.
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let scratch =
            std::env::temp_dir().join(format!("nam_rs_oracle_{}_{stem}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let in_path = scratch.join("in.json");
        let out_path = scratch.join("out.json");
        std::fs::write(&in_path, serde_json::to_string(&signal).unwrap()).unwrap();
        let status = Command::new(&oracle)
            .arg(&path)
            .arg(&in_path)
            .arg(&out_path)
            .status()
            .unwrap();
        assert!(
            status.success(),
            "oracle failed on {:?}",
            path.file_name().unwrap()
        );
        let want: Vec<f32> =
            serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
        std::fs::remove_dir_all(&scratch).ok();

        let rf = net.receptive_field();
        let max_err = got[rf..]
            .iter()
            .zip(&want[rf..])
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "{:?}: max steady-state err {max_err:.2e}",
            path.file_name().unwrap()
        );
        assert!(
            max_err <= TOLERANCE,
            "{:?}: max steady-state error {max_err} > {TOLERANCE}",
            path.file_name().unwrap()
        );
        checked += 1;
    }
    eprintln!("real A2 capture parity: checked {checked} file(s)");
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
