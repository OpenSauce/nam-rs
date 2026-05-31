//! Tests for `.nam` file parsing (the on-disk format → [`NamModel`]).

use nam_rs::{ModelConfig, NamModel, DEFAULT_SAMPLE_RATE};

/// A minimal but structurally-valid WaveNet `.nam`, with `sample_rate` omitted.
const MINIMAL_WAVENET: &str = r#"{
    "version": "0.5.4",
    "architecture": "WaveNet",
    "config": {
        "layers": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": 2,
                "head_size": 1,
                "kernel_size": 3,
                "dilations": [1, 2],
                "activation": "Tanh",
                "gated": false,
                "head_bias": false
            }
        ],
        "head": null,
        "head_scale": 0.5
    },
    "weights": [0.1, -0.2, 0.3]
}"#;

#[test]
fn parses_minimal_wavenet_config() {
    let model = NamModel::from_json_str(MINIMAL_WAVENET).expect("should parse");

    assert_eq!(model.version, "0.5.4");
    assert_eq!(model.architecture, "WaveNet");
    let cfg = match &model.config {
        ModelConfig::WaveNet(c) => c,
        other => panic!("expected WaveNet config, got {other:?}"),
    };
    let layers = &cfg.layers;
    assert_eq!(layers.len(), 1);
    let layer = &layers[0];
    assert_eq!(layer.channels, 2);
    assert_eq!(layer.kernel_size, 3);
    assert_eq!(layer.dilations, vec![1, 2]);
    assert_eq!(layer.activation, "Tanh");
    assert!(!layer.gated);
    assert!(!layer.head_bias);

    assert!((cfg.head_scale - 0.5).abs() < 1e-9);
    assert_eq!(model.weights, vec![0.1_f32, -0.2, 0.3]);
}

#[test]
fn sample_rate_defaults_to_48k_when_absent() {
    let model = NamModel::from_json_str(MINIMAL_WAVENET).expect("should parse");
    assert!(model.sample_rate.is_none());
    assert!((model.expected_sample_rate() - DEFAULT_SAMPLE_RATE).abs() < 1e-9);
}

#[test]
fn sample_rate_is_read_when_present() {
    let json = MINIMAL_WAVENET.replace(
        "\"weights\": [0.1, -0.2, 0.3]",
        "\"sample_rate\": 44100.0, \"weights\": [0.1, -0.2, 0.3]",
    );
    let model = NamModel::from_json_str(&json).expect("should parse");
    assert!((model.expected_sample_rate() - 44100.0).abs() < 1e-9);
}

#[test]
fn rejects_malformed_json() {
    assert!(NamModel::from_json_str("{ not json").is_err());
}

#[test]
fn rejects_wrong_typed_config_field() {
    // A config field with the wrong JSON type must fail to parse, not panic.
    let json = MINIMAL_WAVENET.replace("\"channels\": 2", "\"channels\": \"lots\"");
    assert!(NamModel::from_json_str(&json).is_err());
}

/// A WaveNet file carrying NAM's metadata block (keys taken from a real .nam).
const WITH_METADATA: &str = r#"{
    "version": "0.5.4",
    "architecture": "WaveNet",
    "config": {
        "layers": [{
            "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
            "kernel_size": 1, "dilations": [1], "activation": "ReLU",
            "gated": false, "head_bias": false
        }],
        "head": null, "head_scale": 1.0
    },
    "weights": [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    "metadata": {
        "loudness": -20.02, "input_level_dbu": 18.3, "output_level_dbu": 12.3,
        "name": "Test", "gear_type": "amp"
    }
}"#;

#[test]
fn parses_loudness_and_calibration_metadata() {
    let m = NamModel::from_json_str(WITH_METADATA).expect("parse");
    // Compare with a tolerance, not `assert_eq!`: these parse f64 -> f32, so an exact
    // bit-match isn't guaranteed across platforms/serde versions.
    let approx = |got: Option<f32>, want: f32| (got.expect("present") - want).abs() < 1e-4;
    assert!(approx(m.loudness(), -20.02));
    assert!(approx(m.input_level_dbu(), 18.3));
    assert!(approx(m.output_level_dbu(), 12.3));
}

#[test]
fn metadata_absent_yields_none() {
    // MINIMAL_WAVENET has no metadata block at all.
    let m = NamModel::from_json_str(MINIMAL_WAVENET).expect("parse");
    assert_eq!(m.loudness(), None);
    assert_eq!(m.input_level_dbu(), None);
    assert_eq!(m.output_level_dbu(), None);
}

#[test]
fn unrelated_metadata_keys_are_ignored() {
    let json = WITH_METADATA.replace(
        "\"loudness\": -20.02, \"input_level_dbu\": 18.3, \"output_level_dbu\": 12.3,",
        "",
    );
    let m = NamModel::from_json_str(&json).expect("parse");
    assert_eq!(m.loudness(), None);
    assert_eq!(m.input_level_dbu(), None);
    assert_eq!(m.output_level_dbu(), None);
    // unrelated keys ("name", "gear_type") must not error.
}

const MINIMAL_LSTM: &str = r#"{
    "version": "0.5.4",
    "architecture": "LSTM",
    "config": { "input_size": 1, "hidden_size": 8, "num_layers": 1 },
    "weights": [0.0],
    "sample_rate": 44100.0
}"#;

#[test]
fn parses_lstm_config() {
    let m = NamModel::from_json_str(MINIMAL_LSTM).expect("parse LSTM");
    assert_eq!(m.architecture, "LSTM");
    match &m.config {
        ModelConfig::Lstm(c) => {
            assert_eq!(c.input_size, 1);
            assert_eq!(c.hidden_size, 8);
            assert_eq!(c.num_layers, 1);
        }
        other => panic!("expected Lstm config, got {other:?}"),
    }
    assert_eq!(m.expected_sample_rate(), 44100.0);
}

#[test]
fn wavenet_config_still_parses_through_enum() {
    let m = NamModel::from_json_str(MINIMAL_WAVENET).expect("parse");
    match &m.config {
        ModelConfig::WaveNet(c) => assert_eq!(c.layers.len(), 1),
        other => panic!("expected WaveNet config, got {other:?}"),
    }
}

#[test]
fn unknown_architecture_fails_to_parse() {
    let json = MINIMAL_WAVENET.replace("\"WaveNet\"", "\"Transformer\"");
    let err = NamModel::from_json_str(&json).unwrap_err();
    assert!(
        format!("{err}").contains("Transformer"),
        "error should name the bad architecture: {err}"
    );
}
