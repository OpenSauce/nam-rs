//! Tests for `.nam` file parsing (the on-disk format → [`NamModel`]).

use nam_rs::{NamModel, DEFAULT_SAMPLE_RATE};

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
    assert_eq!(model.config.layers.len(), 1);

    let layer = &model.config.layers[0];
    assert_eq!(layer.channels, 2);
    assert_eq!(layer.kernel_size, 3);
    assert_eq!(layer.dilations, vec![1, 2]);
    assert_eq!(layer.activation, "Tanh");
    assert!(!layer.gated);
    assert!(!layer.head_bias);

    assert!((model.config.head_scale - 0.5).abs() < 1e-9);
    assert_eq!(model.weights, vec![0.1_f32, -0.2, 0.3]);
}

#[test]
fn sample_rate_defaults_to_48k_when_absent() {
    let model = NamModel::from_json_str(MINIMAL_WAVENET).expect("should parse");
    assert!(model.sample_rate.is_none());
    assert!((model.sample_rate() - DEFAULT_SAMPLE_RATE).abs() < 1e-9);
}

#[test]
fn sample_rate_is_read_when_present() {
    let json = MINIMAL_WAVENET.replace(
        "\"weights\": [0.1, -0.2, 0.3]",
        "\"sample_rate\": 44100.0, \"weights\": [0.1, -0.2, 0.3]",
    );
    let model = NamModel::from_json_str(&json).expect("should parse");
    assert!((model.sample_rate() - 44100.0).abs() < 1e-9);
}

#[test]
fn rejects_malformed_json() {
    assert!(NamModel::from_json_str("{ not json").is_err());
}
