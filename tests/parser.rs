//! Tests for `.nam` file parsing (the on-disk format → [`NamModel`]).

use nam_rs::{ActivationSpec, ModelConfig, NamModel, SlimmableConfig, DEFAULT_SAMPLE_RATE};
use nam_rs::{Error, Model};

fn build_fixture(name: &str) -> Result<Model, Error> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let json = std::fs::read_to_string(path).expect("read fixture");
    let model = NamModel::from_json_str(&json)?;
    Model::from_nam(&model)
}

#[test]
fn a2_max_features_are_rejected_not_run() {
    // wavenet_a2_max.nam carries bottleneck/FiLM/groups/dict-activation etc.
    // It must error cleanly (UnsupportedFeature or WeightCountMismatch), never panic.
    match build_fixture("wavenet_a2_max.nam") {
        Err(Error::UnsupportedFeature(_))
        | Err(Error::WeightCountMismatch { .. })
        | Err(Error::UnsupportedActivation(_)) => {}
        other => panic!("expected a clean rejection, got {other:?}"),
    }
}

#[test]
fn mono_condition_dsp_is_supported_not_blanket_guarded() {
    // condition_dsp is no longer a blanket-guarded feature. A *mono* condition_dsp
    // (`condition_size == 1`) builds and runs — full forward parity is covered by the
    // parity suite's `condition_dsp_matches_namcore_oracle`. It must build with no
    // error at all (not merely avoid a "condition_dsp"-worded one).
    let mut m = build_fixture("condition_dsp_mono.nam").expect("mono condition_dsp must build");
    let mut buf = vec![0.1_f32; 256];
    m.process_buffer(&mut buf); // must not panic
}

#[test]
fn multi_channel_condition_dsp_builds_and_runs() {
    // A multi-channel-output condition_dsp is now supported: the NAMCore
    // `wavenet_condition_dsp.nam` example feeds arrays with `condition_size == 3`, fed by
    // a nested WaveNet emitting 3 output channels. It must BUILD and RUN (the N rows of
    // the nested model become the N-wide conditioning) — never reject, never panic.
    // Steady-state parity vs the oracle is covered by the parity suite.
    let mut m =
        build_fixture("wavenet_condition_dsp.nam").expect("multi-channel condition_dsp must build");
    let mut buf = vec![0.1_f32; 256];
    m.process_buffer(&mut buf); // must not panic
}

#[test]
fn slimmable_wavenet_still_builds_and_runs() {
    // The benign `slimmable` training key must NOT trip the guard.
    let mut m = build_fixture("slimmable_wavenet.nam").expect("should build");
    let mut buf = vec![0.1_f32; 64];
    m.process_buffer(&mut buf); // must not panic
}

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
    assert_eq!(layer.kernel_sizes[0], 3);
    assert_eq!(layer.dilations, vec![1, 2]);
    assert!(
        matches!(&layer.activations[0], nam_rs::ActivationSpec::Named { name, negative_slope: None } if name == "Tanh"),
        "got {:?}",
        layer.activations[0]
    );
    assert!(!matches!(
        layer.gating_modes[0],
        nam_rs::GatingMode::Gated | nam_rs::GatingMode::Blended
    ));
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
        "gain": 0.824, "name": "Test", "modeled_by": "somebody",
        "gear_make": "Marshall", "gear_model": "JMP-50", "gear_type": "amp",
        "tone_type": "overdrive", "trainer": "TONE3000",
        "date": {"year": 2026, "month": 5, "day": 15,
                 "hour": 18, "minute": 45, "second": 6}
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
fn metadata_typed_parses_all_fields_in_one_call() {
    let m = NamModel::from_json_str(WITH_METADATA).expect("parse");
    let md = m.metadata_typed();
    let approx = |got: Option<f32>, want: f32| (got.expect("present") - want).abs() < 1e-4;
    assert!(approx(md.loudness, -20.02));
    assert!(approx(md.input_level_dbu, 18.3));
    assert!(approx(md.output_level_dbu, 12.3));
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
    // Swap the calibration keys for ones nam-rs knows nothing about: the unknown keys
    // must be skipped silently rather than failing the parse.
    let json = WITH_METADATA.replace(
        "\"loudness\": -20.02, \"input_level_dbu\": 18.3, \"output_level_dbu\": 12.3,",
        "\"some_future_key\": 1, \"another\": {\"nested\": true},",
    );
    let m = NamModel::from_json_str(&json).expect("parse");
    assert_eq!(m.loudness(), None);
    assert_eq!(m.input_level_dbu(), None);
    assert_eq!(m.output_level_dbu(), None);
    // ...while the keys that *are* known still come through.
    assert_eq!(m.metadata_typed().name.as_deref(), Some("Test"));
}

#[test]
fn parses_descriptive_metadata() {
    let md = NamModel::from_json_str(WITH_METADATA)
        .expect("parse")
        .metadata_typed();
    assert_eq!(md.name.as_deref(), Some("Test"));
    assert_eq!(md.modeled_by.as_deref(), Some("somebody"));
    assert_eq!(md.gear_make.as_deref(), Some("Marshall"));
    assert_eq!(md.gear_model.as_deref(), Some("JMP-50"));
    assert_eq!(md.gear_type.as_deref(), Some("amp"));
    assert_eq!(md.tone_type.as_deref(), Some("overdrive"));
    assert_eq!(md.trainer.as_deref(), Some("TONE3000"));
    assert!((md.gain.expect("gain") - 0.824).abs() < 1e-4);

    let date = md.date.expect("date");
    assert_eq!(
        (
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second
        ),
        (2026, 5, 15, 18, 45, 6)
    );
}

/// `gear_type` and `tone_type` are `String`, not enums, on purpose: files in the wild
/// carry values outside NAM's own enums (TONE3000 writes `"full-rig"` and
/// `"T3K-Null"`). A typed enum would reject those files outright.
#[test]
fn nonstandard_gear_and_tone_types_still_parse() {
    let json = WITH_METADATA
        .replace("\"gear_type\": \"amp\"", "\"gear_type\": \"full-rig\"")
        .replace(
            "\"tone_type\": \"overdrive\"",
            "\"tone_type\": \"T3K-Null\"",
        );
    let md = NamModel::from_json_str(&json)
        .expect("parse")
        .metadata_typed();
    assert_eq!(md.gear_type.as_deref(), Some("full-rig"));
    assert_eq!(md.tone_type.as_deref(), Some("T3K-Null"));
}

/// Real files write explicit JSON `null` for absent calibration levels, and write
/// integers where the schema says float. Both must parse as cleanly as an absent key.
///
/// Note this passes with or without the `lenient` deserializer — plain
/// `#[serde(default)] Option<T>` already handles nulls and integer-valued floats. It
/// is a regression test for that baseline, *not* coverage of `lenient`; the shapes
/// `lenient` actually exists for are covered by the two tests below.
#[test]
fn null_and_integer_metadata_values_parse() {
    let json = WITH_METADATA.replace(
        "\"input_level_dbu\": 18.3, \"output_level_dbu\": 12.3,",
        "\"input_level_dbu\": null, \"output_level_dbu\": 14,",
    );
    let md = NamModel::from_json_str(&json)
        .expect("parse")
        .metadata_typed();
    assert_eq!(md.input_level_dbu, None);
    assert!((md.output_level_dbu.expect("present") - 14.0).abs() < 1e-4);
}

/// One field with an unexpected shape must not take the rest of the block down with
/// it. `Metadata` is parsed in one `from_value` call that falls back to `Default` on
/// error, so without per-field leniency a single junk value would silently erase
/// every other field — including the calibration numbers the DSP path cares about.
#[test]
fn one_malformed_field_does_not_discard_the_others() {
    let json = WITH_METADATA.replace(
        "\"date\": {\"year\": 2026, \"month\": 5, \"day\": 15,\n                 \"hour\": 18, \"minute\": 45, \"second\": 6}",
        "\"date\": \"last Tuesday\"",
    );
    assert!(json.contains("last Tuesday"), "replacement must apply");

    let md = NamModel::from_json_str(&json)
        .expect("parse")
        .metadata_typed();
    assert_eq!(md.date, None, "the malformed field itself yields None");
    // ...and everything else survives.
    assert!((md.loudness.expect("loudness") - -20.02).abs() < 1e-4);
    assert_eq!(md.gear_type.as_deref(), Some("amp"));
    assert_eq!(md.name.as_deref(), Some("Test"));
}

/// The other shapes `lenient` exists for: a scalar field holding the wrong JSON type.
/// Without per-field leniency either of these would fail the whole `from_value` call
/// and, via `metadata_typed`'s fallback to `Default`, silently erase the calibration
/// numbers the DSP path depends on.
#[test]
fn wrong_typed_scalar_fields_cost_only_themselves() {
    // A number where a string belongs, and a string where a number belongs.
    let json = WITH_METADATA
        .replace("\"gear_type\": \"amp\"", "\"gear_type\": 5")
        .replace("\"loudness\": -20.02", "\"loudness\": \"quite loud\"");
    let md = NamModel::from_json_str(&json)
        .expect("parse")
        .metadata_typed();

    assert_eq!(md.gear_type, None, "the malformed field itself yields None");
    assert_eq!(md.loudness, None, "the malformed field itself yields None");
    // The neighbours are untouched — including the other calibration levels.
    assert!((md.input_level_dbu.expect("input level") - 18.3).abs() < 1e-4);
    assert!((md.output_level_dbu.expect("output level") - 12.3).abs() < 1e-4);
    assert_eq!(md.name.as_deref(), Some("Test"));
    assert_eq!(md.tone_type.as_deref(), Some("overdrive"));
    assert!(md.date.is_some());
    // A `gear_type` that failed to parse is unknown, never a guess.
    assert_eq!(md.includes_cab(), None);
}

#[test]
fn includes_cab_classifies_known_gear_types() {
    let with_gear = |gear: &str| {
        let json =
            WITH_METADATA.replace("\"gear_type\": \"amp\"", &format!("\"gear_type\": {gear}"));
        NamModel::from_json_str(&json)
            .expect("parse")
            .includes_cab()
    };

    // Speaker/cab in the signal chain. NAM's vocabulary...
    for gear in ["\"amp_cab\"", "\"amp_pedal_cab\""] {
        assert_eq!(with_gear(gear), Some(true), "{gear} should report a cab");
    }
    // ...and TONE3000's, which hyphenates. `full-rig` is documented by TONE3000 as
    // an alias for `amp-cab`, and `ir` is a directly captured cab response.
    for gear in ["\"amp-cab\"", "\"cab\"", "\"full-rig\"", "\"ir\""] {
        assert_eq!(with_gear(gear), Some(true), "{gear} should report a cab");
    }
    // Gear that stops before the speaker, from both vocabularies.
    for gear in [
        "\"amp\"",
        "\"preamp\"",
        "\"pedal\"",
        "\"pedal_amp\"",
        "\"outboard\"",
        "\"space\"",
    ] {
        assert_eq!(with_gear(gear), Some(false), "{gear} should report no cab");
    }
    // Case, surrounding whitespace, and `-` vs `_` are not significant, so the two
    // spellings of every cross-vocabulary value agree.
    assert_eq!(with_gear("\"  AMP_CAB \""), Some(true));
    assert_eq!(with_gear("\"full_rig\""), with_gear("\"full-rig\""));
    assert_eq!(with_gear("\"amp_cab\""), with_gear("\"amp-cab\""));

    // An unrecognised or absent value is "unknown", never a guess. NAM's `studio`
    // is undocumented and reads both ways; TONE3000's `experimental` promises
    // nothing about the chain. Both stay unknown on purpose.
    assert_eq!(with_gear("\"studio\""), None);
    assert_eq!(with_gear("\"experimental\""), None);
    assert_eq!(with_gear("\"something-new\""), None);
    assert_eq!(with_gear("null"), None);

    // `cab` is matched as a whole `_`-separated token, never a substring: these contain
    // "cab" while meaning something else entirely. A false `Some(true)` is the
    // dangerous direction — it tells the caller to drop an IR the model needs.
    for gear in ["\"cabless\"", "\"cable\"", "\"cabaret\""] {
        assert_eq!(with_gear(gear), None, "{gear} must not read as a cab");
    }
    // ...but an unseen spelling that does name a cab component still classifies,
    // without needing a new match arm.
    assert_eq!(with_gear("\"pedal_amp_cab\""), Some(true));
    // Documented limit of the token rule: a *negated* spelling still reads as
    // cab-inclusive, since `cab` is one of its tokens. No vendor forms values this
    // way, and sniffing for negation in free text is the guesswork this function
    // exists to avoid — so the behaviour is pinned here rather than defended against.
    assert_eq!(with_gear("\"amp_no_cab\""), Some(true));
    assert_eq!(
        NamModel::from_json_str(MINIMAL_WAVENET)
            .expect("parse")
            .includes_cab(),
        None
    );
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

/// Builds a WaveNet config JSON with the given raw `activation` snippet.
fn wavenet_with_activation(activation_json: &str) -> String {
    format!(
        r#"{{"version":"0.7.0","architecture":"WaveNet","config":{{"layers":[{{
            "input_size":1,"condition_size":1,"channels":1,"head_size":1,
            "kernel_size":1,"dilations":[1],"activation":{activation_json},
            "gated":false,"head_bias":false}}],"head":null,"head_scale":1.0}},
            "weights":[1.0,2.0,0.0,0.0,1.0,0.0,1.0,1.0]}}"#
    )
}

fn first_layer_activation(json: &str) -> ActivationSpec {
    let m = NamModel::from_json_str(json).expect("parse");
    match &m.config {
        ModelConfig::WaveNet(c) => c.layers[0].activations[0].clone(),
        other => panic!("expected WaveNet, got {other:?}"),
    }
}

#[test]
fn activation_bare_string_parses() {
    let a = first_layer_activation(&wavenet_with_activation(r#""LeakyReLU""#));
    assert!(
        matches!(a, ActivationSpec::Named { name, negative_slope: None } if name == "LeakyReLU")
    );
}

#[test]
fn activation_dict_default_slope_parses() {
    let a = first_layer_activation(&wavenet_with_activation(r#"{"type":"LeakyReLU"}"#));
    assert!(
        matches!(a, ActivationSpec::Named { name, negative_slope: None } if name == "LeakyReLU")
    );
}

#[test]
fn activation_dict_explicit_slope_parses() {
    let a = first_layer_activation(&wavenet_with_activation(
        r#"{"type":"LeakyReLU","negative_slope":0.1}"#,
    ));
    match a {
        ActivationSpec::Named {
            name,
            negative_slope: Some(s),
        } => {
            assert_eq!(name, "LeakyReLU");
            assert!((s - 0.1).abs() < 1e-6);
        }
        other => panic!("expected Named with slope, got {other:?}"),
    }
}

#[test]
fn activation_list_form_broadcasts_per_layer() {
    // A per-layer activation list (length == dilations.len()) now parses successfully.
    // `wavenet_with_activation` has one dilation, so a one-element list is valid.
    let a = first_layer_activation(&wavenet_with_activation(r#"["ReLU"]"#));
    assert!(
        matches!(a, ActivationSpec::Named { ref name, .. } if name == "ReLU"),
        "got {a:?}"
    );
}

#[test]
fn activation_dict_without_type_is_unsupported() {
    let a = first_layer_activation(&wavenet_with_activation(r#"{"negative_slope":0.01}"#));
    assert!(matches!(a, ActivationSpec::Unsupported(_)));
}

#[test]
fn activation_dict_non_numeric_slope_is_unsupported() {
    // A present-but-malformed `negative_slope` (here a string) must be rejected, not
    // silently treated as the 0.01 default — consistent with the crate's fail-loud
    // handling of unmodeled activation shapes.
    let a = first_layer_activation(&wavenet_with_activation(
        r#"{"type":"LeakyReLU","negative_slope":"0.1"}"#,
    ));
    assert!(matches!(a, ActivationSpec::Unsupported(_)));
}

#[test]
fn activation_dict_null_slope_uses_default() {
    // An explicit null slope means "no value" → runtime default, like an absent key.
    let a = first_layer_activation(&wavenet_with_activation(
        r#"{"type":"LeakyReLU","negative_slope":null}"#,
    ));
    assert!(
        matches!(a, ActivationSpec::Named { name, negative_slope: None } if name == "LeakyReLU")
    );
}

#[test]
fn parses_slimmable_container() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/slimmable_container.nam");
    let json = std::fs::read_to_string(path).expect("read container");
    let m = NamModel::from_json_str(&json).expect("parse container");
    assert_eq!(m.architecture, "SlimmableContainer");
    let cfg: &SlimmableConfig = match &m.config {
        ModelConfig::Slimmable(c) => c,
        other => panic!("expected Slimmable config, got {other:?}"),
    };
    assert_eq!(cfg.submodels.len(), 3);
    // max_values are ascending [0.33, 0.66, 1.0].
    let maxes: Vec<f32> = cfg.submodels.iter().map(|s| s.max_value).collect();
    assert!((maxes[0] - 0.33).abs() < 1e-6);
    assert!((maxes[2] - 1.0).abs() < 1e-6);
    // Submodels are mixed architecture: [LSTM, WaveNet, WaveNet].
    assert_eq!(cfg.submodels[0].model.architecture, "LSTM");
    assert_eq!(cfg.submodels[1].model.architecture, "WaveNet");
    assert_eq!(cfg.submodels[2].model.architecture, "WaveNet");
}

/// The real downloaded A2 captures (gitignored for licensing) must now PARSE
/// cleanly (the `missing field head_size` bug is fixed) and, until the forward-pass
/// phases land, be rejected at build time with a clear UnsupportedFeature — never a
/// parse error, never a panic. Skips when the files are absent (e.g. CI).
#[test]
fn real_a2_captures_parse_and_are_cleanly_guarded() {
    use nam_rs::{Error, Model, NamModel};
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/examples");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut checked = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("nam") {
            continue;
        }
        let json = std::fs::read_to_string(&path).unwrap();
        let model = NamModel::from_json_str(&json)
            .unwrap_or_else(|e| panic!("{:?} failed to PARSE: {e}", path.file_name().unwrap()));
        match Model::from_nam(&model) {
            Ok(_) | Err(Error::UnsupportedFeature(_)) => {}
            Err(other) => panic!(
                "{:?}: unexpected error {other:?}",
                path.file_name().unwrap()
            ),
        }
        checked += 1;
    }
    eprintln!("real A2 capture smoke: checked {checked} file(s)");
}

/// The typed [`Metadata`] schema must match what real files actually write. Every key
/// we claim to type has to survive the parse: a misspelled field name or a wrong Rust
/// type would otherwise be swallowed by the lenient parser into a silent `None`,
/// which no synthetic-JSON test would catch. Skips when the files are absent (CI).
#[test]
fn real_captures_metadata_matches_the_typed_schema() {
    use nam_rs::NamModel;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/examples");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut checked = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("nam") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let model = NamModel::from_json_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let md = model.metadata_typed();
        let Some(raw) = model.metadata.as_ref().and_then(|v| v.as_object()) else {
            continue;
        };

        // A string in the file must arrive as a string in the struct...
        for (key, got) in [
            ("name", &md.name),
            ("modeled_by", &md.modeled_by),
            ("gear_make", &md.gear_make),
            ("gear_model", &md.gear_model),
            ("gear_type", &md.gear_type),
            ("tone_type", &md.tone_type),
            ("trainer", &md.trainer),
        ] {
            if let Some(want) = raw.get(key).and_then(|v| v.as_str()) {
                assert_eq!(got.as_deref(), Some(want), "{name}: metadata.{key} lost");
            }
        }
        // ...a number as a number (real files write bare integers for the levels)...
        for (key, got) in [
            ("loudness", md.loudness),
            ("gain", md.gain),
            ("input_level_dbu", md.input_level_dbu),
            ("output_level_dbu", md.output_level_dbu),
        ] {
            if let Some(want) = raw.get(key).and_then(|v| v.as_f64()) {
                let got = got.unwrap_or_else(|| panic!("{name}: metadata.{key} lost"));
                assert!(
                    (f64::from(got) - want).abs() < 1e-3,
                    "{name}: metadata.{key} = {got}, file says {want}"
                );
            }
        }
        // ...and the two nested blocks as themselves. Check every date component: a
        // month/day transposition in the struct is exactly the slip that would survive
        // a spot-check of year alone.
        if let Some(want) = raw.get("date").and_then(|v| v.as_object()) {
            let date = md
                .date
                .unwrap_or_else(|| panic!("{name}: metadata.date lost"));
            let got = [
                i64::from(date.year),
                i64::from(date.month),
                i64::from(date.day),
                i64::from(date.hour),
                i64::from(date.minute),
                i64::from(date.second),
            ];
            for (component, got) in ["year", "month", "day", "hour", "minute", "second"]
                .into_iter()
                .zip(got)
            {
                assert_eq!(
                    got,
                    want[component].as_i64().unwrap(),
                    "{name}: metadata.date.{component}"
                );
            }
        }
        // `Option<Value>` maps an explicit JSON `null` to `None`, so compare against
        // "present and non-null" rather than mere key presence.
        let training_in_file = raw.get("training").is_some_and(|v| !v.is_null());
        assert_eq!(
            training_in_file,
            md.training.is_some(),
            "{name}: metadata.training lost"
        );

        eprintln!(
            "  {name}: gear_type={:?} includes_cab={:?}",
            md.gear_type,
            model.includes_cab()
        );
        checked += 1;
    }
    eprintln!("real capture metadata: checked {checked} file(s)");
}
