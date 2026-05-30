//! Parsing of the on-disk `.nam` file format.
//!
//! A `.nam` file is a JSON object. The fields here mirror NAM's
//! `export_config()` / `export_weights()` output (see crate-level attribution).
//! Only the WaveNet architecture is modelled for now; LSTM support is future work.

use serde::Deserialize;

use crate::error::Error;

/// Sample rate assumed when a `.nam` file omits the `sample_rate` field.
///
/// Matches NAM's documented default.
pub const DEFAULT_SAMPLE_RATE: f64 = 48_000.0;

/// A parsed `.nam` model file.
///
/// This is the *file representation* — the raw config + flat weight blob. To run
/// inference, build a [`crate::WaveNet`] from it.
#[derive(Debug, Clone, Deserialize)]
pub struct NamModel {
    /// `.nam` format version string (e.g. `"0.5.4"`).
    pub version: String,
    /// Model architecture, e.g. `"WaveNet"`.
    pub architecture: String,
    /// Architecture-specific configuration.
    pub config: WaveNetConfig,
    /// Flat weight blob. The final element is `head_scale` (see NAM
    /// `export_weights`). Stored as `f32` to match NAM Core's inference precision.
    pub weights: Vec<f32>,
    /// Training sample rate. Absent in older files; see [`Self::sample_rate`].
    #[serde(default)]
    pub sample_rate: Option<f64>,
    /// Opaque training/gear metadata. Not used for inference.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

impl NamModel {
    /// Read and parse a `.nam` model from a file on disk.
    ///
    /// Convenience over [`std::fs::read_to_string`] + [`Self::from_json_str`].
    /// Returns [`Error::Io`] if the file can't be read, or [`Error::Json`] if its
    /// contents aren't valid `.nam` JSON.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    /// Parse a `.nam` model from a JSON string already in memory.
    pub fn from_json_str(json: &str) -> Result<Self, Error> {
        Ok(serde_json::from_str(json)?)
    }

    /// The model's sample rate, falling back to [`DEFAULT_SAMPLE_RATE`] when the
    /// file does not specify one.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE)
    }
}

/// WaveNet configuration: a sequence of layer-arrays plus a final output scale.
#[derive(Debug, Clone, Deserialize)]
pub struct WaveNetConfig {
    /// One config per layer-array (NAM standard models have two).
    pub layers: Vec<LayerArrayConfig>,
    /// Optional separate head. `null` in standard models.
    #[serde(default)]
    pub head: Option<serde_json::Value>,
    /// Output gain applied after the head.
    pub head_scale: f32,
}

/// Configuration for a single WaveNet layer-array (a stack of dilated layers
/// sharing channel/kernel parameters).
#[derive(Debug, Clone, Deserialize)]
pub struct LayerArrayConfig {
    /// Number of input channels into the array (1 for the first array).
    pub input_size: usize,
    /// Conditioning signal width (1 for standard amp models).
    pub condition_size: usize,
    /// Hidden channel count.
    pub channels: usize,
    /// Output channels of each layer's head 1x1.
    pub head_size: usize,
    /// Dilated-convolution kernel size (typically 3).
    pub kernel_size: usize,
    /// Per-layer dilation factors, e.g. `[1, 2, 4, ..., 512]`.
    pub dilations: Vec<usize>,
    /// Activation function name, e.g. `"Tanh"`.
    pub activation: String,
    /// Whether the layer uses a gated activation (`tanh * sigmoid`).
    pub gated: bool,
    /// Whether the head 1x1 has a bias term.
    pub head_bias: bool,
}
