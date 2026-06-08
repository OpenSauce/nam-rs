//! Parsing of the on-disk `.nam` file format.
//!
//! A `.nam` file is a JSON object. The fields here mirror NAM's
//! `export_config()` / `export_weights()` output (see crate-level attribution).
//! WaveNet, LSTM, and SlimmableContainer architectures are parsed here (see
//! [`ModelConfig`]);
//! the runtime forward passes live in their own modules.

use serde::de::{self, Deserializer};
use serde::Deserialize;

use crate::error::Error;

/// How a layer-array's `activation` field was specified in the `.nam`.
///
/// NAM A1 writes a bare string (`"Tanh"`); A2 may write a dict
/// (`{"type": "LeakyReLU", "negative_slope": 0.01}`). A per-layer *list* (a
/// distinct activation per layer) is not modeled and is captured as
/// [`ActivationSpec::Unsupported`], which the runtime rejects with
/// [`crate::Error::UnsupportedFeature`] rather than silently mis-running.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationSpec {
    /// A single named activation, with an optional negative slope (LeakyReLU).
    Named {
        /// Activation name, e.g. `"Tanh"`, `"ReLU"`, `"LeakyReLU"`.
        name: String,
        /// LeakyReLU negative slope, if the file specified one. `None` → the
        /// runtime applies NAM's default of `0.01`.
        negative_slope: Option<f32>,
    },
    /// A shape this crate does not model (e.g. a per-layer activation list).
    Unsupported(serde_json::Value),
}

impl<'de> Deserialize<'de> for ActivationSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = serde_json::Value::deserialize(deserializer)?;
        Ok(match &v {
            serde_json::Value::String(s) => ActivationSpec::Named {
                name: s.clone(),
                negative_slope: None,
            },
            serde_json::Value::Object(map) => match map.get("type") {
                Some(serde_json::Value::String(t)) => match map.get("negative_slope") {
                    // Absent or explicit-null slope → runtime default (0.01).
                    None | Some(serde_json::Value::Null) => ActivationSpec::Named {
                        name: t.clone(),
                        negative_slope: None,
                    },
                    // Present and numeric → use it.
                    Some(slope) if slope.as_f64().is_some() => ActivationSpec::Named {
                        name: t.clone(),
                        negative_slope: slope.as_f64().map(|x| x as f32),
                    },
                    // Present but not a number → malformed; reject rather than silently
                    // defaulting (a corrupt/upstream-format error must not pass silently).
                    Some(_) => ActivationSpec::Unsupported(v.clone()),
                },
                _ => ActivationSpec::Unsupported(v),
            },
            _ => ActivationSpec::Unsupported(v),
        })
    }
}

/// Sample rate assumed when a `.nam` file omits the `sample_rate` field.
///
/// Matches NAM's documented default.
pub const DEFAULT_SAMPLE_RATE: f64 = 48_000.0;

/// A parsed `.nam` model file.
///
/// This is the *file representation* — the raw config + flat weight blob. To run
/// inference, build a [`crate::WaveNet`] from it.
#[derive(Debug, Clone)]
pub struct NamModel {
    /// `.nam` format version string (e.g. `"0.5.4"`).
    pub version: String,
    /// Model architecture, e.g. `"WaveNet"`.
    pub architecture: String,
    /// Architecture-specific configuration (dispatched on [`Self::architecture`]).
    pub config: ModelConfig,
    /// Flat weight blob. The final element is `head_scale` (see NAM
    /// `export_weights`). Stored as `f32` to match NAM Core's inference precision.
    pub weights: Vec<f32>,
    /// Training sample rate. Absent in older files; see [`Self::expected_sample_rate`].
    pub sample_rate: Option<f64>,
    /// Opaque training/gear metadata. Not used for inference.
    pub metadata: Option<serde_json::Value>,
}

/// LSTM configuration (NAM `_export_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct LstmConfig {
    /// Input width (1 for mono amp models).
    pub input_size: usize,
    /// Hidden state dimension `H`.
    pub hidden_size: usize,
    /// Number of stacked LSTM layers `L`.
    pub num_layers: usize,
}

/// One entry in a [`SlimmableConfig`]: a complete standalone submodel plus the
/// width-dial threshold at which it becomes active.
#[derive(Debug, Clone, Deserialize)]
pub struct SlimmableSubmodel {
    /// Upper width-dial value this submodel covers (NAM Core `max_value`).
    pub max_value: f32,
    /// The submodel itself — a full standalone `.nam` of any architecture.
    pub model: NamModel,
}

/// `SlimmableContainer` configuration: an ordered list of standalone submodels
/// selected at runtime by a width dial. The container holds no weights of its own.
#[derive(Debug, Clone, Deserialize)]
pub struct SlimmableConfig {
    /// Submodels in ascending `max_value` order; the last is the full-width model.
    pub submodels: Vec<SlimmableSubmodel>,
}

/// Architecture-specific configuration, tagged by `NamModel.architecture`.
#[derive(Debug, Clone)]
pub enum ModelConfig {
    /// WaveNet: a stack of dilated-convolution layer-arrays. Runnable via
    /// [`crate::WaveNet`].
    WaveNet(WaveNetConfig),
    /// LSTM: stacked recurrent layers plus a linear head.
    Lstm(LstmConfig),
    /// SlimmableContainer: a width-selectable set of standalone submodels.
    Slimmable(SlimmableConfig),
}

impl<'de> Deserialize<'de> for NamModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Parse the file shape with `config` left raw, then dispatch on
        // `architecture` to type it. This reads the sibling `architecture` field,
        // which `#[serde(deserialize_with)]` on a single field cannot do.
        #[derive(Deserialize)]
        struct Raw {
            version: String,
            architecture: String,
            config: serde_json::Value,
            weights: Vec<f32>,
            #[serde(default)]
            sample_rate: Option<f64>,
            #[serde(default)]
            metadata: Option<serde_json::Value>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let config = match raw.architecture.as_str() {
            "WaveNet" => {
                let raw_wn: RawWaveNetConfig =
                    serde_json::from_value(raw.config).map_err(de::Error::custom)?;
                ModelConfig::WaveNet(raw_wn.normalize().map_err(de::Error::custom)?)
            }
            "LSTM" => {
                ModelConfig::Lstm(serde_json::from_value(raw.config).map_err(de::Error::custom)?)
            }
            "SlimmableContainer" => ModelConfig::Slimmable(
                serde_json::from_value(raw.config).map_err(de::Error::custom)?,
            ),
            other => {
                return Err(de::Error::custom(format!(
                    "unsupported model architecture: {other:?}"
                )))
            }
        };

        Ok(NamModel {
            version: raw.version,
            architecture: raw.architecture,
            config,
            weights: raw.weights,
            sample_rate: raw.sample_rate,
            metadata: raw.metadata,
        })
    }
}

/// Loudness/level-calibration fields NAM may write into `metadata`. All optional;
/// older or minimal files omit them. Unknown metadata keys are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Metadata {
    /// Perceived loudness of the model's output, in LUFS (NAM's `loudness`).
    #[serde(default)]
    pub loudness: Option<f32>,
    /// Analog level (dBu) corresponding to 0 dBFS at the model input.
    #[serde(default)]
    pub input_level_dbu: Option<f32>,
    /// Analog level (dBu) corresponding to 0 dBFS at the model output.
    #[serde(default)]
    pub output_level_dbu: Option<f32>,
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

    /// The sample rate, in Hz, the model expects its input to be at — falling back
    /// to [`DEFAULT_SAMPLE_RATE`] when the file does not specify one.
    ///
    /// **You must feed the model audio at this rate.** `nam-rs` runs the forward pass
    /// at whatever rate you hand it and does *not* resample. A model captured at one
    /// rate fed audio at another produces silently wrong output: its dilations and
    /// recurrence are defined in samples, not seconds. If your host runs at a
    /// different rate, resample to this rate before [`crate::Model::process_buffer`]
    /// and back afterwards — resampling is the caller's responsibility. Mirrors NAM
    /// Core's `GetExpectedSampleRate()`.
    #[must_use]
    pub fn expected_sample_rate(&self) -> f64 {
        self.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE)
    }

    /// The typed [`Metadata`] (loudness + calibration levels), parsed from the raw
    /// `metadata` block in one shot. Returns defaults (all `None`) when there is no
    /// metadata block or it lacks these keys; unknown keys are ignored.
    ///
    /// Prefer this over the single-field accessors ([`Self::loudness`], etc.) when you
    /// want several fields: each single-field accessor re-clones and re-parses the raw
    /// JSON, whereas this parses once. (All are cold-path / load-time, so neither is on
    /// the audio thread.)
    #[must_use]
    pub fn metadata_typed(&self) -> Metadata {
        match &self.metadata {
            Some(v) => serde_json::from_value(v.clone()).unwrap_or_default(),
            None => Metadata::default(),
        }
    }

    /// Output loudness in LUFS, if the file records it.
    #[must_use]
    pub fn loudness(&self) -> Option<f32> {
        self.metadata_typed().loudness
    }

    /// Input calibration level in dBu (analog level at 0 dBFS in), if present.
    #[must_use]
    pub fn input_level_dbu(&self) -> Option<f32> {
        self.metadata_typed().input_level_dbu
    }

    /// Output calibration level in dBu (analog level at 0 dBFS out), if present.
    #[must_use]
    pub fn output_level_dbu(&self) -> Option<f32> {
        self.metadata_typed().output_level_dbu
    }
}

/// Activation gating mode for a WaveNet layer (NAMCore `GatingMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingMode {
    /// No gating: `out = activation(z)`.
    None,
    /// Gated: `out = primary(z_a) * secondary(z_b)` (classic `tanh*sigmoid`).
    Gated,
    /// Blended: `out = α·primary(z_a) + (1-α)·z_a`, `α = secondary(z_b)`.
    Blended,
}

impl GatingMode {
    /// Parse a NAMCore gating-mode name (`"none"`/`"gated"`/`"blended"`).
    pub(crate) fn from_name(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "gated" => Ok(Self::Gated),
            "blended" => Ok(Self::Blended),
            other => Err(format!("unknown gating_mode: {other:?}")),
        }
    }
}

/// A layer's residual 1×1 (`layer1x1`): maps the activated bottleneck back to
/// `channels`. Active by default (the A1 `_1x1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layer1x1Config {
    /// Whether the 1×1 is present (inactive ⇒ identity residual, needs bottleneck==channels).
    pub active: bool,
    /// Grouped-conv group count (1 = dense).
    pub groups: usize,
}

impl Layer1x1Config {
    pub(crate) fn from_json(v: Option<&serde_json::Value>) -> Self {
        match v {
            None => Self {
                active: true,
                groups: 1,
            },
            Some(o) => Self {
                active: o.get("active").and_then(|x| x.as_bool()).unwrap_or(true),
                groups: o
                    .get("groups")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .unwrap_or(1),
            },
        }
    }
}

/// A layer's head 1×1 (`head1x1`): an optional 1×1 producing this layer's head
/// contribution. Inactive by default (then the head contribution is the activated
/// bottleneck directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head1x1Config {
    /// Whether the head 1×1 is present.
    pub active: bool,
    /// Output channels (defaults to `channels` when active and unspecified).
    pub out_channels: Option<usize>,
    /// Grouped-conv group count.
    pub groups: usize,
}

impl Head1x1Config {
    pub(crate) fn from_json(v: Option<&serde_json::Value>) -> Self {
        match v {
            None => Self {
                active: false,
                out_channels: None,
                groups: 1,
            },
            Some(o) => Self {
                active: o.get("active").and_then(|x| x.as_bool()).unwrap_or(false),
                out_channels: o
                    .get("out_channels")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize),
                groups: o
                    .get("groups")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .unwrap_or(1),
            },
        }
    }
}

/// One FiLM block (`*_pre_film` / `*_post_film`): conditions a scale (+ optional
/// shift) from the conditioning signal. Absent or `false` ⇒ inactive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilmConfig {
    /// Whether this FiLM site is applied.
    pub active: bool,
    /// Whether it adds a shift term (else scale-only).
    pub shift: bool,
    /// Grouped-conv group count for the conditioning 1×1.
    pub groups: usize,
}

impl FilmConfig {
    /// The inactive default (absent key or explicit `false`).
    pub const INACTIVE: Self = Self {
        active: false,
        shift: false,
        groups: 1,
    };

    pub(crate) fn from_json(v: Option<&serde_json::Value>) -> Self {
        match v {
            None => Self::INACTIVE,
            Some(serde_json::Value::Bool(false)) => Self::INACTIVE,
            Some(o) => Self {
                active: o.get("active").and_then(|x| x.as_bool()).unwrap_or(true),
                shift: o.get("shift").and_then(|x| x.as_bool()).unwrap_or(true),
                groups: o
                    .get("groups")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .unwrap_or(1),
            },
        }
    }
}

/// Post-stack head (`config.head`): a stack of `activation → Conv1D` applied after
/// the layer-arrays. `None` for A1 / current A2 defaults.
#[derive(Debug, Clone)]
pub struct PostStackHeadConfig {
    /// Hidden channel count between head convs.
    pub channels: usize,
    /// Final output channels.
    pub out_channels: usize,
    /// Per-conv kernel sizes (one conv per entry).
    pub kernel_sizes: Vec<usize>,
    /// Activation applied before each head conv.
    pub activation: ActivationSpec,
}

/// WaveNet configuration: layer-arrays, optional post-stack head + condition DSP,
/// and the output scale. Per-layer quantities are normalized into `Vec`s.
#[derive(Debug, Clone)]
pub struct WaveNetConfig {
    /// One config per layer-array.
    pub layers: Vec<LayerArrayConfig>,
    /// Optional post-stack head (`config.head`).
    pub post_stack_head: Option<PostStackHeadConfig>,
    /// Output gain (note: the runtime value is the trailing weight).
    pub head_scale: f32,
    /// Input channels (default 1).
    pub in_channels: usize,
    /// Optional nested conditioning DSP.
    pub condition_dsp: Option<Box<NamModel>>,
}

#[derive(serde::Deserialize)]
struct RawWaveNetConfig {
    layers: Vec<RawLayerArrayConfig>,
    #[serde(default)]
    head: Option<serde_json::Value>,
    head_scale: f32,
    #[serde(default)]
    in_channels: Option<usize>,
    #[serde(default)]
    condition_dsp: Option<serde_json::Value>,
}

impl RawWaveNetConfig {
    fn normalize(self) -> Result<WaveNetConfig, String> {
        let layers = self
            .layers
            .into_iter()
            .map(RawLayerArrayConfig::normalize)
            .collect::<Result<Vec<_>, _>>()?;

        let post_stack_head = match self.head {
            Some(h) if !h.is_null() => {
                let channels =
                    h.get("channels")
                        .and_then(|x| x.as_u64())
                        .ok_or("post-stack head missing channels")? as usize;
                let out_channels = h
                    .get("out_channels")
                    .and_then(|x| x.as_u64())
                    .ok_or("post-stack head missing out_channels")?
                    as usize;
                let kernel_sizes: Vec<usize> = h
                    .get("kernel_sizes")
                    .and_then(|x| x.as_array())
                    .ok_or("post-stack head missing kernel_sizes")?
                    .iter()
                    .map(|k| {
                        k.as_u64()
                            .map(|v| v as usize)
                            .ok_or("kernel_sizes entry not an int".to_string())
                    })
                    .collect::<Result<_, _>>()?;
                let activation = serde_json::from_value::<ActivationSpec>(
                    h.get("activation")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                )
                .map_err(|e| e.to_string())?;
                Some(PostStackHeadConfig {
                    channels,
                    out_channels,
                    kernel_sizes,
                    activation,
                })
            }
            _ => None,
        };

        let condition_dsp = match self.condition_dsp {
            Some(v) if !v.is_null() => {
                let m = serde_json::from_value::<NamModel>(v).map_err(|e| e.to_string())?;
                Some(Box::new(m))
            }
            _ => None,
        };

        Ok(WaveNetConfig {
            layers,
            post_stack_head,
            head_scale: self.head_scale,
            in_channels: self.in_channels.unwrap_or(1),
            condition_dsp,
        })
    }
}

/// Configuration for one WaveNet layer-array, normalized so every per-layer
/// quantity is a `Vec` of length `dilations.len()`. Built from the on-disk JSON by
/// [`RawLayerArrayConfig::normalize`]; A1 files fill the A2 fields with defaults.
#[derive(Debug, Clone)]
pub struct LayerArrayConfig {
    /// Input channels into the array (1 for the first array).
    pub input_size: usize,
    /// Conditioning signal width.
    pub condition_size: usize,
    /// Hidden channel count between layers.
    pub channels: usize,
    /// Internal per-layer width (defaults to `channels`).
    pub bottleneck: usize,
    /// Per-layer dilation factors; its length defines the number of layers.
    pub dilations: Vec<usize>,
    /// Per-layer dilated-conv kernel sizes (length == `dilations.len()`).
    pub kernel_sizes: Vec<usize>,
    /// Per-layer primary activations (length == `dilations.len()`).
    pub activations: Vec<ActivationSpec>,
    /// Per-layer gating modes (length == `dilations.len()`).
    pub gating_modes: Vec<GatingMode>,
    /// Per-layer secondary activations (for gating); element may be the default
    /// (a `Named{"Sigmoid"}`) where unspecified. Length == `dilations.len()`.
    pub secondary_activations: Vec<ActivationSpec>,
    /// Grouped-conv groups for the dilated conv.
    pub groups_input: usize,
    /// Grouped-conv groups for the input mixer.
    pub groups_input_mixin: usize,
    /// Head rechannel output width.
    pub head_size: usize,
    /// Head rechannel kernel size (1 for A1; e.g. 16 for A2 conv heads).
    pub head_kernel_size: usize,
    /// Whether the head rechannel has a bias.
    pub head_bias: bool,
    /// Residual 1×1 config.
    pub layer1x1: Layer1x1Config,
    /// Head 1×1 config.
    pub head1x1: Head1x1Config,
    /// FiLM: applied to the layer input before the dilated conv.
    pub conv_pre_film: FilmConfig,
    /// FiLM: applied to the dilated-conv output.
    pub conv_post_film: FilmConfig,
    /// FiLM: applied to the conditioning before the input mixer.
    pub input_mixin_pre_film: FilmConfig,
    /// FiLM: applied to the input-mixer output.
    pub input_mixin_post_film: FilmConfig,
    /// FiLM: applied to the conv+mixin sum before activation.
    pub activation_pre_film: FilmConfig,
    /// FiLM: applied to the activation output.
    pub activation_post_film: FilmConfig,
    /// FiLM: applied to the layer1x1 output (BLENDED branch only, per NAMCore).
    pub layer1x1_post_film: FilmConfig,
    /// FiLM: applied to the head1x1 output.
    pub head1x1_post_film: FilmConfig,
}

/// On-disk shape of a layer-array config: optional / either-or fields exactly as
/// NAM writes them. Converted to [`LayerArrayConfig`] by [`Self::normalize`].
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct RawLayerArrayConfig {
    input_size: usize,
    condition_size: usize,
    channels: usize,
    #[serde(default)]
    bottleneck: Option<usize>,
    dilations: Vec<usize>,
    #[serde(default)]
    kernel_size: Option<usize>,
    #[serde(default)]
    kernel_sizes: Option<Vec<usize>>,
    activation: serde_json::Value,
    #[serde(default)]
    gating_mode: Option<serde_json::Value>,
    #[serde(default)]
    gated: Option<bool>,
    #[serde(default)]
    secondary_activation: Option<serde_json::Value>,
    #[serde(default)]
    groups_input: Option<usize>,
    #[serde(default)]
    groups_input_mixin: Option<usize>,
    #[serde(default)]
    head: Option<serde_json::Value>,
    #[serde(default)]
    head_size: Option<usize>,
    #[serde(default)]
    head_bias: Option<bool>,
    #[serde(default)]
    layer1x1: Option<serde_json::Value>,
    #[serde(default)]
    head1x1: Option<serde_json::Value>,
    #[serde(default)]
    conv_pre_film: Option<serde_json::Value>,
    #[serde(default)]
    conv_post_film: Option<serde_json::Value>,
    #[serde(default)]
    input_mixin_pre_film: Option<serde_json::Value>,
    #[serde(default)]
    input_mixin_post_film: Option<serde_json::Value>,
    #[serde(default)]
    activation_pre_film: Option<serde_json::Value>,
    #[serde(default)]
    activation_post_film: Option<serde_json::Value>,
    #[serde(default)]
    layer1x1_post_film: Option<serde_json::Value>,
    #[serde(default)]
    head1x1_post_film: Option<serde_json::Value>,
}

impl RawLayerArrayConfig {
    pub(crate) fn normalize(self) -> Result<LayerArrayConfig, String> {
        let n = self.dilations.len();
        if n == 0 {
            return Err("layer-array has no dilations".into());
        }

        let kernel_sizes = match (self.kernel_size, self.kernel_sizes) {
            (Some(_), Some(_)) => {
                return Err("layer-array specifies both kernel_size and kernel_sizes".into())
            }
            (Some(k), None) => vec![k; n],
            (None, Some(ks)) => {
                if ks.len() != n {
                    return Err(format!(
                        "kernel_sizes length {} != number of layers {n}",
                        ks.len()
                    ));
                }
                ks
            }
            (None, None) => {
                return Err("layer-array specifies neither kernel_size nor kernel_sizes".into())
            }
        };

        let activations = broadcast_activations(&self.activation, n)?;

        let gating_modes = match (&self.gating_mode, self.gated) {
            (Some(v), _) => broadcast_gating(v, n)?,
            (None, Some(true)) => vec![GatingMode::Gated; n],
            (None, _) => vec![GatingMode::None; n],
        };

        let secondary_activations = match &self.secondary_activation {
            Some(v) => broadcast_secondary(v, n)?,
            None => vec![default_sigmoid(); n],
        };

        let (head_size, head_kernel_size, head_bias) = match &self.head {
            Some(h) if !h.is_null() => {
                let out = h
                    .get("out_channels")
                    .and_then(|x| x.as_u64())
                    .ok_or("layer head missing out_channels")? as usize;
                let k = h
                    .get("kernel_size")
                    .and_then(|x| x.as_u64())
                    .ok_or("layer head missing kernel_size")? as usize;
                let bias = h.get("bias").and_then(|x| x.as_bool()).unwrap_or(true);
                (out, k, bias)
            }
            _ => {
                let hs = self
                    .head_size
                    .ok_or("layer-array missing head_size (and no head object)")?;
                (hs, 1, self.head_bias.unwrap_or(false))
            }
        };

        Ok(LayerArrayConfig {
            input_size: self.input_size,
            condition_size: self.condition_size,
            channels: self.channels,
            bottleneck: self.bottleneck.unwrap_or(self.channels),
            dilations: self.dilations,
            kernel_sizes,
            activations,
            gating_modes,
            secondary_activations,
            groups_input: self.groups_input.unwrap_or(1),
            groups_input_mixin: self.groups_input_mixin.unwrap_or(1),
            head_size,
            head_kernel_size,
            head_bias,
            layer1x1: Layer1x1Config::from_json(self.layer1x1.as_ref()),
            head1x1: Head1x1Config::from_json(self.head1x1.as_ref()),
            conv_pre_film: FilmConfig::from_json(self.conv_pre_film.as_ref()),
            conv_post_film: FilmConfig::from_json(self.conv_post_film.as_ref()),
            input_mixin_pre_film: FilmConfig::from_json(self.input_mixin_pre_film.as_ref()),
            input_mixin_post_film: FilmConfig::from_json(self.input_mixin_post_film.as_ref()),
            activation_pre_film: FilmConfig::from_json(self.activation_pre_film.as_ref()),
            activation_post_film: FilmConfig::from_json(self.activation_post_film.as_ref()),
            layer1x1_post_film: FilmConfig::from_json(self.layer1x1_post_film.as_ref()),
            head1x1_post_film: FilmConfig::from_json(self.head1x1_post_film.as_ref()),
        })
    }
}

/// A `Named{"Sigmoid"}` activation, the gating secondary default.
fn default_sigmoid() -> ActivationSpec {
    ActivationSpec::Named {
        name: "Sigmoid".into(),
        negative_slope: None,
    }
}

/// Broadcast a single activation or expand a per-layer list to length `n`.
fn broadcast_activations(v: &serde_json::Value, n: usize) -> Result<Vec<ActivationSpec>, String> {
    match v {
        serde_json::Value::Array(items) => {
            if items.len() != n {
                return Err(format!(
                    "activation list length {} != number of layers {n}",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|e| {
                    serde_json::from_value::<ActivationSpec>(e.clone()).map_err(|e| e.to_string())
                })
                .collect()
        }
        other => {
            let a = serde_json::from_value::<ActivationSpec>(other.clone())
                .map_err(|e| e.to_string())?;
            Ok(vec![a; n])
        }
    }
}

/// Broadcast/expand `secondary_activation`; JSON `null` elements become the
/// Sigmoid default (NAMCore's default secondary).
fn broadcast_secondary(v: &serde_json::Value, n: usize) -> Result<Vec<ActivationSpec>, String> {
    let one = |e: &serde_json::Value| -> Result<ActivationSpec, String> {
        if e.is_null() {
            Ok(default_sigmoid())
        } else {
            serde_json::from_value::<ActivationSpec>(e.clone()).map_err(|e| e.to_string())
        }
    };
    match v {
        serde_json::Value::Array(items) => {
            if items.len() != n {
                return Err(format!(
                    "secondary_activation list length {} != {n}",
                    items.len()
                ));
            }
            items.iter().map(one).collect()
        }
        other => Ok(vec![one(other)?; n]),
    }
}

/// Broadcast a single gating name or expand a per-layer list to length `n`.
fn broadcast_gating(v: &serde_json::Value, n: usize) -> Result<Vec<GatingMode>, String> {
    match v {
        serde_json::Value::String(s) => {
            let g = GatingMode::from_name(s)?;
            Ok(vec![g; n])
        }
        serde_json::Value::Array(items) => {
            if items.len() != n {
                return Err(format!(
                    "gating_mode list length {} != number of layers {n}",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|e| {
                    e.as_str()
                        .ok_or_else(|| "gating_mode entry is not a string".to_string())
                        .and_then(GatingMode::from_name)
                })
                .collect()
        }
        _ => Err("gating_mode is neither a string nor a list".into()),
    }
}

#[cfg(test)]
mod layer_array_normalize_tests {
    use super::*;

    fn norm(v: serde_json::Value) -> LayerArrayConfig {
        let raw: RawLayerArrayConfig = serde_json::from_value(v).unwrap();
        raw.normalize().unwrap()
    }

    #[test]
    fn a1_layer_broadcasts_scalar_kernel_and_string_activation() {
        let la = norm(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 2, "head_size": 1,
            "kernel_size": 3, "dilations": [1, 2, 4], "activation": "Tanh",
            "gated": false, "head_bias": false
        }));
        assert_eq!(la.channels, 2);
        assert_eq!(la.bottleneck, 2);
        assert_eq!(la.kernel_sizes, vec![3, 3, 3]);
        assert_eq!(la.gating_modes, vec![GatingMode::None; 3]);
        assert_eq!(la.head_size, 1);
        assert_eq!(la.head_kernel_size, 1);
        assert!(!la.head_bias);
        assert!(la.layer1x1.active);
        assert!(!la.head1x1.active);
        assert_eq!(la.groups_input, 1);
        assert_eq!(la.activations.len(), 3);
        assert!(matches!(&la.activations[0], ActivationSpec::Named { name, .. } if name == "Tanh"));
        let g = norm(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 2, "head_size": 1,
            "kernel_size": 3, "dilations": [1], "activation": "Tanh",
            "gated": true, "head_bias": true
        }));
        assert_eq!(g.gating_modes, vec![GatingMode::Gated]);
    }

    #[test]
    fn a2_flexible_layer_parses_per_layer_vectors_and_nested_head() {
        let la = norm(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 3, "bottleneck": 3,
            "dilations": [1, 3, 7],
            "kernel_sizes": [6, 6, 15],
            "activation": [
                {"type": "LeakyReLU", "negative_slope": 0.01},
                {"type": "LeakyReLU", "negative_slope": 0.01},
                {"type": "LeakyReLU", "negative_slope": 0.01}
            ],
            "head": {"out_channels": 1, "kernel_size": 16, "bias": true},
            "head1x1": {"active": false, "out_channels": 1, "groups": 1},
            "layer1x1": {"active": true, "groups": 1},
            "groups_input": 1, "groups_input_mixin": 1,
            "gating_mode": ["none", "none", "none"],
            "secondary_activation": [null, null, null],
            "conv_pre_film": {"active": false, "shift": true, "groups": 1}
        }));
        assert_eq!(la.kernel_sizes, vec![6, 6, 15]);
        assert_eq!(la.gating_modes, vec![GatingMode::None; 3]);
        assert_eq!(la.head_size, 1);
        assert_eq!(la.head_kernel_size, 16);
        assert!(la.head_bias);
        assert_eq!(la.bottleneck, 3);
        assert_eq!(la.activations.len(), 3);
        assert!(!la.conv_pre_film.active);
    }

    #[test]
    fn both_kernel_forms_is_an_error() {
        let raw: RawLayerArrayConfig = serde_json::from_value(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
            "kernel_size": 3, "kernel_sizes": [3], "dilations": [1],
            "activation": "Tanh", "gated": false, "head_bias": false
        }))
        .unwrap();
        assert!(raw.normalize().is_err());
    }

    #[test]
    fn kernel_sizes_length_mismatch_is_an_error() {
        let raw: RawLayerArrayConfig = serde_json::from_value(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
            "kernel_sizes": [3, 3], "dilations": [1],
            "activation": "Tanh", "gated": false, "head_bias": false
        }))
        .unwrap();
        assert!(raw.normalize().is_err());
    }

    #[test]
    fn activation_list_length_mismatch_is_an_error() {
        let raw: RawLayerArrayConfig = serde_json::from_value(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
            "kernel_size": 3, "dilations": [1, 2],
            "activation": ["Tanh"], "gated": false, "head_bias": false
        }))
        .unwrap();
        assert!(raw.normalize().is_err());
    }
}

#[cfg(test)]
mod a2_subconfig_tests {
    use super::*;

    #[test]
    fn gating_mode_from_str() {
        assert_eq!(GatingMode::from_name("none").unwrap(), GatingMode::None);
        assert_eq!(GatingMode::from_name("gated").unwrap(), GatingMode::Gated);
        assert_eq!(
            GatingMode::from_name("blended").unwrap(),
            GatingMode::Blended
        );
        assert!(GatingMode::from_name("wat").is_err());
    }

    #[test]
    fn film_absent_or_false_is_inactive() {
        assert_eq!(FilmConfig::from_json(None), FilmConfig::INACTIVE);
        assert_eq!(
            FilmConfig::from_json(Some(&serde_json::json!(false))),
            FilmConfig::INACTIVE
        );
    }

    #[test]
    fn film_object_defaults_active_shift_groups() {
        let v = serde_json::json!({});
        let f = FilmConfig::from_json(Some(&v));
        assert_eq!(
            f,
            FilmConfig {
                active: true,
                shift: true,
                groups: 1
            }
        );
        let v = serde_json::json!({"active": false, "shift": false, "groups": 2});
        assert_eq!(
            FilmConfig::from_json(Some(&v)),
            FilmConfig {
                active: false,
                shift: false,
                groups: 2
            }
        );
    }

    #[test]
    fn layer1x1_defaults_active_true_groups_1() {
        assert_eq!(
            Layer1x1Config::from_json(None),
            Layer1x1Config {
                active: true,
                groups: 1
            }
        );
        let v = serde_json::json!({"active": true, "groups": 1});
        assert_eq!(
            Layer1x1Config::from_json(Some(&v)),
            Layer1x1Config {
                active: true,
                groups: 1
            }
        );
    }

    #[test]
    fn head1x1_defaults_inactive() {
        let h = Head1x1Config::from_json(None);
        assert_eq!(
            h,
            Head1x1Config {
                active: false,
                out_channels: None,
                groups: 1
            }
        );
        let v = serde_json::json!({"active": false, "out_channels": 1, "groups": 1});
        assert_eq!(
            Head1x1Config::from_json(Some(&v)),
            Head1x1Config {
                active: false,
                out_channels: Some(1),
                groups: 1
            }
        );
    }
}

#[cfg(test)]
mod wavenet_config_tests {
    use super::*;

    fn parse(json: &str) -> WaveNetConfig {
        match NamModel::from_json_str(json).unwrap().config {
            ModelConfig::WaveNet(c) => c,
            other => panic!("expected WaveNet, got {other:?}"),
        }
    }

    #[test]
    fn a1_config_parses_unchanged() {
        let c = parse(
            r#"{
            "version":"0.5.4","architecture":"WaveNet","config":{
                "layers":[{"input_size":1,"condition_size":1,"channels":2,"head_size":1,
                    "kernel_size":3,"dilations":[1,2],"activation":"Tanh",
                    "gated":false,"head_bias":false}],
                "head":null,"head_scale":2.0},
            "weights":[]}"#,
        );
        assert_eq!(c.layers.len(), 1);
        assert_eq!(c.head_scale, 2.0);
        assert!(c.post_stack_head.is_none());
        assert!(c.condition_dsp.is_none());
        assert_eq!(c.layers[0].kernel_sizes, vec![3, 3]);
    }

    #[test]
    fn a2_flexible_container_submodel_config_parses() {
        let c = parse(
            r#"{
            "version":"0.7.0","architecture":"WaveNet","config":{
                "layers":[{"input_size":1,"condition_size":1,"channels":3,"bottleneck":3,
                    "dilations":[1,3,7],"kernel_sizes":[6,6,15],
                    "activation":[{"type":"LeakyReLU"},{"type":"LeakyReLU"},{"type":"LeakyReLU"}],
                    "head":{"out_channels":1,"kernel_size":16,"bias":true},
                    "head1x1":{"active":false},"layer1x1":{"active":true,"groups":1},
                    "gating_mode":["none","none","none"]}],
                "head":null,"head_scale":0.5},
            "weights":[]}"#,
        );
        assert_eq!(c.layers[0].head_kernel_size, 16);
        assert_eq!(c.layers[0].kernel_sizes, vec![6, 6, 15]);
        assert!(c.post_stack_head.is_none());
    }

    #[test]
    fn post_stack_head_parses() {
        let c = parse(
            r#"{
            "version":"0.6.0","architecture":"WaveNet","config":{
                "layers":[{"input_size":1,"condition_size":1,"channels":2,"head_size":2,
                    "kernel_size":3,"dilations":[1],"activation":"Tanh",
                    "gated":false,"head_bias":false}],
                "head":{"channels":4,"out_channels":1,"kernel_sizes":[1,1],"activation":"ReLU"},
                "head_scale":1.0},
            "weights":[]}"#,
        );
        let h = c.post_stack_head.expect("post-stack head present");
        assert_eq!(h.channels, 4);
        assert_eq!(h.out_channels, 1);
        assert_eq!(h.kernel_sizes, vec![1, 1]);
    }

    #[test]
    fn condition_dsp_parses_as_nested_model() {
        let c = parse(
            r#"{
            "version":"0.6.0","architecture":"WaveNet","config":{
                "layers":[{"input_size":1,"condition_size":1,"channels":2,"head_size":1,
                    "kernel_size":3,"dilations":[1],"activation":"Tanh",
                    "gated":false,"head_bias":false}],
                "head":null,"head_scale":1.0,
                "condition_dsp":{"version":"0.5.4","architecture":"WaveNet","config":{
                    "layers":[{"input_size":1,"condition_size":1,"channels":1,"head_size":1,
                        "kernel_size":1,"dilations":[1],"activation":"Tanh",
                        "gated":false,"head_bias":false}],
                    "head":null,"head_scale":1.0},"weights":[]}},
            "weights":[]}"#,
        );
        let dsp = c.condition_dsp.expect("condition_dsp present");
        assert_eq!(dsp.architecture, "WaveNet");
    }
}
