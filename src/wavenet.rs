//! Real-time WaveNet inference.
//!
//! [`WaveNet`] is built once from a parsed [`NamModel`] (which may allocate), then
//! run on the audio thread via [`WaveNet::process_buffer`], which never allocates.
//! All scratch buffers are pre-allocated in [`WaveNet::new`].
//!
//! The forward pass is a port of NAM's WaveNet, built bottom-up from the `conv`,
//! `layer`, and `array` submodules (each unit-tested) and validated end-to-end
//! against the reference in `tests/parity.rs`.

use crate::error::Error;
use crate::model::{LayerArrayConfig, NamModel, WaveNetConfig};

mod array;
mod conv;
mod layer;

use array::LayerArray;
use layer::{Activation, Layer};

/// The only architecture this crate runs.
const ARCHITECTURE: &str = "WaveNet";

/// A ready-to-run WaveNet, with all scratch buffers pre-allocated.
#[derive(Debug)]
pub struct WaveNet {
    arrays: Vec<LayerArray>,
    head_scale: f32,
    /// Samples of input history the deepest dilated tap reaches back over; equals
    /// the model's warmup length / processing latency in samples.
    receptive_field: usize,
    /// Channel width of the first array (its incoming head is silence this wide).
    channels0: usize,
    /// Head signal carried between arrays (two buffers, ping-ponged).
    head_a: Vec<f32>,
    head_b: Vec<f32>,
    /// Layer signal carried between arrays (two buffers, ping-ponged).
    sig_a: Vec<f32>,
    sig_b: Vec<f32>,
}

impl WaveNet {
    /// Build a runnable model from a parsed `.nam` file.
    ///
    /// All allocation happens here. Fails if the architecture is unsupported, an
    /// activation is unknown, or the flat weight blob does not match the config.
    pub fn new(model: &NamModel) -> Result<Self, Error> {
        if model.architecture != ARCHITECTURE {
            return Err(Error::UnsupportedArchitecture(model.architecture.clone()));
        }
        let cfg = &model.config;

        let expected = expected_weight_count(cfg);
        if expected != model.weights.len() {
            return Err(Error::WeightCountMismatch {
                expected,
                found: model.weights.len(),
            });
        }

        let mut r = Reader::new(&model.weights);
        let mut arrays = Vec::with_capacity(cfg.layers.len());
        for la in &cfg.layers {
            arrays.push(build_array(&mut r, la)?);
        }
        let head_scale = r.take(1)[0];

        let max_ch = arrays.iter().map(LayerArray::channels).max().unwrap_or(1);
        let max_head = arrays.iter().map(LayerArray::head_size).max().unwrap_or(1);
        let head_w = max_ch.max(max_head).max(1);
        let sig_w = max_ch.max(1);
        let channels0 = arrays.first().map_or(0, LayerArray::channels);

        Ok(Self {
            arrays,
            head_scale,
            receptive_field: receptive_field(cfg),
            channels0,
            head_a: vec![0.0; head_w],
            head_b: vec![0.0; head_w],
            sig_a: vec![0.0; sig_w],
            sig_b: vec![0.0; sig_w],
        })
    }

    /// Receptive field in samples: how far back the deepest dilated tap reaches.
    ///
    /// This is the model's warmup length and its processing latency. The first
    /// `receptive_field()` output samples of a fresh (or freshly [`reset`](Self::reset))
    /// model are a startup transient computed against zero-filled history, so they
    /// reflect the streaming zero-init convention (matching NAM Core / NeuralAudio)
    /// rather than a training-time forward pass that pre-pads the whole input.
    pub fn receptive_field(&self) -> usize {
        self.receptive_field
    }

    /// Process a buffer of mono samples in place.
    ///
    /// **Real-time contract:** no heap allocation, locks, or syscalls. Enforced by
    /// `tests/rt_safety.rs`.
    pub fn process_buffer(&mut self, io: &mut [f32]) {
        for sample in io.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Process a single mono sample, returning one output sample.
    ///
    /// Equivalent to a one-element [`Self::process_buffer`]; convenient for
    /// callers that are not buffer-oriented. Allocation-free.
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let cond = [x];
        let n = self.arrays.len();
        if n == 0 {
            return self.head_scale * x;
        }

        // First array: input and condition are the mono sample; the incoming head
        // is silence of the array's channel width.
        self.head_a[..self.channels0].fill(0.0);
        {
            let ch = self.arrays[0].channels();
            let hs = self.arrays[0].head_size();
            self.arrays[0].process_sample(
                &cond,
                &cond,
                &self.head_a[..ch],
                &mut self.head_b[..hs],
                &mut self.sig_b[..ch],
            );
        }
        std::mem::swap(&mut self.head_a, &mut self.head_b);
        std::mem::swap(&mut self.sig_a, &mut self.sig_b);

        for i in 1..n {
            let in_w = self.arrays[i - 1].channels();
            let ch = self.arrays[i].channels();
            let hs = self.arrays[i].head_size();
            self.arrays[i].process_sample(
                &self.sig_a[..in_w],
                &cond,
                &self.head_a[..ch],
                &mut self.head_b[..hs],
                &mut self.sig_b[..ch],
            );
            std::mem::swap(&mut self.head_a, &mut self.head_b);
            std::mem::swap(&mut self.sig_a, &mut self.sig_b);
        }

        // After the final swap, head_a holds the last array's head output.
        self.head_scale * self.head_a[0]
    }

    /// Reset all internal state (ring buffers) to silence.
    pub fn reset(&mut self) {
        for a in &mut self.arrays {
            a.reset();
        }
        self.head_a.fill(0.0);
        self.head_b.fill(0.0);
        self.sig_a.fill(0.0);
        self.sig_b.fill(0.0);
    }
}

/// Receptive field implied by `config`: `1 + Σ (kernel_size - 1) · dilation` over
/// every dilated layer in every array. The stacked dilated convs compose additively,
/// so this is the number of past input samples the final output depends on.
fn receptive_field(cfg: &WaveNetConfig) -> usize {
    let mut rf = 1;
    for la in &cfg.layers {
        for &d in &la.dilations {
            rf += (la.kernel_size - 1) * d;
        }
    }
    rf
}

/// Number of `f32`s `config` implies in the flat weight blob, including the final
/// `head_scale`.
fn expected_weight_count(cfg: &WaveNetConfig) -> usize {
    let mut total = 0;
    for la in &cfg.layers {
        let mid = if la.gated {
            2 * la.channels
        } else {
            la.channels
        };
        total += la.channels * la.input_size; // rechannel (no bias)
        let per_layer = mid * la.channels * la.kernel_size // conv weights
            + mid                                          // conv bias
            + mid * la.condition_size                      // input mixer (no bias)
            + la.channels * la.channels                    // 1x1 weights
            + la.channels; // 1x1 bias
        total += la.dilations.len() * per_layer;
        total += la.head_size * la.channels; // head rechannel weights
        if la.head_bias {
            total += la.head_size;
        }
    }
    total + 1 // head_scale
}

fn build_array(r: &mut Reader, la: &LayerArrayConfig) -> Result<LayerArray, Error> {
    let activation = Activation::from_name(&la.activation)?;
    let mid = if la.gated {
        2 * la.channels
    } else {
        la.channels
    };

    let rechannel_w = r.take(la.channels * la.input_size);
    let mut layers = Vec::with_capacity(la.dilations.len());
    for &d in &la.dilations {
        let conv_w = r.take(mid * la.channels * la.kernel_size);
        let conv_b = r.take(mid);
        let mix_w = r.take(mid * la.condition_size);
        let one_w = r.take(la.channels * la.channels);
        let one_b = r.take(la.channels);
        layers.push(Layer::new(
            la.channels,
            la.condition_size,
            la.kernel_size,
            d,
            activation,
            la.gated,
            conv_w,
            conv_b,
            mix_w,
            one_w,
            one_b,
        ));
    }
    let head_w = r.take(la.head_size * la.channels);
    let head_b = if la.head_bias {
        Some(r.take(la.head_size))
    } else {
        None
    };

    Ok(LayerArray::new(
        la.input_size,
        la.channels,
        la.head_size,
        rechannel_w,
        layers,
        head_w,
        head_b,
    ))
}

/// Sequential reader over the flat weight blob, consumed in `export_weights`
/// order. The caller validates the total count up front, so `take` never
/// over-runs.
struct Reader<'a> {
    w: &'a [f32],
    i: usize,
}

impl<'a> Reader<'a> {
    fn new(w: &'a [f32]) -> Self {
        Self { w, i: 0 }
    }

    fn take(&mut self, n: usize) -> Vec<f32> {
        let chunk = self.w[self.i..self.i + n].to_vec();
        self.i += n;
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1 array, 1 layer, 1 channel, ReLU. Weight order:
    // rechannel=1, conv_w=2, conv_b=0.5, mix_w=1, one_w=3, one_b=0.1,
    // head_rechannel=0.5, head_scale=10.
    const TINY: &str = r#"{
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
                "kernel_size": 1, "dilations": [1], "activation": "ReLU",
                "gated": false, "head_bias": false
            }],
            "head": null, "head_scale": 10.0
        },
        "weights": [1.0, 2.0, 0.5, 1.0, 3.0, 0.1, 0.5, 10.0]
    }"#;

    #[test]
    fn tiny_model_matches_hand_computed_forward() {
        let model = NamModel::from_json_str(TINY).unwrap();
        let mut wn = WaveNet::new(&model).unwrap();

        // x=0.5, cond=0.5: z = 2*0.5 + 0.5 + 1*0.5 = 2.0 ; relu=2.0
        // head = 0.5*2.0 = 1.0 ; out = head_scale * 1.0 = 10.0
        let mut buf = [0.5_f32];
        wn.process_buffer(&mut buf);
        assert!((buf[0] - 10.0).abs() < 1e-5, "got {}", buf[0]);
    }

    #[test]
    fn receptive_field_sums_dilated_taps() {
        // 1 + Σ(k-1)·d. Mirrors the reference model: kernel 3, dilations [1,2] then [8].
        let mk = |dilations: Vec<usize>| LayerArrayConfig {
            input_size: 1,
            condition_size: 1,
            channels: 1,
            head_size: 1,
            kernel_size: 3,
            dilations,
            activation: "Tanh".into(),
            gated: false,
            head_bias: false,
        };
        let cfg = WaveNetConfig {
            layers: vec![mk(vec![1, 2]), mk(vec![8])],
            head: None,
            head_scale: 1.0,
        };
        // (3-1)*1 + (3-1)*2 + (3-1)*8 = 2 + 4 + 16 = 22, + 1 = 23.
        assert_eq!(receptive_field(&cfg), 23);

        // TINY (kernel 1, dilation 1) reaches back over no past samples: rf = 1.
        let model = NamModel::from_json_str(TINY).unwrap();
        assert_eq!(WaveNet::new(&model).unwrap().receptive_field(), 1);
    }

    #[test]
    fn reset_restores_from_fresh_result() {
        let model = NamModel::from_json_str(TINY).unwrap();
        let mut wn = WaveNet::new(&model).unwrap();
        let mut warm = [0.3_f32, -0.7, 0.2];
        wn.process_buffer(&mut warm);
        wn.reset();
        let mut a = [0.5_f32];
        wn.process_buffer(&mut a);
        assert!((a[0] - 10.0).abs() < 1e-5, "got {}", a[0]);
    }

    #[test]
    fn wrong_weight_count_is_rejected() {
        let bad = TINY.replace(
            "[1.0, 2.0, 0.5, 1.0, 3.0, 0.1, 0.5, 10.0]",
            "[1.0, 2.0, 0.5, 1.0, 3.0, 0.1, 0.5]",
        );
        let model = NamModel::from_json_str(&bad).unwrap();
        match WaveNet::new(&model) {
            Err(Error::WeightCountMismatch { expected, found }) => {
                assert_eq!(expected, 8);
                assert_eq!(found, 7);
            }
            other => panic!("expected WeightCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_architecture_is_rejected() {
        let bad = TINY.replace("\"WaveNet\"", "\"LSTM\"");
        let model = NamModel::from_json_str(&bad).unwrap();
        assert!(matches!(
            WaveNet::new(&model),
            Err(Error::UnsupportedArchitecture(_))
        ));
    }
}
