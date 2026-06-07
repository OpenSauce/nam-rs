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
use crate::reader::Reader;

mod array;
mod conv;
mod layer;

use array::LayerArray;
use conv::MAX_BLOCK;
use layer::{Activation, Layer};

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
    /// Planar `[width][MAX_BLOCK]` block-path twins of the carry buffers, plus a
    /// scratch copy of the conditioning chunk. Used by [`WaveNet::process_buffer`].
    head_a_blk: Vec<f32>,
    head_b_blk: Vec<f32>,
    sig_a_blk: Vec<f32>,
    sig_b_blk: Vec<f32>,
    cond_blk: Vec<f32>,
}

impl WaveNet {
    /// Build a runnable model from a parsed `.nam` file.
    ///
    /// All allocation happens here. Fails if the architecture is unsupported, an
    /// activation is unknown, or the flat weight blob does not match the config.
    pub fn new(model: &NamModel) -> Result<Self, Error> {
        let cfg = match &model.config {
            crate::model::ModelConfig::WaveNet(cfg) => cfg,
            crate::model::ModelConfig::Lstm(_) => {
                return Err(Error::UnsupportedArchitecture(model.architecture.clone()))
            }
        };

        let expected = expected_weight_count(cfg)?;
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
        // The up-front check guarantees `expected == weights.len()`; this asserts the
        // other half of the invariant — that building consumed exactly `expected`, so
        // `expected_weight_count` and the `build_array` consumption order agree.
        debug_assert_eq!(
            r.remaining(),
            0,
            "build_array consumed fewer weights than expected_weight_count claimed"
        );

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
            head_a_blk: vec![0.0; head_w * MAX_BLOCK],
            head_b_blk: vec![0.0; head_w * MAX_BLOCK],
            sig_a_blk: vec![0.0; sig_w * MAX_BLOCK],
            sig_b_blk: vec![0.0; sig_w * MAX_BLOCK],
            cond_blk: vec![0.0; MAX_BLOCK],
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
    /// Runs the block kernel: each `MAX_BLOCK`-sized chunk is pushed through one
    /// array (and one layer) at a time, keeping each weight matrix hot across the
    /// whole chunk. Bit-for-bit equivalent to looping [`Self::process_sample`], and
    /// it shares the same streaming history, so the two are interchangeable.
    ///
    /// **Real-time contract:** no heap allocation, locks, or syscalls. Enforced by
    /// `tests/rt_safety.rs`.
    pub fn process_buffer(&mut self, io: &mut [f32]) {
        if self.arrays.is_empty() {
            for s in io.iter_mut() {
                *s *= self.head_scale;
            }
            return;
        }
        let mut off = 0;
        while off < io.len() {
            let n = (io.len() - off).min(MAX_BLOCK);
            self.process_chunk(&mut io[off..off + n], n);
            off += n;
        }
    }

    /// Run one `n <= MAX_BLOCK` chunk through every array via the planar block path.
    /// `chunk` is the mono input and is overwritten with the output.
    fn process_chunk(&mut self, chunk: &mut [f32], n: usize) {
        // The conditioning is the mono input; copy it out before we overwrite `chunk`.
        self.cond_blk[..n].copy_from_slice(chunk);

        // First array: input and condition are the mono signal; the incoming head is
        // silence of the array's channel width.
        self.head_a_blk[..self.channels0 * n].fill(0.0);
        {
            let ch = self.arrays[0].channels();
            let hs = self.arrays[0].head_size();
            self.arrays[0].process_block(
                &self.cond_blk[..n],
                &self.cond_blk[..n],
                &self.head_a_blk[..ch * n],
                &mut self.head_b_blk[..hs * n],
                &mut self.sig_b_blk[..ch * n],
                n,
            );
        }
        std::mem::swap(&mut self.head_a_blk, &mut self.head_b_blk);
        std::mem::swap(&mut self.sig_a_blk, &mut self.sig_b_blk);

        for i in 1..self.arrays.len() {
            let in_w = self.arrays[i - 1].channels();
            let ch = self.arrays[i].channels();
            let hs = self.arrays[i].head_size();
            self.arrays[i].process_block(
                &self.sig_a_blk[..in_w * n],
                &self.cond_blk[..n],
                &self.head_a_blk[..ch * n],
                &mut self.head_b_blk[..hs * n],
                &mut self.sig_b_blk[..ch * n],
                n,
            );
            std::mem::swap(&mut self.head_a_blk, &mut self.head_b_blk);
            std::mem::swap(&mut self.sig_a_blk, &mut self.sig_b_blk);
        }

        // After the final swap, head_a_blk holds the last array's head output, whose
        // head_size is 1 — row 0 is the per-sample head signal.
        for (t, s) in chunk.iter_mut().enumerate() {
            *s = self.head_scale * self.head_a_blk[t];
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
///
/// Uses checked arithmetic: an absurd or adversarial config whose dimensions overflow
/// `usize` returns [`Error::ConfigTooLarge`] rather than panicking (debug) or wrapping
/// to a wrong, small count (release).
fn expected_weight_count(cfg: &WaveNetConfig) -> Result<usize, Error> {
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(Error::ConfigTooLarge);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(Error::ConfigTooLarge);

    let mut total = 0usize;
    for la in &cfg.layers {
        let mid = if la.gated {
            mul(2, la.channels)?
        } else {
            la.channels
        };
        total = add(total, mul(la.channels, la.input_size)?)?; // rechannel (no bias)

        let per_layer = add(
            add(
                add(
                    add(
                        mul(mul(mid, la.channels)?, la.kernel_size)?, // conv weights
                        mid,                                          // conv bias
                    )?,
                    mul(mid, la.condition_size)?, // input mixer (no bias)
                )?,
                mul(la.channels, la.channels)?, // 1x1 weights
            )?,
            la.channels, // 1x1 bias
        )?;
        total = add(total, mul(la.dilations.len(), per_layer)?)?;

        total = add(total, mul(la.head_size, la.channels)?)?; // head rechannel weights
        if la.head_bias {
            total = add(total, la.head_size)?;
        }
    }
    add(total, 1) // head_scale
}

fn build_array(r: &mut Reader, la: &LayerArrayConfig) -> Result<LayerArray, Error> {
    let activation = match &la.activation {
        crate::model::ActivationSpec::Named { name, .. } => Activation::from_name(name)?,
        crate::model::ActivationSpec::Unsupported(v) => {
            return Err(Error::UnsupportedFeature(format!("activation: {v}")))
        }
    };
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
            activation: crate::model::ActivationSpec::Named { name: "Tanh".into(), negative_slope: None },
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

    /// A structurally valid config whose dimensions overflow `usize` must return
    /// `ConfigTooLarge`, not panic (debug) or wrap to a wrong count (release).
    #[test]
    fn absurd_dimensions_error_instead_of_overflowing() {
        let json = TINY.replace("\"channels\": 1", "\"channels\": 4294967296");
        let model = NamModel::from_json_str(&json).unwrap();
        assert!(matches!(WaveNet::new(&model), Err(Error::ConfigTooLarge)));
    }

    /// Pins the weight-count invariant `take` relies on: `expected_weight_count` must
    /// equal exactly what `build_array` (+ head_scale) consumes, across config shapes.
    /// Building with that many weights succeeds (and the `debug_assert` in `new` fires
    /// if consumption drifts below it); one fewer / one more is a count mismatch.
    #[test]
    fn weight_count_matches_consumption_across_shapes() {
        #[allow(clippy::too_many_arguments)]
        let mk = |input_size,
                  channels,
                  head_size,
                  kernel_size,
                  dilations: Vec<usize>,
                  gated,
                  head_bias| LayerArrayConfig {
            input_size,
            condition_size: 1,
            channels,
            head_size,
            kernel_size,
            dilations,
            activation: crate::model::ActivationSpec::Named { name: "Tanh".into(), negative_slope: None },
            gated,
            head_bias,
        };
        let layer_sets = vec![
            vec![mk(1, 1, 1, 1, vec![1], false, false)],
            vec![mk(1, 2, 1, 3, vec![1, 2], false, false)],
            vec![mk(1, 4, 2, 3, vec![1, 2, 4], true, false)], // gated
            vec![mk(1, 3, 1, 3, vec![1], false, true)],       // head_bias
            // two arrays: the second takes the first's channels as its input_size.
            vec![
                mk(1, 4, 1, 3, vec![1, 2], false, false),
                mk(4, 2, 1, 3, vec![1], true, true),
            ],
        ];
        for layers in layer_sets {
            let cfg = WaveNetConfig {
                layers,
                head: None,
                head_scale: 1.0,
            };
            let n = expected_weight_count(&cfg).unwrap();
            let mk_model = |count: usize| NamModel {
                version: "0".into(),
                architecture: "WaveNet".into(),
                config: crate::model::ModelConfig::WaveNet(cfg.clone()),
                weights: vec![0.0; count],
                sample_rate: None,
                metadata: None,
            };
            assert!(WaveNet::new(&mk_model(n)).is_ok(), "exact count n={n}");
            assert!(matches!(
                WaveNet::new(&mk_model(n - 1)),
                Err(Error::WeightCountMismatch { .. })
            ));
            assert!(matches!(
                WaveNet::new(&mk_model(n + 1)),
                Err(Error::WeightCountMismatch { .. })
            ));
        }
    }

    #[test]
    fn wavenet_new_rejects_non_wavenet() {
        let lstm = r#"{
            "version": "0.5.4", "architecture": "LSTM",
            "config": { "input_size": 1, "hidden_size": 4, "num_layers": 1 },
            "weights": [0.0]
        }"#;
        let model = NamModel::from_json_str(lstm).unwrap();
        assert!(matches!(
            WaveNet::new(&model),
            Err(Error::UnsupportedArchitecture(_))
        ));
    }

    /// End-to-end on the realistic standard model: the block `process_buffer` must
    /// equal a per-sample `process_sample` loop over the same signal, including
    /// across `MAX_BLOCK` chunk boundaries. `tests/parity.rs` drives `process_buffer`
    /// (the block path) and pins it to the reference NAM oracle within 1e-5; this
    /// test additionally ties the block path to the per-sample path, so the two are
    /// transitively guaranteed equivalent and both oracle-correct.
    #[test]
    fn process_buffer_equals_process_sample_loop_on_standard_model() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/reference_standard.nam");
        let json = std::fs::read_to_string(path).expect("read standard fixture");
        let model = NamModel::from_json_str(&json).expect("parse standard fixture");

        // A signal longer than MAX_BLOCK so chunking is exercised.
        let len = 2 * MAX_BLOCK + 137;
        let signal: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.013).sin() * 0.5 + (i as f32 * 0.27).sin() * 0.2)
            .collect();

        let mut per_sample = WaveNet::new(&model).unwrap();
        let want: Vec<f32> = signal
            .iter()
            .map(|&x| per_sample.process_sample(x))
            .collect();

        let mut block = WaveNet::new(&model).unwrap();
        let mut got = signal.clone();
        block.process_buffer(&mut got);

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "sample {i}: block {g}, per-sample {w}"
            );
        }
    }
}
