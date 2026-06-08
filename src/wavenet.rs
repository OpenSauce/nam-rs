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
use crate::model::{GatingMode, LayerArrayConfig, NamModel, WaveNetConfig};
use crate::reader::Reader;

mod activation;
mod array;
mod conv;
mod film;
mod gating;
mod layer;

use activation::Activation;
use array::LayerArray;
use conv::MAX_BLOCK;
use layer::Layer;

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
            crate::model::ModelConfig::Lstm(_) | crate::model::ModelConfig::Slimmable(_) => {
                return Err(Error::UnsupportedArchitecture(model.architecture.clone()))
            }
        };

        check_unsupported_features(cfg)?;
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
        // Head-carry invariant: each array's head output feeds the next array's head
        // input, and the per-sample/per-block paths size that carried buffer by the
        // *consumer's* channel count (`arrays[i].channels()`). So the producer's head
        // width must match: `arrays[i-1].head_size() == arrays[i].channels()`. Holds
        // for every real model and the 2-array A1 parity fixture; guard it explicitly
        // so a future multi-array A2 model that chains mismatched widths fails loudly
        // here instead of silently reading stale/garbage head rows.
        for i in 1..arrays.len() {
            let produced = arrays[i - 1].head_size();
            let consumed = arrays[i].channels();
            if produced != consumed {
                return Err(Error::UnsupportedFeature(format!(
                    "layer-array head-carry width mismatch: array {} head_size {produced} \
                     != array {i} channels {consumed}",
                    i - 1
                )));
            }
        }

        let head_scale = r.take(1)[0];
        // The up-front check guarantees `expected == weights.len()`; this asserts the
        // other half of the invariant — that building consumed exactly `expected`, so
        // `expected_weight_count` and the `build_array` consumption order agree.
        // A hard assert (not `debug_assert`): if the count formula and the consumption
        // order ever drift, under-consumption would otherwise leave the model silently
        // mis-built in release. (Over-consumption already panics in `Reader::take`.)
        assert_eq!(
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

/// Reject A2 features whose forward pass is not implemented yet, with a clear
/// [`Error::UnsupportedFeature`] (rather than silently mis-running). This is the
/// inverse of the old key-denylist: we now check typed fields. Each rejected case
/// is implemented in a later phase, at which point its check is removed here.
fn check_unsupported_features(cfg: &WaveNetConfig) -> Result<(), Error> {
    if cfg.post_stack_head.is_some() {
        return Err(Error::UnsupportedFeature("post-stack head".into()));
    }
    if cfg.condition_dsp.is_some() {
        return Err(Error::UnsupportedFeature("condition_dsp".into()));
    }
    if cfg.in_channels != 1 {
        return Err(Error::UnsupportedFeature("in_channels != 1".into()));
    }
    for la in &cfg.layers {
        if la.bottleneck != la.channels {
            return Err(Error::UnsupportedFeature("bottleneck != channels".into()));
        }
        if !la.layer1x1.active {
            return Err(Error::UnsupportedFeature("inactive layer1x1".into()));
        }
        if la.layer1x1.groups != 1 {
            return Err(Error::UnsupportedFeature("grouped layer1x1".into()));
        }
        if la.head1x1.active {
            return Err(Error::UnsupportedFeature("head1x1".into()));
        }
        if la.groups_input != 1 || la.groups_input_mixin != 1 {
            return Err(Error::UnsupportedFeature("grouped input conv".into()));
        }
        for f in [
            &la.conv_pre_film,
            &la.conv_post_film,
            &la.input_mixin_pre_film,
            &la.input_mixin_post_film,
            &la.activation_pre_film,
            &la.activation_post_film,
            &la.layer1x1_post_film,
            &la.head1x1_post_film,
        ] {
            if f.active {
                return Err(Error::UnsupportedFeature("FiLM".into()));
            }
        }
        let first = la.gating_mode();
        if la.gating_modes.iter().any(|&g| g != first) {
            return Err(Error::UnsupportedFeature("mixed gating modes".into()));
        }
        if first == GatingMode::Blended {
            return Err(Error::UnsupportedFeature("blended gating".into()));
        }
        if first == GatingMode::Gated && la.secondary_activations.iter().any(|s| !is_sigmoid(s)) {
            return Err(Error::UnsupportedFeature(
                "non-sigmoid secondary activation".into(),
            ));
        }
    }
    Ok(())
}

/// True if `spec` is the (default) Sigmoid secondary activation.
fn is_sigmoid(spec: &crate::model::ActivationSpec) -> bool {
    matches!(spec, crate::model::ActivationSpec::Named { name, .. } if name == "Sigmoid")
}

/// Receptive field implied by `config`: per layer `(kernel_size - 1)·dilation`,
/// plus `(head_kernel_size - 1)` per array for the (possibly multi-tap) head.
fn receptive_field(cfg: &WaveNetConfig) -> usize {
    let mut rf = 1;
    for la in &cfg.layers {
        for (k, &d) in la.kernel_sizes.iter().zip(&la.dilations) {
            rf += (k - 1) * d;
        }
        rf += la.head_kernel_size - 1;
    }
    rf
}

/// Number of `f32`s `config` implies in the flat weight blob, including the final
/// `head_scale`.
///
/// Uses checked arithmetic: an absurd or adversarial config whose dimensions overflow
/// `usize` returns [`Error::ConfigTooLarge`] rather than panicking (debug) or wrapping
/// to a wrong, small count (release).
/// Number of weights one layer-array consumes from the flat blob, in exactly
/// [`build_array`]'s `take` order. This is the **single arithmetic source** for the
/// weight layout: [`expected_weight_count`] sums it across arrays for the up-front
/// validation, and [`build_array`] asserts it consumes precisely this many, so the
/// count formula and the consumption order cannot silently drift apart.
fn array_weight_count(la: &LayerArrayConfig) -> Result<usize, Error> {
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(Error::ConfigTooLarge);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(Error::ConfigTooLarge);

    let gated = la.gating_mode() != GatingMode::None;
    let mid = if gated {
        mul(2, la.bottleneck)?
    } else {
        la.bottleneck
    };
    let head1x1_out = la.head1x1.out_channels.unwrap_or(la.channels);
    let cond = la.condition_size;

    // Grouped Conv1d weight count: out*in*kernel/groups (compact). Caller adds bias.
    let conv_w = |out: usize, in_ch: usize, k: usize, groups: usize| -> Result<usize, Error> {
        Ok(mul(mul(out, in_ch)?, k)? / groups) // dims validated divisible at build
    };
    // FiLM: out_rows = (shift?2:1)*input_dim; weights out_rows*cond/groups + out_rows bias.
    let film = |f: &crate::model::FilmConfig, input_dim: usize| -> Result<usize, Error> {
        if !f.active {
            return Ok(0);
        }
        let out_rows = if f.shift {
            mul(2, input_dim)?
        } else {
            input_dim
        };
        add(conv_w(out_rows, cond, 1, f.groups)?, out_rows)
    };

    let mut total = mul(la.channels, la.input_size)?; // rechannel (no bias)

    for &k in &la.kernel_sizes {
        let conv = add(conv_w(mid, la.channels, k, la.groups_input)?, mid)?;
        let mixin = conv_w(mid, cond, 1, la.groups_input_mixin)?; // no bias
        let mut layer = add(conv, mixin)?;
        if la.layer1x1.active {
            let l = add(
                conv_w(la.channels, la.bottleneck, 1, la.layer1x1.groups)?,
                la.channels,
            )?;
            layer = add(layer, l)?;
        }
        if la.head1x1.active {
            let h = add(
                conv_w(head1x1_out, la.bottleneck, 1, la.head1x1.groups)?,
                head1x1_out,
            )?;
            layer = add(layer, h)?;
        }
        // 8 FiLMs in NAMCore order.
        layer = add(layer, film(&la.conv_pre_film, la.channels)?)?;
        layer = add(layer, film(&la.conv_post_film, mid)?)?;
        layer = add(layer, film(&la.input_mixin_pre_film, cond)?)?;
        layer = add(layer, film(&la.input_mixin_post_film, mid)?)?;
        layer = add(layer, film(&la.activation_pre_film, mid)?)?;
        layer = add(layer, film(&la.activation_post_film, la.bottleneck)?)?;
        layer = add(layer, film(&la.layer1x1_post_film, la.channels)?)?;
        layer = add(layer, film(&la.head1x1_post_film, head1x1_out)?)?;
        total = add(total, layer)?;
    }

    // head rechannel reads head_in rows.
    let head_in = if la.head1x1.active {
        head1x1_out
    } else {
        la.bottleneck
    };
    total = add(
        total,
        mul(mul(la.head_size, head_in)?, la.head_kernel_size)?,
    )?;
    if la.head_bias {
        total = add(total, la.head_size)?;
    }
    Ok(total)
}

fn expected_weight_count(cfg: &WaveNetConfig) -> Result<usize, Error> {
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(Error::ConfigTooLarge);
    let mut total = 0usize;
    for la in &cfg.layers {
        total = add(total, array_weight_count(la)?)?;
    }
    add(total, 1) // head_scale
}

fn build_array(r: &mut Reader, la: &LayerArrayConfig) -> Result<LayerArray, Error> {
    let gated = la.gating_mode() == GatingMode::Gated;
    let mid = if gated { 2 * la.channels } else { la.channels };

    let before = r.remaining();
    let rechannel_w = r.take(la.channels * la.input_size);
    let mut layers = Vec::with_capacity(la.dilations.len());
    for (i, &d) in la.dilations.iter().enumerate() {
        let k = la.kernel_sizes[i];
        let activation = Activation::from_spec(&la.activations[i])?;
        let conv_w = r.take(mid * la.channels * k);
        let conv_b = r.take(mid);
        let mix_w = r.take(mid * la.condition_size);
        let one_w = r.take(la.channels * la.channels);
        let one_b = r.take(la.channels);
        layers.push(Layer::new(
            la.channels,
            la.condition_size,
            k,
            d,
            activation,
            gated,
            conv_w,
            conv_b,
            mix_w,
            one_w,
            one_b,
        ));
    }
    let head_w = r.take(la.head_size * la.channels * la.head_kernel_size);
    let head_b = if la.head_bias {
        Some(r.take(la.head_size))
    } else {
        None
    };

    // Self-check: this array must have consumed exactly what the single count source
    // claims. Fires immediately (and locally) if a future edit changes the `take`
    // order here without updating `array_weight_count`, or vice versa.
    debug_assert_eq!(
        before - r.remaining(),
        array_weight_count(la)?,
        "build_array consumption drifted from array_weight_count"
    );

    Ok(LayerArray::new(
        la.input_size,
        la.channels,
        la.channels, // head_in (Task 3 computes the real head-contribution width)
        la.head_size,
        la.head_kernel_size,
        rechannel_w,
        layers,
        head_w,
        head_b,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a normalized LayerArrayConfig for tests via the parser (keeps it honest).
    fn mk_layer(json: serde_json::Value) -> crate::model::LayerArrayConfig {
        let raw: crate::model::RawLayerArrayConfig = serde_json::from_value(json).unwrap();
        raw.normalize().unwrap()
    }

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
    fn array_weight_count_includes_a2_subblocks() {
        // channels=4, bottleneck=2, condition=1, GATED (mid=2*bn=4), kernel 3 x1 layer,
        // head_size=1 head_kernel=1 no head_bias, groups all 1, head1x1 active out=3,
        // layer1x1 active, conv_post_film (shift) + activation_post_film (no shift) active.
        // Per-layer weights (NAMCore order):
        //   conv:     mid*channels*k + mid          = 4*4*3 + 4   = 52
        //   mixin:    mid*condition                 = 4*1         = 4
        //   layer1x1: channels*bottleneck + channels= 4*2 + 4     = 12
        //   head1x1:  head1x1_out*bottleneck + out   = 3*2 + 3     = 9
        //   conv_post_film: shift -> out_rows=2*mid=8; 8*condition + 8 = 8+8 = 16
        //   activation_post_film: no shift -> out_rows=bottleneck=2; 2*1 + 2 = 4
        // rechannel: channels*input = 4*1 = 4
        // head_rechannel: head_size*head_in*head_k = 1*3*1 = 3 (head_in=head1x1_out=3, no bias)
        // array total = 4 + (52+4+12+9+16+4) + 3 = 4 + 97 + 3 = 104
        let la = mk_layer(serde_json::json!({
            "input_size": 1, "condition_size": 1, "channels": 4, "bottleneck": 2,
            "kernel_sizes": [3], "dilations": [1],
            "activation": [{"type":"Tanh"}],
            "gating_mode": ["gated"],
            "layer1x1": {"active": true, "groups": 1},
            "head1x1": {"active": true, "out_channels": 3, "groups": 1},
            "head": {"out_channels": 1, "kernel_size": 1, "bias": false},
            "conv_post_film": {"active": true, "shift": true, "groups": 1},
            "activation_post_film": {"active": true, "shift": false, "groups": 1}
        }));
        assert_eq!(array_weight_count(&la).unwrap(), 104);
    }

    #[test]
    fn receptive_field_sums_dilated_taps() {
        // rf = 1 + Σ(k-1)·d + Σ(head_kernel_size-1) over all arrays. Here the head
        // kernel is 1 so its term vanishes; mirrors the reference model: kernel 3,
        // dilations [1,2] then [8].
        let cfg = WaveNetConfig {
            layers: vec![
                mk_layer(serde_json::json!({
                    "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
                    "kernel_size": 3, "dilations": [1, 2], "activation": "Tanh",
                    "gated": false, "head_bias": false
                })),
                mk_layer(serde_json::json!({
                    "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
                    "kernel_size": 3, "dilations": [8], "activation": "Tanh",
                    "gated": false, "head_bias": false
                })),
            ],
            post_stack_head: None,
            head_scale: 1.0,
            in_channels: 1,
            condition_dsp: None,
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
        let layer_sets: Vec<Vec<crate::model::LayerArrayConfig>> = vec![
            vec![mk_layer(serde_json::json!({
                "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
                "kernel_size": 1, "dilations": [1], "activation": "Tanh",
                "gated": false, "head_bias": false
            }))],
            vec![mk_layer(serde_json::json!({
                "input_size": 1, "condition_size": 1, "channels": 2, "head_size": 1,
                "kernel_size": 3, "dilations": [1, 2], "activation": "Tanh",
                "gated": false, "head_bias": false
            }))],
            vec![mk_layer(serde_json::json!({
                "input_size": 1, "condition_size": 1, "channels": 4, "head_size": 2,
                "kernel_size": 3, "dilations": [1, 2, 4], "activation": "Tanh",
                "gated": true, "head_bias": false
            }))], // gated
            vec![mk_layer(serde_json::json!({
                "input_size": 1, "condition_size": 1, "channels": 3, "head_size": 1,
                "kernel_size": 3, "dilations": [1], "activation": "Tanh",
                "gated": false, "head_bias": true
            }))], // head_bias
            // two arrays: the second takes the first's channels as its input_size,
            // and the first's head_size must equal the second's channels (the
            // head-carry invariant guarded in `WaveNet::new`).
            vec![
                mk_layer(serde_json::json!({
                    "input_size": 1, "condition_size": 1, "channels": 4, "head_size": 2,
                    "kernel_size": 3, "dilations": [1, 2], "activation": "Tanh",
                    "gated": false, "head_bias": false
                })),
                mk_layer(serde_json::json!({
                    "input_size": 4, "condition_size": 1, "channels": 2, "head_size": 1,
                    "kernel_size": 3, "dilations": [1], "activation": "Tanh",
                    "gated": true, "head_bias": true
                })),
            ],
        ];
        for layers in layer_sets {
            let cfg = WaveNetConfig {
                layers,
                post_stack_head: None,
                head_scale: 1.0,
                in_channels: 1,
                condition_dsp: None,
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

    #[test]
    fn a2_leaky_relu_and_conv_head_parse_and_build() {
        // LeakyReLU + multi-tap conv head (kernel_size=16) are now fully supported.
        // Verify the config parses and builds (weight count determines valid input).
        let json = r#"{
            "version":"0.7.0","architecture":"WaveNet","config":{
                "layers":[{"input_size":1,"condition_size":1,"channels":2,"bottleneck":2,
                    "dilations":[1,2],"kernel_sizes":[3,3],
                    "activation":[{"type":"LeakyReLU"},{"type":"LeakyReLU"}],
                    "head":{"out_channels":1,"kernel_size":16,"bias":true},
                    "layer1x1":{"active":true,"groups":1},
                    "gating_mode":["none","none"]}],
                "head":null,"head_scale":0.5},
            "weights":[]}"#;
        let model0 = NamModel::from_json_str(json).expect("parses cleanly now");
        let cfg = match &model0.config {
            crate::model::ModelConfig::WaveNet(c) => c,
            _ => unreachable!(),
        };
        let n = expected_weight_count(cfg).unwrap();
        let model = NamModel {
            version: "0.7.0".into(),
            architecture: "WaveNet".into(),
            config: crate::model::ModelConfig::WaveNet(cfg.clone()),
            weights: vec![0.0; n],
            sample_rate: None,
            metadata: None,
        };
        assert!(
            WaveNet::new(&model).is_ok(),
            "LeakyReLU + multi-tap conv head should now build"
        );
    }

    #[test]
    fn a1_still_builds_and_runs_after_typed_config() {
        let model = NamModel::from_json_str(TINY).unwrap();
        let mut wn = WaveNet::new(&model).unwrap();
        let mut buf = [0.5_f32];
        wn.process_buffer(&mut buf);
        assert!((buf[0] - 10.0).abs() < 1e-5, "got {}", buf[0]);
    }

    /// Non-power-of-2 and out-of-order dilations with kernel 6 must size buffers
    /// correctly: receptive field is order-independent, and the block path equals the
    /// per-sample path. Mirrors A2's `[1,5,29,97,227]`-style dilations.
    #[test]
    fn non_pow2_out_of_order_dilations_size_correctly() {
        let json = r#"{
            "version":"0.7.0","architecture":"WaveNet","config":{"layers":[{
                "input_size":1,"condition_size":1,"channels":2,"head_size":1,
                "kernel_size":6,"dilations":[97,1,227,5,29],"activation":"ReLU",
                "gated":false,"head_bias":false}],"head":null,"head_scale":0.5},
            "weights":[]}"#;
        // Parse the config, then fill weights to the exact expected count so build succeeds.
        let model0 = NamModel::from_json_str(json).unwrap();
        let cfg = match &model0.config {
            crate::model::ModelConfig::WaveNet(c) => c,
            _ => unreachable!(),
        };
        let n = expected_weight_count(cfg).unwrap();
        let weights: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let model = NamModel {
            version: "0.7.0".into(),
            architecture: "WaveNet".into(),
            config: crate::model::ModelConfig::WaveNet(cfg.clone()),
            weights,
            sample_rate: None,
            metadata: None,
        };

        // rf = 1 + (k-1)*sum(dilations), order-independent.
        let want_rf = 1 + (6 - 1) * (97 + 1 + 227 + 5 + 29);
        let mut per_sample = WaveNet::new(&model).unwrap();
        assert_eq!(per_sample.receptive_field(), want_rf);

        // block path == per-sample path over a signal longer than MAX_BLOCK.
        let len = MAX_BLOCK + 200;
        let signal: Vec<f32> = (0..len).map(|i| (i as f32 * 0.017).sin() * 0.4).collect();
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

    #[test]
    fn multitap_head_config_now_builds() {
        // Weight count: rechannel 2 + 2 layers*(conv 2*2*3+2=14, mix 2, one 2*2+2=6 =>22)
        // + head(1*2*4=8 + bias 1 =9) + head_scale 1 = 56.
        let json = r#"{
            "version":"0.7.0","architecture":"WaveNet","config":{
                "layers":[{"input_size":1,"condition_size":1,"channels":2,"bottleneck":2,
                    "dilations":[1,2],"kernel_sizes":[3,3],
                    "activation":[{"type":"ReLU"},{"type":"ReLU"}],
                    "head":{"out_channels":1,"kernel_size":4,"bias":true},
                    "layer1x1":{"active":true,"groups":1},
                    "gating_mode":["none","none"]}],
                "head":null,"head_scale":0.5},
            "weights":[]}"#;
        let model0 = NamModel::from_json_str(json).unwrap();
        let cfg = match &model0.config {
            crate::model::ModelConfig::WaveNet(c) => c,
            _ => unreachable!(),
        };
        let n = expected_weight_count(cfg).unwrap();
        assert_eq!(n, 56, "expected 56 weights for this conv-head config");
        let weights: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let model = NamModel {
            version: "0.7.0".into(),
            architecture: "WaveNet".into(),
            config: crate::model::ModelConfig::WaveNet(cfg.clone()),
            weights,
            sample_rate: None,
            metadata: None,
        };
        let mut wn = WaveNet::new(&model).expect("conv-head model builds");
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let want: Vec<f32> = {
            let mut w = WaveNet::new(&model).unwrap();
            signal.iter().map(|&x| w.process_sample(x)).collect()
        };
        let mut got = signal.clone();
        wn.process_buffer(&mut got);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "sample {i}: block {g} vs per-sample {w}"
            );
        }
    }
}
