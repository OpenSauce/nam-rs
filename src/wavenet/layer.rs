//! A single WaveNet layer: dilated conv + conditioning mix-in, an activation,
//! a residual 1x1, and a head contribution.
//!
//! Ported from NeuralAudio `WaveNetLayerT` (and cross-checked against NAM's
//! Python `_Layer`). For input `x` and conditioning `c`:
//!
//! ```text
//! z    = conv(x) + input_mixin(c)
//! post = activation(z)                 // gated: tanh(z_a) * sigmoid(z_b)
//! head_accum += post                   // accumulated for the head path
//! out  = one_by_one(post) + x          // residual to the next layer
//! ```

use super::activation::Activation;
use super::conv::{Conv1d, MAX_BLOCK};

/// One dilated WaveNet layer with all scratch buffers pre-allocated.
#[derive(Debug, Clone)]
pub(super) struct Layer {
    conv: Conv1d,
    mixin: Conv1d,
    one_by_one: Conv1d,
    activation: Activation,
    gated: bool,
    channels: usize,
    /// `mid`-wide scratch: dilated-conv output, then `+= mix`.
    block: Vec<f32>,
    /// `mid`-wide scratch: input-mixer output.
    mix: Vec<f32>,
    /// `channels`-wide scratch: post-activation value.
    post: Vec<f32>,
    /// Planar block-path scratch: `[mid][MAX_BLOCK]` twins of `block`/`mix`, and
    /// `[channels][MAX_BLOCK]` for `post`.
    block_blk: Vec<f32>,
    mix_blk: Vec<f32>,
    post_blk: Vec<f32>,
}

impl Layer {
    /// Build a layer from its (already de-interleaved) weight tensors.
    ///
    /// When `gated`, the dilated conv produces `2 * channels` outputs (the conv
    /// and input-mixer weight tensors must be sized accordingly).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        channels: usize,
        condition_size: usize,
        kernel: usize,
        dilation: usize,
        activation: Activation,
        gated: bool,
        conv_w: Vec<f32>,
        conv_b: Vec<f32>,
        mix_w: Vec<f32>,
        one_w: Vec<f32>,
        one_b: Vec<f32>,
    ) -> Self {
        let mid = if gated { 2 * channels } else { channels };
        Self {
            conv: Conv1d::new(channels, mid, kernel, dilation, conv_w, Some(conv_b)),
            mixin: Conv1d::new(condition_size, mid, 1, 1, mix_w, None),
            one_by_one: Conv1d::new(channels, channels, 1, 1, one_w, Some(one_b)),
            activation,
            gated,
            channels,
            block: vec![0.0; mid],
            mix: vec![0.0; mid],
            post: vec![0.0; channels],
            block_blk: vec![0.0; mid * MAX_BLOCK],
            mix_blk: vec![0.0; mid * MAX_BLOCK],
            post_blk: vec![0.0; channels * MAX_BLOCK],
        }
    }

    /// Process one sample.
    ///
    /// - `input`: this layer's input, `channels` wide.
    /// - `condition`: the conditioning signal, `condition_size` wide.
    /// - `head_accum`: `channels`-wide head accumulator; the post-activation is
    ///   *added* to it.
    /// - `out`: `channels`-wide residual output for the next layer.
    pub(super) fn process_sample(
        &mut self,
        input: &[f32],
        condition: &[f32],
        head_accum: &mut [f32],
        out: &mut [f32],
    ) {
        self.conv.process_sample(input, &mut self.block);
        self.mixin.process_sample(condition, &mut self.mix);
        for (b, m) in self.block.iter_mut().zip(&self.mix) {
            *b += *m;
        }

        if self.gated {
            for c in 0..self.channels {
                let a = self.activation.apply(self.block[c]);
                let g = Activation::Sigmoid.apply(self.block[c + self.channels]);
                self.post[c] = a * g;
            }
        } else {
            for c in 0..self.channels {
                self.post[c] = self.activation.apply(self.block[c]);
            }
        }

        for (h, p) in head_accum.iter_mut().zip(&self.post) {
            *h += *p;
        }

        self.one_by_one.process_sample(&self.post, out);
        for (o, x) in out.iter_mut().zip(input) {
            *o += *x;
        }
    }

    /// Block twin of [`Self::process_sample`]. All slices are **planar** `[ch][t]`,
    /// `n <= MAX_BLOCK`: `input`/`out`/`head_accum` are `channels * n`, `condition`
    /// is `condition_size * n`. Equivalent to `n` per-sample calls; allocation-free.
    pub(super) fn process_block(
        &mut self,
        input: &[f32],
        condition: &[f32],
        head_accum: &mut [f32],
        out: &mut [f32],
        n: usize,
    ) {
        let mid = self.block.len();
        let block = &mut self.block_blk[..mid * n];
        let mix = &mut self.mix_blk[..mid * n];
        let post = &mut self.post_blk[..self.channels * n];

        self.conv.process_block(input, block, n);
        self.mixin.process_block(condition, mix, n);
        for (b, m) in block.iter_mut().zip(mix.iter()) {
            *b += *m;
        }

        // Planar rows: value branch is channel `c` at `c*n`, gate branch (if gated)
        // is channel `c + channels` at `(c + channels)*n`.
        if self.gated {
            for c in 0..self.channels {
                let (vrow, grow) = (c * n, (c + self.channels) * n);
                for t in 0..n {
                    let a = self.activation.apply(block[vrow + t]);
                    let g = Activation::Sigmoid.apply(block[grow + t]);
                    post[c * n + t] = a * g;
                }
            }
        } else {
            for c in 0..self.channels {
                for t in 0..n {
                    post[c * n + t] = self.activation.apply(block[c * n + t]);
                }
            }
        }

        for (h, p) in head_accum.iter_mut().zip(post.iter()) {
            *h += *p;
        }

        self.one_by_one.process_block(post, out, n);
        for (o, x) in out.iter_mut().zip(input.iter()) {
            *o += *x;
        }
    }

    pub(super) fn reset(&mut self) {
        self.conv.reset();
        self.mixin.reset();
        self.one_by_one.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_layer_residual_and_head_accumulate() {
        // channels=1, condition=1, kernel=1, dilation=1, ReLU, not gated.
        // conv: block = 2*input + 0.5 ; mixin: mix = 1*condition ; z = block+mix
        // post = relu(z) ; out = 3*post + 0.1 + input (residual) ; head += post
        let mut layer = Layer::new(
            1,
            1,
            1,
            1,
            Activation::Relu,
            false,
            vec![2.0],
            vec![0.5],
            vec![1.0],
            vec![3.0],
            vec![0.1],
        );

        let mut head = vec![0.0];
        let mut out = vec![0.0];

        // Sample 1: input=0.5, cond=0.5 -> z = 1.0+0.5+0.5 = 2.0, relu=2.0
        layer.process_sample(&[0.5], &[0.5], &mut head, &mut out);
        assert_eq!(out, vec![6.6]); // 3*2 + 0.1 + 0.5
        assert_eq!(head, vec![2.0]);

        // Sample 2: input=0.5, cond=-3.0 -> z = 1.0+0.5-3.0 = -1.5, relu=0
        layer.process_sample(&[0.5], &[-3.0], &mut head, &mut out);
        assert_eq!(out, vec![0.6]); // 3*0 + 0.1 + 0.5
        assert_eq!(head, vec![2.0]); // unchanged: += 0
    }

    #[test]
    fn gated_layer_multiplies_tanh_path_by_sigmoid_gate() {
        // channels=1, gated -> mid=2. Make the gate branch (block[1]) == 0 so
        // sigmoid(0)=0.5 exactly; ReLU on the value branch keeps it exact.
        // conv weights [out=2][in=1][k=1] = [value=2.0, gate=0.0]; bias [0,0].
        let mut layer = Layer::new(
            1,
            1,
            1,
            1,
            Activation::Relu,
            true,
            vec![2.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0],
            vec![0.0],
        );

        let mut head = vec![0.0];
        let mut out = vec![0.0];
        // input=1, cond=0 -> block=[2,0]; post = relu(2)*sigmoid(0) = 2*0.5 = 1
        layer.process_sample(&[1.0], &[0.0], &mut head, &mut out);
        assert_eq!(head, vec![1.0]);
        assert_eq!(out, vec![2.0]); // 1*post + input residual = 1 + 1
    }

    #[test]
    fn tanh_activation_matches_reference_value() {
        let mut layer = Layer::new(
            1,
            1,
            1,
            1,
            Activation::Tanh,
            false,
            vec![2.0],
            vec![0.5],
            vec![1.0],
            vec![3.0],
            vec![0.1],
        );
        let mut head = vec![0.0];
        let mut out = vec![0.0];
        // z = 2.0 ; post = tanh(2.0) ; out = 3*post + 0.1 + 0.5
        layer.process_sample(&[0.5], &[0.5], &mut head, &mut out);
        let post = 2.0_f32.tanh();
        assert!((head[0] - post).abs() < 1e-6, "head={}", head[0]);
        assert!((out[0] - (3.0 * post + 0.6)).abs() < 1e-6, "out={}", out[0]);
    }

    /// Block path reproduces the per-sample path for a full layer, gated and not,
    /// multi-channel, dilated, with a per-sample head seed carried in planar form.
    #[test]
    fn process_block_equals_process_sample_loop() {
        for gated in [false, true] {
            let channels = 3usize;
            let cond_sz = 2usize;
            let kernel = 3usize;
            let dilation = 4usize;
            let mid = if gated { 2 * channels } else { channels };
            let mk = |len: usize, salt: usize| -> Vec<f32> {
                (0..len)
                    .map(|i| (((i * 31 + salt * 7) % 29) as f32 - 14.0) * 0.07)
                    .collect()
            };
            let conv_w = mk(mid * channels * kernel, 1);
            let conv_b = mk(mid, 2);
            let mix_w = mk(mid * cond_sz, 3);
            let one_w = mk(channels * channels, 4);
            let one_b = mk(channels, 5);

            let total = 130usize;
            let inp: Vec<Vec<f32>> = (0..total)
                .map(|t| {
                    (0..channels)
                        .map(|c| ((t * 3 + c) as f32 * 0.21).sin())
                        .collect()
                })
                .collect();
            let cond: Vec<Vec<f32>> = (0..total)
                .map(|t| {
                    (0..cond_sz)
                        .map(|c| ((t * 5 + c) as f32 * 0.17).cos())
                        .collect()
                })
                .collect();
            let seed: Vec<Vec<f32>> = (0..total)
                .map(|t| (0..channels).map(|c| ((t + c) as f32) * 0.01).collect())
                .collect();

            let mk_layer = || {
                Layer::new(
                    channels,
                    cond_sz,
                    kernel,
                    dilation,
                    Activation::Tanh,
                    gated,
                    conv_w.clone(),
                    conv_b.clone(),
                    mix_w.clone(),
                    one_w.clone(),
                    one_b.clone(),
                )
            };

            // Reference: per-sample.
            let mut a = mk_layer();
            let mut out_ref = vec![vec![0.0; channels]; total];
            let mut head_ref = vec![vec![0.0; channels]; total];
            for t in 0..total {
                let mut head = seed[t].clone();
                let mut out = vec![0.0; channels];
                a.process_sample(&inp[t], &cond[t], &mut head, &mut out);
                out_ref[t] = out;
                head_ref[t] = head;
            }

            // Under test: block path in two chunks.
            let mut b = mk_layer();
            for (lo, len) in [(0usize, 70usize), (70, 60)] {
                let mut bin = vec![0.0; channels * len];
                let mut bcond = vec![0.0; cond_sz * len];
                let mut bhead = vec![0.0; channels * len];
                for lt in 0..len {
                    for c in 0..channels {
                        bin[c * len + lt] = inp[lo + lt][c];
                        bhead[c * len + lt] = seed[lo + lt][c];
                    }
                    for c in 0..cond_sz {
                        bcond[c * len + lt] = cond[lo + lt][c];
                    }
                }
                let mut bout = vec![0.0; channels * len];
                b.process_block(&bin, &bcond, &mut bhead, &mut bout, len);
                for lt in 0..len {
                    for c in 0..channels {
                        let go = bout[c * len + lt];
                        let gh = bhead[c * len + lt];
                        assert!(
                            (go - out_ref[lo + lt][c]).abs() < 1e-5,
                            "gated={gated} t{} c{c} out: got {go}, want {}",
                            lo + lt,
                            out_ref[lo + lt][c]
                        );
                        assert!(
                            (gh - head_ref[lo + lt][c]).abs() < 1e-5,
                            "gated={gated} t{} c{c} head: got {gh}, want {}",
                            lo + lt,
                            head_ref[lo + lt][c]
                        );
                    }
                }
            }
        }
    }
}
