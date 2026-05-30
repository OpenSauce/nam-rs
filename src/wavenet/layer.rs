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

use super::conv::Conv1d;
use crate::error::Error;

/// Pointwise activation applied after the dilated conv + mix-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Activation {
    Tanh,
    Relu,
    Sigmoid,
}

impl Activation {
    pub(super) fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "Tanh" => Ok(Self::Tanh),
            "ReLU" => Ok(Self::Relu),
            "Sigmoid" => Ok(Self::Sigmoid),
            other => Err(Error::UnsupportedActivation(other.to_string())),
        }
    }

    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::Sigmoid => sigmoid(x),
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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
                let g = sigmoid(self.block[c + self.channels]);
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

    #[test]
    fn unknown_activation_is_rejected() {
        assert!(Activation::from_name("Swish").is_err());
        assert_eq!(Activation::from_name("Tanh").unwrap(), Activation::Tanh);
    }
}
