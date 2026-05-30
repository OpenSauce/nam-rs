//! A layer-array: an input rechannel, a stack of dilated [`Layer`]s, and a head
//! rechannel that maps the accumulated head signal to this array's `head_size`.
//!
//! Ported from NeuralAudio `WaveNetLayerArrayT`. The head accumulator is seeded
//! with the *incoming* head (the previous array's head output, or silence for the
//! first array), each layer adds its post-activation to it, and the head
//! rechannel projects the result:
//!
//! ```text
//! x = rechannel(input)
//! head_accum = head_in            // channels wide
//! for layer: x, head_accum = layer(x, condition, head_accum)
//! head_out = head_rechannel(head_accum)
//! ```

use super::conv::Conv1d;
use super::layer::Layer;

/// A stack of dilated layers sharing channel/kernel parameters.
#[derive(Debug, Clone)]
pub(super) struct LayerArray {
    rechannel: Conv1d,
    layers: Vec<Layer>,
    head_rechannel: Conv1d,
    channels: usize,
    head_size: usize,
    /// `channels`-wide scratch: current layer-signal.
    cur: Vec<f32>,
    /// `channels`-wide scratch: next layer-signal (ping-ponged with `cur`).
    nxt: Vec<f32>,
    /// `channels`-wide head accumulator.
    head_accum: Vec<f32>,
}

impl LayerArray {
    pub(super) fn new(
        input_size: usize,
        channels: usize,
        head_size: usize,
        rechannel_w: Vec<f32>,
        layers: Vec<Layer>,
        head_w: Vec<f32>,
        head_b: Option<Vec<f32>>,
    ) -> Self {
        Self {
            rechannel: Conv1d::new(input_size, channels, 1, 1, rechannel_w, None),
            layers,
            head_rechannel: Conv1d::new(channels, head_size, 1, 1, head_w, head_b),
            channels,
            head_size,
            cur: vec![0.0; channels],
            nxt: vec![0.0; channels],
            head_accum: vec![0.0; channels],
        }
    }

    pub(super) fn channels(&self) -> usize {
        self.channels
    }

    pub(super) fn head_size(&self) -> usize {
        self.head_size
    }

    /// Process one sample.
    ///
    /// - `input`: `input_size` wide.
    /// - `condition`: conditioning signal.
    /// - `head_in`: `channels`-wide incoming head (silence for the first array).
    /// - `head_out`: `head_size`-wide head output.
    /// - `array_out`: `channels`-wide output for the next array.
    pub(super) fn process_sample(
        &mut self,
        input: &[f32],
        condition: &[f32],
        head_in: &[f32],
        head_out: &mut [f32],
        array_out: &mut [f32],
    ) {
        self.rechannel.process_sample(input, &mut self.cur);
        self.head_accum.copy_from_slice(head_in);

        for i in 0..self.layers.len() {
            self.layers[i].process_sample(
                &self.cur,
                condition,
                &mut self.head_accum,
                &mut self.nxt,
            );
            std::mem::swap(&mut self.cur, &mut self.nxt);
        }

        self.head_rechannel
            .process_sample(&self.head_accum, head_out);
        array_out.copy_from_slice(&self.cur);
    }

    pub(super) fn reset(&mut self) {
        self.rechannel.reset();
        self.head_rechannel.reset();
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::{Activation, Layer};
    use super::*;

    fn relu_layer(
        channels: usize,
        conv_w: Vec<f32>,
        conv_b: Vec<f32>,
        mix_w: Vec<f32>,
        one_w: Vec<f32>,
        one_b: Vec<f32>,
    ) -> Layer {
        Layer::new(
            channels,
            1,
            1,
            1,
            Activation::Relu,
            false,
            conv_w,
            conv_b,
            mix_w,
            one_w,
            one_b,
        )
    }

    #[test]
    fn single_layer_array_rechannels_and_projects_head() {
        // rechannel r=1 ; layer: z=2*x+0.5+cond ; relu ; out=3*post+0.1+x ;
        // head_rechannel h=0.5, no bias ; head_in = silence.
        let layer = relu_layer(1, vec![2.0], vec![0.5], vec![1.0], vec![3.0], vec![0.1]);
        let mut array = LayerArray::new(1, 1, 1, vec![1.0], vec![layer], vec![0.5], None);

        let mut head_out = vec![0.0];
        let mut array_out = vec![0.0];
        array.process_sample(&[0.5], &[0.5], &[0.0], &mut head_out, &mut array_out);

        // cur=0.5 ; z=2.0 ; post=2.0 ; array_out=3*2+0.1+0.5=6.6 ; head=0.5*2=1.0
        assert_eq!(array_out, vec![6.6]);
        assert_eq!(head_out, vec![1.0]);
    }

    #[test]
    fn head_accumulates_across_stacked_layers() {
        // Two identity-ish layers (conv w=1,b=0 ; ignore cond ; one w=1,b=0 ; relu).
        // rechannel=1 ; head_rechannel=1, no bias ; head_in=silence.
        let l0 = relu_layer(1, vec![1.0], vec![0.0], vec![0.0], vec![1.0], vec![0.0]);
        let l1 = relu_layer(1, vec![1.0], vec![0.0], vec![0.0], vec![1.0], vec![0.0]);
        let mut array = LayerArray::new(1, 1, 1, vec![1.0], vec![l0, l1], vec![1.0], None);

        let mut head_out = vec![0.0];
        let mut array_out = vec![0.0];
        array.process_sample(&[2.0], &[0.0], &[0.0], &mut head_out, &mut array_out);

        // cur=2 ; L0: post=2, head=2, out=2+2=4 ; L1: post=4, head=6, out=4+4=8
        assert_eq!(array_out, vec![8.0]);
        assert_eq!(head_out, vec![6.0]); // 1 * (2 + 4)
    }

    #[test]
    fn incoming_head_is_carried_and_head_bias_applies() {
        // One layer that contributes 0 to head (relu of negative). head_in=[10].
        // head_rechannel weight=2, bias=[1] -> head_out = 2*10 + 1 = 21.
        let layer = relu_layer(1, vec![1.0], vec![0.0], vec![0.0], vec![1.0], vec![0.0]);
        let mut array =
            LayerArray::new(1, 1, 1, vec![1.0], vec![layer], vec![2.0], Some(vec![1.0]));

        let mut head_out = vec![0.0];
        let mut array_out = vec![0.0];
        // input=-5 -> cur=-5 ; z=-5 ; relu=0 ; head stays 10 ; out=0 + (-5) = -5
        array.process_sample(&[-5.0], &[0.0], &[10.0], &mut head_out, &mut array_out);

        assert_eq!(head_out, vec![21.0]);
        assert_eq!(array_out, vec![-5.0]);
    }

    #[test]
    fn channels_and_head_size_reported() {
        let layer = relu_layer(1, vec![1.0], vec![0.0], vec![0.0], vec![1.0], vec![0.0]);
        let array = LayerArray::new(1, 1, 2, vec![1.0], vec![layer], vec![1.0, 1.0], None);
        assert_eq!(array.channels(), 1);
        assert_eq!(array.head_size(), 2);
    }
}
