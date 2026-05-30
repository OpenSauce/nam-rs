//! Causal, dilated 1-D convolution with a pre-allocated ring buffer.
//!
//! This is the one stateful primitive in the WaveNet forward pass. Weight layout
//! matches NAM's `export_weights` (PyTorch `Conv1d`): `[out_ch][in_ch][kernel]`
//! row-major, followed by an optional `[out_ch]` bias.
//!
//! A 1x1 convolution (`kernel = 1`, `dilation = 1`) degenerates to a per-sample
//! matrix multiply; the same type is reused for the rechannel / input-mixer /
//! 1x1 / head-rechannel layers.

/// A dilated causal convolution that processes one sample at a time, keeping the
/// receptive-field history in a pre-allocated ring buffer (no allocation on the
/// audio thread).
#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    /// `[out_ch][in_ch][kernel]`, row-major.
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
    /// Ring of the most recent `ring_len` input columns, `in_ch` values each.
    ring: Vec<f32>,
    ring_len: usize,
    /// Index of the column written most recently.
    pos: usize,
}

impl Conv1d {
    pub(super) fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(weights.len(), out_ch * in_ch * kernel, "conv weight count");
        if let Some(b) = &bias {
            assert_eq!(b.len(), out_ch, "conv bias count");
        }
        let ring_len = (kernel - 1) * dilation + 1;
        Self {
            in_ch,
            out_ch,
            kernel,
            dilation,
            weights,
            bias,
            ring: vec![0.0; in_ch * ring_len],
            ring_len,
            pos: ring_len - 1,
        }
    }

    #[cfg(test)]
    pub(super) fn out_ch(&self) -> usize {
        self.out_ch
    }

    /// Push one input column (`in_ch` values) and write the convolution result
    /// (`out_ch` values) into `out`.
    pub(super) fn process_sample(&mut self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_ch);
        debug_assert_eq!(out.len(), self.out_ch);

        // Advance and store the newest column.
        self.pos = (self.pos + 1) % self.ring_len;
        let base = self.pos * self.in_ch;
        self.ring[base..base + self.in_ch].copy_from_slice(input);

        for o in 0..self.out_ch {
            let mut acc = self.bias.as_ref().map_or(0.0, |b| b[o]);
            let wo = o * self.in_ch * self.kernel;
            for k in 0..self.kernel {
                // Tap k reads the input at time offset -(kernel-1-k)*dilation.
                let back = (self.kernel - 1 - k) * self.dilation;
                let col = (self.pos + self.ring_len - back) % self.ring_len;
                let rbase = col * self.in_ch;
                for j in 0..self.in_ch {
                    acc += self.weights[wo + j * self.kernel + k] * self.ring[rbase + j];
                }
            }
            out[o] = acc;
        }
    }

    /// Clear the history to silence.
    pub(super) fn reset(&mut self) {
        self.ring.iter_mut().for_each(|x| *x = 0.0);
        self.pos = self.ring_len - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(conv: &mut Conv1d, xs: &[&[f32]]) -> Vec<Vec<f32>> {
        let mut out = vec![0.0; conv.out_ch()];
        xs.iter()
            .map(|x| {
                conv.process_sample(x, &mut out);
                out.clone()
            })
            .collect()
    }

    #[test]
    fn kernel2_dilation1_single_channel() {
        // out[t] = a*x[t-1] + b*x[t] + c, history starts at silence.
        // weights [out][in][k] = [a (oldest tap), b (current tap)].
        let mut conv = Conv1d::new(1, 1, 2, 1, vec![0.5, 2.0], Some(vec![0.1]));
        let got = run(&mut conv, &[&[1.0], &[2.0], &[3.0]]);
        assert_eq!(got, vec![vec![2.1], vec![4.6], vec![7.1]]);
    }

    #[test]
    fn kernel2_dilation2_skips_a_sample() {
        // out[t] = 0.5*x[t-2] + 2*x[t], no bias.
        let mut conv = Conv1d::new(1, 1, 2, 2, vec![0.5, 2.0], None);
        let got = run(&mut conv, &[&[1.0], &[2.0], &[3.0], &[4.0]]);
        assert_eq!(got, vec![vec![2.0], vec![4.0], vec![6.5], vec![9.0]]);
    }

    #[test]
    fn one_by_one_is_a_matmul() {
        // kernel=1, dilation=1: out[o] = sum_j W[o][j]*x[j] + bias[o].
        // W (2x2) = [[1,2],[3,4]], bias=[10,20].
        let mut conv = Conv1d::new(2, 2, 1, 1, vec![1.0, 2.0, 3.0, 4.0], Some(vec![10.0, 20.0]));
        let got = run(&mut conv, &[&[1.0, 1.0], &[2.0, 0.0]]);
        // x=[1,1]: [1+2+10, 3+4+20] = [13,27]; x=[2,0]: [2+10, 6+20]=[12,26]
        assert_eq!(got, vec![vec![13.0, 27.0], vec![12.0, 26.0]]);
    }

    /// Independent naive causal convolution oracle (single channel).
    fn naive(xs: &[f32], w: &[f32], dilation: usize) -> Vec<f32> {
        let k = w.len();
        (0..xs.len())
            .map(|t| {
                (0..k)
                    .map(|tap| {
                        let back = (k - 1 - tap) * dilation;
                        let x = if t >= back { xs[t - back] } else { 0.0 };
                        w[tap] * x
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn ring_buffer_matches_naive_over_long_signal() {
        // kernel=3, dilation=4 -> ring_len=9; feed 32 samples to wrap repeatedly.
        let w = vec![0.3, -1.1, 2.0];
        let mut conv = Conv1d::new(1, 1, 3, 4, w.clone(), None);
        let xs: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin()).collect();
        let got: Vec<f32> = xs
            .iter()
            .map(|&x| {
                let mut o = [0.0];
                conv.process_sample(&[x], &mut o);
                o[0]
            })
            .collect();
        let want = naive(&xs, &w, 4);
        for (g, e) in got.iter().zip(&want) {
            assert!((g - e).abs() < 1e-6, "got {g}, want {e}");
        }
    }

    #[test]
    fn reset_clears_history() {
        let mut conv = Conv1d::new(1, 1, 2, 1, vec![0.5, 2.0], None);
        let _ = run(&mut conv, &[&[1.0], &[2.0]]);
        conv.reset();
        // After reset, history is silence again: out[0] = 2*1 = 2.0
        let got = run(&mut conv, &[&[1.0]]);
        assert_eq!(got, vec![vec![2.0]]);
    }
}
