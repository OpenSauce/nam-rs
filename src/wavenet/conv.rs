//! Causal, dilated 1-D convolution with a pre-allocated history.
//!
//! This is the one stateful primitive in the WaveNet forward pass. Weight layout
//! matches NAM's `export_weights` (PyTorch `Conv1d`): `[out_ch][in_ch][kernel]`
//! row-major, followed by an optional `[out_ch]` bias.
//!
//! A 1x1 convolution (`kernel = 1`, `dilation = 1`) degenerates to a per-sample
//! matrix multiply; the same type is reused for the rechannel / input-mixer /
//! 1x1 / head-rechannel layers.
//!
//! Two equivalent entry points share one streaming history: the per-sample
//! [`Conv1d::process_sample`], and the block [`Conv1d::process_block`]. The
//! history is planar and linear, so the block path appends the block and reads
//! every tap straight out of it. Its kernel computes a tile of output channels
//! by time steps at once and keeps the partial sums in registers across all
//! taps and input channels; each output still sums its bias and then its
//! products in the per-sample order, so the two paths agree bit for bit. A
//! 1x1 convolution has no history and reads the block in place. The block
//! path is the lever behind `WaveNet::process_buffer`.

/// Largest block the planar kernel processes in one call; `process_buffer` chunks
/// longer inputs to this. Scratch is sized to it up front (off the audio thread).
pub(super) const MAX_BLOCK: usize = 1024;

/// A dilated causal convolution with a pre-allocated planar history (no
/// allocation on the audio thread).
///
/// The history is one linear row per input channel, `[in_ch][stride]`. The
/// newest `hist_len` columns sit just before the write cursor `w`; a block is
/// written at `w` and the taps read straight out of the row. When a block would
/// run past the end of the row, the last `hist_len` columns are moved to the
/// front first, so that copy happens once per `stride - hist_len` samples
/// instead of every block. This is the scheme of NAM Core's `RingBuffer`
/// ("rewind").
#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    /// `out_ch / groups`.
    out_per_group: usize,
    /// `in_ch / groups`.
    in_per_group: usize,
    /// `[out_ch][in_ch][kernel]`, row-major.
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
    /// `(kernel - 1) * dilation`: how many past columns a tap can reach.
    hist_len: usize,
    /// Planar `[in_ch][stride]` history and block staging. Empty for a 1x1
    /// convolution, which keeps no history.
    hist: Vec<f32>,
    stride: usize,
    /// The column the next input goes to. Always `>= hist_len`.
    w: usize,
    /// Run the AVX2 build of the block kernel. Set from [`avx2_available`]
    /// when the conv is built; tests clear it to compare the two builds. The
    /// kernel checks the CPU again before it trusts it.
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
}

/// Output channels one tile of the block kernel computes at once.
const TILE_OUT: usize = 4;
/// Time steps one tile of the block kernel computes at once: two SSE
/// registers per output in the portable build, two AVX registers in the AVX2
/// build. On aarch64, NEON has 32 vector registers where SSE has 16, so the
/// portable tile there is four registers per output: each weight loaded and
/// each step of index arithmetic serves twice the time steps. The summation
/// order does not depend on the width, so every width gives the same bits.
#[cfg(not(target_arch = "aarch64"))]
const TILE_T: usize = 8;
#[cfg(target_arch = "aarch64")]
const TILE_T: usize = 16;
#[cfg(target_arch = "x86_64")]
const TILE_T_AVX2: usize = 16;

/// Whether this CPU has AVX2, checked at run time so one binary runs on
/// every x86-64 CPU. The AVX2 builds of the hot loops enable AVX2 only, not
/// FMA: a fused multiply-add rounds once where the portable code rounds twice,
/// so FMA would change the output. Without it both builds do the same
/// operations in the same order and agree bit for bit.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

impl Conv1d {
    /// Dense (ungrouped) convolution — `groups = 1`.
    pub(super) fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Self {
        Self::new_grouped(in_ch, out_ch, kernel, dilation, 1, weights, bias)
    }

    /// Grouped convolution. Weights are **compact** (no off-block zeros), laid out in
    /// NAMCore order: group-major, then out-per-group, then in-per-group, then tap —
    /// i.e. `idx(g,i,j,k) = (((g*out_per_group + i)*in_per_group) + j)*kernel + k`.
    /// `groups = 1` reduces this to the dense `[out][in][kernel]` layout exactly.
    pub(super) fn new_grouped(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert!(groups >= 1, "conv groups must be >= 1");
        assert_eq!(in_ch % groups, 0, "in_ch must be divisible by groups");
        assert_eq!(out_ch % groups, 0, "out_ch must be divisible by groups");
        assert_eq!(
            weights.len(),
            out_ch * in_ch * kernel / groups,
            "grouped conv weight count"
        );
        if let Some(b) = &bias {
            assert_eq!(b.len(), out_ch, "conv bias count");
        }
        let hist_len = (kernel - 1)
            .checked_mul(dilation)
            .expect("conv history overflows usize");
        // At least a full block past the history, and at least the history
        // again, so the move to the front copies no more than it frees.
        let stride = if hist_len == 0 {
            0
        } else {
            hist_len
                .checked_add(hist_len.max(MAX_BLOCK))
                .expect("conv history overflows usize")
        };
        Self {
            in_ch,
            out_ch,
            kernel,
            dilation,
            out_per_group: out_ch / groups,
            in_per_group: in_ch / groups,
            weights,
            bias,
            hist_len,
            hist: vec![0.0; in_ch * stride],
            stride,
            w: hist_len,
            #[cfg(target_arch = "x86_64")]
            avx2: avx2_available(),
        }
    }

    // Geometry accessors consumed by the post-stack head (`head.rs`).
    pub(super) fn out_ch(&self) -> usize {
        self.out_ch
    }

    pub(super) fn in_ch(&self) -> usize {
        self.in_ch
    }

    #[cfg(test)]
    pub(super) fn kernel(&self) -> usize {
        self.kernel
    }

    /// Make room for `n` new columns at the write cursor.
    #[inline]
    fn make_room(&mut self, n: usize) {
        if self.hist_len == 0 || self.w + n <= self.stride {
            return;
        }
        let (h, s, w) = (self.hist_len, self.stride, self.w);
        for j in 0..self.in_ch {
            let row = j * s;
            self.hist.copy_within(row + w - h..row + w, row);
        }
        self.w = h;
    }

    /// Push one input column (`in_ch` values) and write the convolution result
    /// (`out_ch` values) into `out`.
    pub(super) fn process_sample(&mut self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_ch);
        debug_assert_eq!(out.len(), self.out_ch);

        if self.hist_len == 0 {
            self.tile_sample(input, 1, 0, out);
            return;
        }
        self.make_room(1);
        let (s, w) = (self.stride, self.w);
        for (j, &x) in input.iter().enumerate() {
            self.hist[j * s + w] = x;
        }
        self.tile_sample(&self.hist, s, w, out);
        self.w += 1;
    }

    /// One output column from the newest input column, which sits at `w` in
    /// rows of length `s` of `src`.
    #[inline]
    fn tile_sample(&self, src: &[f32], s: usize, w: usize, out: &mut [f32]) {
        let opg = self.out_per_group;
        let ipg = self.in_per_group;
        for o in 0..self.out_ch {
            let mut acc = self.bias.as_ref().map_or(0.0, |b| b[o]);
            // `in_base` = first input channel of this output's group; `wo` = start of
            // output o's rows in the compact buffer (same offset the block path uses).
            let g = o / opg;
            let in_base = g * ipg;
            let wo = o * ipg * self.kernel;
            for k in 0..self.kernel {
                let back = (self.kernel - 1 - k) * self.dilation;
                for jl in 0..ipg {
                    // jl is the local (in-group) input index; jl*kernel+k walks taps.
                    acc += self.weights[wo + jl * self.kernel + k]
                        * src[(in_base + jl) * s + w - back];
                }
            }
            out[o] = acc;
        }
    }

    /// Process `n` input columns at once, in **planar** layout: `block_in` is
    /// `in_ch * n` laid out `[channel * n + t]`, and `block_out` is `out_ch * n` the
    /// same way. Bit-for-bit equivalent to `n` calls of [`Self::process_sample`] and
    /// leaves the streaming history in the identical state, so the two entry points
    /// are freely interchangeable across calls: every output sums its bias and then
    /// its products in the same order as the per-sample path.
    ///
    /// `n` must be `<= MAX_BLOCK`; callers chunk longer runs. Allocation-free.
    pub(super) fn process_block(&mut self, block_in: &[f32], block_out: &mut [f32], n: usize) {
        debug_assert!(n <= MAX_BLOCK);
        debug_assert_eq!(block_in.len(), self.in_ch * n);
        debug_assert_eq!(block_out.len(), self.out_ch * n);
        if n == 0 {
            return;
        }

        if self.hist_len == 0 {
            self.tiles(block_in, n, 0, block_out, n);
            return;
        }
        self.make_room(n);
        let (s, w) = (self.stride, self.w);
        for j in 0..self.in_ch {
            self.hist[j * s + w..j * s + w + n].copy_from_slice(&block_in[j * n..j * n + n]);
        }
        self.tiles(&self.hist, s, w, block_out, n);
        self.w += n;
    }

    /// Every output of a block whose first column sits at `w` in rows of
    /// length `s` of `src`.
    fn tiles(&self, src: &[f32], s: usize, w: usize, block_out: &mut [f32], n: usize) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 && avx2_available() {
            // SAFETY: the CPU reports AVX2.
            unsafe { self.tiles_avx2(src, s, w, block_out, n) };
            return;
        }
        self.tiles_with::<TILE_T>(src, s, w, block_out, n);
    }

    /// [`Self::tiles`] compiled with AVX2 enabled.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn tiles_avx2(&self, src: &[f32], s: usize, w: usize, block_out: &mut [f32], n: usize) {
        self.tiles_with::<TILE_T_AVX2>(src, s, w, block_out, n);
    }

    #[inline(always)]
    fn tiles_with<const T: usize>(
        &self,
        src: &[f32],
        s: usize,
        w: usize,
        block_out: &mut [f32],
        n: usize,
    ) {
        let opg = self.out_per_group;
        let mut o = 0;
        while o < self.out_ch {
            // A tile never crosses a group, so all its outputs read the same inputs.
            let group_end = (o / opg + 1) * opg;
            if o + TILE_OUT <= group_end {
                self.tile::<TILE_OUT, T>(o, src, s, w, block_out, n);
                o += TILE_OUT;
            } else {
                self.tile::<1, T>(o, src, s, w, block_out, n);
                o += 1;
            }
        }
    }

    /// Outputs `o..o + O` of the block at `w` in `src`, `T` time steps at a
    /// time. The partial sums stay in registers across every tap and input
    /// channel.
    #[inline(always)]
    fn tile<const O: usize, const T: usize>(
        &self,
        o: usize,
        src: &[f32],
        s: usize,
        w: usize,
        block_out: &mut [f32],
        n: usize,
    ) {
        let ipg = self.in_per_group;
        let kernel = self.kernel;
        let in_base = (o / self.out_per_group) * ipg;
        let row_len = ipg * kernel;
        let weights = &self.weights[o * row_len..(o + O) * row_len];
        let mut bias = [0.0; O];
        if let Some(b) = &self.bias {
            bias.copy_from_slice(&b[o..o + O]);
        }

        // Full tiles first: the column is read straight from the row, with
        // no branch and no copy, so the sums stay in registers. The rest of
        // the block runs in steps of 4, then 1. Each output sums its bias
        // and products in the same order at any width, so the same bits.
        let full = n - n % T;
        let mut t = 0;
        while t < full {
            self.tile_step::<O, T>(
                o, in_base, row_len, weights, &bias, src, s, w, t, block_out, n,
            );
            t += T;
        }
        while t + 4 <= n {
            self.tile_step::<O, 4>(
                o, in_base, row_len, weights, &bias, src, s, w, t, block_out, n,
            );
            t += 4;
        }
        while t < n {
            self.tile_step::<O, 1>(
                o, in_base, row_len, weights, &bias, src, s, w, t, block_out, n,
            );
            t += 1;
        }
    }

    /// One `O` by `T` tile starting at time step `t`; `t + T <= n`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tile_step<const O: usize, const T: usize>(
        &self,
        o: usize,
        in_base: usize,
        row_len: usize,
        weights: &[f32],
        bias: &[f32; O],
        src: &[f32],
        s: usize,
        w: usize,
        t: usize,
        block_out: &mut [f32],
        n: usize,
    ) {
        debug_assert!(t + T <= n, "a tile step runs past the block");
        let kernel = self.kernel;
        let mut acc = [[0.0_f32; T]; O];
        for (a, &b) in acc.iter_mut().zip(bias) {
            *a = [b; T];
        }
        for k in 0..kernel {
            let back = (kernel - 1 - k) * self.dilation;
            for jl in 0..self.in_per_group {
                let at = (in_base + jl) * s + w + t - back;
                let column: &[f32; T] = src[at..at + T].try_into().unwrap();
                for (oo, a) in acc.iter_mut().enumerate() {
                    let wt = weights[oo * row_len + jl * kernel + k];
                    for (sum, &x) in a.iter_mut().zip(column) {
                        *sum += wt * x;
                    }
                }
            }
        }
        for (oo, a) in acc.iter().enumerate() {
            let row = (o + oo) * n + t;
            block_out[row..row + T].copy_from_slice(a);
        }
    }

    /// Clear the history to silence.
    pub(super) fn reset(&mut self) {
        self.hist.iter_mut().for_each(|x| *x = 0.0);
        self.w = self.hist_len;
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

    /// Expected values taken verbatim from NeuralAmpModelerCore's
    /// `tools/test/test_conv1d.cpp` (MIT) — an oracle independent of our numpy
    /// fixture generator.
    #[test]
    fn matches_namcore_conv1d_vectors() {
        // test_process_basic: weights {1,2}, input [1,2,3,4] -> [2,5,8,11].
        let mut basic = Conv1d::new(1, 1, 2, 1, vec![1.0, 2.0], None);
        assert_eq!(
            run(&mut basic, &[&[1.0], &[2.0], &[3.0], &[4.0]]),
            vec![vec![2.0], vec![5.0], vec![8.0], vec![11.0]]
        );

        // test_process_with_bias: weights {1,0}, bias 5, input [2,3] -> [5,7].
        let mut biased = Conv1d::new(1, 1, 2, 1, vec![1.0, 0.0], Some(vec![5.0]));
        assert_eq!(
            run(&mut biased, &[&[2.0], &[3.0]]),
            vec![vec![5.0], vec![7.0]]
        );

        // test_process_dilation: weights {1,2}, dilation 2, [1,2,3,4] -> [2,4,7,10].
        let mut dil = Conv1d::new(1, 1, 2, 2, vec![1.0, 2.0], None);
        assert_eq!(
            run(&mut dil, &[&[1.0], &[2.0], &[3.0], &[4.0]]),
            vec![vec![2.0], vec![4.0], vec![7.0], vec![10.0]]
        );
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

    /// The block path must reproduce the per-sample path exactly, including history
    /// carried across successive (differently sized) blocks. Planar in/out.
    #[test]
    fn process_block_equals_process_sample_loop() {
        // A few shapes: multi-channel, kernels 1..3, dilations that wrap the ring.
        let cases = [
            (1_usize, 1_usize, 1_usize, 1_usize),
            (2, 3, 1, 1),
            (3, 2, 2, 1),
            (2, 2, 3, 4),
            (4, 5, 2, 7),
        ];
        for (in_ch, out_ch, kernel, dilation) in cases {
            let wlen = out_ch * in_ch * kernel;
            // Deterministic pseudo-random weights/bias/input.
            let w: Vec<f32> = (0..wlen)
                .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.1)
                .collect();
            let bias: Vec<f32> = (0..out_ch).map(|o| (o as f32 + 1.0) * 0.05).collect();
            let total = 200_usize;
            let xs: Vec<Vec<f32>> = (0..total)
                .map(|t| {
                    (0..in_ch)
                        .map(|j| ((t * in_ch + j) as f32 * 0.31).sin())
                        .collect()
                })
                .collect();

            // Reference: per-sample.
            let mut a = Conv1d::new(
                in_ch,
                out_ch,
                kernel,
                dilation,
                w.clone(),
                Some(bias.clone()),
            );
            let mut want = vec![0.0; out_ch];
            let want_all: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| {
                    a.process_sample(x, &mut want);
                    want.clone()
                })
                .collect();

            // Under test: block path, split into uneven chunks to exercise history.
            let mut b = Conv1d::new(in_ch, out_ch, kernel, dilation, w, Some(bias));
            let chunks = [50usize, 1, 99, 50];
            let mut t0 = 0;
            for &len in &chunks {
                // Planar block_in: [channel][time].
                let mut bin = vec![0.0; in_ch * len];
                for (lt, x) in xs[t0..t0 + len].iter().enumerate() {
                    for (j, &v) in x.iter().enumerate() {
                        bin[j * len + lt] = v;
                    }
                }
                let mut bout = vec![0.0; out_ch * len];
                b.process_block(&bin, &mut bout, len);
                for lt in 0..len {
                    for o in 0..out_ch {
                        let got = bout[o * len + lt];
                        let exp = want_all[t0 + lt][o];
                        assert!(
                            (got - exp).abs() < 1e-5,
                            "shape {in_ch}x{out_ch} k{kernel} d{dilation} t{} o{o}: got {got}, want {exp}",
                            t0 + lt
                        );
                    }
                }
                t0 += len;
            }
        }
    }

    /// A history long enough to be moved to the front of its rows many times,
    /// fed in blocks of every size up to `MAX_BLOCK` and single samples, against
    /// a naive oracle that sums in the per-sample order, so bit for bit.
    #[test]
    fn history_survives_moving_to_the_front() {
        let (in_ch, out_ch, kernel, dilation) = (3_usize, 5_usize, 3_usize, 300_usize);
        let w: Vec<f32> = (0..out_ch * in_ch * kernel)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.1)
            .collect();
        let bias: Vec<f32> = (0..out_ch).map(|o| o as f32 * 0.25).collect();
        let total = 9_000_usize;
        let x = |t: usize, j: usize| ((t * in_ch + j) as f32 * 0.013).sin();
        let want = |t: usize, o: usize| {
            let mut acc = bias[o];
            for k in 0..kernel {
                let back = (kernel - 1 - k) * dilation;
                for j in 0..in_ch {
                    let v = if t >= back { x(t - back, j) } else { 0.0 };
                    acc += w[(o * in_ch + j) * kernel + k] * v;
                }
            }
            acc
        };

        let mut conv = Conv1d::new(
            in_ch,
            out_ch,
            kernel,
            dilation,
            w.clone(),
            Some(bias.clone()),
        );
        let chunks = [MAX_BLOCK, 1, 700, MAX_BLOCK, 3, 64, MAX_BLOCK, 1, 511];
        let mut t0 = 0;
        for &len in chunks.iter().cycle() {
            if t0 >= total {
                break;
            }
            let len = len.min(total - t0);
            if len == 1 {
                let input: Vec<f32> = (0..in_ch).map(|j| x(t0, j)).collect();
                let mut out = vec![0.0; out_ch];
                conv.process_sample(&input, &mut out);
                for (o, &got) in out.iter().enumerate() {
                    let exp = want(t0, o);
                    assert_eq!(got.to_bits(), exp.to_bits(), "t{t0} o{o}: {got} vs {exp}");
                }
            } else {
                let bin: Vec<f32> = (0..in_ch)
                    .flat_map(|j| (t0..t0 + len).map(move |t| x(t, j)))
                    .collect();
                let mut bout = vec![0.0; out_ch * len];
                conv.process_block(&bin, &mut bout, len);
                for lt in 0..len {
                    for o in 0..out_ch {
                        let (got, exp) = (bout[o * len + lt], want(t0 + lt, o));
                        assert_eq!(
                            got.to_bits(),
                            exp.to_bits(),
                            "t{} o{o}: {got} vs {exp}",
                            t0 + lt
                        );
                    }
                }
            }
            t0 += len;
        }
    }

    /// The AVX2 build of the block kernel gives the portable bits: dense and
    /// grouped convs, kernels up to 15, a 1x1 conv, and blocks that end in a
    /// partial tile. Skipped on a CPU without AVX2, where only one build runs.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_portable_kernels_agree_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        // (in_ch, out_ch, kernel, dilation, groups)
        let cases = [
            (8_usize, 8_usize, 6_usize, 239_usize, 1_usize),
            (16, 8, 3, 512, 1),
            (8, 8, 15, 13, 1),
            (8, 1, 16, 1, 1),
            (6, 4, 2, 3, 2),
            (4, 4, 3, 7, 4),
            (5, 7, 1, 1, 1),
        ];
        for (in_ch, out_ch, kernel, dilation, groups) in cases {
            let wlen = out_ch * in_ch * kernel / groups;
            let w: Vec<f32> = (0..wlen)
                .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.07)
                .collect();
            let bias: Vec<f32> = (0..out_ch).map(|o| o as f32 * 0.1 - 0.3).collect();
            let mut portable = Conv1d::new_grouped(
                in_ch,
                out_ch,
                kernel,
                dilation,
                groups,
                w.clone(),
                Some(bias.clone()),
            );
            portable.avx2 = false;
            let mut avx2 =
                Conv1d::new_grouped(in_ch, out_ch, kernel, dilation, groups, w, Some(bias));
            avx2.avx2 = true;
            let mut t0 = 0;
            for &len in [MAX_BLOCK, 37, 1, 255, MAX_BLOCK, 3, 128]
                .iter()
                .cycle()
                .take(14)
            {
                let bin: Vec<f32> = (0..in_ch * len)
                    .map(|i| ((t0 * in_ch + i) as f32 * 0.011).sin())
                    .collect();
                let mut a = vec![0.0; out_ch * len];
                let mut b = vec![0.0; out_ch * len];
                portable.process_block(&bin, &mut a, len);
                avx2.process_block(&bin, &mut b, len);
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "{in_ch}x{out_ch} k{kernel} d{dilation} g{groups} len{len} idx{i}"
                    );
                }
                t0 += len;
            }
        }
    }

    /// Every tile width gives the same bits, so the width a build picks (8,
    /// or 16 on aarch64 and in the AVX2 build) is a speed choice only.
    /// Checked directly on the kernel, on every architecture.
    #[test]
    fn tile_width_does_not_change_the_bits() {
        // (in_ch, out_ch, kernel, dilation, groups)
        let cases = [
            (8_usize, 8_usize, 6_usize, 239_usize, 1_usize),
            (16, 8, 3, 512, 1),
            (8, 1, 16, 1, 1),
            (6, 4, 2, 3, 2),
            (8, 8, 3, 5, 2), // grouped, full 4-output tiles in each group
            (5, 7, 1, 1, 1),
        ];
        for (in_ch, out_ch, kernel, dilation, groups) in cases {
            let wlen = out_ch * in_ch * kernel / groups;
            let w: Vec<f32> = (0..wlen)
                .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.07)
                .collect();
            let bias: Vec<f32> = (0..out_ch).map(|o| o as f32 * 0.1 - 0.3).collect();
            let conv = Conv1d::new_grouped(in_ch, out_ch, kernel, dilation, groups, w, Some(bias));
            for n in [1_usize, 7, 37, 128, MAX_BLOCK] {
                // A history of `hist_len` columns, then the block.
                let s = conv.hist_len + n;
                let src: Vec<f32> = (0..in_ch * s)
                    .map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * 2.0)
                    .collect();
                let w0 = conv.hist_len;
                let mut by4 = vec![0.0; out_ch * n];
                let mut by8 = vec![0.0; out_ch * n];
                let mut by16 = vec![0.0; out_ch * n];
                conv.tiles_with::<4>(&src, s, w0, &mut by4, n);
                conv.tiles_with::<8>(&src, s, w0, &mut by8, n);
                conv.tiles_with::<16>(&src, s, w0, &mut by16, n);
                for i in 0..out_ch * n {
                    assert_eq!(
                        (by4[i].to_bits(), by16[i].to_bits()),
                        (by8[i].to_bits(), by8[i].to_bits()),
                        "{in_ch}x{out_ch} k{kernel} d{dilation} g{groups} n{n} idx{i}"
                    );
                }
            }
        }
    }

    #[test]
    fn grouped_constructor_accepts_compact_weight_count() {
        // out=4, in=4, kernel=2, groups=2 -> out_per_group=2, in_per_group=2.
        // Compact weight count = 4*4*2 / 2 = 16. Plus 4 bias entries.
        let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let conv = Conv1d::new_grouped(4, 4, 2, 1, 2, w, Some(vec![0.0; 4]));
        assert_eq!(conv.out_ch(), 4);
    }

    #[test]
    fn grouped_process_block_equals_process_sample_loop() {
        // (in_ch, out_ch, kernel, dilation, groups) — all divisible by groups.
        let cases = [
            (4_usize, 4_usize, 1_usize, 1_usize, 2_usize),
            (4, 4, 2, 2, 2), // grouped, dilated
            (4, 4, 3, 1, 4), // depthwise (groups == in == out)
            (6, 4, 2, 1, 2), // out_per_group=2, in_per_group=3
            (4, 6, 2, 3, 2), // out_per_group=3, in_per_group=2, dilation wraps
        ];
        for (in_ch, out_ch, kernel, dilation, groups) in cases {
            let wlen = out_ch * in_ch * kernel / groups;
            let w: Vec<f32> = (0..wlen)
                .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.1)
                .collect();
            let bias: Vec<f32> = (0..out_ch).map(|o| (o as f32 + 1.0) * 0.05).collect();
            let total = 200_usize;
            let xs: Vec<Vec<f32>> = (0..total)
                .map(|t| {
                    (0..in_ch)
                        .map(|j| ((t * in_ch + j) as f32 * 0.31).sin())
                        .collect()
                })
                .collect();

            // Reference: per-sample.
            let mut a = Conv1d::new_grouped(
                in_ch,
                out_ch,
                kernel,
                dilation,
                groups,
                w.clone(),
                Some(bias.clone()),
            );
            let mut want = vec![0.0; out_ch];
            let want_all: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| {
                    a.process_sample(x, &mut want);
                    want.clone()
                })
                .collect();

            // Under test: block path, uneven chunks to exercise history.
            let mut b = Conv1d::new_grouped(in_ch, out_ch, kernel, dilation, groups, w, Some(bias));
            let chunks = [50usize, 1, 99, 50];
            let mut t0 = 0;
            for &len in &chunks {
                let mut bin = vec![0.0; in_ch * len];
                for (lt, x) in xs[t0..t0 + len].iter().enumerate() {
                    for (j, &v) in x.iter().enumerate() {
                        bin[j * len + lt] = v;
                    }
                }
                let mut bout = vec![0.0; out_ch * len];
                b.process_block(&bin, &mut bout, len);
                for lt in 0..len {
                    for o in 0..out_ch {
                        let got = bout[o * len + lt];
                        let exp = want_all[t0 + lt][o];
                        assert!(
                            (got - exp).abs() < 1e-5,
                            "shape {in_ch}x{out_ch} k{kernel} d{dilation} g{groups} t{} o{o}: got {got}, want {exp}",
                            t0 + lt
                        );
                    }
                }
                t0 += len;
            }
        }
    }

    #[test]
    fn grouped_1x1_is_block_diagonal_matmul() {
        // in=4, out=4, kernel=1, groups=2. out_per_group = in_per_group = 2.
        // Block-diagonal: out[0,1] read in[0,1]; out[2,3] read in[2,3].
        // Compact weights (group-major, out_pg, in_pg, tap):
        //   g0: W[[1,2],[3,4]]   g1: W[[5,6],[7,8]]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bias = vec![10.0, 20.0, 30.0, 40.0];
        let mut conv = Conv1d::new_grouped(4, 4, 1, 1, 2, w, Some(bias));
        let mut out = vec![0.0; 4];
        conv.process_sample(&[1.0, 1.0, 1.0, 1.0], &mut out);
        // out0 = 1*1 + 2*1 + 10 = 13 ; out1 = 3*1 + 4*1 + 20 = 27
        // out2 = 5*1 + 6*1 + 30 = 41 ; out3 = 7*1 + 8*1 + 40 = 55
        assert_eq!(out, vec![13.0, 27.0, 41.0, 55.0]);

        // Cross-group isolation: perturbing in[2,3] must not change out[0,1].
        let mut out2 = vec![0.0; 4];
        conv.reset();
        conv.process_sample(&[1.0, 1.0, 9.0, -9.0], &mut out2);
        assert_eq!(out2[0], 13.0);
        assert_eq!(out2[1], 27.0);
    }

    #[test]
    fn grouped_depthwise_kernel2() {
        // Depthwise: in=out=groups=2, kernel=2 -> out_per_group=in_per_group=1.
        // Channel 0 weights {1,2}; channel 1 weights {10,20}. No bias.
        // Compact order: g0(i0,j0): taps {1,2}; g1(i0,j0): taps {10,20}.
        let w = vec![1.0, 2.0, 10.0, 20.0];
        let mut conv = Conv1d::new_grouped(2, 2, 2, 1, 2, w, None);
        let mut out = vec![0.0; 2];
        // t=0, history silent: ch0 = 2*1 = 2 ; ch1 = 20*3 = 60
        conv.process_sample(&[1.0, 3.0], &mut out);
        assert_eq!(out, vec![2.0, 60.0]);
        // t=1: ch0 = 1*1 + 2*5 = 11 ; ch1 = 10*3 + 20*7 = 170
        conv.process_sample(&[5.0, 7.0], &mut out);
        assert_eq!(out, vec![11.0, 170.0]);
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
