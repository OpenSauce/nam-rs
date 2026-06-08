//! Feature-wise linear modulation (FiLM).
//!
//! A FiLM block modulates a signal by a per-channel scale (and optionally a
//! shift) computed from a conditioning signal. The scale/shift vector is the
//! output of a `1x1` convolution (`Conv1d` with kernel = 1, dilation = 1, bias)
//! over the conditioning input, producing `(shift ? 2 : 1) * input_dim` rows:
//! the top `input_dim` are the scale, the bottom `input_dim` (when `shift`) are
//! the shift. The modulation is then `out = in * scale (+ shift)`, elementwise.
//!
//! Ported from NeuralAmpModelerCore v0.5.3 `NAM/film.h` (`FiLM::Process` /
//! `Process_`). The conditioning `1x1` is the same `Conv1d` primitive used
//! everywhere else, so grouped FiLM reuses the block-diagonal grouped conv.
//!
//! Like every primitive here it exposes a per-sample path and a planar block
//! path (`[channel * n + t]`) that is bit-equivalent to `n` per-sample calls,
//! plus in-place (`_`-suffixed) variants. All scratch is pre-allocated in
//! [`FiLM::new`]; the `process_*` methods never allocate.

use super::conv::{Conv1d, MAX_BLOCK};

/// A FiLM block with all scratch buffers pre-allocated.
// Fields/methods become used as the per-sample and block paths land in the
// following tasks; allow dead_code for the skeleton-only state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(super) struct FiLM {
    /// `condition_dim -> (shift ? 2 : 1) * input_dim`, kernel 1, bias, grouped.
    cond_to_scale_shift: Conv1d,
    /// Width of the modulated signal (the scale width).
    input_dim: usize,
    /// Whether a shift term is added (scale-only when false).
    shift: bool,
    /// `out_rows`-wide scratch holding the conv'd scale (+ shift) for one sample.
    ss: Vec<f32>,
    /// Planar `[out_rows][MAX_BLOCK]` block-path twin of `ss`.
    ss_blk: Vec<f32>,
}

#[allow(dead_code)]
impl FiLM {
    /// Build a FiLM block. See the module-level contract for parameter semantics.
    pub(super) fn new(
        condition_dim: usize,
        input_dim: usize,
        shift: bool,
        groups: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let out_rows = if shift { 2 * input_dim } else { input_dim };
        let cond_to_scale_shift =
            Conv1d::new_grouped(condition_dim, out_rows, 1, 1, groups, weights, Some(bias));
        Self {
            cond_to_scale_shift,
            input_dim,
            shift,
            ss: vec![0.0; out_rows],
            ss_blk: vec![0.0; out_rows * MAX_BLOCK],
        }
    }

    /// Rows the internal Conv1x1 emits: `(shift ? 2 : 1) * input_dim`.
    #[cfg(test)]
    pub(super) fn out_rows(&self) -> usize {
        self.ss.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_scale_only_and_scale_shift_dims() {
        // scale-only: out rows = input_dim = 2, cond_dim = 3, groups = 1.
        let f = FiLM::new(3, 2, false, 1, vec![0.0; 2 * 3], vec![0.0; 2]);
        assert_eq!(f.out_rows(), 2);
        // scale+shift: out rows = 2*input_dim = 4.
        let g = FiLM::new(3, 2, true, 1, vec![0.0; 4 * 3], vec![0.0; 4]);
        assert_eq!(g.out_rows(), 4);
    }
}
