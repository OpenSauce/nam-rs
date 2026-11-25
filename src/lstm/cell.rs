//! One LSTM layer: gates from `W·[x; h] + b`, then the standard cell update.
//!
//! `W` is the combined `(4H) × (input_size + H)` matrix in NAM `export_weights`
//! order, row-major, gate order **i, f, g, o** (PyTorch). The exporter pre-sums
//! `bias_ih + bias_hh`, so there is a single bias vector of length `4H`.

use crate::wavenet::{sigmoid, tanh};

#[derive(Debug)]
pub(crate) struct LstmCell {
    hidden_size: usize,
    input_size: usize,
    /// `W` transposed: (input_size + H) × (4H), so one input column updates
    /// every gate with a contiguous, vectorizable row.
    wt: Vec<f32>,
    /// 4H.
    b: Vec<f32>,
    /// Exported initial hidden/cell (length H each); `reset()` restores to these.
    h0: Vec<f32>,
    c0: Vec<f32>,
    // --- mutable state + scratch (pre-allocated) ---
    h: Vec<f32>,     // H
    c: Vec<f32>,     // H
    xh: Vec<f32>,    // input_size + H
    gates: Vec<f32>, // 4H
}

impl LstmCell {
    pub(crate) fn new(
        input_size: usize,
        hidden_size: usize,
        w: Vec<f32>,
        b: Vec<f32>,
        h0: Vec<f32>,
        c0: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(w.len(), 4 * hidden_size * (input_size + hidden_size));
        debug_assert_eq!(b.len(), 4 * hidden_size);
        debug_assert_eq!(h0.len(), hidden_size);
        debug_assert_eq!(c0.len(), hidden_size);
        let row = input_size + hidden_size;
        let mut wt = vec![0.0; w.len()];
        for g in 0..4 * hidden_size {
            for k in 0..row {
                wt[k * 4 * hidden_size + g] = w[g * row + k];
            }
        }
        Self {
            hidden_size,
            input_size,
            wt,
            b,
            h: h0.clone(),
            c: c0.clone(),
            h0,
            c0,
            xh: vec![0.0; input_size + hidden_size],
            gates: vec![0.0; 4 * hidden_size],
        }
    }

    /// Advance one timestep with input `x` (length `input_size`); returns the new
    /// hidden state `h` (length `hidden_size`). Allocation-free.
    pub(crate) fn process(&mut self, x: &[f32]) -> &[f32] {
        let h = self.hidden_size;
        let row = self.input_size + h;

        // xh = [x ; h_prev]
        self.xh[..self.input_size].copy_from_slice(x);
        self.xh[self.input_size..].copy_from_slice(&self.h);

        // gates = W · xh + b. Column by column, so each gate still sums its
        // bias and then its products in order k = 0, 1, ...
        let gates = &mut self.gates[..4 * h];
        gates.copy_from_slice(&self.b);
        for (k, &x) in self.xh[..row].iter().enumerate() {
            let col = &self.wt[k * 4 * h..(k + 1) * 4 * h];
            for (g, &w) in gates.iter_mut().zip(col) {
                *g += w * x;
            }
        }

        // cell update, gate order i,f,g,o
        let (ifg, o) = gates.split_at(3 * h);
        let (i_f, g_) = ifg.split_at(2 * h);
        let (i, f) = i_f.split_at(h);
        for j in 0..h {
            let c = sigmoid(f[j]) * self.c[j] + sigmoid(i[j]) * tanh(g_[j]);
            self.c[j] = c;
            self.h[j] = sigmoid(o[j]) * tanh(c);
        }
        &self.h
    }

    /// Restore `h`/`c` to the exported initial state.
    pub(crate) fn reset(&mut self) {
        self.h.copy_from_slice(&self.h0);
        self.c.copy_from_slice(&self.c0);
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cell_matches_hand_computed_step() {
        // H=1, input_size=1. Row-major W (4H x (in+H) = 4x2), gate order i,f,g,o.
        // W rows: i:[1,0] f:[0,0] g:[2,0] o:[0,0]; bias=[0,0,0,0]; h0=c0=0.
        let w = vec![
            1.0, 0.0, // i: w_xi=1, w_hi=0
            0.0, 0.0, // f
            2.0, 0.0, // g: w_xg=2
            0.0, 0.0, // o
        ];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let mut cell = LstmCell::new(1, 1, w, b, vec![0.0], vec![0.0]);

        // x=0.5: i=sig(0.5)=0.622459, f=sig(0)=0.5, g=tanh(1.0)=0.761594, o=sig(0)=0.5
        // c = f*0 + i*g = 0.622459*0.761594 = 0.474061
        // h = o*tanh(c) = 0.5*tanh(0.474061) = 0.5*0.441510 = 0.220755
        // (hand arithmetic; 1e-3 tol — the 1e-5 gate is the parity test vs canonical NAM)
        let h = cell.process(&[0.5]);
        assert!((h[0] - 0.2208).abs() < 1e-3, "got {}", h[0]);
    }

    #[test]
    fn reset_restores_initial_state() {
        let w = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let b = vec![0.0; 4];
        let mut cell = LstmCell::new(1, 1, w, b, vec![0.0], vec![0.0]);
        let first = cell.process(&[0.5])[0];
        cell.process(&[0.5]); // advance state
        cell.reset();
        let after = cell.process(&[0.5])[0];
        assert!((first - after).abs() < 1e-7, "{first} vs {after}");
    }
}
