//! Sequential reader over a `.nam` flat weight blob, consumed in NAM's
//! `export_weights` order. Callers validate the total count up front, so `take`
//! never over-runs.

pub(crate) struct Reader<'a> {
    w: &'a [f32],
    i: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(w: &'a [f32]) -> Self {
        Self { w, i: 0 }
    }

    /// Take the next `n` weights as an owned `Vec`.
    pub(crate) fn take(&mut self, n: usize) -> Vec<f32> {
        let chunk = self.w[self.i..self.i + n].to_vec();
        self.i += n;
        chunk
    }
}
