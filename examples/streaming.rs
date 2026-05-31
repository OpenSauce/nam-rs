//! Block-wise streaming — the way you'd actually drive nam-rs from a real-time
//! audio host (JACK callback, VST3/CLAP `process()`, etc.).
//!
//! ```bash
//! cargo run --release --example streaming -- path/to/model.nam
//! ```
//!
//! The key pattern: do all the allocating work (parse + build) once, *off* the
//! audio thread, then call [`WaveNet::process_buffer`] once per audio block on the
//! hot thread. State carries across blocks, so feeding the signal block-by-block
//! gives bit-identical output to one monolithic call — this example proves it.

use nam_rs::{NamModel, WaveNet};

/// A typical host audio block size, in samples.
const BLOCK: usize = 128;

fn main() -> Result<(), nam_rs::Error> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: streaming <model.nam>");

    // --- Setup (off the audio thread): parsing and `WaveNet::new` allocate. ---
    let model = NamModel::from_file(&path)?;
    let mut amp = WaveNet::new(&model)?;
    let layer_arrays = match &model.config {
        nam_rs::ModelConfig::WaveNet(c) => c.layers.len(),
        nam_rs::ModelConfig::Lstm(_) => 0,
    };
    println!(
        "loaded {path}: {} layer-arrays, receptive field {} samples \
         (~{:.1} ms at {} Hz) — the startup transient before output settles",
        layer_arrays,
        amp.receptive_field(),
        amp.receptive_field() as f64 / model.sample_rate() * 1000.0,
        model.sample_rate(),
    );

    // A test signal long enough to clear the warmup transient.
    let signal: Vec<f32> = (0..8 * BLOCK)
        .map(|i| 0.5 * (i as f32 * 0.05).sin())
        .collect();

    // --- Hot path: process one block at a time, reusing state across blocks. ---
    // In a real host this loop body *is* your `process()` callback. It allocates
    // nothing (see tests/rt_safety.rs); the host owns the block buffer.
    let mut streamed = signal.clone();
    for block in streamed.chunks_mut(BLOCK) {
        amp.process_buffer(block);
    }

    // Streaming must equal one big call — `reset()` returns to silence first so the
    // two runs start from the same state.
    amp.reset();
    let mut oneshot = signal.clone();
    amp.process_buffer(&mut oneshot);

    let max_diff = streamed
        .iter()
        .zip(&oneshot)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "streamed {} samples in {}-sample blocks; max deviation from a single \
         whole-buffer call: {max_diff:e}",
        signal.len(),
        BLOCK,
    );
    assert_eq!(max_diff, 0.0, "block size must not change the output");

    Ok(())
}
