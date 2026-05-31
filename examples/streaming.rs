//! Block-wise streaming — the way you'd actually drive nam-rs from a real-time
//! audio host (JACK callback, VST3/CLAP `process()`, etc.).
//!
//! ```bash
//! cargo run --release --example streaming -- path/to/model.nam
//! ```
//!
//! The key pattern: do all the allocating work (parse + build) once, *off* the
//! audio thread, then call `Model::process_buffer` once per audio block on the
//! hot thread. State carries across blocks, so feeding the signal block-by-block
//! gives bit-identical output to one monolithic call — this example proves it.

use nam_rs::{Model, NamModel};

/// A typical host audio block size, in samples.
const BLOCK: usize = 128;

fn main() -> Result<(), nam_rs::Error> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: streaming <model.nam>");

    // --- Setup (off the audio thread): parsing and `Model::from_nam` allocate. ---
    let model = NamModel::from_file(&path)?;
    let mut amp = Model::from_nam(&model)?;
    let sr = model.sample_rate();
    // Describe the model in architecture-appropriate terms: WaveNet has a
    // receptive-field warmup, LSTM is recurrent (no warmup transient).
    let summary = match &amp {
        Model::WaveNet(w) => {
            let rf = w.receptive_field();
            format!(
                "WaveNet, receptive field {rf} samples (~{:.1} ms at {sr} Hz) — the \
                 startup transient before output settles",
                rf as f64 / sr * 1000.0,
            )
        }
        _ => "LSTM (recurrent) — no warmup transient".to_owned(),
    };
    println!("loaded {path}: {summary}");

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
