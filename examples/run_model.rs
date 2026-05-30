//! Run a `.nam` model over a test signal and print output statistics.
//!
//! ```bash
//! cargo run --release --example run_model -- path/to/model.nam [num_samples]
//! ```
//!
//! Useful for smoke-testing that a real model loads and produces sane (finite,
//! bounded) output before wiring nam-rs into a host.

use nam_rs::{NamModel, WaveNet};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: run_model <model.nam> [num_samples]");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);

    let json = std::fs::read_to_string(&path).expect("read model file");
    let model = NamModel::from_json_str(&json).expect("parse .nam");
    println!(
        "loaded {path}: arch={} version={} sample_rate={}",
        model.architecture,
        model.version,
        model.sample_rate()
    );

    let mut wavenet = WaveNet::new(&model).expect("build WaveNet");

    // Deterministic test signal: an impulse followed by two decaying tones.
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32;
            let env = (-(t / n as f32) * 3.0).exp();
            (0.5 * (t * 0.05).sin() + 0.3 * (t * 0.0131).sin()) * env
        })
        .collect();
    if !buf.is_empty() {
        buf[0] = 1.0;
    }
    let in_peak = buf.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));

    wavenet.process_buffer(&mut buf);

    let any_nan = buf.iter().any(|x| !x.is_finite());
    let out_peak = buf.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    let rms = (buf.iter().map(|x| x * x).sum::<f32>() / buf.len().max(1) as f32).sqrt();

    println!("samples={n} in_peak={in_peak:.4} out_peak={out_peak:.6} out_rms={rms:.6} any_nan={any_nan}");
    let head: Vec<f32> = buf.iter().take(8).copied().collect();
    println!("first 8 outputs: {head:?}");
}
