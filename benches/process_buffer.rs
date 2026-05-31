//! Hot-path benchmark for `process_buffer`, the block entry point.
//!
//! Covers both architectures on *realistic* models (not the tiny identity ones), so
//! the numbers reflect a real amp-sim load. Throughput is set per-sample, so
//! Criterion reports time/sample directly (read the per-iteration time and divide by
//! `BLOCK`, or invert the reported elements/s).
//!
//! ```bash
//! cargo bench
//! ```
//!
//! To compare a block-kernel prototype against this per-sample baseline, run
//! `cargo bench` on the baseline, then again after the change — Criterion prints the
//! delta automatically.

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nam_rs::{Model, NamModel};

const BLOCK: usize = 512;

/// Realistic fixtures committed to the repo. The tiny identity models are too small
/// to be representative of a real inference load.
const MODELS: &[(&str, &str)] = &[
    ("wavenet_standard", "tests/fixtures/reference_standard.nam"),
    (
        "lstm_standard",
        "tests/fixtures/reference_lstm_standard.nam",
    ),
];

fn load(rel: &str) -> Model {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let json = std::fs::read_to_string(path).expect("read model");
    let model = NamModel::from_json_str(&json).expect("parse model");
    Model::from_nam(&model).expect("build model")
}

fn bench_process_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("process_buffer");
    group.throughput(Throughput::Elements(BLOCK as u64));

    for (name, rel) in MODELS {
        // `per_sample`: the per-sample loop (the old `process_buffer` body, the
        // baseline). `block`: the planar block kernel. Both share the same model and
        // run in the same bench invocation, so the reported delta is a clean A/B on
        // identical hardware and weights — no reliance on a saved historical baseline.
        let mut ps = load(rel);
        let mut block = load(rel);
        let mut buffer = vec![0.0_f32; BLOCK];
        // Warm both to steady state.
        for s in buffer.iter_mut() {
            *s = ps.process_sample(*s);
        }
        block.process_buffer(&mut buffer);

        group.bench_function(format!("{name}/per_sample"), |b| {
            b.iter(|| {
                for s in buffer.iter_mut() {
                    *s = ps.process_sample(black_box(*s));
                }
            });
        });
        group.bench_function(format!("{name}/block"), |b| {
            b.iter(|| block.process_buffer(black_box(&mut buffer)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_process_buffer);
criterion_main!(benches);
