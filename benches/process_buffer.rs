//! CPU-per-sample benchmark for the WaveNet hot path.
//!
//! Runs the committed reference model over a 512-sample block. To benchmark a
//! larger, more realistic model, point `MODEL` at a standard `.nam` file.
//!
//! ```bash
//! cargo bench
//! ```

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nam_rs::{NamModel, WaveNet};

const MODEL: &str = "tests/fixtures/reference.nam";
const BLOCK: usize = 512;

fn load() -> WaveNet {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL);
    let json = std::fs::read_to_string(path).expect("read model");
    let model = NamModel::from_json_str(&json).expect("parse model");
    WaveNet::new(&model).expect("build WaveNet")
}

fn bench_process_buffer(c: &mut Criterion) {
    let mut wavenet = load();
    let mut buffer = vec![0.0_f32; BLOCK];
    // Warm the ring buffers so we measure steady state.
    wavenet.process_buffer(&mut buffer);

    let mut group = c.benchmark_group("wavenet");
    group.throughput(Throughput::Elements(BLOCK as u64));
    group.bench_function("process_buffer/512", |b| {
        b.iter(|| wavenet.process_buffer(black_box(&mut buffer)));
    });
    group.finish();
}

criterion_group!(benches, bench_process_buffer);
criterion_main!(benches);
