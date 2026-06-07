//! Real-time safety harness: the audio hot path MUST NOT allocate.
//!
//! `WaveNet::process_buffer` is the function called from the audio thread. It may
//! not allocate, lock, or make syscalls. We enforce the allocation half here with
//! `assert_no_alloc`: any heap activity inside the guarded closure fails the test.
//!
//! `WaveNet::new` is allowed to allocate (it runs off the audio thread), so it sits
//! outside the guard.

use assert_no_alloc::*;
use nam_rs::{Lstm, Model, NamModel, WaveNet};

#[cfg(debug_assertions)]
#[global_allocator]
static ALLOC: AllocDisabler = AllocDisabler;

#[test]
fn process_buffer_does_not_allocate() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference.nam");
    let json = std::fs::read_to_string(path).expect("reference.nam (see tests/fixtures/README.md)");
    let model = NamModel::from_json_str(&json).expect("parse model");
    let mut wn = WaveNet::new(&model).expect("build WaveNet");
    // Longer than the block kernel's MAX_BLOCK (1024) so the multi-chunk loop in
    // `process_buffer` runs under the alloc guard, not just a single chunk.
    let mut buffer = vec![0.0_f32; 2100];

    // Warm up any lazy-but-bounded state outside the guard, then assert the steady
    // state allocates nothing.
    wn.process_buffer(&mut buffer);
    assert_no_alloc(|| {
        wn.process_buffer(&mut buffer);
    });
}

#[test]
fn lstm_process_buffer_does_not_allocate() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference_lstm.nam");
    let json = std::fs::read_to_string(path).expect("reference_lstm.nam");
    let model = NamModel::from_json_str(&json).expect("parse model");
    let mut net = Lstm::new(&model).expect("build Lstm");
    let mut buffer = vec![0.0_f32; 512];

    net.process_buffer(&mut buffer); // warm up off-guard
    assert_no_alloc(|| {
        net.process_buffer(&mut buffer);
    });
}

#[test]
fn slimmable_process_and_select_do_not_allocate() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/slimmable_container.nam");
    let json = std::fs::read_to_string(path).expect("slimmable_container.nam");
    let model = NamModel::from_json_str(&json).expect("parse container");
    let mut m = Model::from_nam(&model).expect("build container");
    let mut buffer = vec![0.0_f32; 512];

    // Warm up each submodel off-guard (select then process), so any lazy-but-bounded
    // state is initialized before the guard.
    for i in 0..m.as_slimmable().unwrap().len() {
        m.as_slimmable_mut().unwrap().select(i);
        m.process_buffer(&mut buffer);
    }

    assert_no_alloc(|| {
        // Switching the active submodel is a single index write — no allocation.
        m.as_slimmable_mut().unwrap().select(2);
        m.process_buffer(&mut buffer);
        m.as_slimmable_mut().unwrap().set_slim_size(0.0);
        m.process_buffer(&mut buffer);
        // reset() resets *every* submodel (iterate a Vec, fill pre-allocated buffers):
        // still allocation-free, so a host may call it on the audio thread.
        m.reset();
    });
}
