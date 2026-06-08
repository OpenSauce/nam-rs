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
fn conv_head_process_buffer_does_not_allocate() {
    // The multi-tap convolutional head (A2) takes a different build path than the
    // 1-tap A1 head (a per-array `Conv1d` ring instead of a plain rechannel). Its
    // ring is pre-allocated in `new`, so the hot path must still be alloc-free —
    // pin that here so a future head-path change can't regress it unnoticed.
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conv_head.nam");
    let json = std::fs::read_to_string(path).expect("conv_head.nam");
    let model = NamModel::from_json_str(&json).expect("parse model");
    let mut wn = WaveNet::new(&model).expect("build WaveNet");
    // Longer than MAX_BLOCK (1024) so the multi-chunk loop runs under the guard.
    let mut buffer = vec![0.0_f32; 2100];

    wn.process_buffer(&mut buffer); // warm up off-guard
    assert_no_alloc(|| {
        wn.process_buffer(&mut buffer);
    });
}

#[test]
fn post_stack_head_process_buffer_does_not_allocate() {
    // The post-stack head (an `activation -> Conv1d` chain after the arrays) runs on
    // the hot path. Its convs and the `head_scale` scaling scratch are pre-allocated
    // in `new`, so `process_buffer` must stay alloc-free — pin it here.
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/post_stack_head.nam");
    let json = std::fs::read_to_string(path).expect("post_stack_head.nam");
    let model = NamModel::from_json_str(&json).expect("parse model");
    let mut wn = WaveNet::new(&model).expect("build WaveNet");
    let mut buffer = vec![0.0_f32; 2100]; // > MAX_BLOCK

    wn.process_buffer(&mut buffer); // warm up off-guard
    assert_no_alloc(|| {
        wn.process_buffer(&mut buffer);
    });
}

#[test]
fn a2_synthetic_process_buffer_does_not_allocate() {
    // The richest A2 path: FiLM + BLENDED gating + layer1x1_post_film. Every FiLM owns
    // pre-allocated scratch, Gating is scratchless, and the Layer's `*_blk` buffers are
    // sized in `new`, so the hot path must allocate nothing. Skips if the fixture is
    // absent (e.g. the oracle could not be built in this environment).
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/a2_blended.nam");
    if !path.exists() {
        eprintln!("skip: a2_blended.nam absent");
        return;
    }
    let json = std::fs::read_to_string(&path).unwrap();
    let model = NamModel::from_json_str(&json).unwrap();
    let mut wn = WaveNet::new(&model).expect("a2_blended builds");
    let mut buffer = vec![0.1_f32; 2048];

    wn.process_buffer(&mut buffer); // warm up off-guard
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
