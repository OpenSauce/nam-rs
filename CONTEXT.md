# nam-rs

The vocabulary of NAM inference as this crate exposes it. Where NAM's own sources disagree
with each other — the Python trainer, NAM Core, and the TONE3000 catalogue all name things
slightly differently — this file records which sense wins here.

## The file and the engine

**`.nam` file**:
The exported artefact: an architecture name, a configuration, and one flat array of weights.
Inert on its own. Parsed into `NamModel`.
_Avoid_: Model file, profile, capture file

**Engine**:
The loaded, stateful thing that turns input samples into output samples. Constructed from a
parsed file, allocates once, then runs on the audio thread. Currently the type is named
`Model`; it is renamed to `Engine` in 0.5.0, because "model" already names the file and
because NAM Core calls its equivalent `DSP` rather than a model.
_Avoid_: Runtime (collides with async runtimes), inference object

**Architecture**:
Which network the file declares — WaveNet, LSTM, or SlimmableContainer. Chosen by the
trainer, read from the file, never selected by the caller.

**Submodel**:
One complete, standalone engine inside a SlimmableContainer. A container's submodels may
mix architectures. A submodel is a whole model, not a layer or a slice of one.
_Avoid_: Variant, sub-network, branch

**Slim size**:
The width dial on a SlimmableContainer — which submodel is active, as a CPU/quality
trade-off. The available settings are its breakpoints.
_Avoid_: Quality, level, tier

## Correctness

**Parity**:
Numeric agreement with the reference implementations: output within `1e-5` per sample for
the same file and input. This is the sense used in the design contract and in the parity
tests. Unqualified "parity" always means this.

**Feature coverage**:
How much of NAM Core's feature surface this crate implements. Distinct from Parity — a
model can be numerically exact on the features it does support while lacking others
entirely. Use this term for gap-against-NAM-Core work, never "parity".
_Avoid_: Feature parity

**Oracle**:
A reference implementation run to produce known-good output for a given file and input —
canonical Python NAM, or the vendored NAM Core. Fixtures come from an oracle; expected
values are never hand-authored.

## Signal

**Receptive field**:
How many past samples the output depends on. WaveNet's is set by its dilations; an LSTM's
is effectively none.

**Warmup**:
The startup transient while an engine's state fills from nothing. A property of the engine's
history, not of the audio.

**Prewarm**:
Deliberately settling an engine against silence before real audio, so the first block is
clean. NAM Core does this when it loads a model; this crate leaves it to the caller.

**Conditioning**:
The side-chain signal fed to a WaveNet's layer arrays alongside the audio. Usually the input
signal itself; a `condition_dsp` replaces it with the output of a nested engine.

**Block**:
One host buffer's worth of samples, processed in a single call. The real-time-safe unit of
work, and the fast path.
_Avoid_: Chunk, frame, window

**Real-time safe**:
Performs no heap allocation, no locking, and no system calls. A claim about the audio
thread, enforced by tests, not a claim about speed.

## Descriptive metadata

**Metadata**:
The file's descriptive block — who captured what, with which gear, when. Never reaches the
forward pass. Every field is optional and authored by a human, so treat it as a claim rather
than a fact.

**Loudness**:
How loud a model is against NAM's standardized input, as RMS dBFS. **Not LUFS** — the
trainer applies no K-weighting and no gating, and neither NAM's Python package nor NAM Core
mentions LUFS anywhere.
_Avoid_: LUFS, perceived loudness

**Calibration levels**:
The analog dBu values corresponding to 0 dBFS in and out — the numbers a host needs for
gain-staging. Distinct from Loudness, which describes the model rather than the interface.

**Gear type**:
What was captured and how much of the chain — amp, pedal, full rig. Free text, not an
enumeration: the NAM trainer and TONE3000 use overlapping but different vocabularies, and
real files carry values from neither.

## The tone3000-rs handoff

[`tone3000-rs`](https://github.com/OpenSauce/tone3000-rs) is the client for the TONE3000
catalogue, and its `Model` is a third distinct sense of the word: API metadata plus a
download URL, describing a file it does not contain. That crate's `CONTEXT.md` owns the
handoff vocabulary; this one does not restate it.
