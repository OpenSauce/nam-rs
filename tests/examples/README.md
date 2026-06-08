# Real A2 test captures (gitignored)

This directory holds real, trained NAM **A2** `.nam` captures used as parity inputs.
They are **gitignored** (commercial / licensing) — committed fixtures live in
`tests/fixtures/` and are generated from the MIT NAMCore oracle instead.

Tests that read this directory **skip cleanly when it is empty** (e.g. in CI).

Expected files (v0.7.0 SlimmableContainer of WaveNet submodels):

| File | Provenance |
|---|---|
| `Boosted-6505+-A2-Chugs-FullRig.nam` | user download |
| `Deluxe Reverb.nam` | user download |
| `MRSH JM50LD I Crunch2 FAT CAB.nam` | user download |

Drop any additional `.nam` files here to extend the real-capture smoke/parity tests.
