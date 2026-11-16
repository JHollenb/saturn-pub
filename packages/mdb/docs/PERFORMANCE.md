# Performance: what is fast, what it costs, what is left

What a stepped run costs in mdb, what you can switch on to make it cheaper, and the costs that
are known and not yet removed. Numbers are labelled: **measured** (run, with the setup named) or
**estimated** (arithmetic or reasoning, not run). Laptop numbers are single runs on a 4-thread
CPU: treat differences under about 15% as noise.

## What is fast now (mdb 0.2.1 and later)

A stepped run checks, on every layer transition, that the state it hands to the native layer is
the state it sealed: K/V, hidden state, tokens, the frozen model, the numeric scope. Before
0.2.1 those checks touched the whole K/V cache on every transition (hashing it, copying it into a
fresh native cache, checking it for non-finite values) and walked every parameter of the model,
so a stepped row cost O(layers x cache) and a prompt stepped from row 0 cost O(layers x rows²).
Since 0.2.1 each transition does work in proportion to what it wrote: the grown K/V row, the
hidden state, the logits. Outputs are bit-identical to before (trail fingerprints, receipts,
tokens, logits, record content).

Measured, laptop CPU, bf16, a random model with 28 layers, 2 K/V heads of 128, hidden size 256:

| case | 0.2.0 | 0.2.1 | speedup |
|---|---|---|---|
| one stepped row, 1024 tokens of context | 802 ms | 98 ms | 8.2x |
| one stepped row, 256 tokens of context | 451 ms | 83 ms | 5.4x |
| 96-token prompt stepped from row 0 (`prefill=0`), `until logits` | 27.2 s | 8.1 s | 3.4x |
| `Run.held_bytes()` after that run (7.8 MB held) | 9.1 s | 0.29 s | 32x |
| every port of one row, 512 tokens, `lookahead="none"` | 2.0 s | 0.66 s | 3.0x |
| same, `lookahead="layer"` | 0.65 s | 0.14 s | 4.7x |
| same, `lookahead="auto"` (new, opt-in) | n/a | 0.14 s | 14.5x against 0.2.0's `"none"` |

Measured, Qwen2.5-1.5B, bf16, RTX 4080:

| case | 0.2.0 | 0.2.1 | speedup |
|---|---|---|---|
| one stepped row, 1024 tokens of context | 644 ms | 109 ms | 5.9x |
| one stepped row, 256 tokens of context | 322 ms | 98 ms | 3.3x |
| 32-token prompt stepped from row 0, `until logits` | 8.3 s | 3.2 s | 2.6x |
| `held_bytes()` after it | 1.0 s | 0.04 s | 28x |
| every port of one row, 256 tokens, `"none"` / `"layer"` / `"auto"` | 1.55 / 0.41 / n/a s | 0.91 / 0.17 / 0.30 s | |
| `examples/04` smoke run, per prompt | 4.4 s, 2.7 s | 1.7 s, 1.0 s | 2.7x |

Analysis memory: `symbols.hop_edge` and `symbols.gate_switch_forecast` no longer keep float64
copies of every layer's MLP weights. Measured: 1343 MB to 0.3 MB held after the calls on an
8-layer, 1024 x 4096 model. Estimated for Qwen2.5-1.5B: about 15 GB to a few MB held, with a
transient peak of about 0.7 GB (one layer at a time). Peak GPU memory for stepped rows was
unchanged (measured, with one exception listed below).

What made the difference, in order of measured effect on a 1024-token stepped row: a tensor that
has not been written since its last hash is not hashed again (about 320 ms per row), nor checked
for non-finite values again (about 270 ms); the native cache holds the state's K/V tensors by
reference instead of a copy (about 43 ms); a state sealed once is not sealed again by the next
step, and string-keyed mappings sort without JSON-encoding each key (about 42 ms); the frozen
check compares a flat snapshot of the module tree instead of walking every parameter (about
86 ms: 184.7 ms per row without it against 98.3 ms with it, and 10.14 s against 5.17 s on the
`prefill=0` run, in the committed `cpu/ablate.jsonl`; the results README's smaller "+36 ms"
comes from a separate rerun that was not committed); grown K/V chunks are version-checked once when they join, not by every cut (18 ms per
row; 3.2 s on a 64-row `prefill=0` run).

## Records (mdb 0.3.0)

Record schema v3 ([RECORDS.md](RECORDS.md)) stores each session's execution profile once instead
of in every record, and adds column claims (`nums`) for tables of numbers. Measured, laptop CPU:

| | v2 | v3 |
|---|---|---|
| 32 `whatif` records, an `each` summary and a trace record (34 records): stored bytes | 205,458 | 180,299 |
| a 28 x 32 grid as labelled numbers (`num` each) | 65,180 B; seal 8.96 ms | 65,218 B; seal 7.08 ms |
| the same grid as one column (`nums`) | not available | 10,040 B; seal 0.38 ms |
| largest record | 64 KiB | 1 MiB |

Times over those 34 records (milliseconds for all of them, median of 20 runs; one rerun on 2026-11-16):

| | v2 | v3 |
|---|---|---|
| seal | 16.4 | 12.7 |
| verify | 12.0 | 12.8 |
| sink write | 2.7 | 3.4 |
| sink read and verify | 12.8 | 14.7 |
| `stored_bytes` | 0.9 | 1.4 |

Seal is faster in v3 (the execution profile is not encoded for every record), verify, sink
write and read are 6-24% slower (the v3 read walks the body once more to find columns), and
`stored_bytes` costs 0.5 ms more over the 34 records. The script and its output are
`packages/mdb-experiments/results/records-v3-20261116/` (`scripts/bench_records.py`,
`scripts/bench_grid.py`; `v2/` is main at 3d251bd, `v3/` this branch); run each with the tree's
`src` on `PYTHONPATH`.

## Choices you make

### `lookahead`: how much a move inside a layer computes

`dbg.run(tokens, lookahead=...)`:

- `"none"` (default): a move to a port runs the native layer only up to that port. No native code
  has computed anything after the cursor, which is what a sealed `forecast` relies on ("sealed
  before native code ran"). Walking every port of a layer re-runs the layer from its input for
  each port: measured 504 probes for one 512-token row of a 28-layer model.
- `"layer"`: one full-layer probe serves every port of the layer (28 probes for the same row).
  Native code has then computed ports past the cursor; `run.status()["computed_past_cursor"]`
  lists them.
- `"auto"` (opt-in): `"layer"` while nothing is staged for the pending layer, `"none"` as soon as
  an edit or cell edit is staged. Measured against `"none"`: 0.66 s to 0.14 s per row walk on the laptop CPU
  (4.8x), 0.91 s to 0.30 s on an RTX 4080 (3.0x). `computed_past_cursor` says which probe ran.

Use `"auto"` (or `"layer"`) for reading and analysis walks and in `each` templates that stage no
edit at the pending layer. Keep `"none"` when you seal forecasts at ports inside a layer.

Why `"none"` stays the default: a forecast sealed at a port promises that no native code has
computed the value it predicts. Under `"layer"` or `"auto"` that is not true for ports later in
the same layer, and `lookahead` and `computed_past_cursor` are fields of forecast records and of
replay recipes, so a different default would also change what existing scripts record. Record
ids alone are no longer a reason (0.3.0 changes every id anyway); the forecast guarantee is.

### Foreign hooks switch the native cache to a copy

The native cache holds the state's K/V tensors by reference only while no forward or
forward-pre hook other than mdb's own is registered on the decoder layer that runs (or
globally). A foreign hook (yours, nnsight's, TransformerLens', a profiler's) could write a K/V
tensor through `.data`, numpy or dlpack, which no version counter sees, so mdb then fills the
cache by copy, as before 0.2.1. Cost, measured on the laptop configuration above with the copy
forced: 98 ms to 141 ms per 1024-token row (+43 ms, +44%), +0.4 s on a 64-row `prefill=0` run.
On a GPU the copy is device memory bandwidth and allocator traffic per transition, O(layers x
cache) per row (estimated; not measured on its own). Remove hooks you do not need while
stepping; mdb's own port and trace hooks do not count.

## Known remaining costs and what is planned

Each entry: what it is, its cost, and the proposed approach. None changes results.

**Validity checks run per transition, not once per call.** The frozen-model check, the
numeric-scope check and the execution-profile check run about three times per layer transition
(before the native call, after it, and when the step is captured), and the determinism check
walks every module on every advance and every probe. Cost: the frozen check is 0.15 ms per call
on the laptop configuration (measured; it was 0.54 ms), so about 13 ms of a 98 ms row at 28
layers (estimated from that figure); the determinism walk was not measured on its own (a faster
walk that was tried was slower than PyTorch's `named_modules`). Proposed: check once per public
call (`step`, `until`, an `each` item) and flip a flag from `train()` / `register_*_hook` instead
of re-walking.

**Seals re-serialize constant parts.** Every seal re-encodes the model identity, the execution
profile and the port surface, which do not change during a session. Cost: not measured on its
own since the repeated seals of one state were removed (that removed about 42 ms per 1024-token
row); before that, the manifest step was 5.7 s of a 64-row `prefill=0` run (measured). Proposed:
serialize the constant parts once per adapter and splice the bytes into each seal.

**Sliding-window and other non-default cache layers are copied.** The by-reference cache covers
the plain dynamic cache layer only. Sliding-window layers (Mistral and Qwen2 configurations with
a sliding window), offloaded caches and lazily initialised layers are filled by copy, with the
cost given above for foreign hooks (estimated to be the same order; not measured on such a
model). Proposed: the same by-reference fill for each cache layer type whose `update` never
writes its inputs in place, checked per type.

**`lookahead` defaults to `"none"`.** See above. Proposed: keep the default; consider making
`each` templates and read-only analysis verbs use `"auto"` when they stage nothing, with
`computed_past_cursor` recorded as it is now. This needs an owner decision because it changes
what forecasts sealed inside such templates promise.

**`"auto"` was slower than `"layer"` on one GPU measurement.** 0.30 s against 0.17 s for the same
28 probes (measured once, RTX 4080). Not investigated.

**About 56 MB more peak GPU memory on 256-token stepped rows.** Measured on Qwen2.5-1.5B
(3090 MB against 3034 MB; also 14 MB more on a `"layer"` port walk); equal for 1024-token rows,
`prefill=0` and the `"none"` walk. The cases ran in sequence in one process, so this may be
memory of the previous case not yet released rather than a per-row cost. Not investigated.

**Writes that bypass PyTorch's version counter are not detected.** A hook or kernel that writes a
state tensor through `.data`, numpy, dlpack or a raw kernel is not seen by the in-place guard
(this was also true before 0.2.1). The by-reference cache is switched off while foreign hooks
are registered for that reason; a write from outside any hook is not caught. Cost of checking
would be a hash of the state per transition, which is what 0.2.1 removed. Proposed: an opt-in
`verify` mode that re-hashes the state after each native call, for debugging such code.

**A committed read at an earlier row rebuilds a session.** Reading a port at a past row (with
`prefill=0`) materialises the K/V chunks and re-checks the state: 9.8-16 ms per layer cold,
0.4 ms from the read cache (measured, before 0.2.1; not re-measured). Proposed: probe directly
on the stored cut without materialising.

**Per-commit bookkeeping grows with the trail.** Each commit snapshots every trail mark (0.70 s
over 3135 commits at about 3000 marks, measured before 0.2.1), and reads find their mark by a
linear scan. Small until a `prefill=0` run reaches thousands of marks. Proposed: snapshot only
the last mark and keep a `(row, port) -> mark` index.

**Edit tensors are re-hashed.** A port edit's digest is recomputed whenever its manifest or key is
read, and every restore re-hashes every edit on the trail (cost not measured). Proposed: compute
the digest once when the edit is made (its tensor is a private copy).

**Holds probe each covered layer before committing.** A hold over every layer at least doubles the
per-layer cost (estimated from the code path). Proposed: take the held value from the
committing pass where possible, or batch the probes.

**Some analysis paths still copy whole weights in float64.** `head_mix` keeps a float64 copy of a
layer's whole `o_proj` (about 0.5 GB for 28 layers of Qwen2.5-1.5B, estimated by arithmetic);
`_Grid` loops heads in Python; batched sweep arms clone untouched K/V layers once per arm.
Proposed: slice the head's columns before casting, one flattened scatter for the grid, `expand`
without `clone` for untouched layers.

**Records.** Bulk results that are not a table of numbers (a trace's per-row records, `each`
item results, a compact branches record's arm rows) are stored as JSON text in a store blob,
not in a binary column format. No record producer emits `nums` columns yet, so tables of
numbers inside records are still one `num` per value (about 70 bytes each). Listing a session's
records parses and verifies every record of the session, also when filtering by kind. Proposed:
producers that write grids switch to `nums`; a columnar payload for row tables; push kind
filters into the index query.

**The K/V cache is a list of chunks.** After `T` stepped rows a layer's K/V is `T` chunks, and
reading the whole layer concatenates them. Proposed, larger change: one preallocated K/V buffer
per layer, a cut is (buffer, length), and the state fingerprint becomes a hash chain over
appended rows, so sealing a transition hashes one row. This changes the seal format and would
need a new schema version.
