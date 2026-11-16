# Changelog

All notable changes to `mdb-runtime` (packages/mdb). The version number is set by the owner.

## [0.2.3] — 2026-11-16

Fixed: with `MDB_RECORDS` set, declaring custody (`custody.declare()` / `dbg.custody(...)`) before
the first sealed record stopped the sink file from opening, so records were sealed but never
written to it.

## [0.2.2] — 2026-11-16

Record schema `mdb-record-v3`. Every record sealed from this version on has a new id, even with
the same content, so anything that compares or cites ids must migrate them (`mdb records migrate`). Record content is unchanged: `examples/00-hello-tiny` prints the same
output except the forecast's record id (`f30d4d21ce1341f0` -> `d0c1d6c733c4bfa1` with torch 2.13
and transformers 5.13.1), and the v2 records of that run migrate to exactly the ids the v3 run
seals. Format, bounds and migration: [docs/RECORDS.md](docs/RECORDS.md).

### Changed (breaking)

- `seal_record` writes `mdb-record-v3`. The record you get (`seal_record`, `sink.records()`,
  `index.get_record`) has the same fields as before. Its stored form, which the id is the sha256
  of, keeps the execution profile as `execution_sha256` (the profile is stored once per project
  session and once per sink file) and a column's values as base64 bytes. A plain `json.loads` of
  a sink line or index row is the stored form; use `records.load_record(line, execution=...)` or
  `sink.records()` for the record.
- A record sink writes one `{"mdb_execution": "mdb-execution-v1", ...}` line per execution
  profile, before the first record that uses it. Code that counts sink lines should count
  records (`sink.records()`), not lines.
- Bounds: a v3 record's stored form may be 1 MiB (was 64 KiB) and hold 65,536 JSON values
  outside columns (was 4,096). The 64-item list bound and 4 KiB text bound are unchanged.
  Consequence: results that were compacted at 64 KiB now fit inline. A `Branches.seal` record
  of 25 arms x 6 probes x 40 values is no longer compact (its arm rows are in the record, not
  behind `arms_payload`). An `each` summary of 64 `whatif` items is 61,428 bytes (v2: 62,168,
  95% of the old cap; measured).

### Added

- Column claims: `records.nums(values, authority, proof=, unit=, shape=, dtype=, store=)`, many
  numbers under one label with a shape and a dtype (`float64`, `float32`, `int64`), and
  `records.nums_values(claim, store=)` to read one back as an array. Up to 65,536 values inline
  (about 8 bytes per float64 value stored, against about 70 for a `num`); more go to a tensor
  blob in the project store by sha256-checked pointer. No producer emits columns yet: existing
  record bodies are unchanged.
- `records.load_record`, `records.migrate_record`, `records.same_content`,
  `records.stored_record`, `Sink.executions()`.
- `mdb records migrate PATH [--out DIR] [--force]` and `mdb records verify PATH`
  (`mdb.core.migrate`): writes `NAME.v3.jsonl` and `NAME.v3.map.json` (v2 id -> v3 id, source
  and target sha256) for a record file or every record file under a directory; never in place.
  Every migrated record is read back from the new file and compared field by field with its
  source before the file is kept. Ids cited inside bodies keep their v2 value (the map resolves
  them). Measured on the repository's 607 stored v2 records (11 files: branches, pool,
  core-file): all migrated and read back equal in 3.4 s; stored size 8.39 MB -> 7.59 MB.
- `docs/RECORDS.md`.

### Kept

- v2 records (`mdb-record-v2`) are read and verified as before everywhere: sinks (v2 and v3 lines
  may share a file), project indexes, core files, `mdb serve`. Their 64 KiB bound and their ids
  are unchanged. Legacy `mdb-circuit-investigation-v1` records are read as before.
- Custody, seals and fingerprints: custody blocks, payload lines, trail fingerprints and receipts
  are unchanged; only record ids and the stored encoding changed.

### Measured (laptop CPU, tiny random Qwen2, one run each)

| | v2 (0.2.1) | v3 |
|---|---|---|
| `each` over 32 prompts (`whatif`), 32 item records + summary, plus a 28-row `row_table` trace record: stored bytes | 205,458 | 180,299 (-12%) |
| same, sink file bytes | 205,492 | 182,210 |
| one `whatif` record (mean of 32) | 5,198 | 4,458 |
| `trace` record (row table, 28 rows; payload 17,724 bytes, unchanged) | 2,291 | 1,552 |
| `prompt-set` summary (32 items) | 36,834 | 36,094 |
| seal all 34 / verify all 34 (ms, median of 20) | 16.5 / 11.9 | 12.9 / 12.7 |
| sink write 34 / read and verify 34 (ms) | 2.7 / 12.8 | 3.3 / 13.9 |
| a 28 x 32 grid as `num`s: bytes, seal, verify | 65,180 B, 8.9 ms, 6.7 ms | 65,218 B, 7.0 ms, 7.0 ms |
| the same grid as one column (`nums`) | not available | 10,040 B, 0.39 ms, 0.38 ms |
| read and verify the repository's 607 stored records (11 files; v2 originals vs their v3 migrations), median of 5 | 0.617 s | 0.668 s |

### Fixed

- `explain`'s first-order column and `mdb-lanes` `circuit_native.consumer_gradient` called
  `DecoderAdapter._cache` without the `layer_index` argument 0.2.1 added, and raised `TypeError`
  (22 test failures in `test_analysis_explain.py` / `test_analysis_casts.py` on 0.2.1).

## [0.2.1] — 2026-11-15

Stepping speedups from the 2026-11-14 architecture review (engine-dev), branch `mdb-step-speedups` from `mdb-v0.2.0`. Most of the change is in `saturn_pub` at the repo root, which mdb steps through. Trail fingerprints, receipts, record ids, tokens and logits are bit-identical to 0.2.0 on tiny Qwen2/Qwen3 CPU in bf16 and fp32 and in `00-hello-tiny`; trail fingerprints, tokens and logits are identical on Qwen2.5-1.5B on the GPU host, and example 04's records match field for field except the fields that differ between any two runs (custody time, payload path, per-arm seconds); evidence and numbers in `packages/mdb-experiments/results/step-speedups-20261115/`. Measured: a stepped row at T=1024 went from 802 to 98 ms on the laptop CPU (the review's probe config) and from 644 to 109 ms on Qwen2.5-1.5B on an RTX 4080; example 04's smoke items ran 2.7x faster.

### Changed (faster, same output)

- A layer transition no longer re-hashes the K/V layers it did not write: `saturn_pub.values.describe_remembered` reuses a tensor's descriptor while the tensor object, version counter, data pointer, shape, stride, dtype and device are unchanged (the same guard the `shares_state` contract uses). `describe` and `StateCut.verify` never use it and always re-hash.
- The finite check of the decoder state skips tensors it already found finite and that are unwritten since (same key).
- The native `DynamicCache` holds the state's K/V tensors by reference (a plain `DynamicLayer` concatenates, never writes) only while no foreign forward or forward-pre hook is registered on the decoder layer's modules or globally (`torch.nn.modules.module`), checked once per transition; mdb's own port and trace hooks (`saturn_pub.core.own_hook`) do not count. With a foreign hook, and for sliding-window, offloaded or lazily appended cache layers, the cache is filled by copy as on main. Writes that bypass the version counter (`.data`, numpy, dlpack, raw kernels) stay undetected, as on main.
- A grown K/V layer's chunks are version-guarded once, when they join (`_Appended.guard`), not re-walked by every cut; `held_bytes` visits each shared tensor and block form once; `each` records its per-item `bytes` with one walk (`Run.held_total()`, same value).
- A session reuses its last seal when it seals the very same state object again with the same parent, slot descriptors, model identity and execution contract (`_record`, then `capture`, then the next `_head`); mappings with plain string keys are sorted without JSON-encoding each key (the same order: tested against the canonical sort).
- The frozen-model guard compares a flat snapshot of the module tree (dict identities, item identities, tensor versions and pointers) in C-level passes, and falls back to the exact `_tensor_guards` comparison when anything differs.
- `symbols.hop_edge` and `symbols.gate_switch_forecast` cast the float64 MLP weights one layer at a time instead of caching every layer's (the 17 GB peak on 1.5B); `cache=` now holds only `c . W_down` per layer and the spectral permutation. Same values (identical JSON on two configs).

### Added

- `lookahead="auto"` (opt-in): one full-layer probe serves every port of a layer while nothing is staged for it; with an edit or cell edit staged it stops at the cursor as `"none"`. `status()["computed_past_cursor"]` says which probe ran. The default stays `"none"`: `lookahead` and `computed_past_cursor` are fields of forecast records and recipes, so changing the default would change record ids and the forecast's sealed-before-native-code guarantee (owner decision).

### Behaviour change

- On the by-reference path (no foreign hooks), a forward hook or port controller that writes any K/V tensor of the state in place before the native layer replaces it (including a pre-hook writing the pending layer's existing rows) now gets `InPlaceWrite` (and the session is poisoned until a restore), as for any other shared state tensor. Before, the write went to a private copy. A post-hook writing the freshly concatenated pending layer is still adopted, as on main. With a foreign hook registered the copying path is used and nothing is refused, as on main. Cell edits (`set_cell`) are copy-on-write and unaffected.

## [0.2.0] — 2026-11-14

Covers `main` (8c019ee, 2026-11-13) to `mdb-pre-release` (2026-11-14): the `mdb-engine-refactor` work of 2026-11-14 plus `claude/mdb-lanes-integration-20261114`. `pyproject.toml` and `mdb.__version__` are 0.2.0.

Why a minor bump. The package is pre-1.0, so a breaking change moves the minor version. This release adds many public objects and changes three defaults or behaviours that existing scripts can hit (see Changed). A patch bump would understate both.

### Added

- **Debugger objects, `docs/BRANCHES-POOLS-HOLDS.md`:**
  - `Run.set` takes REPL selectors and cross-run `ValueRef`s (`donor.ref(addr)`) carrying their origin pointer.
  - Holds: `run.hold(addr, source=)` for attention weights, norm scales (`norm_scale`, `attn_norm_scale`, `mlp_norm_scale`) and cells; `run.release`, `holds.final_held_logits`.
  - `Branches` (`dbg.branches`): named arms from one checkpoint, one shared floor, built-in nulls (`random`, `self`, `answer`), stability band, kernel path, sealed `branches` record with custody; `probes={name: fn}` read on every stepped arm before restore; batched mode for whole-cell and native-port arms (`a q k v n g sg u f A o m`) via `sweep.PortPatch`.
  - `Pool` (`dbg.pool`): sealed value sets with leave-one-out and class means, donor rules, matched-norm random, `save`/`load` with integrity checks.
  - Core files: `run.dump(path, row=, capture=)` and `dbg.run_from(path)` with a null check on load; cross-dtype reload is a labelled replica.
  - Transcoders: `sae.open_transcoders`, `TranscoderBasis` (mwhanna layout, `to_saelens()`), `components.transcoder_binding` as an MLP-write `ComponentBinding`.
  - Custody and sinks: `dbg.custody(script=, prereg=, kernel_path=)`, `dbg.records_to(...)`; env `MDB_SCRIPT`, `MDB_PREREG`, `MDB_KERNEL_PATH`, `MDB_RECORDS`.
- **Subspaces:** `analysis.subspace.SubspaceBinding` (basis + authority `derived-exact` / `fitted` / `supplied`), `readout_subspace`, `class_mean_subspace`, `from_pyvene`, `decompose()` into a `ComponentBinding`, `interchange` on `Branches`, `iia`.
- **Symbol table:** name-keyed stats, frames and JSON schema (`to_frame`, `to_arrow`, `to_parquet`, `json_schema`), `CandidateSet`, new columns (contrast, mask_delta, share, mass_nonsink, sink_ratio, role/slot/map_digest, labels, causal), components section, `LabelSymbol` port, replay by record id; `circuit_map` (Slot/Edge/Map, bind, freeze, cell census); `symbols.gate_switch_forecast` (first-order forecasts with and without gate switches, beside `hop_edge`); `cell_readout` / `CellTarget`.
- **Features and watches:** component bindings over residual SAE bases, feature addresses `feature[f]@t<r>.L<l>.h` and `@cache` cells, `FeatureCondition` in watch/break; multi-site watches (`FeatureCondition(text, contexts)` with `features.feature_sites({layer: basis}, row=r)`).
- **More verbs:** `Run.feed` + `feed` op, trace watch replay (`TraceCondition`), `rows.row_table`, `probe.suffix_sweep`.
- **Named symbols** from a transplant result, usable as addresses (`symbol NAME = ...`).
- **Interop:** `mdb.interop` (nnsight, TransformerLens, pyvene Qwen3 shim, circuit-tracer names, `circuit_tracer_graph` adapter, SAELens names and local-path loader, `names.tl_hook`), `interop.clone.shared_clone`; `docs/INTEROP.md`.
- **Sweeps:** planned layer calls printed, cap at 50000 without `max_calls=`, snapshot rollback, sealed `sweep` record.
- **Examples 00–07**: `00-hello-tiny` runs on a laptop CPU with no download (checked by `tests/test_example_hello_tiny.py`); 01–07 are real-model experiments, each as `.py` and a generated `.ipynb`.
- **Docs:** install line and CPU hello world at the top of the README, a glossary, "Two floors" (the 1-spacing verdict floor vs the 2-spacing margin noise floor), a comparison with TransformerLens, nnsight, SAELens, pyvene and circuit-tracer, `docs/README.md`, `docs/PROOF-IDS.md`, and a rewritten `docs/INSTALL.md`.
- `CHANGELOG.md` (this file).

### Changed

- `Branches.run(mode="batched")` defaults to `recheck="band"`: only arms with `1/k <= |dM|/floor <= k` are re-measured stepped (was: the caller's choice). Every batched row carries `recheck = {done, why}`. `recheck=0`, `"all"` and an int `k` keep their old meaning.
- `records_to(path)` refuses (`FileExistsError`) an existing non-empty file that this process did not open, unless `append=True` (`MDB_RECORDS_APPEND=1`). With no path, a directory or `{run}` in the name it writes a per-process, per-mrun-job file (`records-<MRUN_JOB_ID>-<time>-pid<pid>.jsonl`). Rerunning examples 04–07 locally into the same `out/` now needs `append=True`, a fresh `MDB_OUT` or deleting `out/records.jsonl` (example 03 already deletes it).
- A layer step on a model whose attention forward was rewritten after attach (nnsight `.source`) now refuses with the reason (`ValueError`) instead of stepping. Before, reads of `qr`/`kr`/`s`/`p` raised `KeyError` and edits at those ports could go unapplied.
- `whatif`, `whatif_many` and `forecast.seal` refuse a run with active holds. Measured on a tiny model: the prediction was the free continuation's while the held native change differed. Release the holds first, or measure the held arm with `Branches`.
- Verdicts: an effect within the floor needs native and emulated to agree on the sign; the worst-case rounding interval is information, not `unresolved`; per-stage containment on native operands.
- Prefill follow-ups: code uses `run.first_row` / `run.root`; records carry `procedure {prefill, first_row, prompt_rows}`; `Run.set` stages without a probe unless `check=True`.
- Lane-era code lives in `packages/mdb-lanes` (`mdb_lanes`); `mdb lanes` dispatches to the plug-in.
- **Package split:** the maintainers' internal experiments, results, audits, lane-era docs, unapproved loose examples, maintainer tools and internal trackers (`issues.jsonl`, `investigations.jsonl`, `features.json`, `PROVENANCE.json`) moved out of `packages/mdb` to `packages/mdb-experiments` in the repository; they are not part of `mdb-runtime`. The sdist went from 145 MB to under 2 MB; the wheel never contained them.

### Fixed

- Custody: record sinks no longer collide across shards. A compact `Branches.seal` record's `arms_payload` (`state: "absent"`) and any other store-less `bounded.put_json` payload is written to open sinks as an `mdb_payload` line (`Sink.load_json`), so per-arm probe values are sealed. Records and ids are unchanged.
- nnsight `.source` after attach (above).
- Forecasts under active holds (above).
- Twelve fixes from a review of the debugger core (pools, core files, holds, branches, custody, transcoders).
- From running the examples: SAELens local-path loader; the stop reason of a watch that is also a breakpoint; `forecast.seal(staged="include")`; `ResidualCapture` offline component binding; stability (kernel-sensitive band, per-model calibration, `kernel_path` on verdicts); `names.tl_hook`; circuit-tracer adapter (pinned environment only).
- A cache-cell value of shape `(d,)` is normalised to the block shape at arm creation.
- `symbols.table`: the `subspaces` parameter was shadowed by co-readout spans; `circuit_map.freeze` keys bindings by symbol name, not page position; `hop_edge` shadowing and `row_table` diagonal; `cell_readout` is affine under a V bias (Qwen2); `interop.tl.processed` reads the bridge flag; `custody._from_env` clears a stale error after a successful re-read.
- Release-gate fixes: opening a record sink twice on one path (a re-run notebook cell) returns the open sink instead of writing every record twice; a suffix-less path that does not exist yet (`Path("out/sinks")`) is a directory, so per-run naming applies (`out/records.jsonl` stays a file; a trailing `/` or an existing directory also work); a `watch` set when its address is already computed on the cursor's row takes that value as its baseline (as gdb does) and `watch` reports where the baseline was taken; the server's reduce route accepts the pointer an op result such as `print` returns; and a client that disconnects while its op or console line runs cancels that job (the op is interrupted and rolled back, a queued job never runs) instead of holding the worker. Example 01's launcher passes `MDB_NPAIRS` through for a smoke run.

### Known issues

- `until logits` after `start prefill=K` stops at the logits of the first stepped row, not the last row; name the row (`until t5.logits`).
- A role in a prompt set binds the first token of a multi-token word; the other tokens are not part of it.
- Ports of rows prefilled in one batched call cannot be read or edited; use `start prefill=K` so those rows are stepped.
- Every automated test uses a tiny random CPU model; real models are checked by hand on the maintainers' GPU.
- Pre-existing test failures (they also fail on the previous release): `test_two_logits_are_indexed_then_cast` (`stage_checks` casts the full logits row); `test_head_selected_patch_on_a_second_device[mps]` on Apple MPS; and, in some environments, `test_golden_transcript[predict_forecast]` and `test_the_script_names_every_registered_op`.
- `whatif` fails on Apple MPS (no float64); CPU and CUDA are the tested devices.
