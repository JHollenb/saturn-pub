# Changelog

All notable changes to `mdb-runtime` (packages/mdb). The version number is set by the owner; the section below is a draft.

## [Unreleased] — DRAFT, proposed 0.2.0 (not bumped, not tagged)

Covers `main` (8c019ee, 2026-11-13) to `mdb-pre-release` (2026-11-14): the `mdb-engine-refactor` work of 2026-11-14 plus `claude/mdb-lanes-integration-20261114`. `pyproject.toml` and `mdb.__version__` still say 0.1.0.

Why a minor bump. The package is pre-1.0, so a breaking change moves the minor version. This release adds many public objects and changes three defaults or behaviours that existing scripts can hit (see Changed). A patch bump would understate both.

### Added

- **Debugger objects (gap audit 2026-11-14), `docs/BRANCHES-POOLS-HOLDS.md`:**
  - `Run.set` takes REPL selectors and cross-run `ValueRef`s (`donor.ref(addr)`) carrying their origin pointer.
  - Holds: `run.hold(addr, source=)` for attention weights, norm scales (`norm_scale`, `attn_norm_scale`, `mlp_norm_scale`) and cells; `run.release`, `holds.final_held_logits`.
  - `Branches` (`dbg.branches`): named arms from one checkpoint, one shared floor, built-in nulls (`random`, `self`, `answer`), stability band, kernel path, sealed `branches` record with custody; `probes={name: fn}` read on every stepped arm before restore; batched mode for whole-cell and native-port arms (`a q k v n g sg u f A o m`) via `sweep.PortPatch`.
  - `Pool` (`dbg.pool`): sealed value sets with leave-one-out and class means, donor rules, matched-norm random, `save`/`load` with integrity checks.
  - Core files: `run.dump(path, row=, capture=)` and `dbg.run_from(path)` with a null check on load; cross-dtype reload is a labelled replica.
  - Transcoders: `sae.open_transcoders`, `TranscoderBasis` (mwhanna layout, `to_saelens()`), `components.transcoder_binding` as an MLP-write `ComponentBinding`.
  - Custody and sinks: `dbg.custody(script=, prereg=, kernel_path=)`, `dbg.records_to(...)`; env `MDB_SCRIPT`, `MDB_PREREG`, `MDB_KERNEL_PATH`, `MDB_RECORDS`.
- **Subspaces:** `analysis.subspace.SubspaceBinding` (basis + authority `derived-exact` / `fitted` / `supplied`), `readout_subspace`, `class_mean_subspace`, `from_pyvene`, `decompose()` into a `ComponentBinding`, `interchange` on `Branches`, `iia`.
- **Symbol table:** name-keyed stats, frames and JSON schema (`to_frame`, `to_arrow`, `to_parquet`, `json_schema`), `CandidateSet`, new columns (contrast, mask_delta, share, mass_nonsink, sink_ratio, role/slot/map_digest, labels, causal), components section, `LabelSymbol` port, replay by record id; `circuit_map` (Slot/Edge/Map, bind, freeze, cell census); `symbols.gate_switch_forecast` (F1/F2/F3 gate-switch forecasts beside `hop_edge`); `cell_readout` / `CellTarget`.
- **Features and watches:** component bindings over residual SAE bases, feature addresses `feature[f]@t<r>.L<l>.h` and `@cache` cells, `FeatureCondition` in watch/break; multi-site watches (`FeatureCondition(text, contexts)` with `features.feature_sites({layer: basis}, row=r)`).
- **B2 verbs:** `Run.feed` + `feed` op, trace watch replay (`TraceCondition`), `rows.row_table`, `probe.suffix_sweep`.
- **Named symbols** from a transplant result, usable as addresses (`symbol NAME = ...`).
- **Interop:** `mdb.interop` (nnsight, TransformerLens, pyvene Qwen3 shim, circuit-tracer names, `circuit_tracer_graph` adapter, SAELens names and local-path loader, `names.tl_hook`), `interop.clone.shared_clone`; `docs/INTEROP.md`.
- **Sweeps:** planned layer calls printed, cap at 50000 without `max_calls=`, snapshot rollback, sealed `sweep` record.
- **Examples 01–07** (owner-approved), each as `.py` and a generated `.ipynb`.
- `CHANGELOG.md` (this file).

### Changed

- `Branches.run(mode="batched")` defaults to `recheck="band"`: only arms with `1/k <= |dM|/floor <= k` are re-measured stepped (was: the caller's choice). Every batched row carries `recheck = {done, why}`. `recheck=0`, `"all"` and an int `k` keep their old meaning.
- `records_to(path)` refuses (`FileExistsError`) an existing non-empty file that this process did not open, unless `append=True` (`MDB_RECORDS_APPEND=1`). With no path, a directory or `{run}` in the name it writes a per-process, per-mrun-job file (`records-<MRUN_JOB_ID>-<time>-pid<pid>.jsonl`). Rerunning examples 04–07 locally into the same `out/` now needs `append=True`, a fresh `MDB_OUT` or deleting `out/records.jsonl` (example 03 already deletes it).
- A layer step on a model whose attention forward was rewritten after attach (nnsight `.source`) now refuses with the reason (`ValueError`) instead of stepping. Before, reads of `qr`/`kr`/`s`/`p` raised `KeyError` and edits at those ports could go unapplied.
- `whatif`, `whatif_many` and `forecast.seal` refuse a run with active holds (issue DC5). Measured on a tiny model: the prediction was the free continuation's while the held native change differed. Release the holds first, or measure the held arm with `Branches`.
- Verdicts: an effect within the floor needs native and emulated to agree on the sign (M59); the worst-case rounding interval is information, not `unresolved` (M56/M57); per-stage containment on native operands.
- Prefill follow-ups (M54): code uses `run.first_row` / `run.root`; records carry `procedure {prefill, first_row, prompt_rows}`; `Run.set` stages without a probe unless `check=True`.
- Lane-era code lives in `packages/mdb-lanes` (`mdb_lanes`); `mdb lanes` dispatches to the plug-in.

### Fixed

- Custody (amplitude audit C7/6a, issue CU1): record sinks no longer collide across shards. A compact `Branches.seal` record's `arms_payload` (`state: "absent"`) and any other store-less `bounded.put_json` payload is written to open sinks as an `mdb_payload` line (`Sink.load_json`), so per-arm probe values are sealed. Records and ids are unchanged.
- Issue I2: nnsight `.source` after attach (above).
- Issue DC5: forecasts under active holds (above).
- Debugger-core audit F1–F12 (pools, core files, holds, branches, custody, transcoders).
- Showcase findings SC1 (SAELens local-path loader), SC2 (watch + breakpoint reason), SC3 (seal `staged=include`), SC5 (`ResidualCapture` offline component binding), SC6 (stability: kernel-sensitive band, per-model calibration, `kernel_path` on verdicts), SC7 (`names.tl_hook`), SC8 (circuit-tracer adapter, pinned env only).
- EX1: a cache-cell value of shape `(d,)` is normalised to the block shape at arm creation.
- `symbols.table`: the `subspaces` parameter was shadowed by co-readout spans; `circuit_map.freeze` keys bindings by symbol name, not page position; `hop_edge` shadowing and `row_table` diagonal; `cell_readout` is affine under a V bias (Qwen2); `interop.tl.processed` reads the bridge flag; `custody._from_env` clears a stale error after a successful re-read.

### Known issues

- Open in `issues.jsonl` (2026-11-14): 33 not fixed or closed (11 medium, 22 low; includes 1 deferred, 1 known-limit, 1 partial), none high after I2. Notable: M58 (`until logits` after `prefill=K` stops at the first stepped row), M61 (a role binds the first subtoken of a multi-token word), I3/S3 (no edits at earlier rows' ports), M15 (all automated tests use tiny CPU models).
- Pre-existing test failures on the laptop: two REPL transcript tests and DC1 (`test_head_selected_patch_on_a_second_device[mps]`, MPS has no float64). See the pre-release report for the full list.
- The amplitude programme's missing arm-B shard records and replica records are not recovered by the sink fix.
