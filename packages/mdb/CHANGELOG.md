# Changelog

All notable changes to `mdb-runtime` (packages/mdb). The version number is set by the owner.

## [0.2.0] — 2026-11-14 (owner-approved version; on mdb-pre-release, not yet tagged or merged to main)

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

### Known issues

- `until logits` after `start prefill=K` stops at the logits of the first stepped row, not the last row; name the row (`until t5.logits`).
- A role in a prompt set binds the first token of a multi-token word; the other tokens are not part of it.
- Ports of rows prefilled in one batched call cannot be read or edited; use `start prefill=K` so those rows are stepped.
- Every automated test uses a tiny random CPU model; real models are checked by hand on the maintainers' GPU.
- Pre-existing test failures (they also fail on the previous release): `test_two_logits_are_indexed_then_cast` (`stage_checks` casts the full logits row); `test_head_selected_patch_on_a_second_device[mps]` on Apple MPS; and, in some environments, `test_golden_transcript[predict_forecast]` and `test_the_script_names_every_registered_op`.
- `whatif` fails on Apple MPS (no float64); CPU and CUDA are the tested devices.
