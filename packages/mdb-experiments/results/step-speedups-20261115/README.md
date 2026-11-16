---
title: "mdb stepping speedups (review items 1-9): measurements and bit-identity evidence"
date: 2026-11-15
type: results
branch: mdb-step-speedups (from mdb-v0.2.0 = main c83e47f)
tags: [mdb, saturn-pub, performance, stepping-engine, bit-identity, mrun]
---

# mdb stepping speedups: results

Branch `mdb-step-speedups` from `mdb-v0.2.0` (main `c83e47f`). Code commit `9cfa483`; Beast payload staged from `1c20d2e` (same code plus these scripts). Source: `../../../../obsidian/reviews/2026-11-14-mdb-architecture-review-engine-dev.md` (top 10).

Labels: MEASURED = run here; ESTIMATE = arithmetic, not run.

## Bit-identity (hard gate)

| check | how | result |
| --- | --- | --- |
| tiny qwen2/qwen3 CPU, bf16 and fp32 | `scripts/ident.py`: trail fingerprints, receipts, tokens, logits sha of every row, port reads (prefill=0), staged-edit + cell-edit run (lookahead=layer), hold run incl. `held_bytes`, restore/reverse, `each` (whatif, reads, print) with a project: stored record ids and item rows incl. `bytes`; saturn_pub layer-granularity session with restore | `cpu/ident_main.json` == `cpu/ident_branch.json` byte for byte (112 keys) MEASURED |
| `00-hello-tiny` | run on main and branch, same venv | raw stdout identical incl. record id `f30d4d21ce1341f0` (`cpu/hello_*.txt`); matches `expected_output.txt` with ids normalised (the expected file's id comes from other library versions) MEASURED |
| CPU microbench | trail digest, last tokens, logits sha per case | identical in every case and every ablation (`cpu/*.jsonl`); `abbc0130f896d66e` = the reviewer's digest MEASURED |
| MLP-weight analyses (item 4) | `scripts/sym_check.py`: `hop_edge(spectral=True)` twice with a shared cache + `gate_switch_forecast`, full JSON | identical JSON main vs branch at (L,d,d_ff) = (4,256,1024) and (8,1024,4096) MEASURED |
| Qwen2.5-1.5B bf16, RTX 4080 (Beast, job-96ae6309702f) | `beast/beast_step.py` on main and branch in one job: stepped rows T=1024 and T=256, prefill=0 T=32, port walks T=256 | trail, tokens, every row's logits sha, port-read digest, held_total: identical in all 5 cases (`ALL` cases "identical") MEASURED |
| example 04 smoke (Qwen3-1.7B-Base + SAE) on Beast | records.jsonl, results JSON, stdout of main vs branch | every model-derived field identical (summary, per-item verdicts, pool content). Record ids differ because the records hold `custody.declared_at` (wall-clock time), the payload path of the script/prereg and per-arm `seconds`: they differ between any two runs, main vs main included. Field diff in `beast/ex04/` (see below) MEASURED |

Example 04 field diff (complete list of differing fields, main | branch): `custody.declared_at` (timestamps 17 s apart), `custody.script`/`custody.prereg` (paths `ex04_main/` vs `ex04_branch/`), `arms[*].seconds` (0.103-0.109 s | 0.044-0.048 s), and the record ids / `origin.record` pointers that hash those fields.

## Tests

| suite | main (mdb-v0.2.0) | branch |
| --- | --- | --- |
| packages/mdb (libvenv: torch 2.13, transformers 5.13.1) | 2 failed / 4280 passed / 29 skipped | 2 failed / 4288 passed / 29 skipped (same 2 failures; +8 new tests) |
| repo root `tests/` (saturn-pub/.venv: torch 2.14, transformers 5.14.1) | 1 failed / 557 passed / 1 skipped | 1 failed / 581 passed / 1 skipped (same failure; +24 new tests) |

Main's failures are the known ones (`test_two_logits_are_indexed_then_cast`, `test_head_selected_patch_on_a_second_device[mps]`; root: `test_debugger_reuses_symbol_watch_and_provenance_stops_at_restore`). The task brief quoted 3 failed / 4276 passed for mdb; in this environment main gave 2 / 4280 (the other listed pre-existing failures are environment-dependent per the CHANGELOG).

First branch run: 18 mdb failures = the 2 known + 16 parametrisations of `test_read_cache_reads_the_slot_without_hashing`, whose control arm asserted that a second `capture()` of an unchanged state hashes the state (it no longer does: seal reuse + remembered descriptors); root: 3 failures in `test_substrate_counts.py`, whose counters patched `values.describe`/`_tensor_guards` (the hash and the frozen check now happen in `_describe_tensor` and `_FrozenSnapshot`). The tests were updated to count at the new points (control arm uses `verify()`, which always re-hashes; a layer transition now hashes 3 tensors: hidden and the grown K and V). New targeted tests: `tests/test_step_speedups.py` (24) and `packages/mdb/tests/test_step_speedups.py` (8).

## CPU microbench (laptop, 4 threads, bf16; reviewer's config: 28 layers, 2 KV heads x 128, hidden 256)

Cumulative, main -> branch (MEASURED, one run each; `cpu/bench_main.jsonl`, `cpu/bench_branch_final.jsonl`):

| case | main | branch | speedup |
| --- | --- | --- | --- |
| stepped row, T=1024 (prefill=T-4) | 801.7 ms | 98.3 ms (101.6 on a rerun) | 8.2x |
| stepped row, T=256 | 450.5 ms | 82.7 ms | 5.4x |
| prefill=0, T=96, `until logits` | 27.21 s | 8.05 s | 3.4x |
| `held_bytes()` after that (same 7,769,080 B) | 9.07 s | 0.29 s | 32x |
| one row port by port, T=512, `none` | 1997 ms (504 probes) | 664 ms (504 probes) | 3.0x |
| same, `layer` | 646 ms | 137 ms | 4.7x |
| same, `auto` (new, opt-in) | n/a | 138 ms (28 probes) | 14.5x vs main `none` |

Per item, branch minus that one item (revert by monkeypatch, `scripts/ablate.py`; `cpu/ablate.jsonl`), T=1024 row / prefill=0 T=64 (branch 98.3 ms / 5.17 s; main 801.7 ms / 15.32 s):

| item | without it | cost it removes |
| --- | --- | --- |
| 1 descriptor memo | 418.3 ms / 5.96 s | +320 ms/row / +0.8 s |
| 2 finite memo | 368.9 ms / 6.60 s | +271 ms/row / +1.4 s |
| 3 cache by reference | 141.2 ms / 5.56 s | +43 ms/row / +0.4 s |
| 6 chunk guard | 116.2 ms / 8.37 s | +18 ms/row / +3.2 s |
| 7 seal reuse + plain-key sort | 140.6 ms / 7.79 s | +42 ms/row / +2.6 s |
| 8 fast frozen check (main's walk restored) | 137.6 ms / 7.67 s | +36 ms/row (against a 101.6 ms rerun) / +2.5 s |
| 5 held_bytes | n/a (not on the stepping path) | 4.10 s -> 0.157 s per `held_bytes()` at T=64 |

Ablations interact (each is measured with all others on), so they do not add up to the cumulative figure. Single runs on a laptop: treat differences under ~15% as noise.

## Item 4: float64 MLP weight memory (MEASURED on small configs, ESTIMATE for 1.5B)

`scripts/sym_check.py`, macOS RSS:

| (L, d, d_ff) | cache held after the calls, main | branch | RSS growth during the calls, main | branch |
| --- | --- | --- | --- | --- |
| (4, 256, 1024) | 42.1 MB | 0.04 MB | 47.1 MB | 25.3 MB |
| (8, 1024, 4096) | 1343 MB | 0.33 MB | 999 MB | 193 MB |

ESTIMATE for Qwen2.5-1.5B (d=1536, d_ff=8960, L=28): main caches 28 x (3 x 13.76M x 8 B + 2 x 8960 x 769 x 16 B) = about 15.4 GB; the branch keeps about 2.4 MB (one `c . W_down` vector and one permutation per layer) and its transient peak is one layer's float64 gate/up (220 MB) plus the two complex rffts of the gate (220 MB) plus temporaries, about 0.6-0.7 GB. Not run on a 1.5B model. Cost: the casts are redone per call instead of once per cache.

## Beast: Qwen2.5-1.5B bf16, RTX 4080, torch 2.13.0+cu130, transformers 5.13.1 (job-96ae6309702f, MEASURED)

| case | main | branch | speedup |
| --- | --- | --- | --- |
| stepped row, T=1024 (4 rows, ms each) | 657.6, 641.8, 644.5, 643.8 | 125.3, 110.3, 108.8, 108.7 | 5.9x (steady state) |
| stepped row, T=256 | 322.3, 321.6, 329.3, 459.4 | 101.7, 98.9, 98.9, 97.2 | 3.3x |
| prefill=0, T=32, `until logits` | 8.34 s | 3.24 s | 2.6x |
| `held_bytes()` after it | 1.008 s | 0.036 s | 28x |
| port walk T=256, `none` | 1.55 s | 0.91 s | 1.7x |
| port walk T=256, `layer` | 0.41 s | 0.17 s | 2.4x |
| port walk T=256, `auto` | n/a | 0.30 s (28 probes) | 5.2x vs main `none` |
| example 04 smoke wall (2 items) | 16.8 s | 9.9 s | 1.7x |
| example 04 per item | 4.38 s, 2.70 s | 1.65 s, 1.01 s | 2.7x |

Peak VRAM (`torch.cuda.max_memory_allocated`, per case): equal for T=1024 rows (3280.3 MB both), prefill=0 (2968 MB both), port walk `none` (3035 MB both); the branch was higher by 56 MB on T=256 rows (3090.4 vs 3034.1) and by 14 MB on the `layer` walk (3049.3 vs 3035.0). Not investigated: the cases run in sequence in one process, so this may be memory of the previous case not yet collected, not a per-row cost. Example 04 reported 3388 MB on both. Peak RSS per process: 4.09 GB both (bench), 4.44 GB both (example 04). The mrun job peak was 4030 MB VRAM, 2752 MB RSS (reservation 5000 / 6000 MB). The `auto` walk was slower than `layer` on the GPU (0.30 vs 0.17 s) with the same 28 probes; one measurement, not investigated.

## Deferred / not done

- Item 9 default: `lookahead` and `computed_past_cursor` are fields of forecast records (`analysis/forecast.py`, `sealed_at`) and of recipes, so a new default would change record ids and the forecast's "sealed before native code ran" premise. Shipped as opt-in `lookahead="auto"`; the default stays `"none"` (owner decision).
- Item 8 "once per public call": not done. The checks were made cheap instead (frozen check 0.54 -> 0.15 ms per call on this config); their number per transition is unchanged (`_head`, after the native call, `capture`). mdb's `check_deterministic` is unchanged (a faster walk I tried was slower than `named_modules`, so it was dropped).
- Item 7 "pre-serialize constant parts": not done; the seal reuse removes the repeated seals of one state, which was the larger part.
- Item 10: not implemented; design note `item10-columnar-records-design.md` (record ids cannot stay stable; needs mdb-record-v3 and an owner decision).
- Long-tail items 11-24 were out of scope.
