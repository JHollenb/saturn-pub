# Expected outputs: SAE features as component bindings of the carrier cell

Source experiment: an earlier, larger run on 64 base items (two shards) with a sealed
preregistration. Model Qwen3-1.7B-Base, bf16, eager attention, one CUDA GPU; SAEs Qwen-Scope
`SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (k = 50, `resid_post`), layers 9..22. One seed. The
preregistration is copied unchanged as `prereg.json`, and its sha256 is declared through
`dbg.custody`.

## Original run (64 base items)

The original run had a 2-item smoke, a first shard (base items 1-32; this example's items come from
it) and a second shard (base items 33-64 and 44 census items).

## Original 64-item numbers

- **L2 (headline).** At L20, removing the named top country component moves `M` by a median
  **0.59** of zeroing the whole cell. A matched-norm random component moves it **0.0**, and the
  error component **0.11** (n = 63 items with a top country component). L2 holds.
- **Partition.** The components sum to the cell on 64/64 items, at L20 and at every layer, with
  a max gap of 1.8e-15. This is a float64 algebraic identity: it checks the component code, not
  the model.
- **SAELens.** The active sets agree on 108/108 residuals. The SAELens side is mdb's own
  `FeatureBasis.to_saelens()` (strict load) run through SAELens's TopK `encode`. SAELens's
  registry loader was not tested, because `SAE.from_pretrained` fails offline
  (`LocalEntryNotFoundError`).
- **L4.** 48/64 items are above floor only inside L17..L21. In practice this means "above floor
  only at L20, the reading layer": 46 at L20, 4 at L18, and 16/64 never above floor.
- L1 half-holds: the top share is >= 0.5 on 41/64 (bar 40), but the error share is <= 0.2 on
  only 42/64 (bar 48). For L3 the falsifier fires: the median lineage onset is L17, and 46/63
  items have an onset >= L15.

## This example

The example runs the first **24** of the 64 base items: the 16 currency items and the first 8
person items, in first-shard order. The script prints its numbers next to the original run's numbers
for the same 24 items. `reference.json` holds those original numbers. With the full 32 items of s0 the job ran past 520 s (see below),
so it was cut to 24 to keep the job short. `MDB_NITEMS` can be set anywhere from 1 to
32. The random draws follow the original run's seeded sequence, so any prefix of the first shard
reproduces the original run's random picks.

### mdb objects (replacing hand-written parts of the original driver)

- `symbols.declare` plus `symbols.table(run, consumer=M, sets=[countries], components=[cb],
  limit=1, causal=True)` replace the bare `component_table` call.
  The page holds the declared `carrier` symbol with its mass, its content and its measured
  liveness, plus the `components` section: activation, exact direct share, content, and SAELens
  ids per component.
- `Binding.component` / `component_binding` and `component_table` build the per-layer splits
  for L15..L23.
- `dbg.branches(...)` makes one `Branches` per item, with the 36 per-layer arms (top, random
  component, error, zero) run as `br.run(mode="batched", recheck=2)`. They share one floor and
  carry `effect_floors` and `above_floor`. `add_null("L20:gauss", "random", like="L20:top")` is
  mdb's matched-norm Gaussian null, measured stepped. `br.seal()` writes a `branches` record.
  Together these replace the hand-kept arm lists, `cell_sweep`, the hand floors and the
  hand-drawn random arm. The preregistered random-component arm is still built,
  as a named arm, because the prereg defines "random" as an active non-country feature
  component. The Gaussian null is an addition.
- `liveness()` (through `causal=True`) is used for the declared carrier: it zeroes the cell for
  the answer row. It does **not** fit L4, because liveness removes whole cells and cannot remove
  one component. L4 therefore stays on the `Branches` arms.
- `dbg.custody(script=, prereg=)` and `dbg.records_to(out/records.jsonl)` record custody: 48
  records for 24 items, one symbol table and one branches record per item.
- Kept as plain Python: feature lineage, the item loop, and the projection strip,
  last-row null and census items (left out of this example).

### Validation run (measured, 24 items)

One job on one CUDA GPU (RTX 4080) through our job scheduler. Validation log:
`validation-job-e0963b937e59.log` in this directory. Succeeded in 474 s wall (471.8 s of script time). Peak VRAM was 3994 MB and peak RSS 10988 MB.
6/6 checks passed.

| Key number (24 items) | now | original run, same items |
|---|---|---|
| L2 median top / zero (L20) | 0.5238 | 0.5238 |
| L2 median random component / zero | -0.0 | -0.0 |
| L2 median error / zero | 0.1752 | 0.1752 |
| L2 median Gaussian null / zero (new, `add_null`) | 0.0192 | not run |
| L1 median top share / error share | 0.6586 / 0.1236 | 0.6586 / 0.1236 |
| L1 top share >= 0.5, error share <= 0.2 (counts) | 14, 15 | 14, 15 |
| L3 median onset; onset >= L15 | L18; 17/24 | L18; 17/24 |
| L4 only inside L17..21 (original run's floor rule) | 19 (L20: 18, L18: 1) | 19 (L20: 18, L18: 1) |
| L4 only inside L17..21 (Branches floor) | 14 (all at L20) | n/a |
| partition max gap (code check) | 1.33e-15 | 1.33e-15 |
| SAELens active set agrees | 24/24 | 24/24 |
| correct | 24/24 | 24/24 |

The following match the original run on 24/24 items: carrier rows, top country features at every
layer, random picks, and every batched arm's `dM`. The max |difference| is exactly 0.0, because
the arms run on the same kernel path in the same order. The liveness of the carrier (stepped,
fresh prefix cache) and the batched `zero` arm differ by at most 0.375. The stepped rechecks
differ from the batched arms by at most 0.375. Both differences are inside the floor.

**One difference, explained: the floor.** The `Branches` floor of a batched arm is 2 bf16
spacings plus **2 x** the null arm's batched-vs-stepped delta, taken over the **whole
vocabulary** (median 0.949). The original run's floor is 2 spacings plus **1 x** that delta, taken
over the **two margin tokens** (median 0.375). Under the Branches floor, L4 counts 14 items
instead of 19: 4 items drop out at L20 and the single L18 item drops out. To reproduce the original rule exactly,
`Branches.batched` now also reports `kernel_delta_null_consumer_tokens` (see the mdb change
below). With that field, the example's original-rule L4 equals the original run's.

Subset caveat: the 24-item L2 median for top is 0.52, against 0.59 over all 64 items. The
preregistered bars (top >= 0.5, random <= 0.15, error <= 0.2) are judged on the 64-item median,
so on this subset they are printed for information only (all three hold here).

### Earlier runs of this example

- A smoke run failed: a symbol-table record with a k = 50 feature binding exceeded the 64 KiB
  record bound (fixed in mdb, below).
- A smoke run failed: a `(d,)` cell value passed the batched path but the stepped recheck refused
  it. The example now passes the block shape `(1, 1, 1, d)`.
- A 2-item smoke run: all numbers equal the original run. Peak VRAM was 3976 MB. The scheduler
  marked it failed only because the `run.sh` log filter returned non-zero under pipefail (fixed).
- A 28-item run took 525 s wall (too long). Every arm `dM` equalled the original run's. L4 under
  the original run's floor rule was 21 vs 23 because the floor used the whole-vocabulary delta,
  which led to the field added below.

### mdb source changes made for this example

- `analysis/symbols.py` `_records`: when a symbol-table record with a `components` section
  exceeds the record bound, each section keeps its totals and partition, and its component rows
  are stored behind a sha256 pointer (`components_full`, `core.bounded.put_json`).
  `table.report` keeps every row. Test:
  `tests/test_analysis_components.py::test_symbol_table_record_holds_a_wide_feature_binding_by_pointer`.
- `analysis/branches.py` `batched`: each batched row adds `kernel_delta_null_consumer_tokens`,
  the null arm's delta over the consumer's own tokens. The floor is unchanged. Test: an
  assertion added to `tests/test_analysis_branches.py::test_batched_cell_arms_with_stepped_recheck`.
