# Expected outputs: composer neurons, sealed nnsight forecasts, mdb stepped check (Qwen3-1.7B-Base)

The original experiment is showcase exp5 (MEASURED, single seed, bf16 eager CUDA on Beast,
job-09feeeb61cbe, nnsight 0.8.2). Evidence: `results/showcase-20261114/exp5/` (`exp5_neurons.py`,
`prereg.json`, `score.py`, `summary.json`, `out/exp5-items.jsonl`, `out/exp5-panels.json`,
`out/exp5-sealed.jsonl`, `logs/job-09feeeb61cbe.log`). It was audited in
`results/showcase-20261114/AUDIT-SHOWCASE.md` (exp5 row and correction 7). `reference.json` holds the
showcase numbers per item, cut from those files. The script prints "showcase vs now" from it.

## Items

This is the whole showcase panel, not a subset: the 64 two-hop base items (`items.json`, cut from
`results/twohop-20261113/next/r1_items.json`). Within each country they alternate A, B, A, ...; that
gives 38 A items and 26 held-out B items in 19 countries. `MDB_COUNTRIES=Spain,Japan` limits the
measured countries (the smoke run). The pools and the nomination always use all 64 items.

## Showcase numbers (job-09feeeb61cbe; audited)

| | value |
|---|---|
| N1: named set beats the random neuron set by >= 0.5 (batched) | 26/26 |
| N1: named set beats the answer-token neuron null | 26/26 |
| median dM: named / random set / answer-token null | -4.06 / 0.0 / 0.06 |
| N2: Spearman between forecast and native single neuron (71 above floor) | 0.944 |
| N2: median forecast/native (prereg band 0.4-0.7) | 0.829 (fails) |
| N4: batched - stepped within 2 spacings | 51/52, max 0.375 |
| median set forecast / batched set dM | 1.14 |

Framing (audit): N1 is a large effect. "Out of sample" means held-out items of the same country,
which share the answer token. That closes item-level selection, not answer-level selection. The
showcase's answer-token neuron null (median 0.06) argues against simple unembedding alignment.

## Validation run of this example: job-a6c4f49f1615 (MEASURED)

The job ran on Beast at priority 70, revision 321cb029, with script sha256 8673d59ec0c8 (the committed
`composer_neurons.py`) and prereg sha256 8ef92b2d26d3 (the showcase prereg, unchanged). It succeeded
in 197 s wall (script 196 s). Peak VRAM was 6890 MB and peak RSS 2386 MB. Log:
`validation-job-a6c4f49f1615.log`.

Smoke run: job-474f29705c57 (`MDB_COUNTRIES=Spain,Japan`, 4 B items, 136 s, peak VRAM 6890 MB). It
used an earlier nomination rule (see "Differences"). Its 4 items match the full run item for item.

### Showcase vs now on the same 26 B items

| | showcase | now |
|---|---|---|
| panels A/B | 38/26 | identical |
| named neuron sets (19 countries, 1216 neurons) | - | identical in 19/19 |
| N1: named set beats the random neuron set by >= 0.5, batched | 26/26 | 26/26 |
| median dM named set, batched | -4.0625 | -4.1875 |
| median dM random neuron set, batched | 0.0 | 0.0 |
| N2: neurons above floor | 71 | 72 |
| N2: Spearman | 0.944 | 0.963 |
| N2: median forecast/native | 0.829 | 0.818 (still outside 0.4-0.7) |
| N3: median ratio L20 / L26 | 0.972 / 0.769 | 0.898 / 0.781 (N3 still fails) |
| N4: batched - stepped within 2 spacings | 51/52 | 51/52, max 0.375 |
| top single neuron | - | same on 24/26 |

### mdb stepped arms (Branches, new in this example)

| | now |
|---|---|
| named set beats the random neuron set by >= 0.5 (stepped) | 26/26 |
| named set beats `add_null("random")` (matched-norm Gaussian on the same neurons) by >= 0.5 | 26/26; null median +0.625, range -2.9 to +3.1 |
| self-patch null `dM` exactly 0 | 26/26 |
| median dM: set / top single / random set / random change / answer-direction | -4.19 / -1.63 / 0.0 / +0.63 / -59.1 |
| named set "stable" (> 4 floors) | 26/26 |
| set dM / answer-direction null, median | 0.070 |
| set forecast sign agrees with the stepped set dM (`supplied.judge`) | 26/26; median forecast/stepped 1.155 |
| records in the sink, all re-verified | 8 pool, 1690 supplied-forecast, 26 branches |

The answer-direction null is not a matched alternative. It is a ceiling. It writes the knock-out's
own write norm (`W_down[:, set] (mean_loo - f)` at each layer's MLP output, on the unedited run)
straight along the readout direction. The named set reaches about 7% of that ceiling, so its write
is mostly not aimed at the answer's unembedding. That fits the showcase reading (effects composed
through later layers rather than direct alignment), but the ceiling does not test it the way the
showcase's answer-token neuron set did. That neuron null was not rerun (see below).

### Differences and why

1. **Leave-one-out knock-out value (by design).** The edits use `pool.mean_for(item)`: the mean of
   `f` over the other 63 items, taken from mdb's stepped run. The showcase used the 64-item mean of
   nnsight's `f`, which includes the item itself (audit caveat). The edit therefore moves `f` by
   64/63 of the showcase change, plus the kernel-path difference in `f` (point 3). This explains the
   sealed set forecasts being about 1.6% larger per item (for example Seville -5.187 -> -5.270) and
   the batched set effects moving by 0 to 0.25 (one or two bf16 spacings at |logit| 10-20). It also
   explains the 2 changed top single neurons (Shenzhen, yuan), where single-neuron effects are
   near-tied. N1, N2, N3 and N4 keep their verdicts.
2. **Nomination statistic.** The nomination uses the preregistered statistic on the library's values
   (the forecast to the 64-item mean of nnsight's `f`, A items only), so the named sets equal the
   showcase's in 19/19 countries. The smoke run nominated with the pool forecast `dM/df .
   (mean_loo_mdb - f_nnsight)` instead, and its ordered lists differed in 17/19 countries. The full
   run measures this directly: a pool-forecast nomination would swap 9 of 1216 neurons, in 8
   countries. Near-tied summed forecasts reorder under bf16-level differences in `f`. Nomination
   (P10) stays plain Python.
3. **nnsight `f` vs mdb stepped `f`.** They differ on 2,955,261 of 3,145,728 elements (64 items x 8
   layers x 6144). The max absolute gap is 56 and the max relative gap is 506 bf16 spacings. I did not
   check where the 506-spacing element sits; it is probably a near-zero `f`, where the spacing is tiny,
   but that is not verified. The two runs follow different kernel paths: a batched full-sequence
   forward and backward vs a batch-1 stepped row.
4. **Nulls.** The showcase's hand-drawn answer-token neuron set (and its score loop) is replaced by
   mdb's `add_null("answer")`, a residual-port null, so it is attached to the m-port write (above).
   The random neuron set is kept as plain Python: it is a selection null (other neurons). mdb's
   `add_null("random")` is a change null at the same address, and both are reported. The script
   replays the showcase's seeded stream, so the random neuron sets are the showcase's: per-item
   random-set values agree within 0.25.
5. **Batched vs stepped in `Branches`.** `Branches.run(mode="batched", recheck=k)` measures only
   state-cell arms (`cache.L<l>.k|v` through `sweep.cell_sweep`). It refuses port edits such as
   these `f[i=...]` knock-outs. So the batched arms come from nnsight `batch_whatif`, and
   `Branches.run()` (stepped) is the authority for the set, the top neuron, the random set and the
   three mdb nulls on every item, not a recheck of 2 arms. N4 compares the nnsight batched set and
   top arms with the `Branches` stepped ones.

## mdb objects used (and the showcase TODOs they replace)

- `dbg.custody(script=, prereg=)` + `dbg.records_to(...)`: every seal keeps its full record with the
  script and prereg sha256. Replaces P8 "only {'id','value'} kept" and "panel membership loose JSON"
  (the pool records carry the item digests).
- `dbg.pool(items, "t@last.L<x>.f")` + `pool.mean_for(item)`: replaces P6 "64-item mean by hand".
  The pool supports the f port; no source change was needed.
- `names.to_nnsight`: replaces the hand-typed address map.
- `supplied.seal` / `supplied.judge` (with `verdict.margin_floor`, `verdict.spacing`): replaces
  `common.floor_of` / `common.spacing`.
- `dbg.branches(run, Margin, at="t<last>.L20.h")` with `f[i=a|b|...]` selector edits carrying the
  pool pointer as origin (P5 `Run.set` selectors; replaces the hand-built value/mask tensors);
  `add_null("random" | "self" | "answer")`; `run()`, `seal()`. Replaces P1 "stepped authority by
  hand" and P6 "random set and answer-token null drawn by hand" (the random neuron set is kept as a
  selection null, see point 4).
- Not replaced: Q2.2 (linearity ratio column) and the `propagate.whatif` second forecast column;
  P7 (A/B split) and P10 (nomination) stay plain Python.
