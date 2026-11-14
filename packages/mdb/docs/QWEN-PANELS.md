# Qwen value and read-cut panels

Status: November 13, 2026, 22:06 PST (verdicts recomputed after the audit `docs/AUDIT-three-lanes-20261113.md`; first written 21:26 PST). **Smoke and all three full jobs finished and were pulled and analysed.** Smoke `job-9020b3bfd5e7`; full jobs `job-b33e6b98c328` (FP32 value + read lease 0), `job-4a433df3f30b` (read lease 1), `job-7c3232ad65bb` (read lease 2 + BF16 value + BF16 smoke read). Results tables (a) to (e) are below, MEASURED. No new job ran for the recomputation: the read verdicts and the BF16 table were recomputed from the saved rows with the fixed verdict code (`qualification/panel_verdicts_recompute_20261113.py` → `qualification/qwen_panels_20261112/full-verdicts-20261113.json`; `full-tables.json` is unchanged and holds the earlier verdicts).

Worst caveats, in order:
1. **Read verdicts (d) changed after the fix.** The `freezing` random controls did not control what they claimed. For every `freezing` set the two random draws are the same set (`rand0` = `rand1`). For L20H5 the "random" head is L20H11, for L20H11 it is L20H5, and rank3's draw contains L20H11, so three of the six cut controls were other named test heads. The other-item control cuts the same heads' reads of `icy`, which carry 0.05% to 2.7% of the target's attention mass, so it is close to an empty cut. With those controls excluded or ruling nothing out, L20H5, L20H11 and rank3 are `control_missing`; L21H0 is `unresolved` (effect 4.4% to 8.4% of the all-cut, below the 5% minimum-fraction choice); named3 and rank12 stay `set_only` with only set size ruled out.
2. In BF16 the readout-rounding bound is not an uncertainty bound. With the measured BF16 compute floor added (0.125 per margin), 15/58 (re-run) and 14/58 (saved) coefficients are resolved; 2 saved coefficients still lie outside the wider bound from FP32.
3. In BF16 the worker engine does not reproduce the saved main-02 margins bit for bit (32/56 arms differ by 0.0625 to 0.125, tokens all equal).

Numbers in the design sections marked DRY RUN or ASSERTED are unchanged.

Modules: [`mdb.qwen_panels`](../src/mdb/qwen_panels.py) (pure Python: case, plans, recipes, analysis), [`qualification/qwen_panel_worker.py`](../qualification/qwen_panel_worker.py) (torch worker), [`qualification/qwen_panel_stage.py`](../qualification/qwen_panel_stage.py) (plans, job layout, stage, submit), [`qualification/qwen_panel_analyze.py`](../qualification/qwen_panel_analyze.py) (tables (a) to (e)). Tests: [`tests/test_qwen_panel.py`](../tests/test_qwen_panel.py).

## The case

This is the value-coalition case of 2026-11-12 (job main-02), reproduced exactly:

- **Model:** Qwen2.5-1.5B base. The checkpoint is `/mnt/big/llm-models/Qwen2.5-1.5B`; `model.safetensors` sha256 must be `a961db72…aec796`, and the worker refuses any other.
- **Input:** plain text, with no chat template and `add_special_tokens=False`.
- **Prompts:** the recipient demonstrates `icy → cold` and `fiery → hot`, then asks about `freezing`. The donor prompt changes `freezing` to `boiling`.
- **Rows:** the source is row 18; the wrong row is 17 (the colon); the first receiver is row 21 (the final colon). The worker re-tokenizes and refuses any token change.
- **Tracked tokens:** ` cold` 9255, ` hot` 4017, ` boiling` 49837.
- **Margin:** hot − cold at the first generated position.
- **Value cells:** (layer 19–27) × (KV group 0/1), at the source row.

**Engine.** The worker uses transformers `Qwen2ForCausalLM` with eager attention, `DynamicCache` and batch 1. Rows 0–18 run in one forward, then rows 19 and 20 run one forward each. This is the token order of `attention_support.opened` in main-02. Value edits go into the cache at this point. Then row 21 and two generated tokens run, one greedy forward each.

**Lanes.** One model load in FP32 (from the BF16 checkpoint, which is exact). The BF16 lane casts the parameters in place, which is also exact because the values are BF16-representable, and leaves buffers as loaded. Rotary `inv_freq` stays FP32, as `from_pretrained(torch_dtype=bfloat16)` leaves it in transformers 5.14.1 (MEASURED on a tiny Qwen2 checkpoint). One model is resident throughout.

## Plans (sealed; the lane is in the hashed `port` block)

- `value-<lane>`: `attention_panels.value_panel(LATE_CELLS, label="late", groups=KV0/KV1/L<l>-both, doses=(0.5,), reference_dose=0)`. It has 56 arms, and its content equals `value_coalition.plan()` (tested). Arm names map to main-02 names through `qwen_panels.LEGACY_NAMES`.
- `read-float32`: `attention_panels.read_panel` on the first receiver reading the source row.
  - **Items:** `freezing` (row 18) and `icy` (row 2). Each item is the other's other-item control: the same heads cutting another source row.
  - **Sets:** L20H5, L21H0 and L20H11 singly and together (`named3`), plus the label-free `rank3` and `rank12`. Every set is run as a cut and as a keep, in both cut types, with `floor_arms` floors.
  - **Random controls:** two random-head draws per set, matched by layer (strata = layer) and disjoint from the set. Mass is per item: the head's native row-21 attention to that item's row, searched within an absolute 0.02 where reachable. The sealed plan's draws did not exclude the other named test heads, and identical draws were emitted twice; since the 2026-11-13 fix, `read_panel` excludes every named test unit from random draws and emits a repeated draw once, and verdicts re-label each stored draw's mass with the relative rule (see `docs/ATTENTION-PANELS.md`). A new plan built from the same attention would therefore differ from the sealed one; the sealed plan is unchanged.
  - **References:** the all-cut and the unmodified input `native`.
- **Label-free sets:** `contrast_ranking` of the receiver's attention to row 18 against its attention to the other word rows (0, 2, 4, …, 20), using the smoke's FP32 attention. The plan records the rank of each named head, a raw-mass ranking (the trivial head nominator) and the `attention_ranking` top-source baselines (position 0, previous token, offsets from the end, uniform) for all heads, `named3` and each top-n set.
- `read-smoke-<lane>`: L20H5 only, with floors, the other item and the all-cut. It is the BF16 backend-floor measurement in the full layout.

**Branch budget.** `mdb.scheduling_contract` allows at most 256 logical branches per job. The full layout is therefore three jobs (`qwen_panel_stage.job_layout`; read-plan leases are capped at 199):

1. FP32 value plan plus read lease 0.
2. Read lease 1.
3. Read lease 2, the BF16 value plan and the BF16 smoke read plan.

Each lease also runs its own `native` copy. Keep arms carry their all-cut through `attention_panels` lease replication. Copies are checked against the primaries in the analysis.

## How to run

```sh
PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
export PYTHONPATH=<worktree>/packages/mdb/src PATH=/Users/jakeholl/domains/common/bin:$PATH MRUN_URL=http://127.0.0.1:19025
cd packages/mdb/qualification
$PY qwen_panel_stage.py plans --profile smoke --out qwen_panels_20261112/smoke          # done, committed
$PY qwen_panel_stage.py stage  RUN_SMOKE --plans qwen_panels_20261112/smoke --vram-mb 10000 --ram-mb 14000
$PY qwen_panel_stage.py submit RUN_SMOKE                                                  # job-9020b3bfd5e7 (succeeded)
# after the smoke: pull, then build the full plans from its FP32 receiver attention
rtk proxy rsync -a beast:/mnt/hdd8tb/mdb/research/<job>/ /Users/jakeholl/domains/artifacts/mdb-qwen-panels-20261112/<job>/
$PY qwen_panel_stage.py plans --profile full --out qwen_panels_20261112/full \
    --attention /Users/jakeholl/domains/artifacts/mdb-qwen-panels-20261112/<job>/qwen-panels/float32/receiver-attention.json
for k in 0 1 2; do $PY qwen_panel_stage.py stage RUN_FULL_$k --plans qwen_panels_20261112/full --job $k --vram-mb <peak x 1.3> --ram-mb <peak x 1.3>; done
$PY qwen_panel_analyze.py --plans qwen_panels_20261112/full --out tables.json <pulled job dirs>
```

The worker prints `QWENPANEL {"kind": "modules", ...}` with the file and sha256 of `mdb`, `attention_port`, `attention_panels`, `attention_ranking`, `analysis_metrics`, `qwen_panels` and `native_replay_recipe` as imported from the staged wheel. It refuses to start if they differ from the hashes recorded at staging.

## What is stored (per lane, per plan)

- `rows.jsonl`, one row per arm:
  - the tracked logits and their ranks;
  - the margin hot − cold with its readout-rounding bound (`analysis_metrics`; dtype = the dtype the logits were produced in);
  - the greedy first three tokens and the top-20 logits;
  - the distance to the reference arm (max |Δlogit| and KL);
  - for read arms: the port metadata (cuts, removed reads, emptied cells) and the lease.
- `logits.npy` (value plans only): full first-position logits as FP32, [arms, 151936]. Read plans keep the top-20 plus the tracked tokens only.
- `recipes.jsonl`: a plain JSON recipe (`mdb-qwen-panel-recipe-v1`) for every arm. Where the arm kind is expressible, there is also a sealed `mdb.native_replay_recipe` recipe (`copy-values` on the native session, in the main-02 token order).
- `results.json` (read plans): the `attention_panels` results record within the lease. The outcome is |margin shift from `native`|, and twin pairs are |margin(delta cut) − margin(recompute cut)|.
- `receiver-attention.json`: row 21's FP32-recomputed attention (`attention_port.Read.probs`), every layer, head and source.
- `summary.json`: the timeline, peak VRAM and RSS, versions, and the module sha256s.

### Kinds not expressible in `native_replay_recipe`

The module is unchanged. These kinds get a plain JSON recipe only:

- `dose` (0.5): `copy-values` copies whole rows, and there is no blend event.
- Every read-cut kind (`cut`, `keep`, `all_cut`, `floor_sham`, `floor_sham_recompute`, `floor_cut_recompute`): there is no attention-edit event.

Expressible kinds: `native`, `replay`, `donor_native`, `sham`, `coalition`, `singleton`, `complement`, `wrong_target`, `wrong_payload`, `reciprocal`, and the read-plan `native`.

DRY RUN (tiny random Qwen2 with 28 layers × 12 heads / 2 KV heads, CPU): the native recipe regenerates the worker's first-position logits within 1.2e-7 in FP32 and exactly in BF16. The smoke repeats this check on the real model.

MEASURED on the real model (smoke `job-9020b3bfd5e7`, CUDA): FP32, max |Δlogit| native recipe vs worker = 1.24e-5 (`native-recipient`) and 1.34e-5 (`coalition-L20-both`), greedy tokens equal. BF16 is **not** exact on the real model: max |Δlogit| = 0.156 and 0.1875, margin −4.75 (recipe) vs −4.625 (worker) for `native-recipient`, 0.375 vs 0.375 for `coalition-L20-both`, tokens equal. The recipe's BF16 margins equal the saved main-02 margins (−4.75, 0.375); the worker's BF16 margin for `native-recipient` does not (see table (a)).

## Results

Scope: one prompt pair and one model; the 18 cells share one context and are not independent replications. Every number in this section is MEASURED unless marked otherwise. Full tables: [`qualification/qwen_panels_20261112/full-tables.json`](../qualification/qwen_panels_20261112/full-tables.json) (written by `qwen_panel_analyze.py --plans qwen_panels_20261112/full`); full plans and label-free ranking: [`qualification/qwen_panels_20261112/full/`](../qualification/qwen_panels_20261112/full/). Pulled job directories: `/Users/jakeholl/domains/artifacts/mdb-qwen-panels-20261112/<job>/`; staged payloads: `.../runs/full-0{0,1,2}`.

### Runs

| job | content | worker wall (s) | scheduler elapsed (s) | peak VRAM: scheduler / torch reserved (MB) | peak RSS: scheduler / worker ru_maxrss (MB) |
|---|---|---|---|---|---|
| `job-9020b3bfd5e7` (smoke) | 5 value arms + 31 read arms per lane, replay checks | 24.8 | 26.8 | 6352 / 6528 | 4833 / 9726 |
| `job-b33e6b98c328` | FP32 value (56) + read lease 0 (186) | 75.7 | 79.6 | 6302 / 6018 | 5643 / 9730 |
| `job-4a433df3f30b` | read lease 1 (191) | 32.7 | 37.6 | 6300 / 6016 | 4997 / 9723 |
| `job-7c3232ad65bb` | read lease 2 (126) + BF16 value (56) + BF16 smoke read (31) | 33.5 | 36.8 | 6300 / 6512 | 8626 / 9724 |

Reservations for the full jobs: VRAM 8487 MB = 6528 × 1.3, RAM 12645 MB = 9726 × 1.3 (the larger of the two RSS figures; the scheduler's sampled peak misses the load transient). Integrity: all four jobs hash the checkpoint to `a961db72…aec796`, import identical module sha256s, and write byte-identical FP32 receiver attention (sha256 `025a58be…`; BF16 `c7b0dc12…` in both BF16 jobs). The 5 smoke value arms and the 31 smoke read arms have the same margins as the same arms in the full runs, in both lanes. Lease copies: 12 replicated arms, all equal to their primaries (margin and tokens). No read arm is missing (491/491).

### (a) BF16 value arms vs the saved main-02 run

56/56 arms: same first token and same three greedy tokens. Margin bit-equal on 24/56; the others differ by +0.125 (25 arms), −0.125 (5), ±0.0625 (1 each). `native-recipient`: −4.625 here vs −4.75 saved. The smoke's native-recipe replay of the same arm gives −4.75, so the gap is between the worker's HF eager BF16 path and the native-session BF16 path main-02 used, not a different case. The value edits themselves are exact in both lanes: `sham-late` and `replay-recipient` vs `native-recipient` max |Δlogit| = 0.0 in FP32 and BF16. Verdict on "reproduced exactly": tokens yes, BF16 margins no (within one BF16 step at logit scale 16 to 32).

### (b) FP32 coefficients vs BF16

All 58 coefficients are resolved in FP32 (readout bound ≤ 6.8e-5). In BF16, 17/58 exceed their readout-rounding bound in the saved run and 17/58 in the re-run. That bound covers only the final rounding of the stored logits, not BF16 compute error through 28 layers, so since 2026-11-13 `Bounded.resolved` is False in BF16 unless a compute-error term is supplied (0/58 "resolved" on the readout bound alone). With a compute-error term of 0.125 per margin (the measured BF16 recompute-sham floor: max |margin shift| of the 12 BF16 sham arms of `read-smoke-bfloat16`, MEASURED; a floor, not a proven bound), the bound of a two-margin effect grows from 0.25 to 0.5, and 15/58 (re-run) and 14/58 (saved) coefficients are resolved. No coefficient the saved run resolved vanishes in FP32. Selected coefficients (margin units, hot − cold; the bounds in this table are the readout-rounding bounds):

| coefficient | FP32 | BF16 re-run (bound) | saved BF16 (bound) |
|---|---|---|---|
| full effect | +7.528 | +8.000 (0.25) | +8.125 (0.25) |
| singleton L20/KV0 | +3.454 | +3.625 (0.25) | +3.875 (0.25) |
| complement loss L20/KV0 | +4.872 | +5.250 (0.25) | +5.375 (0.25) |
| singleton L20/KV1 | +0.797 | +0.875 (0.25) | +0.875 (0.25) |
| singleton L21/KV0 | +1.587 | +1.500 (0.25) | +1.625 (0.25) |
| complement loss L21/KV0 | +2.447 | +2.625 (0.25) | +2.750 (0.25) |
| singleton L23/KV0 | +0.708 | +0.625 (0.25) | +0.625 (0.25) |
| complement loss L24/KV1 | −0.650 | −0.625 (0.25) | −0.500 (0.25) |
| singleton L26/KV1 | −0.835 | −0.8125 (0.22) | −0.750 (0.22) |
| interaction L20 | +0.499 | +0.500 (0.5, unresolved) | +0.375 (0.5, unresolved) |
| KV-group interaction | −0.301 | −0.375 (0.5, unresolved) | −0.500 (0.5, unresolved) |
| sum of singletons | +5.724 | +5.8125 (4.47) | +6.750 (4.47) |
| nonadditivity (full − sum singletons) | +1.804 | +2.1875 (4.47, unresolved) | +1.375 (4.47, unresolved) |

- Saved BF16 coefficients that differ from FP32 by more than their own readout bound: 5/58 (full effect −0.597, singleton L20/KV0 −0.421, complement loss L20/KV0 −0.503, complement loss L21/KV0 −0.303, paired effect L20 −0.374; FP32 minus saved). The BF16 re-run has 2/58 outside its readout bound (full effect −0.472, complement loss L20/KV0 −0.378). The BF16 readout-rounding bound does not cover BF16 compute error for the largest effects. With the 0.125-per-margin compute term: re-run 0/58 outside, saved 2/58 still outside (full effect, complement loss L20/KV0, against a bound of 0.5). Largest |FP32 − saved| over all 58: 1.03 (sum of singletons, bound 4.47).
- Unresolved in the saved run but |FP32| ≥ 0.25: interaction L20 (+0.499), KV-group interaction (−0.301), nonadditivity (+1.804).
- The two "sign changes" the analysis flags (interaction L19: FP32 +0.0003 vs saved −0.125; interaction L21: FP32 −0.0024 vs saved +0.125) are against saved values inside their 0.5 bound; FP32 puts both near zero.
- L20 pair fit M(a, b) = c0 + ca·a + cb·b + cab·a·b, FP32: c0 −4.647, ca +3.454, cb +0.797, cab +0.499 (all resolved). Saved BF16: −4.75, +3.875, +0.875, +0.375 (cab unresolved). The L20 KV0 × KV1 interaction is positive and about one seventh of ca.

### (c) Max source attention vs cell effects (FP32, 18 cells)

| y | Pearson | Spearman | leave-one-out Pearson (min to max) | leave-one-out Spearman (min to max) |
|---|---|---|---|---|
| singleton effect | 0.812 | 0.360 | 0.631 (drop L20/KV0) to 0.901 (drop L26/KV1) | 0.240 to 0.498 |
| complement loss | 0.823 | 0.263 | 0.656 (drop L20/KV0) to 0.872 (drop L20/KV1) | 0.125 to 0.380 |

The Pearson figure rests on a few large cells (L20/KV0 at 0.927 max attention); the rank correlation is weak. Max source attention does not order the 18 cells by effect.

### (d) Read cuts on the first receiver (row 21 reading row 18), FP32

Native: margin −4.647, tokens ` cold`, `\n`, `Description`. All-cut of row 21's reads of row 18: shift +4.767 (zero, first token ` hot`) and +4.493 (renorm, ` warm`). All-cut of row 21's reads of `icy` (row 2): −0.001 (zero), −0.088 (renorm). Shift = margin(arm) − margin(native); floors are the recompute-sham / delta-vs-recompute floors of each arm (≤ 7.6e-6 throughout).

Shifts, recoveries and controls (MEASURED from the rows; unchanged by the fix):

| set (item `freezing`) | cut: zero / renorm | cut as fraction of the all-cut: zero / renorm | keep: zero / renorm (recovery) | random control (cut, zero / renorm) | other-item control (cut): shift, mass share | 
|---|---|---|---|---|---|
| L20H5 | +1.846 / +2.580 | 0.387 / 0.574 | +3.164 / +2.710 (0.34 / 0.40) | +0.005 / +0.090, less mass; **the draw is L20H11, a named test head** (both draws) | ≤ 0.001; 0.0011 |
| L21H0 | +0.210 / +0.378 | 0.044 / 0.084 | +3.775 / +3.660 (0.21 / 0.19) | −0.014 / +0.069, less mass (both draws the same set) | ≤ 0.003; 0.027 |
| L20H11 | +0.005 / +0.090 | 0.001 / 0.020 | +4.337 / +4.265 (0.09 / 0.05) | +1.846 / +2.580, more mass; **the draw is L20H5** (both draws) | ≤ 0.001; 0.0069 |
| named3 | +2.865 / +3.936 | 0.601 / 0.876 | +2.411 / +2.058 (0.49 / 0.54) | +0.078 / +0.120, less mass (both draws the same set) | ≤ 0.003; 0.0105 |
| rank3 (L20H5, L18H0, L17H0) | +2.776 / +3.126 | 0.582 / 0.696 | +2.332 / +2.176 (0.51 / 0.52) | +0.123 / +0.197, less mass; **the draw contains L20H11** (both draws) | ≤ 0.001; 0.0005 |
| rank12 | +4.200 / +4.710 | 0.881 / 1.048 | +0.906 / +0.837 (0.81 / 0.81) | +0.213 / +0.410, less mass (both draws the same set) | ≤ 0.010; 0.0065 |

The mass share is the other-item control's mass (row 21's attention to row 2 through the same heads) over the target's (to row 18), from the per-item masses stored in the sealed plan. Outcomes are |margin shift|, so a control that moves the margin the other way (L21H0's −0.014) still counts as that much damage.

Verdicts, before and after the 2026-11-13 fix (`full-tables.json` vs `full-verdicts-20261113.json`; MEASURED by re-running both versions of `verdicts()` on the same 491 rows):

| set | `freezing` cut: before → after | `freezing` keep | `icy` cut: before → after | `icy` keep |
|---|---|---|---|---|
| L20H5 | set_only → **control_missing** | not_supported (low recovery) | not_supported → **unresolved** (effect 0.4% of the all-cut in the zero cut type) | not_supported |
| L21H0 | set_only → **unresolved** (effect 4.4% / 8.4% of the all-cut; below the 5% choice) | not_supported | not_supported | not_supported |
| L20H11 | set_only → **control_missing** | not_supported | not_supported | not_supported |
| named3 | set_only → set_only (set size ruled out; mass and generic damage open) | not_supported | not_supported → **unresolved** (effect 3.0% in renorm) | not_supported |
| rank3 | set_only → **control_missing** | not_supported | not_supported → **control_missing** | not_supported |
| rank12 | set_only → set_only (set size ruled out; mass and generic damage open) | set_only | not_supported → **set_only** (mass and set size ruled out; generic damage open) | not_supported |

Counts over the 24 records: before 7 set_only / 17 not_supported; after 4 set_only / 13 not_supported / 4 control_missing / 3 unresolved. With the named-head check reported as a flag only (not excluding the draw), the counts are 6 set_only / 13 not_supported / 2 control_missing / 3 unresolved. Three records depend on the 5% minimum-effect fraction (freezing L21H0, icy L20H5, icy named3); with no bar they would be set_only. The 5% is a choice made in this fix pass, after the data were seen; it is not in the sealed plan.

What changed and why:
- **Named heads as random controls (excluded).** L20H5's and L20H11's draws are each other, and rank3's draw contains L20H11. A draw that contains another named test head of the item no longer counts as a random control. Both draws of every `freezing` set are identical; the second is now listed as a duplicate and counted once.
- **Near-empty other-item cuts (rule nothing out).** For `freezing`, the other-item control removes 0.05% to 2.7% of the target's mass. An other-item control now rules out generic damage only when its mass share is at least 0.10 (a choice; `OTHER_ITEM_MIN_SHARE`). Before the fix, "generic damage is ruled out" rested on these near-empty cuts.
- **Relative mass labels.** The stored labels used an absolute tolerance of 0.02. Re-labelled with the relative rule (within 10% of the target's mass, absolute floor 0.001), the 48 random-control labels change as: matched → more 12, matched → less 6, matched → matched 6, less → less 20, more → more 4. For `icy`, 18 of the 24 draws that were called "matched" are not: they carry up to 14× the target's mass. A heavier control cannot refute, which is why several `icy` records are no longer `not_supported`. The `icy` other-item control (the same heads on row 18) carries 37× to 1,937× the target's mass and also cannot refute now.

How to read this:
- **Set size is not ruled out for L20H5, L20H11 or rank3**: their only random control is another named head (the earlier text said set size was ruled out for every set; for L20H11 the earlier record itself ruled out only generic damage). For named3 and rank12, set size is ruled out by a draw that holds no named head; mass is not, because the same-layer random sets cannot reach the named heads' attention to row 18 and the matcher falls back to the closest-mass set. Generic damage is not ruled out for any `freezing` set, because the other-item cut is near empty. Whether a head with equal mass and a different role exists is not yet known; that needs another prompt or a cross-layer mass match.
- `icy` rank12 is `set_only` on a 0.0096 margin shift. That is 10% (renorm) to 800% (zero) of the `icy` all-cut, because the `icy` all-cut itself barely moves the margin (−0.0012 zero, −0.088 renorm). The relative bar does not catch an item whose all-cut is this small; read this record as "this item hardly matters to the margin", not as a needed set.
- L20H5 alone carries the largest single-head share (2.58 of the 4.49 renorm all-cut). L20H11 alone does almost nothing (0.005 zero, 0.090 renorm). Cut effects are not additive: named3 3.94 (renorm) vs 2.58 + 0.38 + 0.09 = 3.05 summed singly.
- Keeping only L20H5's read (every other head's read of row 18 cut) recovers 34 to 40% of the all-cut shift. Keeping the 12 label-free heads recovers 81%. First tokens: every keep arm keeps the native ` cold` except renorm keep-L20H11 (` warm`); the rank12 cut (both cut types) and the all-cut are the only cuts that change it (` warm`, or ` hot` for the zero all-cut). The margin recovery above is the finer measure.
- Label-free ranking (`contrast_ranking` on row 21's FP32 attention, row 18 against the other word rows): L20H5 rank 0, L21H0 rank 4, L20H11 rank 5; rank3 recovers L20H5 only, rank12 recovers all three. The trivial raw-mass ranking (most attention to row 18) recovers the same: L20H5 in the top 3, all three in the top 12. The contrast ranking adds nothing over raw mass on this prompt. Top-source baselines: over all 336 heads, position 0 is the best trivial predictor of each head's top source (0.586); "row 18" is 0.062. Within named3 and rank3, "offset 3 from the end" (= row 18) matches 1.0, the same as the nominator, so row 18 is not distinguishable from a fixed offset on this one prompt.

### (e) Backend floors

| lane | recompute sham: |margin shift| max / median (n) | recompute sham: max \|Δlogit\| | delta vs recompute twin: \|margin gap\| max / median (n) | value `sham-late` / `replay-recipient` vs native: max \|Δlogit\| |
|---|---|---|---|---|
| FP32 (full read panel) | 3.8e-6 / 0 (196) | 1.16e-5 | 9.5e-6 / 1.9e-6 (98) | 0.0 / 0.0 |
| BF16 (smoke read panel) | 0.125 / 0.125 (12; none zero) | 0.25 | 0.125 / 0.125 (6; 2 zero) | 0.0 / 0.0 |

In BF16 the recompute path alone moves the margin by one step (0.125): a BF16 read cut whose effect is below about 0.25 is not resolvable on this backend. The L20H5 cut in BF16 (+1.875 zero, +2.625 renorm; smoke and full runs agree) stays far above it. Native-recipe tolerance (smoke): 1.3e-5 in FP32, 0.19 in BF16 (see above).
