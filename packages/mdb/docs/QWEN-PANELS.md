# Qwen value and read-cut panels

Status: November 12, 2026. Code and the sealed smoke plans are in place. **No model run has finished yet.** The smoke job `job-9020b3bfd5e7` is queued: the Beast mrun agent was stopped through systemd at 17:24:45 PST and had not restarted when this was written (18:18 PST). Every number below is either from the CPU dry run on a tiny random model (marked DRY RUN) or ASSERTED. The results section is empty until the runs finish.

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
  - **Random controls:** two random-head draws per set, matched by layer (strata = layer) and disjoint from the set. Mass is per item: the head's native row-21 attention to that item's row, matched within 0.02 where reachable.
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
$PY qwen_panel_stage.py submit RUN_SMOKE                                                  # job-9020b3bfd5e7 (queued)
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

## Results

Not yet measured; this section is filled when the runs finish. Scope: one prompt pair and one model; the 18 cells share one context and are not independent replications.
