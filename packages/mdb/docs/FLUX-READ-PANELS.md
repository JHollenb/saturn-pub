# FLUX read panels: sealed attention panels on the FLUX.2 Klein lane

This lane runs an `mdb.attention_panels` plan on FLUX.2 Klein 4B through `mdb.attention_port`. It replaces the one-off worker `saturn/experiments/2026-11-11-flux-read-graph/read_graph_worker.py` for read-cut experiments. It has three parts:

- `src/mdb/flux_read_panels.py`: framework-free logic. It checks plans and leases, builds and verifies regeneration recipes, maps worker rows to panel results, and computes the headfree "gone" score.
- `qualification/flux_read_panel_worker.py`: the mrun worker. It has two modes, panel and `--regenerate`.
- `qualification/flux_read_panel_stage.py`: offline commands to build plans, stage, submit and report.

Unit tests are in `tests/test_flux_read_panel.py`. Sealed plans, tables and verdicts are in `qualification/flux_read_panels_20261112/`.

## End to end

Use the mdb interpreter, with `PYTHONPATH` pointing at the checkout under test. The payload wheel is built from whichever `mdb` is imported, and `stage` prints `mdb_source`.

```bash
PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
export PYTHONPATH=<worktree>/packages/mdb/src
cd packages/mdb/qualification
$PY flux_read_panel_stage.py plans flux_read_panels_20261112/plans          # 1 build + 2 seal + leases
$PY flux_read_panel_stage.py stage RUN --plans .../plans --job smoke         # 3 stage (also: legacy, full:N)
$PY flux_read_panel_stage.py submit RUN [--attempt N]                        # 4 submit (mrun, priority 70)
rtk proxy rsync -a beast:/mnt/hdd8tb/mdb/research/<job>/ <results>/<job>/    # 5 collect
$PY flux_read_panel_stage.py report --plans .../plans --results <results>/job-* --out OUT   # 6 verdicts
$PY flux_read_panel_stage.py stage RUN --plans .../plans --job regenerate \
    --recipes <recipes> --arms smoke-legacy@4242:z:own60:ball full-00@4242:m:own60:red   # 7 regenerate, then submit
```

1. **Build.** `plans` calls `read_panel` once per word and merges the results with `merge_panels`. One call per word is needed because `RandomControls.mass` is a single vector shared by every item; see Workarounds. It then adds the prompt-level arms with `prompt_level_arms`. It writes four plans:
   - `smoke-legacy` / `smoke-new`: the word `ball`, seed 4242;
   - `legacy`: the headfree arm list exactly as executed, 224 branches;
   - `full`: all six words and both seeds, every control, 2,568 branches.
2. **Seal.** Each plan is written in canonical JSON, so the file's sha256 is the plan hash, with `<name>.seal.json` beside it. `leases_within` splits the plan into leases. Each lease's arms plus its reference roots must fit in the scheduler's 256 work items (`scheduling_contract.MAX_LOCAL_WORK_ITEMS`). This gives a lease cap of 251 for `full`, which has 11 leases.
3. **Stage.** Each job carries one sealed plan file and lease per pass in the payload. The smoke job has two passes, `legacy` and `native` precision. The run config holds the plan's sha256 and the lease record. `source_hashes` declares the `mdb/attention_port.py`, `mdb/attention_panels.py` and `mdb/flux_read_panels.py` bytes, and the bootstrap checks them against the installed wheel.
4. **Run.** The worker does the following, in order:
   - It refuses any plan whose sha256 differs from the run config's, and any lease whose seal, plan or branches do not match (`load_plan`, `check_lease`).
   - It prints `FLUXPANEL_MODULES`: the file path and sha256 of each shared module as imported, and the payload seal's entry for it. It raises if they differ.
   - It loads Klein once and encodes every prompt the lease needs: the native prompt and each word-deleted prompt. Then it releases the text encoder.
   - For every (context = seed) it runs the reference roots without the port: native, `del` (word deleted from the prompt) and `iso`. `iso` is the read-graph `hybrid(..., "own")` conditioning: the word is present only in its own rows, the rows after it come from the deleted prompt's encoding shifted by the deleted token count, and the padding rows come from the deleted prompt.
   - It runs each lease branch through `with port.apply(*port_cuts(arm), forwards=4)` and checks `port.calls == 100` and `port.forward_count == 4`. Arms that cut nothing (`iso:`, `del:`, `native:`) also run through the port, and each is checked to be bit-identical to its root (`equals_root`).
   - There are no retries.
5. **Collect.** Results go to `beast:/mnt/hdd8tb/mdb/research/<job>/flux-read-panel/`.
6. **Report.** `report` merges the per-lease `results.<tag>.px_mad.json` files with `merge_results`, which refuses other plans and duplicate outcomes. It then runs `attach_results`, `arm_effects` and `verdicts`, and writes the legacy identity table, the gone tables, the regeneration table and the verdict records.
7. **Regenerate.** `--arms TAG@CONTEXT:ARM_ID` picks a recipe by pass tag, context and arm. The pass tag is needed because the same arm id exists in a legacy pass and a native pass. The chosen recipes are copied into a new payload, and the worker runs `--regenerate CONTEXT:ARM_ID ... --recipes recipes`.

## What is stored per arm

| object | stored | where |
|---|---|---|
| decoded image | yes (PNG) | `images/<pass>/<context>/<arm>.png` |
| recipe (sealed JSON, ~10-22 KB) | yes | `recipes/<pass>/<context>/<arm>.json` |
| scalars | yes | `metrics.<pass>.json` |
| panel results (px MAD, latent rel vs the arm's reference) | yes | `results.<pass>.<metric>.json` |
| final latent | **no** (hash only), unless `retain=["latents"]` | `latents/...npy` |
| attention captures | **no** (this worker runs no observers) | |
| weights | never | checkpoint records and parameter pointers only |

Each `metrics.<pass>.json` row holds the following scalars:

- distances (`px_mad`, `px_max`, `latent_rel`, `rgb_equal`, `latent_equal`) to the arm's reference arm, and to the native, `del` and `iso` roots;
- `vs_floor_of`, the direct delta/recompute twin gap, on `/cut_recompute` floors;
- the latent sha256 (float32 bytes), the decoded-RGB sha256 and the PNG sha256;
- `port_calls`, `forward_count` and the recompute counters from `port.log`;
- seconds and peak allocated VRAM;
- for legacy passes, the pixel comparison with the saved 2026-11-11 reference PNG.

Per job the worker also writes:

- `checkpoint.json`: every checkpoint file with its byte count and safetensors header sha256, plus the full file sha256 when `hash_checkpoint` is set (smoke and regenerate jobs);
- `parameters.json`: one pointer per attention tensor of the transformer, using the `parameter_artifacts` source fields `file`, `tensor`, `dtype`, `shape` and `data_offsets`. `payload_sha256` is not computed, so these are pointers, not restorable artifacts;
- `roots.json`, `tokens.json`, `modules.json`, `summary.json`, `timeline.json`, and `files.json` (the sha256 of every output file).

### The recipe

A recipe has the fields `schema`, `arm`, `model`, `software`, `input`, `noise`, `port` and `expected`, plus a `fingerprint`, which is the sha256 of the canonical JSON. `validate_recipe` refuses an edited or incomplete recipe.

- **Model**:
  - the adapter's `model_identity`, which hashes the content of the transformer and VAE;
  - the model path and pipeline class;
  - the sha256 of `checkpoint.json` and `parameters.json`;
  - the attention blocks the cut touches.
- **Input**:
  - the prompt text;
  - how the conditioning is built: `native`, `deletion` or `hybrid_own`, with the word, the deleted prompt and the row plan;
  - the tokenizer settings.
- **Noise**: the seed, the CUDA generator, the height, width and steps, and the schedule rule.
- **Port**:
  - the backend, `precision`, `zero_method`, `empty` and `forwards`;
  - the cut specs (`Cut.to_dict()`: receivers, sources, heads, blocks, steps, `renormalize`, `sham`, per-cut `zero_method`);
  - the row table, as runs, for every row name the cuts use.
- **Software**:
  - the Python, torch, diffusers, transformers, numpy, Pillow, saturn_pub, CUDA and cuDNN versions;
  - the GPU and host;
  - a fingerprint of the adapter's execution ABI;
  - the sha256 of the worker and of the three shared modules.
- **Expected**: the conditioning, initial noise, final latent, decoded RGB and PNG hashes.

### What `--regenerate` does and does not guarantee

`--regenerate` rebuilds each named arm from the root alone, meaning the checkpoint, prompt text and seed. It re-encodes the prompt, rebuilds the conditioning from the recipe's construction, draws the noise from the seed, and rebuilds the port from the recipe's precision, zero method, row table and cut specs (`cut_from_dict`). It then reports, hash by hash, whether the conditioning, initial noise, final latent, decoded RGB and PNG equal the recipe's. It also reports any software-lane field that differs (`lane_drift`). It refuses to run a recipe whose model identity differs from the loaded model.

It **does** guarantee that a match means the same bytes were produced from the root on this host and this software lane. That lane is the CUDA device, torch/diffusers/transformers versions, saturn_pub wheel, TF32 setting and shared-module bytes.

It **does not** guarantee:

- the same bytes on another GPU, driver or library version: BF16 kernels are not bit-stable across lanes, and a mismatch there is reported, not repaired;
- that the text encoder weights are unchanged. They are not part of `model_identity`; the conditioning hash catches a change in its output;
- the full checkpoint-file hash, except in jobs run with `hash_checkpoint`.

This is the FLUX counterpart of the Qwen hydration path in `native_replay_recipe`, which has its own Qwen-only profile and was not modified.

## Reproduction of the 2026-11-11 headfree run (job-a0bca63168b4)

Klein 4B, 256 px, 4 steps, BF16, TF32 off, prompts `cat` and `car`, seeds 4242 and 9001.

Every number below is MEASURED from the saved job outputs unless it says otherwise. Per-job wall time and VRAM are in `qualification/flux_read_panels_20261112/jobs.json`. The full text report is `report/report.txt`, written by the `report` command. The primary verdict report is `report/e-report-20261113.txt` with `report/e-summary-20261113.json` and `report/e-verdicts-fixed-20261113.json`: all 78 records recomputed from the saved results with the verdict logic of 2026-11-13 by `qualification/panel_verdicts_recompute_20261113.py` (no new runs). `report/d-verdicts-*.json` hold the sealed-logic verdicts and are kept as history. Images and recipes are on Beast under `/mnt/hdd8tb/mdb/research/<job>/flux-read-panel/` and on the laptop under `artifacts/mdb-flux-read-panels-20261112/results/`. They are not committed.

The sealed plans are in `qualification/flux_read_panels_20261112/plans/`:

| plan | sha256 | arms | branches | contents |
|---|---|---|---|---|
| `smoke-legacy` | `ce7dc5cf...` | 20 | 20 | `ball`, seed 4242, the old arms |
| `smoke-new` | `7e14e78c...` | 218 | 218 | `ball`, seed 4242, the full panel |
| `legacy` | `b52a26a9...` | 112 | 224 | the headfree arm list as executed |
| `full` | `94ef887e...` | 1,284 | 2,568 | 156 test, 336 control, 762 floor and 30 reference arms; 11 leases of ≤251 branches |

The full panel holds, per word:

- every old arm;
- both cut types for every set;
- the floor arms of every cut (`/sham_recompute` and `/cut_recompute` for zero cuts, `/sham` for renormalising cuts);
- a matched random set for cut and keep (`rand0-*`, disjoint, stratum and block matched, mass searched within an absolute tolerance of 0.02). The random-control mass is each word's own write-mass share from the seed-4242 capture only (`flux_read_panel_stage.word_masses`), applied to both contexts: whether a draw is mass-matched in the 9001 context is not measured. Re-labelled with the relative rule of 2026-11-13 (within 10% of the target's mass, absolute floor 0.001), 22 of the 78 draws are matched, 52 carry less mass and 4 more; the absolute search had flagged 24 as matched, including the `own240-minus-own60` draws of ball, rain and snow with 4.3× to 11.3× the target's mass (target 0.0015 to 0.0039, control 0.0167 to 0.0189);
- the other-item control (`@rows-<next word>`);
- `own240-minus-own60`;
- prompt-level `native:<pid>` and `img<-alltext` in both types, with floors.

All 78 random draws were emitted, and `panel.identity` records whether each one separates identity from mass. The draws did not exclude the word's other named test sets: all 78 contain at least one unit of another named set of the same word (MEASURED; the overlap is about what a uniform draw gives, median 1.4× the expected count, because the named sets of a word cover a large part of the 600 heads). A test checks that the `legacy` plan equals the 224 executed (seed, arm) pairs. Another checks that every one of those arms appears in `full` with the same kind, cut type, rows, units and condition.

### Jobs

Klein 4B on Beast (RTX 4080). Priority 70, 10,400 MB VRAM request (7,991 MB peak × 1.3; the smoke used 12,000 MB).

| job | what | mrun wall s | worker s | peak alloc MB | branches |
|---|---|---|---|---|---|
| job-a34e4d6fe9dc | smoke (`ball`, 4242): legacy pass + new pass | 620 | 540 | 7,991 | 20/20 + 140/218 (budget stop) |
| job-2b516ba78da6 | legacy, all 6 words, both seeds | 250 | 211 | 7,991 | 224/224 + 28 roots |
| job-68368184de20 | full lease 0 | 292 | 261 | 7,991 | 234/234 |
| job-9b9ec68fc42a | full lease 1 | 354 | 295 | 7,991 | 232/232 |
| job-c4b924f3295d | full lease 2 | 429 | 360 | 7,991 | 247/247 |
| job-ea0bcbfe960f | full lease 3 | 269 | 237 | 7,991 | 242/242 |
| job-17f61da18312 | full lease 4 | 400 | 382 | 7,991 | 242/242 |
| job-baa6927a1a6c | full lease 5 | 333 | 296 | 7,991 | 239/239 |
| job-de42c09696dd | full lease 6 | 253 | 239 | 7,991 | 242/242 |
| job-eb99a2f1d947 | full lease 7 (attempt 2) | 259 | 246 | 7,991 | 249/249 |
| job-03723dc2595b | full lease 8 (attempt 3) | 196 | 194 | 7,991 | 247/247 |
| job-2c64f0609f8f | full lease 9 | 278 | 252 | 7,991 | 247/247 |
| job-0ef55a497c49 | full lease 10 | 147 | 141 | 7,991 | 147/147 |
| job-af5bfaf0fde2 | regenerate 3 arms (separate job) | 25 | 21 | 7,991 | 3/3 match |

All 2,568 branches of the `full` plan ran (sum over the 11 leases). Failed attempts, kept in `jobs.json` and not used:

- job-d8790bc10fbb (lease 7) and job-a41e454eefe1 (lease 8): `lost`, "lease expired (agent silent)". The Beast mrun host agent was stopped by systemd at 16:40:11 PST on 2026-11-12. Lease 7's worker spent 866 s importing from the HDD-backed uv cache and its 540 s budget stopped it before the first branch; lease 8 likewise budget-stopped after roots only. Partial outputs stay on Beast and are excluded from the report.
- job-871e9014eda6 (lease 8, attempt 2): `killed_vram`. mrun's wedged-queue clamp lowered its reservation from 10,400 to 5,776 MB while another user's job held the GPU, and the job was killed at that ceiling during model load. Attempt 3 ran on an idle GPU.

mrun's guarded admission is idempotent on (experiment, command, payload, reservation, priority, ...), so resubmitting a lost job returns the lost job id. `submit --attempt N` appends `-attemptN` to the experiment name to admit a fresh job on byte-identical payloads (checked with `diff -r` against the original staged payload).

Timing:

- Mean time per branch was 0.40 s on the legacy lane and 0.46 s on the new-precision lanes. Keep arms took about 0.6 s.
- The first run in each job took 10-29 s of CUDA warm-up.
- Imports took 28-866 s and dominate the variance in job wall time.

### (a) Legacy precision against the saved reference PNGs

| compared | pixel-identical | not identical |
|---|---|---|
| legacy arms, all 6 words × 2 seeds plus prompt-level (job-2b516ba78da6) | 224 / 224 | none |
| smoke legacy arms (job-a34e4d6fe9dc) | 20 / 20 | none |
| roots: native, del and iso per (word, seed) | 31 / 31 | none |

The comparison was done twice, both times against the saved images:

- in the job, against `beast:/mnt/hdd8tb/mdb/research/job-a0bca63168b4/read-graph-result/`;
- on the laptop, against `saturn/experiments/2026-11-11-flux-read-graph/runs/headfree/result/`, giving 244 of 244 identical.

`native:<pid>` equals both the old `native.<pid>.<seed>.png` and the old `arm.<pid>.<seed>.sham.png`. The gone table computed from the legacy pass equals `runs/headfree/headfree_table.json` with a maximum absolute difference of 0 over all 6 words, 2 seeds and 15 columns.

Determinism across jobs: 188 (context, arm or root) keys were computed in two or more succeeded jobs, 160 arms and 28 roots (recount by the 2026-11-13 audit; this text said 183 with 23 roots before, and the recount was not repeated in the fix pass). Every one has identical latent and RGB hashes. Inside each lease, every `iso:`, `del:` and `native:` arm run through the port is bit-identical to its port-free root.

### (b) Gone scores at new precision against the old table

The gone score is `gone = (d(arm) - d(iso)) / (d(none) - d(iso))`, with `d = 1 - MAD(arm, del) / MAD(native, del)`, as in `headfree_table.py`. The "old" row is also the legacy-precision result, which is identical.

The "floor bound" row is the arm's floor in gone units: `floor_px / (MAD(native, del) × span)`. The floor in px is the larger of two distances: the recompute sham's distance from iso, and the direct delta-to-recompute twin gap. Within one lane (two arms sharing the same del, iso and none images), the triangle inequality makes a gone difference smaller than this bound inseparable from arithmetic noise. The inequality does not cover a comparison across lanes (old precision vs new): the span differs between lanes, and the floor measures only the new lane's sham drift. Measured directly, the cross-lane image distance of an arm exceeds this floor bound in 56 of the 180 cells (up to 1.92×; audit 2026-11-13).

All 6 words × 2 seeds are complete. Over the 180 cells of the report's 15 columns, the largest |new − old| is 0.08 (red 9001 own240, floor bound 0.198). At full precision, 0 of the 180 gone differences exceed their floor bound; the one apparent miss, black 4242 `m:own60` (0.42 vs 0.45), is a rounding artifact: |Δ| = 0.02358 against a bound of 0.02508. That the new precision stays within this floor is an OBSERVATION (0/180 gone-table cells exceed it), not a consequence of the triangle inequality, since the direct image distance exceeds the bound in 56/180 cells. The table shows a column subset; `report/b-gone-full.json` holds every column, including all `m:` twins, `own240-minus-own60`, the random controls and the other-item controls.

| word | seed | source | own60 | own240 | rand240 | modI | nounI | keep-own60 | keep-nounI | m:own60 |
|---|---|---|---|---|---|---|---|---|---|---|
| red | 4242 | old (= legacy, exact) | 0.48 | 0.66 | 0.12 | 0.25 | 0.03 | 0.16 | 0.66 | 0.31 |
|  |  | new precision | 0.47 | 0.64 | 0.13 | 0.27 | 0.02 | 0.14 | 0.66 | 0.29 |
|  |  | floor bound | 0.092 | 0.094 | 0.075 | 0.083 | 0.075 | 0.072 | 0.109 | 0.082 |
| red | 9001 | old (= legacy, exact) | 0.61 | 0.73 | 0.23 | 0.46 | -0.01 | 0.11 | 0.93 | 0.36 |
|  |  | new precision | 0.61 | 0.81 | 0.23 | 0.46 | 0.00 | 0.14 | 1.00 | 0.37 |
|  |  | floor bound | 0.140 | 0.198 | 0.099 | 0.097 | 0.115 | 0.118 | 0.119 | 0.111 |
| black | 4242 | old (= legacy, exact) | 0.31 | 0.72 | -0.03 | 0.14 | 0.11 | 0.03 | 0.58 | 0.45 |
|  |  | new precision | 0.31 | 0.71 | -0.05 | 0.13 | 0.12 | 0.04 | 0.58 | 0.42 |
|  |  | floor bound | 0.047 | 0.054 | 0.067 | 0.052 | 0.042 | 0.033 | 0.035 | 0.025 |
| black | 9001 | old (= legacy, exact) | 0.39 | 0.77 | 0.10 | 0.21 | 0.08 | 0.03 | 0.67 | 0.34 |
|  |  | new precision | 0.39 | 0.77 | 0.11 | 0.21 | 0.07 | 0.03 | 0.67 | 0.33 |
|  |  | floor bound | 0.035 | 0.034 | 0.031 | 0.034 | 0.032 | 0.033 | 0.030 | 0.024 |
| ball | 4242 | old (= legacy, exact) | 0.51 | 0.66 | 0.25 | -0.09 | 0.28 | -0.20 | 0.30 | 0.30 |
|  |  | new precision | 0.51 | 0.68 | 0.24 | -0.08 | 0.30 | -0.19 | 0.29 | 0.29 |
|  |  | floor bound | 0.071 | 0.060 | 0.083 | 0.057 | 0.089 | 0.064 | 0.086 | 0.071 |
| ball | 9001 | old (= legacy, exact) | 0.47 | 0.49 | 0.42 | -0.02 | 0.23 | -0.04 | 0.80 | 0.38 |
|  |  | new precision | 0.46 | 0.48 | 0.41 | -0.03 | 0.22 | -0.04 | 0.80 | 0.37 |
|  |  | floor bound | 0.041 | 0.053 | 0.045 | 0.042 | 0.063 | 0.049 | 0.056 | 0.041 |
| snow | 4242 | old (= legacy, exact) | 0.27 | 0.46 | 0.49 | 0.27 | 0.04 | 0.65 | 0.81 | 0.32 |
|  |  | new precision | 0.26 | 0.45 | 0.46 | 0.27 | 0.03 | 0.66 | 0.81 | 0.32 |
|  |  | floor bound | 0.020 | 0.016 | 0.047 | 0.017 | 0.014 | 0.024 | 0.020 | 0.020 |
| snow | 9001 | old (= legacy, exact) | 0.13 | 0.17 | 0.16 | 0.11 | 0.09 | 0.56 | 0.74 | 0.17 |
|  |  | new precision | 0.12 | 0.17 | 0.15 | 0.11 | 0.09 | 0.56 | 0.73 | 0.17 |
|  |  | floor bound | 0.020 | 0.011 | 0.022 | 0.017 | 0.037 | 0.016 | 0.014 | 0.013 |
| night | 4242 | old (= legacy, exact) | 0.56 | 0.85 | 0.45 | 0.20 | 0.09 | 0.69 | 1.05 | 0.54 |
|  |  | new precision | 0.56 | 0.85 | 0.46 | 0.19 | 0.07 | 0.69 | 1.06 | 0.55 |
|  |  | floor bound | 0.053 | 0.073 | 0.064 | 0.036 | 0.040 | 0.058 | 0.028 | 0.053 |
| night | 9001 | old (= legacy, exact) | 0.28 | 0.57 | 0.20 | 0.12 | -0.01 | 0.48 | 0.99 | 0.22 |
|  |  | new precision | 0.28 | 0.58 | 0.20 | 0.12 | -0.01 | 0.48 | 0.99 | 0.22 |
|  |  | floor bound | 0.033 | 0.045 | 0.020 | 0.020 | 0.025 | 0.024 | 0.016 | 0.017 |
| rain | 4242 | old (= legacy, exact) | -0.10 | 0.22 | -0.06 | 0.15 | -0.24 | 0.38 | 0.89 | -0.12 |
|  |  | new precision | -0.10 | 0.22 | -0.06 | 0.16 | -0.25 | 0.40 | 0.90 | -0.12 |
|  |  | floor bound | 0.137 | 0.128 | 0.110 | 0.159 | 0.129 | 0.135 | 0.095 | 0.137 |
| rain | 9001 | old (= legacy, exact) | 0.09 | 0.66 | 0.27 | 0.23 | -0.15 | 1.41 | 1.49 | 0.03 |
|  |  | new precision | 0.10 | 0.70 | 0.26 | 0.24 | -0.14 | 1.41 | 1.49 | 0.03 |
|  |  | floor bound | 0.127 | 0.122 | 0.108 | 0.124 | 0.187 | 0.226 | 0.109 | 0.118 |

### (c) Regeneration from recipes

The separate job job-af5bfaf0fde2 rebuilt three arms from their recipes alone (checkpoint, prompt text, seed), on the same host and software lane. All three matched every recorded hash: conditioning, initial noise, final latent, decoded RGB and PNG. `lane_drift` was empty, `port.calls == 100` and `forward_count == 4` for each, and the rebuilt cut specs equal the recipe's (`report/c-regenerate.json`).

| arm | precision | cut | conditioning | init noise | latent | RGB | PNG |
|---|---|---|---|---|---|---|---|
| `smoke-legacy@4242:z:own60:ball` | legacy | zero | = | = | = | = | = |
| `full-00@4242:m:own60:red` | native | renormalising | = | = | = | = | = |
| `full-00@4242:z:keep-own60:red` | native | delta zero, 540 heads | = | = | = | = | = |

The job confirmed the staged shared-module bytes (`attention_port` e315da2a, `attention_panels` 31a8595b, `flux_read_panels` f50dfcfe) and worker 1cfb7550. Its 21 s worker time includes the load.

### (d) Verdicts: `attach_results` → `arm_effects` → `verdicts`

The outcome for each arm is its px MAD from its reference arm, `iso:<word>`. The picture-judgement slot is empty (`picture=None`): no picture judgement was made. Every count in this section is MEASURED by running the named version of `verdicts()` on the same saved results (`report/d-results-px.json`, plan `94ef887e...`); no new job ran.

**Primary: all 78 records under the verdict logic of 2026-11-13** (`report/e-report-20261113.txt`, `report/e-summary-20261113.json`, records in `report/e-verdicts-fixed-20261113.json`; written by `qualification/panel_verdicts_recompute_20261113.py`).

| verdict logic | supported | set_only | not_supported | control_missing |
|---|---|---|---|---|
| sealed: `attention_panels` 5e09d95 (sha 31a8595b), the module the plan was sealed with | 27 | 0 | 51 | 0 |
| pre-fix: 166b334 (sha 4a39dbaa; the same file at the audited tip 420b38e), the branch's logic before this fix pass | 2 | 13 | 63 | 0 |
| **fixed: this fix pass (primary)** | **0** | **15** | **46** | **17** |
| fixed, with the named-unit check reported as a flag only | 2 | 14 | 61 | 1 |

The sealed logic predates the verdict fixes of 2026-11-12 (9451fc8, fde00c9) and of 2026-11-13, so its table is history, not a verdict. It counted only controls flagged `separates_identity`: every other-item control (including other-item KEEP controls, which are not sufficiency controls: on ball, `keep-own60@rows-snow` does more damage than the all-cut itself) and random controls only when mass-matched. Random or external sets of the same size that were not mass-matched were ignored even when they did equal damage (`snow own120`: `rand120` 0.1156 against the test's 0.1166, sealed `supported`). Between sealed and pre-fix logic, 25 of the 78 records change; between pre-fix and fixed, 22 change.

What the fix pass changed, and its effect here:
- **Random draws that contain a named test unit are excluded.** All 78 FLUX random draws contain at least one head of another named set of the same word (the overlap is near what a uniform draw gives, because a word's named sets cover a large part of the 600 heads). Under the rule, no FLUX random control is valid: no record can be `supported`, every keep record that is not refuted first by low recovery is `control_missing`, and set size is ruled out for no record. This is the main reason for the 17 `control_missing`. With the check reported as a flag only, `black keep-modI` and `night mod` stay `supported` (each beats a disjoint random control that is mass-matched by the relative rule, on seed-4242 mass). Whether those two survive a random draw that avoids the named sets is not yet known; it needs a new draw and a new run.
- **Other-item controls carry a mass relation** (other word's write-mass share through the same heads over this word's, seed-4242 capture). They rule out generic damage only with a share of at least 0.10 (a choice), and a heavier one (more than 10% above) cannot refute. Shares range 0.34 to 2.07 over the named sets and up to 12.8 for `own240-minus-own60` (snow), so no FLUX other-item cut is near empty. `ball noun` and `night modI` stay `not_supported`: their other-item controls (shares 1.07 and 1.06) are matched by the relative rule, so they may refute.
- **Relative mass labels.** `snow own240-minus-own60` was `not_supported` only because of a "matched" random control carrying 11× the target's mass; re-labelled `more`, it cannot refute, and the record is `set_only` (generic damage ruled out by the other-item control). `rain own240-minus-own60` (random control 4.3× the target's mass) is `control_missing`.
- **Effect fraction.** Every record now carries its effect as a fraction of the all-cut effect (necessity) or its recovery (sufficiency). The smallest necessity fraction of any FLUX record is 0.056 (`black own240-minus-own60`, `not_supported` anyway); no FLUX verdict depends on the 5% minimum-fraction choice (MEASURED: re-run with the bar at 0 changes no record).

**The 0.7 recovery threshold is post hoc.** `DEFAULT_RECOVERY = 0.7` is not in the sealed plan; it was added to the verdict code after the plan was sealed. Every keep verdict that is `not_supported` only because recovery is below 0.7 is marked in the last column, with the status it would have without a recovery threshold. Under the fixed logic that status is `control_missing` (no valid random keep control) for 20 records. With the named-unit check as a flag only, 18 records depend on the threshold: 15 would be at most `set_only` without it and 3 (`snow`, `night`, `rain` `keep-modI`) at most `supported`; `ball keep-own60` is one of the 15 (recovery 0.698 / 0.796 zero, 0.567 / 0.736 renorm, seeds 4242 / 9001; zero 4242 misses 0.7 by 0.0024). So `ball keep-own60` is reported as **at most `set_only`**: not_supported only via a threshold that was not sealed, and mass is open in any case.

Per record (necessity fraction = effect / all-cut effect; sufficiency fraction = recovery; over both cut types and both seeds):

| record | claim | sealed logic | pre-fix logic | **fixed** | fixed, named check as flag | effect fraction min–max | depends on the post-hoc 0.7 recovery threshold |
|---|---|---|---|---|---|---|---|
| red own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.432–0.690 | – |
| red own120 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.392–0.685 | – |
| red own240 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.493–0.782 | – |
| red mod | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.541–0.819 | – |
| red noun | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.315–0.366 | – |
| red modI | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.273–0.551 | – |
| red nounI | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.220–0.242 | – |
| red own240-minus-own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.145–0.187 | – |
| red keep-own60 | sufficient | not_supported | not_supported | **control_missing** | not_supported | 0.655–0.740 | flag: at most set_only without it |
| red keep-own120 | sufficient | not_supported | not_supported | **control_missing** | not_supported | 0.681–0.760 | flag: at most set_only without it |
| red keep-own240 | sufficient | not_supported | not_supported | **control_missing** | not_supported | 0.672–0.804 | flag: at most set_only without it |
| red keep-modI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.557–0.615 | fixed: control_missing without it; flag: at most set_only without it |
| red keep-nounI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.119–0.380 | fixed: control_missing without it |
| black own60 | necessary | supported | set_only | **set_only** | set_only | 0.345–0.488 | – |
| black own120 | necessary | supported | set_only | **set_only** | set_only | 0.579–0.709 | – |
| black own240 | necessary | supported | set_only | **set_only** | set_only | 0.670–0.755 | – |
| black mod | necessary | supported | set_only | **set_only** | set_only | 0.814–0.890 | – |
| black noun | necessary | not_supported | not_supported | **control_missing** | not_supported | 0.115–0.182 | – |
| black modI | necessary | not_supported | not_supported | **set_only** | not_supported | 0.141–0.296 | – |
| black nounI | necessary | not_supported | not_supported | **control_missing** | not_supported | 0.124–0.189 | – |
| black own240-minus-own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.056–0.081 | – |
| black keep-own60 | sufficient | supported | set_only | **control_missing** | set_only | 0.878–0.920 | – |
| black keep-own120 | sufficient | supported | set_only | **control_missing** | set_only | 0.906–0.933 | – |
| black keep-own240 | sufficient | supported | set_only | **control_missing** | set_only | 0.923–0.951 | – |
| black keep-modI | sufficient | supported | supported | **control_missing** | supported | 0.766–0.832 | – |
| black keep-nounI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.129–0.486 | fixed: control_missing without it |
| ball own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.409–0.602 | – |
| ball own120 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.426–0.633 | – |
| ball own240 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.487–0.692 | – |
| ball mod | necessary | not_supported | not_supported | **control_missing** | not_supported | 0.109–0.277 | – |
| ball noun | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.402–0.603 | – |
| ball modI | necessary | not_supported | not_supported | **control_missing** | not_supported | 0.096–0.161 | – |
| ball nounI | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.259–0.456 | – |
| ball own240-minus-own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.215–0.306 | – |
| ball keep-own60 | sufficient | supported | not_supported | **control_missing** | not_supported | 0.567–0.796 | flag: at most set_only without it |
| ball keep-own120 | sufficient | supported | not_supported | **control_missing** | not_supported | 0.654–0.837 | flag: at most set_only without it |
| ball keep-own240 | sufficient | supported | not_supported | **control_missing** | not_supported | 0.668–0.778 | flag: at most set_only without it |
| ball keep-modI | sufficient | not_supported | not_supported | **not_supported** | not_supported | -0.004–0.206 | fixed: control_missing without it |
| ball keep-nounI | sufficient | supported | not_supported | **not_supported** | not_supported | 0.153–0.573 | fixed: control_missing without it; flag: at most set_only without it |
| snow own60 | necessary | supported | set_only | **set_only** | set_only | 0.331–0.451 | – |
| snow own120 | necessary | supported | not_supported | **not_supported** | not_supported | 0.348–0.530 | – |
| snow own240 | necessary | supported | not_supported | **not_supported** | not_supported | 0.389–0.581 | – |
| snow mod | necessary | supported | set_only | **set_only** | set_only | 0.336–0.536 | – |
| snow noun | necessary | supported | set_only | **set_only** | set_only | 0.337–0.455 | – |
| snow modI | necessary | not_supported | not_supported | **set_only** | not_supported | 0.278–0.395 | – |
| snow nounI | necessary | supported | not_supported | **set_only** | not_supported | 0.101–0.319 | – |
| snow own240-minus-own60 | necessary | not_supported | not_supported | **set_only** | set_only | 0.285–0.427 | – |
| snow keep-own60 | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.259–0.620 | fixed: control_missing without it; flag: at most set_only without it |
| snow keep-own120 | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.399–0.664 | fixed: control_missing without it; flag: at most set_only without it |
| snow keep-own240 | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.475–0.675 | fixed: control_missing without it; flag: at most set_only without it |
| snow keep-modI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.213–0.450 | fixed: control_missing without it; flag: at most supported without it |
| snow keep-nounI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.134–0.290 | fixed: control_missing without it |
| night own60 | necessary | supported | set_only | **set_only** | set_only | 0.324–0.568 | – |
| night own120 | necessary | supported | set_only | **set_only** | set_only | 0.446–0.646 | – |
| night own240 | necessary | supported | set_only | **set_only** | set_only | 0.587–0.810 | – |
| night mod | necessary | supported | supported | **set_only** | supported | 0.310–0.521 | – |
| night noun | necessary | not_supported | not_supported | **control_missing** | not_supported | 0.118–0.249 | – |
| night modI | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.150–0.272 | – |
| night nounI | necessary | not_supported | not_supported | **control_missing** | not_supported | 0.102–0.161 | – |
| night own240-minus-own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.084–0.201 | – |
| night keep-own60 | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.307–0.526 | fixed: control_missing without it; flag: at most set_only without it |
| night keep-own120 | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.400–0.597 | fixed: control_missing without it; flag: at most set_only without it |
| night keep-own240 | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.546–0.687 | fixed: control_missing without it; flag: at most set_only without it |
| night keep-modI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.199–0.301 | fixed: control_missing without it; flag: at most supported without it |
| night keep-nounI | sufficient | not_supported | not_supported | **not_supported** | not_supported | 0.004–0.076 | fixed: control_missing without it |
| rain own60 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.302–0.560 | – |
| rain own120 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.423–0.715 | – |
| rain own240 | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.584–0.717 | – |
| rain mod | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.483–0.538 | – |
| rain noun | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.398–0.686 | – |
| rain modI | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.293–0.503 | – |
| rain nounI | necessary | not_supported | not_supported | **not_supported** | not_supported | 0.359–0.561 | – |
| rain own240-minus-own60 | necessary | not_supported | not_supported | **control_missing** | control_missing | 0.388–0.578 | – |
| rain keep-own60 | sufficient | supported | not_supported | **not_supported** | not_supported | 0.251–0.371 | fixed: control_missing without it |
| rain keep-own120 | sufficient | supported | not_supported | **not_supported** | not_supported | 0.229–0.415 | fixed: control_missing without it |
| rain keep-own240 | sufficient | supported | not_supported | **not_supported** | not_supported | 0.376–0.542 | fixed: control_missing without it; flag: at most set_only without it |
| rain keep-modI | sufficient | supported | not_supported | **not_supported** | not_supported | 0.175–0.289 | fixed: control_missing without it; flag: at most supported without it |
| rain keep-nounI | sufficient | supported | not_supported | **not_supported** | not_supported | 0.130–0.226 | fixed: control_missing without it |

**Secondary, superseded: the sealed-logic table (18 records, as first reported).** These rows are copied from `report/d-verdicts-selected.json` without editing. They were computed with `attention_panels` 31a8595b (5e09d95), the module sealed with the plan, which predates the verdict fixes; do not read them as verdicts. For a keep claim the separating control must have a **larger** effect, so a positive margin means the control's effect exceeds the keep arm's. Random controls marked mass-confounded (`separates_identity=False`) were not counted by that logic and are left out of the column below.

| item | unit set | claim | status | both types agree | zero cut 4242: effect / floor | renorm 4242: effect / floor | separating controls 4242 (zero): effect, margin, beyond floor | 9001 |
|---|---|---|---|---|---|---|---|---|
| red | own60 | necessary | not_supported | True | not_supported: 0.0311 / 0.0033 | not_supported: 0.0270 / 0.0029 | own60@rows-black (other_item) 0.0400, -0.0088, False | zero not_supported, renorm not_supported |
| red | modI | necessary | not_supported | True | not_supported: 0.0235 / 0.0030 | supported: 0.0204 / 0.0025 | modI@rows-black (other_item) 0.0255, -0.0020, False | zero not_supported, renorm not_supported |
| red | nounI | necessary | not_supported | True | not_supported: 0.0122 / 0.0027 | not_supported: 0.0112 / 0.0027 | rand0-nounI (random) 0.0122, +0.0000, False; nounI@rows-black (other_item) 0.0273, -0.0151, False | zero not_supported, renorm not_supported |
| red | keep-own60 | sufficient | not_supported | True | not_supported: 0.0131 / 0.0026 | not_supported: 0.0161 / 0.0025 | keep-own60@rows-black (other_item) 0.0126, -0.0005, False | zero not_supported, renorm not_supported |
| red | keep-modI | sufficient | not_supported | True | supported: 0.0219 / 0.0026 | not_supported: 0.0226 / 0.0026 | keep-modI@rows-black (other_item) 0.0262, +0.0042, True | zero not_supported, renorm not_supported |
| red | keep-nounI | sufficient | not_supported | True | not_supported: 0.0410 / 0.0039 | supported: 0.0327 / 0.0027 | keep-rand0-nounI (random) 0.0415, +0.0005, False; keep-nounI@rows-black (other_item) 0.0760, +0.0350, True | zero not_supported, renorm not_supported |
| black | own60 | necessary | supported | True | supported: 0.0615 / 0.0031 | supported: 0.0387 / 0.0017 | own60@rows-ball (other_item) 0.0144, +0.0471, True | zero supported, renorm supported |
| black | modI | necessary | not_supported | False | supported: 0.0373 / 0.0034 | not_supported: 0.0163 / 0.0019 | rand0-modI (random) 0.0143, +0.0230, True; modI@rows-ball (other_item) 0.0072, +0.0301, True | zero supported, renorm not_supported |
| black | nounI | necessary | not_supported | True | not_supported: 0.0170 / 0.0028 | not_supported: 0.0212 / 0.0028 | rand0-nounI (random) 0.0196, -0.0026, False; nounI@rows-ball (other_item) 0.0320, -0.0149, False | zero not_supported, renorm not_supported |
| black | keep-own60 | sufficient | supported | True | supported: 0.0101 / 0.0022 | supported: 0.0137 / 0.0018 | keep-own60@rows-ball (other_item) 0.0724, +0.0623, True | zero supported, renorm supported |
| black | keep-modI | sufficient | supported | True | supported: 0.0252 / 0.0064 | supported: 0.0262 / 0.0019 | keep-rand0-modI (random) 0.0766, +0.0514, True; keep-modI@rows-ball (other_item) 0.1009, +0.0757, True | zero supported, renorm supported |
| black | keep-nounI | sufficient | not_supported | True | not_supported: 0.1099 / 0.0023 | not_supported: 0.0855 / 0.0021 | keep-rand0-nounI (random) 0.0328, -0.0771, False; keep-nounI@rows-ball (other_item) 0.0686, -0.0413, False | zero not_supported, renorm not_supported |
| ball | own60 | necessary | not_supported | True | not_supported: 0.0537 / 0.0038 | not_supported: 0.0351 / 0.0038 | own60@rows-snow (other_item) 0.0536, +0.0001, False | zero supported, renorm supported |
| ball | modI | necessary | not_supported | True | not_supported: 0.0135 / 0.0031 | not_supported: 0.0138 / 0.0029 | rand0-modI (random) 0.0291, -0.0156, False; modI@rows-snow (other_item) 0.0667, -0.0532, False | zero not_supported, renorm not_supported |
| ball | nounI | necessary | not_supported | True | not_supported: 0.0338 / 0.0048 | not_supported: 0.0222 / 0.0048 | nounI@rows-snow (other_item) 0.0358, -0.0020, False | zero supported, renorm supported |
| ball | keep-own60 | sufficient | supported | True | supported: 0.0297 / 0.0035 | supported: 0.0372 / 0.0035 | keep-own60@rows-snow (other_item) 0.1990, +0.1694, True | zero supported, renorm supported |
| ball | keep-modI | sufficient | not_supported | True | not_supported: 0.0913 / 0.0026 | not_supported: 0.0681 / 0.0026 | keep-rand0-modI (random) 0.0554, -0.0359, False; keep-modI@rows-snow (other_item) 0.2117, +0.1204, True | zero not_supported, renorm not_supported |
| ball | keep-nounI | sufficient | supported | True | supported: 0.0447 / 0.0046 | supported: 0.0367 / 0.0030 | keep-nounI@rows-snow (other_item) 0.2291, +0.1843, True | zero supported, renorm supported |

The 18-record comparison first reported here ("changes 4 of the 18 records", `report-head-logic/`) covered only a subset; over all 78 records the pre-fix logic changes 25 records against the sealed logic (table above). The `report` command needed one change to run under the pre-fix logic: control entries no longer always carry `separates_identity`, so the text report reads it with `.get`.

## Workarounds and gaps in the shared modules

None of these blocked the run. The shared modules were not edited; each workaround lives in `flux_read_panels.py`.

1. **Per-item random-control mass** (`attention_panels.py:289-303`, `RandomControls`; `:342-349`, `_draw_random`). `RandomControls.mass` is one vector for every item. A word's own write-mass share is item-specific, so the mass-matched random draw needs one `read_panel` call per word. The per-word panels are then joined with `merge_panels`.
   - Needed: `RandomControls.mass` as `{item: vector}`, or a callable that takes the item.
2. **Prompt-level arms** (`attention_panels.py:561-584`). `read_panel` emits only per-item arms, with receivers and sources templated by the item. The read-graph `sham` and `img<-alltext` arms belong to the prompt as a whole, so `prompt_level_arms` builds them as `Arm` records that follow the generator's naming, floor and twin conventions.
   - Needed: a `prompt_references=` option, or extra all-cut templates without `{item}`.
3. **No `Cut.from_dict`** (`attention_port.py:157-166` has only `to_dict`). Regenerating a port from a recipe needs the inverse, so `flux_read_panels.cut_from_dict` provides it. It is tested on every cut kind, including shams and `floor_arms`, as a byte-equal round trip.
   - Needed: `Cut.from_dict` in the port.
4. **`metadata()` after exit loses the cuts** (`attention_port.py:435-443`). `__exit__` sets `self.cuts = ()`, so `port.metadata()` read after the `with` block reports no cuts. The worker therefore reads it inside the block.
   - Needed: keep the last apply's metadata, for example in `port.last_metadata`.
5. **References that are not arm ids** (`attention_panels.py:568, 574`). `iso:<w>` and `del:<w>` carry `reference="native"`, which is not an arm id unless `references` includes `"native"`. The worker resolves `native` to the root native image of the item's prompt.
   - Needed: either always emit a native reference arm, or document `"native"` as a reserved root name.
6. Already documented in `ATTENTION-PORT.md`: `port.log` recompute counters are empty under `precision="legacy"`, so legacy rows have `recomputed=None`.

Scheduler limit: one mrun job may run at most 256 local work items (`scheduling_contract.MAX_LOCAL_WORK_ITEMS`). One lease plus its reference roots is therefore one job, and `leases_within` sizes the lease cap so that every lease fits.
