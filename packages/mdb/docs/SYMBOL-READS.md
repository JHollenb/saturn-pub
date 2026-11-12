# Symbol reads: one writer for "this head read this source with this much mass"

Status: November 12, 2026. Module: [`mdb.symbol_reads`](../src/mdb/symbol_reads.py). Tests: [`tests/test_symbol_reads.py`](../tests/test_symbol_reads.py). FLUX converter: [`qualification/flux_read_graph_symbols.py`](../qualification/flux_read_graph_symbols.py).

Three sources record the same kind of fact: a head read a source with some mass. Before this module they did so in three ways. Native discovery built passive records by hand. `symbol_attention` built Project-backed Qwen records. The FLUX.2 Klein read-graph experiment had no records at all. All three now produce the same record through one writer.

## Two observation surfaces, one record

| Surface | Entry point | Canonical records | When to use |
|---|---|---|---|
| Project-backed | `project_reads(project, session_id, lifecycle, source=..., **attach_attention_kwargs)` or `reads_from_lifecycle(lifecycle, source=...)` | `symbol_attention.attach_attention`, unchanged | A retained native Session with cuts, receipts and a continue receipt |
| Passive | `PassiveSurface` + `passive_binding` / `passive_definition` / `passive_use` / `passive_lifecycle`, then `read_for_use` | Built by the writer | A non-resumable observation: a passive `StateCut` (native discovery), or a saved capture with no cut (FLUX) |

Both surfaces end in `symbol_read(...)`, the single constructor. It returns a sealed `SymbolRead`:

| Field | Meaning |
|---|---|
| `source` | `ReadSource`: name, token `rows`, `assignment` (`supplied` / `nominated` / `learned`), `provenance`, `binding_fingerprint` |
| `reader` | `Reader`: `block`, `head`, `step` (or `steps` when aggregated), optional `kv_group` (from `attention_port.kv_group`), `block_label`, `receivers`, `receiver_row` |
| `mass`, `mass_kind` | Observed mass and what it measures (attention probability, write mass, ...) |
| `surface`, `backend` | `project-session` or `passive-observation`; backend name |
| `binding_fingerprint`, `use_id`, `use_fingerprint`, `lifecycle_fingerprint` | Anchors into the canonical `SymbolBinding` / `SymbolUse` / `SymbolLifecycle` |
| `sets` | IDs of `ReadSet`s this reader belongs to (membership only) |
| `causal` | `CausalStatus`; `unknown` unless a per-read intervention result was attached |
| `evidence`, `metadata` | Pointers and small scalars |

A `ReadSet` is a set of readers of one source: `name`, `source`, `members` (pointer + count + order-free digest), `selection` (how the set was chosen), `causal` (set-level results) and `member_reads` (fingerprints of member reads that exist as records). `set_id` depends on source, name, membership and selection, not on results.

`save_reads(path, reads, sets, meta=...)` writes one sealed JSON document to a new file. It checks that every set's `member_reads` and every read's `sets` resolve within the file. `load_reads(path)` verifies every seal and rebuilds the objects.

## Causal status

`CausalStatus()` with no results has `status == "unknown"` and the label `unknown (observed only; no intervention)`. The view prints that label on every read that carries it. An observed read is therefore never shown with a blank causal status.

A `CausalResult` records one intervention:

| Field | Values |
|---|---|
| `intervention` | `cut`: the reads are removed. `keep`: only these reads remain |
| `cut_type` | `renormalising`: boolean mask, so the remaining attention renormalises. `no-renormalisation`: the contribution is removed and other weights are left untouched |
| `effect`, `effect_units`, `effect_per_seed` | The arm's score, or `None` when the score cannot be read (state the reason) |
| `floor`, `floor_source` | Size of the no-op or sham arm in the same units, and where it came from (in-run or cross-run) |
| `control` | The matched control arm and its effect. `None` if no control was run |
| `verdict` | See the next table |
| `plan_sha256`, `run_id` | Required: sha256 of the plan file and the job id |
| `claim`, `reason`, `evidence`, `metadata` | What was claimed, why this verdict, and pointers |

| Verdict | Meaning | Constraint enforced |
|---|---|---|
| `supported` | Effect present, and the matched control does not show it | A control and a readable effect are required |
| `not_supported` | No effect, or the control shows the same effect | none |
| `unresolved` | Seeds disagree, the effect is partial, or the score cannot be read | none |
| `control_missing` | Effect present, but no matched control was run | Must not carry a control |

Set-level results go on the `ReadSet` (`with_result`). Nothing copies them onto member reads. A set result says that the set as a whole was cut or kept; it does not show that any member head is necessary on its own. `SymbolRead.with_causal` exists for genuine per-read interventions, such as one head cut by itself.

The canonical `SymbolUse` and `SymbolLifecycle` keep their fixed `authority.causal_effect = "not-established"`. That field describes what the record itself grants. Intervention evidence lives in the read and read-set records described here.

## What `supplied`, `nominated` and `learned` mean

These values say how a source's role or name was assigned.

- `supplied`: the experimenter named the source, for example a test word and its tokenizer rows. Provenance must say who supplied it (`by`).
- `nominated`: a ranking chose the source. Provenance must include the `ranking`, the `score`, and an `evaluation` record that passes `attention_ranking.require_trivial_baselines`, meaning position 0, previous token and uniform were computed on the same rows. A nomination without baselines is refused.
- `learned`: a fitted model proposed the source, as with native discovery's sparse predictor and component catalog. Provenance must name the `predictor`.

Reader sets have their own `selection` record. In the FLUX conversion, the own-word head sets are nominated by a label-free contrast ranking; the source word itself is still `supplied`.

## Pointer rule and bounds

Records follow the replay direction in [the replayable-debugger plan](REPLAYABLE-CIRCUIT-DEBUGGER-PLAN.md): they reference data and never carry it.

- `file_pointer(path, array=, selector=, json_pointer=)` records path + sha256 (+ array name and selector).
- `parameter_pointer(model_identity, parameter, selector=, checkpoint_sha256=)` records a model parameter by identity.
- `run_pointer(run_id, plan_sha256=)` records a run.

`check_bounded` is applied to every read, read set, result, passive use and passive definition. It refuses tensor-like values (anything with `shape`, `ndim`, `detach` or `__array__`), raw bytes, non-finite numbers, any sequence longer than 64 items (`MAX_INLINE_ITEMS`), text over 4 KiB and records over 64 KiB. Large memberships such as a 240-head set are stored as pointer + count + digest.

## Native discovery compatibility

`native_discovery_symbols.register_observations` now builds its binding, definition, use and lifecycle through the passive writer: `PassiveSurface.from_cut`, `passive_binding`, `passive_definition`, `passive_use` and `passive_lifecycle`. The cuts, receipts, symbol tables and JSON files are unchanged.

- Measured: re-running the new function on saved inputs of `artifacts/mdb-native-discovery-20261112/{consumer,main,smoke-02}` gives byte-identical symbol directories. That is 4 + 4 + 1 documents with 44, 44 and 40 files each, including the `LocalStore` cuts and receipts. The same check passes with the pre-change code. smoke-02's saved report predates the `inactive_feature_abstentions` count key; its files are still identical.
- `tests/test_symbol_reads.py` keeps a frozen copy of the pre-consolidation function (b7318c7; unchanged since be8c719) as an oracle and compares bytes on synthetic inputs.
- `reads=True` (off by default) also writes `reads.json`. Sources there are `learned`, with the sealed predicted source mass as score. Default runs therefore keep the exact pre-consolidation file set. Saved-run verification (`verify_run`) reads only the manifest's file hashes, and no saved file changes.

## FLUX read-graph conversion

```sh
PYTHONPATH=packages/mdb/src:src python packages/mdb/qualification/flux_read_graph_symbols.py \
    --experiment saturn/experiments/2026-11-11-flux-read-graph --out NEW_DIR
```

The converter runs no model. It reads saved files only; captures and tables are referenced by path + sha256.

The output contains:

- `lifecycles.json`: one canonical lifecycle for each of the six test words (cat: red, black, ball, snow; car: night, rain). Each holds a `supplied` binding over the word's text rows from `tokens.json` and one observed use per head in the word's `own60` set: 360 uses. Each use carries image←row write mass (`tok_w` summed over 4 steps and the word's rows) for seeds 4242 and 9001, the word share, the contrast rank and score, and per-step mass.
- `reads.json`: 360 `SymbolRead`s, all with causal status `unknown`, and 114 `ReadSet`s. The sets are 54 headfree sets from `head_sets3.json` and 60 headx sets from `runs/headx/plan.json`; empty sets are skipped. 175 set-level results are attached.
- `summary.json` and `symbol-reads.html`.

Checks recorded in `summary.json` (measured on this machine): own60/120/240 recomputed with `attention_ranking.contrast_ranking` match `head_sets3.json` for all six words, and the headfree plan's sets equal `head_sets3.json`.

Verdicts come from a mechanical rule over the audited picture grids in `AUDIT.md`, where the picture decides:

1. Each seed's picture call maps to a level: Y = 0, Y− = 1, P = 2, N = 3.
2. A cut has an effect if both seeds are at 1 or higher. A keep carries the word if both seeds are at 1 or lower.
3. The effect is then compared with a matched control:
   - headfree: own120 vs rand120, own240 vs rand240, mod vs noun, modI vs nounI.
   - headx: each category set vs heavy60.
   - own60 cut/keep and headx keep arms: the cross-run Addendum 4 random 60-head arms (`rand60a`, `keep-rand60a`, pixel mean). These are marked `in_run: false`.
4. snow and rain are always `unresolved` and carry no score or floor numbers, because the pixel score is unusable for them (AUDIT L3, X2).
5. The headx `<category>X` arms have no picture call. They take the audit's X1 statement and are `not_supported`.

Measured result counts:

| | supported | not_supported | unresolved | control_missing |
|---|---|---|---|---|
| cut | 26 | 51 | 40 | 3 |
| keep | 12 | 18 | 18 | 7 |

Of the 58 unresolved results, 47 are snow or rain and 11 have mixed pictures or controls. Examples are own240 on night and ball (AUDIT L1) and keep-nounI on ball. The `control_missing` results are keep-own120, keep-own240 and the renormalising `m:own60` arm.

What the conversion does not claim:

- Two prompts, two seeds, and about three independent cut contrasts: the colour words (red and black share 29 of 60 own heads and the same pools), ball, and night. Group sets are one per prompt and are shared by its words.
- Results are set-level only. No head is shown to be individually necessary, and the 60 member reads stay `unknown`.
- The cross-run random-60 controls come from another job with the same worker family and use the pixel mean only.
- Write mass omits the adaLN gate (FINDINGS). It is a read observation, not a measure of what is written.
- Hand-assigned categories (headx) are recorded as selection metadata, not as findings about categories.
- No checkpoint hash for FLUX.2 Klein was available. Model identity is the plan's model name and path; no parameter pointers were added because parameter names were not verified offline.

## View

`symbol_attention_view.write_view(directory)` keeps the Qwen join view for directories with `linked-symbols.json`. It adds one causal-status line, `unknown (observed only; no intervention)`, to the selection panel. Measured: re-rendering `artifacts/mdb-attention-symbols-20261111/join-02` gives HTML identical to the saved file apart from that line.

For a directory that has `reads.json` and no `linked-symbols.json`, `write_view` dispatches to `write_reads_view`. That view is an offline table with no external assets: read sets with arm, verdict, effect, picture calls, control, floor and run, followed by observed reads with their causal label, filterable by source. The Qwen view was not reused for reads because it is hard-wired to 28 layers × 12 heads, a fixed gold source row and K/V-origin lineage. FLUX has 25 blocks × 24 heads, aggregated steps and set-level results.
