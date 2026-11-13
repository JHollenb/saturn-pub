# Symbol reads: one writer for "this head read this source with this much mass"

Status: November 12, 2026 (schema v2, after audit 8). Module: [`mdb.symbol_reads`](../src/mdb/symbol_reads.py). Tests: [`tests/test_symbol_reads.py`](../tests/test_symbol_reads.py). FLUX converter: [`qualification/flux_read_graph_symbols.py`](../qualification/flux_read_graph_symbols.py).

Three sources record the same kind of fact: a head read a source with some mass. Before this module they did so in three ways. Native discovery built passive records by hand. `symbol_attention` built Project-backed Qwen records. The FLUX.2 Klein read-graph experiment had no records at all. All three now produce the same record through one writer.

## Two observation surfaces, one record

| Surface | Entry point | Canonical records | When to use |
|---|---|---|---|
| Project-backed | `project_reads(project, session_id, lifecycle, source=..., **attach_attention_kwargs)` or `reads_from_lifecycle(lifecycle, source=...)` | `symbol_attention.attach_attention`, unchanged | A retained native Session with cuts, receipts and a continue receipt |
| Passive | `PassiveSurface` + `passive_binding` / `passive_definition` / `passive_use` / `passive_lifecycle`, then `read_for_use` | Built by the writer | A non-resumable observation: a passive `StateCut` (native discovery), or a saved capture with no cut (FLUX) |

Both surfaces end in `symbol_read(...)`, the single constructor. It returns a sealed `SymbolRead` (schema `mdb-symbol-read-v2`):

| Field | Meaning |
|---|---|
| `source` | `ReadSource`: name, token `rows`, `assignment` (`supplied` / `nominated` / `learned`), `provenance`, `binding_fingerprint` |
| `reader` | `Reader`: `block`, `head`, `step` (or `steps` when aggregated), optional `kv_group` (from `attention_port.kv_group`), `block_label`, `receivers`, `receiver_row` |
| `mass`, `mass_kind` | Observed mass and what it measures (attention probability, write mass, ...). `mass_kind` is required |
| `surface`, `backend` | `project-session` or `passive-observation`; backend name |
| `binding_fingerprint`, `use_id`, `use_fingerprint`, `lifecycle_fingerprint` | Anchors into the canonical `SymbolBinding` / `SymbolUse` / `SymbolLifecycle` |
| `sets` | IDs of `ReadSet`s this reader belongs to (membership only) |
| `causal` | `CausalStatus`; `unknown` unless a read-scope intervention result was attached |
| `evidence`, `metadata` | Pointers and small scalars |
| `model` | Optional `model_identity(...)` record: name plus `identity_status` (see Pointers) |

A `ReadSet` (schema `mdb-symbol-read-set-v2`) is a set of readers of one source: `name`, `source`, `members` (pointer + count + order-free digest), `selection` (how the set was chosen), `causal` (set-scope results) and `member_reads` (fingerprints of member reads that exist as records). `set_id` depends on source, name, membership and selection, not on results.

`selection.kind` must be one of `ranked`, `random`, `group`, `category`, `category-unique`, `control`, `supplied`. It is deliberately disjoint from the source-role vocabulary: `nominated` is refused on a set (it means a source role chosen by a ranking that carries the trivial baselines), so a top-k set is `ranked`.

`save_reads(path, reads, sets, meta=...)` writes one sealed JSON document (schema `mdb-symbol-reads-v2`) to a new file. It checks that every set's `member_reads` and every read's `sets` resolve within the file, that `meta` is within its budget, and, when `meta.verdict_rule` is present, that its sha256 matches its rule and every result names that rule id and sha256. `load_reads(path)` verifies every seal and the same rule check. v1 documents are refused (the record shape changed).

## Causal status

`CausalStatus()` with no results has `status == "unknown"` and the label `unknown (observed only; no intervention)`. The view prints that label on every read that carries it.

A `CausalResult` records one intervention. Its fields are typed and validated on construction:

| Field | Values and checks |
|---|---|
| `intervention` | `cut` (the reads are removed) or `keep` (only these reads remain) |
| `cut_type` | `renormalising` (boolean mask; remaining attention renormalises) or `no-renormalisation` (contribution removed, other weights untouched) |
| `scope` | Required. `read` (one read intervened alone) or `set` (a whole `ReadSet`). `SymbolRead` accepts only `read`; `ReadSet` only `set`; `SymbolRead.with_causal` refuses a set result and `ReadSet.with_result` a read result |
| `basis` | Required. What decided the verdict: `score`, `picture` (audited picture calls decide, scores attached as context), `picture_only` (picture calls decide, score omitted as unusable), `audit_statement` (an audit sentence, no per-arm call), `none` |
| `effect`, `effect_per_seed`, `effect_units` | Finite numbers or `None`; strings and booleans are refused |
| `floor`, `floor_source`, `floor_incomplete` | `floor` is a number >= 0 and needs a `floor_source`. `floor_incomplete` is a bool (`True` when the floor is borrowed from another run or omitted) |
| `control` | The ONE control the verdict uses, as a structured record: `arm`, `run_id` (must equal the result's), `in_run` (must be `true`), `matched_on` (non-empty list), `judged_by` (`picture`, `score`, `picture+score`), `judgement` (`clean`, `same`, `mixed`); optional `effect`, `size`, `set_size`, `mass_held`, `set_mass_held`, `rules_out` |
| `reference_controls` | Other arms, recorded but not counted: other runs (`in_run: false`, `verdict_effect: "none"`), unmatched in-run arms (`verdict_effect: "downgrade_only"`), other-item controls. Same required keys as `control` plus `verdict_effect` |
| `picture` | `{"calls": [...], "levels": [...]}`; required for basis `picture` / `picture_only` |
| `claim` | `necessary` for cut, `sufficient` for keep (vocabulary shared with `attention_panels`); derived and checked |
| `claim_text` | The claim in words |
| `ruled_out`, `remaining_alternatives` | Partition of the claim's alternatives: necessary `generic_damage`, `mass`, `set_size`; sufficient `mass`, `set_size`. Each ruled-out item must appear in the `rules_out` of a clean in-run record. `set_size` needs `size >= 0.9 × set_size`; `mass` needs `matched_on` to include `mass` or `mass_held >= set_mass_held` on every seed |
| `claim_scope` | Only on `supported`: `needed_set` (necessary) or `sufficient_set` (sufficient) while any alternative remains; `identity` only when none remains |
| `contrast_id` | Required on set-scope results: the independent contrast the result belongs to |
| `rule_id`, `rule_sha256` | The verdict rule that produced the verdict (both or neither) |
| `plan_sha256`, `plan`, `run_id` | sha256 of the plan file; optional plan file pointer whose digest must equal it; non-empty run id without surrounding whitespace. `result.verify_plan(path)` checks a plan file against the hash |
| `reason`, `evidence`, `metadata` | Why this verdict, pointers, small extras (for example an audit caveat) |

| Verdict | Meaning | Constraint enforced |
|---|---|---|
| `supported` | Effect present and the counted control does not show it | A counted in-run control judged `clean`; basis `score`/`picture`/`picture_only`; unless `picture_only`, numeric `effect`, numeric `floor`, a `floor_incomplete` bool, and cut `effect > floor` / keep `control.effect - effect > floor`; `claim_scope` set |
| `not_supported` | No effect, or the counted control shows the same | `claim_scope` empty |
| `unresolved` | Seeds disagree, the effect is partial, a control partly shows it, or nothing decides | `claim_scope` empty |
| `control_missing` | Effect present, no counted control | No `control` (unmatched arms go under `reference_controls`) |

`supported` here keeps the four-status scheme of the audited converter. `attention_panels` calls a result that holds only against some alternatives `set_only`; here that is `supported` with `claim_scope: needed_set` and a non-empty `remaining_alternatives`.

Set-level results go on the `ReadSet`. A set result says that the set as a whole was cut or kept; it does not show that any member head is necessary on its own.

The canonical `SymbolUse` and `SymbolLifecycle` keep their fixed `authority.causal_effect = "not-established"`. That field describes what the record itself grants. Intervention evidence lives in the read and read-set records described here.

## What `supplied`, `nominated` and `learned` mean

These values say how a source's role or name was assigned.

- `supplied`: the experimenter named the source, for example a test word and its tokenizer rows. Provenance must say who supplied it (`by`).
- `nominated`: a ranking chose the source. Provenance must include the `ranking`, the `score`, and an `evaluation` record that passes `attention_ranking.require_trivial_baselines`, meaning position 0, previous token and uniform were computed on the same rows. A nomination without baselines is refused.
- `learned`: a fitted model proposed the source, as with native discovery's sparse predictor and component catalog. Provenance must name the `predictor`.

Reader sets have their own `selection.kind` (above). In the FLUX conversion the own-word head sets are `ranked` by a label-free contrast ranking; the source word itself is still `supplied`.

## Pointers, identity and bounds

Records follow the "Reference and storage contract" of [the replayable-debugger plan](REPLAYABLE-CIRCUIT-DEBUGGER-PLAN.md): a locator says where, a digest says what, and moving a file must not change the reference.

- `declare_root(name, path)` names a storage root. A document records its roots once (FLUX: `meta.roots`).
- `file_pointer(path, root=..., array=, selector=, json_pointer=, sha256=, role=)` gives `{"kind": "file", "locator": {"root", "path" (relative)}, "digest": {"sha256", "size"}}`. A file outside the root is refused. `resolve_pointer(pointer, roots)` and `verify_pointer(pointer, roots)` (size, then sha256) turn it back into a checked path.
- `model_identity(name, checkpoint_sha256=..., resolve_by=...)`: with a checkpoint hash `identity_status` is `verified`; without one it is `unresolved` and `resolve_by` must say what would resolve it.
- `parameter_pointer(model, parameter, selector=, checkpoint_sha256=, resolve_by=)` carries such a `model` record.
- `run_pointer(run_id, plan_sha256=, plan=)` records a run and optionally its plan's file pointer.
- `verdict_rule(rule_id, rule)` returns `{"id", "sha256", "rule"}` for `meta.verdict_rule`.

`native_discovery_symbols` (reads=True only) still builds its evidence pointer by hand in the old path form; that file was outside this change.

What `check_bounded` enforces, recursively, on every read, read set, causal result, control record, passive use (signed observations + metadata + evidence refs), passive definition (metadata + evidence refs), lifecycle metadata and document `meta`:

- no tensor-like value (anything with `shape`, `ndim`, `detach` or `__array__`, including 0-d numpy arrays) and no raw bytes;
- numpy scalars (`np.bool_`, `np.int64`, `np.float32`, ...) are converted to Python `bool` / `int` / `float`;
- no non-finite number; text at most 4 KiB (`MAX_TEXT_BYTES`); mappings at most 256 keys with text keys;
- a sequence holds at most 64 items (`MAX_INLINE_ITEMS`), counting every item of the sequences nested directly inside it, so a 48 × 48 nested list (2352 items) is refused; a mapping inside a sequence starts its own count;
- at most 4096 values in total per checked value (`MAX_RECORD_ITEMS`);
- at most 64 KiB of canonical JSON per checked value (`MAX_RECORD_BYTES`; `MAX_META_BYTES` = 64 KiB for `meta`);
- nesting depth at most 16.

Large memberships such as a 240-head set are stored as pointer + count + digest.

## Native discovery compatibility

`native_discovery_symbols.register_observations` builds its binding, definition, use and lifecycle through the passive writer: `PassiveSurface.from_cut`, `passive_binding`, `passive_definition`, `passive_use` and `passive_lifecycle`. The cuts, receipts, symbol tables and JSON files are unchanged.

- Measured (after the schema v2 change): re-running the function on the saved inputs of `artifacts/mdb-native-discovery-20261112/smoke-02` gives 40 of 40 files byte-identical to the saved symbol directory, and the frozen pre-consolidation copy gives the same. Earlier (before v2) the same check also passed on `consumer` and `main`; those two were not re-run after this change.
- `tests/test_symbol_reads.py` keeps a frozen copy of the pre-consolidation function (b7318c7; unchanged since be8c719) as an oracle and compares bytes on synthetic inputs.
- `reads=True` (off by default) also writes `reads.json`. Default runs keep the exact pre-consolidation file set.

## FLUX read-graph conversion

```sh
PYTHONPATH=packages/mdb/src:src python packages/mdb/qualification/flux_read_graph_symbols.py \
    --experiment saturn/experiments/2026-11-11-flux-read-graph --out NEW_DIR
```

`--out` is required and must not exist. The converter runs no model. It reads saved files only; files are referenced by locator (root `flux-read-graph`) and digest.

The output contains:

- `lifecycles.json`: one canonical lifecycle for each of the six test words (cat: red, black, ball, snow; car: night, rain). Each holds a `supplied` binding over the word's text rows from `tokens.json` and one observed use per head in the word's `own60` set: 360 uses. Each use carries image←row write mass (`tok_w` summed over 4 steps and the word's rows) for seeds 4242 and 9001, the word share, the contrast rank and score, and per-step mass. Lifecycle metadata carries the model identity record.
- `reads.json`: 360 `SymbolRead`s, all with causal status `unknown`, and 114 `ReadSet`s (54 headfree sets from `head_sets3.json`, 60 headx sets from `runs/headx/plan.json`; empty sets skipped) with 175 set-scope results. `meta` holds `roots`, `inputs`, `jobs`, `checks`, `model`, `scope` and `verdict_rule`.
- `summary.json` (counts per status, per contrast × status, per intervention × status) and `symbol-reads.html`.

Checks recorded in `summary.json` (measured on this machine): own60/120/240 recomputed with `attention_ranking.contrast_ranking` match `head_sets3.json` for all six words, and the headfree plan's sets equal `head_sets3.json`.

### Verdict rule

The rule is written once, in the converter's `RULE`, and stored as `meta.verdict_rule` with id `flux-read-graph-picture-rule-v2` and its sha256. Every result names both. In short:

1. Each seed's picture call (AUDIT.md picture grids) maps to a level: Y = 0, Y− = 1, P = 2, N = 3.
2. Set state. Cut: present if every seed ≥ 1, absent if every seed is 0. Keep: carried if every seed ≤ 1, lost if every seed ≥ 2. Otherwise mixed → `unresolved`. Absent → `not_supported`.
3. No picture call: a headx `<cat>X` cut (not snow) takes the AUDIT X1 sentence → `not_supported` with basis `audit_statement` (16 results); otherwise `unresolved`, basis `none`.
4. Present, with a counted control: the control is judged by the same thresholds (symmetric: a Y− control is a weak effect). Control absent → `supported`; control present and at least as strong on every seed → `not_supported`; otherwise `unresolved`.
5. Present, no counted control: if an in-run screen shows the effect on any seed → `unresolved`; else `control_missing`.
6. Other-job references are recorded with a pixel judgement (|gone| ≤ .15 for cut, gone ≥ .7 for keep) and never change a verdict.
7. snow and rain: the pixel score is unusable (AUDIT L3, X2); picture calls decide (basis `picture_only`) and no effect, floor or control numbers are recorded.

Which arm controls which:

| Claim | Counted (in-run, matched) | Screen (in-run, downgrade only) | Reference (other job, never counted) |
|---|---|---|---|
| headfree cut own120 / own240 | rand120 / rand240 (joint/single counts; not mass-matched) | | |
| headfree cut mod ↔ noun, modI ↔ nounI | the other pool (same ranking procedure) | | |
| headx cut colour/object/scene/light/attribute | heavy60 (joint/single counts; holds less mass, AUDIT H2) | | |
| headfree cut own60 | none | rand120, rand240 | Addendum 4 rand60a |
| headfree keep own60 | none | | Addendum 4 keep-rand60a |
| headfree keep modI ↔ nounI | none | the other pool's keep | |
| headx keep categories | none | | Addendum 4 keep-rand60a |
| headfree keep own120 / own240, m:own60 | none | | |

Alternatives are filled in only for effects that are present. A clean counted control rules out `set_size` when its size is at least 0.9 × the set's, and `mass` only when it holds at least the set's word mass on every seed. An other-item control (the same units cut on another informative word's rows in the same run, picture Y/Y; snow and rain cannot witness) rules out `generic_damage`.

Headfree floors are borrowed from the headx `z:sham-iso` arm of the same word (another job; headfree ran no sham, AUDIT L5): `floor_source` starts with `borrowed:` and `floor_incomplete` is true. own120/own240 results carry the AUDIT L1 caveat ("not attributable to the word's own heads") in `metadata.caveat`.

### Measured result counts

Status × contrast (175 set results):

| contrast | supported | not_supported | unresolved | control_missing |
|---|---|---|---|---|
| colour-words (red, black) | 13 | 33 | 2 | 16 |
| ball | 5 | 16 | 5 | 6 |
| night | 1 | 20 | 9 | 2 |
| snow | 0 | 19 | 7 | 6 |
| rain | 0 | 12 | 0 | 3 |
| all | 19 | 100 | 23 | 33 |

By intervention: cut 19 supported / 79 not_supported / 17 unresolved / 5 control_missing; keep 0 / 21 / 6 / 28.

The 19 `supported` results are all cuts with an in-run picture control: red mod, modI, own120, own240, attribute, colour, light; black mod, modI (weak: Y−/Y−), own120, own240, attribute, colour; ball noun, nounI, own120, object, scene; night mod. All carry `claim_scope: needed_set`. `mass` is ruled out on one: night `mod` against `noun`, which holds more of night's mass (.55/.56 vs .48/.47 by seed) and leaves night intact. `generic_damage` stays open there because the car prompt has no informative other word. No result reaches `identity`.

Differences from the audit-8 corrected table (21 / 100 / 20 / 34), all from applying the rule as written:

- night headx `color` and `light` cuts: `unresolved`, not `supported`. Their counted control heavy60 is Y− on seed 4242, which is the same pattern that makes night own120 vs rand120 `unresolved` under the symmetric rule.
- ball own60 cut: `unresolved`, not `control_missing`. Its screen rand240 shrinks the ball on seed 9001 (Y− 66%, AUDIT L1), the same kind of evidence that makes night own60 `unresolved`.

What the conversion does not claim:

- Two prompts, two seeds (seed-9001 mass is within .06 of 4242, so it is a held-out image seed, not a held-out selection), and about three independent cut contrasts: the colour words (red and black share 29 of 60 own heads and the same pools), ball, and night. Counts of results are not counts of independent evidence; every result names its `contrast_id`.
- Results are set-level only. No head is shown to be individually necessary, and the 60 member reads stay `unknown`.
- No keep result is supported: no in-run random keep arm was run.
- Write mass omits the adaLN gate (FINDINGS). It is a read observation, not a measure of what is written.
- Hand-assigned categories (headx) are recorded as selection metadata, not as findings about categories.
- No checkpoint hash for FLUX.2 Klein was available: `model.identity_status` is `unresolved`, with what would resolve it. No parameter pointers were added because parameter names were not verified offline.

## View

`symbol_attention_view.write_view(directory)` keeps the Qwen join view for directories with `linked-symbols.json`. Its selection panel shows a causal-status line taken from each row's data (`causal_label`: the row's recorded `causal` status, else the `CausalStatus` default). Measured: re-rendering `artifacts/mdb-attention-symbols-20261111/join-02` gives HTML identical to the saved file apart from that line and the added `causal_label` data field; every row's label is `unknown (observed only; no intervention)`.

For a directory that has `reads.json` and no `linked-symbols.json`, `write_view` dispatches to `write_reads_view`, an offline page with no external assets. In order:

- the scope block (prompts, seeds, independent contrasts and which sources belong to each);
- a status × contrast count table;
- the model identity status and the verdict rule (id, sha256, full rule);
- read sets grouped by contrast. Each row shows arm, claim, verdict, basis, picture calls, effect, the counted control (in-run or other job, size-matched only or mass-matched, set and control sizes, word mass held by the set and by the control per seed, picture, judgement), reference controls (marked not counted, or screen), the floor with its source (`borrowed:` when from another run), claim scope with ruled-out and remaining alternatives, any audit caveat, the reason and the run;
- observed reads with mass, mass kind and causal label, filterable by source.
