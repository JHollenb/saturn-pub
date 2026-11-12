# Attention panels: one generator for read-cut and value-replacement experiments

`mdb.attention_panels` builds the arm list of an attention experiment from a named unit set, a universe of units and a few options. It adds every mandatory control, counts the branches, splits the panel into leases, seals the plan by hash, and turns per-arm scalar outcomes into a verdict for each unit set. It replaces the hand-written arm lists in the FLUX read-graph worker (`headcut` block) and the `{arm, kind, cells, dose}` schemas in `value_coalition.plan`, `frozen_value_pair.arms` and `attention_source_resolver.arms`, together with the kind → (side, payload side, target row, payload row) mapping that three qualification workers each copied.

The module is framework-free. Importing it loads neither torch nor numpy (a test checks this). numpy is imported only to draw random controls, and `attention_port` only when an arm is mapped to port cuts.

## The arm record

`Arm` is one frozen, JSON-serialisable record:

| field | meaning |
|---|---|
| `arm` | id, e.g. `z:own60:red`; `z` = no-renormalisation cut, `m` = renormalising cut |
| `family`, `kind` | `read` or `value`; the kind (below) |
| `role` | `test`, `control`, `floor` or `reference` |
| `item`, `rows_item` | the item the result is about; the item whose rows are cut (they differ only for an other-item control) |
| `units`, `unit_set` | the (block, head) pairs or (layer, kv group) cells; `None` = every unit. `unit_set` names the set (`keep-own60` = complement of `own60`) |
| `receivers`, `sources` | row names the port resolves (default `("image", "others:<item>")` ← `("own:<item>",)`) |
| `cut_type`, `zero_method` | `zero` (`renormalize=False`) or `renorm` (`renormalize=True`); the per-cut zero method (`None` = the port's own, recorded in the plan's `execution` block) |
| `condition` | input variant the forward runs under (`iso:red`, `delete:red`, `None`); the port does not see it |
| `reference` | arm whose output the outcome is measured against |
| `control_for`, `control` | test arms this arm controls; `random` / `other_item` / `external` |
| `separates_identity` | True only for a random control that holds mass fixed (`mass_relation == "matched"`) |
| `mass_relation` | a random control's mass against the target's: `matched`, `matched_by_reuse`, `less`, `more`, `unknown` |
| `rules_out` | the alternative explanations this control rules out when the test beats it beyond the floor (`set_size`, `mass`, `generic_damage`) |
| `floor_of`, `twin`, `claim` | the cut whose floor this arm measures; the same arm in the other cut type; `necessary` (cut) or `sufficient` (keep) |
| `dose`, `target_row`, `payload_row` | value family |
| `bundle` | lease atom: a test set, its twins, floors and controls share a bundle |

`rules_out` and `mass_relation` are omitted from the JSON when they hold their defaults, and the plan's `execution` block is omitted when empty, so a plan written before these fields existed reloads with the same sha256 (tested). For such a plan, `verdicts` fills them in: other-item arms (then marked `separates_identity=True`) rule out `generic_damage`, and random arms are profiled from their stored draw report.

Kinds:

- **Read cuts.** `cut` cuts the set. `keep` cuts the set's complement over the universe. Each is emitted in both cut types and linked by `twin`.
- **Read floors.** `floor_sham_recompute` and `floor_cut_recompute` are emitted for every no-renormalisation cut. `floor_sham` is emitted for every renormalising cut.
- **Read references.** `native` is the unmodified input. One shared `native` arm is emitted whenever an arm is scored against it (the `iso:<word>` and `del:<word>` arms always are), so it is counted in `branch_count` and runs in a lease. `condition` is the conditioned input with no cut (`iso:<word>`). `all_cut` cuts in every head (`z:none`, `m:none`). `deletion` runs with the word deleted from the prompt. `sham_all` routes every head and row through the FP32 recompute path and removes nothing (the old `z:sham-iso`).
- **Value (Codex family).** The test kinds are `singleton` (one cell from the donor), `complement` (every cell but one), `coalition` and `dose` (fractional). The control kinds are `wrong_target`, `wrong_payload`, `reciprocal`, `sham` and `fixed_distance`. The reference kinds are `native`, `replay` and `donor_native`.

## The generators

```python
from mdb.attention_panels import ReadPanelSpec, RandomControls, read_panel, value_panel, sets_from_ranking

panel = read_panel(ReadPanelSpec(
    items=("red", "black", "ball", "snow"),             # one prompt's test words
    universe=[(b, h) for b in range(25) for h in range(24)],
    sets={w: {"own60": ..., "own120": ..., "own240": ...} for w in words},   # or sets_from_ranking(contrast_ranking(...), (60, 120, 240))
    contexts=("cat|4242", "cat|9001"),                   # every arm runs in every (prompt, seed)
    cut_sets=("own60", "own120", "own240"), keep_sets=("own60",),
    nested=("own60", "own120", "own240"),                # adds own120-minus-own60, own240-minus-own60
    random=RandomControls(n=1, strata=joint_or_single, mass=word_mass, mass_tolerance=0.02, disjoint=True),
    precision="native", row_table=item_rows(word_rows, text_rows, total_rows), forwards_per_image=STEPS,
))
```

For every test set, and for every item, the read generator emits:

1. The test arm in **both cut types** (`z:<set>:<item>`, `m:<set>:<item>`), linked by `twin`.
2. Its **floor arms**. Each no-renormalisation cut gets `/sham_recompute` (the recompute sham of exactly its cells) and `/cut_recompute` (the recompute twin). Each renormalising cut gets `/sham`. These are built with `attention_port.floor_arms`, so the panel and the port agree on what a floor is. Every cut gets them, controls and references included, and they run in every context: every (prompt, seed), not only the first.
3. **Keep arms** for `keep_sets`: `z:keep-<set>:<item>` cuts in every unit outside the set.
4. **Matched random controls for both cut and keep.** `attention_ranking.matched_control_sets` matches each draw on per-stratum counts and on the number of distinct blocks. Mass matching and disjointness are optional. The random keep arm is the complement of the matched random set. Each draw's report is stored in `panel.controls["<item>:<set>"]`, with its `mass_relation`. A mass-matched draw that is not disjoint and overlaps the target by more than the chance mean + `overlap_sd` sd (default 3, a `RandomControls` parameter recorded in the plan) is `matched_by_reuse`: it shares the identity it should vary, so it does not rule out mass. `panel.identity["<item>:<set>"]` says whether the draws separate identity from mass.
5. An **other-item control**: the same units applied to another item's rows (`z:<set>@rows-<other>:<item>`). By default the other item is the next item, cycling. For a cut claim it rules out `generic_damage` ("cutting these units damages any word"). It does **not** hold mass fixed (the other item's mass through these units differs), so it does not separate identity from mass. For a keep claim it is a reported reference only.
6. **Nested-size arms**: `<big>-minus-<small>`, the added units alone. They are treated as test sets with the same controls.
7. **References** (`native`, `iso:<item>`, `del:<item>`, `z:none:<item>`, `m:none:<item>`) with floors.
8. `external_controls={"rand120": "own120"}` for hand-drawn control sets. They inherit the target set's cut types, have `mass_relation="unknown"` and rule nothing out: they can refute a claim but never support one.

**Refusals.** Each of the following raises unless `waive={key: "written reason"}` is passed: fewer than both cut types on a test set (`both_cut_types`), `floors=False` (`floors`), no random control (`random_cut`, `random_keep`), an item with no other item (`other_item`), and keep sets over a universe that is not a full (block, head) grid (`partial_universe`). A waiver is recorded in the plan and in every verdict. It never makes a result count: a missing cut type still gives `control_missing`. These raise with no waiver possible: an empty set, a set listing a unit twice, a test or control set equal to the whole universe, `RandomControls(n=0)` (use `random=None` plus the waiver), `other_items` mapping an item to itself or to a non-item, a `condition` without `"condition"` in `references` (the outcome would include the conditioning change), and keep sets without `"all_cut"` in `references`. If a random draw is structurally impossible (for example, too few disjoint units), the generator does not raise. It records `status: control_missing` with the failing constraint.

`value_panel(cells, groups=..., doses=(0.5,), controls=..., references=..., comparators=..., fixed_distance=..., waive=None, allow_extrapolation=False)` emits the Codex family. The `sham` control and the `native` reference are mandatory unless `waive={"sham" | "native_reference": "written reason"}`. Doses (and `reference_dose`) must lie in [0, 1] unless `allow_extrapolation=True`. The references form one lease atom (`ref:<label>`); the coalition test, its doses and every control of it (sham, wrong payload, wrong target, reciprocal, fixed distance) form another (`coalition:<label>`), so a sham never lands in a different lease from its coalition. Arms with identical executable content (legacy kind, cell set, dose) are emitted once, and the dropped names go to `panel.aliases`; `resolve_aliases(panel, outcomes)` maps results named by an alias onto the emitted arm. With two cells, both complements equal singletons. `replay` is exempt from this deduplication because it repeats `native` on purpose.

## Execution settings (hashed)

A read plan carries an `execution` block that is part of its sha256:

| key | meaning |
|---|---|
| `precision` | `native` or `legacy` (`ReadPanelSpec.precision`, else a valid `port["precision"]`, else `native`; the two may not contradict) |
| `zero_method` | per cut type: `zero` = the per-arm zero method or the port's default (`port_zero_method`, default `delta`); `renorm` = none; floors always recompute |
| `empty` | the port's empty-row policy |
| `row_table` | sha256, names and sizes of the row table (`ReadPanelSpec.row_table`), or `None` |
| `forwards_per_image` | or `None` |
| `universe` | `blocks`, `heads`, `n_units`, `full_grid`, `units_sha256`, strata counts and digest |
| `unbound` | which of `row_table` / `forwards_per_image` the plan does not bind |

- `port_kwargs(panel, rows=None)` returns the `AttentionPort(backend, **kwargs)` arguments the plan implies (`rows`, `zero_method`, `precision`, `empty`). If the plan binds a row table, `rows` must be given and must match its digest.
- `check_port(panel, port, forwards=None)` raises unless a live `AttentionPort` has the plan's precision, zero method, empty policy and (when bound) row table and forwards per image. `port_cuts(arm, panel=panel, port=port)` runs it before mapping, so a plan cannot execute under another precision without failing.
- `panel.check_universe(observed_blocks, observed_heads)`: **the worker MUST call this against the live model before running keep arms.** A keep arm cuts the complement of its set over the plan's universe; if the universe is not every (block, head) of the model, the keep arm leaves units uncut. It raises on any geometry mismatch, and on a partial universe unless the plan waived `partial_universe`.

## Mappings

- `port_cuts(arm, *, zero_method=None, panel=None, port=None)` returns a tuple of `attention_port.Cut` specs. It is empty for `condition`, `native` and `deletion`. For a floor arm it returns `floor_arms(base)[...]`. Rows are names, so the caller passes a row table: `item_rows(word_rows, text_rows, total_rows)` builds `image`, `own:<item>` and `others:<item>`.
- `value_spec(arm)` returns `{side, payload_side, target_row, payload_row, cells, dose}`, or `None` for arms that replace nothing. Rows are symbolic: `source`, `wrong` (source − 1 in the 2026-11-12 workers) and `fixed-distance`. `run_side(arm)` gives the prompt the forward runs on. This is the one copy of the mapping in `value_coalition_worker.py` (139–147), `frozen_value_pair_worker.py` (49–55) and `resolver_execution_worker.py` (79–89). The workers are hash-bound and were not edited. Details the workers hide (read from their source):
  - the resolver worker takes the target row from the nomination on the **run** side and the payload row from the nomination on the **payload** side, which differ for `sham` and `reciprocal`;
  - `fixed-distance` is `len(tokens) − 4` of the respective side's prompt;
  - the frozen-pair and resolver workers call `value_coalition.replacement` without a dose, so they run every arm at full dose whatever `dose` says; only the value-coalition worker honours `dose` (and its `replacement` accepts only 0.5 and 1).
- `blend(before, payload, dose)` is the one dose blend. Dose 1 returns `payload.clone()` (the payload's dtype). Dose 0 returns `before.clone()`. Any other dose is an FP32 lerp rounded once to `before`'s dtype, so with mixed dtypes dose 1 and dose 0.999 return different dtypes. It is bit-identical to `value_coalition.replacement` (tested in bf16). `forward_interventions._blend` interpolates in the parent's own dtype, so it differs for bf16 parents. It is hash-bound (via `forward_panels.py`) and was left unchanged. `attention_support.Panel.install` (staged payloads only) copies at full dose, which equals `blend(..., 1)`.
- `to_legacy(panel)` writes the old `{arm, kind, cells, dose}` rows.

## Mandatory controls and why each exists

| control | why |
|---|---|
| both cut types, counted only if they agree | In the FLUX read-graph run the renormalising mask produced object duplication that never appeared under the no-renormalisation cut, in 4 of 4 cases (`saturn/experiments/2026-11-11-flux-read-graph`, AUDIT A5: `z:text<-img` showed no duplication in 4/4 while `text<-img` under the mask showed 3+ cats and 8+ balls). |
| recompute sham + recompute twin for every no-renormalisation cut | A delta sham is exactly native by construction (`native_out − 0`), so it says nothing about the cut's arithmetic. The floor is the larger of the recompute sham's effect and the twin gap \|delta − recompute\| (`docs/ATTENTION-PORT.md`, Precision). |
| matched sham for every renormalising cut | There is no delta form. The FP32 sham on the same cells is the floor. |
| random **keep** arm | The label-free head-set run had none, so "240 own heads suffice" could not be told apart from "any 240 heads suffice" (FINDINGS "Flaws in this run"; AUDIT L1). |
| random sets that report their mass relation | In that run the random sets were neither mass-matched nor disjoint from the own sets (10 to 24 heads shared). `rand120` for snow held .47 of the mass, more than `rand240` (.30) (AUDIT L1). See the worked example: for the own sets, random draws cannot hold mass fixed at all. |
| other-item control (same units, another item's rows) | It holds the units fixed and changes the rows. If the effect stays the same, cutting these units damages any word, not this item (`generic_damage`). It does not hold mass fixed. |
| the unmodified `native` run | `iso:<item>` and `del:<item>` are scored against it, so it must run in every context. |
| floors in every (prompt, seed) | Floors depend on the trajectory, so a floor measured only on the first seed does not cover the others. |
| picture judgement slot | Pixel scores and pictures disagreed on many arms of that run, and the picture decided (AUDIT: rain, snow, ball `rand120`, red `own60`). |

## Budget and leases

`panel.branch_count` is the number of arms times the number of contexts. `panel.check_budget(cap)` refuses a panel over `cap`. `panel.leases(cap=256, max_total=None, seal=None)` splits deterministically and context-major. The atom is a (context, bundle) pair, so a test arm, its twin, its floors and its controls always land in the same lease. Atoms are packed in plan order, and an atom larger than `cap` is refused. Each lease carries the plan hash, the seal hash (`None` without `seal`; a given seal is verified first and enters `lease_sha256`), its index, its own `lease_sha256`, and `depends_on`: the (context, arm) references its branches are scored against that run in another lease. A reference (for example `z:none:<item>` for a keep arm, or `native`) can sit in another lease, so outcomes are scored only after every lease of a context has run. The FLUX lane cap is 256 branches per lease (`FLUX_LANE_CAP`).

## Sealing and results

- `panel.canonical()` / `panel.sha256()` return the canonical JSON form (sorted keys, no NaN) and its sha256. `panel.seal(predictions=...)` returns `{schema, plan_sha256, predictions, predictions_sha256, seal_sha256}`.
- `verify_seal(panel, seal)` (or `panel.verify_seal(seal)`) recomputes the plan hash from the panel, the predictions hash from the stored predictions, and the seal hash from the record. Editing the predictions (or any field) after sealing is detected. A seal from `seal_with_forecast_transaction` is checked for the plan hash only; its predictions are verified by `ForecastTransaction.verify()`.
- `seal_with_forecast_transaction(panel, root, predictions, model_identity=, input_identity=)` uses `ForecastTransaction` without bending it. It uses the capture-optional declaration (`frontier={'kind': 'empty'}`, with no bound model and no weights). The plan's canonical JSON is published as a frontier, the predictions as one forecast, and the result is globally sealed. **What blocks the rest:** `ForecastTransaction.execute` runs a callback in-process, and its model guards assume a bound CPU FP32 model. A FLUX panel runs on a remote mrun lane in BF16/CUDA, so outcomes are not attached through `execute`. They are attached by plan hash with `attach_results`.
- `results_record(panel, outcomes, pair_distances=None, seal=None)` takes `outcomes[context][arm_id]`, the scalar distance (≥ 0) of that arm's output from its `reference` arm (for example pixel MAD against `iso:<item>`). `pair_distances[context]["a|b"]` is the direct distance between two arms. `twin_pairs(panel)` lists the `<cut>|<cut>/cut_recompute` pairs the worker must measure in every context; each is the twin gap of that cut's floor. Without it the gap falls back to \|o(cut) − o(cut_recompute)\|, a lower bound that can be 0, and the floor is marked incomplete (see below).
- `attach_results(panel, results, seal=None)` refuses a different plan hash, a seal that does not verify or that the results do not cite, unknown arms (an alias name points to `resolve_aliases`) or contexts, pair distances for unknown contexts or arms, and non-finite or negative outcomes and pair distances.

## Analysis

`arm_effects(panel, results, floor_factor=1.0)` gives each cut arm's effect, its floor and the floor's parts per context, plus `inside_floor`, `floor_incomplete` and the `floor_factor` used.

`verdicts(panel, results, judgements=None, floor_factor=1.0, recovery_threshold=0.7, twin_gap_waiver=None, floor_factor_waiver=None)` returns one record per (item, unit set, claim):

| key | meaning |
|---|---|
| `status` | `supported`, `set_only`, `not_supported`, `unresolved` or `control_missing` (after any picture judgement) |
| `scalar_status`, `why` | the status from the numbers alone, and why |
| `claim_scope`, `claim_statement` | for `supported`: `identity`; for `set_only`: `needed_set` (cut) or `sufficient_set` (keep), with a statement that begins `WEAKER CLAIM` and names what is not shown; `None` otherwise |
| `ruled_out`, `remaining_alternatives` | alternatives ruled out in every context of both cut types, and those still open |
| `refuted_by` | (cut type, context, control) for every refuting control |
| `findings` | e.g. `low_recovery`, `any_same_size_set_suffices` |
| `floor_incomplete`, `twin_gap_waiver` | whether any floor used a lower-bound twin gap; the written waiver, if any |
| `thresholds` | `floor_factor`, `floor_factor_waiver`, the refute rule, `overlap_sd`, `mass_tolerance`, and for keep claims `recovery_threshold` |
| `cut_types` | per type: status, per-context rows (effect, floor, floor parts, every control's effect, margin, floor, `refutes`, `rules_out`, `confounded_with_mass`), `ruled_out`, `refuting_controls`, `mass_confounded_controls`, `separating_controls` |
| `cut_types_agree`, `missing_cut_types`, `waivers`, `identity` | as named |
| `picture`, `overridden_by_picture`, `picture_blocked_by`, `picture_capped` | the picture judgement and what it could and could not change |

## What a verdict does and does not say

A verdict answers one claim about one unit set for one item. It never says more than the controls that were run and beaten allow. `floor_factor` (default 1) scales every floor; `floor_factor <= 0` is refused unless `floor_factor_waiver` gives a written reason. For one control, *floor* below means `floor_factor × max(test floor, control floor)`.

**Alternatives and what rules each out.** A control rules out an alternative only when the test beats it by more than the floor, with both outcomes and both floors present:

| control | rules out | does not rule out |
|---|---|---|
| random set, mass matched within tolerance (`matched`) | `set_size` (any same-size, stratum- and block-matched set), `mass` (any set with this much of the item's mass) | `generic_damage` |
| random set, heavier than the target (`more`) | `set_size`, `mass` (a heavier set did less damage) | `generic_damage` |
| random set, less mass (`less`), mass unknown (`unknown`) or matched only by reusing target units (`matched_by_reuse`) | `set_size` | `mass` (recorded `confounded_with_mass`), `generic_damage` |
| other-item control (cut claims) | `generic_damage` (cutting these units damages any word) | `mass`, `set_size` |
| other-item control (keep claims) | nothing: reported reference only | |
| external (hand-drawn) control | nothing (can only refute) | |

**Necessity (cut): "these units are needed for this item".** Per cut type and context, in this order:

1. The test effect must exceed its floor; otherwise `unresolved`.
2. **Refutation, whatever the confounds.** If any control whose mass is not known to exceed the target's (random `matched`/`matched_by_reuse`/`less`/`unknown`, other-item, external) has an effect not smaller than the test's by more than the floor (`control effect ≥ test effect − floor`), the claim is `not_supported` and the control is listed in `refuted_by`. A heavier random set (`more`) that does as much damage neither refutes nor supports, since its extra mass explains it.
3. If a twin gap was not measured (no direct delta-vs-recompute pair distance), the floor is a lower bound: `unresolved`, reason `twin_gap_not_measured`, and the record carries `floor_incomplete: true`. Only `twin_gap_waiver="written reason"` lets the lower bound stand; the waiver is recorded and `floor_incomplete` stays true.
4. At least one beaten control must rule out a generic-damage explanation (`set_size` or `generic_damage`); otherwise `control_missing`.
5. Then, with `ruled_out` the alternatives ruled out in every context of both cut types: `supported` (scope `identity`) if `set_size`, `mass` and `generic_damage` are all ruled out; otherwise `set_only` (scope `needed_set`), with `remaining_alternatives` listing what is open.

**Sufficiency (keep): "these units alone carry the item".** Per cut type and context:

1. The all-cut arm's effect must exceed its own floor (there is something to recover); otherwise `unresolved`.
2. `recovery = 1 − keep_effect / all_cut_effect` must reach `recovery_threshold` (default .7, recorded); otherwise `not_supported` with finding `low_recovery`.
3. A matched random **keep** control of the same size must recover clearly less: `random keep effect − keep effect > floor`. If one recovers as much, `not_supported` with finding `any_same_size_set_suffices` (any same-size set carries the item, which is itself a result about the model). No random keep control with outcome and floor: `control_missing`. The other-item keep control is reported under `references` and never counted.
4. Incomplete floors: as for necessity.
5. `supported` (scope `identity`) needs `mass` ruled out too (a `matched` or `more` random keep beaten); otherwise `set_only` (scope `sufficient_set`).

**Across contexts and cut types.** Within a cut type the worst context wins (control_missing > not_supported > unresolved > set_only > supported). Both cut types must be present (`control_missing` otherwise, even when waived) and both must pass; if one refutes and the other passes, the result is `not_supported`.

**`set_only` is not `supported`.** It is a distinct status so that `status == "supported"` can never be read off a weaker claim. Its `claim_statement` begins `WEAKER CLAIM (needed_set)` or `WEAKER CLAIM (sufficient_set)` and lists what is NOT shown. For the FLUX own sets, no random draw can hold mass fixed (worked example below), so the best a cut verdict on those sets can be is `set_only` with `mass` remaining.

**Picture judgements.** `judgements[(item, unit_set, claim)] = {"status", "by", "note"}` records a human or auditor's reading of the pictures on whether the effect is real. A key that names no (item, unit set, claim) of the panel raises. A judgement may downgrade any status to `not_supported` or `unresolved`, and may move `not_supported`/`unresolved` to `supported`/`set_only`, but an upgrade is capped by the scope the beaten controls allow (`picture_capped`), and it is refused (`picture_blocked_by`, status unchanged) when a cut type is missing, a control is missing, a floor is incomplete without a waiver, or no beaten control rules out a generic-damage explanation. A picture can say an effect is real; it cannot stand in for a control that was not run.

### Worked example 1: a necessity claim refuted by a random control

These numbers were computed through `verdicts` on hand-set outcomes (MEASURED from the code, not from a model). Set `s` of `red` holds .70 of red's mass; the disjoint random draw holds .06 (`less`). Every floor is .01, and the twin gaps were measured (0).

| cut type | test | random (mass .06) | other item | random margin | outcome |
|---|---|---|---|---|---|
| zero | .50 | .55 | .05 | −.05 ≤ .01 | random **refutes** |
| renorm | .48 | .46 | .04 | .02 > .01 | random beaten: rules out `set_size` |

Status: `not_supported`, `refuted_by = [z:rand0-s:red]`. The random set has less mass than the target, so the commit-5e09d95 rules called it "mass-confounded" and ignored it, and the beaten other-item control made the verdict `supported` (MEASURED by running that version on these numbers). Now a lighter set doing as much damage refutes the claim: the damage does not need these units. The renorm type passing does not rescue it, because both cut types must agree. `ruled_out = [generic_damage]` (the other item was beaten in both types); `remaining_alternatives = [mass, set_size]`.

### Worked example 2: a keep claim with low recovery

Same set, keep claim, both cut types, floors .01:

| keep effect | all-cut effect | recovery | random keep effect | random recovery | status |
|---|---|---|---|---|---|
| .55 | .60 | 1 − .55/.60 = .083 | .58 | .033 | `not_supported`, finding `low_recovery` |
| .10 | 1.00 | .90 | .80 | .20 | `set_only`, scope `sufficient_set`, remaining `[mass]` |

In the first row the commit-5e09d95 rules returned `supported` (MEASURED by running that version on these numbers, with an other-item keep effect of .90): the other-item keep counted as support, and nothing required the keep effect itself to be small. But keeping only the set leaves almost the whole all-cut damage in place: the set does not carry the item. In the second row the set recovers 90%, and a same-size random set recovers 20%, so `set_size` is ruled out. The random keep set had less mass (`less`), so mass is not ruled out: the verdict is the weaker `sufficient_set` claim, not `supported`.

## Reproduced from existing plans (tests)

- **FLUX `runs/headfree/plan.json`, all 6 words.** The test transcribes the worker's per-word `del:` arm and its `headcut` block, then compares the generator's arms with them by name, cut type, receivers, sources, head set and condition. The generator call uses: cut types `("zero",)`, a `renorm` override for the `mask` sets, `rand120/240` as external controls, no floors, and every waiver. They match exactly. The arm names equal the `metrics.json` arm names of the executed run, minus the prompt-level `sham` and `img<-alltext`, plus the `native` reference the panel now emits (MEASURED from the saved files). The full panel for the same plan contains every executed arm unchanged and adds the floors, the renormalising twins, the random cut and keep arms, the other-item arms and the nested arms.
- **`value_coalition.plan()`.** 56 rows with identical (kind, cells, dose) content. The names differ.
- **`frozen_value_pair.arms()` (10) and `attention_source_resolver.arms()` (9).** Identical content. The one listed difference is that the dose written on native arms is inert, because the workers skip replacement for them. `value_coalition` writes 0 and the other two write 1, so `value_panel(reference_dose=...)` reproduces each.

## Worked example: `red` in the cat prompt

The mass is red's own image←row write-mass share per (block, head), from the saved `runs/full-a/result/capture.cat.4242.npz` (`tok_w` [4, 25, 24, 512]). This is the same source `head_sets3.py` ranked. The strata are joint (blocks 0–4) and single (5–24), and the mass tolerance is 0.02. The numbers were MEASURED offline from saved files; no model was run.

| set | target mass | disjoint matched draw | non-disjoint matched draw |
|---|---|---|---|
| own60 | .646 | max reachable .206 → **cannot hold mass fixed** | mass .663 reached by reusing 45 of 60 target heads (chance 6.1 ± 2.2) → `matched_by_reuse` |
| own120 | .712 | max .223 → cannot | 71 of 120 reused (chance 24.5 ± 3.9) → `matched_by_reuse` |
| own240 | .762 | max .227 → cannot | 130 of 240 reused (chance 98.7 ± 5.8) → `matched_by_reuse` |
| own120-minus-own60 | .067 | matched (.052) → `matched` | matched (.049), overlap 6 (chance 7.0) → `matched` |

The old `rand120` held .142 of the mass against own120's .712. For the own sets, therefore, no control can rule out `mass`: the other-item control rules out `generic_damage` but does not hold mass fixed. A cut verdict on own60/120/240 can be at most `set_only` (scope `needed_set`, remaining `mass`). The added-units-alone set has a mass-matched random control, so it alone can reach `supported`.

Panel size with cut sets own60/120/240, the 2 nested-minus sets and keep own60, for 4 words and 2 seeds (MEASURED from the saved headfree plan):

- Each word gets 97 arms: 12 test, 24 control, 57 floor and 4 reference. That is 6 bundles of 15 arms plus a reference bundle of 7. Each bundle of 15 holds the 2 test cut types with their 3 floor arms, the random control with its 3 floor arms in both types, and the other-item control with its 3 floor arms in both types.
- Over 4 words plus the shared `native` arm that is 389 arms, and over 2 seeds 778 branches.
- `leases(256)` gives 4 leases of 247, 247, 254 and 30 branches.

```python
port = AttentionPort(backend, **port_kwargs(panel, rows=row_table))
check_port(panel, port, forwards=STEPS)
panel.check_universe(n_blocks, n_heads)      # before any keep arm runs
ids = panel.by_id()
for lease in panel.leases(256, seal=seal):
    for context, arm_id in lease["branches"]:
        arm = ids[arm_id]
        cuts = port_cuts(arm, panel=panel, port=port)   # () for native/iso/del; (Cut,) otherwise
        with port.apply(*cuts, forwards=STEPS):          # arm.condition: the worker builds the iso/deleted input
            images[context, arm_id] = run(arm.condition, seed_of(context))
# after every lease has run (references may sit in another lease: lease["depends_on"])
for (context, arm_id), image in images.items():
    ref = ids[arm_id].reference
    if ref is not None:
        outcomes.setdefault(context, {})[arm_id] = px_mad(image, images[context, ref])
pairs = {c: {k: px_mad(images[c, k.split("|")[0]], images[c, k.split("|")[1]]) for k in twin_pairs(panel)}
         for c in panel.contexts}
records = verdicts(panel, results_record(panel, outcomes, pair_distances=pairs, seal=seal))
```

Read `docs/ATTENTION-PORT.md` for the port, and `mdb.attention_ranking` for rankings and matched sets.
