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
| `cut_type`, `zero_method` | `zero` (`renormalize=False`) or `renorm` (`renormalize=True`); the per-cut zero method (`None` = port default) |
| `condition` | input variant the forward runs under (`iso:red`, `delete:red`, `None`); the port does not see it |
| `reference` | arm whose output the outcome is measured against |
| `control_for`, `control`, `separates_identity` | test arms this arm controls; `random` / `other_item` / `external`; whether it can separate unit identity from mass |
| `floor_of`, `twin`, `claim` | the cut whose floor this arm measures; the same arm in the other cut type; `necessary` (cut) or `sufficient` (keep) |
| `dose`, `target_row`, `payload_row` | value family |
| `bundle` | lease atom: a test set, its twins, floors and controls share a bundle |

Kinds:

- **Read cuts.** `cut` cuts the set. `keep` cuts the set's complement over the universe. Each is emitted in both cut types and linked by `twin`.
- **Read floors.** `floor_sham_recompute` and `floor_cut_recompute` are emitted for every no-renormalisation cut. `floor_sham` is emitted for every renormalising cut.
- **Read references.** `condition` is the conditioned input with no cut (`iso:<word>`). `native` is the unmodified input. `all_cut` cuts in every head (`z:none`, `m:none`). `deletion` runs with the word deleted from the prompt. `sham_all` routes every head and row through the FP32 recompute path and removes nothing (the old `z:sham-iso`).
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
))
```

For every test set, and for every item, the read generator emits:

1. The test arm in **both cut types** (`z:<set>:<item>`, `m:<set>:<item>`), linked by `twin`.
2. Its **floor arms**. Each no-renormalisation cut gets `/sham_recompute` (the recompute sham of exactly its cells) and `/cut_recompute` (the recompute twin). Each renormalising cut gets `/sham`. These are built with `attention_port.floor_arms`, so the panel and the port agree on what a floor is. Every cut gets them, controls and references included, and they run in every context: every (prompt, seed), not only the first.
3. **Keep arms** for `keep_sets`: `z:keep-<set>:<item>` cuts in every unit outside the set.
4. **Matched random controls for both cut and keep.** `attention_ranking.matched_control_sets` matches each draw on per-stratum counts and on the number of distinct blocks. Mass matching and disjointness are optional. The random keep arm is the complement of the matched random set. Each draw's report is stored in `panel.controls["<item>:<set>"]`. Whether the draws can separate identity from mass is stated in `panel.identity["<item>:<set>"]`.
5. An **other-item control**: the same units applied to another item's rows (`z:<set>@rows-<other>:<item>`). By default the other item is the next item, cycling. This control separates unit identity from mass when random draws cannot.
6. **Nested-size arms**: `<big>-minus-<small>`, the added units alone. They are treated as test sets with the same controls.
7. **References** (`iso:<item>`, `del:<item>`, `z:none:<item>`, `m:none:<item>`) with floors.
8. `external_controls={"rand120": "own120"}` for hand-drawn control sets. They inherit the target set's cut types and are marked `separates_identity=None` (unknown).

**Refusals.** Each of the following raises unless `waive={key: "written reason"}` is passed: fewer than both cut types on a test set (`both_cut_types`), `floors=False` (`floors`), no random control (`random_cut`, `random_keep`), and an item with no other item (`other_item`). A waiver is recorded in the plan and in every verdict. It never makes a result count: a missing cut type still gives `control_missing`. If a random draw is structurally impossible (for example, too few disjoint units), the generator does not raise. It records `status: control_missing` with the failing constraint.

`value_panel(cells, groups=..., doses=(0.5,), controls=..., references=..., comparators=..., fixed_distance=...)` emits the Codex family. Arms with identical executable content (legacy kind, cell set, dose) are emitted once, and the dropped names go to `panel.aliases`. With two cells, both complements equal singletons. `replay` is exempt from this deduplication because it repeats `native` on purpose.

## Mappings

- `port_cuts(arm)` returns a tuple of `attention_port.Cut` specs. It is empty for `condition`, `native` and `deletion`. For a floor arm it returns `floor_arms(base)[...]`. Rows are names, so the caller passes a row table: `item_rows(word_rows, text_rows, total_rows)` builds `image`, `own:<item>` and `others:<item>`.
- `value_spec(arm)` returns `{side, payload_side, target_row, payload_row, cells, dose}`, or `None` for arms that replace nothing. Rows are symbolic: `source`, `wrong` (source − 1 in the 2026-11-12 workers) and `fixed-distance`. `run_side(arm)` gives the prompt the forward runs on. This is the one copy of the mapping in `value_coalition_worker.py` (139–147), `frozen_value_pair_worker.py` (49–55) and `resolver_execution_worker.py` (79–89). The workers are hash-bound and were not edited.
- `blend(before, payload, dose)` is the one dose blend. Dose 1 copies the payload exactly. Dose 0 copies `before`. Any other dose is an FP32 lerp rounded once to `before`'s dtype. It is bit-identical to `value_coalition.replacement` (tested in bf16). `forward_interventions._blend` interpolates in the parent's own dtype, so it differs for bf16 parents. It is hash-bound (via `forward_panels.py`) and was left unchanged. `attention_support.Panel.install` (staged payloads only) copies at full dose, which equals `blend(..., 1)`.
- `to_legacy(panel)` writes the old `{arm, kind, cells, dose}` rows.

## Mandatory controls and why each exists

| control | why |
|---|---|
| both cut types, counted only if they agree | In the FLUX read-graph run the renormalising mask produced object duplication that never appeared under the no-renormalisation cut, in 4 of 4 cases (`saturn/experiments/2026-11-11-flux-read-graph`, AUDIT A5: `z:text<-img` showed no duplication in 4/4 while `text<-img` under the mask showed 3+ cats and 8+ balls). |
| recompute sham + recompute twin for every no-renormalisation cut | A delta sham is exactly native by construction (`native_out − 0`), so it says nothing about the cut's arithmetic. The floor is the larger of the recompute sham's effect and the twin gap \|delta − recompute\| (`docs/ATTENTION-PORT.md`, Precision). |
| matched sham for every renormalising cut | There is no delta form. The FP32 sham on the same cells is the floor. |
| random **keep** arm | The label-free head-set run had none, so "240 own heads suffice" could not be told apart from "any 240 heads suffice" (FINDINGS "Flaws in this run"; AUDIT L1). |
| random sets that report whether they separate identity from mass | In that run the random sets were neither mass-matched nor disjoint from the own sets (10 to 24 heads shared). `rand120` for snow held .47 of the mass, more than `rand240` (.30) (AUDIT L1). See the worked example: for the own sets, random draws cannot separate the two at all. |
| other-item control (same units, another item's rows) | It holds the units fixed and changes the rows. If the effect stays the same, the units matter and the item does not. This is the separator to use when random draws cannot match mass without reusing the target's units. |
| floors in every (prompt, seed) | Floors depend on the trajectory, so a floor measured only on the first seed does not cover the others. |
| picture override slot | Pixel scores and pictures disagreed on many arms of that run, and the picture decided (AUDIT: rain, snow, ball `rand120`, red `own60`). |

## Budget and leases

`panel.branch_count` is the number of arms times the number of contexts. `panel.check_budget(cap)` refuses a panel over `cap`. `panel.leases(cap=256, max_total=None)` splits deterministically and context-major. The atom is a (context, bundle) pair, so a test arm, its twin, its floors and its controls always land in the same lease. Atoms are packed in plan order, and an atom larger than `cap` is refused. Each lease carries the plan hash, its index and its own `lease_sha256`. The FLUX lane cap is 256 branches per lease (`FLUX_LANE_CAP`).

## Sealing and results

- `panel.canonical()` / `panel.sha256()` return the canonical JSON form (sorted keys, no NaN) and its sha256. `panel.seal(predictions=...)` returns `{plan_sha256, predictions, predictions_sha256, seal_sha256}`.
- `seal_with_forecast_transaction(panel, root, predictions, model_identity=, input_identity=)` uses `ForecastTransaction` without bending it. It uses the capture-optional declaration (`frontier={'kind': 'empty'}`, with no bound model and no weights). The plan's canonical JSON is published as a frontier, the predictions as one forecast, and the result is globally sealed. **What blocks the rest:** `ForecastTransaction.execute` runs a callback in-process, and its model guards assume a bound CPU FP32 model. A FLUX panel runs on a remote mrun lane in BF16/CUDA, so outcomes are not attached through `execute`. They are attached by plan hash with `attach_results`.
- `results_record(panel, outcomes, pair_distances=None, seal=None)` takes `outcomes[context][arm_id]`, the scalar distance of that arm's output from its `reference` arm (for example pixel MAD against `iso:<item>`). `pair_distances[context]["a|b"]` is the direct distance between two arms. When it is given for `cut|cut/cut_recompute`, it is used as the twin gap. Otherwise the gap is \|o(cut) − o(cut_recompute)\|, a lower bound on that distance, and the record says so.
- `attach_results(panel, results, seal=None)` refuses a different plan hash, a seal the results do not cite, unknown arms or contexts, and non-finite outcomes.

## Analysis

`arm_effects(panel, results)` gives each cut arm's effect, its floor and the floor's parts per context, plus `inside_floor`.

`verdicts(panel, results, judgements=None)` returns one record per (item, unit set, claim). Each record has `status`, `scalar_status`, `why`, `cut_types` (per type: status, per-context rows with effect, floor, control margins, and which controls separate identity and which are mass-confounded), `cut_types_agree`, `missing_cut_types`, `waivers`, `identity`, `picture` and `overridden_by_picture`. The statuses are:

- `unresolved`: the test effect, or for a keep claim the all-cut effect, is inside its floor.
- `control_missing`: a floor arm, an outcome, a cut type or every identity-separating control is missing. Identity-separating controls are mass-matched random sets that do not reuse the target, and other-item controls.
- `supported`: the effect beats every identity-separating control by more than the larger of the two floors. For a keep claim, the effect must be smaller than each control's effect by more than that.
- `not_supported`: anything else, including two cut types that disagree.

Across contexts the worst status wins, in the order control_missing > not_supported > unresolved > supported. Random controls that are mass-confounded are reported but never counted. A judgement, `judgements[(item, unit_set, claim)] = {"status", "by", "note"}`, overrides the scalar status, and `overridden_by_picture` records whether it changed it.

## Reproduced from existing plans (tests)

- **FLUX `runs/headfree/plan.json`, all 6 words.** The test transcribes the worker's per-word `del:` arm and its `headcut` block, then compares the generator's arms with them by name, cut type, receivers, sources, head set and condition. The generator call uses: cut types `("zero",)`, a `renorm` override for the `mask` sets, `rand120/240` as external controls, no floors, and every waiver. They match exactly. The arm names equal the `metrics.json` arm names of the executed run, minus the prompt-level `sham` and `img<-alltext` (MEASURED from the saved files). The full panel for the same plan contains every executed arm unchanged and adds the floors, the renormalising twins, the random cut and keep arms, the other-item arms and the nested arms.
- **`value_coalition.plan()`.** 56 rows with identical (kind, cells, dose) content. The names differ.
- **`frozen_value_pair.arms()` (10) and `attention_source_resolver.arms()` (9).** Identical content. The one listed difference is that the dose written on native arms is inert, because the workers skip replacement for them. `value_coalition` writes 0 and the other two write 1, so `value_panel(reference_dose=...)` reproduces each.

## Worked example: `red` in the cat prompt

The mass is red's own image←row write-mass share per (block, head), from the saved `runs/full-a/result/capture.cat.4242.npz` (`tok_w` [4, 25, 24, 512]). This is the same source `head_sets3.py` ranked. The strata are joint (blocks 0–4) and single (5–24), and the mass tolerance is 0.02. The numbers were MEASURED offline from saved files; no model was run.

| set | target mass | disjoint matched draw | non-disjoint matched draw |
|---|---|---|---|
| own60 | .646 | max reachable .206 → **cannot separate** | mass .663 reached by reusing 45 of 60 target heads (chance 6.1 ± 2.2) → **cannot separate** |
| own120 | .712 | max .223 → cannot separate | 71 of 120 reused (chance 24.5 ± 3.9) → cannot separate |
| own240 | .762 | max .227 → cannot separate | 130 of 240 reused (chance 98.7 ± 5.8) → cannot separate |
| own120-minus-own60 | .067 | matched (.052) → separable | matched (.049), overlap 6 (chance 7.0) → separable |

The old `rand120` held .142 of the mass against own120's .712. For the own sets, therefore, only the other-item control (`z:own60@rows-black:red`) can separate identity from mass, and the plan says so in `panel.identity`. The added-units-alone set has a usable random control.

Panel size with cut sets own60/120/240, the 2 nested-minus sets and keep own60, for 4 words and 2 seeds:

- Each word gets 97 arms: 12 test, 24 control, 57 floor and 4 reference. That is 6 bundles of 15 arms plus a reference bundle of 7. Each bundle of 15 holds the 2 test cut types with their 3 floor arms, the random control with its 3 floor arms in both types, and the other-item control with its 3 floor arms in both types.
- Over 4 words that is 388 arms, and over 2 seeds 776 branches.
- `leases(256)` gives 4 leases of 246, 246, 254 and 30 branches.

```python
ids = panel.by_id()
for context, arm_id in lease["branches"]:     # one lease of panel.leases(256)
    arm = ids[arm_id]
    cuts = port_cuts(arm)                    # () for iso/del; (Cut,) otherwise
    with port.apply(*cuts, forwards=STEPS):  # arm.condition: the worker builds the iso/deleted input
        images[context, arm_id] = run(arm.condition, seed_of(context))
for (context, arm_id), image in images.items():
    reference = images[context, ids[arm_id].reference] if ids[arm_id].reference in ids else native[context]
    outcomes.setdefault(context, {})[arm_id] = px_mad(image, reference)
records = verdicts(panel, results_record(panel, outcomes))
```

Read `docs/ATTENTION-PORT.md` for the port, and `mdb.attention_ranking` for rankings and matched sets.
