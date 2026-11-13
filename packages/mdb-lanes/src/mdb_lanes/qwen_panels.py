"""Qwen value and read-cut panels for the temperature prompt pair, on the shared modules.

Pure Python (numpy only, imported lazily). Torch-side execution lives in
``qualification/qwen_panel_worker.py``; this module holds what the worker, the stage script and
the analysis share, so each is tested once:

* the case (Qwen2.5-1.5B base, plain text, no chat template; the value-coalition case of
  2026-11-12, job main-02), its rows, tracked tokens and value cells;
* plan builders: :func:`value_plan` (the 56-arm ``value_coalition.plan()`` panel through
  ``attention_panels.value_panel``) and :func:`read_plan` (read cuts of the first receiver's
  reads of the source row through ``attention_panels.read_panel``), both with the numerical lane
  in the hashed ``port`` block;
* :func:`label_free_sets`: (layer, head) ranking by ``attention_ranking.contrast_ranking`` with
  the trivial baselines of ``attention_ranking`` beside it;
* recipes: :func:`plain_recipe` for every arm and :func:`native_recipe` (the
  ``mdb.native_replay_recipe`` schema) where the arm kind is expressible there;
* analysis: :func:`value_margins`, :func:`coefficients`, :func:`l20_fit`,
  :func:`cell_correlations`, :func:`saved_agreement`, :func:`read_results`.
"""
from __future__ import annotations

import math

from . import attention_panels as P

MODEL = "qwen2.5-1.5b"
MODEL_PATH = "/mnt/big/llm-models/Qwen2.5-1.5B"
WEIGHT_FILE_SHA256 = "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
RECIPIENT_TEXT = ("Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\n"
                  "Description: freezing\nTemperature:")
DONOR_TEXT = ("Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\n"
              "Description: boiling\nTemperature:")
# MEASURED in main-02 results/experiment.json (token_ids, token_labels); the worker re-tokenizes
# and refuses any difference.
RECIPIENT_IDS = (5009, 25, 65904, 198, 40690, 25, 9255, 198, 5009, 25, 63842, 198, 40690, 25, 4017, 198,
                 5009, 25, 42218, 198, 40690, 25)
DONOR_IDS = RECIPIENT_IDS[:18] + (49837,) + RECIPIENT_IDS[19:]
SOURCE_ROW, WRONG_ROW, RECEIVER_ROW = 18, 17, 21
ITEM_ROWS = {"freezing": SOURCE_ROW, "icy": 2, "fiery": 10}
CONTENT_ROWS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)     # word tokens (not ':' or newline)
TRACKED = {"cold": 9255, "hot": 4017, "boiling": 49837}      # ' cold', ' hot', ' boiling'
LAYERS, HEADS, KV_HEADS = 28, 12, 2
LATE_CELLS = tuple((layer, kv) for layer in range(19, 28) for kv in range(2))
NAMED_HEADS = {"L20H5": (20, 5), "L21H0": (21, 0), "L20H11": (20, 11)}
CONTEXT = "temperature:freezing>boiling"
LANES = ("float32", "bfloat16")
GENERATED = 3
ENGINE = ("transformers Qwen2ForCausalLM, eager attention, DynamicCache, batch 1; rows 0..18 in one forward, "
          "rows 19 and 20 one forward each (the order of attention_support.opened in main-02); value edits "
          "on the cache at that point; then row 21 and two generated tokens, one forward each (greedy)")
RESULTS_SCHEMA = "mdb-qwen-panel-results-v1"
RECIPE_SCHEMA = "mdb-qwen-panel-recipe-v1"
GROUPS = {**{f"KV{kv}": [c for c in LATE_CELLS if c[1] == kv] for kv in range(KV_HEADS)},
          **{f"L{layer}-both": [(layer, 0), (layer, 1)] for layer in range(19, 28)}}


class QwenPanelError(ValueError):
    """A plan, result or recipe that does not fit the case."""


def lane_port(lane, **extra):
    if lane not in LANES:
        raise QwenPanelError(f"lane must be one of {LANES}")
    return {"lane": lane, "engine": ENGINE, "model": MODEL, "weight_file_sha256": WEIGHT_FILE_SHA256,
            "source_row": SOURCE_ROW, "wrong_row": WRONG_ROW, "receiver_row": RECEIVER_ROW,
            "generated_tokens": GENERATED, **extra}


# ------------------------------------------------------------------------------------- plans

def value_plan(lane):
    """The 56-branch ``value_coalition.plan()`` panel, with the lane in the hashed port block."""
    return P.value_panel(LATE_CELLS, label="late", groups=GROUPS, doses=(0.5,), reference_dose=0,
                         contexts=(CONTEXT,), item="freezing", port=lane_port(lane))


def row_table(items=("freezing", "icy")):
    table = {"receiver": [RECEIVER_ROW]}
    for item in items:
        table[f"own:{item}"] = [ITEM_ROWS[item]]
    return table


def universe():
    return tuple((layer, head) for layer in range(LAYERS) for head in range(HEADS))


def layer_strata():
    return tuple(f"L{layer}" for layer, _ in universe())


def _check_attention(attention):
    import numpy as np
    a = np.asarray(attention, np.float64)
    if a.shape != (LAYERS, HEADS, RECEIVER_ROW + 1):
        raise QwenPanelError(f"receiver attention must be [{LAYERS}, {HEADS}, {RECEIVER_ROW + 1}] (row 21 of the "
                             "native recipient, every layer and head, every visible source)")
    if not np.isfinite(a).all() or (a < 0).any():
        raise QwenPanelError("receiver attention must be finite and non-negative")
    return a


def label_free_sets(attention, sizes=(3, 12)):
    """Rank (layer, head) for the source row against the other content rows; top-``n`` sets.

    ``attention[layer, head, source]``: the first receiver's (row 21) attention in the native
    recipient. Profiles are ``attention[..., j]`` for every content row ``j``; the ranking is
    ``contrast_ranking(profiles, index of row 18)``. Returns the sets, the ranking summary, the
    rank of each named head, a raw-mass ranking (the trivial head nominator: most attention to
    row 18) and the trivial top-source baselines of ``attention_ranking`` on the receiver row of
    every head (all heads, each set, the named heads).
    """
    import numpy as np
    from .attention_ranking import contrast_ranking, evaluate_top_source
    a = _check_attention(attention)
    profiles = np.stack([a[:, :, j] for j in CONTENT_ROWS])
    item = CONTENT_ROWS.index(SOURCE_ROW)
    ranking = contrast_ranking(profiles, item)
    sets = P.sets_from_ranking(ranking, sizes, prefix="rank")
    order = [int(i) for i in ranking["order"]]
    rank_of = {tuple(int(x) for x in np.unravel_index(f, (LAYERS, HEADS))): r for r, f in enumerate(order)}
    raw = np.lexsort((np.arange(LAYERS * HEADS), -a[:, :, SOURCE_ROW].ravel()))
    raw_rank = {tuple(int(x) for x in np.unravel_index(int(f), (LAYERS, HEADS))): r for r, f in enumerate(raw)}

    def baselines(units):
        rows = np.stack([a[layer, head] for layer, head in units])
        return evaluate_top_source(np.full(len(units), SOURCE_ROW), rows.argmax(-1),
                                   visible=np.full(len(units), RECEIVER_ROW + 1),
                                   query_positions=np.full(len(units), RECEIVER_ROW), target_weights=rows,
                                   offsets_from_end=(0, 1, 2, 3), name="source row 18")
    named = tuple(NAMED_HEADS.values())
    summary = {
        "content_rows": list(CONTENT_ROWS), "item_row": SOURCE_ROW, "rest": ranking["rest"],
        "top24": [[int(x) for x in np.unravel_index(f, (LAYERS, HEADS))] + [float(ranking["scores"][f])]
                  for f in order[:24]],
        "named_rank": {k: rank_of[v] for k, v in NAMED_HEADS.items()},
        "named_raw_mass_rank": {k: raw_rank[v] for k, v in NAMED_HEADS.items()},
        "raw_mass_top12": [[int(x) for x in np.unravel_index(int(f), (LAYERS, HEADS))] for f in raw[:12]],
        "recovered": {f"rank{n}": sorted(k for k, v in NAMED_HEADS.items() if v in set(sets[f"rank{n}"]))
                      for n in sizes},
        "recovered_by_raw_mass": {f"raw{n}": sorted(k for k, v in NAMED_HEADS.items()
                                                    if raw_rank[v] < n) for n in sizes},
        "top_source_baselines": {"all_heads": baselines(universe()), "named3": baselines(named),
                                 **{f"rank{n}": baselines(sets[f"rank{n}"]) for n in sizes}},
    }
    return {name: [list(u) for u in units] for name, units in sets.items()}, summary


def read_sets(attention, sizes=(3, 12)):
    sets, summary = label_free_sets(attention, sizes)
    named = {k: [list(v)] for k, v in NAMED_HEADS.items()}
    named["named3"] = [list(v) for v in NAMED_HEADS.values()]
    return {**named, **sets}, summary


def read_plan(lane, attention, *, n_random=2, seed=20261112, mass_tolerance=0.02, sizes=(3, 12)):
    """Read-cut panel on the first receiver (row 21) reading the source row (``own:freezing``).

    Items: ``freezing`` (row 18) and ``icy`` (row 2); each is the other's other-item control
    (same heads, the other row). Sets: the three named reader heads singly and together, and the
    label-free top-``sizes`` sets; cut and keep, both cut types, floors; matched random heads
    drawn in the same layers (strata = layer), disjoint from the set, per-item mass = the head's
    native attention from row 21 to the item's row, matched within ``mass_tolerance`` where a
    same-layer set can reach it. Reference: the unmodified input (``native``) and the all-cut.
    """
    import numpy as np
    a = _check_attention(attention)
    sets, _ = read_sets(a, sizes)
    names = tuple(sets)
    mass = {item: tuple(float(x) for x in a[:, :, ITEM_ROWS[item]].ravel()) for item in ("freezing", "icy")}
    spec = P.ReadPanelSpec(
        items=("freezing", "icy"), universe=universe(), sets={"freezing": sets, "icy": sets},
        contexts=(CONTEXT,), cut_sets=names, keep_sets=names,
        random=P.RandomControls(n=n_random, strata=layer_strata(), mass=mass, mass_tolerance=mass_tolerance,
                                disjoint=True, seed=seed),
        other_items={"freezing": ("icy",), "icy": ("freezing",)},
        receivers=("receiver",), sources=("own:{item}",), condition=None, references=("all_cut",),
        port=lane_port(lane, attention_sha256=P.sha256_json(np.round(a, 9).tolist())),
        precision="native", row_table=row_table(), forwards_per_image=GENERATED)
    return P.read_panel(spec)


def smoke_read_plan(lane):
    """One read-cut set (L20H5) with its floors, other-item control and all-cut, for the smoke."""
    sets = {"L20H5": [list(NAMED_HEADS["L20H5"])]}
    spec = P.ReadPanelSpec(
        items=("freezing", "icy"), universe=universe(), sets={"freezing": sets, "icy": sets},
        contexts=(CONTEXT,), cut_sets=("L20H5",), random=None,
        waive={"random_cut": "smoke: mechanics and floors only; the full panel draws random heads"},
        other_items={"freezing": ("icy",), "icy": ("freezing",)},
        receivers=("receiver",), sources=("own:{item}",), condition=None, references=("all_cut",),
        port=lane_port(lane), precision="native", row_table=row_table(), forwards_per_image=GENERATED)
    return P.read_panel(spec)


# ------------------------------------------------------------------------------- value specs

def resolve_value(arm):
    """``value_spec`` with symbolic rows resolved for this case (``None`` for arms with no edit)."""
    spec = P.value_spec(arm)
    if spec is None:
        return None
    rows = {"source": SOURCE_ROW, "wrong": WRONG_ROW}
    for key in ("target_row", "payload_row"):
        if spec[key] not in rows:
            raise QwenPanelError(f"row {spec[key]!r} is not defined for this case")
        spec[key] = rows[spec[key]]
    for layer, kv in spec["cells"]:
        if not (0 <= layer < LAYERS and 0 <= kv < KV_HEADS):
            raise QwenPanelError(f"cell {(layer, kv)} outside the {LAYERS} x {KV_HEADS} model")
    return spec


def check_value_geometry(panel, layers, kv_heads):
    """Value-plan analogue of ``Panel.check_universe``: every cell exists in the live model."""
    if (int(layers), int(kv_heads)) != (LAYERS, KV_HEADS):
        raise QwenPanelError(f"live model has {layers} layers x {kv_heads} KV heads; the plan is for "
                             f"{LAYERS} x {KV_HEADS}")
    for arm in panel.arms:
        if arm.units is not None:
            for layer, kv in arm.units:
                if not (0 <= layer < layers and 0 <= kv < kv_heads):
                    raise QwenPanelError(f"{arm.arm}: cell {(layer, kv)} outside the live model")
    return True


# ------------------------------------------------------------------------------------ recipes

def plain_recipe(panel, arm, *, lane, cuts=None, port_kwargs=None):
    """A JSON regeneration recipe for one arm on the worker's engine (every arm kind)."""
    body = {"schema": RECIPE_SCHEMA, "plan_sha256": panel.sha256(), "arm": arm.arm, "family": arm.family,
            "kind": arm.kind, "lane": lane, "model": MODEL, "weight_file_sha256": WEIGHT_FILE_SHA256,
            "engine": ENGINE, "run_side": "recipient", "token_ids": list(RECIPIENT_IDS),
            "decode": f"greedy argmax, {GENERATED} tokens from row {RECEIVER_ROW}"}
    if arm.family == "value":
        side = P.run_side(arm)
        body["run_side"] = side
        body["token_ids"] = list(DONOR_IDS if side == "donor" else RECIPIENT_IDS)
        spec = resolve_value(arm)
        body["edit"] = None if spec is None else {
            **spec, "payload_token_ids": list(DONOR_IDS if spec["payload_side"] == "donor" else RECIPIENT_IDS),
            "blend": "mdb.attention_panels.blend(before, payload, dose) on cache values[:, kv, target_row]",
            "when": "after rows 0..20 are cached, before row 21"}
    else:
        body["port_kwargs"] = port_kwargs
        body["cuts"] = [c.to_dict() for c in (cuts or ())]
        body["forwards"] = GENERATED
    return body


def native_recipe(arm, *, model_identity, execution):
    """(recipe body, None) in the ``native_replay_recipe`` schema, or (None, reason) when the arm
    kind is not expressible there (the module copies whole value rows; it has no blend and no
    attention edit). The body is unsealed; seal it with ``native_replay_recipe.seal``."""
    from .native_replay_recipe import SCHEMA
    if arm.family != "value" and arm.kind != "native":
        return None, "read cut: native_replay_recipe has no attention-edit event"
    spec = resolve_value(arm) if arm.family == "value" else None
    if spec is not None and spec["dose"] != 1:
        return None, f"dose {spec['dose']:g}: copy-values copies whole rows; no blend event"
    run = P.run_side(arm) if arm.family == "value" else "recipient"
    sides = [run] if spec is None else list(dict.fromkeys([run, spec["payload_side"]]))
    ids = {"recipient": RECIPIENT_IDS, "donor": DONOR_IDS}
    roots, events = [], []
    for side in sides:
        tokens = ids[side]
        roots.append({"id": side, "token_ids": list(tokens[:SOURCE_ROW + 1])})
        events.append({"id": f"{side}-prompt-{SOURCE_ROW}", "session": side, "kind": "continue", "steps": 33})
        for index in range(SOURCE_ROW + 1, len(tokens)):
            stage = {"id": f"{side}-stage-{index}", "session": side, "kind": "prompt-token",
                     "token_id": int(tokens[index])}
            if index == len(tokens) - 1:
                stage["id"] = f"{side}-parent"
            events.append(stage)
            if index < len(tokens) - 1:
                events.append({"id": f"{side}-prompt-{index}", "session": side, "kind": "continue", "steps": 33})
    if spec is not None:
        events.append({"id": "edit", "session": run, "kind": "copy-values", "source": f"{spec['payload_side']}-parent",
                       "cells": [list(c) for c in spec["cells"]], "target_row": spec["target_row"],
                       "source_row": spec["payload_row"]})
    for step in range(GENERATED):
        events.append({"id": f"readout-{step}", "session": run, "kind": "continue", "steps": 31})
        events.append({"id": f"commit-{step}", "session": run, "kind": "continue", "steps": 2})
    return {"schema": SCHEMA, "model_identity": model_identity, "execution": dict(execution),
            "roots": roots, "events": events}, None


# ----------------------------------------------------------------------------------- analysis

def bounded_margin(arm, logits, dtype):
    """hot - cold margin as an ``analysis_metrics.Bounded`` with the readout rounding bound of
    ``dtype`` (the dtype the logits were produced in); keys (arm, token) so reused logits cancel."""
    from .analysis_metrics import bounded_values
    hot, cold = bounded_values([logits["hot"], logits["cold"]], dtype=dtype, keys=[(arm, "hot"), (arm, "cold")])
    return hot - cold


def value_margins(arms, dtype):
    """{arm: Bounded margin} from worker rows ``{arm, tracked: {word: logit}}``."""
    return {row["arm"]: bounded_margin(row["arm"], row["tracked"], dtype) for row in arms}


def coefficients(margin):
    """The coefficients of value_coalition_bounds.coefficients (same definitions)."""
    from .analysis_metrics import combine
    m0, full = margin["native-recipient"], margin["coalition-late"]
    out = {"full_effect": full - m0}
    singles = []
    for layer, kv in LATE_CELLS:
        single = margin[f"only-L{layer}-KV{kv}"] - m0
        singles.append(single)
        out[f"singleton L{layer}/KV{kv}"] = single
        out[f"complement_loss L{layer}/KV{kv}"] = full - margin[f"without-L{layer}-KV{kv}"]
    for layer in range(19, 28):
        both = margin[f"coalition-L{layer}-both"]
        out[f"paired_effect L{layer}"] = both - m0
        out[f"interaction L{layer}"] = (both - margin[f"only-L{layer}-KV0"] - margin[f"only-L{layer}-KV1"] + m0)
    out["KV_group_interaction"] = full - margin["coalition-KV0"] - margin["coalition-KV1"] + m0
    out["sum_singleton_effects"] = combine((1, s) for s in singles)
    out["nonadditivity (full - sum singletons)"] = out["full_effect"] - out["sum_singleton_effects"]
    return out


# panel arm id -> value_coalition.plan() arm name (the saved run's names)
LEGACY_NAMES = {"coalition-late": "donor-late", "dose0.5-late": "half-donor-late", "sham-late": "sham-late",
                "wrong-payload-late": "wrong-payload", "wrong-target-late": "wrong-target",
                "reciprocal-late": "reciprocal-late", "coalition-KV0": "only-KV0", "coalition-KV1": "only-KV1",
                **{f"coalition-L{layer}-both": f"only-L{layer}-both" for layer in range(19, 28)}}


def legacy_name(arm_id):
    return LEGACY_NAMES.get(arm_id, arm_id)


def l20_fit(margin):
    """M(a, b) = c0 + ca a + cb b + cab a b through the four L20 points (exact, 4 points)."""
    m00 = margin["native-recipient"]
    m10, m01, m11 = margin["only-L20-KV0"], margin["only-L20-KV1"], margin["coalition-L20-both"]
    return {"c0": m00, "ca (L20/KV0)": m10 - m00, "cb (L20/KV1)": m01 - m00, "cab": m11 - m10 - m01 + m00}


def cell_correlations(max_attention, coeffs):
    """Max source attention per cell vs singleton effect and complement loss (Pearson, Spearman, LOO)."""
    from .analysis_metrics import correlation_report
    labels = [f"L{layer}/KV{kv}" for layer, kv in LATE_CELLS]
    x = [max_attention[label] for label in labels]
    out = {}
    for key in ("singleton", "complement_loss"):
        y = [coeffs[f"{key} {label}"].value for label in labels]
        out[key] = correlation_report(x, y, labels=labels)
    return out


def max_source_attention(attention):
    """{"L<layer>/KV<kv>": max over the 6 query heads of that KV group of attention[layer, head, 18]}."""
    a = _check_attention(attention)
    group = HEADS // KV_HEADS
    return {f"L{layer}/KV{kv}": float(a[layer, kv * group:(kv + 1) * group, SOURCE_ROW].max())
            for layer, kv in LATE_CELLS}


def saved_agreement(saved_arms, rows):
    """Arm-by-arm first token and margin against the saved main-02 ``analysis/arms.json``."""
    saved = {a["arm"]: a for a in saved_arms}
    out, differences = [], []
    for row in rows:
        old = saved.get(legacy_name(row["arm"]))
        if old is None:
            raise QwenPanelError(f"saved run has no arm for {row['arm']!r}")
        record = {"arm": row["arm"], "saved_arm": old["arm"], "first_token": row["tokens_text"][0],
                  "saved_own_text": old["own_text"], "own_text": "".join(row["tokens_text"]),
                  "first_token_equal": old["own_text"].startswith(row["tokens_text"][0]),
                  "three_tokens_equal": old["own_text"] == "".join(row["tokens_text"]),
                  "margin": row["margin"], "saved_margin": old["hot_minus_cold"],
                  "margin_equal": row["margin"] == old["hot_minus_cold"],
                  "margin_diff": row["margin"] - old["hot_minus_cold"]}
        out.append(record)
        if not (record["first_token_equal"] and record["margin_equal"]):
            differences.append(record)
    return {"arms": len(out), "first_token_equal": sum(r["first_token_equal"] for r in out),
            "margin_equal": sum(r["margin_equal"] for r in out), "differences": differences, "rows": out}


def read_results(panel, rows, *, metric="|hot-cold margin shift| from the reference arm (logits)"):
    """``attention_panels`` results record from worker rows ``{arm, margin}``.

    Outcome of an arm: |margin(arm) - margin(reference)|. Pair distance of a delta cut and its
    recompute twin: |margin(cut) - margin(twin)| (the same metric measured directly between them).
    """
    by = {r["arm"]: r for r in rows}
    ids = panel.by_id()
    outcomes = {}
    for name, row in by.items():
        arm = ids[name]
        if arm.reference is None:
            continue
        if arm.reference not in by:
            raise QwenPanelError(f"{name}: reference {arm.reference!r} was not run")
        outcomes[name] = abs(row["margin"] - by[arm.reference]["margin"])
    pairs = {}
    for key in P.twin_pairs(panel):
        a, b = key.split("|")
        if a in by and b in by:
            pairs[key] = abs(by[a]["margin"] - by[b]["margin"])
    return P.results_record(panel, {CONTEXT: outcomes}, pair_distances={CONTEXT: pairs}, metric=metric)


def signed_shift(rows, arm, reference="native"):
    by = {r["arm"]: r for r in rows}
    return by[arm]["margin"] - by[reference]["margin"]


def finite(x):
    return x is not None and math.isfinite(float(x))
