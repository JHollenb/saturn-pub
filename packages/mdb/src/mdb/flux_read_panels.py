"""FLUX.2 Klein read-panel lane: plan and lease checks, regeneration recipes, distances, gone scores.

Framework-free: plain JSON and arithmetic. Importing this module loads neither torch nor numpy;
:func:`cut_from_dict` imports :mod:`mdb.attention_port` (itself torch-free at import).

The model-side worker is ``qualification/flux_read_panel_worker.py``; the launcher and the
offline report are ``qualification/flux_read_panel_stage.py``. See ``docs/FLUX-READ-PANELS.md``.

Storage contract (metadata first): per arm the worker keeps the decoded PNG, a sealed JSON
*recipe* and scalar distances. Latents and attention captures are not written unless the run
asks for them (``retain``). A recipe names the model (adapter identity of transformer + VAE
content, checkpoint file records, parameter pointers by name), the prompt text and how the
conditioning was built, the seed, geometry and schedule, the port's cut specs, precision and
row table, the software lane, and the expected hashes (conditioning, initial noise, final
latent, decoded RGB, PNG bytes). It regenerates the arm from the root on the same host and
software lane; it is not a cross-lane numerical guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import replace

from .attention_panels import Arm, Panel, PanelError, canonical_json, item_rows, sha256_json

RUN_SCHEMA = "mdb-flux-read-panel-run-v1"
RECIPE_SCHEMA = "mdb-flux-read-panel-recipe-v1"
RECIPE_FIELDS = ("schema", "arm", "model", "software", "input", "noise", "port", "expected")
CONDITIONING_KINDS = ("native", "deletion", "hybrid_own")
RETAIN_OPTIONS = ("latents",)


# ------------------------------------------------------------------------------------- prompts

def delete_word(prompt, word):
    """Word-deleted prompt, verbatim from the 2026-11-11 read-graph worker (``delete_word``)."""
    out = re.sub(r"\s*\b" + re.escape(word) + r"\b", "", prompt, count=1)
    out = re.sub(r"\s+", " ", out).replace(" ,", ",").strip()
    return out


def hybrid_own_plan(record_original, record_deleted, word):
    """Row plan of the read-graph ``hybrid(pid, word, "own")`` conditioning (index lists only).

    The word is present only in its own text rows: rows after it are copied from the deleted
    prompt's encoding (shifted by the deleted token count), padding rows from the deleted
    prompt's padding; rows before it keep the original encoding. Returns ``None`` when the
    two encodings do not align (the read-graph worker skipped such arms).
    """
    own = sorted(r for w in record_original["words"] if w["word"] == word for r in w["rows"])
    n_o, n_d = record_original["n_real"], record_deleted["n_real"]
    shift = n_o - n_d
    if (not own or record_original["real_rows"] != list(range(n_o))
            or record_deleted["real_rows"] != list(range(n_d)) or shift < 1):
        return None
    later = list(range(max(own) + 1, n_o))
    ids_o, ids_d = record_original["input_ids"], record_deleted["input_ids"]
    return {"own_rows": own, "shift": shift, "later_rows": later,
            "later_source_rows": [j - shift for j in later], "pad_from": n_o,
            "before_rows": list(range(0, min(own))),
            "later_id_mismatch": int(sum(ids_o[j] != ids_d[j - shift] for j in later))}


def safe_name(arm_id):
    """File-name form of an arm id (the read-graph worker's ``re.sub('[^A-Za-z0-9]+', '_', name)``)."""
    return re.sub("[^A-Za-z0-9]+", "_", arm_id)


def legacy_file_name(prompt_id, seed, arm_id):
    """Saved reference PNG name of the 2026-11-11 run for an arm id of this lane (None if none).

    ``native:<pid>`` maps to the capture-run native image; ``m:img<-alltext:<pid>`` to the
    prompt-level ``img<-alltext`` arm; word arms keep their names.
    """
    if arm_id == f"native:{prompt_id}":
        return f"native.{prompt_id}.{seed}.png"
    if arm_id == f"m:img<-alltext:{prompt_id}":
        return f"arm.{prompt_id}.{seed}.{safe_name('img<-alltext')}.png"
    if "/" in arm_id or "@" in arm_id:
        return None
    return f"arm.{prompt_id}.{seed}.{safe_name(arm_id)}.png"


# --------------------------------------------------------------------------------- plan + lease

def load_plan(plan_json, expected_sha256):
    """Panel from its JSON, refusing a plan whose canonical sha256 differs from the expected one."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise PanelError("an expected plan sha256 (64 hex characters) is required")
    panel = Panel.from_json(plan_json)
    actual = panel.sha256()
    if actual != expected_sha256:
        raise PanelError(f"plan sha256 {actual[:12]} differs from the sealed {expected_sha256[:12]}; refusing to run")
    return panel


def check_lease(panel, lease):
    """Refuse a lease that is not a lease of this plan (hash, index seal, membership)."""
    plan_sha = panel.sha256()
    if lease.get("plan_sha256") != plan_sha:
        raise PanelError("lease belongs to a different plan")
    body = {"plan_sha256": plan_sha, "lease": lease["lease"], "branches": lease["branches"]}
    if sha256_json(body) != lease.get("lease_sha256"):
        raise PanelError("lease seal differs from its branches")
    ids = panel.by_id()
    seen = set()
    for context, arm in lease["branches"]:
        if context not in panel.contexts or arm not in ids:
            raise PanelError(f"lease branch {context}:{arm} is not in the plan")
        if (context, arm) in seen:
            raise PanelError(f"lease repeats branch {context}:{arm}")
        seen.add((context, arm))
    return lease


def lease_roots(panel, lease, items):
    """Reference roots a lease needs: native per (context, prompt), deletion + hybrid per (context, word).

    ``items[item] = {"prompt": pid, "word": word or None}``. Roots run without the port and are
    the references every distance is measured against; they are always recomputed in the job.
    """
    ids = panel.by_id()
    roots = []
    for context, arm_id in lease["branches"]:
        item = ids[arm_id].item
        spec = items[item]
        wanted = [("native", spec["prompt"], None)]
        if spec.get("word"):
            wanted += [("del", spec["prompt"], spec["word"]), ("iso", spec["prompt"], spec["word"])]
        for kind, pid, word in wanted:
            key = (context, kind, pid, word)
            if key not in roots:
                roots.append(key)
    return roots


def work_items(panel, lease, items):
    """Model branches a lease costs: its arms plus its reference roots."""
    return len(lease["branches"]) + len(lease_roots(panel, lease, items))


def leases_within(panel, items, limit=256, cap=256):
    """Largest lease cap <= ``cap`` whose every lease (arms + roots) fits ``limit`` work items."""
    for c in range(cap, 0, -1):
        leases = panel.leases(c)
        if all(work_items(panel, lease, items) <= limit for lease in leases):
            return c, leases
    raise PanelError("no lease cap fits the work-item limit")


# ------------------------------------------------------------------------------------ panel build

def merge_panels(panels, *, extra_arms=(), notes=()):
    """One read panel from per-item panels (same contexts), plus extra arms appended at the end.

    Exists because :class:`mdb.attention_panels.RandomControls` takes one per-unit ``mass``
    for every item; per-item mass needs one :func:`read_panel` call per item.
    """
    panels = list(panels)
    contexts = panels[0].contexts
    if any(p.contexts != contexts or p.family != "read" for p in panels):
        raise PanelError("merged panels need the same contexts and the read family")
    arms, sets, controls, identity, waivers, port = [], {}, {}, {}, {}, {}
    for p in panels:
        arms += p.arms
        sets.update(p.sets)
        controls.update(p.controls)
        identity.update(p.identity)
        waivers.update(p.waivers)
        port.update(p.port)
    out = Panel(family="read", arms=arms + list(extra_arms), contexts=list(contexts), sets=sets,
                controls=controls, identity=identity, waivers=waivers, port=port,
                notes=[n for p in panels for n in p.notes] + list(notes))
    return out


def prompt_level_arms(prompt_id, *, cut_types=("zero", "renorm"), floors=True):
    """The read-graph prompt-level arms: ``native:<pid>`` (old ``sham``) and ``img<-alltext``.

    ``img<-alltext`` = every image row loses its reads of every text row, in every head, under
    the native conditioning. Emitted in the given cut types with :func:`floor_arms` floors
    (``floors=False`` for the as-executed legacy reproduction).
    """
    bundle = f"prompt:{prompt_id}"
    native = f"native:{prompt_id}"
    arms = [Arm(arm=native, family="read", kind="native", role="reference", item=prompt_id, bundle=bundle,
                notes=("port installed, nothing cut (the read-graph `sham` arm); identical to the native root",))]
    prefix = {"zero": "z", "renorm": "m"}
    ids = {t: f"{prefix[t]}:img<-alltext:{prompt_id}" for t in cut_types}
    for t in cut_types:
        twin = ids["renorm" if t == "zero" else "zero"] if len(cut_types) == 2 else None
        base = Arm(arm=ids[t], family="read", kind="all_cut", role="reference", item=prompt_id, units=None,
                   unit_set="img<-alltext", rows_item=prompt_id, receivers=("image",), sources=("text",),
                   cut_type=t, reference=native, twin=twin, bundle=bundle)
        arms.append(base)
        if not floors:
            continue
        common = dict(role="floor", floor_of=base.arm, twin=None)
        if t == "renorm":
            arms.append(replace(base, arm=base.arm + "/sham", kind="floor_sham", **common))
        else:
            arms.append(replace(base, arm=base.arm + "/sham_recompute", kind="floor_sham_recompute", **common))
            arms.append(replace(base, arm=base.arm + "/cut_recompute", kind="floor_cut_recompute", **common))
    return arms


def unit_mass(tok_w, rows, heads):
    """Per-(block, head) share of an item's image<-row write mass, flat in universe order.

    ``tok_w``: nested lists or array [steps, blocks, heads, text_rows] (read-graph capture);
    ``rows``: the item's own text rows. Same quantity ``head_sets3.py`` ranked.
    """
    totals = []
    for b in range(len(tok_w[0])):
        for h in range(heads):
            totals.append(sum(float(tok_w[s][b][h][r]) for s in range(len(tok_w)) for r in rows))
    z = sum(totals)
    if not z > 0:
        raise PanelError("item has no write mass")
    return tuple(t / z for t in totals)


# --------------------------------------------------------------------------------------- rows

def runs(rows):
    """Sorted integer rows -> [[start, stop), ...] runs (compact JSON)."""
    out = []
    for r in sorted({int(x) for x in rows}):
        if out and out[-1][1] == r:
            out[-1][1] = r + 1
        else:
            out.append([r, r + 1])
    return out


def unruns(spans):
    return [r for a, b in spans for r in range(int(a), int(b))]


def row_table(word_rows, text_rows, total_rows):
    """Port row table for one prompt: ``image``, ``text`` and ``own:<w>`` / ``others:<w>`` per word."""
    table = item_rows(word_rows, text_rows, total_rows)
    table["text"] = list(range(int(text_rows)))
    return table


# --------------------------------------------------------------------------------------- cuts

def cut_from_dict(d):
    """Rebuild an :class:`mdb.attention_port.Cut` from ``Cut.to_dict()`` (no such hook exists there)."""
    from .attention_port import Cut

    def rows(spec):
        return tuple(x if isinstance(x, str) else tuple(sorted({int(i) for i in x})) for x in spec)

    heads = d.get("heads")
    return Cut(receivers=rows(d["receivers"]), sources=rows(d["sources"]),
               heads=None if heads is None else frozenset((None if b is None else int(b), int(h)) for b, h in heads),
               blocks=None if d.get("blocks") is None else frozenset(int(b) for b in d["blocks"]),
               steps=None if d.get("steps") is None else frozenset(int(s) for s in d["steps"]),
               renormalize=bool(d["renormalize"]), sham=bool(d["sham"]), zero_method=d.get("zero_method"))


# ------------------------------------------------------------------------------------ recipes

def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def seal_recipe(body):
    """Seal a recipe body (finite JSON, fixed top-level fields) with its sha256 fingerprint."""
    value = json.loads(canonical_json(body))
    if "fingerprint" in value:
        raise PanelError("recipe is already sealed")
    _check_recipe_fields(value)
    return {**value, "fingerprint": fingerprint(value)}


def _check_recipe_fields(value):
    if set(value) != set(RECIPE_FIELDS) or value["schema"] != RECIPE_SCHEMA:
        raise PanelError(f"recipe needs exactly the fields {RECIPE_FIELDS} and schema {RECIPE_SCHEMA}")
    if not isinstance(value["model"].get("model_identity"), str):
        raise PanelError("recipe needs the native model identity")
    cond = value["input"].get("conditioning", {})
    if cond.get("kind") not in CONDITIONING_KINDS:
        raise PanelError(f"conditioning kind must be one of {CONDITIONING_KINDS}")
    if cond["kind"] != "native" and not cond.get("word"):
        raise PanelError("deletion / hybrid conditioning needs the word")
    for key in ("seed", "height", "width", "steps"):
        v = value["noise"].get(key)
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
            raise PanelError(f"noise.{key} must be a non-negative integer")
    port = value["port"]
    if port is not None:
        if port.get("precision") not in ("native", "legacy") or port.get("zero_method") not in ("delta", "recompute"):
            raise PanelError("port precision / zero_method missing")
        if port.get("forwards") != value["noise"]["steps"]:
            raise PanelError("port forwards must equal the denoising steps (one apply = one trajectory)")
        cut_from_dict_all(port.get("cuts", []))
    for key in ("latent_sha256", "image_rgb_sha256"):
        h = value["expected"].get(key)
        if not isinstance(h, str) or len(h) != 64:
            raise PanelError(f"expected.{key} must be a sha256")


def cut_from_dict_all(cuts):
    return tuple(cut_from_dict(c) for c in cuts)


def validate_recipe(recipe):
    """Check a sealed recipe's fingerprint and fields; returns a private copy."""
    value = json.loads(canonical_json(recipe))
    expected = value.pop("fingerprint", None)
    if fingerprint(value) != expected:
        raise PanelError("recipe seal differs (the recipe was edited after sealing)")
    _check_recipe_fields(value)
    return {**value, "fingerprint": expected}


def compare_regeneration(recipe, observed, *, lane=None):
    """Compare a regenerated arm's hashes with its recipe; ``observed`` holds the same hash keys."""
    exp = recipe["expected"]
    keys = [k for k in ("conditioning_sha256", "init_latent_sha256", "latent_sha256", "image_rgb_sha256",
                        "png_sha256") if exp.get(k) is not None]
    rows = {k: {"expected": exp[k], "observed": observed.get(k), "equal": exp[k] == observed.get(k)} for k in keys}
    out = {"recipe": recipe["fingerprint"], "context": recipe["arm"]["context"], "arm": recipe["arm"]["arm"],
           "hashes": rows, "match": all(r["equal"] for r in rows.values())}
    if lane is not None:
        drift = {k: {"recipe": recipe["software"].get(k), "now": v} for k, v in lane.items()
                 if recipe["software"].get(k) != v}
        out["lane_drift"] = drift
        out["same_lane"] = not drift
    return out


# --------------------------------------------------------------------------- distances / results

def results_from_rows(panel, rows, *, metric="px_mad"):
    """Panel results record from worker metric rows (outcome = distance to the arm's reference).

    ``rows``: dicts with ``context``, ``arm``, ``vs_reference[metric]`` and, for
    ``floor_cut_recompute`` arms, ``vs_floor_of[metric]`` (the direct delta/recompute twin gap).
    """
    outcomes, pairs = {}, {}
    ids = panel.by_id()
    for r in rows:
        if r["arm"] not in ids:
            continue
        outcomes.setdefault(r["context"], {})[r["arm"]] = float(r["vs_reference"][metric])
        if r.get("vs_floor_of") is not None:
            a = ids[r["arm"]]
            pairs.setdefault(r["context"], {})[f"{a.floor_of}|{a.arm}"] = float(r["vs_floor_of"][metric])
    from .attention_panels import results_record
    return results_record(panel, outcomes, pair_distances=pairs, metric=f"{metric} vs reference arm")


def merge_results(panel, records):
    """Merge per-lease results records of one plan (refuses duplicates and other plans)."""
    from .attention_panels import attach_results, results_record
    outcomes, pairs, metric = {}, {}, None
    for rec in records:
        attach_results(panel, rec)
        metric = metric or rec.get("metric")
        for ctx, rows in rec["outcomes"].items():
            for arm, v in rows.items():
                if arm in outcomes.setdefault(ctx, {}):
                    raise PanelError(f"outcome {ctx}:{arm} appears in two results records")
                outcomes[ctx][arm] = v
        for ctx, rows in rec.get("pair_distances", {}).items():
            pairs.setdefault(ctx, {}).update(rows)
    return results_record(panel, outcomes, pair_distances=pairs, metric=metric)


# ---------------------------------------------------------------------------------- gone score

def d_score(mad_arm_del, mad_native_del):
    """``d = 1 - MAD(arm, del) / MAD(native, del)`` (headfree_table.py)."""
    if not mad_native_del > 0:
        raise PanelError("MAD(native, del) must be positive")
    return 1.0 - float(mad_arm_del) / float(mad_native_del)


def gone(d_arm, d_iso, d_none):
    """``gone = (d(arm) - d(iso)) / (d(none) - d(iso))`` (headfree_table.py)."""
    span = float(d_none) - float(d_iso)
    if span == 0:
        raise PanelError("d(none) equals d(iso): the gone scale is undefined")
    return (float(d_arm) - float(d_iso)) / span


def gone_table(px_vs_del, mad_native_del, words, columns, *, contexts, none="z:none:{w}", iso="iso:{w}",
               floors=None):
    """Gone scores per word, context and column, with the per-context mean.

    ``px_vs_del[(context, arm_id)]`` = MAD(arm image, deletion image); ``mad_native_del[(context,
    word)]`` = MAD(native, del). ``columns``: arm-id templates with ``{w}``. ``floors[(context,
    arm_id)]`` (px, against iso) adds ``floor_gone_bound`` = floor / (MAD(native, del) * span):
    by the triangle inequality ``|gone(arm) - gone(arm')| <= MAD(arm, arm') / (MAD(native, del)
    * span)``, so an arm whose floor is ``f`` px cannot be told apart from a gone change below
    that bound.
    """
    out = []
    for w in words:
        per = []
        for ctx in contexts:
            base = mad_native_del.get((ctx, w))
            need = [none.format(w=w), iso.format(w=w)]
            if base is None or any((ctx, a) not in px_vs_del for a in need):
                continue
            d = lambda a: d_score(px_vs_del[(ctx, a)], base)
            d_iso, d_none = d(iso.format(w=w)), d(none.format(w=w))
            row = {"context": ctx, "span": d_none - d_iso, "mad_native_del": base, "gone": {}, "floor_gone_bound": {}}
            for col in columns:
                arm = col.format(w=w)
                if (ctx, arm) not in px_vs_del:
                    continue
                row["gone"][col] = gone(d(arm), d_iso, d_none)
                if floors and floors.get((ctx, arm)) is not None:
                    row["floor_gone_bound"][col] = floors[(ctx, arm)] / (base * abs(d_none - d_iso))
            per.append(row)
        if not per:
            continue
        mean = {}
        for col in columns:
            vals = [p["gone"][col] for p in per if col in p["gone"]]
            if len(vals) == len(per):
                mean[col] = sum(vals) / len(vals)
        out.append({"word": w, "contexts": per, "mean": mean})
    return out


def format_gone(table, columns, *, label=lambda c: c.replace(":{w}", "")):
    heads = [label(c) for c in columns]
    lines = [f"{'word':<8} {'ctx':>5} {'span':>5} " + " ".join(f"{h[-11:]:>11}" for h in heads)]
    for row in table:
        for p in row["contexts"]:
            lines.append(f"{row['word']:<8} {p['context']:>5} {p['span']:5.2f} " + " ".join(
                f"{p['gone'][c]:11.2f}" if c in p["gone"] else f"{'-':>11}" for c in columns))
        lines.append(f"{row['word']:<8} {'mean':>5} {'':>5} " + " ".join(
            f"{row['mean'][c]:11.2f}" if c in row["mean"] else f"{'-':>11}" for c in columns))
    return "\n".join(lines)


def finite(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)
