"""Unit ranking by contrast, matched control sets, and mandatory trivial source baselines.

Framework-free (numpy only). Three tools that every attention/read-mass analysis should share:

* ``contrast_ranking`` / ``group_contrast_ranking``: rank units (heads, (block, head) pairs, any
  unit shape) for one item by ``share * log((share + eps) / (rest + eps))``, where ``share`` is the
  item's read-mass profile normalised to sum to one and ``rest`` is the mean share of the other
  items (or a caller-supplied baseline profile). Ported from the FLUX read-graph head-set scripts
  (saturn/experiments/2026-11-11-flux-read-graph/head_sets2.py and head_sets3.py).
* ``matched_control_sets``: deterministic random unit sets matched to a target set on per-stratum
  counts, number of distinct blocks touched, optionally total mass, and optionally disjointness,
  with a report of what was and was not matched. A random set that is not mass-matched tests mass,
  not identity; the report always shows both masses.
* ``evaluate_top_source`` / ``evaluate_route_forecast``: score a nominator's top-source agreement
  together with fixed position 0, fixed offsets from the end, previous token and uniform chance
  on the same rows. The baselines are always computed; there is no flag to disable them.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9
BASELINE_POLICY = "trivial baselines always computed on the same rows; cannot be disabled"
REQUIRED_BASELINES = ("position_0", "previous_token", "uniform")


# --------------------------------------------------------------------------- contrast ranking

def shares(profiles):
    """Normalise each item's non-negative profile (leading item axis) to sum to one."""
    p = np.asarray(profiles, dtype=np.float64)
    if p.ndim < 2:
        raise ValueError("profiles need a leading item axis and at least one unit axis")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("read mass must be finite and non-negative")
    total = p.reshape(len(p), -1).sum(1)
    if (total <= 0).any():
        raise ValueError("every item needs positive total read mass")
    return p / total.reshape((-1,) + (1,) * (p.ndim - 1))


def _baseline_share(baseline, unit_shape):
    b = np.asarray(baseline, dtype=np.float64)
    if b.shape != unit_shape:
        raise ValueError("baseline profile shape must equal the unit shape")
    return shares(b[None])[0]


def contrast_score(share, rest, *, eps=EPS):
    """``share * log((share + eps) / (rest + eps))`` elementwise (the ported FLUX score)."""
    share, rest = np.asarray(share, np.float64), np.asarray(rest, np.float64)
    return share * np.log((share + eps) / (rest + eps))


def _ranked(score, unit_shape, top):
    flat = np.asarray(score, np.float64).ravel()
    # Descending score; ties broken by lower flat index (deterministic across numpy versions).
    order = np.lexsort((np.arange(flat.size), -flat))
    chosen = order if top is None else order[:top]
    return {"order": order, "scores": flat, "unit_shape": tuple(unit_shape),
            "top": [tuple(int(v) for v in np.unravel_index(i, unit_shape)) for i in chosen]}


def contrast_ranking(profiles, item, *, rest_items=None, baseline=None, eps=EPS, top=None):
    """Rank units for ``item`` against the mean share of other items or a supplied baseline.

    ``rest_items`` selects which items form the comparison (default: all items except ``item``);
    pass it to exclude, for example, other occurrences of the same word. ``baseline`` replaces the
    rest mean with a caller-supplied profile (normalised here). Returns flat ``order`` (best first),
    flat ``scores``, ``unit_shape`` and the ``top`` units as index tuples.
    """
    s = shares(profiles)
    n = len(s)
    if not 0 <= item < n:
        raise ValueError("item index out of range")
    if baseline is not None:
        if rest_items is not None:
            raise ValueError("pass either rest_items or baseline, not both")
        rest = _baseline_share(baseline, s.shape[1:])
    else:
        rest_items = [i for i in range(n) if i != item] if rest_items is None else list(rest_items)
        if not rest_items or item in rest_items or any(not 0 <= i < n for i in rest_items):
            raise ValueError("rest items must be a nonempty set of other items")
        rest = s[rest_items].mean(0)
    result = _ranked(contrast_score(s[item], rest, eps=eps), s.shape[1:], top)
    result["rest"] = "baseline" if baseline is not None else "mean share of %d other items" % len(rest_items)
    return result


def group_contrast_ranking(profiles, labels, group, *, include=None, eps=EPS, top=None):
    """Category version (head_sets2 ``ranked(role)``): mean share of items labelled ``group``
    against the mean share of all included labelled items.

    ``labels`` gives one label per item (``None`` = unlabelled, ignored). ``include`` is a boolean
    mask of items allowed into either mean; pass ``include = item_prompt != held_out_prompt`` for
    the leave-one-prompt-out ranking.
    """
    s = shares(profiles)
    labels = list(labels)
    if len(labels) != len(s):
        raise ValueError("one label per item is required")
    include = np.ones(len(s), bool) if include is None else np.asarray(include, bool)
    if include.shape != (len(s),):
        raise ValueError("include mask must have one entry per item")
    pool = [i for i in range(len(s)) if include[i] and labels[i] is not None]
    members = [i for i in pool if labels[i] == group]
    if not members:
        raise ValueError("no included item carries the requested group label")
    mean, base = s[members].mean(0), s[pool].mean(0)
    result = _ranked(contrast_score(mean, base, eps=eps), s.shape[1:], top)
    result["rest"] = "mean share of %d included labelled items" % len(pool)
    result["members"] = len(members)
    return result


# --------------------------------------------------------------------------- matched controls

def _counts(labels, units):
    keys, counts = np.unique(labels[units], return_counts=True)
    return {k.item() if hasattr(k, "item") else k: int(c) for k, c in zip(keys, counts, strict=True)}


def matched_control_sets(target, n_units, *, strata=None, blocks=None, mass=None,
                         mass_tolerance=None, disjoint=False, seed=0, n_sets=1,
                         max_tries=500, swap_steps=4000):
    """Random unit sets matched to ``target`` (flat unit indices in ``range(n_units)``).

    Always matched: count per stratum (``strata`` = one label per unit; default one stratum).
    Matched when ``blocks`` is given: number of distinct blocks touched (and, when every block lies
    in one stratum, distinct blocks per stratum). Matched when ``mass`` and ``mass_tolerance`` are
    given: total mass within an absolute tolerance, by block- and stratum-preserving swaps.
    ``disjoint=True`` draws only from units outside the target. Deterministic given ``seed``.

    Returns a list of ``{"units": sorted int array, "report": {...}}``; each report lists matched
    and ``unmatched`` constraints, overlap with the target, and target vs control mass whenever
    ``mass`` is supplied (even if mass matching was not requested).
    """
    target = np.unique(np.asarray(target, dtype=np.int64))
    if target.size == 0 or target.min() < 0 or target.max() >= n_units:
        raise ValueError("target must be a nonempty set of unit indices")
    strata = np.zeros(n_units, np.int64) if strata is None else np.asarray(strata)
    if strata.shape != (n_units,):
        raise ValueError("one stratum label per unit is required")
    if blocks is not None:
        blocks = np.asarray(blocks)
        if blocks.shape != (n_units,):
            raise ValueError("one block label per unit is required")
    if mass is not None:
        mass = np.asarray(mass, np.float64)
        if mass.shape != (n_units,) or not np.isfinite(mass).all():
            raise ValueError("one finite mass per unit is required")
    if mass_tolerance is not None and (mass is None or mass_tolerance < 0):
        raise ValueError("mass matching needs per-unit mass and a non-negative tolerance")
    rng = np.random.default_rng(seed)
    allowed = np.ones(n_units, bool)
    if disjoint:
        allowed[target] = False
    quota = _counts(strata, target)
    block_mode = None
    if blocks is not None:
        per_block = {}
        for u in range(n_units):
            per_block.setdefault(blocks[u].item(), set()).add(strata[u].item())
        block_mode = "per-stratum" if all(len(v) == 1 for v in per_block.values()) else "total"
        target_blocks = set(blocks[target].tolist())
        block_quota = ({s: len({blocks[u].item() for u in target if strata[u] == s}) for s in quota}
                       if block_mode == "per-stratum" else len(target_blocks))
    out = []
    for index in range(n_sets):
        chosen, attempts, why = None, 0, []
        while chosen is None and attempts < max_tries:
            attempts += 1
            chosen = _draw(rng, strata, blocks, allowed, quota, block_mode,
                           block_quota if blocks is not None else None, why)
        if chosen is None:
            counts = {}
            for reason in why:
                counts[reason] = counts.get(reason, 0) + 1
            failed = "; ".join(f"{r} (x{c})" for r, c in sorted(counts.items(), key=lambda kv: -kv[1]))
            raise ValueError(f"no structurally matched control set found in {max_tries} tries "
                             f"(disjoint={bool(disjoint)}); constraint(s) that failed: {failed}")
        swaps = 0
        if mass_tolerance is not None:
            chosen, swaps = _match_mass(rng, chosen, strata, blocks, allowed, mass,
                                        float(mass[target].sum()), mass_tolerance, swap_steps)
        out.append({"units": np.sort(chosen), "report": _report(
            target, np.sort(chosen), strata, blocks, mass, mass_tolerance, disjoint, seed,
            index, attempts, swaps, block_mode)})
    return out


def _draw(rng, strata, blocks, allowed, quota, block_mode, block_quota, why=None):
    why = [] if why is None else why
    picked = []
    if blocks is None:
        for s, c in quota.items():
            pool = np.flatnonzero(allowed & (strata == s))
            if len(pool) < c:
                why.append(f"stratum {s!r}: needs {c} units, only {len(pool)} allowed")
                return None
            picked += rng.choice(pool, size=c, replace=False).tolist()
        return np.array(picked, np.int64)
    if block_mode == "per-stratum":
        for s, c in quota.items():
            pool_blocks = np.unique(blocks[allowed & (strata == s)])
            k = block_quota[s]
            if len(pool_blocks) < k:
                why.append(f"stratum {s!r}: needs {k} distinct blocks, only {len(pool_blocks)} allowed")
                return None
            chosen_blocks = rng.choice(pool_blocks, size=k, replace=False)
            in_blocks = allowed & (strata == s) & np.isin(blocks, chosen_blocks)
            if in_blocks.sum() < c:
                why.append(f"stratum {s!r}: {k} chosen blocks hold {int(in_blocks.sum())} allowed units, need {c}")
                return None
            first = [rng.choice(np.flatnonzero(in_blocks & (blocks == b))) for b in chosen_blocks]
            rest = np.setdiff1d(np.flatnonzero(in_blocks), first)
            picked += first + rng.choice(rest, size=c - k, replace=False).tolist()
        return np.array(picked, np.int64)
    # Mixed strata within blocks: choose k blocks, cover each, then fill quotas; caller retries.
    pool_blocks = np.unique(blocks[allowed])
    if len(pool_blocks) < block_quota:
        why.append(f"distinct blocks: needs {block_quota}, only {len(pool_blocks)} allowed")
        return None
    chosen_blocks = rng.choice(pool_blocks, size=block_quota, replace=False)
    remaining = dict(quota)
    for b in rng.permutation(chosen_blocks):
        options = [u for u in np.flatnonzero(allowed & (blocks == b))
                   if remaining.get(strata[u].item(), 0) > 0 and u not in picked]
        if not options:
            why.append(f"mixed-stratum block {b!r}: no allowed unit left that fits the remaining per-stratum counts")
            return None
        u = int(rng.choice(options))
        picked.append(u)
        remaining[strata[u].item()] -= 1
    in_blocks = allowed & np.isin(blocks, chosen_blocks)
    for s, c in remaining.items():
        pool = np.setdiff1d(np.flatnonzero(in_blocks & (strata == s)), picked)
        if len(pool) < c:
            why.append(f"stratum {s!r}: needs {c} more units inside the chosen blocks, only {len(pool)} allowed")
            return None
        picked += rng.choice(pool, size=c, replace=False).tolist() if c else []
    return np.array(picked, np.int64)


def _match_mass(rng, chosen, strata, blocks, allowed, mass, goal, tolerance, steps):
    chosen = list(int(u) for u in chosen)
    current = float(mass[chosen].sum())
    swaps = 0
    for _ in range(steps):
        gap = goal - current
        if abs(gap) <= tolerance:
            break
        u = chosen[int(rng.integers(len(chosen)))]
        selected = np.zeros(len(mass), bool)
        selected[chosen] = True
        ok = allowed & ~selected & (strata == strata[u])
        if blocks is not None:
            # Keep the number of distinct blocks: a sole representative may move to its own block
            # or to an untouched block; otherwise the unit may move within touched blocks.
            touched = np.isin(blocks, blocks[chosen])
            if np.sum(blocks[chosen] == blocks[u]) == 1:
                ok &= (blocks == blocks[u]) | ~touched
            else:
                ok &= touched
        candidates = np.flatnonzero(ok)
        if not len(candidates):
            continue
        new_gap = np.abs(gap - (mass[candidates] - mass[u]))
        best = int(candidates[int(np.argmin(new_gap))])
        if new_gap.min() < abs(gap):
            chosen[chosen.index(u)] = best
            current += float(mass[best] - mass[u])
            swaps += 1
    return np.array(chosen, np.int64), swaps


def _report(target, units, strata, blocks, mass, tolerance, disjoint, seed, index, attempts,
            swaps, block_mode):
    unmatched = []
    tc, cc = _counts(strata, target), _counts(strata, units)
    report = {"seed": seed, "set_index": index, "attempts": attempts, "mass_swaps": swaps,
              "size": {"target": int(target.size), "control": int(units.size)},
              "strata_counts": {"target": tc, "control": cc, "matched": tc == cc}}
    if tc != cc:
        unmatched.append("strata_counts")
    if blocks is None:
        report["distinct_blocks"] = {"status": "not requested"}
    else:
        tb, cb = len(set(blocks[target].tolist())), len(set(blocks[units].tolist()))
        report["distinct_blocks"] = {"target": tb, "control": cb, "matched": tb == cb, "mode": block_mode}
        if tb != cb:
            unmatched.append("distinct_blocks")
    if mass is None:
        report["mass"] = {"status": "not supplied"}
    else:
        tm, cm = float(mass[target].sum()), float(mass[units].sum())
        report["mass"] = {"target": tm, "control": cm, "tolerance": tolerance,
                          "matched": None if tolerance is None else bool(abs(tm - cm) <= tolerance),
                          "status": "matched within tolerance" if tolerance is not None and abs(tm - cm) <= tolerance
                          else "NOT matched: control tests mass as well as identity"}
        if tolerance is None or abs(tm - cm) > tolerance:
            unmatched.append("mass")
    overlap = int(np.intersect1d(target, units).size)
    report["overlap_with_target"] = overlap
    report["disjoint"] = {"requested": bool(disjoint), "matched": overlap == 0}
    if disjoint and overlap:
        unmatched.append("disjoint")
    report["unmatched"] = unmatched
    return report


# --------------------------------------------------------------------------- trivial baselines

def _int_rows(name, value, n=None):
    a = np.asarray(value)
    if a.ndim != 1 or (n is not None and len(a) != n) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a 1-D integer array with one entry per row")
    return a.astype(np.int64)


def _offset(k):
    k = int(k)
    if k < 0:
        raise ValueError("offsets_from_end must be non-negative")
    return k


def evaluate_top_source(predicted, target, *, visible, query_positions=None, target_weights=None,
                        offsets_from_end=(0,), name="nominator"):
    """Top-source agreement of a nominator beside trivial baselines on the same rows.

    ``predicted`` / ``target``: nominated and native top source per row. ``visible``: number of
    sources visible to each row (sources ``0 .. visible-1``). ``query_positions`` (default
    ``visible - 1``, must satisfy ``0 <= query <= visible``) defines the previous-token baseline
    ``query - 1``, which is scored only on rows with query >= 1 (the row count is reported). ``offsets_from_end`` adds
    ``visible - 1 - k`` baselines (k = 0 is the most recent visible source; offset 0 is always
    included, even for ``offsets_from_end=()``). Uniform is reported as
    its exact expected agreement ``mean(1 / visible)``. With ``target_weights`` (rows x sources
    native weights), every entry also reports the native weight at the nominated source.
    """
    target = _int_rows("target", target)
    n = len(target)
    if n == 0:
        raise ValueError("at least one row is required")
    predicted = _int_rows("predicted", predicted, n)
    visible = _int_rows("visible", visible, n)
    query = visible - 1 if query_positions is None else _int_rows("query_positions", query_positions, n)
    if (visible < 1).any() or (target < 0).any() or (target >= visible).any():
        raise ValueError("native top source must lie inside the visible sources")
    if (query < 0).any() or (query > visible).any():
        raise ValueError("query_positions must satisfy 0 <= query <= visible (query - 1 must be a "
                         "visible source for the previous-token baseline)")
    offsets = sorted({0} | {_offset(k) for k in offsets_from_end})
    weights = None
    if target_weights is not None:
        weights = np.asarray(target_weights, np.float64)
        if weights.ndim != 2 or len(weights) != n or weights.shape[1] < visible.max():
            raise ValueError("target weights must be rows x sources")

    def score(choice, rows=None):
        choice = np.asarray(choice, np.int64)
        keep = np.ones(n, bool) if rows is None else rows
        valid = (choice >= 0) & (choice < visible)
        count = int(keep.sum())
        row = {"rows": count, "top_source_agreement":
               float(np.mean((valid & (choice == target))[keep])) if count else None}
        if weights is not None:
            row["native_weight_at_choice"] = float(np.mean(np.where(
                valid, weights[np.arange(n), np.clip(choice, 0, weights.shape[1] - 1)], 0.0)[keep])) if count else None
        return row

    # Query 0 has no previous token (the clamp to position 0 would score it correct whenever the
    # native top source is position 0), so the previous-token baseline is scored on query >= 1 only.
    has_previous = query >= 1
    previous = score(query - 1, has_previous)
    previous["excluded_rows"] = int(n - has_previous.sum())
    previous["note"] = "rows with query position 0 excluded (no previous token)"
    if has_previous.any():
        previous["nominator_top_source_agreement_same_rows"] = float(
            np.mean((predicted == target)[has_previous] & ((predicted >= 0) & (predicted < visible))[has_previous]))
    baselines = {"position_0": score(np.zeros(n, np.int64)), "previous_token": previous}
    for k in offsets:
        baselines[f"offset_{k}_from_end"] = score(np.maximum(visible - 1 - k, 0))
    uniform = {"rows": n, "top_source_agreement": float(np.mean(1.0 / visible)), "kind": "expected value"}
    if weights is not None:
        uniform["native_weight_at_choice"] = float(np.mean(
            [weights[i, :visible[i]].sum() / visible[i] for i in range(n)]))
    baselines["uniform"] = uniform
    nominator = {"name": name, **score(predicted)}
    scored = [k for k in baselines if baselines[k]["top_source_agreement"] is not None]
    best = max(scored, key=lambda k: baselines[k]["top_source_agreement"])
    margin = nominator["top_source_agreement"] - baselines[best]["top_source_agreement"]
    return {"rows": n, "nominator": nominator, "baselines": baselines,
            "best_trivial": {"name": best, "top_source_agreement": baselines[best]["top_source_agreement"]},
            "margin_over_best_trivial": margin, "beats_every_trivial_baseline": bool(margin > 0),
            "baseline_policy": BASELINE_POLICY}


def require_trivial_baselines(record):
    """Raise unless an evaluation record carries the mandatory trivial baselines."""
    baselines = record.get("baselines") if isinstance(record, dict) else None
    if not baselines or any(k not in baselines for k in REQUIRED_BASELINES) or "best_trivial" not in record:
        raise ValueError("source evaluation lacks the mandatory trivial baselines")
    return record


def evaluate_route_forecast(predicted_routes, native_routes, *, offsets_from_end=(0,), name="nominator"):
    """Apply ``evaluate_top_source`` to attention arrays ``[..., queries, sources]``.

    Queries are the last ``queries`` positions of the ``sources`` axis (causal self-attention when
    the two are equal), so row ``q`` sees ``sources - queries + q + 1`` sources.
    """
    p, t = np.asarray(predicted_routes), np.asarray(native_routes)
    if p.shape != t.shape or t.ndim < 2:
        raise ValueError("predicted and native routes need identical [..., queries, sources] shapes")
    queries, sources = t.shape[-2:]
    if queries > sources:
        raise ValueError("more queries than sources")
    q = sources - queries + np.arange(queries)
    lead = int(np.prod(t.shape[:-2], dtype=np.int64))
    rows = t.reshape(lead * queries, sources)
    return evaluate_top_source(p.reshape(-1, sources).argmax(-1), rows.argmax(-1),
                               visible=np.tile(q + 1, lead), query_positions=np.tile(q, lead),
                               target_weights=rows, offsets_from_end=offsets_from_end, name=name)


def trivial_route_baselines(native_routes, *, offsets_from_end=(0,)):
    """Trivial top-source baselines alone for native attention ``[..., queries, sources]``."""
    t = np.asarray(native_routes)
    baselines = evaluate_route_forecast(t, t, offsets_from_end=offsets_from_end,
                                        name="native-self (ceiling)")["baselines"]
    baselines["previous_token"].pop("nominator_top_source_agreement_same_rows", None)  # nominator-specific
    return baselines
