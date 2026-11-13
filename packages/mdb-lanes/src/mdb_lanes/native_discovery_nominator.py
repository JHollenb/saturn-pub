"""Learning-free source nominator: contrast ranking over captured native attention (numpy only).

For every captured route layer (span) and head, the discovery documents' native attention mass is
pooled into source classes relative to each query row: ``position_0``, ``offset_k`` (source
``query - k`` for ``k = 0 .. offsets-1``, current token at ``k = 0``) and ``other`` (everything
further back). ``mdb.attention_ranking.contrast_ranking`` ranks the classes of each (layer, head)
against the mass a uniform causal reader would put in the same classes over the same rows, and
the highest-ranked concrete class (``other`` names no single source) becomes that head's rule.
On a held-out document the rule yields one nominated source per (layer, head, query) from the
document length alone, so the nomination can be sealed before any native held-out computation.

No parameter is fitted by gradient descent and no held-out or development data is read.
"""
from __future__ import annotations

import numpy as np

from .attention_ranking import contrast_ranking, evaluate_route_forecast

SCHEMA = "mdb-native-discovery-v1/contrast-nominator"


def class_names(offsets):
    return ["position_0"] + [f"offset_{k}" for k in range(offsets)] + ["other"]


def class_index(length, offsets):
    """``[T, T]`` class of source ``s`` for query ``q`` (``-1`` above the causal diagonal)."""
    q = np.arange(length)[:, None]
    s = np.arange(length)[None, :]
    distance = q - s
    out = np.where(distance < offsets, 1 + distance, 1 + offsets)
    out = np.where(s == 0, 0, out)
    return np.where(distance < 0, -1, out).astype(np.int64)


def class_mass(routes, offsets):
    """Summed attention mass per class over query rows: ``[..., heads, T, T] -> [..., heads, C]``."""
    routes = np.asarray(routes, np.float64)
    index = class_index(routes.shape[-1], offsets)
    out = np.zeros(routes.shape[:-2] + (offsets + 2,))
    for c in range(offsets + 2):
        out[..., c] = (routes * (index == c)).sum((-1, -2))
    return out


def uniform_class_mass(length, offsets):
    """Mass a uniform causal reader puts in each class, summed over the ``length`` query rows."""
    index = class_index(length, offsets)
    uniform = np.tril(np.ones((length, length))) / np.arange(1, length + 1)[:, None]
    return np.array([(uniform * (index == c)).sum() for c in range(offsets + 2)])


def fit_contrast_nominator(route_arrays, *, offsets=8, document_ids=()):
    """Per-(layer, head) source rule from discovery attention ``[stages, heads, T, T]`` per document."""
    route_arrays = [np.asarray(r, np.float64) for r in route_arrays]
    if not route_arrays or any(r.ndim != 4 or r.shape[:2] != route_arrays[0].shape[:2] for r in route_arrays):
        raise ValueError("route arrays must share [stages, heads, T, T] leading geometry")
    if type(offsets) is not int or offsets < 1:
        raise ValueError("offsets must be a positive integer")
    stages, heads = route_arrays[0].shape[:2]
    mass = sum(class_mass(r, offsets) for r in route_arrays)          # [stages, heads, C]
    chance = sum(uniform_class_mass(r.shape[-1], offsets) for r in route_arrays)  # [C]
    names = class_names(offsets)
    profiles = mass.reshape(stages * heads, -1)
    rules, records = np.zeros((stages, heads), np.int64), []
    for item in range(stages * heads):
        ranked = contrast_ranking(profiles, item, baseline=chance)
        versus_heads = contrast_ranking(profiles, item)
        order = [int(i) for i in ranked["order"]]
        choice = next(c for c in order if names[c] != "other")
        stage, head = divmod(item, heads)
        rules[stage, head] = choice
        share = profiles[item] / profiles[item].sum()
        records.append({"stage": stage, "head": head, "rule": names[choice],
                        "ranking": [names[c] for c in order],
                        "contrast_scores": {names[c]: float(ranked["scores"][c]) for c in range(len(names))},
                        "discovery_mass_share": {names[c]: float(share[c]) for c in range(len(names))},
                        "top_class_against_other_heads": names[int(versus_heads["order"][0])]})
    return {"schema": SCHEMA, "offsets": offsets, "classes": names, "rules": rules.tolist(),
            "heads": records, "documents": list(document_ids), "learning": "none",
            "ranking": "mdb.attention_ranking.contrast_ranking against the uniform-causal class mass over the same discovery rows",
            "fit_scope": "discovery native attention only"}


def nominate_sources(nominator, length):
    """Nominated source ``[stages, heads, T]`` for a document of ``length`` tokens."""
    rules = np.asarray(nominator["rules"], np.int64)
    q = np.arange(length)
    out = np.zeros(rules.shape + (length,), np.int64)
    for (stage, head), rule in np.ndenumerate(rules):
        if rule == 0:
            out[stage, head] = 0
        else:
            out[stage, head] = np.maximum(q - (rule - 1), 0)  # too-early rows fall back to position 0
    return out


def one_hot_routes(sources):
    """``[..., T]`` source indices to ``[..., T, T]`` one-hot causal routes."""
    sources = np.asarray(sources, np.int64)
    length = sources.shape[-1]
    if (sources > np.arange(length)).any() or (sources < 0).any():
        raise ValueError("nominated sources must be causal")
    return (np.arange(length) == sources[..., None]).astype(np.float32)


def score_nominations(sources, native_routes, *, name="contrast nominator"):
    """Per-layer top-source agreement beside the mandatory trivial baselines."""
    native_routes = np.asarray(native_routes)
    routes = one_hot_routes(sources)
    if routes.shape != native_routes.shape:
        raise ValueError("nominations and native routes differ in geometry")
    return [{"span": i, "source_top1": evaluate_route_forecast(routes[i], native_routes[i], name=name)}
            for i in range(len(routes))]
