"""Post-hoc (not sealed) route baselines for a saved native-discovery run (numpy only, no model).

Adds, beside the sealed per-document records, three references the run itself does not compute:

* ``offset_k_from_end`` for k = 0..7 (the run records k = 0 and previous token only);
* ``raw_mass_per_head``: per (layer, head), the source class (position 0 or offset 0..7) with the
  most raw native attention mass over the discovery documents; the trivial per-head counterpart of
  the contrast nominator (which ranks the same classes against a uniform causal reader);
* ``heldout_oracle_per_head``: per (layer, head), the class that best matches that held-out
  document's own native top sources. It reads the held-out answer, so it is a ceiling for any
  per-head positional rule, not a baseline.

For every held-out document and route layer it prints each learned arm and the contrast nominator
beside the best trivial baseline (position 0, previous token, offsets 0..7 from end, uniform, raw
mass per head), using the run's own scorer (``mdb_lanes.attention_ranking.evaluate_route_forecast``).

  python posthoc_route_baselines.py <stage> --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mdb_lanes.attention_ranking import evaluate_route_forecast, trivial_route_baselines
from mdb_lanes.forecast_transaction import load_forecast_tree
from mdb_lanes.native_discovery import read_arrays, verify_run
from mdb_lanes.native_discovery_nominator import class_index, class_mass, class_names, nominate_sources, score_nominations

OFFSETS = 8


def per_head_oracle(native, offsets=OFFSETS):
    """Per (stage, head) class maximising agreement with this document's own native top sources."""
    stages, heads, t, _ = native.shape
    top = native.argmax(-1)                       # [stages, heads, T]
    rules = np.zeros((stages, heads), np.int64)
    for c in range(offsets + 1):                   # classes 0 (position 0) .. offsets (offset_{offsets-1})
        cand = nominate_sources({"rules": np.full((stages, heads), c)}, t)
        score = (cand == top).mean(-1)
        if c == 0:
            best = score
        else:
            better = score > best
            rules[better], best = c, np.where(better, score, best)
    return rules


def non_sink_agreement(choices, native_layer, offsets=OFFSETS):
    """Top-source agreement on rows whose native top source is not position 0.

    ``choices``: name -> ``[heads, T]`` nominated source per row. Every such row has query >= 1,
    so previous token is always defined and no offset baseline gets a clamped position-0 hit.
    """
    top = native_layer.argmax(-1)                  # [heads, T]
    heads, t = top.shape
    keep = top != 0
    q = np.broadcast_to(np.arange(t), (heads, t))
    rows = int(keep.sum())
    out = {"rows": rows, "fraction_of_rows": rows / top.size}
    if rows == 0:
        return out
    trivial = {f"offset_{k}_from_end": float((np.maximum(q - k, 0) == top)[keep].mean()) for k in range(offsets)}
    trivial["previous_token"] = trivial["offset_1_from_end"]
    trivial["uniform"] = float((1.0 / (q + 1))[keep].mean())
    out["trivial"] = trivial
    out["predictors"] = {name: float((np.asarray(c) == top)[keep].mean()) for name, c in choices.items()}
    return out


def analyse(stage):
    root = Path(stage) / "evidence"
    verify_run(root)
    report = json.loads((root / "report.json").read_text())
    held = {r["document"]: r for r in report["heldout"]}
    discovery = []
    for path in sorted((root / "traces").glob("*.json")):
        record = json.loads(path.read_text())
        if record["split"] == "discovery":
            discovery.append(read_arrays(root / "traces", record["arrays"])["routes"])
    mass = sum(class_mass(np.asarray(r, np.float64), OFFSETS) for r in discovery)   # [stages, heads, C]
    raw_rules = mass[..., :-1].argmax(-1)          # exclude "other" (names no single source)
    names = class_names(OFFSETS)
    out = {"raw_mass_rules": [[names[c] for c in row] for row in raw_rules.tolist()],
           "discovery_documents": len(discovery), "documents": {}}
    for forecast in sorted(p for p in (root / "forecasts").iterdir() if p.is_dir()):
        observed = json.loads((forecast / "observations/ordinary_native.json").read_text())
        native_tree = load_forecast_tree(forecast, observed["value"])
        doc = native_tree["trace"]["document_id"]
        native = np.asarray(native_tree["native_routes"])
        t = native.shape[-1]
        row = held[doc]
        raw = score_nominations(nominate_sources({"rules": raw_rules}, t), native, name="raw mass per head")
        oracle_rules = per_head_oracle(native)
        oracle = score_nominations(nominate_sources({"rules": oracle_rules}, t), native, name="held-out oracle per head")
        predicted = {}
        for path in sorted((forecast / "forecasts").glob("*.json")):
            sealed = json.loads(path.read_text())
            tree = load_forecast_tree(forecast, sealed["predictions"])
            if "routes" in tree and sealed["name"] in row["arms"]:
                predicted[sealed["name"]] = np.asarray(tree["routes"])
        contrast_sources = None
        for path in sorted((forecast / "forecasts").glob("*.json")):
            sealed = json.loads(path.read_text())
            if sealed["name"] == "contrast_nominator":
                contrast_sources = np.asarray(load_forecast_tree(forecast, sealed["predictions"])["sources"])
        raw_sources = nominate_sources({"rules": raw_rules}, t)
        oracle_sources = nominate_sources({"rules": oracle_rules}, t)
        if set(predicted) != set(row["arms"]):
            raise ValueError("sealed learned routes do not match the report's arms")
        layers = []
        for i, b in enumerate(report["boundaries"][:-1]):
            base = trivial_route_baselines(native[i], offsets_from_end=range(OFFSETS))
            trivial = {k: v["top_source_agreement"] for k, v in base.items()}
            trivial["raw_mass_per_head"] = raw[i]["source_top1"]["nominator"]["top_source_agreement"]
            best = max(trivial, key=trivial.get)
            # The run's own sealed set (offsets k >= 1 from end clamp early rows to position 0).
            core = {k: trivial[k] for k in ("position_0", "previous_token", "offset_0_from_end", "uniform")}
            # Recomputed from the sealed predicted routes; cross-checked against the report when present.
            learned = {}
            for arm, routes in predicted.items():
                value = evaluate_route_forecast(routes[i], native[i])["nominator"]["top_source_agreement"]
                saved = row["arms"][arm][i].get("source_top1")
                if saved is not None and abs(saved["nominator"]["top_source_agreement"] - value) > 1e-12:
                    raise ValueError("recomputed learned agreement differs from the report")
                learned[arm] = value
            entry = {"layer": b, "trivial": trivial, "best_trivial": best,
                     "best_trivial_agreement": trivial[best], "best_core_trivial_agreement": max(core.values()),
                     "learned": learned,
                     "heldout_oracle_per_head": oracle[i]["source_top1"]["nominator"]["top_source_agreement"]}
            choices = {arm: routes[i].argmax(-1) for arm, routes in predicted.items()}
            choices["raw_mass_per_head"] = raw_sources[i]
            choices["heldout_oracle_per_head"] = oracle_sources[i]
            if contrast_sources is not None:
                choices["contrast_nominator"] = contrast_sources[i]
            entry["non_sink_rows"] = non_sink_agreement(choices, native[i])
            if "contrast_nominator" in row:
                entry["contrast_nominator"] = row["contrast_nominator"][i]["source_top1"]["nominator"]["top_source_agreement"]
            layers.append(entry)
        out["documents"][doc] = layers
    # Summary per layer.
    summary = {}
    docs = list(out["documents"])
    for i, b in enumerate(report["boundaries"][:-1]):
        cells = [out["documents"][d][i] for d in docs]
        arms = list(cells[0]["learned"])
        s = {"n": len(cells),
             "mean_best_trivial": float(np.mean([c["best_trivial_agreement"] for c in cells])),
             "mean_best_core_trivial": float(np.mean([c["best_core_trivial_agreement"] for c in cells])),
             "best_trivial_names": sorted({c["best_trivial"] for c in cells}),
             "mean_trivial": {k: float(np.mean([c["trivial"][k] for c in cells])) for k in cells[0]["trivial"]},
             "mean_heldout_oracle_per_head": float(np.mean([c["heldout_oracle_per_head"] for c in cells])),
             "learned": {a: {"mean": float(np.mean([c["learned"][a] for c in cells])),
                             "max": float(np.max([c["learned"][a] for c in cells])),
                             "docs_beating_best_trivial": int(sum(c["learned"][a] > c["best_trivial_agreement"] for c in cells))}
                         for a in arms}}
        if "contrast_nominator" in cells[0]:
            s["contrast_nominator"] = {
                "mean": float(np.mean([c["contrast_nominator"] for c in cells])),
                "docs_beating_best_trivial_incl_raw_mass": int(sum(c["contrast_nominator"] > c["best_trivial_agreement"] for c in cells)),
                "docs_beating_raw_mass_per_head": int(sum(c["contrast_nominator"] > c["trivial"]["raw_mass_per_head"] for c in cells)),
                "docs_below_raw_mass_per_head": int(sum(c["contrast_nominator"] < c["trivial"]["raw_mass_per_head"] for c in cells))}
        ns = [c["non_sink_rows"] for c in cells if c["non_sink_rows"]["rows"]]
        if ns:
            s["non_sink_rows"] = {
                "documents_with_rows": len(ns), "rows": int(sum(x["rows"] for x in ns)),
                "mean_fraction_of_rows": float(np.mean([c["non_sink_rows"]["fraction_of_rows"] for c in cells])),
                "mean_trivial": {k: float(np.mean([x["trivial"][k] for x in ns])) for k in ns[0]["trivial"]},
                "mean_best_trivial": float(np.mean([max(x["trivial"].values()) for x in ns])),
                "predictors": {k: {"mean": float(np.mean([x["predictors"][k] for x in ns])),
                                   "docs_beating_best_trivial": int(sum(x["predictors"][k] > max(x["trivial"].values()) for x in ns))}
                               for k in ns[0]["predictors"]}}
        summary[f"L{b}"] = s
    out["summary"] = summary
    out["learned_cells"] = len(docs) * len(report["boundaries"][:-1]) * len(cells[0]["learned"])
    out["learned_cells_beating_best_trivial"] = int(sum(
        c["learned"][a] > c["best_trivial_agreement"] for d in docs for c in out["documents"][d] for a in c["learned"]))
    out["learned_non_sink_cells_beating_best_trivial"] = int(sum(
        c["non_sink_rows"]["predictors"][a] > max(c["non_sink_rows"]["trivial"].values())
        for d in docs for c in out["documents"][d] if c["non_sink_rows"]["rows"] for a in c["learned"]))
    return out


def show(out):
    print(f"discovery documents for raw-mass rules: {out['discovery_documents']}")
    print(f"learned cells beating the best trivial baseline: {out['learned_cells_beating_best_trivial']}/{out['learned_cells']}; "
          f"on non-position-0 rows: {out['learned_non_sink_cells_beating_best_trivial']}/{out['learned_cells']}")
    for layer, s in out["summary"].items():
        print(f"\n{layer} (n={s['n']} held-out documents)")
        print("  trivial means: " + " ".join(f"{k}={v:.3f}" for k, v in s["mean_trivial"].items()))
        print(f"  mean per-document best trivial {s['mean_best_trivial']:.3f} ({', '.join(s['best_trivial_names'])}); "
              f"run's sealed four only {s['mean_best_core_trivial']:.3f}; "
              f"held-out oracle per head (ceiling, reads answer) {s['mean_heldout_oracle_per_head']:.3f}")
        for a, v in s["learned"].items():
            print(f"  {a:34s} mean {v['mean']:.3f} max {v['max']:.3f} docs > best trivial {v['docs_beating_best_trivial']}/{s['n']}")
        if "contrast_nominator" in s:
            c = s["contrast_nominator"]
            print(f"  {'contrast_nominator (no learning)':34s} mean {c['mean']:.3f} docs > best trivial incl. raw mass "
                  f"{c['docs_beating_best_trivial_incl_raw_mass']}/{s['n']}; vs raw mass per head: above "
                  f"{c['docs_beating_raw_mass_per_head']}, below {c['docs_below_raw_mass_per_head']}")
        if "non_sink_rows" in s:
            n = s["non_sink_rows"]
            print(f"  rows whose native top source is not position 0: {n['rows']} rows, mean share {n['mean_fraction_of_rows']:.3f}")
            print("    trivial means: " + " ".join(f"{k}={v:.3f}" for k, v in n["mean_trivial"].items()
                                              if k in ("offset_0_from_end", "previous_token", "offset_2_from_end", "uniform")))
            print(f"    mean per-document best trivial {n['mean_best_trivial']:.3f}")
            for k, v in n["predictors"].items():
                print(f"    {k:34s} mean {v['mean']:.3f} docs > best trivial {v['docs_beating_best_trivial']}/{n['documents_with_rows']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json", type=Path, help="write the full tables here (must not exist)")
    args = parser.parse_args()
    result = analyse(args.stage)
    show(result)
    if args.json:
        with args.json.open("x") as stream:
            json.dump(result, stream, indent=1, sort_keys=True)
