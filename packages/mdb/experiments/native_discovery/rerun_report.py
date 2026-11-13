"""Artifact-only tables for a position-robust native-discovery run (numpy only, no model).

Reads ``<stage>/evidence/report.json`` after ``verify_run`` and prints, per arm, the mean and
per-document output KL at the last position, uncentred / centred / position-0-excluded relative
MSE at each boundary, the fraction of documents with a first-token activation, top-source
agreement beside every trivial baseline per layer, and paired comparisons (sign count, mean and
median difference, paired bootstrap 95% interval over held-out documents).

  python rerun_report.py /path/to/rerun-2-main --json tables.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mdb.native_discovery import verify_run

METRICS = ("relative_mse", "centered_relative_mse", "relative_mse_excluding_position_0",
           "centered_relative_mse_excluding_position_0")
BASELINE_STATES = ("hold_state", "mean_write", "split_mean_write")
SINK_THRESHOLD = 0.5  # a document "has a first-token activation" at a boundary above this share
BOOTSTRAP = 10000


def bootstrap(diff, *, seed=20261112):
    diff = np.asarray(diff, np.float64)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired(candidate, reference, *, lower_is_better=True):
    """Paired summary of ``candidate - reference`` over documents."""
    c, r = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    d = c - r
    better = int(np.sum(d < 0) if lower_is_better else np.sum(d > 0))
    worse = int(np.sum(d > 0) if lower_is_better else np.sum(d < 0))
    return {"n": int(len(d)), "candidate_better": better, "candidate_worse": worse,
            "ties": int(len(d) - better - worse), "mean_difference": float(d.mean()),
            "median_difference": float(np.median(d)), "bootstrap_95": bootstrap(d),
            "candidate_mean": float(c.mean()), "reference_mean": float(r.mean()),
            "direction": "candidate - reference; " + ("lower" if lower_is_better else "higher") + " is better"}


def tables(stage):
    root = Path(stage) / "evidence"
    verify_run(root)
    report = json.loads((root / "report.json").read_text())
    held = report["heldout"]
    boundaries = report["boundaries"]
    learned = list(held[0]["arms"])
    docs = [row["document"] for row in held]

    def states(row, arm):
        return [s["state"] for s in row["arms"][arm]] if arm in row["arms"] else row[arm]

    def consumer(row, arm):
        return row["native_output_consumer"][arm] if arm in row["arms"] else row[arm + "_consumer"]

    arms = [a for a in BASELINE_STATES if a in held[0]] + learned
    out = {"documents": docs, "boundaries": boundaries, "options": report.get("options"),
           "output_kl": {}, "top1": {}, "state": {}, "sink": {}, "routes": {}, "paired": {}}
    for arm in arms:
        kl = [consumer(r, arm)["native_to_predicted_kl"] for r in held]
        top = [consumer(r, arm)["top1_agreement"] for r in held]
        out["output_kl"][arm] = {"mean": float(np.mean(kl)), "per_document": kl}
        out["top1"][arm] = {"agree": int(np.sum(top)), "n": len(top)}
        out["state"][arm] = {}
        for i, b in enumerate(boundaries[1:]):
            out["state"][arm][f"L{b}"] = {
                m: {"mean": float(np.mean([states(r, arm)[i][m] for r in held])),
                    "median": float(np.median([states(r, arm)[i][m] for r in held])),
                    "per_document": [states(r, arm)[i][m] for r in held]} for m in METRICS}
    for i, b in enumerate(boundaries[1:]):
        share = [r["hold_state"][i]["position_0_energy_fraction"] for r in held]
        out["sink"][f"L{b}"] = {"documents_above_threshold": int(np.sum(np.asarray(share) > SINK_THRESHOLD)),
                                "n": len(share), "threshold": SINK_THRESHOLD, "per_document": share}
    # Top-source agreement per captured route layer, with every trivial baseline beside it.
    route_arms = learned + (["contrast_nominator"] if "contrast_nominator" in held[0] else [])
    for i, b in enumerate(boundaries[:-1]):
        layer = {"baselines": {}}
        for name in held[0]["route_top1_trivial_baselines"][i]:
            values = [r["route_top1_trivial_baselines"][i][name]["top_source_agreement"] for r in held]
            if all(v is not None for v in values):
                layer["baselines"][name] = {"mean": float(np.mean(values)), "per_document": values}
        for arm in route_arms:
            rec = [(r["contrast_nominator"][i] if arm == "contrast_nominator" else r["arms"][arm][i])["source_top1"]
                   for r in held]
            layer[arm] = {"mean": float(np.mean([x["nominator"]["top_source_agreement"] for x in rec])),
                          "per_document": [x["nominator"]["top_source_agreement"] for x in rec],
                          "documents_beating_every_trivial_baseline": int(sum(x["beats_every_trivial_baseline"] for x in rec)),
                          "mean_margin_over_per_document_best_trivial": float(np.mean([x["margin_over_best_trivial"] for x in rec]))}
        best = max(layer["baselines"], key=lambda k: layer["baselines"][k]["mean"])
        layer["best_trivial_by_mean"] = best
        out["routes"][f"L{b}"] = layer
    # Paired comparisons.
    p = out["paired"]
    if "split_mean_write" in held[0]:
        for arm in learned:
            p[f"{arm} vs split_mean_write: output KL"] = paired(
                out["output_kl"][arm]["per_document"], out["output_kl"]["split_mean_write"]["per_document"])
            for b in boundaries[1:]:
                for m in ("relative_mse_excluding_position_0", "centered_relative_mse_excluding_position_0"):
                    p[f"{arm} vs split_mean_write: {m} L{b}"] = paired(
                        out["state"][arm][f"L{b}"][m]["per_document"],
                        out["state"]["split_mean_write"][f"L{b}"][m]["per_document"])
    for arm in [a for a in learned if a.startswith("sparse_reconstruction")]:
        for b in boundaries[1:-1]:
            p[f"{arm} vs mean_write: relative_mse_excluding_position_0 L{b}"] = paired(
                out["state"][arm][f"L{b}"]["relative_mse_excluding_position_0"]["per_document"],
                out["state"]["mean_write"][f"L{b}"]["relative_mse_excluding_position_0"]["per_document"])
    if "contrast_nominator" in held[0]:
        for i, b in enumerate(boundaries[:-1]):
            layer = out["routes"][f"L{b}"]
            best = layer["best_trivial_by_mean"]
            p[f"contrast_nominator vs {best} (best trivial by mean): top-source agreement L{b}"] = paired(
                layer["contrast_nominator"]["per_document"], layer["baselines"][best]["per_document"],
                lower_is_better=False)
    return out


def show(t):
    b = t["boundaries"]
    print(f"held-out documents: {len(t['documents'])}; boundaries {b}")
    print("\nfirst-token activation (position-0 share of native energy > 0.5):")
    for k, v in t["sink"].items():
        print(f"  {k}: {v['documents_above_threshold']}/{v['n']}")
    print("\noutput KL at last position (mean; top-1 agreement):")
    for arm, v in sorted(t["output_kl"].items(), key=lambda kv: kv[1]["mean"]):
        print(f"  {arm:34s} {v['mean']:9.3f}   top1 {t['top1'][arm]['agree']}/{t['top1'][arm]['n']}")
    print("\nrelative MSE, mean over documents (uncentred | centred | pos0-excluded | centred pos0-excluded):")
    for arm, layers in t["state"].items():
        cells = "  ".join(f"{k}: " + " ".join(f"{layers[k][m]['mean']:.4g}" for m in METRICS) for k in layers)
        print(f"  {arm:34s} {cells}")
    print("\ntop-source agreement per layer (mean over documents):")
    for layer, v in t["routes"].items():
        base = " ".join(f"{k}={x['mean']:.3f}" for k, x in v["baselines"].items())
        print(f"  {layer}: baselines {base}")
        for arm, x in v.items():
            if arm in ("baselines", "best_trivial_by_mean"):
                continue
            print(f"     {arm:34s} {x['mean']:.3f}  beats-all-trivial docs {x['documents_beating_every_trivial_baseline']}")
    print("\npaired comparisons (candidate - reference):")
    for name, v in t["paired"].items():
        print(f"  {name}: better {v['candidate_better']}/{v['n']}, mean {v['mean_difference']:.4g} "
              f"[{v['bootstrap_95'][0]:.4g}, {v['bootstrap_95'][1]:.4g}], median {v['median_difference']:.4g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json", type=Path, help="write the full tables here (must not exist)")
    args = parser.parse_args()
    result = tables(args.stage)
    show(result)
    if args.json:
        with args.json.open("x") as stream:
            json.dump(result, stream, indent=1, sort_keys=True)
