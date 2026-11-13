"""Pearson, Spearman and leave-one-out range for the value-coalition attention/effect figures.

Reproduces the descriptive correlations in ``analysis/arithmetic-diagnostics.json`` (computed by
``arithmetic-diagnostics.py`` with the definition in ``value_coalition_render.py``): for each of
the 18 late layer/KV cells, the largest native-recipient source attention among the six query
heads sharing that KV group, against the singleton effect and against the complement loss.
The Pearson figure alone is reported there; this prints it with Spearman and the range over
dropping each cell. The 18 cells share one context and one downstream network, so they are not
independent replications. Artifact-only, numpy.

  python value_coalition_correlation.py /path/to/mdb-value-coalition-20261112/main-02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[0:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1].parent / "mdb" / "src")]
from mdb_lanes.analysis_metrics import correlation_report  # noqa: E402


def main(run):
    run = Path(run)
    trends = json.loads((run / "analysis/trends.json").read_text())
    heads = json.loads((run / "analysis/head-uses.json").read_text())
    cells = trends["cells"]
    labels = [f"L{c['layer']}/KV{c['kv_head']}" for c in cells]
    mass = [max(h["source_mass"] for h in heads if h["arm"] == "native-recipient"
                and h["layer"] == c["layer"] and h["kv_head"] == c["kv_head"]) for c in cells]
    saved = json.loads((run / "analysis/arithmetic-diagnostics.json").read_text())["descriptive_cell_correlations"]
    out = {}
    for key, saved_key in (("singleton_effect", "max_native_attention_vs_singleton"),
                           ("complement_loss", "max_native_attention_vs_complement_loss")):
        report = correlation_report(mass, [c[key] for c in cells], labels=labels)
        if abs(report["pearson"] - saved[saved_key]) > 1e-12:
            raise ValueError("Pearson recomputation differs from the saved diagnostic: " + key)
        loo = report["leave_one_out"]
        drop = {r["dropped"]: r for r in loo["rows"]}
        print(f"max native source attention vs {key} (n={report['n']})")
        print(f"  Pearson {report['pearson']:.3f} (saved {saved[saved_key]:.3f})   Spearman {report['spearman']:.3f}")
        print(f"  leave-one-out Pearson {loo['pearson']['min']:.3f} (drop {loo['pearson']['min_dropped']})"
              f" .. {loo['pearson']['max']:.3f} (drop {loo['pearson']['max_dropped']})")
        print(f"  leave-one-out Spearman {loo['spearman']['min']:.3f} (drop {loo['spearman']['min_dropped']})"
              f" .. {loo['spearman']['max']:.3f} (drop {loo['spearman']['max_dropped']})")
        print(f"  without L20/KV0: Pearson {drop['L20/KV0']['pearson']:.3f}, Spearman {drop['L20/KV0']['spearman']:.3f}")
        out[key] = report
    print("Cells share one context and network; these are descriptive, not independent replications.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    main(**vars(parser.parse_args()))
