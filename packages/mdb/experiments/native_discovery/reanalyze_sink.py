"""Artifact-only re-analysis of a saved native-discovery run with the shared position-robust metrics.

For each held-out document and boundary: uncentred / centred relative MSE, both with position 0
excluded, and position 0's share of target energy, for every sealed forecast arm, the sealed
global mean-write baseline, and a position-split mean write (position 0 vs rest) fit here on the
run's discovery traces only. For each captured route layer: learned top-source agreement beside
the trivial baselines (position 0, offset 0 from end, previous token, uniform).

No model load, no forward, numpy only. Reads ``<stage>/evidence`` (the extracted run).

  python reanalyze_sink.py /path/to/artifacts/mdb-native-discovery-20261112/main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mdb.analysis_metrics import fit_mean_writes, mean_write_forecast, relative_error
from mdb.attention_ranking import evaluate_route_forecast
from mdb.forecast_transaction import load_forecast_tree
from mdb.native_discovery import cross_split_prefixes, read_arrays, verify_run

KEYS = ("relative_mse", "centered_relative_mse", "relative_mse_excluding_position_0",
        "centered_relative_mse_excluding_position_0", "position_0_energy_fraction")


def traces(root):
    out = {}
    for path in sorted((root / "traces").glob("*.json")):
        record = json.loads(path.read_text())
        out[record["document_id"]] = (record, read_arrays(root / "traces", record["arrays"]))
    return out


def reanalyze(stage):
    root = Path(stage) / "evidence"
    verify_run(root)
    captured = traces(root)
    discovery = [a["states"] for r, a in captured.values() if r["split"] == "discovery"]
    fitted = fit_mean_writes(discovery)
    boundaries = json.loads((root / "report.json").read_text())["boundaries"]
    documents = []
    for forecast in sorted(p for p in (root / "forecasts").iterdir() if p.is_dir()):
        observed = json.loads((forecast / "observations/ordinary_native.json").read_text())
        native = load_forecast_tree(forecast, observed["value"])
        document = native["trace"]["document_id"]
        states, routes = np.asarray(native["native_states"]), np.asarray(native["native_routes"])
        record, arrays = captured[document]
        if not np.array_equal(states, arrays["states"]):
            raise ValueError("sealed observation differs from saved trace")
        arms = {}
        for path in sorted((forecast / "forecasts").glob("*.json")):
            sealed = json.loads(path.read_text())
            arms[sealed["name"]] = load_forecast_tree(forecast, sealed["predictions"])
        state_rows, route_rows = {}, {}
        for name, prediction in arms.items():
            state_rows[name] = [{k: relative_error(s, states[i + 1])[k] for k in KEYS}
                                for i, s in enumerate(prediction["state"])]
            if "routes" in prediction:
                route_rows[name] = [evaluate_route_forecast(prediction["routes"][i], routes[i], name=name)
                                    for i in range(len(routes))]
        split = mean_write_forecast(states[0], fitted, split=True)
        state_rows["position_split_mean_write (post hoc, discovery-fit)"] = [
            {k: relative_error(split[i], states[i + 1])[k] for k in KEYS} for i in range(len(split))]
        documents.append({"document": document, "first_tokens": record["token_pieces"][:3],
                          "states": state_rows, "routes": route_rows})
    windows = [(r["split"], a["tokens"].tolist()) for r, a in captured.values()]
    shared = cross_split_prefixes(windows, refuse_at=10 ** 6)
    return {"boundaries": boundaries, "documents": documents,
            "longest_cross_split_shared_prefix": max(shared),
            "discovery_documents": len(discovery)}


def show(result):
    print(f"boundaries {result['boundaries']}; position-split means fit on "
          f"{result['discovery_documents']} discovery traces; longest cross-split shared token "
          f"prefix {result['longest_cross_split_shared_prefix']}")
    for doc in result["documents"]:
        print(f"\n== {doc['document']} first tokens {doc['first_tokens']}")
        print("   arm / boundary                     rel     centred  rel-ex0  cen-ex0  pos0-energy")
        for name, rows in doc["states"].items():
            for i, row in enumerate(rows):
                vals = " ".join(f"{row[k]:8.4f}" if row[k] is not None else "    None" for k in KEYS)
                print(f"   {name[:30]:30s} L{result['boundaries'][i + 1]:<3d} {vals}")
        print("   route top-source agreement at boundary layers (nominator | pos0 prev last uniform)")
        for name, rows in doc["routes"].items():
            cells = []
            for i, r in enumerate(rows):
                b = r["baselines"]
                cells.append(f"L{result['boundaries'][i]}: {r['nominator']['top_source_agreement']:.3f} | "
                             f"{b['position_0']['top_source_agreement']:.3f} "
                             f"{b['previous_token']['top_source_agreement']:.3f} "
                             f"{b['offset_0_from_end']['top_source_agreement']:.3f} "
                             f"{b['uniform']['top_source_agreement']:.3f}")
            print(f"   {name[:22]:22s} " + "   ".join(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json", type=Path, help="also write the full result here (must not exist)")
    args = parser.parse_args()
    result = reanalyze(args.stage)
    show(result)
    if args.json:
        with args.json.open("x") as stream:
            json.dump(result, stream, indent=1, sort_keys=True)
