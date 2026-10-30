"""Publish compact, source-bound results after the frozen route forecast test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(analysis_path, output, index_path, figure_prefix=None):
    analysis = json.loads(analysis_path.read_text())
    cases = []
    for result in analysis["results"]:
        arms = {}
        for name, arm in result["arms"].items():
            continuation = arm["continuation"]
            arms[name] = {"first": continuation["first_decode"],
                "second": continuation["second_decode"],
                "first_forecast": arm["first_register_forecast"],
                "second_forecast": arm["second_register_forecast"],
                "native_relative_registers": arm["native_relative_registers"],
                "own_state_local_fixture": continuation.get("own_state_local_fixture"),
                "successor_roots_status": continuation.get("successor_input_roots", {}).get("status")}
        cases.append({"case_id": result["case_id"], "supplied_family": result["family"],
            "reader": result["reader"], "arms": arms, "signed_effects": result["signed_effects"],
            "route_ranking": result["route_ranking"], "native_correctness": result["native_correctness"]})
    site_arms = ("post0_zero", "post1_zero", "post2_zero")
    metrics = {
        "post0_prior_registers": [sum(c["arms"]["post0_zero"]["first_forecast"]["exact"] for c in cases), 6],
        "later_zero_native_registers": [sum(c["arms"][a]["native_relative_registers"]["exact"]
            for c in cases for a in site_arms[1:]), 12],
        "U_restore_native_registers": [sum(c["arms"]["projection_native"]["native_relative_registers"]["exact"] for c in cases), 6],
        "U_restore_native_RDI": [sum(c["arms"]["projection_native"]["first"]["register_words"][7]
            == c["arms"]["identity"]["first"]["register_words"][7] for c in cases), 6],
        "U_closer_than_S_at_reader": [sum(c["route_ranking"]["matches"] for c in cases), 6],
        "post0_effect_sign": [sum(c["signed_effects"]["post0_zero_minus_identity"]["matches"] for c in cases), 6],
        "U_effect_sign": [sum(c["signed_effects"]["projection_native_minus_post0_zero"]["matches"] for c in cases), 6],
        "site_full_register_forecasts": [sum(c["arms"][a]["first_forecast"]["exact"] for c in cases for a in site_arms), 18],
        "no_effect_site_baseline": [sum(r["status"] == "supported" for r in analysis[
            "comparison_baseline_counts"]["no_effect_at_any_post"]["outcomes"]), 18],
        "own_state_local_successor_registers": [sum(c["arms"][a]["own_state_local_fixture"]["exact"]
            for c in cases for a in c["arms"]), 48],
    }
    summary = {"schema": "mdb-predictive-route-summary-v1", "analysis_sha256": sha(analysis_path),
        "source": analysis["source"], "execution": analysis["execution"],
        "mechanics_status": analysis["mechanics_status"], "instrument_status": analysis["instrument_status"],
        "trend_status": "bounded-transfer-with-counterexamples", "terminal_status": "unestablished",
        "metrics": metrics, "cases": cases, "counterexamples": analysis["counterexamples"],
        "counterexample_count_scope": "17 forecast rows include repeated no-op and horizon outcomes; not independent discoveries",
        "pattern_updates": [
            {"id": "pattern_0001", "status": "scope-expansion-contradicted",
             "observation": "Prior-state retention extends to these SUB fixtures; XOR corrupts extra bits and MOV retains native decisions"},
            {"id": "route_0002", "status": "split-consumer-dependent-support",
             "observation": "Target RDI recovery6/6, full-bank recovery5/6, reader-distance ranking4/6; retain separate applicability"},
            {"id": "pattern_0003", "status": "supported-within-six-fixtures",
             "observation": "Both later post deletions preserve native register decisions; continuous writes still change"}],
        "limits": analysis["limits"], "supplied": analysis["supplied"],
        "empirical_scope": analysis["empirical_old_evidence_scope"],
        "prospective_scope": analysis["proposed_applicability"]}
    for path, value in ((output, summary), (index_path, analysis["generic_record_index"])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    if figure_prefix:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
        positions = list(range(len(cases)))
        for offset, arm, label, color in ((-.18, "projection_native", "Restore projected write U", "#28649d"),
                                         (.18, "skip_native", "Restore skip state S", "#cf793a")):
            values = [c["route_ranking"]["distances_from_native"][arm] for c in cases]
            bars = axis.bar([p + offset for p in positions], values, width=.34, label=label, color=color)
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_xticks(positions, [c["case_id"] + (" *" if not c["route_ranking"]["matches"] else "")
            + "\nRDI bit " + str(c["reader"]["bit"]) for c in cases])
        axis.set_ylabel("Absolute reader-logit distance from native (lower is closer)")
        axis.set_title("Frozen forecast: restoring U is closer than restoring S\nObserved in 4/6 cases; * marks a counterexample", loc="left")
        axis.set_ylim(0, 27)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper right")
        for extension in ("png", "pdf"):
            figure.savefig(figure_prefix.with_suffix("." + extension), dpi=180)
        plt.close(figure)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.analysis, args.output, args.index, args.figure_prefix), sort_keys=True))


if __name__ == "__main__":
    main()
