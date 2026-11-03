"""Write the forwardless/timing qualification record from one collected run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mdb.compiled_state import verify_report

RESULT = "forwardless-result"
RULES = {"agreement": "match when |mdb - earlier| <= 0.005 (same 32,768-row sample, n_err ~700)",
         "error_recall": "match when |mdb - earlier| <= 0.05",
         "speedup": "match when mdb / earlier is within [0.8, 1.25]",
         "milliseconds": "match when mdb / earlier is within [0.8, 1.25]",
         "exact_through": "match when every step equals the earlier value"}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(part, claim, earlier, measured, kind, **extra):
    if earlier is None or measured is None:
        match = None
    elif kind in ("speedup", "milliseconds"):
        match = 0.8 <= measured / earlier <= 1.25
    elif kind == "exact_through":
        match = list(measured) == list(earlier)
    else:
        match = abs(measured - earlier) <= (0.005 if kind == "agreement" else 0.05)
    return {"part": part, "claim": claim, "kind": kind, "earlier": earlier, "mdb": measured, "match": match, **extra}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if v < 2 else f"{v:.2f}"
    return json.dumps(v) if isinstance(v, list) else str(v)


def record(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "report.json").read_text())  # collector's copy of the worker summary
    base = run_dir / "extracted" / RESULT
    report = verify_report(json.loads((base / "report.json").read_text()))
    qual_path = base / "qualification.json"
    q = json.loads(qual_path.read_text())
    if (file_sha(qual_path) != summary["qualification_sha256"] or report["fingerprint"] != summary["report_fingerprint"]
            or q["addition"]["report_sha256"] != file_sha(base / "report.json")):
        raise ValueError("collected files differ from the worker summary")
    job = json.loads((run_dir / "job.json").read_text())
    cmp = q["config"]["comparison"]
    fs = report["feature_sets"]
    rows = []
    for name, (agree, recall) in cmp["part_e"]["agreement"].items():
        f, L = name[:-4], name[-3:-1]
        a = fs.get(f, {}).get("layers", {}).get(L, {}).get("agreement")
        if a:
            rows.append(_row("E", f"{name} answer agreement", agree, a["row_agreement"], "agreement", n=a["n"]))
            rows.append(_row("E", f"{name} error recall", recall, a.get("error_recall"), "error_recall",
                             n_native_wrong=a.get("n_native_wrong")))
    part = fs.get("F4-+pairs", {}).get("partial", {}).get("20")
    if part:
        for label, key, (agree, recall) in (("T3c(20)", "compiled_kv", cmp["part_f"]["agreement"]["T3c(20) sweep 32768"]),
                                            ("T3c-nativeKV(20)", "native_kv_oracle",
                                             cmp["part_f"]["agreement"]["T3c-nativeKV(20) sweep 32768"])):
            rows.append(_row("F", f"{label} answer agreement", agree, part[key]["row_agreement"], "agreement",
                             n=part[key]["n"]))
            rows.append(_row("F", f"{label} error recall", recall, part[key].get("error_recall"), "error_recall",
                             n_native_wrong=part[key].get("n_native_wrong")))
    timing = q["addition"]["timing"]
    for b in ("1", "256"):
        arms = timing.get(b, {}).get("forward", {}).get("arms", {})
        for name, earlier in cmp["part_e"]["speedup"].get(b, {}).items():
            got = arms.get(name, {})
            rows.append(_row("E", f"{name} speedup at batch {b}", earlier, got.get("speedup_vs_baseline"), "speedup",
                             mdb_ms=got.get("median_ms"), earlier_ms=cmp["part_e"]["timing_ms"][b].get(name)))
        for name, earlier in cmp["part_f"]["speedup"].get(b, {}).items():
            got = arms.get(name.replace("T3c(", "T3c-F4-+pairs("), {})
            rows.append(_row("F", f"{name} speedup at batch {b}", earlier, got.get("speedup_vs_baseline"), "speedup",
                             mdb_ms=got.get("median_ms"), earlier_ms=cmp["part_f"]["timing_ms"][b].get(name)))
        native = arms.get("native_prefix_cached", {}).get("median_ms")
        rows.append(_row("E", f"native prefix-cached forward at batch {b} (ms)", cmp["part_e"]["timing_ms"][b]
                         ["native_prefix_cached"], native, "milliseconds"))
    g1 = q["g1"]
    seen = g1["chain"]["SEEN:full_vocab"]
    rows.append(_row("G", "G1 4-token zero-layer chain exact_through (SEEN)", cmp["part_g"]["chain"]["exact_through"],
                     seen["exact_through"], "exact_through", n=seen["n"]))
    for b in ("1", "256"):
        serving = g1["timing"].get(b, {}).get("serving", {}).get("arms", {})
        rows.append(_row("G", f"G1 chain vs KV-cached native decode at batch {b}",
                         cmp["part_g"]["speedup"][b]["predict_chain_L28"],
                         serving.get("chain_L28", {}).get("speedup_vs_baseline"), "speedup",
                         mdb_ms=serving.get("chain_L28", {}).get("median_ms"),
                         earlier_ms=cmp["part_g"]["timing_ms"][b]["predict_chain_L28"],
                         mdb_baseline_ms=serving.get("native_kv_decode_4", {}).get("median_ms"),
                         earlier_baseline_ms=cmp["part_g"]["timing_ms"][b]["native_kv_decode"]))
    table = ["| Part | Claim | Earlier | mdb | Match |", "|---|---|---|---|---|"]
    for r in rows:
        table.append(f"| {r['part']} | {r['claim']} | {_fmt(r['earlier'])} | {_fmt(r['mdb'])} | "
                     f"{'n/a' if r['match'] is None else ('yes' if r['match'] else 'NO')} |")
    serving = {"addition": [{k: s[k] for k in ("tau", "native_prompts", "served", "forwards_per_1000_prompts", "refits")}
                            | {"served_exact": s["audit"]["served_exact"], "predicted_exact": s["audit"]["predicted_exact"]}
                            for s in q["addition"]["serving"]],
               "g1": [{k: s[k] for k in ("tau", "native_prompts", "served", "forwards_per_1000_prompts", "refits")}
                      | {"served_exact": s["audit"]["served_exact"], "predicted_exact": s["audit"]["predicted_exact"]}
                      for s in g1["serving"]]}
    gates = {"addition_measure": report["parity"]["gate"], "addition_ladder": q["addition"]["parity_ladder"]["gate"],
             "g1_ladder": g1["parity"]["gate"],
             "max_abs_dlogit": {"addition": max(c["max_abs_dlogit"] for k in ("resume", "kv_injection", "partial_vs_hook")
                                                for c in q["addition"]["parity_ladder"][k].values()),
                                "addition_manual_vs_hf": q["addition"]["parity_ladder"]["manual_native_vs_hf"],
                                "g1": max(c["max_abs_dlogit"] for k in ("resume", "kv_injection", "partial_vs_hook")
                                          for c in g1["parity"][k].values()),
                                "g1_manual_vs_hf": g1["parity"]["manual_native_vs_hf"]}}
    return {"schema": "mdb-forwardless-qualification-record-v1", "job_id": summary["job_id"],
            "job_state": job.get("state"), "request_sha256": summary["request_sha256"],
            "report_fingerprint": report["fingerprint"], "qualification_sha256": summary["qualification_sha256"],
            "model": report["model"], "execution": report["execution"], "environment": q["environment"],
            "rows": q["addition"]["rows"], "g1_setup": g1["setup"], "gates": gates, "rules": RULES,
            "comparison": rows, "chains": {"addition": q["addition"]["chain"], "g1": g1["chain"]},
            "serving": serving, "comparison_sources": {k: v["source"] for k, v in cmp.items()},
            "limits": q["config"]["limits"], "elapsed_s": q["elapsed_s"], "table": "\n".join(table)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = record(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(value["table"])


if __name__ == "__main__":
    main()
