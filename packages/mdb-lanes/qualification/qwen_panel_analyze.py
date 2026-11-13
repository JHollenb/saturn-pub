"""Analyse pulled Qwen panel runs (artifact-only; numpy; no model).

  python qwen_panel_analyze.py --plans PLANS_DIR --out TABLES.json RUN_DIR [RUN_DIR ...]

RUN_DIR: a pulled ``<job_id>`` directory holding ``qwen-panels/<lane>/<plan>[-leaseN]/rows.jsonl``.
Writes one JSON with tables (a)-(e) and prints them:
(a) BF16 value arms vs the saved main-02 run (first token, margin);
(b) FP32 margins, every value-coalition coefficient with its readout-rounding bound, the L20 pair
    fit, beside the saved BF16 coefficient (resolved / unresolved, sign change, vanished);
(c) max source attention per cell vs singleton effect and complement loss (Pearson, Spearman, LOO);
(d) read cuts on the first receiver: per head set, cut/keep x cut type shift, floor, controls, verdict;
(e) backend floors (recompute sham, delta-vs-recompute gap) per lane, plus value sham / replay.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path[0:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1].parent / "mdb" / "src")]
from mdb_lanes import attention_panels as P  # noqa: E402
from mdb_lanes import qwen_panels as Q  # noqa: E402

MAIN02 = Path("/Users/jakeholl/domains/artifacts/mdb-value-coalition-20261112/main-02")


def rows_of(runs):
    """{(lane, plan name): [rows]} with lease copies separated and checked against primaries."""
    out, copies = {}, {}
    attention = {}
    for run in runs:
        base = Path(run) / "qwen-panels"
        for lane_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            lane = lane_dir.name
            att = lane_dir / "receiver-attention.json"
            if att.exists():
                attention.setdefault(lane, json.loads(att.read_text())["attention"])
            for plan_dir in sorted(p for p in lane_dir.iterdir() if p.is_dir()):
                name = plan_dir.name.split("-lease")[0]
                for line in (plan_dir / "rows.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    row["_job"] = Path(run).name
                    (copies if row.get("replicated") else out).setdefault((lane, name), []).append(row)
    checks = []
    for key, rows in copies.items():
        primary = {r["arm"]: r for r in out.get(key, [])}
        for r in rows:
            p = primary.get(r["arm"])
            if p is not None:
                checks.append({"lane": key[0], "plan": key[1], "arm": r["arm"], "jobs": [p["_job"], r["_job"]],
                               "margin_equal": p["margin"] == r["margin"], "tokens_equal": p["tokens"] == r["tokens"]})
    for key, rows in out.items():
        names = [r["arm"] for r in rows]
        if len(names) != len(set(names)):
            raise ValueError(f"{key}: an arm ran twice as a primary")
    return out, checks, attention


def coefficient_table(fp32, bf16, saved):
    table = []
    for name in fp32:
        f, b, s = fp32[name], bf16.get(name), saved.get(name)
        row = {"coefficient": name, "fp32": f.value, "fp32_bound": f.bound, "fp32_resolved": f.resolved}
        for tag, v in (("bf16_rerun", b), ("saved_bf16", s)):
            if v is not None:
                row.update({tag: v.value, tag + "_bound": v.bound, tag + "_resolved": v.resolved})
        if s is not None:
            row["sign_change_vs_saved"] = (f.value > 0) != (s.value > 0) and f.resolved and s.value != 0
            row["saved_reported_but_unresolved"] = not s.resolved
            row["fp32_vanishes"] = not f.resolved
        table.append(row)
    return table


def analyse(plans_dir, runs):
    plans_dir = Path(plans_dir)
    rows, copy_checks, attention = rows_of(runs)
    out = {"runs": [str(r) for r in runs], "lease_copy_checks": {
        "n": len(copy_checks), "all_equal": all(c["margin_equal"] and c["tokens_equal"] for c in copy_checks),
        "unequal": [c for c in copy_checks if not (c["margin_equal"] and c["tokens_equal"])]}}
    saved_arms = json.loads((MAIN02 / "analysis/arms.json").read_text())
    margins = {}
    for lane in Q.LANES:
        if (lane, f"value-{lane}") in rows:
            value_rows = rows[(lane, f"value-{lane}")]
            produced = value_rows[0]["logits_dtype"]
            margins[lane] = Q.value_margins(value_rows, produced)
            out.setdefault("value_rows", {})[lane] = [
                {k: r[k] for k in ("arm", "margin", "margin_bound", "tokens_text", "tracked")} for r in value_rows]
    # (a)
    if "bfloat16" in margins:
        out["a_bf16_vs_saved"] = Q.saved_agreement(saved_arms, rows[("bfloat16", "value-bfloat16")])
    # (b)
    saved_rows = [{"arm": a.arm, "tracked": {w: next(s for s in saved_arms if s["arm"] == Q.legacy_name(a.arm))
                                             ["words"][w]["logit"] for w in ("hot", "cold")}}
                  for a in Q.value_plan("bfloat16").arms]
    saved_m = Q.value_margins(saved_rows, "bfloat16")
    coeffs = {lane: Q.coefficients(m) for lane, m in margins.items() if len(m) == 56}
    coeffs["saved"] = Q.coefficients(saved_m)
    if "float32" in coeffs:
        out["b_coefficients"] = coefficient_table(coeffs["float32"], coeffs.get("bfloat16", {}), coeffs["saved"])
        fits = {lane: Q.l20_fit(margins[lane]) for lane in margins if len(margins[lane]) == 56}
        fits["saved"] = Q.l20_fit(saved_m)
        out["b_l20_fit"] = {lane: {k: v.as_dict() for k, v in f.items()} for lane, f in fits.items()}
        out["b_fp32_margins"] = {a: {"margin": m.value, "bound": m.bound} for a, m in margins["float32"].items()}
        # (c)
        cells = Q.max_source_attention(attention["float32"])
        out["c_correlations_fp32"] = {"max_source_attention": cells, **Q.cell_correlations(cells, coeffs["float32"])}
    # (d) read cuts
    read_key = ("float32", "read-float32")
    if read_key in rows:
        panel = P.Panel.from_json(json.loads((plans_dir / "read-float32.plan.json").read_text()))
        rrows = rows[read_key]
        missing = sorted({a.arm for a in panel.arms} - {r["arm"] for r in rrows})
        out["d_missing_arms"] = missing
        results = Q.read_results(panel, rrows)
        verdicts = P.verdicts(panel, results)
        effects = {(e["arm"]): e for e in P.arm_effects(panel, results)}
        by = {r["arm"]: r for r in rrows}
        native = by["native"]["margin"]
        table = []
        for v in verdicts:
            for t, pt in v["cut_types"].items():
                arm = pt["arm"]
                ctx = pt["contexts"][0]
                controls = [{"control": c["control"], "kind": c["kind"], "shift": by[c["control"]]["margin"] - native,
                             "beyond_floor": c.get("beyond_floor"), "refutes": c.get("refutes"),
                             "mass_relation": c.get("mass_relation")} for c in ctx.get("controls", [])]
                refs = [{"control": c["control"], "kind": c["kind"], "shift": by[c["control"]]["margin"] - native}
                        for c in ctx.get("references", [])]
                table.append({"item": v["item"], "unit_set": v["unit_set"], "claim": v["claim"], "cut_type": t,
                              "arm": arm, "shift": by[arm]["margin"] - native, "tokens": by[arm]["tokens_text"],
                              "floor": effects[arm]["floor"], "floor_parts": effects[arm]["floor_parts"],
                              "all_cut_shift": (by[f"{P.PREFIX[t]}:none:{v['item']}"]["margin"] - native),
                              "recovery": ctx.get("recovery"), "controls": controls, "references": refs,
                              "cut_type_status": pt["status"], "status": v["status"]})
        out["d_read_table"] = table
        out["d_verdicts"] = verdicts
        out["d_native"] = {"margin": native, "tokens": by["native"]["tokens_text"]}
        label = plans_dir / "label-free-ranking.json"
        if label.exists():
            out["d_label_free"] = json.loads(label.read_text())["summary"]
    # (e) floors
    floors = {}
    for (lane, name), rr in rows.items():
        by = {r["arm"]: r for r in rr}
        if name.startswith("read"):
            sham = [abs(r["margin"] - by[r["reference"]]["margin"]) for r in rr
                    if r["kind"] in ("floor_sham_recompute", "floor_sham") and r["reference"] in by]
            sham_logit = [r["to_reference"]["max_abs_logit"] for r in rr
                          if r["kind"] in ("floor_sham_recompute", "floor_sham") and "to_reference" in r]
            gaps = [abs(r["margin"] - by[r["floor_of"]]["margin"]) for r in rr
                    if r["kind"] == "floor_cut_recompute" and r["floor_of"] in by]

            def stats(xs):
                return None if not xs else {"n": len(xs), "max": max(xs), "median": statistics.median(xs),
                                            "zero": sum(x == 0 for x in xs)}
            floors[f"{lane}:{name}"] = {"recompute_sham_margin": stats(sham),
                                        "recompute_sham_max_abs_logit": stats(sham_logit),
                                        "delta_vs_recompute_margin_gap": stats(gaps)}
        else:
            ref = by.get("native-recipient")
            floors[f"{lane}:{name}"] = {k: None if k not in by else {
                "margin_shift": by[k]["margin"] - ref["margin"],
                "max_abs_logit": by[k].get("to_reference", {}).get("max_abs_logit")} for k in ("sham-late", "replay-recipient")}
    out["e_floors"] = floors
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    result = analyse(args.plans, args.runs)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("lease_copy_checks", "e_floors", "d_missing_arms")},
                     indent=1, default=str))
