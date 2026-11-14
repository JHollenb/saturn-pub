"""Recompute FLUX and Qwen read-panel verdicts from SAVED results with the 2026-11-13 verdict code.

Artifact-only (numpy + json; no model, no torch, no job). Writes NEW files only; the sealed plans,
the saved results and the earlier reports are read, never rewritten.

  PY=/Users/jakeholl/domains/saturn-pub/.venv/bin/python
  PYTHONPATH=<wt>/packages/mdb-lanes/src:<wt>/packages/mdb/src:<wt>/src $PY panel_verdicts_recompute_20261113.py [--out DIR]

--out DIR writes the four outputs under DIR (same relative names) instead of in place, to re-check them.

Verdict logic versions compared (each is a module file at a git commit, run on the same results):
  sealed  = attention_panels.py at 5e09d95 (sha256 31a8595b...), the module the FLUX plan was sealed with
  prefix  = attention_panels.py at 4bbe3e7 (sha256 4a39dbaa...), the branch's logic before this fix pass
            (first recorded as 166b334 at packages/mdb/src/mdb/; byte-identical, moved to mdb_lanes)
  fixed   = the working tree's attention_panels.py (this fix pass)
Outputs:
  qualification/flux_read_panels_20261112/report/e-verdicts-fixed-20261113.json   (all 78 records, fixed logic)
  qualification/flux_read_panels_20261112/report/e-summary-20261113.json         (counts per logic + per record)
  qualification/flux_read_panels_20261112/report/e-report-20261113.txt           (the tables)
  qualification/qwen_panels_20261112/full-verdicts-20261113.json                 (Qwen read verdicts + BF16 table)
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # the checkout (worktree) root
sys.path.insert(0, str(HERE))
from mdb_lanes import attention_panels as P  # noqa: E402
from mdb_lanes import qwen_panels as Q  # noqa: E402

FLUX_Q = HERE / "flux_read_panels_20261112"
QWEN_Q = HERE / "qwen_panels_20261112"
QWEN_RUNS = [Path("/Users/jakeholl/domains/artifacts/mdb-qwen-panels-20261112") / j
             for j in ("job-b33e6b98c328", "job-4a433df3f30b", "job-7c3232ad65bb")]
# (commit, path of attention_panels.py at that commit); the module moved to mdb-lanes in 9ec645b
COMMITS = {"sealed": ("5e09d95", "packages/mdb/src/mdb/attention_panels.py"),
           "prefix": ("4bbe3e7", "packages/mdb-lanes/src/mdb_lanes/attention_panels.py")}
FIXED = ROOT / "packages/mdb-lanes/src/mdb_lanes/attention_panels.py"
OUT = None                   # --out: write outputs under this directory instead of in place
NO_THRESHOLD = 1e-9          # "no recovery threshold" for the threshold-dependence check


def module_at(commit, tag):
    """attention_panels.py at ``commit`` loaded as mdb_lanes.<tag> (relative imports resolve to the current
    mdb_lanes)."""
    commit, source_path = commit
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{source_path}"],
                         capture_output=True, text=True, check=True).stdout
    path = Path(__file__).with_name(f".{tag}_attention_panels.py")
    spec = importlib.util.spec_from_loader(f"mdb_lanes._{tag}_attention_panels", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "mdb_lanes"
    sys.modules[spec.name] = mod                       # dataclasses look their module up here
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode()).hexdigest()


def save(path, value):
    path = Path(path)
    if OUT is not None:
        path = OUT / path.relative_to(HERE)
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; outputs of this script are new files (never overwritten)")
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + "\n")


def counts(records):
    return dict(sorted(collections.Counter(r["status"] for r in records).items()))


def key(r):
    return f"{r['item']} {r['unit_set']}"


def threshold_note(status, without):
    """Wording for a keep record that is not_supported only because of the post-hoc recovery threshold."""
    cap = f"at most {without}" if without in ("supported", "set_only") else without
    return f"{status} only via the post-hoc 0.7 recovery threshold; without it: {cap}"


def relabel_counts(panel, records):
    """Counts of (stored label -> relabelled) over the random controls of the zero-cut test arms."""
    seen, out = set(), collections.Counter()
    for r in records:
        t = r["cut_types"].get("zero")
        for c in (t or {}).get("contexts", [{}])[0].get("controls", []):
            if c["kind"] == "random" and c["control"] not in seen:
                seen.add(c["control"])
                out[f"{c.get('mass_relation_stored')} -> {c.get('mass_relation')}"] += 1
    return dict(sorted(out.items()))


# ============================================================================================ FLUX

def flux():
    import flux_read_panel_stage as stage            # read-only use: word_masses() (seed-4242 capture)
    plan = json.loads((FLUX_Q / "plans/full.plan.json").read_text())
    seal = json.loads((FLUX_Q / "plans/full.seal.json").read_text())
    results = json.loads((FLUX_Q / "report/d-results-px.json").read_text())
    masses = {w: dict(zip(stage.universe(), vec)) for w, vec in stage.word_masses().items()}   # {word: {unit: mass}}
    out, logic = {}, {}
    for tag in ("sealed", "prefix"):
        mod, digest = module_at(COMMITS[tag], tag)
        panel = mod.Panel.from_json(plan)
        assert panel.sha256() == seal["plan_sha256"]
        out[tag] = mod.verdicts(panel, results)
        logic[tag] = {"commit": COMMITS[tag][0], "module_sha256": digest}
    panel = P.Panel.from_json(plan)
    assert panel.sha256() == seal["plan_sha256"], "sealed plan hash not reproduced"
    fixed = P.verdicts(panel, results, item_mass=masses)
    flagged = P.verdicts(panel, results, item_mass=masses, exclude_named_draws=False)
    loose = P.verdicts(panel, results, item_mass=masses, recovery_threshold=NO_THRESHOLD)
    loose_flag = P.verdicts(panel, results, item_mass=masses, recovery_threshold=NO_THRESHOLD,
                            exclude_named_draws=False)
    nobar = P.verdicts(panel, results, item_mass=masses, min_effect_fraction=0.0)
    saved_sealed = json.loads((FLUX_Q / "report/d-verdicts-all.json").read_text())
    assert [r["status"] for r in saved_sealed] == [r["status"] for r in out["sealed"]], \
        "sealed-logic rerun differs from report/d-verdicts-all.json"
    logic["fixed"] = {"module_sha256": hashlib.sha256(FIXED.read_bytes()).hexdigest()}
    by = {name: {key(r): r for r in recs} for name, recs in
          (("sealed", out["sealed"]), ("prefix", out["prefix"]), ("fixed", fixed), ("flag", flagged),
           ("loose", loose), ("loose_flag", loose_flag), ("nobar", nobar))}
    rows = []
    for r in fixed:
        k = key(r)
        dep = (r["claim"] == "sufficient" and "low_recovery" in r["findings"] and
               by["loose"][k]["status"] != r["status"])
        dep_flag = (r["claim"] == "sufficient" and "low_recovery" in by["flag"][k]["findings"] and
                    by["loose_flag"][k]["status"] != by["flag"][k]["status"])
        ef = r["effect_fraction"]
        rows.append({"record": k, "claim": r["claim"], "sealed": by["sealed"][k]["status"],
                     "prefix": by["prefix"][k]["status"], "fixed": r["status"],
                     "fixed_named_check_flag_only": by["flag"][k]["status"],
                     "reported": threshold_note(r["status"], by["loose"][k]["status"]) if dep else r["status"],
                     "reported_named_check_flag_only": (threshold_note(by["flag"][k]["status"],
                                                                       by["loose_flag"][k]["status"])
                                                        if dep_flag else by["flag"][k]["status"]),
                     "depends_on_post_hoc_recovery_threshold": dep or dep_flag,
                     "status_without_min_effect_bar": by["nobar"][k]["status"],
                     "effect_fraction_min": ef["min"], "effect_fraction_max": ef["max"],
                     "findings": r["findings"], "ruled_out": r["ruled_out"],
                     "remaining": r["remaining_alternatives"],
                     "excluded_controls": sorted({c for t in r["cut_types"].values()
                                                  for c in t["excluded_controls"]})})
    summary = {"plan_sha256": seal["plan_sha256"], "records": len(fixed), "logic": logic,
               "counts": {"sealed": counts(out["sealed"]), "prefix": counts(out["prefix"]), "fixed": counts(fixed),
                          "fixed_named_check_flag_only": counts(flagged)},
               "post_hoc_recovery_threshold": {
                   "value": P.DEFAULT_RECOVERY, "in_sealed_plan": False,
                   "records_whose_verdict_depends_on_it_fixed": [x["record"] for x in rows
                                                                 if x["reported"] != x["fixed"]],
                   "records_whose_verdict_depends_on_it_named_flag_only": [
                       x["record"] for x in rows if x["reported_named_check_flag_only"] != x["fixed_named_check_flag_only"]]},
               "min_effect_fraction": {"value": P.DEFAULT_MIN_EFFECT_FRACTION, "in_sealed_plan": False,
                                       "records_whose_verdict_depends_on_it": [
                                           x["record"] for x in rows if x["status_without_min_effect_bar"] != x["fixed"]]},
               "random_mass_labels_stored_vs_relabelled": relabel_counts(panel, fixed),
               "item_mass": "each word's own image<-row write-mass share per (block, head), full-a capture seed 4242 "
                            "(flux_read_panel_stage.word_masses); also the mass the sealed random draws were matched on, "
                            "applied to both contexts (9001 mass not measured)",
               "rows": rows}
    save(FLUX_Q / "report/e-verdicts-fixed-20261113.json", fixed)
    save(FLUX_Q / "report/e-summary-20261113.json", summary)
    lines = ["FLUX read panels: verdicts recomputed from report/d-results-px.json (no new runs)",
             f"plan {seal['plan_sha256'][:12]}, {len(fixed)} records; logic: " +
             ", ".join(f"{k} {v.get('commit', 'worktree')} ({v['module_sha256'][:8]})" for k, v in logic.items()),
             "", "counts:"]
    for k, v in summary["counts"].items():
        lines.append(f"  {k:<30} {v}")
    lines += ["", f"{'record':<28} {'claim':<10} {'sealed':<14} {'prefix':<14} {'fixed':<15} {'fixed,flag-only':<15} "
              f"{'eff.frac min..max':<17} reported (fixed)"]
    for x in rows:
        ef = ("-" if x["effect_fraction_min"] is None else
              f"{x['effect_fraction_min']:.3f}..{x['effect_fraction_max']:.3f}")
        lines.append(f"{x['record']:<28} {x['claim']:<10} {x['sealed']:<14} {x['prefix']:<14} {x['fixed']:<15} "
                     f"{x['fixed_named_check_flag_only']:<15} {ef:<17} {x['reported']}")
    report_txt = FLUX_Q / "report/e-report-20261113.txt"
    if OUT is not None:
        report_txt = OUT / report_txt.relative_to(HERE)
    report_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    print(json.dumps({k: summary[k] for k in ("post_hoc_recovery_threshold", "min_effect_fraction",
                                              "random_mass_labels_stored_vs_relabelled")}, indent=1))
    return summary


# ============================================================================================ Qwen

def qwen():
    import qwen_panel_analyze as A                   # read-only use: rows_of, coefficient_table
    rows, _, attention = A.rows_of(QWEN_RUNS)
    plan = json.loads((QWEN_Q / "full/read-float32.plan.json").read_text())
    seal = json.loads((QWEN_Q / "full/read-float32.seal.json").read_text())
    panel = P.Panel.from_json(plan)
    assert panel.sha256() == seal["plan_sha256"], "sealed Qwen plan hash not reproduced"
    rrows = rows[("float32", "read-float32")]
    results = Q.read_results(panel, rrows)
    before = json.loads((QWEN_Q / "full-tables.json").read_text())["d_verdicts"]
    prefix_mod, prefix_sha = module_at(COMMITS["prefix"], "prefix")
    prefix = prefix_mod.verdicts(prefix_mod.Panel.from_json(plan), results)
    assert [r["status"] for r in prefix] == [r["status"] for r in before], "prefix rerun differs from full-tables.json"
    fixed = P.verdicts(panel, results)               # other-item masses come from the stored per-item reports
    flagged = P.verdicts(panel, results, exclude_named_draws=False)
    nobar = {key(r): r["status"] for r in P.verdicts(panel, results, min_effect_fraction=0.0)}
    by = {r["arm"]: r for r in rrows}
    native = by["native"]["margin"]
    table = []
    for b, v, f in zip(before, fixed, flagged):
        assert key(b) == key(v) == key(f)
        per = {}
        for t, pt in v["cut_types"].items():
            ctx = pt["contexts"][0]
            per[t] = {"shift": by[pt["arm"]]["margin"] - native,
                      "all_cut_shift": by[f"{P.PREFIX[t]}:none:{v['item']}"]["margin"] - native,
                      "effect_fraction": ctx.get("effect_fraction"), "recovery": ctx.get("recovery"),
                      "controls": [{"control": c["control"], "kind": c["kind"], "mass_relation": c.get("mass_relation"),
                                    "mass_relation_stored": c.get("mass_relation_stored"),
                                    "other_item_share": c.get("other_item_share"),
                                    "excluded": c.get("excluded"), "refutes": c.get("refutes"),
                                    "rules_out": c.get("rules_out"),
                                    "shift": by[c["control"]]["margin"] - native if c["control"] in by else None}
                                   for c in ctx.get("controls", [])]}
        table.append({"record": key(v), "claim": v["claim"], "before": b["status"], "fixed": v["status"],
                      "fixed_named_check_flag_only": f["status"], "status_without_min_effect_bar": nobar[key(v)],
                      "ruled_out": v["ruled_out"],
                      "remaining": v["remaining_alternatives"], "findings": v["findings"],
                      "effect_fraction": {"min": v["effect_fraction"]["min"], "max": v["effect_fraction"]["max"]},
                      "cut_types": per})
    # (b) BF16 coefficients with an explicit compute-error term from the measured BF16 recompute-sham floor
    smoke = rows[("bfloat16", "read-smoke-bfloat16")]
    sb = {r["arm"]: r for r in smoke}
    sham = [abs(r["margin"] - sb[r["reference"]]["margin"]) for r in smoke
            if r["kind"] in ("floor_sham_recompute", "floor_sham") and r["reference"] in sb]
    ce = max(sham)
    saved_arms = json.loads((A.MAIN02 / "analysis/arms.json").read_text())
    saved_rows = [{"arm": a.arm, "tracked": {w: next(s for s in saved_arms if s["arm"] == Q.legacy_name(a.arm))
                                             ["words"][w]["logit"] for w in ("hot", "cold")}}
                  for a in Q.value_plan("bfloat16").arms]
    fp32 = Q.coefficients(Q.value_margins(rows[("float32", "value-float32")], "float32"))
    coef = {}
    for tag, src in (("bf16_rerun", rows[("bfloat16", "value-bfloat16")]), ("saved_bf16", saved_rows)):
        coef[tag] = {"rounding_only": Q.coefficients(Q.value_margins(src, "bfloat16")),
                     "with_compute": Q.coefficients(Q.value_margins(src, "bfloat16", compute_error=ce))}
    btable = []
    for name, f in fp32.items():
        row = {"coefficient": name, "fp32": f.value, "fp32_bound": f.bound, "fp32_resolved": f.resolved}
        for tag in coef:
            r0, r1 = coef[tag]["rounding_only"][name], coef[tag]["with_compute"][name]
            row.update({tag: r1.value, f"{tag}_readout_bound": r0.bound, f"{tag}_bound": r1.bound,
                        f"{tag}_readout_rounding_resolved": r0.readout_rounding_resolved,
                        f"{tag}_resolved_without_compute_term": r0.resolved,
                        f"{tag}_resolved": r1.resolved,
                        f"{tag}_fp32_outside_readout_bound": abs(r0.value - f.value) > r0.bound,
                        f"{tag}_fp32_outside_bound": abs(r1.value - f.value) > r1.bound})
        btable.append(row)

    def n(pred):
        return sum(1 for r in btable if pred(r))
    bsum = {"compute_error_per_margin": ce,
            "compute_error_source": f"max |margin shift| of {len(sham)} BF16 recompute-sham arms "
                                    "(read-smoke-bfloat16, job-7c3232ad65bb): a MEASURED floor, not a proven bound",
            "coefficients": len(btable), "fp32_resolved": n(lambda r: r["fp32_resolved"])}
    for tag in coef:
        bsum[tag] = {"readout_rounding_resolved": n(lambda r: r[f"{tag}_readout_rounding_resolved"]),
                     "resolved_without_compute_term": n(lambda r: r[f"{tag}_resolved_without_compute_term"]),
                     "resolved_with_compute_term": n(lambda r: r[f"{tag}_resolved"]),
                     "fp32_outside_readout_bound": [r["coefficient"] for r in btable
                                                    if r[f"{tag}_fp32_outside_readout_bound"]],
                     "fp32_outside_bound_with_compute_term": [r["coefficient"] for r in btable
                                                              if r[f"{tag}_fp32_outside_bound"]]}
    out = {"plan_sha256": seal["plan_sha256"], "runs": [str(r) for r in QWEN_RUNS],
           "logic": {"prefix": {"commit": COMMITS["prefix"][0], "module_sha256": prefix_sha},
                     "fixed": {"module_sha256": hashlib.sha256(FIXED.read_bytes()).hexdigest()}},
           "counts": {"before": counts(before), "fixed": counts(fixed), "fixed_named_check_flag_only": counts(flagged)},
           "min_effect_fraction": {"value": P.DEFAULT_MIN_EFFECT_FRACTION, "in_sealed_plan": False,
                                   "records_whose_verdict_depends_on_it": [
                                       r["record"] for r in table if r["status_without_min_effect_bar"] != r["fixed"]]},
           "random_mass_labels_stored_vs_relabelled": relabel_counts(panel, fixed),
           "d_table": table, "d_verdicts_fixed": fixed, "b_bf16": {"summary": bsum, "table": btable}}
    save(QWEN_Q / "full-verdicts-20261113.json", out)
    print(json.dumps({k: out[k] for k in ("counts", "min_effect_fraction", "random_mass_labels_stored_vs_relabelled")},
                     indent=1))
    for r in table:
        print(f"  {r['record']:<22} {r['claim']:<10} {r['before']:<14} -> {r['fixed']:<15} "
              f"(flag-only {r['fixed_named_check_flag_only']:<15}) frac {r['effect_fraction']} {r['findings']}")
    print(json.dumps(bsum, indent=1))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    OUT = ap.parse_args().out
    flux()
    qwen()
