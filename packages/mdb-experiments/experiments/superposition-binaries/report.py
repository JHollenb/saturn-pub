"""Merge study-S score.json files (job dirs) and print the PREREG tables and pass conditions.

    python report.py DIR/job-a DIR/job-b ... [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROGRAMS = ("bsort-O0", "bsort-O2", "fsm-O0", "fsm-O2", "fnv-O0", "fnv-O2")
ORDER = ("control", "rot", "rot_n0.1", "rot_n0.3", "rot_n1.0", "p3216", "p1024", "p512", "p256", "p128",
         "p64", "p32", "p16", "p8", "p4")
PRIMARY = "sae_w96_k4"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def merge(dirs):
    pooled, per_pc, src = {}, {}, {}
    for d in dirs:
        r = json.loads((Path(d) / "score.json").read_text())
        job = Path(d).name
        for c, v in r["pooled"].items():
            pooled[c] = v
            src[("pooled", c)] = job
        for c, v in r["per_pc"].items():
            per_pc[c] = v
            src[("per_pc", c)] = job
    return pooled, per_pc, src


def avg(pooled, cond, method, key="macro_iou", sub=None):
    vals = []
    for p in PROGRAMS:
        m = pooled[cond][p]["methods"].get(method)
        if m is None:
            return None
        m = m[sub] if sub else m
        vals.append(m.get(key))
    return mean(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pooled, per_pc, src = merge(a.dirs)
    conds = [c for c in ORDER if c in pooled]
    kkc = lambda c, p: next(k for k in pooled[c][p]["methods"] if k.endswith("_kc"))  # noqa: E731
    out = {"pooled": {}, "per_pc": {}, "grid": {}, "value_split": {}, "pass": {}}
    print("## pooled matched IoU / purity (mean of 6 programs)\n")
    print("| condition | job | raw | pca | ica | kmeans k=Kc | sae w96 k4 | sae oracle-cut | sae group IoU | sae fvu |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        row = {m: (avg(pooled, c, m), avg(pooled, c, m, "macro_precision")) for m in ("raw", "pca", "ica", PRIMARY)}
        km = mean([pooled[c][p]["methods"][kkc(c, p)]["macro_iou"] for p in PROGRAMS])
        kmp = mean([pooled[c][p]["methods"][kkc(c, p)]["macro_precision"] for p in PROGRAMS])
        row["kmeans_kc"] = (km, kmp)
        oc = avg(pooled, c, PRIMARY, sub="oracle_cut")
        gi = avg(pooled, c, PRIMARY, "macro_group_iou")
        fvu = avg(pooled, c, PRIMARY, "fvu")
        out["pooled"][c] = {**{k: {"iou": v[0], "purity": v[1]} for k, v in row.items()},
                            "sae_oracle_cut_iou": oc, "sae_group_iou": gi, "sae_fvu": fvu, "job": src[("pooled", c)]}
        f = lambda t: f"{t[0]:.3f} / {t[1]:.3f}"  # noqa: E731
        print(f"| {c} | {src[('pooled', c)]} | {f(row['raw'])} | {f(row['pca'])} | {f(row['ica'])} | "
              f"{f(row['kmeans_kc'])} | {f(row[PRIMARY])} | {oc:.3f} | {gi:.3f} | {fvu:.3f} |")
    if "control" in pooled and "raw_native" in pooled["control"][PROGRAMS[0]]["methods"]:
        rn = avg(pooled, "control", "raw_native", "macro_group_iou")
        rni = avg(pooled, "control", "raw_native")
        print(f"\ncontrol raw native (bit > 0.5): best-unit IoU {rni:.3f}, group IoU {rn:.3f}")
        out["control_raw_native"] = {"iou": rni, "group_iou": rn}
    print("\n## SAE grid: matched IoU / purity / group IoU / value purity (baseline) / classes with >=2 pure units\n")
    for c in conds:
        ms = [m for m in pooled[c][PROGRAMS[0]]["methods"] if m.startswith("sae_")]
        if len(ms) < 2:
            continue
        for m in ms:
            vp = mean([pooled[c][p]["methods"][m]["value_split"]["value_purity"] for p in PROGRAMS])
            vb = mean([pooled[c][p]["methods"][m]["value_split"]["baseline_class_modal_value_share"] for p in PROGRAMS])
            two = mean([sum(v >= 2 for v in pooled[c][p]["methods"][m]["value_split"]["pure_units_per_class"].values())
                        for p in PROGRAMS])
            live = mean([pooled[c][p]["methods"][m]["live"] for p in PROGRAMS])
            r = {"iou": avg(pooled, c, m), "purity": avg(pooled, c, m, "macro_precision"),
                 "group_iou": avg(pooled, c, m, "macro_group_iou"), "value_purity": vp, "value_baseline": vb,
                 "classes_2plus_pure": two, "live": live, "fvu": avg(pooled, c, m, "fvu")}
            out["grid"].setdefault(c, {})[m] = r
            print(f"{c:9s} {m:14s} iou {r['iou']:.3f} pur {r['purity']:.3f} group {r['group_iou']:.3f} "
                  f"vp {vp if vp is None else round(vp, 3)} (base {vb if vb is None else round(vb, 3)}) "
                  f"2+pure {two:.2f} live {live:.0f} fvu {r['fvu']:.3f}")
    # pass conditions
    def cond_ok(c):
        if c not in pooled:
            return None
        s, r_, pc, ic = (avg(pooled, c, m) for m in (PRIMARY, "raw", "pca", "ica"))
        return {"sae": s, "raw": r_, "pca": pc, "ica": ic, "sae_minus_raw": s - r_, "gg": s - r_ >= 0.20,
                "ge": s >= max(pc, ic) - 0.02, "ok": (s - r_ >= 0.20) and (s >= max(pc, ic) - 0.02)}
    P = {c: cond_ok(c) for c in ("rot", "p1024", "p256", "p64")}
    out["pass"]["conditions"] = P
    if all(P.values()):
        out["pass"]["P(i)"] = P["rot"]["ok"]
        out["pass"]["P(ii)"] = all(P[c]["ok"] for c in ("p1024", "p256", "p64"))
        out["pass"]["SAEs earn their place"] = out["pass"]["P(i)"] and out["pass"]["P(ii)"]
    sweep = [c for c in ORDER if c.startswith("p") and c in pooled]
    if "p3216" in pooled:
        ref = avg(pooled, "p3216", PRIMARY)
        broken = [int(c[1:]) for c in sweep if avg(pooled, c, PRIMARY) < 0.8 * ref]
        out["pass"]["break_d"] = max(broken) if broken else None
        out["pass"]["break_ref_p3216"] = ref
    if "control" in pooled:
        s, r_ = avg(pooled, "control", PRIMARY), avg(pooled, "control", "raw")
        sp, rp = avg(pooled, "control", PRIMARY, "macro_precision"), avg(pooled, "control", "raw", "macro_precision")
        out["pass"]["control_raw_matches_sae"] = {"raw_iou": r_, "sae_iou": s, "raw_purity": rp, "sae_purity": sp,
                                                  "holds": r_ >= s - 0.02 and rp >= sp - 0.02}
    print("\n## pass conditions\n")
    print(json.dumps(out["pass"], indent=1))
    print("\n## per-pc macro bit IoU (mean over pcs, then programs)\n")
    for c, v in per_pc.items():
        ms = v[PROGRAMS[0]]["macro_bit_iou_mean_over_pcs"].keys()
        row = {m: mean([v[p]["macro_bit_iou_mean_over_pcs"][m] for p in PROGRAMS]) for m in ms}
        out["per_pc"][c] = {**row, "job": src[("per_pc", c)], "pcs": sum(v[p]["pcs"] for p in PROGRAMS),
                            "class_purity_trivially_1": all(v[p]["class_purity_trivially_1"] for p in PROGRAMS)}
        print(c, src[("per_pc", c)], json.dumps({m: round(x, 3) for m, x in row.items()}), "pcs", out["per_pc"][c]["pcs"])
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
