#!/usr/bin/env python3
"""Independent B2 audit recompute (auditor, 2026-11-14). Does NOT import analyze_b2.py.

Reads watch/data-b2/*.jsonl.gz, watch/subset.json, watch/r_1113.json and the 11-13 data/*.jsonl.gz.
P1/P2 are recomputed from the raw per-step letter logits and to_consumer (not from the stored
watch results); AUROC by midranks; own cluster bootstrap (seed 7, 2000 reps).
Writes audit/recompute_b2.json.  Usage: python3 audit/recompute_b2.py  (from watch/)
"""
import glob
import gzip
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parent
L = "ABCD"


def load():
    recs, dup = {}, 0
    for p in sorted(glob.glob(str(W / "data-b2" / "*.jsonl.gz"))):
        for line in gzip.open(p, "rt"):
            r = json.loads(line)
            dup += r["uid"] in recs
            recs[r["uid"]] = r
    return list(recs.values()), dup


def midrank_auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        ranks[allv[i][0]] = (i + j + 1) / 2
        i = j
    rs = sum(ranks[v] for v in pos)
    n1, n0 = len(pos), len(neg)
    return (rs - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(dpairs, apairs, reps=2000, seed=7):
    rng = random.Random(seed)

    def grp(x):
        g = defaultdict(list)
        for c, v in x:
            g[c].append(v)
        return [g[c] for c in sorted(g)]
    gd, ga = grp(dpairs), grp(apairs)
    vals = sorted(midrank_auc([v for g in (rng.choice(gd) for _ in gd) for v in g],
                              [v for g in (rng.choice(ga) for _ in ga) for v in g])
                  for _ in range(reps))
    return [round(vals[int(0.025 * reps)], 3), round(vals[int(0.975 * reps)], 3)]


def tp_at(pos, neg, k):
    thr = sorted(neg, reverse=True)[k]
    return sum(p > thr for p in pos), sum(n > thr for n in neg)


def per_step(r):
    X, R = r["X"], r["R"]
    xi, ri = L.index(X), L.index(R)
    out = []
    for s in r["sweep"]:
        nat = s["letters"]["native"]
        m2 = s["letters"]["M2"]
        m3 = s["letters"]["M3"]
        m = nat[xi] - nat[ri]
        am = L[max(range(4), key=lambda j: nat[j])]
        abl = m2[xi] - m2[ri]
        tc = s["to_consumer"]
        p1 = (am == X) and (abl < 0)
        p1m3 = (am == X) and (m3[xi] - m3[ri] < 0)
        # P2: share = tc / margin (registered), undefined at margin 0 -> False
        msc = s["margin"]
        p2 = False if msc == 0 else (tc["hint"] / msc > tc["chain_so_far"] / msc)
        out.append(dict(k=s["step"], frac=s["frac"], m=m, m_stored=s["margin"], am=am,
                        am_stored=s["argmax_letter"], abl=abl, abl_stored=s["ablated"]["hint"],
                        delta=m - abl, p1=p1, p1m3=p1m3, p2=p2))
    return out


def main():
    recs, dup = load()
    subset = json.load(open(W / "subset.json"))
    rfix = json.load(open(W / "r_1113.json"))["R"]
    res = {"n_records": len(recs), "duplicates_replaced": dup}
    sub_uids = {it["uid"]: it for it in subset["items"]}
    res["uids_match_subset"] = set(sub_uids) == {r["uid"] for r in recs}
    res["labels_match_subset"] = all(sub_uids[r["uid"]]["label"] == r["label"] for r in recs)
    res["R_matches_r1113"] = sum(rfix[r["uid"]] == r["R"] for r in recs)
    res["X_matches_subset"] = sum(sub_uids[r["uid"]]["x"] == r["X"] for r in recs)

    # ---- consistency of stored fields vs raw letters, and stored watches vs my P1
    worst = dict(margin=0.0, abl=0.0)
    am_mismatch = p1_mismatch = p2_mismatch = steps_bad = 0
    S = {}
    for r in recs:
        ps = per_step(r)
        K = r["K"]
        want = sorted({round(j * K / 33) for j in range(34)})
        steps_bad += [p["k"] for p in ps] != want
        for p in ps:
            worst["margin"] = max(worst["margin"], abs(p["m"] - p["m_stored"]))
            worst["abl"] = max(worst["abl"], abs(p["abl"] - p["abl_stored"]))
            am_mismatch += p["am"] != p["am_stored"]
        p1m = [p["p1"] for p in ps]
        p1_mismatch += p1m != r["watches"]["P1"]["holds"]
        p2_mismatch += [p["p2"] for p in ps] != r["watches"]["P2"]["holds"]
        n = len(ps)
        first = next((p["frac"] for p in ps if p["p1"]), math.inf)
        negm = [-p["m"] for p in ps]
        S[r["uid"]] = dict(
            label=r["label"], cluster=r["cluster"], K=K, nh=r["n_hint_rows"],
            ff=sum(p1m) / n, tstar=first,
            crossing=any(a and not b for a, b in zip(p1m, p1m[1:])),
            n_cross=sum(a != b for a, b in zip(p1m, p1m[1:])),
            integ=st.median(negm), final=-ps[-1]["m"], fm=ps[-1]["m"],
            meanneg=st.mean(negm), k0=-ps[0]["m"], p1k0=ps[0]["p1"], p1K=ps[-1]["p1"],
            p2=sum(p["p2"] for p in ps) / n, p1m3=sum(p["p1m3"] for p in ps) / n,
            fracX=sum(p["am"] == r["X"] for p in ps) / n,
            med_delta=st.median(p["delta"] for p in ps), delta_K=ps[-1]["delta"],
            ps=ps, m1113=r["margin_1113"])
    res["consistency"] = dict(steps_schedule_mismatch=steps_bad, stored_margin_max_abs=worst["margin"],
                              stored_ablated_max_abs=worst["abl"], argmax_mismatch=am_mismatch,
                              P1_holds_mismatch_items=p1_mismatch, P2_holds_mismatch_items=p2_mismatch)

    D = [s for s in S.values() if s["label"] == "direct"]
    A = [s for s in S.values() if s["label"] == "anyway"]
    Rz = [s for s in S.values() if s["label"] == "rationalized"]
    res["n"] = dict(direct=len(D), anyway=len(A), rationalized=len(Rz))

    def block(key, d=D, a=A, ci=True):
        pos, neg = [s[key] for s in d], [s[key] for s in a]
        o = dict(auroc=round(midrank_auc(pos, neg), 4))
        if ci:
            o["ci95"] = boot([(s["cluster"], s[key]) for s in d], [(s["cluster"], s[key]) for s in a])
        if len(neg) > 16:
            o["tp6"], o["fa6"] = tp_at(pos, neg, 6)
            o["tp16"], o["fa16"] = tp_at(pos, neg, 16)
        return o

    for key in ("ff", "integ", "final", "p2", "meanneg", "k0", "p1m3", "fracX", "med_delta"):
        res[key] = block(key)
    sd = [s for s in D if s["fm"] < 2]
    sa = [s for s in A if s["fm"] < 2]
    res["stratum_lt2"] = dict(n_direct=len(sd), n_anyway=len(sa),
                              ff=block("ff", sd, sa, False)["auroc"],
                              integ=block("integ", sd, sa, False)["auroc"],
                              final=block("final", sd, sa, False)["auroc"],
                              ff_ci=boot([(s["cluster"], s["ff"]) for s in sd],
                                         [(s["cluster"], s["ff"]) for s in sa]),
                              final_ci=boot([(s["cluster"], s["final"]) for s in sd],
                                            [(s["cluster"], s["final"]) for s in sa]))
    # paired bootstrap of the stratum difference ff - final
    rng = random.Random(11)
    diffs = []
    for _ in range(2000):
        bd = [rng.choice(sd) for _ in sd]
        ba = [rng.choice(sa) for _ in sa]
        diffs.append(midrank_auc([s["ff"] for s in bd], [s["ff"] for s in ba])
                     - midrank_auc([s["final"] for s in bd], [s["final"] for s in ba]))
    diffs.sort()
    res["stratum_lt2"]["ff_minus_final_ci95_itemboot"] = [round(diffs[50], 3), round(diffs[1949], 3)]
    # predictions
    res["T1"] = dict(median_tstar_direct=st.median(s["tstar"] for s in D),
                     median_ff_anyway=st.median(s["ff"] for s in A))
    res["T1"]["holds"] = res["T1"]["median_tstar_direct"] <= 0.5 and res["T1"]["median_ff_anyway"] <= 0.10
    res["T2"] = dict(rat=round(sum(s["crossing"] for s in Rz) / len(Rz), 3),
                     direct=round(sum(s["crossing"] for s in D) / len(D), 3))
    res["T2"]["holds"] = res["T2"]["rat"] >= 0.30 and res["T2"]["direct"] <= 0.10
    res["T3"] = dict(tp6=res["ff"]["tp6"], holds=res["ff"]["tp6"] >= 45)
    res["descriptives"] = {lab: dict(median_ff=st.median(s["ff"] for s in g),
                                     fired_any=sum(s["ff"] > 0 for s in g),
                                     median_cross=st.median(s["n_cross"] for s in g),
                                     median_final_margin=round(st.median(s["fm"] for s in g), 3),
                                     p1_at_k0=sum(s["p1k0"] for s in g), p1_at_K=sum(s["p1K"] for s in g),
                                     median_K=st.median(s["K"] for s in g),
                                     median_hint_rows=st.median(s["nh"] for s in g))
                           for lab, g in (("direct", D), ("anyway", A), ("rationalized", Rz))}
    # tstar distribution (direct): fraction firing at k=0 exactly
    res["T1"]["direct_tstar_eq0"] = sum(s["tstar"] == 0 for s in D)
    res["T1"]["direct_tstar_le_0.1"] = sum(s["tstar"] <= 0.1 for s in D)
    res["T1"]["anyway_tstar_eq0"] = sum(s["tstar"] == 0 for s in A)

    # ---- leakage: chain length, hint rows
    res["leak"] = dict(K_auroc=round(midrank_auc([s["K"] for s in D], [s["K"] for s in A]), 3),
                       hint_rows_auroc=round(midrank_auc([s["nh"] for s in D], [s["nh"] for s in A]), 3),
                       ff_vs_K_spearman_all=None)

    # ---- circularity: margin-only placebos (no hint information)
    allsteps = [p for s in D + A for p in s["ps"]]
    c = st.median(p["delta"] for p in allsteps)  # pooled, label-free constant hint push
    for r in recs:
        s = S[r["uid"]]
        n = len(s["ps"])
        s["plac_c"] = sum(p["am"] == r["X"] and p["m"] < c for p in s["ps"]) / n
        s["plac_c_final"] = float(s["ps"][-1]["am"] == r["X"] and s["ps"][-1]["m"] < c)
    res["placebo_constant_push"] = dict(c=round(c, 4), ff_placebo=block("plac_c"),
                                        stratum_lt2=block("plac_c", sd, sa, False)["auroc"])
    # strata of final margin
    strata = {}
    for lo, hi in ((-99, 1), (1, 2), (2, 4), (4, 99)):
        d = [s for s in D if lo <= s["fm"] < hi]
        a = [s for s in A if lo <= s["fm"] < hi]
        e = dict(n_direct=len(d), n_anyway=len(a))
        if d and a:
            e.update(ff=round(midrank_auc([s["ff"] for s in d], [s["ff"] for s in a]), 3),
                     final=round(midrank_auc([s["final"] for s in d], [s["final"] for s in a]), 3),
                     integ=round(midrank_auc([s["integ"] for s in d], [s["integ"] for s in a]), 3),
                     placebo=round(midrank_auc([s["plac_c"] for s in d], [s["plac_c"] for s in a]), 3))
        e["median_ff_direct"] = st.median(s["ff"] for s in d) if d else None
        e["median_ff_anyway"] = st.median(s["ff"] for s in a) if a else None
        strata[f"[{lo},{hi})"] = e
    res["final_margin_strata"] = strata
    # matched pairs from subset.json
    pr = []
    for du, au, md, ma in subset["pairs"]:
        sdd, saa = S[du], S[au]
        pr.append(dict(dm=sdd["fm"], am=saa["fm"], ff=(sdd["ff"] > saa["ff"]) + 0.5 * (sdd["ff"] == saa["ff"]),
                       fin=(sdd["final"] > saa["final"]) + 0.5 * (sdd["final"] == saa["final"]),
                       gap=abs(md - ma)))
    def pairstat(ps):
        return dict(n=len(ps), ff_wins=round(sum(p["ff"] for p in ps) / len(ps), 3),
                    final_wins=round(sum(p["fin"] for p in ps) / len(ps), 3)) if ps else dict(n=0)
    res["pairs"] = dict(all=pairstat(pr), gap_lt_1=pairstat([p for p in pr if p["gap"] < 1]),
                        gap_lt_2=pairstat([p for p in pr if p["gap"] < 2]),
                        median_gap=round(st.median(p["gap"] for p in pr), 3))
    # logistic: label ~ final margin (+ ff); in-sample LR test (numpy-free Newton)
    def logit_fit(X, y, iters=50):
        p = len(X[0])
        b = [0.0] * p
        for _ in range(iters):
            g = [0.0] * p
            Hm = [[0.0] * p for _ in range(p)]
            ll = 0.0
            for xi, yi in zip(X, y):
                z = sum(bj * xj for bj, xj in zip(b, xi))
                z = max(min(z, 30), -30)
                mu = 1 / (1 + math.exp(-z))
                ll += yi * math.log(mu + 1e-12) + (1 - yi) * math.log(1 - mu + 1e-12)
                for i in range(p):
                    g[i] += (yi - mu) * xi[i]
                    for j in range(p):
                        Hm[i][j] += mu * (1 - mu) * xi[i] * xi[j]
            # solve Hm d = g (Gauss)
            M = [row[:] + [gi] for row, gi in zip(Hm, g)]
            for i in range(p):
                piv = max(range(i, p), key=lambda r_: abs(M[r_][i]))
                M[i], M[piv] = M[piv], M[i]
                for r_ in range(p):
                    if r_ != i:
                        f = M[r_][i] / M[i][i]
                        M[r_] = [a - f * bb for a, bb in zip(M[r_], M[i])]
            b = [bj + M[i][p] / M[i][i] for i, bj in enumerate(b)]
        return b, ll
    DA = D + A
    y = [1 if s["label"] == "direct" else 0 for s in DA]
    lfm = [math.copysign(math.log1p(abs(s["fm"])), s["fm"]) for s in DA]
    b0, ll0 = logit_fit([[1, f] for f in lfm], y)
    b1, ll1 = logit_fit([[1, f, s["ff"]] for f, s in zip(lfm, DA)], y)
    b2, ll2 = logit_fit([[1, f, s["p1K"] * 1.0] for f, s in zip(lfm, DA)], y)
    b3, ll3 = logit_fit([[1, f, s["p1K"] * 1.0, s["ff"]] for f, s in zip(lfm, DA)], y)
    res["logistic_slog_final_margin"] = dict(
        ll_margin=round(ll0, 3), ll_margin_ff=round(ll1, 3), LR_ff=round(2 * (ll1 - ll0), 3),
        coef_ff=round(b1[2], 3), ll_margin_p1K=round(ll2, 3), LR_p1K=round(2 * (ll2 - ll0), 3),
        ll_margin_p1K_ff=round(ll3, 3), LR_ff_given_p1K=round(2 * (ll3 - ll2), 3),
        note="LR ~ chi2(1): 3.84 = p .05, 6.63 = p .01; in-sample, clustered items ignored")
    # exactness
    ch = [r["checks"] for r in recs]
    fs = [d for c_ in ch for d in c_.get("feed_spot", [])]
    res["exactness"] = dict(null_vs_fresh_max=max(c_["null_vs_fresh_max_abs"] for c_ in ch),
                            trace_check_max=max(c_["trace_check_max_abs"] for c_ in ch if c_["trace_check_max_abs"] is not None),
                            trace_check_n=sum(c_["trace_check_max_abs"] is not None for c_ in ch),
                            feed_native_max=max(d["native_max_abs"] for d in fs),
                            feed_M2_max=max(d["M2_abs"] for d in fs), feed_n=len(fs),
                            feed_supplied=sorted({d["supplied"] for d in fs}),
                            R_agrees=sum(c_["R_agrees"] for c_ in ch))
    # cross-check vs the 11-13 run at k = K (same prompt + chain + suffix)
    old = {}
    for p in glob.glob(str(W.parent / "data" / "*.jsonl.gz")):
        for line in gzip.open(p, "rt"):
            o = json.loads(line)
            if str(o.get("f")) in ("1.0", "1") and "margin_ablated" in o:
                old[o["uid"]] = o
    dm, da, nmatch = [], [], 0
    for r in recs:
        o = old.get(r["uid"])
        if o is None:
            continue
        nmatch += 1
        ps = S[r["uid"]]["ps"]
        dm.append(abs(ps[-1]["m"] - float(o["margin"])))
        da.append(abs(ps[-1]["abl"] - float(o["margin_ablated"])))
    res["vs_1113_at_K"] = dict(n=nmatch, margin_max_abs=max(dm), margin_median_abs=st.median(dm),
                               M2_max_abs=max(da), M2_median_abs=st.median(da))
    # P1 at K vs 11-13 strict flip on the same items
    res["PH2"] = dict(direct_P1_at_K=sum(s["p1K"] for s in D), anyway_P1_at_K=sum(s["p1K"] for s in A))
    runs = []
    for s in D:
        c_ = 0
        for p in s["ps"] + [dict(p1=False)]:
            if p["p1"]:
                c_ += 1
            elif c_:
                runs.append(c_)
                c_ = 0
    res["PH2"]["direct_median_run_len"] = st.median(runs)
    for s in S.values():
        s.pop("ps")
    (HERE / "recompute_b2.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
