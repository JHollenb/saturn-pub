"""Score the amplitude programme (prereg.json) from out/<model>/*.jsonl -> summary.json + tables.

python score.py            (reads data/ and out/, writes summary.json, prints markdown tables)
"""
import glob
import json
import math
import statistics as st
from pathlib import Path

D = Path(__file__).resolve().parent
P = json.load(open(D / "data/panel.json"))
PS = json.load(open(D / "data/panel_supp.json"))
ITEM = {i["id"]: i for i in P["items"] + PS["items"]}
DOSES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"]


def rows(model, kind):
    out = {}
    for f in sorted(glob.glob(str(D / "out" / model / f"{kind}-{model}-*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out[r["id"]] = r
    return list(out.values())


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def frac(a, b):
    return f"{a}/{b}" + (f" ({a / b:.2f})" if b else "")


def armA(model):
    A = [r for r in rows(model, "armA") if "excluded" not in r]
    res = {}
    for stratum in ("right_operand", "readable"):
        it = [r for r in A if r["stratum"] == stratum]
        mono, flips, stars, crit, stable_flip = 0, 0, [], [], 0
        span = {k: 0 for k in ("1", "4", "16")}
        tb = 0
        per = []
        for r in it:
            a = ITEM[r["id"]]["answer"]
            Ms = [r["stepped"][f"T{x:g}"]["M"] for x in DOSES]
            fl = r["floor"]
            ok = all(Ms[i + 1] >= Ms[i] - fl for i in range(len(Ms) - 1))
            mono += ok
            star = next((x for x in DOSES if r["stepped"][f"T{x:g}"]["argmax"] == a), None)
            if star is not None:
                flips += 1
                stars.append(star)
                crit.append(r["amp"] * star)
                stable_flip += (r["stepped"][f"T{star:g}"]["stability"] == "stable")
            tb += any(r["batched"][f"T{x:g}"]["argmax"] == a for x in DOSES)
            for k in span:
                span[k] += any(r["batched"].get(f"S{k}_{x:g}", {}).get("argmax") == a for x in DOSES)
            nul = {k: v["argmax"] == a for k, v in r["nulls"].items()}
            per.append({"id": r["id"], "amp": r["amp"], "alpha_star": star, "monotone": ok, "M0": r["M0"],
                        "M_by_dose": Ms, "floor": fl, "null_repairs": nul,
                        "self_dM": r["nulls"]["null_self"]["dM"]})
        res[stratum] = {"n": len(it), "monotone": mono, "flipped": flips, "alpha_star_median": med(stars),
                        "alpha_stars": stars, "critical_amp": crit,
                        "critical_amp_median": med(crit),
                        "critical_q75_over_q25": (q(crit, .75) / q(crit, .25)) if len(crit) >= 2 and q(crit, .25) > 0 else None,
                        "flip_stable_4floors": stable_flip, "batched_T_flips": tb, "span_flips": span,
                        "null_repairs": sum(any(p["null_repairs"][k] for k in ("null_rnd3", "null_rnd8")) for p in per),
                        "self_dM_max": max((abs(p["self_dM"]) for p in per), default=None),
                        "items": per}
    # P2: amplitudes
    E = [r for r in rows(model, "armE1") if "excluded" not in r]
    ro = [r for r in A if r["stratum"] == "right_operand"]
    ctry = {ITEM[r["id"]]["country_token"] for r in ro}
    corr = [r["amp"] for r in E if ITEM[r["id"]]["country_token"] in ctry]
    corr_all = [r["amp"] for r in E]
    res["P2"] = {"auc_correct_vs_right_operand": auc(corr, [r["amp"] for r in ro]),
                 "n_correct_same_country": len(corr),
                 "auc_correct_all_vs_right_operand": auc(corr_all, [r["amp"] for r in ro]),
                 "median_amp_correct": med(corr), "median_amp_right_operand": med([r["amp"] for r in ro]),
                 "median_amp_readable": med([r["amp"] for r in A if r["stratum"] == "readable"]),
                 "auc_correct_vs_readable": auc(corr, [r["amp"] for r in A if r["stratum"] == "readable"])}
    # E1
    lose, ratio = 0, []
    for r in E:
        a = ITEM[r["id"]]["answer"]
        if r["base_argmax"] != a:
            continue
        lose += r["batched"]["T0"]["argmax"] != a
        if r["M0"] > 0:
            ratio.append(r["batched"]["T0"]["M"] / r["M0"])
    res["E1"] = {"n": len(ratio), "lose_at_alpha0": lose, "median_M0ratio": med(ratio),
                 "lose_at": {k: sum(1 for r in E if r["base_argmax"] == ITEM[r["id"]]["answer"]
                                    and r["batched"][k]["argmax"] != ITEM[r["id"]]["answer"]) for k in ("T0", "T0.25", "T0.5", "T0.75")}}
    res["mdb_vs_score_correctness_disagree"] = sum(1 for r in rows(model, "armA") if r.get("mdb_correct"))
    return res


def armB(model, LC):
    B = rows(model, "armB")
    out = {"n": len(B)}
    key = f"amp_L{LC}"
    strata = {}
    arms_all = []
    for r in B:
        a = ITEM[r["id"]]["answer"]
        true = {k: v for k, v in r["arms"].items() if k.startswith("B")}
        rnd = {k: v for k, v in r["arms"].items() if k.startswith("R")}
        rep = [k for k, v in true.items() if v["argmax"] == a]
        s = strata.setdefault(r["stratum"], {"n": 0, "any": 0, "arms": 0, "rep_arms": 0, "rnd_rep_arms": 0,
                                             "rnd_arms": 0, "any_stable": 0})
        s["n"] += 1
        s["any"] += bool(rep)
        s["any_stable"] += any(true[k]["stability"] == "stable" for k in rep)
        s["arms"] += len(true)
        s["rep_arms"] += len(rep)
        s["rnd_arms"] += len(rnd)
        s["rnd_rep_arms"] += sum(v["argmax"] == a for v in rnd.values())
        if not r["excluded"] and key in r["base_probes"]:
            a0 = r["base_probes"][key]
            for k, v in true.items():
                a1 = v["probes"].get(key)
                raise_ = (a1 - a0 >= 0.3) or (a0 > 0.1 and a1 / a0 >= 1.3)
                arms_all.append({"id": r["id"], "arm": k, "stratum": r["stratum"], "a0": a0, "a1": a1,
                                 "raise": raise_, "repair": v["argmax"] == a, "dM": v["dM"],
                                 "stable": v["stability"] == "stable"})
    out["strata"] = strata
    rep = [x for x in arms_all if x["repair"]]
    ra = [x for x in arms_all if x["raise"]]
    nr = [x for x in arms_all if not x["raise"]]
    out["repairing_arms"] = len(rep)
    out["repairing_arms_raise"] = sum(x["raise"] for x in rep)
    out["repairing_arms_no_rise_lt_0.1"] = sum((x["a1"] - x["a0"]) < 0.1 for x in rep)
    out["raise_arms"] = len(ra)
    out["repair_rate_raise"] = (sum(x["repair"] for x in ra) / len(ra)) if ra else None
    out["repair_rate_noraise"] = (sum(x["repair"] for x in nr) / len(nr)) if nr else None
    out["median_da_by_pair"] = {}
    pairs = sorted({x["arm"] for x in arms_all})
    for p in pairs:
        out["median_da_by_pair"][p] = med([x["a1"] - x["a0"] for x in arms_all if x["arm"] == p])
    out["corr_da_dM"] = _spearman([x["a1"] - x["a0"] for x in arms_all], [x["dM"] for x in arms_all])
    out["repairs"] = [{k: x[k] for k in ("id", "arm", "stratum", "a0", "a1", "dM", "stable")} for x in rep]
    return out


def _spearman(x, y):
    if len(x) < 3:
        return None
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for n, i in enumerate(o):
            r[i] = n
        return r
    a, b = rk(x), rk(y)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q_ - mb) for p, q_ in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q_ - mb) ** 2 for q_ in b))
    return num / den if den else None


def armC(model):
    C = rows(model, "armC")
    A = {r["id"]: r for r in rows(model, "armA")}
    out = {"n": len(C), "flipped_fp32": sum(r["alpha_star"] is not None for r in C)}
    agree = 0
    for r in C:
        a = ITEM[r["id"]]["answer"]
        bf = next((x for x in DOSES if A.get(r["id"], {}).get("stepped", {}).get(f"T{x:g}", {}).get("argmax") == a), None)
        agree += bf == r["alpha_star"]
    out["alpha_star_bf16_equals_fp32"] = agree
    out["core_check_ok"] = sum(bool(r["core_check"]["ok"]) for r in C)
    gaps = [abs(r["closure_gap"]) for r in C if "closure_gap" in r]
    out["closure_gap_max"] = max(gaps) if gaps else None
    comp = [r["composer"] for r in C if "composer" in r]
    out["composer_n"] = len(comp)
    out["jaccard_median"] = med([c["jaccard"] for c in comp])
    out["jaccard_random_median"] = med([c["random_mean"] for c in comp])
    out["jaccard_random_max"] = max((c["random_max"] for c in comp), default=None)
    out["composer_items"] = [{"id": r["id"], "stratum": r["stratum"], **r["composer"]} for r in C if "composer" in r]
    out["switch_count_median"] = {}
    for r in C:
        for x, n in (r.get("switch_count") or {}).items():
            out["switch_count_median"].setdefault(x, []).append(n)
    out["switch_count_median"] = {x: med(v) for x, v in out["switch_count_median"].items()}
    return out


def forecast(model, LC, L):
    F = rows(model, "forecast")
    out = {"n": len(F)}
    ratios = {"F1": []}
    mism = {"F1": 0}
    for k in ("8", "32", "128"):
        ratios[f"F2_{k}"], ratios[f"F3_{k}"] = [], []
        mism[f"F2_{k}"] = mism[f"F3_{k}"] = 0
    ratios["F3_0"], mism["F3_0"] = [], 0
    nflip = 0
    for r in F:
        d = r["doses"]["1"]
        nat, M0 = d["native_delta"], r["M0"]
        flip = (M0 + nat) < 0
        nflip += flip
        fs = {"F1": d["F1"], **{f"F2_{k}": v for k, v in d["F2"].items()}, **{f"F3_{k}": v for k, v in d["F3"].items()}}
        for k, v in fs.items():
            if nat:
                ratios[k].append(v / nat)
            mism[k] += ((M0 + v) < 0) != flip
    out["native_flips"] = nflip
    out["median_ratio"] = {k: med(v) for k, v in ratios.items()}
    out["flip_mismatch"] = mism
    # log-slopes of the exact MLP term per layer and of the re-read total
    slopes = {}
    rer = []
    for r in F:
        ds = r["doses"]
        for lo, hi in (("0.5", "1"), ("1", "2")):
            for x, t1 in ds[lo]["mlp_by_layer"].items():
                t2 = ds[hi]["mlp_by_layer"][x]
                if abs(t1) > 0.05 and abs(t2) > 0.05 and t1 * t2 > 0:
                    slopes.setdefault(int(x), []).append(math.log2(t2 / t1))
            a1, a2 = ds[lo]["reread"], ds[hi]["reread"]
            if abs(a1) > 0.05 and abs(a2) > 0.05 and a1 * a2 > 0:
                rer.append(math.log2(a2 / a1))
    out["mlp_log_slope_by_layer"] = {x: med(v) for x, v in sorted(slopes.items())}
    early = [s for x, v in slopes.items() if LC + 1 <= x <= LC + 4 for s in v]
    late = [s for x, v in slopes.items() if LC + 5 <= x <= L - 1 for s in v]
    out["mlp_log_slope_band"] = med(early)
    out["mlp_log_slope_tail"] = med(late)
    out["reread_log_slope"] = med(rer)
    out["closure_gap_max"] = max(abs(r["doses"][k]["closure_gap"]) for r in F for k in r["doses"]) if F else None
    out["core_check_ok"] = sum(bool(r["core_check"]["ok"]) for r in F)
    out["switches_median_by_layer"] = {}
    for r in F:
        for x, n in r["doses"]["1"]["switches"].items():
            out["switches_median_by_layer"].setdefault(x, []).append(n)
    out["switches_median_by_layer"] = {x: med(v) for x, v in out["switches_median_by_layer"].items()}
    return out


if __name__ == "__main__":
    summary = {}
    for m in MODELS:
        if not (D / "out" / m).is_dir():
            continue
        S = json.load(open(D / "data" / f"score-{m}.json"))["model"]
        LC, L = S["L_c"], S["layers"]
        summary[m] = {"L_c": LC, "A": armA(m), "B": armB(m, LC), "C": armC(m), "F": forecast(m, LC, L)}
    json.dump(summary, open(D / "summary.json", "w"), indent=1, default=str)
    for m, s in summary.items():
        A = s["A"]
        print(f"## {m}")
        for k in ("right_operand", "readable"):
            a = A[k]
            print(k, "n", a["n"], "mono", a["monotone"], "flip", a["flipped"], "alpha*", a["alpha_stars"],
                  "med", a["alpha_star_median"], "crit med", a["critical_amp_median"], "q75/q25", a["critical_q75_over_q25"],
                  "stable", a["flip_stable_4floors"], "spans", a["span_flips"], "Tb", a["batched_T_flips"],
                  "null rep", a["null_repairs"], "self", a["self_dM_max"])
        print("P2", A["P2"]); print("E1", A["E1"]); print("disagree", A["mdb_vs_score_correctness_disagree"])
        B = dict(s["B"]); B.pop("repairs"); B.pop("median_da_by_pair")
        print("B", B)
        C = dict(s["C"]); C.pop("composer_items")
        print("C", C)
        print("F", s["F"])
