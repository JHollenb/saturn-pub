"""Verdicts for job 2 against the preregistration (python3 analyze.py; reads files in this dir).

Writes analysis.json and prints the tables. Margin M_d = logit(a) - logit(b_d) at the answer row.
Batched sweep effects are taken against the null arm of the same chunk (same batch numerics);
stepped arms against the stepped native run. Floor per item (prereg): max(2 x bf16 spacing at
the larger |logit|, |M_null - M_native| of every chunk, |M_self - M_native|, |M_again - M_native|).
"""
import glob
import json
import math
import statistics as st
from collections import Counter

BATCH = 96
cand = {i["id"]: i for i in json.load(open("candidates.json"))["items"]}
panel = {p["id"]: p for p in json.load(open("job1-panel.json"))["panel"]}
fore = {f["id"]: f for f in json.load(open("job1-forecast.json"))["forecasts"]}
recs = {}
for path in sorted(glob.glob("job2-*.jsonl")):
    for line in open(path):
        r = json.loads(line)
        recs[r["id"]] = r
med = lambda xs: st.median(xs) if xs else float("nan")  # noqa: E731


def spacing(x):
    x = abs(x)
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


def rank_corr(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for k, i in enumerate(order):
            out[i] = float(k)
        return out
    a, b = ranks(xs), ranks(ys)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


items, out = [], {}
for rid, r in recs.items():
    item, f, p = cand[rid], fore[rid], panel[rid]
    nd = len(p["donors"])
    nat = r["native"]
    a_nat = [nat[0] - nat[1 + k] for k in range(nd)]
    sw = r["sweep"]
    nulls = sw["null"]
    st_ = r["stepped"]
    self_d = [abs((st_["self.V"][0] - st_["self.V"][1 + k]) - a_nat[k]) for k in range(nd)]
    again_d = [abs((st_["native_again"][0] - st_["native_again"][1 + k]) - a_nat[k])
               for k in range(nd)]
    null_d = [abs((n[0] - n[1 + k]) - a_nat[k]) for n in nulls for k in range(nd)]
    big = max(abs(x) for x in nat)
    floor = max(2 * spacing(big), max(null_d), max(self_d), max(again_d))
    # sweep: per-site mean effect (V, K) over different-bridge donors, vs the chunk null
    eff = {"v": {}, "k": {}}
    ctl = {}
    for i, (key, lg, am) in enumerate(zip(sw["keys"], sw["logits"], sw["argmax"])):
        n = nulls[i // BATCH]
        who, t, row, layer, g = key
        site = (row, layer, g)
        if who.startswith("d"):
            k = int(who[1:])
            dm = (lg[0] - lg[1 + k]) - (n[0] - n[1 + k])
            eff[t].setdefault(site, []).append((dm, am))
        else:
            ctl[site] = am
    b = r["tokens"][1:1 + nd]
    ev = {s: st.mean(x[0] for x in v) for s, v in eff["v"].items()}
    ek = {s: st.mean(x[0] for x in v) for s, v in eff["k"].items()}
    flipv = {s: sum(x[1] == b[k] for k, x in enumerate(v)) / len(v) for s, v in eff["v"].items()}
    gt = min(ev, key=ev.get)
    order_gt = sorted(ev, key=ev.get)
    span = f["span"]
    L, KV = len(f["c"][0]), len(f["c"][0][0])

    def site_of(x):
        return (x["row"], x["layer"], x["kv"]) if x else None

    fsites = {k: site_of(f[k]) for k in ("F1", "F2", "F3")}
    cgrid = {(span[i], layer, g): f["c"][i][layer][g] for i in range(len(span))
             for layer in range(L) for g in range(KV)}
    sites = list(ev)
    rho = rank_corr([cgrid[s] for s in sites], [-ev[s] for s in sites])
    rec = {"id": rid, "family": item["family"], "floor": floor, "gt": gt, "gt_effect": ev[gt],
           "gt_flip_batched": flipv[gt], "span_len": len(span), "r0": f["r0"],
           "native_margins": a_nat, "rank_corr_c_vs_effect": rho,
           "second_site": order_gt[1], "second_effect": ev[order_gt[1]],
           "k_best": min(ek, key=ek.get), "k_best_effect": min(ek.values()),
           "k_max_abs": max(abs(v) for v in ek.values()),
           "v_sites_above_floor": sum(abs(v) > floor for v in ev.values()),
           "k_sites_above_floor": sum(abs(v) > floor for v in ek.values())}
    for k_, s in fsites.items():
        if s is None:
            continue
        rec[k_] = {"site": s, "hit1": s == gt, "hit3": s in order_gt[:3],
                   "layer_hit": s[1] == gt[1], "rank": order_gt.index(s) + 1,
                   "effect": ev[s]}
    # stepped arms: dM, flip, fraction moved
    dp = r["donor_prompt"]
    m_don = [dp[k][0] - dp[k][1 + k] for k in range(nd)]
    arms = {}
    for name, v in st_.items():
        if name in ("self.V", "native_again"):
            continue
        label, kind, who = name.split(".")
        argmax = v[-1]
        if who.startswith("d"):
            k = int(who[1:])
            dm = (v[0] - v[1 + k]) - a_nat[k]
            frac = dm / (m_don[k] - a_nat[k]) if m_don[k] != a_nat[k] else float("nan")
            arms.setdefault(f"{label}.{kind}", []).append(
                {"dm": dm, "flip": argmax == b[k], "keep": argmax == r["tokens"][0],
                 "frac": frac, "above_floor": abs(dm) > floor})
        else:
            arms.setdefault(f"{label}.{kind}.ctl", []).append(
                {"keep": argmax == r["tokens"][0], "dm": (v[0] - v[1]) - a_nat[0]})
    rec["arms"] = arms
    # batched vs stepped at the GT and F1 sites (V)
    val = []
    for label in ("GT", "F1"):
        s = gt if label == "GT" else fsites["F1"]
        for k in range(nd):
            batched = eff["v"][s][k][0]
            stepped = arms[f"{label}.V"][k]["dm"]
            val.append(abs(batched - stepped))
    rec["batched_vs_stepped_max"] = max(val)
    items.append(rec)

n = len(items)
print(f"{n} recipients with job-2 records")


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


summary = {"n": n}
for k in ("F1", "F2", "F3"):
    got = [x[k] for x in items if k in x]
    summary[k] = {"n": len(got), "hit1": rate([g["hit1"] for g in got]),
                  "hit3": rate([g["hit3"] for g in got]),
                  "layer_hit": rate([g["layer_hit"] for g in got]),
                  "median_rank": med([g["rank"] for g in got])}
summary["chance_hit1_median"] = med([1 / (x["span_len"] * 28 * 8) for x in items])
summary["gt_sites"] = Counter(f"L{x['gt'][1]}.kv{x['gt'][2]}" for x in items).most_common(6)
summary["gt_row_is_r0"] = rate([x["gt"][0] == x["r0"] for x in items])
summary["gt_layer_18_24"] = rate([18 <= x["gt"][1] <= 24 for x in items])
summary["gt_effect_median"] = med([x["gt_effect"] for x in items])
summary["second_site"] = Counter(f"L{x['second_site'][1]}.kv{x['second_site'][2]}"
                                 for x in items).most_common(6)
summary["second_effect_median"] = med([x["second_effect"] for x in items])
summary["rank_corr_median"] = med([x["rank_corr_c_vs_effect"] for x in items])
summary["floor_median"] = med([x["floor"] for x in items])
summary["floor_max"] = max(x["floor"] for x in items)
summary["batched_vs_stepped_max"] = max(x["batched_vs_stepped_max"] for x in items)
summary["batched_vs_stepped_within_floor"] = rate([x["batched_vs_stepped_max"] <= x["floor"]
                                                    for x in items])
summary["k_max_abs_median"] = med([x["k_max_abs"] for x in items])
summary["k_sites_above_floor_median"] = med([x["k_sites_above_floor"] for x in items])
summary["v_sites_above_floor_median"] = med([x["v_sites_above_floor"] for x in items])
arm_names = sorted({a for x in items for a in x["arms"]})
summary["arms"] = {}
for a in arm_names:
    rows = [e for x in items for e in x["arms"].get(a, [])]
    if a.endswith(".ctl"):
        summary["arms"][a] = {"n": len(rows), "keep": rate([e["keep"] for e in rows]),
                              "median_dm": med([e["dm"] for e in rows])}
    else:
        summary["arms"][a] = {"n": len(rows), "flip": rate([e["flip"] for e in rows]),
                              "keep": rate([e["keep"] for e in rows]),
                              "median_frac": med([e["frac"] for e in rows]),
                              "median_dm": med([e["dm"] for e in rows]),
                              "above_floor": rate([e["above_floor"] for e in rows])}
by_family = {}
for fam in sorted({x["family"] for x in items}):
    xs = [x for x in items if x["family"] == fam]
    by_family[fam] = {"n": len(xs), "F1_hit1": rate([x["F1"]["hit1"] for x in xs]),
                      "gt_effect_median": med([x["gt_effect"] for x in xs]),
                      "GT.V_flip": rate([e["flip"] for x in xs for e in x["arms"]["GT.V"]])}
summary["by_family"] = by_family
json.dump({"summary": summary, "items": items}, open("analysis.json", "w"), indent=1,
          default=str)
print(json.dumps(summary, indent=1, default=str))
