"""Reconciliation of entry-1 numbers with explicit definitions (python3 reconcile.py).

Every number in the tracker's entry 1 that is not printed by analyze.py or census.py is computed
here, with the definition written next to it. Writes reconcile.json. Inputs: job2-*.jsonl,
job3-*.jsonl, job4.jsonl, job1-panel.json, candidates.json, analysis.json.
"""
import glob
import json
import math
import statistics as st
from collections import Counter, defaultdict

out = {}
cand = {i["id"]: i for i in json.load(open("candidates.json"))["items"]}
spec = json.load(open("candidates.json"))
captok = {v: k for k, v in spec["capital_tokens"].items()}
panel = {p["id"]: p for p in json.load(open("job1-panel.json"))["panel"]}
rows1 = {r["id"]: r for r in json.load(open("job1-panel.json"))["rows"]}
recs = {}
for path in glob.glob("job2-*.jsonl"):
    for line in open(path):
        r = json.loads(line)
        recs[r["id"]] = r


def spacing(x):
    x = abs(x)
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


# 1. P6: batched null vs stepped native, per item: max over chunks and donors of
#    |M_null - M_native| <= 2 x bf16 spacing, spacing taken (a) at |M_native| (the prereg text
#    "at |M|"), (b) at the largest |logit| among the kept tokens (what analyze.py's floor uses).
at_m = at_logit = 0
for rid, r in recs.items():
    nd = len(panel[rid]["donors"])
    nat = r["native"]
    ok_m = ok_l = True
    for n in r["sweep"]["null"]:
        for k in range(nd):
            m_nat = nat[0] - nat[1 + k]
            d = abs((n[0] - n[1 + k]) - m_nat)
            ok_m &= d <= 2 * spacing(m_nat)
            ok_l &= d <= 2 * spacing(max(abs(nat[0]), abs(nat[1 + k])))
    at_m += ok_m
    at_logit += ok_l
out["P6"] = {"items": len(recs), "within_2_spacing_at_abs_M (prereg wording)": at_m,
             "within_2_spacing_at_larger_abs_logit_of_the_pair": at_logit,
             "note": "verdicts.json said 64/64; that number came from an ad-hoc check whose "
                     "spacing used the largest |logit| over all kept tokens (incl. the control's "
                     "answer), not saved; superseded by this file"}

# 2. P5 split of errors whose hop 1 and hop 2 alone are both right.
#    other-capital: argmax is a capital token of the candidate set; copy: the argmax text
#    (stripped, lowercased) is a prefix of a word of the descriptor; else non-capital city/other.
split = Counter()
for rid, r in rows1.items():
    if r["correct"] or not (r["hop1_correct"] and r["hop2_correct"]):
        continue
    words = cand[rid]["descriptor"].lower().split()
    t = r["argmax_text"].strip().lower()
    if r["argmax"] in captok:
        split["other-capital"] += 1
    elif t and any(w.startswith(t) for w in words):
        split["copy (word-prefix of the descriptor)"] += 1
    else:
        split["non-capital other (mostly a city of the country)"] += 1
out["P5_split"] = dict(split)

# 3./4. Natural-error repair and readout, two class definitions.
j4 = [json.loads(line) for line in open("job4.jsonl")]
err = [r for r in j4 if not r["correct"]]
defs = {
    "A_wrong_capital_by_answer": lambda r: rows1[r["id"]]["argmax"] in captok,
    "B_my_tracker_set (lira, lev, sol, peso, shilling)":
        lambda r: r["id"] in {"currency:lira", "currency:lev", "currency:sol", "currency:peso",
                              "currency:shilling"},
}
out["repair"] = {}
for name, f in defs.items():
    a = [r for r in err if f(r)]
    b = [r for r in err if not f(r)]

    def tally(group):
        reps = [p for r in group for p in r.get("repairs", [])]
        return {"items": [r["id"] for r in group] if len(group) < 10 else len(group),
                "repairs": len(reps), "cell_fixed": sum(p["carrier_cell"]["fixed"] for p in reps),
                "band_fixed": sum(p["band_v_8_27"]["fixed"] for p in reps),
                "readout_best_country_is_true": sum(r["naming"].get("true_best_of_countries", False)
                                                    for r in group),
                "readout_best_country": {r["id"]: r["naming"].get("best_country") for r in group}
                if len(group) < 10 else None}
    out["repair"][name] = {"class": tally(a), "rest": tally(b)}
out["repair"]["note"] = ("readout best country = best of the candidate panel's single-token "
                         "country tokens (Armenia is not among them; lev's full-vocabulary top-1 is "
                         "' Armenia', its best panel country is ' Russia')")

# 7. cell_sweep timing per item (arms, seconds) from the saved records.
times = [(r["sweep"]["arms"], r["sweep"]["seconds"]) for r in recs.values()]
out["sweep_timing"] = {"items": len(times), "median_seconds": st.median(t for _, t in times),
                       "median_arms": st.median(a for a, _ in times),
                       "median_ms_per_arm": st.median(1000 * t / a for a, t in times),
                       "smoke_job-febd18f58dd4_first_item": "1,120 arms, 0.24 s (job log)"}

# 8. F2 vs F1 sites.
fore = {f["id"]: f for f in json.load(open("job1-forecast.json"))["forecasts"]}
same = sum(1 for f in fore.values() if f["F2"] and (f["F2"]["row"], f["F2"]["layer"], f["F2"]["kv"])
           == (f["F1"]["row"], f["F1"]["layer"], f["F1"]["kv"]))
out["F2_equals_F1_site"] = f"{same}/{len(fore)}"

# 9. P4 control failures with the argmax class.
fails = []
for rid, r in recs.items():
    ctl = panel[rid]["control"]
    for a in ("F1.V.ctl", "GT.V.ctl", "band8.V.ctl"):
        v = r["stepped"].get(a)
        if v and v[-1] != r["tokens"][0]:
            city = ctl.startswith("city:")
            fails.append({"recipient": rid, "arm": a, "control": ctl,
                          "argmax_is_capital": v[-1] in captok, "control_is_city": city})
out["P4_failures"] = {a: {"n": sum(f["arm"] == a for f in fails),
                          "city_controls": sum(f["arm"] == a and f["control_is_city"] for f in fails),
                          "other_capital": [f["recipient"] for f in fails
                                            if f["arm"] == a and f["argmax_is_capital"]]}
                      for a in ("F1.V.ctl", "GT.V.ctl", "band8.V.ctl")}

# 6./10. Bridge features: naming match definitions and a cross-bridge null.
ADJ = json.load(open("bridge_names.json"))
j3 = [json.loads(line) for p in sorted(glob.glob("job3-*.jsonl")) for line in open(p)]
j3 = [r for r in j3 if "pairs" in r]


def names(feature, bridge, mode):
    keys = ADJ[bridge]
    toks = feature["lens"] if mode != "top3" else feature["lens"][:3]
    if mode == "exact":
        return any(t.strip().lower() in keys for t in toks)
    return any(any(k in t.lower() for k in keys) for t in toks)


naming = {}
for mode in ("substring", "top3", "exact"):
    n = top1 = top3 = 0
    for r in j3:
        b = cand[r["id"]]["bridge"]
        for p in r["pairs"]:
            own = [f for f in p["top_features"] if f["delta_act"] < 0 and f["along"] > 0][:3]
            n += 1
            flags = [names(f, b, mode) for f in own]
            top1 += bool(flags and flags[0])
            top3 += any(flags)
    naming[mode] = {"pairs": n, "top1_names_bridge": top1, "any_of_top3_names_bridge": top3}
out["bridge_feature_naming"] = {
    "definition": "recipient-side features of a pair = top_features with delta_act < 0 and along > 0, "
                  "first 3 by |along|; 'names the bridge' = a logit-lens token matches the bridge's "
                  "name list in bridge_names.json (country, adjective, capital, Chinese name): "
                  "substring = any key is a substring of any of the 5 lens tokens; top3 = same on "
                  "the first 3 lens tokens; exact = a stripped lowercased token equals a key",
    **naming}
# cross-bridge null: for each feature that is a top-3 recipient-side feature of some recipient,
# the share of that bridge's recipients carrying it vs the share of other bridges' recipients
carry = defaultdict(set)
by_bridge_recips = defaultdict(set)
for r in j3:
    b = cand[r["id"]]["bridge"]
    by_bridge_recips[b].add(r["id"])
    for p in r["pairs"]:
        for f in [f for f in p["top_features"] if f["delta_act"] < 0 and f["along"] > 0][:3]:
            carry[f["f"]].add(r["id"])
null = []
for f, holders in carry.items():
    bridges = Counter(cand[x]["bridge"] for x in holders)
    b, k = bridges.most_common(1)[0]
    nb = len(by_bridge_recips[b])
    if nb < 2 or k < 2:
        continue
    others = len(j3) - nb
    out_k = len(holders) - k
    null.append({"feature": f, "bridge": b, "in_bridge": f"{k}/{nb}",
                 "other_bridges": f"{out_k}/{others}",
                 "other_bridge_list": sorted(set(cand[x]["bridge"] for x in holders) - {b})})
null.sort(key=lambda x: -int(x["in_bridge"].split("/")[0]))
out["bridge_feature_cross_bridge_null"] = null
json.dump(out, open("reconcile.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
