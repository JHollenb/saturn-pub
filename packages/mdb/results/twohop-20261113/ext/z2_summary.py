"""Z2 verdicts (prereg_hop.json) from z2-*.jsonl: python3 z2_summary.py FILE.jsonl; writes FILE-summary.json."""
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict

rows = [json.loads(line) for line in open(sys.argv[1])]


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


def q(xs):
    xs = sorted(xs)
    return {"median": st.median(xs), "q1": st.quantiles(xs, n=4)[0], "q3": st.quantiles(xs, n=4)[2],
            "n": len(xs)} if len(xs) > 1 else {"values": xs}


def spacing(x):
    x = abs(x)
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        k = i
        while k + 1 < len(xs) and xs[order[k + 1]] == xs[order[i]]:
            k += 1
        for m in range(i, k + 1):
            r[order[m]] = (i + k) / 2
        i = k + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den


out = {"n": len(rows)}
for arm in ("donor", "ablate", "city"):
    rs = [r for r in rows if arm in r["arms"]]
    if not rs:
        continue
    A = [r["arms"][arm] for r in rs]
    E = [a["edge"] for a in A]
    nat = [e["consumer"]["native_delta"]["value"] for e in E]
    d = {"pairs": len(A)}
    floors = [2 * spacing(max(abs(x) for x in a["M0"] + a["M1"])) for a in A]
    gaps = [e["closure"]["gap"]["value"] for e in E]
    d["closure_within_floor"] = rate(abs(g) <= f for g, f in zip(gaps, floors))
    d["closure_abs_gap"] = q([abs(g) for g in gaps])
    d["residual_add_rounding_abs"] = q([abs(e["residual_add_rounding"]["value"]) for e in E])
    for k in ("direct", "mlp", "reread", "norm"):
        d[f"share_{k}"] = q([e["shares"][k]["value"] for e in E if "value" in e["shares"][k]])
    largest = Counter(max(("direct", "mlp", "reread", "norm"),
                          key=lambda k: abs(e["shares"][k].get("value", 0))) for e in E)
    d["largest_term"] = dict(largest)
    m0 = [a["M0"][0] - a["M0"][1] for a in A]
    m1 = [a["M1"][0] - a["M1"][1] for a in A]
    obs = [(x > 0) != (y > 0) for x, y in zip(m0, m1)]
    pred = [(x > 0) != (x + e["direct"]["value"] + e["totals"]["mlp"]["value"] > 0)
            for x, e in zip(m0, E)]
    d["observed_flips"] = sum(obs)
    d["flip_mismatch"] = rate(o != p for o, p in zip(obs, pred))
    pred_d = [(x > 0) != (x + e["direct"]["value"] > 0) for x, e in zip(m0, E)]
    d["flip_mismatch_direct_only"] = rate(o != p for o, p in zip(obs, pred_d))
    d["spearman_mlp_native"] = spearman([e["totals"]["mlp"]["value"] for e in E], nat)
    d["spearman_direct_native"] = spearman([e["direct"]["value"] for e in E], nat)
    tops = [m["top_share"]["value"] for e in E for m in e["mlp"]]
    d["top8_share_per_layer"] = q(tops)
    per_item_max = [max(m["top_share"]["value"] for m in e["mlp"]) for e in E]
    d["top8_share_ge_0.5_any_layer_items"] = rate(x >= 0.5 for x in per_item_max)
    lin = [e["totals"]["mlp_linear"]["value"] / e["totals"]["mlp_exact"]["value"] for e in E
           if e["totals"]["mlp_exact"]["value"]]
    d["linear_share"] = q(lin)
    d["mlp_by_layer_median"] = {m["layer"]: st.median(e2["mlp"][i]["exact"]["value"] for e2 in E)
                                for i, m in enumerate(E[0]["mlp"])}
    d["shared_gate_abs_median_by_layer"] = {
        m["layer"]: st.median(abs(e2["mlp"][i]["shared_gate"]["value"]) for e2 in E)
        for i, m in enumerate(E[0]["mlp"])}
    if arm == "donor":
        d["native_vs_job2_max_abs"] = max(abs(n - a["job2_delta"]) for n, a in zip(nat, A))
        d["table_hop_matches"] = rate(a["table_hop_matches"] for a in A)
        sp = [m["spectral"] for e in E for m in e["mlp"]]
        d["spectral"] = {
            "credit_top32": q([s["credit_top_bins_share"]["value"] for s in sp]),
            "credit_top32_permuted": q([s["credit_top_bins_share_permuted"]["value"] for s in sp]),
            "native_minus_permuted": q([s["credit_top_bins_share"]["value"]
                                        - s["credit_top_bins_share_permuted"]["value"] for s in sp]),
            "neuron_top192": q([s["neuron_top_share_same_fraction"]["value"] for s in sp]),
            "energy_top32": q([s["energy_top_bins_share"]["value"] for s in sp]),
            "energy_top32_permuted": q([s["energy_top_bins_share_permuted"]["value"] for s in sp]),
            "parseval_max_error": max(s["parseval_max_error"]["value"] for s in sp),
            "neuron_beats_channel": rate(s["neuron_top_share_same_fraction"]["value"]
                                         > s["credit_top_bins_share"]["value"] for s in sp)}
    if arm == "ablate":
        d["against_unbound"] = q([e["totals"]["mlp"]["value"] / a["unbound_mlp"] for e, a in zip(E, A)])
    # recurrence of named neurons within a country
    names = {r["id"]: {n["name"] for m in r["arms"][arm]["edge"]["mlp"] for n in m["neurons"]}
             for r in rs}
    country = {r["id"]: r["bridge"] for r in rs}

    def recurring(labels):
        by = defaultdict(list)
        for i, c in labels.items():
            by[c].append(i)
        found = {}
        for c, ids in by.items():
            if len(ids) < 3:
                continue
            cnt = Counter(n for i in ids for n in names[i])
            rec = sorted(n for n, k in cnt.items() if k >= 3)
            if rec:
                found[c] = rec
        return found
    obs_rec = recurring(country)
    d["recurring_by_country"] = {c: v[:10] for c, v in obs_rec.items()}
    d["countries_with_recurrence"] = len(obs_rec)
    d["recurring_neurons_total"] = sum(len(v) for v in obs_rec.values())
    rng = random.Random(0)
    ids, labs = list(country), list(country.values())
    null = []
    for _ in range(200):
        rng.shuffle(labs)
        null.append(sum(len(v) for v in recurring(dict(zip(ids, labs))).values()))
    d["recurrence_null_mean"] = st.mean(null)
    d["recurrence_null_ge_observed"] = sum(x >= d["recurring_neurons_total"] for x in null) / 200
    # neurons named in every country (generic): named on >= 25% of all items
    allc = Counter(n for v in names.values() for n in v)
    d["generic_neurons(>=25% items)"] = sorted(n for n, k in allc.items() if k >= 0.25 * len(names))[:20]
    out[arm] = d
D = out["donor"]
med = lambda x: x["median"]  # noqa: E731
out["verdicts"] = {
    "H1 closure >= 60/64": D["closure_within_floor"][0] >= 60,
    "H2 mlp share >= 0.5, reread not largest": med(D["share_mlp"]) >= 0.5
    and med(D["share_reread"]) < max(med(D["share_mlp"]), med(D["share_direct"])),
    "H3 against_unbound median in [0.05, 0.5]": 0.05 <= med(out["ablate"]["against_unbound"]) <= 0.5,
    "H4 flip mismatch <= 0.20": D["flip_mismatch"][2] <= 0.20,
    "H5 spearman(mlp, native) >= 0.6": D["spearman_mlp_native"] >= 0.6,
    "H6 top8 share median < 0.5": med(D["top8_share_per_layer"]) < 0.5,
    "H7 recurrence (ablate) >= 4 countries and null < 5%": out["ablate"]["countries_with_recurrence"] >= 4
    and out["ablate"]["recurrence_null_ge_observed"] < 0.05,
    "H8 linear share in [0.3, 0.9]": 0.3 <= med(D["linear_share"]) <= 0.9,
    "H9 |channel - permuted| <= 0.02 and neuron > channel": abs(med(D["spectral"]["native_minus_permuted"])) <= 0.02
    and med(D["spectral"]["neuron_top192"]) > med(D["spectral"]["credit_top32"]),
    "F_mlp": med(D["share_mlp"]) < 0.3 and D["largest_term"].get("reread", 0) == max(D["largest_term"].values()),
    "F_dense": D["top8_share_ge_0.5_any_layer_items"][0] == 0 and out["ablate"]["countries_with_recurrence"] == 0
    and D["countries_with_recurrence"] == 0,
    "F_flip": D["closure_within_floor"][2] >= 0.9 and D["flip_mismatch"][2] > 0.20,
    "F_spectral (retired)": abs(med(D["spectral"]["native_minus_permuted"])) <= 0.02,
}
json.dump(out, open(sys.argv[1].replace(".jsonl", "-summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
