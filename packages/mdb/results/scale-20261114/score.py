"""Score one model of the scale study (prereg_scale.json 'operational'), CPU.

python score.py MODEL_DIR CAPDIR SWEEP_PREFIXES CAPINFO_JSONS FORECAST_PREFIXES ARMS_JSONS OUT
(comma lists where plural). Everything per family; nothing pooled is a verdict.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch

from mdb.analysis import circuit_map as cm
from mdb.analysis.reads import layer_module
from mdb.repl.__main__ import _load
import scale_common as sc

MODEL, CAPDIR, SWP, CINF, FCP, ARMS, OUT = sys.argv[1:8]
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cpu", True, None)
ad = dbg.adapter
L, H, KV = ad.layers, ad.num_heads, ad.num_kv_heads
G = H // KV
final = getattr(ad.backbone, ad.spec.final_norm)
gamma, eps_f = final.weight.double(), float(final.variance_epsilon)
W_U = ad.model.lm_head.weight
P = json.load(open("panel.json"))
items = P["items"]
by_id = {i["id"]: i for i in items}
B1T = json.load(open("b1_sets.json"))["truths"]
SETS = P["sets"]
RO = {k: gamma[None, :] * W_U[v].double() for k, v in SETS.items()}
capinfo = sc.capture_info(CINF.split(","))
correct = sc.correct_map(P, capinfo)
sweep = sc.load_sweep(SWP.split(","))
car = sc.carriers(P, sweep, correct)
LC, KC = car["L_c"], car["kv_c"]
fc_t, fc_j = {}, {}
for p in FCP.split(","):
    fc_t.update(torch.load(p + ".pt", weights_only=False))
    fc_j.update(json.load(open(p + ".json")))
sw_j = {}
for p in SWP.split(","):
    sw_j.update(json.load(open(p + ".json")))
arms = {}
for p in ARMS.split(","):
    d = json.load(open(p))
    for k, v in d["items"].items():
        arms.setdefault(k, {}).update(v)
WR = {x: cm.group_writers(layer_module(ad, x).self_attn.o_proj.weight, H, KV) for x in range(L)}
FAMS = list(P["counts"])
rng = random.Random(0)
TOKS = json.load(open(CINF.split(",")[0]))["tokens"]
TIX = {t: i for i, t in enumerate(TOKS)}


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def iqr(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 4:
        return None
    q = st.quantiles(xs, n=4)
    return [q[0], q[2]]


def ranks(x):
    return torch.argsort(torch.argsort(x)).double()


def spearman(a, b):
    if a.numel() < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return float(ra @ rb) / den if den else None


def name(setk, w):
    """index of the best candidate of set setk for write w (d,)"""
    return int((RO[setk] @ w).argmax())


def rank_of(setk, w, tok):
    if tok not in SETS[setk]:
        return None
    s = RO[setk] @ w
    return int((s > s[SETS[setk].index(tok)]).sum()) + 1


def write(cap, x, k, r):
    return WR[x][k] @ cap["v"][x][k, r].double()


def proj_out(u):
    u = u / u.norm()
    return lambda w: w - u * (u @ w)


per = {}
caps = {}
for it in items:
    iid = it["id"]
    cap = sc.load_cap(CAPDIR, iid)
    caps[iid] = {"v": cap["v"], "rows": {r: {"p": cap["rows"][r]["p"]} for r in cap["rows"]},
                 "ans_p": cap["ans"]["p"], "ans_h": cap["ans"]["h"], "logits": cap["logits"]}
    del cap

# ---------------------------------------------------------------- per item
for it in items:
    iid, fam = it["id"], it["family"]
    cap = caps[iid]
    span, last = it["span"], len(it["ids"]) - 1
    r_at = car["row_at_item"][iid]
    ctok = it["country_token"]
    w = write(cap, LC, KC, r_at)
    rec = {"family": fam, "block": it["block"], "correct": correct[iid], "row_at_item": r_at,
           "item_carrier": car["item"][iid]}
    rec["country_rank"] = rank_of("countries", w, ctok)
    rec["names_country"] = rec["country_rank"] == 1
    rec["readable_error"] = (not correct[iid]) and (rec["country_rank"] is None or rec["country_rank"] > 3)
    # co-readouts with the country direction projected out of write and candidates
    u = gamma * W_U[ctok].double()
    po = proj_out(u)
    co = {}
    capital = it["capital_token"]
    targets = {"cities": capital}
    if it["block"] == "depth":
        targets["languages"] = it["answer"]
        if B1T.get(iid, {}).get("persons_surname"):
            targets["persons_surname"] = B1T[iid]["persons_surname"]
    for sk, tt in targets.items():
        if tt is None or tt not in SETS[sk]:
            continue
        raw = SETS[sk][name(sk, w)] == tt
        ROp = torch.stack([po(rw) for rw in RO[sk]])
        prj = SETS[sk][int((ROp @ po(w)).argmax())] == tt
        lex = SETS[sk][name(sk, u)] == tt
        co[sk] = {"raw": raw, "projected": prj, "lexical_null": lex, "counts": prj and not lex}
    rec["co"] = co
    # L_first at the carrier row, layers 0..L_c
    names_by_layer = [{name("countries", write(cap, x, k, r_at)) for k in range(KV)} for x in range(LC + 1)]
    ti = SETS["countries"].index(ctok)
    fake = rng.choice([i for i in range(len(SETS["countries"])) if i != ti])
    first = lambda c: next((x for x in range(LC + 1) if c in names_by_layer[x]), None)
    stable = next((x for x in range(LC + 1) if all(ti in names_by_layer[y] for y in range(x, LC + 1))), None)
    rec["L_first"], rec["L_first_fake"], rec["L_first_stable"] = first(ti), first(fake), stable
    # write window and concentration (sweep)
    s = sweep[iid]
    ic = car["item"][iid]
    rr = ic["cell"][2]
    sel = s["cells"][:, 2] == rr
    above = [int(s["cells"][j, 0]) for j in torch.nonzero(sel).reshape(-1)
             if abs(float(s["dM"][j])) > sc.floor(s["la"][j], s["lb"][j])]
    rec["write_window"] = (max(above) - min(above) + 1) if above else None
    band = sw_j[iid]["band_dM"]
    order = s["dM"].argsort()
    rec["concentration"] = abs(float(s["dM"][order[0]])) / abs(band) if band else None
    rec["top2_share"] = (abs(float(s["dM"][order[0]])) + abs(float(s["dM"][order[1]]))) / abs(band) if band else None
    c1, c2 = s["cells"][order[0]].tolist(), s["cells"][order[1]].tolist()
    rec["runner_up"] = c2
    rec["top2_adjacent_same_row"] = c1[2] == c2[2] and abs(c1[0] - c2[0]) == 1
    rec["band_dM"] = band
    # forecasts: effect size, hit@k
    f = fc_t.get(iid)
    if f is not None:
        assert torch.equal(f["cells"], s["cells"]), iid
        fl = torch.tensor([sc.floor(a, b) for a, b in zip(s["la"].tolist(), s["lb"].tolist())])
        m = s["dM"].abs() > fl
        nat = s["dM"][m].double()
        out = {"n_cells": int(m.sum())}
        for key in ("fo", "prop"):
            fv = f[key][m].double()
            rho = spearman(fv, nat)
            sh = []
            for _ in range(200):
                perm = torch.tensor(rng.sample(range(fv.numel()), fv.numel())) if fv.numel() else None
                sh.append(spearman(fv[perm], nat) if perm is not None else None)
            sh = sorted(x for x in sh if x is not None)
            out[key] = {"spearman": rho, "shuffle95": sh[int(0.95 * len(sh))] if sh else None,
                        "hit1": int(f[key].argmin()) == int(order[0]),
                        "hit3": int(order[0]) in [int(i) for i in f[key].argsort()[:3]]}
        j0 = int(order[0])
        out["fo_over_native_at_carrier"] = float(f["fo"][j0]) / float(s["dM"][j0]) if float(s["dM"][j0]) else None
        reps = []
        for arm in fc_j[iid]["arms"]:
            idx = [n for n in range(len(s["cells"])) if s["cells"][n].tolist() == arm["cell"]][0]
            reps.append({"cell": arm["cell"], "native": float(s["dM"][idx]), "fo": arm["fo"],
                         "replica": arm["replica_dM"], "prop": arm["prop"]})
        out["replica_arms"] = reps
        out["replica_null_err"] = fc_j[iid]["replica_null_err"]
        rec["forecast"] = out
    # row rules
    pa = cap["ans_p"][LC]                                            # (H, rows)
    grp = [h for h in range(H) if h // G == KC]
    cand = [r for r in span if r != 0]
    mass = {r: float(pa[grp, r].sum()) for r in cand}
    marg = {}
    for r in cand:
        sc_ = RO["countries"] @ write(cap, LC, KC, r)
        oth = sc_.clone()
        oth[ti] = -float("inf")
        marg[r] = float(sc_[ti] - oth.max())
    rules = {"R_argmax": max(cand, key=lambda r: (mass[r], r)), "R_last": span[-1], "R_first": span[0],
             "S": max(cand, key=lambda r: (mass[r] * marg[r], r))}
    rec["rows"] = {k: {"row": v, "is_causal": v == r_at,
                       "names": name("countries", write(cap, LC, KC, v)) == ti} for k, v in rules.items()}
    # site nulls
    nonsink = pa[grp, 1:last].sum(0)
    mrow = 1 + int(nonsink.argmax())
    sh_cell = rng.choice([c["cell"][:2] for c in car["item"].values()])
    lens = {}
    for xx in (LC, LC + 4):
        if xx < L:
            h = cap["ans_h"][xx].double()
            hn = h * torch.rsqrt(h.pow(2).mean() + eps_f)
            lens[f"lens_L{xx}"] = name("countries", hn) == ti
    rec["nulls"] = {"fixed_last_span": name("countries", write(cap, LC, KC, span[-1])) == ti,
                    "mass_row": name("countries", write(cap, LC, KC, mrow)) == ti, "mass_row_is": mrow,
                    "random_span_row": sum(name("countries", write(cap, LC, KC, r)) == ti for r in span) / len(span),
                    "shuffled_cell": name("countries", write(cap, sh_cell[0], sh_cell[1], r_at)) == ti,
                    **lens}
    # mass on the carrier row, non-sink renormalised
    tot = float(nonsink.sum())
    rec["carrier_mass_nonsink"] = float(pa[grp, r_at].sum()) / tot if tot else None
    if fam == "dist-city":
        dd = it["distractor"]
        di = SETS["countries"].index(dd["country_token"]) if dd["country_token"] in SETS["countries"] else None
        rec["dist"] = {"distractor_names_its_country": any(name("countries", write(cap, LC, KC, r)) == di for r in dd["rows"]),
                       "mass_distractor": sum(float(pa[grp, r].sum()) for r in dd["rows"]) / tot if tot else None,
                       "mass_true": rec["carrier_mass_nonsink"]}
    # upstream slot typing (cells below L_c at span rows, causal above floor)
    if fam in ("d3-per", "d4-work", "d3-lm"):
        t1 = B1T.get(iid, {})
        want = {"countries": ctok, "cities": t1.get("city"), "persons_surname": t1.get("persons_surname")}
        firsts = {}
        for sk, tt in want.items():
            if tt is None or tt not in SETS[sk]:
                continue
            ci = SETS[sk].index(tt)
            hit = None
            for j in range(len(s["cells"])):
                x, k, r = s["cells"][j].tolist()
                if x >= LC or abs(float(s["dM"][j])) <= sc.floor(s["la"][j], s["lb"][j]):
                    continue
                if name(sk, write(cap, x, k, r)) == ci and (hit is None or x < hit):
                    hit = x
            firsts[sk] = hit
        rec["slot_first_causal"] = firsts
    # arms
    a = arms.get(iid, {})
    if "closure" in a:
        cl = a["closure"]
        rec["closure"] = {"same_keeps": cl.get("same_country_mean", {}).get("argmax") == it["answer"] if "same_country_mean" in cl else None,
                          "donor_moves": cl.get("donor_country_mean", {}).get("argmax") == by_id[it["donor"]]["answer"] if "donor_country_mean" in cl else None,
                          "repair_dist": cl.get("true_cell_into_distractor", {}).get("argmax") == it["answer"] if "true_cell_into_distractor" in cl else None}
    if "backpatch" in a:
        rec["backpatch_repaired"] = a["backpatch"]["argmax"] == it["answer"]
    if "b6" in a:
        lg = cap["logits"]
        toks = SETS["cities"] if it["block"] == "base" else SETS["languages"]
        rest = [t for t in toks if t != it["answer"]]
        nat_m = float(lg[TIX[it["answer"]]]) - max(float(lg[TIX[t]]) for t in rest)
        rec["b6_drop"] = nat_m - (a["b6"]["a"] - a["b6"]["max_rest"])
    for dose in ("0.5", "1.0"):
        if dose in a:
            e = a[dose]
            nd = e["native_delta"]
            rec.setdefault("split", {})[dose] = {k: (e[k] / nd if nd else None) for k in ("direct", "mlp", "reread", "norm")}
            rec["split"][dose]["native_delta"] = nd
    per[iid] = rec

# cross-item null: same (index from span end, L_c, kv_c) in the next family item with another country
for fam in FAMS:
    ids = [i["id"] for i in items if i["family"] == fam]
    for n_, iid in enumerate(ids):
        it = by_id[iid]
        other = next((j for j in ids[n_ + 1:] + ids[:n_] if by_id[j]["bridge"] != it["bridge"]), None)
        if other is None:
            continue
        k_from_end = it["span"][::-1].index(per[iid]["row_at_item"])
        osp = by_id[other]["span"]
        if k_from_end < len(osp):
            r2 = osp[::-1][k_from_end]
            per[iid]["nulls"]["cross_item"] = name("countries", write(caps[other], LC, KC, r2)) == SETS["countries"].index(it["country_token"])

# projected-language chance (derangement of countries over depth items)
dep = [i for i in items if i["block"] == "depth"]
perm = list(range(len(dep)))
while any(p == q for p, q in enumerate(perm)) or any(dep[p]["bridge"] == dep[q]["bridge"] for p, q in enumerate(perm)):
    rng.shuffle(perm)
lang_shuf = 0
for p, q in enumerate(perm):
    it, ot = dep[p], dep[q]
    w = write(caps[it["id"]], LC, KC, per[it["id"]]["row_at_item"])
    po = proj_out(gamma * W_U[ot["country_token"]].double())
    ROp = torch.stack([po(rw) for rw in RO["languages"]])
    lang_shuf += SETS["languages"][int((ROp @ po(w)).argmax())] == it["answer"]


# ---------------------------------------------------------------- per family tables
def frac(xs):
    xs = [x for x in xs if x is not None]
    return [sum(bool(x) for x in xs), len(xs)]


fam_tab = {}
for fam in FAMS:
    rs = [per[i["id"]] for i in items if i["family"] == fam]
    ok = [r for r in rs if r["correct"]]
    bad = [r for r in rs if not r["correct"]]
    t = {"n": len(rs), "correct": len(ok), "acc": len(ok) / len(rs),
         "carrier_agree_correct": frac([tuple(r["item_carrier"]["cell"][:2]) == (LC, KC) for r in ok]),
         "names_country_correct": frac([r["names_country"] for r in ok]),
         "names_country_wrong": frac([r["names_country"] for r in bad]),
         "readable_error_share_wrong": frac([r["readable_error"] for r in bad]),
         "write_window_le5_correct": frac([r["write_window"] is not None and r["write_window"] <= 5 for r in ok]),
         "concentration_correct": {"median": med([r["concentration"] for r in ok]), "iqr": iqr([r["concentration"] for r in ok])},
         "top2_share_correct": med([r["top2_share"] for r in ok]),
         "top2_adjacent_same_row": frac([r["top2_adjacent_same_row"] for r in ok]),
         "L_first": {"med_correct": med([r["L_first"] if r["L_first"] is not None else LC + 1 for r in ok]),
                     "med_wrong": med([r["L_first"] if r["L_first"] is not None else LC + 1 for r in bad]),
                     "med_fake_correct": med([r["L_first_fake"] if r["L_first_fake"] is not None else LC + 1 for r in ok]),
                     "med_stable_correct": med([r["L_first_stable"] if r["L_first_stable"] is not None else LC + 1 for r in ok])},
         "co": {}, "nulls": {}, "rows": {}, "forecast": {}, "closure": {}}
    for sk in ("cities", "languages", "persons_surname"):
        cs = [r["co"][sk] for r in rs if sk in r["co"]]
        if cs:
            t["co"][sk] = {k: frac([c[k] for c in cs]) for k in ("raw", "projected", "lexical_null", "counts")}
    for nk in ("fixed_last_span", "mass_row", "shuffled_cell", "cross_item", f"lens_L{LC}", f"lens_L{LC + 4}"):
        t["nulls"][nk] = frac([r["nulls"].get(nk) for r in rs])
    t["nulls"]["random_span_row_expect"] = sum(r["nulls"]["random_span_row"] for r in rs)
    t["nulls"]["carrier_names"] = frac([r["names_country"] for r in rs])
    for rk in ("R_argmax", "R_last", "R_first", "S"):
        t["rows"][rk] = {"is_causal": frac([r["rows"][rk]["is_causal"] for r in rs]),
                         "names": frac([r["rows"][rk]["names"] for r in rs])}
    diff = [r for r in rs if len({r["rows"][k]["row"] for k in r["rows"]}) > 1]
    t["rows"]["where_rules_differ"] = {rk: frac([r["rows"][rk]["is_causal"] for r in diff]) for rk in ("R_argmax", "R_last", "R_first", "S")}
    fr = [r["forecast"] for r in rs if "forecast" in r]
    for key in ("fo", "prop"):
        t["forecast"][key] = {"spearman_med": med([f[key]["spearman"] for f in fr]),
                              "shuffle95_med": med([f[key]["shuffle95"] for f in fr]),
                              "hit1": frac([f[key]["hit1"] for f in fr]), "hit3": frac([f[key]["hit3"] for f in fr])}
    t["forecast"]["fo_over_native"] = {"median": med([f["fo_over_native_at_carrier"] for f in fr]),
                                       "iqr": iqr([f["fo_over_native_at_carrier"] for f in fr])}
    rep = [a for f in fr for a in f["replica_arms"] if abs(a["native"]) > 0]
    t["forecast"]["replica_over_native"] = {"median": med([a["replica"] / a["native"] for a in rep]),
                                            "iqr": iqr([a["replica"] / a["native"] for a in rep])}
    t["forecast"]["fo_over_native_replica_cells"] = {"median": med([a["fo"] / a["native"] for a in rep])}
    t["forecast"]["replica_null_err_max"] = max([f["replica_null_err"] for f in fr], default=None)
    t["closure"] = {"same_keeps": frac([r.get("closure", {}).get("same_keeps") for r in rs]),
                    "donor_moves": frac([r.get("closure", {}).get("donor_moves") for r in rs])}
    t["b6_drop"] = {"median": med([r.get("b6_drop") for r in rs])}
    t["backpatch"] = {"late": frac([r.get("backpatch_repaired") for r in bad if r["L_first"] is not None]),
                      "never": frac([r.get("backpatch_repaired") for r in bad if r["L_first"] is None])}
    if fam == "dist-city":
        t["dist_wrong"] = {"true_carrier_names_true": frac([r["names_country"] for r in bad]),
                           "distractor_names_its": frac([r["dist"]["distractor_names_its_country"] for r in bad]),
                           "mass_distractor_gt_true": frac([(r["dist"]["mass_distractor"] or 0) > (r["dist"]["mass_true"] or 0) for r in bad]),
                           "repair": frac([r.get("closure", {}).get("repair_dist") for r in bad])}
    sp = {d: [r["split"][d] for r in ok if "split" in r and d in r["split"]] for d in ("0.5", "1.0")}
    t["split"] = {d: {k: {"median": med([x[k] for x in v]), "iqr": iqr([x[k] for x in v])} for k in ("direct", "mlp", "reread", "norm")}
                  for d, v in sp.items() if v}
    if fam in ("d3-per", "d4-work", "d3-lm"):
        t["slot_first_causal"] = {sk: {"med_correct": med([r["slot_first_causal"].get(sk) for r in ok]),
                                       "med_wrong": med([r["slot_first_causal"].get(sk) for r in bad]),
                                       "found": frac([r["slot_first_causal"].get(sk) is not None for r in rs])}
                                  for sk in ("persons_surname", "cities", "countries")}
    fam_tab[fam] = t

# model-level (still per family underneath)
deep = [per[i["id"]] for i in items if i["family"] in ("d3-lm", "d3-per", "d4-work")]
slack = lambda r: -1 if r["L_first"] is None else LC - r["L_first"]
pos = [slack(r) for r in deep if r["correct"]]
neg = [slack(r) for r in deep if not r["correct"]]
auc = (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))) if pos and neg else None
spl = {d: [per[i["id"]]["split"][d] for i in items if per[i["id"]].get("split", {}).get(d)] for d in ("0.5", "1.0")}
both = [i["id"] for i in items if "split" in per[i["id"]] and len(per[i["id"]]["split"]) == 2]
b6 = {"language_drop_median": med([per[i["id"]].get("b6_drop") for i in items if i["block"] == "depth"]),
      "capital_drop_median": med([per[i["id"]].get("b6_drop") for i in items if i["block"] == "base"])}
model = {"model": Path(MODEL).name, "layers": L, "heads": H, "kv": KV, "L_c": LC, "kv_c": KC,
         "frac_depth": LC / L, "tail": L - LC, "mode": [car["mode_n"], car["mode_of"]], "mode_counts": car["mode_counts"],
         "split": {d: {k: {"median": med([x[k] for x in v]), "iqr": iqr([x[k] for x in v])} for k in ("direct", "mlp", "reread", "norm")}
                   for d, v in spl.items() if v},
         "split_dose_move_median": med([abs(per[i]["split"]["1.0"]["mlp"] - per[i]["split"]["0.5"]["mlp"]) for i in both
                                        if per[i]["split"]["1.0"]["mlp"] is not None and per[i]["split"]["0.5"]["mlp"] is not None]),
         "B2_auc_slack_d3d4": auc, "B6": b6,
         "projected_language_hits": frac([per[i["id"]]["co"].get("languages", {}).get("projected") for i in dep]),
         "projected_language_shuffled_hits": [lang_shuf, len(dep)]}
json.dump({"model": model, "families": fam_tab, "per_item": per, "carriers": car}, open(OUT, "w"), indent=1, default=str)
print(json.dumps(model, indent=1, default=str))
