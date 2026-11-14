"""Run 1 scoring (after forecasts-*.json are committed). python r1_score.py MODEL CAPDIR OUT

Inputs: forecasts-1a.json, forecasts-1b.json (+ -l0pred.pt), arms-1a.jsonl, arms-1b.jsonl,
r1_items.json, candidates.json, candidates_ext.json, z1-full.jsonl (content strips), and the
captures in CAPDIR (cap/, arm/). Prereg: next/prereg_run1.json.
"""
import json
import math
import random
import zlib
import statistics as st
import sys
from pathlib import Path

import torch

from mdb.repl.__main__ import _load

MODEL, CAPDIR, OUT = sys.argv[1:4]
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
L, H, KV, DH = ad.layers, ad.num_heads, ad.num_kv_heads, ad.head_dim
blocks = getattr(ad.backbone, ad.spec.layers)
final = getattr(ad.backbone, ad.spec.final_norm)
W_U = ad.model.lm_head.weight.detach()
gamma = final.weight.detach()
tok = dbg.tokenizer
items = {i["id"]: i for i in json.load(open("r1_items.json"))["items"]}
for _i in items.values():  # hard items carry capital_token None; their answer is the capital token
    if _i.get("capital_token") is None and _i.get("group") == "hard":
        _i["capital_token"] = _i["answer"]
base_c = json.load(open("candidates.json"))
ext_c = json.load(open("candidates_ext.json"))
allc = base_c["items"] + ext_c["items"]
COUNTRIES = sorted({i["country_token"] for i in allc if i.get("country_token")})
CAPITALS = sorted(set(base_c["capital_tokens"].values()) | {i["capital_token"] for i in ext_c["items"]
                                                           if i.get("capital_token")}
                  | {i["capital_token"] for i in items.values() if i.get("capital_token")})
ENT = sorted({tok(" " + i["descriptor"], add_special_tokens=False)["input_ids"][0] for i in allc})
ALL = sorted(set(COUNTRIES) | set(CAPITALS) | set(ENT))
g_of = torch.arange(H) // (H // KV)
BRIDGE = {}
for i in allc:
    if i.get("country_token") and i.get("bridge"):
        BRIDGE.setdefault(i["bridge"], {"country": i["country_token"]})
for b_, t_ in base_c["capital_tokens"].items():
    BRIDGE.setdefault(b_, {})["capital"] = t_
L0PRED = {}
for tag in ("1a", "1b"):
    L0PRED.update(torch.load(f"forecasts-{tag}-l0pred.pt", weights_only=False))


def bf16_spacing(x):
    x = x.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(x)) - 7)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


def q(xs):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if len(xs) < 2:
        return {"values": xs}
    qs = st.quantiles(xs, n=4)
    return {"median": st.median(xs), "q1": qs[0], "q3": qs[2], "n": len(xs)}


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
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def best(write, tokens):
    s = (W_U[tokens] * gamma) @ write
    return tokens[int(s.argmax())]


def readout(write, item):
    out = {}
    for name, toks, truth in (("countries", COUNTRIES, item["country_token"]),
                              ("capitals", CAPITALS, item["capital_token"]),
                              ("all", ALL, item["country_token"])):
        b = best(write, toks)
        out[name] = {"best": b, "text": tok.decode([b]), "ok": b == truth}
    out["all"]["kind"] = ("country" if out["all"]["best"] in COUNTRIES else
                          "capital" if out["all"]["best"] in CAPITALS else "entity")
    return out


def cell_write(cap, x, r, g, v=None):
    p = cap[f"L{x}.p"].reshape(H, -1)
    v = cap[f"v{x}"][0, g, r].float() if v is None else v.float().reshape(-1)
    W = blocks[x].self_attn.o_proj.weight
    return sum(W[:, h * DH:(h + 1) * DH] @ (p[h, r] * v) for h in range(H) if int(g_of[h]) == g)


fc = {}
for tag in ("1a", "1b"):
    fc.update(json.load(open(f"forecasts-{tag}.json")))
arms = {}
for tag in ("1a", "1b"):
    for line in open(f"arms-{tag}.jsonl"):
        r = json.loads(line)
        arms[r["id"]] = r
z1 = {json.loads(line)["id"]: json.loads(line) for line in open("z1-full.jsonl")}
res = {"forecast": {}, "lifetime": {}, "contrast": {}, "cuts": {}, "code": {}, "two_by_two": {},
       "coalition": {}, "knockout": {}, "address": {}, "readouts": {}, "hop_ladders": {}}
F = res["forecast"]
per_item_sp, swap_flip, fc_err = [], {"pred": [], "obs": []}, {}
for iid, a in arms.items():
    if iid not in fc or "sweep" not in a:
        continue
    f = fc[iid]["forecast"]
    M0 = a["native"]["logits"][0] - a["native"]["logits"][1]
    xs, ys = [], []
    for key, dm in zip(a["sweep"]["keys"], a["sweep"]["dM"]):
        if key[0] == "cell":
            xs.append(f[f"cell|{key[1]}|{key[2]}|{key[3]}"])
            ys.append(dm)
    big = [i for i, y in enumerate(ys) if abs(y) > 0.5]
    ib = max(range(len(ys)), key=lambda i: abs(ys[i]))
    ifc = max(range(len(xs)), key=lambda i: abs(xs[i]))
    top5 = set(sorted(range(len(ys)), key=lambda i: -abs(ys[i]))[:5])
    per_item_sp.append({"id": iid, "group": a["group"], "sp": spearman(xs, ys),
                        "near_zero_frac": sum(abs(y) < 0.07 for y in ys) / len(ys), "n_big": len(big),
                        "sp_big": spearman([xs[i] for i in big], [ys[i] for i in big]) if len(big) >= 3 else None,
                        "top_cell_same": ib == ifc, "fc_top_in_native_top5": ifc in top5,
                        "ratio_big": st.median(xs[i] / ys[i] for i in big) if big else None})
    for kind in ("t0_zero", "t0_mask", "relay_zero", "relay_mask"):
        dm = a["cuts"][kind]["logits"][0] - a["cuts"][kind]["logits"][1] - M0
        fc_err.setdefault(f"cut {kind}", []).append((f[f"cut|{kind}"], dm, f[f"cut32|{kind}"]))
    for n_, v in a.get("swaps", {}).items():
        dm = v["logits"][0] - v["logits"][1] - M0
        fc_err.setdefault(f"swap {n_}", []).append((f[f"swap|{n_}"], dm, None))
        if n_ == "donor0":
            swap_flip["pred"].append((M0 > 0) != (M0 + f["swap|donor0"] > 0))
            swap_flip["obs"].append((M0 > 0) != (M0 + dm > 0))
    for n_ in ("B", "AB"):
        if n_ in a.get("two_by_two", {}):
            v = a["two_by_two"][n_]
            fc_err.setdefault(f"2x2 {n_}", []).append((f[f"2x2|{n_}"], v["logits"][0] - v["logits"][1] - M0, None))
    for n_, v in a.get("knockout", {}).items():
        fc_err.setdefault(f"knockout {n_}", []).append((f[f"knockout|{n_}"], v["logits"][0] - v["logits"][1] - M0, None))
for g in ("base", "hard"):
    sp = [x["sp"] for x in per_item_sp if x["group"] == g and not math.isnan(x["sp"])]
    F[f"liveness_spearman_{g}"] = q(sp)
    rows_ = [x for x in per_item_sp if x["group"] == g]
    F[f"liveness_diag_{g}"] = {"near_zero_frac": q([x["near_zero_frac"] for x in rows_]),
                               "n_big": q([x["n_big"] for x in rows_]),
                               "sp_big_items_ge3": q([x["sp_big"] for x in rows_ if x["sp_big"] is not None]),
                               "top_cell_same": rate(x["top_cell_same"] for x in rows_),
                               "fc_top_in_native_top5": rate(x["fc_top_in_native_top5"] for x in rows_),
                               "fc_over_native_big": q([x["ratio_big"] for x in rows_ if x["ratio_big"] is not None])}
F["donor_swap_flip_mismatch"] = rate(p != o for p, o in zip(swap_flip["pred"], swap_flip["obs"]))
F["donor_swap_observed_flips"] = sum(swap_flip["obs"])
for k, v in fc_err.items():
    F[k] = {"spearman_fc_vs_native": spearman([x[0] for x in v], [x[1] for x in v]),
            "abs_err_over_abs_native": q([abs(x[0] - x[1]) / max(abs(x[1]), 1e-6) for x in v]),
            "sign_agree": rate((x[0] > 0) == (x[1] > 0) for x in v if abs(x[1]) > 0.25), "n": len(v)}
    if v and v[0][2] is not None:
        F[k]["f32p_abs_err_over_abs_native"] = q([abs(x[2] - x[1]) / max(abs(x[1]), 1e-6) for x in v])

# lifetime (base): causal strip vs Z1 content strip
first_c, first_causal, empty_outside = [], [], 0
for iid, a in arms.items():
    if a["group"] != "base":
        continue
    layers_hit = sorted({k[2] for k, dm in zip(a["sweep"]["keys"], a["sweep"]["dM"])
                         if k[0] == "cell" and k[1] in items[iid]["span"] and abs(dm) > 0.5})
    fcl = layers_hit[0] if layers_hit else None
    if not [x for x in layers_hit if x != 20]:
        empty_outside += 1
    z = z1.get(iid)
    strip = z["table"]["r0rest"]["strip"]["countries"] if z else None
    ok = z["table"]["r0rest"]["content"]["countries"]["names_truth"] if z else False
    fcont = strip.index("C") if (strip and ok and "C" in strip) else None
    first_causal.append(fcl)
    first_c.append(fcont)
res["lifetime"] = {
    "first_causal_layer": q([x for x in first_causal if x is not None]),
    "first_content_C": q([x for x in first_c if x is not None]),
    "causal_later_than_content_by_gt8": rate(c - s > 8 for c, s in zip(first_causal, first_c)
                                             if c is not None and s is not None),
    "single_cell_strip_empty_outside_L20": [empty_outside, len(first_causal)]}

# contrast vs share ranking against the liveness sweep (base + hard)
sp_share, sp_contrast, top_sh, top_co, spb_sh, spb_co = [], [], [], [], [], []
for iid, a in arms.items():
    if iid not in fc:
        continue
    cap = torch.load(Path(CAPDIR) / "cap" / a["group"] / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    toks = cap["toks"].tolist()
    hl = cap[f"L{L - 1}.h_out"].reshape(-1)
    rho = float((hl.pow(2).mean() + final.variance_epsilon) ** -0.5)
    c = rho * gamma * (W_U[toks[0]] - W_U[toks[1]])
    sh, co, ef = [], [], []
    for key, dm in zip(a["sweep"]["keys"], a["sweep"]["dM"]):
        if key[0] != "cell":
            continue
        _, r, x, g = key
        p = cap[f"L{x}.p"].reshape(H, -1)
        A = cap[f"L{x}.A"].reshape(H, DH)
        v = cap[f"v{x}"][0, g, r].float()
        W = blocks[x].self_attn.o_proj.weight
        s_ = sum(float(c @ (W[:, h * DH:(h + 1) * DH] @ (p[h, r] * v))) for h in range(H) if int(g_of[h]) == g)
        c_ = sum(float(c @ (W[:, h * DH:(h + 1) * DH] @ (p[h, r] * (v - A[h])))) for h in range(H)
                 if int(g_of[h]) == g)
        sh.append(abs(s_))
        co.append(abs(c_))
        ef.append(abs(dm))
    if ef:
        itop = max(range(len(ef)), key=lambda i: ef[i])
        top_sh.append(max(range(len(sh)), key=lambda i: sh[i]) == itop)
        top_co.append(max(range(len(co)), key=lambda i: co[i]) == itop)
        bg = [i for i, e_ in enumerate(ef) if e_ > 0.5]
        if len(bg) >= 3:
            spb_sh.append(spearman([sh[i] for i in bg], [ef[i] for i in bg]))
            spb_co.append(spearman([co[i] for i in bg], [ef[i] for i in bg]))
    sp_share.append(spearman(sh, ef))
    sp_contrast.append(spearman(co, ef))
    # address rule and readouts
    it = items[iid]
    span, j = it["span"], int(cap["j"])
    p20 = cap["L20.p"].reshape(H, -1)
    score = {r: float(p20[12, r] + p20[13, r]) for r in span if r != 0}
    r_arg = max(score, key=score.get)
    res["address"].setdefault(a["group"], []).append(
        {"gt": it["row"], "argmax": r_arg, "last": it["last_row"], "first": span[0],
         "random": random.Random(zlib.crc32(iid.encode())).choice(span)})
    rd = res["readouts"].setdefault(a["group"], {"declared": [], "B_answer_h": [], "B_answer_hout": []})
    rd["declared"].append(readout(cell_write(cap, 20, it["row"], 6), it))
    for nm_, port in (("B_answer_h", "L20.h"), ("B_answer_hout", "L20.h_out")):
        rd[nm_].append(readout(cap[port].reshape(-1), it))
    armc = torch.load(Path(CAPDIR) / "arm" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    # mask-cut f32 check at L0 (measured Delta o_0 vs the closed forms, bf16 vs replica-f32 p)
    l0 = L0PRED.get(iid, {})
    for kind in ("t0_zero", "t0_mask", "relay_zero", "relay_mask"):
        key = f"{kind}.L0.o"
        if key not in armc or f"{kind}|bf" not in l0:
            continue
        o_cut = armc[key].reshape(-1)
        meas = o_cut - cap["L0.o"].reshape(-1)
        sp_ = bf16_spacing(o_cut) + bf16_spacing(cap["L0.o"].reshape(-1))
        row_ = {"group": a["group"]}
        for st_ in ("bf", "f32"):
            e = (meas - l0[f"{kind}|{st_}"]).abs()
            row_[f"rel_err_{st_}"] = float(e.norm() / meas.norm().clamp_min(1e-9))
            row_[f"within2_{st_}"] = bool((e <= 2 * sp_).all())
            row_[f"frac_within2_{st_}"] = float((e <= 2 * sp_).float().mean())
        res["cuts"].setdefault(kind, []).append(row_)
    # hop ladders of code swaps (base)
    if a["group"] == "base":
        lad = {}
        for n_ in a.get("swaps", {}):
            pre = f"swap_{n_}."
            if pre + "L20.o" not in armc:
                continue
            d = float(c @ (armc[pre + "L20.o"].reshape(-1) - cap["L20.o"].reshape(-1)))
            mlp = sum(float(c @ (armc[pre + f"L{x}.m"].reshape(-1) - cap[f"L{x}.m"].reshape(-1))) for x in range(20, L))
            rr = sum(float(c @ (armc[pre + f"L{x}.o"].reshape(-1) - cap[f"L{x}.o"].reshape(-1))) for x in range(21, L))
            lg = armc[pre + "logits"]
            nat = float(lg[toks[0]] - lg[toks[1]]) - (float(cap["logits"][toks[0]] - cap["logits"][toks[1]]))
            content = readout(cell_write(cap, 20, it["row"], 6, cap["swap_values"][n_]), it)
            lad[n_] = {"native": nat, "direct": d, "mlp": mlp, "reread": rr, "content": content}
        res["hop_ladders"][iid] = lad
res["contrast"] = {"share_spearman": q(sp_share), "contrast_spearman": q(sp_contrast),
                   "contrast_beats_share": rate(c_ > s_ for c_, s_ in zip(sp_contrast, sp_share)
                                                if not (math.isnan(c_) or math.isnan(s_))),
                   "share_top_cell_is_native_top": rate(top_sh), "contrast_top_cell_is_native_top": rate(top_co),
                   "share_spearman_big": q(spb_sh), "contrast_spearman_big": q(spb_co)}
for g, rows in res["address"].items():
    res["address"][g] = {k: rate(r[k] == r["gt"] for r in rows) for k in ("argmax", "last", "first", "random")}
for g, rd in res["readouts"].items():
    res["readouts"][g] = {k: {"country": rate(x["countries"]["ok"] for x in v),
                              "capital": rate(x["capitals"]["ok"] for x in v),
                              "all": rate(x["all"]["ok"] for x in v),
                              "all_kind": {kk: sum(x["all"]["kind"] == kk for x in v) for kk in ("country", "capital", "entity")}}
                          for k, v in rd.items()}
for kind, rows in res["cuts"].items():
    res["cuts"][kind] = {"rel_err_bf": q([r["rel_err_bf"] for r in rows]),
                         "rel_err_f32": q([r["rel_err_f32"] for r in rows]),
                         "f32_better": rate(r["rel_err_f32"] < r["rel_err_bf"] for r in rows),
                         "within2_bf": rate(r["within2_bf"] for r in rows),
                         "within2_f32": rate(r["within2_f32"] for r in rows),
                         "frac_elems_within2_f32": q([r["frac_within2_f32"] for r in rows]),
                         "frac_elems_within2_bf": q([r["frac_within2_bf"] for r in rows])}
# relay band vs cell transplant; code; 2x2; coalition; knock-out (natives)
rel, code, tbt, coal, ko = [], {}, [], [], {}
for iid, a in arms.items():
    if a["group"] != "base":
        continue
    M0 = a["native"]["logits"][0] - a["native"]["logits"][1]
    band = [dm for k, dm in zip(a["sweep"]["keys"], a["sweep"]["dM"]) if k[0] == "relay_band"][0]
    cell = a["swaps"]["donor0"]["logits"][0] - a["swaps"]["donor0"]["logits"][1] - M0
    rel.append(abs(band) > 0.5 * abs(cell))
    for n_, v in a["swaps"].items():
        oc = BRIDGE.get(a.get("other_country"), {})
        cont = res["hop_ladders"].get(iid, {}).get(n_, {}).get("content")
        code.setdefault(n_, []).append({"names_other": bool(cont) and cont["countries"]["best"] == oc.get("country"),
                                        "flips_other_capital": v["argmax"] == oc.get("capital"),
                                        "dM": v["logits"][0] - v["logits"][1] - M0,
                                        "kept": v["argmax"] == items[iid]["answer"], "family": a["family"],
                                        "content": res["hop_ladders"].get(iid, {}).get(n_, {}).get("content")})
    t = a.get("two_by_two", {})
    if "B" in t:
        d = {k_: t[k_]["logits"][0] - t[k_]["logits"][1] - M0 for k_ in ("A", "B", "AB")}
        tbt.append({"I": d["AB"] - d["A"] - d["B"], "A": d["A"], "B": d["B"]})
    for cpair in a.get("coalition", []):
        coal.append(abs(cpair["pair"]) < abs(sum(cpair["single"])))
    if "knockout" in a:
        d = {k_: v["logits"][0] - v["logits"][1] - M0 for k_, v in a["knockout"].items()}
        for k_, v in d.items():
            ko.setdefault(k_, []).append(v)
        ko.setdefault("named_beats_random", []).append(abs(d["named"]) - abs(d["random"]) >= 0.5)
        ko.setdefault("named_beats_answer_null", []).append(abs(d["named"]) - abs(d["answer_null"]) >= 0.5)
res["cuts"]["relay_band_gt_half_cell"] = rate(rel)
res["code"] = {n_: {"dM": q([x["dM"] for x in v]), "abs_dM": q([abs(x["dM"]) for x in v]),
                    "kept": rate(x["kept"] for x in v),
                    "kept_non_city": rate(x["kept"] for x in v if x["family"] != "city"),
                    "kept_city": rate(x["kept"] for x in v if x["family"] == "city"),
                    "content_country_ok": rate(x["content"]["countries"]["ok"] for x in v if x["content"]),
                    "content_capital_ok": rate(x["content"]["capitals"]["ok"] for x in v if x["content"]),
                    "content_names_other_country": rate(x["names_other"] for x in v),
                    "argmax_is_other_capital": rate(x["flips_other_capital"] for x in v)}
               for n_, v in code.items()}
res["code"]["country_lt_panel"] = None
if "country_mean" in code:
    pairs = []
    for iid, a in arms.items():
        if a["group"] == "base" and "country_mean" in a["swaps"]:
            M0 = a["native"]["logits"][0] - a["native"]["logits"][1]
            dc = a["swaps"]["country_mean"]["logits"][0] - a["swaps"]["country_mean"]["logits"][1] - M0
            dp = a["swaps"]["panel_mean"]["logits"][0] - a["swaps"]["panel_mean"]["logits"][1] - M0
            pairs.append(abs(dc) < abs(dp))
    res["code"]["country_lt_panel"] = rate(pairs)
res["two_by_two"] = {"n": len(tbt), "abs_I_le_quarter_abs_A": rate(abs(x["I"]) <= 0.25 * abs(x["A"]) for x in tbt),
                     "I": q([x["I"] for x in tbt]), "A": q([x["A"] for x in tbt]), "B": q([x["B"] for x in tbt])}
res["coalition"] = {"sub_additive": rate(coal)}
res["knockout"] = {k_: (q(v) if k_ in ("named", "random", "answer_null") else rate(v)) for k_, v in ko.items()}
F["per_item"] = per_item_sp
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("hop_ladders",)}, indent=1, default=str)[:20000])
