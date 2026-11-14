"""TP1 offline analysis (prereg_tp1.json T1-T4 + falsifier, addendum I1-I4).

python tp1_analyze.py MODEL ARRDIR OUT   (reads ARRDIR/p*.pt, ./tp1-results.json, ./tp1-abl-*.json, ./tp1-tl.json)
"""
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL, ARR, OUT = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(MODEL)
EOT = tok.convert_tokens_to_ids("<|endoftext|>")
FUNC = {"the", "a", "an", "of", "in", "on", "at", "to", "is", "are", "was", "and", "or", "so", "with", "by",
        "for", "from", "as", "that", "this", "it", "its"}
res_main = json.load(open("tp1-results.json"))
P = [torch.load(f, weights_only=False) for f in sorted(glob.glob(f"{ARR}/p*.pt"))]
assert len(P) == len(res_main["summary"]), (len(P), len(res_main["summary"]))
NL = P[0]["h"].shape[0]       # 29: layer inputs 0..27 + final-norm input


def q(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if len(xs) < 2:
        return {"values": xs}
    qs = st.quantiles(xs, n=4)
    return {"median": st.median(xs), "q1": qs[0], "q3": qs[2], "n": len(xs)}


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


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


def tclass(r, t):
    if r == 0:
        return "t0"
    if t == EOT:
        return "special"
    s = tok.decode([t])
    if s.strip() == "":
        return "whitespace"
    if not any(c.isalnum() for c in s):
        return "punct"
    if s.strip().lower() in FUNC:
        return "function"
    return "content"


def pbucket(r):
    for lo, hi, name in ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, 31, "16-31")):
        if lo <= r <= hi:
            return name
    return "32+"


H = [p["h"].float() for p in P]                     # (NL, n, d)
G = [p["grad"].float() for p in P]
hn = [h.norm(dim=-1) for h in H]                    # (NL, n)
gn = [g.norm(dim=-1) for g in G]
ids = [p["ids"] for p in P]
out = {"n_prompts": len(P), "rows": sum(len(i) for i in ids)}

# ---- T1: top-1% energy (||h||) entries over non-last rows, layers 1-27
ent = [(float(hn[k][l, r]), float(gn[k][l, r]), k, r, l) for k in range(len(P)) for l in range(1, 28)
       for r in range(len(ids[k]) - 1)]
ent.sort(key=lambda e: -e[0])
ntop = max(1, len(ent) // 100)
top = ent[:ntop]
out["T1"] = {"n_entries": len(ent), "n_top": ntop,
             "grad_norm_top_median": st.median(e[1] for e in top),
             "grad_norm_all_median": st.median(e[1] for e in ent),
             "top_class_counts": dict(sorted(defaultdict(int, {}).items())),
             "pass": st.median(e[1] for e in top) < st.median(e[1] for e in ent)}
cc = defaultdict(int)
for e in top:
    cc[tclass(e[3], ids[e[2]][e[3]])] += 1
out["T1"]["top_class_counts"] = dict(cc)
ent_nt0 = [e for e in ent if e[3] != 0]
top_nt0 = ent_nt0[:max(1, len(ent_nt0) // 100)]
out["T1"]["non_t0"] = {"grad_norm_top_median": st.median(e[1] for e in top_nt0),
                       "grad_norm_all_median": st.median(e[1] for e in ent_nt0),
                       "top_class_counts": dict(defaultdict(int, {}))}
cc = defaultdict(int)
for e in top_nt0:
    cc[tclass(e[3], ids[e[2]][e[3]])] += 1
out["T1"]["non_t0"]["top_class_counts"] = dict(cc)

# ---- falsifier: per prompt Spearman(||h||, ||g||*||h||) over non-t0 rows x layers 1-27, median over prompts
sp_all, sp_nolast, sp_g = [], [], []
for k in range(len(P)):
    n = len(ids[k])
    for rows, dest in ((range(1, n), sp_all), (range(1, n - 1), sp_nolast)):
        a = [float(hn[k][l, r]) for l in range(1, 28) for r in rows]
        b = [float(hn[k][l, r] * gn[k][l, r]) for l in range(1, 28) for r in rows]
        dest.append(spearman(a, b))
    a = [float(hn[k][l, r]) for l in range(1, 28) for r in range(1, n - 1)]
    b = [float(gn[k][l, r]) for l in range(1, 28) for r in range(1, n - 1)]
    sp_g.append(spearman(a, b))
out["falsifier"] = {"spearman_h_vs_gh_non_t0": q(sp_all), "spearman_h_vs_gh_non_t0_non_last": q(sp_nolast),
                    "spearman_h_vs_g_non_t0_non_last": q(sp_g),
                    "fires": st.median(x for x in sp_all if not math.isnan(x)) > 0.5}

# ---- T2: constancy by position vs by token at mid layers
T2 = {}
for l in (7, 14, 21):
    by_pos, by_tok = defaultdict(list), defaultdict(list)
    for k in range(len(P)):
        for r, t in enumerate(ids[k]):
            by_pos[r].append(H[k][l, r])
            by_tok[t].append(H[k][l, r])
    pm = {r: (torch.stack(v).mean(0), len(v)) for r, v in by_pos.items()}
    tm = {t: (torch.stack(v).mean(0), len(v)) for t, v in by_tok.items()}
    occ_elsewhere = defaultdict(int)
    for k in range(len(P)):
        for r, t in enumerate(ids[k]):
            if r > 0:
                occ_elsewhere[t] += 1
    r0, fw = [], []
    for k in range(len(P)):
        for r, t in enumerate(ids[k]):
            h = H[k][l, r]
            cp = float(torch.nn.functional.cosine_similarity(h, pm[r][0], dim=0))
            ct = float(torch.nn.functional.cosine_similarity(h, tm[t][0], dim=0))
            if r == 0 and occ_elsewhere[t] > 0:
                r0.append((cp, ct, pm[r][1], tm[t][1]))
            if r >= 2 and r < len(ids[k]) - 1 and tclass(r, t) == "function" and tm[t][1] >= 3 and pm[r][1] >= 3:
                fw.append((cp, ct, pm[r][1], tm[t][1]))
    T2[f"L{l}"] = {"row0": {"n": len(r0), "cos_pos": q([x[0] for x in r0]), "cos_tok": q([x[1] for x in r0]),
                            "pos_gt_tok": rate(x[0] > x[1] for x in r0)},
                   "function_mid": {"n": len(fw), "cos_pos": q([x[0] for x in fw]),
                                    "cos_tok": q([x[1] for x in fw]), "tok_gt_pos": rate(x[1] > x[0] for x in fw),
                                    "n_pos_mean": q([x[2] for x in fw]), "n_tok_mean": q([x[3] for x in fw])}}
out["T2"] = T2

# ---- T3: ablations
abl = []
for f in sorted(glob.glob("tp1-abl-*.json")):
    abl += json.load(open(f))["ablations"]
out["ablations_n"] = len(abl)
T3 = {}
for tag in ("top1pct", "median"):
    rows = [a for a in abl if a["set"] == tag]
    both = [a for a in rows if a.get("position") and a.get("global")]
    T3[tag] = {"n": len(rows),
               "pos_lt_global": rate(abs(a["position"]["dM"]) < abs(a["global"]["dM"]) for a in both),
               "abs_dM_global": q([abs(a["global"]["dM"]) for a in rows if a.get("global")]),
               "abs_dM_position": q([abs(a["position"]["dM"]) for a in rows if a.get("position")]),
               "abs_dM_token": q([abs(a["token"]["dM"]) for a in rows if a.get("token")]),
               "kept_global": rate(a["global"]["argmax_kept"] for a in rows if a.get("global")),
               "kept_position": rate(a["position"]["argmax_kept"] for a in rows if a.get("position")),
               "kept_token": rate(a["token"]["argmax_kept"] for a in rows if a.get("token")),
               "classes": dict(defaultdict(int))}
    cc = defaultdict(int)
    for a in rows:
        cc[tclass(a["row"], a["token"])] += 1
    T3[tag]["classes"] = dict(cc)
T3["pass"] = (T3["top1pct"]["pos_lt_global"][2] or 0) >= 0.7
out["T3"] = T3

# ---- T4: pseudo-BOS takes over the sink
summ = res_main["summary"]
T4 = {}
for kind in ("plain", "lead_The", "lead_newline", "lead_eot"):
    ks = [k for k in range(len(P)) if summ[k]["kind"] == kind]
    row0_max, ratio1 = [], []
    for k in ks:
        n = len(ids[k])
        for l in range(1, 28):
            row0_max.append(int(hn[k][l, :n - 1].argmax()) == 0)
        med = float(hn[k][14, 2:n - 1].median()) if n > 3 else float("nan")
        ratio1.append(float(hn[k][14, 1]) / med)
    T4[kind] = {"n_prompts": len(ks), "row0_is_max_norm_layers1_27": rate(row0_max),
                "row1_norm_over_median_L14": q(ratio1),
                "row0_token": sorted({tok.decode([ids[k][0]]) for k in ks})[:6],
                "row1_token": sorted({tok.decode([ids[k][1]]) for k in ks})[:6]}
out["T4"] = T4

# ---- class map at L14 (and t0 / sink across layers)
cmap = defaultdict(lambda: defaultdict(list))
for k in range(len(P)):
    n = len(ids[k])
    recv = P[k]["received"]          # (L, n) attention mass received, all heads and query rows
    for r in range(n - 1):
        c = tclass(r, ids[k][r])
        for key, val in (("h_norm", hn[k][14, r]), ("grad_norm", gn[k][14, r]),
                         ("grad_dot_h", (G[k][14, r] * H[k][14, r]).sum()),
                         ("received_L14", recv[14, r] / max(1, n - r)), ("vnorm_L14", P[k]["vnorm"][14, r]),
                         ("centred_frac_L14", (H[k][14, r] - H[k][14, r].mean()).norm() / hn[k][14, r])):
            cmap[c][key].append(float(val))
        for key, val in (("h_norm", hn[k][14, r]), ("grad_norm", gn[k][14, r])):
            cmap["pos " + pbucket(r)][key].append(float(val))
out["class_map_L14"] = {c: {k_: q(v) for k_, v in d.items()} for c, d in cmap.items()}

# ---- I1 (from main), I2 (TL + identity), I3 (SAE), I4 (SAE check)
chk = res_main["checks"]
out["I1"] = [{"prompt": c["prompt"], "direction": c["direction"], "nnsight": c["nnsight_grad_dot_u"],
              "mdb_linear": c["mdb_linear"],
              "rel_diff": abs(c["nnsight_grad_dot_u"] - c["mdb_linear"]) / abs(c["mdb_linear"])} for c in chk]
cf = []
for e in top:
    k, r, l = e[2], e[3], e[4]
    h = H[k][l, r]
    cf.append(float((h - h.mean()).norm() / h.norm()))
out["I2"] = {"top1pct_centred_frac": q(cf)}
shares = []
for l in (1, 7, 14, 21, 27):
    allh = torch.cat([H[k][l, :len(ids[k]) - 1] for k in range(len(P))])
    e_nat = allh.norm(dim=-1) ** 2
    c = allh - allh.mean(-1, keepdim=True)
    e_c = c.norm(dim=-1) ** 2
    idx = e_nat.argsort(descending=True)[:max(1, len(e_nat) // 100)]
    shares.append({"layer": l, "share_native": float(e_nat[idx].sum() / e_nat.sum()),
                   "share_centred": float(e_c[idx].sum() / e_c.sum())})
out["I2"]["top1pct_energy_share"] = shares
try:
    tl = json.load(open("tp1-tl.json"))["tl"]
    out["I2"]["tl_state"] = tl.get("state")
    out["I2"]["tl_identity_rel_err"] = q([r_["identity_rel_err"] for r_ in tl.get("rows", [])])
    out["I2"]["tl_processed"] = tl.get("processed")
except FileNotFoundError:
    out["I2"]["tl_state"] = "no tp1-tl.json"
sae = res_main["sae"]
I3 = {}
for layer in sae["layers"]:
    hidx = layer + 1
    art, oth, t0e = [], [], []
    thr = sorted((float(hn[k][hidx, r]) for k in range(len(P)) for r in range(len(ids[k]) - 1)), reverse=True)
    cut = thr[max(1, len(thr) // 100) - 1]
    for k in range(len(P)):
        errs = sae["per_prompt"][str(k)][str(layer)] if str(k) in sae["per_prompt"] else \
            sae["per_prompt"][k][layer]
        for r in range(len(ids[k]) - 1):
            (art if float(hn[k][hidx, r]) >= cut else oth).append(errs[r])
            if r == 0:
                t0e.append(errs[r])
    I3[f"L{layer}"] = {"top1pct": q(art), "others": q(oth), "t0": q(t0e),
                       "top_gt_others": st.median(art) > st.median(oth)}
I3["layers_top_gt_others"] = sum(v["top_gt_others"] for v in I3.values() if isinstance(v, dict))
out["I3"] = I3
out["I4"] = rate(c["same_active_set"] for c in sae["check"])
json.dump(out, open(OUT, "w"), indent=1, default=str)
print(json.dumps(out, default=str)[:3000])
