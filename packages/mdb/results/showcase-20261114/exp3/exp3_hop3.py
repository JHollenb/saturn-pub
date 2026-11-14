"""Job C (exp3/prereg.json): transcoder features nominate answer-row MLP writers of the
language, mdb measures them.  python exp3_hop3.py OUTDIR [PICK]

Pass 1 (per item, 132 = 105 correct d2-d4 + 27 wrong d3/d4): native run to the logits; at the
answer row, the mwhanna lowl0 transcoders (Qwen3-1.7B post-trained; L20..L25) encode t<last>.L<l>.n
(SAELens hook mlp.hook_in = mdb n; ReLU, no b_dec on the input) and every active feature's
direct effect rho_F c . a_s d_s on M[language, runner-up language] and M[city, runner-up city]
is computed in float64. The whole-MLP writes m at L20..25 are kept for the mean ablation.
Nomination (frozen rule, prereg): per layer, features with direct effect >= 0.25 on the item's
language margin that do so on >= 3 items of the same language and on <= 1/50 of the items of
other languages; per item the top 5 such features by direct effect.
Pass 2 (per item with >= 1 nominated feature): from one checkpoint at t<last>.L20.h, arms
(supplied vector edits via analysis.supplied.vector_edit): each nominated feature's write
removed (t<last>.L<l>.m add -a d), the joint top-5 removal, a random active non-nominated
transcoder feature at the top feature's layer rescaled to the top write's norm, and the
self-excluded mean ablation of the whole MLP at the top feature's layer. The joint arm is also
run as its own run and split by symbols.hop_edge (direct, later MLP, re-read, norm, rounding;
neurons=False, and neurons=True on the first 8 correct items).
The log prints item ids and seconds only.
"""
import collections
import json
import random
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, ".")
from common import SCORE, SETS, TC_DIR, floor_of, items, load, log  # noqa: E402
from mdb.analysis import supplied, symbols  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.reads import readout  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
LAYERS = list(range(20, 26))
dbg = load()
ad = dbg.adapter
dev = ad.model.lm_head.weight.device
TC = {}
for x in LAYERS:
    st = load_file(f"{TC_DIR}/layer_{x}.safetensors")
    TC[x] = {"W_enc": st["W_enc"].to(dev), "b_enc": st["b_enc"].float(), "W_dec": st["W_dec"],
             "W_enc_cpu": st["W_enc"], "b_dec": st["b_dec"]}
log("load")
T = SETS["truths"]
LANGS, CITIES = SETS["sets"]["languages"], SETS["sets"]["cities"]
final = getattr(ad.backbone, ad.spec.final_norm)
GAMMA = f64(final.weight).reshape(-1).cpu()
W_U = ad.model.lm_head.weight
VER = {"library": "mwhanna/qwen3-1.7b-transcoders-lowl0 (read directly; SAELens mwhanna_transcoder layout)",
       "version": "safetensors bf16, config model_name Qwen/Qwen3-1.7B"}
AMAP = {"t<last>.L<l>.n": "blocks.<l>.mlp.hook_in", "t<last>.L<l>.m": "blocks.<l>.hook_mlp_out"}
rng = random.Random(20261114)
pool = sorted([i for i in items("depth") if i["family"] in ("d2", "d3-lm", "d3-per", "d4-work")
               and (SCORE[i["id"]]["correct"] or i["family"] != "d2")], key=lambda i: i["id"])
SHARD = None
TAG = "-smoke" if PICK else ""
if PICK in ("s0", "s1"):
    SHARD, TAG = int(PICK[1]), "-" + PICK
elif PICK == "smoke":
    SHARD = "smoke"
elif PICK:
    pool = [pool[int(k)] for k in PICK.split(",")]


def runner(lg, truth, toks):
    rest = [t for t in toks if t != truth]
    return max(rest, key=lambda t: float(lg[t]))


def encode(x, n):
    pre = (TC[x]["W_enc"] @ n.to(dev, torch.bfloat16)).float().cpu() + TC[x]["b_enc"]
    ids = (pre > 0).nonzero().reshape(-1)
    rows = TC[x]["W_enc_cpu"].index_select(0, ids).double()
    acts = torch.relu(rows @ n.double().cpu() + TC[x]["b_enc"].index_select(0, ids).double())
    keep = acts > 0
    return ids[keep].tolist(), acts[keep]


def dec(x, ids):
    return TC[x]["W_dec"].index_select(0, torch.tensor(ids, dtype=torch.long)).double()   # (n, d)


# ---------------- pass 1
P1 = {}
msum = {x: torch.zeros(ad.d_model if hasattr(ad, "d_model") else 2048, dtype=torch.float64) for x in LAYERS}
for item in pool:
    t0 = time.monotonic()
    iid, ids = item["id"], item["ids"]
    last = len(ids) - 1
    tr = T[iid]
    run = dbg.run(ids, name="nat")
    run.until(f"t{last}.logits")
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    lang, city = tr["language"], tr["city"]
    rl, rc = runner(lg, lang, LANGS), runner(lg, city, CITIES)
    hl = f64(run.read(f"t{last}.L{ad.layers - 1}.h_out")).reshape(-1).cpu()
    rho = float((hl.pow(2).mean() + float(final.variance_epsilon)) ** -0.5)
    cl = rho * GAMMA * (f64(W_U[lang]) - f64(W_U[rl])).cpu()
    cc = rho * GAMMA * (f64(W_U[city]) - f64(W_U[rc])).cpu()
    rec = {"id": iid, "family": tr["family"], "correct": SCORE[iid]["correct"], "lang": lang,
           "city": city, "runner_lang": rl, "runner_city": rc,
           "M_lang": float(lg[lang] - lg[rl]), "M_city": float(lg[city] - lg[rc]),
           "floor": floor_of(lg, lang, rl), "floor_city": floor_of(lg, city, rc),
           "argmax": int(lg.argmax()), "layers": {}}
    ms = {}
    for x in LAYERS:
        n = run.read(f"t{last}.L{x}.n").reshape(-1)
        fid, acts = encode(x, n)
        D = dec(x, fid)
        writes = D * acts[:, None]
        rec["layers"][x] = {"ids": fid, "acts": acts.tolist(), "de_lang": (writes @ cl).tolist(),
                            "de_city": (writes @ cc).tolist()}
        m = f64(run.read(f"t{last}.L{x}.m")).reshape(-1).cpu()
        ms[x] = m
        msum[x] += m
        recon = writes.sum(0) + TC[x]["b_dec"].double()
        rec["layers"][x]["rel_err"] = float((m - recon).norm() / m.norm())
        rec["layers"][x]["error_de_lang"] = float((m - recon) @ cl)
    P1[iid] = (rec, ms)
    del run
    log(f"P1 {iid} {time.monotonic() - t0:.2f}s")

# ---------------- nomination (frozen rule)
qual = collections.defaultdict(lambda: collections.defaultdict(set))   # (layer, f) -> lang -> items
for iid, (rec, _) in P1.items():
    for x in LAYERS:
        L_ = rec["layers"][x]
        for f, de in zip(L_["ids"], L_["de_lang"]):
            if de >= 0.25:
                qual[(x, f)][rec["lang"]].add(iid)
nominated = {}                                                          # (layer, f) -> lang
n_by_lang = collections.Counter(r["lang"] for r, _ in P1.values())
for key, by in qual.items():
    for lang, its in by.items():
        if len(its) < 3:
            continue
        other = sum(len(v) for lg_, v in by.items() if lg_ != lang)
        if other <= (len(P1) - n_by_lang[lang]) / 50:
            nominated[key] = lang
log(f"NOMINATED {len(nominated)}")
json.dump({"features": [[x, f, lang] for (x, f), lang in sorted(nominated.items())],
           "counts": {f"L{x}.f{f}": {str(k): len(v) for k, v in qual[(x, f)].items()} for (x, f) in nominated}},
          open(f"{OUT}/exp3-nominated{TAG}.json", "w"), indent=1)

# ---------------- pass 2
fh = open(f"{OUT}/exp3-items{TAG}.jsonl", "w")
cache = {}
n_neuron = 0
for item in (pool if SHARD is None else pool[:3] if SHARD == "smoke" else pool[SHARD::2]):
    t0 = time.monotonic()
    iid, ids = item["id"], item["ids"]
    last = len(ids) - 1
    rec, ms = P1[iid]
    feats = []
    for x in LAYERS:
        L_ = rec["layers"][x]
        for f, a, de in zip(L_["ids"], L_["acts"], L_["de_lang"]):
            if nominated.get((x, f)) == rec["lang"]:
                feats.append((de, x, f, a))
    feats = sorted(feats, reverse=True)[:5]
    rec["nominated"] = [[x, f, a, de] for de, x, f, a in feats]
    # nominated features' activations on this item regardless of qualification (H4)
    rec["lang_feature_acts"] = {}
    for (x, f), lang in nominated.items():
        if lang == rec["lang"]:
            L_ = rec["layers"][x]
            rec["lang_feature_acts"][f"L{x}.f{f}"] = L_["acts"][L_["ids"].index(f)] if f in L_["ids"] else 0.0
    toks = [rec["lang"], rec["runner_lang"], rec["city"], rec["runner_city"]]
    arms = {}
    if feats:
        run = dbg.run(ids, name="arms")
        run.until(f"t{last}.L20.h")
        ck = run.checkpoint("pre")

        def go(edits, label):
            for x, vec in sorted(edits, key=lambda e: e[0]):
                run.until(f"t{last}.L{x}.m")
                supplied.vector_edit(run, f"t{last}.L{x}.m", vec.to(torch.bfloat16).reshape(1, 1, -1),
                                     op="add", library=VER["library"], version=VER["version"], address_map=AMAP)
            run.until(f"t{last}.logits")
            lg = run.read(f"t{last}.logits").reshape(-1).float()
            arms[label] = {"M_lang": float(lg[toks[0]] - lg[toks[1]]) - rec["M_lang"],
                           "M_city": float(lg[toks[2]] - lg[toks[3]]) - rec["M_city"],
                           "argmax": int(lg.argmax())}
            run.restore(ck)

        writes = []
        for de, x, f, a in feats:
            w = -(dec(x, [f])[0] * a)
            writes.append((x, w))
            go([(x, w)], f"L{x}.f{f}")
        go(writes, "joint")
        de0, x0, f0, a0 = feats[0]
        L0 = rec["layers"][x0]
        nomset = {f for (x, f) in nominated if x == x0}
        cands = [(f, a) for f, a in zip(L0["ids"], L0["acts"]) if f not in nomset]
        if cands:
            fr, ar = rng.choice(cands)
            wr = dec(x0, [fr])[0] * ar
            wr = -wr * (writes[0][1].norm() / (wr.norm() + 1e-30))
            go([(x0, wr)], "random")
            arms["random"]["feature"] = [x0, fr]
        mean = (msum[x0] - ms[x0]) / (len(P1) - 1)
        go([(x0, mean - ms[x0])], f"mean_L{x0}")
        # joint arm as its own run, split by hop_edge against the native run
        nat = dbg.run(ids, name="nat2")
        nat.until(f"t{last}.logits")
        ed = dbg.run(ids, name="joint")
        for x, vec in sorted(writes, key=lambda e: e[0]):
            ed.until(f"t{last}.L{x}.m")
            ed.set(f"t{last}.L{x}.m", vec.to(torch.bfloat16).reshape(1, 1, -1), "add")
        ed.until(f"t{last}.logits")
        reader = readout(nat, Margin(last, toks[0], toks[1]))
        neurons = rec["correct"] and n_neuron < 8
        n_neuron += int(neurons)
        try:
            edge = symbols.hop_edge(nat, ed, reader, cell_layer=min(x for x, _ in writes),
                                    top_neurons=4, cache=cache, neurons=neurons)
            arms["joint"]["edge"] = {
                "native_delta": edge["consumer"]["native_delta"]["value"],
                "direct": edge["direct"]["value"], "mlp": edge["totals"]["mlp"]["value"],
                "reread": edge["totals"]["reread"]["value"], "norm": edge["norm"]["value"],
                "rounding": edge["rounding"]["value"],
                "mlp_by_layer": [[e["layer"], e["captured"]["value"]] for e in edge["mlp"]],
                "neurons": ([[e["layer"], e["nonlinear"]["value"], e["linear"]["value"],
                              [[n_["name"], n_["effect"]["value"]] for n_ in e.get("neurons", [])]]
                             for e in edge["mlp"]] if neurons else None)}
        except Exception as e:  # noqa: BLE001
            arms["joint"]["edge"] = {"refused": f"{type(e).__name__}: {str(e)[:200]}"}
        del run, nat, ed
    rec["arms"] = arms
    rec["seconds"] = round(time.monotonic() - t0, 2)
    rec["layers"] = {str(x): {k: v for k, v in L_.items() if k in ("rel_err", "error_de_lang")}
                     | {"n_active": len(L_["ids"])} for x, L_ in rec["layers"].items()}
    fh.write(json.dumps(rec, default=float) + "\n")
    fh.flush()
    log(f"P2 {iid} {rec['seconds']}s")
log("TOTAL")
