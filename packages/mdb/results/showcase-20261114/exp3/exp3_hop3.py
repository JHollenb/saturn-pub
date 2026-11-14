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
# TODO(mdb-feature P3): sae.open_set('transcoder', TC_DIR, layers=...) with read_site='L<l>.n', write_site='L<l>.m' — safetensors transcoders are loaded and moved to device by hand; sae.py loads residual SAEs only.
TC = {}
for x in LAYERS:
    st = load_file(f"{TC_DIR}/layer_{x}.safetensors")
    TC[x] = {"W_enc": st["W_enc"].to(dev), "b_enc": st["b_enc"].float(), "W_dec": st["W_dec"],
             "W_enc_cpu": st["W_enc"], "b_dec": st["b_dec"]}
log("load")
T = SETS["truths"]
LANGS, CITIES = SETS["sets"]["languages"], SETS["sets"]["cities"]
# TODO(mdb): use reads.readout(run, Margin).vector/.rho (analysis/reads.py:108-128) instead of manual final-norm gamma/rho and W_U rows (gap audit §1 final-norm readout row).
final = getattr(ad.backbone, ad.spec.final_norm)
GAMMA = f64(final.weight).reshape(-1).cpu()
W_U = ad.model.lm_head.weight
VER = {"library": "mwhanna/qwen3-1.7b-transcoders-lowl0 (read directly; SAELens mwhanna_transcoder layout)",
       "version": "safetensors bf16, config model_name Qwen/Qwen3-1.7B"}
# TODO(mdb): use interop.names.to_nnsight / core.foreign_names.ct_node instead of a hand-typed address map (gap audit §1 library address maps row).
AMAP = {"t<last>.L<l>.n": "blocks.<l>.mlp.hook_in", "t<last>.L<l>.m": "blocks.<l>.hook_mlp_out"}
rng = random.Random(20261114)
# TODO(mdb-feature P7): pset.shard(K, N) with roles/vars — pool construction and SHARD/PICK slicing are hand-rolled.
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


# TODO(mdb-feature P3): FeatureBasis.encode for transcoders (bf16 GPU screen + float64 exact re-encode) — encode/decode of transcoders is hand-written here and again in exp4.
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
    # TODO(mdb): use symbols.table + an edge-mode watch on the answer row's candidate-set argmax over L*, with margins (Q2 item 4), instead of one read at the logits (gap audit §3/§5 Q4.2).
    run.until(f"t{last}.logits")
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    lang, city = tr["language"], tr["city"]
    rl, rc = runner(lg, lang, LANGS), runner(lg, city, CITIES)
    # TODO(mdb): use reads.readout(run, Margin(last, lang, rl)).vector/.rho instead of hand-rolled rho and cl/cc direct-effect vectors (gap audit §1 final-norm readout row).
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
        # TODO(mdb-feature P3): MLP-write binding: component_binding(run, name, 't<r>.L<l>.m', transcoder_basis) with the Unbound row = error — recon, rel_err and error_de_lang are recomputed by hand (error share of each MLP write).
        m = f64(run.read(f"t{last}.L{x}.m")).reshape(-1).cpu()
        ms[x] = m
        msum[x] += m
        recon = writes.sum(0) + TC[x]["b_dec"].double()
        rec["layers"][x]["rel_err"] = float((m - recon).norm() / m.norm())
        rec["layers"][x]["error_de_lang"] = float((m - recon) @ cl)
    P1[iid] = (rec, ms)
    del run
    log(f"P1 {iid} {time.monotonic() - t0:.2f}s")

# TODO(mdb-feature P10): nominate(scores, rule, k) -> sealed SuspectList, then branches.measure(list, nulls=[...]) — frozen nomination rule is coded by hand (5 such rules across scripts).
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
# TODO(mdb-feature P7): each/pset.runs(..., shard=K/N) — item loop and shard slicing hand-rolled.
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
        # TODO(mdb-feature P1): run.branches(at=ck)... br.add per nominated feature, 'joint', 'random', 'mean' — one checkpoint + go() loop is the Branches object; the joint arm below needs two extra fresh runs.
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
            # TODO(mdb-feature P3): feature_value(f, act, basis) supplied vector from the transcoder binding — feature write -a*d_f is built by hand.
            w = -(dec(x, [f])[0] * a)
            writes.append((x, w))
            go([(x, w)], f"L{x}.f{f}")
        go(writes, "joint")
        de0, x0, f0, a0 = feats[0]
        L0 = rec["layers"][x0]
        # TODO(mdb-feature P1): br.add_null('random', like='top', norm='matched', pool=active_non_nominated) — matched-norm random null drawn by hand.
        # TODO(mdb-feature P6): pool.random(matched_norm=..., seed=...) — same.
        nomset = {f for (x, f) in nominated if x == x0}
        cands = [(f, a) for f, a in zip(L0["ids"], L0["acts"]) if f not in nomset]
        if cands:
            fr, ar = rng.choice(cands)
            wr = dec(x0, [fr])[0] * ar
            wr = -wr * (writes[0][1].norm() / (wr.norm() + 1e-30))
            go([(x0, wr)], "random")
            arms["random"]["feature"] = [x0, fr]
        # TODO(mdb-feature P6): pool.mean(by=..., leave_out=item) over the MLP write — leave-one-out class mean coded by hand (self-inclusion bug hit once in TP1).
        mean = (msum[x0] - ms[x0]) / (len(P1) - 1)
        go([(x0, mean - ms[x0])], f"mean_L{x0}")
        # TODO(mdb): use branches from the one checkpoint (P1) so hop_edge attaches to any branch instead of two extra fresh runs (nat2, joint) (gap audit §4 exp3 / exp3:211-217).
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
