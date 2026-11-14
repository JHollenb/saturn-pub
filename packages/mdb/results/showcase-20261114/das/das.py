"""DAS at the carrier cell (owner-approved; das/prereg.json): train pyvene low-rank rotated-space
interventions on held-out-by-country pairs, build the training-free and supervised subspaces, and
score every arm's IIA on exp7's 64 Run 1a pairs through mdb's stepped engine.

python das.py admit OUT [PICK]       native mdb runs: admission + carrier row + cell value per item
python das.py train OUT KS STEPS     pyvene DAS per k (+ symtab / meandiff / random bases) -> bases.pt
python das.py score OUT ARMS [PICK]  mdb stepped IIA per test pair for the named arm families

Carrier cell: L20 KV6 value at the frozen R_argmax row (common.CARRIER). Train items: das/
train_candidates.json (countries disjoint from all exp7 test countries); test: the 64 Run 1a base
pairs (r1_items.json group 'base', recipient + its donor), never used in training or in any basis.
The log prints item ids, counts, losses and seconds only.
"""
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from common import CG, CL, SAE_DIR, carrier_row, floor_of, items, load, log  # noqa: E402

MODE, OUT = sys.argv[1], sys.argv[2]
SEED = 20261114
TRAIN = json.load(open("train_candidates.json"))
dbg = load()
ad = dbg.adapter
DH, KV, H = ad.head_dim, ad.num_kv_heads, ad.num_heads
log(f"load mode={MODE}")


def test_pairs():
    base = sorted(items("base"), key=lambda i: i["id"])
    by_id = {i["id"]: i for i in items("base")}
    out = []
    for item in base:
        did = item["donor"]["id"]
        donor = by_id.get(did) or {"id": did, "ids": item["donor"]["ids"], "span": [item["donor"]["row"]],
                                   "answer": item["donor"]["answer"], "bridge": item["donor"]["bridge"]}
        out.append((item, donor))
    return out


def native(item):
    """mdb native run (prefill = first span row): (crow, logits, cell value fp32)."""
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="nat", prefill=item["span"][0])
    run.until(f"t{last}.logits")
    crow = carrier_row(run, item)
    lg = run.read(f"t{last}.logits").reshape(-1).float().cpu()
    v = run.read_cache(CL, "v", staged=False)[0, CG, crow].float().cpu().clone()
    del run
    return crow, lg, v


if MODE == "admit":
    pick = sys.argv[3] if len(sys.argv) > 3 else ""
    todo = [dict(i, split="train") for i in TRAIN["items"]]
    seen = set()
    for item, donor in test_pairs():
        for x in (item, donor):
            if x["id"] not in seen:
                seen.add(x["id"])
                todo.append(dict(x, split="test"))
    if pick:
        todo = [todo[int(k)] for k in pick.split(",")]
    rows, cells = [], {}
    for x in todo:
        t0 = time.monotonic()
        crow, lg, v = native(x)
        top = lg.topk(2)
        rows.append({"id": x["id"], "split": x["split"], "bridge": x.get("bridge"), "answer": x["answer"],
                     "argmax": int(top.indices[0]), "correct": int(top.indices[0]) == x["answer"],
                     "crow": crow, "n": len(x["ids"]), "span": x["span"],
                     "margin_top2": float(top.values[0] - top.values[1])})
        cells[x["id"]] = v
        log(f"A {x['split']} {x['id']} {time.monotonic() - t0:.2f}s")
    json.dump(rows, open(f"{OUT}/admit{'-smoke' if pick else ''}.json", "w"), indent=0)
    torch.save(cells, f"{OUT}/cells{'-smoke' if pick else ''}.pt")
    log(f"admitted {sum(r['correct'] for r in rows if r['split'] == 'train')}/"
        f"{sum(r['split'] == 'train' for r in rows)} train")
    log("TOTAL")
    sys.exit(0)

TAG = os.environ.get("ADMIT_TAG", "")                 # "-smoke": the smoke's admit subset
ADMIT = {r["id"]: r for r in json.load(open(f"{OUT}/admit{TAG}.json"))}
CELLS = torch.load(f"{OUT}/cells{TAG}.pt")
if TAG:
    _tp = test_pairs
    test_pairs = lambda: [(i, d) for i, d in _tp() if i["id"] in ADMIT and d["id"] in ADMIT]  # noqa: E731


def train_items():
    return [dict(i, crow=ADMIT[i["id"]]["crow"]) for i in TRAIN["items"]
            if i["id"] in ADMIT and ADMIT[i["id"]]["correct"]]


def symtab_rows(tokens):
    """Cell-space readout directions W_g^T (gamma_F * W_U[t]) (rows, DH) float64: the symbol table's
    content readout (symbols.content, P19) of a KV-group-6 write at unit mass, mapped into the cell."""
    from mdb.analysis.circuit_map import group_writers
    from mdb.analysis.finite import f64
    from mdb.analysis.reads import layer_module

    final = getattr(ad.backbone, ad.spec.final_norm)
    gamma = f64(final.weight).reshape(-1).cpu()
    w_u = ad.model.lm_head.weight
    rows = f64(w_u.index_select(0, torch.tensor(tokens, device=w_u.device))).cpu() * gamma
    w_g = group_writers(layer_module(ad, CL).self_attn.o_proj.weight, H, KV)[CG].cpu()   # (d, DH)
    return rows @ w_g


def pca(rows, k):
    x = rows - rows.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(x, full_matrices=False)
    return vt[:k].T.contiguous(), s


if MODE == "train":
    import pyvene as pv
    from mdb.interop.clone import shared_clone
    from mdb.interop.pyvene_names import qwen3_shim

    KS = [int(k) for k in sys.argv[3].split(",")]
    STEPS = int(sys.argv[4])
    BS, LR = 32, 1e-2
    qwen3_shim()
    tr = train_items()
    log(f"train items {len(tr)} countries {len({i['bridge'] for i in tr})}")
    pairs = [(b, s) for b in tr for s in tr if b["answer"] != s["answer"]]
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    probe = pairs[:256]                                     # train-fit probe (in training set)
    tests = test_pairs()
    model = shared_clone(ad.model)
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device

    def batch(seqs):
        n = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return {"input_ids": ids.to(dev), "attention_mask": mask.to(dev)}

    def forward(im, bases, sources, brows, srows):
        B = len(bases)
        loc = {"sources->base": ([([[CG]] * B, [[r] for r in srows])], [([[CG]] * B, [[r] for r in brows])])}
        _, cf = im(batch(bases), [batch(sources)], loc)
        last = torch.tensor([len(s) - 1 for s in bases], device=dev)
        return cf.logits[torch.arange(B, device=dev), last].float()

    def evaluate(im, prs, brow_of, srow_of, ans_b, ans_s):
        out = []
        with torch.no_grad():
            for i in range(0, len(prs), BS):
                chunk = prs[i:i + BS]
                lg = forward(im, [b["ids"] for b, _ in chunk], [s["ids"] for _, s in chunk],
                             [brow_of(b) for b, _ in chunk], [srow_of(s) for _, s in chunk])
                for j, (b, s) in enumerate(chunk):
                    out.append(float(lg[j, ans_b(b)] - lg[j, ans_s(s)]))
        return out

    bases_out = {"meta": {"KS": KS, "steps": STEPS, "bs": BS, "lr": LR, "seed": SEED,
                          "train_items": [i["id"] for i in tr], "pairs": len(pairs)}}
    t_rows = lambda x: ADMIT[x["id"]]["crow"]  # noqa: E731
    for k in KS:
        t0 = time.monotonic()
        torch.manual_seed(SEED + k)
        cfg = pv.IntervenableConfig(representations=[pv.RepresentationConfig(
            layer=CL, component="head_value_output", unit="h.pos", max_number_of_units=1,
            low_rank_dimension=k)], intervention_types=pv.LowRankRotatedSpaceIntervention)
        im = pv.IntervenableModel(cfg, model=model)
        im.set_device(dev, set_model=False)
        im.disable_model_gradients()
        opt = torch.optim.Adam(im.get_trainable_parameters(), lr=LR)
        curve = []
        for step in range(STEPS):
            chunk = [pairs[(step * BS + j) % len(pairs)] for j in range(BS)]
            with torch.enable_grad():
                lg = forward(im, [b["ids"] for b, _ in chunk], [s["ids"] for _, s in chunk],
                             [b["crow"] for b, _ in chunk], [s["crow"] for _, s in chunk])
                loss = torch.nn.functional.cross_entropy(lg, torch.tensor([s["answer"] for _, s in chunk], device=dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
            curve.append(float(loss))
            if step % 25 == 0 or step == STEPS - 1:
                log(f"k={k} step {step} loss {float(loss):.3f}")
        iv = list(im.interventions.values())[0]
        W = iv.rotate_layer.weight.detach().double().cpu()                  # (DH, k), orthonormal cols
        fit = evaluate(im, probe, lambda x: x["crow"], lambda x: x["crow"], lambda x: x["answer"], lambda x: x["answer"])
        pv_test = evaluate(im, tests, t_rows, t_rows, lambda x: x["answer"], lambda x: x["answer"])
        bases_out[f"das{k}"] = W
        bases_out[f"das{k}_info"] = {"loss_curve": curve, "train_probe_M": fit, "pyvene_test_M": pv_test,
                                     "orth_err": float((W.T @ W - torch.eye(k, dtype=W.dtype)).abs().max()),
                                     "seconds": round(time.monotonic() - t0, 1)}
        log(f"k={k} done {time.monotonic() - t0:.1f}s probe flips {sum(m < 0 for m in fit)}/{len(fit)}")
        del im, opt
    # training-free symbol-table bases: country readout directions (and capital directions, 2b)
    full = json.load(open("r1_items.json"))["items"]
    countries = sorted({i["country_token"] for i in TRAIN["items"] if i.get("country_token")}
                       | {i["country_token"] for i in full if i.get("country_token")}
                       | {i["donor"]["country_token"] for i in full if i.get("donor", {}).get("country_token")})
    capitals = sorted({i["answer"] for i in TRAIN["items"]} | {i["answer"] for i, _ in tests} | {d["answer"] for _, d in tests})
    S, s_sv = pca(symtab_rows(countries), max(KS))
    C, c_sv = pca(symtab_rows(capitals), max(KS))
    # supervised mean-difference: PCA of per-country class means of TRAIN cell values (fp64)
    by = {}
    for i in tr:
        by.setdefault(i["bridge"], []).append(CELLS[i["id"]].double())
    means = torch.stack([torch.stack(v).mean(0) for v in by.values()])
    Mb, m_sv = pca(means, max(KS))
    g = torch.Generator().manual_seed(SEED)
    R = torch.linalg.qr(torch.randn(DH, max(KS), generator=g, dtype=torch.float64))[0]
    for k in KS:
        bases_out[f"symtab{k}"] = S[:, :k]
        bases_out[f"symcap{k}"] = C[:, :k]
        bases_out[f"meandiff{k}"] = Mb[:, :k]
        bases_out[f"random{k}"] = R[:, :k]
    bases_out["meta"].update({"country_tokens": countries, "capital_tokens": capitals,
                              "symtab_sv": s_sv.tolist(), "symcap_sv": c_sv.tolist(), "meandiff_sv": m_sv.tolist(),
                              "meandiff_countries": sorted(by)})
    torch.save(bases_out, f"{OUT}/bases.pt")
    log("TOTAL")
    sys.exit(0)

if MODE == "score":
    from mdb.analysis import sae
    from mdb.analysis.components import component_binding

    ARMS = sys.argv[3].split(",")                 # families: whole, das, symtab, symcap, meandiff, random, sae
    pick = sys.argv[4] if len(sys.argv) > 4 else ""
    tag = sys.argv[5] if len(sys.argv) > 5 else ("smoke" if pick else "full")
    BASES = torch.load(f"{OUT}/bases.pt")
    KS = BASES["meta"]["KS"]
    tests = test_pairs()
    if pick:
        tests = [tests[int(k)] for k in pick.split(",")]
    if "sae" in ARMS:
        B19 = sae.open_set("qwen-scope", SAE_DIR, layers=[19], cache=2).get(19)
        COUNTRY = {int(f) for f in json.load(open("census.json"))["features"]}
    fh = open(f"{OUT}/score-{tag}.jsonl", "w")
    DON = {}
    for item, donor in tests:
        t0 = time.monotonic()
        a, b, did = item["answer"], donor["answer"], donor["id"]
        ids, last = item["ids"], len(item["ids"]) - 1
        run = dbg.run(ids, name="nat", prefill=item["span"][0])
        run.until(f"t{last}.logits")
        crow = carrier_row(run, item)
        lg0 = run.read(f"t{last}.logits").reshape(-1).float()
        v = run.read_cache(CL, "v", staged=False)[0, CG, crow].double().cpu()
        Qsae = None
        if "sae" in ARMS:
            cb = component_binding(run, "c", f"cache.L{CL}.v[kv{CG},row={crow}]", B19, features=[])
            acts_r = set(B19.encode(run.read(f"t{crow}.L{CL}.h"))["ids"])
        del run
        if did not in DON:
            drun = dbg.run(donor["ids"], name="don", prefill=donor["span"][0])
            dl = len(donor["ids"]) - 1
            drun.until(f"t{dl}.logits")
            dcrow = carrier_row(drun, donor)
            dv = drun.read_cache(CL, "v", staged=False)[0, CG, dcrow].double().cpu()
            acts_d = set(B19.encode(drun.read(f"t{dcrow}.L{CL}.h"))["ids"]) if "sae" in ARMS else set()
            DON[did] = (dcrow, dv, acts_d)
            del drun
        dcrow, dv, acts_d = DON[did]
        arms_v = {}
        if "whole" in ARMS:
            arms_v["whole"] = dv
        for fam in ("das", "symtab", "symcap", "meandiff", "random"):
            if fam in ARMS:
                for k in KS:
                    Q = BASES[f"{fam}{k}"].double()
                    arms_v[f"{fam}{k}"] = v + Q @ (Q.T @ (dv - v))
        if "sae" in ARMS:
            feats = sorted((acts_r | acts_d) & COUNTRY)
            if feats:
                Mx = torch.stack([cb.image(B19.direction(f)) for f in feats], 1)
                Qs, Rr = torch.linalg.qr(Mx)
                rank = int((Rr.diagonal().abs() > 1e-9 * Rr.diagonal().abs().max()).sum())
                Qsae = Qs[:, :rank]
            else:
                Qsae = torch.zeros(DH, 0, dtype=torch.float64)
            arms_v["sae"] = v + Qsae @ (Qsae.T @ (dv - v))
        run = dbg.run(ids, name="br", prefill=crow)
        run.until(f"t{crow}.L{CL}.h")
        ck = run.checkpoint("pre")
        shape = run.port_shape("v")
        out = {}
        for name, val in arms_v.items():
            full = torch.zeros(shape, dtype=ad.dtype, device="cuda")
            mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
            full.reshape(-1)[CG * DH:(CG + 1) * DH] = val.to("cuda", ad.dtype)
            mask.reshape(-1)[CG * DH:(CG + 1) * DH] = True
            run.set(f"t{crow}.L{CL}.v", full, mask=mask)
            run.until(f"t{last}.logits")
            lg = run.read(f"t{last}.logits").reshape(-1).float()
            out[name] = {"M": float(lg[a] - lg[b]), "argmax": int(lg.argmax())}
            run.restore(ck)
        del run
        rec = {"id": item["id"], "donor": did, "crow": crow, "donor_crow": dcrow, "M0": float(lg0[a] - lg0[b]),
               "floor": floor_of(lg0, a, b), "arms": out, "delta_norm": float((dv - v).norm()),
               "seconds": round(time.monotonic() - t0, 2)}
        if Qsae is not None:
            rec["sae_rank"] = int(Qsae.shape[1])
            rec["sae_basis"] = Qsae.tolist()
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        log(f"S {item['id']} {rec['seconds']}s")
    log("TOTAL")
