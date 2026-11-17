"""Null re-check on Qwen3-1.7B-Base (entries 1, 9, 10 of the mdb tracker), one model load.

    python q3_recheck.py MODEL DATA_DIR OUT [N_ITEMS]

Claims (saturn experiments/2026-11-16-null-recheck/PREREG.md):
  C1  K-only single cell at the carrier (L20 KV6, gt row) has no effect   (entry 1, 0/123 flips)
  C2  K-only bands over the descriptor span from L0/8/16/20 do nothing    (entry 1, 0/123 flips)
  C3  V-only band from L24 does nothing: "the read finishes by L23"     (entry 1, 0/123 flips)
  C4  no live single V cell at the carrier row outside L17-L21            (entry 1 Run 1 P2, entry 10)
  C5  single-donor zero-effect span V cells                              (entry 1 Run 1 P1, descriptive)
  C6  t0's L14 input replaced by another prompt's costs nothing          (entry 9 T3)

C1-C5 use mdb's batched native cell sweep (``mdb.analysis.sweep.cell_sweep``, the instrument the
original job 2 used): cell edits read by the answer row only, one null arm per chunk. Donors: 16
same-family items with a different answer (the panel's canonical donors first, then same-length
items, then the rest, seeded). Rows are aligned by position from the end of the descriptor span; a
donor key from another position is RoPE-rotated to the recipient's row (one bf16 rounding; values
need no rotation). C6 is a batched full forward with a pre-hook on layer 14 (row 0 of every lane),
16 donors = prompts with a different first token. Verdicts: mdb ``donors.informative`` / ``pooled``.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

from common import dump, floor_of, informative, log, pool, rotate_delta, spacing

from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load

N_DONORS, SEED, BATCH = 16, 20261116, 96
K_BANDS, V_BANDS = (0, 8, 16, 20), (20, 24)
WINDOW = range(17, 22)   # L17..L21
LIVE = 0.5               # Run 1 P2's live threshold (|dM| > 0.5)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def aligned(rspan, dspan):
    """(recipient row, donor row) pairs aligned from the end of the span."""
    n = min(len(rspan), len(dspan))
    return list(zip(rspan[-n:], dspan[-n:]))


def donors_for(item, entry, cand):
    rng = random.Random(f"{SEED}:{item['id']}")
    pool_ = [c for c in cand.values() if c["family"] == item["family"] and c["id"] != item["id"]
             and c["answer"] != item["answer"]]
    canon = [cand[d] for d in entry["donors"] if d in cand and cand[d]["answer"] != item["answer"]]
    same_len = [c for c in pool_ if c not in canon and len(c["ids"]) == len(item["ids"])]
    rest = [c for c in pool_ if c not in canon and c not in same_len]
    rng.shuffle(same_len)
    rng.shuffle(rest)
    return (canon + same_len + rest)[:N_DONORS], len(canon), len(pool_)


def main(model_dir, data, out, n_items=None):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    dbg = _load(Path(model_dir), "bf16", DEV, False, None)
    adapter = dbg.adapter
    model = adapter.model
    L, KV = adapter.layers, adapter.num_kv_heads
    rot = model.model.rotary_emb
    assert float(getattr(rot, "attention_scaling", 1.0)) == 1.0
    inv_freq = rot.inv_freq.float().cpu()
    log(f"load: {L} layers, {KV} kv heads, attn {model.config._attn_implementation}")
    import json
    cand = {i["id"]: i for i in json.load(open(data / "candidates.json"))["items"]}
    panel = json.load(open(data / "job1-panel.json"))["panel"]
    gt = json.load(open(data / "gt.json"))
    if n_items:
        panel = panel[:n_items]

    caches = {}

    def cache_of(cid):
        if cid not in caches:
            ids = cand[cid]["ids"]
            run = dbg.run(ids, name="c")
            k = [run.read_cache(layer, "k", staged=False)[0].clone() for layer in range(L)]
            v = [run.read_cache(layer, "v", staged=False)[0].clone() for layer in range(L)]
            run.until(f"t{len(ids) - 1}.logits")
            caches[cid] = (k, v, run.read().reshape(-1).float().cpu())
        return caches[cid]

    cols = {k: [] for k in ("item", "kind", "start", "row", "layer", "kv", "donor", "dM", "floor",
                            "inf", "to_donor", "changed")}
    KIND = {"V1": 0, "K1": 1, "Kband": 2, "Vband": 3}
    meta, xcheck_inputs = [], None
    for ii, entry in enumerate(panel):
        item = cand[entry["id"]]
        ids, span, a = item["ids"], item["span"], item["answer"]
        g = gt[item["id"]]
        grow, glayer, gkv = g["row"], g["layer"], g["kv"]
        donors, n_canon, n_avail = donors_for(item, entry, cand)
        rk, rv, native = cache_of(item["id"])
        toks = [a] + sorted({d["answer"] for d in donors})
        col = {t: i for i, t in enumerate(toks)}
        arms, keys = [], []
        bands, bkeys = [], []
        rK, rV = torch.stack(rk), torch.stack(rv)            # [L, KV, T, hd]
        for kd, d in enumerate(donors):
            dk, dv, _ = cache_of(d["id"])
            dK, dV = torch.stack(dk), torch.stack(dv)
            pairs = aligned(span, d["span"])
            # rotated donor keys at each aligned row, all layers and heads at once: [L, KV, hd]
            krot = {r: rotate_delta(dK[:, :, dr], r - dr, inv_freq) for r, dr in pairs}
            for r, dr in pairs:
                vinf = (rV[:, :, r] != dV[:, :, dr]).any(-1).cpu()   # mdb informative: bitwise differ
                for layer in range(L):
                    for kv in range(KV):
                        arms.append([Cell(layer, "v", r, kv, dV[layer, kv, dr])])
                        keys.append(("V1", 0, r, layer, kv, kd, bool(vinf[layer, kv])))
            if grow in krot:
                val = krot[grow][glayer, gkv]
                arms.append([Cell(glayer, "k", grow, gkv, val)])
                keys.append(("K1", 0, grow, glayer, gkv, kd, informative(rK[glayer, gkv, grow], val)))
            for kind, starts in (("Kband", K_BANDS), ("Vband", V_BANDS)):
                for s in starts:
                    cells, inf = [], False
                    for r, dr in pairs:
                        src = krot[r] if kind == "Kband" else dV[:, :, dr]
                        base = rK[:, :, r] if kind == "Kband" else rV[:, :, r]
                        inf = inf or bool((src[s:] != base[s:]).any())
                        t = "k" if kind == "Kband" else "v"
                        cells += [Cell(layer, t, r, kv, src[layer, kv]) for layer in range(s, L) for kv in range(KV)]
                    bands.append(cells)
                    bkeys.append((kind, s, -1, -1, -1, kd, inf))
        run = dbg.run(ids, name="sweep")
        res = [cell_sweep(run, arms, tokens=toks, batch=BATCH, stepped_logits=native),
               cell_sweep(run, bands, tokens=toks, batch=BATCH, stepped_logits=native)]
        # per-donor floor: 2 spacings at the larger |logit| + spread of the null M over every chunk
        nulls = torch.cat([r["null"] for r in res])
        floors = []
        for d in donors:
            m = nulls[:, 0] - nulls[:, col[d["answer"]]]
            floors.append(floor_of(native[a], native[d["answer"]], float(m.max() - m.min())))
        for r_, kk in zip(res, (keys, bkeys)):
            lg, am = r_["logits"].double(), r_["argmax"]
            n = len(kk)
            idx = torch.arange(n)
            nl = r_["null"].double()[idx // BATCH]
            kd_t = torch.tensor([x[5] for x in kk])
            bans = torch.tensor([donors[x[5]]["answer"] for x in kk])
            bcol = torch.tensor([col[donors[x[5]]["answer"]] for x in kk])
            dm = ((lg[idx, 0] - lg[idx, bcol]) - (nl[idx, 0] - nl[idx, bcol])).tolist()
            fl = torch.tensor(floors, dtype=torch.float64)[kd_t].tolist()
            tod = (am == bans).tolist()
            chg = (am != a).tolist()
            for j, (kind, s, row, layer, kv, kd, inf) in enumerate(kk):
                for name, v in (("item", ii), ("kind", KIND[kind]), ("start", s), ("row", row),
                                ("layer", layer), ("kv", kv), ("donor", kd), ("dM", dm[j]),
                                ("floor", fl[j]), ("inf", -1 if inf is None else int(inf)),
                                ("to_donor", tod[j]), ("changed", chg[j])):
                    cols[name].append(v)
        meta.append({"id": item["id"], "family": item["family"], "len": len(ids), "span": span,
                     "gt": [grow, glayer, gkv], "native_correct": int(native.argmax()) == a,
                     "donors": [d["id"] for d in donors], "same_len": [len(d["ids"]) == len(ids) for d in donors],
                     "n_canonical": n_canon, "n_available": n_avail, "floors": floors,
                     "null_vs_stepped": [r["null_vs_stepped"] for r in res], "arms": len(arms) + len(bands)})
        if xcheck_inputs is None:
            xcheck_inputs = (item, donors[:2], grow, glayer, gkv)
        log(f"{ii} {item['id']}: {len(arms)}+{len(bands)} arms, donors {len(donors)} "
            f"(canon {n_canon}, avail {n_avail}), null_vs_stepped {[r['null_vs_stepped'] for r in res]}")
    np.savez_compressed(out / "q3_cells.npz", **{k: np.array(v) for k, v in cols.items()})
    summary = {"items": len(meta), "meta": meta}
    summary["C6"] = tp1(model, data, out)
    summary["crosscheck"] = crosscheck(dbg, cache_of, xcheck_inputs, cols, inv_freq)
    summary["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None
    dump(out / "q3_summary.json", summary)
    log("done")


def tp1(model, data, out):
    """C6: row 0 of the L14 input replaced by 16 other prompts' row 0 (different first token)."""
    import json
    d = json.load(open(data / "tp1_items.json"))
    prompts = {p["k"]: p for p in d["prompts"]}
    layers = model.model.layers
    state = {"cap": None, "patch": None}

    def pre(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        if state["cap"] is not None:
            state["cap"].append(hs[:, 0].detach().clone())
        if state["patch"] is not None:
            hs = hs.clone()
            hs[1:, 0] = state["patch"].to(hs.dtype)
            if args:
                return (hs, *args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        return None

    h = layers[14].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        row0, nat1 = {}, {}
        for k, p in prompts.items():
            state["cap"] = []
            lg = model(torch.tensor([p["ids"]], device=DEV)).logits[0, -1].float().cpu()
            row0[k] = state["cap"][0][0]
            nat1[k] = lg
        state["cap"] = None
        pos_mean = torch.stack([row0[k].float() for k in prompts]).mean(0).to(torch.bfloat16)
        by_first = {}
        for k, p in prompts.items():
            by_first.setdefault(p["ids"][0], []).append(k)
        recs = []
        for t in d["top1pct"]:
            k = t["prompt"]
            p = prompts[k]
            others = sorted(f for f in by_first if f != p["ids"][0])
            rng = random.Random(f"{SEED}:tp1:{k}")
            rng.shuffle(others)
            dks = [rng.choice(sorted(by_first[f])) for f in others[:N_DONORS]]
            vals = torch.stack([row0[j] for j in dks] + [pos_mean.to(row0[k].device)])
            state["patch"] = vals
            ids = torch.tensor([p["ids"]] * (len(dks) + 2), device=DEV)
            lg = model(ids).logits[:, -1].float().cpu()
            state["patch"] = None
            t1, t2 = p["t1"], p["t2"]
            M = lg[:, t1] - lg[:, t2]
            m1 = float(nat1[k][t1] - nat1[k][t2])
            floor = floor_of(nat1[k][t1], nat1[k][t2], abs(float(M[0]) - m1))
            rows = [{"donor": j, "first": prompts[j]["ids"][0], "dM": float(M[i + 1] - M[0]), "floor": floor,
                     "informative": informative(row0[k], row0[j]), "argmax_kept": int(lg[i + 1].argmax()) == t1}
                    for i, j in enumerate(dks)]
            recs.append({"prompt": k, "M0": m1, "orig_position_dM": t["orig_position_dM"],
                         "position_mean_dM": float(M[-1] - M[0]), "pooled": pool(rows), "per_donor": rows,
                         "batched_null_vs_batch1": float(M[0]) - m1})
            log(f"C6 prompt {k}: {recs[-1]['pooled']['label']}, max {recs[-1]['pooled']['max_floor_units']:.2f} floors")
        return recs
    finally:
        h.remove()


def crosscheck(dbg, cache_of, inputs, cols, inv_freq):
    """Two K1 and two V1 arms of the first item re-measured stepped (``Run.set_cell`` then
    ``until logits`` on a fresh run, job 2's authority path); dM against the stepped native."""
    out = []
    try:
        item, dons, grow, glayer, gkv = inputs
        ids = item["ids"]
        last = len(ids) - 1
        a = item["answer"]
        nat = cache_of(item["id"])[2]
        for kd, d in enumerate(dons):
            dk, dv, _ = cache_of(d["id"])
            dg = dict(aligned(item["span"], d["span"])).get(grow)
            b = d["answer"]
            for t, val in (("k", rotate_delta(dk[glayer][gkv, dg], grow - dg, inv_freq)),
                           ("v", dv[glayer][gkv, dg])):
                run = dbg.run(ids, name="x")
                run.set_cell(glayer, t, val.reshape(1, 1, 1, -1), rows=[grow], kv=[gkv])
                run.until(f"t{last}.logits")
                lg = run.read().reshape(-1).float().cpu()
                stepped = float((lg[a] - lg[b]) - (nat[a] - nat[b]))
                kind = 1 if t == "k" else 0
                batched = [cols["dM"][i] for i in range(len(cols["dM"])) if cols["item"][i] == 0
                           and cols["kind"][i] == kind and cols["donor"][i] == kd and cols["row"][i] == grow
                           and cols["layer"][i] == glayer and cols["kv"][i] == gkv]
                out.append({"tensor": t, "donor": d["id"], "stepped_dM": stepped,
                            "batched_dM": batched[0] if batched else None})
                log(f"crosscheck {out[-1]}")
    except Exception as e:  # noqa: BLE001
        out.append({"error": f"{type(e).__name__}: {str(e)[:400]}"})
        log(f"crosscheck failed: {out[-1]}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
