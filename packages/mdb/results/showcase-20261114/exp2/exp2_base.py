"""Job A (prereg exp2/prereg.json + exp1/prereg.json census section).

python exp2_base.py OUTDIR [PICK]

Per base item (Run 1a, 64): native run (prefill = first span row) to the logits; carrier row by
the frozen R_argmax rule; mdb component bindings (Binding.component, B-a) of the carrier cell
L20 KV6 over the Qwen-Scope L19 basis (SAELens-checked) and of the cells L15..L23 KV6 at the
same row over the L14..L22 bases; component_table against M[answer, donor answer] with the
country content readout; the always-last-row cell as a null; the lifetime strips of the top
country feature (exact projection of the fixed L19 direction over all layers; decoder-cosine
lineage L9..L19); one batched native cell_sweep (B-c supplied vectors) per item with 4 arms per
layer (remove top country component, remove a random active non-country component rescaled to
the country component's norm, remove the error component, zero the cell).
Census (exp1): the L19 active set at the carrier row on the 64 base items and the 44 d1/d2
depth items (never scored by exp1).
The log prints item ids and seconds only.
"""
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from common import (CG, CL, SAE_DIR, SETS, SCORE, carrier_row, dump, floor_of, items, kept,  # noqa: E402
                    load, log, margin)
from mdb.analysis import sae  # noqa: E402
from mdb.analysis.components import component_binding, component_table  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.sweep import Cell, cell_sweep  # noqa: E402
from mdb.analysis.symbols import Binding, Candidates  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
dbg = load()
ad = dbg.adapter
FIRST = None
SWEEP_LAYERS = list(range(15, 24))
LINEAGE = list(range(9, 20))
need = sorted(set(LINEAGE) | {x - 1 for x in SWEEP_LAYERS})
SAES = sae.open_set("qwen-scope", SAE_DIR, layers=need, cache=len(need) + 1)
B = {x: SAES.get(x) for x in need}
B19 = B[19]
log("load")
COUNTRIES = SETS["sets"]["countries"]
rng = random.Random(20261114)

# ---- SAELens: registry load attempt (offline) and the strict to_saelens() copy of L19
prov = {"basis": B19.describe(), "sae_set": SAES.describe()}
try:
    from sae_lens import SAE
    import sae_lens
    prov["sae_lens_version"] = sae_lens.__version__
    try:
        reg = SAE.from_pretrained("qwen-scope-3-1.7b-base-w32k-l50", "layer19", device="cpu")
        reg = reg[0] if isinstance(reg, tuple) else reg
        prov["registry_load"] = {"ok": True, "metadata": dict(reg.cfg.metadata.__dict__)
                                 if hasattr(reg.cfg.metadata, "__dict__") else str(reg.cfg.metadata)}
    except Exception as e:  # noqa: BLE001
        reg = None
        prov["registry_load"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    SL = B19.to_saelens()
    prov["to_saelens"] = {"ok": True, "metadata": str(SL.cfg.metadata)[:600]}
except Exception as e:  # noqa: BLE001
    SL = None
    prov["to_saelens"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
log("saelens")


def sl_agrees(x, ids):
    if SL is None:
        return None
    with torch.no_grad():
        acts = SL.encode(x.float().reshape(1, -1).cpu())
    return sorted(int(i) for i in acts[0].nonzero().reshape(-1).tolist()) == sorted(ids)


def comp_rows(tab):
    out = []
    for r in tab["components"]:
        c = r.get("content", {}).get("countries", {})
        out.append({"c": r["component"], "f": r.get("feature"),
                    "a": (r["activation"]["value"] if "activation" in r else None),
                    "e": r["to_consumer"]["value"], "best": c.get("best"),
                    "names_truth": c.get("names_truth"), "norm": r["value_norm"]["value"]})
    return out


def top_country(rows):
    cands = [r for r in rows if r["f"] is not None and r["names_truth"]]
    return max(cands, key=lambda r: r["e"]) if cands else None


# TODO(mdb-feature P9): feature_lineage(f, layer, set) (decoder-cosine chain, labelled derived-local) as a binding column — feature lineage is hand-rolled over B[x].W_dec.
def lineage(f19, acts_by_layer):
    """Backward chain of f19 by max decoder cosine; per layer (feature, cos, active)."""
    chain = {19: (f19, 1.0, f19 in acts_by_layer[19])}
    d = B19.direction(f19).float()
    for x in range(18, LINEAGE[0] - 1, -1):
        W = B[x].W_dec                                     # (d, F) float32
        cos = (d @ W) / (W.norm(dim=0) * d.norm() + 1e-12)
        g = int(cos.argmax())
        chain[x] = (g, float(cos[g]), g in acts_by_layer[x])
        d = B[x].direction(g).float()
    onset = 19
    for x in range(19, LINEAGE[0] - 1, -1):
        g, c, on = chain[x]
        if c >= 0.7 and on:
            onset = x
        else:
            break
    return {"chain": {str(k): v for k, v in chain.items()}, "onset": onset,
            "censored": onset == LINEAGE[0]}


# TODO(mdb-feature P7): for item, run in pset.runs(...) — per-item run setup and the PICK/shard slicing below are hand-rolled.
def run_item(item, full):
    global FIRST
    t = time.monotonic()
    ids, span = item["ids"], item["span"]
    last = len(ids) - 1
    run = dbg.run(ids, name="nat", prefill=span[0])
    run.until(f"t{last}.logits")
    if FIRST is None:
        FIRST = run.program.names[0]
    crow = carrier_row(run, item)
    nat = run.read(f"t{last}.logits")
    country = item.get("country_token") or SETS["truths"][item["id"]]["country"]
    rec = {"id": item["id"], "group": item["group"], "family": item["family"],
           "carrier_row": crow, "last_span": span[-1], "country": country}
    h19 = run.read(f"t{crow}.L{CL}.h")
    enc = B19.encode(h19)
    rec["l19"] = {"ids": enc["ids"], "acts": [float(a) for a in enc["acts"]]}
    rec["saelens_agrees"] = sl_agrees(h19, enc["ids"])
    if not full:
        ans = item.get("answer")
        rec["correct"] = SCORE.get(item["id"], {}).get("correct")
        rec["native_argmax"] = int(nat.reshape(-1).argmax())
        rec["answer"] = ans
        return rec, time.monotonic() - t
    ans, foil = item["answer"], item["donor"]["answer"]
    rec.update(answer=ans, foil=foil, native=kept(nat, [ans, foil]),
               correct=int(nat.reshape(-1).argmax()) == ans)
    M = Margin(last, ans, foil)
    cands = [Candidates("countries", tuple(COUNTRIES), truth=country)]
    # TODO(mdb): use symbols.table(consumer, sets=, content_layers=True) with declared carrier/desc/t0/relay/question instead of component_table only (gap audit §4 exp2 first row: reader/mass/content columns and the memo's nulls were dropped).
    # TODO(mdb): use MapTable permutation null (circuit_map.py:507) instead of dropped cross-item / shuffled-id nulls (gap audit §4 exp2).
    # L1 / partition: the carrier cell over the L19 basis
    cb = Binding.component(run, "carrier", f"cache.L{CL}.v[kv{CG},row={crow}]", B19)
    tab = component_table(run, cb, M, sets=cands)
    rows = comp_rows(tab)
    rec["l20"] = {"cell_e": tab["cell_to_consumer"]["value"], "linear_e": tab["linear_to_consumer"]["value"],
                  "rounding_e": tab["rounding_to_consumer"]["value"], "mass": tab["mass"]["value"],
                  "partition_gap": tab["partition"]["gap"]["value"], "partition_holds": tab["partition"]["holds"],
                  "closure": {k: (v["value"] if isinstance(v, dict) and "value" in v else v)
                              for k, v in tab["closure"].items() if k != "meaning"},
                  "rows": rows, "top": top_country(rows)}
    feats = [r for r in rows if r["f"] is not None]
    nonc = [r for r in feats if not r["names_truth"]]
    rec["l20"]["random"] = rng.choice(nonc) if nonc else None
    # null: the always-last-row cell
    if span[-1] != crow:
        cbl = Binding.component(run, "last", f"cache.L{CL}.v[kv{CG},row={span[-1]}]", B19)
        tl = component_table(run, cbl, M, sets=cands)
        rl = comp_rows(tl)
        rec["l20_lastrow"] = {"cell_e": tl["cell_to_consumer"]["value"], "top": top_country(rl),
                              "error": [r for r in rl if r["c"] == "error"][0]}
    # strips
    # TODO(mdb-feature P9): projection strip + lineage as binding columns over a layer-indexed FeatureContext — exact projection strip over layers is hand-computed from raw reads.
    hs = {x: run.read(f"t{crow}.L{x}.h_out").reshape(-1) for x in range(ad.layers)}
    top = rec["l20"]["top"]
    if top is not None:
        d = B19.direction(top["f"])
        d = d / d.norm()
        rec["projection_strip"] = [float(d @ hs[x].double().cpu()) for x in range(ad.layers)]
        acts_by = {x: set(B[x].encode(hs[x])["ids"]) for x in LINEAGE}
        rec["lineage"] = lineage(top["f"], acts_by)
    # TODO(mdb): use symbols liveness() per layer with floors instead of remove-component arms per layer built by hand into cell_sweep (gap audit §4 exp2 'liveness() for L4').
    # TODO(mdb-feature P1): br.add / br.add_null per (layer, arm) with floor attached — arm list + labels + floor bookkeeping are parallel hand-kept lists.
    # sweep arms over L15..L23 at the carrier row
    arms, labels = [], []
    per_layer = {}
    for x in SWEEP_LAYERS:
        cbx = component_binding(run, f"c{x}", f"cache.L{x}.v[kv{CG},row={crow}]", B[x - 1])
        tx = component_table(run, cbx, M, sets=cands)
        rx = comp_rows(tx)
        tp = top_country(rx)
        nc = [r for r in rx if r["f"] is not None and not r["names_truth"]]
        rnd = rng.choice(nc) if nc else None
        per_layer[x] = {"cell_e": tx["cell_to_consumer"]["value"], "top": tp, "random": rnd,
                        "error_e": [r for r in rx if r["c"] == "error"][0]["e"],
                        "partition_holds": tx["partition"]["holds"]}
        dt = run.read_cache(x, "v", staged=False).dtype
        if tp is not None:
            arms.append([Cell(x, "v", crow, CG, cbx.edited(remove=(tp["c"],)).to(dt))])
            labels.append((x, "top"))
            # TODO(mdb-feature P1): br.add_null('random', like='top', norm='matched', pool=active_nonfeature) — matched-norm random component arm drawn and rescaled by hand.
            # TODO(mdb-feature P6): pool.random(matched_norm=..., seed=...) — same hand-coded random draw.
            if rnd is not None:
                vr = cbx.get(rnd["c"]).value
                scale = cbx.get(tp["c"]).value.norm() / (vr.norm() + 1e-30)
                arms.append([Cell(x, "v", crow, CG, cbx.edited(add=-vr * scale).to(dt))])
                labels.append((x, "random"))
        arms.append([Cell(x, "v", crow, CG, cbx.edited(remove=("error",)).to(dt))])
        labels.append((x, "error"))
        arms.append([Cell(x, "v", crow, CG, torch.zeros_like(cbx.native).to(dt))])
        labels.append((x, "zero"))
    rec["layers"] = {str(k): v for k, v in per_layer.items()}
    del run
    rb = dbg.run(ids, name="sweep", prefill=span[0])
    rb.until(f"t{last}.L0.{FIRST}")
    res = cell_sweep(rb, arms, tokens=[ans, foil], stepped_logits=nat)
    base_m = margin(nat, ans, foil)
    rec["sweep"] = {"base_m": base_m, "null_vs_stepped": res["null_vs_stepped"],
                    "floor": floor_of(nat, ans, foil, res["null_vs_stepped"] or 0.0),
                    "arms": [{"layer": lx, "arm": kind,
                              "dM": float(res["logits"][i, 0] - res["logits"][i, 1]) - base_m,
                              "argmax": int(res["argmax"][i])}
                             for i, (lx, kind) in enumerate(labels)]}
    del rb
    return rec, time.monotonic() - t


# TODO(mdb-feature P7): pset = dbg.prompt_set(...); pset.shard(K, N) — PICK / s0 / s1 slicing and JSONL loops below are hand-rolled.
base = items("base")
census_extra = [i for i in items("depth") if i["family"] in ("d1", "d2")]
TAG = "-smoke" if PICK and PICK[0] != "s" else ""
if PICK == "s0":
    base, census_extra, TAG = base[:32], [], "-s0"
elif PICK == "s1":
    base, TAG = base[32:], "-s1"
elif PICK:
    base = [base[int(k)] for k in PICK.split(",")]
    census_extra = census_extra[:2]
recs = []
with open(f"{OUT}/exp2-items{TAG}.jsonl", "w") as fh:
    for item in base:
        rec, s = run_item(item, True)
        fh.write(json.dumps(rec, default=float) + "\n")
        fh.flush()
        log(f"A {item['id']} {s:.2f}s")
with open(f"{OUT}/census-items{TAG}.jsonl", "w") as fh:
    for item in census_extra:
        rec, s = run_item(item, False)
        fh.write(json.dumps(rec, default=float) + "\n")
        fh.flush()
        log(f"C {item['id']} {s:.2f}s")
# TODO(mdb-feature P8): record sink / attached project — provenance and per-item records are not persisted as sealed records.
dump(prov, f"{OUT}/provenance{TAG}.json")
log("TOTAL")
