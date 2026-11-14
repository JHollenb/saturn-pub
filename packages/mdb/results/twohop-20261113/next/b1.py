"""B1 native job (next/prereg_b1.json). python b1.py MODEL MODE OUTDIR [PICK]

MODE F: freeze the two-hop map on the 64 base items of Run 1a (no 1c item is touched):
  circuit_map.item_report per item + the span cell census, circuit_map.freeze -> b1-map.json.
MODE D: the 154 depth items (1c): (a) native pass with prefill=span[0], p and h at the span rows
  for layers 0..22 saved; (b) carrier-cell transplant from the donor + hop_edge on M; (c) E1
  upstream transplant (span rows before the carrier, all KV heads, layers 0..19) and its
  same-country control, with the cell consumer (hop_edge to the carrier cell) and the full M.
  Arms go to ./arms-b1<-smoke>.jsonl and OUTDIR/arm/<id>.pt; the log prints ids and seconds only.
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.analysis import circuit_map as cm
from mdb.analysis import symbols
from mdb.analysis.forecast import Margin
from mdb.analysis.reads import layer_module, readout
from mdb.repl.__main__ import _load

MODEL, MODE, OUTDIR = sys.argv[1:4]
PICK = sys.argv[4] if len(sys.argv) > 4 else ""
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
ad = dbg.adapter
L, H, KV, DH = ad.layers, ad.num_heads, ad.num_kv_heads, ad.head_dim
CL, CG = 20, 6
SETS = json.load(open("b1_sets.json"))
spec = json.load(open("r1_items.json"))
OUT = Path(OUTDIR)
for sub in ("span", "arm"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
final = getattr(ad.backbone, ad.spec.final_norm)
gamma = final.weight.double().cpu()
W_U = ad.model.lm_head.weight
WRITERS = {x: cm.group_writers(layer_module(ad, x).self_attn.o_proj.weight, H, KV) for x in range(L)}
CARRIER = cm.Slot("carrier", "carrier", "R_argmax", layers=(CL,), kv=(CG,), rule_layer=CL,
                  rule_heads=(12, 13))
MAP = cm.Map([CARRIER, cm.Slot("desc", "descriptor", "rest")],
             [cm.Edge("desc", "carrier", "read")],
             {"kind": "margin", "answer": "item answer", "foil": "donor answer"},
             {"countries": SETS["sets"]["countries"], "cities": SETS["sets"]["cities"]})


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def ro_rows(tokens):
    return gamma[None, :] * W_U.double().cpu()[tokens]


RO_C = ro_rows(SETS["sets"]["countries"])
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def kept(lg, toks):
    lg = lg.reshape(-1).float()
    return {"logits": [float(lg[t]) for t in toks], "argmax": int(lg.argmax())}


def census_best(run, span, truth):
    values = {x: run.read_cache(x, "v", staged=False) for x in range(L)}
    cen = cm.cell_census(values, WRITERS, span, {"countries": RO_C})["scores"]["countries"]
    toks = SETS["sets"]["countries"]
    t = toks.index(truth)
    other = cen.clone()
    other[..., t] = -float("inf")
    margin = cen[..., t] - other.max(-1).values                 # (L, R, KV)
    flat = int(margin.reshape(-1).argmax())
    layer, rem = divmod(flat, len(span) * KV)
    ri, g = divmod(rem, KV)
    return {"layer": layer, "row": span[ri], "kv": g, "margin": float(margin.reshape(-1)[flat])}


if MODE == "F":
    items = [i for i in spec["items"] if i["group"] == "base"]
    if PICK:
        items = [items[int(k)] for k in PICK.split(",")]
    reports = []
    for item in items:
        t = time.monotonic()
        ids, last = item["ids"], len(item["ids"]) - 1
        run = dbg.run(ids, name="nat")
        run.until(f"t{last}.logits")
        it = {"name": item["id"], "span": item["span"], "gt": item["row"], "last": last}
        rep = cm.item_report(MAP, run, it, consumer=Margin(last, item["answer"], item["donor"]["answer"]),
                             truths={"countries": item["country_token"],
                                     "cities": item["capital_token"]})
        crow = int(rep["slots"]["carrier"][0].split("row=")[1].rstrip("]"))
        best = census_best(run, item["span"], item["country_token"])
        best["at_carrier"] = best["row"] == crow and (best["layer"], best["kv"]) == (CL, CG)
        rep["best_cell"] = best
        c0 = rep["table"]["bindings"][0].get("content", {}).get("countries", {})
        rep["carrier_names_country"] = {"verdict": "yes" if c0.get("names_truth") else "no"}
        rep["carrier_row_is_gt"] = {"verdict": "yes" if crow == item["row"] else "no"}
        reports.append(rep)
        print(f"F {item['id']} {time.monotonic() - t:.2f}s", flush=True)
    frozen = cm.freeze(MAP, reports, ["best_cell.layer", "best_cell.kv", "best_cell.at_carrier",
                                      "carrier_names_country.verdict", "carrier_row_is_gt.verdict"])
    frozen.constants["hop_shares_run1_donor"] = {"direct": 0.453, "mlp": 0.553, "reread": -0.010,
                                                 "source": "audited Run 1 ladders (AUDIT-RUN1-TP1.md P5)"}
    frozen.constants["best_cells"] = [r["best_cell"] for r in reports]
    doc = frozen.to_json()
    doc["digest"] = frozen.digest()
    json.dump(doc, open(f"b1-map{'-smoke' if PICK else ''}.json", "w"), indent=1, default=str)
    print(f"MAP digest {doc['digest']}", flush=True)
    print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ MODE D
T = SETS["truths"]
DON = json.load(open("b1_donors.json"))
by_id = {i["id"]: i for i in spec["items"] if i["group"] == "depth"}
items = sorted(by_id.values(), key=lambda i: i["id"])
if PICK:
    items = [items[int(k)] for k in PICK.split(",")]
ARMS = open(f"arms-b1{'-smoke' if PICK else ''}{('-' + PICK.replace(',', '_')[:20]) if PICK else ''}.jsonl", "w")
DONOR_CACHE = {}
HOP_CACHE = {}


def item_view(i):
    return {"name": i["id"], "span": i["span"], "last": len(i["ids"]) - 1}


def donor_state(did):
    if did not in DONOR_CACHE:
        d = by_id[did]
        r = dbg.run(d["ids"], name="donor")
        r.until(f"t{len(d['ids']) - 1}.logits")
        crow = cm.slot_rows(CARRIER, r, item_view(d))[0]
        DONOR_CACHE[did] = {
            "carrier_row": crow,
            "cell": r.read_cache(CL, "v", staged=False)[:, CG:CG + 1, crow:crow + 1].clone(),
            "span_v": {x: r.read_cache(x, "v", staged=False)[:, :, d["span"]].clone() for x in range(CL)}}
    return DONOR_CACHE[did]


def edge_view(edge):
    return {"native_delta": edge["consumer"]["native_delta"]["value"],
            "direct": edge["direct"]["value"], "upstream": edge["upstream"]["value"],
            "mlp": edge["totals"]["mlp"]["value"], "reread": edge["totals"]["reread"]["value"],
            "norm": edge["norm"]["value"], "rounding": edge["rounding"]["value"],
            "mlp_by_layer": [[e["layer"], e["captured"]["value"]] for e in edge["mlp"]],
            "reread_by_layer": [[e["layer"], e["effect"]["value"]] for e in edge["reread"]],
            "shares": {k: (v["value"] if isinstance(v, dict) and "value" in v else None)
                       for k, v in edge["shares"].items() if k != "meaning"}}


for item in items:
    t_item = time.monotonic()
    iid, ids, span = item["id"], item["ids"], item["span"]
    last = len(ids) - 1
    tr = T[iid]
    rec = {"id": iid, "family": tr["family"]}
    tens = {}
    # (a) native pass, span rows stepped with ports
    run = dbg.run(ids, name="nat", prefill=span[0])
    run.until(f"t{last}.logits")
    crow = cm.slot_rows(CARRIER, run, item_view(item))[0]
    rec["carrier_row"] = crow
    torch.save({"span": span, "h": {r: torch.stack([run.read(f"t{r}.L{x}.h").float().cpu().reshape(-1)
                                                     for x in range(23)]) for r in span},
                "p": {r: [run.read(f"t{r}.L{x}.p").float().cpu() for x in range(23)] for r in span}},
               OUT / "span" / f"{safe(iid)}.pt")
    tens["carrier_v_native"] = run.read_cache(CL, "v", staged=False)[0, CG, crow].float().cpu()
    # (b) carrier-cell transplant
    did = DON[iid]["cell"]
    ds = donor_state(did)
    foil = T[did]["language"]
    edited = dbg.run(ids, name="cellx", prefill=span[0])
    edited.until(f"t{last}.L{CL}.h")
    edited.set_cell(CL, "v", ds["cell"], rows=[crow], kv=[CG])
    edited.until(f"t{last}.logits")
    reader = readout(run, Margin(last, tr["language"], foil))
    edge = symbols.hop_edge(run, edited, reader, cell_layer=CL, top_neurons=4, cache=HOP_CACHE)
    rec["cell"] = {"donor": did, "donor_row": ds["carrier_row"], "foil": foil,
                   "native": kept(run.read(f"t{last}.logits"), [tr["language"], foil]),
                   "edited": kept(edited.read(f"t{last}.logits"), [tr["language"], foil]),
                   "edge": edge_view(edge)}
    tens["donor_cell_v"] = ds["cell"].reshape(-1).float().cpu()
    del edited
    # (c) E1 upstream transplant and its control
    if "e1" in DON[iid]:
        src = [r for r in span if r < crow]
        rec["e1_src_rows"] = src
        for kind in ("e1", "control"):
            did2 = DON[iid].get(kind)
            if did2 is None or not src:
                rec[kind] = {"state": "excluded", "reason": "no donor" if did2 is None else "no source row"}
                continue
            d2 = donor_state(did2)
            k_idx = [span.index(r) for r in src]
            other = T[did2]["country"] if T[did2]["country"] != tr["country"] else \
                T[DON[iid]["e1"]]["country"] if DON[iid].get("e1") else SETS["sets"]["countries"][0]
            target = symbols.CellTarget(crow, CL, CG, tr["country"], other)
            creader = symbols.cell_readout(run, target)
            edited = dbg.run(ids, name="e1x", prefill=span[0])
            edited.until(f"t{crow}.L0.h")
            for x in range(CL):
                edited.set_cell(x, "v", d2["span_v"][x][:, :, k_idx], rows=src, kv=list(range(KV)))
            edited.until(f"t{last}.logits")
            cedge = symbols.hop_edge(run, edited, creader, cell_layer=0, top_neurons=2, cache=HOP_CACHE)
            foil2 = T[DON[iid]["e1"]]["language"] if DON[iid].get("e1") else foil
            rec[kind] = {"donor": did2, "other_country": other, "foil": foil2,
                         "native": kept(run.read(f"t{last}.logits"), [tr["language"], foil2]),
                         "edited": kept(edited.read(f"t{last}.logits"), [tr["language"], foil2]),
                         "cell_native": creader.native,
                         "edge": edge_view(cedge)}
            tens[f"{kind}_carrier_v"] = edited.read_cache(CL, "v", staged=False)[0, CG, crow].float().cpu()
            del edited
    torch.save(tens, OUT / "arm" / f"{safe(iid)}.pt")
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    ARMS.write(json.dumps(rec) + "\n")
    ARMS.flush()
    print("B1 " + json.dumps({"id": iid, "s": rec["seconds"]}), flush=True)    # custody: no values
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
