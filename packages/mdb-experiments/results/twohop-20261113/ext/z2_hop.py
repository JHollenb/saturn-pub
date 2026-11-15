"""Z2: the hop-2 ladder through symbols.hop_edge (prereg_hop.json).

python z2_hop.py MODEL OUT [PICK]

Inputs in the working dir: candidates.json, job1-panel.json, gt.json, job2_gtv.json. One model
load; per panel recipient: the native run, then three edited runs of the L20 KV6 cell at the
gt row (donor 0's cell; the panel-mean cell; a same-country different-city cell for city items),
each read by hop_edge against the native run. The donor arm also goes through symbols.table
(hops={'bridge': {..., 'edge': ...}}) so the record path is exercised.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from mdb.analysis import symbols
from mdb.analysis.finite import f64
from mdb.analysis.forecast import Margin
from mdb.analysis.reads import readout
from mdb.repl.__main__ import _load

MODEL, OUT = sys.argv[1:3]
PICK = sys.argv[3] if len(sys.argv) > 3 else ""
LAYER, GROUP = 20, 6
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
L = dbg.adapter.layers
spec = json.load(open("candidates.json"))
cand = {i["id"]: i for i in spec["items"]}
full_panel = json.load(open("job1-panel.json"))["panel"]
panel = [full_panel[int(k)] for k in PICK.split(",")] if PICK else full_panel
gt = json.load(open("gt.json"))
j2 = json.load(open("job2_gtv.json"))
correct = {r["id"]: r["correct"] for r in json.load(open("job1-panel.json"))["rows"]}
city_by_country = defaultdict(list)
for i in spec["items"]:
    if i["family"] == "city":
        city_by_country[i["bridge"]].append(i["id"])
CACHE = {}
OUTF = open(f"{OUT}.jsonl", "w")


def cell_of(ids, row):
    run = dbg.run(ids, name="cell")
    return run.read_cache(LAYER, "v", staged=False)[:, GROUP:GROUP + 1, row:row + 1].clone()


# the panel-mean cell over all 64 recipients' own cells (independent of PICK)
t = time.monotonic()
cells = [cell_of(cand[e["id"]]["ids"], gt[e["id"]]["row"]) for e in full_panel]
MEAN = torch.stack([c.float() for c in cells]).mean(0).to(cells[0].dtype)
print(f"load+mean {time.monotonic() - T0:.1f}s (mean cell {time.monotonic() - t:.1f}s); "
      f"{len(panel)} items", flush=True)


def edited_run(ids, j, r0, value):
    er = dbg.run(ids, name="edit")
    er.set_cell(LAYER, "v", value, rows=[r0], kv=[GROUP])
    er.until(f"t{j}.logits")
    return er


def logit_pair(run, j, a, b):
    lg = run.read(f"t{j}.logits").reshape(-1)
    return float(f64(lg[a])), float(f64(lg[b]))


def unbound_mlp(run, j, reader):
    c = f64(reader.vector).reshape(-1)
    total = float(c @ f64(run.read(f"t{j}.L0.h")).reshape(-1))
    for x in range(L):
        total += float(c @ f64(run.read(f"t{j}.L{x}.m")).reshape(-1))
    return total


def arm(run, item, j, r0, a, b, value, spectral):
    reader = readout(run, Margin(j, a, b))
    er = edited_run(item["ids"], j, r0, value)
    t = time.monotonic()
    edge = symbols.hop_edge(run, er, reader, cell_layer=LAYER, top_neurons=8,
                            spectral=spectral, cache=CACHE)
    return er, reader, edge, round(time.monotonic() - t, 2)


for entry in panel:
    t_item = time.monotonic()
    item = cand[entry["id"]]
    ids, j, a = item["ids"], item["n"] - 1, item["answer"]
    r0 = gt[item["id"]]["row"]
    d0 = cand[entry["donors"][0]]
    b0 = d0["answer"]
    run = dbg.run(ids, name="main")
    run.until(f"t{j}.logits")
    rec = {"id": item["id"], "family": item["family"], "bridge": item["bridge"], "r0": r0,
           "donor0": d0["id"]}
    arms = {}
    # donor arm (spectral control here) and the table path
    er, reader, edge, sec = arm(run, item, j, r0, a, b0, cell_of(d0["ids"], r0), True)
    saved = j2[item["id"]]
    arms["donor"] = {"edge": edge, "hop_seconds": sec,
                     "M0": logit_pair(run, j, a, b0), "M1": logit_pair(er, j, a, b0),
                     "job2_delta": (saved["GT.V.d0"][0] - saved["GT.V.d0"][1])
                     - (saved["native"][0] - saved["native"][1]),
                     "unbound_mlp": unbound_mlp(run, j, reader)}
    symbols.declare(run, "bridge", [f"cache.L{LAYER}.v[kv{GROUP},row={r0}]"])
    native = edge["consumer"]["native_delta"]["value"]
    table = symbols.table(run, consumer=Margin(j, a, b0), limit=1, hops={"bridge": {
        "native_delta": native, "direct_delta": edge["direct"]["value"], "edge": edge,
        "replacement": f"V cell L20 KV6 from {d0['id']}"}})
    view = table.report["bindings"][0]
    arms["donor"]["table_hop_matches"] = view["hop"]["edge"] == edge
    arms["donor"]["table_to_consumer"] = view["to_consumer"].get("value")
    # ablate arm: the panel-mean cell
    er, reader, edge, sec = arm(run, item, j, r0, a, b0, MEAN, False)
    arms["ablate"] = {"edge": edge, "hop_seconds": sec, "M0": logit_pair(run, j, a, b0),
                      "M1": logit_pair(er, j, a, b0), "unbound_mlp": arms["donor"]["unbound_mlp"]}
    # city arm
    if item["family"] == "city":
        others = [c for c in city_by_country[item["bridge"]] if c != item["id"]]
        others.sort(key=lambda c: not correct.get(c, False))
        if others:
            dc = cand[others[0]]
            lg = run.read(f"t{j}.logits").reshape(-1).float()
            top2 = [int(x) for x in lg.topk(2).indices]
            runner = top2[1] if top2[0] == a else top2[0]
            rd = dc["row"] - (item["row"] - r0)
            er, reader, edge, sec = arm(run, item, j, r0, a, runner, cell_of(dc["ids"], rd), False)
            arms["city"] = {"edge": edge, "hop_seconds": sec, "donor": dc["id"],
                            "M0": logit_pair(run, j, a, runner), "M1": logit_pair(er, j, a, runner)}
    rec["arms"] = arms
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    e = arms["donor"]["edge"]
    print("Z2 " + json.dumps({
        "id": item["id"], "native": e["consumer"]["native_delta"]["value"],
        "job2": arms["donor"]["job2_delta"], "direct": e["direct"]["value"],
        "mlp": e["totals"]["mlp"]["value"], "reread": e["totals"]["reread"]["value"],
        "norm": e["norm"]["value"], "gap": e["closure"]["gap"]["value"],
        "lin": e["totals"]["mlp_linear"]["value"],
        "spec": [round(m["spectral"]["credit_top_bins_share"]["value"], 3) for m in e["mlp"][:3]],
        "perm": [round(m["spectral"]["credit_top_bins_share_permuted"]["value"], 3)
                 for m in e["mlp"][:3]],
        "table_ok": arms["donor"]["table_hop_matches"], "s": rec["seconds"]}), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
