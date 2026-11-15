"""J1: the cell's code, country-mean vs panel-mean vs city-family-mean (next/prereg_j1.json).

python j1_code.py MODEL OUT [PICK]

Inputs: candidates.json, job1-panel.json, gt.json. One model load. For each of the 64 entry-1
recipients, the L20 KV6 V cell at its gt row is replaced by (i) the panel-mean cell, (ii) the
same-country mean (correct candidates sharing the bridge, excluding the item, each at its gt row
when it is a panel recipient, else its last descriptor row), (iii) a random other country's mean,
(iv) for city items the city-family mean (all other correct city items), (v) donor 0's cell.
Each edited run is read by symbols.hop_edge (consumer M[answer, donor-0 answer]) and by
symbols.table content (countries, capitals, entities) on the declared bridge cell.
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from mdb.analysis import symbols
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
spec = json.load(open("candidates.json"))
cand = {i["id"]: i for i in spec["items"]}
pj = json.load(open("job1-panel.json"))
full_panel = pj["panel"]
correct = {r["id"]: r["correct"] for r in pj["rows"]}
gt = json.load(open("gt.json"))
panel = [full_panel[int(k)] for k in PICK.split(",")] if PICK else full_panel
CACHE = {}
OUTF = open(f"{OUT}.jsonl", "w")


def first_token(text):
    return tok(" " + text, add_special_tokens=False)["input_ids"][0]


COUNTRIES = tuple(sorted({i["country_token"] for i in spec["items"]}))
CAPITALS = tuple(sorted(set(spec["capital_tokens"].values())))
ENT_OF = {i["id"]: first_token(i["descriptor"]) for i in spec["items"]}
ENTITIES = tuple(sorted(set(ENT_OF.values())))


def row_of(cid):
    return gt[cid]["row"] if cid in gt else cand[cid]["row"]


cell_cache = {}


def cell_of(cid):
    if cid not in cell_cache:
        c = cand[cid]
        run = dbg.run(c["ids"], name="cell")
        r = row_of(cid)
        cell_cache[cid] = run.read_cache(LAYER, "v", staged=False)[:, GROUP:GROUP + 1, r:r + 1].clone()
    return cell_cache[cid]


def mean_of(ids):
    cells = [cell_of(x) for x in ids]
    return torch.stack([x.float() for x in cells]).mean(0).to(cells[0].dtype)


ok = [i["id"] for i in spec["items"] if correct.get(i["id"])]
by_country = defaultdict(list)
for x in ok:
    by_country[cand[x]["bridge"]].append(x)
t = time.monotonic()
PANEL_MEAN = mean_of([e["id"] for e in full_panel])
CITY = [x for x in ok if cand[x]["family"] == "city"]
print(f"load {time.monotonic() - T0:.1f}s; cells {time.monotonic() - t:.1f}s; {len(panel)} items; "
      f"correct candidates {len(ok)}", flush=True)


def sets_for(item):
    return [symbols.Candidates("countries", COUNTRIES, item["country_token"]),
            symbols.Candidates("capitals", CAPITALS, item["answer"]),
            symbols.Candidates("entities", ENTITIES, ENT_OF[item["id"]])]


def content(run, item, r0, cons):
    symbols.declare(run, "bridge", [f"cache.L{LAYER}.v[kv{GROUP},row={r0}]"])
    tab = symbols.table(run, consumer=cons, sets=sets_for(item), limit=1)
    c = tab.report["bindings"][0]["content"]
    return {s: {"best": c[s]["best"], "text": tok.decode([c[s]["best"]]),
                "names_truth": c[s].get("names_truth")} for s in ("countries", "capitals", "entities")}


def arm(run, item, j, r0, cons, reader, value):
    er = dbg.run(item["ids"], name="edit")
    er.set_cell(LAYER, "v", value, rows=[r0], kv=[GROUP])
    er.until(f"t{j}.logits")
    lg = er.read().reshape(-1).float()
    edge = symbols.hop_edge(run, er, reader, cell_layer=LAYER, cache=CACHE)
    am = int(lg.argmax())
    return {"dM": edge["consumer"]["native_delta"]["value"], "direct": edge["direct"]["value"],
            "mlp": edge["totals"]["mlp"]["value"], "gap": edge["closure"]["gap"]["value"],
            "argmax": am, "argmax_text": tok.decode([am]), "flips": am != item["answer"],
            "content": content(er, item, r0, cons)}


for k, entry in enumerate(panel):
    t_item = time.monotonic()
    item = cand[entry["id"]]
    ids, j, a = item["ids"], item["n"] - 1, item["answer"]
    r0 = gt[item["id"]]["row"]
    d0 = cand[entry["donors"][0]]
    cons = Margin(j, a, d0["answer"])
    run = dbg.run(ids, name="main")
    run.until(f"t{j}.logits")
    reader = readout(run, cons)
    lg = run.read().reshape(-1).float()
    rec = {"id": item["id"], "family": item["family"], "bridge": item["bridge"], "r0": r0,
           "M0": float(lg[a] - lg[d0["answer"]]), "native_content": content(run, item, r0, cons)}
    members = [x for x in by_country[item["bridge"]] if x != item["id"]]
    others = sorted(c for c, v in by_country.items() if c != item["bridge"] and len(v) >= 2)
    other = random.Random(k).choice(others)
    arms = {"panel_mean": PANEL_MEAN, "donor0": cell_of(d0["id"]),
            "other_country_mean": mean_of(by_country[other])}
    if members:
        arms["country_mean"] = mean_of(members)
    if item["family"] == "city":
        arms["city_family_mean"] = mean_of([x for x in CITY if x != item["id"]])
    rec["n_country_members"] = len(members)
    rec["other_country"] = other
    rec["arms"] = {name: arm(run, item, j, r0, cons, reader, v) for name, v in arms.items()}
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    print("J1 " + json.dumps({"id": item["id"], "M0": rec["M0"], **{
        n: [round(v["dM"], 2), v["argmax_text"], v["content"]["countries"]["text"],
            v["content"]["entities"]["text"]] for n, v in rec["arms"].items()}, "s": rec["seconds"]}),
          flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
