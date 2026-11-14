"""Entity vs bridge (prereg_ext_addendum.json part B), Qwen3-1.7B-Base, carrier L20 KV6 at r0.

python eb_driver.py MODEL candidates.json job1-panel.json OUT START STOP [eb2]

EB1 (recipients START..STOP-1 of the entry-1 panel): one single-request MEMIT patch per
recipient (mdb_lanes.memit, layers 4-8) rewrites hop 1 (descriptor -> country) to the first
donor's country. With the patch installed (a removable down_proj overlay): hop-1 efficacy, the
two-hop answer, the carrier readout (attention-weighted L20 KV6 write at r0 projected through the
final-norm gain and the unembedding, best of the panel's country tokens) and the restore test
(native cell under the patch). Without the patch: the plant test (patched cell, and the patched
V band L8-27 at r0, set into the unpatched model). Locality: 8 other recipients' two-hop argmax.
EB2 (with arg eb2, START/STOP ignored): city recipients x same-country different-city donors.
All model runs are mdb stepped runs (Run.set_cell for cells).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.reads import head_mix
from mdb.repl.__main__ import _load

MODEL, CAND, PANEL, OUT = sys.argv[1:5]
START, STOP = int(sys.argv[5]), int(sys.argv[6])
MODE = sys.argv[7] if len(sys.argv) > 7 else "eb1"
LAYER, G = 20, 6
T0 = time.monotonic()
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok, model = dbg.tokenizer, dbg.model
L = dbg.adapter.layers
spec = json.load(open(CAND))
cand = {i["id"]: i for i in spec["items"]}
panel = json.load(open(PANEL))["panel"]
countries = sorted({i["country_token"] for i in spec["items"] if i["country_token"]})
norm = getattr(dbg.adapter.backbone, dbg.adapter.spec.final_norm)
W = model.lm_head.weight
OUTF = open(f"{OUT}.jsonl", "w")
TARGET = {"Korea": "South Korea", "Britain": "England", "Czech": "Czech Republic",
          "Netherlands": "Holland"}
ATTR = {"currency": "The {} is the currency of", "person": "{} was born in the country of",
        "landmark": "{} is located in the country of", "city": "{} is a city in the country of"}
PARA = {"currency": "The country that uses the {} as its currency is",
        "person": "The country where {} was born is",
        "landmark": "The country where {} can be found is",
        "city": "The country where the city of {} lies is"}


def first(word):
    return tok.encode(" " + word, add_special_tokens=False)[0]


def stepped(ids, cells=()):
    run = dbg.run(ids, name="arm")
    for layer, value, rows, kv in cells:
        run.set_cell(layer, "v", value, rows=rows, kv=kv)
    run.until(f"t{len(ids) - 1}.logits")
    return run


def readout(run, item):
    mix = head_mix(run, item["n"] - 1, LAYER, {})
    write = sum(mix.o_parts()[h, item["row"]] for h in range(mix.p.shape[0]) if mix.kappa[h] == G)
    x = (write * norm.weight.double().cpu()).to(device=W.device, dtype=W.dtype)
    proj = (W[countries] @ x).float()
    return countries[int(proj.argmax())]


def cls3(token, new, old):
    return "new" if token == new else "old" if token == old else "other"


def bare_argmax(text):
    ids = tok.encode(text, add_special_tokens=False)
    run = dbg.run(ids, name="bare")
    run.until(f"t{len(ids) - 1}.logits")
    return int(run.read().reshape(-1).argmax())


if MODE == "eb2":
    cities = [i for i in spec["items"] if i["family"] == "city"]
    rows = []
    for entry in panel:
        rec = cand[entry["id"]]
        if rec["family"] != "city":
            continue
        base = stepped(rec["ids"])
        nat = int(base.read().reshape(-1).argmax())
        for d in [c for c in cities if c["bridge"] == rec["bridge"] and c["id"] != rec["id"]]:
            drun = dbg.run(d["ids"], name="donor")
            dv = [drun.read_cache(layer, "v", staged=False)[:, :, d["row"]:d["row"] + 1].clone()
                  for layer in range(L)]
            copy_tok = first(d["descriptor"])
            one = int(stepped(rec["ids"], [(LAYER, dv[LAYER][:, G:G + 1], [rec["row"]], [G])])
                      .read().reshape(-1).argmax())
            band = int(stepped(rec["ids"], [(layer, dv[layer], [rec["row"]], None)
                                            for layer in range(8, L)]).read().reshape(-1).argmax())

            def score(t):
                return "kept" if t == rec["answer"] else "copied" if t == copy_tok else "other"
            row = {"recipient": rec["id"], "donor": d["id"], "native": tok.decode([nat]),
                   "cell": score(one), "cell_text": tok.decode([one]),
                   "band": score(band), "band_text": tok.decode([band])}
            rows.append(row)
            OUTF.write(json.dumps(row) + "\n")
            print("EB2 " + json.dumps(row), flush=True)
    print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------- EB1: MEMIT hop-1 edits
from mdb_lanes.memit import (MEMITHyperParams, generate_context_templates,  # noqa: E402
                             layer_second_moments, memit)

hp = MEMITHyperParams()
cache = Path("/mnt/ssd1tb/cache/beast/twohop-memit")
cache.mkdir(parents=True, exist_ok=True)
c0_path = cache / "C0-qwen3-1.7b-base-L4-8-200k.pt"
text_path = Path("/mnt/usb-storage/wikitext103_train_paragraphs_20mb.txt")
t = time.monotonic()
if c0_path.exists():
    C0, counted = torch.load(c0_path, map_location="cuda")
    C0 = {int(k): v.cuda() for k, v in C0.items()}
else:
    texts = text_path.read_text().split("\n")
    with torch.no_grad():
        C0, counted = layer_second_moments(model, tok, texts, list(hp.layers), n_tokens=200_000,
                                           max_len=512, batch=2)
    torch.save(({k: v.cpu() for k, v in C0.items()}, counted), str(c0_path) + ".tmp")
    os.replace(str(c0_path) + ".tmp", c0_path)
mom2 = {"dataset": str(text_path), "tokens": counted, "cache": str(c0_path)}
templates = generate_context_templates(model, tok, hp.context_template_length_params, hp.seed)
print(f"C0 tokens {counted} ({time.monotonic() - t:.1f}s); templates {len(templates)}; "
      f"load {time.monotonic() - T0:.1f}s", flush=True)

recips = panel[START:STOP]
natives = {}
for entry in panel:
    rec = cand[entry["id"]]
    run = stepped(rec["ids"])
    natives[rec["id"]] = {"argmax": int(run.read().reshape(-1).argmax()), "read": readout(run, rec),
                          "cell": run.read_cache(LAYER, "v", staged=False)[:, G:G + 1,
                                                                          rec["row"]:rec["row"] + 1].clone()}
for k, entry in enumerate(recips):
    t = time.monotonic()
    rec = cand[entry["id"]]
    don = cand[entry["donors"][0]]
    fam = rec["family"]
    target = TARGET.get(don["bridge"], don["bridge"])
    req = {"id": rec["id"], "prompt": ATTR[fam], "subject": rec["descriptor"],
           "target_new": " " + target}
    patch, diag = memit(model, tok, [req], C0, hp, context_templates=templates, mom2_info=mom2)
    fit_s = time.monotonic() - t
    new_c, old_c = don["answer"], rec["answer"]
    nb, ob = don["country_token"], rec["country_token"]
    others = [cand[e["id"]] for e in panel if e["id"] != rec["id"]][(k * 7) % 50:(k * 7) % 50 + 8]
    with patch.install(model):
        h1 = bare_argmax(ATTR[fam].format(rec["descriptor"]))
        h1p = bare_argmax(PARA[fam].format(rec["descriptor"]))
        run = stepped(rec["ids"])
        two = int(run.read().reshape(-1).argmax())
        read = readout(run, rec)
        edited_cell = run.read_cache(LAYER, "v", staged=False)[:, G:G + 1, rec["row"]:rec["row"] + 1].clone()
        edited_band = [run.read_cache(layer, "v", staged=False)[:, :, rec["row"]:rec["row"] + 1].clone()
                       for layer in range(L)]
        restore = int(stepped(rec["ids"], [(LAYER, natives[rec["id"]]["cell"], [rec["row"]], [G])])
                      .read().reshape(-1).argmax())
        loc = [int(stepped(o["ids"]).read().reshape(-1).argmax()) == natives[o["id"]]["argmax"]
               for o in others]
    plant = int(stepped(rec["ids"], [(LAYER, edited_cell, [rec["row"]], [G])]).read().reshape(-1).argmax())
    plant_band = int(stepped(rec["ids"], [(layer, edited_band[layer], [rec["row"]], None)
                                          for layer in range(8, L)]).read().reshape(-1).argmax())
    undo = int(stepped(rec["ids"]).read().reshape(-1).argmax()) == natives[rec["id"]]["argmax"]
    tgt_first = first(target)
    row = {"recipient": rec["id"], "donor": don["id"], "new_bridge": don["bridge"],
           "old_bridge": rec["bridge"], "target_new": target,
           "hop1_top1": tok.decode([h1]), "hop1_efficacy": h1 == tgt_first,
           "hop1_paraphrase_top1": tok.decode([h1p]), "hop1_paraphrase_efficacy": h1p == tgt_first,
           "two_hop": cls3(two, new_c, old_c), "two_hop_text": tok.decode([two]),
           "readout": cls3(read, nb, ob), "readout_text": tok.decode([read]),
           "native_readout": cls3(natives[rec["id"]]["read"], nb, ob),
           "restore_native_cell": cls3(restore, new_c, old_c),
           "plant_edited_cell": cls3(plant, new_c, old_c), "plant_text": tok.decode([plant]),
           "plant_edited_band": cls3(plant_band, new_c, old_c),
           "plant_band_text": tok.decode([plant_band]),
           "cell_change_norm": float((edited_cell.float() - natives[rec["id"]]["cell"].float()).norm()),
           "cell_native_norm": float(natives[rec["id"]]["cell"].float().norm()),
           "locality": sum(loc) / len(loc), "undo_exact_argmax": undo,
           "z_nll_final": diag["requests"][rec["id"]].get("nll_final") if "requests" in diag else None,
           "fit_s": round(fit_s, 1), "seconds": round(time.monotonic() - t, 1)}
    OUTF.write(json.dumps(row) + "\n")
    OUTF.flush()
    print("EB1 " + json.dumps(row), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
