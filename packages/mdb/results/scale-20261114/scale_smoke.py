"""Feasibility smoke (no scoring): peak VRAM and seconds per model for the B1 pipeline steps on
one two-hop item and the longest d4 item. python scale_smoke.py MODEL_DIR [MODEL_DIR ...]
Prints only resource numbers and tokenizer agreement, no outcome values."""
import gc
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

torch.set_grad_enabled(False)
base = json.load(open("candidates.json"))["items"]
ext = json.load(open("candidates_ext.json"))["items"]
two = base[0]
d4 = max((i for i in ext if i["family"] == "d4-work"), key=lambda i: i["n"])
q3_ids = {i["id"]: i["ids"] for i in base + ext}
for path in sys.argv[1:]:
    t0 = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    dbg = _load(Path(path), "bf16", "cuda", True, None)
    ad, tok = dbg.adapter, dbg.tokenizer
    load_s = time.monotonic() - t0
    weights_mb = torch.cuda.memory_allocated() / 2**20
    L, H, KV = ad.layers, ad.num_heads, ad.num_kv_heads
    agree = sum(tok.encode(i["prompt"], add_special_tokens=False) == q3_ids[i["id"]] for i in base + ext)
    cl = round(20 / 28 * L)
    car = cm.Slot("carrier", "carrier", "R_argmax", layers=(cl,), kv=(KV - 2,), rule_layer=cl,
                  rule_heads=(H - 4, H - 3))
    m = cm.Map([car, cm.Slot("desc", "descriptor", "rest")], [], {}, {"countries": json.load(open("b1_sets.json"))["sets"]["countries"]})
    steps = {}
    for name, item in (("two", two), ("d4", d4)):
        ids = tok.encode(item["prompt"], add_special_tokens=False)
        last = len(ids) - 1
        span = [r for r in item["span"] if r < last] or [last - 1]
        t = time.monotonic()
        run = dbg.run(ids, name="nat", prefill=span[0])
        run.until(f"t{last}.logits")
        rep = cm.item_report(m, run, {"name": name, "span": span, "last": last},
                             consumer=Margin(last, 0, 1), truths={"countries": item.get("country_token")})
        crow = cm.slot_rows(car, run, {"span": span, "last": last})[0]
        WR = {x: cm.group_writers(layer_module(ad, x).self_attn.o_proj.weight, H, KV) for x in range(L)}
        values = {x: run.read_cache(x, "v", staged=False) for x in range(L)}
        cm.cell_census(values, WR, span, {"c": torch.randn(4, ad.model.config.hidden_size, dtype=torch.float64)})
        cell = run.read_cache(cl, "v", staged=False)[:, KV - 2:KV - 1, crow:crow + 1].clone() * 0.5
        ed = dbg.run(ids, name="ed", prefill=span[0])
        ed.until(f"t{last}.L{cl}.h")
        ed.set_cell(cl, "v", cell, rows=[crow], kv=[KV - 2])
        ed.until(f"t{last}.logits")
        symbols.hop_edge(run, ed, readout(run, Margin(last, 0, 1)), cell_layer=cl, top_neurons=2)
        src = [r for r in span if r < crow]
        if src:
            ce = dbg.run(ids, name="e1", prefill=span[0])
            ce.until(f"t{crow}.L0.h")
            for x in range(cl):
                v = run.read_cache(x, "v", staged=False)[:, :, src].clone() * 0.5
                ce.set_cell(x, "v", v, rows=src, kv=list(range(KV)))
            ce.until(f"t{last}.logits")
            r = symbols.cell_readout(run, symbols.CellTarget(crow, cl, KV - 2, 0, 1))
            symbols.hop_edge(run, ce, r, cell_layer=0, top_neurons=2)
            del ce
        steps[name] = {"n_tokens": len(ids), "seconds": round(time.monotonic() - t, 2)}
        del run, ed, values
    peak = torch.cuda.max_memory_allocated() / 2**20
    print("SMOKE " + json.dumps({"model": Path(path).name, "layers": L, "heads": H, "kv": KV,
                                 "d_model": ad.model.config.hidden_size, "load_s": round(load_s, 1),
                                 "weights_mb": round(weights_mb), "peak_alloc_mb": round(peak),
                                 "tokenizer_agrees_with_qwen3": [agree, len(base) + len(ext)],
                                 "steps": steps}), flush=True)
    del dbg, ad
    gc.collect()
    torch.cuda.empty_cache()
