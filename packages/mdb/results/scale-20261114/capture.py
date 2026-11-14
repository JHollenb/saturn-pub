"""Job C (prereg_scale.json): capture the 128-item panel on one model, no edits.

python capture.py MODEL_DIR OUTDIR [PICK]
Per item (stepped from the first span or distractor row, so those rows carry ports):
  k, v: every layer, every row (bf16, (L, KV, rows, dh));
  rows: for each stepped prompt row r < last, h and p at every layer (f32);
  ans: the answer row's h p o r m h_out at every layer, n g sg u f at layers >= L // 2 (f32);
  logits at the union of the candidate sets and the answers, argmax, top 5.
Writes OUTDIR/cap/<id>.pt and ./capture-<model>.json (ids, argmax, shapes, seconds). The log
prints ids and seconds only.
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.repl.__main__ import _load

MODEL, OUTDIR = sys.argv[1:3]
PICK = sys.argv[3] if len(sys.argv) > 3 else ""
TAG = sys.argv[4] if len(sys.argv) > 4 else ("-smoke" if PICK else "")
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
ad = dbg.adapter
L, H, KV, DH = ad.layers, ad.num_heads, ad.num_kv_heads, ad.head_dim
P = json.load(open("panel.json"))
items = P["items"]
if PICK:
    sel = [int(k) for k in PICK.split(",")] if "," in PICK or "-" not in PICK else \
        list(range(int(PICK.split("-")[0]), int(PICK.split("-")[1])))
    items = [items[k] for k in sel]
TOKS = sorted({t for v in P["sets"].values() for t in v} | {i["answer"] for i in P["items"]})
OUT = Path(OUTDIR) / "cap"
OUT.mkdir(parents=True, exist_ok=True)
name = Path(MODEL).name
info = {"model": name, "layers": L, "heads": H, "kv": KV, "head_dim": DH,
        "d_model": ad.model.config.hidden_size,
        "gqa_map": [h // (H // KV) for h in range(H)],
        "tied_embeddings": bool(getattr(ad.model.config, "tie_word_embeddings", False)),
        "v_bias": getattr(ad.backbone.layers[0].self_attn.v_proj, "bias", None) is not None,
        "q_norm": hasattr(ad.backbone.layers[0].self_attn, "q_norm"),
        "tokens": TOKS, "items": {}}
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def f32(x):
    return x.detach().float().cpu()


for item in items:
    t = time.monotonic()
    ids, span = item["ids"], item["span"]
    last = len(ids) - 1
    start = min(span + item.get("distractor", {}).get("rows", []))
    run = dbg.run(ids, name="nat", prefill=start)
    run.until(f"t{last}.logits")
    cap = {"id": item["id"], "span": span, "last": last, "prefill": start,
           "k": torch.stack([run.read_cache(x, "k", staged=False)[0].cpu() for x in range(L)]),
           "v": torch.stack([run.read_cache(x, "v", staged=False)[0].cpu() for x in range(L)]),
           "rows": {r: {"h": torch.stack([f32(run.read(f"t{r}.L{x}.h")).reshape(-1) for x in range(L)]),
                        "p": torch.stack([f32(run.read(f"t{r}.L{x}.p")).reshape(H, -1) for x in range(L)])}
                    for r in range(start, last)},
           "ans": {k: torch.stack([f32(run.read(f"t{last}.L{x}.{k}")).reshape(-1) for x in range(L)])
                   for k in ("h", "o", "r", "m", "h_out")}}
    cap["ans"]["p"] = torch.stack([f32(run.read(f"t{last}.L{x}.p")).reshape(H, -1) for x in range(L)])
    for k in ("n", "g", "sg", "u", "f"):
        cap["ans"][k] = {x: f32(run.read(f"t{last}.L{x}.{k}")).reshape(-1) for x in range(L // 2, L)}
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    cap["logits"] = lg[TOKS].cpu()
    cap["argmax"] = int(lg.argmax())
    cap["top5"] = [int(i) for i in lg.topk(5).indices]
    torch.save(cap, OUT / f"{safe(item['id'])}.pt")
    info["items"][item["id"]] = {"argmax": cap["argmax"], "top5": cap["top5"], "cache_rows": int(cap["k"].shape[2]),
                                 "seconds": round(time.monotonic() - t, 2)}
    del run, cap
    print("C " + json.dumps({"id": item["id"], "s": info["items"][item["id"]]["seconds"]}), flush=True)
info["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
json.dump(info, open(f"capture-{name}{TAG}.json", "w"), indent=1)
print(f"TOTAL {time.monotonic() - T0:.1f}s peak {info['peak_alloc_mb']} MB", flush=True)
