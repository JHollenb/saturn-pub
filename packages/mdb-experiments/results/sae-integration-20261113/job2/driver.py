"""Job 2: SAE x mdb on Qwen3-1.7B-Base, one model load, engine verbs only (dispatch).

Every command is an mdb op through ``mdb.ops.dispatch``; its rendered text goes to stdout and
its JSON to PART.jsonl. The driver only chooses which feature ids to feed the next command
from earlier op results (no model arithmetic here).
"""
import json
import sys
import time
from pathlib import Path

from mdb.ops import OpError, dispatch
from mdb.repl.__main__ import _load
from mdb.repl.render import render, render_error

MODEL, SAE, PART = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
T0 = time.monotonic()
dbg = _load(MODEL, "bf16", "cuda", True, None)
print(f"load {time.monotonic() - T0:.1f}s", flush=True)
tok = dbg.tokenizer
LOG = open(f"{PART}.jsonl", "w")
ctx = dbg.ops(None)


def op(name, args, show=True):
    t = time.monotonic()
    try:
        r = dispatch(ctx, name, args)
        r.setdefault("op", name)
        text = render(r)
    except OpError as error:
        r = {"op": name, "error": str(error)}
        text = render_error(error)
    except Exception as error:  # noqa: BLE001 - recorded, the job goes on
        r = {"op": name, "error": f"{type(error).__name__}: {error}"}
        text = f"error (internal): {r['error']}"
    dt = time.monotonic() - t
    if show:
        print(f"(mdb) {name} {json.dumps(args)}  [{dt:.2f}s]\n{text}", flush=True)
    LOG.write(json.dumps({"op": name, "args": args, "seconds": dt, "result": r}) + "\n")
    LOG.flush()
    return r


def one(word):
    ids = tok.encode(word, add_special_tokens=False)
    assert len(ids) == 1, (word, ids)
    return ids[0]


def top_effects(res, sign=+1, n=3):
    feats = res.get("features", [])
    picked = [f["feature"] for f in feats if "effect" in f and sign * f["effect"]["value"] > 0]
    return picked[:n]


op("sae", {"action": "load", "name": "qs", "path": SAE, "cache": 28, "verify": True})
Pa, Ro, Fr, It = one(" Paris"), one(" Rome"), one(" France"), one(" Italy")
A, B = f"M[{Pa},{Ro}]", f"M[{Fr},{It}]"
P1 = "The Eiffel Tower is located in the city of"
P2 = P1 + " Paris, It is the capital of"
CF = "The Colosseum is located in the city of Rome, It is the capital of"


def study(prompt, consumer, sites, units, splice_layers, trace_sites):
    op("start", {"prompt": prompt})
    op("until", {"target": "logits"})
    op("print", {"address": "logits", "top": 3})
    op("features", {"address": "L*.h_out", "consumer": consumer})                    # item 1
    for shift in (-1, 1):                                       # site alignment check (memo a)
        op("features", {"address": "L*.h_out", "consumer": consumer, "shift": shift})
    found = {}
    for layer in sites:
        found[layer] = op("features", {"address": f"L{layer}.h_out", "consumer": consumer,
                                       "top": 10, "lens": 5, "circuit": layer == sites[-1]})
    built = found[sites[-1]].get("circuit") if sites else None
    if built:
        op("circuit", {"target": "show", "id": built["id"]})
    for site, unit in units:                                                         # item 2
        op("features", {"address": site, "match": unit, "top": 8, "lens": 5})
    for layer in splice_layers:                                                      # item 1 (native)
        op("whatif", {"target": f"L{layer}.h_out", "value": "error*-1", "op": "add",
                      "consumer": consumer})
    for layer in sites:                                                              # item 3
        picked = top_effects(found[layer], +1, 3)
        for f in picked:
            op("whatif", {"target": f"L{layer}.h_out", "value": f"feature[{f},coef]*-1",
                          "op": "add", "consumer": consumer})
        if picked:
            expr = "".join(f"-feature[{f},coef]" for f in picked).lstrip("+")
            op("whatif", {"target": f"L{layer}.h_out", "value": expr, "op": "add",
                          "consumer": consumer})
            op("whatif", {"target": f"L{layer}.h_out", "value": f"feature[{picked[0]},coef]*1",
                          "op": "add", "consumer": consumer})
    for layer in sites[:2]:                                     # steering dose curve
        picked = top_effects(found[layer], +1, 1)
        for alpha in (0.5, 1, 2, 4):
            if picked:
                op("whatif", {"target": f"L{layer}.h_out",
                              "value": f"feature[{picked[0]},coef]*{alpha}", "op": "add",
                              "consumer": consumer})
    for layer in trace_sites:                                                        # item 4
        picked = top_effects(found.get(layer, {}), +1, 2)
        for f in picked:
            op("feature", {"feature": f, "layer": layer, "top": 4})
    return found


def chain(prompt, consumer_ids):
    """Native replacement model: every site L0..L26.h_out replaced by its SAE reconstruction
    (staged at L{l+1}.h on the already-edited value), then the native logits."""
    op("start", {"prompt": prompt, "name": "chain"})
    for layer in range(1, 28):
        op("until", {"target": f"L{layer}.h"}, show=False)
        op("set", {"target": f"L{layer}.h", "value": "error*-1", "op": "add"}, show=False)
    op("until", {"target": "logits"})
    op("print", {"address": "logits", "top": 5})
    row = int(ctx.run.where.split(".")[0][1:])
    logits = ctx.run.read(f"t{row}.logits").reshape(-1).double()
    a, b = consumer_ids
    print(f"CHAIN margin {float(logits[a] - logits[b]):.4f} [measured, under 27 splices] argmax "
          f"{int(logits.argmax())} {tok.decode([int(logits.argmax())])!r}", flush=True)
    op("features", {"address": "L26.h_out", "consumer": f"M[{a},{b}]"}, show=True)
    op("frame", {"address": "main:"}, show=False)


print("===== recall", flush=True)
study(P1, A, [20, 22, 23], [("L22.h_out", "L22.f[i=5773]"), ("L20.h_out", "L20.A[h13]"),
                             ("L20.h_out", "L20.A[h13,src=2]")],
      [4, 8, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [20, 22])
print(f"T {time.monotonic() - T0:.1f}s", flush=True)

print("===== followup", flush=True)
found = study(P2, B, [20, 21, 22, 23],
              [("L23.h_out", "L23.f[i=2298]"), ("L22.h_out", "L22.f[i=5773]"),
               ("L20.h_out", "L20.A[h13]"), ("L20.h_out", "L20.A[h13,src=11]")],
              [8, 14, 16, 18, 19, 20, 21, 22, 23, 24, 26], [20, 23])
print(f"T {time.monotonic() - T0:.1f}s", flush=True)

print("===== native replacement model (all sites spliced)", flush=True)
chain(P1, (Pa, Ro))
chain(P2, (Fr, It))
print(f"T {time.monotonic() - T0:.1f}s", flush=True)

print("===== (c) steering vs transplant, rome counterfactual", flush=True)
op("checkpoint", {"name": "main_logits"})
op("start", {"prompt": CF, "name": "rome"})
op("until", {"target": "logits"})
op("print", {"address": "logits", "top": 3})
rome = {}
for layer in (20, 21, 22, 23):
    rome[layer] = op("features", {"address": f"L{layer}.h_out", "consumer": B, "top": 10,
                                  "lens": 5})
op("frame", {"address": "main:"})
op("where", {})
for cell in ("cache.L20.v[row=11]", "cache.L20.v[kv6,row=11]", "cache.L20.k[kv6,row=11]",
             "cache.L22.v[row=11]"):
    op("whatif", {"target": cell, "value": f"rome:{cell}", "op": "replace", "consumer": B})
for layer in (16, 18, 20, 21, 22, 23):
    site = f"L{layer}.h_out"
    op("whatif", {"target": site, "value": "recon@rome+error@rome", "op": "replace",
                  "consumer": B})                              # whole last-row residual
    op("whatif", {"target": site, "value": "recon@rome-recon", "op": "add", "consumer": B})
    op("whatif", {"target": site, "value": "error@rome-error", "op": "add", "consumer": B})
for layer in (20, 21, 22, 23):
    mine = top_effects(found[layer], +1, 3)
    theirs = top_effects(rome[layer], -1, 3)
    chosen = list(dict.fromkeys(mine + theirs))
    expr = "".join(f"+feature[{f},coef]@rome-feature[{f},coef]" for f in chosen).lstrip("+")
    if chosen:
        op("whatif", {"target": f"L{layer}.h_out", "value": expr, "op": "add", "consumer": B})
    if mine and theirs:
        op("compare", {"a": f"L{layer}.h_out=" + "".join(
            f"-feature[{f},coef]" for f in mine).lstrip("+") + " add",
            "b": f"L{layer}.h_out=" + "".join(
            f"+feature[{f},coef]@rome" for f in theirs).lstrip("+") + " add",
            "consumer": B})
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
