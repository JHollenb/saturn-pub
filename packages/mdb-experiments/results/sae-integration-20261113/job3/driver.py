"""Job 3: variable census (memo idea A, reader and source side) + per-layer causal error
accounting (idea B) on aligned entity pairs, Qwen3-1.7B-Base, one model load, engine verbs only.

Template "The city of {X} is the capital of" (entity row 3, pending row 7), consumer
M[country_X, country_Y] for the pair (X, Y). The driver picks feature ids / layers from earlier
op results and reads native logits through Run.read; every intervention is an mdb op.
"""
import json
import sys
import time
from pathlib import Path

from mdb.ops import OpError, dispatch
from mdb.repl.__main__ import _load
from mdb.repl.render import render, render_error

MODEL, SAE, PART, NPAIRS = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
READER_LAYERS = [int(x) for x in sys.argv[5].split(",")]
SOURCE_LAYERS = [int(x) for x in sys.argv[6].split(",")]
SPLICE_LAYERS = [int(x) for x in sys.argv[7].split(",")]
T0 = time.monotonic()
dbg = _load(MODEL, "bf16", "cuda", True, None)
tok = dbg.tokenizer
LOG = open(f"{PART}.jsonl", "w")
ctx = dbg.ops(None)
ROWS = []


def op(name, args, show=True):
    t = time.monotonic()
    try:
        r = dispatch(ctx, name, args)
        r.setdefault("op", name)
        text = render(r)
    except OpError as error:
        r = {"op": name, "error": str(error)}
        text = render_error(error)
    except Exception as error:  # noqa: BLE001
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
    return ids[0] if len(ids) == 1 else None


def margin(a, b):
    row = int(ctx.run.where.split(".")[0][1:])
    logits = ctx.run.read(f"t{row}.logits").reshape(-1).double()
    return float(logits[a] - logits[b]), int(logits.argmax())


def dm(result):
    m = result.get("measured", {})
    return m.get("delta", {}).get("value") if "delta" in m else None


op("sae", {"action": "load", "name": "qs", "path": SAE, "cache": 28})
# ---- prelude: Eiffel follow-up vs Colosseum, the residual feature/error patches (job2's arms
# that the '@RUN-' parse bug refused) and the entity-feature swaps, sealed whatif each
if PRELUDE := (len(sys.argv) <= 9 or sys.argv[9] != "noprelude"):
    P2 = "The Eiffel Tower is located in the city of Paris, It is the capital of"
    CF = "The Colosseum is located in the city of Rome, It is the capital of"
    Fr, It = one(" France"), one(" Italy")
    B = f"M[{Fr},{It}]"
    op("start", {"prompt": CF, "name": "rome"}, show=False)
    op("until", {"target": "logits"}, show=False)
    rome = {L: op("features", {"address": f"L{L}.h_out", "consumer": B, "top": 10},
                  show=False) for L in (20, 21, 22, 23)}
    op("start", {"prompt": P2, "name": "main"}, show=False)
    op("until", {"target": "logits"}, show=False)
    mine = {L: op("features", {"address": f"L{L}.h_out", "consumer": B, "top": 10},
                  show=False) for L in (20, 21, 22, 23)}
    for L in (16, 18, 20, 21, 22, 23):
        for value in ("recon@rome-recon", "error@rome-error"):
            op("whatif", {"target": f"L{L}.h_out", "value": value, "op": "add", "consumer": B})

    def picks(res, sign, n):
        return [f["feature"] for f in res.get("features", [])
                if sign * f["effect"]["value"] > 0][:n]

    for L in (20, 21, 22, 23):
        for n in (1, 3):
            chosen = list(dict.fromkeys(picks(mine[L], +1, n) + picks(rome[L], -1, n)))
            expr = "".join(f"+feature[{f},coef]@rome-feature[{f},coef]" for f in chosen)
            op("whatif", {"target": f"L{L}.h_out", "value": expr.lstrip("+"), "op": "add",
                          "consumer": B})
    print(f"PRELUDE T {time.monotonic() - T0:.1f}s", flush=True)

TABLE = [("Paris", "France"), ("Rome", "Italy"), ("Berlin", "Germany"), ("Moscow", "Russia"),
         ("Athens", "Greece"), ("Beijing", "China"), ("Madrid", "Spain"), ("London", "England"),
         ("Prague", "Czech"), ("Vienna", "Austria"), ("Amsterdam", "Netherlands"),
         ("Lisbon", "Portugal"), ("Budapest", "Hungary"), ("Tokyo", "Japan"), ("Cairo", "Egypt"),
         ("Seoul", "Korea"), ("Copenhagen", "Denmark"), ("Stockholm", "Sweden"),
         ("Brussels", "Belgium")]
TEMPLATE = "The city of {} is the capital of"
entries = []
for city, country in TABLE:
    ids = tok.encode(TEMPLATE.format(city), add_special_tokens=False)
    if one(" " + city) is None or one(" " + country) is None or len(ids) != 8:
        print("skip", city, country, len(ids))
        continue
    entries.append((city, country, ids, one(" " + country)))
OFFSET = int(sys.argv[10]) if len(sys.argv) > 10 else 0
pairs = [(entries[i], entries[(i + 1) % len(entries)])
         for i in range(len(entries))][OFFSET:OFFSET + NPAIRS]
print(f"{len(entries)} entries, {len(pairs)} pairs; load {time.monotonic() - T0:.1f}s", flush=True)

DETAIL = int(sys.argv[8]) if len(sys.argv) > 8 else 2


def native(ida, ka, kb, base, steps):
    """A fresh main run, edits staged at their stops (``(stop, target, value, op)``), run to
    logits natively; returns the margin change against ``base`` (measured)."""
    op("start", {"prompt": ida, "name": "main"}, show=False)
    for stop, target, value, kind in steps:
        if stop is not None:
            op("until", {"target": stop}, show=False)
        args = {"target": target, "value": value}
        if kind:
            args["op"] = kind
        res = op("set", args, show=False)
        if "error" in res:
            return None
    op("until", {"target": "logits"}, show=False)
    return margin(ka, kb)[0] - base


for index, ((xa, ya, ida, ka), (xb, yb, idb, kb)) in enumerate(pairs):
    t_pair = time.monotonic()
    C = f"M[{ka},{kb}]"
    row = {"pair": f"{xa}->{xb}", "consumer": C}
    print(f"===== pair {xa} ({ya}) <- {xb} ({yb})", flush=True)
    op("start", {"prompt": idb, "name": "cf"}, show=False)
    op("until", {"target": "logits"}, show=False)
    row["cf_margin"], row["cf_argmax"] = margin(ka, kb)
    op("start", {"prompt": idb[:4], "name": "tc"}, show=False)
    op("until", {"target": "logits"}, show=False)
    op("start", {"prompt": ida, "name": "main"}, show=False)
    op("until", {"target": "logits"}, show=False)
    base, row["argmax"] = margin(ka, kb)
    row["base"] = base
    row["argmax_text"] = tok.decode([row["argmax"]])
    prof = op("features", {"address": "L*.h_out", "consumer": C}, show=index < DETAIL)
    row["profile"] = [{"layer": r["layer"], "direct": r["direct"]["value"],
                       "error": r["error"]["value"], "features": r["features_sum"]["value"],
                       "bias": r["bias"]["value"],
                       "err_norm_frac": r["error_fraction_of_norm"]["value"]}
                      for r in prof.get("layers", [])]
    direct_err = {p["layer"]: p["error"] for p in row["profile"]}
    # idea B: causal error splice per layer (native), amplification vs the direct share
    row["splice"] = {}
    for layer in SPLICE_LAYERS:
        d = native(ida, ka, kb, base, [(f"L{layer + 1}.h", f"L{layer + 1}.h", "error*-1", "add")])
        row["splice"][layer] = {"native": d, "error_direct": direct_err.get(layer)}
        if index < DETAIL:
            op("start", {"prompt": ida, "name": "main"}, show=False)
            op("until", {"target": "logits"}, show=False)
            op("whatif", {"target": f"L{layer}.h_out", "value": "error*-1", "op": "add",
                          "consumer": C})
    # (a) reader side: value-cell transplant at the entity row, per layer (native)
    row["cell"] = {}
    for layer in READER_LAYERS:
        cell = f"cache.L{layer}.v[row=3]"
        row["cell"][layer] = native(ida, ka, kb, base, [(None, cell, f"cf:{cell}", None)])
    row["cell_k"] = {}
    good = {k: v for k, v in row["cell"].items() if v is not None}
    best = min(good, key=lambda k: good[k]) if good else None
    row["best_cell_layer"] = best
    if best is not None:
        cell = f"cache.L{best}.k[row=3]"
        row["cell_k"][best] = native(ida, ka, kb, base, [(None, cell, f"cf:{cell}", None)])
        # memo idea A, reader side: Delta x at L{best}.h_out (= L{best+1}.h) split exactly
        op("start", {"prompt": ida, "name": "T"}, show=False)
        op("set", {"target": f"cache.L{best}.v[row=3]",
                   "value": f"cf:cache.L{best}.v[row=3]"}, show=False)
        op("until", {"target": "logits"}, show=False)
        site = f"L{best + 1}.h"
        row["reader"] = {}
        for key, value, kind in (("delta", "recon@T+error@T", "replace"),
                                 ("phi", "recon@T-recon", "add"),
                                 ("E", "error@T-error", "add")):
            row["reader"][key] = native(ida, ka, kb, base, [(site, site, value, kind)])
        if index < DETAIL:
            op("start", {"prompt": ida, "name": "main"}, show=False)
            op("until", {"target": "logits"}, show=False)
            for value, kind in (("recon@T-recon", "add"), ("error@T-error", "add")):
                op("whatif", {"target": f"L{best}.h_out", "value": value, "op": kind,
                              "consumer": C})
            op("features", {"address": f"L{best}.h_out", "consumer": C, "top": 8, "lens": 4})
    # source side: the entity row's residual at L{l}.h_out, full / feature part / error part,
    # realised on the pending entity row of a truncated run and carried into the full prompt as
    # its K/V cells of layers l+1..27 (null arm = the truncated run unedited: stepped vs prefill)
    row["source"] = {}
    for layer in SOURCE_LAYERS:
        out = {}
        for arm, value, kind in (("null", None, None), ("full", "recon@tc+error@tc", "replace"),
                                 ("phi", "recon@tc-recon", "add"),
                                 ("E", "error@tc-error", "add")):
            op("start", {"prompt": ida[:4], "name": "tm"}, show=False)
            op("until", {"target": f"L{layer + 1}.h"}, show=False)
            if value is not None:
                res = op("set", {"target": f"L{layer + 1}.h", "value": value, "op": kind},
                         show=False)
                if "error" in res:
                    out[arm] = None
                    continue
            op("until", {"target": "logits"}, show=False)
            steps = [(None, f"cache.L{k}.{t}[row=3]", f"tm:cache.L{k}.{t}[row=3]", None)
                     for k in range(layer + 1, 28) for t in ("k", "v")]
            out[arm] = native(ida, ka, kb, base, steps)
        row["source"][layer] = out
    steps = [(None, f"cache.L{k}.{t}[row=3]", f"cf:cache.L{k}.{t}[row=3]", None)
             for k in range(28) for t in ("k", "v")]
    row["all_cells"] = native(ida, ka, kb, base, steps)
    row["seconds"] = time.monotonic() - t_pair
    ROWS.append(row)
    print("ROW " + json.dumps({k: v for k, v in row.items() if k not in ("profile",)},
                              default=str), flush=True)
    print(f"T {time.monotonic() - T0:.1f}s", flush=True)
json.dump(ROWS, open(f"{PART}-rows.json", "w"), default=str)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
