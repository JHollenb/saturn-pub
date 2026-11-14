"""Shared setup for the showcase jobs (advisor memo 2026-11-14): model load, panels, carrier rule.

Panels are the two-hop Run 1 items (``r1_items.json``), the B1 candidate sets and truths
(``b1_sets.json``) and the B1 correctness (``b1-score.json``), copied into the payload from
``results/twohop-20261113/next/``. The carrier is the frozen B1 rule: L20 KV6 at R_argmax, the
argmax of the answer row's L20 h12+h13 attention over the span (``circuit_map.slot_rows``).
"""
import json
import os
import time
from pathlib import Path

import torch

from mdb.analysis import circuit_map as cm

MODEL = "/mnt/big/llm-models/Qwen3-1.7B-Base"
SAE_DIR = "/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
TC_DIR = "/mnt/big/llm-models/qwen3-1.7b-transcoders-lowl0"
CL, CG = 20, 6
T0 = time.monotonic()
HERE = Path(os.environ.get("DATA", "."))
SPEC = json.load(open(HERE / "r1_items.json"))
SETS = json.load(open(HERE / "b1_sets.json"))
SCORE = json.load(open(HERE / "b1-score.json"))["per_item"]
CARRIER = cm.Slot("carrier", "carrier", "R_argmax", layers=(CL,), kv=(CG,), rule_layer=CL,
                  rule_heads=(12, 13))


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def load():
    from mdb.repl.__main__ import _load

    torch.set_grad_enabled(False)
    return _load(Path(MODEL), "bf16", "cuda", True, None)


def items(group):
    return [i for i in SPEC["items"] if i["group"] == group]


def carrier_row(run, item):
    return cm.slot_rows(CARRIER, run, {"name": item["id"], "span": item["span"],
                                       "last": len(item["ids"]) - 1})[0]


def kept(logits, toks):
    lg = logits.reshape(-1).float()
    return {"logits": [float(lg[t]) for t in toks], "argmax": int(lg.argmax())}


def margin(logits, a, b):
    lg = logits.reshape(-1).float()
    return float(lg[a] - lg[b])


def spacing(x):
    """bf16 spacing at |x| (8 bits of mantissa incl. hidden: 2^(e-7))."""
    import math
    x = abs(float(x))
    if x == 0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(x)) - 7)


def floor_of(logits, a, b, extra=0.0):
    """2 bf16 spacings at the larger |logit| of the pair (the B1/P6 wording that held) plus a
    measured kernel-path delta."""
    lg = logits.reshape(-1).float()
    return 2 * spacing(max(abs(float(lg[a])), abs(float(lg[b])))) + abs(extra)


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)
