"""Shared definitions for the scale study (prereg_scale.json 'operational'): floors, carriers."""
import collections
import json
import math
from pathlib import Path

import torch


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def floor(la, lb):
    """2 bf16 spacings at the larger |logit| of the pair."""
    m = max(abs(float(la)), abs(float(lb)), 1e-30)
    return 2 * 2.0 ** (math.floor(math.log2(m)) - 7)


def correct_map(panel, capinfo):
    return {i["id"]: capinfo[i["id"]]["argmax"] == i["answer"] for i in panel["items"]}


def capture_info(paths):
    out = {}
    for p in paths:
        out.update(json.load(open(p))["items"])
    return out


def carriers(panel, sweep, correct, fixed=None):
    """item carrier (argmax of -dM above floor), model carrier (modal (layer, kv) over correct base
    items), carrier_at_item row (argmin dM over span rows at the model carrier's (layer, kv))."""
    item = {}
    for it in panel["items"]:
        s = sweep[it["id"]]
        i = int(s["dM"].argmin())
        x, k, r = (int(v) for v in s["cells"][i])
        fl = floor(s["la"][i], s["lb"][i])
        item[it["id"]] = {"cell": [x, k, r], "dM": float(s["dM"][i]), "floor": fl,
                          "above_floor": abs(float(s["dM"][i])) > fl}
    base_ok = [it["id"] for it in panel["items"] if it["block"] == "base" and correct[it["id"]]
               and item[it["id"]]["above_floor"]]
    cnt = collections.Counter(tuple(item[i]["cell"][:2]) for i in base_ok)
    if fixed is not None:          # supplement panel: the model carrier frozen from the main panel
        (lc, kc), n = tuple(fixed), cnt.get(tuple(fixed), 0)
    else:
        (lc, kc), n = cnt.most_common(1)[0]
    at = {}
    for it in panel["items"]:
        s = sweep[it["id"]]
        sel = (s["cells"][:, 0] == lc) & (s["cells"][:, 1] == kc)
        idx = torch.nonzero(sel).reshape(-1)
        j = int(idx[s["dM"][idx].argmin()])
        at[it["id"]] = int(s["cells"][j, 2])
    return {"L_c": int(lc), "kv_c": int(kc), "mode_n": n, "mode_of": len(base_ok),
            "mode_counts": [[list(k), v] for k, v in cnt.most_common(8)], "item": item, "row_at_item": at}


def load_sweep(prefixes):
    out = {}
    for p in prefixes:
        out.update(torch.load(p + ".pt", weights_only=False))
    return out


def load_cap(capdir, iid):
    return torch.load(Path(capdir) / "cap" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
