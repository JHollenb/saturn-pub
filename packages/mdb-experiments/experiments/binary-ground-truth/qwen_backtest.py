"""Qwen back-test of the binary lessons (PREREG-QWEN.md in saturn experiments/2026-11-16-binary-ground-truth).

    python qwen_backtest.py MODEL ITEMS_JSON OUT [N_PER_FAMILY]

One model load (mdb ``load_local``, bf16 eager). Per item, batched native forwards: residual
``h_out`` at (row, layer) replaced by a donor's same-position value, arms ``total`` and ``frozen``
(every downstream ``p`` held at the base's pattern). The verdict rules are mdb's:
``analysis.donors.informative`` / ``pooled`` and ``analysis.paths.argmax_rows`` / ``diverged``.
The eager attention is re-implemented here so ``p`` can be held per lane; it is checked bitwise
against the stock forward on every base. A few patches are re-measured through
``analysis.paths.paired`` (stepped mdb, the authority) at the end.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.donors import EFFECT, NO_EFFECT, informative, pooled
from mdb.analysis.paths import argmax_rows

T0 = time.monotonic()
LAYERS = (0, 3, 7, 11, 15, 19, 23, 26)
N_DONORS, SEED, BATCH = 16, 20261116, 192
DEV = os.environ.get("DEV", "cuda")


def log(m):
    print(f"[{time.monotonic() - T0:7.1f}s] {m}", flush=True)


def spacing(x):
    x = abs(float(x))
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x else 2.0 ** -133


class Lab:
    """Batched forwards with per-lane h_out patches, per-lane p freezing and argmax capture."""

    def __init__(self, model):
        from transformers.models.qwen3 import modeling_qwen3 as mq

        self.m, self.mq = model, mq
        self.layers = model.model.layers
        self.L = len(self.layers)
        self.original = mq.eager_attention_forward
        self.patch = {}          # layer -> (lane idx tensor, row idx tensor, values [n, d])
        self.freeze_from = None  # [B] long: hold p at layers >= this (base_w)
        self.base_w = None       # list per layer [H, T, T]
        self.record_w = False
        self.am = None           # [B, L, H, T] argmax per p row (-1 tie)
        self.w = []
        self.h_keep = None       # layers whose h_out to keep
        self.h = {}
        lab = self

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            ks = mq.repeat_kv(key, module.num_key_value_groups)
            vs = mq.repeat_kv(value, module.num_key_value_groups)
            w = torch.matmul(query, ks.transpose(2, 3)) * scaling
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    w = w.masked_fill(~attention_mask[:, :, :, :ks.shape[-2]], torch.finfo(w.dtype).min)
                else:
                    w = w + attention_mask[:, :, :, :ks.shape[-2]]
            w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
            i = module.layer_idx
            if lab.freeze_from is not None:
                sel = lab.freeze_from <= i
                if bool(sel.any()):
                    w = w.clone()
                    w[sel] = lab.base_w[i].to(w.dtype)
            if lab.record_w:
                lab.w.append(w[0].clone())
            if lab.am is not None:
                lab.am[:, i] = argmax_rows(w).reshape(w.shape[0], w.shape[1], w.shape[2]).to(torch.int16)
            out = torch.matmul(w, vs).transpose(1, 2).contiguous()
            return out, w

        self.eager = eager
        for li, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(li))

    def _hook(self, li):
        def hook(mod, args, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self.h_keep is not None and li in self.h_keep:
                self.h[li] = hs.detach().clone()
            if li in self.patch:
                lanes, rows, vals = self.patch[li]
                hs = hs.clone()
                hs[lanes, rows] = vals.to(hs.dtype)
                return (hs, *out[1:]) if isinstance(out, tuple) else hs
            return None
        return hook

    @torch.no_grad()
    def forward(self, ids, *, mine=True):
        self.mq.eager_attention_forward = self.eager if mine else self.original
        try:
            x = torch.as_tensor(ids, device=DEV)
            hs = self.m.model(input_ids=x).last_hidden_state[:, -1]
            return self.m.lm_head(hs).float()
        finally:
            self.mq.eager_attention_forward = self.original


def main(model_dir, items_json, out, per_family=8):
    from mdb.engine.models import load_local

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    model = load_local(Path(model_dir), dtype="bf16", device=DEV)
    model.eval()
    lab = Lab(model)
    log(f"load: {lab.L} layers, attn {model.config._attn_implementation}")
    allitems = [i for i in json.load(open(items_json))["items"] if i["group"] == "base"]
    byid = {i["id"]: i for i in allitems}

    # every base item once: native argmax, h_out at the grid layers (donor values)
    H, native = {}, {}
    lab.h_keep = set(LAYERS)
    for it in allitems:
        lab.h = {}
        lg = lab.forward([it["ids"]])
        H[it["id"]] = {l: lab.h[l][0] for l in LAYERS}
        native[it["id"]] = lg[0]
    lab.h_keep = None
    log("donor cache")
    fams = sorted({i["family"] for i in allitems})
    chosen = []
    for f in fams:
        fi = [i for i in allitems if i["family"] == f]
        ok = [i for i in fi if int(native[i["id"]].argmax()) == i["answer"]]
        chosen += (ok + [i for i in fi if i not in ok])[:per_family]
        log(f"family {f}: {len(ok)}/{len(fi)} correct")
    rng = random.Random(SEED)
    records, items_meta, cross = [], [], []
    for it in chosen:
        ids, last = it["ids"], len(it["ids"]) - 1
        a, b = it["answer"], it["donor"]["answer"]
        same = [i for i in allitems if i["family"] == it["family"] and i["id"] != it["id"]]
        canon = it["donor"]["id"] if it["donor"]["id"] in byid else None
        order = ([byid[canon]] if canon else []) + [i for i in same if i["id"] != canon]
        other = [i for i in allitems if i["family"] != it["family"]]
        donors = order[:N_DONORS - 1] + [rng.choice(other)]
        # base: stock vs re-implemented eager, and the base pattern for freezing
        stock = lab.forward([ids], mine=False)
        lab.record_w, lab.w = True, []
        mine = lab.forward([ids])
        lab.record_w = False
        lab.base_w = lab.w
        bitwise = bool(torch.equal(stock, mine))
        lanes = []                                 # (layer, row, donor index, arm)
        for l in LAYERS:
            for r in it["span"]:
                for k, d in enumerate(donors):
                    if r < len(d["ids"]):
                        for arm in ("total", "frozen"):
                            lanes.append((l, r, k, arm))
        res = {}
        base_M = []
        for s in range(0, len(lanes), BATCH - 1):
            chunk = lanes[s:s + BATCH - 1]
            B = len(chunk) + 1
            lab.patch = {}
            for l in LAYERS:
                sel = [(j + 1, r, k) for j, (ll, r, k, _) in enumerate(chunk) if ll == l]
                if sel:
                    lab.patch[l] = (torch.tensor([x[0] for x in sel], device=DEV),
                                    torch.tensor([x[1] for x in sel], device=DEV),
                                    torch.stack([H[donors[x[2]]["id"]][l][x[1]] for x in sel]))
            ff = [lab.L + 1] + [l + 1 if arm == "frozen" else lab.L + 1 for (l, _, _, arm) in chunk]
            lab.freeze_from = torch.tensor(ff, device=DEV)
            lab.am = torch.empty(B, lab.L, model.config.num_attention_heads, len(ids), dtype=torch.int16,
                                 device=DEV)
            lg = lab.forward([ids] * B)
            lab.patch, lab.freeze_from = {}, None
            M = (lg[:, a] - lg[:, b]).double().cpu()
            base_M.append(float(M[0]))
            am = lab.am
            lab.am = None
            base_am = am[0]
            for j, (l, r, k, arm) in enumerate(chunk):
                e = {"dM": float(M[j + 1] - M[0])}
                if arm == "total":
                    d = am[j + 1, l + 1:, :, r:]
                    bb = base_am[l + 1:, :, r:]
                    tie = (d < 0) | (bb < 0)
                    mv = (d != bb) & ~tie
                    e.update(moved=int(mv.sum()), ties=int(tie.sum()),
                             moved_last=int(mv[:, :, -1].sum()),
                             heads=int(mv.any(-1).sum()))
                res[(l, r, k, arm)] = e
        spread = max(base_M) - min(base_M)
        nat = mine[0]
        floor = 2 * spacing(max(abs(float(nat[a])), abs(float(nat[b])))) + spread
        items_meta.append({"id": it["id"], "family": it["family"], "span": it["span"], "len": len(ids),
                           "correct": int(nat.argmax()) == a, "M_base": float(nat[a] - nat[b]),
                           "floor": floor, "base_lane_spread": spread, "eager_bitwise_stock": bitwise,
                           "donors": [d["id"] for d in donors], "canonical": canon,
                           "batches": len(base_M)})
        for l in LAYERS:
            for r in it["span"]:
                for k, d in enumerate(donors):
                    if (l, r, k, "total") not in res:
                        continue
                    t, f = res[(l, r, k, "total")], res[(l, r, k, "frozen")]
                    records.append({
                        "item": it["id"], "family": it["family"], "layer": l, "row": r, "donor": k,
                        "donor_id": d["id"], "canonical": d["id"] == canon,
                        "same_family": d["family"] == it["family"],
                        "informative": informative(H[it["id"]][l][r], H[d["id"]][l][r]),
                        "dM_total": t["dM"], "dM_frozen": f["dM"], "floor": floor,
                        "effect": abs(t["dM"]) > floor, "frozen_effect": abs(f["dM"]) > floor,
                        "moved": t["moved"], "moved_last": t["moved_last"], "ties": t["ties"],
                        "heads": t["heads"]})
        if len(cross) < 4 and it is chosen[0]:
            cross.append((it, donors))
        log(f"{it['id']}: {len(lanes)} lanes, {len(base_M)} batches, bitwise {bitwise}, floor {floor:.4g}")
    (out / "records.json").write_text(json.dumps({"items": items_meta, "records": records}))
    summary = summarize(records, items_meta)
    summary["stepped_crosscheck"] = crosscheck(model, lab, cross, records)
    summary["seconds"] = round(time.monotonic() - T0, 1)
    summary["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


def summarize(records, items_meta):
    cells = {}
    for x in records:
        cells.setdefault((x["item"], x["layer"], x["row"]), []).append(x)
    verdict = lambda x: EFFECT if x["effect"] else NO_EFFECT
    canon_null = canon_flip = canon_null_uninf = 0
    any_null = any_flip = 0
    for rows in cells.values():
        pool = pooled([{"verdict": verdict(x), "informative": x["informative"]} for x in rows])
        c = [x for x in rows if x["canonical"]]
        if c and not c[0]["effect"]:
            canon_null += 1
            canon_flip += pool["verdict"] == EFFECT
            canon_null_uninf += c[0]["informative"] is False
        for x in rows:
            if not x["effect"]:
                any_null += 1
                any_flip += pool["verdict"] == EFFECT
    inf = [x for x in records if x["informative"]]
    div = [x for x in inf if x["moved"] > 0]
    dive = [x for x in div if x["effect"]]
    ratios = sorted(abs(x["dM_frozen"]) / abs(x["dM_total"]) for x in dive)
    med = ratios[len(ratios) // 2] if ratios else None
    by_layer = {}
    for l in LAYERS:
        il = [x for x in inf if x["layer"] == l]
        dl = [x for x in il if x["moved"] > 0 and x["effect"]]
        rl = sorted(abs(x["dM_frozen"]) / abs(x["dM_total"]) for x in dl)
        by_layer[l] = {"informative": len(il), "path_diverged": sum(x["moved"] > 0 for x in il),
                       "path_diverged_last_row": sum(x["moved_last"] > 0 for x in il),
                       "diverged_effect": len(dl), "median_frozen_over_total": rl[len(rl) // 2] if rl else None}
    frac = lambda n, d: (n / d) if d else None
    return {
        "items": len(items_meta), "correct": sum(i["correct"] for i in items_meta),
        "eager_bitwise_all": all(i["eager_bitwise_stock"] for i in items_meta),
        "cells": len(cells), "patches": len(records), "informative": len(inf),
        "a": {"canonical_single_nulls": canon_null, "flip_to_effect_16": canon_flip,
              "G1_fraction": frac(canon_flip, canon_null),
              "G2_canonical_null_uninformative": frac(canon_null_uninf, canon_null),
              "any_single_nulls": any_null, "any_flip": any_flip, "any_fraction": frac(any_flip, any_null),
              "patches_effect": sum(x["effect"] for x in records),
              "patches_uninformative": sum(x["informative"] is False for x in records)},
        "b": {"G3_path_diverged_all_heads": frac(len(div), len(inf)),
              "G4_path_diverged_last_row": frac(sum(x["moved_last"] > 0 for x in inf), len(inf)),
              "diverged_effect": len(dive), "G5_median_frozen_over_total": med,
              "frac_ratio_below_half": frac(sum(r < 0.5 for r in ratios), len(ratios)),
              "q25_q75": [ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4]] if ratios else None,
              "median_moved_selections": sorted(x["moved"] for x in div)[len(div) // 2] if div else None,
              "uninformative_diverged": sum(x["moved"] > 0 for x in records if x["informative"] is False),
              "by_layer": by_layer}}


def crosscheck(model, lab, cross, records):
    """A few patches re-measured through mdb paths.paired (stepped, the authority)."""
    out = []
    try:
        from mdb.analysis.forecast import Margin
        from mdb.analysis.paths import paired
        from mdb.ops.debugger import attach

        dbg = attach(model, None)
        for it, donors in cross:
            ids, last = it["ids"], len(it["ids"]) - 1
            M = Margin(last, it["answer"], it["donor"]["answer"])
            for l, k in ((7, 0), (19, 1), (0, 2)):
                r = it["span"][-1]
                d = donors[k]
                drun = dbg.run(d["ids"], name=f"d{k}", prefill=0)
                drun.until(f"t{len(d['ids']) - 1}.logits")
                val = drun.read(f"t{r}.L{l}.h_out").clone()
                run = dbg.run(ids, name="x", prefill=it["span"][0])
                run.until(f"t{last}.logits")
                p = paired(run, [(f"t{r}.L{l}.h_out", val, "replace", None)], M)
                rec = next(x for x in records if x["item"] == it["id"] and x["layer"] == l
                           and x["row"] == r and x["donor"] == k)
                out.append({"item": it["id"], "layer": l, "row": r, "donor": d["id"],
                            "stepped": {"total": p["total"]["dM"]["value"], "frozen": p["frozen"]["dM"]["value"],
                                        "verdict": p["verdict"], "moved": p["path"]["moved"]},
                            "batched": {"total": rec["dM_total"], "frozen": rec["dM_frozen"],
                                        "moved": rec["moved"]}})
                log(f"crosscheck {out[-1]}")
    except Exception as e:  # noqa: BLE001
        out.append({"error": f"{type(e).__name__}: {str(e)[:400]}"})
        log(f"crosscheck failed: {out[-1]}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
