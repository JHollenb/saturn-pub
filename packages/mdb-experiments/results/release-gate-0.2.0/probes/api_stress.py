"""Release-gate 0.2.0, part 2: Python debugger objects on a small real model (CPU, laptop).

Qwen2.5-0.5B (base, bf16, eager, CPU). Each section runs under try/except and prints
API <section> ok|FAIL seconds detail; a final API_JSON line summarises. Sections: selectors and
origins (Run.set + ValueRef), holds (+ forecast/whatif refusals), Branches (stepped, batched
port + cell arms, band recheck, nulls, probes, seal into a sink), Pool, core files (dump /
run_from), record sink payloads, multi-site feature watch (random bases), symbol table with
subspaces.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mdb
from mdb.analysis import features, forecast, propagate, sae, subspace, symbols
from mdb.engine import holds
from mdb.analysis.branches import edit
from mdb.analysis.forecast import Margin
from mdb.engine import breakpoints as bp

MODEL = os.environ.get("MDB_MODEL")
OUT = Path(os.environ.get("MDB_OUT", "out-api"))
OUT.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(0)
RES = {}


def section(name):
    def deco(fn):
        t0 = time.monotonic()
        try:
            detail = fn()
            RES[name] = {"ok": True, "s": round(time.monotonic() - t0, 2), "detail": detail}
            print(f"API {name} ok {time.monotonic() - t0:.2f}s {json.dumps(detail, default=str)[:900]}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            RES[name] = {"ok": False, "s": round(time.monotonic() - t0, 2),
                         "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}
            print(f"API {name} FAIL {time.monotonic() - t0:.2f}s {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc()[-2500:], flush=True)
        return fn
    return deco


t0 = time.monotonic()
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager",
                                             local_files_only=True).eval()
dbg = mdb.attach(model, tok)
print(f"loaded+attached {time.monotonic() - t0:.1f}s  layers {dbg.adapter.layers}", flush=True)
L = dbg.adapter.layers
MID = L // 2
P = tok("The Eiffel Tower is located in the city of")["input_ids"]
Q = tok("The Colosseum is located in the city of")["input_ids"]
assert len(P) == len(Q), (len(P), len(Q))
last = len(P) - 1
A, B = tok(" Paris")["input_ids"][0], tok(" Rome")["input_ids"][0]
M = Margin(last, A, B)
sink = dbg.records_to(str(OUT) + "/")
custody = dbg.custody(script=__file__, kernel_path="mdb-stepped")


def margin_of(run):
    run.until(f"t{last}.logits")
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    return float(lg[A] - lg[B])


base = dbg.run(P, name="base")
donor = dbg.run(Q, name="donor")
M_base, M_donor = margin_of(base), margin_of(donor)
print(f"base margin {M_base}  donor margin {M_donor}", flush=True)


@section("selectors_origins")
def _():
    ref = donor.ref(f"t{last}.L{MID}.v[kv1]")
    r = dbg.run(P, name="sel")
    r.until(f"t{last}.L{MID}.h")          # Run.set stages at the pending layer only
    r.set(f"t{last}.L{MID}.v[kv1]", ref)
    r.set(f"t@last.L{MID}.p[h3,src=0:3]", 0.0)
    m = margin_of(r)
    cell = dbg.run(P, name="cell")
    cell.set(f"cache.L{MID}.v[kv1,row=5]", donor.ref(f"cache.L{MID}.v[kv1,row=5]"))
    mc = margin_of(cell)
    return {"pointer_keys": sorted(ref.pointer), "margin_after_v_and_p_edit": m, "margin_cell_edit": mc,
            "edits": len(r.committed_edits())}


@section("holds_and_refusals")
def _():
    r = dbg.run(P, name="held")
    h1 = r.hold(f"t@last.L>{MID}.p", source=base)
    h2 = r.hold(f"t@last.L>{MID}.norm_scale", source=base)
    out = {"holds": [str(h) for h in r.hold_objects()] if hasattr(r, "hold_objects") else None}
    refusals = {}
    try:
        forecast.seal(r, M)
        refusals["forecast.seal"] = "NOT refused"
    except Exception as e:  # noqa: BLE001
        refusals["forecast.seal"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        propagate.whatif(r, propagate.Edit(f"t{last}.L{MID + 1}.m", 0.0, "scale"), M)
        refusals["whatif"] = "NOT refused"
    except Exception as e:  # noqa: BLE001
        refusals["whatif"] = f"{type(e).__name__}: {str(e)[:120]}"
    out["refusals"] = refusals
    r.set(f"t{last}.L{MID}.m", 0.0, "scale") if False else None
    m_self = margin_of(r)               # held against base itself: must equal base
    out["held_self_margin"] = m_self
    out["held_self_equals_base"] = m_self == M_base
    fl = holds.final_held_logits(r, last, base)
    out["final_held_logits_margin"] = float(fl.reshape(-1).float()[A] - fl.reshape(-1).float()[B])
    for h in list(r.holds):
        r.release(h["name"] if isinstance(h, dict) else h)
    try:
        forecast.seal(r, M)
        out["seal_after_release"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["seal_after_release"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


@section("branches")
def _():
    r = dbg.run(P, name="br")
    probes = {"h_out_norm": lambda run: float(run.read(f"t{last}.L{L - 1}.h_out").float().norm())}
    br = dbg.branches(r, M, at=f"t{last}.L{MID}.h", name="rg-branches", probes=probes)
    br.add("m_scale0", [edit(f"t{last}.L{MID + 2}.m", 0.0, op="scale")])
    br.add("f_sel", [edit(f"t{last}.L{MID + 1}.f[i=1|2|3|4|5]", 0.0)])
    br.add("A_head", [edit(f"t{last}.L{MID}.A[h3]", 0.0)])
    br.add("v_donor", [edit(f"t{last}.L{MID}.v[kv1]", donor.ref(f"t{last}.L{MID}.v[kv1]"))])
    br.add("o_donor", [edit(f"t{last}.L{MID + 3}.o", donor.ref(f"t{last}.L{MID + 3}.o"))])
    br.add("cell_v", [edit(f"cache.L{MID + 1}.v[kv0,row=5]", donor.ref(f"cache.L{MID + 1}.v[kv0,row=5]"))])
    br.add_null("rnd", "random", like="o_donor", seed=0)
    br.add_null("self", "self", like="o_donor")
    br.add_null("ans", "answer", like="o_donor")
    t = time.monotonic()
    stepped = {x["name"]: x for x in br.run()}
    ts = time.monotonic() - t
    t = time.monotonic()
    batched = {x["name"]: x for x in br.run(mode="batched", again=True)}
    tb = time.monotonic() - t
    t = time.monotonic()
    ball = {x["name"]: x for x in br.batched(recheck="all")}
    ta = time.monotonic() - t
    rows = {}
    for n in stepped:
        rows[n] = {"stepped": stepped[n]["dM"]["value"], "batched": batched[n]["dM"]["value"],
                   "floor": stepped[n].get("floor", {}).get("value"),
                   "band_recheck": batched[n].get("recheck"),
                   "all_kernel_delta": (ball[n].get("kernel_delta") or {}).get("value"),
                   "probe": (stepped[n].get("probes") or {})}
    # re-run stepped so the seal holds stepped rows
    br.run(again=True)
    rec = br.seal()
    cmp = br.compare("o_donor", "v_donor")
    return {"M0": br.M0, "floor": br.floor, "rows": rows, "s_stepped": ts, "s_batched_band": tb,
            "s_batched_all": ta, "seal_id": rec.get("id", "")[:16] if isinstance(rec, dict) else str(rec)[:60],
            "compare_keys": sorted(cmp)[:8], "self_null_zero": stepped["self"]["dM"]["value"] == 0.0}


@section("pool")
def _():
    cities = ["The Eiffel Tower is located in the city of", "The Colosseum is located in the city of",
              "The Brandenburg Gate is located in the city of", "The Kremlin is located in the city of",
              "The Acropolis is located in the city of", "Big Ben is located in the city of"]
    items = [{"id": f"c{i}", "ids": tok(c)["input_ids"], "country": ["fr", "it", "de", "ru", "gr", "uk"][i]}
             for i, c in enumerate(cities)]
    pool = dbg.pool(items, f"t@last.L{MID}.f", labels=("country",), name="rg-pool")
    m = pool.mean_for(items[0])
    d = pool.donor(items[0], rule="other:country")
    rnd = pool.random(items[0], seed=0)
    rec = pool.seal()
    path = OUT / "pool.pt"
    pool.save(path)
    from mdb.analysis.pools import Pool
    back = Pool.load(path)
    return {"mean_shape": list(m.value.shape), "mean_excluded": m.pointer.get("excluded"),
            "donor": d.pointer.get("rule"), "random_norm": float(rnd.value.float().norm()),
            "loaded_n": len(getattr(back, "items", []) or []), "path_bytes": path.stat().st_size}


@section("core_file")
def _():
    path = OUT / "core.pt"
    if path.exists():
        path.unlink()
    src = dbg.run(P, name="coresrc")
    src.until(f"t{last}.logits")
    K = last
    src.dump(str(path), row=K, capture=(f"t{last}.logits",))
    r = dbg.run_from(str(path))
    m = margin_of(r)
    return {"bytes": path.stat().st_size, "core_check": r.core_check, "margin_reloaded": m,
            "equals_base": m == M_base, "replica": (r.core or {}).get("replica")}


@section("sink_payloads")
def _():
    recs = sink.records()
    pays = sink.payloads()
    kinds = {}
    for x in recs:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    return {"path": str(sink.path), "records": len(recs), "payload_lines": len(pays), "kinds": kinds,
            "custody_on_records": sum("custody" in x["body"] for x in recs)}


@section("multi_site_watch")
def _():
    width, g, F = int(model.config.hidden_size), torch.Generator().manual_seed(21), 64
    layers = [0, 1, 2, 3]
    bases = {l: sae.FeatureBasis(f"r{l}", l, "h", F // 2, torch.randn(F, width, generator=g),
                                 torch.randn(F, generator=g) * 0.1, torch.randn(width, F, generator=g) / 8,
                                 torch.randn(width, generator=g) * 0.1) for l in layers}
    probe_runs = {}
    for l in layers:
        pr = dbg.run(P)
        pr.until(f"t{last}.L{l}.h")
        probe_runs[l] = set(features.activations(pr, bases[l], last, l))
    feat = next(f for f in range(F) if sum(f in probe_runs[l] for l in layers) >= 2)
    cond = features.FeatureCondition(f"feature[{feat}] > 0", features.feature_sites(bases, row=last))
    fresh = dbg.run(P, prefill=last - 1)       # start one row early so t<last>.L0.h is reached
    table = bp.Table()
    table.add_watch(f"t{last}.L*.h", fresh.program, mode="level", compiled=cond)
    table.add_break(f"t{last}.L2.h", fresh.program)
    hits = []
    t = time.monotonic()
    for _ in range(50):
        res = bp.run_forward(fresh, table, budget=4)
        if res["reason"] not in ("watch", "watch+breakpoint", "breakpoint"):
            break
        hits.append((res["reason"], res["where"], [h.get("site") for h in res["hits"]]))
    back = []
    for _ in range(50):
        res = bp.run_reverse(fresh, table)
        if res["reason"] not in ("watch", "watch+breakpoint", "breakpoint"):
            break
        back.append(res["where"])
    return {"feature": feat, "live_layers": [l for l in layers if feat in probe_runs[l]], "hits": hits,
            "reverse": back, "reverse_matches": back == [h[1] for h in reversed(hits)],
            "s_walk": time.monotonic() - t}


@section("symbols_subspaces")
def _():
    r = dbg.run(P, name="sym")
    r.until(f"t{last}.logits")
    hd = dbg.adapter.head_dim
    g = torch.Generator().manual_seed(2)
    q = subspace.orthonormal(torch.randn(hd, 4, generator=g, dtype=torch.float64))
    part = subspace.subspace_binding("part", MID, 1, q, authority="supplied")
    rest = subspace.subspace_binding("rest", MID, 1, torch.eye(hd, dtype=torch.float64) - q @ q.T)
    t = time.monotonic()
    tab = symbols.table(r, consumer=M, subspaces=[part.bound(5), rest.bound(5)])
    rep = tab.report
    rows = rep.get("subspaces", [])
    tot = sum(x["subspace_to_consumer"]["value"] for x in rows)
    fr = tab.to_frame() if hasattr(tab, "to_frame") else None
    return {"s": time.monotonic() - t, "n_bindings": len(rep.get("bindings", rep.get("rows", []))),
            "subspace_rows": [x["name"] for x in rows], "sum_vs_cell": [tot, rows[0]["cell_to_consumer"]["value"]],
            "frame_shape": getattr(fr, "shape", None), "report_bytes": len(json.dumps(rep, default=str))}


print("API_JSON " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tb"} for k, v in RES.items()},
                               default=str))
print(f"API passed {sum(v['ok'] for v in RES.values())}/{len(RES)}  total {time.monotonic() - t0:.1f}s")
