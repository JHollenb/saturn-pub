"""Job B (exp1/prereg.json): watch on the carrier cell's SAE country features, repair by one
named component, continue.  python exp1_depth.py OUTDIR [PICK]

Per d3/d4 item (110): a native run to the logits gives the carrier row (frozen R_argmax rule)
and the native answer; the run is restored to t<last>.start and continued by
breakpoints.run_forward with an mdb watch whose predicate is a FeatureCondition over the
Qwen-Scope L19 country features (census.json, frozen) plus an unconditional break at the same
stop (t<last>.L20.h); at the stop a checkpoint, then arms R1 (named component swap), R2 (whole
cell donor), R3 (error component swap), RND (matched-norm random feature swap), each:
set_cell, continue to the logits, read, restore. R1 on wrong items is sealed with
forecast.seal after the edit commits and verified at the logits.
The log prints item ids and seconds only.
"""
import json
import random
import sys
import time
import types

import torch

sys.path.insert(0, ".")
from common import (CG, CL, MODEL, SAE_DIR, SCORE, SETS, carrier_row, floor_of, items, kept,  # noqa: E402
                    load, log)
from mdb.analysis import forecast, sae  # noqa: E402
from mdb.analysis.components import component_binding  # noqa: E402
from mdb.analysis.features import FeatureCondition, FeatureContext  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.engine import breakpoints as bp  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
CEN = json.load(open("census.json"))
LABEL = {int(f): int(c) for f, c in CEN["features"].items()}
TRUTHF = {int(c): (int(v["feature"]), float(v["a_median"])) for c, v in CEN["a_truth"].items()}
COUNTRY_FEATS = sorted(LABEL)
dbg = load()
ad = dbg.adapter
B19 = sae.open_set("qwen-scope", SAE_DIR, layers=[19], cache=2).get(19)
try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(MODEL)
except Exception:  # noqa: BLE001
    TOK = None
T = SETS["truths"]
log("load")
WATCH = "best_feature(COUNTRIES) >= 0 and best_feature(COUNTRIES) != truth"
FAMS = ("d3-lm", "d3-per", "d4-work")
depth = sorted([i for i in items("depth") if i["family"] in FAMS], key=lambda i: i["id"])
by_id = {i["id"]: i for i in items("depth")}
rng = random.Random(20261114)
if PICK:
    depth = [depth[int(k)] for k in PICK.split(",")]
DONORS = {}


def text(tok):
    return TOK.decode([tok]).strip().lower() if TOK is not None else str(tok)


def entity(iid):
    return iid.split(":", 1)[1].lower()


def pick_donor(iid):
    me = T[iid]
    pool = [k for k, v in SCORE.items() if v["correct"] and k != iid and T[k]["country"] == me["country"]]
    order = {me["family"]: 0, "d3-lm": 1, "d3-per": 1, "d4-work": 1, "d2": 2, "d1": 3}
    pool.sort(key=lambda k: (order.get(T[k]["family"], 4), k))
    return pool[0] if pool else None


def donor_state(did):
    if did not in DONORS:
        d = by_id[did]
        last = len(d["ids"]) - 1
        r = dbg.run(d["ids"], name="donor", prefill=d["span"][0])
        r.until(f"t{last}.logits")
        crow = carrier_row(r, d)
        cb = component_binding(r, "donor", f"cache.L{CL}.v[kv{CG},row={crow}]", B19, features=[])
        DONORS[did] = {"row": crow, "cell": r.read_cache(CL, "v", staged=False)[0, CG, crow].clone(),
                       "error": cb.get("error").value.clone(), "family": T[did]["family"]}
        del r
    return DONORS[did]


def arm(run, ckw, crow, last, value, toks, seal=False):
    dt = run.read_cache(CL, "v", staged=False).dtype
    run.set_cell(CL, "v", value.to(dt).reshape(1, 1, 1, -1), rows=[crow], kv=[CG])
    out = {}
    if seal:
        try:
            run.until(f"t{last}.L{CL + 1}.h")
            sealed = forecast.seal(run, Margin(last, toks[0], toks[1]))
            out["sealed"] = {"id": sealed.get("id"), "predicted": sealed.get("predicted")}
        except Exception as e:  # noqa: BLE001
            out["sealed"] = {"refused": f"{type(e).__name__}: {str(e)[:200]}"}
    run.until(f"t{last}.logits")
    lg = run.read(f"t{last}.logits")
    out.update(kept(lg, toks))
    out["text"] = text(out["argmax"])
    if seal and "id" in out.get("sealed", {}):
        try:
            v = forecast.verify(run, [out["sealed"]["id"]])[0]
            out["verify"] = {"verdict": v.get("verdict"), "native": v.get("native"),
                             "predicted": v.get("predicted"), "floor": v.get("floor")}
        except Exception as e:  # noqa: BLE001
            out["verify"] = {"refused": f"{type(e).__name__}: {str(e)[:200]}"}
    run.restore(ckw)
    return out


recs = []
fh = open(f"{OUT}/exp1-items{'-smoke' if PICK else ''}.jsonl", "w")
for item in depth:
    t0 = time.monotonic()
    iid, ids, span = item["id"], item["ids"], item["span"]
    last = len(ids) - 1
    tr = T[iid]
    lang, country = tr["language"], tr["country"]
    correct = SCORE[iid]["correct"]
    run = dbg.run(ids, name="nat", prefill=span[0])
    run.until(f"t{last}.start")
    ck0 = run.checkpoint("pre")
    run.until(f"t{last}.logits")
    crow = carrier_row(run, item)
    nat = run.read(f"t{last}.logits").reshape(-1).float()
    top2 = nat.topk(2).indices.tolist()
    foil = top2[0] if top2[0] != lang else top2[1]
    toks = [lang, foil]
    rec = {"id": iid, "family": tr["family"], "correct": correct, "carrier_row": crow,
           "last_span": span[-1], "country": country, "language": lang, "foil": foil,
           "native": kept(nat, toks), "native_text": text(int(nat.argmax()))}
    run.restore(ck0)
    ctx = FeatureContext(B19, crow, CL, CG, sets={"COUNTRIES": COUNTRY_FEATS}, labels=LABEL,
                         candidates={"C": SETS["sets"]["countries"]}, constants={"truth": country})
    table = bp.Table()
    w = table.add_watch(f"t{last}.L{CL}.h", run.program, mode="level",
                        compiled=FeatureCondition(WATCH, ctx))
    table.add_break(f"t{last}.L{CL}.h", run.program)
    out = bp.run_forward(run, table, budget=4)
    rec["stop"] = {"reason": out["reason"], "where": out["where"],
                   "watch_fired": any(h["id"] == w.id for h in out["hits"])}
    env = types.SimpleNamespace(run=run)
    rec["best_feature_label"] = ctx.best_feature(run, "COUNTRIES")
    rec["content_C"] = ctx.content(run, "C")
    if span[-1] != crow:
        ctx_l = FeatureContext(B19, span[-1], CL, CG, sets={"COUNTRIES": COUNTRY_FEATS}, labels=LABEL,
                               constants={"truth": country})
        rec["lastrow_watch"] = FeatureCondition(WATCH, ctx_l).evaluate(env)
    else:
        rec["lastrow_watch"] = rec["stop"]["watch_fired"]
    ckw = run.checkpoint("wrong")
    cb = component_binding(run, "carrier", f"cache.L{CL}.v[kv{CG},row={crow}]", B19)
    acts = cb.acts
    active_c = [(acts[f], f) for f in COUNTRY_FEATS if acts.get(f, 0) > 0]
    wrong = [(a, f) for a, f in active_c if LABEL[f] != country]
    fw = max(wrong)[1] if wrong else None
    tf = TRUTHF.get(country)
    rec["active_country_features"] = [[f, LABEL[f], a] for a, f in sorted(active_c, reverse=True)]
    rec["truth_feature"] = list(tf) if tf else None
    delta = torch.zeros_like(cb.native)
    if tf is not None:
        delta = delta + cb.feature_value(tf[0], tf[1], B19)
    if fw is not None:
        delta = delta - cb.feature_value(fw, acts[fw], B19)
    rec["r1"] = {"wrong_feature": fw, "has_truth_feature": tf is not None,
                 "delta_norm": float(delta.norm()), "cell_norm": float(cb.native.norm())}
    arms = {}
    if float(delta.norm()) > 0:
        arms["R1"] = arm(run, ckw, crow, last, cb.native + delta, toks, seal=not correct)
    did = pick_donor(iid)
    if did is not None:
        ds = donor_state(did)
        rec["donor"] = {"id": did, "family": ds["family"]}
        arms["R2"] = arm(run, ckw, crow, last, ds["cell"].double().cpu(), toks)
        arms["R2"]["copy"] = arms["R2"]["text"] in entity(did).replace("-", " ").split()
        arms["R3"] = arm(run, ckw, crow, last, cb.native - cb.get("error").value + ds["error"], toks)
    if "R1" in arms:
        arms["R1"]["copy"] = arms["R1"]["text"] in entity(iid).replace("-", " ").split()
        nonc = [f for f in cb.active if f not in LABEL]
        if nonc:
            fr = rng.choice(nonc)
            while True:
                fa = rng.randrange(B19.d_sae)
                if fa not in LABEL and fa not in acts:
                    break
            dn = -cb.feature_value(fr, acts[fr], B19) + cb.feature_value(fa, acts[fr], B19)
            dn = dn * (delta.norm() / (dn.norm() + 1e-30))
            arms["RND"] = arm(run, ckw, crow, last, cb.native + dn, toks)
            arms["RND"]["features"] = [fr, fa]
    rec["arms"] = arms
    rec["floor"] = floor_of(nat, lang, foil)
    rec["seconds"] = round(time.monotonic() - t0, 2)
    fh.write(json.dumps(rec, default=float) + "\n")
    fh.flush()
    log(f"B {iid} {rec['seconds']}s")
    del run
log("TOTAL")
