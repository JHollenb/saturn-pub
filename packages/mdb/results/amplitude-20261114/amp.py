"""Amplitude law of latent hops (prereg.json; advisor memo 2026-11-14 open-problems §2.1 + §2.3).

python amp.py native  MODEL_DIR OUTDIR PHASE [SHARD]   bf16 GPU, mdb stepped/batched native arms
    PHASE pools : Pools at the carrier cell (correct items, band layers, wrong items), sealed
          A     : arm A dose ladders (P_true per item, class-mean span k=1,4,16) + nulls; E1 dose-down
                  on correct items; core files of every item the replica phase uses
          B     : arm B true back-patch at the carrier row, 16 (s, t) pairs + nulls, probes = carrier
                  amplitude over the band layers
python amp.py replica MODEL_DIR OUTDIR DEVICE [SHARD]  fp32 replica from the core files (bf16 prefix
          cache, fp32 answer row): arm C (dose ladder, alpha*, hop_edge, gate switches vs the exp5
          composer set) and §2.3 (donor-swap F1/F2/F3, dose ladder log-slopes)

SHARD = "i/n" (items i, i+n, ...). Pools are read from OUTDIR (written by PHASE pools). Every record
is sealed with custody (MDB_SCRIPT / MDB_PREREG or dbg.custody) into records-*.jsonl in the cwd.
The log prints item ids, counts and seconds only (no outcomes).
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

from mdb.analysis import symbols
from mdb.analysis.branches import edit
from mdb.analysis.finite import f64
from mdb.analysis.forecast import Margin
from mdb.analysis.pools import Pool
from mdb.analysis.reads import readout
from mdb.analysis.subspace import class_mean_subspace, subspace_binding
from mdb.engine import core_file
from mdb.repl.__main__ import _load

T0 = time.monotonic()
MODE, MODEL, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
NAME = os.environ.get("MODEL_NAME", MODEL.name)
HERE = Path(os.environ.get("DATA", "."))
torch.set_grad_enabled(False)
DOSES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
DOWN = [0.0, 0.25, 0.5, 0.75]
SPAN_K = [1, 4, 16]
SEED = 20261114


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


# ---- panel, scores, strata (scale study inputs; hopmath.py §3 definitions) -----------------------
P = json.load(open(HERE / "panel.json"))
PS = json.load(open(HERE / "panel_supp.json"))
S = json.load(open(HERE / f"score-{NAME}.json"))
SS = json.load(open(HERE / f"supscore-{NAME}.json"))
LC, KC, L = S["model"]["L_c"], S["model"]["kv_c"], S["model"]["layers"]
BAND = [x for x in range(LC - 3, LC + 2)]
PER, SPER = S["per_item"], SS["per_item"]


def prep(it, supp):
    pr = (SPER if supp else PER)[it["id"]]
    return dict(it, supp=supp, country=int(it["country_token"]), roles={"carrier": pr["row_at_item"]},
                carrier=pr["row_at_item"], correct=bool(pr["correct"]), rank=pr["country_rank"],
                last=len(it["ids"]) - 1)


MAIN = [prep(i, False) for i in P["items"]]
BY_ID_ALL = {i["id"]: i for i in MAIN}
CORRECT = [i for i in MAIN if i["correct"]]
WRONG = [prep(i, False) for i in P["items"] if i["family"] in ("d3-lm", "d3-per", "d4-work")
         and not PER[i["id"]]["correct"]]
WRONG += [prep(i, True) for i in PS["items"] if i["family"] == "d5" and not SPER[i["id"]]["correct"]]
for w in WRONG:
    w["stratum"] = "readable" if (w["rank"] is None or w["rank"] > 3) else "right_operand"
if os.environ.get("SMOKE"):                       # smoke: a few items, same code path
    n = int(os.environ["SMOKE"])
    keep = {w["country"] for w in WRONG[:n]}
    based = [i for i in CORRECT if i["block"] == "base" and i["donor"] in BY_ID_ALL
             and BY_ID_ALL[i["donor"]]["correct"]][:n]
    keep |= {BY_ID_ALL[i["donor"]]["country"] for i in based}
    CORRECT = [i for i in CORRECT if i["country"] in keep][:6 * n] + based
    CORRECT = list({i["id"]: i for i in CORRECT}.values())
    WRONG = WRONG[:n]
CORRECT_BASE = [i for i in CORRECT if i["block"] == "base"]
BY_ID = {i["id"]: i for i in MAIN}


def shard(xs):
    sh = sys.argv[5] if len(sys.argv) > 5 else "0/1"
    i, n = (int(v) for v in sh.split("/"))
    return [x for k, x in enumerate(xs) if k % n == i], sh.replace("/", "of")


def cell_addr(layer, row):
    return f"cache.L{layer}.v[kv{KC},row={row}]"


def pool_path(layer, kind):
    return OUT / f"pool-{kind}-L{layer}.json"


def lab(x, authority="measured"):
    return {"value": float(x), "authority": authority}


def val(x):
    return x["value"] if isinstance(x, dict) else x


# ---- directions from a correct-items pool (Pool LOO means; the item is never in its own mean) -----
class Code:
    """The item's true-country code at one layer: m_all = leave-one-out mean over correct items,
    m_c = leave-one-out same-country mean, d = (m_c - m_all)/|m_c - m_all|; amplitude
    a(v) = (v - m_all) . d / |m_c - m_all| (1.0 = the same-country correct items' mean code)."""

    def __init__(self, pool, item):
        self.m_all = f64(pool.mean_for(item).value).reshape(-1)
        cm = pool.class_mean_for(item, by="country")
        self.n_class = cm.pointer["n"]
        code = f64(cm.value).reshape(-1) - self.m_all
        self.N = float(code.norm())
        self.d = code / self.N
        self.pointers = {"m_all": _ptr(pool.mean_for(item)), "m_c": _ptr(cm)}

    def amp(self, v):
        return float((f64(v).reshape(-1).cpu() - self.m_all) @ self.d) / self.N

    def dose(self, v, alpha):
        v = f64(v).reshape(-1).cpu()
        return v + (alpha - 1.0) * float((v - self.m_all) @ self.d) * self.d


def _ptr(ref):
    p = dict(ref.pointer)
    p.pop("items", None)
    return p


def code_or_none(pool, item):
    """``Code`` or None when the pool refuses the class mean (no same-country correct item)."""
    try:
        return Code(pool, item)
    except Exception as e:
        if "no item left" in str(e):
            return None
        raise


def span_bindings(pool):
    """Rank-k PCA of the centred class means over every country of the correct pool (fitted;
    test-informed at the class level: the wrong item's own country is a class, the item is not
    in any mean)."""
    vals = {}
    for it, v in zip(pool.items, pool.values):
        vals.setdefault(str(it["labels"]["country"]), []).append(v)
    return {k: class_mean_subspace(f"span{k}", LC, KC, vals, k, provenance={
        "data": f"pool {pool.name} ({len(pool)} correct main-panel items, {len(vals)} countries)",
        "training": "none (class means)", "test_informed": "class level"}) for k in SPAN_K
        if k < len(vals)}


def other_top(logits, a):
    lg = logits.reshape(-1).float().clone()
    lg[a] = -float("inf")
    return int(lg.argmax())


def custody(dbg, tag):
    prereg = Path(__file__).resolve().parent / "prereg.json"
    dbg.custody(script=__file__, prereg=prereg if prereg.is_file() else None,
                kernel_path="mdb-stepped", note=f"amplitude {tag} {NAME}")
    return dbg.records_to(f"records-{tag}-{NAME}.jsonl")    # every sealed record is emitted here


# =================================================================================================
if MODE == "native":
    PHASE = sys.argv[4]
    dbg = _load(MODEL, "bf16", os.environ.get("DEVICE", "cuda"), True, None)
    log(f"load {NAME} L_c={LC} kv={KC} phase={PHASE}")
    sink = custody(dbg, f"native-{PHASE}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "core").mkdir(exist_ok=True)

    if PHASE == "pools":
        for x in BAND:
            t = time.monotonic()
            pool = dbg.pool(CORRECT, f"cache.L{x}.v[kv{KC},row=@carrier]", labels=("country",),
                            name=f"carrier-correct-L{x}")
            pool.seal()
            pool.save(pool_path(x, "correct"))
            log(f"pool correct L{x} n={len(pool)} {time.monotonic() - t:.1f}s")
        pool = dbg.pool(WRONG, f"cache.L{LC}.v[kv{KC},row=@carrier]", labels=("country",),
                        name=f"carrier-wrong-L{LC}")
        pool.seal()
        pool.save(pool_path(LC, "wrong"))
        log(f"pool wrong L{LC} n={len(pool)}")
        json.dump({"model": NAME, "L_c": LC, "kv_c": KC, "band": BAND, "n_correct": len(CORRECT),
                   "wrong": [{k: w[k] for k in ("id", "family", "stratum", "rank", "supp", "carrier")}
                             for w in WRONG]}, open(f"pools-{NAME}.json", "w"), indent=0)

    pools = {x: Pool.load(pool_path(x, "correct")) for x in BAND}
    PC = pools[LC]
    WP = Pool.load(pool_path(LC, "wrong"))

    if PHASE == "A":
        spans = span_bindings(PC)
        out = open(f"armA-{NAME}-{shard(WRONG)[1]}.jsonl", "w")
        todo, _ = shard(WRONG)
        for it in todo:
            t = time.monotonic()
            r, last, a = it["carrier"], it["last"], it["answer"]
            code = code_or_none(PC, it)
            v = f64(WP.value(it)).reshape(-1)
            base = dbg.run(it["ids"], name="A0", prefill=last)
            base.until(f"t{last}.logits")
            lg0 = base.read(f"t{last}.logits")
            b = other_top(lg0, a)
            rec = {"id": it["id"], "family": it["family"], "stratum": it["stratum"], "rank": it["rank"],
                   "native_argmax": int(lg0.reshape(-1).argmax()), "b": b,
                   "mdb_correct": int(lg0.reshape(-1).argmax()) == a}
            core = OUT / "core" / f"{it['id'].replace(':', '_').replace(' ', '_').replace('/', '_')}.pt"
            crec = core_file.dump(base, core, row=last)
            rec["core"] = {"path": str(core), "record": crec["id"]}
            if code is None:
                rec["excluded"] = "no correct main-panel item of this country (pool class mean refused)"
                out.write(json.dumps(rec) + "\n")
                log(f"A {it['id']} excluded {time.monotonic() - t:.1f}s")
                continue
            rec.update(amp=code.amp(v), code_norm=code.N, n_class=code.n_class, pointers=code.pointers,
                       v_norm=float(v.norm()), span_mass={k: val(sb.mass(v, code.m_all)) for k, sb in spans.items()})
            run = dbg.run(it["ids"], name="A", prefill=last)
            br = dbg.branches(run, Margin(last, a, b), name=f"armA:{it['id']}")
            cell = cell_addr(LC, r)
            shape = WP.value(it).shape
            sb_true = subspace_binding(f"true:{it['id']}", LC, KC, code.d[:, None], authority="fitted",
                                       provenance={"rule": "LOO same-country class mean minus LOO all-correct mean, rank 1",
                                                   "test_informed": "per item (its true country)"})
            rec["true_digest"] = sb_true.digest()
            for al in DOSES:
                br.add(f"T{al:g}", [edit(cell, code.dose(v, al).reshape(shape).to(torch.bfloat16),
                                         origin={"kind": "amplitude-dose", "subspace": "true", "alpha": al})])
            for k, sb in spans.items():
                for al in DOSES:
                    vv = v + (al - 1.0) * sb.project(v - code.m_all)
                    br.add(f"S{k}_{al:g}", [edit(cell, vv.reshape(shape).to(torch.bfloat16),
                                                 origin={"kind": "amplitude-dose", "subspace": f"span{k}",
                                                         "alpha": al, "digest": sb.digest()})])
            rows = br.run(mode="batched")
            stepped = br.run_arms([f"T{al:g}" for al in DOSES], again=True)
            br.add_null("null_rnd3", "random", like="T3", seed=SEED)
            br.add_null("null_rnd8", "random", like="T8", seed=SEED + 1)
            br.add_null("null_self", "self", like="T2")
            nulls = br.run_arms(["null_rnd3", "null_rnd8", "null_self"])

            def slim(row):
                return {"M": val(row["M"]), "dM": val(row["dM"]), "argmax": row["argmax"],
                        "floor": val(row.get("floor", {"value": None})),
                        "floors": val(row.get("effect_floors") or {"value": None}),
                        "stability": row.get("stability"), "kernel_path": row["kernel_path"]}
            rec.update(M0=br.M0, floor=br.floor, base_argmax=br.base_argmax,
                       batched={rw["name"]: slim(rw) for rw in rows},
                       stepped={rw["name"]: slim(rw) for rw in stepped},
                       nulls={rw["name"]: slim(rw) for rw in nulls})
            out.write(json.dumps(rec) + "\n")
            out.flush()
            log(f"A {it['id']} {time.monotonic() - t:.1f}s")
        # E1 (sealed exploratory): dose-down on correct items + core files of correct base items (§2.3)
        todoc, _ = shard(CORRECT)
        oute = open(f"armE1-{NAME}-{shard(WRONG)[1]}.jsonl", "w")
        for it in todoc:
            t = time.monotonic()
            r, last, a = it["carrier"], it["last"], it["answer"]
            code = code_or_none(PC, it)
            run = dbg.run(it["ids"], name="E1", prefill=last)
            run.until(f"t{last}.logits")
            lg0 = run.read(f"t{last}.logits")
            b = other_top(lg0, a)
            rec = {"id": it["id"], "family": it["family"], "block": it["block"]}
            if it["block"] == "base" and it["donor"] in BY_ID:
                core = OUT / "core" / f"{it['id'].replace(':', '_').replace(' ', '_').replace('/', '_')}.pt"
                crec = core_file.dump(run, core, row=last)
                rec["core"] = {"path": str(core), "record": crec["id"]}
            if code is None:
                rec["excluded"] = "only item of its country"
                oute.write(json.dumps(rec) + "\n")
                continue
            v = f64(PC.value(it)).reshape(-1)
            shape = PC.value(it).shape
            run = dbg.run(it["ids"], name="E1b", prefill=last)
            br = dbg.branches(run, Margin(last, a, b), name=f"E1:{it['id']}")
            for al in DOWN:
                br.add(f"T{al:g}", [edit(cell_addr(LC, r), code.dose(v, al).reshape(shape).to(torch.bfloat16),
                                         origin={"kind": "amplitude-dose", "subspace": "true", "alpha": al})])
            rows = br.run(mode="batched")
            rec.update(amp=code.amp(v), code_norm=code.N, n_class=code.n_class, M0=br.M0, floor=br.floor,
                       base_argmax=br.base_argmax, b=b,
                       batched={rw["name"]: {"M": val(rw["M"]), "argmax": rw["argmax"],
                                             "floors": val(rw.get("effect_floors") or {"value": None})} for rw in rows})
            oute.write(json.dumps(rec) + "\n")
            oute.flush()
            log(f"E1 {it['id']} {time.monotonic() - t:.1f}s")

    if PHASE == "B":
        S_LAYERS = [LC + 2, LC + 4, LC + 6, L - 1]
        T_LAYERS = [LC - 8, LC - 6, LC - 4, LC - 2]
        todo, tag = shard(WRONG)
        out = open(f"armB-{NAME}-{tag}.jsonl", "w")
        for it in todo:
            t = time.monotonic()
            r, last, a = it["carrier"], it["last"], it["answer"]
            codes = {x: code_or_none(pools[x], it) for x in BAND}
            r0 = dbg.run(it["ids"], name="B0", prefill=last)
            r0.until(f"t{last}.logits")
            b = other_top(r0.read(f"t{last}.logits"), a)

            def probe_for(x):
                c = codes[x]
                return lambda run: c.amp(run.read_cache(x, "v", staged=False)[0, KC, r])
            probes = {f"amp_L{x}": probe_for(x) for x in BAND if codes[x] is not None}
            probes["v_norm_Lc"] = lambda run: float(f64(run.read_cache(LC, "v", staged=False)[0, KC, r]).norm())
            run = dbg.run(it["ids"], name="B", prefill=r)
            br = dbg.branches(run, Margin(last, a, b), at=f"t{r}.L{min(T_LAYERS)}.h",
                              name=f"armB:{it['id']}", probes=probes)
            ref = br.reference
            for s in S_LAYERS:
                for tl in T_LAYERS:
                    br.add(f"B{s}_{tl}", [edit(f"t{r}.L{tl}.h", ref.ref(f"t{r}.L{s}.h"))])
            for s in S_LAYERS:
                for tl in T_LAYERS:
                    br.add_null(f"R{s}_{tl}", "random", like=f"B{s}_{tl}", seed=SEED + s * 100 + tl)
            br.add_null("self", "self", like=f"B{LC + 4}_{LC - 4}")
            rows = br.run_arms()
            rec = {"id": it["id"], "family": it["family"], "stratum": it["stratum"], "rank": it["rank"],
                   "b": b, "M0": br.M0, "floor": br.floor, "base_argmax": br.base_argmax,
                   "base_probes": {k: (val(v) if not isinstance(v, list) else [val(x) for x in v])
                                   for k, v in br.base_probes.items()},
                   "excluded": codes[LC] is None,
                   "arms": {rw["name"]: {"M": val(rw["M"]), "dM": val(rw["dM"]), "argmax": rw["argmax"],
                                         "floors": val(rw.get("effect_floors") or {"value": None}),
                                         "stability": rw.get("stability"),
                                         "probes": {k: val(v) for k, v in rw.get("probes", {}).items()}}
                            for rw in rows}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            log(f"B {it['id']} {len(rows)} arms {time.monotonic() - t:.1f}s")
    log("DONE")

# =================================================================================================
if MODE == "replica":
    DEV = sys.argv[4]
    dbg = _load(MODEL, "fp32", DEV, True, None)
    log(f"load fp32 {DEV} {NAME}")
    sink = custody(dbg, "replica")
    PC = Pool.load(pool_path(LC, "correct"))
    WP = Pool.load(pool_path(LC, "wrong"))
    named = json.load(open(HERE / "exp5-panels.json"))["named"] if NAME == "Qwen3-1.7B-Base" else {}
    SW_LAYERS = list(range(LC + 1, LC + 5))
    wcache = {}
    PART = os.environ.get("PART", "C,F")

    _cores = {}

    def core_run(it, name):
        """A replica run of the item's core file (loaded and digest-checked once per item); the
        null check runs on the base run only."""
        if it["id"] not in _cores:
            _cores.clear()
            core = OUT / "core" / f"{it['id'].replace(':', '_').replace(' ', '_').replace('/', '_')}.pt"
            _cores[it["id"]] = core_file.load(core)
        return dbg.run_from(_cores[it["id"]], name=name.replace(".", "p"), check=(name == "base"))

    def to_logits(run, last):
        run.until(f"t{last}.logits")
        return run.read(f"t{last}.logits").reshape(-1)

    if "C" in PART:
        todo, tag = shard(WRONG)
        out = open(f"armC-{NAME}-{tag}.jsonl", "w")
        rng = random.Random(SEED)
        for it in todo:
            t = time.monotonic()
            code = code_or_none(PC, it)
            if code is None:
                continue
            r, last, a = it["carrier"], it["last"], it["answer"]
            v = f64(WP.value(it)).reshape(-1)
            base = core_run(it, "base")
            lg0 = to_logits(base, last)
            b = other_top(lg0, a)
            rec = {"id": it["id"], "stratum": it["stratum"], "b": b, "core_check": base.core_check,
                   "M0": float(lg0[a] - lg0[b]), "amp": code.amp(v), "ladder": {}}
            alpha_star, edited_star = None, None
            for al in DOSES:
                ed = core_run(it, f"d{al:g}")
                ed.set(cell_addr(LC, r), code.dose(v, al).reshape(WP.value(it).shape).to(torch.float32))
                lg = to_logits(ed, last)
                rec["ladder"][f"{al:g}"] = {"M": float(lg[a] - lg[b]), "argmax": int(lg.argmax())}
                if alpha_star is None and int(lg.argmax()) == a:
                    alpha_star, edited_star = al, ed
            rec["alpha_star"] = alpha_star
            if alpha_star is None and os.environ.get("SMOKE"):     # smoke only: exercise the edge code
                edited_star = ed
            if edited_star is not None:
                reader = readout(base, Margin(last, a, b))
                edge = symbols.hop_edge(base, edited_star, reader, cell_layer=LC, top_neurons=256, cache=wcache)
                rec["closure_gap"] = val(edge["closure"]["gap"])
                rec["shares"] = {k: val(v_) for k, v_ in edge["shares"].items() if k != "meaning"}
                switches, effects = [], {}
                for layer in edge["mlp"]:
                    x = layer["layer"]
                    g0 = f64(base.read(f"t{last}.L{x}.g")).reshape(-1)
                    g1 = f64(edited_star.read(f"t{last}.L{x}.g")).reshape(-1)
                    sw = ((g0 > 0) != (g1 > 0))
                    for n in layer["neurons"]:
                        i = int(n["name"].split(".f")[1])
                        effects[(x, i)] = abs(val(n["effect"]))
                        if x in SW_LAYERS and bool(sw[i]):
                            switches.append((abs(val(n["effect"])), x, i))
                    if x in SW_LAYERS:
                        rec.setdefault("switch_count", {})[x] = int(sw.sum())
                        rec.setdefault("mlp_exact", {})[x] = val(layer["exact"])
                        rec.setdefault("mlp_linear", {})[x] = val(layer["linear"])
                switches.sort(reverse=True)
                rec["switch_top"] = [[x, i, e] for e, x, i in switches[:64]]
                country = it["bridge"]
                comp = [tuple(p) for p in named.get(country, []) if p[0] in SW_LAYERS]
                if comp:
                    N = len(comp)
                    sset = {(x, i) for _, x, i in switches[:N]}
                    jac = len(sset & set(comp)) / len(sset | set(comp)) if (sset or comp) else 0.0
                    d_ff = int(g0.numel())
                    rnd = []
                    for _ in range(200):
                        rs = {(rng.choice(SW_LAYERS), rng.randrange(d_ff)) for _ in range(N)}
                        rnd.append(len(rs & set(comp)) / len(rs | set(comp)))
                    rec["composer"] = {"country": country, "N": N, "n_switch_listed": len(switches),
                                       "jaccard": jac, "overlap": len(sset & set(comp)),
                                       "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd),
                                       "composer_any_switch": sum(1 for p in comp if p in {(x, i) for _, x, i in switches})}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            log(f"C {it['id']} {time.monotonic() - t:.1f}s")

    if "F" in PART:
        todo, tag = shard(CORRECT_BASE)
        out = open(f"forecast-{NAME}-{tag}.jsonl", "w")
        for it in todo:
            t = time.monotonic()
            don = BY_ID.get(it["donor"])
            if don is None:
                continue
            try:
                md = f64(PC.class_mean_for({"id": "__donor_country__", "country": don["country"]},
                                           by="country").value).reshape(-1)
            except Exception:
                continue
            r, last, a, b = it["carrier"], it["last"], it["answer"], don["answer"]
            if a == b:
                continue
            v = f64(PC.value(it)).reshape(-1)
            base = core_run(it, "base")
            lg0 = to_logits(base, last)
            reader = readout(base, Margin(last, a, b))
            rec = {"id": it["id"], "donor": don["id"], "M0": float(lg0[a] - lg0[b]),
                   "core_check": base.core_check, "doses": {}}
            for al in (0.25, 0.5, 1.0, 2.0):
                ed = core_run(it, f"f{al:g}")
                ed.set(cell_addr(LC, r), (v + al * (md - v)).reshape(PC.value(it).shape).to(torch.float32))
                lg = to_logits(ed, last)
                edge = symbols.hop_edge(base, ed, reader, cell_layer=LC, top_neurons=0, cache=wcache)
                d = {"native_delta": float((lg[a] - lg[b]) - (lg0[a] - lg0[b])), "argmax": int(lg.argmax()),
                     "direct": val(edge["direct"]), "reread": val(edge["totals"]["reread"]),
                     "closure_gap": val(edge["closure"]["gap"]),
                     "mlp_by_layer": {e["layer"]: val(e["exact"]) for e in edge["mlp"]},
                     "reread_by_layer": {e["layer"]: val(e["effect"]) for e in edge["reread"]}}
                if al == 1.0:
                    fc = symbols.gate_switch_forecast(base, ed, reader, cell_layer=LC, ks=(8, 32, 128), cache=wcache)
                    d["F1"] = val(fc["F1"])
                    d["F2"] = {k: val(x) for k, x in fc["F2"].items()}
                    d["F3"] = {k: val(x) for k, x in fc["F3"].items()}
                    d["switches"] = {e["layer"]: e["gate_sign_switches"] for e in fc["layers"]}
                rec["doses"][f"{al:g}"] = d
            out.write(json.dumps(rec) + "\n")
            out.flush()
            log(f"F {it['id']} {time.monotonic() - t:.1f}s")
    log("DONE")
