"""Z1: the latent chain seen through the symbol table (prereg_chain.json).

python z1_chain.py MODEL OUT [PICK]

Inputs in the working dir: candidates.json, job1-panel.json, gt.json, job2_gtv.json.
Per panel recipient: the native table with declared bridge / r0rest / desc bindings and candidate
sets (symbols.table(sets=, content_layers=, hops=)); the baselines B_mass, B_lens (L20.h, L24.h on
the run truncated at r0), B_answer (answer row L20.h_out); arm T_diff (donor 0's V cell at L20 KV6
into r0, re-take the table); arm T_city (same-country different-city donor, city recipients).
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from mdb.analysis import symbols
from mdb.analysis.forecast import Margin
from mdb.analysis.reads import head_mix, readout
from mdb.repl.__main__ import _load

MODEL, OUT = sys.argv[1:3]
PICK = sys.argv[3] if len(sys.argv) > 3 else ""   # comma-separated panel indices (smoke)
LAYER, GROUP = 20, 6
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
L, H, KV = dbg.adapter.layers, dbg.adapter.num_heads, dbg.adapter.num_kv_heads
spec = json.load(open("candidates.json"))
cand = {i["id"]: i for i in spec["items"]}
panel = json.load(open("job1-panel.json"))["panel"]
if PICK:
    panel = [panel[int(k)] for k in PICK.split(",")]
gt = json.load(open("gt.json"))
j2 = json.load(open("job2_gtv.json"))


def first_token(text):
    return tok(" " + text, add_special_tokens=False)["input_ids"][0]


COUNTRIES = tuple(sorted({i["country_token"] for i in spec["items"]}))
CAPITALS = tuple(sorted(set(spec["capital_tokens"].values())))
ENT_OF = {i["id"]: first_token(i["descriptor"]) for i in spec["items"]}
ENTITIES = tuple(sorted(set(ENT_OF.values())))
ALL = tuple(dict.fromkeys(COUNTRIES + CAPITALS + ENTITIES))
SET_KIND = {**{t: "entity" for t in ENTITIES}, **{t: "capital" for t in CAPITALS},
            **{t: "country" for t in COUNTRIES}}
city_by_country = defaultdict(list)
for i in spec["items"]:
    if i["family"] == "city":
        city_by_country[i["bridge"]].append(i["id"])
correct = {r["id"]: r["correct"] for r in json.load(open("job1-panel.json"))["rows"]}
OUTF = open(f"{OUT}.jsonl", "w")
print(f"load {time.monotonic() - T0:.1f}s; {len(panel)} items; sets countries {len(COUNTRIES)} "
      f"capitals {len(CAPITALS)} entities {len(ENTITIES)} all {len(ALL)}", flush=True)


def sets_for(item):
    return [symbols.Candidates("countries", COUNTRIES, item["country_token"]),
            symbols.Candidates("capitals", CAPITALS, item["answer"]),
            symbols.Candidates("entities", ENTITIES, ENT_OF[item["id"]]),
            symbols.Candidates("all", ALL, item["country_token"])]


def declare(run, item, r0):
    symbols.declare(run, "bridge", [f"cache.L{LAYER}.v[kv{GROUP},row={r0}]"])
    rest = ([f"t{r0}.L{x}.v" for x in range(L) if x != LAYER]
            + [f"cache.L{LAYER}.v[kv{g},row={r0}]" for g in range(KV) if g != GROUP])
    symbols.declare(run, "r0rest", rest)
    others = [f"t{r}" for r in item["span"] if r != r0]
    if others:
        symbols.declare(run, "desc", others)
    return 3 if others else 2


def brief(content):
    """best token, its text and kind, names_truth and truth_rank per set (bounded)."""
    out = {}
    for name in ("countries", "capitals", "entities", "all"):
        c = content[name]
        out[name] = {"best": c["best"], "text": tok.decode([c["best"]]),
                     "kind": SET_KIND.get(c["best"]), "names_truth": c.get("names_truth"),
                     "truth_rank": c.get("truth_rank"), "margin": c["margin"]["value"]}
    out["vocab_top"] = content["vocab_top"].get("text")
    return out


def take_table(run, item, consumer, hops=None):
    t = time.monotonic()
    n_decl = declare(run, item, item["row"])
    table = symbols.table(run, consumer=consumer, sets=sets_for(item), content_layers=True,
                          hops=hops, limit=n_decl, tokenizer=tok)
    views = {b["symbol"]: b for b in table.report["bindings"]}
    out = {}
    for name, v in views.items():
        if "content" not in v or "countries" not in v["content"]:
            out[name] = {"content": v.get("content")}
            continue
        out[name] = {"content": brief(v["content"]), "strip": v.get("content_lifetime"),
                     "to_consumer": v["to_consumer"].get("value"), "status": v["status"],
                     "hop": v.get("hop")}
    out["_seconds"] = round(time.monotonic() - t, 2)
    return out


def bridge_write(run, j, r0):
    mix = head_mix(run, j, LAYER, {})
    parts = mix.o_parts()
    return sum(parts[h, r0] for h in range(H) if mix.kappa[h] == GROUP)


def read_vec(run, sets, basis, vec):
    c = symbols.content(run, vec, sets, basis, tokenizer=tok)
    return brief(c)


def edited(item, donor, consumer):
    """Recipient run with donor's V cell (its own r0) at L20 KV6 written into the recipient's r0."""
    drun = dbg.run(donor["ids"], name="donor")
    dv = drun.read_cache(LAYER, "v", staged=False).clone()
    rd, r0 = donor["row"], item["row"]
    er = dbg.run(item["ids"], name="edit")
    er.set_cell(LAYER, "v", dv[:, GROUP:GROUP + 1, rd:rd + 1], rows=[r0], kv=[GROUP])
    er.until(f"t{item['n'] - 1}.logits")
    logits = er.read().reshape(-1).float()
    return er, logits


for entry in panel:
    t_item = time.monotonic()
    item = cand[entry["id"]]
    ids, j, r0, a = item["ids"], item["n"] - 1, item["row"], item["answer"]
    assert (gt[item["id"]]["row"], gt[item["id"]]["layer"], gt[item["id"]]["kv"]) == (r0, LAYER, GROUP)
    d0 = cand[entry["donors"][0]]
    b0 = d0["answer"]
    cons = Margin(j, a, b0)
    rec = {"id": item["id"], "family": item["family"], "bridge": item["bridge"], "r0": r0,
           "span": item["span"], "donor0": d0["id"], "truth": {
               "country": tok.decode([item["country_token"]]), "capital": tok.decode([a]),
               "entity": tok.decode([ENT_OF[item["id"]]])}}
    # native run
    run = dbg.run(ids, name="main")
    run.until(f"t{j}.logits")
    native = run.read().reshape(-1).float()
    rec["native_argmax"] = int(native.argmax())
    rec["native_M"] = float(native[a] - native[b0])
    reader = readout(run, cons)
    w_nat = bridge_write(run, j, r0)
    # T_diff
    er, el = edited(item, d0, cons)
    w_don = bridge_write(er, j, r0)
    direct = float((w_don - w_nat).double().cpu().reshape(-1) @ reader.vector.double().cpu().reshape(-1))
    native_delta = float(el[a] - el[b0]) - rec["native_M"]
    saved = j2[item["id"]]
    saved_delta = (saved["GT.V.d0"][0] - saved["GT.V.d0"][1]) - (saved["native"][0] - saved["native"][1])
    rec["hop_check"] = {"native_delta": native_delta, "job2_GT.V.d0_delta": saved_delta,
                        "direct_delta": direct}
    et = take_table(er, item, cons)
    eb = et["bridge"]["content"]
    rec["T_diff"] = {"argmax": int(el.argmax()), "argmax_text": tok.decode([int(el.argmax())]),
                     "answer_moves": int(el.argmax()) == b0,
                     "answer_kept": int(el.argmax()) == a,
                     "content_country": eb["countries"]["text"],
                     "content_moves": eb["countries"]["best"] == d0["country_token"],
                     "content_all": eb["all"]["text"], "table_seconds": et["_seconds"]}
    rec["T_diff"]["together"] = rec["T_diff"]["content_moves"] == rec["T_diff"]["answer_moves"]
    # native table with the hop
    tab = take_table(run, item, cons, hops={"bridge": {
        "native_delta": native_delta, "direct_delta": direct,
        "replacement": f"V cell L20 KV6 from {d0['id']} (stepped set_cell)"}})
    rec["table"] = tab
    basis = symbols.readout_basis(run, j)
    sets = sets_for(item)
    # B_mass: the most attended (row in span, layer, group) write
    best = (-1.0, None)
    writes = {}
    for x in range(L):
        mix = head_mix(run, j, x, {})
        parts = mix.o_parts()
        for g in range(KV):
            hs = [h for h in range(H) if mix.kappa[h] == g]
            for r in item["span"]:
                m = float(sum(mix.p[h, r] for h in hs))
                if m > best[0]:
                    best = (m, (r, x, g))
                    writes[(r, x, g)] = sum(parts[h, r] for h in hs)
    rec["B_mass"] = {"site": list(best[1]), "mass": best[0],
                     **read_vec(run, sets, basis, writes[best[1]])}
    # B_answer: answer row residual at L20.h_out
    rec["B_answer"] = read_vec(run, sets, basis,
                               run.read(f"t{j}.L{LAYER}.h_out").reshape(-1).double().cpu())
    # B_lens: truncated run, r0 residual at L20.h and L24.h, own RMS
    tr = dbg.run(ids[:r0 + 1], name="trunc")
    gamma = basis[1]
    eps = float(getattr(dbg.adapter.backbone, dbg.adapter.spec.final_norm).variance_epsilon)
    rec["B_lens"] = {}
    for x in (20, 24):
        tr.until(f"t{r0}.L{x}.h")
        v = tr.read().reshape(-1).double().cpu()
        rho = float((v.pow(2).mean() + eps) ** -0.5)
        rec["B_lens"][f"L{x}"] = read_vec(tr, sets, (rho, gamma), v)
    # T_city
    if item["family"] == "city":
        others = [c for c in city_by_country[item["bridge"]] if c != item["id"]]
        others.sort(key=lambda c: not correct.get(c, False))
        if others:
            dc = cand[others[0]]
            top2 = [int(x) for x in native.topk(2).indices]
            runner = top2[1] if top2[0] == a else top2[0]
            ccons = Margin(j, a, runner)
            cr, cl = edited(item, dc, ccons)
            ct = take_table(cr, item, ccons)
            cb = ct["bridge"]["content"]
            am = int(cl.argmax())
            rec["T_city"] = {"donor": dc["id"], "donor_correct": correct.get(dc["id"]),
                             "argmax": am, "argmax_text": tok.decode([am]),
                             "answer_moves": am != a,
                             "answer_is_donor_city_token": am == ENT_OF[dc["id"]],
                             "content_entity": cb["entities"]["text"],
                             "content_moves": cb["entities"]["best"] == ENT_OF[dc["id"]],
                             "content_entity_was": tab["bridge"]["content"]["entities"]["text"],
                             "content_country": cb["countries"]["text"],
                             "country_unchanged": cb["countries"]["best"]
                             == tab["bridge"]["content"]["countries"]["best"],
                             "content_all": cb["all"]["text"]}
            rec["T_city"]["together"] = rec["T_city"]["content_moves"] == rec["T_city"]["answer_moves"]
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    bc = tab["bridge"]["content"]
    print("Z1 " + json.dumps({"id": item["id"], "country": bc["countries"]["text"],
                              "ok": bc["countries"]["names_truth"], "all": bc["all"]["text"],
                              "lens24": rec["B_lens"]["L24"]["countries"]["text"],
                              "Tdiff": rec["T_diff"]["together"], "hop": rec["hop_check"],
                              "strip": tab["r0rest"]["strip"]["countries"],
                              "s": rec["seconds"], "tab_s": tab["_seconds"]}), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
