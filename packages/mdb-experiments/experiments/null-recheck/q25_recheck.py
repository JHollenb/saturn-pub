"""Null re-check on Qwen2.5-1.5B (tracker entries 4 and 5), one model load.

    python q25_recheck.py MODEL DATA_DIR OUT

Panel: the 27 capital pairs of interop J2 (job-ec87a6e1c6f2), template "The tower is located in the
city of {city}, It is the capital of" -> " {country}", city row 8, bf16. Recipient i keeps its J2
donor as donor 0 (canonical) and gets the next 15 kept cities in J2's shuffled order (16 donors,
all with a different country). Consumer per donor: M = logit(country_i) - logit(country_donor).

Claims (saturn experiments/2026-11-16-null-recheck/PREREG.md):
  C7  L22 KV1 key cell at the city row has no effect     (J2 mean -0.02; entry 4 Eiffel -0.06)
  C8  L22 KV0 value cell has no effect                    (J2 mean -0.00)
  C9  L22 head 7 is not used (zeroed -0.07)               -> resampled from donors
  C10 L23 neuron f4197 is Eiffel-specific (zeroed -0.04)  -> resampled from donors
  C11 city info has left its row by L26 (resid -0.06)     -> h at the L26 (and L24) input, row 8
C7-C10: mdb's batched native sweep (cell_sweep, cells and answer-row port patches; reader scope as
J2's default). C11: batched full forward, pre-hook on the layer (rows >= 8 recomputed).
"""
import random
import sys
from pathlib import Path

import torch

from common import dump, floor_of, informative, log, pool

from mdb.analysis.sweep import Cell, PortPatch, cell_sweep
from mdb.engine.ports import PortEdit
from mdb.repl.__main__ import _load

CAPITALS = """Paris France|Rome Italy|Berlin Germany|Madrid Spain|Tokyo Japan|Beijing China|
Moscow Russia|Cairo Egypt|Athens Greece|Vienna Austria|Lisbon Portugal|Dublin Ireland|
Oslo Norway|Stockholm Sweden|Helsinki Finland|Warsaw Poland|Budapest Hungary|Bangkok Thailand|
Ottawa Canada|Canberra Australia|Nairobi Kenya|Lima Peru|Santiago Chile|Havana Cuba|
Tehran Iran|Baghdad Iraq|Kabul Afghanistan|Manila Philippines|Hanoi Vietnam|Jakarta Indonesia|
Ankara Turkey|Kyiv Ukraine|Bucharest Romania|Sofia Bulgaria|Belgrade Serbia|
Copenhagen Denmark|Brussels Belgium|Amsterdam Netherlands|Bern Switzerland|Damascus Syria|
Beirut Lebanon|Amman Jordan|Caracas Venezuela|Bogota Colombia|Quito Ecuador|
Montevideo Uruguay|Islamabad Pakistan|Dhaka Bangladesh|Kathmandu Nepal|Doha Qatar|
Tunis Tunisia|Algiers Algeria|Rabat Morocco|Accra Ghana|Abuja Nigeria|Khartoum Sudan|
Brasilia Brazil|Reykjavik Iceland|Tallinn Estonia|Riga Latvia|Vilnius Lithuania|Minsk Belarus|
Tbilisi Georgia|Yerevan Armenia|Baku Azerbaijan|Tashkent Uzbekistan|Taipei Taiwan|
Kampala Uganda|Harare Zimbabwe|Lusaka Zambia|Luanda Angola|Dakar Senegal|Kinshasa Congo|
Muscat Oman|Sanaa Yemen|Bratislava Slovakia|Ljubljana Slovenia|Zagreb Croatia|Tirana Albania"""
STEM = "The tower is located in the city of"
N_DONORS, BATCH = 16, 96
RESID_LAYERS = (22, 24, 26)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HEAD, NEURON, ZERO_HEAD_CTRL = 7, 4197, 6


def main(model_dir, data, out):
    import json
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    dbg = _load(Path(model_dir), "bf16", DEV, True, None)
    tok, adapter = dbg.tokenizer, dbg.adapter
    model = adapter.model
    L, KV, hd = adapter.layers, adapter.num_kv_heads, adapter.head_dim
    log(f"load: {L} layers, {KV} kv heads, attn {model.config._attn_implementation}")
    j2 = json.load(open(data / "j2_items.json"))
    pairs, ROW = j2["pairs"], j2["row"]
    kept = [p[0] for p in pairs]
    assert all(pairs[i][1] == kept[(i + 1) % len(kept)] for i in range(len(pairs))), "J2 pair order"
    country = dict(e.strip().split(" ", 1) for e in CAPITALS.replace("\n", "").split("|"))
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    assert len(enc(STEM)) == ROW
    items = {}
    for c in kept:
        ids = enc(f"{STEM} {c}, It is the capital of")
        assert ids[ROW] == enc(" " + c)[0]
        items[c] = {"ids": ids, "ans": enc(" " + country[c])[0]}
    last = len(items[kept[0]]["ids"]) - 1
    assert all(len(x["ids"]) == last + 1 for x in items.values())

    layers = model.model.layers
    patch = {}
    cap = None

    def make_pre(li):
        def pre(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            if cap is not None:
                cap[li] = hs[0, ROW].detach().clone()
            if li not in patch:
                return None
            hs = hs.clone()
            lanes, vals = patch[li]
            hs[lanes, ROW] = vals.to(hs.dtype)
            if args:
                return (hs, *args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        return pre

    hooks = [layers[li].register_forward_pre_hook(make_pre(li), with_kwargs=True) for li in RESID_LAYERS]
    st = {}
    for c in kept:
        run = dbg.run(items[c]["ids"], name="c")
        k = [run.read_cache(layer, "k", staged=False)[0].clone() for layer in range(L)]
        v = [run.read_cache(layer, "v", staged=False)[0].clone() for layer in range(L)]
        run.until(f"t{last}.logits")
        st[c] = {"k": k, "v": v, "logits": run.read().reshape(-1).float().cpu(),
                 "A": run.read(f"t{last}.L22.A").clone(), "f": run.read(f"t{last}.L23.f").clone()}
    for c in kept:
        cap = {}
        model(torch.tensor([items[c]["ids"]], device=DEV))
        st[c]["h"] = cap
    cap = None
    a_shape, f_shape = tuple(st[kept[0]]["A"].shape), tuple(st[kept[0]]["f"].shape)
    log(f"captures: A {a_shape} f {f_shape} h {tuple(st[kept[0]]['h'][26].shape)}")
    mask_h = torch.zeros(a_shape, dtype=torch.bool)
    mask_h[..., HEAD * hd:(HEAD + 1) * hd] = True
    mask_h6 = torch.zeros(a_shape, dtype=torch.bool)
    mask_h6[..., ZERO_HEAD_CTRL * hd:(ZERO_HEAD_CTRL + 1) * hd] = True
    mask_f = torch.zeros(f_shape, dtype=torch.bool)
    mask_f[..., NEURON] = True

    recs = []
    try:
        for i, c in enumerate(kept):
            dons = [kept[(i + 1 + j) % len(kept)] for j in range(N_DONORS)]
            a = items[c]["ans"]
            s = st[c]
            toks = [a] + [items[d]["ans"] for d in dons]
            assert len(set(toks)) == len(toks)
            arms, keys = [], []
            for kd, d in enumerate(dons):
                sd = st[d]
                for name, cell, base in (
                        ("k22kv1", Cell(22, "k", ROW, 1, sd["k"][22][1, ROW]), s["k"][22][1, ROW]),
                        ("v22kv0", Cell(22, "v", ROW, 0, sd["v"][22][0, ROW]), s["v"][22][0, ROW]),
                        ("v22kv1", Cell(22, "v", ROW, 1, sd["v"][22][1, ROW]), s["v"][22][1, ROW])):
                    arms.append([cell])
                    keys.append((name, kd, informative(base, cell.value)))
                arms.append([PortPatch(22, "A", PortEdit("A", sd["A"], "replace", mask=mask_h))])
                keys.append(("A22h7", kd, informative(s["A"][mask_h.to(s["A"].device)],
                                                      sd["A"][mask_h.to(sd["A"].device)])))
                arms.append([PortPatch(23, "f", PortEdit("f", sd["f"], "replace", mask=mask_f))])
                keys.append(("f23n4197", kd, informative(s["f"][..., NEURON], sd["f"][..., NEURON])))
            for name, pp in (("A22h7_zero", PortPatch(22, "A", PortEdit("A", 0.0, "scale", mask=mask_h))),
                             ("A22h6_zero", PortPatch(22, "A", PortEdit("A", 0.0, "scale", mask=mask_h6))),
                             ("f23n4197_zero", PortPatch(23, "f", PortEdit("f", 0.0, "replace", mask=mask_f)))):
                arms.append([pp])
                keys.append((name, 0, True))
            run = dbg.run(items[c]["ids"], name="sweep")
            res = cell_sweep(run, arms, tokens=toks, batch=BATCH, stepped_logits=s["logits"])
            lg, nl = res["logits"].double(), res["null"].double()
            native = s["logits"]
            floors = []
            for kd in range(N_DONORS):
                m = nl[:, 0] - nl[:, 1 + kd]
                floors.append(floor_of(native[a], native[toks[1 + kd]], float(m.max() - m.min())))
            rows = []
            for j, (name, kd, inf) in enumerate(keys):
                n0 = nl[j // BATCH]
                dm = float((lg[j, 0] - lg[j, 1 + kd]) - (n0[0] - n0[1 + kd]))
                rows.append({"arm": name, "donor": dons[kd], "k": kd, "dM": dm, "floor": floors[kd],
                             "informative": inf, "to_donor": int(res["argmax"][j]) == toks[1 + kd],
                             "changed": int(res["argmax"][j]) != a})
            # C11: residual h at the layer input, row 8, full forward (rows >= 8 recomputed)
            B = 1 + N_DONORS * len(RESID_LAYERS)
            patch.clear()
            lanes = 1
            for li in RESID_LAYERS:
                vals = torch.stack([st[d]["h"][li].reshape(-1) for d in dons])
                patch[li] = (torch.arange(lanes, lanes + N_DONORS, device=DEV), vals)
                lanes += N_DONORS
            lgr = model(torch.tensor([items[c]["ids"]] * B, device=DEV)).logits[:, -1].double().cpu()
            patch.clear()
            m1 = float(native[a] - native[toks[1]])
            off = float(lgr[0, a] - lgr[0, toks[1]]) - m1
            lane = 1
            for li in RESID_LAYERS:
                for kd, d in enumerate(dons):
                    b = toks[1 + kd]
                    dm = float((lgr[lane, a] - lgr[lane, b]) - (lgr[0, a] - lgr[0, b]))
                    rows.append({"arm": f"h{li}", "donor": d, "k": kd, "dM": dm,
                                 "floor": floor_of(native[a], native[b], abs(off)),
                                 "informative": informative(s["h"][li], st[d]["h"][li]),
                                 "to_donor": int(lgr[lane].argmax()) == b, "changed": int(lgr[lane].argmax()) != a})
                    lane += 1
            recs.append({"city": c, "donors": dons, "M0_canonical": m1, "null_vs_stepped": res["null_vs_stepped"],
                         "resid_null_offset": off, "rows": rows})
            log(f"{i} {c}: {len(arms)} sweep arms, null_vs_stepped {res['null_vs_stepped']}, resid offset {off:.4f}")
    finally:
        for h in hooks:
            h.remove()
    summary = {"pairs": len(recs), "records": recs, "crosscheck": crosscheck(dbg, items, kept, st, last, ROW)}
    summary["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None
    dump(out / "q25_records.json", summary)
    log("done")


def crosscheck(dbg, items, kept, st, last, row):
    """Pair 0: the f4197 resample through mdb's multi-donor ``donors.whatif`` (2 donors, stepped,
    mdb's own verdict and donor label), and the K cell stepped through ``Run.set_cell``."""
    out = []
    try:
        from mdb.analysis import donors as D
        from mdb.analysis.forecast import Margin

        c = kept[0]
        a = items[c]["ans"]
        nat = st[c]["logits"]
        f_shape = tuple(st[c]["f"].shape)
        mask = torch.zeros(f_shape, dtype=torch.bool)
        mask[..., NEURON] = True
        for d in (kept[1], kept[2]):
            b = items[d]["ans"]
            run = dbg.run(items[c]["ids"], name="x")
            run.until(f"t{last}.L23.f")
            rep = D.whatif(run, f"t{last}.L23.f", [st[d]["f"]], Margin(last, a, b), mask=mask)
            delta = rep["per_donor"][0]["delta"]
            out.append({"arm": "f23n4197", "recipient": c, "donor": d, "mdb_label": rep["label"],
                        "stepped_delta": delta.get("value") if isinstance(delta, dict) else delta})
            log(f"crosscheck {out[-1]}")
            run = dbg.run(items[c]["ids"], name="y")
            run.set_cell(22, "k", st[d]["k"][22][1, row].reshape(1, 1, 1, -1), rows=[row], kv=[1])
            run.until(f"t{last}.logits")
            lg = run.read().reshape(-1).float().cpu()
            out.append({"arm": "k22kv1", "recipient": c, "donor": d,
                        "stepped_dM": float((lg[a] - lg[b]) - (nat[a] - nat[b]))})
            log(f"crosscheck {out[-1]}")
    except Exception as e:  # noqa: BLE001
        out.append({"error": f"{type(e).__name__}: {str(e)[:400]}"})
        log(f"crosscheck failed: {out[-1]}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:4])
