"""Job 3: SAE census of the bridge variable at the measured carrier, and bridge features.

python job3_census.py MODEL SAE candidates.json job1-panel.json GT.json OUT

GT.json: {recipient id: {"row": r, "layer": l, "kv": g}} (the job-2 ground-truth carrier).
Prereg census: the descriptor-row residual x = L<l>.h (SAE layer l-1, resid_post) of recipient
and donor, read on runs truncated at row r (it is their pending row); x = b + Phi + E (exact,
float64). Arms: null (x_rec), full (x_don), phi (x_rec + Phi_don - Phi_rec), E (x_rec + E_don -
E_rec), each rounded once to bf16 and set at L<l>.h of a truncated recipient run, which then
runs natively; its value cell cache.L<l>.v[kv g, row r] is the only thing transplanted into the
recipient's full run (stepped, run to logits). Also: the direct donor-cell transplant (P3 check).

Bridge features: for every (recipient, donor) the SAE features ranked by their linearised share
of the value-cell change dv_g = W_v,g (w_in * (x_don - x_rec) / rms(x_rec)) (derived-local),
with the logit-lens tokens of the top decoder directions.
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.analysis import sae as saemod
from mdb.repl.__main__ import _load

MODEL, SAE, CAND, PANEL, GTF, OUT = sys.argv[1:7]
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
L, KV, D = dbg.adapter.layers, dbg.adapter.num_kv_heads, dbg.adapter.head_dim
cand = {i["id"]: i for i in json.load(open(CAND))["items"]}
panel = {p["id"]: p for p in json.load(open(PANEL))["panel"]}
gts = json.load(open(GTF))
need = sorted({g["layer"] - 1 for g in gts.values() if g["layer"] >= 1})
saes = saemod.open_set("qs", SAE, layers=need, cache=3)
print(f"load {time.monotonic() - T0:.1f}s; {len(gts)} carriers; SAE layers {need}", flush=True)
OUTF = open(f"{OUT}.jsonl", "w")
blocks = dbg.adapter.backbone.layers if hasattr(dbg.adapter.backbone, "layers") else None


def truncated(ids, r, layer, value=None):
    """Run ids[:r+1] (row r pending) to L<layer>.h, optionally replace it, run to logits;
    returns (x read at L<layer>.h before any edit, the value cell row r of layer `layer`)."""
    run = dbg.run(ids[:r + 1], name="tr")
    run.until(f"t{r}.L{layer}.h")
    x = run.read().clone()
    if value is not None:
        run.set(f"t{r}.L{layer}.h", value.to(dtype=x.dtype, device=x.device).reshape(x.shape))
    run.until(f"t{r}.logits")
    return x, run.read_cache(layer, "v", staged=False)[:, :, r:r + 1].clone()


def full(ids, layer, cell, r, g):
    run = dbg.run(ids, name="full")
    if cell is not None:
        run.set_cell(layer, "v", cell, rows=[r], kv=[g])
    run.until(f"t{len(ids) - 1}.logits")
    return run.read().reshape(-1).float().cpu()


# items ordered by carrier layer so each SAE layer loads once
order = sorted(gts, key=lambda i: gts[i]["layer"])
for rid in order:
    gt = gts[rid]
    r, layer, g = gt["row"], gt["layer"], gt["kv"]
    item = cand[rid]
    ids, a = item["ids"], item["answer"]
    rec = {"id": rid, "carrier": gt, "pairs": []}
    if layer < 1:
        rec["skipped"] = "carrier at layer 0: its input is the embedding, no SAE"
        OUTF.write(json.dumps(rec) + "\n")
        continue
    t = time.monotonic()
    basis = saes.get(layer - 1)
    assert basis.covers(layer, "h")
    base = full(ids, layer, None, r, g)
    x_rec, v_rec_tr = truncated(ids, r, layer)
    run_full = dbg.run(ids, name="cells")
    v_rec_full = run_full.read_cache(layer, "v", staged=False)[:, :, r:r + 1].clone()
    rec["trunc_vs_prefill_cell_max_abs"] = float((v_rec_tr.float() - v_rec_full.float()).abs().max())
    dec_rec = basis.decompose(x_rec)
    rec["rec_error_fraction"] = dec_rec.stats()["error_fraction_of_norm"]["value"]
    # linearised value-cell map at x_rec (derived-local): dv_g = W_v,g (w_in * dx) / rms(x_rec)
    module = dbg.adapter.model.model.layers[layer]
    w_in = module.input_layernorm.weight.double().cpu()
    eps = float(module.input_layernorm.variance_epsilon)
    xr = x_rec.double().cpu().reshape(-1)
    rms = float((xr.pow(2).mean() + eps) ** 0.5)
    Wv = module.self_attn.v_proj.weight[g * D:(g + 1) * D].double().cpu()   # rows sliced, then cast
    for k, did in enumerate(panel[rid]["donors"]):
        donor = cand[did]
        b = donor["answer"]
        x_don, _ = truncated(donor["ids"], r, layer)
        dec_don = basis.decompose(x_don)
        arms = {"null": dec_rec.x, "full": dec_don.x,
                "phi": dec_rec.x + (dec_don.recon - dec_rec.recon),
                "E": dec_rec.x + (dec_don.error - dec_rec.error)}
        pair = {"donor": did, "don_error_fraction": dec_don.stats()["error_fraction_of_norm"]["value"]}
        m0 = float(base[a] - base[b])
        for name, value in arms.items():
            _, cell = truncated(ids, r, layer, value)
            lg = full(ids, layer, cell[:, g:g + 1], r, g)
            pair[name] = float(lg[a] - lg[b]) - m0
            pair[name + "_argmax"] = int(lg.argmax())
        run_d = dbg.run(donor["ids"], name="dcell")
        dcell = run_d.read_cache(layer, "v", staged=False)[:, g:g + 1, r:r + 1].clone()
        lg = full(ids, layer, dcell, r, g)
        pair["direct"] = float(lg[a] - lg[b]) - m0
        pair["direct_argmax"] = int(lg.argmax())
        # bridge features: linearised share of each feature change in dv_g
        acts = {}
        for dec, sign in ((dec_rec, -1.0), (dec_don, 1.0)):
            for f, act in zip(dec.ids, dec.acts.tolist()):
                acts[f] = acts.get(f, 0.0) + sign * act
        ids_f = list(acts)
        dirs = basis.directions(ids_f)                                       # (d, n) float64
        delta = torch.tensor([acts[f] for f in ids_f], dtype=torch.float64)
        dv_f = Wv @ ((w_in[:, None] * dirs) / rms) * delta                   # (D, n)
        dv_total = Wv @ (w_in * (dec_don.x - dec_rec.x) / rms)
        unit = dv_total / dv_total.norm()
        along = (unit @ dv_f)                                                # share along dv
        top = along.abs().topk(min(8, len(ids_f)))
        pair["dv_norm_lin"] = float(dv_total.norm())
        pair["dv_error_along"] = float(unit @ (Wv @ (w_in * (dec_don.error - dec_rec.error) / rms)))
        pair["dv_bias_free_features_along"] = float(along.sum())
        pair["top_features"] = []
        for i in top.indices.tolist():
            f = ids_f[i]
            lens = saemod.logit_lens(run_full, basis.direction(f), top=5)
            pair["top_features"].append({"f": f, "delta_act": acts[f], "along": float(along[i]),
                                         "lens": [tok.decode([x]) for x in lens]})
        rec["pairs"].append(pair)
    rec["seconds"] = time.monotonic() - t
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    print("CENSUS " + json.dumps({"id": rid, "carrier": gt,
                                  "pairs": [{k: p[k] for k in ("full", "phi", "E", "null", "direct")}
                                            for p in rec["pairs"]],
                                  "feat": [[x["f"], round(x["along"], 3), x["lens"][:3]]
                                           for x in rec["pairs"][0]["top_features"][:3]],
                                  "s": round(rec["seconds"], 1)}), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
