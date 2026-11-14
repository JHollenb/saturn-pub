"""Job G (exp7/prereg.json): interchange intervention accuracy at the carrier cell, with a floor
and a training-free subspace (pyvene x mdb branches x SAE pushforward). No DAS (needs owner
approval).  python exp7_iia.py OUTDIR [PICK]

Per Run 1a base pair (recipient = base item, source = its different-bridge donor; 64 pairs),
consumer M[answer, donor answer]:
- mdb: native run (prefill = first span row) -> carrier row (frozen R_argmax rule) and M0; the
  donor's carrier cell v (L20 KV6). Branches in sequence scope from one checkpoint at
  t<crow>.L20.h of a run with prefill = crow (every reader of the carrier row sees the edit,
  matching pyvene's positional interchange): (a) whole cell = donor v; (b) Pi_SAE: v + Q Q^T
  (v_donor - v) with Q the orthonormal basis of the cell images W_V^g (rho gamma d_f) of the
  frozen exp1 census country features active (L19, carrier row) in the recipient or the donor
  (Binding.component map at the recipient's rho); (c) a random subspace of the same rank (seed).
- pyvene 0.1.8 (Qwen3 shim mdb.interop.pyvene_compat): the whole-cell interchange as a declared
  config (v_proj.output of layer 20, unit pos, subspace = KV group 6 of the KV partition), base
  position = recipient carrier row, source position = donor carrier row, on a shared-weight clone;
  also its no-intervention forward (kernel-path delta against mdb's stepped native).
IIA flip: M0 > floor and M_after < -floor (floor = 2 bf16 spacings at the larger |logit|).
The log prints item ids and seconds only.
"""
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from common import CG, CL, SAE_DIR, carrier_row, floor_of, items, load, log  # noqa: E402
from mdb.analysis import sae  # noqa: E402
from mdb.analysis.components import component_binding  # noqa: E402
from mdb.interop.clone import shared_clone  # noqa: E402
from mdb.interop.pyvene_compat import register_qwen3, value_cell_config  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
dbg = load()
ad = dbg.adapter
DH, KV = ad.head_dim, ad.num_kv_heads
B19 = sae.open_set("qwen-scope", SAE_DIR, layers=[19], cache=2).get(19)
CEN = json.load(open("census.json"))
COUNTRY = {int(f) for f in CEN["features"]}
import pyvene as pv  # noqa: E402

added = register_qwen3()
cfg = value_cell_config(CL, CG, DH, KV)
pv_cfg = {k: v for k, v in cfg.items() if k != "kv"}
IM = pv.IntervenableModel(pv_cfg, model=shared_clone(ad.model))
log("load")
base = sorted(items("base"), key=lambda i: i["id"])
by_id = {i["id"]: i for i in items("base")}
if PICK:
    base = [base[int(k)] for k in PICK.split(",")]
rng = random.Random(20261114)
gen = torch.Generator().manual_seed(20261114)
fh = open(f"{OUT}/exp7-items{'-smoke' if PICK else ''}.jsonl", "w")
DON = {}


def native(item):
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="nat", prefill=item["span"][0])
    run.until(f"t{last}.logits")
    crow = carrier_row(run, item)
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    cb = component_binding(run, "c", f"cache.L{CL}.v[kv{CG},row={crow}]", B19, features=[])
    acts = set(B19.encode(run.read(f"t{crow}.L{CL}.h"))["ids"])
    return run, crow, lg, cb, acts


for item in base:
    t0 = time.monotonic()
    a = item["answer"]
    did = item["donor"]["id"]
    donor = by_id.get(did) or {"id": did, "ids": item["donor"]["ids"], "span": [item["donor"]["row"]],
                               "answer": item["donor"]["answer"]}
    b = item["donor"]["answer"]
    run, crow, lg0, cb, acts_r = native(item)
    del run
    if did not in DON:
        drun, dcrow, _, dcb, acts_d = native(donor)
        DON[did] = (dcrow, dcb.native.clone(), acts_d)
        del drun
    dcrow, dv, acts_d = DON[did]
    feats = sorted((acts_r | acts_d) & COUNTRY)
    if feats:
        M = torch.stack([cb.image(B19.direction(f)) for f in feats], 1)        # (dh, n)
        Q, R = torch.linalg.qr(M)
        rank = int((R.diagonal().abs() > 1e-9 * R.diagonal().abs().max()).sum())
        Q = Q[:, :rank]
    else:
        Q, rank = torch.zeros(DH, 0, dtype=torch.float64), 0
    Qr = torch.linalg.qr(torch.randn(DH, max(rank, 1), generator=gen, dtype=torch.float64))[0][:, :rank]
    v = cb.native
    arms_v = {"whole": dv.double().cpu(), "sae": v + Q @ (Q.T @ (dv.double().cpu() - v)),
              "random": v + Qr @ (Qr.T @ (dv.double().cpu() - v))}
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="br", prefill=crow)
    run.until(f"t{crow}.L{CL}.h")
    ck = run.checkpoint("pre")
    shape = run.port_shape("v")
    out = {}
    for name, val in arms_v.items():
        full = torch.zeros(shape, dtype=ad.dtype, device="cuda")
        mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
        full.reshape(-1)[CG * DH:(CG + 1) * DH] = val.to("cuda", ad.dtype)
        mask.reshape(-1)[CG * DH:(CG + 1) * DH] = True
        run.set(f"t{crow}.L{CL}.v", full, mask=mask)
        run.until(f"t{last}.logits")
        lg = run.read(f"t{last}.logits").reshape(-1).float()
        out[name] = {"M": float(lg[a] - lg[b]), "argmax": int(lg.argmax())}
        run.restore(ck)
    del run
    with torch.no_grad():
        bi = torch.tensor([ids], device="cuda")
        si = torch.tensor([donor["ids"]], device="cuda")
        _, cf = IM({"input_ids": bi}, [{"input_ids": si}], {"sources->base": ([[[dcrow]]], [[[crow]]])},
                   subspaces=[[[CG]]])
        lp = cf.logits[0, -1].float()
        lc = IM.model(bi).logits[0, -1].float()
    rec = {"id": item["id"], "donor": did, "crow": crow, "donor_crow": dcrow, "M0": float(lg0[a] - lg0[b]),
           "floor": floor_of(lg0, a, b), "rank": rank, "features": feats, "arms": out,
           "pyvene": {"M": float(lp[a] - lp[b]), "M_clean": float(lc[a] - lc[b]), "argmax": int(lp.argmax())},
           "seconds": round(time.monotonic() - t0, 2)}
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    log(f"G {item['id']} {rec['seconds']}s")
json.dump({"pyvene": pv.__version__ if hasattr(pv, "__version__") else "0.1.8", "registered": added,
           "config": {k: str(v) for k, v in cfg.items()}}, open(f"{OUT}/exp7-prov{'-smoke' if PICK else ''}.json", "w"), indent=1)
log("TOTAL")
