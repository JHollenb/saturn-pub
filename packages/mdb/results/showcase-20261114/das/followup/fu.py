"""11b follow-up (followup/prereg.json): vocabulary size vs test identity for the country readout span (Q1)
and a rank-matched SAE sweep (Q2), scored on exp7's 64 Run 1a pairs exactly as das.py (same pairs, carrier
row, floor, stepped interchange), built from mdb objects: subspace.readout_subspace / subspace_binding /
SubspaceBinding.swap / mass, branches.Branches (one checkpoint per pair, every arm at one floor).

python fu.py OUT [PICK] [TAG]    one record per pair -> OUT/score-TAG.jsonl   (no training, no DAS)
"""
import hashlib
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from common import CG, CL, SAE_DIR, carrier_row, floor_of, items, load, log  # noqa: E402
from mdb.analysis import sae  # noqa: E402
from mdb.analysis import subspace as sub  # noqa: E402
from mdb.analysis.branches import Branches, edit  # noqa: E402
from mdb.analysis.components import component_binding  # noqa: E402
from mdb.analysis.finite import EXACT  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.verdict import spacing  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
TAG = sys.argv[3] if len(sys.argv) > 3 else ("smoke" if PICK else "full")
SHAS = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in ("fu.py", "prereg.json", "vocab.json")}
dbg = load()
ad = dbg.adapter
DH = ad.head_dim
cust = dbg.custody(script="fu.py", prereg="prereg.json", note="11b follow-up Q1/Q2")
log(f"load custody script {cust.get('script_sha256', '')[:12]} prereg {cust.get('prereg_sha256', '')[:12]}")
V = json.load(open("vocab.json"))
KS_Q1, KS_Q2, KS_RO = [8, 16, 32, 64], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 64]
CEN = json.load(open("census.json"))
CENSUS = sorted(int(f) for f in CEN["features"])
COUNTRY = set(CENSUS)
assert len(CENSUS) == 58


def test_pairs():
    base = sorted(items("base"), key=lambda i: i["id"])
    by_id = {i["id"]: i for i in items("base")}
    out = []
    for item in base:
        did = item["donor"]["id"]
        donor = by_id.get(did) or {"id": did, "ids": item["donor"]["ids"], "span": [item["donor"]["row"]],
                                   "answer": item["donor"]["answer"], "bridge": item["donor"]["bridge"]}
        out.append((item, donor))
    return out


def readout(name, tokens, k, setname):
    return sub.readout_subspace(ad, name, CL, CG, tokens, k, set_name=setname)


# global (pair-independent) readout spans
GLOBAL = {}
for k in KS_RO:
    GLOBAL[f"full69_k{k}"] = readout(f"full69_k{k}", V["full69"], k, "full69 (11b vocabulary)")
for k in KS_Q1:
    GLOBAL[f"a69_k{k}"] = readout(f"a69_k{k}", V["a69"], k, "a69 (no test country, 69 tokens)")
    GLOBAL[f"aall_k{k}"] = readout(f"aall_k{k}", V["aall"], k, "aall (no test country, all single-token)")
log(f"global readout spans {len(GLOBAL)}")

B19 = sae.open_set("qwen-scope", SAE_DIR, layers=[19], cache=2).get(19)
IMG = None          # (DH, 58) unit-rho cell images of the 58 census decoder directions (set on the first pair)
RANK_A = RANK_B = None


def census_rank_B():
    """Rule B (census evidence; test-informed: the census items include the 64 base items): on-share
    (k/n on the feature's own country) desc, other-count asc, feature id asc."""
    def key(f):
        on, other = CEN["evidence"][str(f)]["on"], CEN["evidence"][str(f)]["other"]
        return (-int(on.split("/")[0]) / int(on.split("/")[1]), int(other.split("/")[0]), f)
    return sorted(CENSUS, key=key)


def build_sae(cb):
    global IMG, RANK_A, RANK_B
    img = torch.stack([cb.image(B19.direction(f)).double().cpu().reshape(-1) for f in CENSUS], 1) / cb.rho
    if IMG is None:
        IMG = img
        norms = IMG.norm(dim=0)
        RANK_A = sorted(range(58), key=lambda j: (-float(norms[j]), CENSUS[j]))     # column indices, rule A
        order_b = census_rank_B()
        RANK_B = [CENSUS.index(f) for f in order_b]
    dev = float((img - IMG).abs().max() / IMG.abs().max())
    assert dev < 1e-9, f"census images are not rho-proportional ({dev})"
    return dev


tests = test_pairs()
if PICK:
    tests = [tests[int(k)] for k in PICK.split(",")]
fh = open(f"{OUT}/score-{TAG}.jsonl", "w")
DON = {}
SAE_SUBS = None
for item, donor in tests:
    t0 = time.monotonic()
    a, b, did = item["answer"], donor["answer"], donor["id"]
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="nat", prefill=item["span"][0])
    run.until(f"t{last}.logits")
    crow = carrier_row(run, item)
    lg0 = run.read(f"t{last}.logits").reshape(-1).float()
    v = run.read_cache(CL, "v", staged=False)[0, CG, crow].double().cpu()
    cb = component_binding(run, "c", f"cache.L{CL}.v[kv{CG},row={crow}]", B19, features=[])
    acts_r = set(B19.encode(run.read(f"t{crow}.L{CL}.h"))["ids"])
    del run
    if did not in DON:
        drun = dbg.run(donor["ids"], name="don", prefill=donor["span"][0])
        dl = len(donor["ids"]) - 1
        drun.until(f"t{dl}.logits")
        dcrow = carrier_row(drun, donor)
        dv = drun.read_cache(CL, "v", staged=False)[0, CG, dcrow].double().cpu()
        acts_d = set(B19.encode(drun.read(f"t{dcrow}.L{CL}.h"))["ids"])
        DON[did] = (dcrow, dv, acts_d)
        del drun
    dcrow, dv, acts_d = DON[did]
    img_dev = build_sae(cb)
    if SAE_SUBS is None:
        SAE_SUBS = {}
        for rule, rank in (("A", RANK_A), ("B", RANK_B)):
            for k in KS_Q2:
                SAE_SUBS[f"sae{rule}_k{k}"] = sub.subspace_binding(
                    f"sae{rule}_k{k}", CL, CG, IMG[:, rank[:k]], authority=EXACT,
                    provenance={"method": "span of cell images W_V^g(gamma d_f) of the top-k census country "
                                          "features", "rule": rule, "features": [CENSUS[j] for j in rank[:k]]})
        log("SAE spans " + " ".join(f"{n}:{s.k}" for n, s in SAE_SUBS.items() if n.endswith("k32")))
    subs = dict(GLOBAL)
    subs.update(SAE_SUBS)
    drop = {int(item["country_token"]), int(item["donor"]["country_token"])}
    assert len(drop) == 2 and drop <= set(V["full69"]), (item["id"], drop)
    lto = [t for t in V["full69"] if t not in drop]
    for k in KS_Q1:
        subs[f"lto_k{k}"] = readout(f"lto_k{k}", lto, k, "full69 minus the pair's two countries")
    feats = sorted((acts_r | acts_d) & COUNTRY)                    # exp7's per-pair set
    if feats:
        Qs = sub.subspace_binding("sae_exp7", CL, CG, torch.stack([cb.image(B19.direction(f)) for f in feats], 1)
                                  .double().cpu(), authority=EXACT, provenance={"features": feats})
        subs["sae"] = Qs
        ordA = [CENSUS[j] for j in RANK_A]
        featsA = [f for f in ordA if f in feats]
        for k in KS_Q2:
            if featsA[:k]:
                cols = [CENSUS.index(f) for f in featsA[:k]]
                subs[f"saeC_k{k}"] = sub.subspace_binding(f"saeC_k{k}", CL, CG, IMG[:, cols], authority=EXACT,
                                                          provenance={"rule": "C", "features": featsA[:k]})
    fl = 2 * spacing(max(abs(float(lg0[a])), abs(float(lg0[b]))), torch.bfloat16)
    assert fl == floor_of(lg0, a, b)
    run = dbg.run(ids, name="br", prefill=crow)
    br = Branches(run, Margin(last, a, b), at=f"t{crow}.L{CL}.h", floor=fl, name=f"fu:{item['id']}")
    shape = run.port_shape("v")
    mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
    mask.reshape(-1)[CG * DH:(CG + 1) * DH] = True

    def full_of(val):
        full = torch.zeros(shape, dtype=ad.dtype, device="cuda")
        full.reshape(-1)[CG * DH:(CG + 1) * DH] = val.reshape(-1).to("cuda", ad.dtype)
        return full

    br.add("whole", [edit(f"t{crow}.L{CL}.v", full_of(dv), mask=mask, origin={"kind": "donor-cell", "arm": "whole"})])
    info = {"whole": {"k": DH, "mass": 1.0}}
    for name, sb in subs.items():
        br.add(name, [edit(f"t{crow}.L{CL}.v", full_of(sb.swap(v, dv)), mask=mask,
                           origin={"kind": "subspace-interchange", "subspace": sb.name, "digest": sb.digest()})])
        info[name] = {"k": sb.k, "mass": float(sb.mass(dv, reference=v)["value"])}
    rows = {r["name"]: r for r in br.run()}
    del run
    out = {n: {"M": float(r["M"]["value"]), "argmax": int(r["argmax"]), "effect_floors": float(r["M"]["value"]) / fl,
               **info[n]} for n, r in rows.items()}
    rec = {"id": item["id"], "donor": did, "crow": crow, "donor_crow": dcrow, "M0": float(lg0[a] - lg0[b]),
           "branches_M0": float(br.M0), "floor": fl, "M0_floors": float(lg0[a] - lg0[b]) / fl, "arms": out,
           "delta_norm": float((dv - v).norm()), "sae_exp7_features": feats, "img_dev": img_dev,
           "lto_dropped": sorted(drop), "script_sha256": SHAS["fu.py"], "prereg_sha256": SHAS["prereg.json"],
           "vocab_sha256": SHAS["vocab.json"], "custody": cust, "seconds": round(time.monotonic() - t0, 2)}
    rec["rankA"], rec["rankB"] = [CENSUS[j] for j in RANK_A], [CENSUS[j] for j in RANK_B]
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    log(f"S {item['id']} {rec['seconds']}s arms {len(out)}")
log("TOTAL")
