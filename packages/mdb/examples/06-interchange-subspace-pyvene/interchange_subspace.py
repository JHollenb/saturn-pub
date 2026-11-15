# %% [markdown]
# # Interchange at the carrier cell: pyvene and mdb, the whole cell and three subspaces
#
# **Question.** In Qwen3-1.7B-Base, a two-hop prompt such as *"The capital of the country whose
# currency is the yen is"* resolves the bridge country (Japan) in one value cell: layer 20, KV
# group 6, at the *carrier row* (the span row that the answer row's L20 heads 12 and 13 attend to
# most). If we copy that cell from a donor prompt with another bridge (the won: Korea), does the
# answer follow the donor? That is an **interchange intervention**; the share of pairs where the
# answer flips is the **interchange intervention accuracy (IIA)**. And is the variable carried by
# a small subspace of the 128-dimensional cell?
#
# **Two tools, one cell.** [pyvene](https://github.com/stanfordnlp/pyvene) states the interchange
# as a declared config (`v_proj.output` of layer 20, KV group 6 as a subspace partition, base and
# source positions). mdb is a debugger: it runs the model one port at a time, takes the donor's
# value with its origin (`Run.ref`), branches the recipient's run from one checkpoint and
# measures every arm natively against **one floor** (the bf16 noise floor of the margin). So we
# can check that the two tools agree, and then use mdb's objects to ask the subspace question:
#
# * `SubspaceBinding`: a named subspace of the cell with its basis authority and provenance.
#   Here the **SAE pushforward**: the images in the cell of the country features of a
#   Qwen-Scope SAE that are active at the carrier row (training-free; median rank 4).
# * `subspace.interchange` on `Branches`: the subspace swap `v + Q Q^T (v_donor - v)` and the
#   whole-cell swap on one branch point, with nulls added to the same object.
# * `symbols.table(subspaces=...)`: the subspace shown as a binding in the carrier's symbol table.
#
# **Measured result** (showcase exp7, `job-c8144767543d`, 64 pairs; framing from
# `results/showcase-20261114/AUDIT-SHOWCASE.md`): pyvene and mdb agree on 63/64 flip verdicts
# (whole-cell IIA 44/64 mdb, 43/64 pyvene). Their margin gap has the same median as the
# no-intervention kernel-path gap (0.125) but a heavier tail (max 0.75 vs 0.375). The honest
# negative: the SAE subspace (median rank 4) moves a median 0.34 of the whole-cell margin change
# but flips only 4/64; a random subspace of the same rank flips 0/64.
#
# This example runs the first 34 of the 64 pairs (`MDB_NPAIRS`), which hold the one pair where
# the two tools disagree and all four SAE flips. Everything runs as a Python library; `MODEL`
# and `SAE` default to Beast paths.

# %%
import json
import os
import statistics as st
import time
from pathlib import Path

HERE = Path(os.environ.get("MDB_EXAMPLE_DIR", Path.cwd()))
MODEL = Path(os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen3-1.7B-Base"))
SAE = os.environ.get("MDB_SAE", "/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50")
NPAIRS = int(os.environ.get("MDB_NPAIRS", "34"))     # first N pairs in id order (the study: 64)
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
CL, CG = 20, 6                                        # the carrier cell: layer 20, KV group 6
SEED = 20261114
SHOW = "city:Gdansk"                                  # the pair shown in the symbol table
T0 = time.monotonic()
DATA = json.load(open(HERE / "items.json"))
REF = json.load(open(HERE / "reference.json"))
PAIRS = DATA["pairs"][:NPAIRS]
RESULT: dict = {"model": str(MODEL), "sae": SAE, "npairs": len(PAIRS)}
print(len(PAIRS), "pairs:", ", ".join(p["recipient"]["id"] for p in PAIRS))

# %% [markdown]
# ## 1. Load the model through mdb, declare custody
#
# `mdb.attach` wraps a loaded Hugging Face model (bf16, eager attention). `dbg.custody` declares
# the script and the preregistration (`prereg.json`, the showcase's sealed predictions for this
# experiment): every record sealed from now on carries their sha256, the library versions and
# the kernel path. `dbg.records_to` appends every sealed record (branches, pools, symbol tables)
# to a JSONL file, so the results persist with no project attached.

# %%
import torch

import mdb
from mdb.engine.models import load_local
from transformers import AutoTokenizer

torch.set_grad_enabled(False)
model = load_local(MODEL, dtype="bf16", device=DEVICE)
tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
dbg = mdb.attach(model, tok)
ad = dbg.adapter
DH, KV = ad.head_dim, ad.num_kv_heads
custody = dbg.custody(script=HERE / "interchange_subspace.py", prereg=HERE / "prereg.json",
                      kernel_path="mdb-stepped", note="examples/06 interchange + subspaces")
sink = dbg.records_to(OUT / "records.jsonl")
print(f"model loaded in {time.monotonic() - T0:.0f}s; head_dim {DH}, {KV} KV groups; "
      f"script sha256 {custody['script_sha256'][:12]}, prereg {custody['prereg_sha256'][:12]}")

# %% [markdown]
# ## 2. The carrier rule, the SAE and the pyvene model
#
# * The carrier row is the frozen B1 rule: the argmax over the bridge span of the answer row's
#   L20 attention, heads 12 and 13 (the two query heads of KV group 6). `circuit_map.slot_rows`
#   evaluates it on a run.
# * Qwen-Scope's layer-19 SAE reads `L19.h_out`, which is mdb's `L20.h`: the input of the layer
#   whose value projection writes the cell. The country features are the 58 features of the
#   frozen exp1 census.
# * pyvene 0.1.8 maps Qwen2 but not Qwen3; `register_qwen3()` copies Qwen2's module mapping to
#   the Qwen3 classes. `value_cell_config` is mdb's cell `cache.L20.v[kv6]` as a pyvene config.
#   pyvene gets a `shared_clone` of the model (same weights, its own module tree), so mdb's
#   interceptors and pyvene's hooks never meet.

# %%
import pyvene as pv

from mdb.analysis import circuit_map as cm
from mdb.analysis import sae
from mdb.interop.clone import shared_clone
from mdb.interop.pyvene_compat import register_qwen3, value_cell_config

CARRIER = cm.Slot("carrier", "carrier", "R_argmax", layers=(CL,), kv=(CG,), rule_layer=CL,
                  rule_heads=(12, 13))


def carrier_row(run, item):
    return cm.slot_rows(CARRIER, run, {"name": item["id"], "span": item["span"],
                                       "last": len(item["ids"]) - 1})[0]


B19 = sae.open_set("qwen-scope", SAE, layers=[19], cache=2).get(19)
COUNTRY = set(DATA["census_features"])
registered = register_qwen3()
PV_CFG = value_cell_config(CL, CG, DH, KV)
IM = pv.IntervenableModel({k: v for k, v in PV_CFG.items() if k != "kv"}, model=shared_clone(model))
print("SAE", B19.site, "k", B19.k, "| census country features", len(COUNTRY),
      "| pyvene registered", registered, "| config", PV_CFG["component"])

# %% [markdown]
# ## 3. Donors: one native run each, the donor cell as a `ValueRef`, and a `Pool`
#
# Each donor prompt runs natively (prefill up to its bridge span, then stepped), to find its
# carrier row and its active SAE features. `Run.ref("t<r>.L20.v[kv6]")` reads the donor's cell
# with the REPL's selector and returns a `ValueRef`: the 128 values plus their **origin pointer**
# (donor run, address, sha256, cut). An edit made with it records where its value came from.
#
# The same cells are also collected as a `Pool` (`dbg.pool`): one value per donor at
# `cache.L20.v[kv6,row=@carrier]`, with each donor's carrier row as its `@carrier` role and the
# same prefill split, sealed as a `pool` record with every item's origin. We check that the pool
# and `Run.ref` hold the same bits, then hand the interchange the pool's donor (`pool.donor`).

# %%
from mdb.analysis.finite import f64

DONORS = {}
for p in PAIRS:
    d = p["donor"]
    if d["id"] in DONORS:
        continue
    last = len(d["ids"]) - 1
    drun = dbg.run(d["ids"], name="donor", prefill=d["span"][0])
    drun.until(f"t{last}.logits")
    dcrow = carrier_row(drun, d)
    acts = set(B19.encode(drun.read(f"t{dcrow}.L{CL}.h"))["ids"])
    DONORS[d["id"]] = {"item": d, "crow": dcrow, "acts": acts,
                       "ref": drun.ref(f"t{dcrow}.L{CL}.v[kv{CG}]")}
    del drun
print(len(DONORS), "donors; example pointer:",
      {k: v for k, v in next(iter(DONORS.values()))["ref"].pointer.items() if k != "cut"})

pool_items = [{"id": k, "ids": v["item"]["ids"], "roles": {"carrier": v["crow"]},
               "span0": v["item"]["span"][0]} for k, v in DONORS.items()]
try:
    POOL = dbg.pool(pool_items, f"cache.L{CL}.v[kv{CG},row=@carrier]", name="donor-carrier-v",
                    prefill=lambda item: item["span0"])
except TypeError:                     # an mdb without Pool(prefill=): keep the Run.ref values
    POOL = None
if POOL is not None:
    POOL.seal()
    same = all(torch.equal(POOL.value(k).reshape(-1).cpu(), v["ref"].value.reshape(-1).cpu())
               for k, v in DONORS.items())
    print(f"pool {POOL.name}: n={len(POOL)}, kernel path {POOL.kernel_path}, "
          f"record {POOL.record['id'][:16]}; bitwise equal to Run.ref on every donor: {same}")
    RESULT["pool"] = {"n": len(POOL), "kernel_path": POOL.kernel_path, "equal_to_run_ref": same,
                      "record": POOL.record["id"]}
else:
    print("this mdb has no Pool(prefill=): donors come from Run.ref only")
    RESULT["pool"] = None


def donor_ref(pair):
    """The donor's cell as a ValueRef: from the pool (pointer names pool, rule, item), else from
    the donor run (pointer names run, address, cut)."""
    did = pair["donor"]["id"]
    if POOL is not None and RESULT["pool"]["equal_to_run_ref"]:
        return POOL.donor(pair["recipient"], rule="any", order=[did])
    return DONORS[did]["ref"]


# %% [markdown]
# ## 4. Subspaces as mdb bindings
#
# A `SubspaceBinding` names a `k`-dimensional subspace of the cell by an orthonormal basis, with
# the **authority of the basis** and its provenance. Three kinds here:
#
# * **SAE pushforward** (per pair, `derived-exact`): each active country feature's direction
#   `d_f` mapped into the cell at the recipient's carrier row, `W_V^g (rho gamma d_f)`
#   (`ComponentBinding.image`: the value projection of KV group 6 after the layer's RMSNorm at the
#   native `rho`). The features are the census country features active at the carrier row of the
#   recipient or the donor. Its rank is the number of such features (median 4).
# * **Random** (per pair, `supplied`): a seeded Gaussian basis of the same rank.
# * **Country readout span** (global, `derived-exact`, `subspace.readout_subspace`): the top 32
#   principal directions of the symbol table's content readout of a declared country-token set,
#   mapped into the cell. `full69` contains the test countries' tokens (test-informed); `a69`
#   contains none of them (test-free). These two arms are the readout-span comparison of the DAS
#   follow-up (section 7), re-measured here on the same pairs.

# %%
from mdb.analysis import subspace as sub
from mdb.analysis.components import component_binding

READOUT = {name: sub.readout_subspace(ad, f"readout-{name}", CL, CG, DATA["readout_tokens"][name],
                                      32, set_name=name)
           for name in ("full69", "a69")}
for name, sb in READOUT.items():
    print(f"readout span {name}: k={sb.k}, authority {sb.authority}, "
          f"explained {sb.provenance['explained']:.3f}, digest {sb.digest()[:12]}")


def sae_subspace(run, item, crow, acts_r, acts_d):
    """The SAE pushforward binding of one pair (None when no country feature is active)."""
    feats = sorted((acts_r | acts_d) & COUNTRY)
    if not feats:
        return None, feats
    cb = component_binding(run, "carrier", f"cache.L{CL}.v[kv{CG},row={crow}]", B19, features=[])
    images = torch.stack([cb.image(B19.direction(f)) for f in feats], 1)       # (d_h, n)
    sb = sub.subspace_binding(
        "sae-country", CL, CG, images, authority="derived-exact",
        provenance={"method": "W_V^g (rho gamma d_f) at the recipient's carrier row",
                    "sae": "Qwen-Scope W32K L0_50 layer 19 (reads L20.h)", "features": feats,
                    "feature_set": "exp1 census country features active at the carrier row of "
                                   "the recipient or the donor",
                    "census_sha256": DATA["census_sha256"], "item": item["id"], "row": crow})
    return sb, feats


# %% [markdown]
# ## 5. The interchange, pair by pair
#
# For each pair (recipient item, donor), consumer `M = logit(answer) - logit(donor answer)`:
#
# 1. **Native run** of the recipient (same prefill split as the donor): `M0`, the floor
#    (`verdict.margin_floor`: 2 bf16 spacings at the larger `|logit|`), the carrier row and the
#    active SAE features.
# 2. `subspace.interchange(dbg, sae_sb, ids, crow, donor_ref, Margin, floor=, keep=True)`: a run
#    with prefill = carrier row, branched at `t<crow>.L20.h`; arm `M` sets the cell to
#    `v + Q Q^T (v_donor - v)`, arm `M_whole` to `v_donor`. With a `ValueRef` source both are
#    selector edits `t<crow>.L20.v[kv6]` whose origin keeps the donor pointer.
# 3. On the same `Branches` object (same checkpoint, same floor): the random subspace, the two
#    readout spans, and the built-in **self-patch null** (`add_null("self")`: the cell replaced by
#    its own value, `dM` must be exactly 0). `seal()` writes one `branches` record per pair.
# 4. **pyvene**: the whole-cell interchange from its declared config (base position = recipient
#    carrier row, source position = donor carrier row), and its no-intervention forward
#    (`M_clean`): pyvene runs one batched forward, mdb steps, so `M_clean - M0` is the kernel-path
#    gap with no intervention at all.
#
# A **flip**: `M0 > floor` and `M_after < -floor` (the showcase rule; `subspace.iia` also counts
# it at 4 floors). The item loop is plain Python: prompt-set iteration (`each`, gap-audit P7) is
# not built in mdb yet.

# %%
from mdb.analysis.branches import edit
from mdb.analysis.forecast import Margin
from mdb.analysis.verdict import margin_floor

REF_BY_ID = {r["id"]: r for r in REF["pairs"]}
ROWS, KEEP = [], {}
for index, pair in enumerate(PAIRS):
    t_pair = time.monotonic()
    item, did = pair["recipient"], pair["donor"]["id"]
    ids, last = item["ids"], len(item["ids"]) - 1
    a, b = item["answer"], pair["donor"]["answer"]
    donor = DONORS[did]
    run = dbg.run(ids, name="recipient", prefill=item["span"][0])
    run.until(f"t{last}.logits")
    crow = carrier_row(run, item)
    lg0 = run.read(f"t{last}.logits").reshape(-1)
    M0, floor = float(lg0[a].float() - lg0[b].float()), margin_floor(lg0, a, b)
    v_run = f64(run.read_cache(CL, "v", staged=False)[0, CG, crow]).cpu()   # this run's cell
    acts_r = set(B19.encode(run.read(f"t{crow}.L{CL}.h"))["ids"])
    sae_sb, feats = sae_subspace(run, item, crow, acts_r, donor["acts"])
    rank = 0 if sae_sb is None else sae_sb.k
    ref = donor_ref(pair)
    consumer = Margin(last, a, b)
    # whole cell + SAE subspace on one branch point (a pair with no active country feature gets a
    # zero-rank subspace: the full-cell binding is used only to carry the whole arm)
    carrier_sb = sae_sb or sub.subspace_binding("whole", CL, CG, torch.eye(DH, dtype=torch.float64))
    res = sub.interchange(dbg, carrier_sb, ids, crow, ref, consumer, floor=floor, keep=True)
    br = res["branches"]
    v_nat = f64(br.reference.read_cache(CL, "v", staged=False)[0, CG, crow]).cpu()
    v_don = f64(ref.value).reshape(-1).cpu()
    target = f"t{crow}.L{CL}.v[kv{CG}]"
    rnd = None
    if rank:
        g = torch.Generator().manual_seed(SEED + index)
        rnd = sub.subspace_binding("random", CL, CG, torch.randn(DH, rank, generator=g,
                                                                 dtype=torch.float64),
                                   provenance={"rule": "Gaussian, same rank as sae-country",
                                               "seed": SEED + index})
        br.add("random", [edit(target, rnd.swap(v_nat, v_don),
                               origin={"kind": "subspace-interchange", "subspace": "random",
                                       "digest": rnd.digest(), "source": ref.pointer})])
    for name, sb in READOUT.items():
        br.add(name, [edit(target, sb.swap(v_nat, v_don),
                           origin={"kind": "subspace-interchange", "subspace": sb.name,
                                   "digest": sb.digest(), "source": ref.pointer})])
    # diagnostic arms: the same swaps against the cell of the native run (prefill = start of the
    # span), as the showcase built them; section 8 uses them to explain any difference
    for name, sb in (("sae_vnative", sae_sb), ("a69_vnative", READOUT["a69"])):
        if sb is not None:
            br.add(name, [edit(target, sb.swap(v_run, v_don),
                               origin={"kind": "subspace-interchange", "subspace": sb.name,
                                       "against": "native-run cell (showcase convention)"})])
    br.add_null("self", "self", like="M_whole")
    br.run()                                           # measures the arms added since
    table = {r["name"]: r for r in br.table()}
    record = br.seal()
    # pyvene: the declared whole-cell interchange, and its clean forward
    bi = torch.tensor([ids], device=DEVICE)
    si = torch.tensor([pair["donor"]["ids"]], device=DEVICE)
    _, cf = IM({"input_ids": bi}, [{"input_ids": si}],
               {"sources->base": ([[[donor["crow"]]]], [[[crow]]])}, subspaces=[[[CG]]])
    lp = cf.logits[0, -1].float()
    lc = IM.model(bi).logits[0, -1].float()
    row = {"id": item["id"], "donor": did, "crow": crow, "donor_crow": donor["crow"],
           "M0": M0, "floor": floor, "branches_M0": br.M0, "rank": rank, "features": feats,
           "whole": table["M_whole"]["M"],
           "sae": table["M"]["M"] if rank else M0,     # no subspace: no change (the showcase rule)
           "random": table["random"]["M"] if rank else M0,
           "full69_k32": table["full69"]["M"], "a69_k32": table["a69"]["M"],
           "self_dM": table["self"]["dM"],
           "sae_vnative": table["sae_vnative"]["M"] if rank else M0,
           "a69_vnative": table["a69_vnative"]["M"],
           "cell_kernel_gap": float((v_nat - v_run).abs().max()),
           "pyvene": float(lp[a] - lp[b]), "pyvene_clean": float(lc[a] - lc[b]),
           "mass_of_delta_sae": res["mass_of_delta"]["value"] if rank else 0.0,
           "record": record["id"], "seconds": round(time.monotonic() - t_pair, 2)}
    ROWS.append(row)
    if item["id"] == SHOW or (SHOW not in {p["recipient"]["id"] for p in PAIRS} and index == 0):
        KEEP = {"run": run, "sb": sae_sb, "rnd": rnd, "crow": crow, "consumer": consumer,
                "row": row}
    else:
        del run
    del br, res
    rf = REF_BY_ID[item["id"]]
    print(f"{item['id']:28} crow {crow} rank {rank}  M0 {M0:7.3f} (showcase {rf['M0']:7.3f})  "
          f"whole {row['whole']:7.3f} ({rf['whole']:7.3f})  pyvene {row['pyvene']:7.3f} "
          f"({rf['pyvene']:7.3f})  sae {row['sae']:7.3f} ({rf['sae']:7.3f})  random "
          f"{row['random']:6.3f}  full69 {row['full69_k32']:6.3f}  a69 {row['a69_k32']:6.3f}  "
          f"self dM {row['self_dM']}  [{row['seconds']}s]", flush=True)
RESULT["rows"] = ROWS

# %% [markdown]
# ## 6. IIA, pyvene against mdb, and the subspace in the symbol table
#
# `subspace.iia` counts flips over the records at 1 floor (the showcase rule) and at 4 floors
# (the audit's stable-verdict margin). The fraction of the whole-cell margin change a subspace
# moves is `(M0 - M_sub) / (M0 - M_whole)`.

# %%
med = lambda xs: st.median(xs) if xs else float("nan")  # noqa: E731
ARMS = ("whole", "pyvene", "sae", "random", "full69_k32", "a69_k32")
iia = {arm: sub.iia([{"M0": r["M0"], "floor": r["floor"], "M": r[arm]} for r in ROWS])
       for arm in ARMS}
iia_branches_m0 = sub.iia([{"M0": r["branches_M0"], "floor": r["floor"], "M": r["whole"]}
                           for r in ROWS])
moved = {arm: med([(r["M0"] - r[arm]) / (r["M0"] - r["whole"]) for r in ROWS
                   if r["M0"] != r["whole"]]) for arm in ("sae", "random", "full69_k32", "a69_k32")}
flip = lambda r, arm: r[arm] < -r["floor"]  # noqa: E731
agree = sum(flip(r, "whole") == flip(r, "pyvene") for r in ROWS)
gap = [abs(r["pyvene"] - r["whole"]) for r in ROWS]
kern = [abs(r["pyvene_clean"] - r["M0"]) for r in ROWS]
summary = {
    "pairs": len(ROWS),
    "flips": {arm: f"{iia[arm]['margin1']['flips']}/{iia[arm]['margin1']['eligible']}" for arm in ARMS},
    "flips_margin4": {arm: f"{iia[arm]['margin4']['flips']}/{iia[arm]['margin4']['eligible']}"
                      for arm in ARMS},
    "whole_flips_with_branches_M0": f"{iia_branches_m0['margin1']['flips']}/"
                                    f"{iia_branches_m0['margin1']['eligible']}",
    "pyvene_mdb_verdict_agree": f"{agree}/{len(ROWS)}",
    "disagree": [r["id"] for r in ROWS if flip(r, "whole") != flip(r, "pyvene")],
    "pyvene_mdb_gap": {"median": med(gap), "max": max(gap)},
    "kernel_path_clean_gap": {"median": med(kern), "max": max(kern)},
    "sae_rank": {"median": med([r["rank"] for r in ROWS]), "max": max(r["rank"] for r in ROWS)},
    "median_frac_moved": moved,
    "self_null_max_abs_dM": max(abs(r["self_dM"]) for r in ROWS),
    "median_mass_of_delta_sae": med([r["mass_of_delta_sae"] for r in ROWS if r["rank"]])}
RESULT["summary"] = summary
print(json.dumps(summary, indent=1))

# %% [markdown]
# Now attach the measured results to the bindings and read the carrier's symbol table for one
# pair (`SHOW`, default `city:Gdansk`, a pair the SAE subspace flips). `symbols.table(...,
# subspaces=[...])` adds a `subspaces` section: per binding its basis authority, provenance,
# digest, the native cell's mass inside the subspace, the direct (fixed-attention) consumer
# share of the in-subspace part, and the attached labels (here this pair's interchange and the
# panel IIA). The table is sealed as a record like every other.

# %%
from mdb.analysis import symbols

show = KEEP["row"]
bindings = []
if KEEP["sb"] is not None:
    sbs = KEEP["sb"].bound(KEEP["crow"])
    sbs.attach("interchange", {"M0": show["M0"], "M": show["sae"], "M_whole": show["whole"],
                               "floor": show["floor"]})
    sbs.attach("iia_panel", iia["sae"]["margin1"])
    bindings.append(sbs)
if KEEP["rnd"] is not None:
    rb = KEEP["rnd"].bound(KEEP["crow"])
    rb.attach("interchange", {"M": show["random"]})
    rb.attach("iia_panel", iia["random"]["margin1"])
    bindings.append(rb)
for name, sb in READOUT.items():
    bound = sb.bound(KEEP["crow"])
    bound.attach("interchange", {"M": show[f"{name}_k32"]})
    bound.attach("iia_panel", iia[f"{name}_k32"]["margin1"])
    bindings.append(bound)
t_table = time.monotonic()
tab = symbols.table(KEEP["run"], consumer=KEEP["consumer"], subspaces=bindings, limit=8)
rows_sub = tab.report["subspaces"]
print(f"symbol table for {show['id']} at {KEEP['run'].where} "
      f"({time.monotonic() - t_table:.1f}s, records {tab.report['records'][:1]})")
for r in rows_sub:
    share = r.get("share_of_cell")
    print(f"  {r['name']:16} k={int(r['k']['value']):3}  {r['basis_authority']:13} "
          f"mass_native {r['mass_native']['value']:.3f}  "
          f"to_consumer {r['subspace_to_consumer']['value']:+.3f} of cell "
          f"{r['cell_to_consumer']['value']:+.3f}  share {share and round(share['value'], 3)}  "
          f"labels {sorted(r['labels'])}")
RESULT["symbol_table"] = {"pair": show["id"], "subspaces": [
    {"name": r["name"], "k": r["k"]["value"], "authority": r["basis_authority"],
     "mass_native": r["mass_native"]["value"],
     "subspace_to_consumer": r["subspace_to_consumer"]["value"],
     "cell_to_consumer": r["cell_to_consumer"]["value"]} for r in rows_sub]}

# %% [markdown]
# ## 7. Context: DAS and the readout span (audited, not re-run here)
#
# The DAS study (tracker entry 11b, `results/showcase-20261114/das/`) trained pyvene DAS
# rotations on 38 other countries and scored them on the same 64 pairs, through the same mdb
# branches. Audited numbers (`das/AUDIT-DAS.md`, `das/followup/AUDIT-FOLLOWUP.md`), flips at
# 1 floor out of 64:
#
# | subspace (k = 32 unless stated) | flips | jobs |
# |---|---|---|
# | whole cell | 44 | job-c8144767543d (exp7), reproduced bitwise in 11b |
# | DAS, trained on 38 other countries (held out) | 16 | train job-14bfa70f5ddb; score job-70dfc80ed3c4, job-f9da01e895ef, job-5ce9e8d2f078 |
# | country readout span, 69 tokens incl. the test countries (`full69`) | 35 | same scoring jobs |
# | readout span with no test country (69 or 91 tokens) | 20-22 | job-8b01f1a12d84, job-33a4b67a28eb, job-60bce10afa0c, job-70ebd29d8f10 |
# | readout span minus only the pair's own two countries | 22 | same follow-up jobs |
# | SAE pushforward, per pair (median rank 4) | 4 | exp7 |
# | random | 0 | 11b |
#
# How to read it (per the audits): the readout span's lead is mostly per-pair identity. Removing
# only the pair's two country tokens costs 13 of 35 flips; the other test countries add nothing
# beyond a same-size generic span. A test-free span (20-22) does at least as well as held-out
# DAS (16), not significantly better on 64 pairs. At rank 4 or below the per-pair SAE set is the
# better handle (4 flips and 0.34 of the margin moved, against 0 and 0.19 for the readout span at
# k=4), but it stops at about 4 dimensions; the rank-matched SAE sweep trails the readout span.
# The country code is high-rank and only partly shared across countries. DAS is not trained in
# this example; section 5 re-measures the `full69` and `a69` k=32 arms on this subset.

# %% [markdown]
# ## 8. Showcase against now, on the same pairs
#
# `reference.json` holds the showcase's per-pair numbers for these pairs (exp7
# `job-c8144767543d`; the readout arms from the follow-up jobs), cut by `make_subset.py`. Exact
# equality is expected for `M0`, the carrier rows, the SAE rank and features, the whole-cell
# margin, both pyvene margins and the readout arms. The SAE and random arms can differ: the
# showcase swapped against the recipient's cell from the native run (prefill = start of the
# span), while `subspace.interchange` swaps against the cell of the branched run itself
# (prefill = carrier row), so the complement of the subspace keeps that run's own value. The
# random basis is also drawn per pair here (`SEED + index`), not from one stream over all 64.
# The diagnostic arms `sae_vnative` and `a69_vnative` redo the swap against the native run's
# cell, the showcase convention, and must then equal the showcase exactly.

# %%
ref_rows = [REF_BY_ID[r["id"]] for r in ROWS]
checks = []
for key in ("M0", "crow", "donor_crow", "rank", "features", "whole", "pyvene", "pyvene_clean",
            "branches_M0", "full69_k32", "a69_k32", "sae", "random"):
    equal = sum(r[key] == f[key] for r, f in zip(ROWS, ref_rows))
    diff = (max(abs(r[key] - f[key]) for r, f in zip(ROWS, ref_rows))
            if key not in ("features",) else None)
    checks.append((key, equal, diff))
    print(f"{key:13} equal on {equal}/{len(ROWS)}" + (f"  max |diff| {diff}" if diff else ""))
for key, ref_key in (("sae_vnative", "sae"), ("a69_vnative", "a69_k32")):
    equal = sum(r[key] == f[ref_key] for r, f in zip(ROWS, ref_rows))
    checks.append((key, equal, None))
    print(f"{key:13} equal to showcase {ref_key} on {equal}/{len(ROWS)}")
moved_cells = sum(r["cell_kernel_gap"] > 0 for r in ROWS)
print(f"recipient cell differs between the native run and the branched run on {moved_cells}/"
      f"{len(ROWS)} pairs (max |dv| {max(r['cell_kernel_gap'] for r in ROWS)})")
RESULT["equal_counts"] = {k: e for k, e, _ in checks}
ref_summary = {"flips": {}, "flips_margin4": {}}
for arm in ARMS:
    for m, key in ((1.0, "flips"), (4.0, "flips_margin4")):
        n = sum(1 for f in ref_rows if f["M0"] > m * f["floor"] and f[arm] < -m * f["floor"])
        ref_summary[key][arm] = n
print(f"\n{'number':34} {'showcase':>10} {'now':>10}")
for arm in ARMS:
    print(f"{'flips @1 floor ' + arm:34} {ref_summary['flips'][arm]:>10} "
          f"{iia[arm]['margin1']['flips']:>10}")
for arm in ARMS:
    print(f"{'flips @4 floors ' + arm:34} {ref_summary['flips_margin4'][arm]:>10} "
          f"{iia[arm]['margin4']['flips']:>10}")
rs = REF["summary"] if len(ROWS) == REF["summary"]["pairs"] else None
if rs:
    for key, now in (("pyvene_mdb_verdict_agree", agree),
                     ("median_frac_moved sae", moved["sae"]),
                     ("median_frac_moved random", moved["random"]),
                     ("pyvene_mdb_gap median", summary["pyvene_mdb_gap"]["median"]),
                     ("pyvene_mdb_gap max", summary["pyvene_mdb_gap"]["max"]),
                     ("kernel_path_clean_gap median", summary["kernel_path_clean_gap"]["median"]),
                     ("kernel_path_clean_gap max", summary["kernel_path_clean_gap"]["max"])):
        path = key.split(" ")
        val = rs[path[0]] if len(path) == 1 else rs[path[0]][path[1]]
        print(f"{key:34} {round(val, 4):>10} {round(now, 4):>10}")
print(f"self-patch null: max |dM| {summary['self_null_max_abs_dM']} (must be 0)")
RESULT["showcase_reference"] = ref_summary

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib not installed: skipping the plot")
if plt is not None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].scatter([r["whole"] for r in ROWS], [r["pyvene"] for r in ROWS], s=14)
    lim = [min(r["whole"] for r in ROWS) - 1, max(r["M0"] for r in ROWS) + 1]
    ax[0].plot(lim, lim, lw=0.8, color="grey")
    ax[0].set_xlabel("mdb whole-cell M")
    ax[0].set_ylabel("pyvene whole-cell M")
    ax[0].set_title("Whole-cell interchange: pyvene vs mdb")
    names = [("sae", "SAE (rank ~4)"), ("random", "random (same rank)"),
             ("full69_k32", "readout span full69, k=32"), ("a69_k32", "readout span a69, k=32")]
    for i, (arm, label) in enumerate(names):
        fr = [(r["M0"] - r[arm]) / (r["M0"] - r["whole"]) for r in ROWS if r["M0"] != r["whole"]]
        ax[1].boxplot(fr, positions=[i], widths=0.5)
    ax[1].set_xticks(range(len(names)), [n for _, n in names], rotation=15, fontsize=7)
    ax[1].axhline(1.0, lw=0.8, color="grey")
    ax[1].set_title("Fraction of the whole-cell margin change moved")
    fig.tight_layout()
    fig.savefig(OUT / "interchange_subspace.png", dpi=130)
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass
RESULT["seconds"] = round(time.monotonic() - T0, 1)
RESULT["records"] = str(OUT / "records.jsonl")
json.dump(RESULT, open(OUT / "interchange_subspace_results.json", "w"), indent=1, default=str)
print(f"done in {RESULT['seconds']}s; wrote {OUT}/interchange_subspace_results.json")
print("RESULT_JSON " + json.dumps({k: v for k, v in RESULT.items() if k != "rows"}, default=str))
