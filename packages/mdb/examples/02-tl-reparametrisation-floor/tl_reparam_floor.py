# %% [markdown]
# # TransformerLens weight processing as a reparametrisation null: how big must a patching effect be?
#
# **Question.** A patching effect is a difference of two numbers computed in bf16. If you compute
# the *same* forward a second way (another kernel order, another batch, the same weights folded
# differently), the effect moves by a few bf16 spacings. How large must an effect be before its
# verdict ("this head matters, with this sign") survives every such re-realisation?
#
# **Two realisations of one forward.**
#
# * **mdb** (the authority): the real Hugging Face model, stepped row by row, every patch a
#   native edit from one checkpoint (`Branches`), each effect `dM` read against one floor (2 bf16
#   spacings at the margin's `|logit|`). This is mdb's margin noise floor (`verdict.margin_floor`,
#   2 spacings), deliberately wider than the 1-spacing verdict floor that `verify` and `whatif`
#   use, because this example compares effects measured on different realisations (batched vs
#   stepped, TransformerLens vs mdb), not a prediction against its own native run.
# * **TransformerLens 4.4** `TransformerBridge` over the same weights: *unprocessed* (U16, the same
#   module) and *processed* (P16, `enable_compatibility_mode()`: RMSNorm gains folded into the next
#   linear layers, unembedding centred). Processing is a pure reparametrisation, so in exact
#   arithmetic both bridges compute the same function. All patch arms of an item run in one batch;
#   U16 also runs one arm at a time (batch 1). The fp32 bridges (U32, P32, on CPU) check that the
#   reparametrisation is pure.
#
# **The patches** (per item, `2H + 1` effects; consumer `M = logit(answer) - logit(donor answer)` at
# the last row): zero each query head's attention output `A` at the carrier layer `L_c` and at
# `L_c + 1`, and the carrier V interchange (the carrier KV group's `v` at the carrier row replaced
# by the donor's, in sequence scope). Qwen3-0.6B-Base: carrier `L20 KV6`; Qwen2.5-0.5B:
# `L21 KV0`.
#
# **What the original run measured** (48 items per model, one seed). This is an effect-size
# table, not a "verdicts don't hold" result:
#
# * Above about 4 floors, **0/214** effects lose their verdict on any path (110 Qwen3 + 104
#   Qwen2.5). Losses sit at 1-2 floors.
# * Sign reversals are rare: 1/274 (Qwen3), 28/228 (Qwen2.5, all from processing).
# * The kernel path dominates on Qwen3; processing dominates on Qwen2.5.
# * The 2-spacing floor is a sound noise bound, but a verdict needs about 4 floors (mdb's
#   `stability` band, `k = 4`).
# * fp32: processed vs unprocessed base margins differ by at most 2.3e-5 (Qwen3) / 1.8e-5
#   (Qwen2.5): processing is a pure reparametrisation.
#
# This example reruns both models on **24 of the 48 items** (every second base item sorted by id,
# 6 per family; `items.json`) in one job, with the hand-rolled parts of the original driver
# replaced by mdb debugger objects, and prints its numbers next to the original run's on the same
# items (`reference.json`, cut from the original raw outputs by `make_reference.py`).

# %%
import gc
import json
import os
import time
from pathlib import Path

HERE = Path(os.environ.get("MDB_EXAMPLE_DIR", ".")).resolve()
MODELS_DIR = Path(os.environ.get("MDB_MODELS", "/mnt/big/llm-models"))
NITEMS = int(os.environ.get("MDB_NITEMS", "24"))      # items per model (the subset has 24)
MODELS = [("Qwen3-0.6B-Base", 20, 6), ("Qwen2.5-0.5B", 21, 0)]   # (model, L_c, carrier KV group)
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.monotonic()
RESULT: dict = {"items_per_model": NITEMS, "models": {}}


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


DATA = json.load(open(HERE / "items.json"))
ITEMS = {i["id"]: i for i in DATA["items"]}
RUN_IDS = DATA["run"][:NITEMS]
REF = json.load(open(HERE / "reference.json"))
print(len(RUN_IDS), "items:", RUN_IDS)

# %% [markdown]
# ## 1. mdb as a library, with custody
#
# `dbg.custody(script=..., prereg=...)` makes every record sealed from now on carry the sha256 of
# this script and of the original run's sealed preregistration (copied here as `prereg.json`), the library
# versions and the kernel path. `dbg.records_to(path)` appends every sealed record (branches,
# supplied forecasts) to a JSONL file, each verifiable on read-back.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mdb
from mdb.analysis import circuit_map as cm
from mdb.analysis import stability, supplied, verdict
from mdb.analysis.branches import edit
from mdb.analysis.forecast import Margin
from mdb.engine.breakpoints import parse_selector, plan, stop_axes
from mdb.interop import names, tl

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)
VER = tl.version()
print("transformer-lens", VER, "torch", torch.__version__)


def load(path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=dtype, attn_implementation="eager",
        local_files_only=True).to(device).eval()


# %% [markdown]
# ## 2. One model: the debugger, the carrier slot and four bridges
#
# * The carrier row is a **`circuit_map.Slot`** with this model's carrier (`layers=(L_c,)`,
#   `kv=(g,)`) and the `R_argmax` rule: the span row (t0 excluded, ties to the later row) where
#   the answer row's `L_c` attention, summed over the KV group's query heads, peaks.
#   `cm.group_heads(adapter, g)` names those query heads (the grouping `group_writers` sums over),
#   so nothing is hard-coded per model.
# * U16 wraps the same module mdb is attached to (`tl.bridge`, a shared clone: same tensors).
#   P16 is a second bf16 copy, processed. U32/P32 are fp32 copies on the CPU (all four bridges on
#   the GPU did not fit in VRAM).

# %%


def open_model(name, LC, KVC):
    path = MODELS_DIR / name
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = load(path, torch.bfloat16, DEVICE)
    dbg = mdb.attach(model, tok)
    ad = dbg.adapter
    slot = cm.Slot("carrier", "carrier", "R_argmax", layers=(LC,), kv=(KVC,), rule_layer=LC,
                   rule_heads=cm.group_heads(ad, KVC))
    bridges = {"U16": tl.bridge(model, tok)}
    bridges["P16"] = tl.bridge(load(path, torch.bfloat16, DEVICE), tok, in_place=True)
    bridges["P16"].enable_compatibility_mode()
    bridges["U32"] = tl.bridge(load(path, torch.float32, "cpu"), tok, in_place=True)
    bridges["P32"] = tl.bridge(load(path, torch.float32, "cpu"), tok, in_place=True)
    bridges["P32"].enable_compatibility_mode()
    return dbg, slot, bridges


# %% [markdown]
# ## 3. The TransformerLens side, indexed by mdb's own selector planner
#
# An earlier version of this experiment indexed TL's 4-D `hook_z` (`batch, pos, head, d_head`) as
# if it were mdb's flat `(batch, pos, head*d_head)`: the wrong slice, silently. Here:
#
# * hook names come from `names.to_tl` (`L20.A` -> `blocks.20.attn.hook_z`, `L20.v` -> the `v`
#   hook), and `names.tl_hook(port, fn)` hands `fn` the value **in mdb's layout** (and refuses a
#   tensor of the wrong rank);
# * which columns `A[h3]` or `v[kv6]` means is decided by `breakpoints.plan`, the same planner
#   `Run.set("t@last.L20.A[h3]", 0)` uses on the mdb side. One selector, one index rule, both
#   tools.

# %%


def cols(dbg, label, selector):
    """The last-axis columns a selector picks on port ``label`` (mdb's planner)."""
    run = dbg.live
    axes = stop_axes(run.program, label)
    width = dbg.adapter.head_dim * (dbg.adapter.num_heads if label.endswith(".A")
                                    else dbg.adapter.num_kv_heads)
    pick = plan(axes, [None, None, width], parse_selector(selector),
                head_dim=dbg.adapter.head_dim)[-1]
    return torch.tensor(list(pick), dtype=torch.long)


def tl_arms(bridge_key, bridge, ids, plan_, crow, donor_v, *, batch=True):
    """Margins of [base, zero head h at L_c (H), at L_c+1 (H), V interchange] on a bridge: all
    arms in one batch, or one forward per arm (``batch=False``)."""
    H, LC, a, b = plan_["H"], plan_["LC"], plan_["a"], plan_["b"]
    n = 2 * H + 2
    tokens = torch.tensor([ids], device=plan_["device"][bridge_key])
    current = {"arms": list(range(n))}

    def zero(layer):
        first = 1 if layer == LC else 1 + H

        def fn(x, hook):                              # x in mdb layout (batch, pos, H*d_head)
            for j, arm in enumerate(current["arms"]):
                if first <= arm < first + H:
                    x[j, -1, plan_["A"][arm - first].to(x.device)] = 0
            return x
        return fn

    def vx(x, hook):
        for j, arm in enumerate(current["arms"]):
            if arm == n - 1:
                x[j, crow, plan_["v"].to(x.device)] = donor_v.to(x.device, x.dtype)
        return x

    hooks = [(names.to_tl(f"L{LC}.A"), names.tl_hook(f"L{LC}.A", zero(LC))),
             (names.to_tl(f"L{LC + 1}.A"), names.tl_hook(f"L{LC + 1}.A", zero(LC + 1))),
             (names.to_tl(f"L{LC}.v"), names.tl_hook(f"L{LC}.v", vx))]
    if batch:
        logits = bridge.run_with_hooks(tokens.expand(n, -1), fwd_hooks=hooks)[:, -1].float().cpu()
    else:
        rows = []
        for arm in range(n):
            current["arms"] = [arm]
            rows.append(bridge.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1].float().cpu())
        logits = torch.stack(rows)
    return (logits[:, a] - logits[:, b]).double().tolist()


def tl_donor_v(bridge_key, bridge, ids, plan_, dcrow):
    got = {}

    def grab(x, hook):
        got["v"] = x[0, dcrow, plan_["v"].to(x.device)].clone()
        return x
    tokens = torch.tensor([ids], device=plan_["device"][bridge_key])
    LC = plan_["LC"]
    bridge.run_with_hooks(tokens, fwd_hooks=[(names.to_tl(f"L{LC}.v"),
                                              names.tl_hook(f"L{LC}.v", grab))])
    return got["v"]


def tl_base(bridge, ids, a, b):
    """fp32 CPU leg: the base forward only (the pure-reparametrisation check)."""
    logits = bridge(torch.tensor([ids]))[:, -1].float().cpu()
    return float((logits[0, a] - logits[0, b]).double())


# %% [markdown]
# ## 4. The mdb side: `Branches` from one checkpoint, a donor `ValueRef`, sealed TL claims
#
# Per item (a plain Python loop):
#
# 1. **Donor.** Run the donor prompt, bind its carrier row with the slot, and take
#    `donor.ref("cache.L_c.v[kv g,row=r]")`: the value **and its origin pointer** (run, address,
#    sha256, cut). The donor is fixed by the panel, so no `Pool` rule is needed here.
# 2. **Carrier row** of the item (`cm.slot_rows`).
# 3. **TL legs** (U16 batched and batch 1, P16 batched, U32/P32 base).
# 4. **Seal** each P16 patch effect as a `supplied-forecast` record (library, version, processing,
#    hook map) *before* mdb measures the arm: the library's number is a claim, never a measurement.
# 5. **`dbg.branches(run, Margin(last, a, b), at="t<last>.L<L_c>.h")`**: one checkpoint, one arm
#    per head written as a selector edit `edit("t@last.L20.A[h3]", 0.0)`, and a self-patch null
#    (the head's own value written back: `dM` must be exactly 0). Every arm row carries `dM`, its
#    edit manifest, the shared `floor`, `effect_floors` and the `stability` band (stable above
#    `k = 4` floors, else kernel-sensitive).
# 6. **V interchange** on a run split at the carrier row (`prefill=crow`, so the edited `v` flows
#    through the carrier row's own later layers: sequence scope), as a second `Branches` that
#    shares the first one's floor: `edit("t<crow>.L_c.v[kv g]", donor value, origin=pointer)`.
# 7. **Judge** each sealed claim against mdb's native `dM` with `supplied.judge` on the item's
#    floor.
#
# **The floor.** `Branches` takes 2 spacings (`verdict.margin_floor`) at the *base* margin's larger
# `|logit|`. The original run took the largest `|logit|` over all of the item's arms. Each arm row
# records its consumer `magnitude`, so both rules come from the same rows; the comparison with the
# original run uses its rule (`item_floor`), and the base-rule floor is reported beside it.

# %%


def study_item(dbg, slot, bridges, item, plan_, sink_counts):
    LC, KVC, H = plan_["LC"], plan_["KVC"], plan_["H"]
    ids, last = item["ids"], len(item["ids"]) - 1
    donor = ITEMS[item["donor"]]
    a, b = item["answer"], donor["answer"]
    plan_.update(a=a, b=b)
    consumer = Margin(last, a, b)
    # 1. donor value with its origin
    drun = dbg.run(donor["ids"], name="donor")
    dlast = len(donor["ids"]) - 1
    drun.until(f"t{dlast}.logits")
    dcrow = cm.slot_rows(slot, drun, {"name": donor["id"], "span": donor["span"], "last": dlast})[0]
    dref = drun.ref(f"cache.L{LC}.v[kv{KVC},row={dcrow}]")
    # 2. the item's carrier row
    run = dbg.run(ids, name="main")
    run.until(f"t{last}.logits")
    crow = cm.slot_rows(slot, run, {"name": item["id"], "span": item["span"], "last": last})[0]
    # 3. TL legs
    tl_m = {}
    for key in ("U16", "P16"):
        dv = tl_donor_v(key, bridges[key], donor["ids"], plan_, dcrow)
        tl_m[key] = tl_arms(key, bridges[key], ids, plan_, crow, dv)
        if key == "U16":
            tl_m["U16_b1"] = tl_arms(key, bridges[key], ids, plan_, crow, dv, batch=False)
    for key in ("U32", "P32"):
        tl_m[key] = tl_base(bridges[key], ids, a, b)
    # 4-5. Branches for the head arms; seal the P16 claims at the branch point first
    br = dbg.branches(run, consumer, at=f"t{last}.L{LC}.h", name=f"heads:{item['id']}")
    arms, targets = [], []
    for layer in (LC, LC + 1):
        for h in range(H):
            arms.append(f"L{layer}.h{h}")
            targets.append(f"t@last.L{layer}.A[h{h}]")
    arms.append("V")
    targets.append(f"t{crow}.L{LC}.v[kv{KVC}]")
    records = []
    for k, (arm, target) in enumerate(zip(arms, targets), start=1):
        records.append(supplied.seal(
            run, target={"kind": "margin", "row": last, "indices": [a, b]},
            edit={"address": target, "op": "replace", "value": "0" if arm != "V" else
                  f"donor {donor['id']} cache.L{LC}.v[kv{KVC},row={dcrow}]"},
            value=tl_m["P16"][k] - tl_m["P16"][0], library="transformer-lens", version=VER,
            address_map={"A": names.to_tl(f"L{LC}.A"), "v": names.to_tl(f"L{LC}.v"),
                         "processing": json.dumps(plan_["processing"])},
            method="processed bridge (enable_compatibility_mode) patch effect, batched"))
    for arm, target in zip(arms[:-1], targets[:-1]):
        br.add(arm, [edit(target, 0.0)])
    br.add_null("self", "self", like=f"L{LC}.h{slot.rule_heads[0]}")
    br.run()
    # 6. V interchange in sequence scope, same floor
    vrun = dbg.run(ids, name="vx", prefill=crow)
    brv = dbg.branches(vrun, consumer, at=f"t{crow}.L{LC}.h", floor=br.floor,
                       name=f"v:{item['id']}")
    brv.add("V", [edit(targets[-1], dref.value.reshape(-1), origin=dict(dref.pointer))])
    brv.run()
    rows = [br.arm(arm).result for arm in arms[:-1]] + [brv.arm("V").result]
    native = [br.M0] + [r["M"]["value"] for r in rows]
    dM = [r["dM"]["value"] for r in rows]
    mag = max([verdict.magnitude(consumer, br.base_logits)] + [r["magnitude"]["value"] for r in rows])
    item_floor = 2 * verdict.spacing(mag, torch.bfloat16)
    # 7. judge the sealed TL claims on mdb's floor
    judged = [supplied.judge(rec, d, magnitude=mag, dtype=torch.bfloat16, kernel_path="mdb-stepped")
              for rec, d in zip(records, dM)]
    sink_counts["branches"] += [br.seal()["id"], brv.seal()["id"]]
    out = {"id": item["id"], "crow": crow, "donor_crow": dcrow, "a": a, "b": b,
           "mdb": native, "dM": dM, "V_dM_vs_main_base": brv.arm("V").result["M"]["value"] - br.M0,
           "V_base_split_delta": brv.M0 - br.M0,
           "floor_base_rule": br.floor, "item_floor": item_floor,
           "self_null_dM": br.arm("self").result["dM"]["value"],
           "stability": [r.get("stability") for r in rows],
           "judged": [{"record": j["record"], "native_effect": j["native_effect"],
                       "sign_agrees": j["sign_agrees"], "stability": j["stability"],
                       "error": j["error"]["value"]} for j in judged],
           "V_origin": brv.arm("V").result["edits"][0].get("origin"), **tl_m}
    del br, brv, run, vrun, drun
    return out


# %% [markdown]
# ## 5. Run both models (one at a time, freed in between)

# %%
custody = None
RECORDS = OUT / "records.jsonl"
sink = None
PER_MODEL = {}
for name, LC, KVC in MODELS:
    t_model = time.monotonic()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dbg, slot, bridges = open_model(name, LC, KVC)
    if custody is None:
        custody = dbg.custody(script=HERE / "tl_reparam_floor.py", prereg=HERE / "prereg.json",
                              kernel_path="mdb-stepped")
    if sink is None:
        sink = dbg.records_to(RECORDS)
    ad = dbg.adapter
    dbg.run(ITEMS[RUN_IDS[0]]["ids"], name="probe")          # a live run for the planner's axes
    plan_ = {"LC": LC, "KVC": KVC, "H": ad.num_heads,
             "A": [cols(dbg, f"L{LC}.A", f"h{h}") for h in range(ad.num_heads)],
             "v": cols(dbg, f"L{LC}.v", f"kv{KVC}"),
             "processing": {k: tl.processing(b) for k, b in bridges.items()},
             "device": {"U16": DEVICE, "P16": DEVICE, "U32": "cpu", "P32": "cpu"}}
    log(f"{name}: loaded; H={ad.num_heads} KV={ad.num_kv_heads} d_head={ad.head_dim}; carrier "
        f"L{LC} KV{KVC}, query heads {slot.rule_heads}")
    rows, counts = [], {"branches": []}
    for item_id in RUN_IDS:
        t_item = time.monotonic()
        rows.append(study_item(dbg, slot, bridges, ITEMS[item_id], plan_, counts))
        log(f"{name} {item_id} crow={rows[-1]['crow']} {time.monotonic() - t_item:.2f}s")
    PER_MODEL[name] = rows
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else None
    RESULT["models"][name] = {"carrier": [LC, KVC], "heads": ad.num_heads,
                              "query_heads_of_group": list(slot.rule_heads),
                              "processing": plan_["processing"], "tl": VER,
                              "hooks": {"A": [names.to_tl(f"L{LC}.A"), names.to_tl(f"L{LC + 1}.A")],
                                        "v": names.to_tl(f"L{LC}.v")},
                              "branches_records": len(counts["branches"]),
                              "seconds": round(time.monotonic() - t_model, 1),
                              "cuda_peak_allocated_mb": peak}
    log(f"{name}: done; cuda peak allocated {peak} MB")
    del dbg, slot, bridges
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
json.dump(PER_MODEL, open(OUT / "tl_reparam_floor_items.json", "w"), default=str)

# %% [markdown]
# ## 6. The effect-size table, from mdb's `stability.calibrate`
#
# `stability.calibrate` takes `(reference effect, other-path effect, floor)` triples and returns,
# per bin of `|reference| / floor` (1-1.5, 1.5-2, 2-4, > 4), how many reference verdicts (sign
# above the floor) are **lost** on the other path (sign reversed, or at or below the floor), how
# many of those reversed sign, and `k`: the smallest margin in floors above which none was lost.
# Four path pairs, each on the item floor:
#
# | path pair | reference | other |
# |---|---|---|
# | total | mdb stepped | TL processed, batched |
# | kernel | mdb stepped | TL unprocessed, batched |
# | batch | TL unprocessed, batch 1 | TL unprocessed, batched |
# | processing | TL unprocessed, batched | TL processed, batched |
#
# The same function scores the original run's saved margins on the same items.

# %%
PAIRS = {"total": ("mdb", "P16"), "kernel": ("mdb", "U16"), "batch": ("U16_b1", "U16"),
         "processing": ("U16", "P16")}
PATHS = {"mdb": "mdb-stepped", "P16": "tl-processed-batched", "U16": "tl-unprocessed-batched",
         "U16_b1": "tl-unprocessed-batch1"}


def effects(r, key):
    if key == "mdb" and "dM" in r:
        return r["dM"]
    return [m - r[key][0] for m in r[key][1:]]


def floor_of(r):
    return r["item_floor"] if "item_floor" in r else 2 * verdict.spacing(r["max_abs_logit"],
                                                                         torch.bfloat16)


def tables(rows, model):
    out = {}
    for pair, (ref_key, other_key) in PAIRS.items():
        triples = [(x, y, floor_of(r)) for r in rows
                   for x, y in zip(effects(r, ref_key), effects(r, other_key))]
        out[pair] = stability.calibrate(triples, model=model,
                                        paths=[PATHS[ref_key], PATHS[other_key]],
                                        source="examples/02-tl-reparametrisation-floor").to_json()
    return out


def show(model, now, then):
    print(f"\n{model}: |effect| / floor bins, lost (sign reversed) -- now | showcase, same items")
    print(f"  {'path':<11}" + "".join(f"{b['bin']:>22}" for b in now["total"]["table"])
          + f"{'all':>16}{'k':>8}")
    for pair in PAIRS:
        cells = "".join("{:>22}".format(f"{x['lost']}/{x['n']} ({x['sign_reversed']}) | "
                                        f"{y['lost']}/{y['n']}")
                        for x, y in zip(now[pair]["table"], then[pair]["table"]))
        print(f"  {pair:<11}{cells}{now[pair]['lost']:>6}/{now[pair]['above_floor']} | "
              f"{then[pair]['lost']}/{then[pair]['above_floor']}{now[pair]['k']:>7.2f}")


SUMMARY = {}
for name, *_ in MODELS:
    rows = PER_MODEL[name]
    ref_rows = [REF["models"][name][r["id"]] for r in rows]
    now, then = tables(rows, name), tables(ref_rows, name)
    show(name, now, then)
    above4 = {p: sum(b["lost"] for b in now[p]["table"] if b["bin"] == ">4") for p in PAIRS}
    n4 = next(b["n"] for b in now["total"]["table"] if b["bin"] == ">4")
    judged = [j for r in rows for j in r["judged"]]
    s = {
        "tables_now": now, "tables_showcase": then,
        "above_4_floors": {"n": n4, "lost_any_path": sum(above4.values())},
        "sign_reversed_total": f"{now['total']['sign_reversed']}/{now['total']['above_floor']}",
        "sign_reversed_kernel": f"{now['kernel']['sign_reversed']}/{now['kernel']['above_floor']}",
        "lost_kernel_vs_processing": [now["kernel"]["lost"], now["processing"]["lost"]],
        "fp32_max_abs_P32_minus_U32": max(abs(r["P32"] - r["U32"]) for r in rows),
        "fp32_max_abs_showcase": max(abs(r["P32"][0] - r["U32"][0]) for r in ref_rows),
        "judged_native_effect": sum(j["native_effect"] for j in judged),
        "judged_sign_disagrees": sum(j["sign_agrees"] is False for j in judged),
        "judged_stable": sum(j["stability"] == "stable" for j in judged),
        "self_null_max_abs_dM": max(abs(r["self_null_dM"]) for r in rows),
        "items_floor_rules_differ": sum(r["item_floor"] != r["floor_base_rule"] for r in rows),
        "V_split_base_delta_max": max(abs(r["V_base_split_delta"]) for r in rows),
        "memo_null_U16_b1_base_bitwise_eq_mdb": sum(r["U16_b1"][0] == r["mdb"][0] for r in rows),
    }
    print(f"  above 4 floors: {n4} effects, lost on any path: {s['above_4_floors']['lost_any_path']}"
          f"; sign reversals total {s['sign_reversed_total']}, kernel {s['sign_reversed_kernel']}")
    print(f"  fp32 pure reparametrisation: max |P32 - U32| = {s['fp32_max_abs_P32_minus_U32']:.3g}"
          f" (showcase on these items {s['fp32_max_abs_showcase']:.3g})")
    print(f"  sealed TL claims judged: {s['judged_native_effect']} native effects, sign disagrees "
          f"{s['judged_sign_disagrees']}, stable (> 4 floors) {s['judged_stable']}")
    print(f"  self-patch null max |dM| {s['self_null_max_abs_dM']}; items whose base-rule floor != "
          f"item floor: {s['items_floor_rules_differ']}/{len(rows)}")
    SUMMARY[name] = s
RESULT["summary"] = SUMMARY

# %% [markdown]
# ## 7. Original run vs now on the same items
#
# The native arms are the same stepped edits from the same checkpoint as the original
# hand-written loop, so the margins should be **bitwise** equal; the TL legs run the same library
# version on the same batches. The V arm's margin is compared directly (both from a run split at
# the carrier row); its `dM` here is against that run's own base (`Branches`), where the original run
# subtracted the unsplit run's base: the two bases differ by the split's kernel path
# (`V_split_base_delta_max`).

# %%
checks = []
for name, *_ in MODELS:
    rows = PER_MODEL[name]
    ref = REF["models"][name]
    rr = [ref[r["id"]] for r in rows]
    checks.append((f"{name}: carrier rows equal (item and donor)",
                   all(r["crow"] == q["crow"] and r["donor_crow"] == q["donor_crow"]
                       for r, q in zip(rows, rr))))
    diff = max(abs(x - y) for r, q in zip(rows, rr) for x, y in zip(r["mdb"], q["mdb"]))
    checks.append((f"{name}: mdb native margins equal the showcase's (max |diff| {diff:g})",
                   diff == 0.0))
    for key in ("U16", "P16", "U16_b1"):
        d = max(abs(x - y) for r, q in zip(rows, rr) for x, y in zip(r[key], q[key]))
        checks.append((f"{name}: TL {key} margins equal the showcase's (max |diff| {d:g})", d == 0.0))
    d32 = max(abs(r[k] - q[k][0]) for r, q in zip(rows, rr) for k in ("U32", "P32"))
    checks.append((f"{name}: fp32 base margins within 1e-4 of the showcase's ({d32:.2g})", d32 <= 1e-4))
    s = SUMMARY[name]
    same = all(x["n"] == y["n"] and x["lost"] == y["lost"]
               for p in PAIRS for x, y in zip(s["tables_now"][p]["table"], s["tables_showcase"][p]["table"]))
    checks.append((f"{name}: loss-by-bin tables equal the showcase's on these items", same))
    checks.append((f"{name}: 0 verdicts lost above 4 floors on any path",
                   s["above_4_floors"]["lost_any_path"] == 0))
    checks.append((f"{name}: fp32 processing is a pure reparametrisation (max |P32-U32| <= 1e-4)",
                   s["fp32_max_abs_P32_minus_U32"] <= 1e-4))
    checks.append((f"{name}: self-patch null dM == 0", s["self_null_max_abs_dM"] == 0.0))
    checks.append((f"{name}: judged native effects == above-floor count",
                   s["judged_native_effect"] == s["tables_now"]["total"]["above_floor"]))
q3, q25 = SUMMARY.get("Qwen3-0.6B-Base"), SUMMARY.get("Qwen2.5-0.5B")
if q3 and q25:
    checks.append(("kernel path dominates on Qwen3 (kernel lost >= processing lost)",
                   q3["lost_kernel_vs_processing"][0] >= q3["lost_kernel_vs_processing"][1]))
    checks.append(("processing dominates on Qwen2.5 (processing lost > kernel lost)",
                   q25["lost_kernel_vs_processing"][1] > q25["lost_kernel_vs_processing"][0]))
RESULT["checks"] = {n: bool(ok) for n, ok in checks}
for n, ok in checks:
    print("PASS" if ok else "DIFF", n)
print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks pass")

# %% [markdown]
# ## 8. The records
#
# Every `branches` record (one per head set and one per V interchange, per item) and every
# `supplied-forecast` record went to `records.jsonl` with custody; `sink.records()` reads them back
# and verifies each one.

# %%
back = sink.records()
kinds = {}
for rec in back:
    kinds[rec["kind"]] = kinds.get(rec["kind"], 0) + 1
custody_body = back[0]["body"].get("custody", {}) if back else {}
RESULT["records"] = {"path": str(RECORDS), "kinds": kinds,
                     "custody_script_sha256": custody_body.get("script_sha256"),
                     "custody_prereg_sha256": custody_body.get("prereg_sha256")}
print("records verified:", kinds)
RESULT["seconds"] = round(time.monotonic() - T0, 1)
json.dump(RESULT, open(OUT / "tl_reparam_floor_results.json", "w"), indent=1, default=str)
print(f"done in {RESULT['seconds']}s; wrote {OUT}/tl_reparam_floor_results.json")
print("RESULT_JSON " + json.dumps({k: v for k, v in RESULT.items() if k != "summary"}
                                  | {"summary": {m: {k: v for k, v in s.items()
                                                     if not k.startswith("tables")}
                                                 for m, s in SUMMARY.items()}}, default=str))
