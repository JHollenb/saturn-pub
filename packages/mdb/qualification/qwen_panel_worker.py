"""Generic Qwen panel worker: sealed ``attention_panels`` plans on one resident Qwen2.5-1.5B.

Runs, per declared numerical lane (``float32`` / ``bfloat16``; one model resident at a time, the
model is loaded once per lane and freed before the next):

* value plans (``family="value"``): every arm through ``attention_panels.value_spec`` + ``blend``
  on the KV cache values of the declared cells (singleton, complement, coalition, dose, wrong
  target, wrong payload, reciprocal, sham, replay, native references);
* read plans (``family="read"``): every arm through ``AttentionPort(QwenBackend(model))`` with
  ``port_cuts(arm, panel=..., port=...)`` (cut, keep, both cut types, ``floor_arms`` floors,
  random and other-item controls, all-cut and native references).

Refuses: a plan whose recomputed sha256 differs from the config or from its seal, a plan built for
another lane, non-eager attention (the port refuses), a live geometry other than the plan's
(``check_universe`` / ``check_value_geometry``), a tokenization other than the saved case, and a
checkpoint whose sha256 differs. No retries. Writes per plan: ``rows.jsonl`` (one row per arm:
tracked logits, margin hot - cold with its rounding bound, greedy first three tokens, top-k,
distances to the reference arm, port metadata), ``logits.npy`` (full first-position logits as
FP32 for value plans; read plans keep top-k + tracked tokens only), ``recipes.jsonl``
(regeneration recipe per arm) and ``results.json`` (read plans: the ``attention_panels``
results record). Prints the sha256 and file of every shared module it imported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import time
from pathlib import Path

TOP_K = 20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def worker():
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    job = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "local"))
    root = Path(config["output_scratch"]) / job / "qwen-panels"
    root.mkdir(parents=True, exist_ok=False)
    save(root / "config.json", config)
    started = time.monotonic()
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("QWENPANEL " + json.dumps(row, sort_keys=True, default=str), flush=True)

    import numpy as np
    import torch
    import mdb
    import mdb.analysis_metrics as am
    import mdb.attention_panels as P
    import mdb.attention_port as ap
    import mdb.attention_ranking as ar
    import mdb.native_replay_recipe as nrr
    import mdb.qwen_panels as Q
    modules = {m.__name__: {"file": m.__file__, "sha256": sha(m.__file__)}
               for m in (mdb, ap, P, ar, am, Q, nrr)}
    expected = config.get("module_sha256") or {}
    event("modules", modules=modules)
    bad = {k: (modules[k]["sha256"], v) for k, v in expected.items() if modules.get(k, {}).get("sha256") != v}
    if bad:
        raise RuntimeError(f"staged module bytes differ from the staging checkout: {bad}")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device(config.get("device", "cuda"))
    cuda = dev.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("admitted CUDA required")
    if not cuda and not config.get("dry_run"):
        raise RuntimeError("only a dry run may use a device other than CUDA")

    weight_file = Path(config["model_path"]) / "model.safetensors"
    event("weights_hash_start", file=str(weight_file))
    weight_sha = sha(weight_file)
    if weight_sha != config["expected_weight_file_sha256"]:
        raise RuntimeError("Qwen model.safetensors differs from the case's checkpoint identity")
    event("weights_hash_done", sha256=weight_sha)

    from transformers import DynamicCache
    from mdb.ar.model import load_ar_model

    lanes_out = {}
    order = [lane["dtype"] for lane in config["lanes"]]
    if order not in (["float32", "bfloat16"], ["float32"], ["bfloat16"]):
        raise RuntimeError("lanes run float32 first, then bfloat16 (one load; parameters cast down in place)")
    # ONE model load (FP32 from the BF16 checkpoint: exact). The BF16 lane casts the parameters in
    # place (FP32 -> BF16 of BF16-representable values: exact) and leaves buffers as loaded, as
    # from_pretrained(torch_dtype=bfloat16) does in transformers 5.14.1 (rotary inv_freq stays FP32;
    # MEASURED on a tiny Qwen2 checkpoint). One resident model throughout.
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    event("load_start")
    spec, model, tokenizer = load_ar_model(config["model"], device=dev.type, torch_dtype=torch.float32,
                                           attn_implementation="eager")
    model.eval().requires_grad_(False)
    if spec.family != "qwen2" or model.config._attn_implementation != "eager":
        raise RuntimeError("eager Qwen2 required")
    event("load_done", registry=spec.registry_name, layers=len(model.model.layers),
          heads=model.config.num_attention_heads, kv_heads=model.config.num_key_value_heads,
          buffers={n: str(b.dtype) for n, b in model.named_buffers()})

    def run_lane(lane):
        dtype_name = lane["dtype"]
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
        if next(model.parameters()).dtype != dtype:
            with torch.no_grad():
                for param in model.parameters():
                    param.data = param.data.to(dtype)
            if cuda:
                torch.cuda.empty_cache()
        if {p.dtype for p in model.parameters()} != {dtype}:
            raise RuntimeError("model parameters are not in the declared lane dtype")
        inv_freq = getattr(model.model.rotary_emb, "inv_freq", None)
        event("lane_start", lane=dtype_name, inv_freq_dtype=None if inv_freq is None else str(inv_freq.dtype),
              vram_allocated_mb=torch.cuda.memory_allocated() / 2 ** 20 if cuda else None)
        ids = {side: tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
               for side, text in (("recipient", Q.RECIPIENT_TEXT), ("donor", Q.DONOR_TEXT))}
        if ids["recipient"] != Q.RECIPIENT_IDS or ids["donor"] != Q.DONOR_IDS:
            raise RuntimeError("tokenization differs from the saved case")
        for word, token in Q.TRACKED.items():
            if tokenizer(" " + word, add_special_tokens=False)["input_ids"] != [token]:
                raise RuntimeError(f"tracked token ' {word}' is not the single id {token}")
        identity = {"model": config["model"], "weight_file_sha256": weight_sha, "lane": dtype_name}

        def prefill(tokens):
            """Rows 0..18 in one forward, rows 19, 20 one forward each (main-02 order)."""
            cache = DynamicCache(config=model.config)
            t = torch.tensor([tokens], device=dev)
            with torch.inference_mode():
                model(input_ids=t[:, :Q.SOURCE_ROW + 1], past_key_values=cache, use_cache=True)
                for j in range(Q.SOURCE_ROW + 1, len(tokens) - 1):
                    model(input_ids=t[:, j:j + 1], past_key_values=cache, use_cache=True)
            if cache.get_seq_length() != len(tokens) - 1:
                raise RuntimeError("prefill cache length differs")
            return cache

        parents = {side: prefill(ids[side]) for side in ids}
        parent_values = {side: [layer.values.clone() for layer in parents[side].layers] for side in parents}

        def generate(cache, first):
            """Three greedy forwards from row 21; returns (first-position logits, tokens, dtype)."""
            tokens, first_logits, produced = [], None, None
            token = first
            with torch.inference_mode():
                for step in range(Q.GENERATED):
                    out = model(input_ids=torch.tensor([[token]], device=dev), past_key_values=cache, use_cache=True)
                    logits = out.logits[0, -1]
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("nonfinite logits")
                    if step == 0:
                        first_logits, produced = logits.float().cpu(), str(logits.dtype).replace("torch.", "")
                    token = int(logits.argmax())
                    tokens.append(token)
            return first_logits, tokens, produced

        def record(arm, logits, tokens, produced, reference_logits):
            tracked = {w: float(logits[t]) for w, t in Q.TRACKED.items()}
            m = Q.bounded_margin(arm.arm, tracked, produced)
            top = torch.topk(logits, TOP_K)
            rank = {w: int((logits > logits[t]).sum()) for w, t in Q.TRACKED.items()}
            row = {"arm": arm.arm, "family": arm.family, "kind": arm.kind, "role": arm.role,
                   "reference": arm.reference, "tracked": tracked, "tracked_rank": rank,
                   "margin": m.value, "margin_bound": m.bound, "margin_resolved": m.resolved,
                   "logits_dtype": produced, "tokens": tokens, "tokens_text": [tokenizer.decode([t]) for t in tokens],
                   "topk": [[int(i), float(v)] for v, i in zip(top.values.tolist(), top.indices.tolist())]}
            if reference_logits is not None:
                d = logits - reference_logits
                p, q = torch.log_softmax(logits.double(), -1), torch.log_softmax(reference_logits.double(), -1)
                row["to_reference"] = {"max_abs_logit": float(d.abs().max()),
                                       "kl_reference_to_arm": float((q.exp() * (q - p)).sum())}
            return row

        observe_port = ap.AttentionPort(ap.QwenBackend(model, identity=identity))

        def receiver_probs(read):
            return read.probs()[0, :, -1, :].double().cpu().numpy()

        obs = ap.observe(receiver_probs, steps={0}, name="receiver_row_probs")
        with observe_port.apply(observers=[obs], forwards=Q.GENERATED):
            native_logits, native_tokens, produced = generate(copy.deepcopy(parents["recipient"]), ids["recipient"][-1])
        attention = obs.stack()[0]                                        # [layers, heads, 22]
        lane_root = root / dtype_name
        lane_root.mkdir()
        save(lane_root / "receiver-attention.json", {
            "what": "FP32 softmax recomputed from the native q/k (attention_port.Read.probs) of row 21, step 0, "
                    "native recipient; [layer][head][source]",
            "lane": dtype_name, "attention": attention.tolist(),
            "native_tokens": native_tokens, "native_tokens_text": [tokenizer.decode([t]) for t in native_tokens]})
        event("observed", lane=dtype_name, tokens=native_tokens,
              max_source_attention={k: round(v, 4) for k, v in Q.max_source_attention(attention).items()})

        lane_summary = {"plans": {}, "identity": identity, "ids": {k: list(v) for k, v in ids.items()}}
        adapter = None
        value_logits = {}

        def native_recipe(arm, entry, inexpressible):
            """Seal a native_replay_recipe for the arm when its kind is expressible there."""
            nonlocal adapter
            if not config.get("native_recipes", True):
                return
            if adapter is None:
                from mdb.prompt_qwen import PromptQwenAdapter
                event("native_adapter_start", lane=dtype_name)
                adapter = PromptQwenAdapter(model)      # hashes the parameters for its model identity
                event("native_adapter_done", lane=dtype_name, model_identity=adapter.model_identity)
            body, why = Q.native_recipe(arm, model_identity=adapter.model_identity, execution=adapter.execution)
            if body is not None:
                entry["native_replay_recipe"] = nrr.seal(body)
            else:
                inexpressible.setdefault(arm.kind, why)
        for item in lane["plans"]:
            plan_raw = (payload / item["file"]).read_bytes()
            panel = P.Panel.from_json(json.loads(plan_raw))
            seal = json.loads((payload / item["seal"]).read_text())
            if panel.sha256() != item["plan_sha256"]:
                raise RuntimeError(f"{item['file']}: plan sha256 {panel.sha256()} differs from the config")
            P.verify_seal(panel, seal)
            if panel.port.get("lane") != dtype_name:
                raise RuntimeError(f"{item['file']} was built for lane {panel.port.get('lane')!r}, not {dtype_name!r}")
            only = item.get("arms")
            arms = [a for a in panel.arms if only is None or a.arm in only]
            if only is not None and len(arms) != len(only):
                raise RuntimeError("config names arms that are not in the plan")
            lease = item.get("lease")
            plan_root = lane_root / (item["name"] + ("" if lease is None else f"-lease{lease['index']}"))
            plan_root.mkdir()
            rows, logits_store, recipes, kept = [], [], [], {}
            inexpressible = {}
            t0 = time.monotonic()
            if panel.family == "value":
                Q.check_value_geometry(panel, len(model.model.layers), model.config.num_key_value_heads)
                ref_logits = {}
                for arm in arms:
                    spec_v = Q.resolve_value(arm)
                    side = P.run_side(arm)
                    cache = copy.deepcopy(parents[side])
                    if spec_v is not None:
                        for layer, kv in spec_v["cells"]:
                            values = cache.layers[layer].values
                            before = values[:, kv, spec_v["target_row"]]
                            payload_row = parent_values[spec_v["payload_side"]][layer][:, kv, spec_v["payload_row"]]
                            values[:, kv, spec_v["target_row"]] = P.blend(before, payload_row, spec_v["dose"])
                    logits, tokens, produced = generate(cache, ids[side][-1])
                    reference = ref_logits.get(arm.reference)
                    row = record(arm, logits, tokens, produced, reference)
                    row.update(run_side=side, edit=spec_v)
                    if arm.kind in ("native", "donor_native"):
                        ref_logits[arm.arm] = logits
                    rows.append(row)
                    logits_store.append(logits.numpy())
                    recipes.append({"plain": Q.plain_recipe(panel, arm, lane=dtype_name)})
                    native_recipe(arm, recipes[-1], inexpressible)
                    kept[arm.arm] = logits
                value_logits.update(kept)
                np.save(plan_root / "logits.npy", np.stack(logits_store).astype(np.float32))
                save(plan_root / "logits-index.json", {"rows": [r["arm"] for r in rows], "dtype": "float32",
                                                       "what": "first generated position, full vocabulary",
                                                       "produced_in": produced})
            else:
                rows_table = Q.row_table(sorted({a.rows_item for a in panel.arms if a.rows_item}))
                port = ap.AttentionPort(ap.QwenBackend(model, identity=identity), **P.port_kwargs(panel, rows_table))
                P.check_port(panel, port, forwards=Q.GENERATED)
                panel.check_universe(len(model.model.layers), model.config.num_attention_heads)
                if lease is None:
                    order = [b for lz in panel.leases(cap=max(panel.branch_count, 1)) for b in lz["branches"]]
                    copies = set()
                else:
                    # One lease of a plan split across jobs: recompute it and refuse a different lease.
                    leases = panel.leases(cap=lease["cap"])
                    chosen = leases[lease["index"]]
                    if chosen["lease_sha256"] != lease["lease_sha256"] or len(leases) != lease["of"]:
                        raise RuntimeError("lease differs from the staged lease (sha256 recomputed)")
                    order = chosen["branches"]
                    copies = {b for _, b in chosen["replicated"]}
                    if not any(b == "native" for _, b in order):
                        order = [[Q.CONTEXT, "native"]] + order        # every lease scores against its own native
                        copies.add("native")
                seen = set()
                for _, name in order:
                    if name in seen or (only is not None and name not in only):
                        continue
                    seen.add(name)
                    arm = panel.by_id()[name]
                    cuts = P.port_cuts(arm, panel=panel, port=port)
                    with port.apply(*cuts, forwards=Q.GENERATED, log=True):
                        logits, tokens, produced = generate(copy.deepcopy(parents["recipient"]), ids["recipient"][-1])
                    meta = port.metadata()
                    meta["emptied"] = list(port.emptied)
                    meta["removed_reads"] = sum(g["removed_reads"] for entry in port.log for g in entry["groups"])
                    reference = kept.get(arm.reference)
                    row = record(arm, logits, tokens, produced, reference)
                    row.update(port=meta, unit_set=arm.unit_set, item=arm.item, cut_type=arm.cut_type,
                               control=arm.control, claim=arm.claim, floor_of=arm.floor_of,
                               replicated=name in copies, lease=lease)
                    rows.append(row)
                    kept[name] = logits
                    recipes.append({"plain": Q.plain_recipe(panel, arm, lane=dtype_name, cuts=cuts,
                                                            port_kwargs=P.port_kwargs(panel, rows_table))})
                    native_recipe(arm, recipes[-1], inexpressible)
                pair_logits = {}
                for key in P.twin_pairs(panel):
                    a, b = key.split("|")
                    if a in kept and b in kept:
                        pair_logits[key] = float((kept[a] - kept[b]).abs().max())
                results = Q.read_results(panel, rows)    # within this lease (copies included)
                save(plan_root / "results.json", results)
                save(plan_root / "pair-max-abs-logit.json", pair_logits)
            with (plan_root / "rows.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            with (plan_root / "recipes.jsonl").open("x") as stream:
                for r in recipes:
                    stream.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
            save(plan_root / "plan-identity.json", {"plan_sha256": panel.sha256(), "seal_sha256": seal["seal_sha256"],
                                                   "file": item["file"], "arms_run": len(rows),
                                                   "arms_in_plan": len(panel.arms),
                                                   "inexpressible_in_native_replay_recipe": inexpressible})
            lane_summary["plans"][item["name"]] = {"arms": len(rows), "seconds": round(time.monotonic() - t0, 3),
                                                   "plan_sha256": panel.sha256(), "inexpressible": inexpressible}
            event("plan_done", lane=dtype_name, plan=item["name"], arms=len(rows),
                  seconds=round(time.monotonic() - t0, 3))

        replays = []
        for name in lane.get("replay_check", ()):
            # Smoke only: regenerate an arm through mdb.native_replay_recipe and compare logits.
            from saturn_pub import StateAddress
            vp = next(P.Panel.from_json(json.loads((payload / i["file"]).read_bytes()))
                      for i in lane["plans"] if i["family"] == "value")
            arm = vp.by_id()[name]
            body, why = Q.native_recipe(arm, model_identity=adapter.model_identity, execution=adapter.execution)
            t1 = time.monotonic()
            replay = nrr.NativeReplay(nrr.seal(body), adapter).run()
            native = replay.materialize("readout-0", StateAddress("logits")).float().reshape(-1).cpu()
            hf = value_logits[name]
            replay_tokens = replay.sessions[P.run_side(arm)].read("tokens").tolist()[0][-Q.GENERATED:]
            replays.append({"arm": name, "seconds": round(time.monotonic() - t1, 3),
                            "max_abs_logit_vs_worker": float((native - hf).abs().max()),
                            "margin_native_replay": float(native[Q.TRACKED["hot"]] - native[Q.TRACKED["cold"]]),
                            "margin_worker": float(hf[Q.TRACKED["hot"]] - hf[Q.TRACKED["cold"]]),
                            "tokens_native_replay": replay_tokens})
            event("replay_check", lane=dtype_name, **replays[-1])
        lane_summary["replay_check"] = replays
        lane_summary["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None
        lane_summary["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20 if cuda else None
        event("lane_done", lane=dtype_name, peak_vram_allocated_mb=lane_summary["peak_vram_allocated_mb"],
              peak_vram_reserved_mb=lane_summary["peak_vram_reserved_mb"])
        return lane_summary

    import gc
    for lane in config["lanes"]:
        lanes_out[lane["dtype"]] = run_lane(lane)      # caches, ports and adapters are local to run_lane
        gc.collect()
    summary = {"schema": "mdb-qwen-panels-run-v1", "job_id": job, "request_sha256": request.get("request_sha256"),
               "worker_sha256": sha(payload / "qwen_panel_worker.py"), "modules": modules,
               "torch": torch.__version__, "lanes": lanes_out,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "wall_s": round(time.monotonic() - started, 3), "timeline": timeline,
               "storage": {"value plans": "full first-position logits as FP32 (logits.npy)",
                           "read plans": f"top-{TOP_K} + tracked tokens (' cold', ' hot', ' boiling') per arm"}}
    import transformers
    summary["transformers"] = transformers.__version__
    save(root / "summary.json", summary)
    print("QWENPANEL_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "timeline"},
                                            sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    worker()
