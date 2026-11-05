"""Finite native joint-component validation of raw-state attention predictions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import time


def worker():
    payload = Path.cwd()
    L = runpy.run_path(str(payload / "flux_attention_state_law.py"))
    S = runpy.run_path(str(payload / "flux_mlp_state_law.py"))
    read, sha, save = S["read"], S["sha"], S["save"]
    config, request = [read(os.environ[k]) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST")]
    if config != request["config"] or any(sha(payload / n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("sealed payload/config differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    import numpy as np
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    from safetensors import safe_open
    from saturn_pub import Adapter, Session
    from saturn_pub.contracts import ExecutionPoint
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.native_navigation import NativeCutNavigator
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(exist_ok=False, parents=True)
    os.chdir(scratch)
    out = Path("binding-minimal-result")
    out.mkdir()
    for name in ("plan.json", "parameter-artifacts.json", "retrieve_weights.py"):
        shutil.copyfile(payload / name, out / name)
    save(out / "request.json", request)
    checkpoint = Path(config["model_path"]) / config["checkpoint_file"]
    if sha(checkpoint) != config["checkpoint_file_sha256"]:
        raise ValueError("complete native checkpoint file identity differs")
    state_dict, weights, parameter_descriptors = {}, {}, {}
    unused = {"attn.add_q_proj.weight", "attn.norm_added_q.weight", "attn.to_add_out.weight",
              "ff_context.linear_in.weight", "ff_context.linear_out.weight"}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as bank:
        for record in read(out / "parameter-artifacts.json")["artifacts"]:
            source = record["source"]
            tensor = bank.get_tensor(source["tensor"])
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            if list(tensor.shape) != source["shape"] or tensor.dtype != torch.bfloat16 or hashlib.sha256(raw).hexdigest() != source["payload_sha256"]:
                raise ValueError("named original parameter recipe differs")
            name = source["tensor"].removeprefix("transformer_blocks.1.")
            state_dict[name] = tensor.clone().contiguous()
            parameter_descriptors[name] = describe(state_dict[name])
            if name not in unused:
                weights[name] = tensor.float().numpy()
                weights[name].flags.writeable = False
    del tensor, raw
    with torch.device("meta"):
        block = native.Flux2TransformerBlock(dim=3072, num_attention_heads=24,
            attention_head_dim=128, mlp_ratio=3., eps=1e-6, bias=False)
    block.load_state_dict(state_dict, strict=True, assign=True)
    block.eval().requires_grad_(False)
    if not hasattr(block.ff.act_fn, "gate_fn") or type(block.attn.processor).__name__ != "Flux2AttnProcessor":
        raise ValueError("qualified native arithmetic layout differs")
    save(out / "native-source.json", dict(diffusers_source_sha256=sha(Path(native.__file__)),
        parameters=parameter_descriptors, scope="Original Diffusers joint1 component, CPU BF16; no ATen replay claim."))
    arrays = {}
    for name, record in config["input_files"].items():
        path = payload / (name + ".npy")
        if sha(path) != record["sha256"]:
            raise ValueError("source input file differs")
        arrays[name] = np.load(path, allow_pickle=False)
    common = {k: arrays[k] for k in ("image", "img_mod", "txt_mod", "cos", "sin")}
    texts = dict(base=arrays["text_base"], changed=arrays["text_changed"],
        midpoint=S["quantize"]((arrays["text_base"].astype(np.float64)+arrays["text_changed"])/2))
    states = {name: dict(common, text=text) for name, text in texts.items()}
    counts = dict.fromkeys(config["expected_counts"], 0)
    counts["component_loads"] = 1
    active = dict(name=None, fields=None, ports=None)
    ix, txix = config["indices"], config["text_indices"]
    handles, receipts = [], []
    original_dispatch = native.dispatch_attention_fn

    def wall():
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise RuntimeError("finite worker wall budget")

    def event(phase, **fields):
        print("MDB_ATTENTION_STATE " + json.dumps(dict(phase=phase, counts=counts.copy(), seconds=time.monotonic()-started, **fields), sort_keys=True), flush=True)

    def count(key):
        def hook(module, args):
            counts[key] += 1
        return hook

    def observe(field, value, port):
        if active["fields"] is None:
            raise ValueError("unscoped native port observation")
        active["fields"][field] = value.detach().float().cpu().numpy().copy()
        active["ports"][field] = dict(port=port, descriptor=describe(value))

    def return_hook(field, selector, port):
        def hook(module, args, output):
            observe(field, selector(output), port)
            return output
        return hook

    def attn_input(module, args, kwargs):
        observe("norm_image", kwargs["hidden_states"][:, ix], "attn:input.hidden_states[selected-image-rows]")
        observe("norm_text", kwargs["encoder_hidden_states"][:, txix], "attn:input.encoder_hidden_states[selected-text-rows]")

    def dispatch(q, k, v, *args, **kwargs):
        counts["native_sdpa_in_block"] += 1
        image_rows = [512+i for i in ix]
        observe("q", q[:, image_rows].permute(0, 2, 1, 3), "native.dispatch_attention_fn:input.q[positioned,image-queries]")
        observe("k", k.permute(0, 2, 1, 3), "native.dispatch_attention_fn:input.k[positioned,all-keys]")
        observe("v", v.permute(0, 2, 1, 3), "native.dispatch_attention_fn:input.v[all-values]")
        result = original_dispatch(q, k, v, *args, **kwargs)
        observe("head", result[:, image_rows].permute(0, 2, 1, 3), "native.dispatch_attention_fn:return[image-queries]")
        return result

    handles.extend([block.register_forward_pre_hook(count("native_blocks")),
        block.attn.register_forward_pre_hook(count("native_attention")),
        block.ff.register_forward_pre_hook(count("native_image_ff")),
        block.ff_context.register_forward_pre_hook(count("native_text_ff")),
        block.attn.register_forward_pre_hook(attn_input, with_kwargs=True)])
    handles.extend(m.register_forward_pre_hook(count("native_linears")) for m in block.modules() if isinstance(m, torch.nn.Linear))
    for field, module, selector in (
        ("q_raw", block.attn.to_q, lambda a: a[:, ix].reshape(1, len(ix), 24, 128)),
        ("image_k_raw", block.attn.to_k, lambda a: a[:, ix].reshape(1, len(ix), 24, 128)),
        ("text_k_raw", block.attn.add_k_proj, lambda a: a[:, txix].reshape(1, len(txix), 24, 128)),
        ("q_norm", block.attn.norm_q, lambda a: a[:, ix]),
        ("image_k_norm", block.attn.norm_k, lambda a: a[:, ix]),
        ("text_k_norm", block.attn.norm_added_k, lambda a: a[:, txix]),
        ("attention", block.attn, lambda a: a[0][:, ix]),
        ("projected", block.ff.linear_in, lambda a: a[:, ix]),
        ("activated", block.ff.act_fn.gate_fn, lambda a: a[:, ix]),
        ("product", block.ff.act_fn, lambda a: a[:, ix]),
        ("mlp", block.ff, lambda a: a[:, ix])):
        path = next(n for n, m in block.named_modules() if m is module)
        handles.append(module.register_forward_hook(return_hook(field, selector, path+":return[selected-rows]")))
    handles.append(block.ff.register_forward_pre_hook(lambda module, args: observe("mlp_input", args[0][:, ix], "ff:input[selected-image-rows]")))

    class JointComponent(Adapter):
        model_identity = config["model"]
        execution = dict(backend="cpu", device="cpu", dtype="bfloat16", threads=4,
                         component="original Diffusers transformer_blocks.1")
        def boundary(self, state):
            return "joint:"+str(1+state["advanced"])
        def point(self, state):
            return ExecutionPoint(family="flux2-klein-joint-component", logical_step=2,
                                  phase="joint", index=1+state["advanced"])
        def addresses(self, state):
            return tuple(state)
        def validate(self, state):
            if state["advanced"] not in (0, 1) or state["image"].shape != (1, 1024, 3072) or state["text"].shape != (1, 512, 3072):
                raise ValueError("standalone component state geometry differs")
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and (value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
                    raise ValueError("component requires finite CPU state: "+name)
                if isinstance(value, torch.Tensor) and value.dtype != (torch.float32 if name in {"cos", "sin"} else torch.bfloat16):
                    raise ValueError("component state dtype differs: "+name)
        def advance(self, state):
            if state["advanced"]:
                raise ValueError("one native component step only")
            with torch.inference_mode():
                text, image = block(hidden_states=state["image"], encoder_hidden_states=state["text"],
                    temb_mod_img=state["img_mod"], temb_mod_txt=state["txt_mod"],
                    image_rotary_emb=(state["cos"], state["sin"]), joint_attention_kwargs=None)
            return dict(state, image=image, text=text, advanced=1)

    adapter, store = JointComponent(), LocalStore(out / "store")
    def receipt(r):
        store.receipt(r)
        receipts.append(r.to_dict())
        counts["receipts"] += 1

    tx = ForecastTransaction(out / "transaction", model_identity=adapter.model_identity,
        input_identity=config["source_audit_sha256"], source_identity=request["request_sha256"],
        frontier=dict(kind="architectural", boundary="original observed block-entry image/text/context", parents=config["parents"],
            projection="Standalone component closure omits latent/conditioning/schedule fields not read by this block; no full-pipeline continuation."),
        execution_scope="Native Diffusers block through explicit MDB standalone adapter; port-only hooks; four separately counted native SDPA kernels; no ATen replay.",
        budgets=dict(max_native_calls=7, max_wall_ms=600000, max_bytes=512 << 20))
    predicted, corners = {}, {}
    def predict(_):
        for name, state in states.items():
            wall()
            predicted[name] = L["forecast"](state, weights, ix, txix)
            counts["numpy_block_forecasts"] += 1
            event("predicted", case=name)
        for name in config["corners"]:
            kname = "changed" if name[1] == "1" else "base"
            vname = "changed" if name[3] == "1" else "base"
            value, _ = L["dense"](predicted["base"]["q"], predicted[kname]["k"], predicted[vname]["v"])
            corners[name] = S["quantize"](value)
            counts["numpy_corner_forecasts"] += 1
        return dict(predictions=dict(blocks=predicted, corners=corners, counts_at_seal=counts.copy(), hypotheses=config["hypotheses"]))
    tx.predict("raw_state", predict)
    seal = tx.seal()
    save(out / "seal-census.json", dict(counts=counts.copy(), seal=seal))
    event("sealed")
    completed, observations = [], {}
    native.dispatch_attention_fn = dispatch
    try:
        with tx:
            def execute(_):
                for name, initial in states.items():
                    wall()
                    active.update(name=name, fields={}, ports={})
                    typed = {k: torch.from_numpy(v.copy()).to(dtype=torch.float32 if k in {"cos", "sin"} else torch.bfloat16) for k, v in initial.items()}
                    typed["advanced"] = 0
                    nav = NativeCutNavigator(Session(adapter, typed), max_steps=1, on_receipt=receipt)
                    before = nav.session.capture(retain=False)
                    store.save(before)
                    source_description = describe(before.payload)
                    r = nav.step()
                    after = nav.session.capture(retain=False)
                    store.save(after)
                    if describe(before.payload) != source_description:
                        raise ValueError("original component parent mutated")
                    observe("image_exit", after.payload["image"][:, ix], "native.block:return.image[selected-image-rows]")
                    fields = active["fields"]
                    if set(fields) != set(predicted[name]):
                        raise ValueError("declared native observation inventory differs")
                    path = out / ("native-"+name+".npz")
                    np.savez_compressed(path, **fields)
                    row = dict(id=name, parent=before.fingerprint, result=after.fingerprint,
                        receipt=r.to_dict(), archive=dict(path=path.name, sha256=sha(path)), ports=active["ports"],
                        source_parent=config["parents"].get(name, "derived midpoint of declared base/changed text"))
                    save(out / ("completed-"+name+".json"), row)
                    observations[name] = fields
                    completed.append(row)
                    active.update(name=None, fields=None, ports=None)
                    event("native_block_complete", case=name)
                corner_values = {}
                for name in config["corners"]:
                    wall()
                    kname = "changed" if name[1] == "1" else "base"
                    vname = "changed" if name[3] == "1" else "base"
                    operands = [torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16) for v in (
                        predicted["base"]["q"], predicted[kname]["k"], predicted[vname]["v"])]
                    counts["native_sdpa_standalone"] += 1
                    with torch.inference_mode():
                        value = torch.nn.functional.scaled_dot_product_attention(*operands, dropout_p=0., is_causal=False)
                    corner_values[name] = value.float().numpy()
                path = out / "native-corners.npz"
                np.savez_compressed(path, **corner_values)
                return dict(rows=completed, corners=dict(path=path.name, sha256=sha(path)), counts=counts.copy())
            tx.execute("native", execute)
    finally:
        native.dispatch_attention_fn = original_dispatch
        for handle in handles:
            handle.remove()
    if counts != config["expected_counts"]:
        raise ValueError("finite physical census differs: "+str(counts))
    for name, value in block.named_parameters():
        if describe(value) != parameter_descriptors[name]:
            raise ValueError("native checkpoint parameter mutated")
    comp = L["routing_components"](observations["base"]["q"], observations["base"]["k"], observations["base"]["v"],
        observations["changed"]["k"], observations["changed"]["v"], 512)
    path = out / "routing-components.npz"
    np.savez_compressed(path, **comp)
    metrics = dict(errors={name: {key: S["difference"](v, predicted[name][key]) for key, v in fields.items()} for name, fields in observations.items()},
        routing={key: S["statistics"](v) for key, v in comp.items()},
        structural={name: dict(q=S["difference"](v["q"], observations["base"]["q"]),
            image_k=S["difference"](v["k"][:, :, 512:], observations["base"]["k"][:, :, 512:]),
            image_v=S["difference"](v["v"][:, :, 512:], observations["base"]["v"][:, :, 512:])) for name, v in observations.items()})
    save(out / "metrics.json", metrics)
    save(out / "rows.json", completed)
    save(out / "receipts.json", receipts)
    summary = dict(schema="mdb-flux-attention-state-law-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        counts=counts, seal=seal, transaction=verify_transaction(tx.root), worker_seconds=time.monotonic()-started,
        mechanics_status="pending-independent-audit", terminal_status="unassessed")
    save(out / "summary.json", summary)
    save(out / "files.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=summary["job_id"], request_sha256=request["request_sha256"],
        output_root=str(out.resolve()), files_manifest_sha256=sha(out / "files.json"), summary_sha256=sha(out / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        out = Path("binding-minimal-result")
        if out.is_dir() and (out / "request.json").exists() and not (out / "files.json").exists():
            S = runpy.run_path(str(Path(__file__).with_name("flux_mlp_state_law.py")))
            S["save"](out / "failure.json", dict(error=repr(error), job_id=os.environ.get("MRUN_JOB_ID"), native_retries=0))
            S["save"](out / "files.json", {p.relative_to(out).as_posix(): S["sha"](p) for p in sorted(out.rglob("*")) if p.is_file()})
        raise
