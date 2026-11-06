"""Seal complete image/text forecasts, then run native and predicted-state futures."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import time


def worker():
    payload = Path.cwd()
    C = runpy.run_path(str(payload / "flux_state_closure.py"))
    S = runpy.run_path(str(payload / "flux_mlp_state_law.py"))
    read, save, sha = S["read"], S["save"], S["sha"]
    config, request = [read(os.environ[k]) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST")]
    assert config == request["config"]
    assert all(sha(payload / n) == h for n, h in request["source_hashes"].items())
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    import numpy as np
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    from safetensors import safe_open
    from saturn_pub import Adapter, Session, Act
    from saturn_pub.contracts import ExecutionPoint
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.native_navigation import NativeCutNavigator
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction, load_forecast_tree

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    out = Path("binding-minimal-result"); out.mkdir()
    for name in ("plan.json", "parameter-artifacts.json", "retrieve_weights.py"):
        shutil.copyfile(payload / name, out / name)
    save(out / "request.json", request)
    checkpoint = Path(config["model_path"]) / config["checkpoint_file"]
    assert sha(checkpoint) == config["checkpoint_file_sha256"]
    counts = dict.fromkeys(config["expected_counts"], 0)
    blocks, descriptors = {}, {}
    # One checkpoint-backed two-block component; both native modules load once.
    with safe_open(str(checkpoint), framework="pt", device="cpu") as bank:
        for index in config["blocks"]:
            state = {}
            for record in read(out / "parameter-artifacts.json")["artifacts"]:
                src = record["source"]
                prefix = f"transformer_blocks.{index}."
                if not src["tensor"].startswith(prefix):
                    continue
                tensor = bank.get_tensor(src["tensor"])
                assert tensor.dtype == torch.bfloat16 and list(tensor.shape) == src["shape"]
                assert hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == src["payload_sha256"]
                state[src["tensor"].removeprefix(prefix)] = tensor.clone().contiguous()
            with torch.device("meta"):
                block = native.Flux2TransformerBlock(dim=3072, num_attention_heads=24,
                    attention_head_dim=128, mlp_ratio=3., eps=1e-6, bias=False)
            block.load_state_dict(state, strict=True, assign=True)
            block.eval().requires_grad_(False)
            blocks[index] = block
            descriptors[str(index)] = {n: describe(p) for n, p in block.named_parameters()}
        del state, tensor, block
    counts["component_loads"] = 1
    save(out / "native-source.json", dict(diffusers_source_sha256=sha(Path(native.__file__)),
        parameters=descriptors, scope="Original two joint blocks, native CPU BF16; port-only observation."))
    arrays = {}
    for name, ref in config["input_files"].items():
        p = payload / (name+".npy")
        assert sha(p) == ref["sha256"]
        arrays[name] = np.load(p, allow_pickle=False)
    roots = C["states"](arrays)
    common = {k: arrays[k] for k in ("img_mod", "txt_mod", "cos", "sin")}
    ix, txix = config["indices"], config["text_indices"]
    active = dict(case=None, block=None, fields=None, ports=None)
    handles, receipts, rows, navigators, j1cuts = [], [], [], {}, {}
    original_dispatch = native.dispatch_attention_fn

    def wall():
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise RuntimeError("finite worker wall budget")

    def event(phase, **fields):
        body = dict(phase=phase, seconds=time.monotonic()-started, counts=counts.copy(), **fields)
        with (out / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True)+"\n")
        print("MDB_STATE_CLOSURE "+json.dumps(body, sort_keys=True), flush=True)

    def count(key):
        def hook(module, args):
            counts[key] += 1
        return hook

    def observe(field, tensor, port):
        if active["fields"] is None:
            raise ValueError("unscoped port observation")
        active["fields"][field] = tensor.detach().float().cpu().numpy().copy()
        active["ports"][field] = dict(port=port, descriptor=describe(tensor))

    def hook_return(field, selector, port):
        def hook(module, args, value):
            observe(field, selector(value), port)
            return value
        return hook

    for index, block in blocks.items():
        handles += [block.register_forward_pre_hook(count("native_blocks")),
            block.attn.register_forward_pre_hook(count("native_attention")),
            block.ff.register_forward_pre_hook(count("native_image_ff")),
            block.ff_context.register_forward_pre_hook(count("native_text_ff"))]
        handles += [m.register_forward_pre_hook(count("native_linears")) for m in block.modules() if isinstance(m, torch.nn.Linear)]
        def attn_input(module, args, kwargs):
            observe("image.norm", kwargs["hidden_states"][:, ix], "attn:input.hidden_states[image-witness]")
            observe("text.norm", kwargs["encoder_hidden_states"][:, txix], "attn:input.encoder_hidden_states[text-witness]")
        handles.append(block.attn.register_forward_pre_hook(attn_input, with_kwargs=True))
        for stream, ff, selected in (("image", block.ff, ix), ("text", block.ff_context, txix)):
            for field, module in (("projected", ff.linear_in), ("activated", ff.act_fn.gate_fn), ("product", ff.act_fn), ("mlp", ff)):
                port = next(n for n, m in block.named_modules() if m is module)
                handles.append(module.register_forward_hook(hook_return(stream+"."+field, lambda v, selected=selected: v[:, selected], port+":return[witness]")))
            def ff_input(module, args, stream=stream, selected=selected):
                observe(stream+".mlp_input", args[0][:, selected], ("ff" if stream == "image" else "ff_context")+":input[witness]")
            handles.append(ff.register_forward_pre_hook(ff_input))
        handles.append(block.attn.register_forward_hook(hook_return("image.attention", lambda v: v[0][:, ix], "attn:return.image[witness]")))
        handles.append(block.attn.register_forward_hook(hook_return("text.attention", lambda v: v[1][:, txix], "attn:return.text[witness]")))

    def dispatch(q, k, v, *args, **kwargs):
        counts["native_sdpa"] += 1
        for stream, positions in (("image", [512+i for i in ix]), ("text", txix)):
            for field, value in (("q", q), ("k", k), ("v", v)):
                observe(stream+"."+field, value[:, positions].permute(0, 2, 1, 3), "native.dispatch_attention_fn:input."+field+"["+stream+"-witness]")
        result = original_dispatch(q, k, v, *args, **kwargs)
        for stream, positions in (("image", [512+i for i in ix]), ("text", txix)):
            observe(stream+".head", result[:, positions].permute(0, 2, 1, 3), "native.dispatch_attention_fn:return["+stream+"-witness]")
        return result

    class JointChain(Adapter):
        model_identity = config["model"]
        execution = dict(backend="cpu", device="cpu", dtype="bfloat16", threads=4,
                         component="original Diffusers transformer_blocks.1-2")
        def boundary(self, state):
            return "joint:"+str(1+state["advanced"])
        def point(self, state):
            return ExecutionPoint(family="flux2-klein-two-joint-component", logical_step=2,
                                  phase="joint", index=1+state["advanced"])
        def addresses(self, state):
            return tuple(state)
        def validate(self, state):
            if state["advanced"] not in (0, 1, 2) or state["image"].shape != (1, 1024, 3072) or state["text"].shape != (1, 512, 3072):
                raise ValueError("chain closure geometry differs")
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and (value.device.type != "cpu" or not bool(torch.isfinite(value).all()) or value.dtype != (torch.float32 if key in {"cos", "sin"} else torch.bfloat16)):
                    raise ValueError("chain state must have finite declared CPU dtypes: "+key)
        def advance(self, state):
            index = 1+state["advanced"]
            assert index == active["block"] and index in blocks
            with torch.inference_mode():
                text, image = blocks[index](hidden_states=state["image"], encoder_hidden_states=state["text"],
                    temb_mod_img=state["img_mod"], temb_mod_txt=state["txt_mod"],
                    image_rotary_emb=(state["cos"], state["sin"]), joint_attention_kwargs=None)
            return dict(state, image=image, text=text, advanced=state["advanced"]+1)

    adapter, store = JointChain(), LocalStore(out / "store")
    def receipt(value):
        store.receipt(value)
        body = value.to_dict()
        receipts.append(body)
        counts["receipts"] += 1
        if body["operation"] == "apply":
            counts["acts"] += 1
        wall()

    tx = ForecastTransaction(out / "transaction", model_identity=adapter.model_identity,
        input_identity=config["source_attention_audit_sha256"], source_identity=request["request_sha256"],
        frontier=dict(kind="architectural", boundary="observed complete raw joint1-entry image/text/context;quarter mix derived before native",
            parents=config["parents"], scope="two-block standalone closure; latent/scheduler/conditioning not read; no full-pipeline continuation"),
        execution_scope="Original native two-block CPU component via MDB cuts/receipts and observation-only hooks; independent full NumPy state forecasts; no native intermediates used by predictor.",
        budgets=dict(max_native_calls=9, max_wall_ms=720000, max_bytes=512 << 20))
    predictions = {}
    for index in config["blocks"]:
        # Only one FP32 parameter expansion at a time; native BF16 blocks stay fixed.
        weights = {n: p.detach().float().numpy() for n, p in blocks[index].named_parameters()}
        def predict(_, index=index):
            values = {}
            for name in config["cases"]:
                wall()
                state = roots[name] if index == 1 else dict(common, image=predictions[1][name]["image"], text=predictions[1][name]["text"])
                values[name] = C["forecast"](state, weights, ix, txix, chunk_rows=config["chunk_rows"])
                counts["numpy_block_forecasts"] += 1
                event("predicted", case=name, block=index)
            return dict(predictions=values)
        tx.predict("joint"+str(index), predict)
        descriptor = read(tx.root / f"forecasts/joint{index}.json")["predictions"]
        predictions[index] = load_forecast_tree(tx.root, descriptor)
        del weights
        gc.collect()
    tx.predict("census", lambda _: dict(predictions=dict(counts_at_seal=counts.copy(), hypotheses=config["hypotheses"])))
    seal = tx.seal()
    save(out / "seal-census.json", dict(counts=counts.copy(), seal=seal))
    event("sealed")

    def capture(nav):
        cut = nav.session.capture(retain=False)
        store.save(cut)
        return cut

    def step(nav, name, index, branch):
        wall()
        active.update(case=name, block=index, fields={}, ports={})
        before = capture(nav)
        original = describe(before.payload)
        r = nav.step()
        after = capture(nav)
        assert describe(before.payload) == original
        for stream in ("image", "text"):
            observe(stream, after.payload[stream], "native.block:return."+stream+"[complete]")
        assert set(active["fields"]) == set(predictions[index][name])
        path = out / (branch+"-"+name+"-j"+str(index)+".npz")
        # Complete exits already live in typed CAS state; archive only witnesses.
        np.savez_compressed(path, **{k:v for k,v in active["fields"].items() if k not in {"image", "text"}})
        row = dict(case=name, block=index, branch=branch, parent=before.fingerprint, result=after.fingerprint,
            receipt=r.to_dict(), ports=active["ports"], complete_fields_from_state=["image", "text"],
            archive=dict(path=path.name, sha256=sha(path)))
        rows.append(row)
        save(out / ("completed-"+path.stem+".json"), row)
        active.update(case=None, block=None, fields=None, ports=None)
        event("native_complete", case=name, block=index, branch=branch)
        return after

    native.dispatch_attention_fn = dispatch
    try:
        with tx:
            def execute(_):
                for name in config["cases"]:
                    typed = {k: torch.from_numpy(v.copy()).to(torch.float32 if k in {"cos", "sin"} else torch.bfloat16) for k, v in roots[name].items()}
                    nav = NativeCutNavigator(Session(adapter, dict(typed, advanced=0)), max_steps=2, on_receipt=receipt)
                    navigators[name] = nav
                    j1cuts[name] = step(nav, name, 1, "native")
                for name in config["cases"]:
                    step(navigators[name], name, 2, "native")
                forks = []
                for name in config["cases"]:
                    # Fork directly from saved J1 cut through public Session; navigator
                    # needs a fresh physical budget because source has already taken2steps.
                    session = navigators[name].session.fork(j1cuts[name])
                    counts["forks"] += 1
                    nav = NativeCutNavigator(session, max_steps=1, on_receipt=receipt)
                    fork_parent = capture(nav)
                    acts = []
                    for stream in ("image", "text"):
                        before = capture(nav)
                        value = torch.from_numpy(np.asarray(predictions[1][name][stream]).copy()).to(torch.bfloat16)
                        act = Act.replace(stream, value, name="sealed-predicted-joint1-"+stream)
                        r = nav.apply(act)
                        after = capture(nav)
                        assert describe(after.payload) == describe({**before.payload, stream:value})
                        acts.append(dict(stream=stream, before=before.fingerprint, after=after.fingerprint,
                            act=act.manifest(), receipt=r.to_dict()))
                    predicted_input = capture(nav)
                    for stream in ("image", "text"):
                        assert np.array_equal(predicted_input.payload[stream].float().numpy(), predictions[1][name][stream])
                    result = step(nav, name, 2, "predicted-input")
                    forks.append(dict(case=name, source=j1cuts[name].fingerprint, fork_parent=fork_parent.fingerprint,
                        predicted_input=predicted_input.fingerprint, acts=acts, result=result.fingerprint))
                save(out / "forks.json", forks)
                return dict(rows=rows, forks=forks, counts=counts.copy())
            tx.execute("native", execute)
    finally:
        native.dispatch_attention_fn = original_dispatch
        for handle in handles:
            handle.remove()
    assert counts == config["expected_counts"], counts
    for index, block in blocks.items():
        assert {n: describe(p) for n, p in block.named_parameters()} == descriptors[str(index)]
    save(out / "rows.json", rows)
    save(out / "receipts.json", receipts)
    summary = dict(schema="mdb-flux-state-closure-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        counts=counts, seal=seal, transaction=verify_transaction(tx.root), worker_seconds=time.monotonic()-started,
        mechanics_status="pending-independent-audit", terminal_status="unassessed")
    save(out / "summary.json", summary)
    save(out / "files.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY "+json.dumps(dict(job_id=summary["job_id"], request_sha256=request["request_sha256"],
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
