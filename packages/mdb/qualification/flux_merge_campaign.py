"""Factor a localized native parallel merge into residual/attention/MLP operands."""
from __future__ import annotations

import itertools
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    support = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, file_sha, identity = (support[key] for key in ("save", "file_sha", "identity"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(file_sha(name) != value for name, value in request["source_hashes"].items()):
        raise ValueError("request/source differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    candidate = runpy.run_path("flux_parallel_merge.py")["merge_candidate"]
    attention_helper = runpy.run_path("flux_attention_sources.py")["attention_sources"] if config.get("attention_sources") else None
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-merge-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, sequence = time.monotonic(), 0
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "projection_evaluations": 0,
              "prompt_encodings": 0, "receipt_emissions": 0, "physical_batch": 1}

    def event(kind, **details):
        nonlocal sequence
        row = {"kind": kind, "seconds": time.monotonic() - started, "sequence": sequence, **details}
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_MERGE_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = None
    store = LocalStore(output / "typed-store")

    def on_receipt(receipt):
        store.receipt(receipt)
        body = receipt.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "receipt": receipt.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        if body["operation"] == "continue":
            counts["adapter_steps"] += body["steps"]
        else:
            counts["acts"] += 1
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("native budget exceeded")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=on_receipt)

    initial = {}
    if config.get("fresh_context"):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu,
            retrieve_timesteps,
        )

        # Encode the two fixed literals once; all later branches reuse their cuts.
        pipe.transformer.to("cpu")
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        pipe.text_encoder.to("cuda")
        prepared = {}
        for name, prompt in config["prompts"].items():
            wall()
            with torch.inference_mode():
                conditioning, txt_ids = pipe.encode_prompt(prompt, device=torch.device("cuda"),
                    num_images_per_prompt=1, max_sequence_length=512)
            counts["prompt_encodings"] += 1
            prepared[name] = (conditioning, txt_ids)
        if not torch.equal(prepared["base"][1], prepared["changed"][1]):
            raise ValueError("prompt pair changes position IDs")
        with torch.inference_mode():
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                prepared["base"][0].dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
            sigmas = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
            timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("unexpected schedule")
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.vae.bn.to("cpu")
        # Construct only after final placement: the adapter freezes storage identity.
        adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
        raw = adapter.session(latent=latent, conditioning=prepared["base"][0], img_ids=img_ids,
            txt_ids=prepared["base"][1], timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
        current = NativeCutNavigator(raw, max_steps=config["max_adapter_steps"], on_receipt=on_receipt)
        current.step()
        base_projected = current.capture()
        store.save(base_projected)
        donor_text = torch.nn.functional.linear(prepared["changed"][0],
            pipe.transformer.context_embedder.weight, pipe.transformer.context_embedder.bias)
        changed_nav = nav(base_projected)
        text_act = Act.replace("text", donor_text, name="initial-working-text-contrast")
        text_receipt = changed_nav.apply(text_act)
        changed_projected = changed_nav.capture()
        store.save(changed_projected)
        expected = dict(base_projected.payload)
        expected["text"] = donor_text
        if describe(changed_projected.payload) != describe(expected):
            raise ValueError("initial text intervention changed undeclared closure")
        initial = {"base_projected": base_projected.fingerprint, "changed_projected": changed_projected.fingerprint,
            "text_act": text_act.manifest(), "text_receipt": text_receipt.to_dict(),
            "prompts": config["prompts"], "seed": config["seed"],
            "conditioning": {name: tensor_descriptor(value[0]) for name, value in prepared.items()}}
        before = {}
        for name, projected in (("base", base_projected), ("changed", changed_projected)):
            session = nav(projected)
            session.until(**config["selector"])
            before[name] = session.capture()
        event("fresh_prefix_ready", counts=dict(counts))
    else:
        adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
        inputs = LocalStore("input-store")
        before = {name: inputs.load(cut_id, device="cuda") for name, cut_id in config["input_cuts"].items()}
    for cut in before.values():
        store.save(cut)
        if cut.model_identity != adapter.model_identity or any(cut.execution_point[k] != v for k, v in config["selector"].items()):
            raise ValueError("input model or selected boundary differs")
    for slot in before["base"].payload:
        if slot not in {"text", "image"} and describe(before["base"].payload[slot]) != describe(before["changed"].payload[slot]):
            raise ValueError("unmatched source closure: " + slot)

    target = pipe.transformer.single_transformer_blocks[config["block"]].attn
    if attention_helper is not None and (type(target.processor).__name__ != "Flux2ParallelSelfAttnProcessor"
            or target.processor._attention_backend is not None or target.processor._parallel_config is not None):
        raise ValueError("attention intervention requires native default backend without parallel configuration")
    projection, ports, after, prefix_receipts = target.to_out, {}, {}, {}
    text_rows = before["base"].payload["text"].shape[1]
    gate = Flux2Modulation.split(before["base"].payload["context"]["single_mod"], 1)[0][2]
    residuals = {name: torch.cat([cut.payload["text"], cut.payload["image"]], dim=1) for name, cut in before.items()}

    def save_arrays(name, values):
        path = output / (name + ".npz")
        np.savez_compressed(path, **{key: value.detach().float().cpu().numpy() for key, value in values.items()})
        return {"path": path.name, "sha256": file_sha(path), "descriptors": {key: tensor_descriptor(value) for key, value in values.items()}}

    # Read-only observation hooks. All interventions happen later as typed image Acts.
    port_records = {}
    for name, cut in before.items():
        captured = {}

        def input_hook(module, args):
            if "operands" in captured:
                raise ValueError("projection entered more than once")
            captured["operands"] = args[0].detach().clone()

        def output_hook(module, args, result):
            captured["projection"] = result.detach().clone()

        hooks = [projection.register_forward_pre_hook(input_hook), projection.register_forward_hook(output_hook)]
        qkv_ports = {}
        if attention_helper is not None:
            def capture_q(module, args, result):
                qkv_ports["q"] = result.detach().clone()

            def capture_k(module, args, result):
                qkv_ports["k"] = result.detach().clone()

            def capture_v(module, args, result):
                qkv = result[..., :3 * target.inner_dim]
                qkv_ports["v"] = qkv.chunk(3, dim=-1)[2].unflatten(-1, (target.heads, -1)).detach().clone()

            hooks += [target.norm_q.register_forward_hook(capture_q), target.norm_k.register_forward_hook(capture_k),
                      target.to_qkv_mlp_proj.register_forward_hook(capture_v)]
        session = nav(cut)
        try:
            receipt = session.step()
        finally:
            for hook in hooks:
                hook.remove()
        if set(captured) != {"operands", "projection"}:
            raise ValueError("native projection ports missing")
        after[name] = session.capture()
        store.save(after[name])
        ports[name] = captured
        if attention_helper is not None:
            rotate = runpy.run_path("flux_functional_denoiser.py")["_rotate"]
            qkv_ports["q"] = rotate(qkv_ports["q"], cut.payload["context"]["rope"])
            qkv_ports["k"] = rotate(qkv_ports["k"], cut.payload["context"]["rope"])
            captured["qkv"] = qkv_ports
        prefix_receipts[name] = receipt.to_dict()
        port_records[name] = save_arrays(name + ".native-ports", {key: value for key, value in captured.items() if key != "qkv"} | {"residual": residuals[name], "gate": gate})
        if attention_helper is not None:
            port_records[name]["positioned_qkv"] = save_arrays(name + ".positioned-qkv", qkv_ports)
    input_cuts = {name: cut.fingerprint for name, cut in before.items()}
    source = {"schema": "mdb-flux-merge-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "input_cuts": input_cuts, "initial": initial,
        "post_cuts": {name: cut.fingerprint for name, cut in after.items()}, "prefix_receipts": prefix_receipts,
        "ports": port_records, "attention_width": target.inner_dim, "text_rows": text_rows,
        "projection_path": f"transformer.single_transformer_blocks.{config['block']}.attn.to_out",
        "attention_contract": {"processor": type(target.processor).__name__, "backend": str(target.processor._attention_backend),
            "parallel_config": str(target.processor._parallel_config), "Q_factor": "source of installed image-query outputs; text-query outputs discarded; changed text remains"},
        "weight": tensor_descriptor(projection.weight), "bias": None if projection.bias is None else tensor_descriptor(projection.bias),
        "scope": "observed native R/A/M ports and two blocks before seal; original frozen projection, typed image-only post-block Acts; no independent edited-image forecast"}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        frontier={"kind": "architectural", "source_sha256": file_sha(output / "source.json"),
            "allowed_prefix": "declared fresh text encodings/projection and matched prefixes, then two selected-block port observations; no RGB outcomes" if initial else "two native selected-block executions to observe actual merge ports; no RGB outcomes"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("merge_hypotheses", lambda context: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows = []

    def finish(session, row):
        event("branch_start", row=row["id"])
        schedulers = []
        while session.point["logical_step"] < config["steps"]:
            wall()
            point = session.point
            session.step()
            if point["phase"] == "scheduler":
                value = session.session.read("latent")
                path = output / f"{row['id']}.step{session.point['logical_step']}.npy"
                np.save(path, value.float().cpu().numpy(), allow_pickle=False)
                schedulers.append({"step": session.point["logical_step"], "path": path.name, "sha256": file_sha(path), "descriptor": tensor_descriptor(value)})
        if counts["images"] >= config["max_images"]:
            raise RuntimeError("image budget exceeded")
        path = output / (row["id"] + ".png")
        adapter.decode(session.session, height=512, width=512).save(path)
        counts["images"] += 1
        row.update(schedulers=schedulers, image={"path": path.name, "sha256": file_sha(path)})
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def install(image, row):
        session = nav(after["changed"])
        act = Act.replace("image", image, name=row["id"])
        receipt = session.apply(act)
        post = session.capture()
        store.save(post)
        expected = dict(after["changed"].payload)
        expected["image"] = image
        if describe(post.payload) != describe(expected):
            raise ValueError("Act changed undeclared closure")
        row.update(parent=after["changed"].fingerprint, post_act=post.fingerprint, act=act.manifest(), receipt=receipt.to_dict())
        finish(session, row)

    with tx:
        event("hypotheses_sealed", counts=dict(counts), seal=seal)

        def execute(context):
            for name in ("base", "changed"):
                finish(nav(after[name]), {"id": name, "kind": "reference", "parent": after[name].fingerprint})
            install(after["changed"].payload["image"], {"id": "copy.sham", "kind": "copy"})
            for bits in itertools.product("01", repeat=3):
                wall()
                mask = "".join(bits)
                with torch.inference_mode():
                    result = candidate(recipient_residual=residuals["changed"], donor_residual=residuals["base"],
                        recipient_operands=ports["changed"]["operands"], donor_operands=ports["base"]["operands"],
                        gate=gate, projection=projection, attention_width=target.inner_dim, text_rows=text_rows, mask=mask)
                counts["projection_evaluations"] += 1
                native_changed = torch.cat([after["changed"].payload["text"], after["changed"].payload["image"]], dim=1)
                row = {"id": "merge." + mask, "kind": "merge", "mask_R_A_M": mask,
                    "operands_descriptor": tensor_descriptor(result["operands"]),
                    "projection": save_arrays("merge." + mask + ".projection", {"value": result["projection"]}),
                    "text_rows_exact_changed": torch.equal(result["updated"][:, :text_rows], native_changed[:, :text_rows]),
                    "self_full_state_exact_changed": torch.equal(result["updated"], native_changed) if mask == "000" else None,
                    "all_donor_image_exact_base": torch.equal(result["updated"][:, text_rows:], after["base"].payload["image"]) if mask == "111" else None}
                install(result["updated"][:, text_rows:], row)
            if attention_helper is not None:
                joins = {}
                for bits in itertools.product("01", repeat=3):
                    wall()
                    mask = "".join(bits)
                    with torch.inference_mode():
                        selected = attention_helper(base=ports["base"]["qkv"], changed=ports["changed"]["qkv"], text_rows=text_rows, mask=mask)
                        operands = ports["changed"]["operands"].clone()
                        operands[:, text_rows:, :target.inner_dim] = selected["attention"][:, text_rows:]
                        result = candidate(recipient_residual=residuals["changed"], donor_residual=residuals["base"],
                            recipient_operands=operands, donor_operands=ports["base"]["operands"],
                            gate=gate, projection=projection, attention_width=target.inner_dim, text_rows=text_rows, mask="100")
                    counts["projection_evaluations"] += 1
                    # Q/K/V selections are recipes over retained source arrays;
                    # retain their hashes without duplicating six large arrays per row.
                    qkv_record = save_arrays("attention." + mask + ".ports", {"attention": selected["attention"]})
                    qkv_record["selected_ports"] = {key: tensor_descriptor(selected[key]) for key in ("q", "k", "v")}
                    if mask in {"000", "111"}:
                        native = ports["changed" if mask == "000" else "base"]["operands"][:, :, :target.inner_dim]
                        if not torch.equal(selected["attention"], native):
                            raise ValueError("attention self reconstruction differs from native ports")
                        joins[mask] = {"equivalent_row": "merge.100" if mask == "000" else "merge.110", "ports": qkv_record,
                            "image_descriptor": tensor_descriptor(result["updated"][:, text_rows:]),
                            "projection_descriptor": tensor_descriptor(result["projection"])}
                        continue
                    row = {"id": "attention." + mask, "kind": "attention", "mask_Q_TKV_IKV": mask,
                        "positioned_qkv": qkv_record, "operands_descriptor": tensor_descriptor(result["operands"]),
                        "projection": save_arrays("attention." + mask + ".projection", {"value": result["projection"]})}
                    install(result["updated"][:, text_rows:], row)
                save(output / "attention-joins.json", joins)
            return {"rows": rows, "counts": dict(counts), "source_sha256": file_sha(output / "source.json")}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-merge-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counts, "mechanics_status": "valid",
        "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started, "seal": seal,
        "transaction_verification": verification, "RGB_forecast": False, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): file_sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config.get("max_archive_bytes", 256 << 20))


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-merge-result")
        if output.is_dir():
            report = {"schema": "mdb-flux-merge-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())["request_sha256"],
                "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            runpy.run_path("flux_trajectory_forecast_campaign.py")["save"](output / "failure.json", report)
            bound = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()).get("max_archive_bytes", 256 << 20)
            runpy.run_path("campaign_support.py")["emit"](output, report, max_compressed_bytes=bound)
        raise
