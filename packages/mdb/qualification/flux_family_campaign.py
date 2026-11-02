"""One resident MDB run: six causal panels with immutable rewind/replay roots."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import time
from pathlib import Path


def create_transaction(root, *, model_identity, input_identity, source_identity, source_sha256, max_wall_seconds):
    """Keep full model weights external; bind source evidence and finite phases."""
    from mdb.forecast_transaction import ForecastTransaction

    return ForecastTransaction(root, model_identity=model_identity, input_identity=input_identity,
        source_identity=source_identity, frontier={"kind": "architectural", "source_sha256": source_sha256,
            "allowed_prefix": "eight encodings,matched prefixes through single4 ports; heldout prefix only; no heldout suffix/RGB"},
        budgets={"max_native_calls": 100000, "max_wall_ms": max_wall_seconds * 1000})


def seal_prediction(transaction, name, value):
    transaction.predict(name, lambda context: {"predictions": value})
    return transaction.seal()


def record_failure(output, report, *, save, file_sha, emit, limits):
    """Preserve a prior publication manifest and never mask the first error."""
    save(output / "failure.json", report)
    manifest = output / "files.json"
    if manifest.exists():
        prior = output / "files-before-failure.json"
        with prior.open("xb") as stream:
            stream.write(manifest.read_bytes())
    entries = {path.relative_to(output).as_posix(): file_sha(path)
               for path in sorted(output.rglob("*")) if path.is_file() and path != manifest}
    manifest.write_text(json.dumps(entries, sort_keys=True, indent=2, allow_nan=False) + "\n")
    try:
        emit(output, report, limits)
        return True
    except Exception as custody_error:
        print("MDB_FLUX_FAMILY_FAILURE_SUMMARY_V1 " + json.dumps({**report,
            "custody_error": f"{type(custody_error).__name__}: {custody_error}",
            "scope": "failure archive unavailable; original exception and on-disk outputs retained"}, sort_keys=True), flush=True)
        return False


def worker():
    utility = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, sha, identity = (utility[key] for key in ("save", "file_sha", "identity"))
    plan = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if request["config"] != plan or any(sha(name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("request/payload differs")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(plan["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    ops = runpy.run_path("flux_family_ops.py")
    rules = runpy.run_path("flux_family_plan.py")
    merge = runpy.run_path("flux_parallel_merge.py")["merge_candidate"]
    rotate = runpy.run_path("flux_functional_denoiser.py")["_rotate"]
    output = Path("flux-family-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", plan)
    if shutil.disk_usage(output).free < 12 << 30:
        raise ValueError("requires twelve GiB free scratch for bounded evidence and transport")
    torch.set_num_threads(plan["cpu_threads"])
    torch.manual_seed(37)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    started, phase = time.monotonic(), "prefix"
    counts = dict(model_loads=0, prompt_encodings=0, prefix_projections=0, adapter_steps=0,
                  acts=0, images=0, rewinds=0, projection_evaluations=0, attention_evaluations=0,
                  receipt_emissions=0, physical_batch=1, native_module_entries=0)
    store = LocalStore(output / "typed-store")
    sequence, rows, finished = 0, [], {}
    source_cpu, histories, pairs = {}, {}, {}
    future_released, heldout_released = False, False
    phase_entries = {}

    def event(kind, **details):
        nonlocal sequence
        value = dict(sequence=sequence, kind=kind, phase=phase, seconds=time.monotonic() - started, **details)
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_FAMILY_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > plan["max_wall_seconds"]:
            raise RuntimeError("finite wall allowance exhausted")

    def receipt(receipt):
        store.receipt(receipt)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "receipt": receipt.fingerprint}) + "\n")
        counts["receipt_emissions"] += 1
        body = receipt.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 0) if body["operation"] == "continue" else 1
        if counts["adapter_steps"] > plan["max_adapter_steps"]:
            raise RuntimeError("native step allowance exhausted")

    def retain(cut):
        store.save(cut)
        return cut.fingerprint

    def nav(cut_id):
        counts["rewinds"] += 1
        return NativeCutNavigator(Session.from_cut(adapter, store.load(cut_id, device="cuda")),
                                  max_steps=plan["max_adapter_steps"], on_receipt=receipt)

    def arrays(name, values):
        path = output / (name + ".npz")
        np.savez_compressed(path, **{key: value.detach().float().cpu().numpy() for key, value in values.items()})
        return {"path": path.name, "sha256": sha(path),
                "descriptors": {key: tensor_descriptor(value) for key, value in values.items()}}

    def read_ports(history):
        return {key: value.to("cuda") for key, value in source_cpu[history].items()}

    model_root = Path(plan["model_path"])
    inventory = {path.relative_to(model_root).as_posix(): sha(path) for path in sorted(model_root.rglob("*"))
                 if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != plan["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    event("load_start")
    pipe = Flux2KleinPipeline.from_pretrained(str(model_root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein profile")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    prepared = {}
    for name, prompt in plan["prompts"].items():
        wall()
        with torch.inference_mode():
            conditioning, txt_ids = pipe.encode_prompt(prompt, device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
        prepared[name] = (conditioning.detach().cpu(), txt_ids.detach().cpu())
        counts["prompt_encodings"] += 1
        event("prompt_encoded", prompt_id=name)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")

    def weight_versions():
        values = [(name, list(value.shape), str(value.dtype), str(value.device), value.data_ptr(), value._version)
                  for module_name, module in (("transformer", pipe.transformer), ("vae", pipe.vae))
                  for name, value in [(module_name + "." + key, tensor) for key, tensor in list(module.named_parameters()) + list(module.named_buffers())]]
        return identity(values)

    initial_weight_versions = weight_versions()

    def native_entry(module, args):
        if phase in {"development_forecast", "heldout_forecast"}:
            raise RuntimeError("native module entered a prediction window")
        counts["native_module_entries"] += 1
        phase_entries[phase] = phase_entries.get(phase, 0) + 1

    hooks = [module.register_forward_pre_hook(native_entry) for component in (pipe.transformer, pipe.vae.decoder)
             for module in component.modules()]
    # Each seed's noise and schedule are prepared once; all histories reuse them.
    seeds = {}
    for seed in sorted({h["seed"] for h in plan["histories"]}):
        with torch.inference_mode():
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                torch.bfloat16, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(seed))
            sigmas = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
            timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("schedule geometry differs")
        seeds[seed] = (latent.detach().cpu(), img_ids.detach().cpu(), timesteps.detach().cpu(), pipe.scheduler.sigmas.detach().cpu())

    target = pipe.transformer.single_transformer_blocks[plan["selector"]["index"]].attn
    if type(target.processor).__name__ != "Flux2ParallelSelfAttnProcessor" or target.processor._attention_backend is not None or target.processor._parallel_config is not None:
        raise ValueError("unqualified native attention dispatcher")
    projection_weight = tensor_descriptor(target.to_out.weight)
    projection_bias = None if target.to_out.bias is None else tensor_descriptor(target.to_out.bias)
    initial_ids = {}
    for history in plan["histories"]:
        wall()
        name = history["id"]
        conditioning, ids = prepared[history["conditioning"]]
        working, work_ids = prepared[history["working_text"]]
        if not torch.equal(ids, work_ids):
            raise ValueError("working-text contrast changes position IDs")
        if name == "anchor.hybrid":
            current = nav(initial_ids["anchor.base"])
            with torch.inference_mode():
                text = torch.nn.functional.linear(working.to("cuda"), pipe.transformer.context_embedder.weight, pipe.transformer.context_embedder.bias)
            act = Act.replace("text", text, name="initial-working-text-contrast")
            rec = current.apply(act)
            initial = retain(current.capture())
            initial_act = {"parent": initial_ids["anchor.base"], "act": act.manifest(), "receipt": rec.to_dict()}
        else:
            latent, img_ids, timesteps, schedule = seeds[history["seed"]]
            raw = adapter.session(latent=latent.to("cuda"), conditioning=conditioning.to("cuda"), img_ids=img_ids.to("cuda"),
                                  txt_ids=ids.to("cuda"), timesteps=timesteps.to("cuda"), sigmas=schedule.to("cuda"))
            current = NativeCutNavigator(raw, max_steps=plan["max_adapter_steps"], on_receipt=receipt)
            current.step()
            counts["prefix_projections"] += 1
            initial = retain(current.capture())
            initial_act = None
        initial_ids[name] = initial
        current.until(**plan["selector"])
        cut = current.capture()
        before_id = retain(cut)
        text_rows = cut.payload["text"].shape[1]
        captured = {}

        def observe_input(module, args):
            if "operands" in captured:
                raise ValueError("projection entered twice")
            captured["operands"] = args[0].detach().clone()

        def observe_output(module, args, result):
            captured["projection"] = result.detach().clone()

        local_hooks = [target.to_out.register_forward_pre_hook(observe_input), target.to_out.register_forward_hook(observe_output)]
        qkv_needed = name in {"anchor.base", "anchor.hybrid"}
        if qkv_needed:
            def observe_q(module, args, result):
                captured["q"] = result.detach().clone()

            def observe_k(module, args, result):
                captured["k"] = result.detach().clone()

            def observe_v(module, args, result):
                captured["v"] = result[..., :3 * target.inner_dim].chunk(3, dim=-1)[2].unflatten(-1, (target.heads, -1)).detach().clone()

            local_hooks += [target.norm_q.register_forward_hook(observe_q), target.norm_k.register_forward_hook(observe_k), target.to_qkv_mlp_proj.register_forward_hook(observe_v)]
        try:
            rec = current.step()
        finally:
            for hook in local_hooks:
                hook.remove()
        after_id = retain(current.capture())
        captured["residual"] = torch.cat([cut.payload["text"], cut.payload["image"]], dim=1)
        captured["gate"] = Flux2Modulation.split(cut.payload["context"]["single_mod"], 1)[0][2]
        if qkv_needed:
            captured["q"] = rotate(captured["q"], cut.payload["context"]["rope"])
            captured["k"] = rotate(captured["k"], cut.payload["context"]["rope"])
        after = store.load(after_id, device="cuda")
        with torch.inference_mode():
            expected = captured["residual"] + captured["gate"] * captured["projection"]
        if not torch.equal(expected, torch.cat([after.payload["text"], after.payload["image"]], dim=1)):
            raise ValueError("native BF16 merge residual equation differs")
        histories[name] = {**history, "before": before_id, "after": after_id, "initial": initial, "initial_act": initial_act,
            "boundary": plan["selector"], "text_rows": text_rows, "attention_width": target.inner_dim, "heads": target.heads,
            "ports": arrays(name + ".ports", captured), "prefix_receipt": rec.to_dict(),
            "projection_weight": projection_weight, "projection_bias": projection_bias,
            "native_equation_exact": True}
        source_cpu[name] = {key: value.detach().cpu() for key, value in captured.items()}
        if sum(value.numel() * value.element_size() for values in source_cpu.values() for value in values.values()) > 1536 << 20:
            raise ValueError("declared CPU source cache exceeded")
        captured.clear()
        del cut, after, current
        torch.cuda.empty_cache()
        event("prefix_ready", history=name, split=history["split"], counts=dict(counts))

    for pair in plan["pairs"]:
        source, recipient = (histories[pair[key]] for key in ("source", "recipient"))
        if (source["text_rows"], source["attention_width"], source["heads"]) != (recipient["text_rows"], recipient["attention_width"], recipient["heads"]):
            raise ValueError("cross-history source geometry differs")
        left, right = (store.load(history["before"], device="cpu") for history in (source, recipient))
        exceptions = {"text", "image", "conditioning"}
        if any(describe(value) != describe(right.payload[key]) for key, value in left.payload.items() if key not in exceptions):
            raise ValueError("cross-history IDs/time/latent/context closure differs")
        a, b = (source_cpu[pair[key]] for key in ("source", "recipient"))
        if not torch.equal(a["gate"], b["gate"]):
            raise ValueError("cross-history gate differs")
        signature = ops["prefix_signature"](a["residual"], b["residual"], a["operands"], b["operands"], source["attention_width"], source["text_rows"])
        # This is P_T(C) from the recipient's own conditioning projection, not
        # an alternative future. The hybrid's text Act occurs after this cut.
        conditioning_cut = recipient["initial_act"]["parent"] if recipient["initial_act"] else recipient["initial"]
        initial_text = store.load(conditioning_cut, device="cpu").payload["text"].double()
        current_text = right.payload["text"].double()
        denominator = torch.linalg.vector_norm(initial_text) * torch.linalg.vector_norm(current_text)
        signature["recipient_conditioning_working_cosine"] = float((initial_text * current_text).sum() / denominator) if float(denominator) else 0.
        pairs[pair["id"]] = {**pair, "signature": signature, "closure_exceptions": sorted(exceptions)}

    a, b = source_cpu["anchor.base"], source_cpu["anchor.hybrid"]
    tr, width = histories["anchor.base"]["text_rows"], target.inner_dim
    ranked = ops["rank_supports"](a["operands"][..., :width], b["operands"][..., :width], tr, target.heads)
    text_energy = ((a["k"][:, :tr].double() - b["k"][:, :tr].double()).square()
                   + (a["v"][:, :tr].double() - b["v"][:, :tr].double()).square()).sum(dim=(0, 2, 3)).tolist()
    rankings = {"heads": ranked["heads"], "image": ranked["image_positions"],
                "text": sorted(range(tr), key=lambda i: (-text_energy[i], i))}
    supports = {axis: {"top": ranking[:max(1, len(ranking) // 4)],
                       "alternative": ranking[-max(1, len(ranking) // 4):]}
                for axis, ranking in rankings.items()}
    source = {"schema": "mdb-flux-family-sources-v1", "model_identity": adapter.model_identity,
              "checkpoint_sha256": identity(inventory), "histories": histories, "pairs": pairs,
              "support_selection": supports, "attention_contract": {"processor": type(target.processor).__name__,
                  "backend": None, "parallel_config": None, "Q": "installed image-query outputs; text outputs discarded"}}
    corner_checks = {}
    anchor_base, anchor_changed = read_ports("anchor.base"), read_ports("anchor.hybrid")
    for name, recipe, original in (
        ("changed", {}, anchor_changed),
        ("base", {key: "base" for key in ("q", "k_text", "v_text", "k_image", "v_image")}, anchor_base),
    ):
        selected = ops["attention_variant"](base=anchor_base, changed=anchor_changed, text_rows=tr, recipe=recipe)
        counts["attention_evaluations"] += 1
        if not torch.equal(selected["attention"], original["operands"][..., :width]):
            raise ValueError("native attention reconstruction corner differs")
        with torch.inference_mode():
            projected = torch.nn.functional.linear(original["operands"], target.to_out.weight, target.to_out.bias)
        counts["projection_evaluations"] += 1
        if not torch.equal(projected, original["projection"]):
            raise ValueError("native full-shape projection reconstruction differs")
        corner_checks[name] = {"attention": arrays("corner." + name, {"value": selected["attention"]}),
                               "projection_descriptor": tensor_descriptor(projected), "native_exact": True}
    source["corner_checks"] = corner_checks
    del anchor_base, anchor_changed, selected, projected
    save(output / "sources.json", source)
    save(output / "support_selection.json", supports)
    event("all_prefixes_ready", counts=dict(counts), source_sha256=sha(output / "sources.json"))

    def finish(current, row, inherited=()):
        if not future_released or (row["split"] == "heldout" and not heldout_released):
            raise ValueError("native suffix released before its seal")
        event("branch_start", row=row["id"], split=row["split"])
        schedulers = list(inherited)
        future_cuts = {}
        while current.point["logical_step"] < plan["steps"]:
            wall()
            if row["id"] in {"kv.KV", "break.image"}:
                for name, point in plan["rescue_points"].items():
                    if all(current.point[key] == value for key, value in point.items()):
                        future_cuts[name] = retain(current.capture())
            point = dict(current.point)
            current.step()
            if point["phase"] == "scheduler":
                value = current.session.read("latent")
                path = output / f"{row['id']}.step{current.point['logical_step']}.npy"
                np.save(path, value.float().cpu().numpy(), allow_pickle=False)
                schedulers.append({"step": current.point["logical_step"], "path": path.name,
                                   "sha256": sha(path), "descriptor": tensor_descriptor(value)})
        if len(schedulers) != 4 or counts["images"] >= plan["max_images"]:
            raise ValueError("scheduler/image work geometry differs")
        path = output / (row["id"] + ".png")
        adapter.decode(current.session, height=512, width=512).save(path)
        counts["images"] += 1
        row.update(schedulers=schedulers, future_cuts=future_cuts, image={"path": path.name, "sha256": sha(path)})
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        finished[row["id"]] = row
        event("branch_complete", row=row["id"], split=row["split"], counts=dict(counts))

    def install(current, image, row):
        parent_cut = current.capture()
        row["parent"] = retain(parent_cut)
        act = Act.replace("image", image, name=row["id"])
        rec = current.apply(act)
        post = current.capture()
        row.update(post_act=retain(post), act=act.manifest(), receipt=rec.to_dict())
        if any(describe(value) != describe(post.payload[key]) for key, value in parent_cut.payload.items() if key != "image"):
            raise ValueError("image Act changed undeclared closure")

    def execute_row(definition):
        row = dict(definition)
        if row["kind"] == "reference":
            current = nav(histories[row["history"]]["after"])
            row["parent"] = histories[row["history"]]["after"]
            finish(current, row)
            return
        pair = pairs[row["pair"]]
        source_history, recipient_history = (histories[pair[key]] for key in ("source", "recipient"))
        if row["kind"] == "rescue":
            broken, clean = finished["break.image"], finished["kv.KV"]
            point = row["when"]
            current = nav(broken["future_cuts"][point])
            donor_id = clean["future_cuts"][point]
            donor, parent = store.load(donor_id, device="cuda"), current.capture()
            selected_rows = supports["image"][row["support"]]
            image = parent.payload["image"].clone()
            image[:, selected_rows] = donor.payload["image"][:, selected_rows]
            row.update(donor=donor_id, selected_rows=selected_rows,
                       broken_row="break.image", clean_row="kv.KV")
            install(current, image, row)
            inherited = [dict(saved, inherited_from="break.image") for saved in broken["schedulers"] if saved["step"] <= current.point["logical_step"]]
            finish(current, row, inherited)
            return
        current = nav(recipient_history["after"])
        if row["kind"] == "copy":
            install(current, current.session.read("image"), row)
            finish(current, row)
            return
        base, changed = (read_ports(pair[key]) for key in ("source", "recipient"))
        tr, width = recipient_history["text_rows"], recipient_history["attention_width"]
        operands = changed["operands"]
        calculation = {"gate_history": pair["recipient"]}
        if row["kind"] == "merge":
            mask = row["mask"]
        else:
            recipe = dict(row.get("recipe", {"k_text": "base", "v_text": "base"}))
            receive_rows = None
            if row["kind"] == "support":
                selected_rows = supports[row["axis"]][row["side"]]
                if row["axis"] == "heads":
                    recipe["heads"] = selected_rows
                elif row["axis"] == "text":
                    recipe["text_positions"] = selected_rows
                else:
                    receive_rows = selected_rows
            selected = ops["attention_variant"](base={key: base[key] for key in ("q", "k", "v")},
                changed={key: changed[key] for key in ("q", "k", "v")}, text_rows=tr, recipe=recipe)
            counts["attention_evaluations"] += 1
            operands = operands.clone()
            if receive_rows is None:
                operands[:, tr:, :width] = selected["attention"][:, tr:]
            else:
                indices = [tr + i for i in receive_rows]
                operands[:, indices, :width] = selected["attention"][:, indices]
            if row["kind"] == "break":
                indices = [tr + i for i in supports["image"]["top"]]
                operands[:, indices, :width] = changed["operands"][:, indices, :width]
                calculation["omit_rows"] = supports["image"]["top"]
            calculation.update(recipe=recipe, receive_rows=receive_rows,
                attention=arrays(row["id"] + ".attention", {"value": selected["attention"]}),
                selected_ports={key: tensor_descriptor(selected[key]) for key in ("q", "k", "v")})
            mask = "100" if row.get("restore_residual", True) else "000"
        with torch.inference_mode():
            result = merge(recipient_residual=changed["residual"], donor_residual=base["residual"],
                recipient_operands=operands, donor_operands=base["operands"], gate=changed["gate"],
                projection=target.to_out, attention_width=width, text_rows=tr, mask=mask)
        counts["projection_evaluations"] += 1
        residual = changed["residual"].clone()
        if mask[0] == "1":
            residual[:, tr:] = base["residual"][:, tr:]
        calculation.update(operands_descriptor=tensor_descriptor(result["operands"]), residual_descriptor=tensor_descriptor(residual),
                           projection=arrays(row["id"] + ".projection", {"value": result["projection"]}))
        row["calculations"] = calculation
        install(current, result["updated"][:, tr:], row)
        del base, changed, result, operands
        finish(current, row)

    def transaction(name):
        # Full weights stay external and identity/version bound. Binding the
        # transformer here would clone its state_dict into the transaction.
        return create_transaction(output / ("transaction-" + name),
            model_identity=adapter.model_identity, input_identity=identity(plan), source_identity=identity(request["source_hashes"]),
            source_sha256=sha(output / "sources.json"), max_wall_seconds=plan["max_wall_seconds"])

    phase = "development_forecast"
    dev = transaction("development")
    with dev:
        dev_seal = seal_prediction(dev, "hypotheses", {"plan": plan, "source_sha256": sha(output / "sources.json")})
        if counts["images"] != 0 or initial_weight_versions != weight_versions():
            raise ValueError("development release before seal or weights drifted")
        event("development_sealed", seal=dev_seal, counts=dict(counts))
        future_released, phase = True, "development_execute"

        def run_development(context):
            for row in plan["rows"]:
                if row["split"] == "development":
                    execute_row(row)
            return {"rows": rows.copy(), "counts": dict(counts), "source_sha256": sha(output / "sources.json")}

        dev.execute("native", run_development)
    save(output / "transaction-development-report.json", dev.report())
    verify_dev = verify_transaction(output / "transaction-development", expected_seal_sha256=dev_seal["seal_sha256"])

    def final_latent(row_id):
        saved = next(value for value in finished[row_id]["schedulers"] if value["step"] == 4)
        return np.load(output / saved["path"], allow_pickle=False).astype(np.float64)

    def projection(row_id, pair):
        source = final_latent("ref." + pair["source"])
        recipient = final_latent("ref." + pair["recipient"])
        delta = recipient - source
        denominator = float(np.sum(delta * delta))
        return float(np.sum((final_latent(row_id) - source) * delta) / denominator) if denominator else None

    phase = "heldout_forecast"
    samples = [{"pair_id": pair["id"], "signature": pair["signature"],
                "targets": {action: projection(pair["id"] + "." + action, pair) for action in rules["ACTIONS"]}}
               for pair in pairs.values() if pair["split"] == "development"]
    resolver = rules["fit_resolver"](samples)
    recipient_resolver = rules["fit_resolver"](samples, features=rules["RECIPIENT_FEATURES"])
    predictions = {pair["id"]: rules["resolve"](resolver, pair["signature"])
                   for pair in pairs.values() if pair["split"] == "heldout"}
    recipient_predictions = {pair["id"]: rules["resolve"](recipient_resolver, pair["signature"])
                             for pair in pairs.values() if pair["split"] == "heldout"}
    frozen = {"model": resolver, "predictions": predictions, "development_rows": sorted(finished),
              "recipient_only_model": recipient_resolver, "recipient_only_predictions": recipient_predictions,
              "prefix_features": {pair["id"]: pair["signature"] for pair in pairs.values() if pair["split"] == "heldout"},
              "heldout_suffixes_before_seal": 0, "fixed_baseline": plan["hypotheses"]["merge"]}
    save(output / "resolver.json", frozen)
    heldout = transaction("heldout")
    with heldout:
        heldout_seal = seal_prediction(heldout, "applicability", frozen)
        if any(row["split"] == "heldout" for row in rows) or initial_weight_versions != weight_versions():
            raise ValueError("heldout quarantine or weight identity violated")
        event("heldout_sealed", seal=heldout_seal, predictions=predictions, counts=dict(counts))
        heldout_released, phase = True, "heldout_execute"

        def run_heldout(context):
            for row in plan["rows"]:
                if row["split"] == "heldout":
                    execute_row(row)
            return {"rows": [row for row in rows if row["split"] == "heldout"], "counts": dict(counts)}

        heldout.execute("native", run_heldout)
    save(output / "transaction-heldout-report.json", heldout.report())
    verify_heldout = verify_transaction(output / "transaction-heldout", expected_seal_sha256=heldout_seal["seal_sha256"])
    if counts["images"] != plan["max_images"] or {row["id"] for row in rows} != {row["id"] for row in plan["rows"]}:
        raise ValueError("declared panel inventory incomplete")
    if initial_weight_versions != weight_versions():
        raise ValueError("model parameters/buffers changed")
    for hook in hooks:
        hook.remove()
    summary = {"schema": "mdb-flux-six-panel-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counts, "model_identity": adapter.model_identity,
        "mechanics_status": "valid", "terminal_status": "unassessed", "scope": plan["scope"],
        "seals": {"development": dev_seal, "heldout": heldout_seal},
        "verification": {"development": verify_dev, "heldout": verify_heldout},
        "phase_native_entries": phase_entries, "weight_versions": {"initial": initial_weight_versions, "final": weight_versions()},
        "RGB_forecast": False, "worker_seconds": time.monotonic() - started}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    runpy.run_path("flux_family_custody.py")["emit"](output, summary, plan["limits"])


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-family-result")
        if output.is_dir():
            utility = runpy.run_path("flux_trajectory_forecast_campaign.py")
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            report = {"schema": "mdb-flux-six-panel-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                      "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            try:
                record_failure(output, report, save=utility["save"], file_sha=utility["file_sha"],
                    emit=runpy.run_path("flux_family_custody.py")["emit"], limits=request["config"]["limits"])
            except Exception as publication_error:
                print("MDB_FLUX_FAMILY_FAILURE_SUMMARY_V1 " + json.dumps({**report,
                    "publication_error": f"{type(publication_error).__name__}: {publication_error}"}, sort_keys=True), flush=True)
        raise
