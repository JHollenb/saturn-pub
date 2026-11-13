"""Factor a localized native parallel merge into residual/attention/MLP operands."""
from __future__ import annotations

import itertools
import json
import os
import runpy
import time
from pathlib import Path

RECIPIENT_GAINS = {"11": (1., 1.), "10": (1., 0.), "01": (0., 1.), "00": (0., 0.),
                   "1h": (1., .5), "h1": (.5, 1.)}


def validate_shape_writer_config(config):
    if config.get("panel") not in {"shape-step1-writer", "shape-step1-recipient-controls"}:
        return False
    block = 5 if config["panel"] == "shape-step1-recipient-controls" else 4
    if (config.get("prefix_selector") != {"logical_step": 1, "phase": "single", "index": 0}
            or config.get("selector") != {"logical_step": 1, "phase": "single", "index": block}
            or config.get("reset_step") != 2 or config.get("block") != block
            or config.get("fresh_context") or config.get("attention_sources")):
        raise ValueError("shape writer requires the declared step1 single0-to-selected-single prefix and step2 reset")
    if block == 5 and config.get("recipient_gain_rows", {"self." + name: list(gains) for name, gains in RECIPIENT_GAINS.items()}) != {
            "self." + name: list(gains) for name, gains in RECIPIENT_GAINS.items()}:
        raise ValueError("recipient gain rows differ from the declared six controls")
    return True


def requires_conditioning_reset(config, row):
    if config.get("panel") == "shape-step1-recipient-controls":
        policy = row.get("continuation_policy", "common-base-reset")
        if policy not in {"common-base-reset", "recipient-own-C"}:
            raise ValueError("unknown recipient continuation policy")
        return policy == "common-base-reset"
    return config.get("panel") == "shape-step1-writer"


def matched_shape_closure(cuts, *, selector):
    """Only this experiment permits a stored-conditioning mismatch."""
    from saturn_pub.values import describe

    if set(cuts) != {"base", "changed"}:
        raise ValueError("shape writer requires base and changed origins")
    base, changed = cuts["base"], cuts["changed"]
    if (base.model_identity != changed.model_identity or dict(base.execution) != dict(changed.execution)
            or base.execution_point != changed.execution_point
            or any(base.execution_point.get(key) != value for key, value in selector.items())):
        raise ValueError("shape writer source model/execution/clock differs")
    left, right = base.payload, changed.payload
    if set(left) != set(right):
        raise ValueError("shape writer source closure inventory differs")
    for slot in left:
        if slot not in {"text", "image", "conditioning"} and describe(left[slot]) != describe(right[slot]):
            raise ValueError("shape writer unmatched nuisance closure: " + slot)


def advance_shape_sources(adapter, origins, *, prefix_selector, selector, store, max_steps, on_receipt=None):
    """Advance original native handoff cuts to selected ports with public lineage."""
    from saturn_pub import Session

    from mdb_lanes.native_navigation import NativeCutNavigator

    matched_shape_closure(origins, selector=prefix_selector)
    selected, receipts = {}, {}
    for name in ("base", "changed"):
        origin = origins[name]
        store.save(origin)
        receipts[name] = []

        def accept(receipt):
            store.receipt(receipt)
            receipts[name].append(receipt.to_dict())
            if on_receipt is not None:
                on_receipt(receipt)

        nav = NativeCutNavigator(Session.from_cut(adapter, origin), max_steps=max_steps, on_receipt=accept)
        nav.until(**selector)
        selected[name] = nav.capture()
        store.save(selected[name])
    matched_shape_closure(selected, selector=selector)
    return selected, receipts


def reset_shape_conditioning(nav, value, *, step, name, source_cut, store):
    """Typed C reset immediately before the declared next projection."""
    from saturn_pub import Act
    from saturn_pub.values import describe

    point = nav.point
    if (point.get("logical_step") != step or point.get("phase") != "project"
            or point.get("local_clock") != 0 or point.get("index") is not None):
        raise ValueError("shape source reset requires the declared project boundary")
    before = nav.capture()
    store.save(before)
    expected = dict(before.payload)
    expected["conditioning"] = value
    act = Act.replace("conditioning", value, name=name)
    receipt = nav.apply(act)
    store.receipt(receipt)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe(expected):
        raise ValueError("shape source reset changed undeclared closure")
    return {"act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": receipt.fingerprint,
            "parent": before.fingerprint, "post_act": after.fingerprint, "point": point,
            "source_cut": source_cut, "undeclared_slots_exact": True}


def worker():
    support = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, file_sha, identity = (support[key] for key in ("save", "file_sha", "identity"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(file_sha(name) != value for name, value in request["source_hashes"].items()):
        raise ValueError("request/source differs")
    shape_mode = validate_shape_writer_config(config)
    recipient_mode = config.get("panel") == "shape-step1-recipient-controls"
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

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    candidate = runpy.run_path("flux_parallel_merge.py")["merge_candidate"]
    own_candidate = runpy.run_path("flux_parallel_merge.py")["recipient_candidate"] if recipient_mode else None
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

    initial, origins, advance_receipts = {}, {}, {}
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
        if shape_mode:
            origins = before
            before, advance_receipts = advance_shape_sources(adapter, origins,
                prefix_selector=config["prefix_selector"], selector=config["selector"], store=store,
                max_steps=config["max_adapter_steps"], on_receipt=on_receipt)
            event("shape_writer_prefix_ready", origin_cuts={name: cut.fingerprint for name, cut in origins.items()},
                input_cuts={name: cut.fingerprint for name, cut in before.items()}, counts=dict(counts))
    for cut in before.values():
        store.save(cut)
        if cut.model_identity != adapter.model_identity or any(cut.execution_point[k] != v for k, v in config["selector"].items()):
            raise ValueError("input model or selected boundary differs")
    for slot in before["base"].payload:
        allowed = {"text", "image", "conditioning"} if shape_mode else {"text", "image"}
        if slot not in allowed and describe(before["base"].payload[slot]) != describe(before["changed"].payload[slot]):
            raise ValueError("unmatched source closure: " + slot)

    target = pipe.transformer.single_transformer_blocks[config["block"]].attn
    if attention_helper is not None and (type(target.processor).__name__ != "Flux2ParallelSelfAttnProcessor"
            or target.processor._attention_backend is not None or target.processor._parallel_config is not None):
        raise ValueError("attention intervention requires native default backend without parallel configuration")
    projection, ports, after, prefix_receipts = target.to_out, {}, {}, {}
    text_rows = before["base"].payload["text"].shape[1]
    gate = Flux2Modulation.split(before["base"].payload["context"]["single_mod"], 1)[0][2]
    own_gate = Flux2Modulation.split(before["changed"].payload["context"]["single_mod"], 1)[0][2] if recipient_mode else None
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
    if shape_mode:
        source.update(origin_cuts={name: cut.fingerprint for name, cut in origins.items()},
            advance_receipts=advance_receipts, prefix_selector=config["prefix_selector"],
            conditioning_reset={"step": config["reset_step"], "phase": "project", "slot": "conditioning",
                "source_cut": origins["base"].fingerprint,
                "value_descriptor": tensor_descriptor(origins["base"].payload["conditioning"])})
    if recipient_mode:
        source.update(recipient_controls={"gains_A_M": {name: {"A": gains[0], "M": gains[1]} for name, gains in RECIPIENT_GAINS.items()},
            "native_reference": "self.native", "identity_reference": "self.11", "recipe_mode": "recipient-only",
            "gate_source_cut": before["changed"].fingerprint, "gate_descriptor": tensor_descriptor(own_gate),
            "scope": "own residual/operands/gate/full original projection; no donor tensor inputs or donor lookup in recipe"},
            continuation_policies={"common-base-reset": source["conditioning_reset"],
                "recipient-own-C": {"source_cut": origins["changed"].fingerprint, "reset": None,
                    "value_descriptor": tensor_descriptor(origins["changed"].payload["conditioning"])}})
    save(output / "source.json", source)
    allowed_prefix = ("declared original step1 handoff cuts advanced natively to single5, then two selected-block observations; no RGB outcomes"
        if recipient_mode else "declared original step1 handoff cuts advanced natively to single4, then two selected-block observations; no RGB outcomes"
        if shape_mode else "declared fresh text encodings/projection and matched prefixes, then two selected-block port observations; no RGB outcomes"
        if initial else "two native selected-block executions to observe actual merge ports; no RGB outcomes")
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        frontier={"kind": "architectural", "source_sha256": file_sha(output / "source.json"),
            "allowed_prefix": allowed_prefix},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("merge_hypotheses", lambda context: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows = []

    def finish(session, row):
        common_reset = requires_conditioning_reset(config, row)
        if recipient_mode:
            row.setdefault("continuation_policy", "common-base-reset")
            row.setdefault("recipe_mode", "reference" if row["kind"] == "reference" else "copy-sham" if row["kind"] == "copy" else "donor-merge")
            if row["continuation_policy"] not in {"common-base-reset", "recipient-own-C"}:
                raise ValueError("unknown recipient continuation policy")
            start = session.capture()
            row["conditioning_at_start"] = tensor_descriptor(start.payload["conditioning"])
            row["conditioning_source_cut"] = origins["base" if row["id"] == "base" else "changed"].fingerprint
            if not common_reset:
                row["source_cut"] = before["changed"].fingerprint
                row["own_conditioning"] = tensor_descriptor(start.payload["conditioning"])
        event("branch_start", row=row["id"])
        schedulers, continuation_acts = [], []
        while session.point["logical_step"] < config["steps"]:
            wall()
            point = session.point
            if common_reset and point["logical_step"] == config["reset_step"] and point["phase"] == "project":
                continuation_acts.append(reset_shape_conditioning(session, origins["base"].payload["conditioning"],
                    step=config["reset_step"], name=row["id"] + ".future-source-base",
                    source_cut=origins["base"].fingerprint, store=store))
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
        if shape_mode:
            if len(continuation_acts) != int(common_reset):
                raise ValueError("shape branch future-source reset count differs from policy")
            final = session.capture()
            store.save(final)
            row.update(continuation_acts=continuation_acts, final_cut=final.fingerprint)
            if recipient_mode and not common_reset and describe(final.payload["conditioning"]) != describe(origins["changed"].payload["conditioning"]):
                raise ValueError("recipient-only future changed its own conditioning")
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
            masks = ("000", "001", "010", "011") if recipient_mode else ("".join(bits) for bits in itertools.product("01", repeat=3))
            for mask in masks:
                wall()
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
            if recipient_mode:
                finish(nav(after["changed"]), {"id": "self.native", "kind": "reference", "parent": after["changed"].fingerprint,
                    "continuation_policy": "recipient-own-C", "recipe_mode": "recipient-native"})
                for name, gains in RECIPIENT_GAINS.items():
                    wall()
                    with torch.inference_mode():
                        result = own_candidate(residual=residuals["changed"], operands=ports["changed"]["operands"],
                            gate=own_gate, projection=projection, attention_width=target.inner_dim, text_rows=text_rows, gains=gains)
                    counts["projection_evaluations"] += 1
                    native_changed = torch.cat([after["changed"].payload["text"], after["changed"].payload["image"]], dim=1)
                    row = {"id": "self." + name, "kind": "self", "recipe_mode": "recipient-only-scale",
                        "continuation_policy": "recipient-own-C", "gains": list(gains),
                        "operands_descriptor": tensor_descriptor(result["operands"]),
                        "projection": save_arrays("self." + name + ".projection", {"value": result["projection"]}),
                        "text_rows_exact_changed": torch.equal(result["updated"][:, :text_rows], native_changed[:, :text_rows]),
                        "self_full_state_exact_changed": torch.equal(result["updated"], native_changed) if name == "11" else None}
                    install(result["updated"][:, text_rows:], row)
                expected_rows = {"base", "changed", "copy.sham", *("merge." + mask for mask in ("000", "001", "010", "011")),
                                 "self.native", *("self." + name for name in RECIPIENT_GAINS)}
                if len(rows) != 14 or {row["id"] for row in rows} != expected_rows:
                    raise ValueError("recipient-control row inventory differs")
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
