"""Seal full image MLP fields, lower them into MDB branches, consume native RGB."""
from __future__ import annotations

import gc
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from pathlib import Path

CONTEXT = {"id": "p1-s71", "prompt_id": "p1", "seed": 71,
    "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
QUERIES = [0, 31, 32, 255, 511, 512, 767, 1023]
NEURONS = list(range(0, 9216, 288))
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 245,
    "acts": 11, "candidate_calls": 9, "receipt_emissions": 256, "images": 9, "vae_calls": 9, "physical_batch": 1}


def validate_config(config):
    helper = runpy.run_path(str(Path(__file__).with_name("flux_mlp_replacement.py")))
    if (config.get("context") != CONTEXT or config.get("height") != 512 or config.get("width") != 512
            or config.get("steps") != 4 or config.get("single_index") != 6
            or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 6}
            or config.get("query_indices") != QUERIES or config.get("neuron_indices") != NEURONS
            or config.get("neuron_chunk") != 256 or config.get("array_storage") != "bf16-bits"
            or config.get("archive_integer_ids") != "native-int64" or config.get("branches") != helper["BRANCHES"]):
        raise ValueError("full MLP replacement scientific geometry differs")
    for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls"):
        if type(config.get("max_" + key)) is not int or config["max_" + key] != COUNTS[key]:
            raise ValueError("full MLP replacement finite census differs: " + key)
    for key, lo, hi in (("max_wall_seconds", 1, 1200), ("cpu_threads", 4, 4), ("archive_compression_level", 1, 9)):
        if type(config.get(key)) is not int or not lo <= config[key] <= hi:
            raise ValueError("invalid full MLP replacement budget: " + key)


def run_panel(*, pipe, adapter, cuts, output, config, store, array, wall, event, counts, helper, capture, predictions):
    """One resident model; seven image Acts, matched native parents and suffixes."""
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    legacy = runpy.run_path("flux_single_contraction.py")
    util = runpy.run_path("flux_single_mlp_campaign.py")
    save, sha = util["save"], util["sha"]
    active, candidates, native, images, prefixes, post, rows, controls = {"branch": "direct"}, {}, {}, {}, {}, {}, [], {}
    def receipt(value):
        wall()
        body = value.to_dict()
        store.receipt(value)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"] - 112, "branch": active["branch"], "receipt": value.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
            raise RuntimeError("finite MLP replacement receipt budget exhausted")
    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=137, on_receipt=receipt)
    low, high = cuts["zero"], cuts["text"]
    if (low.model_identity != high.model_identity or low.execution != high.execution or set(low.payload) != set(high.payload)
            or any(describe(value) != describe(high.payload[key]) for key, value in low.payload.items() if key not in {"image", "text"})):
        raise ValueError("prepared sources differ outside working image/text")
    for cut in cuts.values():
        if any(cut.execution_point[key] != value for key, value in {"logical_step": 3, "phase": "single", "index": 6, "local_clock": 6}.items()):
            raise ValueError("source is not current-step pre-single6")
        store.save(cut)
    controls.update({key + "_source_exact": torch.equal(low.payload[key], high.payload[key]) for key in ("image", "text")})
    block = pipe.transformer.single_transformer_blocks[6]
    if not isinstance(block.attn.to_out, torch.nn.Linear):
        raise ValueError("replacement consumer is not the qualified native linear port")
    with capture(block, text_rows=low.payload["text"].shape[1], query_indices=QUERIES, neuron_indices=NEURONS) as captured:
        for spec in config["branches"]:
            wall()
            name, base, role = spec["id"], spec["base"], spec["source"]
            event("candidate_start", candidate=name)
            counts["candidate_calls"] += 1
            if spec["feature"] == "native":
                value = legacy["capture_candidate"](block, cuts[base])
                native[base] = value
                replacement = None
            else:
                reference = native[role]["ports"]["original_concat"][:, native[role]["text_rows"]:, native[role]["attention_width"]:]
                if spec["feature"] == "predicted":
                    payload = helper["prepare_mlp_payload"](predictions[role]["activation"], reference, roll=spec["roll"])
                else:
                    payload = torch.roll(reference, spec["roll"], dims=1) if spec["roll"] else reference
                value = helper["capture_replacement"](block, cuts[base], native[base], payload,
                    source_parent=cuts[role].fingerprint)
                replacement = {**{key: spec[key] for key in ("feature", "source", "roll")},
                    "payload": array(payload), "controls": value["replacement_controls"]}
            candidates[name] = {key: value[key] for key in ("input_parent", "source_parent", "selection", "text_rows", "attention_width", "equation_exact", "partition_controls")}
            candidates[name].update(ports={key: array(tensor) for key, tensor in value["ports"].items()}, replacement=replacement)
            images[name] = value["image"]
            event("candidate_complete", candidate=name)
        # Direct native captures and native navigator execution are independently compared.
        for role in ("zero", "text"):
            active["branch"] = "prefix." + role
            n = navigator(cuts[role])
            entry = n.step()
            post[role] = current = n.capture()
            store.save(current)
            prefixes[role] = {"input_cut": cuts[role].fingerprint, "post_single6": current.fingerprint, "receipt": entry.to_dict()}
            for slot in ("image", "text"):
                controls[role + "_candidate_" + slot + "_exact_native"] = torch.equal(native[role][slot], current.payload[slot])
        save(output / "prefixes.json", {"prefixes": prefixes})
        save(output / "candidates.json", {"candidates": candidates})
        for spec in config["branches"]:
            active["branch"] = spec["id"]
            n = navigator(post[spec["base"]])
            row = {**spec, "parent": post[spec["base"]].fingerprint, "start_point": n.point,
                "acts": [], "continuation_policy": "recipient-stored-C-no-further-project"}
            if spec["feature"] != "native":
                row["acts"].append(legacy["install_port"](n, "image", images[spec["id"]], name=spec["id"] + ".image", store=store))
            path = []
            while n.point["logical_step"] < 4:
                wall()
                path.append(n.step().to_dict())
            if len(path) != 15:
                raise ValueError("replacement suffix is not thirteen singles/readout/scheduler")
            final = n.capture()
            store.save(final)
            if describe(final.payload["conditioning"]) != describe(cuts[spec["base"]].payload["conditioning"]):
                raise ValueError("replacement suffix changed stored conditioning")
            counts["vae_calls"] += 1
            image = adapter.decode(n.session, height=512, width=512)
            file = output / (spec["id"] + ".png")
            image.save(file)
            counts["images"] += 1
            row.update(final_cut=final.fingerprint, final_latent=array(final.payload["latent"]),
                final_conditioning=tensor_descriptor(final.payload["conditioning"]), continuation_receipts=path,
                image={"path": file.name, "sha256": sha(file)})
            save(output / "rows" / (spec["id"] + ".json"), row)
            rows.append(row)
            event("branch_complete", row=spec["id"])
    observed = {role: {"call_index": row["call_index"], **{key: array(row[key]) for key in
        ("layernorm", "modulated_input", "u", "v", "activation")}} for role, row in captured["rows"].items()}
    save(output / "entry-observations.json", {"rows": observed, "calls": captured["calls"],
        "target_entries_before_seal": 0, "scope": "First two native source calls only; all nine direct calls and two native prefixes are counted."})
    by_id = {row["id"]: row for row in rows}
    controls["sham_png_equals_zero"] = by_id["sham"]["image"]["sha256"] == by_id["zero"]["image"]["sha256"]
    return {"rows": rows, "counts": dict(counts), "control_observations": controls,
        "source_sha256": sha(output / "source.json"), "candidates_sha256": sha(output / "candidates.json"),
        "entry_observations_sha256": sha(output / "entry-observations.json"), "target_calls": captured["calls"]}


def worker():
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    print("MDB_FLUX_MLP_REPLACEMENT_BOOTSTRAP " + json.dumps({"phase": "worker_enter", "unix_time": time.time()}), flush=True)
    util = runpy.run_path("flux_single_mlp_campaign.py")
    save, sha, identity, json_value = (util[key] for key in ("save", "sha", "identity", "json_value"))
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("replacement request/source bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-mlp-replacement-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"plan_sha256": sha(output / "plan.json"), "request_sha256": request["request_sha256"],
        "source_identity": identity(request["source_hashes"]), "job_id": request["job_id"],
        "work": {key: 0 for key in COUNTS if key != "physical_batch"}, "scope": config["scope"]})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    imports_started = time.monotonic()
    print("MDB_FLUX_MLP_REPLACEMENT_BOOTSTRAP " + json.dumps({"phase": "runtime_imports_start", "worker_seconds": imports_started - started}), flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.tensor_archive import write_tensor_archive

    helper = runpy.run_path("flux_mlp_replacement.py")
    mlp = runpy.run_path("flux_single_mlp.py")
    engine = runpy.run_path("flux_single_contexts.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("MLP replacement requires admitted CUDA")
    imports_finished, sequence = time.monotonic(), 0
    save(output / "startup.json", {"runtime_import_seconds": imports_finished - imports_started,
        "worker_preimport_seconds": imports_started - started, "clock_scope": "Worker entry through runtime imports; bootstrap environment resolution is outside this clock"})
    counts = {key: 0 for key in COUNTS}
    counts["physical_batch"] = 1
    progress = {"phase": "checkpoint_inventory", "wall_calls": 0, "last_heartbeat": imports_finished}
    def wall():
        now = time.monotonic()
        if now - started > config["max_wall_seconds"]:
            raise RuntimeError("finite full MLP replacement wall budget exhausted")
        progress["wall_calls"] += 1
        if now - progress["last_heartbeat"] >= 15:
            progress["last_heartbeat"] = now
            event("heartbeat", phase=progress["phase"], wall_calls=progress["wall_calls"])
    def event(kind, **fields):
        nonlocal sequence
        body = json_value({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_MLP_REPLACEMENT_PHASE " + json.dumps(body, sort_keys=True), flush=True)
    arrays = {}
    def array(value):
        wall()
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in arrays:
            arrays[key] = write_tensor_archive(output / "arrays" / (key + ".npy"), value, root=output)
        return arrays[key]
    root = Path(config["model_path"])
    event("checkpoint_inventory_start")
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("replacement checkpoint inventory differs")
    progress["phase"] = "model_loading"
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires deterministic five-joint/twenty-single distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
        (type(pipe), type(pipe.text_encoder), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[6]),
         type(pipe.transformer.single_transformer_blocks[6].attn.processor), type(pipe.vae), type(pipe.scheduler), Flux2KleinAdapter)}
    event("pipeline_loaded")
    progress["phase"] = "encoder_placement"
    pipe.text_encoder.to("cuda")
    event("encoder_resident")
    progress["phase"] = "prompt_encoding"
    wall()
    with torch.inference_mode():
        conditioning, txt_ids = pipe.encode_prompt(CONTEXT["prompt"], device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
    counts["prompt_encodings"] += 1
    event("encoding_complete")
    encodings = {"conditioning": array(conditioning), "txt_ids": array(txt_ids)}
    conditioning, txt_ids = conditioning.cpu(), txt_ids.cpu()
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    event("encoder_released")
    progress["phase"] = "denoiser_vae_placement"
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    event("denoiser_vae_resident")
    progress["phase"] = "adapter_construction"
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    source = {"schema": "mdb-flux-mlp-replacement-source-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory),
        "checkpoint_files": inventory, "execution": dict(adapter.execution), "implementation_sources": implementations,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "context": CONTEXT, "encoding": encodings, "encodings": encodings, "counts_after_encoding": dict(counts),
        "cpu_thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "blas_configuration": np.__config__.show(mode="dicts"), "scope": config["scope"]}
    event("prompt_encoded")
    progress["phase"] = "noise_schedule"
    event("noise_schedule_start")
    with torch.inference_mode():
        conditioning, txt_ids = conditioning.to("cuda"), txt_ids.to("cuda")
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512, conditioning.dtype,
            torch.device("cuda"), torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("replacement native schedule differs")
        event("noise_schedule_ready")
        progress["phase"] = "session_construction"
        event("session_start")
        session = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
            timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
    event("session_ready")
    progress["phase"] = "source_builder"
    event("source_builder_start")
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=output / "builder", context=CONTEXT,
        job_id=request["job_id"], wall=wall)
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key]
    event("source_builder_complete")
    builder_ref = {"path": "builder/builder.json", "sha256": sha(output / "builder/builder.json")}
    progress["phase"] = "source_preparation"
    event("source_preparation_start")
    cuts, preparation = util["prepare_sources"](adapter=adapter, cuts=cuts, output=output / "preparation", builder_ref=builder_ref, wall=wall)
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key] + preparation["counts"][key]
    preparation_ref = {"path": "preparation/preparation.json", "sha256": sha(output / "preparation/preparation.json")}
    source.update(builder=builder_ref, preparation=preparation_ref)
    save(output / "source.json", source)
    event("sources_prepared")
    progress["phase"] = "source_binding"
    block = pipe.transformer.single_transformer_blocks[6]
    h, i = 3072, 9216
    if block.norm.elementwise_affine or block.norm.eps != 1e-6 or tuple(block.norm.normalized_shape) != (h,):
        raise ValueError("native replacement normalization differs")
    fused = block.attn.to_qkv_mlp_proj
    if tuple(fused.weight.shape) != (3 * h + 2 * i, h):
        raise ValueError("native replacement fused projection profile differs")
    weight = fused.weight[3 * h:3 * h + 2 * i]
    bias = fused.bias[3 * h:3 * h + 2 * i] if fused.bias is not None else None
    binding = {"schema": "mdb-flux-mlp-replacement-binding-v1", "inputs": {role: {"cut": cut.fingerprint,
        "image": array(cut.payload["image"]), "single_mod": array(cut.payload["context"]["single_mod"])} for role, cut in cuts.items()},
        "weight_rows": array(weight), "bias_rows": array(bias) if bias is not None else None,
        "query_indices": QUERIES, "neuron_indices": NEURONS, "weight_row_range": [3 * h, 3 * h + 2 * i],
        "hidden_width": h, "mlp_width": i, "eps": block.norm.eps, "elementwise_affine": False, "neuron_chunk": config["neuron_chunk"],
        "linear_path": "single_transformer_blocks.6.attn.to_qkv_mlp_proj",
        "native_parameters": {name: tensor_descriptor(value) for name, value in fused.named_parameters()},
        "counts_before_seal": dict(counts), "target_entries_before_seal": 0, "builder": builder_ref, "preparation": preparation_ref,
        "scope": "Full image M[1,1024,9216], actual BF16 paired native weights and observed pre6 state/time. No native current-target norm/U/V/M/attention input. Earlier-timestep module6 calls allowed."}
    save(output / "binding.json", binding)
    operands = {role: {"image": np.array(cut.payload["image"].float().cpu().numpy(), copy=True),
        "mod": np.array(cut.payload["context"]["single_mod"].float().cpu().numpy(), copy=True)} for role, cut in cuts.items()}
    weight_np = np.array(weight.float().cpu().numpy(), copy=True)
    bias_np = np.array(bias.float().cpu().numpy(), copy=True) if bias is not None else None
    for value in [weight_np, *([bias_np] if bias_np is not None else []), *(value for row in operands.values() for value in row.values())]:
        value.setflags(write=False)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(binding),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": binding},
        execution_scope="Observed pre6 state to complete image M prediction, then native consumer substitution and nine native RGB futures; trusted callback dataflow and external CUDA census",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000, "max_bytes": 512 << 20})
    release = {"native": False, "blocked": 0, "target_calls": 0}
    def guard(module, args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("native model entered before full MLP forecast seal")
        if module is block:
            release["target_calls"] += 1
    handles = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    try:
        with tx:
            tx.frontier("source_binding", binding)
            progress["phase"] = "full_field_prediction"
            event("prediction_start")
            tx.predict("mlp_field", lambda _: {"predictions": {"cases": {role: helper["predict_mlp_field"](row["image"], row["mod"], weight_np, bias_np,
                eps=block.norm.eps, hidden_width=h, mlp_width=i, neuron_chunk=config["neuron_chunk"], query_indices=QUERIES, neuron_indices=NEURONS)
                for role, row in operands.items()}}})
            tx.predict("replacement_hypotheses", lambda _: {"predictions": config["hypotheses"]})
            seal = tx.seal()
            event("full_field_forecast_sealed", seal=seal, target_entries_before_seal=release["blocked"])
            release["native"] = True
            progress["phase"] = "native_panel"
            tx.execute("native", lambda context: run_panel(pipe=pipe, adapter=adapter, cuts=cuts, output=output, config=config,
                store=LocalStore(output / "typed-store"), array=array, wall=wall, event=event, counts=counts, helper=helper,
                capture=mlp["capture_source_calls"], predictions=context.predictions["mlp_field"]["cases"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output / "transaction-report.json", tx.report())
    if counts != COUNTS or release["target_calls"] != 11 or release["blocked"] != 0:
        raise ValueError("completed full MLP replacement physical census differs")
    summary = {"schema": "mdb-flux-mlp-replacement-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "counts": dict(counts), "builder_counts": {"adapter_steps": 108, "acts": 4, "receipt_emissions": 112},
        "panel_counts": {"adapter_steps": 137, "acts": 7, "receipt_emissions": 144, "candidate_calls": 9, "images": 9, "vae_calls": 9},
        "seal": seal, "transaction_verification": verify_transaction(tx.root), "model_identity": adapter.model_identity,
        "control_observations": tx.observation("native")["control_observations"], "target_calls": release["target_calls"],
        "worker_seconds": time.monotonic() - started, "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete")
    progress["phase"] = "archive_export"
    # Diagnostic stderr must not interleave with stdout custody transport.
    faulthandler.cancel_dump_traceback_later()
    print("MDB_FLUX_MLP_REPLACEMENT_ARCHIVE_START " + json.dumps({"raw_output_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()), "counts": counts}), flush=True)
    util["emit"](output, summary, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-mlp-replacement-result")
        if output.is_dir():
            util = runpy.run_path("flux_single_mlp_campaign.py")
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-mlp-replacement-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            util["save"](output / "failure.json", failure)
            util["emit"](output, failure, request["config"])
        raise
