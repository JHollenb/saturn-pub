"""Color/shape composition, predicted MLP consumers and matched endpoint cost."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if config != request["config"] or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("prediction-future source/config identity differs")
    if (config["seed"] != 60665 or config["height"] != 512 or config["width"] != 512
            or config["steps"] != 4 or set(config["prompts"]) != {"base", "color", "shape", "joint"}):
        raise ValueError("prediction-future scientific geometry differs")
    started = time.monotonic()
    payload_root = Path.cwd()
    helper = runpy.run_path(str(payload_root / "flux_predicted_future.py"))
    mlp = runpy.run_path(str(payload_root / "flux_mlp_replacement.py"))
    legacy = runpy.run_path(str(payload_root / "flux_single_contraction.py"))
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    output = Path("predicted-future-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "request.json", request)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import write_tensor_archive

    metric = helper["field_metrics"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("admitted CUDA is required")
    counts = {"loads": 0, "encodes": 0, "noise_draws": 0, "native_transitions": 0,
        "acts": 0, "receipts": 0, "source_blocks": 0, "validation_blocks": 0,
        "cached_projections": 0, "images": 0}
    receipts = []
    sequence = 0

    def event(kind, **values):
        nonlocal sequence
        row = {"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **values}
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_PREDICTED_FUTURE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite prediction-future budget exhausted")

    def receipt(value):
        body = value.to_dict()
        counts["receipts"] += 1
        counts["native_transitions" if body["operation"] == "continue" else "acts"] += 1
        receipts.append(body)
        if counts["native_transitions"] > 523 or counts["acts"] > 5:
            raise RuntimeError("finite transition/Act budget exhausted")

    def array(name, value):
        return write_tensor_archive(output / "arrays" / (name + ".npy"), value, root=output)

    def cpu(value):
        return value.detach().float().cpu().numpy()

    def sync():
        torch.cuda.synchronize()

    root = Path(config["model_path"])
    event("inventory")
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires deterministic distilled Klein 5/20 architecture")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    encodings = {}
    for role in ("base", "color", "shape", "joint"):
        wall()
        with torch.inference_mode():
            e, ids = pipe.encode_prompt(config["prompts"][role], device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
        encodings[role] = (e.cpu(), ids.cpu())
        counts["encodes"] += 1
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store = LocalStore(output / "store")
    source = {"model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory),
        "checkpoint_files": inventory, "execution": dict(adapter.execution),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "mdb-runtime", "saturn-pub")},
        "implementations": {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
            (type(pipe), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[6]), Flux2KleinAdapter)}}
    save(output / "source.json", source)
    with torch.inference_mode():
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512, torch.bfloat16,
            torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, _steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
    pre, navs = {}, {}
    for role, (e, ids) in encodings.items():
        session = adapter.session(latent=latent.clone(), conditioning=e.to("cuda"), img_ids=img_ids,
            txt_ids=ids.to("cuda"), timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
        nav = NativeCutNavigator(session, max_steps=112, on_receipt=receipt)
        while not (nav.point["logical_step"] == 3 and nav.point["phase"] == "single" and nav.point["index"] == 6):
            wall()
            nav.step()
        pre[role], navs[role] = nav.capture(), nav
        store.save(pre[role])
        event("prefix_captured", role=role, parent=pre[role].fingerprint)
    block = pipe.transformer.single_transformer_blocks[6]
    fused = block.attn.to_qkv_mlp_proj
    weight = fused.weight[9216:].detach()
    bias = None if fused.bias is None else fused.bias[9216:].detach()
    if tuple(weight.shape) != (18432, 3072) or block.norm.elementwise_affine or block.norm.eps != 1e-6:
        raise ValueError("MLP parameter/normalization profile differs")
    binding = {"inputs": {role: {"parent": cut.fingerprint, "image": array(role + ".entry", cut.payload["image"]),
        "mod": array(role + ".mod", cut.payload["context"]["single_mod"])} for role, cut in pre.items()},
        "weight": array("weight", weight), "bias": array("bias", bias) if bias is not None else None,
        "eps": block.norm.eps, "weight_identity": tensor_descriptor(fused.weight),
        "observed_inputs": "Actual current incoming state/modulation/weights, including joint prefix; no current-site outputs",
        "counts_before_forecast": dict(counts)}
    save(output / "binding.json", binding)
    predicted, forecast_records, prediction_seconds = {}, {}, {}
    entered = {"current": 0, "released": False}

    def guard(module, args):
        if not entered["released"]:
            entered["current"] += 1
            raise ValueError("current target entered before prediction seal")

    guard_handle = block.register_forward_pre_hook(guard)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(binding),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": binding},
        execution_scope="Complete incoming-state M prediction; actual joint incoming prefix is observed. Frozen base-attention consumer and native suffixes.",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    with tx:
        tx.frontier("binding", binding)

        def forecast(_):
            weight_np, bias_np = cpu(weight), None if bias is None else cpu(bias)
            for role, cut in pre.items():
                wall()
                tick = time.monotonic()
                values = mlp["predict_mlp_field"](cpu(cut.payload["image"]), cpu(cut.payload["context"]["single_mod"]),
                    weight_np, bias_np, eps=block.norm.eps, neuron_chunk=256)
                prediction_seconds[role] = time.monotonic() - tick
                predicted[role] = torch.from_numpy(np.array(values["activation"], copy=True)).to("cuda", torch.bfloat16)
                forecast_records[role] = array(role + ".predicted", predicted[role])
            additive = mlp["_bf16"](cpu(predicted["color"]).astype(np.float64) + cpu(predicted["shape"]) - cpu(predicted["base"]))
            predicted["additive"] = torch.from_numpy(np.array(additive, copy=True)).to("cuda", torch.bfloat16)
            forecast_records["additive"] = array("additive.predicted", predicted["additive"])
            save(output / "forecasts.json", {"fields": forecast_records, "prediction_seconds": prediction_seconds,
                "joint_output_observed": False, "joint_incoming_state_observed": True,
                "additive_rule": "B(M_color + M_shape - M_base); no joint field in this rule"})
            return {"predictions": {"fields": forecast_records, "prediction_seconds": prediction_seconds,
                "semantic_predictions": config["semantic_predictions"]}}

        tx.predict("fields", forecast)
        seal = tx.seal()
        event("sealed", seal=seal, blocked_entries=entered["current"])
        entered["released"] = True
        guard_handle.remove()

        def execute(_):
            native, measured, post, errors = {}, {}, {}, {}
            for role, cut in pre.items():
                wall()
                native[role] = legacy["capture_candidate"](block, cut)
                counts["source_blocks"] += 1
                n = native[role]
                measured[role] = n["ports"]["original_concat"][:, n["text_rows"]:, n["attention_width"]:]
                errors[role] = metric(cpu(predicted[role]), cpu(measured[role]))
                errors[role]["native_field"] = array(role + ".measured", measured[role])
                navs[role].step()
                post[role] = navs[role].capture()
                if not torch.equal(post[role].payload["image"], n["image"]) or not torch.equal(post[role].payload["text"], n["text"]):
                    raise ValueError("candidate/native prefix differs")
                store.save(post[role])
            base = native["base"]
            save(output / "base-consumer.json", {"concat": array("base.concat", base["ports"]["original_concat"]),
                "residual": array("base.residual", base["ports"]["residual"]), "gate": array("base.gate", base["ports"]["gate"]),
                "output_weight": array("output.weight", block.attn.to_out.weight),
                "output_bias": None if block.attn.to_out.bias is None else array("output.bias", block.attn.to_out.bias),
                "text_rows": base["text_rows"], "parent": pre["base"].fingerprint})
            fields = {"sham": measured["base"], "transfer.measured": measured["joint"],
                "transfer.predicted": predicted["joint"], "transfer.additive": predicted["additive"],
                "transfer.roll": torch.roll(predicted["joint"], 512, dims=1)}
            cached, consumer_checks = {}, {}
            for name, values in fields.items():
                cached[name] = helper["consume_cached"](block, pre["base"], base, values)
                counts["cached_projections"] += 1
            # Validate every endpoint in the dose pool; validation is counted work,
            # not a claimed physical saving in this exploratory run.
            grid, times = [], {"full_block": [], "cached_projection": []}
            original_m = measured["base"]
            for dose in helper["dose_values"]():
                wall()
                values = (original_m.float() + dose * (predicted["joint"].float() - original_m.float())).to(torch.bfloat16)
                sync()
                tick = time.monotonic()
                fast = helper["consume_cached"](block, pre["base"], base, values)
                sync()
                fast_seconds = time.monotonic() - tick
                counts["cached_projections"] += 1
                tick = time.monotonic()
                full = mlp["capture_replacement"](block, pre["base"], base, values,
                    capture_path=str(payload_root / "flux_single_contraction.py"))
                sync()
                full_seconds = time.monotonic() - tick
                counts["validation_blocks"] += 1
                exact_i, exact_t = torch.equal(fast["image"], full["image"]), torch.equal(fast["text"], full["text"])
                consumer_checks[str(dose)] = {"image_exact": exact_i, "text_exact": exact_t}
                times["full_block"].append(full_seconds)
                times["cached_projection"].append(fast_seconds)
                grid.append({"dose": dose, "exact_image": exact_i, "exact_text": exact_t,
                    "cached_seconds": fast_seconds, "full_block_seconds": full_seconds,
                    "image_effect_l2": float(torch.linalg.vector_norm((fast["image"] - base["image"]).float()).item())})
                del full, fast
            save(output / "dose-pool.json", {"rows": grid, "times": times,
                "scope": "17 same-parent M-only endpoints; all17 full blocks executed as separately counted validation. Zero pool VAE/suffix calls. Cached kernel includes clone, footprint checks, native to_out and gated addition; full baseline uses instrumented capture helper, not optimized production inference."})
            save(output / "field-errors.json", errors)
            interaction = predicted["joint"].float() - predicted["color"].float() - predicted["shape"].float() + predicted["base"].float()
            save(output / "interaction.json", {"predicted_field_interaction_l2": float(torch.linalg.vector_norm(interaction).item()),
                "additive_vs_joint": metric(cpu(predicted["additive"]), cpu(predicted["joint"])),
                "scope": "Interaction between different observed incoming native histories; no claim of an isolated semantic algebra"})
            rows = []
            def continuation(name, nav, act=None):
                path = []
                while nav.point["logical_step"] < 4:
                    wall()
                    path.append(nav.step().to_dict())
                cut = nav.capture()
                store.save(cut)
                image = adapter.decode(nav.session, height=512, width=512)
                image.save(output / (name + ".png"))
                counts["images"] += 1
                row = {"id": name, "final_cut": cut.fingerprint, "final_latent": array(name + ".latent", cut.payload["latent"]),
                    "path": path, "act": act, "png": name + ".png", "png_sha256": sha(output / (name + ".png"))}
                rows.append(row)
                event("image_complete", branch=name)
            for role in pre:
                continuation(role, navs[role])
            for name in fields:
                nav = NativeCutNavigator(Session.from_cut(adapter, post["base"]), max_steps=15, on_receipt=receipt)
                act = legacy["install_port"](nav, "image", cached[name]["image"], name=name, store=store)
                continuation(name, nav, act)
            save(output / "rows.json", rows)
            save(output / "receipts.json", receipts)
            save(output / "consumer-checks.json", consumer_checks)
            return {"counts": dict(counts), "field_errors_sha256": sha(output / "field-errors.json"),
                "rows_sha256": sha(output / "rows.json"), "consumer_checks": consumer_checks}

        tx.execute("native_futures", execute)
    summary = {"schema": "mdb-predicted-future-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counts, "seal": seal,
        "transaction_verification": verify_transaction(tx.root), "worker_seconds": time.monotonic() - started,
        "model_identity": adapter.model_identity, "mechanics_status": "pending-independent-audit",
        "terminal_status": "unassessed", "scope": config["scope"]}
    if counts != {"loads": 1, "encodes": 4, "noise_draws": 1, "native_transitions": 523,
            "acts": 5, "receipts": 528, "source_blocks": 4, "validation_blocks": 17, "cached_projections": 22, "images": 9}:
        raise ValueError("completed prediction-future physical census differs: " + str(counts))
    save(output / "summary.json", summary)
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    event("complete")
    # timeline changes above, so refresh the complete manifest before transport.
    (output / "files.json").unlink()
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload_root / "flux_family_custody.py"))["emit"](output, summary, compression_level=1)


if __name__ == "__main__":
    worker()
