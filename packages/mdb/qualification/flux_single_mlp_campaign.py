"""Fresh-seed single6 MLP entry forecast followed by nine native futures."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

CONTEXT = {"id": "p1-s67", "prompt_id": "p1", "seed": 67,
    "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
QUERIES = [0, 31, 32, 255, 511, 512, 767, 1023]
NEURONS = list(range(0, 9216, 288))
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 245,
    "acts": 14, "candidate_calls": 5, "receipt_emissions": 259, "images": 9, "vae_calls": 9, "physical_batch": 1}


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if (config.get("context") != CONTEXT or config.get("height") != 512 or config.get("width") != 512
            or config.get("steps") != 4 or config.get("single_index") != 6
            or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 6}
            or config.get("query_indices") != QUERIES or config.get("neuron_indices") != NEURONS
            or config.get("output_dims") != list(range(0, 3072, 96)) or config.get("array_storage") != "bf16-bits"
            or config.get("archive_integer_ids") != "native-int64"):
        raise ValueError("fresh MLP-entry scientific geometry differs")
    helper = runpy.run_path(str(Path(__file__).with_name("flux_single_contraction.py")))
    if config.get("branches") != helper["BRANCHES"]:
        raise ValueError("fresh MLP-entry branch recipes differ")
    for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls"):
        if type(config.get("max_" + key)) is not int or config["max_" + key] != COUNTS[key]:
            raise ValueError("fresh MLP-entry finite census differs: " + key)
    for key, low, high in (("max_wall_seconds", 1, 780), ("cpu_threads", 1, 16), ("archive_compression_level", 1, 9)):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("invalid fresh MLP-entry budget: " + key)


def emit(output, report, config):
    manifest = output / "files.json"
    if manifest.exists():
        manifest.unlink()
    save(manifest, {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    runpy.run_path("flux_family_custody.py")["emit"](output, json_value(report),
        limits=config.get("transport_limits"), compression_level=config["archive_compression_level"])


def prepare_sources(*, adapter, cuts, output, builder_ref, wall):
    """12 native transitions and one image Act, after the unchanged 96/3 builder."""
    from saturn_pub import Session
    from saturn_pub.store import LocalStore

    from mdb.native_navigation import NativeCutNavigator

    engine = runpy.run_path("flux_single_contexts.py")
    output.mkdir(parents=True, exist_ok=False)
    store, emissions = LocalStore(output / "typed-store"), []
    counts = {"adapter_steps": 0, "acts": 0, "receipt_emissions": 0}
    def receipt(value):
        body = value.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        counts["receipt_emissions"] += 1
        if counts["adapter_steps"] > 12 or counts["acts"] > 1:
            raise RuntimeError("finite source-preparation budget exhausted")
        store.receipt(value)
        emissions.append({"sequence": len(emissions), "receipt": value.fingerprint})
    navs = {role: NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=12, on_receipt=receipt) for role, cut in cuts.items()}
    first = {}
    for role in ("zero", "text"):
        wall()
        store.save(cuts[role])
        entry = navs[role].step()
        post = navs[role].capture()
        store.save(post)
        first[role] = {"input_cut": cuts[role].fingerprint, "post_cut": post.fingerprint, "receipt": entry.to_dict()}
    act = engine["replace_slot"](navs["zero"], "image", navs["text"].capture().payload["image"],
        name="prepare.low-post0-image-from-own-high-history", store=store)
    matched = {role: nav.capture().fingerprint for role, nav in navs.items()}
    paths = {}
    for role in ("zero", "text"):
        paths[role] = []
        for _ in range(5):
            wall()
            paths[role].append(navs[role].step().to_dict())
    result = {role: nav.capture() for role, nav in navs.items()}
    for cut in result.values():
        store.save(cut)
        if any(cut.execution_point[key] != value for key, value in {"logical_step": 3, "phase": "single", "index": 6, "local_clock": 6}.items()):
            raise ValueError("prepared source is not current-step pre-single6")
    if counts != {"adapter_steps": 12, "acts": 1, "receipt_emissions": 13}:
        raise ValueError("source preparation completed census differs")
    record = {"schema": "mdb-flux-single-mlp-preparation-v1", "builder": builder_ref, "first_single0": first,
        "image_act": act, "matched_post0": matched, "paths": paths, "counts": counts,
        "input_cuts": {role: cut.fingerprint for role, cut in cuts.items()},
        "pre6_cuts": {role: cut.fingerprint for role, cut in result.items()},
        "scope": "Direct post0 matched-image/different-text construction followed by native1..5. No new K/V equivalence assay; earlier-step module6 calls are allowed builder work."}
    save(output / "preparation.json", record)
    save(output / "receipt-emissions.json", emissions)
    return result, record


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("fresh MLP-entry request/source bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-single-mlp-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"plan_sha256": sha(output / "plan.json"), "request_sha256": request["request_sha256"],
        "source_identity": identity(request["source_hashes"]), "job_id": request["job_id"],
        "work": {key: 0 for key in COUNTS if key != "physical_batch"},
        "scope": "Fixed fresh seed, queries and neurons before loading. Early MLP forecast is conditional on observed pre6 state; no raw-prompt image forecast."})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.tensor_archive import write_tensor_archive

    engine = runpy.run_path("flux_single_contexts.py")
    panel_engine = runpy.run_path("flux_single_contraction_campaign.py")
    mlp = runpy.run_path("flux_single_mlp.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("fresh MLP-entry panel requires admitted CUDA")
    started, sequence = time.monotonic(), 0
    counts = {key: 0 for key in COUNTS}
    counts["physical_batch"] = 1
    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite fresh MLP-entry wall budget exhausted")
    def event(kind, **fields):
        nonlocal sequence
        value = json_value({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_SINGLE_MLP_PHASE " + json.dumps(value, sort_keys=True), flush=True)
    arrays = {}
    def array(value):
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in arrays:
            arrays[key] = write_tensor_archive(output / "entry-arrays" / (key + ".npy"), value, root=output)
        return arrays[key]
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("fresh MLP-entry checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires deterministic five-joint/twenty-single distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    implementation_sources = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
        (type(pipe), type(pipe.text_encoder), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[6]),
         type(pipe.transformer.single_transformer_blocks[6].attn.processor), type(pipe.vae), type(pipe.scheduler), Flux2KleinAdapter)}
    pipe.text_encoder.to("cuda")
    wall()
    with torch.inference_mode():
        conditioning, txt_ids = pipe.encode_prompt(CONTEXT["prompt"], device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
    counts["prompt_encodings"] += 1
    encodings = {"conditioning": array(conditioning), "txt_ids": array(txt_ids)}
    conditioning, txt_ids = conditioning.cpu(), txt_ids.cpu()
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    source = {"schema": "mdb-flux-single-mlp-source-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory),
        "checkpoint_files": inventory, "execution": dict(adapter.execution), "implementation_sources": implementation_sources,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "context": CONTEXT, "encodings": encodings, "counts_after_encoding": dict(counts), "scope": config["scope"]}
    event("prompt_encoded")
    with torch.inference_mode():
        conditioning, txt_ids = conditioning.to("cuda"), txt_ids.to("cuda")
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512, conditioning.dtype,
            torch.device("cuda"), torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("fresh native schedule differs")
        session = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
            timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=output / "builder", context=CONTEXT,
        job_id=request["job_id"], wall=wall)
    builder_ref = {"path": "builder/builder.json", "sha256": sha(output / "builder/builder.json")}
    cuts, preparation = prepare_sources(adapter=adapter, cuts=cuts, output=output / "preparation", builder_ref=builder_ref, wall=wall)
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key] + preparation["counts"][key]
    preparation_ref = {"path": "preparation/preparation.json", "sha256": sha(output / "preparation/preparation.json")}
    source.update(builder=builder_ref, preparation=preparation_ref, encoding=encodings)
    save(output / "source.json", source)
    event("sources_prepared")
    block = pipe.transformer.single_transformer_blocks[6]
    hidden_width, mlp_width = 3072, 9216
    if block.norm.elementwise_affine or block.norm.eps != 1e-6 or tuple(block.norm.normalized_shape) != (hidden_width,):
        raise ValueError("native MLP entry normalization differs")
    rows = [3 * hidden_width + j for j in NEURONS] + [3 * hidden_width + mlp_width + j for j in NEURONS]
    fused = block.attn.to_qkv_mlp_proj
    if tuple(fused.weight.shape) != (3 * hidden_width + 2 * mlp_width, hidden_width):
        raise ValueError("native fused QKV/MLP projection profile differs")
    with torch.inference_mode():
        indices = torch.tensor(rows, device=fused.weight.device, dtype=torch.long)
        weight = fused.weight.index_select(0, indices)
        bias = fused.bias.index_select(0, indices) if fused.bias is not None else None
    eps = block.norm.eps
    binding = {"schema": "mdb-flux-single-mlp-binding-v1", "inputs": {role: {"cut": cut.fingerprint,
        "image": array(cut.payload["image"]), "single_mod": array(cut.payload["context"]["single_mod"])} for role, cut in cuts.items()},
        "weight_rows": array(weight), "bias_rows": array(bias) if bias is not None else None,
        "query_indices": QUERIES, "neuron_indices": NEURONS, "weight_row_indices": rows,
        "hidden_width": hidden_width, "mlp_width": mlp_width, "eps": block.norm.eps, "elementwise_affine": False,
        "linear_path": "single_transformer_blocks.6.attn.to_qkv_mlp_proj",
        "native_parameters": {name: tensor_descriptor(value) for name, value in fused.named_parameters()},
        "counts_before_seal": dict(counts), "target_entries_before_seal": 0, "builder": builder_ref, "preparation": preparation_ref,
        "row_extraction": "Actual BF16 native parameter rows, paired first/second MLP half order. Full parameter identity authenticated; no partial-row Merkle proof.",
        "scope": "Eight image positions, 32 paired neurons; observed pre6 state/time modulation plus selected actual native weights. No target current-step single6 output available. Earlier-step module6 calls allowed."}
    save(output / "entry-binding.json", binding)
    # Predictor closure receives only incoming state, modulation and selected weights.
    operands = {role: {"image": np.array(cut.payload["image"].float().cpu().numpy(), copy=True),
        "mod": np.array(cut.payload["context"]["single_mod"].float().cpu().numpy(), copy=True)} for role, cut in cuts.items()}
    weight_np = np.array(weight.float().cpu().numpy(), copy=True)
    bias_np = np.array(bias.float().cpu().numpy(), copy=True) if bias is not None else None
    tx = ForecastTransaction(output / "mlp-entry-transaction", model_identity=adapter.model_identity,
        input_identity=identity(binding), source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": binding},
        execution_scope="Observed fresh pre6 state to sampled LayerNorm/time-affine/fused-MLP/SwiGLU prediction; trusted callback dataflow, external CUDA hooks/census",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    release = {"native": False, "blocked": 0}
    def guard(module, args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("current-step single6 target entered before early MLP seal")
    hook = block.register_forward_pre_hook(guard)
    try:
        with tx:
            tx.frontier("source_binding", binding)
            tx.predict("entry_mlp", lambda _: {"predictions": {"cases": {role: mlp["predict_entry_samples"](values["image"], values["mod"], weight_np, bias_np,
                eps=eps, query_indices=QUERIES, neuron_indices=NEURONS, hidden_width=hidden_width, mlp_width=mlp_width)
                for role, values in operands.items()}}})
            seal = tx.seal()
            event("entry_forecast_sealed", seal=seal, target_entries_before_seal=release["blocked"])
            release["native"] = True
            def execute(_):
                panel_config = {**config, "max_adapter_steps": 137, "max_acts": 10,
                    "max_wall_seconds": max(1, int(config["max_wall_seconds"] - (time.monotonic() - started)))}
                metadata = {**{key: source[key] for key in ("checkpoint_sha256", "implementation_sources", "versions")},
                    "context_id": CONTEXT["id"], "context": CONTEXT, "outer_preseal_counts": dict(counts),
                    "source_roles": {"zero": "direct matched-post0 low-T trajectory", "text": "own high-history trajectory; I/T may differ at pre6"},
                    "early_seal_sha256": seal["seal_sha256"], "timing_scope": "Fresh builder108/4 before early seal; zero panel replay; no current-step single6 before early seal"}
                selection = {"schema": "mdb-flux-single-mlp-selection-v1", "kind": "fresh-native-builder", "context_id": CONTEXT["id"],
                    "builder": builder_ref, "preparation": preparation_ref, "input_cuts": {role: cut.fingerprint for role, cut in cuts.items()}}
                with mlp["capture_source_calls"](block, text_rows=cuts["zero"].payload["text"].shape[1], query_indices=QUERIES, neuron_indices=NEURONS) as captured:
                    panel = panel_engine["run_panel"](pipe=pipe, adapter=adapter, inputs=cuts, output=output / "panel", config=panel_config,
                        request=request, metadata=metadata, selection=selection, prepared_inputs=True)
                observed = {role: {"call_index": row["call_index"], **{key: array(row[key]) for key in ("layernorm", "modulated_input", "u", "v", "activation")}}
                    for role, row in captured["rows"].items()}
                save(output / "entry-observations.json", {"rows": observed, "calls": captured["calls"], "seal": seal,
                    "target_entries_before_seal": release["blocked"], "scope": "First two direct target calls only; later edited/prefix/future target calls excluded from these samples"})
                for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "receipt_emissions"):
                    counts[key] += panel["counts"][key]
                native_rows = {role: {"call_index": row["call_index"], **{key: row[key].float().cpu().numpy() for key in ("layernorm", "modulated_input", "u", "v", "activation")}}
                    for role, row in captured["rows"].items()}
                return {"rows": native_rows, "calls": captured["calls"], "counts": dict(counts), "panel_summary_sha256": sha(output / "panel/summary.json"),
                    "entry_observations_sha256": sha(output / "entry-observations.json")}
            tx.execute("native_entry", execute)
    finally:
        hook.remove()
    save(output / "entry-transaction-report.json", tx.report())
    if counts != COUNTS:
        raise ValueError("complete fresh MLP-entry physical census differs")
    outer = {"schema": "mdb-flux-single-mlp-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "counts": dict(counts), "builder_counts": {"adapter_steps": 108, "acts": 4, "receipt_emissions": 112},
        "panel_summary_sha256": sha(output / "panel/summary.json"), "entry_seal": seal,
        "entry_transaction_verification": verify_transaction(tx.root), "model_identity": adapter.model_identity,
        "worker_seconds": time.monotonic() - started, "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "scope": config["scope"]}
    save(output / "outer-summary.json", outer)
    event("complete")
    print("MDB_FLUX_SINGLE_MLP_ARCHIVE_START " + json.dumps({"raw_output_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()), "counts": counts}), flush=True)
    emit(output, outer, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-single-mlp-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-single-mlp-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            emit(output, failure, request["config"])
        raise
