"""One resident native Klein lease, three fresh matched single0 K/V panels."""
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

PROMPTS = {
    "p0": "A red sphere to the left of a blue cube on a plain white background. Two separate objects, simple studio product photograph.",
    "p1": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
CONTEXTS = [{"id": name, "prompt_id": prompt, "prompt": PROMPTS[prompt], "seed": seed}
    for name, prompt, seed in (("p0-s53", "p0", 53), ("p1-s47", "p1", 47), ("p1-s53", "p1", 53))]


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(child) for child in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_value(value), stream, sort_keys=True, indent=2, allow_nan=False)
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
    if (config.get("contexts") != CONTEXTS or config.get("height") != 512 or config.get("width") != 512
            or config.get("steps") != 4 or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 0}
            or config.get("guidance_scale", 1.) != 1.):
        raise ValueError("fresh-context scientific geometry differs from fixed panel")
    if config.get("attention_query_indices", [0, 31, 32, 255, 511, 512, 767, 1023]) != [0, 31, 32, 255, 511, 512, 767, 1023]:
        raise ValueError("fresh-context image queries differ from fixed eight-query panel")
    expected_branches = [("zero", "zero", "zero"), ("text", "text", "text"), ("sham", "zero", "zero"),
        ("K.image", "K", None), ("V.image", "V", None), ("KV.image", "KV", None),
        ("KV.both", "KV", "KV"), ("text.text", None, "text"), ("coherent", "text", "text")]
    if config.get("branches") != [{"id": name, "kind": "reference" if name in {"zero", "text"} else "edited",
            "image_source": image, "text_source": text} for name, image, text in expected_branches]:
        raise ValueError("fresh-context branch recipes differ")
    for name, minimum, maximum in (("max_adapter_steps", 861, 900), ("max_acts", 39, 39),
            ("max_candidate_calls", 15, 15), ("max_images", 27, 27), ("max_vae_calls", 27, 27),
            ("max_wall_seconds", 1, 900), ("cpu_threads", 1, 16), ("archive_compression_level", 1, 9)):
        if type(config.get(name)) is not int or not minimum <= config[name] <= maximum:
            raise ValueError("invalid finite fresh-context budget: " + name)


def emit(output, report, config):
    # Both normal and failure reports may contain immutable transaction mappings.
    runpy.run_path("flux_family_custody.py")["emit"](output, json_value(report),
        limits=config.get("transport_limits"), compression_level=config["archive_compression_level"])


def manifest(output):
    path = output / "files.json"
    if path.exists():
        path.unlink()
    save(path, {item.relative_to(output).as_posix(): sha(item) for item in sorted(output.rglob("*")) if item.is_file()})


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("fresh-context request/source bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-single-contexts-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"plan_sha256": sha(output / "plan.json"), "request_sha256": request["request_sha256"],
        "source_identity": identity(request["source_hashes"]), "job_id": request["job_id"],
        "work": {"model_loads": 0, "prompt_encodings": 0, "noise_draws": 0, "adapter_steps": 0, "candidate_calls": 0},
        "scope": "Frozen fresh applicability/recipes before loading or model work; not a numerical RGB forecast"})
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

    engine = runpy.run_path("flux_single_contexts.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("fresh-context panel requires admitted CUDA")
    started = time.monotonic()
    counts = {"model_loads": 0, "prompt_encodings": 0, "noise_draws": 0, "adapter_steps": 0,
        "acts": 0, "candidate_calls": 0, "images": 0, "vae_calls": 0, "receipt_emissions": 0, "physical_batch": 1}
    sequence = 0
    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite fresh-context wall budget exhausted")
    def event(kind, **fields):
        nonlocal sequence
        value = json_value({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started,
            "counts": dict(counts), **fields})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_SINGLE_CONTEXTS_PHASE " + json.dumps(value, sort_keys=True), flush=True)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("fresh checkpoint inventory differs")
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
        (type(pipe), type(pipe.text_encoder), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[0]),
         type(pipe.transformer.single_transformer_blocks[0].attn.processor), type(pipe.vae), type(pipe.scheduler), Flux2KleinAdapter)}
    cache, conditioning_records = {}, {}
    pipe.text_encoder.to("cuda")
    for prompt_id, prompt in PROMPTS.items():
        wall()
        with torch.inference_mode():
            conditioning, txt_ids = pipe.encode_prompt(prompt, device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
        counts["prompt_encodings"] += 1
        cache[prompt_id] = (conditioning.cpu(), txt_ids.cpu())
        path = output / "encoded" / (prompt_id + ".npy")
        path.parent.mkdir(exist_ok=True)
        np.save(path, conditioning.float().cpu().numpy(), allow_pickle=False)
        ids_path = path.with_name(prompt_id + "-txt-ids.npy")
        np.save(ids_path, txt_ids.float().cpu().numpy(), allow_pickle=False)
        conditioning_records[prompt_id] = {"prompt": prompt, "conditioning": {"path": path.relative_to(output).as_posix(), "sha256": sha(path)},
            "txt_ids": {"path": ids_path.relative_to(output).as_posix(), "sha256": sha(ids_path)}}
        event("prompt_encoded", prompt_id=prompt_id)
    del conditioning, txt_ids
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    source = {"schema": "mdb-flux-single-contexts-source-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "checkpoint_files": inventory, "execution": dict(adapter.execution),
        "implementation_sources": implementation_sources,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "contexts": CONTEXTS, "encodings": conditioning_records, "counts_after_encoding": dict(counts), "scope": config["scope"]}
    save(output / "source.json", source)
    results = {}
    for context in CONTEXTS:
        wall()
        event("context_start", context_id=context["id"])
        with torch.inference_mode():
            conditioning, txt_ids = (tensor.to("cuda") for tensor in cache[context["prompt_id"]])
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                conditioning.dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(context["seed"]))
            counts["noise_draws"] += 1
            schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
            timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
                mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
            if steps != 4:
                raise ValueError("fresh native schedule differs")
            session = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
                timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
        before_builder = dict(counts)
        builder_root = output / "builders" / context["id"]
        cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=builder_root,
            context=context, job_id=request["job_id"], wall=wall)
        for key in ("adapter_steps", "acts", "receipt_emissions"):
            counts[key] += builder["counts"][key]
        panel_config = {**config, "context": context, "max_adapter_steps": 191, "max_acts": 10,
            "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9,
            "max_wall_seconds": max(1, int(config["max_wall_seconds"] - (time.monotonic() - started)))}
        builder_ref = {"path": (builder_root / "builder.json").relative_to(output).as_posix(), "sha256": sha(builder_root / "builder.json")}
        metadata = {**{key: source[key] for key in ("checkpoint_sha256", "execution", "implementation_sources", "versions")},
            "context_id": context["id"], "context": context, "job_id": request["job_id"],
            "source_selection": {"kind": "fresh-native-builder", "builder": builder_ref},
            "builder": builder_ref, "parent_counts_before_builder": before_builder, "parent_counts_after_builder": dict(counts),
            "allowed_prior_work": "Two cached own prompt encodes; fresh seeded Z0 and 96 native builder transitions/three typed Acts; no panel direct/prefix/VAE work"}
        panel = engine["run_panel"](pipe=pipe, adapter=adapter, cuts=cuts, output=output / "contexts" / context["id"],
            config=panel_config, request=request, source_metadata=metadata)
        for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "receipt_emissions"):
            counts[key] += panel["counts"][key]
            if key != "receipt_emissions" and counts[key] > config["max_" + key]:
                raise RuntimeError("finite cumulative fresh-context budget exhausted: " + key)
        results[context["id"]] = {"context": context, "builder": builder_ref,
            "builder_counts": builder["counts"], "panel_counts": panel["counts"], "counts_after": dict(counts),
            "panel_summary": {"path": "contexts/" + context["id"] + "/summary.json",
                "sha256": sha(output / "contexts" / context["id"] / "summary.json")}}
        event("context_complete", context_id=context["id"])
        del session, cuts, builder, panel, conditioning, txt_ids, latent, img_ids
        gc.collect()
        torch.cuda.empty_cache()
    expected = {"model_loads": 1, "prompt_encodings": 2, "noise_draws": 3, "adapter_steps": 861,
        "acts": 39, "candidate_calls": 15, "images": 27, "vae_calls": 27, "receipt_emissions": 900}
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError("complete cross-context physical census differs")
    summary = {"schema": "mdb-flux-single-contexts-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "model_identity": adapter.model_identity, "source_sha256": sha(output / "source.json"), "contexts": results,
        "counts": counts, "worker_seconds": time.monotonic() - started, "mechanics_status": "pending-independent-audit",
        "terminal_status": "unassessed", "RGB_forecast": False, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete")
    event("archive_start", raw_output_bytes_before_event=sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))
    manifest(output)
    emit(output, summary, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-single-contexts-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            config = request["config"]
            failure = {"schema": "mdb-flux-single-contexts-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            manifest(output)
            emit(output, failure, config)
        raise
