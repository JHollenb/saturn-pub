"""One fresh-prompt, recipient-only native joint4 write campaign.

All futures consume their own conditioning and predicted states. Native scoped
candidate calculations are followed by truthful unmodified adapter steps and
typed image/text Acts. Partial-latent VAE previews never advance denoising.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

BRANCHES = [
    {"id": "native", "attention_gain": 1., "mlp_gain": 1., "mlp_policy": "recompute", "rollback_text": False},
    {"id": "sham", "attention_gain": 1., "mlp_gain": 1., "mlp_policy": "recompute", "rollback_text": False},
    {"id": "Ahalf.recompute", "attention_gain": .5, "mlp_gain": 1., "mlp_policy": "recompute", "rollback_text": False},
    {"id": "Azero.recompute", "attention_gain": 0., "mlp_gain": 1., "mlp_policy": "recompute", "rollback_text": False},
    {"id": "Ahalf.hold", "attention_gain": .5, "mlp_gain": 1., "mlp_policy": "hold-native", "rollback_text": False},
    {"id": "Azero.hold", "attention_gain": 0., "mlp_gain": 1., "mlp_policy": "hold-native", "rollback_text": False},
    {"id": "Mhalf", "attention_gain": 1., "mlp_gain": .5, "mlp_policy": "recompute", "rollback_text": False},
    {"id": "Mzero", "attention_gain": 1., "mlp_gain": 0., "mlp_policy": "recompute", "rollback_text": False},
    {"id": "bothzero", "attention_gain": 0., "mlp_gain": 0., "mlp_policy": "recompute", "rollback_text": False},
    {"id": "text.rollback", "attention_gain": 1., "mlp_gain": 1., "mlp_policy": "recompute", "rollback_text": True},
    {"id": "Azero.recompute.textrollback", "attention_gain": 0., "mlp_gain": 1., "mlp_policy": "recompute", "rollback_text": True},
    {"id": "Azero.hold.textrollback", "attention_gain": 0., "mlp_gain": 1., "mlp_policy": "hold-native", "rollback_text": True}]
PREVIEW_ROWS = ["native", "Azero.recompute", "Azero.hold", "Mzero"]


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
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
    if (config.get("selector") != {"logical_step": 1, "phase": "joint", "index": 4}
            or config.get("branches") != BRANCHES or config.get("preview_rows") != PREVIEW_ROWS
            or config.get("preview_steps") != [2, 3] or config.get("seed") != 47
            or config.get("steps") != 4 or config.get("height") != 512 or config.get("width") != 512
            or config.get("guidance_scale", 1.) != 1. or not isinstance(config.get("prompt"), str) or not config["prompt"]):
        raise ValueError("fresh joint-self campaign differs from fixed context/branch/preview contract")
    for name, ceiling, minimum in (("max_adapter_steps", 1200, 1), ("max_candidate_calls", 13, 12),
            ("max_final_images", 12, 12), ("max_preview_images", 8, 8), ("max_images", 20, 20),
            ("max_vae_calls", 21, 21), ("max_wall_seconds", 840, 1)):
        if type(config.get(name)) is not int or not minimum <= config[name] <= ceiling:
            raise ValueError("invalid finite joint-self budget: " + name)
    compression_level = config.get("archive_compression_level", 9)
    if type(compression_level) is not int or not 1 <= compression_level <= 9:
        raise ValueError("archive_compression_level must be an integer in [1,9]")


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(name) != expected for name, expected in request["source_hashes"].items()):
        raise ValueError("request/source binding differs")
    validate_config(config)
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
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    helper = runpy.run_path("flux_joint_self.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("joint-self campaign requires admitted CUDA")
    output = Path("flux-joint-self-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, timeline, active = time.monotonic(), [], {"branch": "prefix"}
    counts = {"model_loads": 0, "prompt_encodings": 0, "physical_batch": 1, "adapter_steps": 0,
              "candidate_calls": 0, "acts": 0, "receipt_emissions": 0, "final_images": 0,
              "preview_images": 0, "images": 0, "vae_calls": 0}

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite joint-self wall budget exhausted")

    def event(kind, **fields):
        row = json_value({"sequence": len(timeline), "seconds": time.monotonic() - started, "kind": kind, **fields})
        timeline.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_JOINT_SELF_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic four-step distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    with torch.inference_mode():
        conditioning, txt_ids = pipe.encode_prompt(config["prompt"], device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
        counts["prompt_encodings"] += 1
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
            conditioning.dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
    if steps != 4:
        raise ValueError("native schedule differs")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store = LocalStore(output / "typed-store")
    array_index = {}

    def array(value):
        wall()
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in array_index:
            path = output / "arrays" / (key + ".npy")
            path.parent.mkdir(exist_ok=True)
            np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)
            array_index[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path),
                "descriptor": descriptor, "stored_dtype": "float32", "stored_shape": list(value.shape)}
        return array_index[key]

    def receipt(value):
        store.receipt(value)
        body = value.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "branch": active["branch"],
                                    "receipt": value.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite native-step budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def calculate(parent, spec, held=None):
        wall()
        if counts["candidate_calls"] >= config["max_candidate_calls"]:
            raise RuntimeError("finite direct candidate-call budget exhausted")
        counts["candidate_calls"] += 1
        return helper["joint_candidate_self"](pipe.transformer.transformer_blocks[config["selector"]["index"]], parent,
            attention_gain=spec["attention_gain"], mlp_gain=spec["mlp_gain"], mlp_policy=spec["mlp_policy"], held_mlp=held)

    def candidate_record(value, parent):
        return {"input_parent": parent.fingerprint, "attention_gain": value["attention_gain"], "mlp_gain": value["mlp_gain"],
            "mlp_policy": value["mlp_policy"], "equation_exact": value["equation_exact"], "arithmetic": value["arithmetic"],
            "ports": {name: array(tensor) for name, tensor in value["ports"].items()},
            "held_source_parent": parent.fingerprint if value["mlp_policy"] == "hold-native" else None}

    original = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
        timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
    initial = original.capture()
    store.save(initial)
    prefix = NativeCutNavigator(original, max_steps=config["max_adapter_steps"], on_receipt=receipt)
    prefix_receipts, prefix_schedulers = [], []
    while any(prefix.point[key] != value for key, value in config["selector"].items()):
        wall()
        before = prefix.point
        value = prefix.step()
        prefix_receipts.append(value.to_dict())
        if before["phase"] == "scheduler":
            cut = prefix.capture()
            store.save(cut)
            prefix_schedulers.append({"step": prefix.point["logical_step"], "cut": cut.fingerprint, "latent": array(cut.payload["latent"])})
    parent = prefix.capture()
    store.save(parent)
    baseline = calculate(parent, BRANCHES[0])
    active["branch"] = "baseline-parity"
    native_receipt = prefix.step()
    native_post = prefix.capture()
    store.save(native_post)
    if not torch.equal(baseline["image"], native_post.payload["image"]) or not torch.equal(baseline["text"], native_post.payload["text"]):
        raise ValueError("own native candidate does not reproduce adapter block")
    baseline_record = candidate_record(baseline, parent)
    held = baseline["held_mlp"]
    source = {"schema": "mdb-flux-joint-self-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "checkpoint_files": inventory, "execution": dict(adapter.execution),
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"], "selector": config["selector"],
        "prompt": config["prompt"], "seed": config["seed"], "initial_cut": initial.fingerprint,
        "prefix_receipts": prefix_receipts, "prefix_schedulers": prefix_schedulers, "parent_cut": parent.fingerprint,
        "native_post_cut": native_post.fingerprint, "baseline_native_receipt": native_receipt.to_dict(),
        "baseline_candidate": {**baseline_record, "held_mlp": {"parent": parent.fingerprint, "attention_gain": 1., "value": array(held["value"])}},
        "own_conditioning": tensor_descriptor(conditioning), "branches": config["branches"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "implementation_sources": {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
            (type(pipe), type(pipe.transformer), type(pipe.transformer.transformer_blocks[4]), type(pipe.vae), type(pipe.scheduler), type(adapter))},
        "array_storage": "content-addressed per-tensor lossless FP32-expanded native BF16 NPY; no model weights serialized",
        "allowed_prefix": "one native prompt encoding, native own-state prefix to step1 joint4; one direct native baseline and one native adapter block for port parity; no RGB outcomes",
        "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
            "allowed_prefix": source["allowed_prefix"]}, budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("joint_self_hypotheses", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows, preview_parity = [], {}

    def decode(nav_value, *, preview=False):
        wall()
        if counts["vae_calls"] >= config["max_vae_calls"]:
            raise RuntimeError("finite VAE-call budget exhausted")
        counts["vae_calls"] += 1
        return helper["decode_live_preview"](pipe, nav_value.capture(), height=512, width=512) if preview else adapter.decode(nav_value.session, height=512, width=512)

    def finish(n, row):
        schedulers, previews = [], {}
        while n.point["logical_step"] < config["steps"]:
            wall()
            before = n.point
            n.step()
            if before["phase"] == "scheduler":
                cut = n.capture()
                store.save(cut)
                step = n.point["logical_step"]
                schedulers.append({"step": step, "cut": cut.fingerprint, "latent": array(cut.payload["latent"])})
                if row["id"] in config["preview_rows"] and step in config["preview_steps"]:
                    if counts["preview_images"] >= config["max_preview_images"] or counts["images"] >= config["max_images"]:
                        raise RuntimeError("finite preview image budget exhausted")
                    image = decode(n, preview=True)
                    if n.capture().fingerprint != cut.fingerprint:
                        raise ValueError("preview changed its live own-state frontier")
                    path = output / (row["id"] + f".preview{step}.png")
                    image.save(path)
                    counts["preview_images"] += 1
                    counts["images"] += 1
                    previews[str(step)] = {"path": path.name, "sha256": sha(path), "cut": cut.fingerprint,
                        "point": dict(cut.execution_point), "scope": "native VAE preview of partial noised live latent; no denoising, no fake completion"}
        final = n.capture()
        store.save(final)
        if describe(final.payload["conditioning"]) != describe(initial.payload["conditioning"]):
            raise ValueError("own future conditioning changed")
        if counts["final_images"] >= config["max_final_images"] or counts["images"] >= config["max_images"]:
            raise RuntimeError("finite final image budget exhausted")
        image = decode(n)
        path = output / (row["id"] + ".png")
        image.save(path)
        counts["final_images"] += 1
        counts["images"] += 1
        if row["id"] == "native":
            other = decode(n, preview=True)
            exact = np.array_equal(np.asarray(image), np.asarray(other))
            png = io.BytesIO()
            other.save(png, format="PNG")
            png_exact = png.getvalue() == path.read_bytes()
            if not exact or not png_exact or n.capture().fingerprint != final.fingerprint:
                raise ValueError("explicit VAE preview helper differs from final native adapter decode")
            preview_parity.update(cut=final.fingerprint, point=dict(final.execution_point), pixel_exact=True, png_exact=True,
                adapter_png_sha256=sha(path), helper_png_sha256=hashlib.sha256(png.getvalue()).hexdigest(),
                adapter_pixel_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                helper_pixel_sha256=hashlib.sha256(np.asarray(other).tobytes()).hexdigest(),
                extra_vae_calls=1, extra_saved_images=0)
        row.update(final_cut=final.fingerprint, final_conditioning=tensor_descriptor(final.payload["conditioning"]),
            image={"path": path.name, "sha256": sha(path)}, schedulers=schedulers, previews=previews)
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    with tx:
        event("hypotheses_sealed", seal=seal, counts=dict(counts), source_sha256=sha(output / "source.json"))

        def execute(_):
            for spec in config["branches"]:
                active["branch"] = spec["id"]
                event("branch_start", row=spec["id"])
                row = {**spec, "parent": parent.fingerprint, "acts": [], "continuation_policy": "own-C-throughout-no-reset"}
                if spec["id"] == "native":
                    n = nav(native_post)
                    row.update(native_post=native_post.fingerprint, native_receipt=native_receipt.to_dict(),
                               candidate=baseline_record, candidate_phase="pre-seal native port/parity observation")
                else:
                    n = nav(parent)
                    value = calculate(parent, spec, held if spec["mlp_policy"] == "hold-native" else None)
                    current_receipt = n.step()
                    post = n.capture()
                    store.save(post)
                    if not torch.equal(value["text"], post.payload["text"]):
                        raise ValueError("image-write candidate changed current native text")
                    row.update(native_post=post.fingerprint, native_receipt=current_receipt.to_dict(),
                               candidate=candidate_record(value, parent), current_text_exact_native=True)
                    install = post.payload["image"].clone() if spec["id"] in {"sham", "text.rollback"} else value["image"]
                    row["acts"].append(helper["install_joint_port"](n, "image", install,
                        name=spec["id"] + ".image", store=store, expected_point=post.execution_point))
                    if spec["rollback_text"]:
                        row["acts"].append(helper["install_joint_port"](n, "text", parent.payload["text"],
                            name=spec["id"] + ".own-pre-joint-text", store=store, expected_point=post.execution_point))
                finish(n, row)
            if (len(rows) != 12 or counts["final_images"] != 12 or counts["preview_images"] != 8
                    or counts["images"] != 20 or counts["vae_calls"] != 21 or counts["prompt_encodings"] != 1):
                raise ValueError("completed physical image/preview/encoding inventory differs")
            return {"rows": rows, "counts": dict(counts), "preview_helper_final_parity": preview_parity,
                    "source_sha256": sha(output / "source.json"), "RGB_forecast": False}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    save(output / "preview-helper-final-parity.json", preview_parity)
    summary = {"schema": "mdb-flux-joint-self-result-v1", "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "mechanics_status": "valid", "terminal_status": "unassessed", "model_identity": adapter.model_identity,
        "counts": counts, "seal": seal, "transaction_verification": verification, "worker_seconds": time.monotonic() - started,
        "RGB_forecast": False, "fresh_single_prompt": True, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
        compression_level=config.get("archive_compression_level", 9))


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-joint-self-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-joint-self-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
            runpy.run_path("campaign_support.py")["emit"](output, failure, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
                compression_level=config.get("archive_compression_level", 9))
        raise
