"""One finite resident lease: eight scene histories and ten same-parent futures."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def worker():
    config, request = (json.loads(Path(os.environ[key]).read_text()) for key in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("request/source binding differs")
    helper = runpy.run_path("flux_scene_inventory.py")
    expected = helper["make_plan"]()
    if any(config[k] != expected[k] for k in ("histories", "rows")) or config["selector"] != {"logical_step": 1, "phase": "joint", "index": 4}:
        raise ValueError("finite scene factorial differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
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

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    output = Path("flux-scene-inventory-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, timeline, active = time.monotonic(), [], {"row": "prefix"}
    counts = {"model_loads": 0, "prompt_encodings": 0, "adapter_steps": 0, "acts": 0,
              "images": 0, "vae_calls": 0, "receipt_emissions": 0, "physical_batch": 1}
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite scene-inventory wall budget exhausted")

    def event(kind, **fields):
        value = {"sequence": len(timeline), "seconds": time.monotonic()-started, "kind": kind, **fields}
        timeline.append(value)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        print("MDB_FLUX_SCENE_INVENTORY_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        save(output/"checkpoint-inventory-mismatch.json", {"expected": config["expected_checkpoint_sha256"],
            "actual": identity(inventory), "files": inventory})
        raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    prepared = {}
    for history in config["histories"]:
        wall()
        with torch.inference_mode():
            conditioning, text_ids = pipe.encode_prompt(history["prompt"], device=torch.device("cuda"),
                num_images_per_prompt=1, max_sequence_length=512)
        prepared[history["id"]] = conditioning.detach().cpu(), text_ids.detach().cpu()
        counts["prompt_encodings"] += 1
        event("encoded", history=history["id"])
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store = LocalStore(output / "typed-store")
    with torch.inference_mode():
        latent, image_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels//4, 512, 512,
            torch.bfloat16, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
    if steps != 4:
        raise ValueError("schedule differs")

    def receipt(value):
        wall()
        store.receipt(value)
        body = value.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "row": active["row"], "receipt": value.fingerprint})+"\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite adapter budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def array(name, value):
        path = output / "arrays" / (name+".npy")
        path.parent.mkdir(exist_ok=True)
        np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)
        return {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "native": tensor_descriptor(value)}

    histories, snapshots, ports, factors = {}, {}, {}, {}
    for history in config["histories"]:
        name = history["id"]
        active["row"] = "prefix."+name
        conditioning, text_ids = prepared[name]
        session = adapter.session(latent=latent.clone(), conditioning=conditioning.to("cuda"), img_ids=image_ids.clone(),
            txt_ids=text_ids.to("cuda"), timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
        n = NativeCutNavigator(session, max_steps=config["max_adapter_steps"], on_receipt=receipt)
        initial = n.capture()
        store.save(initial)
        n.until(logical_step=0, phase="single", index=4)
        early = n.capture()
        store.save(early)
        n.until(**config["selector"])
        entry = n.capture()
        store.save(entry)
        captured = {}

        def attention_hook(module, args, result):
            captured["attention"] = result[0].detach().clone()

        def mlp_hook(module, args, result):
            captured["mlp"] = result.detach().clone()

        handles = [pipe.transformer.transformer_blocks[4].attn.register_forward_hook(attention_hook),
                   pipe.transformer.transformer_blocks[4].ff.register_forward_hook(mlp_hook)]
        try:
            block_receipt = n.step()
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {"attention", "mlp"}:
            raise ValueError("actual native joint ports missing")
        post = n.capture()
        store.save(post)
        (_, _, ga), (_, _, gm) = Flux2Modulation.split(entry.payload["context"]["img_mod"], 2)
        if not torch.equal((entry.payload["image"] + ga*captured["attention"]) + gm*captured["mlp"], post.payload["image"]):
            raise ValueError("actual native sequential joint equation does not close")
        captured.update(gate_attention=ga, gate_mlp=gm)
        histories[name] = {"initial": initial.fingerprint, "early": early.fingerprint,
                           "entry": entry.fingerprint, "post": post.fingerprint,
                           "joint_receipt": block_receipt.to_dict(), "factors": history["factors"]}
        ports[name] = {key: array(name+"."+key, value) for key, value in captured.items()}
        snapshots[name] = {"early": early, "entry": entry, "post": post}
        factors[name] = tuple(history["factors"])
        event("prefix_observed", history=name, counts=dict(counts))

    # All state comparisons and supports are constructed before any final VAE call.
    source = {"schema": "mdb-flux-scene-inventory-source-v1", "histories": histories, "ports": ports,
              "model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory), "checkpoint_files": inventory,
              "execution": dict(adapter.execution), "source_hashes": request["source_hashes"],
              "request_sha256": request["request_sha256"], "image_ids": array("image_ids", image_ids),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}}
    save(output / "source.json", source)
    diagnostics = {}
    for boundary in ("early", "entry", "post"):
        diagnostics[boundary] = {}
        for slot in ("image", "text", "conditioning", "latent"):
            values = {factors[name]: cuts[boundary].payload[slot].detach().float().cpu().numpy() for name, cuts in snapshots.items()}
            diagnostics[boundary][slot] = helper["factorial_diagnostics"](values)
    for slot in ("attention", "mlp"):
        diagnostics[slot] = helper["factorial_diagnostics"]({factors[name]: np.load(output/p[slot]["path"], allow_pickle=False)
                                                            for name, p in ports.items()})
    save(output / "state-relationships.json", diagnostics)
    base = snapshots["s0c0r0"]["post"]
    supports = {}
    for factor, name in (("shapeA", "s1c0r0"), ("colorB", "s0c1r0"), ("relation", "s0c0r1")):
        source_cut = snapshots[name]["post"]
        supports[factor] = helper["propose_support"](base.payload["image"].float().cpu().numpy(),
            source_cut.payload["image"].float().cpu().numpy(), image_ids.cpu().numpy())
    save(output / "supports.json", supports)
    predictions = {"hypotheses": config["hypotheses"], "supports_sha256": sha(output/"supports.json"),
                   "state_relationships_sha256": sha(output/"state-relationships.json"),
                   "scope": "observed native prefixes and contrast donors; no final RGB seen; intervention-response forecasts, not numerical RGB prediction"}
    tx = ForecastTransaction(output/"transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output/"source.json"),
            "allowed_prefix": "eight own-state natural prefixes through step1 joint4 exit; actual native A/M ports; no final images"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"]*1000})
    tx.predict("scene_inventory", lambda _: {"predictions": predictions})
    seal = tx.seal()
    rows = []

    def execute(_):
        event("predictions_sealed", seal=seal, counts=dict(counts))
        for spec in config["rows"]:
            wall()
            active["row"] = spec["id"]
            parent = snapshots[spec["history"]]["post"]
            n = nav(parent)
            row = {**spec, "parent": parent.fingerprint, "acts": [], "continuation": "recipient natural text and conditioning retained throughout"}
            if spec["kind"] in {"sham", "patch"}:
                replacement = parent.payload["image"].clone()
                if spec["kind"] == "patch":
                    donor = snapshots[spec["source"]]["post"]
                    indices = list(range(replacement.shape[1])) if spec["mode"] == "full" else supports[spec["factor"]][spec["mode"]]
                    replacement[:, indices] = donor.payload["image"][:, indices]
                    row["selected_rows"] = indices
                    row["source_cut"] = donor.fingerprint
                act = Act.replace("image", replacement, name=spec["id"])
                rec = n.apply(act)
                after = n.capture()
                store.save(after)
                if describe(after.payload) != describe({**parent.payload, "image": replacement}):
                    raise ValueError("image intervention changed undeclared state")
                row["acts"].append({"manifest": act.manifest(), "receipt": rec.to_dict(),
                                    "post": after.fingerprint, "undeclared_slots_exact": True})
            schedulers = []
            while n.point["logical_step"] < 4:
                before = n.point
                n.step()
                if before["phase"] == "scheduler":
                    cut = n.capture()
                    schedulers.append({"step": n.point["logical_step"], "cut": cut.fingerprint,
                                       "latent": array(spec["id"]+".latent"+str(n.point["logical_step"]), cut.payload["latent"])})
            final = n.capture()
            store.save(final)
            if describe(final.payload["conditioning"]) != describe(parent.payload["conditioning"]):
                raise ValueError("recipient future conditioning changed")
            if counts["images"] >= len(config["rows"]):
                raise ValueError("image budget exceeded")
            counts["vae_calls"] += 1
            image = adapter.decode(n.session, height=512, width=512)
            path = output/(spec["id"]+".png")
            image.save(path)
            counts["images"] += 1
            row.update(final_cut=final.fingerprint, schedulers=schedulers,
                       image={"path": path.name, "sha256": sha(path)})
            save(output/"rows"/(spec["id"]+".json"), row)
            rows.append(row)
            event("branch_complete", row=spec["id"], counts=dict(counts))
        return {"rows": rows, "counts": dict(counts), "numerical_RGB_forecast": False}

    with tx:
        tx.execute("native", execute)
    save(output/"transaction-report.json", tx.report())
    verification = verify_transaction(output/"transaction", expected_seal_sha256=seal["seal_sha256"])
    native = next(r for r in rows if r["id"] == "native.s0c0r0")
    sham = next(r for r in rows if r["id"] == "sham")
    if native["image"]["sha256"] != sham["image"]["sha256"] or any(
            not np.array_equal(np.load(output/a["latent"]["path"]), np.load(output/b["latent"]["path"]))
            for a, b in zip(native["schedulers"], sham["schedulers"])):
        raise ValueError("same-parent sham fails native image/scheduler replay")
    if len(rows) != 18 or counts["prompt_encodings"] != 8 or counts["model_loads"] != 1:
        raise ValueError("physical run inventory differs")
    summary = {"schema": "mdb-flux-scene-inventory-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "model_identity": adapter.model_identity,
        "mechanics_status": "valid", "instrument_status": "pixel instrument evaluated offline",
        "terminal_status": "unassessed", "counts": counts, "seal": seal, "transaction_verification": verification,
        "sham_exact": True, "worker_seconds": time.monotonic()-started, "numerical_RGB_forecast": False}
    save(output/"summary.json", summary)
    event("complete", counts=dict(counts))
    save(output/"files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=512 << 20, compression_level=1)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-scene-inventory-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-scene-inventory-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                       "request_sha256": request["request_sha256"], "error": f"{type(error).__name__}: {error}", "mechanics_status": "errored"}
            save(output/"failure.json", failure)
            runpy.run_path("campaign_support.py")["emit"](output, failure, max_compressed_bytes=512 << 20, compression_level=1)
        raise
