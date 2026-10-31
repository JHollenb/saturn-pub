"""Native-consumer source/carrier mapping; semantic hypotheses, no image forecast."""
from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    support = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, sha, file_sha, identity = (support[name] for name in ("save", "sha", "file_sha", "identity"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(file_sha(name) != value for name, value in request["source_hashes"].items()):
        raise ValueError("request or staged source differs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import verify_transaction
    from mdb.forward_recording import tensor_descriptor

    routes = runpy.run_path("flux_concept_routes.py")
    branch_carriers = routes["branch_carriers"]
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-concept-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, timeline = time.monotonic(), []
    counts = {"adapter_steps": 0, "images": 0, "acts": 0, "prefix_steps": 0, "physical_batch": 1}

    def event(kind, **details):
        row = support["json_value"]({"kind": kind, "sequence": len(timeline), "seconds": time.monotonic() - started, **details})
        timeline.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_CONCEPT_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(files) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    prepared = {}
    for prompt in config["prompts"]:
        wall()
        with torch.inference_mode():
            conditioning, txt_ids = pipe.encode_prompt(prompt["text"], device=torch.device("cuda"),
                num_images_per_prompt=1, max_sequence_length=512)
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                conditioning.dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
            sigmas = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
            timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("unexpected schedule")
        prepared[prompt["id"]] = {"latent": latent, "conditioning": conditioning, "img_ids": img_ids,
            "txt_ids": txt_ids, "timesteps": timesteps.clone(), "sigmas": pipe.scheduler.sigmas.clone()}
    for name in ("latent", "img_ids", "txt_ids", "timesteps", "sigmas"):
        if not all(torch.equal(row[name], prepared["base"][name]) for row in prepared.values()):
            raise ValueError("prompt contrast changes nuisance initial state or IDs")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store = LocalStore(output / "typed-store")
    sessions, cuts, cut_records, mlp = {}, {}, {}, {}

    def advance(session, n=1):
        wall()
        if counts["adapter_steps"] + n > config["max_adapter_steps"]:
            raise RuntimeError("finite adapter budget exhausted")
        receipt = session.continue_(n)
        store.receipt(receipt)
        counts["adapter_steps"] += n
        return receipt

    # Allowed observed prefixes stop before step1 joint0. No RGB is decoded.
    for prompt_id, args in prepared.items():
        session = adapter.session(**args)
        sessions[prompt_id], cuts[prompt_id], cut_records[prompt_id] = session, {}, {}
        for step in config["cut_steps"]:
            before = session.capture()
            store.save(before)
            advance(session)
            after = session.capture()
            store.save(after)
            cuts[prompt_id][step] = (before, after)
            cut_records[prompt_id][str(step)] = {"parent": before.fingerprint, "projected": after.fingerprint}
            if step == 0:
                captured, target = {}, pipe.transformer.transformer_blocks[0]

                def input_r(module, args):
                    captured["r"] = args[0].detach().clone()

                def input_x(module, args):
                    captured["x"] = args[0].detach().clone()

                def output_y(module, args, result):
                    captured["y"] = result.detach().clone()

                hooks = [target.norm2.register_forward_pre_hook(input_r), target.ff.register_forward_pre_hook(input_x),
                         target.ff.register_forward_hook(output_y)]
                try:
                    advance(session)
                finally:
                    for hook in hooks:
                        hook.remove()
                gate = Flux2Modulation.split(after.payload["context"]["img_mod"], 2)[1][2]
                captured["gate"] = gate
                captured["write"] = gate * captured["y"]
                captured["post_image"] = session.read("image")
                if not torch.equal(captured["r"] + captured["write"], captured["post_image"]):
                    raise ValueError("native MLP ports do not close residual equation")
                path = output / (prompt_id + ".mlp.npz")
                np.savez_compressed(path, **{key: value.float().cpu().numpy() for key, value in captured.items()})
                mlp[prompt_id] = {"path": path.name, "sha256": file_sha(path), "ports":
                    {key: tensor_descriptor(value) for key, value in captured.items()}, "equation_exact": True,
                    "scope": "actual native hooks at step0 joint0; not a full GPU ATen symbol trace"}
                while session.capture(retain=False).execution_point["logical_step"] == 0:
                    advance(session)
        event("prefix_ready", prompt=prompt_id, cuts=cut_records[prompt_id], counts=dict(counts))
    counts["prefix_steps"] = counts["adapter_steps"]
    source = {"schema": "mdb-flux-concept-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(files), "request_sha256": request["request_sha256"],
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "mdb-runtime", "saturn-pub")},
        "implementations": {str(Path(inspect.getfile(kind)).resolve()): file_sha(inspect.getfile(kind))
            for kind in (type(adapter), type(pipe), type(pipe.transformer), type(pipe.transformer.transformer_blocks[0]), type(pipe.vae))},
        "cuts": cut_records, "mlp": mlp, "observed_prefix_steps": counts["prefix_steps"],
        "prediction_scope": "predeclared semantic route hypotheses only; no numerical forecast of altered images"}
    save(output / "source.json", source)
    results = []

    def finish(session, row):
        event("branch_start", row=row["id"])
        schedulers = []
        while session.capture(retain=False).execution_point["logical_step"] < config["steps"]:
            phase = session.capture(retain=False).execution_point["phase"]
            advance(session)
            if phase == "scheduler":
                state = session.capture(retain=False).payload
                values = state["latent"].float().cpu().numpy()
                path = output / (row["id"] + f".step{state['step']}.npy")
                np.save(path, values, allow_pickle=False)
                schedulers.append({"step": state["step"], "path": path.name, "sha256": file_sha(path),
                                   "descriptor": tensor_descriptor(state["latent"])})
        if counts["images"] >= config["max_images"]:
            raise RuntimeError("finite image budget exhausted")
        image = adapter.decode(session, height=512, width=512)
        path = output / (row["id"] + ".png")
        image.save(path)
        counts["images"] += 1
        row.update(image={"path": path.name, "sha256": file_sha(path)}, schedulers=schedulers)
        results.append(row)
        save(output / "rows" / (row["id"] + ".json"), row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    tx, seal = routes["prepare_hypotheses"](output / "transaction", model_identity=source["model_identity"],
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        cuts=cut_records, source_sha256=file_sha(output / "source.json"), config=config)
    with tx:
        event("hypotheses_sealed", seal=seal, counts=dict(counts))

        def execute(context):
            for prompt_id in prepared:
                finish(sessions[prompt_id].fork(cuts[prompt_id][1][1]), {"id": prompt_id + ".native", "prompt": prompt_id,
                    "kind": "native", "parent": cuts[prompt_id][1][1].fingerprint})
            for step in config["cut_steps"]:
                parent, projected = cuts["base"][step]
                for donor_id in ("base", "relation", "color", "shape"):
                    for arm in (["sham"] if donor_id == "base" else config["arms"]):
                        row_id = f"step{step}.{donor_id}.{arm}"
                        branch, receipts = branch_carriers(sessions["base"], parent, projected, cuts[donor_id][step][1],
                            arm, name=row_id, on_receipt=store.receipt)
                        counts["adapter_steps"] += 1
                        counts["acts"] += len(receipts)
                        post = branch.capture()
                        store.save(post)
                        finish(branch, {"id": row_id, "prompt": "base", "donor": donor_id, "kind": arm, "step": step,
                            "parent": parent.fingerprint, "recipient_projected": projected.fingerprint,
                            "donor_projected": cuts[donor_id][step][1].fingerprint, "post_act": post.fingerprint,
                            "acts": receipts, "undeclared_slots_exact": True})
            return {"rows": results, "counts": dict(counts), "source_sha256": file_sha(output / "source.json")}

        tx.execute("native", execute)
    report = tx.report()
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    save(output / "transaction-report.json", report)
    summary = {"schema": "mdb-flux-concept-routes-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "mechanics_status": "valid", "terminal_status": "unassessed",
        "counts": dict(counts), "model_identity": source["model_identity"], "seal": seal,
        "transaction_verification": verification, "worker_seconds": time.monotonic() - started,
        "scope": config["scope"], "RGB_forecast": False}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): file_sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config["max_archive_bytes"])


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-concept-result")
        if output.is_dir():
            support = runpy.run_path("flux_trajectory_forecast_campaign.py")
            report = {"schema": "mdb-flux-concept-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())["request_sha256"],
                "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            support["save"](output / "failure.json", report)
            runpy.run_path("campaign_support.py")["emit"](output, report, max_compressed_bytes=512 << 20)
        raise
