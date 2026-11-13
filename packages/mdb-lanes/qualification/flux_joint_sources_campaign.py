"""Joint4 direct-versus-MLP mediation and complete Q/textKV/imageKV factorial."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def worker():
    h = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, sha, identity = (h[k] for k in ("save", "file_sha", "identity"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(sha(k) != v for k, v in request["source_hashes"].items()):
        raise ValueError("request/source differs")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    candidate = runpy.run_path("flux_joint_sources.py")["source_candidate"]
    install = runpy.run_path("flux_joint_writes.py")["install_image"]
    reset = runpy.run_path("flux_handoff_campaign.py")["reset_conditioning"]
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-joint-sources-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started = time.monotonic()
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "candidate_block_calls": 0,
              "receipt_emissions": 0, "prompt_encodings": 0, "physical_batch": 1}

    def event(kind, **fields):
        row = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        with (output / "timeline.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print("MDB_FLUX_JOINT_SOURCES_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    block = pipe.transformer.transformer_blocks[4]
    if (type(block.attn.processor).__name__ != "Flux2AttnProcessor" or block.attn.processor._attention_backend is not None
            or block.attn.processor._parallel_config is not None):
        raise ValueError("requires native default attention backend")
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    cuts = {k: inputs.load(v, device="cuda") for k, v in config["input_cuts"].items()}
    for cut in cuts.values():
        if cut.model_identity != adapter.model_identity or any(cut.execution_point[k] != v for k, v in
            {"logical_step": 1, "phase": "joint", "index": 4}.items()):
            raise ValueError("source identity/clock differs")
        store.save(cut)
    if cuts["base"].execution_point != cuts["changed"].execution_point:
        raise ValueError("source clocks differ")
    for key in cuts["base"].payload:
        if key not in {"text", "image", "conditioning"} and describe(cuts["base"].payload[key]) != describe(cuts["changed"].payload[key]):
            raise ValueError("source nuisance differs: " + key)

    def receipt(r):
        store.receipt(r)
        body = r.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as f:
            f.write(json.dumps({"sequence": counts["receipt_emissions"], "receipt": r.fingerprint}) + "\n")
        counts["receipt_emissions"] += 1
        if body["operation"] == "continue":
            counts["adapter_steps"] += body["steps"]
        else:
            counts["acts"] += 1
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite transition budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def calculate(cut, **kwargs):
        wall()
        counts["candidate_block_calls"] += 1
        if counts["candidate_block_calls"] > config["max_candidate_calls"]:
            raise RuntimeError("finite candidate budget exhausted")
        return candidate(block, cut, **kwargs)

    def arrays(name, values):
        path = output / (name + ".npz")
        np.savez_compressed(path, **{k: v.detach().float().cpu().numpy() for k, v in values.items()})
        return {"path": path.name, "sha256": sha(path), "descriptors": {k: tensor_descriptor(v) for k, v in values.items()}}

    source = {"schema": "mdb-flux-joint-sources-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "input_cuts": config["input_cuts"], "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
            "allowed_prefix": "two audited joint4-entry parents; earlier intervention images inform hypotheses; no new selected-block or RGB calls"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("joint_source_hypotheses", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    observed, values, posts, rows = {}, {}, {}, []

    def observe(name):
        value = calculate(cuts[name])
        n = nav(cuts[name])
        r = n.step()
        post = n.capture()
        store.save(post)
        if not torch.equal(post.payload["image"], value["image"]) or not torch.equal(post.payload["text"], value["text"]):
            raise ValueError("candidate instrumentation differs from native adapter")
        values[name], posts[name] = value, post
        observed[name] = {"parent": cuts[name].fingerprint, "post": post.fingerprint, "native_receipt": r.to_dict(),
            "ports": arrays("ports." + name, value["ports"]),
            "qkv": arrays("positioned-qkv." + name, value["qkv"]["observed"]),
            "text_rows": value["text_rows"], "dispatch": value["qkv"]["dispatch"], "native_image_and_text_exact": True}

    def finish(n, row):
        if any(n.point[k] != v for k, v in {"logical_step": 1, "phase": "single", "index": 0}.items()):
            raise ValueError("text clamp at wrong clock")
        before = n.capture()
        store.save(before)
        act = Act.replace("text", posts["base"].payload["text"], name=row["id"] + ".common-text")
        r = n.apply(act)
        after = n.capture()
        store.save(after)
        if describe(after.payload) != describe({**before.payload, "text": posts["base"].payload["text"]}):
            raise ValueError("text clamp changed undeclared closure")
        row.setdefault("acts", []).append({"parent": before.fingerprint, "post_act": after.fingerprint,
            "act": act.manifest(), "receipt": r.to_dict()})
        schedulers = []
        while n.point["logical_step"] < config["steps"]:
            wall()
            point = n.point
            if point["logical_step"] == 2 and point["phase"] == "project":
                row["acts"].append(reset(n, cuts["base"].payload["conditioning"], step=2,
                    name=row["id"] + ".common-future-source", store=store))
            n.step()
            if point["phase"] == "scheduler":
                step = n.point["logical_step"]
                schedulers.append({"step": step, **arrays(row["id"] + f".step{step}", {"latent": n.session.read("latent")})})
        final = n.capture()
        store.save(final)
        path = output / (row["id"] + ".png")
        adapter.decode(n.session, height=512, width=512).save(path)
        counts["images"] += 1
        if counts["images"] > config["max_images"]:
            raise RuntimeError("finite image budget exhausted")
        row.update(final_cut=final.fingerprint, schedulers=schedulers, image={"path": path.name, "sha256": sha(path)})
        rows.append(row)
        save(output / "rows" / (row["id"] + ".json"), row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def arm(spec):
        n = nav(cuts["changed"])
        kwargs = {}
        if spec["kind"] == "output":
            kwargs["attention"] = values["base"]["ports"]["attention"]
            if spec.get("lock_changed_mlp"):
                kwargs["mlp"] = values["changed"]["ports"]["mlp"]
        else:
            kwargs.update(donor=values["base"]["qkv"]["observed"], mask=spec.get("mask", "000"),
                          text_query=spec.get("text_query", False))
        value = calculate(cuts["changed"], **kwargs)
        r = n.step()
        native_post = n.capture()
        act = install(n, value["image"], name=spec["id"] + ".image-candidate", store=store)
        row = {**spec, "parent": cuts["changed"].fingerprint, "native_post": native_post.fingerprint,
            "native_receipt": r.to_dict(), "acts": [act],
            "ports": arrays(spec["id"] + ".ports", value["ports"]),
            "effective_qkv": {k: tensor_descriptor(v) for k, v in value["qkv"]["effective"].items()},
            "dispatch": value["qkv"]["dispatch"], "equation_exact": value["equation_exact"],
            "image_matches_changed": torch.equal(value["image"], values["changed"]["image"]),
            "computed_attention_matches_base": torch.equal(value["ports"]["attention"], values["base"]["ports"]["attention"]),
            "note": "Candidate text return discarded; exact base text clamp follows at same joint4 exit. Native receipts use original unmodified block."}
        finish(n, row)

    with tx:
        event("hypotheses_sealed", seal=seal)

        def execute(_):
            for name in ("base", "changed"):
                observe(name)
            save(output / "observed-sources.json", observed)
            for name in ("base", "changed"):
                finish(nav(posts[name]), {"id": name, "kind": "reference", "parent": posts[name].fingerprint})
            for spec in config["branches"]:
                arm(spec)
            return {"rows": rows, "counts": dict(counts), "observed_sources_sha256": sha(output / "observed-sources.json")}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    checked = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-joint-sources-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "model_identity": adapter.model_identity,
        "mechanics_status": "valid", "terminal_status": "unassessed", "counts": counts,
        "worker_seconds": time.monotonic() - started, "seal": seal, "transaction_verification": checked,
        "scope": config["scope"], "RGB_forecast": False}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    event("publishing", raw_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=512 << 20)


if __name__ == "__main__":
    worker()
