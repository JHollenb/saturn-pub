"""Localize step1 joint1–4 attention and image-MLP writes through native RGB."""
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
    if request["config"] != config or any(sha(k) != v for k, v in request["source_hashes"].items()):
        raise ValueError("request/source mismatch")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
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

    candidate = runpy.run_path("flux_joint_writes.py")["joint_candidate"]
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
    output = Path("flux-joint-writes-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started = time.monotonic()
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "candidate_block_calls": 0,
              "prompt_encodings": 0, "physical_batch": 1}

    def event(kind, **fields):
        row = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        with (output / "timeline.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print("MDB_FLUX_JOINT_WRITES_PHASE " + json.dumps(row, sort_keys=True), flush=True)

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
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    origins = {k: inputs.load(v, device="cuda") for k, v in config["input_cuts"].items()}
    for cut in origins.values():
        if cut.model_identity != adapter.model_identity or any(cut.execution_point[k] != v for k, v in
            {"logical_step": 1, "phase": "joint", "index": 1}.items()):
            raise ValueError("source identity/clock mismatch")
        store.save(cut)
    for slot in origins["base"].payload:
        if slot not in {"image", "text", "conditioning"} and describe(origins["base"].payload[slot]) != describe(origins["changed"].payload[slot]):
            raise ValueError("source nuisance mismatch: " + slot)

    def receipt(r):
        store.receipt(r)
        body = r.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite transition budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def calculate(cut, *, attention=None, mlp=None):
        wall()
        index = cut.execution_point["index"]
        value = candidate(pipe.transformer.transformer_blocks[index], cut, attention=attention, mlp=mlp)
        counts["candidate_block_calls"] += 1
        if counts["candidate_block_calls"] > 50:
            raise RuntimeError("finite candidate budget exhausted")
        return value

    def arrays(name, values):
        path = output / (name + ".npz")
        np.savez_compressed(path, **{k: v.detach().float().cpu().numpy() for k, v in values.items()})
        return {"path": path.name, "sha256": sha(path), "descriptors": {k: tensor_descriptor(v) for k, v in values.items()}}

    # Declare hypotheses before any new native joint block or image continuation.
    source = {"schema": "mdb-flux-joint-writes-source-v1", "model_identity": adapter.model_identity,
              "checkpoint_sha256": identity(inventory), "input_cuts": config["input_cuts"], "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
                  "allowed_prefix": "observed step1 after-joint0 parents; historical coarse outcomes; no new blocks or RGB"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("joint_write_hypotheses", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    donors, frontiers, rows = {}, {}, []

    def native_prefix(name):
        n = nav(origins[name])
        for index in range(1, 5):
            before = n.capture()
            store.save(before)
            value = calculate(before)
            n.step()
            after = n.capture()
            store.save(after)
            if not torch.equal(value["image"], after.payload["image"]) or not torch.equal(value["text"], after.payload["text"]):
                raise ValueError("direct candidate does not reproduce native block")
            record = {"parent": before.fingerprint, "post": after.fingerprint,
                      "native_receipt": n.session.receipts[-1].to_dict(), "equation_exact": value["equation_exact"],
                      "ports": arrays(f"ports.{name}.j{index}", value["ports"])}
            donors[name, index] = value["ports"]
            source.setdefault("prefixes", {}).setdefault(name, {})[str(index)] = record
        frontiers[name] = n.capture()

    def finish(n, row):
        actions = row.setdefault("acts", [])
        # End the observation window by disposing of changed text identically.
        if n.point != frontiers["base"].execution_point:
            raise ValueError("text clamp at wrong clock")
        before = n.capture()
        store.save(before)
        act = Act.replace("text", frontiers["base"].payload["text"], name=row["id"] + ".text-base")
        r = n.apply(act)
        post = n.capture()
        store.save(post)
        if describe(post.payload) != describe({**before.payload, "text": frontiers["base"].payload["text"]}):
            raise ValueError("text clamp closure differs")
        actions.append({"parent": before.fingerprint, "post_act": post.fingerprint, "act": act.manifest(), "receipt": r.to_dict()})
        schedulers = []
        while n.point["logical_step"] < config["steps"]:
            wall()
            point = n.point
            if point["logical_step"] == 2 and point["phase"] == "project":
                actions.append(reset(n, origins["base"].payload["conditioning"], step=2,
                    name=row["id"] + ".future-C-base", store=store))
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
        row.update(final_cut=final.fingerprint, image={"path": path.name, "sha256": sha(path)}, schedulers=schedulers)
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def intervention(spec):
        n = nav(origins["changed"])
        row = {**spec, "parent": origins["changed"].fingerprint, "acts": [], "calculations": []}
        if spec.get("initial_base_image"):
            row["acts"].append(install(n, origins["base"].payload["image"], name=spec["id"] + ".initial-I-base", store=store))
        for index in range(1, 5):
            if index not in spec["blocks"]:
                n.step()
                continue
            before = n.capture()
            store.save(before)
            donor = donors["changed" if spec.get("sham") else "base", index]
            value = calculate(before,
                attention=donor["attention"] if "A" in spec["ports"] else None,
                mlp=donor["mlp"] if "M" in spec["ports"] else None)
            # Native post-block receipt remains truthful; candidate applied separately.
            n.step()
            native = n.capture()
            if not torch.equal(value["text"], native.payload["text"]):
                raise ValueError("image-port edit unexpectedly changed current text")
            record = install(n, value["image"], name=spec["id"] + f".j{index}.image", store=store)
            row["acts"].append(record)
            row["calculations"].append({"input": before.fingerprint, "native_post": native.fingerprint,
                "block": index, "donor": "changed" if spec.get("sham") else "base",
                "ports": spec["ports"], "equation_exact": value["equation_exact"],
                "values": arrays(spec["id"] + f".j{index}.candidate", {"image": value["image"], **value["ports"]})})
        finish(n, row)

    with tx:
        event("hypotheses_sealed", seal=seal)

        def execute(_):
            for name in ("base", "changed"):
                native_prefix(name)
            save(output / "observed-prefixes.json", source)
            for name in ("base", "changed"):
                finish(nav(frontiers[name]), {"id": name, "kind": "reference", "parent": frontiers[name].fingerprint})
            for spec in config["branches"]:
                intervention(spec)
            return {"rows": rows, "counts": dict(counts), "observed_prefixes_sha256": sha(output / "observed-prefixes.json")}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-joint-writes-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counts, "mechanics_status": "valid",
        "instrument_status": "native-consumer-tested", "terminal_status": "unassessed", "RGB_forecast": False,
        "worker_seconds": time.monotonic() - started, "model_identity": adapter.model_identity,
        "seal": seal, "transaction_verification": verification, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=512 << 20)


if __name__ == "__main__":
    worker()
