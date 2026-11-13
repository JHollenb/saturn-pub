"""Same-parent shape/source timing assay; native futures, no numerical RGB forecast."""
from __future__ import annotations

import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def replace_field(session, slot, value, *, name, store):
    """One typed edit, with a complete declared-closure check and durable cuts."""
    from saturn_pub import Act
    from saturn_pub.values import describe

    before = session.capture()
    store.save(before)
    expected = dict(before.payload)
    expected[slot] = value
    act = Act.replace(slot, value, name=name)
    receipt = session.apply(act)
    store.receipt(receipt)
    after = session.capture()
    if describe(after.payload) != describe(expected):
        raise ValueError("edit changed undeclared state")
    store.save(after)
    return {"slot": slot, "parent": before.fingerprint, "after": after.fingerprint,
            "act": act.manifest(), "receipt": receipt.to_dict(), "undeclared_slots_exact": True}


def prepare_hypotheses(root, *, model_identity, parent, source_hashes, config):
    from mdb_lanes.forecast_transaction import ForecastTransaction

    tx = ForecastTransaction(root, model_identity=model_identity,
        input_identity=parent, source_identity=source_hashes,
        frontier={"kind": "architectural", "boundary": "fresh seed47 projected parent before joint0; observed encoder states"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    try:
        tx.predict("shape_timing_hypotheses", lambda context: {"predictions": config["hypotheses"]})
        return tx, tx.seal()
    except BaseException:
        tx.close()
        raise


def result_summary(*, job_id, request_sha256, counts, rows, seal, seconds, verification):
    return {"schema": "mdb-flux-shape-timing-v1", "job_id": job_id,
        "request_sha256": request_sha256, "mechanics_status": "valid", "terminal_status": "unassessed",
        "RGB_forecast": False, "counts": counts, "rows": rows, "seal": seal,
        "worker_seconds": seconds, "transaction_verification": verification}


def worker():
    support = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, sha, identity = (support[k] for k in ("save", "file_sha", "identity"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(sha(n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("request/source differs")
    for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[n] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator

    branch_carriers = runpy.run_path("flux_concept_routes.py")["branch_carriers"]
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-shape-timing-result")
    output.mkdir(exist_ok=False)
    started, counts = time.monotonic(), {"transitions": 0, "acts": 0, "images": 0, "encodings": 0, "loads": 1}

    def event(kind, **detail):
        row = {"kind": kind, "seconds": time.monotonic() - started, **detail}
        with (output / "timeline.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print("MDB_FLUX_SHAPE_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(files) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.scheduler.config.stochastic_sampling or pipe.transformer.config.guidance_embeds:
        raise ValueError("requires deterministic distilled Klein")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    inputs, store = LocalStore("input-store"), LocalStore(output / "typed-store")
    donors = {n: inputs.load(h, device="cuda") for n, h in config["input_cuts"].items()}
    for cut in donors.values():
        if cut.model_identity != adapter.model_identity:
            raise ValueError("source model identity differs")
    b, d = donors["base"].payload, donors["shape"].payload
    for slot in ("context", "img_ids", "txt_ids", "sigmas", "timesteps", "image", "latent"):
        if describe(b[slot]) != describe(d[slot]):
            raise ValueError("source nuisance differs: " + slot)
    with torch.inference_mode():
        latent, ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
            b["conditioning"].dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
    session = adapter.session(latent=latent, conditioning=b["conditioning"], img_ids=ids,
        txt_ids=b["txt_ids"], timesteps=b["timesteps"].clone(), sigmas=b["sigmas"].clone())
    preproject = session.capture()
    store.save(preproject)
    store.receipt(session.continue_(1))
    counts["transitions"] += 1
    parent = session.capture()
    store.save(parent)
    source = {"checkpoint_sha256": identity(files), "model_identity": adapter.model_identity,
        "parent": parent.fingerprint, "preproject": preproject.fingerprint, "input_cuts": config["input_cuts"], "request": request,
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}}
    save(output / "source.json", source)
    save(output / "plan.json", config)
    tx, seal = prepare_hypotheses(output / "transaction", model_identity=adapter.model_identity,
        parent=parent.fingerprint, source_hashes=request["source_hashes"], config=config)
    event("hypotheses_sealed", seal=seal, counts=dict(counts))
    rows, histories = {}, {}

    def edit(s, slot, value, name):
        row = replace_field(s, slot, value, name=name, store=store)
        counts["acts"] += 1
        return row

    def finish(s, name, *, acts, prefix=(), capture_history=False):
        wall()
        nav = NativeCutNavigator(s, max_steps=112, on_receipt=store.receipt)
        history, schedulers = {}, list(prefix)
        if nav.point["phase"] == "project":
            history[nav.point["logical_step"]] = nav.capture()
        event("branch_start", branch=name)
        while nav.point["logical_step"] < 4:
            wall()
            if counts["transitions"] >= config["max_transitions"]:
                raise RuntimeError("transition budget exhausted")
            phase = nav.point["phase"]
            nav.step()
            counts["transitions"] += 1
            if phase == "scheduler":
                state = nav.session.capture(retain=False).payload
                step = state["step"]
                path = output / f"{name}.step{step}.npy"
                np.save(path, state["latent"].float().cpu().numpy(), allow_pickle=False)
                schedulers.append({"step": step, "path": path.name, "sha256": sha(path)})
                if capture_history and step < 4:
                    history[step] = nav.capture()
                    store.save(history[step])
        if counts["images"] >= config["max_images"]:
            raise RuntimeError("image budget exhausted")
        path = output / (name + ".png")
        adapter.decode(nav.session, height=512, width=512).save(path)
        counts["images"] += 1
        row = {"id": name, "acts": acts, "schedulers": schedulers, "image": {"path": path.name, "sha256": sha(path)}}
        rows[name] = row
        save(output / "rows" / (name + ".json"), row)
        event("branch_complete", branch=name, counts=dict(counts))
        return history

    def execute(context):
        finish(Session.from_cut(adapter, parent), "base", acts=[])
        for name, arm in (("early", "text"), ("coherent", "both")):
            s, acts = branch_carriers(session, preproject, parent, donors["shape"], arm,
                name=name, on_receipt=store.receipt)
            counts["transitions"] += 1
            counts["acts"] += len(acts)
            store.save(s.capture())
            histories[name] = finish(s, name, acts=acts, capture_history=True)
        s, acts = branch_carriers(session, preproject, parent, donors["shape"], "conditioning",
            name="source_only", on_receipt=store.receipt)
        counts["transitions"] += 1
        counts["acts"] += len(acts)
        store.save(s.capture())
        finish(s, "source_only", acts=acts)
        for history, target in (("early", d["conditioning"]), ("coherent", b["conditioning"])):
            for step in (1, 2, 3):
                name = history + f".switch{step}"
                cut = histories[history][step]
                s = Session.from_cut(adapter, cut)
                acts = [edit(s, "conditioning", target, name + ".conditioning")]
                prefix = [r for r in rows[history]["schedulers"] if r["step"] <= step]
                finish(s, name, acts=acts, prefix=prefix)
        return {"rows": rows, "counts": dict(counts)}

    with tx:
        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    summary = result_summary(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        counts=counts, rows=rows, seal=seal, seconds=time.monotonic() - started,
        verification=verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"]))
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config["max_archive_bytes"])


if __name__ == "__main__":
    worker()
