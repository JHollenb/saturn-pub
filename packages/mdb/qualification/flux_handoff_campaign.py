"""Locate within-step carrier handoff using real typed restores and RGB futures."""
from __future__ import annotations

import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    support = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, file_sha, identity = (support[key] for key in ("save", "file_sha", "identity"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(file_sha(name) != value for name, value in request["source_hashes"].items()):
        raise ValueError("staged request/source differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-handoff-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, sequence = time.monotonic(), 0
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "prompt_encodings": 0, "physical_batch": 1}

    def event(kind, **details):
        nonlocal sequence
        row = {"kind": kind, "sequence": sequence, "seconds": time.monotonic() - started, **details}
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_HANDOFF_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite worker wall budget exhausted")

    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    # Prompt encoder is intentionally never run: these are verified old cuts.
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    cuts = {key: inputs.load(value, device="cuda") for key, value in config["input_cuts"].items()}
    for cut in cuts.values():
        if cut.model_identity != adapter.model_identity:
            raise ValueError("hydrated model identity differs")
        store.save(cut)
    # Source contrast changes only C/T. No changed latent, position, gates or schedule.
    base, relation = cuts["base"], cuts["relation"]
    for slot in base.payload:
        if slot not in {"text", "conditioning"} and describe(base.payload[slot]) != describe(relation.payload[slot]):
            raise ValueError("source contrast changes nuisance slot: " + slot)

    def on_receipt(receipt):
        body = receipt.to_dict()
        store.receipt(receipt)
        if body["operation"] == "continue":
            counts["adapter_steps"] += body["steps"]
        else:
            counts["acts"] += 1
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite native operation budget exhausted")

    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=on_receipt)

    a, d = navigator(base), navigator(base)
    seed_act = Act.replace("text", relation.payload["text"], name="seed.changed-working-text")
    seed_receipt = d.apply(seed_act)
    changed = d.capture()
    store.save(changed)
    expected = dict(base.payload)
    expected["text"] = relation.payload["text"]
    if describe(changed.payload) != describe(expected):
        raise ValueError("changed branch altered undeclared initial state")
    source = {"schema": "mdb-flux-handoff-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "request_sha256": request["request_sha256"],
        "packages": {key: importlib.metadata.version(key) for key in ("mdb-runtime", "saturn-pub", "torch", "diffusers", "transformers")},
        "input_cuts": config["input_cuts"], "base": base.fingerprint, "changed": changed.fingerprint,
        "seed_act": seed_act.manifest(), "seed_receipt": seed_receipt.to_dict(),
        "predecessor": config["predecessor"], "prompt_encodings": 0,
        "scope": "same seed31 known contrast; native causal continuation, not a numerical image forecast"}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        frontier={"kind": "architectural", "source_sha256": file_sha(output / "source.json"),
            "base": base.fingerprint, "changed": changed.fingerprint,
            "allowed_prefix": "previous observed timestep0 projection; no new native block or RGB outcome yet"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("handoff_hypotheses", lambda context: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows, frontiers = [], {"base": {}, "changed": {}}

    def finish(nav, row, *, capture_reference=None):
        event("branch_start", row=row["id"])
        schedulers = []
        while nav.point["logical_step"] < config["steps"]:
            wall()
            point = nav.point
            if capture_reference is not None and point["logical_step"] == 0:
                for boundary in config["boundaries"]:
                    if all(point[key] == value for key, value in boundary["selector"].items()):
                        cut = nav.capture()
                        store.save(cut)
                        frontiers[capture_reference][boundary["id"]] = cut
            nav.step()
            if point["phase"] == "scheduler":
                step = nav.point["logical_step"]
                latent = nav.session.read("latent")
                path = output / f"{row['id']}.step{step}.npy"
                np.save(path, latent.float().cpu().numpy(), allow_pickle=False)
                schedulers.append({"step": step, "path": path.name, "sha256": file_sha(path), "descriptor": tensor_descriptor(latent)})
        if counts["images"] >= config["max_images"]:
            raise RuntimeError("finite image budget exhausted")
        path = output / (row["id"] + ".png")
        adapter.decode(nav.session, height=512, width=512).save(path)
        counts["images"] += 1
        row.update(image={"path": path.name, "sha256": file_sha(path)}, schedulers=schedulers)
        rows.append(row)
        save(output / "rows" / (row["id"] + ".json"), row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def restore(boundary, slots, name, *, sham=False):
        parent, donor = frontiers["changed"][boundary], frontiers["base"][boundary]
        nav, acts = navigator(parent), []
        expected = dict(parent.payload)
        for slot in slots:
            value = parent.payload[slot] if sham else donor.payload[slot]
            act = Act.replace(slot, value, name=name + "." + slot)
            receipt = nav.apply(act)
            acts.append({"act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": receipt.fingerprint})
            expected[slot] = value
        post = nav.capture()
        store.save(post)
        if describe(post.payload) != describe(expected):
            raise ValueError("restoration changed undeclared closure")
        finish(nav, {"id": name, "kind": "sham" if sham else "restore", "boundary": boundary,
            "slots": slots, "parent": parent.fingerprint, "donor": donor.fingerprint, "post_act": post.fingerprint,
            "acts": acts, "undeclared_slots_exact": True})

    with tx:
        event("hypotheses_sealed", seal=seal, counts=dict(counts))

        def execute(context):
            finish(a, {"id": "base", "kind": "reference", "parent": base.fingerprint}, capture_reference="base")
            finish(d, {"id": "changed", "kind": "reference", "parent": changed.fingerprint}, capture_reference="changed")
            if any(set(values) != {b["id"] for b in config["boundaries"]} for values in frontiers.values()):
                raise ValueError("missing requested breakpoint")
            save(output / "frontiers.json", {key: {name: cut.fingerprint for name, cut in values.items()} for key, values in frontiers.items()})
            for boundary in config["boundaries"]:
                for slot in ("text", "image"):
                    restore(boundary["id"], [slot], boundary["id"] + ".restore_" + slot)
            late = config["boundaries"][-1]["id"]
            restore(late, ["text", "image"], "late.sham", sham=True)
            restore(late, ["text", "image"], "late.restore_both")
            return {"rows": rows, "counts": dict(counts), "source_sha256": file_sha(output / "source.json")}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-handoff-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counts, "mechanics_status": "valid",
        "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started,
        "model_identity": adapter.model_identity, "seal": seal, "transaction_verification": verification,
        "RGB_forecast": False, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): file_sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=256 << 20)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-handoff-result")
        if output.is_dir():
            report = {"schema": "mdb-flux-handoff-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())["request_sha256"],
                "error": f"{type(error).__name__}: {error}", "mechanics_status": "errored"}
            runpy.run_path("flux_trajectory_forecast_campaign.py")["save"](output / "failure.json", report)
            runpy.run_path("campaign_support.py")["emit"](output, report, max_compressed_bytes=256 << 20)
        raise
