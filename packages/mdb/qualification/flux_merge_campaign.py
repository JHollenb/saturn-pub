"""Factor a localized native parallel merge into residual/attention/MLP operands."""
from __future__ import annotations

import itertools
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
        raise ValueError("request/source differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    candidate = runpy.run_path("flux_parallel_merge.py")["merge_candidate"]
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-merge-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, sequence = time.monotonic(), 0
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "projection_evaluations": 0,
              "prompt_encodings": 0, "receipt_emissions": 0, "physical_batch": 1}

    def event(kind, **details):
        nonlocal sequence
        row = {"kind": kind, "seconds": time.monotonic() - started, "sequence": sequence, **details}
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_MERGE_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    before = {name: inputs.load(cut_id, device="cuda") for name, cut_id in config["input_cuts"].items()}
    for cut in before.values():
        store.save(cut)
        if cut.model_identity != adapter.model_identity or any(cut.execution_point[k] != v for k, v in config["selector"].items()):
            raise ValueError("input model or selected boundary differs")
    for slot in before["base"].payload:
        if slot not in {"text", "image"} and describe(before["base"].payload[slot]) != describe(before["changed"].payload[slot]):
            raise ValueError("unmatched source closure: " + slot)

    def on_receipt(receipt):
        store.receipt(receipt)
        body = receipt.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "receipt": receipt.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        if body["operation"] == "continue":
            counts["adapter_steps"] += body["steps"]
        else:
            counts["acts"] += 1
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("native budget exceeded")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=on_receipt)

    target = pipe.transformer.single_transformer_blocks[config["block"]].attn
    projection, ports, after, prefix_receipts = target.to_out, {}, {}, {}
    text_rows = before["base"].payload["text"].shape[1]
    gate = Flux2Modulation.split(before["base"].payload["context"]["single_mod"], 1)[0][2]
    residuals = {name: torch.cat([cut.payload["text"], cut.payload["image"]], dim=1) for name, cut in before.items()}

    def save_arrays(name, values):
        path = output / (name + ".npz")
        np.savez_compressed(path, **{key: value.detach().float().cpu().numpy() for key, value in values.items()})
        return {"path": path.name, "sha256": file_sha(path), "descriptors": {key: tensor_descriptor(value) for key, value in values.items()}}

    # Read-only observation hooks. All interventions happen later as typed image Acts.
    port_records = {}
    for name, cut in before.items():
        captured = {}

        def input_hook(module, args):
            if "operands" in captured:
                raise ValueError("projection entered more than once")
            captured["operands"] = args[0].detach().clone()

        def output_hook(module, args, result):
            captured["projection"] = result.detach().clone()

        hooks = [projection.register_forward_pre_hook(input_hook), projection.register_forward_hook(output_hook)]
        session = nav(cut)
        try:
            receipt = session.step()
        finally:
            for hook in hooks:
                hook.remove()
        if set(captured) != {"operands", "projection"}:
            raise ValueError("native projection ports missing")
        after[name] = session.capture()
        store.save(after[name])
        ports[name] = captured
        prefix_receipts[name] = receipt.to_dict()
        port_records[name] = save_arrays(name + ".native-ports", {**captured, "residual": residuals[name], "gate": gate})
    source = {"schema": "mdb-flux-merge-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "input_cuts": config["input_cuts"],
        "post_cuts": {name: cut.fingerprint for name, cut in after.items()}, "prefix_receipts": prefix_receipts,
        "ports": port_records, "attention_width": target.inner_dim, "text_rows": text_rows,
        "projection_path": f"transformer.single_transformer_blocks.{config['block']}.attn.to_out",
        "weight": tensor_descriptor(projection.weight), "bias": None if projection.bias is None else tensor_descriptor(projection.bias),
        "scope": "observed native R/A/M ports and two blocks before seal; original frozen projection, typed image-only post-block Acts; no independent edited-image forecast"}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request["source_hashes"]),
        frontier={"kind": "architectural", "source_sha256": file_sha(output / "source.json"),
            "allowed_prefix": "two native selected-block executions to observe actual merge ports; no RGB outcomes"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("merge_hypotheses", lambda context: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows = []

    def finish(session, row):
        event("branch_start", row=row["id"])
        schedulers = []
        while session.point["logical_step"] < config["steps"]:
            wall()
            point = session.point
            session.step()
            if point["phase"] == "scheduler":
                value = session.session.read("latent")
                path = output / f"{row['id']}.step{session.point['logical_step']}.npy"
                np.save(path, value.float().cpu().numpy(), allow_pickle=False)
                schedulers.append({"step": session.point["logical_step"], "path": path.name, "sha256": file_sha(path), "descriptor": tensor_descriptor(value)})
        if counts["images"] >= config["max_images"]:
            raise RuntimeError("image budget exceeded")
        path = output / (row["id"] + ".png")
        adapter.decode(session.session, height=512, width=512).save(path)
        counts["images"] += 1
        row.update(schedulers=schedulers, image={"path": path.name, "sha256": file_sha(path)})
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def install(image, row):
        session = nav(after["changed"])
        act = Act.replace("image", image, name=row["id"])
        receipt = session.apply(act)
        post = session.capture()
        store.save(post)
        expected = dict(after["changed"].payload)
        expected["image"] = image
        if describe(post.payload) != describe(expected):
            raise ValueError("Act changed undeclared closure")
        row.update(parent=after["changed"].fingerprint, post_act=post.fingerprint, act=act.manifest(), receipt=receipt.to_dict())
        finish(session, row)

    with tx:
        event("hypotheses_sealed", counts=dict(counts), seal=seal)

        def execute(context):
            for name in ("base", "changed"):
                finish(nav(after[name]), {"id": name, "kind": "reference", "parent": after[name].fingerprint})
            install(after["changed"].payload["image"], {"id": "copy.sham", "kind": "copy"})
            for bits in itertools.product("01", repeat=3):
                wall()
                mask = "".join(bits)
                with torch.inference_mode():
                    result = candidate(recipient_residual=residuals["changed"], donor_residual=residuals["base"],
                        recipient_operands=ports["changed"]["operands"], donor_operands=ports["base"]["operands"],
                        gate=gate, projection=projection, attention_width=target.inner_dim, text_rows=text_rows, mask=mask)
                counts["projection_evaluations"] += 1
                native_changed = torch.cat([after["changed"].payload["text"], after["changed"].payload["image"]], dim=1)
                row = {"id": "merge." + mask, "kind": "merge", "mask_R_A_M": mask,
                    "operands_descriptor": tensor_descriptor(result["operands"]),
                    "projection": save_arrays("merge." + mask + ".projection", {"value": result["projection"]}),
                    "text_rows_exact_changed": torch.equal(result["updated"][:, :text_rows], native_changed[:, :text_rows]),
                    "self_full_state_exact_changed": torch.equal(result["updated"], native_changed) if mask == "000" else None,
                    "all_donor_image_exact_base": torch.equal(result["updated"][:, text_rows:], after["base"].payload["image"]) if mask == "111" else None}
                install(result["updated"][:, text_rows:], row)
            return {"rows": rows, "counts": dict(counts), "source_sha256": file_sha(output / "source.json")}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-merge-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counts, "mechanics_status": "valid",
        "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started, "seal": seal,
        "transaction_verification": verification, "RGB_forecast": False, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): file_sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=256 << 20)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-merge-result")
        if output.is_dir():
            report = {"schema": "mdb-flux-merge-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())["request_sha256"],
                "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            runpy.run_path("flux_trajectory_forecast_campaign.py")["save"](output / "failure.json", report)
            runpy.run_path("campaign_support.py")["emit"](output, report, max_compressed_bytes=256 << 20)
        raise
