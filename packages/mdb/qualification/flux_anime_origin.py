"""Reciprocal depth curve for a late appearance-bearing image residual."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def worker():
    helper = runpy.run_path("flux_scene_inventory_campaign.py")
    sha, identity, save = (helper[k] for k in ("sha", "identity", "save"))
    config, request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("staged request/source mismatch")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.native_navigation import NativeCutNavigator

    payload_root = Path.cwd()
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    output = Path("flux-anime-origin-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    external = Path(config["external_input_root"]).resolve(strict=True)
    for relative, binding in config["external_inputs"].items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "typed-store":
            raise ValueError("invalid input CAS path")
        origin = external / rel
        if origin.is_symlink() or not origin.resolve().is_relative_to(external) or origin.stat().st_size != binding["bytes"] or sha(origin) != binding["sha256"]:
            raise ValueError("historical CAS differs")
        dest = Path("input-store").joinpath(*rel.parts[1:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write(origin.read_bytes())
        if sha(dest) != binding["sha256"]:
            raise ValueError("hydrated input differs")
    save(output / "input-custody.json", {"root": str(external), "files": config["external_inputs"]})
    started, active = time.monotonic(), {"row": "loading"}
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "candidate_block_calls": 0, "receipt_emissions": 0,
              "model_loads": 0, "prompt_encodings": 0, "physical_batch": 1}

    def event(kind, **fields):
        value = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        with (output / "timeline.jsonl").open("a") as f:
            f.write(json.dumps(value, sort_keys=True) + "\n")
        print("MDB_FLUX_ANIME_ORIGIN_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(47)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = Path(config["model_path"])
    inventory = {p.relative_to(model).as_posix(): sha(p) for p in sorted(model.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or not torch.cuda.is_available():
        raise ValueError("checkpoint/CUDA differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    origins = {k: inputs.load(v, device="cuda") for k, v in config["input_cuts"].items()}
    for cut in origins.values():
        if cut.model_identity != adapter.model_identity or dict(cut.execution) != dict(adapter.execution) or cut.execution_point["logical_step"] != 3 or cut.execution_point["phase"] != "project":
            raise ValueError("input identity/execution/clock differs")
        store.save(cut)

    def receipt(r):
        wall()
        store.receipt(r)
        b = r.to_dict()
        counts["adapter_steps" if b["operation"] == "continue" else "acts"] += b.get("steps", 1)
        counts["receipt_emissions"] += 1
        with (output / "receipt-emissions.jsonl").open("a") as f:
            f.write(json.dumps({"row": active["row"], "receipt": r.fingerprint}) + "\n")
        if counts["adapter_steps"] > 600 or counts["acts"] > 34:
            raise RuntimeError("finite transition/Act budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=600, on_receipt=receipt)

    save(output / "source.json", {"model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory),
        "execution": dict(adapter.execution), "input_cuts": config["input_cuts"], "source_hashes": request["source_hashes"],
        "request_sha256": request["request_sha256"], "actual_execution_scope": "CUDA BF16 B1 unmodified public NativeCutNavigator; zero candidate block calls"})
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"), "allowed_prefix": config["scope"]},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("origin", lambda _: {"predictions": config["hypotheses"]})
    seal, frontiers, rows = tx.seal(), {}, []

    def finish(n, row):
        while n.point["logical_step"] < 4:
            n.step()
        final = n.capture()
        store.save(final)
        image = output / (row["id"] + ".png")
        adapter.decode(n.session, height=512, width=512).save(image)
        counts["images"] += 1
        if counts["images"] > 32:
            raise RuntimeError("finite image budget exhausted")
        row.update(final_cut=final.fingerprint, image={"path": image.name, "sha256": sha(image)})
        if row["id"] in config["historical_pngs"]:
            row["historical_control_exact"] = sha(image) == config["historical_pngs"][row["id"]]
            if not row["historical_control_exact"]:
                raise ValueError("historical control mismatch")
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def execute(_):
        for role in ("base", "source"):
            active["row"] = "native." + role
            n = nav(origins[role])
            for point in config["points"]:
                n.until(logical_step=3, phase=point["phase"], index=point["index"])
                cut = n.capture()
                store.save(cut)
                frontiers[role + "/" + point["name"]] = cut.fingerprint
            finish(n, {"id": "native." + role, "parent": origins[role].fingerprint})
        save(output / "frontiers.json", frontiers)
        for point in config["points"]:
            for recipient, donor in (("base", "source"), ("source", "base")):
                before, source = (store.load(frontiers[r + "/" + point["name"]], device="cuda") for r in (recipient, donor))
                for support in ("upper", "lower"):
                    active["row"] = key = recipient + "." + support + "." + point["name"]
                    n = nav(before)
                    image = before.payload["image"].clone()
                    selected = config["part_rows"][support]
                    image[:, selected] = source.payload["image"][:, selected]
                    r = n.apply(Act.replace("image", image, name=key))
                    post = n.capture()
                    store.save(post)
                    if describe(post.payload) != describe({**before.payload, "image": image}):
                        raise ValueError("band Act changed undeclared state")
                    finish(n, {"id": key, "parent": before.fingerprint, "source_cut": source.fingerprint,
                        "recipient": recipient, "support": support, "site": point, "post_act": post.fingerprint,
                        "act_receipt": r.fingerprint, "nonimage_and_outside_exact": True})
        for role in ("base", "source"):
            active["row"] = key = "sham." + role
            before = store.load(frontiers[role + "/single10"], device="cuda")
            n = nav(before)
            r = n.apply(Act.replace("image", before.payload["image"].clone(), name=key))
            post = n.capture()
            store.save(post)
            finish(n, {"id": key, "parent": before.fingerprint, "post_act": post.fingerprint, "act_receipt": r.fingerprint})
        return {"rows": rows, "counts": dict(counts)}

    with tx:
        event("predictions_sealed", seal=seal, counts=dict(counts))
        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    summary = {"schema": "mdb-anime-origin-v1", "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"], "counts": counts,
        "worker_seconds": time.monotonic() - started, "output_root": str(Path.cwd()), "mechanics_status": "valid", "terminal_status": "unassessed",
        "seal": seal, "verification": verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload_root / "campaign_support.py"))["emit"](output, summary, max_compressed_bytes=512 << 20, compression_level=3)


if __name__ == "__main__":
    worker()
