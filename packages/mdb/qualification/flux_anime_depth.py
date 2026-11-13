"""Coarse time/depth map, working-state transplants and carried-state rescues."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def worker():
    helpers = runpy.run_path("flux_scene_inventory_campaign.py")
    sha, identity, save = (helpers[k] for k in ("sha", "identity", "save"))
    config, request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("staged request/source differs")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from safetensors.torch import save_file
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.native_navigation import NativeCutNavigator

    output = Path("flux-anime-depth-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    root = Path(config["external_input_root"]).resolve(strict=True)
    for name, binding in config["external_inputs"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "typed-store":
            raise ValueError("invalid historical CAS path")
        source = root / relative
        if (source.is_symlink() or not source.resolve().is_relative_to(root) or source.stat().st_size != binding["bytes"]
                or sha(source) != binding["sha256"]):
            raise ValueError("historical input differs")
        dest = Path("input-store").joinpath(*relative.parts[1:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(source.read_bytes())
        if sha(dest) != binding["sha256"]:
            raise ValueError("hydrated input differs")
    save(output / "input-custody.json", {"source_root": str(root), "files": config["external_inputs"]})
    started, active = time.monotonic(), {"row": "loading"}
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "model_loads": 0, "prompt_encodings": 0,
              "candidate_block_calls": 0, "scheduler_equation_forecasts": 0, "physical_batch": 1, "receipt_emissions": 0}

    def event(kind, **fields):
        value = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        print("MDB_FLUX_ANIME_DEPTH_PHASE " + json.dumps(value, sort_keys=True), flush=True)

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
    model_root = Path(config["model_path"])
    inventory = {p.relative_to(model_root).as_posix(): sha(p) for p in sorted(model_root.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or not torch.cuda.is_available():
        raise ValueError("checkpoint/admitted CUDA differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(model_root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    pipe.set_progress_bar_config(disable=True)
    if pipe.scheduler.config.stochastic_sampling:
        raise ValueError("deterministic flow lane required")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    origins = {k: inputs.load(v, device="cuda") for k, v in config["input_cuts"].items()}
    for cut in origins.values():
        if (cut.model_identity != adapter.model_identity or dict(cut.execution) != dict(adapter.execution)
                or any(cut.execution_point[k] != v for k, v in config.get("start_point", {"logical_step": 2, "phase": "project"}).items())):
            raise ValueError("input identity/execution/clock differs")
        store.save(cut)
    if len(pipe.transformer.single_transformer_blocks) != 20 or len(pipe.transformer.transformer_blocks) != 5:
        raise ValueError("qualified depth differs")

    def receipt(r):
        wall()
        store.receipt(r)
        body = r.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        counts["receipt_emissions"] += 1
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"row": active["row"], "receipt": r.fingerprint}) + "\n")
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite native budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def clock(point):
        return f"s{point['logical_step']}.{point['phase']}." + str(point.get("index"))

    def difference(a, b):
        delta = a.float() - b.float()
        return {"exact": torch.equal(a, b), "mean_abs": float(delta.abs().mean()), "rms": float(delta.square().mean().sqrt()),
                "max_abs": float(delta.abs().max()), "changed_elements": int((a != b).sum()), "elements": a.numel()}

    source = {"schema": "mdb-flux-anime-depth-source-v1", "model_identity": adapter.model_identity, "execution": dict(adapter.execution),
        "checkpoint_sha256": identity(inventory), "input_cuts": config["input_cuts"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "actual_execution_scope": "CUDA BF16 B1 native Navigator; ledger counts actual transitions; zero candidate blocks",
        "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
        "allowed_prefix": config["scope"]}, budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("depth", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    frontiers, rows, traces, reference_features = {}, [], [], {}
    points_to_store = {"s2.single.0", "s2.single.10", "s2.single.19", "s2.scheduler.None", "s3.project.None",
                       "s3.single.10", "s3.single.19", "s3.scheduler.None"}

    def finish(n, row, *, observe=None, expected_latent=None, predicted_next_step=None):
        commits = []
        while n.point["logical_step"] < 4:
            point = n.point
            if observe is not None:
                observe(n)
            n.step()
            if point["phase"] == "scheduler":
                post = n.capture()
                store.save(post)
                record = {"step": n.point["logical_step"], "cut": post.fingerprint}
                if expected_latent is not None and n.point["logical_step"] == predicted_next_step:
                    record["forecast_exact"] = torch.equal(expected_latent, post.payload["latent"])
                    if not record["forecast_exact"]:
                        raise ValueError("source-velocity scheduler equation forecast did not close")
                if n.point["logical_step"] == 3:
                    donor = frontiers.get(row["scene"] + ".left/s3.project.None")
                    if donor is not None:
                        record["left_vs_single"] = difference(post.payload["latent"][:, config["left_rows"][row["scene"]]],
                            donor.payload["latent"][:, config["left_rows"][row["scene"]]])
                commits.append(record)
        final = n.capture()
        store.save(final)
        path = output / (row["id"] + ".png")
        adapter.decode(n.session, height=512, width=512).save(path)
        counts["images"] += 1
        if counts["images"] > config.get("max_images", 22):
            raise RuntimeError("finite image budget exhausted")
        row.update(final_cut=final.fingerprint, schedulers=commits, image={"path": path.name, "sha256": sha(path)})
        expected = config["historical_pngs"].get(row["id"])
        if expected is not None:
            row["historical_png_exact"] = sha(path) == expected
            if not row["historical_png_exact"]:
                raise ValueError("historical PNG differs")
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def trace(scene, variant):
        key = scene + "." + variant
        active["row"] = key
        n = nav(origins[key])
        ids = config["left_rows"][scene]

        def observe(n):
            cut = n.capture()
            tag = clock(n.point)
            values = {k: (v[:, ids] if k in {"image", "latent"} else v).detach().cpu()
                      for k in ("image", "latent", "text") if (v := cut.payload[k]) is not None}
            if variant == "left":
                reference_features[tag] = values
                metrics = {k: {"rms": float(v.float().square().mean().sqrt())} for k, v in values.items()}
            else:
                metrics = {k: difference(v, reference_features[tag][k]) for k, v in values.items()}
            traces.append({"row": key, "clock": n.point, "cut": cut.fingerprint, "metrics": metrics,
                           "measurement_scope": "Actual native tensors; selected-left image/latent, full text. Full cuts retained only at declared frontiers."})
            if tag in points_to_store:
                store.save(cut)
                frontiers[key + "/" + tag] = cut

        finish(n, {"id": key, "scene": scene, "kind": "native_trace", "parent": origins[key].fingerprint}, observe=observe)

    def transplant(scene, spec):
        key = scene + "." + spec["id"]
        active["row"] = key
        tag = spec["clock"]
        before = frontiers[scene + ".both/" + tag]
        donor = frontiers[scene + (".both/" if spec.get("sham") else ".left/") + tag]
        n, actions = nav(before), []
        expected_latent = None
        forecast = None
        if spec["slots"] == ["image", "text"]:
            step = before.execution_point["logical_step"]
            source_scheduler = frontiers[scene + ".left/" + f"s{step}.scheduler.None"]
            delta_sigma = before.payload["sigmas"][step + 1] - before.payload["sigmas"][step]
            # Preserve the native scalar/tensor promotion: dt*BF16 velocity first,
            # then add to FP32 carried latent and cast back. No future-branch reads.
            expected_latent = (before.payload["latent"].float() + delta_sigma * source_scheduler.payload["noise_prediction"]).to(torch.bfloat16)
            path = output / (key + ".next-latent-forecast.safetensors")
            save_file({"latent": expected_latent.contiguous().cpu()}, str(path))
            counts["scheduler_equation_forecasts"] += 1
            forecast = {"path": path.name, "sha256": sha(path), "source_velocity_cut": source_scheduler.fingerprint,
                "recipient_carried_cut": before.fingerprint, "next_step": step + 1,
                "scope": "Source natural velocity observed; independently evaluate native Euler update before this transplanted branch. Not RGB-from-prompt prediction."}
            save(output / (key + ".forecast.json"), forecast)
            event("numeric_forecast_persisted", row=key, forecast=forecast)
        for slot in spec["slots"]:
            pre = n.capture()
            value = pre.payload[slot].clone()
            if spec["support"] != "full":
                ids = config["left_rows"][scene] if spec["support"] == "left" else config["part_rows"][scene][spec["support"]]
                value[:, ids] = donor.payload[slot][:, ids]
            else:
                value = donor.payload[slot].clone()
            act = Act.replace(slot, value, name=key + "." + slot)
            r = n.apply(act)
            post = n.capture()
            store.save(post)
            if describe(post.payload) != describe({**pre.payload, slot: value}):
                raise ValueError("typed transplant changed undeclared closure")
            actions.append({"slot": slot, "support": spec["support"], "parent": pre.fingerprint, "post": post.fingerprint,
                            "act": act.manifest(), "receipt": r.to_dict()})
        finish(n, {"id": key, "scene": scene, "kind": "transplant", "spec": spec, "parent": before.fingerprint,
            "source": donor.fingerprint, "acts": actions, "forecast": forecast}, expected_latent=expected_latent,
            predicted_next_step=forecast["next_step"] if forecast else None)

    with tx:
        event("predictions_sealed", seal=seal, counts=dict(counts))

        def execute(_):
            for scene in ("festival", "bridge"):
                reference_features.clear()
                if config.get("stage") == "parts":
                    for variant in ("left", "both"):
                        frontiers[scene + "." + variant + "/s3.single.19"] = origins[scene + "." + variant]
                else:
                    for variant in ("left", "both"):
                        trace(scene, variant)
                for spec in config["branches"]:
                    transplant(scene, spec)
            save(output / "frontiers.json", {k: v.fingerprint for k, v in frontiers.items()})
            save(output / "traces.json", traces)
            return {"rows": rows, "counts": counts}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    summary = {"schema": "mdb-flux-anime-depth-v1", "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "counts": counts, "mechanics_status": "valid", "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started,
        "seal": seal, "verification": verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=512 << 20, compression_level=3)


if __name__ == "__main__":
    worker()
