"""Native early text-K/V and late image/text routes from paired own Z3 cuts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

BRANCHES = [
    {"id": "zero", "kind": "reference", "image_source": "zero", "text_source": "zero"},
    {"id": "own", "kind": "reference", "image_source": "own", "text_source": "own"},
    {"id": "sham", "kind": "late", "image_source": "zero", "text_source": "zero"},
    {"id": "late.image", "kind": "late", "image_source": "own", "text_source": None},
    {"id": "late.text", "kind": "late", "image_source": None, "text_source": "own"},
    {"id": "late.both", "kind": "late", "image_source": "own", "text_source": "own"},
    {"id": "early.kv.image", "kind": "early", "image_source": "kv", "text_source": None},
    {"id": "early.kv.both", "kind": "early", "image_source": "kv", "text_source": "kv"},
    {"id": "early.coherent", "kind": "early", "image_source": "own", "text_source": "own"}]


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(child) for child in value]
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
    if (config.get("branches") != BRANCHES or set(config.get("input_cuts", {})) != {"zero", "own"}
            or config.get("input_store") != "input-store" or config.get("source_selection") != "source-selection.json"
            or config.get("steps") != 4 or config.get("height") != 512 or config.get("width") != 512 or config.get("seed") != 47):
        raise ValueError("rescue routes differ from the declared paired own-Z3 panel")
    for name, minimum, maximum in (("max_adapter_steps", 220, 260), ("max_acts", 11, 11),
            ("max_candidate_calls", 3, 3), ("max_images", 9, 9), ("max_vae_calls", 9, 9),
            ("max_wall_seconds", 1, 600), ("archive_compression_level", 1, 9)):
        if type(config.get(name)) is not int or not minimum <= config[name] <= maximum:
            raise ValueError("invalid finite rescue-route budget: " + name)


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged route source/request binding differs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    helper, joint = runpy.run_path("flux_rescue_routes.py"), runpy.run_path("flux_joint_self.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("rescue routes require admitted CUDA")
    output = Path("flux-rescue-routes-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    counts = {"model_loads": 0, "prompt_encodings": 0, "noise_draws": 0, "physical_batch": 1,
        "adapter_steps": 0, "acts": 0, "candidate_calls": 0, "receipt_emissions": 0, "images": 0, "vae_calls": 0}
    started, timeline, active = time.monotonic(), [], {"branch": "load"}

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite rescue-route wall budget exhausted")

    def event(kind, **fields):
        value = json_value({"sequence": len(timeline), "kind": kind, "seconds": time.monotonic() - started, **fields})
        timeline.append(value)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_RESCUE_ROUTES_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    selection = json.loads(Path(config["source_selection"]).read_text())
    if (selection["input_cuts"] != config["input_cuts"] or selection["prior_job_id"] != "job-a1afc382b4fe"
            or selection["expected_pngs"] != config["expected_pngs"]):
        raise ValueError("paired prior occurrence selection differs")
    for path, digest in selection["files"].items():
        if not Path(path).resolve().is_relative_to(Path(config["input_store"]).resolve()) or sha(path) != digest:
            raise ValueError("paired source CAS bytes/path differ")
    prior = json.loads(Path(selection["source_record"]["path"]).read_text())
    prior_audit = json.loads(Path("prior-audit.json").read_text())
    if (sha(selection["source_record"]["path"]) != selection["source_record"]["sha256"]
            or sha("prior-audit.json") != selection["audit_sha256"] or prior_audit["mechanics_status"] != "valid"
            or prior["request_sha256"] != selection["request_sha256"]
            or prior_audit["source"]["job_id"] != selection["prior_job_id"]
            or prior_audit["source"]["archive_sha256"] != selection["archive_sha256"]
            or prior_audit["source"]["request_sha256"] != selection["request_sha256"]):
        raise ValueError("paired predecessor source/audit custody differs")
    prior_rows = {}
    for name, row_id in (("zero", "L1.C0"), ("own", "L1.C0.rescue3")):
        record = selection["prior_rows"][name]
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("prior row bytes differ")
        row = json.loads(Path(record["path"]).read_text())
        if (row["id"] != row_id or row["image"]["sha256"] != selection["expected_pngs"][name]
                or next(item["post_cut"] for item in row["project_reads"] if item["logical_step"] == 3) != config["input_cuts"][name]):
            raise ValueError("paired post-project source occurrence differs")
        prior_rows[name] = row
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or prior["checkpoint_sha256"] != identity(inventory):
        raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True,
        text_encoder=None, tokenizer=None)
    counts["model_loads"] += 1
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires native deterministic distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False).to("cuda")
    pipe.vae.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    if adapter.model_identity != prior["model_identity"] or dict(adapter.execution) != prior["execution"]:
        raise ValueError("paired native model/execution profile differs")
    inputs, store = LocalStore(config["input_store"]), LocalStore(output / "typed-store")
    cuts = {name: inputs.load(identifier, device="cuda") for name, identifier in config["input_cuts"].items()}
    helper["validate_paired_cuts"](cuts["zero"], cuts["own"])
    for cut in cuts.values():
        store.save(cut)
    release = {"native": False, "entries_before_release": 0}

    def guard(module, args):
        if not release["native"]:
            release["entries_before_release"] += 1
            raise ValueError("native route entered before hypothesis seal")

    guards = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    array_index = {}

    def array(value):
        wall()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite observed source/candidate/route array")
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in array_index:
            path = output / "arrays" / (key + ".npy")
            path.parent.mkdir(exist_ok=True)
            np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)
            array_index[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "descriptor": descriptor,
                "stored_dtype": "float32", "stored_shape": list(value.shape), "native_stride": list(value.stride())}
        return array_index[key]

    block = pipe.transformer.transformer_blocks[0]
    source = {"schema": "mdb-flux-rescue-routes-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "execution": dict(adapter.execution), "source_selection": selection,
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"], "input_cuts": config["input_cuts"],
        "input_arrays": {name: {slot: array(cut.payload[slot]) for slot in ("latent", "conditioning", "image", "text")} for name, cut in cuts.items()},
        "source_parameters": {name: tensor_descriptor(value) for name, value in block.named_parameters()},
        "prior_execution": prior["execution"], "prior_implementation_sources": prior["implementation_sources"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "implementation_sources": {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
            (type(pipe), type(pipe.transformer), type(block), type(block.attn.processor), type(pipe.vae), type(pipe.scheduler), type(adapter))},
        "preseal_counts": dict(counts), "preseal_native_entries": release["entries_before_release"], "branches": config["branches"],
        "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
            "allowed_prefix": "paired own-Z3 post-project cuts hydrated; no new native project, block, scheduler, attention dispatch or VAE before seal"},
        execution_scope="CUDA BF16 paired own-Z3 native suffix; no new encoder/noise; external Navigator ledger counts native transitions and direct candidate calls",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("rescue_route_hypotheses", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows, prefix_values, prefixes, candidates, controls = [], {}, {}, {}, {}

    def receipt(value):
        store.receipt(value)
        body = value.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "branch": active["branch"],
                "receipt": value.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"] or counts["acts"] > config["max_acts"]:
            raise RuntimeError("finite native route-step/Act budget exhausted")

    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def calculate(name, cut, kv=None):
        wall()
        if counts["candidate_calls"] >= config["max_candidate_calls"]:
            raise RuntimeError("finite direct candidate-call budget exhausted")
        counts["candidate_calls"] += 1
        event("candidate_start", candidate=name, input_parent=cut.fingerprint)
        value = helper["capture_candidate"](block, cut, joint_candidate=joint["joint_candidate_self"], text_kv_source=kv)
        candidates[name] = {key: value[key] for key in ("input_parent", "donor_parent", "text_rows", "dispatch_calls", "partition_controls")}
        candidates[name].update(ports={key: array(tensor) for key, tensor in value["ports"].items()},
            observed_qkv={key: array(tensor) for key, tensor in value["observed_qkv"].items()},
            used_qkv={key: array(tensor) for key, tensor in value["used_qkv"].items()},
            attention_dispatch_output=array(value["attention_dispatch_output"]), equation_exact=value["equation_exact"])
        event("candidate_complete", candidate=name, counts=dict(counts))
        return value

    def finish(n, row):
        continuation = []
        while n.point["logical_step"] < 4:
            wall()
            continuation.append(n.step().to_dict())
        final = n.capture()
        store.save(final)
        expected_c = cuts["own" if row["id"] == "own" else "zero"].payload["conditioning"]
        if describe(final.payload["conditioning"]) != describe(expected_c):
            raise ValueError("route branch changed undeclared stored conditioning")
        if counts["images"] >= config["max_images"] or counts["vae_calls"] >= config["max_vae_calls"]:
            raise RuntimeError("finite route image/VAE budget exhausted")
        counts["vae_calls"] += 1
        image = adapter.decode(n.session, height=512, width=512)
        path = output / (row["id"] + ".png")
        image.save(path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint, final_latent=array(final.payload["latent"]),
            final_conditioning=tensor_descriptor(final.payload["conditioning"]), expected_conditioning=tensor_descriptor(expected_c),
            continuation_receipts=continuation, image={"path": path.name, "sha256": sha(path)})
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    with tx:
        event("hypotheses_sealed", seal=seal, counts=dict(counts), native_entries_before_release=release["entries_before_release"])
        release["native"] = True

        def execute(_):
            native_candidates = {name: calculate(name, cuts[name]) for name in ("zero", "own")}
            own = native_candidates["own"]
            kv = calculate("kv", cuts["zero"], {"parent": cuts["own"].fingerprint, "text_rows": own["text_rows"], **own["observed_qkv"]})
            for name in ("zero", "own"):
                active["branch"] = "prefix." + name
                n, emitted = navigator(cuts[name]), []
                first = n.step()
                emitted.append(first.to_dict())
                post = n.capture()
                store.save(post)
                prefix_values[name] = {"joint0_post": post}
                while n.point["phase"] != "single" or n.point["index"] != 0:
                    emitted.append(n.step().to_dict())
                single = n.capture()
                store.save(single)
                prefix_values[name]["single0"] = single
                prefixes[name] = {"input_cut": cuts[name].fingerprint, "joint0_post": post.fingerprint,
                    "single0": single.fingerprint, "prefix_receipts": emitted,
                    "observed_text_statistics": {label: {"l2_norm": float(torch.linalg.vector_norm(frontier.payload["text"].float())),
                        "zero_fraction": float((frontier.payload["text"] == 0).float().mean()),
                        "scope": "Observed working T; zero label names source C history, not a zero-valued tensor."}
                        for label, frontier in (("joint0_post", post), ("single0", single))}}
                controls[name + "_candidate_image_exact_native"] = torch.equal(native_candidates[name]["image"], post.payload["image"])
                controls[name + "_candidate_text_exact_native"] = torch.equal(native_candidates[name]["text"], post.payload["text"])
            controls["kv_candidate_image_exact_own_candidate"] = torch.equal(kv["image"], own["image"])
            controls["kv_candidate_image_exact_own_native"] = torch.equal(kv["image"], prefix_values["own"]["joint0_post"].payload["image"])
            save(output / "prefixes.json", {"schema": "mdb-flux-rescue-prefixes-v1", "prefixes": prefixes})
            save(output / "candidates.json", {"schema": "mdb-flux-rescue-candidates-v1", "candidates": candidates})
            for spec in config["branches"]:
                active["branch"] = spec["id"]
                early = spec["kind"] == "early"
                reference = "own" if spec["id"] == "own" else "zero"
                cut = prefix_values[reference]["joint0_post" if early else "single0"]
                n = navigator(cut)
                row = {**spec, "parent": cut.fingerprint, "start_point": n.point, "prefix_source": reference, "acts": [],
                    "continuation_policy": "own-stored-C" if reference == "own" else "zero-stored-C-no-more-projections"}
                event("branch_start", row=spec["id"], parent=cut.fingerprint)
                if spec["kind"] != "reference":
                    for slot, key in (("image", "image_source"), ("text", "text_source")):
                        selected = spec[key]
                        if selected is None:
                            continue
                        value = kv[slot] if selected == "kv" else prefix_values[selected]["joint0_post" if early else "single0"].payload[slot]
                        row["acts"].append(helper["install_route_port"](n, slot, value, name=spec["id"] + "." + slot,
                            store=store, joint_install=joint["install_joint_port"]))
                finish(n, row)
            expected = {"adapter_steps": 220, "acts": 11, "candidate_calls": 3, "images": 9, "vae_calls": 9,
                "prompt_encodings": 0, "noise_draws": 0}
            if any(counts[key] != value for key, value in expected.items()):
                raise ValueError("completed physical route census differs")
            by_id = {row["id"]: row for row in rows}
            controls.update(zero_png_equals_prior=by_id["zero"]["image"]["sha256"] == selection["expected_pngs"]["zero"],
                own_png_equals_prior=by_id["own"]["image"]["sha256"] == selection["expected_pngs"]["own"],
                sham_png_equals_zero=by_id["sham"]["image"]["sha256"] == by_id["zero"]["image"]["sha256"],
                late_both_png_equals_own=by_id["late.both"]["image"]["sha256"] == by_id["own"]["image"]["sha256"],
                early_coherent_png_equals_own=by_id["early.coherent"]["image"]["sha256"] == by_id["own"]["image"]["sha256"])
            return {"rows": rows, "counts": dict(counts), "prefixes": prefixes, "candidates": candidates,
                "control_observations": controls, "source_sha256": sha(output / "source.json"), "RGB_forecast": False}

        tx.execute("native", execute)
    for handle in guards:
        handle.remove()
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-rescue-routes-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "model_identity": adapter.model_identity, "counts": counts,
        "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "seal": seal,
        "transaction_verification": verification, "worker_seconds": time.monotonic() - started,
        "control_observations": controls, "RGB_forecast": False, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    event("archive_start", raw_output_bytes_before_event=sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        scope="all retained output files before this archive-start event and final files manifest; no model weights serialized")
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
        compression_level=config["archive_compression_level"])


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-rescue-routes-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-rescue-routes-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
            runpy.run_path("campaign_support.py")["emit"](output, failure, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
                compression_level=config.get("archive_compression_level", 9))
        raise
