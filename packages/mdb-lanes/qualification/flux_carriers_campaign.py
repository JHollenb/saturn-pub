"""Finite own-context/latent carrier panel from one imported native Z2 cut.

All candidate construction precedes the seal; every suffix and RGB is native.
Two rescue futures share their actual edited Z3 prefix, not a native reference.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

BRANCHES = [
    {"id": "native", "alpha": 1., "beta": 1., "rescue_from": None},
    {"id": "sham", "alpha": 1., "beta": 1., "rescue_from": None},
    {"id": "L1.Chalf", "alpha": 1., "beta": .5, "rescue_from": None},
    {"id": "L1.C0", "alpha": 1., "beta": 0., "rescue_from": None},
    {"id": "L0.C1", "alpha": 0., "beta": 1., "rescue_from": None},
    {"id": "L0.Chalf", "alpha": 0., "beta": .5, "rescue_from": None},
    {"id": "L0.C0", "alpha": 0., "beta": 0., "rescue_from": None},
    {"id": "Lhalf.C1", "alpha": .5, "beta": 1., "rescue_from": None},
    {"id": "Lhalf.C0", "alpha": .5, "beta": 0., "rescue_from": None},
    {"id": "L1.C0.rescue3", "alpha": 1., "beta": 0., "rescue_from": "L1.C0"},
    {"id": "L0.C0.rescue3", "alpha": 0., "beta": 0., "rescue_from": "L0.C0"}]
PREVIEW_ROWS = ["native", "L1.C0", "L0.C1", "L0.C0"]
NATIVE_PIPELINE_SHA256 = "8a57a8a7f1c22fc0c5bc5d3fd1bd9bc76c17c1f4d6644cc73370651dd955358e"


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
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
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if (config.get("branches") != BRANCHES or config.get("preview_rows") != PREVIEW_ROWS or config.get("preview_steps") != [3]
            or config.get("steps") != 4 or config.get("height") != 512 or config.get("width") != 512
            or config.get("seed") != 47 or not isinstance(config.get("prompt"), str) or not config["prompt"]
            or config.get("input_store") != "input-store" or config.get("source_selection") != "source-selection.json"
            or set(config.get("input_cuts", {})) != {"Z0", "Z2"}):
        raise ValueError("carrier campaign differs from fixed own-state panel")
    for name, minimum, maximum in (("max_adapter_steps", 560, 650), ("max_images", 15, 15),
            ("max_final_images", 11, 11), ("max_preview_images", 4, 4), ("max_vae_calls", 16, 16),
            ("max_wall_seconds", 1, 600), ("archive_compression_level", 1, 9)):
        if type(config.get(name)) is not int or not minimum <= config[name] <= maximum:
            raise ValueError("invalid finite carrier budget: " + name)


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged carrier source/request differs")
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

    helper, preview = runpy.run_path("flux_carriers.py"), runpy.run_path("flux_joint_self.py")["decode_live_preview"]
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("carrier suffix requires admitted CUDA")
    output = Path("flux-carriers-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    started, timeline, active = time.monotonic(), [], {"branch": "load"}
    counts = {"model_loads": 0, "prompt_encodings": 0, "noise_draws": 0, "physical_batch": 1,
        "adapter_steps": 0, "acts": 0, "receipt_emissions": 0, "project_reads": 0,
        "rescue_prefix_reuses": 0, "final_images": 0, "preview_images": 0, "images": 0, "vae_calls": 0}

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite carrier wall budget exhausted")

    def event(kind, **fields):
        row = json_value({"sequence": len(timeline), "kind": kind, "seconds": time.monotonic() - started, **fields})
        timeline.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_CARRIERS_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    selection = json.loads(Path(config["source_selection"]).read_text())
    if selection["input_cuts"] != config["input_cuts"] or len(selection["files"]) != 20:
        raise ValueError("selected two-cut CAS inventory differs")
    for path, digest in selection["files"].items():
        if not Path(path).resolve().is_relative_to(Path(config["input_store"]).resolve()) or sha(path) != digest:
            raise ValueError("source CAS path/hash differs")
    prior = json.loads(Path(selection["source_record"]["path"]).read_text())
    prior_audit, prior_row = (json.loads(Path(name).read_text()) for name in ("prior-audit.json", "prior-native-row.json"))
    if (sha(selection["source_record"]["path"]) != selection["source_record"]["sha256"]
            or sha("prior-audit.json") != selection["audit_sha256"]
            or sha("prior-native-row.json") != selection["prior_native_row_sha256"]
            or prior_audit["mechanics_status"] != "valid"
            or prior_audit["source"]["job_id"] != selection["prior_job_id"]
            or prior_audit["source"]["archive_sha256"] != selection["archive_sha256"]
            or prior_audit["source"]["request_sha256"] != selection["request_sha256"]
            or prior["request_sha256"] != selection["request_sha256"]
            or prior["seed"] != config["seed"] or prior["seed"] != 47 or prior["prompt"] != config["prompt"]
            or prior["initial_cut"] != config["input_cuts"]["Z0"]
            or prior_row["id"] != "native"
            or next(item["cut"] for item in prior_row["schedulers"] if item["step"] == 2) != config["input_cuts"]["Z2"]
            or prior_row["image"]["sha256"] != selection["expected_native_png_sha256"]
            or config.get("expected_native_png_sha256") != selection["expected_native_png_sha256"]):
        raise ValueError("prior source/audit/native occurrence lineage differs")
    pipeline_sources = {name: digest for name, digest in prior["implementation_sources"].items() if name.endswith("pipeline_flux2_klein.py")}
    if len(pipeline_sources) != 1 or next(iter(pipeline_sources.values())) != NATIVE_PIPELINE_SHA256:
        raise ValueError("initial native unit-normal preparation implementation differs")
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
        raise ValueError("native source model/execution profile differs")
    if sha(inspect.getfile(type(pipe))) != NATIVE_PIPELINE_SHA256:
        raise ValueError("loaded native pipeline differs from prior initial-noise implementation")
    store, inputs = LocalStore(output / "typed-store"), LocalStore(config["input_store"])
    initial, parent = (inputs.load(config["input_cuts"][key], device="cuda") for key in ("Z0", "Z2"))
    helper["validate_sources"](initial, parent)
    for cut in (initial, parent):
        store.save(cut)
    release = {"native": False, "entries_before_release": 0}

    def guard(module, args):
        if not release["native"]:
            release["entries_before_release"] += 1
            raise ValueError("native suffix entered before carrier hypothesis seal")

    guards = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    array_index = {}

    def array(value):
        wall()
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in array_index:
            path = output / "arrays" / (key + ".npy")
            path.parent.mkdir(exist_ok=True)
            np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)
            array_index[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "descriptor": descriptor,
                "stored_dtype": "float32", "stored_shape": list(value.shape), "native_stride": list(value.stride())}
        return array_index[key]

    payloads, recipes = {}, {}
    for spec in config["branches"]:
        if spec["rescue_from"] is None:
            value = helper["carrier_payloads"](initial, parent, alpha=spec["alpha"], beta=spec["beta"])
            payloads[spec["id"]] = value
            recipes[spec["id"]] = {**value["recipe"], "latent": array(value["latent"]), "conditioning": array(value["conditioning"])}
    base = payloads["native"]
    context_projection = pipe.transformer.context_embedder
    projection_parameters = {"class": type(context_projection).__name__, "weight": None, "bias": None}
    if isinstance(context_projection, torch.nn.Linear):
        projection_parameters.update(weight=tensor_descriptor(context_projection.weight),
            bias=None if context_projection.bias is None else tensor_descriptor(context_projection.bias))
    source = {"schema": "mdb-flux-carriers-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "execution": dict(adapter.execution), "source_selection": selection,
        "prior_implementation_sources": prior["implementation_sources"], "prior_execution": prior["execution"],
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
        "initial_cut": initial.fingerprint, "parent_cut": parent.fingerprint,
        "initial_latent": array(initial.payload["latent"]), "parent_latent": array(parent.payload["latent"]),
        "own_conditioning": array(parent.payload["conditioning"]), "sigmas": parent.payload["sigmas"].tolist(),
        "timesteps": parent.payload["timesteps"].tolist(), "sigma_index": 2, "sigma2": float(parent.payload["sigmas"][2]),
        "anchor": array(base["anchor"]), "residual": array(base["residual"]), "recipes": recipes,
        "branches": config["branches"], "context_embedder_parameters": projection_parameters,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "implementation_sources": {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
            (type(pipe), type(pipe.transformer), type(pipe.vae), type(pipe.scheduler), type(adapter))},
        "preseal_counts": dict(counts), "preseal_native_entries": release["entries_before_release"],
        "initial_noise_provenance": "Prior seed47 native prepare_latents unit-normal draw, source/audit/archive/checkpoint pinned; sigma0=1. No new encoding or noise draw.",
        "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
            "allowed_prefix": "two source-qualified own cuts hydrated; no native project/block/scheduler/VAE or new prompt encoding"},
        execution_scope="CUDA BF16 native own-StateCut suffix; no new encoder/noise; FP32 recipe arithmetic with BF16 payloads; externally managed Navigator execution counted by dedicated public transition/Act ledger, not tx.work.native_calls",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("carrier_hypotheses", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows, row_cuts, parity, controls = [], {}, {}, {}

    def receipt(value):
        store.receipt(value)
        body = value.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "branch": active["branch"],
                "receipt": value.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite native carrier-step budget exhausted")

    def project(n):
        pre = n.capture()
        store.save(pre)
        seen, handles = {}, []

        def capture_input(name):
            def hook(module, args):
                if name in seen:
                    raise ValueError("projection module entered more than once")
                seen[name] = args[0].detach().clone()
            return hook

        def capture_output(name):
            def hook(module, args, value):
                seen[name] = value.detach().clone()
            return hook

        try:
            handles = [context_projection.register_forward_pre_hook(capture_input("conditioning")),
                context_projection.register_forward_hook(capture_output("text")),
                pipe.transformer.x_embedder.register_forward_pre_hook(capture_input("latent")),
                pipe.transformer.x_embedder.register_forward_hook(capture_output("image"))]
            value = n.step()
        finally:
            for handle in handles:
                handle.remove()
        post = n.capture()
        store.save(post)
        if set(seen) != {"conditioning", "latent", "text", "image"} or any(not torch.equal(seen[name], cut.payload[name])
                for name, cut in (("conditioning", pre), ("latent", pre), ("text", post), ("image", post))):
            raise ValueError("actual native projection input/output differs from typed state")
        counts["project_reads"] += 1
        return {"logical_step": pre.payload["step"], "pre_cut": pre.fingerprint, "post_cut": post.fingerprint,
            "receipt": value.to_dict(), **{name: array(tensor) for name, tensor in seen.items()},
            "actual_module_ports_exact_typed_state": True}

    def decode(n, *, live=False):
        wall()
        if counts["vae_calls"] >= config["max_vae_calls"]:
            raise RuntimeError("finite VAE-call budget exhausted")
        counts["vae_calls"] += 1
        return preview(pipe, n.capture(), height=512, width=512) if live else adapter.decode(n.session, height=512, width=512)

    def finish(n, row, expected_c):
        schedulers = [] if row["rescue_from"] is None else [{"step": 3, "cut": row["parent"], "latent": array(n.session.read("latent"))}]
        projects, previews = [], {}
        while n.point["logical_step"] < 4:
            wall()
            point = n.point
            if point["phase"] == "project":
                projects.append(project(n))
            else:
                n.step()
            if point["phase"] == "scheduler":
                cut = n.capture()
                store.save(cut)
                row_cuts.setdefault(row["id"], {})[cut.payload["step"]] = cut
                schedulers.append({"step": cut.payload["step"], "cut": cut.fingerprint, "latent": array(cut.payload["latent"])})
                if row["id"] in config["preview_rows"] and cut.payload["step"] == 3:
                    image = decode(n, live=True)
                    if n.capture().fingerprint != cut.fingerprint:
                        raise ValueError("live VAE preview changed carrier state")
                    path = output / (row["id"] + ".preview3.png")
                    image.save(path)
                    counts["preview_images"] += 1
                    counts["images"] += 1
                    previews["3"] = {"path": path.name, "sha256": sha(path), "cut": cut.fingerprint,
                        "point": dict(cut.execution_point), "scope": "native VAE on live noised Z3, no denoising or feedback"}
        final = n.capture()
        store.save(final)
        if describe(final.payload["conditioning"]) != describe(expected_c):
            raise ValueError("future conditioning policy changed")
        image = decode(n)
        path = output / (row["id"] + ".png")
        image.save(path)
        counts["final_images"] += 1
        counts["images"] += 1
        if counts["images"] > config["max_images"] or counts["preview_images"] > config["max_preview_images"] or counts["final_images"] > config["max_final_images"]:
            raise RuntimeError("finite carrier image inventory exceeded")
        if row["id"] == "native":
            other = decode(n, live=True)
            buffer = io.BytesIO()
            other.save(buffer, format="PNG")
            pixel_exact = np.array_equal(np.asarray(image), np.asarray(other))
            png_exact = buffer.getvalue() == path.read_bytes()
            if n.capture().fingerprint != final.fingerprint:
                raise ValueError("live preview helper changed completed carrier state")
            parity.update(cut=final.fingerprint, point=dict(final.execution_point), pixel_exact=bool(pixel_exact), png_exact=png_exact,
                adapter_png_sha256=sha(path), helper_png_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                adapter_pixel_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                helper_pixel_sha256=hashlib.sha256(np.asarray(other).tobytes()).hexdigest(), extra_vae_calls=1, extra_saved_images=0)
            controls["preview_helper_final_pixel_exact"] = bool(pixel_exact)
            controls["preview_helper_final_png_exact"] = png_exact
            controls["native_png_equals_prior"] = sha(path) == selection["expected_native_png_sha256"]
        row.update(final_cut=final.fingerprint, final_conditioning=tensor_descriptor(final.payload["conditioning"]),
            expected_conditioning=tensor_descriptor(expected_c), image={"path": path.name, "sha256": sha(path)},
            schedulers=schedulers, project_reads=projects, previews=previews)
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    with tx:
        event("hypotheses_sealed", seal=seal, counts=dict(counts), native_entries_before_release=release["entries_before_release"])
        release["native"] = True

        def execute(_):
            for spec in config["branches"]:
                active["branch"] = spec["id"]
                rescue = spec["rescue_from"]
                cut = parent if rescue is None else row_cuts[rescue][3]
                n = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)
                row = {**spec, "parent": cut.fingerprint, "acts": [],
                    "continuation_policy": "stored-C-dose-until-Z3-then-own-C" if rescue else "stored-C-dose-through-final"}
                event("branch_start", row=spec["id"], parent=cut.fingerprint, rescue_from=rescue)
                if rescue:
                    counts["rescue_prefix_reuses"] += 1
                    expected_c = parent.payload["conditioning"]
                    row["shared_prefix"] = {"row_id": rescue, "cut": cut.fingerprint, "logical_step": 3, "adapter_steps_reused": 28}
                    row["recipe"] = {"conditioning": array(expected_c), "source": "own original stored C restored at Z3"}
                    row["acts"].append(helper["install_carrier_port"](n, "conditioning", expected_c,
                        name=spec["id"] + ".own-C-rescue", store=store, expected_step=3))
                else:
                    value = payloads[spec["id"]]
                    expected_c = value["conditioning"]
                    row["recipe"] = recipes[spec["id"]]
                    if spec["id"] != "native":
                        for slot in ("latent", "conditioning"):
                            row["acts"].append(helper["install_carrier_port"](n, slot, value[slot],
                                name=spec["id"] + "." + slot, store=store, expected_step=2))
                finish(n, row, expected_c)
            expected = {"adapter_steps": 560, "acts": 18, "project_reads": 20, "final_images": 11,
                "preview_images": 4, "images": 15, "vae_calls": 16, "rescue_prefix_reuses": 2,
                "prompt_encodings": 0, "noise_draws": 0}
            if any(counts[key] != value for key, value in expected.items()):
                raise ValueError("completed shared-prefix physical inventory differs")
            by_id = {row["id"]: row for row in rows}
            controls["sham_png_equals_native"] = by_id["sham"]["image"]["sha256"] == by_id["native"]["image"]["sha256"]
            controls["native_expected_prior_png_sha256"] = selection["expected_native_png_sha256"]
            return {"rows": rows, "counts": dict(counts), "preview_helper_final_parity": parity,
                "control_observations": controls, "source_sha256": sha(output / "source.json"), "RGB_forecast": False}

        tx.execute("native", execute)
    for handle in guards:
        handle.remove()
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    save(output / "preview-helper-final-parity.json", parity)
    summary = {"schema": "mdb-flux-carriers-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "model_identity": adapter.model_identity, "counts": counts,
        "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "seal": seal, "transaction_verification": verification,
        "worker_seconds": time.monotonic() - started, "control_observations": controls, "RGB_forecast": False, "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
        compression_level=config["archive_compression_level"])


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-carriers-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-carriers-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
            runpy.run_path("campaign_support.py")["emit"](output, failure, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
                compression_level=config.get("archive_compression_level", 9))
        raise
