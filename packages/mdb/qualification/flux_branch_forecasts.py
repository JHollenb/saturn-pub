"""Sealed numerical forecasts of four common-parent diffusion image futures.

Stored prompt carriers are observations. A fresh latent and one base projection
create the parent. Three text-only Acts define edited initial states. Every
functional trajectory and RGB image is sealed before any native suffix runs.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import runpy
import time
from collections.abc import Mapping
from pathlib import Path


def boundary_arrays(info, state):
    tag = f"step{info['step']}.{info['phase']}.{info['index']}"
    kind = info["phase"]
    arrays = ({"image": state["image"], "text": state["text"]} if kind in {"project", "joint", "single"}
              else {"noise_prediction": state["noise_prediction"]} if kind == "readout" else {"latent": state["latent"]})
    if kind == "project":
        arrays.update({"context." + key: state["context"][key] for key in ("temb", "img_mod", "txt_mod", "single_mod")})
    return tag, arrays


def pack(value):
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu()
        array = array.float() if array.dtype == torch.bfloat16 else array
        return {"tensor_dtype": str(value.dtype), "array": array.numpy().copy()}
    if isinstance(value, Mapping):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(pack(item) for item in value)
    return value


def unpack(value, *, device):
    import numpy as np
    import torch

    if isinstance(value, Mapping):
        if set(value) == {"tensor_dtype", "array"}:
            return torch.from_numpy(np.array(value["array"], copy=True)).to(device=device, dtype=getattr(torch, value["tensor_dtype"].split(".")[-1]))
        return {key: unpack(item, device=device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(unpack(item, device=device) for item in value)
    return value


class BoundaryCache:
    """Immutable full predictions in RAM; declared witnesses/endpoints on disk."""

    def __init__(self, *, witness_values, max_bytes):
        self.values = {}
        self.bytes = 0
        self.witness_values, self.max_bytes = witness_values, max_bytes

    def snapshot(self, key, value, *, retain_full=False):
        import hashlib

        import numpy as np
        import torch

        from mdb.forward_recording import tensor_descriptor

        if key in self.values:
            raise ValueError("duplicate forecast cache key")
        cpu = value.detach().cpu().contiguous()
        expanded = cpu.float().numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()
        if self.bytes + expanded.nbytes > self.max_bytes:
            raise RuntimeError("declared CPU forecast cache budget exceeded")
        frozen = np.frombuffer(expanded.tobytes(), dtype=expanded.dtype).reshape(expanded.shape)
        self.values[key] = frozen
        self.bytes += frozen.nbytes
        positions = np.unique(np.linspace(0, frozen.size - 1, min(self.witness_values, frozen.size), dtype=np.int64))
        row = {"descriptor": tensor_descriptor(cpu), "float_array_sha256": hashlib.sha256(frozen.tobytes()).hexdigest(),
               "witness_indices": positions, "witness_values": frozen.ravel()[positions], "full_retained": retain_full}
        if retain_full:
            row["full_values"] = frozen
        return row

    def inspect(self, key, actual, sealed, *, compare):
        import hashlib

        from mdb.forward_recording import tensor_descriptor

        frozen = self.values[key]
        if hashlib.sha256(frozen.tobytes()).hexdigest() != sealed["float_array_sha256"]:
            raise ValueError("sealed full forecast RAM bytes changed")
        expanded = actual.detach().float().cpu().numpy()
        row = {"descriptor": tensor_descriptor(actual), "full_error": compare(expanded, frozen),
               "witness_indices": sealed["witness_indices"], "witness_values": expanded.ravel()[sealed["witness_indices"]]}
        if sealed["full_retained"]:
            row["full_values"] = expanded.copy()
        return row


def manifest_fields(manifest):
    return {row["key"]["value"]: row["value"] for row in manifest["payload"]["items"]}


def load_carrier(record_path, blob_path, *, device):
    """Hydrate one authenticated tensor port, without claiming a replayable cut."""
    import hashlib

    from safetensors.torch import load_file
    from saturn_pub.values import describe, digest

    record = json.loads(Path(record_path).read_text())
    cut, page = record["cut"], record["page"]
    if digest(cut["manifest"]) != cut["fingerprint"] or digest(page) != cut["slots"]["text"]:
        raise ValueError("source carrier manifest/page identity differs")
    if hashlib.sha256(Path(blob_path).read_bytes()).hexdigest() != page["tensor_blob"]:
        raise ValueError("source carrier blob differs")
    encoded = page["encoded"]
    if encoded["kind"] != "tensor":
        raise ValueError("carrier must be one tensor")
    tensors = load_file(str(blob_path), device=device)
    if set(tensors) != {encoded["name"]}:
        raise ValueError("carrier tensor inventory differs")
    value = tensors[encoded["name"]]
    if describe(value) != page["value"] or page["value"] != manifest_fields(cut["manifest"])["text"]:
        raise ValueError("carrier tensor/source descriptor differs")
    return {"text": value, "manifest": cut["manifest"]}


def text_branch_parents(adapter, parent, supplied, branch_ids, store):
    """Actual typed text replacements, preserving every other parent field."""
    from saturn_pub import Act, Session
    from saturn_pub.values import describe

    parents, acts = {"base": parent}, {}
    for name in branch_ids[1:]:
        manifest = supplied[name]["manifest"]
        if manifest["model_identity"] != parent.model_identity or manifest["execution_point"] != parent.manifest()["execution_point"]:
            raise ValueError("source carrier execution contract differs")
        for slot in ("img_ids", "txt_ids", "timesteps", "sigmas", "context"):
            if manifest_fields(manifest)[slot] != describe(parent.payload[slot]):
                raise ValueError("source carrier context differs: " + slot)
        branch = Session.from_cut(adapter, parent)
        act = Act.replace("text", supplied[name]["text"], name="initial-text-" + name)
        receipt = branch.apply(act)
        store.receipt(receipt)
        edited = branch.capture()
        expected = dict(parent.payload)
        expected["text"] = supplied[name]["text"]
        if describe(edited.payload) != describe(expected):
            raise ValueError("text Act changed undeclared closure")
        store.save(edited)
        parents[name], acts[name] = edited, {"act": act.manifest(), "receipt": receipt.to_dict()}
    return parents, acts


def worker():
    support = runpy.run_path("flux_trajectory_forecast_campaign.py")
    save, file_sha, identity, compare = (support[key] for key in ("save", "file_sha", "identity", "compare"))
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(file_sha(name) != expected for name, expected in request["source_hashes"].items()):
        raise ValueError("staged request/source differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    functions = runpy.run_path("flux_functional_denoiser.py")
    vae_functions = runpy.run_path("flux_functional_vae.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    output = Path("flux-branch-forecast-result")
    output.mkdir(exist_ok=False)
    started, sequence, phase = time.monotonic(), 0, {"name": "prefix"}
    counts = {"prefix_module_entries": 0, "prediction_module_entries": 0, "native_module_entries": 0,
              "selected_block_entries": 0, "vae_decoder_entries": 0, "predicted_transitions": 0,
              "native_transitions": 0, "prompt_encodings": 0, "prefix_projections": 0, "acts": 0, "images": 0}

    def event(kind, **details):
        nonlocal sequence
        row = support["json_value"]({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, **details})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_BRANCH_FORECAST_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(files) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs")
    save(output / "plan.json", config)
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
    base_observed = inputs.load(config["input_cuts"]["base"], device="cuda")
    if base_observed.model_identity != adapter.model_identity:
        raise ValueError("supplied base contract differs")
    store.save(base_observed)
    supplied = {"base": {"text": base_observed.payload["text"], "manifest": base_observed.manifest()}}
    for name in config["branch_ids"][1:]:
        supplied[name] = load_carrier("carriers/" + name + ".json", "carriers/" + name + ".safetensors", device="cuda")
    save(output / "carrier-sources.json", {name: value["manifest"] for name, value in supplied.items()})
    selected_ids = {id(m) for m in (*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks)}

    def guard(module, args):
        wall()
        if phase["name"] == "prediction":
            counts["prediction_module_entries"] += 1
            raise RuntimeError("native module entry forbidden during functional forecasts")
        counts["prefix_module_entries" if phase["name"] == "prefix" else "native_module_entries"] += 1
        if id(module) in selected_ids:
            counts["selected_block_entries"] += 1
        if module is pipe.vae.decoder:
            counts["vae_decoder_entries"] += 1

    hook = torch.nn.modules.module.register_module_forward_pre_hook(guard)
    tx = None
    try:
        old = base_observed.payload
        with torch.inference_mode():
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                old["conditioning"].dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
        base_session = adapter.session(latent=latent, conditioning=old["conditioning"], img_ids=img_ids,
            txt_ids=old["txt_ids"], timesteps=old["timesteps"].clone(), sigmas=old["sigmas"].clone())
        receipt = base_session.continue_(1)
        store.receipt(receipt)
        counts["prefix_projections"] += 1
        parent = base_session.capture()
        store.save(parent)
        if torch.equal(latent, old["latent"]) or parent.payload["phase"] != "joint" or parent.payload["cursor"] != 0:
            raise ValueError("fresh latent or pre-joint0 frontier differs")
        parents, acts = text_branch_parents(adapter, parent, supplied, config["branch_ids"], store)
        counts["acts"] = len(acts)
        if counts["selected_block_entries"] or counts["vae_decoder_entries"]:
            raise ValueError("native suffix entered during parent construction")
        parent_descriptors = {name: describe(cut.payload) for name, cut in parents.items()}
        parameters = {prefix + "." + name: value for prefix, module in (("transformer", pipe.transformer), ("vae", pipe.vae))
                      for name, value in (*module.named_parameters(), *module.named_buffers())}
        signatures = {name: (id(value), value.data_ptr(), value._version) for name, value in parameters.items()}

        def weight_check(stage, expected=None):
            wall()
            current = {prefix + "." + name: value for prefix, module in (("transformer", pipe.transformer), ("vae", pipe.vae))
                       for name, value in (*module.named_parameters(), *module.named_buffers())}
            if set(current) != set(parameters) or any((id(v), v.data_ptr(), v._version) != signatures[n] for n, v in current.items()):
                raise ValueError("registered weight/buffer identities changed")
            values = {name: tensor_descriptor(value) for name, value in current.items()}
            if expected is not None and values != expected:
                raise ValueError("weight/buffer bytes changed")
            save(output / ("weights-" + stage + ".json"), values)
            return values

        phase["name"] = "prediction"
        weights = weight_check("before-prediction")
        source = {"request": request, "checkpoint_files": files, "parents": {n: c.fingerprint for n, c in parents.items()},
            "base_parent": parent.fingerprint, "acts": acts, "execution": dict(adapter.execution),
            "model_identity": {"checkpoint_sha256": identity(files), "adapter": adapter.model_identity},
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
            "implementation_sources": {inspect.getfile(k): file_sha(inspect.getfile(k)) for k in (type(pipe), type(adapter), type(pipe.transformer), type(pipe.vae))},
            "observed_scope": config["observed"], "arithmetic": config["arithmetic"],
            "image_processor_config": dict(pipe.image_processor.config), "RAM_only_intermediates": True}
        save(output / "source.json", source)
        tx = ForecastTransaction(output / "transaction", model_identity=source["model_identity"], input_identity=source["parents"],
            source_identity=request["source_hashes"], frontier={"kind": "architectural", "boundary": config["observed"]},
            budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000,
                     "max_bytes": 512 << 20, "max_files": 8192})
        cache = BoundaryCache(witness_values=config["witness_values"], max_bytes=config["max_cache_bytes"])
        event("frontiers_captured", counts=dict(counts), parents=source["parents"])
        with tx:
            for name, cut in parents.items():
                tx.frontier(name, {"state": pack(cut.payload), "manifest": cut.manifest(), "weights_digest": identity(weights)})

            def forecast(name):
                def calculate(context):
                    state, rows = unpack(context.frontiers[name]["state"], device="cuda"), []

                    def on_boundary(info, state):
                        wall()
                        tag, arrays = boundary_arrays(info, state)
                        rows.append({"tag": tag, "event": info, "arrays": {key: cache.snapshot(name + "/" + tag + "/" + key, value,
                            retain_full=info["phase"] in {"readout", "scheduler"}) for key, value in arrays.items()}})
                        counts["predicted_transitions"] += 1
                        if counts["predicted_transitions"] > config["max_total_transitions"]:
                            raise RuntimeError("forecast budget exceeded")

                    with torch.inference_mode():
                        final = functions["functional_trajectory"](pipe.transformer, state, on_boundary=on_boundary, max_transitions=111)
                        decoded = vae_functions["decode_functional"](pipe, final["latent"], final["img_ids"], 512, 512)
                    vae = {key: cache.snapshot(name + "/vae/" + key, decoded[key], retain_full=key == "decoded") for key in ("decoder_input", "decoded")}
                    pixels = decoded["pixels"][0]
                    Image.fromarray(pixels).save(output / (name + ".forecast.png"))
                    event("forecast_complete", branch=name, transitions=len(rows), counts=dict(counts), cache_bytes=cache.bytes)
                    return {"predictions": {"boundaries": rows, "vae": vae, "pixels": pixels,
                        "forecast_png_sha256": file_sha(output / (name + ".forecast.png")), "own_state": True, "native_refreshes": 0}}
                return calculate

            for name in config["branch_ids"]:
                tx.predict(name, forecast(name))
            if counts["predicted_transitions"] != config["max_total_transitions"] or counts["prediction_module_entries"]:
                raise ValueError("forecast work/exclusion differs")
            if {name: describe(cut.payload) for name, cut in parents.items()} != parent_descriptors:
                raise ValueError("native parents mutated during prediction")
            weight_check("at-release", weights)
            seal = tx.seal()
            event("all_forecasts_sealed", seal=seal["seal_sha256"], counts=dict(counts))
            tx.release()
            phase["name"] = "native"

            def execute(context):
                results = {}
                for name in config["branch_ids"]:
                    event("native_branch_start", branch=name)
                    prediction, rows = context.predictions[name], []
                    nav = NativeCutNavigator(Session.from_cut(adapter, parents[name]), max_steps=111, on_receipt=store.receipt)
                    for forecast_row in prediction["boundaries"]:
                        wall()
                        point = nav.point
                        info = {"step": point["logical_step"], "phase": point["phase"], "index": point["local_clock"]}
                        receipt = nav.step()
                        tag, arrays = boundary_arrays(info, nav.session.capture(retain=False).payload)
                        if tag != forecast_row["tag"]:
                            raise ValueError("native/forecast boundary order differs")
                        rows.append({"tag": tag, "receipt": receipt.to_dict(), "arrays": {key: cache.inspect(name + "/" + tag + "/" + key,
                            value, forecast_row["arrays"][key], compare=compare) for key, value in arrays.items()}})
                        counts["native_transitions"] += 1
                    observed = {}

                    def before_decode(module, args):
                        observed["decoder_input"] = args[0].detach().clone()

                    def after_decode(module, args, result):
                        observed["decoded"] = result.detach().clone()

                    hooks = [pipe.vae.decoder.register_forward_pre_hook(before_decode), pipe.vae.decoder.register_forward_hook(after_decode)]
                    try:
                        image = adapter.decode(nav.session, height=512, width=512)
                    finally:
                        for item in hooks:
                            item.remove()
                    image.save(output / (name + ".native.png"))
                    pixels = np.asarray(image, dtype=np.uint8)
                    results[name] = {"boundaries": rows, "vae": {key: cache.inspect(name + "/vae/" + key, value,
                        prediction["vae"][key], compare=compare) for key, value in observed.items()}, "pixels": pixels,
                        "pixel_error": compare(pixels, prediction["pixels"]), "native_png_sha256": file_sha(output / (name + ".native.png"))}
                    counts["images"] += 2
                    event("native_branch_complete", branch=name, pixel_error=results[name]["pixel_error"], counts=dict(counts))
                return {"branches": results, "counts": dict(counts), "future_inputs": "separate native own-state continuations; no forecast payload installed"}

            tx.execute("native", execute)
        weight_check("after-native", weights)
        observation = tx.observation("native")
        summary = {"schema": "mdb-flux-branch-forecast-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
            "request_sha256": request["request_sha256"], "counts": counts, "mechanics_status": "valid", "terminal_status": "unassessed",
            "pixel_errors": {name: value["pixel_error"] for name, value in observation["branches"].items()},
            "seal": seal, "transaction_verification": verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"]),
            "worker_seconds": time.monotonic() - started, "RGB_forecast": True, "own_state": True, "native_refreshes": 0,
            "scope": config["scope"], "arithmetic": config["arithmetic"], "limits": config["limits"]}
        save(output / "transaction-report.json", tx.report())
        save(output / "summary.json", summary)
        event("complete", summary=summary)
        save(output / "files.json", {p.relative_to(output).as_posix(): file_sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
        runpy.run_path("campaign_support.py")["emit"](output, support["json_value"](summary), max_compressed_bytes=config["max_archive_bytes"])
    finally:
        hook.remove()
        if tx is not None:
            tx.close()


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        support = runpy.run_path("flux_trajectory_forecast_campaign.py")
        output = Path("flux-branch-forecast-result")
        if output.is_dir():
            report = {"schema": "mdb-flux-branch-forecast-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())["request_sha256"],
                "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            support["save"](output / "failure.json", report)
            runpy.run_path("campaign_support.py")["emit"](output, report, max_compressed_bytes=256 << 20)
        raise
