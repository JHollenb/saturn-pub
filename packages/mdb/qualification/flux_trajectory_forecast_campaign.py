"""Forecast an own-state Klein continuation and RGB before native comparison.

The functional graph shares Torch CUDA primitives with the reference. It never
calls native model forwards, adapter transitions, scheduler or decoder methods.
One observed initial frontier is supplied; later states are predicted only.
"""
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


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def json_value(value):
    """Copy public read-only observation containers into JSON containers."""
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def compare(actual, predicted):
    import numpy as np

    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("incompatible or nonfinite comparison arrays")
    delta = actual - predicted
    l2, reference = float(np.linalg.norm(delta.ravel())), float(np.linalg.norm(actual.ravel()))
    index = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    return {"shape": list(actual.shape), "elements": actual.size, "changed_elements": int(np.count_nonzero(delta)),
        "exact": bool(np.array_equal(actual, predicted)), "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()), "l2": l2, "native_l2": reference,
        "relative_l2": l2 / reference if reference else None, "signed_mean": float(delta.mean()),
        "largest_effect": {"coordinate": [int(x) for x in index], "native_minus_forecast": float(delta[index])}}


def worker():
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if request["config"] != config:
        raise ValueError("configuration differs from the staged request")
    for name, expected in request["source_hashes"].items():
        if file_sha(name) != expected:
            raise ValueError("staged source differs: " + name)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from PIL import Image
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    functions = runpy.run_path("flux_functional_denoiser.py")
    vae_functions = runpy.run_path("flux_functional_vae.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    if any(config[name] != value for name, value in (("height", 512), ("width", 512), ("steps", 4), ("guidance_scale", 1.))):
        raise ValueError("requires the declared 512px four-step distilled profile")
    output = Path("flux-trajectory-result")
    output.mkdir(exist_ok=False)
    started, timeline = time.monotonic(), []
    counts = {"prefix_module_entries": 0, "prediction_module_entries": 0, "native_module_entries": 0,
              "selected_block_entries": 0, "vae_decoder_entries": 0, "predicted_transitions": 0, "native_transitions": 0}
    phase = {"name": "prefix"}

    def event(kind, **details):
        row = json_value({"sequence": len(timeline), "kind": kind, "seconds": time.monotonic() - started, **details})
        timeline.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_TRAJECTORY_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    root = Path(config["model_path"])
    save(output / "plan.json", config)
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(files) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the declared file inventory")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein without guidance embeddings")
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    hooks, selected_ids = [], {id(m) for m in (*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks)}

    def guard(module, args):
        wall()
        if phase["name"] == "prediction":
            counts["prediction_module_entries"] += 1
            raise RuntimeError("native module entry forbidden during functional forecast")
        counts["prefix_module_entries" if phase["name"] == "prefix" else "native_module_entries"] += 1
        if id(module) in selected_ids:
            counts["selected_block_entries"] += 1
        if module is pipe.vae.decoder:
            counts["vae_decoder_entries"] += 1

    hooks.append(torch.nn.modules.module.register_module_forward_pre_hook(guard))
    transaction = None
    try:
        pipe.text_encoder.to("cuda")
        with torch.inference_mode():
            conditioning, txt_ids = pipe.encode_prompt(config["prompt"], device=torch.device("cuda"),
                num_images_per_prompt=1, max_sequence_length=512)
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                conditioning.dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(config["seed"]))
            sigmas = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
            timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("schedule geometry differs")
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        pipe.vae.to("cuda")
        pipe.vae.bn.to("cpu")
        adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
        session = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
                                  timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
        session.continue_(1)
        parent = session.capture()
        if parent.execution_point["phase"] != "joint" or parent.execution_point["index"] != 0:
            raise ValueError("observed initial parent is not before joint0")
        store = LocalStore(output / "typed-store")
        store.save(parent)
        if counts["selected_block_entries"] or counts["vae_decoder_entries"]:
            raise ValueError("selected suffix entered before prediction")
        phase["name"] = "prediction"
        event("frontier_captured", parent=parent.fingerprint, counts=dict(counts))

        def pack(value):
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu()
                array = array.float() if array.dtype == torch.bfloat16 else array
                return {"tensor_dtype": str(value.dtype), "array": array.numpy().copy()}
            if isinstance(value, Mapping):
                return {key: pack(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return tuple(pack(item) for item in value)
            return value

        def unpack(value):
            if isinstance(value, Mapping):
                if set(value) == {"tensor_dtype", "array"}:
                    return torch.from_numpy(np.array(value["array"], copy=True)).to(device="cuda", dtype=getattr(torch, value["tensor_dtype"].split(".")[-1]))
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(unpack(item) for item in value)
            return value

        modules = {"transformer": pipe.transformer, "vae": pipe.vae}
        parameter_map = {prefix + "." + name: value for prefix, module in modules.items()
                         for name, value in (*module.named_parameters(), *module.named_buffers())}
        parameter_signatures = {name: (id(value), value.data_ptr(), value._version) for name, value in parameter_map.items()}

        def weight_check(stage, *, expected=None):
            wall()
            if any(getattr(pipe, name) is not module for name, module in modules.items()):
                raise ValueError("pipeline module binding changed at " + stage)
            current = {prefix + "." + name: value for prefix, module in modules.items()
                       for name, value in (*module.named_parameters(), *module.named_buffers())}
            if set(current) != set(parameter_map):
                raise ValueError("registered parameter/buffer inventory changed at " + stage)
            result = {}
            for name, value in current.items():
                if (id(value), value.data_ptr(), value._version) != parameter_signatures[name]:
                    raise ValueError("native parameter/buffer identity changed: " + name)
                result[name] = tensor_descriptor(value)
            if expected is not None and result != expected:
                raise ValueError("native parameter bytes changed at " + stage)
            save(output / ("weights-" + stage + ".json"), result)
            return result

        weights = weight_check("before-prediction")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}
        implementation = {inspect.getfile(kind): file_sha(inspect.getfile(kind)) for kind in
                          (type(pipe), type(pipe.transformer), type(pipe.vae), type(pipe.vae.decoder), type(pipe.scheduler), type(adapter))}
        source = {"request": request, "checkpoint_files": files, "parent_fingerprint": parent.fingerprint,
            "execution": dict(adapter.execution), "versions": versions, "implementation_sources": implementation,
            "vae_config": dict(pipe.vae.config), "image_processor_config": dict(pipe.image_processor.config),
            "model_identity": {"checkpoint_sha256": identity(files), "adapter": adapter.model_identity},
            "arithmetic": config["arithmetic"], "low_level_kernels": functions.get("SHARED_KERNELS"),
            "observed_scope": config["observed"], "RAM_only_intermediates": True}
        save(output / "source.json", source)
        transaction = ForecastTransaction(output / "transaction", model_identity=source["model_identity"],
            input_identity=parent.fingerprint, source_identity=request["source_hashes"],
            frontier={"kind": "architectural", "boundary": "observed typed pre-joint0 image/text/context"},
            budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000,
                     "max_bytes": 192 << 20, "max_files": 2048})
        cache, order = {}, []

        def arrays_at(event_info, state):
            tag = f"step{event_info['step']}.{event_info['phase']}.{event_info['index']}"
            kind = event_info["phase"]
            arrays = ({"image": state["image"], "text": state["text"]} if kind in {"project", "joint", "single"}
                      else {"noise_prediction": state["noise_prediction"]} if kind == "readout"
                      else {"latent": state["latent"]})
            if kind == "project":
                arrays.update({"context." + key: state["context"][key] for key in ("temb", "img_mod", "txt_mod", "single_mod")})
            return tag, arrays

        def snapshot(tag, array, *, retain_full=False):
            array = array.detach().to("cpu").contiguous()
            descriptor = tensor_descriptor(array)
            # Byte-backed NumPy arrays cannot become writable; native comparison
            # reads these exact bytes after verifying their declared descriptors.
            widened = array.float().numpy() if array.dtype == torch.bfloat16 else array.numpy()
            frozen = np.frombuffer(widened.tobytes(), dtype=widened.dtype).reshape(widened.shape)
            indices = np.unique(np.linspace(0, frozen.size - 1, min(config["witness_values"], frozen.size), dtype=np.int64))
            cache[tag] = frozen
            result = {"descriptor": descriptor, "float_array_sha256": sha(frozen.tobytes()),
                      "witness_indices": indices, "witness_values": frozen.ravel()[indices], "full_retained": retain_full}
            if retain_full:
                result["full_values"] = frozen
            return result

        with transaction:
            transaction.frontier("initial", {"state": pack(parent.payload), "manifest": parent.manifest(),
                "native_entry_counts": dict(counts), "weights_digest": identity(weights),
                "scope": config["scope"], "arithmetic": config["arithmetic"]})

            def predict(context):
                state = unpack(context.frontiers["initial"]["state"])
                boundary_rows = []

                def on_boundary(info, current):
                    wall()
                    tag, arrays = arrays_at(info, current)
                    order.append(tag)
                    boundary_rows.append({"tag": tag, "event": info,
                        "arrays": {name: snapshot(tag + "/" + name, value, retain_full=info["phase"] in {"readout", "scheduler"})
                                   for name, value in arrays.items()}})
                    counts["predicted_transitions"] += 1
                    if counts["predicted_transitions"] > config["max_transitions"]:
                        raise RuntimeError("functional transition budget exhausted")
                    if info["phase"] == "scheduler":
                        event("predicted_scheduler", step=info["step"], transitions=counts["predicted_transitions"])

                with torch.inference_mode():
                    final = functions["functional_trajectory"](pipe.transformer, state, on_boundary=on_boundary,
                                                              max_transitions=config["max_transitions"])
                    decoded = vae_functions["decode_functional"](pipe, final["latent"], final["img_ids"], 512, 512)
                vae_rows = {name: snapshot("vae/" + name, decoded[key], retain_full=key == "decoded")
                            for name, key in (("decoder_input", "decoder_input"), ("decoded", "decoded"))}
                pixels = decoded["pixels"]
                if isinstance(pixels, torch.Tensor):
                    pixels = pixels.detach().cpu().numpy()
                pixels = np.asarray(pixels, dtype=np.uint8)
                if pixels.shape == (1, 512, 512, 3):
                    pixels = pixels[0]
                if pixels.shape != (512, 512, 3):
                    raise ValueError("functional RGB geometry differs")
                Image.fromarray(pixels).save(output / "forecast.png")
                return {"predictions": {"boundaries": boundary_rows, "vae": vae_rows,
                    "pixels": pixels, "final_latent": final["latent"].float().cpu().numpy(),
                    "forecast_png_sha256": file_sha(output / "forecast.png"),
                    "boundary_cache_bytes": sum(value.nbytes for value in cache.values()),
                    "own_state": True, "native_refreshes": 0, "RGB_forecast": True}}

            transaction.predict("trajectory", predict)
            if counts["predicted_transitions"] != config["max_transitions"] or counts["prediction_module_entries"]:
                raise ValueError("forecast transition/exclusion contract differs")
            if session.capture(retain=False).fingerprint != parent.fingerprint:
                raise ValueError("native parent changed during prediction")
            weight_check("at-release", expected=weights)
            seal = transaction.seal()
            event("trajectory_sealed", seal=seal["seal_sha256"], counts=dict(counts), cached_arrays=len(cache))
            transaction.release()
            phase["name"] = "native"

            def execute(context):
                prediction = context.predictions["trajectory"]
                rows, native_order, index = [], [], 0

                def inspect_array(tag, actual, sealed):
                    frozen = cache[tag]
                    if sha(frozen.tobytes()) != sealed["float_array_sha256"]:
                        raise ValueError("full forecast RAM bytes changed: " + tag)
                    actual_array = actual.detach().float().cpu().numpy()
                    positions = np.asarray(sealed["witness_indices"], dtype=np.int64)
                    return {"descriptor": tensor_descriptor(actual), "full_error": compare(actual_array, frozen),
                            "witness_indices": positions, "witness_values": actual_array.ravel()[positions]}

                while index < len(prediction["boundaries"]):
                    wall()
                    before = session.capture(retain=False)
                    info = {"step": int(before.payload["step"]), "phase": before.payload["phase"],
                            "index": int(before.payload["cursor"])}
                    receipt = session.continue_(1)
                    store.receipt(receipt)
                    state = session.capture(retain=False).payload
                    tag, arrays = arrays_at(info, state)
                    if index >= len(prediction["boundaries"]) or tag != prediction["boundaries"][index]["tag"]:
                        raise ValueError("native and functional boundary order differs")
                    sealed_row = prediction["boundaries"][index]
                    row = {"tag": tag, "receipt": receipt.to_dict(), "arrays":
                           {name: inspect_array(tag + "/" + name, value, sealed_row["arrays"][name]) for name, value in arrays.items()}}
                    if info["phase"] in {"readout", "scheduler"}:
                        for name, value in arrays.items():
                            row["arrays"][name]["full_values"] = value.float().cpu().numpy().copy()
                    rows.append(row)
                    native_order.append(tag)
                    index += 1
                    counts["native_transitions"] += 1
                    if counts["native_transitions"] > config["max_native_transitions"]:
                        raise RuntimeError("native transition budget exhausted")
                    if info["phase"] == "scheduler":
                        event("native_scheduler", step=info["step"], error=row["arrays"]["latent"]["full_error"])
                if native_order != order:
                    raise ValueError("trajectory order differs")
                final_payload = session.capture(retain=False).payload
                if final_payload["step"] != len(final_payload["timesteps"]):
                    raise ValueError("native trajectory did not reach the declared final timestep")
                native_vae = {}

                def vae_input(module, args):
                    native_vae["decoder_input"] = args[0].detach().clone()

                def vae_output(module, args, result):
                    native_vae["decoded"] = result.detach().clone()

                observing = [pipe.vae.decoder.register_forward_pre_hook(vae_input), pipe.vae.decoder.register_forward_hook(vae_output)]
                try:
                    native_image = adapter.decode(session, height=512, width=512)
                finally:
                    for hook in observing:
                        hook.remove()
                pixels = np.asarray(native_image, dtype=np.uint8)
                native_image.save(output / "native.png")
                vae_comparisons = {name: inspect_array("vae/" + name, value, prediction["vae"][name]) for name, value in native_vae.items()}
                vae_comparisons["decoded"]["full_values"] = native_vae["decoded"].float().cpu().numpy()
                return {"boundaries": rows, "vae": vae_comparisons, "pixels": pixels,
                    "pixel_error": compare(pixels, prediction["pixels"]), "native_png_sha256": file_sha(output / "native.png"),
                    "counts": dict(counts), "future_inputs": "native own-state only; no forecast payload installed"}

            transaction.execute("native", execute)
        weight_check("after-native", expected=weights)
        verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
        observation = transaction.observation("native")
        summary = {"schema": "mdb-flux-trajectory-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
            "request_sha256": request["request_sha256"], "mechanics_status": "valid", "terminal_status": "unassessed",
            "model_identity": source["model_identity"], "counts": counts, "seal": seal,
            "transaction_verification": verification, "pixel_error": observation["pixel_error"],
            "worker_seconds": time.monotonic() - started, "RGB_forecast": True,
            "own_state": True, "native_refreshes": 0, "arithmetic": config["arithmetic"],
            "scope": config["scope"], "limits": config["limits"]}
        save(output / "transaction-report.json", transaction.report())
        save(output / "summary.json", summary)
        event("complete", pixel_error=observation["pixel_error"], counts=counts)
        save(output / "files.json", {p.relative_to(output).as_posix(): file_sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
        runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config["max_archive_bytes"])
    finally:
        for hook in hooks:
            hook.remove()
        if transaction is not None:
            transaction.close()


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-trajectory-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            report = {"schema": "mdb-flux-trajectory-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                      "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", report)
            runpy.run_path("campaign_support.py")["emit"](output, report, max_compressed_bytes=128 << 20)
        raise
