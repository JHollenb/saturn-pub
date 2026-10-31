"""Sealed conditional Klein image-MLP writes with public typed RGB consumers.

GPU prefixes stop before joint.0.ff. NumPy forecasts are conditional on those
observed prefixes. Only local FF/write arithmetic is forecast; RGB and scene
relationships are subsequent native observations, not predicted images.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from pathlib import Path

ARMS = ("native", "sham", "forecastself", "donorfull", "donorleft", "donorright")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checkpoint_inventory(root):
    return {path.relative_to(root).as_posix(): file_sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}


def quantize_bf16(value):
    """Explicit NumPy FP32 -> BF16 round-to-nearest-even, widened back to FP32."""
    import numpy as np

    values = np.asarray(value, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("BF16 payload inputs must be finite")
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)).view(np.float32)
    if not np.isfinite(rounded).all():
        raise ValueError("BF16 payload cast overflowed")
    return np.frombuffer(rounded.tobytes(order="C"), dtype=np.float32).reshape(values.shape)


def coordinate_masks(img_ids):
    """Native IDs are (T,H,W,L); split the actual W coordinates in half."""
    import numpy as np

    coordinates = np.asarray(img_ids)
    if coordinates.ndim != 3 or coordinates.shape[0] != 1 or coordinates.shape[2] != 4:
        raise ValueError("expected native B1 (T,H,W,L) image position IDs")
    columns = np.unique(coordinates[0, :, 2])
    if len(columns) < 2 or len(columns) % 2:
        raise ValueError("spatial controls require an even nonempty image-column grid")
    left = np.isin(coordinates[:, :, 2], columns[:len(columns) // 2])[:, :, None]
    right = ~left
    return {"left": left, "right": right,
            "description": {"coordinate_axis": 2, "native_axes": ["T", "H", "W", "L"],
                            "left_columns": columns[:len(columns) // 2].tolist(),
                            "right_columns": columns[len(columns) // 2:].tolist(),
                            "scope": "fixed coordinate halves, not learned object masks"}}


class _PrefixStop(BaseException):
    pass


def worker():
    raw_config = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    config = json.loads(raw_config)
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if request.get("config") != config:
        raise ValueError("worker configuration differs from sealed request")
    for name, expected in request["source_hashes"].items():
        if file_sha(name) != expected:
            raise ValueError("staged source differs: " + name)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config.get("cpu_threads", 4))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub import Act
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_ports import NativePortCatalog
    from mdb.forward_recording import record_forward, tensor_descriptor

    formulas = runpy.run_path("flux_mlp_equations.py")
    prompts = config["prompts"]
    if (len(prompts) != 2 or {row["id"] for row in prompts} != {"left", "right"}
            or tuple(config.get("arms", ARMS)) != ARMS
            or any(config.get(name, value) != value for name, value in
                   (("seed", 17), ("height", 512), ("width", 512), ("steps", 4),
                    ("guidance_scale", 1.), ("target_block", 0), ("target_step", 0)))):
        raise ValueError("requires sealed two-scene 512px/4step/seed17/guidance1 joint.0 six-arm panel")
    if config.get("max_images", 12) < len(prompts) * len(ARMS):
        raise ValueError("image budget does not admit the declared panel")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise ValueError("this finite numerical profile requires an admitted CUDA worker")
    output = Path("flux-mlp-result")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    max_wall = config.get("max_wall_seconds", 840)
    counts = {"prefix_attempts": 0, "prefix_stops": 0, "target_ff_entries": 0,
              "vae_entries": 0, "denied_entries": 0, "admitted_adapter_steps": 0,
              "completed_adapter_steps": 0, "writer_records": 0, "acts": 0, "images": 0}
    events, phase = [], {"released": False, "prefix_active": False}

    def event(kind, **detail):
        row = {"sequence": len(events), "kind": kind, "elapsed_seconds": time.monotonic() - started, **detail}
        events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_MLP_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def wall_check():
        if time.monotonic() - started > max_wall:
            raise RuntimeError("finite worker wall budget exhausted")

    root = Path(config["model_path"])
    files = checkpoint_inventory(root)
    if not files or identity(files) != config["expected_checkpoint_sha256"]:
        raise ValueError("local pipeline checkpoint inventory differs from pre-bound bytes")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds:
        raise ValueError("requires plain distilled Klein without guidance embeddings")
    for component in (pipe.transformer, pipe.text_encoder, pipe.vae):
        component.eval().requires_grad_(False)
    if any(hasattr(module, "_hf_hook") for module in pipe.transformer.modules()):
        raise ValueError("caller offload hooks are outside the public adapter profile")
    # Encode both conditions before the transformer becomes resident. The text
    # encoder is then parked on CPU for the entire typed trajectory experiment.
    pipe.text_encoder.to("cuda")
    prepared = {}
    for prompt in prompts:
        wall_check()
        with torch.inference_mode():
            conditioning, text_ids = pipe.encode_prompt(prompt["text"], device=torch.device("cuda"),
                num_images_per_prompt=1, max_sequence_length=512)
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4,
                512, 512, conditioning.dtype, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(17))
        sigmas = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, resolved_steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=sigmas,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if resolved_steps != 4 or len(timesteps) != 4:
            raise ValueError("native schedule differs from four-step profile")
        prepared[prompt["id"]] = {"latent": latent, "conditioning": conditioning, "img_ids": img_ids,
            "txt_ids": text_ids, "timesteps": timesteps.clone(), "sigmas": pipe.scheduler.sigmas.clone()}
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    # Public adapter requires CPU BN statistics, while native VAE decoder
    # parameters may remain on CUDA. Freeze this placement before wrapping.
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    target = pipe.transformer.transformer_blocks[0]
    if (target.ff.linear_in.bias is not None or target.ff.linear_out.bias is not None
            or next(target.ff.parameters()).dtype != torch.bfloat16):
        raise ValueError("expected bias-free native BF16 Klein image FF")
    transitions_per_image = 4 * (adapter.joint_layers + adapter.single_layers + 3)
    required_steps = 2 * len(prompts) + len(prompts) * len(ARMS) * (transitions_per_image - 2)
    if required_steps > config.get("max_adapter_steps", 2048):
        raise ValueError("declared native adapter step geometry exceeds finite budget")
    model_identity = {"checkpoint_sha256": identity(files), "adapter": adapter.model_identity,
                      "target": "transformer_blocks.0.ff", "step": 0}
    implementation_sources = {str(Path(inspect.getfile(kind)).resolve()): file_sha(inspect.getfile(kind))
        for kind in (type(pipe), type(pipe.transformer), type(target), type(target.ff), type(target.norm2),
                     type(pipe.vae), type(pipe.scheduler), type(adapter))}
    semantics = {"schema": "mdb-flux-scene-hypothesis-v1",
        "hypothesis": "Replacing the recipient image-MLP write with the mirrored donor write shifts red-cube/blue-sphere horizontal ordering toward the donor scene.",
        "predicted_direction": {"left": "donor right tends to increase red_x minus blue_x",
                                "right": "donor left tends to decrease red_x minus blue_x"},
        "masks": "fixed native (T,H,W,L) W-coordinate halves; not learned object masks",
        "controls": "exact native sham and analytic forecastself distinguish copy and approximation effects",
        "status": "predeclared semantic hypothesis; no pixel or object outcome guaranteed",
        "numeric_prediction_scope": "conditional selected FF output, gated write, and local image residual only",
        "RGB_prediction": False, "evaluation_scope": "RGB/object/relationship observations assessed separately"}
    save(output / "plan.json", config)
    save(output / "semantic-hypothesis.json", semantics)
    save(output / "source.json", {"request": request, "checkpoint_files": files, "model_identity": model_identity,
        "native_execution": dict(adapter.execution), "implementation_sources": implementation_sources,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "CUDA": {"device": torch.cuda.get_device_name(), "runtime": torch.version.cuda,
                 "cudnn": torch.backends.cudnn.version(), "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
        "placement": "text encoder CUDA only for conditioning then CPU; transformer CUDA BF16; VAE parameters CUDA and BN CPU",
        "device_transition": "native BF16 CUDA inputs/weights widened to immutable FP32 CPU; independent equations FP64; supplied writes rounded BF16 explicitly"})
    weights, weight_binding = {}, {}
    for name in ("linear_in", "linear_out"):
        parameter = getattr(target.ff, name).weight
        widened = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        weights[name] = np.frombuffer(widened.tobytes(), dtype=np.float32).reshape(widened.shape)
        weight_binding[name] = {"parameter": "transformer.transformer_blocks.0.ff." + name + ".weight",
            "native": tensor_descriptor(parameter), "external_fp32_shape": list(widened.shape),
            "external_fp32_sha256": hashlib.sha256(memoryview(weights[name]).cast("B")).hexdigest(),
            "recipe": "detach native checkpoint BF16 parameter; to CPU FP32 contiguous; immutable C-order NumPy bytes",
            "external_matrix_serialized": False}
        del widened
    save(output / "external-weights.json", {"checkpoint_sha256": identity(files), "weights": weight_binding,
        "authority": "externally supplied full selected FF matrices from pinned checkpoint; not compressed or inferred"})
    hooks, sessions, parents, prefixes = [], {}, {}, {}
    store = LocalStore(output / "typed-store")
    transaction = None

    def weight_check(stage):
        checks = {}
        for name, binding in weight_binding.items():
            actual = tensor_descriptor(getattr(target.ff, name).weight)
            widened = getattr(target.ff, name).weight.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            current_sha = hashlib.sha256(memoryview(widened).cast("B")).hexdigest()
            copy_sha = hashlib.sha256(memoryview(weights[name]).cast("B")).hexdigest()
            if (any(actual[key] != binding["native"][key] for key in ("shape", "dtype", "sha256"))
                    or current_sha != binding["external_fp32_sha256"] or copy_sha != current_sha
                    or weights[name].flags.writeable):
                raise ValueError("external/native FF weight binding changed at " + stage)
            checks[name] = {"native": actual, "external_fp32_sha256": current_sha, "exact": True}
        save(output / ("weight-check-" + stage + ".json"), checks)

    def ff_guard(module, args):
        wall_check()
        counts["target_ff_entries"] += 1
        if not phase["released"]:
            counts["denied_entries"] += 1
            raise RuntimeError("selected native FF forbidden before global forecast seal")

    def vae_guard(module, args):
        wall_check()
        counts["vae_entries"] += 1
        if not phase["released"]:
            counts["denied_entries"] += 1
            raise RuntimeError("native VAE forbidden before global forecast seal")

    def advance(session, steps, branch):
        wall_check()
        counts["admitted_adapter_steps"] += steps
        if counts["admitted_adapter_steps"] > config.get("max_adapter_steps", 2048):
            raise RuntimeError("adapter transition budget exhausted")
        receipt = session.continue_(steps)
        counts["completed_adapter_steps"] += steps
        store.receipt(receipt)
        with (output / "transition-receipts.jsonl").open("a") as stream:
            stream.write(json.dumps({"branch": branch, "receipt": receipt.to_dict()}, sort_keys=True) + "\n")
        return receipt

    hooks.append(target.ff.register_forward_pre_hook(ff_guard))
    hooks.append(pipe.vae.decoder.register_forward_pre_hook(vae_guard))
    try:
        for prompt in prompts:
            key = prompt["id"]
            session = adapter.session(**prepared[key])
            advance(session, 1, key + ".project")
            parent = session.capture()
            if parent.execution_point["phase"] != "joint" or parent.execution_point["index"] != 0:
                raise ValueError("typed parent is not before step0 joint0")
            store.save(parent)
            sessions[key], parents[key] = session, parent
            partial = session.fork(parent)
            before = partial.capture(retain=False)
            observed = {}

            def observe_r(module, args, saved=observed):
                saved["r"] = args[0].detach().clone()

            def stop_x(module, args, saved=observed):
                saved["x"] = args[0].detach().clone()
                counts["prefix_stops"] += 1
                raise _PrefixStop()

            observing = [target.norm2.register_forward_pre_hook(observe_r),
                         target.ff.register_forward_pre_hook(stop_x, prepend=True)]
            counts["prefix_attempts"] += 1
            phase["prefix_active"] = True
            event("prefix_start", scene=key, parent=parent.fingerprint)
            try:
                partial.continue_()
            except _PrefixStop:
                pass
            else:
                raise RuntimeError("genuine prefix failed to stop before selected FF")
            finally:
                phase["prefix_active"] = False
                for hook in observing:
                    hook.remove()
            after = partial.capture(retain=False)
            if before.fingerprint != after.fingerprint or set(observed) != {"r", "x"}:
                raise ValueError("partial native block changed typed parent or missed its exact boundary")
            context = parent.payload["context"]
            gate = Flux2Modulation.split(context["img_mod"], 2)[1][2]
            ids = parent.payload["img_ids"].detach().cpu().numpy().copy()
            masks = coordinate_masks(ids)
            prefixes[key] = {"r": observed["r"].float().cpu().numpy().copy(),
                "x": observed["x"].float().cpu().numpy().copy(), "gate": gate.float().cpu().numpy().copy(),
                "img_ids": ids, "masks": masks, "prompt": prompt,
                "parent_fingerprint": parent.fingerprint, "parent_manifest": parent.manifest(),
                "native_descriptors": {name: tensor_descriptor(value) for name, value in observed.items()},
                "prefix_contract": {"kind": "genuine CUDA partial joint block", "stop_before": "transformer_blocks.0.ff",
                    "captured_r": "norm2 actual argument", "captured_x": "FF actual argument before its body",
                    "native_target_FF_computed": False, "typed_parent_unchanged": True,
                    "CPU_record_prefix_used": False}, "typed_prefix_before": before.manifest(), "typed_prefix_after": after.manifest()}
            event("prefix_complete", scene=key, target_ff_entries=counts["target_ff_entries"], parent_unchanged=True)
            del partial, observed, context, gate
        if not np.array_equal(prefixes["left"]["img_ids"], prefixes["right"]["img_ids"]):
            raise ValueError("donor/recipient native spatial coordinate order differs")
        if counts["target_ff_entries"] or counts["vae_entries"] or counts["denied_entries"]:
            raise RuntimeError("native selected suffix exclusion failed")
        pre_release = dict(counts)

        class Writer(torch.nn.Module):
            def forward(self, r, write):
                return r + write

        writer = Writer().eval()
        transaction = ForecastTransaction(output / "transaction", model=writer,
            model_identity=model_identity, input_identity=identity({key: value.fingerprint for key, value in parents.items()}),
            source_identity=request["source_hashes"], frontier={"kind": "architectural", "boundary": "observed-native-joint0-FF-input",
                "scope": "CPU FP32 paramless writer; native consumer is separately bound CUDA BF16 public adapter"},
            budgets={"max_native_calls": 16, "max_wall_ms": int(max_wall * 1000), "max_bytes": 768 << 20, "max_files": 4096})
        with transaction:
            transaction.frontier("prefixes", prefixes)
            transaction.frontier("external_weights", {"checkpoint_sha256": identity(files), "weights": weight_binding})
            transaction.frontier("semantic_hypothesis", semantics)
            transaction.frontier("exclusion", {"pre_release_counters": pre_release, "target_ff_entries": 0, "VAE_entries": 0})
            weight_check("pre-prediction")
            for key in ("left", "right"):
                def predict(context, scene=key):
                    prefix = context.frontiers["prefixes"][scene]
                    result = formulas["forecast"](prefix["x"], prefix["r"], prefix["gate"], weights,
                                                   chunk_tokens=config.get("chunk_tokens", 32))
                    return {"predictions": {"scene": scene, "values": result,
                            "scope": "conditional selected FF y, gated write and local residual only; no RGB forecast"},
                        "payloads": {"write": quantize_bf16(result["write"])}}
                transaction.predict(key, predict, payload_provenance={"write": {
                    "formula": "donor gate multiplied by independent donor FF output",
                    "cast_policy": "FP64->FP32->BF16 round-to-nearest-even widened to FP32; later exact native BF16 cast"}})
            seal = transaction.seal()
            event("global_forecast_sealed", seal_sha256=seal["seal_sha256"], pre_release_counters=dict(counts))
            transaction.release()
            weight_check("at-release")
            phase["released"] = True
            event("native_suffix_released", seal_sha256=seal["seal_sha256"])

            def execute(context):
                rows, native_audits = {}, {}
                for recipient in ("left", "right"):
                    donor = "right" if recipient == "left" else "left"
                    source = sessions[recipient].fork(parents[recipient])
                    native = {}

                    def r_hook(module, args):
                        native["r"] = args[0].detach().clone()

                    def x_hook(module, args):
                        native["x"] = args[0].detach().clone()

                    def y_hook(module, args, value):
                        native["y"] = value.detach().clone()

                    observing = [target.norm2.register_forward_pre_hook(r_hook), target.ff.register_forward_pre_hook(x_hook),
                                 target.ff.register_forward_hook(y_hook)]
                    try:
                        advance(source, 1, recipient + ".native_joint0")
                    finally:
                        for hook in observing:
                            hook.remove()
                    native_post = source.capture()
                    store.save(native_post)
                    base_image, base_text = source.read("image"), source.read("text")
                    gate = Flux2Modulation.split(parents[recipient].payload["context"]["img_mod"], 2)[1][2]
                    actual_write = gate * native["y"]
                    reconstructed = native["r"] + actual_write
                    exact_update = bool(torch.equal(base_image, reconstructed))
                    exact_prefix = {name: bool(np.array_equal(native[name].float().cpu().numpy(), prefixes[recipient][name])) for name in ("r", "x")}
                    if not exact_update or not all(exact_prefix.values()):
                        raise RuntimeError("native FF/update arithmetic or preseal prefix join differs")
                    native_audits[recipient] = {"r": native["r"].float().cpu().numpy().copy(),
                        "x": native["x"].float().cpu().numpy().copy(), "y": native["y"].float().cpu().numpy().copy(),
                        "write": actual_write.float().cpu().numpy().copy(), "image": base_image.float().cpu().numpy().copy(),
                        "native_descriptors": {name: tensor_descriptor(value) for name, value in
                            {**native, "write": actual_write, "image": base_image, "text": base_text}.items()},
                        "exact_update": exact_update, "exact_prefix": exact_prefix,
                        "postblock_manifest": native_post.manifest(), "parent_fingerprint": parents[recipient].fingerprint}
                    for arm in ARMS:
                        row_id = recipient + "." + arm
                        event("branch_start", row=row_id)
                        branch = source.fork(native_post)
                        cpu_capture = None
                        act_record = None
                        mask = None
                        origin = None
                        if arm in {"native", "sham"}:
                            candidate = base_image.clone()
                            if arm == "sham":
                                act = Act.replace("image", candidate, name=row_id)
                                receipt = branch.apply(act)
                                act_record = {"act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": store.receipt(receipt)}
                                counts["acts"] += 1
                        else:
                            origin = recipient if arm == "forecastself" else donor
                            write = context.payloads[origin]["write"]
                            r_cpu = torch.from_numpy(np.array(prefixes[recipient]["r"], copy=True))
                            w_cpu = torch.from_numpy(np.array(write, copy=True))
                            cpu_capture = record_forward(writer, (r_cpu, w_cpu), model_identity=model_identity,
                                input_identity=identity({"recipient": recipient, "write_source": origin, "arm": arm}),
                                execution={"backend": "cpu", "dtype": "float32", "scope": "parameterless writer"}, capture_ports=True)
                            counts["writer_records"] += 1
                            save(output / "writer-traces" / (row_id + ".json"), cpu_capture.trace)
                            supplied = cpu_capture.output.to(device="cuda", dtype=torch.bfloat16)
                            if arm in {"donorleft", "donorright"}:
                                side = "left" if arm == "donorleft" else "right"
                                mask = torch.from_numpy(np.array(prefixes[recipient]["masks"][side], copy=True)).to("cuda")
                                candidate = torch.where(mask, supplied, base_image)
                            else:
                                candidate = supplied
                            act = Act.replace("image", candidate, name=row_id)
                            receipt = branch.apply(act)
                            act_record = {"act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": store.receipt(receipt),
                                "source_forecast": origin, "payload_name": "write", "writer_trace": cpu_capture.trace["fingerprint"],
                                "writer_output_port": NativePortCatalog(cpu_capture.trace).resolve("module:<root>:return")}
                            counts["acts"] += 1
                        image_after = branch.read("image")
                        text_exact = bool(torch.equal(branch.read("text"), base_text))
                        unmasked_exact = True if mask is None else bool(torch.equal(image_after[~mask.expand_as(image_after)], base_image[~mask.expand_as(base_image)]))
                        if not text_exact or not unmasked_exact:
                            raise RuntimeError("Act changed recipient text or unmasked native image rows")
                        applied = branch.capture()
                        advance(branch, transitions_per_image - 2, row_id + ".native_suffix")
                        final = branch.capture()
                        event("RGB_decode_start", row=row_id)
                        rgb = adapter.decode(branch, height=512, width=512)
                        pixels = np.asarray(rgb, dtype=np.uint8)
                        image_path = output / "images" / (row_id + ".png")
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        rgb.save(image_path)
                        counts["images"] += 1
                        if counts["images"] > config.get("max_images", 12):
                            raise RuntimeError("native RGB image budget exhausted")
                        rows[row_id] = {"recipient": recipient, "donor": origin, "arm": arm,
                            "parent_fingerprint": parents[recipient].fingerprint, "native_postblock_fingerprint": native_post.fingerprint,
                            "applied_manifest": applied.manifest(), "final_manifest": final.manifest(), "Act": act_record,
                            "supplied_image_descriptor": tensor_descriptor(candidate), "text_preserved_exact": text_exact,
                            "unmasked_native_exact": unmasked_exact, "native_sham_image_exact": bool(torch.equal(candidate, base_image)) if arm in {"native", "sham"} else None,
                            "mask": None if mask is None else {"side": "left" if arm == "donorleft" else "right",
                                "rows": int(mask.sum()), "source": "sealed native image-coordinate halves"},
                            "final_latent": branch.read("latent").float().cpu().numpy().copy(),
                            "RGB": {"path": image_path.relative_to(output).as_posix(), "png_sha256": file_sha(image_path),
                                "pixel_sha256": sha(pixels.tobytes()), "shape": list(pixels.shape), "dtype": "uint8",
                                "authority": "public adapter native VAE+image_processor PIL output; no RGB equation forecast"},
                            "future_scope": "all remaining native blocks, flow commits and VAE; measured after sealed first-write intervention"}
                        event("branch_complete", row=row_id, RGB_sha256=rows[row_id]["RGB"]["pixel_sha256"])
                        del branch, image_after, applied, final, candidate, rgb, pixels, cpu_capture
                        gc.collect()
                    native_row, sham_row = rows[recipient + ".native"], rows[recipient + ".sham"]
                    sham_control = {"final_latent_exact": bool(np.array_equal(native_row["final_latent"], sham_row["final_latent"])),
                        "RGB_exact": native_row["RGB"]["pixel_sha256"] == sham_row["RGB"]["pixel_sha256"]}
                    native_audits[recipient]["sham_control"] = sham_control
                    if not all(sham_control.values()):
                        raise RuntimeError("exact native sham changed the final native RGB consumer")
                return {"rows": rows, "native_audits": native_audits,
                        "scope": "local FF/write forecasts; RGB/objects are subsequent native observations"}

            transaction.execute("native", execute)
        weight_check("post-execution")
        verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
        summary = {"schema": "mdb-flux-mlp-forecast-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
            "request_sha256": request["request_sha256"], "counts": counts, "pre_release_counters": pre_release,
            "model_identity": model_identity, "native_execution": dict(adapter.execution), "global_seal": seal,
            "transaction_verification": verification, "mechanics_status": "valid", "images": counts["images"],
            "worker_seconds": time.monotonic() - started, "semantic_status": "unassessed; predeclared hypothesis retained",
            "terminal_status": "unassessed", "limits": ["Observed genuine GPU prefix conditions the independent CPU FP64 equations.",
                "The selected FF weights are externally supplied and checkpoint-bound, not serialized or compressed.",
                "CPU FP32 writer capture and CUDA BF16 native consumer have separate numerical contracts.",
                "Analytic forecastself is distinct from the exact native sham; precision-induced effects remain evidence.",
                "Spatial masks are fixed coordinate halves, not discovered object masks.",
                "Only selected FF/write/update is predicted; RGB, objects, and relationships are native consumer observations."]}
        save(output / "transaction-report.json", transaction.report())
        save(output / "summary.json", summary)
        raw_before_manifest = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        event("archive_start", raw_output_bytes_before_manifest=raw_before_manifest)
        save(output / "files.json", {path.relative_to(output).as_posix(): file_sha(path)
                                     for path in sorted(output.rglob("*")) if path.is_file()})
        raw_total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        print("MDB_FLUX_ARCHIVE_GEOMETRY " + json.dumps({"raw_output_bytes": raw_total,
            "compressed_cap_bytes": config.get("max_archive_bytes", 512 << 20)}), flush=True)
        archive_started = time.monotonic()
        runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20))
        # Do not modify archived files after the immutable manifest was made.
        print("MDB_FLUX_ARCHIVE_TIMING " + json.dumps({"seconds": time.monotonic() - archive_started,
            "raw_output_bytes": raw_total, "scope": "archive/compression/stdout publication; separate from worker scientific phases"}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        if transaction is not None:
            transaction.close()


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-mlp-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            report = {"schema": "mdb-flux-mlp-forecast-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", report)
            failure_config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
            runpy.run_path("campaign_support.py")["emit"](output, report,
                max_compressed_bytes=failure_config.get("max_archive_bytes", 512 << 20))
        raise
