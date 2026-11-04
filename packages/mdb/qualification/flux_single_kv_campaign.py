"""Five native single0 K/V calculations and nine typed native continuations."""
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
    {"id": "text", "kind": "reference", "image_source": "text", "text_source": "text"},
    {"id": "sham", "kind": "edited", "image_source": "zero", "text_source": "zero"},
    {"id": "K.image", "kind": "edited", "image_source": "K", "text_source": None},
    {"id": "V.image", "kind": "edited", "image_source": "V", "text_source": None},
    {"id": "KV.image", "kind": "edited", "image_source": "KV", "text_source": None},
    {"id": "KV.both", "kind": "edited", "image_source": "KV", "text_source": "KV"},
    {"id": "text.text", "kind": "edited", "image_source": None, "text_source": "text"},
    {"id": "coherent", "kind": "edited", "image_source": "text", "text_source": "text"}]
ATTENTION_QUERY_INDICES = [0, 31, 32, 255, 511, 512, 767, 1023]


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


def emit_report(output, report, **kwargs):
    # Transaction observations are immutable Mapping views, including nested
    # comparison dictionaries. The shared emitter expects ordinary JSON values.
    runpy.run_path("campaign_support.py")["emit"](output, json_value(report), **kwargs)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if (config.get("branches") != BRANCHES or set(config.get("input_cuts", {})) != {"zero", "text"}
            or config.get("input_store") != "input-store" or config.get("source_selection") != "source-selection.json"
            or config.get("steps") != 4 or config.get("height") != 512 or config.get("width") != 512 or config.get("seed") != 47):
        raise ValueError("single-KV panel differs from fixed own-history assay")
    if config.get("attention_query_indices", ATTENTION_QUERY_INDICES) != ATTENTION_QUERY_INDICES:
        raise ValueError("sampled attention queries differ from fixed predeclared indices")
    for name, minimum, maximum in (("max_adapter_steps", 191, 240), ("max_acts", 10, 10),
            ("max_candidate_calls", 5, 5), ("max_images", 9, 9), ("max_vae_calls", 9, 9),
            ("max_wall_seconds", 1, 600), ("archive_compression_level", 1, 9)):
        if type(config.get(name)) is not int or not minimum <= config[name] <= maximum:
            raise ValueError("invalid finite single-KV budget: " + name)


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(name) != expected for name, expected in request["source_hashes"].items()):
        raise ValueError("staged single-KV source/request differs")
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

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    helper = runpy.run_path("flux_single_kv.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("single-KV requires admitted CUDA")
    output = Path("flux-single-kv-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    counts = {"model_loads": 0, "physical_batch": 1, "prompt_encodings": 0, "noise_draws": 0,
        "adapter_steps": 0, "acts": 0, "candidate_calls": 0, "receipt_emissions": 0, "images": 0, "vae_calls": 0}
    started, timeline, active = time.monotonic(), [], {"branch": "load"}

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite single-KV wall budget exhausted")

    def event(kind, **fields):
        value = json_value({"sequence": len(timeline), "kind": kind, "seconds": time.monotonic() - started, **fields})
        timeline.append(value)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_SINGLE_KV_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    selection = json.loads(Path(config["source_selection"]).read_text())
    if (selection["input_cuts"] != config["input_cuts"] or selection["expected_pngs"] != config["expected_pngs"]
            or selection["prior_job_id"] != "job-b0da807f5d60"):
        raise ValueError("prior single source selection differs")
    for path, expected in selection["files"].items():
        if not Path(path).resolve().is_relative_to(Path(config["input_store"]).resolve()) or sha(path) != expected:
            raise ValueError("single source CAS path/hash differs")
    prior = json.loads(Path(selection["source_record"]["path"]).read_text())
    audit = json.loads(Path("prior-audit.json").read_text())
    if (sha(selection["source_record"]["path"]) != selection["source_record"]["sha256"]
            or sha("prior-audit.json") != selection["audit_sha256"] or audit["mechanics_status"] != "valid"
            or audit["source"]["job_id"] != selection["prior_job_id"]
            or audit["source"]["archive_sha256"] != selection["archive_sha256"]
            or audit["source"]["request_sha256"] != selection["request_sha256"] or prior["request_sha256"] != selection["request_sha256"]):
        raise ValueError("single predecessor source/audit custody differs")
    for name, row_id in (("zero", "zero"), ("text", "late.text")):
        record = selection["prior_rows"][name]
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("prior single source row bytes differ")
        row = json.loads(Path(record["path"]).read_text())
        selected_cut = row["parent"] if name == "zero" else row["acts"][-1]["post_act"]
        if row["id"] != row_id or selected_cut != config["input_cuts"][name] or row["image"]["sha256"] != selection["expected_pngs"][name]:
            raise ValueError("single source occurrence differs")
        if name == "text" and row["acts"][-1]["slot"] != "text":
            raise ValueError("text source is not the actual text-only Act frontier")
    if "prefix_record" in selection and sha(selection["prefix_record"]["path"]) != selection["prefix_record"]["sha256"]:
        raise ValueError("optional prior prefix record differs")
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or prior["checkpoint_sha256"] != identity(inventory):
        raise ValueError("single source checkpoint differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True,
        text_encoder=None, tokenizer=None)
    counts["model_loads"] += 1
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires deterministic distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False).to("cuda")
    pipe.vae.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    if adapter.model_identity != prior["model_identity"] or dict(adapter.execution) != prior["execution"]:
        raise ValueError("single source native model/execution differs")
    inputs, store = LocalStore(config["input_store"]), LocalStore(output / "typed-store")
    cuts = {name: inputs.load(identifier, device="cuda") for name, identifier in config["input_cuts"].items()}
    helper["validate_paired_cuts"](cuts["zero"], cuts["text"])
    for cut in cuts.values():
        store.save(cut)
    release = {"native": False, "entries_before_release": 0}

    def guard(module, args):
        if not release["native"]:
            release["entries_before_release"] += 1
            raise ValueError("native single suffix entered before hypothesis seal")

    guards = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    array_index = {}

    def array(value):
        wall()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite observed single array")
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in array_index:
            path = output / "arrays" / (key + ".npy")
            path.parent.mkdir(exist_ok=True)
            np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)
            array_index[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "descriptor": descriptor,
                "stored_dtype": "float32", "stored_shape": list(value.shape), "native_stride": list(value.stride())}
        return array_index[key]

    block = pipe.transformer.single_transformer_blocks[0]
    source = {"schema": "mdb-flux-single-kv-source-v1", "model_identity": adapter.model_identity,
        "checkpoint_sha256": identity(inventory), "execution": dict(adapter.execution), "source_selection": selection,
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"], "input_cuts": config["input_cuts"],
        "input_arrays": {name: {slot: array(cut.payload[slot]) for slot in ("latent", "conditioning", "image", "text")} for name, cut in cuts.items()},
        "source_parameters": {name: tensor_descriptor(value) for name, value in block.named_parameters()},
        "prior_execution": prior["execution"], "prior_implementation_sources": prior["implementation_sources"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "implementation_sources": {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
            (type(pipe), type(pipe.transformer), type(block), type(block.attn.processor), type(pipe.vae), type(pipe.scheduler), type(adapter))},
        "branches": config["branches"], "preseal_counts": dict(counts), "preseal_native_entries": release["entries_before_release"],
        "retention": "Image-only MLP preactivation/activation; full native fused attention+MLP projection return; no full fused input projection or weights retained.",
        "attention_forecast_plan": {"query_indices": ATTENTION_QUERY_INDICES,
            "allowed_observations": "two native zero/text direct calls before the second seal; no prefix/edited calls",
            "novel_conditional_targets": ["K", "V"], "represented_consistency_control": "KV",
            "scope": "Source-reviewed observed-QKV-only NumPy dataflow, not hermetic isolation or independent MLP/RGB prediction"},
        "scope": config["scope"]}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
            "allowed_prefix": "paired own-history pre-single0 cuts hydrated; no new native module/attention/scheduler/VAE call before seal"},
        execution_scope="CUDA BF16 own-history parallel single0 native suffix; no encoder/noise; external public Navigator and direct-candidate ledgers",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("single_kv_hypotheses", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    rows, prefixes, post_cuts, candidates, values, controls, attention_summary = [], {}, {}, {}, {}, {}, {}

    def receipt(value):
        store.receipt(value)
        body = value.to_dict()
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "branch": active["branch"],
                "receipt": value.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"] or counts["acts"] > config["max_acts"]:
            raise RuntimeError("finite single transition/Act budget exhausted")

    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def calculate(name, cut, kv=None, selected="native"):
        if counts["candidate_calls"] >= config["max_candidate_calls"]:
            raise RuntimeError("finite single candidate-call budget exhausted")
        counts["candidate_calls"] += 1
        event("candidate_start", candidate=name, input_parent=cut.fingerprint)
        value = helper["capture_candidate"](block, cut, source=kv, selection=selected)
        candidates[name] = {key: value[key] for key in ("input_parent", "source_parent", "selection", "text_rows", "dispatch_calls", "equation_exact", "partition_controls", "dispatch_profile")}
        candidates[name].update(ports={key: array(tensor) for key, tensor in value["ports"].items()},
            observed_qkv={key: array(tensor) for key, tensor in value["observed_qkv"].items()},
            used_qkv={key: array(tensor) for key, tensor in value["used_qkv"].items()}, attention_dispatch_output=array(value["attention_dispatch_output"]))
        event("candidate_complete", candidate=name, counts=dict(counts))
        values[name] = value

    def finish(n, row):
        continuation = []
        while n.point["logical_step"] < 4:
            wall()
            continuation.append(n.step().to_dict())
        final = n.capture()
        store.save(final)
        if describe(final.payload["conditioning"]) != describe(cuts["zero"].payload["conditioning"]):
            raise ValueError("single route changed stored C0")
        if counts["images"] >= config["max_images"] or counts["vae_calls"] >= config["max_vae_calls"]:
            raise RuntimeError("finite single image/VAE budget exhausted")
        counts["vae_calls"] += 1
        image = adapter.decode(n.session, height=512, width=512)
        path = output / (row["id"] + ".png")
        image.save(path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint, final_latent=array(final.payload["latent"]),
            final_conditioning=tensor_descriptor(final.payload["conditioning"]), continuation_receipts=continuation,
            image={"path": path.name, "sha256": sha(path)})
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    with tx:
        event("hypotheses_sealed", seal=seal, counts=dict(counts), native_entries_before_release=release["entries_before_release"])
        release["native"] = True

        def execute(_):
            for name in ("zero", "text"):
                calculate(name, cuts[name])
            native_text = values["text"]
            kv_source = {"parent": cuts["text"].fingerprint, "text_rows": native_text["text_rows"], **native_text["observed_qkv"]}
            binding = {"native_candidates": {name: {"input_parent": cuts[name].fingerprint,
                "qkv": candidates[name]["observed_qkv"], "dispatch_profile": candidates[name]["dispatch_profile"]} for name in ("zero", "text")},
                "query_indices": ATTENTION_QUERY_INDICES, "text_rows": native_text["text_rows"],
                "main_seal_sha256": seal["seal_sha256"], "preseal_counts": dict(counts),
                "scope": "Only observed source QKV operands enter the numerical forecast; native outputs remain resident, so this is reviewed dataflow rather than hermetic isolation."}
            profiles = [value["dispatch_profile"] for value in (values["zero"], values["text"])]
            supported = all(profile == {"extra_positional_arguments": 0, "attn_mask_present": False, "dropout_p": 0.,
                "is_causal": False, "scale": None, "enable_gqa": False, "attention_kwargs_present": False,
                "backend": None, "parallel_config_present": False} for profile in profiles)
            qkv_numpy = {name: {key: tensor.detach().float().cpu().numpy().copy() for key, tensor in values[name]["observed_qkv"].items()}
                for name in ("zero", "text")}
            attention_tx = ForecastTransaction(output / "attention-transaction", model_identity=adapter.model_identity,
                input_identity=identity(binding), source_identity=identity(request["source_hashes"]),
                frontier={"kind": "architectural", "source_binding_sha256": identity(binding)},
                execution_scope="Independent NumPy FP64 sampled attention from two observed CUDA BF16 native QKV sources; no model calls in prediction",
                budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
            with attention_tx:
                attention_tx.frontier("source_binding", binding)
                attention_tx.predict("sampled_attention", lambda context: {"predictions": helper["forecast_attention_samples"](
                    qkv_numpy["zero"], qkv_numpy["text"], text_rows=native_text["text_rows"], query_indices=ATTENTION_QUERY_INDICES)
                    if supported else {"instrument_status": "not-applicable", "query_indices": ATTENTION_QUERY_INDICES,
                        "cases": {}, "reason": "native dispatcher uses an unsupported mask/dropout/scale/cache/backend profile; main native panel retained"}})
                attention_seal = attention_tx.seal()
                event("attention_forecast_sealed", seal=attention_seal, counts=dict(counts), source_binding_sha256=identity(binding))

                def edited(context):
                    for name in ("K", "V", "KV"):
                        calculate(name, cuts["zero"], kv_source, name)
                    forecast = context.predictions["sampled_attention"]
                    samples = {}
                    if forecast["instrument_status"] == "applicable":
                        for name in ("K", "V", "KV"):
                            actual = values[name]["attention_dispatch_output"][:, [native_text["text_rows"] + index for index in ATTENTION_QUERY_INDICES]].detach().float().cpu().numpy()
                            comparisons = {}
                            for label, key in (("fp64", "attention"), ("cast_bf16", "attention_bf16")):
                                delta = actual.astype(np.float64) - forecast["cases"][name][key].astype(np.float64)
                                comparisons[label] = {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
                                    "signed_mean": float(delta.mean()), "l2": float(np.linalg.norm(delta.ravel()))}
                            samples[name] = {"attention": actual, "actual_native_dtype": str(values[name]["attention_dispatch_output"].dtype),
                                "source_dispatch_array": candidates[name]["attention_dispatch_output"], "comparisons": comparisons,
                                "actual_rms": float(np.sqrt(np.mean(actual.astype(np.float64) ** 2))), "text_mass_measured": False}
                    return {"instrument_status": forecast["instrument_status"], "samples": samples, "counts": dict(counts)}

                attention_tx.execute("edited_candidates", edited)
            save(output / "attention-transaction-report.json", attention_tx.report())
            attention_verification = verify_transaction(output / "attention-transaction", expected_seal_sha256=attention_seal["seal_sha256"])
            observed_attention = attention_tx.observation("edited_candidates")
            attention_summary.update(seal=attention_seal, verification=attention_verification, source_binding=binding,
                instrument_status=observed_attention["instrument_status"],
                comparisons={name: {"comparisons": value["comparisons"], "actual_rms": value["actual_rms"]}
                    for name, value in observed_attention["samples"].items()}, text_mass_measured=False,
                scope="K/V-only novel conditional forecasts; KV represented native-source consistency control; no error gates or independent MLP/RGB forecast")
            save(output / "attention-forecast.json", attention_summary)
            for name in ("zero", "text"):
                active["branch"] = "prefix." + name
                n = navigator(cuts[name])
                native_receipt = n.step()
                post = n.capture()
                store.save(post)
                post_cuts[name] = post
                prefixes[name] = {"input_cut": cuts[name].fingerprint, "post_single0": post.fingerprint, "receipt": native_receipt.to_dict()}
                for slot in ("image", "text"):
                    controls[name + "_candidate_" + slot + "_exact_native"] = torch.equal(values[name][slot], post.payload[slot])
            for name in values:
                controls[name + "_image_mlp_preactivation_exact_zero"] = torch.equal(values[name]["ports"]["image_mlp_preactivation"], values["zero"]["ports"]["image_mlp_preactivation"])
                controls[name + "_image_mlp_activation_exact_zero"] = torch.equal(values[name]["ports"]["image_mlp_activation"], values["zero"]["ports"]["image_mlp_activation"])
            controls["KV_image_exact_text_candidate"] = torch.equal(values["KV"]["image"], values["text"]["image"])
            controls["KV_image_exact_text_native"] = torch.equal(values["KV"]["image"], post_cuts["text"].payload["image"])
            save(output / "prefixes.json", {"schema": "mdb-flux-single-kv-prefixes-v1", "prefixes": prefixes})
            save(output / "candidates.json", {"schema": "mdb-flux-single-kv-candidates-v1", "candidates": candidates})
            for spec in config["branches"]:
                active["branch"] = spec["id"]
                reference = "text" if spec["id"] == "text" else "zero"
                parent = post_cuts[reference]
                n = navigator(parent)
                row = {**spec, "parent": parent.fingerprint, "prefix_source": reference, "start_point": n.point,
                    "acts": [], "continuation_policy": "stored-C0-no-further-projection"}
                event("branch_start", row=spec["id"], parent=parent.fingerprint)
                if spec["kind"] != "reference":
                    for slot, field in (("image", "image_source"), ("text", "text_source")):
                        selected = spec[field]
                        if selected is not None:
                            tensor = post_cuts[selected].payload[slot] if selected in post_cuts else values[selected][slot]
                            row["acts"].append(helper["install_port"](n, slot, tensor, name=spec["id"] + "." + slot, store=store))
                finish(n, row)
            expected = {"adapter_steps": 191, "acts": 10, "candidate_calls": 5, "images": 9, "vae_calls": 9,
                "prompt_encodings": 0, "noise_draws": 0}
            if any(counts[key] != value for key, value in expected.items()):
                raise ValueError("completed single-KV physical census differs")
            by_id = {row["id"]: row for row in rows}
            controls.update(zero_png_equals_prior=by_id["zero"]["image"]["sha256"] == selection["expected_pngs"]["zero"],
                text_png_equals_prior=by_id["text"]["image"]["sha256"] == selection["expected_pngs"]["text"],
                sham_png_equals_zero=by_id["sham"]["image"]["sha256"] == by_id["zero"]["image"]["sha256"],
                coherent_png_equals_text=by_id["coherent"]["image"]["sha256"] == by_id["text"]["image"]["sha256"])
            return {"rows": rows, "counts": dict(counts), "prefixes": prefixes, "candidates": candidates,
                "control_observations": controls, "source_sha256": sha(output / "source.json"), "RGB_forecast": False,
                "numerical_attention_forecast": attention_summary}

        tx.execute("native", execute)
    for handle in guards:
        handle.remove()
    save(output / "transaction-report.json", tx.report())
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    summary = {"schema": "mdb-flux-single-kv-result-v1", "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "model_identity": adapter.model_identity, "counts": counts, "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed",
        "seal": seal, "transaction_verification": verification, "worker_seconds": time.monotonic() - started,
        "control_observations": controls, "RGB_forecast": False, "scope": config["scope"]}
    summary["numerical_attention_forecast"] = attention_summary
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    event("archive_start", raw_output_bytes_before_event=sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        scope="all retained files before this archive event/final manifest; no full fused input projection or weights retained")
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    emit_report(output, summary, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
        compression_level=config["archive_compression_level"])


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-single-kv-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-single-kv-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
            emit_report(output, failure, max_compressed_bytes=config.get("max_archive_bytes", 512 << 20),
                compression_level=config.get("archive_compression_level", 9))
        raise
