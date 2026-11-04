"""One-resident imported single6 A/M contraction panel, native consumer only."""
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

QUERIES = [0, 31, 32, 255, 511, 512, 767, 1023]
OUTPUT_DIMS = list(range(0, 3072, 96))


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
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
    if (config.get("single_index") != 6 or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 6}
            or config.get("input_store") != "input-store" or config.get("source_selection") != "source-selection.json"
            or set(config.get("input_cuts", {})) != {"zero", "text"} or config.get("query_indices") != QUERIES
            or config.get("output_dims") != OUTPUT_DIMS or config.get("height") != 512 or config.get("width") != 512 or config.get("steps") != 4):
        raise ValueError("single6 contraction scientific geometry differs")
    recipes = [("zero", "zero", "zero"), ("text", "text", "text"), ("sham", "zero", "zero"),
        ("A.image", "A", None), ("M.image", "M", None), ("AM.image", "AM", None),
        ("text.text", None, "text"), ("coherent", "text", "text"), ("AM.text", "AM", "text")]
    if config.get("branches") != [{"id": name, "kind": "reference" if name in {"zero", "text"} else "edited",
            "image_source": image, "text_source": text} for name, image, text in recipes]:
        raise ValueError("single6 contraction nine-arm recipes differ")
    for name, lo, hi in (("max_adapter_steps", 143, 143), ("max_acts", 10, 10), ("max_candidate_calls", 5, 5),
            ("max_images", 9, 9), ("max_vae_calls", 9, 9), ("max_wall_seconds", 1, 600),
            ("cpu_threads", 1, 16), ("archive_compression_level", 1, 9)):
        if type(config.get(name)) is not int or not lo <= config[name] <= hi:
            raise ValueError("invalid finite contraction budget: " + name)


def verify_selection(config):
    selection = json.loads(Path(config["source_selection"]).read_text())
    if (selection["schema"] != "mdb-flux-single-contraction-selection-v1" or selection["prior_job_id"] != "job-b8e38f8a326f"
            or selection["context_id"] != "p1-s53" or selection["input_cuts"] != config["input_cuts"]
            or any(selection[key] != config[key] for key in ("expected_pngs", "expected_final_latents"))):
        raise ValueError("contraction source occurrence/reference differs")
    records = {key: selection[key] for key in ("source_record", "audit_record", "prior_request", "prior_manifest", "prior_report")}
    records.update({"row." + role: record for role, record in selection["prior_rows"].items()})
    loaded = {}
    for key, record in records.items():
        path = Path(record["path"])
        if path.is_symlink() or not path.resolve().is_relative_to(Path.cwd()) or sha(path) != record["sha256"]:
            raise ValueError("contraction selected metadata path/hash differs: " + key)
        loaded[key] = json.loads(path.read_text())
    prior, audit, request, manifest, report = (loaded[key] for key in ("source_record", "audit_record", "prior_request", "prior_manifest", "prior_report"))
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or request["request_sha256"] != selection["request_sha256"] or prior["request_sha256"] != selection["request_sha256"]
            or audit["mechanics_status"] != "valid" or audit["source"]["job_id"] != selection["prior_job_id"]
            or audit["source"]["archive_sha256"] != selection["archive_sha256"] or audit["source"]["request_sha256"] != selection["request_sha256"]
            or report["job_id"] != selection["prior_job_id"] or report["archive_sha256"] != selection["archive_sha256"]
            or report["request_sha256"] != selection["request_sha256"] or report["files_manifest_sha256"] != records["prior_manifest"]["sha256"]):
        raise ValueError("contraction prior request/audit/archive binding differs")
    for target, record in selection["source_files"].items():
        if sha(target) != record["sha256"] or Path(target).stat().st_size != record["bytes"] or manifest[record["prior_path"]] != record["sha256"]:
            raise ValueError("contraction staged metadata differs from producer manifest")
    for path, digest in selection["files"].items():
        item = Path(path)
        if item.is_symlink() or not item.resolve().is_relative_to(Path(config["input_store"]).resolve()) or sha(item) != digest:
            raise ValueError("contraction imported CAS path/hash differs")
    for role in ("zero", "text"):
        row = loaded["row." + role]
        if row["id"] != role or row["image"]["sha256"] != selection["expected_pngs"][role] or row["final_latent"]["descriptor"] != selection["expected_final_latents"][role]:
            raise ValueError("contraction prior reference row differs")
        trajectory = row["reference_trajectory"]
        imported = next(entry for entry in trajectory if entry["point"]["phase"] == "single" and entry["point"]["index"] == 3)
        if imported["cut"] != selection["input_cuts"][role] or not imported["retained_full_cut"]:
            raise ValueError("contraction imported source is not the retained post-single2 occurrence")
        for key, index in (("expected_frontiers", 6), ("expected_outputs", 7)):
            entry = next(item for item in trajectory if item["point"]["phase"] == "single" and item["point"]["index"] == index)
            expected = {name: entry[name] for name in ("cut", "point", "receipt_id")}
            expected["arrays"] = {slot: record["descriptor"] for slot, record in entry["arrays"].items()}
            if selection[key][role] != expected:
                raise ValueError("contraction selected replay descriptor is not the exact prior trajectory occurrence")
    return selection, prior


def validate_native_sources(prior, current):
    def names(value):
        result = {}
        for path, digest in value.items():
            name = Path(path).name
            if name in result:
                raise ValueError("ambiguous native implementation basename")
            result[name] = digest
        return result
    old, new = names(prior), names(current)
    if not new or any(old.get(key) != value for key, value in new.items()):
        raise ValueError("contraction native implementation changed from prior history")
    return new


def run_panel(*, pipe, adapter, inputs, output, config, request, metadata, selection, prepared_inputs=False):
    """Injectable resident-model panel; no loading, encoding, random draws or jobs."""
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    if type(prepared_inputs) is not bool or config.get("array_storage", "float32") not in {"float32", "bf16-bits"}:
        raise ValueError("invalid optional contraction input/storage profile")
    helper = runpy.run_path("flux_single_contraction.py")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "plan.json").exists():
        save(output / "plan.json", config)
    elif json.loads((output / "plan.json").read_text()) != config:
        raise ValueError("prepared contraction plan differs")
    store, array_index = LocalStore(output / "typed-store"), {}
    counts = {"model_loads": 0, "prompt_encodings": 0, "noise_draws": 0, "physical_batch": 1,
        "adapter_steps": 0, "acts": 0, "candidate_calls": 0, "images": 0, "vae_calls": 0, "receipt_emissions": 0}
    started, active, sequence = time.monotonic(), {"branch": "replay"}, 0
    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite contraction panel wall exhausted")
    def event(kind, **fields):
        nonlocal sequence
        body = json_value({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_SINGLE_CONTRACTION_PHASE " + json.dumps(body, sort_keys=True), flush=True)
    def receipt(value):
        body = value.to_dict()
        store.receipt(value)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"], "branch": active["branch"], "receipt": value.fingerprint}, sort_keys=True) + "\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if counts["adapter_steps"] > config["max_adapter_steps"] or counts["acts"] > config["max_acts"]:
            raise RuntimeError("finite contraction receipt budget exhausted")
    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)
    def array(value):
        wall()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite contraction observed array")
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in array_index:
            path = output / "arrays" / (key + ".npy")
            path.parent.mkdir(exist_ok=True)
            if config.get("array_storage", "float32") == "bf16-bits":
                from mdb.tensor_archive import write_tensor_archive
                array_index[key] = write_tensor_archive(path, value, root=output)
            else:
                np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)
                array_index[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "descriptor": descriptor,
                    "stored_dtype": "float32", "stored_shape": list(value.shape), "native_stride": list(value.stride())}
        return array_index[key]
    def frontier_check(cut, expected):
        if cut.fingerprint != expected["cut"] or dict(cut.execution_point) != expected["point"]:
            raise ValueError("source replay complete cut/point differs from predecessor")
        if any(tensor_descriptor(cut.payload[slot]) != descriptor for slot, descriptor in expected["arrays"].items()):
            raise ValueError("source replay live-array descriptor differs from predecessor")
    cuts, replay = {}, {}
    for role, cut in inputs.items():
        point = cut.execution_point
        start = 6 if prepared_inputs else 3
        if any(point[key] != value for key, value in {"logical_step": 3, "phase": "single", "index": start, "local_clock": start}.items()):
            raise ValueError("source replay requires retained post-single2 cut")
        store.save(cut)
        if prepared_inputs:
            cuts[role] = cut
            replay[role] = {"input_cut": cut.fingerprint, "pre6_cut": cut.fingerprint, "receipts": []}
            continue
        active["branch"] = "replay." + role
        n = navigator(cut)
        path = [n.step().to_dict() for _ in range(3)]
        current = n.capture()
        frontier_check(current, selection["expected_frontiers"][role])
        store.save(current)
        cuts[role] = current
        replay[role] = {"input_cut": cut.fingerprint, "pre6_cut": current.fingerprint, "receipts": path,
            "complete_cut_exact_prior": True, "live_descriptors_exact_prior": True}
    if prepared_inputs:
        low, high = cuts["zero"], cuts["text"]
        if (low.model_identity != high.model_identity or low.execution != high.execution or set(low.payload) != set(high.payload)
                or any(describe(value) != describe(high.payload[key]) for key, value in low.payload.items() if key not in {"image", "text"})):
            raise ValueError("prepared sources differ in identity/schema or outside image/text")
        for key in ("image", "text"):
            a, b = low.payload[key], high.payload[key]
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError("prepared source geometry/dtype/device/finite differs")
        save(output / "source-replay.json", {"kind": "prepared-inputs", "inputs": replay,
            "counts": {"adapter_steps": 0, "acts": 0, "receipt_emissions": 0}})
    else:
        helper["validate_sources"](cuts["zero"], cuts["text"])
        save(output / "source-replay.json", replay)
    block = pipe.transformer.single_transformer_blocks[6]
    if not isinstance(block.attn.to_out, torch.nn.Linear):
        raise ValueError("single6 native projection is not the qualified full linear module")
    source = {**metadata, "schema": "mdb-flux-single-contraction-source-v1", "job_id": request["job_id"],
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"], "source_selection": selection,
        "model_identity": adapter.model_identity, "execution": dict(adapter.execution),
        "input_cuts": {role: cut.fingerprint for role, cut in cuts.items()}, "imported_cuts": {role: cut.fingerprint for role, cut in inputs.items()},
        "input_arrays": {role: {slot: array(cut.payload[slot]) for slot in ("latent", "conditioning", "image", "text")} for role, cut in cuts.items()},
        "source_parameters": {name: tensor_descriptor(value) for name, value in block.named_parameters()},
        "linear_path": "single_transformer_blocks.6.attn.to_out", "preseal_counts": dict(counts),
        "query_indices": config["query_indices"], "output_dims": config["output_dims"], "branches": config["branches"], "scope": config["scope"]}
    if prepared_inputs:
        source.pop("imported_cuts")
        source.update(prepared_inputs=True, prepared_input_cuts={role: cut.fingerprint for role, cut in inputs.items()},
            source_contrast_observations={key + "_exact": torch.equal(cuts["zero"].payload[key], cuts["text"].payload[key]) for key in ("image", "text")})
    if config.get("array_storage", "float32") != "float32":
        source["array_storage"] = config["array_storage"]
    save(output / "source.json", source)
    release = {"native": False, "blocked": 0}
    def guard(module, args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("single6 native suffix entered before main seal")
    guards = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    rows, candidates, values, prefixes, post, controls = [], {}, {}, {}, {}, {}
    def calculate(name, cut, source=None, selected="native"):
        wall()
        if counts["candidate_calls"] >= config["max_candidate_calls"]:
            raise RuntimeError("finite contraction direct-call budget exhausted")
        event("candidate_start", candidate=name, input_parent=cut.fingerprint)
        counts["candidate_calls"] += 1
        value = helper["capture_candidate"](block, cut, source=source, selection=selected)
        candidates[name] = {key: value[key] for key in ("input_parent", "source_parent", "selection", "text_rows", "attention_width", "equation_exact", "partition_controls")}
        candidates[name]["ports"] = {name: array(tensor) for name, tensor in value["ports"].items()}
        values[name] = value
        event("candidate_complete", candidate=name)
    def finish(n, row):
        path = []
        while n.point["logical_step"] < 4:
            wall()
            path.append(n.step().to_dict())
        final = n.capture()
        store.save(final)
        if describe(final.payload["conditioning"]) != describe(cuts["zero"].payload["conditioning"]):
            raise ValueError("contraction suffix changed stored conditioning")
        if counts["vae_calls"] >= config["max_vae_calls"] or counts["images"] >= config["max_images"]:
            raise RuntimeError("finite contraction VAE/image budget exhausted")
        counts["vae_calls"] += 1
        image = adapter.decode(n.session, height=512, width=512)
        file = output / (row["id"] + ".png")
        image.save(file)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint, final_latent=array(final.payload["latent"]), final_conditioning=tensor_descriptor(final.payload["conditioning"]),
            continuation_receipts=path, image={"path": file.name, "sha256": sha(file)})
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"])
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(source["input_cuts"]),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
            "allowed_prefix": ("Externally prepared observed current-step pre6 cuts; zero panel source replay; earlier-timestep module6 calls allowed in the declared fresh builder, no current-step module6 call before the outer early-MLP seal"
                if prepared_inputs else "Six native replay steps on authenticated own-history predecessor cuts; known predecessor references already executed; no direct current-site captures before this seal")},
        execution_scope="CUDA BF16 native single6 conditional operand assay; external receipt/direct-call census authoritative",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    projection_summary = {}
    try:
        with tx:
            tx.predict("contraction_hypotheses", lambda _: {"predictions": config["hypotheses"]})
            main_seal = tx.seal()
            event("hypotheses_sealed", seal=main_seal, blocked_native_entries=release["blocked"])
            release["native"] = True
            def execute(_):
                for role in ("zero", "text"):
                    calculate(role, cuts[role])
                high = values["text"]
                donor = {"parent": cuts["text"].fingerprint, "concat": high["ports"]["original_concat"],
                    "text_rows": high["text_rows"], "attention_width": high["attention_width"]}
                with torch.inference_mode():
                    indices = torch.tensor(config["output_dims"], dtype=torch.long, device=block.attn.to_out.weight.device)
                    weight = block.attn.to_out.weight.index_select(0, indices)
                    bias = block.attn.to_out.bias.index_select(0, indices) if block.attn.to_out.bias is not None else None
                binding = {"source_candidates": {role: {"input_parent": cuts[role].fingerprint, "ports": candidates[role]["ports"]} for role in ("zero", "text")},
                    "linear_path": source["linear_path"], "weight_rows": array(weight), "bias_rows": array(bias) if bias is not None else None,
                    "weight_parameter": source["source_parameters"]["attn.to_out.weight"],
                    "bias_parameter": source["source_parameters"].get("attn.to_out.bias"),
                    "row_extraction": "Actual native BF16 parameter.index_select(0, output_indices), losslessly widened FP32 for owned NumPy arithmetic; token-ID unrelated output-channel row order is explicit",
                    "query_indices": config["query_indices"], "output_indices": config["output_dims"], "text_rows": high["text_rows"],
                    "attention_width": high["attention_width"], "counts_before_seal": dict(counts), "main_seal_sha256": main_seal["seal_sha256"]}
                native_numpy = {role: {"concat": values[role]["ports"]["original_concat"].float().cpu().numpy(),
                    "residual": values[role]["ports"]["residual"].float().cpu().numpy(), "gate": values[role]["ports"]["gate"].float().cpu().numpy()} for role in ("zero", "text")}
                weight_numpy = np.array(weight.float().cpu().numpy(), copy=True)
                bias_numpy = np.array(bias.float().cpu().numpy(), copy=True) if bias is not None else None
                for value in [weight_numpy, *([bias_numpy] if bias_numpy is not None else []),
                        *(value for operands in native_numpy.values() for value in operands.values())]:
                    value.setflags(write=False)
                numerical = ForecastTransaction(output / "projection-transaction", model_identity=adapter.model_identity,
                    input_identity=identity(binding), source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": binding},
                    execution_scope="NumPy FP64 sampled fused projection/write conditional on two observed native source operands and fixed actual weight rows",
                    budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
                with numerical:
                    numerical.frontier("source_binding", binding)
                    numerical.predict("sampled_projection", lambda _: {"predictions": helper["forecast_samples"](
                        native_numpy["zero"], native_numpy["text"], weight_numpy, bias_numpy,
                        query_indices=config["query_indices"], output_indices=config["output_dims"], text_rows=high["text_rows"], attention_width=high["attention_width"])})
                    numeric_seal = numerical.seal()
                    event("projection_forecast_sealed", seal=numeric_seal, binding_sha256=identity(binding))
                    def edited(context):
                        for name in ("A", "M", "AM"):
                            calculate(name, cuts["zero"], donor, name)
                        samples = {}
                        positions = [high["text_rows"] + index for index in config["query_indices"]]
                        for name in ("A", "M", "AM"):
                            actual = {key: values[name]["ports"][port][:, positions][:, :, config["output_dims"]].float().cpu().numpy()
                                for key, port in (("projection", "projection"), ("updated", "updated"))}
                            errors = {}
                            for key in ("projection_fp64", "projection_bf16", "updated_fp64", "updated_bf16"):
                                delta = actual[key.split("_")[0]].astype(np.float64) - context.predictions["sampled_projection"]["cases"][name][key]
                                errors[key] = {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))), "l2": float(np.linalg.norm(delta.ravel())), "signed_mean": float(delta.mean())}
                            samples[name] = {**actual, "errors_actual_minus_prediction": errors}
                        return {"samples": samples, "counts": dict(counts)}
                    numerical.execute("edited_candidates", edited)
                save(output / "projection-transaction-report.json", numerical.report())
                projection_summary.update(seal=numeric_seal, verification=verify_transaction(numerical.root), binding=binding,
                    samples={name: {"errors_actual_minus_prediction": sample["errors_actual_minus_prediction"],
                        "sample_shape": list(sample["projection"].shape), "actual_native_dtype": str(values[name]["ports"]["projection"].dtype),
                        **{key: array(torch.from_numpy(np.array(sample[key], copy=True))) for key in ("projection", "updated")}}
                        for name, sample in numerical.observation("edited_candidates")["samples"].items()})
                save(output / "projection-forecast.json", projection_summary)
                for role in ("zero", "text"):
                    active["branch"] = "prefix." + role
                    n = navigator(cuts[role])
                    value = n.step()
                    current = n.capture()
                    store.save(current)
                    post[role] = current
                    prefixes[role] = {"input_cut": cuts[role].fingerprint, "post_single6": current.fingerprint, "receipt": value.to_dict()}
                    if not prepared_inputs:
                        controls[role + "_prefix_cut_equals_prior"] = current.fingerprint == selection["expected_outputs"][role]["cut"]
                    for slot in ("image", "text"):
                        controls[role + "_candidate_" + slot + "_exact_native"] = torch.equal(values[role][slot], current.payload[slot])
                for name in ("A", "M", "AM"):
                    controls[name + "_text_exact_zero_candidate"] = torch.equal(values[name]["text"], values["zero"]["text"])
                controls["AM_projection_image_exact_text_candidate"] = torch.equal(values["AM"]["ports"]["projection"][:, high["text_rows"]:], high["ports"]["projection"][:, high["text_rows"]:])
                save(output / "prefixes.json", {"prefixes": prefixes})
                save(output / "candidates.json", {"candidates": candidates})
                for spec in config["branches"]:
                    active["branch"] = spec["id"]
                    role = "text" if spec["id"] == "text" else "zero"
                    n = navigator(post[role])
                    row = {**spec, "parent": post[role].fingerprint, "prefix_source": role, "start_point": n.point,
                        "acts": [], "continuation_policy": "recipient-stored-C-no-further-project"}
                    if spec["kind"] != "reference":
                        for slot, key in (("image", "image_source"), ("text", "text_source")):
                            selected = spec[key]
                            if selected is not None:
                                tensor = post[selected].payload[slot] if selected in post else values[selected][slot]
                                row["acts"].append(helper["install_port"](n, slot, tensor, name=spec["id"] + "." + slot, store=store))
                    finish(n, row)
                by_id = {row["id"]: row for row in rows}
                controls.update(sham_png_equals_zero=by_id["sham"]["image"]["sha256"] == by_id["zero"]["image"]["sha256"],
                    coherent_png_equals_text=by_id["coherent"]["image"]["sha256"] == by_id["text"]["image"]["sha256"])
                if not prepared_inputs:
                    for role in ("zero", "text"):
                        controls[role + "_png_equals_prior"] = by_id[role]["image"]["sha256"] == selection["expected_pngs"][role]
                        controls[role + "_latent_equals_prior"] = by_id[role]["final_latent"]["descriptor"] == selection["expected_final_latents"][role]
                return {"rows": rows, "counts": dict(counts), "control_observations": controls, "source_sha256": sha(output / "source.json")}
            tx.execute("native", execute)
    finally:
        for handle in guards:
            handle.remove()
    expected = {"adapter_steps": 137 if prepared_inputs else 143, "acts": 10, "candidate_calls": 5, "images": 9, "vae_calls": 9,
        "receipt_emissions": 147 if prepared_inputs else 153}
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError("single6 completed physical census differs")
    save(output / "transaction-report.json", tx.report())
    summary = {"schema": "mdb-flux-single-contraction-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "counts": counts, "seal": main_seal, "transaction_verification": verify_transaction(tx.root),
        "projection_forecast_seal": projection_summary["seal"], "control_observations": controls,
        "model_identity": adapter.model_identity, "worker_seconds": time.monotonic() - started,
        "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete")
    return json_value(summary)


def emit(output, report, config):
    manifest = output / "files.json"
    if manifest.exists():
        manifest.unlink()
    save(manifest, {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    runpy.run_path("flux_family_custody.py")["emit"](output, json_value(report), limits=config.get("transport_limits"), compression_level=config["archive_compression_level"])


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("contraction request/source bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-single-contraction-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"plan_sha256": sha(output / "plan.json"), "request_sha256": request["request_sha256"],
        "model_loads": 0, "adapter_steps": 0, "candidate_calls": 0, "scope": "Fixed imported own-history single6 conditional assay, no globally novel reference suffix claim"})
    selection, prior = verify_selection(config)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    started = time.monotonic()
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("contraction assay requires admitted CUDA")
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or prior["checkpoint_sha256"] != identity(inventory):
        raise ValueError("contraction checkpoint differs from source history")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True, text_encoder=None, tokenizer=None)
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires qualified deterministic distilled five-joint/twenty-single profile")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False).to("cuda")
    pipe.vae.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
        (type(pipe), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[6]),
         type(pipe.transformer.single_transformer_blocks[6].attn.processor), type(pipe.vae), type(pipe.scheduler), type(adapter))}
    binding = validate_native_sources(prior["implementation_sources"], implementations)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}
    if adapter.model_identity != prior["model_identity"] or dict(adapter.execution) != prior["execution"] or versions != prior["versions"]:
        raise ValueError("contraction native model/execution/packages differ")
    inputs = {role: LocalStore(config["input_store"]).load(identifier, device="cuda") for role, identifier in config["input_cuts"].items()}
    summary = run_panel(pipe=pipe, adapter=adapter, inputs=inputs, output=output, config=config, request=request, selection=selection,
        metadata={"checkpoint_sha256": identity(inventory), "context_id": selection["context_id"], "versions": versions,
            "implementation_sources": implementations, "native_source_binding": binding,
            "source_roles": {"zero": "low own-history trajectory", "text": "high own-history trajectory; both I and T differ at pre6"},
            "outer_preseal_counts": {"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0},
            "timing_scope": "max_wall_seconds bounds panel replay/capture/futures; outer lease additionally bounds load and export"})
    outer = {**summary, "counts": {**summary["counts"], "model_loads": 1}, "worker_seconds": time.monotonic() - started,
        "panel_summary_sha256": sha(output / "summary.json")}
    save(output / "outer-summary.json", outer)
    print("MDB_FLUX_SINGLE_CONTRACTION_ARCHIVE_START " + json.dumps({"raw_output_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()), "counts": outer["counts"]}), flush=True)
    emit(output, outer, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-single-contraction-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-single-contraction-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            emit(output, failure, request["config"])
        raise
