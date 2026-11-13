"""Reusable resident-model matched single0 panel and fresh source builder."""
from __future__ import annotations

import hashlib
import json
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def replace_slot(nav, slot, value, *, name, store):
    """One public typed replacement preserving the exact slot layout/closure."""
    import torch
    from saturn_pub import Act
    from saturn_pub.values import describe

    before = nav.capture()
    reference = before.payload[slot]
    if (value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device
            or not bool(torch.isfinite(value).all())):
        raise ValueError("builder replacement geometry/dtype/device/finite contract differs")
    with torch.inference_mode():
        installed = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
        installed.copy_(value)
    store.save(before)
    act = Act.replace(slot, installed, name=name)
    receipt = nav.apply(act)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe({**before.payload, slot: installed}):
        raise ValueError("builder Act changed an undeclared slot")
    return {"slot": slot, "point": nav.point, "parent": before.fingerprint, "post_act": after.fingerprint,
        "act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": receipt.fingerprint,
        "undeclared_slots_exact": True}


def build_sources(*, adapter, session, output, context, job_id=None, max_steps=100, wall=lambda: None):
    """Native Z2→zero-C Z3→paired histories→only-working-T transplant.

    The builder owns a separate typed store. Its 96 native transitions and three
    Acts are counted independently from the 191-transition nine-future panel.
    """
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore

    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    store = LocalStore(output / "typed-store")
    counts = {"adapter_steps": 0, "acts": 0, "receipt_emissions": 0}
    emissions = []
    def receipt(value):
        body = value.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        counts["receipt_emissions"] += 1
        if counts["adapter_steps"] > max_steps or counts["acts"] > 3:
            raise RuntimeError("finite source-builder budget exhausted")
        store.receipt(value)
        emissions.append({"sequence": len(emissions), "receipt": value.fingerprint})
    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=max_steps, on_receipt=receipt)
    def advance(nav, step, phase, index=None):
        path = []
        while any(nav.point[key] != value for key, value in {"logical_step": step, "phase": phase, "index": index}.items()):
            wall()
            path.append(nav.step().to_dict())
        store.save(nav.capture())
        return path
    initial = session.capture()
    store.save(initial)
    native = navigator(initial)
    paths = {"to_Z2": advance(native, 2, "project")}
    z2 = native.capture()
    own_c = z2.payload["conditioning"]
    acts = {"zero_conditioning": replace_slot(native, "conditioning", torch.zeros_like(own_c), name="builder.zero-C-at-Z2", store=store)}
    zero_z2 = native.capture()
    paths["to_Z3"] = advance(native, 3, "project")
    z3 = native.capture()
    zero, own = navigator(z3), navigator(z3)
    acts["restore_conditioning"] = replace_slot(own, "conditioning", own_c, name="builder.own-C-at-Z3", store=store)
    paths["zero_to_single0"] = advance(zero, 3, "single", 0)
    paths["own_to_single0"] = advance(own, 3, "single", 0)
    zero_cut, own_cut = zero.capture(), own.capture()
    matched = navigator(zero_cut)
    acts["transplant_text"] = replace_slot(matched, "text", own_cut.payload["text"], name="builder.own-history-T-only", store=store)
    text_cut = matched.capture()
    runpy.run_path("flux_single_kv.py")["validate_paired_cuts"](zero_cut, text_cut)
    arrays = {}
    for name, tensor in {"initial_latent": initial.payload["latent"], "own_conditioning": own_c,
            "Z2_latent": z2.payload["latent"], "Z3_latent": z3.payload["latent"]}.items():
        path = output / (name + ".npy")
        np.save(path, tensor.detach().float().cpu().numpy(), allow_pickle=False)
        arrays[name] = {"path": path.name, "sha256": sha(path), "descriptor": tensor_descriptor(tensor),
            "stored_dtype": "float32", "stored_shape": list(tensor.shape), "native_stride": list(tensor.stride())}
    if counts["adapter_steps"] != 96 or counts["acts"] != 3:
        raise ValueError("fresh source-builder census differs from five-joint/twenty-single four-step profile")
    record = {"schema": "mdb-flux-single-context-builder-v1", "context": context, "job_id": job_id,
        "initial": initial.fingerprint, "Z2": z2.fingerprint, "zeroZ2": zero_z2.fingerprint, "Z3": z3.fingerprint,
        "zero_single0": zero_cut.fingerprint, "own_single0": own_cut.fingerprint, "matched_text": text_cut.fingerprint,
        "paths": paths, "acts": acts, "arrays": arrays, "counts": counts,
        "scope": "Own freshly encoded C and seeded initial noise; text-only history transplant into zero-C single0 recipient"}
    save(output / "builder.json", record)
    save(output / "receipt-emissions.json", emissions)
    return {"zero": zero_cut, "text": text_cut}, record


def run_panel(*, pipe, adapter, cuts, output, config, request, source_metadata, output_prepared=False):
    """Run one matched pre-single0 panel on supplied resident native components.

    No model load, prompt encode, random draw or source construction occurs here.
    All per-context counts refer to physical panel work, excluding the builder.
    """
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    output = Path(output)
    if output_prepared:
        if not output.is_dir() or json.loads((output / "plan.json").read_text()) != config or (output / "source.json").exists():
            raise ValueError("prepared panel output/plan differs or already contains a source")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "plan.json", config)
    helper = runpy.run_path("flux_single_kv.py")
    single_index = config.get("single_index", 0)
    helper["validate_paired_cuts"](cuts["zero"], cuts["text"], single_index=single_index)
    queries = config.get("attention_query_indices", ATTENTION_QUERY_INDICES)
    counts = {"model_loads": 0, "physical_batch": 1, "prompt_encodings": 0, "noise_draws": 0,
        "adapter_steps": 0, "acts": 0, "candidate_calls": 0, "receipt_emissions": 0, "images": 0, "vae_calls": 0}
    started, timeline, active = time.monotonic(), [], {"branch": "panel"}
    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite panel wall budget exhausted")
    def event(kind, **fields):
        value = json_value({"sequence": len(timeline), "kind": kind, "seconds": time.monotonic() - started, **fields})
        timeline.append(value)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_SINGLE_CONTEXT_PHASE " + json.dumps(value, sort_keys=True), flush=True)
    store = LocalStore(output / "typed-store")
    for cut in cuts.values():
        store.save(cut)
    release = {"native": False, "entries_before_release": 0}

    def guard(module, args):
        if not release["native"]:
            release["entries_before_release"] += 1
            raise ValueError("native single suffix entered before hypothesis seal")

    guards = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    try:
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

        block = pipe.transformer.single_transformer_blocks[single_index]
        source = {**source_metadata, "schema": "mdb-flux-single-kv-source-v1", "model_identity": adapter.model_identity,
            "execution": dict(adapter.execution), "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
            "input_cuts": {name: cut.fingerprint for name, cut in cuts.items()},
            "input_arrays": {name: {slot: array(cut.payload[slot]) for slot in ("latent", "conditioning", "image", "text")} for name, cut in cuts.items()},
            "source_parameters": {name: tensor_descriptor(value) for name, value in block.named_parameters()},
            "branches": config["branches"], "preseal_counts": dict(counts), "preseal_native_entries": release["entries_before_release"],
            "attention_forecast_plan": {"query_indices": queries,
                "allowed_observations": "two native zero/text direct calls before second seal; no panel prefix/edited calls",
                "novel_conditional_targets": ["K", "V"], "represented_consistency_control": "KV"}, "scope": config["scope"]}
        if single_index:
            source.update(selector_index=single_index, selector={"logical_step": 3, "phase": "single",
                "index": single_index, "local_clock": single_index})
        save(output / "source.json", source)
        tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity({"config": config, "input_cuts": source["input_cuts"]}),
            source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
                "allowed_prefix": ("fresh source builder completed; no panel native call before this structural seal" if single_index == 0
                    else "authenticated imported observed post-single0 cuts hydrated; prior reference suffixes were executed in predecessor; no panel native call before structural seal")},
            execution_scope=("CUDA BF16 matched single0 panel; separately counted fresh builder and encoder/noise; public Navigator/direct-candidate ledgers"
                if single_index == 0 else "CUDA BF16 matched single1 imported-source panel; separately counted model load; public Navigator/direct-candidate ledgers"),
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
            value = helper["capture_candidate"](block, cut, source=kv, selection=selected, single_index=single_index)
            candidates[name] = {key: value[key] for key in ("input_parent", "source_parent", "selection", "text_rows", "dispatch_calls", "equation_exact", "partition_controls", "dispatch_profile")}
            candidates[name].update(ports={key: array(tensor) for key, tensor in value["ports"].items()},
                observed_qkv={key: array(tensor) for key, tensor in value["observed_qkv"].items()},
                used_qkv={key: array(tensor) for key, tensor in value["used_qkv"].items()}, attention_dispatch_output=array(value["attention_dispatch_output"]))
            event("candidate_complete", candidate=name, counts=dict(counts))
            values[name] = value

        def trajectory(n, native_receipt, kind):
            body = native_receipt.to_dict()
            point, arrays = n.point, {}
            selected = point["phase"] in {"readout", "scheduler"} or point["logical_step"] == 4 or (
                point["phase"] == "single" and point["index"] in {2, 3, 10})
            cut = n.session.capture(retain=False) if selected else None
            slots = ("latent",) if point["logical_step"] == 4 else ("image", "text", "latent")
            for slot in slots:
                value = n.session.read(slot)
                if value is not None:
                    full = array(value)
                    wide = value.detach().float().cpu().numpy().astype(np.float64)
                    arrays[slot] = {"descriptor": tensor_descriptor(value), "full_array": full,
                        "rms": float(np.sqrt(np.mean(wide * wide))), "l2": float(np.linalg.norm(wide.ravel()))}
            if cut is not None and cut.payload.get("noise_prediction") is not None:
                value = cut.payload["noise_prediction"]
                wide = value.detach().float().cpu().numpy().astype(np.float64)
                arrays["noise_prediction"] = {"descriptor": tensor_descriptor(value), "full_array": array(value),
                    "rms": float(np.sqrt(np.mean(wide * wide))), "l2": float(np.linalg.norm(wide.ravel()))}
            if selected:
                if cut.fingerprint != body["result"]:
                    raise ValueError("reference trajectory receipt/current cut differs")
                store.save(cut)
            return {"kind": kind, "receipt_id": native_receipt.fingerprint, "cut": body["result"], "point": point,
                "arrays": arrays, "retained_full_cut": selected}

        def finish(n, row):
            continuation = []
            retain = config.get("capture_reference_trajectory", False) and row["kind"] == "reference"
            if retain:
                row["reference_trajectory"] = [trajectory(n, prefix_receipts[row["prefix_source"]], "prefix")]
            while n.point["logical_step"] < 4:
                wall()
                value = n.step()
                continuation.append(value.to_dict())
                if retain:
                    row["reference_trajectory"].append(trajectory(n, value, "continuation"))
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

            prefix_receipts = {}
            def execute(_):
                for name in ("zero", "text"):
                    calculate(name, cuts[name])
                native_text = values["text"]
                kv_source = {"parent": cuts["text"].fingerprint, "text_rows": native_text["text_rows"], **native_text["observed_qkv"]}
                binding = {"native_candidates": {name: {"input_parent": cuts[name].fingerprint,
                    "qkv": candidates[name]["observed_qkv"], "dispatch_profile": candidates[name]["dispatch_profile"]} for name in ("zero", "text")},
                    "query_indices": queries, "text_rows": native_text["text_rows"],
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
                        qkv_numpy["zero"], qkv_numpy["text"], text_rows=native_text["text_rows"], query_indices=queries)
                        if supported else {"instrument_status": "not-applicable", "query_indices": queries,
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
                                actual = values[name]["attention_dispatch_output"][:, [native_text["text_rows"] + index for index in queries]].detach().float().cpu().numpy()
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
                    prefix_receipts[name] = native_receipt
                    post = n.capture()
                    store.save(post)
                    post_cuts[name] = post
                    prefixes[name] = {"input_cut": cuts[name].fingerprint, f"post_single{single_index}": post.fingerprint, "receipt": native_receipt.to_dict()}
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
                                row["acts"].append(helper["install_port"](n, slot, tensor, name=spec["id"] + "." + slot,
                                    store=store, single_index=single_index))
                    finish(n, row)
                expected = {"adapter_steps": 2 + 9 * (21 - single_index), "acts": 10, "candidate_calls": 5, "images": 9, "vae_calls": 9,
                    "prompt_encodings": 0, "noise_draws": 0}
                if any(counts[key] != value for key, value in expected.items()):
                    raise ValueError("completed single-KV physical census differs")
                by_id = {row["id"]: row for row in rows}
                controls.update(sham_png_equals_zero=by_id["sham"]["image"]["sha256"] == by_id["zero"]["image"]["sha256"],
                    coherent_png_equals_text=by_id["coherent"]["image"]["sha256"] == by_id["text"]["image"]["sha256"])
                return {"rows": rows, "counts": dict(counts), "prefixes": prefixes, "candidates": candidates,
                    "control_observations": controls, "source_sha256": sha(output / "source.json"), "RGB_forecast": False,
                    "numerical_attention_forecast": attention_summary}

            tx.execute("native", execute)
        save(output / "transaction-report.json", tx.report())
        verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
        summary = {"schema": "mdb-flux-single-kv-result-v1", "job_id": request["job_id"], "context_id": source_metadata["context_id"], "request_sha256": request["request_sha256"],
            "model_identity": adapter.model_identity, "counts": counts, "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed",
            "seal": seal, "transaction_verification": verification, "worker_seconds": time.monotonic() - started,
            "control_observations": controls, "RGB_forecast": False, "scope": config["scope"]}
        summary["numerical_attention_forecast"] = attention_summary
        save(output / "summary.json", summary)
        event("complete", counts=dict(counts))
        return json_value(summary)
    finally:
        for handle in guards:
            handle.remove()
