"""Fixed-recipient image Q × whole-K/V-bank conditional primitive forecasts.

All source QKV were observed in an earlier authenticated run. Four native
SDPA/image-tail primitive forecasts are sealed before four full native joint0
targets. The targets still form their own QKV, then substitute the declared
sealed inputs at dispatch and return the original SDPA result unchanged.
"""
from __future__ import annotations

import argparse
import faulthandler
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path

READER = "flux_image_reader_forecast.py"
INVENTORY = "audit_flux_image_tail_microscope.py"
BRANCHES = tuple(dict(id=f"q{q}-b{b}", sourceQindex=q, sourceBankindex=b) for q, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
SOURCE_ROLES = {"0": "sham", "1": "midpoint"}


def helpers(config):
    here = Path(__file__).parent
    reader = runpy.run_path(str(here / READER))
    microscope, core = reader["helpers"](config)
    for name in (READER, INVENTORY):
        microscope["require"](microscope["sha"](here / name) == config["helper_source_sha256"][name], "qualified helper differs: " + name)
    return reader, microscope, core


def validate_config(config):
    reader, microscope, _ = helpers(config)
    reader["validate_config"](config)
    for key in ("source_manifest_sha256", "source_request_sha256", "source_forecast_seal_sha256", "source_input_seal_sha256"):
        value = config[key]
        microscope["require"](isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid source identity: " + key)
    microscope["require"](isinstance(config["source_job_id"], str) and config["source_job_id"], "source job required")
    return config


def load_source(config, reader, microscope, core):
    """Authenticate prior actual QKV and join the older sealed observation path."""
    import numpy as np
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    root = Path(config["source_root"])
    require, sha = microscope["require"], microscope["sha"]
    inventory = runpy.run_path(str(Path(__file__).with_name(INVENTORY)))["validate_inventory"](root, config["source_manifest_sha256"])
    read = lambda name: json.loads((root / name).read_text())
    summary, source, plan, inputs = [read(name) for name in ("summary.json", "source.json", "plan.json", "input-seal.json")]
    require(all(row["request_sha256"] == config["source_request_sha256"] for row in (summary, source, plan, inventory)), "prior source request differs")
    require(all(row["job_id"] == config["source_job_id"] for row in (summary, source, inventory)), "prior source job differs")
    require(source["worker_sha256"] == config["helper_source_sha256"][READER], "prior producer source differs")
    require(all(plan[name] == config[name] for name in ("module_config", "native_source_sha256", "expected_model_identity", "joint_parameters")), "prior native profile/parameter occurrence differs")
    require(sha(root / "input-seal.json") == config["source_input_seal_sha256"] == summary["input_seal_sha256"] == source["input_seal_sha256"], "prior input seal differs")
    require(summary["forecast_seal_sha256"] == config["source_forecast_seal_sha256"], "prior forecast seal differs")
    transaction = root / "transaction"
    verification = verify_transaction(transaction, expected_seal_sha256=config["source_forecast_seal_sha256"], max_bytes=plan["transaction_max_bytes"])
    frontier = read("transaction/frontiers/observed_qkv.json")
    frontier = load_forecast_tree(transaction, frontier["value"])
    encoded = read("transaction/forecasts/image_reader.json")
    predictions = load_forecast_tree(transaction, encoded["predictions"])
    require(predictions["target_entries_before_seal"] == 0, "prior forecast exposed complete target")
    sources, joins = {}, {}
    for index, role in SOURCE_ROLES.items():
        prefix, target = read("prefixes/" + role + ".json"), read("targets/" + role + ".json")
        cut = inputs["inputs"][role]["cut"]
        require(all(row["input_cut"] == cut for row in (prefix, target, frontier[role], predictions["cases"][role])), "prior role/cut join differs")
        require(prefix["input_seal_sha256"] == frontier[role]["input_seal_sha256"] == config["source_input_seal_sha256"], "prior frontier/input binding differs")
        values, layouts, parity = {}, {}, {}
        for name in ("q", "k", "v"):
            actual = core["H"]["load_array"](root, target["actual"][name])
            earlier = core["H"]["load_array"](root, prefix["values"][name])
            sealed = np.asarray(frontier[role]["values"][name])
            require(earlier.dtype == sealed.dtype == np.float32 and earlier.shape == sealed.shape
                    and np.array_equal(np.ascontiguousarray(earlier).view(np.uint32), np.ascontiguousarray(sealed).view(np.uint32)),
                    "source sidecar does not match sealed prefix QKV")
            sealed_layout = json.loads(json.dumps(frontier[role]["layouts"][name], default=dict))
            require(microscope["identity"](prefix["layouts"][name]) == microscope["identity"](sealed_layout), "source sealed layout differs")
            descriptor = target["actual_descriptors"][name]
            require(descriptor["shape"] == list(actual.shape) and descriptor["dtype"] == "torch.bfloat16", "source native QKV descriptor differs")
            require(microscope["historical_descriptor"](actual)["bf16_le_sha256"] == descriptor["sha256"], "source QKV native byte identity differs")
            values[name] = actual
            layouts[name] = dict(shape=descriptor["shape"], stride=descriptor["stride"])
            parity[name] = dict(actual_minus_prior_sealed=microscope["measure"](actual, sealed),
                                actual_bf16_bytes_match_prior_sealed=microscope["historical_descriptor"](actual)["bf16_le_sha256"] == microscope["historical_descriptor"](sealed)["bf16_le_sha256"],
                                actual_record=target["actual"][name], prefix_record=prefix["values"][name], native_descriptor=descriptor)
        require(values["q"].shape == values["k"].shape == values["v"].shape, "source full-bank geometry differs")
        sources[index] = dict(values=values, layouts=layouts, input_cut=cut, source_role=role)
        joins[index] = dict(role=role, input_cut=cut, parity=parity)
    midpoint = core["H"]["load_array"](root, inputs["inputs"]["midpoint"]["image"])
    controls = dict(q0_b0=predictions["controls"]["wrong_fixed_head"]["image"],
                    q1_b1=core["H"]["load_array"](root, read("targets/midpoint.json")["actual"]["image"]))
    return dict(root=root, sources=sources, inputs=inputs, midpoint_image=midpoint, controls=controls,
                joins=joins, verification=verification, inventory=inventory)


def capture_intervened(block, payload, native, chosen, declared, microscope, counts):
    """Capture receiving QKV, replace dispatch inputs, forward original SDPA once."""
    from mdb_lanes.forward_recording import tensor_descriptor

    original, calls, consumed = native.dispatch_attention_fn, [], {}

    def intervene(q, k, v, *args, **kwargs):
        calls.append(True)
        if len(calls) != 1:
            raise ValueError("extra native consumer dispatch")
        for name, receiving in zip(("q", "k", "v"), (q, k, v)):
            supplied = chosen[name]
            descriptor = tensor_descriptor(supplied)
            if (supplied.shape != receiving.shape or supplied.dtype != receiving.dtype or supplied.device != receiving.device
                    or list(supplied.stride()) != list(declared[name]["stride"])
                    or any(descriptor[key] != declared[name][key] for key in ("sha256", "dtype"))
                    or tuple(descriptor["shape"]) != tuple(declared[name]["shape"])):
                raise ValueError("sealed QKV consumer payload differs")
            consumed[name] = dict(descriptor, stride=list(supplied.stride()))
        return original(chosen["q"], chosen["k"], chosen["v"], *args, **kwargs)

    native.dispatch_attention_fn = intervene
    try:
        # capture_joint observes receiving arguments before the inner substitution.
        actual, descriptors, operations = microscope["capture_joint"](block, payload, native, counts)
        if len(calls) != 1:
            raise ValueError("native consumer missing")
        receiving = {name: actual.pop(name) for name in ("q", "k", "v")}
        receiving_descriptors = {name: descriptors.pop(name) for name in ("q", "k", "v")}
        return actual, descriptors, operations, receiving, receiving_descriptors, consumed
    finally:
        native.dispatch_attention_fn = original


def contrast_accounts(native_images, predicted_images):
    """Every response/error uses the same native Y11−Y00 fixed-recipient RMS."""
    import numpy as np
    pairs = dict(query_at_bank0=("q1-b0", "q0-b0"), query_at_bank1=("q1-b1", "q0-b1"),
                 bank_at_query0=("q0-b1", "q0-b0"), bank_at_query1=("q1-b1", "q1-b0"),
                 total=("q1-b1", "q0-b0"))
    total = native_images["q1-b1"].astype(np.float64) - native_images["q0-b0"].astype(np.float64)
    denominator = float(np.sqrt(np.mean(total * total)))
    def contrast(values, name):
        if name == "interaction":
            return (values["q1-b1"].astype(np.float64) - values["q1-b0"].astype(np.float64)
                    - values["q0-b1"].astype(np.float64) + values["q0-b0"].astype(np.float64))
        high, low = pairs[name]
        return values[high].astype(np.float64) - values[low].astype(np.float64)
    def statistics(value):
        rms = float(np.sqrt(np.mean(value * value)))
        return dict(rms=rms, relative_to_native_total_rms=None if denominator == 0 else rms / denominator,
                    signed_mean=float(value.mean()), maxabs=float(np.max(np.abs(value))))
    records = {}
    for name in (*pairs, "interaction"):
        actual, predicted = contrast(native_images, name), contrast(predicted_images, name)
        an, pn = np.linalg.norm(actual), np.linalg.norm(predicted)
        records[name] = dict(native=statistics(actual), predicted=statistics(predicted),
                             prediction_minus_native=statistics(predicted - actual),
                             cosine=None if an == 0 or pn == 0 else float(np.vdot(actual, predicted) / (an * pn)))
    return dict(native_total_rms=denominator, denominator="native Y11 minus Y00 at identical midpoint residual/modulation",
                records=records, interaction="Y11-Y10-Y01+Y00", scope="Fixed-recipient numerical response, not old midpoint-minus-sham response or semantic attribution.")


def run(config, *, parents=None, weights=None, progress=None):
    validate_config(config)
    import numpy as np
    import torch
    from diffusers.models.transformers import transformer_flux2 as native
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor

    from saturn_pub.core import StateCut
    from saturn_pub.values import describe

    started = time.monotonic()
    reader, m, core = helpers(config)
    require, save, sha = m["require"], m["save"], m["sha"]
    require(sha(inspect.getfile(native)) == config["native_source_sha256"], "native module differs")
    tiny = config.get("profile") == "tiny-cpu"
    require(tiny or (parents is None and weights is None), "injected source requires tiny-cpu")
    device = config.get("device", "cuda")
    require(device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    torch.set_num_threads(config.get("cpu_threads", 4))
    output, parameter_root, input_root = Path(config["output"]), Path(config["parameter_root"]).resolve(), Path(config["input_root"]).resolve()
    require(not any(output.resolve().is_relative_to(root) for root in (parameter_root, input_root, Path(config["source_root"]).resolve())), "output overlaps original source")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    phase = dict(name="setup", seal=None, target_entries=0, denied_target_attempts=0)
    forecast_counts = dict.fromkeys(("native_attention", "linear", "layernorm", "swiglu"), 0)
    target_counts = dict.fromkeys(("native_blocks", "native_attention", "linear", "layernorm", "rmsnorm", "swiglu"), 0)
    guard, bank = None, None
    original_policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic, torch.get_deterministic_debug_mode())
    def event(kind, **detail):
        row = dict(kind=kind, seconds=time.monotonic() - started, **detail)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
        require(row["seconds"] <= config.get("max_wall_seconds", 600), "finite query-bank wall exceeded")
    def archive(value):
        return core["H"]["array"](output, value)
    def wide(value):
        return value.detach().float().cpu().numpy().copy()
    try:
        core["configure_eager_determinism"]()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        if not tiny:
            for key, name in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
                require(Path(config.get(key, name)).is_file(), "required custody sidecar missing: " + name)
            require(sha(config["retrieval_script"]) == m["PARAMETER_RETRIEVER_SHA256"], "qualified retriever differs")
        source = load_source(config, reader, m, core)
        for name, record in config.get("input_files", {}).items():
            path = input_root / name
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.resolve().is_relative_to(input_root)
                    and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "original source changed: " + name)
        if weights is None:
            require(all(row["parameter_tensor"] == "transformer_blocks.0." + key for key, row in config["joint_parameters"].items()), "native parameter mapping differs")
            bank = core["H"]["RecipeBank"](json.loads(Path(config["parameter_manifest"]).read_text()), parameter_root, max_cache_bytes=512 << 20)
            weights = {name: bank.load(record) for name, record in config["joint_parameters"].items()}
        source_parameters = {name: core["R"]["descriptor"](value) for name, value in weights.items()}
        with torch.device("meta"):
            block = native.Flux2TransformerBlock(**config["module_config"])
        require(set(weights) == set(block.state_dict()), "complete original joint0 required")
        block.load_state_dict({name: torch.from_numpy(np.array(value, copy=True)).to(device, torch.bfloat16) for name, value in weights.items()}, assign=True, strict=True)
        block.eval().requires_grad_(False)
        require(type(block.attn.processor) is native.Flux2AttnProcessor, "qualified processor differs")
        before_parameters = {name: describe(value) for name, value in block.state_dict().items()}
        if parents is None:
            from saturn_pub.store import LocalStore
            parents = {role: LocalStore(input_root / spec.get("store", role)).load(spec["cut"], device=device) for role, spec in config["inputs"].items()}
        before_parents = {}
        require(set(parents) == {"sham", "edit"}, "two original source parents required")
        for role, cut in parents.items():
            cut.verify()
            require(cut.fingerprint == config["inputs"][role]["cut"] and cut.model_identity == config["expected_model_identity"], "source cut/model differs")
            require(cut.fingerprint == source["inputs"]["original_source_cuts"][role], "source original-cut join differs")
            before_parents[role] = describe(cut.payload)
        sham = parents["sham"]
        p = sham.payload
        require((p["step"], p["phase"], p["cursor"]) == (1, "joint", 0), "requires actual step1 pre-joint0")
        image_layout = dict(shape=list(p["image"].shape), stride=list(p["image"].stride()))
        image = reader["restore_bf16"](source["midpoint_image"], image_layout, device)
        recipient = StateCut(sham.model_identity, sham.execution, sham.boundary, sham.fingerprint, {**p, "image": image}, sham.execution_point, sham.surface)
        require(recipient.fingerprint == source["inputs"]["inputs"]["midpoint"]["cut"], "reconstructed sealed midpoint cut differs")
        require(all(describe(recipient.payload[name]) == describe(p[name]) for name in p if name != "image"), "recipient changed carried state")
        payload = recipient.payload
        text_rows = payload["text"].shape[1]
        if not tiny:
            require(tuple(payload["image"].shape) == (1, 1024, 3072) and tuple(payload["text"].shape) == (1, 512, 3072), "production recipient geometry differs")
        modules = core["make_modules"](m["remap_image_weights"](weights), device=device, eps=config["module_config"]["eps"])
        block_modules = set(block.modules())
        def target_guard(module, args):
            if module in block_modules:
                if phase["name"] != "target":
                    phase["denied_target_attempts"] += 1
                    raise ValueError("complete block entered forecast phase")
                if module is block:
                    require(phase["seal"] is not None, "target before forecast seal")
                    phase["target_entries"] += 1
                    require(phase["target_entries"] <= 4, "extra query-bank target")
        guard = torch.nn.modules.module.register_module_forward_pre_hook(target_guard)
        layout = lambda value: dict(shape=list(value.shape), stride=list(value.stride()))
        frontier = {**source["sources"], "recipient": dict(image=wide(payload["image"]), image_layout=layout(payload["image"]),
                    img_mod=wide(payload["context"]["img_mod"]), img_mod_layout=layout(payload["context"]["img_mod"]),
                    cut=recipient.fingerprint, text_rows=text_rows)}
        save(output / "source-joins.json", dict(joins=source["joins"], old_transaction_verification=source["verification"],
             original_job=config["source_job_id"], original_request_sha256=config["source_request_sha256"], original_manifest_sha256=config["source_manifest_sha256"]))
        save(output / "source.json", dict(schema="mdb-image-query-bank-source-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"),
             worker_sha256=sha(__file__), source_job_id=config["source_job_id"], source_manifest_sha256=config["source_manifest_sha256"],
             source_request_sha256=config["source_request_sha256"], source_forecast_seal_sha256=config["source_forecast_seal_sha256"],
             source_input_seal_sha256=config["source_input_seal_sha256"], source_root=str(source["root"]), helper_source_sha256=config["helper_source_sha256"],
             recipient_cut=recipient.fingerprint, recipient_manifest=recipient.manifest(), parameters=source_parameters, joint_parameters=config["joint_parameters"],
             source_scope="Prior actual positioned QKV; no new prefix. Original full-block QKV formation runs before sealed dispatch substitution."))
        save(output / "native-contract.json", dict(device=device, dtype="torch.bfloat16", source_reduction_policy=True, tf32=False, deterministic=True,
             processor=type(block.attn.processor).__name__, backend=block.attn.processor._attention_backend, parallel_config=repr(block.attn.processor._parallel_config),
             query_geometry="full original text+image rows", bank="complete coupled K/V text+image bank", recipient="same midpoint image/residual/modulation/text/RoPE/clock for every branch",
             intervention="dispatch inputs only; original SDPA return is not replaced; no fabricated Act receipt"))
        source_changes = {}
        for name in ("q", "k", "v"):
            a, b = source["sources"]["0"]["values"][name], source["sources"]["1"]["values"][name]
            source_changes[name] = dict(text_rows=m["measure"](b[:, :text_rows], a[:, :text_rows]), image_rows=m["measure"](b[:, text_rows:], a[:, text_rows:]))
        save(output / "source-qkv-changes.json", source_changes)
        event("source_bound", target_entries=0, recipient_cut=recipient.fingerprint)
        with torch.inference_mode():
            phase["name"] = "forecast"
            with ForecastTransaction(output / "transaction", model_identity=config["expected_model_identity"], input_identity=recipient.fingerprint,
                    source_identity=dict(job_id=config["source_job_id"], manifest=config["source_manifest_sha256"], seal=config["source_forecast_seal_sha256"]),
                    frontier={"kind": "architectural"}, array_storage="files-v1",
                    budgets={"max_bytes": config.get("transaction_max_bytes", 512 << 20), "max_wall_ms": int(config.get("max_wall_seconds", 600) * 1000), "max_native_calls": 1},
                    execution_scope="Externally counted native SDPA/image-tail primitive forecasts from prior saved QKV; full-block targets externally excluded until seal.") as tx:
                tx.frontier("saved_qkv", frontier)
                def forecast(context):
                    f = context.frontiers["saved_qkv"]
                    rec = f["recipient"]
                    predictions = {}
                    for branch in BRANCHES:
                        qindex, bindex = str(branch["sourceQindex"]), str(branch["sourceBankindex"])
                        q = reader["restore_bf16"](f[qindex]["values"]["q"], f[qindex]["layouts"]["q"], device)
                        k, v = [reader["restore_bf16"](f[bindex]["values"][name], f[bindex]["layouts"][name], device) for name in ("k", "v")]
                        require(q.shape == k.shape == v.shape and q.shape[1] == rec["image"].shape[1] + int(rec["text_rows"]), "full source QKV geometry differs")
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                            forecast_counts["native_attention"] += 1
                            head = native.dispatch_attention_fn(q, k, v, attn_mask=None, backend=block.attn.processor._attention_backend, parallel_config=block.attn.processor._parallel_config)
                            head = head.flatten(2, 3).to(q.dtype)[:, int(rec["text_rows"]):]
                            if head.is_cuda:
                                torch.cuda.synchronize()
                        values = m["image_fields"](core["native_tail"](head, reader["restore_bf16"](rec["image"], rec["image_layout"], device),
                                    reader["restore_bf16"](rec["img_mod"], rec["img_mod_layout"], device), modules, counts=forecast_counts))
                        predictions[branch["id"]] = dict(image=wide(values["image"]), head=wide(head), primitive_ports={name: archive(value) for name, value in values.items()},
                              sourceQindex=branch["sourceQindex"], sourceBankindex=branch["sourceBankindex"], recipient_cut=rec["cut"],
                              consumed_qkv_descriptors={name: dict(tensor_descriptor(value), stride=list(value.stride())) for name, value in zip(("q", "k", "v"), (q, k, v))},
                              selected_aten_operations=[dict(name=item.key, calls=item.count) for item in trace.key_averages() if any(token in item.key for token in ("scaled_dot", "bmm", "softmax"))])
                    require(phase["target_entries"] == phase["denied_target_attempts"] == 0, "target attempted during forecast")
                    return {"predictions": dict(cases=predictions, forecast_counts=forecast_counts, target_entries_before_seal=0)}
                tx.predict("image_query_bank", forecast)
                require(phase["target_entries"] == phase["denied_target_attempts"] == 0, "target exposure before seal")
                phase["seal"] = tx.seal()["seal_sha256"]
                event("cohort_sealed", seal_sha256=phase["seal"], target_entries=0, forecast_counts=forecast_counts.copy())
                def execute(context):
                    phase["name"] = "target"
                    f, predictions, targets = context.frontiers["saved_qkv"], context.predictions["image_query_bank"]["cases"], {}
                    for branch in BRANCHES:
                        key, pred = branch["id"], predictions[branch["id"]]
                        qindex, bindex = str(pred["sourceQindex"]), str(pred["sourceBankindex"])
                        chosen = {name: reader["restore_bf16"](f[qindex if name == "q" else bindex]["values"][name],
                                   f[qindex if name == "q" else bindex]["layouts"][name], device) for name in ("q", "k", "v")}
                        actual, descriptors, operations, receiving, receiving_desc, consumed_desc = capture_intervened(block, recipient.payload, native, chosen,
                                    pred["consumed_qkv_descriptors"], m, target_counts)
                        target = dict(recipient_cut=recipient.fingerprint, sourceQindex=pred["sourceQindex"], sourceBankindex=pred["sourceBankindex"],
                                      actual={name: archive(value) for name, value in actual.items()}, actual_descriptors=descriptors,
                                      receiving_qkv={name: archive(value) for name, value in receiving.items()}, receiving_descriptors=receiving_desc,
                                      consumed_qkv={name: archive(value) for name, value in chosen.items()}, consumed_descriptors=consumed_desc,
                                      forecast_minus_native=m["measure"](pred["image"], wide(actual["image"])), selected_aten_operations=operations,
                                      operation=dict(recipient_path="joint0.attn.dispatch", sources={"q": "saved actual Q" + qindex, "k": "saved actual K" + bindex, "v": "saved actual V" + bindex},
                                      receiving_qkv_formed_natively=True, original_sdpa_calls=1, return_replaced=False, act_receipt=None))
                        if key in ("q0-b0", "q1-b1"):
                            reference = source["controls"]["q0_b0" if key == "q0-b0" else "q1_b1"]
                            target["retrospective_control"] = dict(native_minus_prior=m["measure"](wide(actual["image"]), reference),
                                reference="prior sealed wrong-fixed-sham-head prediction at midpoint residual" if key == "q0-b0" else "prior complete midpoint native image return")
                        targets[key] = target
                        save(output / "targets" / (key + ".json"), target)
                        event("target_completed", branch=key, target_entries=phase["target_entries"], target_counts=target_counts.copy())
                    return dict(targets=targets, forecast_counts=forecast_counts, target_counts=target_counts, recipient_cut=recipient.fingerprint)
                tx.execute("native_targets", execute)
                targets = tx.observation("native_targets")["targets"]
                predictions = tx.prediction("image_query_bank")["cases"]
                native_images = {key: core["H"]["load_array"](output, row["actual"]["image"]) for key, row in targets.items()}
                predicted_images = {key: row["image"] for key, row in predictions.items()}
                save(output / "contrasts.json", contrast_accounts(native_images, predicted_images))
            save(output / "transaction-report.json", tx.report())
            save(output / "transaction-verification.json", verify_transaction(output / "transaction", expected_seal_sha256=phase["seal"], max_bytes=config.get("transaction_max_bytes", 512 << 20)))
        require(target_counts == dict(native_blocks=4, native_attention=4, linear=48, layernorm=16, rmsnorm=16, swiglu=8), "native target census differs")
        require(forecast_counts == dict(native_attention=4, linear=12, layernorm=4, swiglu=4), "forecast primitive census differs")
        require(phase["target_entries"] == 4 and phase["denied_target_attempts"] == 0, "target phase violations")
        for role, cut in parents.items():
            cut.verify()
            require(describe(cut.payload) == before_parents[role], "source state changed")
        recipient.verify()
        require({name: describe(value) for name, value in block.state_dict().items()} == before_parameters, "native parameters changed")
        require({name: core["R"]["descriptor"](value) for name, value in weights.items()} == source_parameters, "source parameters changed")
        if bank is not None:
            for relative, signature in bank.signatures.items():
                stat = (parameter_root / relative).stat()
                require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature, "checkpoint changed")
        for name, record in config.get("input_files", {}).items():
            require(sha(input_root / name) == record["sha256"], "input source changed")
        runpy.run_path(str(Path(__file__).with_name(INVENTORY)))["validate_inventory"](source["root"], config["source_manifest_sha256"])
        for key, name in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
            path = config.get(key, name)
            if Path(path).is_file():
                shutil.copyfile(path, output / name)
            else:
                require(tiny, "required custody sidecar missing: " + name)
        if not tiny:
            require(sha(output / "retrieve_weights.py") == m["PARAMETER_RETRIEVER_SHA256"], "qualified retriever differs")
        summary = dict(schema="mdb-image-query-bank-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"),
                       source_job_id=config["source_job_id"], recipient_cut=recipient.fingerprint, branches=[row["id"] for row in BRANCHES],
                       forecast_primitive_counts=forecast_counts, target_counts=target_counts, complete_native_targets=4, new_prefix_entries=0,
                       denied_target_attempts=0, forecast_seal_sha256=phase["seal"], transaction_array_storage="files-v1", seconds=time.monotonic() - started,
                       vae_calls=0, suffix_steps=0, encodes=0, noise_draws=0, immutable_sources_and_parameters_unchanged=True, mechanics_status="complete", terminal_status="unassessed",
                       scope="Conditional native primitive query/whole-bank cross, same midpoint recipient. All numerical/control misses retained; no independent Flash or semantic claim.")
        save(output / "summary.json", summary)
        return summary
    finally:
        if guard is not None:
            guard.remove()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_policy
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original_flags[:4]
        torch.set_deterministic_debug_mode(original_flags[4])


def finalize_directory(output, **kwargs):
    return runpy.run_path(str(Path(__file__).with_name(READER)))["finalize_directory"](output, **kwargs)


def worker(config_path):
    faulthandler.dump_traceback_later(90, repeat=True)
    print("MDB_IMAGE_QUERY_BANK_SETUP bootstrap", flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    _, m, _ = helpers(config)
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        m["require"](request["config"] == config, "request/config differs")
        for name, digest in request["source_hashes"].items():
            m["require"](m["sha"](Path.cwd() / name) == digest, "staged source differs: " + name)
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "binding-minimal-result"))
    summary = run(config, progress=lambda row: print("MDB_IMAGE_QUERY_BANK_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    ready = finalize_directory(config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
