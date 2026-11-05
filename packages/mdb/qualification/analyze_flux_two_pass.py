"""Audit two complete self-fed denoising passes from saved source evidence.

The independent numerical program never calls the producer or a native model.
Typed joins, custody and prospective order are mechanics; accuracy is observed.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREV = runpy.run_path(str(HERE / "analyze_flux_final_latent.py"))
BASE, H, CHAIN, MATH, NEW, OLD = (PREV[key] for key in ("BASE", "H", "CHAIN", "MATH", "NEW", "OLD"))
require, exact, sha, read, compare, boundary, cut_context, parameters = (
    PREV[key] for key in ("require", "exact", "sha", "read", "compare", "boundary", "cut_context", "parameters"))
PORTS, ALL_PORTS = PREV["PORTS"], PREV["ALL_PORTS"]
ORDER = ["native", "sham", "bridge.predicted", "full.predicted"]
PHYSICAL_ORDER = ["full.predicted", "native", "sham", "bridge.predicted"]
COUNTS = PREV["COUNTS"]


def tree_exact(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and exact(left, right)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(tree_exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(
            tree_exact(a, b) for a, b in zip(left, right, strict=True))
    return left is None and right is None


def independent_pass_events(latent, conditioning, img_ids, txt_ids, schedule, binding, load_parameters):
    """Yield independently computed outputs, always advancing our own state."""
    spec = binding["project"]
    weights = load_parameters("project", spec)
    project = NEW["independent_project"](latent, conditioning, img_ids, txt_ids, schedule["timestep"], weights,
        axes_dims_rope=spec["axes_dims_rope"], rope_theta=spec["rope_theta"], timestep_channels=spec["timestep_channels"])
    del weights
    context, image, text = project["context"], project["image"], project["text"]
    yield "project", None, project
    for index in range(5):
        spec = binding["joint" + str(index)]
        weights = load_parameters("joint", spec)
        result = NEW["independent_joint"](image, text, context, weights, heads=spec["heads"],
            layernorm_eps=spec["layernorm_eps"], qk_eps=spec["qk_eps"])
        del weights
        image, text = result["image"], result["text"]
        yield "joint", index, result
    current = np.concatenate((text, image), axis=1)
    del project, image, text
    for index in range(20):
        spec = binding["final_blocks"][str(index)]
        weights = load_parameters("single", spec)
        current = CHAIN["independent_transition"](current, context["single_mod"], weights,
            heads=spec["heads"], layernorm_eps=spec["layernorm_eps"], qk_eps=spec["qk_eps"],
            rope_cos=context["rope"][0], rope_sin=context["rope"][1])
        del weights
        yield "single", index, {"output": current}
    spec = binding["readout"]
    weights = load_parameters("readout", spec)
    result = MATH["independent_readout"](current[:, conditioning.shape[1]:], context["temb"], **weights, eps=spec["eps"])
    del weights
    yield "readout", None, result
    commit = MATH["independent_flow_commit"](latent, result["noise"], schedule["sigma"], schedule["sigma_next"])
    yield "commit", None, commit


def load_bank(e, kind, spec):
    if kind in ("project", "joint"):
        return parameters(e, spec)
    if kind == "single":
        return BASE["load_parameters"](e, spec, BASE["PARAMS"])
    require(kind == "readout", "unknown independent parameter role")
    ports = {**{"norm_out." + key: value for key, value in spec["native_parameters"].items()},
        **{"proj_out." + key: value for key, value in spec["projection_native_parameters"].items()}}
    return BASE["load_parameters"](e, spec, {"modulation_weight": "norm_out.linear.weight",
        "modulation_bias": "norm_out.linear.bias", "projection_weight": "proj_out.weight",
        "projection_bias": "proj_out.bias"}, ports)


def closure_state(e, cut, latent, context, image, text):
    actual = {"latent": e.slot(cut, "latent"), "context": cut_context(e, cut) if context is not None else
        e.typed["unpack"](e.fields(cut)["context"]), "image": e.slot(cut, "image") if image is not None else
        e.typed["unpack"](e.fields(cut)["image"]), "text": e.slot(cut, "text") if text is not None else
        e.typed["unpack"](e.fields(cut)["text"])}
    require(tree_exact(actual, {"latent": latent, "context": context, "image": image, "text": text}),
        "complete predicted closure was not installed")


def check_derived_terminal(e, record, prefix_cut, action, final_latent, helper_sha):
    """Derive terminal descriptors without any future native state operand."""
    unsigned = {key: value for key, value in record.items() if key != "fingerprint"}
    require(BASE["canonical_hash"](unsigned) == record["fingerprint"]
        and record["schema"] == "mdb-flux-derived-terminal-v1" and record["native_steps"] == 0
        and record["acts_applied"] == record["execution_receipt_emissions"] == 0
        and record["source_cut"] == record["scaffold_parent"] == prefix_cut
        and record["source_hashes"]["derivation_helper"] == helper_sha,
        "non-native terminal derivation identity differs")
    source = e.fields(prefix_cut)
    require(record["source_preconditions"] == {key: BASE["canonical_hash"](source[key]) for key in ALL_PORTS},
        "derived terminal source preconditions differ")
    source_cut = e.store["cuts"][prefix_cut]
    require(record["source_model_identity"] == source_cut["model_identity"]
        and record["source_execution"] == source_cut["execution"]
        and record["placeholder_latent_source"]["cut"] == prefix_cut
        and record["placeholder_latent_source"]["slot"] == "latent", "derived terminal source identity differs")
    scaffold = record["scaffold_cut"]
    require(record["source_point"] == e.point(prefix_cut) and record["scaffold_point"] == e.point(scaffold)
        and e.store["cuts"][scaffold]["parent"] == prefix_cut and action["parent"] == scaffold
        and record["action"] == action["act"], "derived shell lineage/Act differs")
    from saturn_pub.values import describe as public_describe
    describe = public_describe
    expected = {**source, "step": describe(len(e.slot(prefix_cut, "timesteps"))), "phase": describe("project"),
        "cursor": describe(0), **{key: describe(None) for key in ("context", "image", "text", "noise_prediction")}}
    require(e.fields(scaffold) == expected
        and OLD["described_mapping"](record["scaffold_payload"]) == expected,
        "scaffold supplied something outside prefix-derived clock/reset")
    installed = action["post_act"]
    actual = e.fields(installed)
    require(all(actual[key] == expected[key] for key in expected if key != "latent")
        and OLD["described_mapping"](record["predicted_terminal_payload"]) == actual,
        "predicted terminal borrowed a native future field")
    closure_state(e, installed, final_latent, None, None, None)
    require(tuple(e.point(installed)[key] for key in ("logical_step", "phase", "index")) ==
        (len(e.slot(prefix_cut, "timesteps")), "project", None), "derived terminal clock/address differs")
    return {"all13_fields_prefix_derived": True, "source_cut": prefix_cut, "scaffold_cut": scaffold,
        "installed_cut": installed, "derivation_fingerprint": record["fingerprint"],
        "scope": "Explicit non-native construction. All terminal values derive from observed source, sealed Z4 and declared clock/reset; later native state is comparison only."}


def audit_pass(e, binding, predicted, schedule, independent_latent, segment, pass_index, conditioning, img_ids, txt_ids,
              sealed_input_latent):
    logical_step = 2 + pass_index
    require(predicted["logical_step"] == logical_step
        and len(predicted["single_steps"]) == 20 and set(predicted["joints"]) == {str(i) for i in range(5)},
        "two-pass forecast occurrence inventory differs")
    output = {"logical_step": logical_step, "project": {}, "joints": {}, "blocks": []}
    current = independent_latent
    previous_image = previous_text = None
    context = predicted["project"]["context"]
    incoming_latent = predicted["project"]["input_latent_descriptor"]
    require(incoming_latent == MATH["identity"](sealed_input_latent), "next project did not receive its own sealed incoming latent")
    for local, (kind, index, independent) in enumerate(independent_pass_events(independent_latent, conditioning,
            img_ids, txt_ids, schedule, binding, lambda role, spec: load_bank(e, role, spec))):
        native = boundary(e, segment["boundaries"][28 * pass_index + local])
        if kind == "project":
            value = predicted["project"]
            output["project"] = {key: {"sealed_minus_independent": compare(value[key], independent[key]),
                "predicted_minus_native": compare(value[key], native[key])} for key in ("image", "text")}
            output["project"]["context"] = {key: compare(context[key], independent["context"][key])
                for key in ("temb", "img_mod", "txt_mod", "single_mod")}
            output["project"]["rope"] = [compare(a, b) for a, b in zip(context["rope"], independent["context"]["rope"], strict=True)]
            output["project"]["context_predicted_minus_native"] = {key: compare(context[key], native["context"][key])
                for key in ("temb", "img_mod", "txt_mod", "single_mod")}
            require(value["conditioning_descriptor"] == MATH["identity"](conditioning), "project conditioning refreshed")
            previous_image, previous_text = value["image"], value["text"]
        elif kind == "joint":
            value = predicted["joints"][str(index)]
            require(value["input_image_descriptor"] == MATH["identity"](previous_image)
                and value["input_text_descriptor"] == MATH["identity"](previous_text), "joint refreshed its predicted input")
            output["joints"][str(index)] = {key: {"sealed_minus_independent": compare(value[key], independent[key]),
                "predicted_minus_native": compare(value[key], native[key])} for key in ("image", "text")}
            previous_image, previous_text = value["image"], value["text"]
            if index == 4:
                current = np.concatenate((previous_text, previous_image), axis=1)
        elif kind == "single":
            entry = predicted["single_steps"][index]
            value = entry["prediction"]
            require(entry["block_index"] == index and value["input_descriptor"] == MATH["identity"](current)
                and value["output_descriptor"] == MATH["identity"](value["output"]), "single refreshed its predicted input")
            current = value["output"]
            native_values = np.concatenate((native["text"], native["image"]), axis=1)
            output["blocks"].append({"block": index, "sealed_minus_independent": compare(current, independent["output"]),
                "predicted_minus_native": {"both": compare(current, native_values),
                    "text": compare(current[:, :conditioning.shape[1]], native["text"]),
                    "image": compare(current[:, conditioning.shape[1]:], native["image"])}})
        elif kind == "readout":
            value = predicted["readout"]
            require(value["input_descriptor"] == MATH["identity"](current[:, conditioning.shape[1]:])
                and value["noise_descriptor"] == MATH["identity"](value["noise"]), "readout own-state join differs")
            output["readout"] = {"sealed_minus_independent": compare(value["noise"], independent["noise"]),
                "predicted_minus_native": compare(value["noise"], native["noise_prediction"])}
        else:
            require(kind == "commit" and local == 27, "pass transition sequence differs")
            value = predicted["commit"]
            require(value["input_descriptor"] == incoming_latent
                and value["noise_descriptor"] == predicted["readout"]["noise_descriptor"]
                and value["latent_descriptor"] == MATH["identity"](value["latent"]), "Euler own-state join differs")
            output["commit"] = {"sealed_minus_independent": compare(value["latent"], independent["latent"]),
                "predicted_minus_native": compare(value["latent"], native["latent"])}
            require(predicted["next_state_contract"]["clock"] == {"step": logical_step + 1, "phase": "project", "cursor": 0}
                and predicted["next_state_contract"]["reset"] ==
                {"context": None, "image": None, "text": None, "noise_prediction": None}, "pass reset/next clock differs")
            independent_latent = independent["latent"]
        print(json.dumps({"phase": "independent_two_pass", "logical_step": logical_step, "kind": kind, "index": index}), flush=True)
        gc.collect()
    return output, independent_latent


def analyze(root):
    started = time.monotonic()
    root = Path(root).resolve()
    request, job, collection = [read(root / name) for name in ("request.json", "job.json", "collection.json")]
    data = root / "extracted/flux-two-pass-result"
    collector = runpy.run_path(str(HERE / "collect_flux_two_pass.py"))
    collector["validate_job"](job, job["job_id"], request)
    inventory = collector["inventory"](data)
    require(inventory == read(root / "collection-remote-before.json") == read(root / "collection-remote-after.json")
        == read(root / "collection-local-inventory.json") and collection["job_id"] == job["job_id"]
        and collection["request_sha256"] == request["request_sha256"]
        and collection["collector_source_sha256"] == sha(HERE / "collect_flux_two_pass.py")
        and all(sha(root / name) == digest for name, digest in collection["bindings"].items())
        and job == read(root / "collection-job.json"), "authenticated collection differs")
    collector["verify_manifest"](data, inventory, job["job_id"], request)
    require(job["result"]["diagnostics"]["command"] == read(root / "admission-command.json")["command"], "executed command differs")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
        == request["request_sha256"], "request changed")
    for name, digest in request["source_hashes"].items():
        require(sha(H["file_path"](root / "payload", name)) == digest, "payload source changed")
    for name, digest in request["independent_math_sources"].items():
        require(sha(root / "independent-math-freeze" / name) == sha(HERE / name) == digest,
            "pre-outcome independent source changed")
    source, summary, plan, binding, prefix, segment = [read(data / name) for name in
        ("source.json", "summary.json", "plan.json", "binding.json", "prefix.json", "native-segment.json")]
    require(plan == request["config"] and summary["counts"] == COUNTS and plan["context"]["seed"] == 109
        and source["context"] == plan["context"] and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["source_hashes"] == request["source_hashes"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["checkpoint_sha256"] == BASE["canonical_hash"](source["checkpoint_files"]) == plan["expected_checkpoint_sha256"],
        "plan/model/source identity differs")
    require(source["execution"]["adapter"] == "mdb-flux-final-latent-closure-v1"
        and source["execution"]["device"]["type"] == "cuda" and source["execution"]["dtype"] == "torch.bfloat16",
        "actual backend differs")
    frozen = read(data / "plan-frozen.json")
    require(frozen["plan_sha256"] == sha(data / "plan.json")
        and frozen["source_identity"] == BASE["canonical_hash"](request["source_hashes"]), "plan freeze changed")
    syntax = ast.parse((root / "payload/flux_reentry_reference.py").read_text())
    pins = ast.literal_eval(next(node.value for node in syntax.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_CONTRACT" for target in node.targets)))
    expected_abi = {key: value for key, value in pins.items() if key not in {"diffusers_version", "torch_qualification_version"}}
    step_syntax = ast.parse((root / "payload/flux_step_boundary_reference.py").read_text())
    step_pins = ast.literal_eval(next(node.value for node in step_syntax.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_CONTRACT" for target in node.targets)))
    expected_step = {key: value for key, value in step_pins.items() if key.endswith("sha256")}
    for name in ("pre_model_source_abi", "native_source_abi"):
        ref = source[name]
        abi = read(data / ref["path"])
        final = abi["prior_final_latent"]
        tail = final["prior_joint_tail"]
        chain = tail["prior_joint_chain"]
        old = chain["prior_native_and_closure"]
        require(sha(data / ref["path"]) == ref["sha256"] and abi["verified"] is True
            and abi["native_advance_inherited"] is True and old["reentry_actual"] == old["reentry_expected"] == expected_abi
            and old["base_step"]["verified"] is True and old["base_step"]["model_calls"] == 0
            and old["base_step"]["actual"] == old["base_step"]["expected"] == expected_step
            and old["closure_module_sha256"] == plan["expected_closure_module_sha256"]
            and chain["joint_closure_module_sha256"] == plan["expected_joint_closure_module_sha256"]
            and tail["tail_closure_module_sha256"] == plan["expected_tail_closure_module_sha256"]
            and final["final_closure_module_sha256"] == plan["expected_final_closure_module_sha256"]
            and abi["derived_terminal_module_sha256"] == plan["expected_derived_terminal_module_sha256"], "native/helper source ABI differs")
    scalar = read(data / "cuda-scalar-qualification.json")
    require(scalar["qualified"] is True and scalar["device"] == "cuda" and scalar["model_calls"] == 0
        and len(scalar["cases"]) == 3 and all(case["values"] == 2048 and case["native_dtype"] == "torch.bfloat16"
            and case["native_vs_bf16_dt_mismatches"] == 0 and case["native_vs_fp32_dt_mismatches"] > 0
            for case in scalar["cases"]), "actual CUDA weak-scalar instrument absent")
    tx = H["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    tx_report = read(data / "transaction-report.json")
    require(tx["phase"] == tx_report["phase"] == "closed" and not tx_report["failure"] and not tx_report["violations"]
        and CHAIN["SHARED"]["transaction_verifications_match"](tx, summary["transaction_verification"]), "transaction did not close")
    tree = H["NumericalTree"](data / "transaction")
    decode = lambda name, field: tree.decode(read(data / "transaction" / name)[field])
    forecast = decode("forecasts/two_pass.json", "predictions")
    observed = decode("observations/native.json", "value")
    require(decode("frontiers/source_binding.json", "value") == binding
        and decode("forecasts/hypotheses.json", "predictions") == plan["hypotheses"], "sealed source binding differs")
    e = BASE["SavedEvidence"](data)
    require(all(cut["model_identity"] == source["model_identity"] and cut["execution"] == source["execution"]
            for cut in e.store["cuts"].values()) and all(receipt["model_identity"] == source["model_identity"]
            and receipt["execution"] == source["execution"] for receipt in e.receipts.values()), "typed identities differ")
    pre = prefix["pre_project"]
    prefix_ids = BASE["path"](e, prefix["receipts"], prefix["initial"], pre, 56)
    require(pre == binding["input_cut"] == segment["input_cut"]
        and tuple(e.point(pre)[key] for key in ("logical_step", "phase", "index")) == (2, "project", None)
        and all(e.typed["unpack"](e.fields(pre)[key]) is None for key in ("context", "image", "text", "noise_prediction")),
        "earlier observation point is not a native empty project boundary")
    from mdb.tensor_archive import load_tensor_archive
    inputs = {key: load_tensor_archive(data, ref) for key, ref in binding["inputs"].items()}
    require(set(inputs) == {"latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"}
        and all(exact(value, e.slot(pre, key)) for key, value in inputs.items()), "observed source inputs differ")
    require(exact(e.array(source["encoding"]["conditioning"]), inputs["conditioning"])
        and exact(load_tensor_archive(data, source["encoding"]["txt_ids"]), inputs["txt_ids"]), "clean encoding differs")
    reference_ids = BASE["path"](e, segment["receipts"], pre, segment["native_final"]["cut"], 56)
    require(len(segment["boundaries"]) == 56 and segment["common_post_joint4"]["cut"] == segment["boundaries"][33]["cut"]
        and segment["native_final"]["cut"] == segment["boundaries"][55]["cut"], "full native trajectory inventory differs")
    for index, record in enumerate(segment["boundaries"]):
        require(record["index"] == index and record["receipt_id"] == reference_ids[index]
            and record["cut"] == segment["receipts"][index]["result"], "native full-array boundary lineage differs")
        boundary(e, record)
        logical, local = 2 + index // 28, index % 28
        point = ((logical, "joint", local) if local < 5 else (logical, "single", local - 5) if local < 25 else
            (logical, "readout", None) if local == 25 else (logical, "scheduler", None) if local == 26 else
            (logical + 1, "project", None))
        require(tuple(e.point(record["cut"])[key] for key in ("logical_step", "phase", "index")) == point,
            "native two-pass clock/operation sequence differs")
    require(forecast["chain_length"] == 56 and len(forecast["passes"]) == 2
        and set(binding["pass_schedules"]) == {"2", "3"} and set(binding["final_blocks"]) == {str(i) for i in range(20)},
        "two-pass own-state inventory differs")
    for index in range(5):
        spec = binding["joint" + str(index)]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["layernorm_eps"], spec["qk_eps"]) == (24, 128, 1e-6, 1e-6),
            "joint source port/profile differs")
    for index in range(20):
        spec = binding["final_blocks"][str(index)]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"single_transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["hidden_width"], spec["mlp_width"], spec["layernorm_eps"], spec["qk_eps"])
                == (24, 128, 3072, 9216, 1e-6, 1e-6), "single source port/profile differs")
    pass_math, independent_latent, sealed_latent = [], inputs["latent"], inputs["latent"]
    for index, prediction in enumerate(forecast["passes"]):
        schedule = {key: e.array(binding["pass_schedules"][str(index + 2)][key])
            for key in ("timestep", "sigma", "sigma_next")}
        require(exact(schedule["timestep"], inputs["timesteps"][index + 2])
            and exact(schedule["sigma"], inputs["sigmas"][index + 2])
            and exact(schedule["sigma_next"], inputs["sigmas"][index + 3])
            and prediction["chain_length"] == 28
            and prediction["context_input_descriptors"] == PREV["context_descriptor"](prediction["project"]["context"]),
            "per-pass time/schedule/own context differs")
        result, independent_latent = audit_pass(e, binding, prediction, schedule, independent_latent, segment,
            index, inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"], sealed_latent)
        pass_math.append(result)
        sealed_latent = prediction["commit"]["latent"]
    require(float(inputs["sigmas"][4]) == 0.0, "terminal sigma is not zero")
    rows, pixels, latents = {}, {}, {}
    for name in ORDER:
        row = read(data / "rows" / (name + ".json"))
        require(row == next(value for value in observed["rows"] if value["id"] == name)
            and len(row["acts"]) == (name != "native") and row["continuation_receipts"] == [], "future record differs")
        parent = row["parent"]
        require(row["entry"]["cut"] == parent, "future entry differs")
        boundary(e, row["entry"])
        if name != "full.predicted":
            require(parent == segment["common_post_joint4" if name == "bridge.predicted" else "native_final"]["cut"],
                "native/sham/bridge parent differs")
        current = parent
        for action in row["acts"]:
            expected = ({"latent": e.slot(parent, "latent"), "context": None, "image": None, "text": None} if name == "sham" else
                {"latent": forecast["passes"][0]["commit"]["latent"], "context": forecast["passes"][1]["project"]["context"],
                 **{key: forecast["passes"][1]["joints"]["4"][key] for key in ("image", "text")}} if name == "bridge.predicted" else
                {"latent": sealed_latent, "context": None, "image": None, "text": None})
            current = PREV["atomic_act"](e, action, parent, expected)
            require(row["installed"]["cut"] == current, "installed capture differs")
            boundary(e, row["installed"])
        length = 22 if name == "bridge.predicted" else 0
        ids = BASE["path"](e, row["bridge_receipts"], current, row["final_cut"], length)
        require(len(row["bridge_boundaries"]) == len(row["bridge_records"]) == length, "bridge full arrays missing")
        for index, record in enumerate(row["bridge_records"]):
            require(record["index"] == index and record["before"]["cut"] == current
                and record["after"] == row["bridge_boundaries"][index] and record["receipt"] == row["bridge_receipts"][index]
                and record["receipt_id"] == ids[index] and record["after"]["cut"] == record["receipt"]["result"], "bridge path/arrays differ")
            before, after = boundary(e, record["before"]), boundary(e, record["after"])
            require(e.point(record["after"]["cut"]) == e.point(segment["boundaries"][34 + index]["cut"]), "matched suffix clock differs")
            if index < 21:
                require(exact(after["latent"], forecast["passes"][0]["commit"]["latent"])
                    and tree_exact(after["context"], forecast["passes"][1]["project"]["context"]), "bridge refreshed predicted Z3/context")
            if index == 0:
                require(exact(before["latent"], forecast["passes"][0]["commit"]["latent"])
                    and tree_exact(before["context"], forecast["passes"][1]["project"]["context"])
                    and all(exact(before[key], forecast["passes"][1]["joints"]["4"][key]) for key in ("image", "text")),
                    "mathematical/native suffix received different complete operands")
            current = record["after"]["cut"]
        final = boundary(e, row["terminal_final"])
        require(row["terminal_final"]["cut"] == row["final_cut"] == current
            and tuple(e.point(current)[key] for key in ("logical_step", "phase", "index")) == (4, "project", None)
            and all(final[key] is None for key in ("context", "image", "text", "noise_prediction")), "terminal clock/reset differs")
        latents[name], pixels[name] = e.array(row["final_latent"]), e.png(row["image"])
        require(exact(latents[name], final["latent"]), "actual VAE input latent differs")
        rows[name] = row
    derivation = read(data / "derived-terminal.json")
    require(rows["full.predicted"]["derivation"] == {"path": "derived-terminal.json", "sha256": sha(data / "derived-terminal.json")},
        "terminal derivation custody differs")
    derived = check_derived_terminal(e, derivation, pre, rows["full.predicted"]["acts"][0], sealed_latent,
        plan["expected_derived_terminal_module_sha256"])
    native_fields = e.fields(rows["native"]["final_cut"])
    for name in ("sham", "bridge.predicted", "full.predicted"):
        require(all(e.fields(rows[name]["final_cut"])[key] == native_fields[key] for key in native_fields if key != "latent")
            and e.point(rows[name]["final_cut"]) == e.point(rows["native"]["final_cut"]), "VAE consumers differ outside latent")
    expected_emissions = [("prefix", rid) for rid in prefix_ids]
    expected_emissions += [("full.predicted", rows["full.predicted"]["acts"][0]["receipt_id"])]
    expected_emissions += [("reference", rid) for rid in reference_ids]
    expected_emissions += [("sham", rows["sham"]["acts"][0]["receipt_id"]),
        ("bridge.predicted", rows["bridge.predicted"]["acts"][0]["receipt_id"])]
    expected_emissions += [("bridge.predicted", value["fingerprint"]) for value in rows["bridge.predicted"]["bridge_receipts"]]
    emissions = CHAIN["SHARED"]["read_jsonl"](data / "receipt-emissions.jsonl")
    require([value["sequence"] for value in emissions] == list(range(137))
        and [(value["branch"], value["receipt"]) for value in emissions] == expected_emissions
        and set(rid for _, rid in expected_emissions) == set(e.receipts), "physical execution receipt census/order differs")
    target_counts = {**{str(i): 3 for i in range(20)}, "norm_out": 3, "proj_out": 3,
        **{key: 2 for key in ("x_embedder", "context_embedder", "time_guidance_embed", "double_stream_modulation_img",
            "double_stream_modulation_txt", "single_stream_modulation")}, "pos_embed": 4,
        **{"joint" + str(i): 2 for i in range(5)}}
    require(summary["target_calls"] == target_counts, "selected-target census differs")
    timeline = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    sealed = [value for value in timeline if value["kind"] == "two_pass_sealed"]
    images = [value for value in timeline if value["kind"] == "predicted_image_complete_before_native_reference"]
    references = [value for value in timeline if value["kind"] == "reference_transition_complete"]
    branches = [value["row"] for value in timeline if value["kind"] == "branch_complete"]
    first = read(data / "first-predicted-image.json")
    require(len(sealed) == len(images) == 1 and len(references) == 56 and branches == PHYSICAL_ORDER
        and sealed[0]["seal"] == summary["seal"] and sealed[0]["blocked_entries"] == 0
        and set(sealed[0]["target_entries_before_seal"]) == set(target_counts)
        and set(sealed[0]["target_entries_before_seal"].values()) == {0}
        and first["target_entries"] == images[0]["target_entries"] == sealed[0]["target_entries_before_seal"]
        and sealed[0]["sequence"] < images[0]["sequence"] < min(value["sequence"] for value in references)
        and first["image"] == images[0]["image"] == rows["full.predicted"]["image"]
        and first["source_cut"] == pre and first["source_unchanged"] is first["parameters_unchanged"] is True
        and first["derivation_sha256"] == sha(data / "derived-terminal.json"), "predicted PNG was not completed before native reference")
    for key, value in {"adapter_steps": 56, "acts": 1, "receipt_emissions": 57, "images": 1, "vae_calls": 1}.items():
        require(first["counts"][key] == images[0]["counts"][key] == value, "first-image physical census differs")
    observations = read(data / "continuation-observations.json")
    require(observations["plan"] == plan["observation_plan"] and observations["model_calls_added"] == 0
        and set(observations["rows"]) == {"native", "bridge.predicted"}, "full suffix observer plan differs")
    bridge = rows["bridge.predicted"]
    bridge_incoming = [bridge["acts"][0]["receipt_id"], *[value["fingerprint"] for value in bridge["bridge_receipts"]]]
    for name, records, incoming in (
            ("native", segment["boundaries"][33:], reference_ids[33:]),
            ("bridge.predicted", [bridge["common_post_joint4"], *bridge["bridge_boundaries"]], bridge_incoming)):
        trajectory = observations["rows"][name]
        require(len(trajectory) == len(records) == len(incoming) == 23, "observer boundary count differs")
        for index, (item, record, rid) in enumerate(zip(trajectory, records, incoming, strict=True)):
            require(item["index"] == index and item["incoming_receipt"] == rid and item["complete_cut_retained"] is True
                and all(item[key] == record[key] for key in ("cut", "point", "closure", "clock")), "observer lineage differs")
            boundary(e, item)
    final_pass = forecast["passes"][1]
    predicted_values = [np.concatenate((final_pass["joints"]["4"]["text"], final_pass["joints"]["4"]["image"]), axis=1)]
    predicted_values += [entry["prediction"]["output"] for entry in final_pass["single_steps"]]
    predicted_values += [final_pass["readout"]["noise"], final_pass["commit"]["latent"]]
    suffix_curve = []
    for index, predicted in enumerate(predicted_values):
        nrecord = segment["boundaries"][33 + index]
        brecord = bridge["common_post_joint4"] if index == 0 else bridge["bridge_boundaries"][index - 1]
        nvalue, bvalue = boundary(e, nrecord), boundary(e, brecord)
        role = "working_image_and_text" if index <= 20 else "readout_noise" if index == 21 else "committed_final_latent"
        slot = "noise_prediction" if index == 21 else "latent"
        narray = np.concatenate((nvalue["text"], nvalue["image"]), axis=1) if index <= 20 else nvalue[slot]
        barray = np.concatenate((bvalue["text"], bvalue["image"]), axis=1) if index <= 20 else bvalue[slot]
        streams = ({key: PREV["suffix_difference"](nvalue[key], bvalue[key],
            predicted[:, :binding["text_rows"]] if key == "text" else predicted[:, binding["text_rows"]:])
            for key in ("image", "text")} if index <= 20 else {})
        suffix_curve.append({"index": index, "role": role, "native_cut": nrecord["cut"], "bridge_cut": brecord["cut"],
            "point": nrecord["point"], "decomposition": PREV["suffix_difference"](narray, barray, predicted), "streams": streams})
    controls = {"sham_final_latent_exact_native": exact(latents["sham"], latents["native"]),
        "sham_png_exact_native": rows["sham"]["image"]["sha256"] == rows["native"]["image"]["sha256"]}
    require(controls == summary["control_observations"] == observed["control_observations"], "control report differs")
    endpoints = {}
    for name in ORDER:
        delta = pixels[name].astype(np.float64) - pixels["native"]
        endpoints[name] = {"latent": compare(latents[name], latents["native"]), "rgb": {
            "mae": float(np.abs(delta).mean()), "rms": float(np.sqrt(np.mean(delta * delta))),
            "signed_mean": float(delta.mean()), "max_abs": float(np.abs(delta).max()),
            "changed_value_count": int(np.count_nonzero(delta)), "value_count": int(delta.size)}}
    rgb = PREV["suffix_difference"](pixels["native"], pixels["bridge.predicted"], pixels["full.predicted"])
    rgb["scope"] = "Actual native VAE response to latent differences; P-B is neither a VAE-kernel error nor independently predicted RGB."
    return {"schema": "mdb-flux-two-pass-audit-v1", "mechanics_status": "valid",
        "instrument_status": "applicable; independent own-state56 recurrence and actual pre-reference predicted PNG",
        "trend_status": "observed; signed discrepancies retained", "terminal_status": "unassessed",
        "job_id": job["job_id"], "request_sha256": request["request_sha256"], "seal": summary["seal"], "counts": COUNTS,
        "passes": pass_math, "endpoint_comparisons": endpoints, "derived_terminal": derived,
        "first_predicted_image": first, "final_latent_decomposition": PREV["suffix_difference"](
            latents["native"], latents["bridge.predicted"], latents["full.predicted"]),
        "final_rgb_decomposition": rgb, "suffix_curve": suffix_curve, "control_observations": controls,
        "audit_seconds": time.monotonic() - started, "models_executed": False, "target_calls": summary["target_calls"],
        "collection": {"files": inventory["file_count"], "bytes": inventory["expanded_bytes"], "sha256": sha(root / "collection.json")},
        "independent_math_sources": request["independent_math_sources"], "auditor_source_sha256": sha(Path(__file__)),
        "scope": "Fresh seed109, observed step2/project after56native. Complete56 own-state transitions toZ4 and actual nativeVAE PNG BEFORE selected native targets. Derived terminal is declared non-native state construction. Matched P-B covers final22suffix only. One history, no raw-prompt/independentRGB forecast, universal error bound or speedup claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("mechanics_status", "job_id", "control_observations", "audit_seconds")}))
