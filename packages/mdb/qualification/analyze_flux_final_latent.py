"""Audit a sealed own-state forecast through the final denoising latent.

Saved arrays only. Whole-suffix finite differences retain signed discrepancies;
custody and state mechanics are distinct from numerical/semantic observations.
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
TAIL = runpy.run_path(str(HERE / "analyze_flux_joint_tail.py"))
OLD = TAIL["OLD"]
BASE, H, CHAIN, MATH, NEW = (TAIL[key] for key in ("BASE", "H", "CHAIN", "MATH", "NEW"))
require, exact, sha, read = (TAIL[key] for key in ("require", "exact", "sha", "read"))
compare, boundary, cut_context, parameters = (TAIL[key] for key in ("compare", "boundary", "cut_context", "parameters"))
PORTS, ALL_PORTS = TAIL["PORTS"], TAIL["ALL_PORTS"]
ORDER = ["native", "sham", "bridge.predicted", "full.predicted"]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 134,
    "acts": 3, "candidate_calls": 0, "receipt_emissions": 137, "images": 4, "vae_calls": 4, "physical_batch": 1}


def context_descriptor(value):
    if isinstance(value, np.ndarray):
        require(value.dtype == np.float32 and value.size and np.isfinite(value).all(), "context descriptor requires finite actual FP32 arrays")
        return {"shape": list(value.shape), "dtype": "float32",
            "sha256": hashlib.sha256(np.asarray(value, dtype="<f4", order="C").tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {key: context_descriptor(child) for key, child in value.items()}
    return [context_descriptor(child) for child in value]


def suffix_difference(native, bridge, predicted):
    result = TAIL["finite_difference"]([native, bridge, predicted])
    result["additional_suffix"] = result.pop("local")
    result["native_normalized_relative_rms"]["additional_suffix"] = result["native_normalized_relative_rms"].pop("local")
    result["scope"] = "P-N=(B-N)+(P-B) from a shared own-predicted joint4 closure. P-B accumulates the whole new suffix; not per-block local attribution."
    result["denominators"] = "Per-comparison inherited/total use N and additional_suffix uses B. Shared native_normalized_relative_rms uses the same N for all three terms."
    return result


def atomic_act(e, action, parent, expected):
    receipt = action["receipt"]
    require(receipt == e.receipts[action["receipt_id"]] and receipt["operation"] == "apply"
        and receipt["parent"] == action["parent"] == parent and receipt["result"] == action["post_act"]
        and receipt["act"] == action["act"] and set(action["act"]["writes"]) == set(PORTS)
        and set(action["act"]["reads"]) == set(ALL_PORTS)
        and action["act"]["operation"]["id"] == "mdb.flux.replace_final_latent_closure", "atomic final closure identity differs")
    fields = e.fields(parent)
    require(action["act"]["preconditions"] == {key: BASE["canonical_hash"](fields[key]) for key in ALL_PORTS},
        "atomic source/clock preconditions differ")
    values = OLD["described_mapping"](OLD["described_mapping"](action["act"]["parameters"])["value"])
    after = action["post_act"]
    require(e.fields(after) == {**fields, **values} and OLD["described_mapping"](receipt["writes"]) == values
        and e.point(after) == e.point(parent), "atomic closure changed undeclared state or clock")
    for name in PORTS:
        if expected[name] is None:
            actual = e.typed["unpack"](e.fields(after)[name])
        else:
            actual = cut_context(e, after) if name == "context" else e.slot(after, name)
        require(OLD["tree_exact"](actual, expected[name]), "Act did not install the sealed complete field")
    return after


def analyze(root):
    started = time.monotonic()
    root = Path(root).resolve()
    request, job, collection = [read(root / name) for name in ("request.json", "job.json", "collection.json")]
    data = root / "extracted/flux-final-latent-result"
    collector = runpy.run_path(str(HERE / "collect_flux_final_latent.py"))
    collector["validate_job"](job, job["job_id"], request)
    inventory = collector["inventory"](data)
    require(inventory == read(root / "collection-remote-before.json") == read(root / "collection-remote-after.json")
        == read(root / "collection-local-inventory.json") and collection["job_id"] == job["job_id"]
        and collection["request_sha256"] == request["request_sha256"]
        and collection["collector_source_sha256"] == sha(HERE / "collect_flux_final_latent.py")
        and all(sha(root / name) == digest for name, digest in collection["bindings"].items())
        and job == read(root / "collection-job.json"), "collection identity/source/custody differs")
    collector["verify_manifest"](data, inventory, job["job_id"], request)
    require(job["result"]["diagnostics"]["command"] == read(root / "admission-command.json")["command"], "executed command differs")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "request changed")
    for name, digest in request["source_hashes"].items():
        require(sha(H["file_path"](root / "payload", name)) == digest, "payload source changed")
    for name, digest in request["independent_math_sources"].items():
        require(sha(root / "independent-math-freeze" / name) == sha(HERE / name) == digest, "pre-outcome independent source changed")
    source, summary, plan, binding, prefix, segment = [read(data / name) for name in
        ("source.json", "summary.json", "plan.json", "binding.json", "prefix.json", "native-segment.json")]
    require(plan == request["config"] and summary["counts"] == COUNTS and plan["context"]["seed"] == 107
        and source["context"] == plan["context"] and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["source_hashes"] == request["source_hashes"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["checkpoint_sha256"] == BASE["canonical_hash"](source["checkpoint_files"]) == plan["expected_checkpoint_sha256"],
        "plan/model/source identity differs")
    require(source["execution"]["adapter"] == "mdb-flux-final-latent-closure-v1"
        and source["execution"]["device"]["type"] == "cuda" and source["execution"]["dtype"] == "torch.bfloat16", "actual backend differs")
    frozen = read(data / "plan-frozen.json")
    require(frozen["plan_sha256"] == sha(data / "plan.json") and frozen["source_identity"] == BASE["canonical_hash"](request["source_hashes"]), "plan freeze changed")
    syntax = ast.parse((root / "payload/flux_reentry_reference.py").read_text())
    source_pins = ast.literal_eval(next(node.value for node in syntax.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_CONTRACT" for target in node.targets)))
    expected_abi = {key: value for key, value in source_pins.items() if key not in {"diffusers_version", "torch_qualification_version"}}
    for key in ("pre_model_source_abi", "native_source_abi"):
        ref = source[key]
        abi = read(data / ref["path"])
        prior_tail = abi["prior_joint_tail"]
        prior_chain = prior_tail["prior_joint_chain"]
        old = prior_chain["prior_native_and_closure"]
        require(sha(data / ref["path"]) == ref["sha256"] and abi["verified"] is True
            and abi["native_advance_inherited"] is True and old["reentry_actual"] == old["reentry_expected"] == expected_abi
            and old["closure_module_sha256"] == plan["expected_closure_module_sha256"]
            and prior_chain["joint_closure_module_sha256"] == plan["expected_joint_closure_module_sha256"]
            and prior_tail["tail_closure_module_sha256"] == plan["expected_tail_closure_module_sha256"]
            and abi["final_closure_module_sha256"] == plan["expected_final_closure_module_sha256"], "native source ABI differs")
    scalar = read(data / "cuda-scalar-qualification.json")
    require(scalar["qualified"] is True and scalar["device"] == "cuda" and scalar["model_calls"] == 0
        and len(scalar["cases"]) == 3
        and [case["dt_fp32"] for case in scalar["cases"]] == [float(np.float32(x)) for x in (-.33333334, -.17320508, .14012345)]
        and all(case["values"] == 2048 and case["native_dtype"] == "torch.bfloat16"
            and case["dt_device"].startswith("cuda") and case["noise_device"].startswith("cuda")
            and case["native_vs_bf16_dt_mismatches"] == 0 and case["native_vs_fp32_dt_mismatches"] > 0 for case in scalar["cases"]),
        "actual CUDA weak-scalar instrument absent")
    tx = H["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    report = read(data / "transaction-report.json")
    require(tx["phase"] == report["phase"] == "closed" and not report["failure"] and not report["violations"]
        and CHAIN["SHARED"]["transaction_verifications_match"](tx, summary["transaction_verification"]), "transaction did not close")
    tree = H["NumericalTree"](data / "transaction")
    decode = lambda name, field: tree.decode(read(data / "transaction" / name)[field])
    forecast = decode("forecasts/final_latent.json", "predictions")
    observed = decode("observations/native.json", "value")
    require(decode("frontiers/source_binding.json", "value") == binding
        and decode("forecasts/hypotheses.json", "predictions") == plan["hypotheses"], "sealed sources/hypotheses differ")
    e = BASE["SavedEvidence"](data)
    require(all(cut["model_identity"] == source["model_identity"] and cut["execution"] == source["execution"] for cut in e.store["cuts"].values())
        and all(receipt["model_identity"] == source["model_identity"] and receipt["execution"] == source["execution"] for receipt in e.receipts.values()), "typed identities differ")
    prefix_ids = BASE["path"](e, prefix["receipts"], prefix["initial"], prefix["pre6"], 68)
    pre = prefix["pre6"]
    require(pre == binding["input_cut"] == segment["input_cut"], "prefix/forecast/reference join differs")
    from mdb.tensor_archive import load_tensor_archive
    require(exact(e.array(source["encoding"]["conditioning"]), e.slot(prefix["initial"], "conditioning"))
        and exact(load_tensor_archive(data, source["encoding"]["txt_ids"]), e.slot(prefix["initial"], "txt_ids")), "clean conditioning/IDs differ")
    reference_ids = BASE["path"](e, segment["receipts"], pre, segment["native_final"]["cut"], 44)
    for index, record in enumerate(segment["boundaries"]):
        require(record["index"] == index and record["receipt_id"] == reference_ids[index]
            and record["cut"] == segment["receipts"][index]["result"], "reference boundary lineage differs")
        boundary(e, record)
    tail = forecast["prefix"]
    chain, first = tail["prefix"], tail["prefix"]["prefix"]
    remaining, independent_z3 = OLD["audit_remaining"](e, binding, first["remaining_step"], segment, plan)
    spec = binding["project"]
    require(exact(e.array(spec["conditioning"]), e.slot(pre, "conditioning"))
        and exact(load_tensor_archive(data, spec["img_ids"]), e.slot(pre, "img_ids"))
        and exact(load_tensor_archive(data, spec["txt_ids"]), e.slot(pre, "txt_ids"))
        and exact(e.array(spec["timestep"]), e.slot(pre, "timesteps")[3]), "next-project carried sources/clock differ")
    weights = parameters(e, spec)
    project = NEW["independent_project"](independent_z3, e.array(spec["conditioning"]), load_tensor_archive(data, spec["img_ids"]),
        load_tensor_archive(data, spec["txt_ids"]), e.array(spec["timestep"]), weights,
        axes_dims_rope=spec["axes_dims_rope"], rope_theta=spec["rope_theta"], timestep_channels=spec["timestep_channels"])
    del weights
    require(first["project"]["input_latent_descriptor"] == first["remaining_step"]["commit"]["latent_descriptor"], "project refreshed latent")
    new_math = {}
    native_project = boundary(e, segment["native_project"])
    new_math["project"] = {key: {"sealed_minus_independent": compare(first["project"][key], project[key]),
        "predicted_minus_native": compare(first["project"][key], native_project[key])} for key in ("image", "text")}
    new_math["project"]["context"] = {key: compare(first["project"]["context"][key], project["context"][key])
        for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    new_math["project"]["rope"] = [compare(a, b) for a, b in zip(first["project"]["context"]["rope"], project["context"]["rope"])]
    new_math["project"]["context_predicted_minus_native"] = {key: compare(first["project"]["context"][key], native_project["context"][key])
        for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    new_math["project"]["rope_predicted_minus_native"] = [compare(a, b)
        for a, b in zip(first["project"]["context"]["rope"], native_project["context"]["rope"])]
    predictions = {"0": first["joint0"], "1": chain["joint1"], **tail["joints"]}
    require(set(predictions) == {str(index) for index in range(5)}, "all-five-joint forecast inventory differs")
    incoming, incoming_prediction = project, first["project"]
    for index in range(5):
        name, predicted = "joint" + str(index), predictions[str(index)]
        spec = binding[name]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["layernorm_eps"], spec["qk_eps"]) == (24, 128, 1e-6, 1e-6)
            and predicted["input_image_descriptor"] == incoming_prediction["image_descriptor"]
            and predicted["input_text_descriptor"] == incoming_prediction["text_descriptor"], "joint inputs/parameter occurrence differ")
        if index >= 2:
            require(tail["input_sources"][str(index)] ==
                f"own complete predicted joint{index - 1} I/T and own predicted project context", "tail input lineage declaration differs")
        weights = parameters(e, spec)
        result = NEW["independent_joint"](incoming["image"], incoming["text"], project["context"], weights,
            heads=spec["heads"], layernorm_eps=spec["layernorm_eps"], qk_eps=spec["qk_eps"])
        del weights
        native = boundary(e, segment["native_post_joint" + str(index)])
        new_math[name] = {key: {"sealed_minus_independent": compare(predicted[key], result[key]),
            "predicted_minus_native": compare(predicted[key], native[key])} for key in ("image", "text")}
        incoming, incoming_prediction = result, predicted
        print(json.dumps({"phase": "independent_joint", "block": index}), flush=True)
        gc.collect()
    final = forecast["final_step"]
    current = np.concatenate((predictions["4"]["text"], predictions["4"]["image"]), axis=1)
    independent = np.concatenate((incoming["text"], incoming["image"]), axis=1)
    require(final["chain_length"] == len(final["steps"]) == 20
        and final["input_descriptor"] == MATH["identity"](current)
        and final["context_source"] == "own predicted project context", "final suffix source/length differs")
    require(final["context_input_descriptors"] == context_descriptor(first["project"]["context"]),
        "final suffix refreshed its own predicted context")
    require(set(binding["final_blocks"]) == {str(index) for index in range(20)}, "final parameter inventory differs")
    final_math = []
    for index in range(20):
        spec, entry = binding["final_blocks"][str(index)], final["steps"][index]
        predicted = entry["prediction"]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"single_transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["hidden_width"], spec["mlp_width"], spec["layernorm_eps"], spec["qk_eps"])
                == (24, 128, 3072, 9216, 1e-6, 1e-6), "final block numerical/source profile differs")
        if index >= 6:
            require(all(spec[key] == binding["blocks"][str(index)][key] for key in ("parameters", "native_parameters")),
                "repeated single bank was rebound instead of reused")
        require(entry["step"] == entry["block_index"] == index
            and entry["input_source"] == ("own complete predicted joint4 I/T" if index == 0 else "previous complete predicted output")
            and entry["input_descriptor"] == predicted["input_descriptor"] == MATH["identity"](current)
            and exact(predicted["sampled"]["input"], current[:, plan["sample_indices"]])
            and predicted["output_descriptor"] == MATH["identity"](predicted["output"]), "final suffix replaced a predicted predecessor")
        weights = BASE["load_parameters"](e, spec, BASE["PARAMS"])
        independent = CHAIN["independent_transition"](independent, project["context"]["single_mod"], weights,
            heads=24, layernorm_eps=1e-6, qk_eps=1e-6, rope_cos=project["context"]["rope"][0], rope_sin=project["context"]["rope"][1])
        del weights
        actual = boundary(e, segment["boundaries"][22 + index])
        native = np.concatenate((actual["text"], actual["image"]), axis=1)
        current = predicted["output"]
        final_math.append({"block": index, "sealed_minus_independent": compare(current, independent),
            "predicted_minus_native": {"both": compare(current, native), "text": compare(current[:, :512], native[:, :512]),
                "image": compare(current[:, 512:], native[:, 512:])}})
        print(json.dumps({"phase": "independent_final_single", "block": index,
            "changed_values": final_math[-1]["sealed_minus_independent"]["bitwise_mismatch_count"]}), flush=True)
        gc.collect()
    readout = binding["readout"]
    ports = {**{"norm_out." + key: value for key, value in readout["native_parameters"].items()},
        **{"proj_out." + key: value for key, value in readout["projection_native_parameters"].items()}}
    names = {"modulation_weight": "norm_out.linear.weight", "modulation_bias": "norm_out.linear.bias",
        "projection_weight": "proj_out.weight", "projection_bias": "proj_out.bias"}
    weights = BASE["load_parameters"](e, readout, names, ports)
    independent_noise = MATH["independent_readout"](independent[:, 512:], project["context"]["temb"], **weights, eps=1e-6)["noise"]
    del weights
    require(final["output_descriptor"] == MATH["identity"](current)
        and final["readout"]["input_descriptor"] == MATH["identity"](current[:, 512:])
        and final["readout"]["noise_descriptor"] == MATH["identity"](final["readout"]["noise"]), "final readout joins differ")
    sigma, next_sigma = [e.array(binding["final_commit"][key]) for key in ("sigma", "sigma_next")]
    schedule = e.slot(pre, "sigmas")
    require(binding["final_commit"]["step_index"] == 3 and exact(sigma, schedule[3]) and exact(next_sigma, schedule[4])
        and float(next_sigma) == 0.0, "final Euler source/clock differs")
    independent_final = MATH["independent_flow_commit"](independent_z3, independent_noise, sigma, next_sigma)["latent"]
    require(final["commit"]["input_descriptor"] == MATH["identity"](first["remaining_step"]["commit"]["latent"])
        and final["commit"]["noise_descriptor"] == final["readout"]["noise_descriptor"]
        and final["commit"]["latent_descriptor"] == MATH["identity"](final["commit"]["latent"])
        and final["next_state_contract"]["clock"] == {"step": 4, "phase": "project", "cursor": 0}
        and final["next_state_contract"]["reset"] == {"context": None, "image": None, "text": None, "noise_prediction": None},
        "final Euler own-Z3 or terminal reset differs")
    native_noise = boundary(e, segment["boundaries"][42])["noise_prediction"]
    native_z4 = boundary(e, segment["native_final"])["latent"]
    final_stage = {"blocks": final_math,
        "readout": {"sealed_minus_independent": compare(final["readout"]["noise"], independent_noise),
            "predicted_minus_native": compare(final["readout"]["noise"], native_noise)},
        "commit": {"sealed_minus_independent": compare(final["commit"]["latent"], independent_final),
            "predicted_minus_native": compare(final["commit"]["latent"], native_z4)}}
    expected_emissions = [("prefix", rid) for rid in prefix_ids] + [("reference", rid) for rid in reference_ids]
    rows, pixels, latents = {}, {}, {}
    for name in ORDER:
        row = read(data / "rows" / (name + ".json"))
        require(row == next(value for value in observed["rows"] if value["id"] == name), "observed future row differs")
        parent = segment["common_post_joint4" if name == "bridge.predicted" else "native_final"]["cut"]
        require(row["parent"] == row["entry"]["cut"] == parent and len(row["acts"]) == (name != "native")
            and row["continuation_receipts"] == [], "future parent/Act/native continuation differs")
        boundary(e, row["entry"])
        current_cut = parent
        for action in row["acts"]:
            expected = ({"latent": e.slot(parent, "latent"), "context": None, "image": None, "text": None} if name == "sham" else
                {"latent": first["remaining_step"]["commit"]["latent"], "context": first["project"]["context"],
                 "image": predictions["4"]["image"], "text": predictions["4"]["text"]} if name == "bridge.predicted" else
                {"latent": final["commit"]["latent"], "context": None, "image": None, "text": None})
            current_cut = atomic_act(e, action, parent, expected)
            expected_emissions.append((name, action["receipt_id"]))
            require(row["installed"]["cut"] == current_cut, "installed capture differs")
            boundary(e, row["installed"])
        length = 22 if name == "bridge.predicted" else 0
        if length:
            require(row["common_post_joint4"] == row["installed"], "bridge common input differs from complete installed prediction")
        bridge_ids = BASE["path"](e, row["bridge_receipts"], current_cut, row["final_cut"], length)
        require(len(row["bridge_boundaries"]) == len(row["bridge_records"]) == length, "bridge full-array census differs")
        for index, record in enumerate(row["bridge_records"]):
            require(record["index"] == index and record["before"]["cut"] == current_cut
                and record["after"] == row["bridge_boundaries"][index]
                and record["receipt"] == row["bridge_receipts"][index]
                and record["receipt_id"] == bridge_ids[index]
                and record["after"]["cut"] == record["receipt"]["result"], "bridge full input/output lineage differs")
            before, after = boundary(e, record["before"]), boundary(e, record["after"])
            require(e.point(record["after"]["cut"]) == e.point(segment["boundaries"][22 + index]["cut"]), "matched suffix clocks differ")
            if index < 21:
                require(exact(after["latent"], first["remaining_step"]["commit"]["latent"])
                    and OLD["tree_exact"](after["context"], first["project"]["context"]), "bridge refreshed carried own-state Z3/context")
            if index == 0:
                require(OLD["tree_exact"](before["context"], first["project"]["context"])
                    and exact(before["latent"], first["remaining_step"]["commit"]["latent"])
                    and all(exact(before[key], predictions["4"][key]) for key in ("image", "text")), "whole suffix initial operands are not matched")
            current_cut = record["after"]["cut"]
        expected_emissions.extend((name, rid) for rid in bridge_ids)
        actual_final = boundary(e, row["terminal_final"])
        require(row["terminal_final"]["cut"] == row["final_cut"] == current_cut
            and tuple(e.point(current_cut)[key] for key in ("logical_step", "phase", "index")) == (4, "project", None)
            and all(actual_final[key] is None for key in ("context", "image", "text", "noise_prediction")), "terminal reset/clock differs")
        latents[name], pixels[name] = e.array(row["final_latent"]), e.png(row["image"])
        require(exact(latents[name], actual_final["latent"]), "actual VAE input latent differs")
        rows[name] = row
    pcut, bcut = rows["full.predicted"]["final_cut"], rows["bridge.predicted"]["final_cut"]
    require(e.point(pcut) == e.point(bcut)
        and all(e.fields(pcut)[key] == e.fields(bcut)[key] for key in e.fields(pcut) if key != "latent"),
        "matched final VAE consumers differ outside latent")
    observations = read(data / "continuation-observations.json")
    require(observations["plan"] == plan["observation_plan"] and observations["model_calls_added"] == 0
        and observations["branch_execution_order"] == ["native", "bridge.predicted"]
        and set(observations["rows"]) == {"native", "bridge.predicted"}, "full suffix observation plan differs")
    for name in ("native", "bridge.predicted"):
        trajectory = observations["rows"][name]
        records = ([segment["common_post_joint4"], *segment["boundaries"][22:]] if name == "native" else
            [rows[name]["common_post_joint4"], *rows[name]["bridge_boundaries"]])
        ids = ([reference_ids[21], *reference_ids[22:]] if name == "native" else
            [rows[name]["acts"][0]["receipt_id"], *[value["fingerprint"] for value in rows[name]["bridge_receipts"]]])
        require(len(trajectory) == len(records) == len(ids) == 23, "full observer boundary count differs")
        for index, (item, record, rid) in enumerate(zip(trajectory, records, ids, strict=True)):
            require(item["index"] == index and item["incoming_receipt"] == rid and item["complete_cut_retained"] is True
                and all(item[key] == record[key] for key in ("cut", "point", "closure", "clock")), "full observer lineage differs")
            boundary(e, item)
    suffix_curve = []
    bridge = rows["bridge.predicted"]
    predicted_values = [np.concatenate((predictions["4"]["text"], predictions["4"]["image"]), axis=1)]
    predicted_values += [entry["prediction"]["output"] for entry in final["steps"]]
    predicted_values += [final["readout"]["noise"], final["commit"]["latent"]]
    for index, predicted in enumerate(predicted_values):
        nrecord = segment["common_post_joint4"] if index == 0 else segment["boundaries"][21 + index]
        brecord = bridge["common_post_joint4"] if index == 0 else bridge["bridge_boundaries"][index - 1]
        nvalue, bvalue = boundary(e, nrecord), boundary(e, brecord)
        role = "working_image_and_text" if index <= 20 else "readout_noise" if index == 21 else "committed_final_latent"
        narray = np.concatenate((nvalue["text"], nvalue["image"]), axis=1) if index <= 20 else nvalue["noise_prediction" if index == 21 else "latent"]
        barray = np.concatenate((bvalue["text"], bvalue["image"]), axis=1) if index <= 20 else bvalue["noise_prediction" if index == 21 else "latent"]
        streams = ({key: suffix_difference(nvalue[key], bvalue[key], predicted[:, :512] if key == "text" else predicted[:, 512:])
            for key in ("image", "text")} if index <= 20 else {})
        suffix_curve.append({"index": index, "role": role, "native_cut": nrecord["cut"], "bridge_cut": brecord["cut"],
            "point": nrecord["point"], "decomposition": suffix_difference(narray, barray, predicted), "streams": streams})
    emissions = CHAIN["SHARED"]["read_jsonl"](data / "receipt-emissions.jsonl")
    require([value["sequence"] for value in emissions] == list(range(137))
        and [(value["branch"], value["receipt"]) for value in emissions] == expected_emissions
        and set(rid for _, rid in expected_emissions) == set(e.receipts), "physical native/Act receipt census differs")
    expected_targets = {**{str(index): 2 if index < 6 else 3 for index in range(20)}, "norm_out": 3, "proj_out": 3,
        **{name: 1 for name in ("x_embedder", "context_embedder", "time_guidance_embed", "double_stream_modulation_img",
            "double_stream_modulation_txt", "single_stream_modulation")}, "pos_embed": 2,
        **{"joint" + str(index): 1 for index in range(5)}}
    require(summary["target_calls"] == expected_targets, "native module guard census differs")
    timeline = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    sealed = [value for value in timeline if value["kind"] == "final_latent_sealed"]
    references = [value for value in timeline if value["kind"] == "reference_transition_complete"]
    require(len(sealed) == 1 and len(references) == 44 and sealed[0]["blocked_entries"] == 0
        and sealed[0]["seal"] == summary["seal"] and set(sealed[0]["target_entries_before_seal"].values()) == {0}
        and max(value["sequence"] for value in timeline if value["kind"] in
            {"block_forecast_complete", "project_forecast_complete", "joint_forecast_complete", "joint1_forecast_complete",
             "joint_tail_forecast_complete", "final_single_forecast_complete", "final_readout_commit_forecast_complete"}) < sealed[0]["sequence"]
        and sealed[0]["sequence"] < min(value["sequence"] for value in references), "selected native target executed before full forecast seal")
    controls = {"sham_final_latent_exact_native": exact(latents["sham"], latents["native"]),
        "sham_png_exact_native": rows["sham"]["image"]["sha256"] == rows["native"]["image"]["sha256"]}
    require(controls == summary["control_observations"] == observed["control_observations"], "control report differs")
    endpoints = {}
    for name in ORDER:
        delta = pixels[name].astype(np.float64) - pixels["native"]
        endpoints[name] = {"latent": compare(latents[name], latents["native"]),
            "rgb": {"mae": float(np.abs(delta).mean()), "rms": float(np.sqrt(np.mean(delta * delta))),
                "signed_mean": float(delta.mean()), "max_abs": float(np.abs(delta).max()),
                "changed_value_count": int(np.count_nonzero(delta)), "value_count": int(delta.size)}}
    rgb_decomposition = suffix_difference(pixels["native"], pixels["bridge.predicted"], pixels["full.predicted"])
    rgb_decomposition["scope"] = "Ordered finite differences of actual native VAE RGB outputs. P-B is the common native VAE response to the final-latent discrepancy; no independently predicted RGB or VAE-kernel numerical error is inferred."
    rgb_decomposition["units"] = "RGB channel levels0..255; native-relative RMS divides by native RGB RMS, separate from all latent and working-stream denominators."
    return {"schema": "mdb-flux-final-latent-audit-v1", "mechanics_status": "valid",
        "instrument_status": "applicable; matched whole final suffix, exact terminal reset and independent self-fed recurrence",
        "trend_status": "observed; signed numerical discrepancies and actual VAE outcomes retained", "terminal_status": "unassessed",
        "job_id": job["job_id"], "request_sha256": request["request_sha256"], "seal": summary["seal"], "counts": COUNTS,
        "remaining_step": remaining, "reentry_math": new_math, "final_step": final_stage, "endpoint_comparisons": endpoints,
        "final_latent_decomposition": suffix_difference(latents["native"], latents["bridge.predicted"], latents["full.predicted"]),
        "final_rgb_decomposition": rgb_decomposition,
        "suffix_curve": suffix_curve, "control_observations": controls,
        "audit_seconds": time.monotonic() - started, "models_executed": False, "target_calls": summary["target_calls"],
        "collection": {"files": inventory["file_count"], "bytes": inventory["expanded_bytes"], "sha256": sha(root / "collection.json")},
        "independent_math_sources": request["independent_math_sources"], "auditor_source_sha256": sha(Path(__file__)),
        "scope": "Fresh seed107 observed step2/pre6;44 own-state transitions predict final Z4 before all selected targets. Four actual native VAE outputs. Only bridge.predicted runs a native22-step suffix. Whole-suffix additional discrepancy, not individual-block local attribution. No raw-prompt or independent RGB forecast, universal dictionary or speedup claim."}


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
