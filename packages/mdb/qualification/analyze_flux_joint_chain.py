"""Audit a sealed joint1 forecast and its matched native incoming-state bridge.

Saved arrays only. Source, lineage and closure checks protect the instrument;
signed numerical discrepancies remain observations, not discovery gates.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import runpy
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = runpy.run_path(str(HERE / "analyze_flux_reentry.py"))
BASE, H, CHAIN, MATH, NEW = (OLD[key] for key in ("BASE", "H", "CHAIN", "MATH", "NEW"))
require, exact, sha, read = (OLD[key] for key in ("require", "exact", "sha", "read"))
compare, boundary, cut_context, parameters = (OLD[key] for key in ("compare", "boundary", "cut_context", "parameters"))
PORTS, ALL_PORTS = OLD["PORTS"], OLD["ALL_PORTS"]
ORDER = ["native", "sham", "inherited.predicted", "joint1.predicted"]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 188,
    "acts": 3, "candidate_calls": 0, "receipt_emissions": 191, "images": 4, "vae_calls": 4, "physical_batch": 1}


def finite_difference(values):
    """Ordered vector identity P-N=(B-N)+(P-B), including cancellation."""
    native, bridge, predicted = [np.asarray(value, np.float64) for value in values]
    inherited, local = bridge - native, predicted - bridge
    total = predicted - native
    dot = float(np.sum(inherited * local))
    denominator = float(np.linalg.norm(inherited.ravel()) * np.linalg.norm(local.ravel()))
    native_rms = float(np.sqrt(np.mean(native * native)))
    return {"inherited": compare(bridge, native), "joint1_local": compare(predicted, bridge),
        "total": compare(predicted, native), "cross_dot": dot,
        "cross_cosine": dot / denominator if denominator else None,
        "native_normalized_relative_rms": {name: float(np.sqrt(np.mean(term * term)) / native_rms)
            if native_rms else None for name, term in (("inherited", inherited), ("joint1_local", local), ("total", total))},
        "denominators": "Per-comparison inherited/total relative RMS uses N, local uses B. native_normalized_relative_rms uses the same N for all three terms.",
        "squared_norm_identity_residual": float(np.sum(total * total)
            - np.sum(inherited * inherited) - np.sum(local * local) - 2 * dot),
        "max_abs_identity_residual": float(np.abs(total - inherited - local).max()),
        "scope": "Ordered finite differences at common joint2; no semantic attribution or monotonicity assumption."}


def atomic_act(e, action, parent, expected):
    receipt = action["receipt"]
    require(receipt == e.receipts[action["receipt_id"]] and receipt["operation"] == "apply"
        and receipt["parent"] == action["parent"] == parent and receipt["result"] == action["post_act"]
        and receipt["act"] == action["act"] and set(action["act"]["writes"]) == set(PORTS)
        and set(action["act"]["reads"]) == set(ALL_PORTS)
        and action["act"]["operation"]["id"] == "mdb.flux.replace_joint_forecast_closure",
        "new atomic joint closure identity differs")
    fields = e.fields(parent)
    require(action["act"]["preconditions"] == {key: BASE["canonical_hash"](fields[key]) for key in ALL_PORTS},
        "atomic source/clock preconditions differ")
    values = OLD["described_mapping"](OLD["described_mapping"](action["act"]["parameters"])["value"])
    after = action["post_act"]
    require(e.fields(after) == {**fields, **values} and OLD["described_mapping"](receipt["writes"]) == values
        and e.point(after) == e.point(parent), "atomic closure changed undeclared state or clock")
    for name in PORTS:
        actual = cut_context(e, after) if name == "context" else e.slot(after, name)
        require(OLD["tree_exact"](actual, expected[name]), "Act did not install the sealed complete field")
    return after


def audit_observer(e, rows, segment, plan):
    record = read(e.data / "continuation-observations.json")
    require(record["plan"] == plan["observation_plan"] and record["branch_execution_order"] == ORDER
        and record["model_calls_added"] == 0 and set(record["rows"]) == set(ORDER), "observer instrument differs")
    full, report = {}, {}
    witnesses = CHAIN["PREVIOUS"]["witness_values"]
    for name in ORDER:
        row, trajectory = rows[name], record["rows"][name]
        require(len(trajectory) == 26, "observer boundary census differs")
        incoming = (row["bridge_receipts"][-1]["fingerprint"] if row["bridge_receipts"] else
            row["acts"][-1]["receipt_id"] if row["acts"] else segment["receipts"][-1]["fingerprint"])
        report[name] = []
        for index, item in enumerate(trajectory):
            cut = row["common_post_joint1"]["cut"] if index == 0 else row["continuation_receipts"][index - 1]["result"]
            rid = incoming if index == 0 else row["continuation_receipts"][index - 1]["fingerprint"]
            require(item["index"] == index and item["cut"] == cut and item["point"] == e.point(cut)
                and item["incoming_receipt"] == rid and item["complete_cut_retained"] == (index in {0, 25})
                and set(item["slots"]) == {"image", "text", "latent", "noise_prediction"}
                and set(item["comparisons"]) == {"native"}, "observer lineage/storage policy differs")
            measurements = {}
            for slot, spec in item["slots"].items():
                native_spec = record["rows"]["native"][index]["slots"][slot]
                require(type(spec["present"]) is bool and spec["present"] == native_spec["present"], "observer presence differs")
                if cut in e.store["cuts"]:
                    require(spec["present"] == (e.fields(cut)[slot]["kind"] == "tensor"), "retained observer presence differs")
                if not spec["present"]:
                    require("array" not in spec and "witness" not in spec, "absent field has invented values")
                    continue
                actual = e.slot(cut, slot) if cut in e.store["cuts"] else None
                require(spec["descriptor"]["dtype"] == native_spec["descriptor"]["dtype"] == "torch.bfloat16"
                    and spec["descriptor"]["shape"] == native_spec["descriptor"]["shape"], "observer dtype/shape differs")
                if actual is not None:
                    require(e.typed["matches"](actual, spec["descriptor"], bf16=True), "retained descriptor differs")
                if name == "native":
                    archived = e.array(spec["array"])
                    require(e.typed["matches"](archived, spec["descriptor"], bf16=True)
                        and (actual is None or exact(actual, archived)), "native observer array differs")
                    full[index, slot] = actual = archived
                else:
                    require("array" not in spec, "candidate intermediate exceeds witness-only policy")
                witness = witnesses(spec, actual)
                if slot == "latent" and index < 25:
                    held = e.slot(row["common_post_joint1"]["cut"], "latent")
                    require(e.typed["matches"](held, spec["descriptor"], bf16=True), "stored Z3 changed before commit")
                    witnesses(spec, held)
                if slot in {"image", "text"}:
                    active = item["point"]["phase"] in {"joint", "single", "readout"}
                    require(spec["meaning"]["meaningful_propagation_observable"] == active
                        and spec["meaning"]["stale"] == (not active), "stale tensor treated as propagation")
                if slot == "noise_prediction":
                    require(item["point"]["phase"] == "scheduler", "noise observed outside readout-to-commit lifetime")
                reference = full[index, slot]
                reference_witness = witnesses(record["rows"]["native"][index]["slots"][slot], reference)
                observed = item["comparisons"]["native"]["slots"][slot]
                require(observed["sample_count"] == reference.size and 0 <= observed["changed_value_count"] <= reference.size,
                    "observer metric denominator differs")
                measurements[slot] = {"worker_full_observation": observed,
                    "independent_fixed64_comparison": compare(witness, reference_witness),
                    "independent_full_comparison": compare(actual, reference) if actual is not None else None,
                    "meaning": spec["meaning"]}
            report[name].append({"index": index, "point": item["point"], "slots": measurements})
    return {"rows": report, "scope": "Native full trajectory, retained candidate endpoints,64 fixed candidate witnesses. Intermediate full candidate metrics are worker observations."}


def analyze(root):
    started = time.monotonic()
    root = Path(root).resolve()
    request, job, collection = [read(root / name) for name in ("request.json", "job.json", "collection.json")]
    data = root / "extracted/flux-joint-chain-result"
    collector = runpy.run_path(str(HERE / "collect_flux_joint_chain.py"))
    collector["validate_job"](job, job["job_id"], request)
    inventory = collector["inventory"](data)
    require(inventory == read(root / "collection-remote-before.json") == read(root / "collection-remote-after.json")
        == read(root / "collection-local-inventory.json") and collection["job_id"] == job["job_id"]
        and collection["request_sha256"] == request["request_sha256"]
        and collection["collector_source_sha256"] == sha(HERE / "collect_flux_joint_chain.py")
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
    require(plan == request["config"] and summary["counts"] == COUNTS and plan["context"]["seed"] == 101
        and source["context"] == plan["context"] and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["source_hashes"] == request["source_hashes"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["checkpoint_sha256"] == BASE["canonical_hash"](source["checkpoint_files"]) == plan["expected_checkpoint_sha256"],
        "plan/model/source identity differs")
    require(source["execution"]["adapter"] == "mdb-flux-joint-forecast-closure-v1"
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
        old = abi["prior_native_and_closure"]
        require(sha(data / ref["path"]) == ref["sha256"] and abi["verified"] is True
            and abi["native_advance_inherited"] is True and old["reentry_actual"] == old["reentry_expected"] == expected_abi
            and old["closure_module_sha256"] == plan["expected_closure_module_sha256"]
            and abi["joint_closure_module_sha256"] == plan["expected_joint_closure_module_sha256"], "native source ABI differs")
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
    forecast = decode("forecasts/joint_chain.json", "predictions")
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
    reference_ids = BASE["path"](e, segment["receipts"], pre, segment["common_post_joint1"]["cut"], 19)
    for index, record in enumerate(segment["boundaries"]):
        require(record["index"] == index and record["receipt_id"] == reference_ids[index]
            and record["cut"] == segment["receipts"][index]["result"], "reference boundary lineage differs")
        boundary(e, record)
    first = forecast["prefix"]
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
    native_project = boundary(e, segment["boundaries"][16])
    new_math["project"] = {key: {"sealed_minus_independent": compare(first["project"][key], project[key]),
        "predicted_minus_native": compare(first["project"][key], native_project[key])} for key in ("image", "text")}
    new_math["project"]["context"] = {key: compare(first["project"]["context"][key], project["context"][key])
        for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    new_math["project"]["rope"] = [compare(a, b) for a, b in zip(first["project"]["context"]["rope"], project["context"]["rope"])]
    new_math["project"]["context_predicted_minus_native"] = {key: compare(first["project"]["context"][key], native_project["context"][key])
        for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    new_math["project"]["rope_predicted_minus_native"] = [compare(a, b)
        for a, b in zip(first["project"]["context"]["rope"], native_project["context"]["rope"])]
    incoming, incoming_prediction = project, first["project"]
    for index in (0, 1):
        name = "joint" + str(index)
        spec = binding[name]
        predicted = first[name] if index == 0 else forecast[name]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["layernorm_eps"], spec["qk_eps"]) == (24, 128, 1e-6, 1e-6)
            and predicted["input_image_descriptor"] == incoming_prediction["image_descriptor"]
            and predicted["input_text_descriptor"] == incoming_prediction["text_descriptor"], "joint inputs/parameter occurrence differ")
        weights = parameters(e, spec)
        result = NEW["independent_joint"](incoming["image"], incoming["text"], project["context"], weights,
            heads=spec["heads"], layernorm_eps=spec["layernorm_eps"], qk_eps=spec["qk_eps"])
        del weights
        native = boundary(e, segment["native_post_joint0" if index == 0 else "common_post_joint1"])
        new_math[name] = {key: {"sealed_minus_independent": compare(predicted[key], result[key]),
            "predicted_minus_native": compare(predicted[key], native[key])} for key in ("image", "text")}
        incoming, incoming_prediction = result, predicted
        print(json.dumps({"phase": "independent_joint", "block": index}), flush=True)
        gc.collect()
    require(forecast["joint1_input_source"] == "own complete predicted joint0 I/T and own predicted project context", "joint1 lineage declaration differs")
    expected_emissions = [("prefix", rid) for rid in prefix_ids] + [("reference", rid) for rid in reference_ids]
    rows, common, pixels, latents = {}, {}, {}, {}
    for name in ORDER:
        row = read(data / "rows" / (name + ".json"))
        require(row == next(value for value in observed["rows"] if value["id"] == name), "observed future row differs")
        parent = segment["native_post_joint0" if name == "inherited.predicted" else "common_post_joint1"]["cut"]
        require(row["parent"] == parent and len(row["acts"]) == (name != "native"), "future installation parent/count differs")
        current = parent
        for action in row["acts"]:
            stage = first["joint0"] if name == "inherited.predicted" else forecast["joint1"]
            expected = ({"latent": e.slot(parent, "latent"), "image": e.slot(parent, "image"),
                "text": e.slot(parent, "text"), "context": cut_context(e, parent)} if name == "sham" else
                {"latent": first["remaining_step"]["commit"]["latent"], "image": stage["image"],
                 "text": stage["text"], "context": first["project"]["context"]})
            current = atomic_act(e, action, parent, expected)
            expected_emissions.append((name, action["receipt_id"]))
            boundary(e, row["installed"])
        bridge_length = int(name == "inherited.predicted")
        bridge_ids = BASE["path"](e, row["bridge_receipts"], current, row["common_post_joint1"]["cut"], bridge_length)
        require(len(row["bridge_boundaries"]) == len(row["bridge_records"]) == bridge_length, "bridge census differs")
        for index, record in enumerate(row["bridge_boundaries"]):
            require(record["cut"] == row["bridge_receipts"][index]["result"], "bridge lineage differs")
            boundary(e, record)
            bridge_record = row["bridge_records"][index]
            require(bridge_record["receipt_id"] == bridge_ids[index]
                and bridge_record["receipt"] == row["bridge_receipts"][index]
                and bridge_record["before"]["cut"] == current and bridge_record["after"] == record,
                "full bridge source/result record differs")
            boundary(e, bridge_record["before"])
            current = record["cut"]
        expected_emissions.extend((name, rid) for rid in bridge_ids)
        current = row["common_post_joint1"]["cut"]
        common[name] = boundary(e, row["common_post_joint1"])
        require(tuple(e.point(current)[key] for key in ("logical_step", "phase", "index")) == (3, "joint", 2), "future did not reach joint2")
        ids = BASE["path"](e, row["continuation_receipts"], current, row["final_cut"], 25)
        expected_emissions.extend((name, rid) for rid in ids)
        require(tuple(e.point(row["final_cut"])[key] for key in ("logical_step", "phase", "index")) == (4, "project", None), "future did not commit Z4")
        latents[name], pixels[name] = e.array(row["final_latent"]), e.png(row["image"])
        require(exact(latents[name], e.slot(row["final_cut"], "latent")), "final latent differs")
        rows[name] = row
    # Native joint1 consumes the sealed joint0 streams and exactly the same
    # predicted project context/Z3 used by mathematical joint1. Its one bridge
    # transition may update only working streams and cursor.
    bridge_input = rows["inherited.predicted"]["installed"]["cut"]
    bridge_output = rows["inherited.predicted"]["common_post_joint1"]["cut"]
    predicted_output = rows["joint1.predicted"]["installed"]["cut"]
    unchanged = [key for key in e.fields(bridge_input) if key not in {"image", "text", "cursor"}]
    require(all(e.fields(bridge_input)[key] == e.fields(bridge_output)[key] for key in unchanged)
        and all(e.fields(bridge_output)[key] == e.fields(predicted_output)[key]
            for key in e.fields(bridge_output) if key not in {"image", "text"}), "matched bridge changed another input/context/clock")
    emissions = CHAIN["SHARED"]["read_jsonl"](data / "receipt-emissions.jsonl")
    require([value["sequence"] for value in emissions] == list(range(191))
        and [(value["branch"], value["receipt"]) for value in emissions] == expected_emissions
        and set(rid for _, rid in expected_emissions) == set(e.receipts), "native/Act census differs")
    expected_targets = {**{str(index): 5 for index in range(6, 20)}, "norm_out": 5, "proj_out": 5,
        **{name: 1 for name in ("x_embedder", "context_embedder", "time_guidance_embed", "double_stream_modulation_img",
            "double_stream_modulation_txt", "single_stream_modulation")}, "pos_embed": 2, "joint0": 1, "joint1": 2}
    require(summary["target_calls"] == expected_targets, "native module guard census differs")
    timeline = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    sealed = [value for value in timeline if value["kind"] == "joint_chain_sealed"]
    references = [value for value in timeline if value["kind"] == "reference_transition_complete"]
    require(len(sealed) == 1 and len(references) == 19 and sealed[0]["blocked_entries"] == 0
        and sealed[0]["seal"] == summary["seal"] and set(sealed[0]["target_entries_before_seal"].values()) == {0}
        and max(value["sequence"] for value in timeline if value["kind"] in
            {"block_forecast_complete", "project_forecast_complete", "joint_forecast_complete", "joint1_forecast_complete"}) < sealed[0]["sequence"]
        and sealed[0]["sequence"] < min(value["sequence"] for value in references), "target executed before prediction seal")
    control = {"sham_final_latent_exact_native": exact(latents["sham"], latents["native"]),
        "sham_png_exact_native": rows["sham"]["image"]["sha256"] == rows["native"]["image"]["sha256"]}
    require(control == summary["control_observations"] == observed["control_observations"], "control report differs")
    endpoints = {}
    for name in ORDER:
        delta = pixels[name].astype(np.float64) - pixels["native"]
        endpoints[name] = {"latent": compare(latents[name], latents["native"]),
            "common": {key: compare(common[name][key], common["native"][key]) for key in ("latent", "image", "text")},
            "rgb": {"mae": float(np.abs(delta).mean()), "rms": float(np.sqrt(np.mean(delta * delta))),
                "signed_mean": float(delta.mean()), "max_abs": float(np.abs(delta).max()),
                "changed_value_count": int(np.count_nonzero(delta)), "value_count": int(delta.size)}}
    decomposition = {key: finite_difference([common[name][key] for name in
        ("native", "inherited.predicted", "joint1.predicted")]) for key in ("latent", "image", "text")}
    return {"schema": "mdb-flux-joint-chain-audit-v1", "mechanics_status": "valid",
        "instrument_status": "applicable; complete matched native bridge, atomic sham and independent self-fed recurrence",
        "trend_status": "observed; signed numerical and endpoint effects retained", "terminal_status": "unassessed",
        "job_id": job["job_id"], "request_sha256": request["request_sha256"], "seal": summary["seal"], "counts": COUNTS,
        "remaining_step": remaining, "reentry_math": new_math, "endpoint_comparisons": endpoints,
        "common_joint2_decomposition": decomposition, "control_observations": control,
        "matched_bridge": {"input_cut": bridge_input, "output_cut": bridge_output,
            "predicted_output_cut": predicted_output, "nonworking_fields_exact": True, "native_steps": 1},
        "continuation": audit_observer(e, rows, segment, plan),
        "audit_seconds": time.monotonic() - started, "models_executed": False, "target_calls": summary["target_calls"],
        "collection": {"files": inventory["file_count"], "bytes": inventory["expanded_bytes"], "sha256": sha(root / "collection.json")},
        "independent_math_sources": request["independent_math_sources"], "auditor_source_sha256": sha(Path(__file__)),
        "scope": "One fresh seed101/prompt, observed step2/pre6. Nineteen self-fed transitions through project/joint0/joint1; four native25-transition suffixes plus actual VAE. No full prompt-to-image independent execution, semantic dictionary, universal family or speedup claim."}


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
