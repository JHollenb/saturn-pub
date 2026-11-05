"""Saved-only independent audit of a sealed cross-step FLUX state forecast.

Custody, source, typed closure and declared execution are integrity checks.
Approximation errors and downstream effects remain signed observations.
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
BASE = runpy.run_path(str(HERE / "analyze_flux_step_boundary.py"))
NEW = runpy.run_path(str(HERE / "flux_reentry_audit_math.py"))
CHAIN, MATH, H = (BASE[key] for key in ("CHAIN", "MATH", "H"))
require, exact, sha, read = (BASE[key] for key in ("require", "exact_bits", "sha", "read"))
PORTS = ("latent", "context", "image", "text")
ALL_PORTS = (*PORTS, "step", "phase", "cursor", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas", "noise_prediction")
ORDER = ["native", "sham", "latent.predicted", "project.predicted", "joint.predicted"]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 219,
    "acts": 4, "candidate_calls": 0, "receipt_emissions": 223, "images": 5, "vae_calls": 5, "physical_batch": 1}


def compare(actual, reference):
    a, r = np.asarray(actual), np.asarray(reference)
    require(a.shape == r.shape and a.size and np.isfinite(a).all() and np.isfinite(r).all(), "comparison geometry/finite differs")
    same_type = a.dtype == r.dtype
    mismatches = (int(np.count_nonzero(np.ascontiguousarray(a).view("V" + str(a.dtype.itemsize))
        != np.ascontiguousarray(r).view("V" + str(r.dtype.itemsize)))) if same_type else None)
    wide_a, wide_r = a.astype(np.float64), r.astype(np.float64)
    delta = wide_a - wide_r
    energy, ref_energy = float(np.sum(delta * delta)), float(np.sum(wide_r * wide_r))
    return {"shape": list(a.shape), "sample_count": int(a.size), "bitwise_mismatch_count": mismatches,
        "bitwise_exact": mismatches == 0 if same_type else False, "l2": float(np.sqrt(energy)),
        "rms": float(np.sqrt(energy / a.size)), "mae": float(np.abs(delta).mean()),
        "signed_mean": float(delta.mean()), "max_abs": float(np.abs(delta).max()),
        "reference_rms": float(np.sqrt(ref_energy / a.size)), "actual_rms": float(np.sqrt(np.mean(wide_a * wide_a))),
        "relative_rms": float(np.sqrt(energy / ref_energy)) if ref_energy else None,
        "sign": "actual/predicted minus reference; zero denominator is undefined, not failed"}


def tree_array(e, value):
    if isinstance(value, dict) and "path" in value and "descriptor" in value:
        return e.array(value)
    if isinstance(value, dict):
        return {key: tree_array(e, item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(tree_array(e, item) for item in value)
    require(value is None, "unexpected closure archive leaf")
    return None


def tree_exact(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and exact(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(tree_exact(a, b) for a, b in zip(left, right))
    return left is None and right is None


def described_mapping(value):
    """Remove one mapping wrapper while retaining every nested typed node."""
    require(isinstance(value, dict) and value.get("kind") == "mapping", "expected typed mapping")
    items = value["items"]
    require(all(item["key"].get("kind") == "scalar" and item["key"].get("type") == "str" for item in items),
        "typed mapping needs string keys")
    result = {item["key"]["value"]: item["value"] for item in items}
    require(len(result) == len(items), "typed mapping has duplicate keys")
    return result


def cut_context(e, cut):
    nested = CHAIN["PREVIOUS"]["nested_slot_array"]
    return {**{key: nested(e, cut, "context", (key,)) for key in ("temb", "img_mod", "txt_mod", "single_mod")},
        "rope": tuple(nested(e, cut, "context", ("rope", index)) for index in (0, 1))}


def boundary(e, record):
    cut = record["cut"]
    require(set(record["closure"]) == {"latent", "context", "image", "text", "noise_prediction"}
        and record["point"] == e.point(cut)
        and record["clock"] == {key: e.typed["unpack"](e.fields(cut)[key]) for key in ("step", "phase", "cursor")},
        "closure archive clock differs")
    result = tree_array(e, record["closure"])
    for key, value in result.items():
        if key == "context" and value is not None:
            require(tree_exact(value, cut_context(e, cut)), "archived complete context differs from typed cut")
        elif value is None:
            require(e.typed["unpack"](e.fields(cut)[key]) is None, "archived absent field differs")
        else:
            require(exact(value, e.slot(cut, key)), "archived tensor differs from typed cut")
    return result


def parameters(e, spec):
    require(spec["parameters"].keys() == spec["native_parameters"].keys(), "new native parameter inventory differs")
    result = {}
    for name, archive in spec["parameters"].items():
        require(all(archive["descriptor"][key] == spec["native_parameters"][name][key] for key in ("shape", "dtype", "sha256")),
            "parameter archive is not the named native port")
        result[name] = MATH["exact_bf16"](e.array(archive))
    return result


def audit_remaining(e, binding, forecast, segment, plan):
    """Every independent block feeds the next, including its readout and commit."""
    pre = binding["input_cut"]
    for slot, archive in binding["inputs"].items():
        require(exact(e.array(archive), e.slot(pre, slot)), "observed-prefix operand differs")
    context = cut_context(e, pre)
    require(exact(e.array(binding["context"]["single_mod"]), context["single_mod"])
        and exact(e.array(binding["context"]["temb"]), context["temb"]), "prefix time context differs")
    for index, name in enumerate(("cos", "sin")):
        require(exact(e.array(binding["rope"][name]), context["rope"][index]), "prefix rotary context differs")
    current = np.concatenate((e.slot(pre, "text"), e.slot(pre, "image")), axis=1)
    independent = current
    require(forecast["chain_length"] == len(forecast["steps"]) == 14
        and forecast["input_descriptor"] == MATH["identity"](current), "remaining-chain geometry differs")
    block_rows = []
    for position, index in enumerate(range(6, 20)):
        spec, entry = binding["blocks"][str(index)], forecast["steps"][position]
        predicted = entry["prediction"]
        require(entry["step"] == position and entry["block_index"] == index
            and entry["input_source"] == ("observed native step2/pre6" if position == 0 else "previous complete predicted output")
            and entry["input_descriptor"] == predicted["input_descriptor"] == MATH["identity"](current)
            and exact(predicted["sampled"]["input"], current[:, plan["sample_indices"]])
            and predicted["output_descriptor"] == MATH["identity"](predicted["output"]), "chain uses another intermediate state")
        require((spec["heads"], spec["head_dim"], spec["hidden_width"], spec["mlp_width"], spec["layernorm_eps"], spec["qk_eps"])
            == (24, 128, 3072, 9216, 1e-6, 1e-6), "selected single numerical profile differs")
        p = BASE["load_parameters"](e, spec, BASE["PARAMS"])
        independent = CHAIN["independent_transition"](independent, context["single_mod"], p,
            heads=24, layernorm_eps=1e-6, qk_eps=1e-6, rope_cos=context["rope"][0], rope_sin=context["rope"][1])
        del p
        actual = boundary(e, segment["boundaries"][position])
        native = np.concatenate((actual["text"], actual["image"]), axis=1)
        current = predicted["output"]
        block_rows.append({"block": index, "sealed_minus_independent": compare(current, independent),
            "predicted_minus_native": {"both": compare(current, native), "text": compare(current[:, :512], native[:, :512]),
                "image": compare(current[:, 512:], native[:, 512:])}})
        print(json.dumps({"phase": "independent_single", "block": index,
            "changed_values": block_rows[-1]["sealed_minus_independent"]["bitwise_mismatch_count"]}), flush=True)
        gc.collect()
    readout = binding["readout"]
    ports = {**{"norm_out." + k: v for k, v in readout["native_parameters"].items()},
        **{"proj_out." + k: v for k, v in readout["projection_native_parameters"].items()}}
    names = {"modulation_weight": "norm_out.linear.weight", "modulation_bias": "norm_out.linear.bias",
        "projection_weight": "proj_out.weight", "projection_bias": "proj_out.bias"}
    require(not readout["elementwise_affine"] and readout["eps"] == 1e-6, "readout norm profile differs")
    p = BASE["load_parameters"](e, readout, names, ports)
    noise = MATH["independent_readout"](independent[:, 512:], context["temb"], **p, eps=1e-6)["noise"]
    del p
    require(forecast["readout"]["input_descriptor"] == MATH["identity"](current[:, 512:])
        and forecast["readout"]["noise_descriptor"] == MATH["identity"](forecast["readout"]["noise"]), "readout joins differ")
    sigma, next_sigma = [e.array(binding["commit"][key]) for key in ("sigma", "sigma_next")]
    schedule = e.slot(pre, "sigmas")
    require(exact(sigma, schedule[2]) and exact(next_sigma, schedule[3]), "Euler schedule differs")
    committed = MATH["independent_flow_commit"](e.slot(pre, "latent"), noise, sigma, next_sigma)["latent"]
    native_noise = boundary(e, segment["boundaries"][14])["noise_prediction"]
    native_z3 = boundary(e, segment["boundaries"][15])["latent"]
    require(forecast["commit"]["input_descriptor"] == MATH["identity"](e.slot(pre, "latent"))
        and forecast["commit"]["noise_descriptor"] == forecast["readout"]["noise_descriptor"]
        and forecast["commit"]["latent_descriptor"] == MATH["identity"](forecast["commit"]["latent"]), "commit joins differ")
    return {"blocks": block_rows, "readout": {"sealed_minus_independent": compare(forecast["readout"]["noise"], noise),
        "predicted_minus_native": compare(forecast["readout"]["noise"], native_noise)},
        "commit": {"sealed_minus_independent": compare(forecast["commit"]["latent"], committed),
            "predicted_minus_native": compare(forecast["commit"]["latent"], native_z3)}}, committed


def atomic_act(e, action, parent, expected):
    receipt = action["receipt"]
    require(receipt == e.receipts[action["receipt_id"]] and receipt["operation"] == "apply"
        and receipt["parent"] == action["parent"] == parent and receipt["result"] == action["post_act"]
        and receipt["act"] == action["act"] and set(action["act"]["writes"]) == set(PORTS)
        and set(action["act"]["reads"]) == set(ALL_PORTS)
        and action["act"]["operation"]["id"] == "mdb.flux.replace_forecast_closure", "atomic closure Act identity differs")
    fields = e.fields(parent)
    require(action["act"]["preconditions"] == {key: BASE["canonical_hash"](fields[key]) for key in ALL_PORTS},
        "atomic source/clock preconditions differ")
    values = described_mapping(described_mapping(action["act"]["parameters"])["value"])
    after = action["post_act"]
    require(e.fields(after) == {**fields, **values} and described_mapping(receipt["writes"]) == values
        and e.point(after) == e.point(parent), "atomic closure changed undeclared state or clock")
    for name in PORTS:
        actual = cut_context(e, after) if name == "context" else e.slot(after, name)
        require(tree_exact(actual, expected[name]), "atomic closure did not install the sealed complete field")
    return after


def finite_difference(values):
    """Exact ordered identity in F64, with no claim of unique attribution."""
    native, latent, project, predicted = [np.asarray(value, np.float64) for value in values]
    terms = [latent - native, project - latent, predicted - project]
    total = predicted - native
    residual = total - sum(terms)
    summary = {name: compare(term, np.zeros_like(term)) for name, term in zip(
        ("inherited_latent", "project_addition", "joint_addition"), terms)}
    cross = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dot = float(np.sum(terms[i] * terms[j]))
        norm = float(np.linalg.norm(terms[i].ravel()) * np.linalg.norm(terms[j].ravel()))
        cross[str(i) + ":" + str(j)] = {"dot": dot, "cosine": dot / norm if norm else None}
    summary.update(total=compare(predicted, native), max_abs_identity_residual=float(np.abs(residual).max()),
        pairwise_cross_terms=cross,
        ordered_scope="P-X=(A-X)+(B-A)+(P-B); finite differences for this replacement order, not semantic mediation fractions")
    return summary


def audit_observer(e, rows, segment, plan):
    record = read(e.data / "continuation-observations.json")
    require(record["plan"] == plan["observation_plan"] and record["branch_execution_order"] == ORDER
        and record["model_calls_added"] == 0 and set(record["rows"]) == set(ORDER), "continuation instrument differs")
    full, report = {}, {}
    witnesses = CHAIN["PREVIOUS"]["witness_values"]
    for name in ORDER:
        row, trajectory = rows[name], record["rows"][name]
        require(len(trajectory) == 27, "continuation observation census differs")
        incoming = (row["bridge_receipts"][-1]["fingerprint"] if row["bridge_receipts"] else
            row["acts"][-1]["receipt_id"] if row["acts"] else segment["receipts"][-1]["fingerprint"])
        report[name] = []
        for index, item in enumerate(trajectory):
            cut = row["common_post_joint0"]["cut"] if index == 0 else row["continuation_receipts"][index - 1]["result"]
            rid = incoming if index == 0 else row["continuation_receipts"][index - 1]["fingerprint"]
            require(item["index"] == index and item["cut"] == cut and item["point"] == e.point(cut)
                and item["incoming_receipt"] == rid and item["complete_cut_retained"] == (index in {0, 26})
                and set(item["slots"]) == {"image", "text", "latent", "noise_prediction"}
                and set(item["comparisons"]) == {"native"}, "observer lineage/storage policy differs")
            measurements = {}
            for slot, spec in item["slots"].items():
                native_spec = record["rows"]["native"][index]["slots"][slot]
                require(type(spec["present"]) is bool and spec["present"] == native_spec["present"], "observer presence profile differs")
                if cut in e.store["cuts"]:
                    require(spec["present"] == (e.fields(cut)[slot]["kind"] == "tensor"), "retained observer presence differs")
                if not spec["present"]:
                    require("array" not in spec and "witness" not in spec, "absent observer field has invented values")
                    continue
                actual = e.slot(cut, slot) if cut in e.store["cuts"] else None
                require(spec["descriptor"]["dtype"] == native_spec["descriptor"]["dtype"] == "torch.bfloat16"
                    and spec["descriptor"]["shape"] == native_spec["descriptor"]["shape"], "observer dtype/shape differs")
                if actual is not None:
                    require(e.typed["matches"](actual, spec["descriptor"], bf16=True), "retained observer descriptor differs")
                if name == "native":
                    archived = e.array(spec["array"])
                    require(e.typed["matches"](archived, spec["descriptor"], bf16=True)
                        and (actual is None or exact(actual, archived)), "native observer array/descriptor differs")
                    full[index, slot] = actual = archived
                else:
                    require("array" not in spec, "candidate intermediate array exceeds witness-only policy")
                witness = witnesses(spec, actual)
                if slot == "latent" and index < 26:
                    held = e.slot(row["common_post_joint0"]["cut"], "latent")
                    require(e.typed["matches"](held, spec["descriptor"], bf16=True), "stored Z3 changed before commit")
                    witnesses(spec, held)
                if slot in {"image", "text"}:
                    active = item["point"]["phase"] in {"joint", "single", "readout"}
                    require(spec["meaning"]["meaningful_propagation_observable"] == active
                        and spec["meaning"]["stale"] == (not active), "stale working tensor treated as propagation")
                if slot == "noise_prediction":
                    require(item["point"]["phase"] == "scheduler", "noise observed outside its readout-to-commit lifetime")
                reference = full[index, slot]
                reference_witness = witnesses(record["rows"]["native"][index]["slots"][slot], reference)
                observed = item["comparisons"]["native"]["slots"][slot]
                require(observed["sample_count"] == reference.size and 0 <= observed["changed_value_count"] <= reference.size,
                    "observer full metric denominator differs")
                measurements[slot] = {"worker_full_observation": observed,
                    "independent_fixed64_comparison": compare(witness, reference_witness),
                    "independent_full_comparison": compare(actual, reference) if actual is not None else None,
                    "meaning": spec["meaning"]}
            report[name].append({"index": index, "point": item["point"], "slots": measurements})
    return {"rows": report, "scope": "Native full trajectory arrays, complete retained candidate endpoints and64 fixed candidate witnesses. Candidate intermediate full norms are worker observations, not independently reconstructed complete states. Stored latent is constant before commit; working streams become absent or stale outside active phases."}


def analyze(root, *, correction=None):
    started = time.monotonic()
    root = Path(root).resolve()
    request, job, collection = [read(root / name) for name in ("request.json", "job.json", "collection.json")]
    data = root / "extracted/flux-reentry-result"
    collector = runpy.run_path(str(HERE / "collect_flux_reentry.py"))
    collector["validate_job"](job, job["job_id"], request)
    inventory = collector["inventory"](data)
    require(inventory == read(root / "collection-remote-before.json") == read(root / "collection-remote-after.json")
        == read(root / "collection-local-inventory.json")
        and collection["job_id"] == job["job_id"] and collection["request_sha256"] == request["request_sha256"]
        and collection["collector_source_sha256"] == sha(HERE / "collect_flux_reentry.py")
        and all(sha(root / name) == digest for name, digest in collection["bindings"].items())
        and job == read(root / "collection-job.json"), "collection identity/source/custody differs")
    collector["verify_manifest"](data, inventory, job["job_id"], request)
    require(job["result"]["diagnostics"]["command"] == read(root / "admission-command.json")["command"], "executed command differs")
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "request changed")
    for name, digest in request["source_hashes"].items():
        require(sha(H["file_path"](root / "payload", name)) == digest, "payload source changed")
    original_auditor = request["independent_math_sources"][Path(__file__).name]
    actual_auditor = sha(Path(__file__))
    repair = None
    if actual_auditor != original_auditor:
        require(correction is not None, "changed auditor requires an explicit correction receipt")
        correction = Path(correction).resolve()
        repaired = read(correction)
        require(repaired["schema"] == "mdb-flux-reentry-audit-correction-v1" and repaired["model_calls"] == 0
            and repaired["request_sha256"] == request["request_sha256"] and repaired["job_id"] == job["job_id"]
            and repaired["original_auditor_sha256"] == original_auditor and repaired["corrected_auditor_sha256"] == actual_auditor,
            "audit correction source/request binding differs")
        repair = {"path": str(correction), "sha256": sha(correction), "record": repaired}
    for name, digest in request["independent_math_sources"].items():
        require(sha(root / "independent-math-freeze" / name) == digest
            and (name == Path(__file__).name or sha(HERE / name) == digest), "pre-outcome independent source changed")
    source, summary, plan, binding, prefix, segment = [read(data / name) for name in
        ("source.json", "summary.json", "plan.json", "binding.json", "prefix.json", "native-segment.json")]
    require(plan == request["config"] and summary["counts"] == COUNTS and plan["context"]["seed"] == 97
        and source["context"] == plan["context"] and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["source_hashes"] == request["source_hashes"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["checkpoint_sha256"] == BASE["canonical_hash"](source["checkpoint_files"]) == plan["expected_checkpoint_sha256"],
        "scientific plan/model/source identity differs")
    require(source["execution"]["adapter"] == "mdb-flux-forecast-closure-v1"
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
        require(sha(data / ref["path"]) == ref["sha256"] and abi["verified"] is True
            and abi["native_advance_inherited"] is True and abi["reentry_actual"] == abi["reentry_expected"] == expected_abi
            and abi["closure_module_sha256"] == plan["expected_closure_module_sha256"], "native source ABI differs")
    scalar = read(data / "cuda-scalar-qualification.json")
    require(scalar["qualified"] is True and scalar["device"] == "cuda" and scalar["model_calls"] == 0
        and len(scalar["cases"]) == 3
        and [c["dt_fp32"] for c in scalar["cases"]] == [float(np.float32(x)) for x in (-.33333334, -.17320508, .14012345)]
        and all(c["values"] == 2048 and c["native_dtype"] == "torch.bfloat16"
            and c["dt_device"].startswith("cuda") and c["noise_device"].startswith("cuda")
            and c["native_vs_bf16_dt_mismatches"] == 0 and c["native_vs_fp32_dt_mismatches"] > 0 for c in scalar["cases"]),
        "actual CUDA weak-scalar fixture absent or inapplicable")
    tx = H["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    report = read(data / "transaction-report.json")
    require(tx["phase"] == report["phase"] == "closed" and not report["failure"] and not report["violations"]
        and CHAIN["SHARED"]["transaction_verifications_match"](tx, summary["transaction_verification"]), "transaction did not close cleanly")
    tree = H["NumericalTree"](data / "transaction")
    decode = lambda name, field: tree.decode(read(data / "transaction" / name)[field])
    forecast = decode("forecasts/reentry.json", "predictions")
    observed = decode("observations/native.json", "value")
    require(decode("frontiers/source_binding.json", "value") == binding
        and decode("forecasts/hypotheses.json", "predictions") == plan["hypotheses"], "sealed source/hypotheses differ")
    e = BASE["SavedEvidence"](data)
    require(all(c["model_identity"] == source["model_identity"] and c["execution"] == source["execution"] for c in e.store["cuts"].values())
        and all(r["model_identity"] == source["model_identity"] and r["execution"] == source["execution"] for r in e.receipts.values()), "typed execution identities differ")
    prefix_ids = BASE["path"](e, prefix["receipts"], prefix["initial"], prefix["pre6"], 68)
    require(prefix["pre6"] == binding["input_cut"] == segment["input_cut"], "prefix/forecast/reference join differs")
    require(exact(e.array(source["encoding"]["conditioning"]), e.slot(prefix["initial"], "conditioning")), "clean own conditioning differs")
    from mdb.tensor_archive import load_tensor_archive
    require(exact(load_tensor_archive(data, source["encoding"]["txt_ids"]), e.slot(prefix["initial"], "txt_ids")), "clean text IDs differ")
    reference_ids = BASE["path"](e, segment["receipts"], prefix["pre6"], segment["common_post_joint0"]["cut"], 18)
    for index, record in enumerate(segment["boundaries"]):
        require(record["index"] == index and record["receipt_id"] == reference_ids[index]
            and record["cut"] == segment["receipts"][index]["result"], "reference boundary lineage differs")
        boundary(e, record)
    remaining, independent_z3 = audit_remaining(e, binding, forecast["remaining_step"], segment, plan)
    project_spec, joint_spec = binding["project"], binding["joint0"]
    pre = prefix["pre6"]
    require(exact(e.array(project_spec["conditioning"]), e.slot(pre, "conditioning"))
        and exact(load_tensor_archive(data, project_spec["img_ids"]), e.slot(pre, "img_ids"))
        and exact(load_tensor_archive(data, project_spec["txt_ids"]), e.slot(pre, "txt_ids"))
        and exact(e.array(project_spec["timestep"]), e.slot(pre, "timesteps")[3]), "next-project carried source/clock differs")
    project_params = parameters(e, project_spec)
    project = NEW["independent_project"](independent_z3, e.array(project_spec["conditioning"]), load_tensor_archive(data, project_spec["img_ids"]),
        load_tensor_archive(data, project_spec["txt_ids"]), e.array(project_spec["timestep"]), project_params,
        axes_dims_rope=project_spec["axes_dims_rope"], rope_theta=project_spec["rope_theta"], timestep_channels=project_spec["timestep_channels"])
    del project_params
    require(forecast["project"]["input_latent_descriptor"] == forecast["remaining_step"]["commit"]["latent_descriptor"], "next project refreshed latent")
    require(forecast["joint0"]["input_image_descriptor"] == forecast["project"]["image_descriptor"]
        and forecast["joint0"]["input_text_descriptor"] == forecast["project"]["text_descriptor"], "joint0 refreshed working streams")
    gc.collect()
    joint_params = parameters(e, joint_spec)
    joint = NEW["independent_joint"](project["image"], project["text"], project["context"], joint_params,
        heads=joint_spec["heads"], layernorm_eps=joint_spec["layernorm_eps"], qk_eps=joint_spec["qk_eps"])
    del joint_params
    new_math = {}
    for stage, independent in (("project", project), ("joint0", joint)):
        native = boundary(e, segment["native_project" if stage == "project" else "common_post_joint0"])
        new_math[stage] = {key: {"sealed_minus_independent": compare(forecast[stage][key], independent[key]),
            "predicted_minus_native": compare(forecast[stage][key], native[key])} for key in ("image", "text")}
    new_math["project"]["context"] = {key: compare(forecast["project"]["context"][key], project["context"][key])
        for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    new_math["project"]["rope"] = [compare(a, b) for a, b in zip(forecast["project"]["context"]["rope"], project["context"]["rope"])]
    native_context = cut_context(e, segment["native_project"]["cut"])
    new_math["project"]["context_predicted_minus_native"] = {key: compare(forecast["project"]["context"][key], native_context[key])
        for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    new_math["project"]["rope_predicted_minus_native"] = [compare(a, b) for a, b in zip(forecast["project"]["context"]["rope"], native_context["rope"])]
    expected_emissions = [("prefix", rid) for rid in prefix_ids] + [("reference", rid) for rid in reference_ids]
    rows, common, pixels, latents = {}, {}, {}, {}
    for name in ORDER:
        row = read(data / "rows" / (name + ".json"))
        require(row == next(v for v in observed["rows"] if v["id"] == name), "observed future row differs")
        parent = segment["native_Z3" if name == "latent.predicted" else "native_project" if name == "project.predicted" else "common_post_joint0"]["cut"]
        require(row["parent"] == parent and len(row["acts"]) == (name != "native"), "future installation parent/count differs")
        current = parent
        for action in row["acts"]:
            if name == "latent.predicted":
                current = BASE["latent_act"](e, action, parent, forecast["remaining_step"]["commit"]["latent"])
            else:
                stage = forecast["project" if name == "project.predicted" else "joint0"]
                expected = ({"latent": e.slot(parent, "latent"), "image": e.slot(parent, "image"),
                    "text": e.slot(parent, "text"), "context": cut_context(e, parent)} if name == "sham" else
                    {"latent": forecast["remaining_step"]["commit"]["latent"], "image": stage["image"],
                     "text": stage["text"], "context": forecast["project"]["context"]})
                current = atomic_act(e, action, parent, expected)
            expected_emissions.append((name, action["receipt_id"]))
            boundary(e, row["installed"])
        bridge_length = 2 if name == "latent.predicted" else 1 if name == "project.predicted" else 0
        bridge_ids = BASE["path"](e, row["bridge_receipts"], current, row["common_post_joint0"]["cut"], bridge_length)
        require(len(row["bridge_boundaries"]) == bridge_length, "complete bridge census differs")
        for index, record in enumerate(row["bridge_boundaries"]):
            require(record["cut"] == row["bridge_receipts"][index]["result"], "bridge closure lineage differs")
            boundary(e, record)
        expected_emissions.extend((name, rid) for rid in bridge_ids)
        current = row["common_post_joint0"]["cut"]
        common[name] = boundary(e, row["common_post_joint0"])
        require(tuple(e.point(current)[k] for k in ("logical_step", "phase", "index")) == (3, "joint", 1), "future did not reach common joint1")
        ids = BASE["path"](e, row["continuation_receipts"], current, row["final_cut"], 26)
        expected_emissions.extend((name, rid) for rid in ids)
        require(tuple(e.point(row["final_cut"])[k] for k in ("logical_step", "phase", "index")) == (4, "project", None), "future did not commit Z4")
        latents[name], pixels[name] = e.array(row["final_latent"]), e.png(row["image"])
        require(exact(latents[name], e.slot(row["final_cut"], "latent")), "final latent differs from typed cut")
        rows[name] = row
    emissions = CHAIN["SHARED"]["read_jsonl"](data / "receipt-emissions.jsonl")
    require([v["sequence"] for v in emissions] == list(range(223))
        and [(v["branch"], v["receipt"]) for v in emissions] == expected_emissions
        and set(rid for _, rid in expected_emissions) == set(e.receipts), "physical native/Act census differs")
    phase_counts = {}
    for branch, rid in expected_emissions:
        receipt = e.receipts[rid]
        if branch == "prefix" or receipt["operation"] != "continue":
            continue
        point = e.point(receipt["parent"])
        key = (point["logical_step"], point["phase"], point["index"])
        phase_counts[key] = phase_counts.get(key, 0) + 1
    require(all(phase_counts.get((2, "single", index)) == 1 and phase_counts.get((3, "single", index)) == 5 for index in range(6, 20))
        and phase_counts.get((3, "project", None)) == 2 and phase_counts.get((3, "joint", 0)) == 3,
        "step-qualified target receipt census differs")
    expected_targets = {**{str(index): 6 for index in range(6, 20)}, "norm_out": 6, "proj_out": 6,
        **{name: 2 for name in ("x_embedder", "context_embedder", "time_guidance_embed", "double_stream_modulation_img",
            "double_stream_modulation_txt", "single_stream_modulation")}, "pos_embed": 4, "joint0": 3}
    require(summary["target_calls"] == expected_targets, "native module guard census differs")
    timeline = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    sealed = [v for v in timeline if v["kind"] == "reentry_sealed"]
    reference_events = [v for v in timeline if v["kind"] == "reference_transition_complete"]
    require(len(sealed) == 1 and len(reference_events) == 18 and sealed[0]["blocked_entries"] == 0
        and sealed[0]["seal"] == summary["seal"] and set(sealed[0]["target_entries_before_seal"].values()) == {0}
        and max(v["sequence"] for v in timeline if v["kind"] in {"block_forecast_complete", "project_forecast_complete", "joint_forecast_complete"}) < sealed[0]["sequence"]
        and sealed[0]["sequence"] < min(v["sequence"] for v in reference_events), "forecast seal preceded neither forecast completion nor targets")
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
        ("native", "latent.predicted", "project.predicted", "joint.predicted")]) for key in ("latent", "image", "text")}
    continuation = audit_observer(e, rows, segment, plan)
    return {"schema": "mdb-flux-reentry-audit-v1", "mechanics_status": "valid", "instrument_status": "applicable; atomic sham and independent recurrence retained",
        "trend_status": "observed; all signed numerical and endpoint effects retained", "terminal_status": "unassessed",
        "job_id": job["job_id"], "request_sha256": request["request_sha256"], "seal": summary["seal"], "counts": COUNTS,
        "remaining_step": remaining, "reentry_math": new_math, "endpoint_comparisons": endpoints,
        "common_joint1_decomposition": decomposition, "control_observations": control,
        "continuation": continuation,
        "audit_seconds": time.monotonic() - started, "models_executed": False, "target_calls": summary["target_calls"],
        "collection": {"files": inventory["file_count"], "bytes": inventory["expanded_bytes"], "sha256": sha(root / "collection.json")},
        "independent_math_sources": request["independent_math_sources"], "auditor_source_sha256": sha(Path(__file__)),
        "preoutcome_auditor_sha256": original_auditor, "audit_correction": repair,
        "scope": "One seed97/prompt, observed step2/pre6. Complete self-fed14 singles, readout, Euler, next project and joint0. Five native26-transition suffixes and actual VAE. No independent remaining next-step suffix/RGB or object-semantic/generalization/speedup claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--correction", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir, correction=args.correction)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("mechanics_status", "job_id", "control_observations", "audit_seconds")}))
