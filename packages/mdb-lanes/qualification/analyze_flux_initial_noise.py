"""Independent saved-only initial-noise112 and observed-Z2 matched56 audit.

No producer or native framework forward/replay is executed. The independently
pinned NumPy program recomputes all168 transitions using checkpoint parameters;
the legacy models_executed flag refers to native forwards, not this arithmetic.
Numerical discrepancies are observations; custody, clocks and own-input joins
are mechanics.
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
TWO = runpy.run_path(str(HERE / "analyze_flux_two_pass.py"))
BASE, H, CHAIN, MATH, NEW, OLD, PREV = (TWO[key] for key in ("BASE", "H", "CHAIN", "MATH", "NEW", "OLD", "PREV"))
require, exact, sha, read, compare, boundary = (TWO[key] for key in ("require", "exact", "sha", "read", "compare", "boundary"))
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 112,
    "acts": 3, "candidate_calls": 0, "receipt_emissions": 115, "images": 4, "vae_calls": 4, "physical_batch": 1}
ORDER = ("full.predicted", "matched.predicted", "native", "sham")
SLOTS = ("latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")
TARGETS = (*map(str, range(20)), "norm_out", "proj_out", "x_embedder", "context_embedder", "time_guidance_embed",
    "double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation", "pos_embed",
    *("joint" + str(index) for index in range(5)))


def geometry(origin):
    require(type(origin) is int and origin in (0, 2), "unknown forecast origin")
    return {"origin": origin, "logical_steps": list(range(origin, 4)), "transitions": 28 * (4 - origin),
        "native_offset": 28 * origin, "terminal_step": 4}


def point_after(index):
    require(type(index) is int and 0 <= index < 112, "native transition index outside finite horizon")
    logical, local = divmod(index, 28)
    if local < 5:
        return logical, "joint", local
    if local < 25:
        return logical, "single", local - 5
    if local == 25:
        return logical, "readout", None
    if local == 26:
        return logical, "scheduler", None
    return logical + 1, "project", None


def guard_contract(history, seal_counts, image_counts, seal_targets, image_targets):
    """A second prospective boundary preserves all earlier physical work."""
    require(history in ("full", "matched"), "unknown guard history")
    earlier, acts = (0, 0) if history == "full" else (56, 1)
    expected = {"adapter_steps": earlier, "acts": acts, "receipt_emissions": earlier + acts,
        "images": acts, "vae_calls": acts}
    require(all(type(seal_counts.get(k)) is int and seal_counts[k] == v for k, v in expected.items()),
        "seal erased historical work or consumed future transitions")
    expected.update(acts=acts + 1, receipt_emissions=earlier + acts + 1, images=acts + 1, vae_calls=acts + 1)
    require(all(type(image_counts.get(k)) is int and image_counts[k] == v for k, v in expected.items()),
        "pre-reference image physical work differs")
    logical = range(0 if history == "full" else 2, 4)
    require(set(seal_targets) == set(image_targets) == {str(i) for i in logical}
        and all(set(targets) == set(TARGETS) and all(type(v) is int and v == 0 for v in targets.values())
            for targets in (*seal_targets.values(), *image_targets.values())),
        "clock-qualified future target executed before image")
    return {"history": history, "earlier_native_transitions": earlier, "pre_reference_image_counts": image_counts,
        "scope": "Clock-qualified future counters; earlier calls remain in the monotonic physical ledger."}


def check_origin(e, origin, logical_step):
    require(origin["point"] == e.point(origin["cut"])
        and tuple(origin["point"][k] for k in ("logical_step", "phase", "index")) == (logical_step, "project", None),
        "forecast origin is not its declared native project clock")
    require(set(origin["inputs"]) == set(SLOTS), "source operand inventory differs")
    inputs = {key: e.array(ref) for key, ref in origin["inputs"].items()}
    require(all(exact(value, e.slot(origin["cut"], key)) for key, value in inputs.items()), "observed origin operand differs")
    require(inputs["timesteps"].shape == (4,) and inputs["sigmas"].shape == (5,)
        and float(inputs["sigmas"][-1]) == 0.0
        and all(e.typed["unpack"](e.fields(origin["cut"])[key]) is None
            for key in ("context", "image", "text", "noise_prediction")), "origin schedule/reset differs")
    return inputs


def audit_pass(e, binding, predicted, schedule, independent_latent, segment, logical_step, conditioning, img_ids,
               txt_ids, sealed_input_latent):
    """One pass uses our preceding computed state, never a native next field."""
    require(predicted["logical_step"] == logical_step and predicted["chain_length"] == 28
        and len(predicted["single_steps"]) == 20 and set(predicted["joints"]) == {str(i) for i in range(5)},
        "forecast pass inventory/clock differs")
    require(predicted["input_latent_descriptor"] == predicted["project"]["input_latent_descriptor"] == MATH["identity"](sealed_input_latent),
        "project refreshed its own incoming sealed latent")
    context = predicted["project"]["context"]
    require(predicted["context_input_descriptors"] == PREV["context_descriptor"](context), "own project context differs")
    result = {"logical_step": logical_step, "transitions": []}
    sealed_register = sealed_input_latent
    image = text = None
    events = TWO["independent_pass_events"](independent_latent, conditioning, img_ids, txt_ids, schedule,
        binding, lambda kind, spec: TWO["load_bank"](e, kind, spec))
    for local, (kind, index, independent) in enumerate(events):
        record = segment["boundaries"][28 * logical_step + local]
        native = boundary(e, record)
        scalar = {"native_transition_index": 28 * logical_step + local, "point": record["point"],
            "native_cut": record["cut"], "kind": kind, "index": index}
        if kind == "project":
            value = predicted["project"]
            require(value["conditioning_descriptor"] == MATH["identity"](conditioning), "project conditioning refreshed")
            scalar["fields"] = {key: {"sealed_minus_independent": compare(value[key], independent[key]),
                "predicted_minus_native": compare(value[key], native[key])} for key in ("image", "text")}
            scalar["context"] = {key: {"sealed_minus_independent": compare(context[key], independent["context"][key]),
                "predicted_minus_native": compare(context[key], native["context"][key])}
                for key in ("temb", "img_mod", "txt_mod", "single_mod")}
            scalar["rope"] = [{"sealed_minus_independent": compare(a, b), "predicted_minus_native": compare(a, c)}
                for a, b, c in zip(context["rope"], independent["context"]["rope"], native["context"]["rope"], strict=True)]
            image, text = value["image"], value["text"]
        elif kind == "joint":
            value = predicted["joints"][str(index)]
            require(value["input_image_descriptor"] == MATH["identity"](image)
                and value["input_text_descriptor"] == MATH["identity"](text), "joint refreshed predicted I/T")
            scalar["fields"] = {key: {"sealed_minus_independent": compare(value[key], independent[key]),
                "predicted_minus_native": compare(value[key], native[key])} for key in ("image", "text")}
            image, text = value["image"], value["text"]
            if index == 4:
                sealed_register = np.concatenate((text, image), axis=1)
        elif kind == "single":
            entry = predicted["single_steps"][index]
            value = entry["prediction"]
            require(entry["block_index"] == entry["step"] == index
                and entry["input_descriptor"] == value["input_descriptor"] == MATH["identity"](sealed_register)
                and value["output_descriptor"] == MATH["identity"](value["output"]), "single refreshed predicted register")
            sealed_register = value["output"]
            scalar["fields"] = {"both": {"sealed_minus_independent": compare(sealed_register, independent["output"]),
                "predicted_minus_native": compare(sealed_register, np.concatenate((native["text"], native["image"]), axis=1))}}
            scalar["streams"] = {key: compare(part, native[key]) for key, part in
                (("text", sealed_register[:, :conditioning.shape[1]]), ("image", sealed_register[:, conditioning.shape[1]:]))}
        elif kind == "readout":
            value = predicted["readout"]
            require(value["input_descriptor"] == MATH["identity"](sealed_register[:, conditioning.shape[1]:])
                and value["noise_descriptor"] == MATH["identity"](value["noise"]), "readout refreshed predicted I")
            scalar["fields"] = {"noise": {"sealed_minus_independent": compare(value["noise"], independent["noise"]),
                "predicted_minus_native": compare(value["noise"], native["noise_prediction"])}}
        else:
            require(kind == "commit" and local == 27, "pass operation sequence differs")
            value = predicted["commit"]
            require(value["input_descriptor"] == MATH["identity"](sealed_input_latent)
                and value["noise_descriptor"] == predicted["readout"]["noise_descriptor"]
                and value["latent_descriptor"] == MATH["identity"](value["latent"]), "Euler refreshed its own operands")
            require(predicted["next_state_contract"]["clock"] == {"step": logical_step + 1, "phase": "project", "cursor": 0}
                and predicted["next_state_contract"]["reset"] ==
                {"context": None, "image": None, "text": None, "noise_prediction": None}
                and predicted["next_state_contract"]["carried_state"] == binding["carried_state"], "pass clock/reset differs")
            scalar["fields"] = {"latent": {"sealed_minus_independent": compare(value["latent"], independent["latent"]),
                "predicted_minus_native": compare(value["latent"], native["latent"])}}
            independent_latent = independent["latent"]
        result["transitions"].append(scalar)
        print(json.dumps({"phase": "independent_initial_noise", "logical_step": logical_step, "kind": kind, "index": index}), flush=True)
        gc.collect()
    require(len(result["transitions"]) == 28, "pass stopped before complete own-state commit")
    return result, independent_latent


def audit_history(e, binding, forecast, segment, *, origin):
    spec = geometry(origin)
    require(forecast["origin_logical_step"] == origin and forecast["chain_length"] == spec["transitions"]
        and len(forecast["passes"]) == len(spec["logical_steps"]), "history inventory differs")
    inputs = check_origin(e, binding["origins"]["full" if origin == 0 else "matched"], origin)
    independent, sealed = inputs["latent"], inputs["latent"]
    passes = []
    for logical, prediction in zip(spec["logical_steps"], forecast["passes"], strict=True):
        schedule = {key: e.array(binding["pass_schedules"][str(logical)][key]) for key in ("timestep", "sigma", "sigma_next")}
        require(exact(schedule["timestep"], inputs["timesteps"][logical])
            and exact(schedule["sigma"], inputs["sigmas"][logical])
            and exact(schedule["sigma_next"], inputs["sigmas"][logical + 1]), "history schedule borrowed a future clock")
        result, independent = audit_pass(e, binding, prediction, schedule, independent, segment, logical,
            inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"], sealed)
        passes.append(result)
        sealed = prediction["commit"]["latent"]
    return {"origin_logical_step": origin, "transitions": spec["transitions"], "passes": passes}, independent


def history_binding(binding, history, snapshot):
    origin = 0 if history == "full" else 2
    require(history in ("full", "matched"), "unknown sealed source history")
    return {**{key: value for key, value in binding.items() if key != "origins"}, "history": history,
        "logical_steps": geometry(origin)["logical_steps"], "origin": binding["origins"][history],
        "counts_before_seal": snapshot["counts"],
        **{key: snapshot[key] for key in ("target_calls", "target_calls_by_step", "clock_qualified_targets_remaining")}}


def parameter_cache(data, cache, *, checkpoint=None, ssh=None):
    """Hydrate only an explicit external cache; never mutate producer evidence."""
    from mdb_lanes.parameter_artifacts import restore, validate_manifest, verify_present

    data, cache = Path(data), Path(cache)
    require(not cache.resolve().is_relative_to(data.resolve()), "parameter cache must be outside immutable result")
    manifest_path = data / "parameter-artifacts.json"
    manifest = validate_manifest(read(manifest_path))
    status = verify_present(manifest, cache)
    restoration = None
    if not status["complete"]:
        require(checkpoint is not None, "parameter cache incomplete: provide --checkpoint or a verified --parameter-root")
        restoration = restore(manifest, cache, checkpoint, ssh=ssh)
        status = verify_present(manifest, cache)
    require(status["complete"], "parameter hydration incomplete")
    return {entry["path"]: entry for entry in manifest["artifacts"]}, {
        "manifest_sha256": sha(manifest_path), "root": str(cache.resolve()), "verification": status,
        "restoration": restoration, "scope": "Exact NPY cache reconstructed from pinned checkpoint recipes. No learned activation or model call; producer result tree unchanged."}


class Evidence(BASE["SavedEvidence"]):
    def __init__(self, data, cache, recipes):
        super().__init__(data)
        self.parameter_root, self.recipes = Path(cache), recipes

    def array(self, record):
        from mdb_lanes.tensor_archive import load_tensor_archive

        if record.get("storage") != "checkpoint-recipe-v1":
            if "encoding" not in record:
                return super().array(record)
            require(record["encoding"] in {"bf16-bits-v1", "native-npy-v1"}, "unsupported modern tensor encoding")
            value = load_tensor_archive(self.data, record)
            require(value.size and np.isfinite(value).all(), "modern native tensor archive is empty or nonfinite")
            return value
        recipe = self.recipes.get(record["path"])
        require(recipe is not None and recipe["sha256"] == record["sha256"]
            and recipe["source"]["shape"] == record["stored_shape"]
            and recipe["source"]["dtype"] == "BF16" and record["descriptor"]["dtype"] == "torch.bfloat16"
            and recipe["source"]["payload_sha256"] == record["descriptor"]["sha256"]
            and record["parameter_tensor"] in recipe.get("checkpoint_tensor_aliases", [recipe["source"]["tensor"]]),
            "external parameter is not the exact recipe/native field")
        return load_tensor_archive(self.parameter_root, record)


def validate_plan(plan, source, summary):
    require(plan["context"]["seed"] == 113 and plan["context"]["id"] == "p1-s113"
        and source["context"] == plan["context"] and plan["logical_steps"] == [0, 1, 2, 3]
        and plan["matched_logical_steps"] == [2, 3] and plan["block_indices"] == list(range(20))
        and plan["selector"] == {"logical_step": 0, "phase": "project", "index": None}
        and plan["matched_selector"] == {"logical_step": 2, "phase": "project", "index": None}
        and plan["branch_execution_order"] == list(ORDER)
        and BASE["canonical_hash"](summary["counts"]) == BASE["canonical_hash"](COUNTS),
        "fresh initial-noise context/finite plan/census differs")
    require(source["execution"]["adapter"] == "mdb-flux-final-latent-closure-v1"
        and source["execution"]["device"]["type"] == "cuda" and source["execution"]["dtype"] == "torch.bfloat16"
        and source["checkpoint_sha256"] == BASE["canonical_hash"](source["checkpoint_files"])
        == plan["expected_checkpoint_sha256"], "actual native model/backend/checkpoint differs")


def reader_ast_proof(original_source, corrected_source):
    """Permit the reader dispatch and exact correction plumbing, no equations."""
    old_tree, new_tree = map(ast.parse, (original_source, corrected_source))
    old = {node.name: node for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    new = {node.name: node for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    require(set(new) == set(old) | {"reader_ast_proof", "audit_source_authority"}, "unexpected corrected function inventory")
    dump = lambda node: ast.dump(node, include_attributes=False)
    clone = lambda node: ast.parse(ast.unparse(node)).body[0]
    def globals_only(tree):
        return [dump(node) for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
            and not (isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))]
    old_globals = globals_only(old_tree)
    require(old_globals == globals_only(new_tree), "top-level helper imports/constants changed")
    hashes = {}
    for name in old:
        if name in {"custody", "analyze"}:
            continue
        before, after = clone(old[name]), clone(new[name])
        if name == "Evidence":
            before.body = [node for node in before.body if not isinstance(node, ast.FunctionDef) or node.name != "array"]
            after.body = [node for node in after.body if not isinstance(node, ast.FunctionDef) or node.name != "array"]
        require(dump(before) == dump(after), "equation/control AST changed: " + name)
        hashes[name] = hashlib.sha256(dump(before).encode()).hexdigest()
    expected = clone(old["custody"])
    expected.args.kwonlyargs.append(ast.arg(arg="audit_correction", annotation=None))
    expected.args.kw_defaults.append(ast.Constant(None))
    replaced = 0
    for index, node in enumerate(expected.body):
        if isinstance(node, ast.For) and "independent_math_sources" in ast.unparse(node.iter):
            expected.body[index] = ast.parse("audit_source_authority(run, request, job, audit_correction)").body[0]
            replaced += 1
    require(replaced == 1 and dump(expected) == dump(new["custody"]), "custody correction changed other validation")
    expected = clone(old["analyze"])
    expected.args.kwonlyargs.append(ast.arg(arg="audit_correction", annotation=None))
    expected.args.kw_defaults.append(ast.Constant(None))
    replaced = 0
    for index, node in enumerate(expected.body):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "custody":
            node.value.keywords.append(ast.keyword(arg="audit_correction", value=ast.Name(id="audit_correction", ctx=ast.Load())))
            expected.body.insert(index + 1, ast.parse("authority = audit_source_authority(run, request, job, audit_correction)").body[0])
            replaced += 1
            break
    returned = expected.body[-1]
    require(isinstance(returned, ast.Return) and isinstance(returned.value, ast.Dict), "unexpected original audit return")
    returned.value.keys.extend([ast.Constant("preoutcome_auditor_sha256"), ast.Constant("audit_correction")])
    returned.value.values.extend([ast.parse('request["independent_math_sources"]["analyze_flux_initial_noise.py"]', mode="eval").body,
        ast.Name(id="authority", ctx=ast.Load())])
    require(replaced == 1 and dump(expected) == dump(new["analyze"]), "correction changed mathematical/control execution")
    return {"unchanged_function_ast_sha256": hashes,
        "unchanged_globals_ast_sha256": hashlib.sha256(json.dumps(old_globals).encode()).hexdigest(), "custody_other_validation_unchanged": True,
        "analyze_control_unchanged": True, "exception": "Evidence.array lossless modern native decoding; explicit source-authority/provenance plumbing"}


def audit_source_authority(run, request, job, correction_path):
    own = "analyze_flux_initial_noise.py"
    cohort = request["independent_math_sources"]
    for name, digest in cohort.items():
        require(sha(run / "independent-math-freeze" / name) == digest, "original frozen independent source changed")
        if name != own:
            require(sha(HERE / name) == digest, "unrelated independent source changed")
    current = sha(HERE / own)
    if correction_path is None:
        require(current == cohort[own], "pre-outcome independent source changed; explicit correction required")
        return None
    path = Path(correction_path).resolve()
    record = read(path)
    require(record["schema"] == "mdb-flux-initial-noise-reader-correction-v1"
        and record["job_id"] == job["job_id"] and record["request_sha256"] == request["request_sha256"]
        and record["original_auditor_sha256"] == cohort[own] and record["corrected_auditor_sha256"] == current
        and record["original_cohort_sha256s"] == cohort and type(record["model_calls"]) is int and record["model_calls"] == 0
        and record["producer_files_changed"] is False and record["scope"] == "lossless-modern-native-reader-only",
        "reader correction identity/scope differs")
    require(set(record["failed_attempt"]) == {"stdout", "stderr", "receipt"}, "failed attempt references incomplete")
    for reference in (*record["failed_attempt"].values(), record["qualification"]):
        require(Path(reference["path"]).is_absolute() and sha(Path(reference["path"])) == reference["sha256"],
            "correction evidence reference changed")
    failure = read(Path(record["failed_attempt"]["receipt"]["path"]))
    qualification = read(Path(record["qualification"]["path"]))
    require(failure["returncode"] != 0 and failure["model_calls"] == 0
        and failure["source_before_sha256"] == failure["source_after_sha256"] == cohort[own]
        and failure["stdout_sha256"] == record["failed_attempt"]["stdout"]["sha256"]
        and failure["stderr_sha256"] == record["failed_attempt"]["stderr"]["sha256"]
        and str(run / "independent-math-freeze" / own) in failure["command"] and str(run) in failure["command"]
        and qualification["returncode"] == 0
        and qualification["analyzer_sha256"] == current, "correction qualification/failure status differs")
    proof = reader_ast_proof((run / "independent-math-freeze" / own).read_text(), (HERE / own).read_text())
    require(record["ast_proof"] == proof, "reader correction AST proof differs")
    return {"path": str(path), "sha256": sha(path), "record": record,
        "status": "post-outcome saved-reader correction; original request/freeze/failed attempt unchanged"}


def custody(run, *, audit_correction=None):
    collector = runpy.run_path(str(HERE / "collect_flux_initial_noise.py"))
    request, job, collection = [read(run / name) for name in ("request.json", "job.json", "collection.json")]
    data = run / "extracted/flux-initial-noise-result"
    collector["validate_job"](job, job["job_id"], request)
    inventory = collector["inventory"](data)
    require(inventory == read(run / "collection-remote-before.json") == read(run / "collection-remote-after.json")
        == read(run / "collection-local-inventory.json") and job == read(run / "collection-job.json")
        and collection["job_id"] == job["job_id"] and collection["request_sha256"] == request["request_sha256"]
        and collection["collector_source_sha256"] == sha(HERE / "collect_flux_initial_noise.py")
        and all(sha(H["file_path"](run, name)) == digest for name, digest in collection["bindings"].items()),
        "authenticated collection/source binding differs")
    collector["verify_manifest"](data, inventory, job["job_id"], request)
    require(job["result"]["diagnostics"]["command"] == read(run / "admission-command.json")["command"], "executed command differs")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
        == request["request_sha256"], "request changed")
    for name, digest in request["source_hashes"].items():
        require(sha(H["file_path"](run / "payload", name)) == digest, "payload source changed")
    audit_source_authority(run, request, job, audit_correction)
    return data, request, job, collection, inventory


def source_abi(data, source, plan, run):
    def constants(name):
        syntax = ast.parse((run / "payload" / name).read_text())
        return ast.literal_eval(next(node.value for node in syntax.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SOURCE_CONTRACT" for target in node.targets)))

    pins = constants("flux_reentry_reference.py")
    expected_reentry = {key: value for key, value in pins.items() if key not in {"diffusers_version", "torch_qualification_version"}}
    pins = constants("flux_step_boundary_reference.py")
    expected_step = {key: value for key, value in pins.items() if key.endswith("sha256")}
    for name in ("pre_model_source_abi", "native_source_abi"):
        ref = source[name]
        abi = read(H["file_path"](data, ref["path"]))
        two = abi["prior_two_pass"]
        final = two["prior_final_latent"]
        tail = final["prior_joint_tail"]
        chain = tail["prior_joint_chain"]
        old = chain["prior_native_and_closure"]
        require(sha(data / ref["path"]) == ref["sha256"] and abi["verified"] is True
            and abi["native_advance_inherited"] is True and two["verified"] is True
            and old["reentry_actual"] == old["reentry_expected"] == expected_reentry
            and old["base_step"]["verified"] is True and old["base_step"]["model_calls"] == 0
            and old["base_step"]["actual"] == old["base_step"]["expected"] == expected_step
            and old["closure_module_sha256"] == plan["expected_closure_module_sha256"]
            and chain["joint_closure_module_sha256"] == plan["expected_joint_closure_module_sha256"]
            and tail["tail_closure_module_sha256"] == plan["expected_tail_closure_module_sha256"]
            and final["final_closure_module_sha256"] == plan["expected_final_closure_module_sha256"]
            and two["derived_terminal_module_sha256"] == plan["expected_derived_terminal_module_sha256"]
            and abi["parameter_artifacts_module_sha256"] == plan["expected_parameter_artifacts_module_sha256"],
            "actual native/helper source ABI differs")
    scalar = read(data / "cuda-scalar-qualification.json")
    require(scalar["qualified"] is True and scalar["device"] == "cuda" and scalar["model_calls"] == 0
        and len(scalar["cases"]) == 3 and all(case["values"] == 2048 and case["native_dtype"] == "torch.bfloat16"
            and case["native_vs_bf16_dt_mismatches"] == 0 and case["native_vs_fp32_dt_mismatches"] > 0
            for case in scalar["cases"]), "actual CUDA scalar-promotion instrument absent")


def bank_profiles(binding):
    require(set(binding["pass_schedules"]) == {str(i) for i in range(4)}
        and set(binding["final_blocks"]) == {str(i) for i in range(20)}, "complete bank/schedule inventory differs")
    for index in range(5):
        spec = binding["joint" + str(index)]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["layernorm_eps"], spec["qk_eps"]) == (24, 128, 1e-6, 1e-6),
            "joint native port/profile differs")
    for index in range(20):
        spec = binding["final_blocks"][str(index)]
        require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"single_transformer_blocks.{index}."
            and (spec["heads"], spec["head_dim"], spec["hidden_width"], spec["mlp_width"], spec["layernorm_eps"], spec["qk_eps"])
                == (24, 128, 3072, 9216, 1e-6, 1e-6), "single native port/profile differs")


def native_path(e, segment, full_cut, matched_cut):
    require(segment["input_cut"] == full_cut and len(segment["receipts"]) == len(segment["boundaries"]) == 112,
        "native full trajectory inventory differs")
    ids = BASE["path"](e, segment["receipts"], full_cut, segment["native_final"]["cut"], 112)
    source_fields = e.fields(full_cut)
    latent_field, context_field = source_fields["latent"], None
    for index, record in enumerate(segment["boundaries"]):
        require(record["index"] == index and record["receipt_id"] == ids[index]
            and record["cut"] == segment["receipts"][index]["result"]
            and tuple(e.point(record["cut"])[k] for k in ("logical_step", "phase", "index")) == point_after(index),
            "native transition lineage/clock differs")
        boundary(e, record)
        fields = e.fields(record["cut"])
        require(all(fields[key] == source_fields[key] for key in
            ("conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")), "native carried conditioning/IDs/schedule changed")
        local = index % 28
        if local == 0:
            context_field = fields["context"]
        if local < 27:
            require(fields["latent"] == latent_field and fields["context"] == context_field,
                "native pass refreshed latent/context before scheduler commit")
        else:
            require(all(e.typed["unpack"](fields[key]) is None for key in ("context", "image", "text", "noise_prediction")),
                "native scheduler did not clear transient fields")
            latent_field = fields["latent"]
    require(segment["boundaries"][55]["cut"] == matched_cut
        and all(segment["boundaries"][-1][key] == value for key, value in segment["native_final"].items()),
        "matched source is not actual native Z2")
    return ids


def verify_transaction(data, summary, binding, history, *, max_bytes):
    root = data / "transaction" / history
    seal = summary["seals"][history]
    verified = H["verify_transaction"](root, expected_seal_sha256=seal["seal_sha256"], max_bytes=max_bytes)
    report = read(root / "report.json")
    require(summary["transaction_reports"][history] == {"path": "transaction/" + history + "/report.json", "sha256": sha(root / "report.json")}
        and verified["phase"] == report["phase"] == "closed" and not report["failure"] and not report["violations"]
        and CHAIN["SHARED"]["transaction_verifications_match"](verified, summary["transaction_verification"][history]),
        "history transaction did not close cleanly")
    tree = H["NumericalTree"](root)
    frontier = tree.decode(read(root / "frontiers/source_binding.json")["value"])
    events = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    seals = [event for event in events if event["kind"] == "history_" + history + "_sealed"]
    require(len(seals) == 1, "history seal event missing or duplicated")
    require(frontier == history_binding(binding, history, seals[0])
        == read(data / ("binding-" + history + ".json")) and "origins" not in frontier,
        "history sealed a different origin or unavailable future binding")
    forecast = tree.decode(read(root / "forecasts/passes.json")["predictions"])
    observation = tree.decode(read(root / "observations/predicted_terminal_consumer.json")["value"])
    require(observation["row"] == next(row for row in summary["rows"] if row["id"] == history + ".predicted")
        and observation["first_image"] == read(data / ("first-image-" + history + ".json"))
        and observation["counts"] == observation["first_image"]["counts"], "persisted consumer observation differs")
    return verified, forecast


def target_profile(multiplier):
    return {key: multiplier * (2 if key == "pos_embed" else 1) for key in TARGETS}


def prospective_order(data, summary, binding, rows):
    events = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    require([event["sequence"] for event in events] == list(range(len(events))), "timeline sequence differs")
    references = [event for event in events if event["kind"] == "reference_transition_complete"]
    require(len(references) == 112, "native reference timeline incomplete")
    images = [event for event in events if event["kind"] == "predicted_image_complete"]
    require([event["history"] for event in images] == ["full", "matched"], "forecast image chronology differs")
    checked = {}
    for history, origin in (("full", 0), ("matched", 2)):
        seals = [event for event in events if event["kind"] == "history_" + history + "_sealed"]
        image = next(event for event in images if event["history"] == history)
        first = read(data / ("first-image-" + history + ".json"))
        require(len(seals) == 1, "history seal occurrence differs")
        sealed = seals[0]
        row = rows[history + ".predicted"]
        require(sealed["seal"] == summary["seals"][history] and sealed["blocked_entries"] == 0
            and sealed["sequence"] < image["sequence"] < references[28 * origin]["sequence"]
            and (origin == 0 or references[55]["sequence"] < sealed["sequence"])
            and first["image"] == image["image"] == row["image"]
            and first["source_cut"] == binding["origins"][history]["cut"]
            and first["source_unchanged"] is first["parameters_unchanged"] is True
            and first["derivation_sha256"] == sha(data / "derivations" / (history + ".json")),
            "forecast image not prospectively rendered from its bound origin")
        checked[history] = guard_contract(history, sealed["counts"], first["counts"],
            sealed["clock_qualified_targets_remaining"], first["clock_qualified_targets_remaining"])
        require(first["counts"] == image["counts"] and first["native_steps_before"] == 28 * origin
            and first["target_calls"] == image["target_calls"] == sealed["target_calls"] == target_profile(origin)
            and first["target_calls_by_step"] == image["target_calls_by_step"] == sealed["target_calls_by_step"]
            and first["clock_qualified_targets_remaining"] == image["clock_qualified_targets_remaining"],
            "image counter snapshot changed or erased earlier target history")
        require(first["target_calls_by_step"] == {str(i): target_profile(int(i < origin)) for i in range(4)},
            "target counter logical clock history differs")
        checked[history].update(seal_sequence=sealed["sequence"], image_sequence=image["sequence"],
            first_future_reference_sequence=references[28 * origin]["sequence"], evidence=first)
    require(summary["target_calls"] == target_profile(4)
        and summary["target_calls_by_step"] == {str(i): target_profile(1) for i in range(4)}, "final native target census differs")
    return checked


def analyze(run, *, parameter_root=None, checkpoint=None, ssh=None, audit_correction=None):
    started = time.monotonic()
    run = Path(run).resolve()
    data, request, job, collection, inventory = custody(run, audit_correction=audit_correction)
    authority = audit_source_authority(run, request, job, audit_correction)
    source, summary, plan, binding, initial, segment = [read(data / name) for name in
        ("source.json", "summary.json", "plan.json", "binding.json", "binding-initial.json", "native-segment.json")]
    require(plan == request["config"] and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["source_hashes"] == request["source_hashes"], "producer source/request identity differs")
    validate_plan(plan, source, summary)
    require(summary["mathematical_steps"] == 168 and summary["branch_execution_order"] == list(ORDER)
        and summary["blocked_entries"] == 0, "completed history/guard inventory differs")
    source_abi(data, source, plan, run)
    frozen = read(data / "plan-frozen.json")
    require(frozen["plan_sha256"] == sha(data / "plan.json")
        and frozen["source_identity"] == BASE["canonical_hash"](request["source_hashes"]), "pre-source plan freeze differs")
    require(set(initial["origins"]) == {"full"} and set(binding["origins"]) == {"full", "matched"}
        and initial["origins"]["full"] == binding["origins"]["full"]
        and {k: v for k, v in initial.items() if k != "origins"} == {k: v for k, v in binding.items() if k != "origins"},
        "initial binding borrowed a future observation or changed shared parameters")
    require(source["binding_initial"] == {"path": "binding-initial.json", "sha256": sha(data / "binding-initial.json")}
        and source["parameter_artifacts"] == binding["parameter_artifacts"] == summary["parameter_artifacts"]
        and source["parameter_artifacts"]["sha256"] == sha(data / "parameter-artifacts.json")
        and source["parameter_artifacts"]["script_sha256"] == sha(data / "retrieve_weights.py")
            == plan["expected_parameter_artifacts_module_sha256"]
        and source["parameter_artifacts"]["parameter_files_written"] == 0
        and not any(name.startswith("parameters/") for name in inventory["files"]), "checkpoint recipe/source storage custody differs")
    bank_profiles(binding)
    cache = Path(parameter_root) if parameter_root is not None else run / "parameter-cache"
    recipes, hydration = parameter_cache(data, cache, checkpoint=checkpoint, ssh=ssh)
    manifest = read(data / "parameter-artifacts.json")
    require(len(recipes) == source["parameter_artifacts"]["artifacts"] == 169
        and manifest["provenance"]["job_id"] == job["job_id"]
        and manifest["provenance"]["request_sha256"] == request["request_sha256"]
        and manifest["provenance"]["producer_source_sha256"] == request["source_hashes"]["flux_initial_noise_campaign.py"]
        and all(recipe["source"]["file_sha256"] == source["checkpoint_files"][recipe["source"]["file"]]
            for recipe in recipes.values()), "recipe checkpoint/producer identity differs")
    e = Evidence(data, cache, recipes)
    require(all(cut["model_identity"] == source["model_identity"] and cut["execution"] == source["execution"]
        for cut in e.store["cuts"].values()) and all(receipt["model_identity"] == source["model_identity"]
        and receipt["execution"] == source["execution"] for receipt in e.receipts.values()), "typed model/execution identities differ")
    full_inputs = check_origin(e, binding["origins"]["full"], 0)
    matched_inputs = check_origin(e, binding["origins"]["matched"], 2)
    require(all(exact(full_inputs[key], matched_inputs[key]) for key in SLOTS if key != "latent"), "matched history changed conditioning/IDs/schedules")
    require(exact(e.array(source["encoding"]["conditioning"]), full_inputs["conditioning"])
        and exact(e.array(source["encoding"]["txt_ids"]), full_inputs["txt_ids"])
        and exact(e.array(source["actual_initial_noise"]), full_inputs["latent"]), "clean native encoding/noise differs")
    full_cut, matched_cut = (binding["origins"][history]["cut"] for history in ("full", "matched"))
    native_ids = native_path(e, segment, full_cut, matched_cut)
    prefix = read(data / "native-prefix.json")
    require(prefix["input_cut"] == full_cut and prefix["matched_origin"] == binding["origins"]["matched"]
        and prefix["boundaries"] == segment["boundaries"][:56] and prefix["receipts"] == segment["receipts"][:56],
        "matched observation is not actual same-history native Z2")
    transactions, final_predictions, histories = {}, {}, {}
    for history, origin in (("full", 0), ("matched", 2)):
        transactions[history], forecast = verify_transaction(data, summary, binding, history,
            max_bytes=plan[history + "_transaction_max_bytes"])
        histories[history], independent_final = audit_history(e, binding, forecast, segment, origin=origin)
        final_predictions[history] = forecast["passes"][-1]["commit"]["latent"].copy()
        del forecast, independent_final
        gc.collect()
    require(transactions["full"]["seal_sha256"] == collection["seals"]["full"]
        == collection["seal_sha256"] and transactions["matched"]["seal_sha256"] == collection["seals"]["matched"],
        "authenticated collection/history seals differ")
    rows, latents, pixels, derived = {}, {}, {}, {}
    for name in ORDER:
        row = read(data / "rows" / (name + ".json"))
        require(row["id"] == name and row == next(value for value in summary["rows"] if value["id"] == name)
            and len(row["acts"]) == (name != "native") and row["continuation_receipts"] == [], "terminal-only row recipe differs")
        parent = row["parent"]
        require(row["entry"]["cut"] == parent, "row entry parent differs")
        boundary(e, row["entry"])
        current = parent
        if name in ("native", "sham"):
            require(parent == segment["native_final"]["cut"], "native/sham parent differs")
        if name != "native":
            history = name.split(".")[0]
            predicted = final_predictions[history] if name != "sham" else e.slot(parent, "latent")
            current = PREV["atomic_act"](e, row["acts"][0], parent,
                {"latent": predicted, "context": None, "image": None, "text": None})
            require(row["installed"]["cut"] == current, "installed terminal record differs")
            boundary(e, row["installed"])
            if name != "sham":
                path = "derivations/" + history + ".json"
                require(row["derivation"] == {"path": path, "sha256": sha(data / path)}, "derivation custody differs")
                derived[history] = TWO["check_derived_terminal"](e, read(data / path), binding["origins"][history]["cut"],
                    row["acts"][0], predicted, plan["expected_derived_terminal_module_sha256"])
        final = boundary(e, row["terminal_final"])
        require(row["terminal_final"]["cut"] == row["final_cut"] == current
            and tuple(e.point(current)[k] for k in ("logical_step", "phase", "index")) == (4, "project", None)
            and all(final[key] is None for key in ("context", "image", "text", "noise_prediction")), "terminal VAE reset differs")
        latents[name], pixels[name] = e.array(row["final_latent"]), e.png(row["image"])
        require(exact(latents[name], final["latent"]), "VAE received a different terminal latent")
        rows[name] = row
    native_fields = e.fields(rows["native"]["final_cut"])
    for name in ORDER:
        require(all(e.fields(rows[name]["final_cut"])[key] == value for key, value in native_fields.items() if key != "latent")
            and e.point(rows[name]["final_cut"]) == e.point(rows["native"]["final_cut"]), "VAE inputs differ outside latent")
    expected = [("full.predicted", rows["full.predicted"]["acts"][0]["receipt_id"])]
    expected += [("reference", rid) for rid in native_ids[:56]]
    expected += [("matched.predicted", rows["matched.predicted"]["acts"][0]["receipt_id"])]
    expected += [("reference", rid) for rid in native_ids[56:]]
    expected += [("sham", rows["sham"]["acts"][0]["receipt_id"])]
    emissions = CHAIN["SHARED"]["read_jsonl"](data / "receipt-emissions.jsonl")
    require([item["sequence"] for item in emissions] == list(range(115))
        and [(item["branch"], item["receipt"]) for item in emissions] == expected
        and set(rid for _, rid in expected) == set(e.receipts)
        and sum(e.receipts[rid].get("steps", 0) for _, rid in expected if e.receipts[rid]["operation"] == "continue") == 112
        and sum(e.receipts[rid]["operation"] == "apply" for _, rid in expected) == 3, "physical execution occurrence census differs")
    guards = prospective_order(data, summary, binding, rows)
    endpoints = {}
    for name in ORDER:
        endpoints[name] = {"latent": compare(latents[name], latents["native"]), "rgb": compare(pixels[name], pixels["native"]),
            "rgb_mae": float(np.abs(pixels[name].astype(np.float64) - pixels["native"]).mean())}
    controls = {"sham_final_latent_exact_native": exact(latents["sham"], latents["native"]),
        "sham_png_exact_native": rows["sham"]["image"]["sha256"] == rows["native"]["image"]["sha256"]}
    require(all(summary["control_observations"][key] == value for key, value in controls.items()), "control report differs")
    return {"schema": "mdb-flux-initial-noise-audit-v1", "mechanics_status": "valid",
        "instrument_status": "native-replay-qualified" if all(controls.values()) else "replay-unresolved",
        "trend_status": "observed; all signed numerical discrepancies retained", "terminal_status": "unassessed",
        "job_id": job["job_id"], "request_sha256": request["request_sha256"], "context": plan["context"],
        "seals": summary["seals"], "counts": COUNTS, "mathematical_transitions": 168,
        "histories": histories, "endpoint_comparisons": endpoints, "derived_terminal": derived,
        "prospective_guards": guards, "control_observations": controls, "transaction_verification": transactions,
        "parameter_hydration": hydration, "target_calls": summary["target_calls"],
        "target_calls_by_step": summary["target_calls_by_step"], "models_executed": False,
        "collection": {"files": inventory["file_count"], "bytes": inventory["expanded_bytes"], "sha256": sha(run / "collection.json")},
        "independent_math_sources": request["independent_math_sources"], "auditor_source_sha256": sha(Path(__file__)),
        "audit_seconds": time.monotonic() - started,
        "scope": "One seed113 native history: full112 from actual initial noise, matched56 from observed same-history nativeZ2. Each independent recurrence advances its own computed state. Native prompt encoding and VAE remain supplied; guard counts are producer instrumentation, not OS isolation/kernel tracing. Four PNGs are native VAE observations, not independent pixel forecasts. No accuracy gate, semantic claim, universal error bound or speedup certification.",
        "preoutcome_auditor_sha256": request["independent_math_sources"]["analyze_flux_initial_noise.py"], "audit_correction": authority}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parameter-root", type=Path, help="Existing external, exact recipe-qualified NPY cache")
    parser.add_argument("--checkpoint", type=Path, help="Optionally restore an incomplete external cache from this checkpoint root")
    parser.add_argument("--ssh", help="Explicit remote host for --checkpoint hydration")
    parser.add_argument("--audit-correction", type=Path, help="Explicit hash-bound post-outcome reader correction receipt")
    args = parser.parse_args()
    require(not args.out.exists(), "refuse to overwrite an audit")
    report = analyze(args.run_dir, parameter_root=args.parameter_root, checkpoint=args.checkpoint, ssh=args.ssh,
        audit_correction=args.audit_correction)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("mechanics_status", "instrument_status", "job_id", "counts", "audit_seconds")}))
