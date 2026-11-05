"""Independent saved-only audit of a sealed fourteen-block diffusion commit.

The complete-block recurrence and new readout/Euler mathematics were pinned in
the request before execution. Native errors are measurements, never gates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import runpy
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHAIN = runpy.run_path(str(HERE / "analyze_flux_transition_chain.py"))
MATH = runpy.run_path(str(HERE / "flux_step_boundary_audit_math.py"))
H = runpy.run_path(str(HERE / "summarize_context_window.py"))
require, exact_bits, field_error, SavedEvidence = (CHAIN[k] for k in ("require", "exact_bits", "field_error", "SavedEvidence"))
sha = CHAIN["sha_file"]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 168,
    "acts": 2, "candidate_calls": 0, "receipt_emissions": 170, "images": 3, "vae_calls": 3, "physical_batch": 1}
RESET = {"image", "text", "context", "noise_prediction"}
VOLATILE = RESET | {"latent", "step", "phase", "cursor"}
PARAMS = {"qkv_mlp_weight": "attn.to_qkv_mlp_proj.weight", "qkv_mlp_bias": "attn.to_qkv_mlp_proj.bias",
    "to_out_weight": "attn.to_out.weight", "to_out_bias": "attn.to_out.bias",
    "norm_q_weight": "attn.norm_q.weight", "norm_k_weight": "attn.norm_k.weight"}


def read(path):
    return H["read"](Path(path))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compare(predicted, native):
    """Signed predicted-minus-native; denominator is native RMS explicitly."""
    result = field_error(np.asarray(predicted), np.asarray(native))
    result["relative_to_native_rms"] = result["relative_rms"]
    result["sign"] = "predicted minus native/reference"
    return result


def stripped(e, cut, excluded):
    return {k: v for k, v in e.fields(cut).items() if k not in excluded}


def declared_mapping(e, cut, excluded):
    payload = e.store["cuts"][cut]["payload"]
    return {**payload, "items": [item for item in payload["items"] if item["key"]["value"] not in excluded]}


def check_boundary(e, record):
    cut = record["cut"]
    require(record["point"] == e.point(cut)
        and record["clock"] == {k: e.typed["unpack"](e.fields(cut)[k]) for k in ("step", "phase", "cursor")},
        "saved boundary clock/cut differs")
    arrays = {k: e.array(v) for k, v in record["arrays"].items()}
    expected = {k for k in ("image", "text", "latent", "noise_prediction") if e.fields(cut)[k]["kind"] == "tensor"}
    require(set(arrays) == expected and all(exact_bits(v, e.slot(cut, k)) for k, v in arrays.items()),
        "saved full boundary arrays differ from typed cut")
    return arrays


def path(e, receipts, start, end, length):
    require(len(receipts) == length, "native path length differs")
    current, ids = start, []
    for r in receipts:
        require(r == e.receipts[r["fingerprint"]] and r["operation"] == "continue" and r["steps"] == 1
            and r["parent"] == current and len(r["transitions"]) == 1,
            "native path receipt/custody/parent differs")
        transition = r["transitions"][0]
        require(transition["before"] == e.point(current) and transition["after"] == e.point(r["result"])
            and transition["native_operation"]["footprint"] == "declared"
            and transition["native_operation"]["operation"] == transition["before"]["next_operation"],
            "native operation/clock footprint differs")
        ids.append(r["fingerprint"])
        current = r["result"]
    require(current == end, "native path endpoint differs")
    return ids


def load_parameters(e, spec, names, native=None):
    ports = spec["native_parameters"] if native is None else native
    require(set(spec["parameters"]) == set(names), "parameter port inventory differs")
    result = {}
    for key, port in names.items():
        record = spec["parameters"][key]
        if record is None:
            require(key.endswith("bias") and port not in ports, "missing native bias binding differs")
            result[key] = None
        else:
            require(port in ports and all(record["descriptor"][k] == ports[port][k] for k in ("shape", "dtype", "sha256")),
                "archived parameter is not the declared native port")
            result[key] = e.array(record)
            MATH["exact_bf16"](result[key])
    return result


def latent_act(e, action, parent, expected):
    r = action["receipt"]
    require(action["slot"] == "latent" and action["undeclared_slots_exact"] is True
        and r == e.receipts[action["receipt_id"]] and r["operation"] == "apply"
        and r["act"] == action["act"] and r["parent"] == action["parent"] == parent
        and r["result"] == action["post_act"] and action["act"]["writes"] == ["latent"]
        and action["act"]["operation"]["id"] == "saturn.builtin.replace", "latent Act identity differs")
    parameter = e.typed["unpack"](action["act"]["parameters"])["value"]
    after = action["post_act"]
    require(e.fields(after) == {**e.fields(parent), "latent": parameter}
        and e.typed["unpack"](r["writes"])["latent"] == parameter
        and e.point(parent) == e.point(after) == action["point"]
        and exact_bits(e.slot(after, "latent"), expected)
        and e.fields(after)["latent"]["stride"] == e.fields(parent)["latent"]["stride"],
        "latent Act payload/stride/complete closure differs")
    return after


def audit_continuation(e, rows, segment, plan):
    record = read(e.data / "continuation-observations.json")
    require(record["plan"] == plan["observation_plan"] and record["branch_execution_order"] == ["native", "sham", "predicted"]
        and record["model_calls_added"] == 0 and set(record["rows"]) == {"native", "sham", "predicted"},
        "continuation instrument profile differs")
    full, results = {}, {}
    previous = CHAIN["PREVIOUS"]
    for name in ("native", "sham", "predicted"):
        row, trajectory = rows[name], record["rows"][name]
        require(len(trajectory) == 29, "continuation boundary census differs")
        initial_incoming = row["acts"][-1]["receipt_id"] if row["acts"] else segment["receipts"][-1]["fingerprint"]
        initial_z3 = e.slot(row["next_input"]["cut"], "latent")
        results[name] = []
        for index, item in enumerate(trajectory):
            cut = row["next_input"]["cut"] if index == 0 else row["continuation_receipts"][index - 1]["result"]
            incoming = initial_incoming if index == 0 else row["continuation_receipts"][index - 1]["fingerprint"]
            require(item["index"] == index and item["cut"] == cut and item["point"] == e.point(cut)
                and item["incoming_receipt"] == incoming and item["complete_cut_retained"] == (index in {0, 28})
                and set(item["slots"]) == {"image", "text", "latent", "noise_prediction"}
                and set(item["comparisons"]) == {"native"}, "continuation clock/lineage/storage policy differs")
            summaries = {}
            for slot, value in item["slots"].items():
                reference_slot = record["rows"]["native"][index]["slots"][slot]
                require(type(value["present"]) is bool and value["present"] == reference_slot["present"],
                    "candidate/reference continuation slot presence differs")
                if cut in e.store["cuts"]:
                    require(value["present"] == (e.fields(cut)[slot]["kind"] == "tensor"),
                        "retained continuation slot presence differs")
                if not value["present"]:
                    require("array" not in value and "witness" not in value, "absent continuation slot invented a value")
                    continue
                require(value["descriptor"]["dtype"] == reference_slot["descriptor"]["dtype"] == "torch.bfloat16"
                    and value["descriptor"]["shape"] == reference_slot["descriptor"]["shape"],
                    "candidate/reference continuation tensor geometry differs")
                actual = None
                if cut in e.store["cuts"]:
                    actual = e.slot(cut, slot)
                    require(e.typed["matches"](actual, value["descriptor"], bf16=True), "retained continuation descriptor differs")
                if name == "native":
                    saved = e.array(value["array"])
                    require(e.typed["matches"](saved, value["descriptor"], bf16=True)
                        and (actual is None or exact_bits(saved, actual)), "native reference array differs")
                    full[(index, slot)] = actual = saved
                else:
                    require("array" not in value, "candidate intermediate arrays exceed declared witness-only policy")
                witness = previous["witness_values"](value, actual)
                if slot == "latent" and index < 28:
                    require(e.typed["matches"](initial_z3, value["descriptor"], bf16=True), "stored Z3 changed before its next commit")
                    previous["witness_values"](value, initial_z3)
                if slot in {"image", "text"}:
                    active = item["point"]["phase"] in {"joint", "single", "readout"}
                    require(value["meaning"]["meaningful_propagation_observable"] == active
                        and value["meaning"]["stale"] == (not active), "stale working tensor treated as propagation")
                reported = item["comparisons"]["native"]["slots"][slot]
                reference = full.get((index, slot))
                if reference is None:
                    require(reported == {"status": "not-present-in-both-current-states"}, "absent reference comparison differs")
                    continue
                ref_witness = previous["witness_values"](record["rows"]["native"][index]["slots"][slot], reference)
                require(reported["sample_count"] == reference.size and 0 <= reported["changed_value_count"] <= reference.size
                    and all(np.isfinite(reported[k]) for k in ("l2", "rms", "mae", "signed_mean", "max_abs")),
                    "candidate full metric observation is malformed")
                summary = {"meaning": value["meaning"], "worker_observed_full_metrics": reported,
                    "independent_fixed64_metrics": previous["compare_arrays"](witness.reshape(1, -1, 1), ref_witness.reshape(1, -1, 1)),
                    "independently_verified_full_metrics": actual is not None}
                if actual is not None:
                    independent = compare(actual, reference)
                    independent["mae"] = float(np.mean(np.abs(actual.astype(np.float64) - reference)))
                    summary["independent_full_metrics"] = independent
                    summary["worker_minus_independent_reduction"] = {k: reported[k] - independent[k]
                        for k in ("l2", "rms", "mae", "signed_mean", "max_abs")}
                summaries[slot] = summary
            results[name].append({"index": index, "point": item["point"], "slots": summaries})
    return {"rows": results, "boundaries_per_branch": 29,
        "scope": "Complete native reference arrays and typed branch endpoints; candidate intermediate full norms remain worker observations. Fixed64 witnesses provide independent sampled checks. Stored Z3 is constant before commit; stale working tensors are excluded from active-propagation interpretations."}


def audit_forecast(e, binding, forecast, segment, plan):
    pre = binding["input_cut"]
    require(binding["block_indices"] == list(range(6, 20)) and binding["text_rows"] == 512
        and binding["point"] == e.point(pre) and binding["clock_state"] == {"step": 2, "phase": "single", "cursor": 6}
        and binding["carried_state"] == declared_mapping(e, pre, VOLATILE), "forecast input state/clock differs")
    for slot, archive in binding["inputs"].items():
        require(exact_bits(e.array(archive), e.slot(pre, slot)), "forecast operand borrowed another cut")
    nested = CHAIN["PREVIOUS"]["nested_slot_array"]
    for key in ("single_mod", "temb"):
        require(exact_bits(e.array(binding["context"][key]), nested(e, pre, "context", (key,))), "forecast time context differs")
    for index, name in enumerate(("cos", "sin")):
        require(exact_bits(e.array(binding["rope"][name]), nested(e, pre, "context", ("rope", index))), "forecast RoPE differs")
    current = np.concatenate((e.slot(pre, "text"), e.slot(pre, "image")), axis=1)
    independent = current
    require(current.shape == (1, 1536, 3072) and forecast["chain_length"] == len(forecast["steps"]) == 14
        and forecast["input_descriptor"] == MATH["identity"](current), "forecast complete chain geometry differs")
    mod = e.array(binding["context"]["single_mod"])
    cos, sin = [e.array(binding["rope"][k]) for k in ("cos", "sin")]
    rows = []
    for position, index in enumerate(range(6, 20)):
        spec, step = binding["blocks"][str(index)], forecast["steps"][position]
        p = step["prediction"]
        source = "observed native step2/pre6" if position == 0 else "previous complete predicted output"
        require(step["step"] == position and step["block_index"] == index and step["input_source"] == source
            and step["input_descriptor"] == p["input_descriptor"] == MATH["identity"](current)
            and exact_bits(p["sampled"]["input"], current[:, plan["sample_indices"]])
            and p["sampled"]["indices"] == plan["sample_indices"]
            and p["output_descriptor"] == MATH["identity"](p["output"])
            and exact_bits(p["sampled"]["output"], p["output"][:, plan["sample_indices"]])
            and "image" not in p and "text" not in p, "forecast chain refreshed/aliased/misdeclared complete state")
        require((spec["heads"], spec["head_dim"], spec["hidden_width"], spec["mlp_width"], spec["layernorm_eps"], spec["qk_eps"])
            == (24, 128, 3072, 9216, 1e-6, 1e-6), "complete-block numerical profile differs")
        profile = spec["attention_profile"]
        require(profile["processor_type"].endswith(".Flux2ParallelSelfAttnProcessor")
            and all(profile[k] is None for k in ("processor_backend", "parallel_config", "attention_mask", "joint_attention_kwargs"))
            and profile["is_causal"] is False and profile["kv_cache"] is False, "attention dispatch profile differs")
        parameters = load_parameters(e, spec, PARAMS)
        independent = CHAIN["independent_transition"](independent, mod, parameters, heads=24,
            layernorm_eps=1e-6, qk_eps=1e-6, rope_cos=cos, rope_sin=sin)
        del parameters
        native = check_boundary(e, segment["boundaries"][position])
        actual = np.concatenate((native["text"], native["image"]), axis=1)
        current = p["output"]
        rows.append({"block_index": index, "sealed_minus_independent": compare(current, independent),
            "predicted_minus_native": {"both": compare(current, actual), "text": compare(current[:, :512], actual[:, :512]),
                "image": compare(current[:, 512:], actual[:, 512:])}})
        print(json.dumps({"phase": "block_audit", "block": index,
            "independent_exact": rows[-1]["sealed_minus_independent"]["bitwise_exact"],
            "native_relative_rms": rows[-1]["predicted_minus_native"]["both"]["relative_rms"]}), flush=True)
        gc.collect()
    require(forecast["output_descriptor"] == MATH["identity"](current), "forecast final block descriptor differs")
    readout = binding["readout"]
    ports = {**{"norm_out." + k: v for k, v in readout["native_parameters"].items()},
        **{"proj_out." + k: v for k, v in readout["projection_native_parameters"].items()}}
    names = {"modulation_weight": "norm_out.linear.weight", "modulation_bias": "norm_out.linear.bias",
        "projection_weight": "proj_out.weight", "projection_bias": "proj_out.bias"}
    require(readout["elementwise_affine"] is False and readout["eps"] == 1e-6, "readout normalization profile differs")
    rp = load_parameters(e, readout, names, ports)
    predicted_readout = MATH["independent_readout"](independent[:, 512:], e.array(binding["context"]["temb"]), **rp, eps=1e-6)
    r = forecast["readout"]
    require(r["input_descriptor"] == MATH["identity"](current[:, 512:])
        and r["noise_descriptor"] == MATH["identity"](r["noise"])
        and r["sampled"]["indices"] == plan["image_sample_indices"]
        and exact_bits(r["sampled"]["noise"], r["noise"][:, plan["image_sample_indices"]]), "readout forecast input/output join differs")
    native_noise = check_boundary(e, segment["boundaries"][14])["noise_prediction"]
    # Additional diagnostic: apply the same independent readout law to native post19.
    native_post19 = check_boundary(e, segment["boundaries"][13])["image"]
    native_input_readout = MATH["independent_readout"](native_post19, e.array(binding["context"]["temb"]), **rp, eps=1e-6)
    del rp
    commit = forecast["commit"]
    sigma, following = [e.array(binding["commit"][k]) for k in ("sigma", "sigma_next")]
    schedule = e.slot(pre, "sigmas")
    require(exact_bits(sigma, schedule[2]) and exact_bits(following, schedule[3])
        and binding["commit"]["step_index"] == 2, "Euler schedule scalar borrowed another step")
    z = e.slot(pre, "latent")
    independently_committed = MATH["independent_flow_commit"](z, predicted_readout["noise"], sigma, following)
    actual_z3 = check_boundary(e, segment["boundaries"][15])["latent"]
    exact_native_commit = MATH["independent_flow_commit"](z, native_noise, sigma, following)
    require(commit["input_descriptor"] == MATH["identity"](z)
        and commit["noise_descriptor"] == r["noise_descriptor"]
        and commit["latent_descriptor"] == MATH["identity"](commit["latent"])
        and forecast["next_state_contract"] == {"clock": {"step": 3, "phase": "project", "cursor": 0},
            "reset": {k: None for k in RESET}, "carried_state": binding["carried_state"]}, "forecast commit/reset/input join differs")
    return {"blocks": rows, "readout": {"sealed_minus_independent": compare(r["noise"], predicted_readout["noise"]),
        "predicted_minus_native": compare(r["noise"], native_noise),
        "independent_on_native_input_minus_native": compare(native_input_readout["noise"], native_noise)},
        "commit": {"sealed_minus_independent": compare(commit["latent"], independently_committed["latent"]),
            "predicted_minus_native": compare(commit["latent"], actual_z3),
            "independent_on_native_noise_minus_native": compare(exact_native_commit["latent"], actual_z3),
            "dt_fp32": independently_committed["dt_fp32"], "dt_bf16": independently_committed["dt_bf16"]},
        "scope": "Every independent stage consumes its preceding independent complete state. Native-input readout/commit diagnostics are retrospective formula checks, never inputs to the sealed forecast."}


def analyze(root):
    started = time.monotonic()
    root = Path(root).resolve()
    request, job, collection = [read(root / n) for n in ("request.json", "job.json", "collection.json")]
    data = root / "extracted/flux-step-boundary-result"
    require(job["state"] == "succeeded" and job["result"]["returncode"] == 0
        and collection["job_id"] == job["job_id"] and collection["request_sha256"] == request["request_sha256"],
        "actual succeeded scheduler job/request/collection differs")
    collector = runpy.run_path(str(HERE / "collect_flux_step_boundary.py"))
    inventory = collector["inventory"](data)
    require(inventory == read(root / "collection-remote-before.json") == read(root / "collection-remote-after.json")
        == read(root / "collection-local-inventory.json")
        and collection["collector_source_sha256"] == sha(HERE / "collect_flux_step_boundary.py")
        and all(sha(root / name) == digest for name, digest in collection["bindings"].items())
        and job == read(root / "collection-job.json"), "authenticated collection inventory/source/bindings differ")
    collector["verify_manifest"](data, inventory, job["job_id"], request)
    require(job["result"]["diagnostics"]["command"] == read(root / "admission-command.json")["command"],
        "actual executed command differs from sealed admission command")
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"],
        "request seal differs")
    for name, digest in request["source_hashes"].items():
        require(sha(H["file_path"](root / "payload", name)) == digest, "staged worker source differs")
    for name, digest in request["independent_math_sources"].items():
        require(sha(HERE / name) == sha(root / "independent-math-freeze" / name) == digest,
            "pre-outcome independent numerical source changed")
    source, summary, plan, frozen, binding, prefix, segment = [read(data / n) for n in
        ("source.json", "summary.json", "plan.json", "plan-frozen.json", "binding.json", "prefix.json", "native-segment.json")]
    require(plan == request["config"] and summary["counts"] == COUNTS
        and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["source_hashes"] == request["source_hashes"]
        and source["context"] == plan["context"] and plan["context"]["seed"] == 89
        and source["checkpoint_sha256"] == canonical_hash(source["checkpoint_files"]) == plan["expected_checkpoint_sha256"]
        and frozen["plan_sha256"] == sha(data / "plan.json") and frozen["request_sha256"] == request["request_sha256"]
        and frozen["source_identity"] == canonical_hash(request["source_hashes"]), "source/checkpoint/plan/census binding differs")
    require(source["execution"]["device"]["type"] == "cuda" and source["execution"]["dtype"] == "torch.bfloat16",
        "actual execution profile differs")
    for suffix, expected in (
        ("diffusers/models/normalization.py", "e92ebbb130082578f3cff3361891a1adaa32331347e7bb09e4c70b208a0a6059"),
        ("diffusers/schedulers/scheduling_flow_match_euler_discrete.py", "8b96f25a6170480fb02decb58904e026250d6d68be5de8a7b8785b4c721e9d80"),
        ("saturn_pub/adapters/flux2.py", "2004b81971a258f1ca6295fe69d24767dc6bb6b92e103fa30cf6ccb0249b5a86"),
    ):
        matches = [digest for name, digest in source["implementation_sources"].items() if name.endswith(suffix)]
        require(matches == [expected], "actual native normalization/scheduler/adapter source differs from pinned law")
    for key in ("pre_model_source_abi", "native_source_abi"):
        reference = source[key]
        require(sha(data / reference["path"]) == reference["sha256"], "native source ABI receipt changed")
        receipt = read(data / reference["path"])
        require(receipt["verified"] is True and receipt["model_calls"] == 0
            and len(receipt["actual"]) == 5 and receipt["actual"] == receipt["expected"],
            "native source ABI was not actually verified")
    tx = H["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    report = read(data / "transaction-report.json")
    require(tx["phase"] == report["phase"] == "closed" and not report["failure"] and not report["violations"],
        "transaction lacks clean durable completion")
    require(CHAIN["SHARED"]["transaction_verifications_match"](tx, summary["transaction_verification"])
        and tx["event_head"] == report["event_head"], "transaction verification/event head differs")
    tree = H["NumericalTree"](data / "transaction")
    decode = lambda name, field: tree.decode(read(data / "transaction" / name)[field])
    forecast = decode("forecasts/step_boundary.json", "predictions")
    require(decode("frontiers/source_binding.json", "value") == binding
        and decode("forecasts/hypotheses.json", "predictions") == plan["hypotheses"], "sealed frontier/hypotheses differ")
    e = SavedEvidence(data)
    require(all(c["model_identity"] == source["model_identity"] and c["execution"] == source["execution"] for c in e.store["cuts"].values())
        and all(r["model_identity"] == source["model_identity"] and r["execution"] == source["execution"] for r in e.receipts.values()),
        "typed model/execution identity differs")
    prefix_ids = path(e, prefix["receipts"], prefix["initial"], prefix["pre6"], 68)
    from mdb.tensor_archive import load_tensor_archive

    require(prefix["pre6"] == binding["input_cut"] == segment["input_cut"]
        and exact_bits(e.array(source["encoding"]["conditioning"]), e.slot(prefix["initial"], "conditioning"))
        and exact_bits(load_tensor_archive(data, source["encoding"]["txt_ids"]), e.slot(prefix["initial"], "txt_ids")), "clean native source/encoding differs")
    expected_before = {**COUNTS, "adapter_steps": 68, "acts": 0, "receipt_emissions": 68, "images": 0, "vae_calls": 0}
    require(binding["counts_before_seal"] == expected_before
        and binding["target_entries_before_seal"] == {**{str(i): 0 for i in range(6, 20)}, "norm_out": 0, "proj_out": 0},
        "pre-seal target/work census differs")
    z3_cut = segment["next_input"]["cut"]
    reference_ids = path(e, segment["receipts"], prefix["pre6"], z3_cut, 16)
    for position, record in enumerate(segment["boundaries"]):
        require(record["index"] == position and record["receipt_id"] == reference_ids[position]
            and record["cut"] == segment["receipts"][position]["result"], "native reference boundary lineage differs")
        check_boundary(e, record)
        if position < 14:
            require(stripped(e, record["cut"], {"image", "text", "phase", "cursor"})
                == stripped(e, prefix["pre6"], {"image", "text", "phase", "cursor"}), "single block changed undeclared state")
    require(stripped(e, z3_cut, VOLATILE) == stripped(e, prefix["pre6"], VOLATILE)
        and {k: e.typed["unpack"](e.fields(z3_cut)[k]) for k in RESET} == {k: None for k in RESET}
        and tuple(e.point(z3_cut)[k] for k in ("logical_step", "phase", "index")) == (3, "project", None),
        "complete commit reset/carry contract differs")
    math_report = audit_forecast(e, binding, forecast, segment, plan)
    observed = decode("observations/native.json", "value")
    require(observed["native_segment_sha256"] == sha(data / "native-segment.json")
        and observed["continuation_observations_sha256"] == sha(data / "continuation-observations.json"), "native observation references differ")
    expected_emissions = [("prefix", rid) for rid in prefix_ids] + [("reference", rid) for rid in reference_ids]
    futures, pixels, latents, rows = {}, {}, {}, {}
    for name in ("native", "sham", "predicted"):
        row = read(data / "rows" / (name + ".json"))
        require(row == next(r for r in observed["rows"] if r["id"] == name)
            and row["parent"] == z3_cut and row["id"] == row["mode"] == name,
            "future row identity/parent differs")
        current = z3_cut
        require(len(row["acts"]) == (0 if name == "native" else 1), "typed latent Act census differs")
        for action in row["acts"]:
            expected = e.slot(z3_cut, "latent") if name == "sham" else forecast["commit"]["latent"]
            current = latent_act(e, action, current, expected)
            expected_emissions.append((name, action["receipt_id"]))
        require(row["next_input"]["cut"] == current, "next-step native input did not consume sealed latent")
        check_boundary(e, row["next_input"])
        ids = path(e, row["continuation_receipts"], current, row["final_cut"], 28)
        expected_emissions.extend((name, rid) for rid in ids)
        require(tuple(e.point(row["final_cut"])[k] for k in ("logical_step", "phase", "index")) == (4, "project", None),
            "native continuation did not commit Z4")
        latents[name], pixels[name] = e.array(row["final_latent"]), e.png(row["image"])
        require(exact_bits(latents[name], e.slot(row["final_cut"], "latent")), "final latent array differs from native typed cut")
        rows[name] = row
        futures[name] = {"native_transitions": len(ids), "typed_latent_acts": len(row["acts"]), "final_cut": row["final_cut"],
            "image": row["image"], "input_cut": current}
    emitted = CHAIN["SHARED"]["read_jsonl"](data / "receipt-emissions.jsonl")
    require([r["sequence"] for r in emitted] == list(range(170))
        and [(r["branch"], r["receipt"]) for r in emitted] == expected_emissions
        and {r for _, r in expected_emissions} == set(e.receipts), "physical receipt occurrence census differs")
    selected_entries = {"2": {str(i): 0 for i in range(6, 20)}, "3": {str(i): 0 for i in range(6, 20)}}
    for branch, rid in expected_emissions:
        receipt = e.receipts[rid]
        if branch == "prefix" or receipt["operation"] != "continue":
            continue
        point = e.point(receipt["parent"])
        key = str(point["index"])
        if point["phase"] == "single" and key in selected_entries[str(point["logical_step"])]:
            selected_entries[str(point["logical_step"])][key] += 1
    require(selected_entries == {"2": {str(i): 1 for i in range(6, 20)}, "3": {str(i): 3 for i in range(6, 20)}}
        and summary["target_calls"] == {**{str(i): 4 for i in range(6, 20)}, "norm_out": 4, "proj_out": 4},
        "current-step targets were confused with later-step targets")
    timeline = CHAIN["SHARED"]["read_jsonl"](data / "timeline.jsonl")
    seal_events = [x for x in timeline if x["kind"] == "step_boundary_sealed"]
    reference_events = [x for x in timeline if x["kind"] == "reference_transition_complete"]
    require(len(seal_events) == 1 and len(reference_events) == 16
        and seal_events[0]["seal"] == summary["seal"] and seal_events[0]["blocked_entries"] == 0
        and seal_events[0]["target_entries_before_seal"] == binding["target_entries_before_seal"]
        and max(x["sequence"] for x in timeline if x["kind"] == "block_forecast_complete") < seal_events[0]["sequence"]
        and seal_events[0]["sequence"] < min(x["sequence"] for x in reference_events), "seal/current-target ordering differs")
    scalar = read(data / "cuda-scalar-qualification.json")
    require(scalar["qualified"] is True and scalar["device"] == "cuda" and scalar["model_calls"] == 0
        and len(scalar["cases"]) == 3
        and [c["dt_fp32"] for c in scalar["cases"]] == [float(np.float32(x)) for x in (-.33333334, -.17320508, .14012345)]
        and all(c["values"] == 2048 and c["native_dtype"] == "torch.bfloat16"
            and c["dt_device"].startswith("cuda") and c["noise_device"].startswith("cuda")
            and c["native_vs_bf16_dt_mismatches"] == 0 and c["native_vs_fp32_dt_mismatches"] > 0 for c in scalar["cases"]),
        "actual CUDA weak-scalar qualification absent")
    control = {"sham_final_latent_exact_native": exact_bits(latents["sham"], latents["native"]),
        "sham_png_exact_native": rows["sham"]["image"]["sha256"] == rows["native"]["image"]["sha256"]}
    require(control == summary["control_observations"] == observed["control_observations"], "sham control report differs")
    endpoint = {}
    for name in ("sham", "predicted"):
        delta = pixels[name].astype(np.float64) - pixels["native"]
        endpoint[name] = {"latent": compare(latents[name], latents["native"]),
            "rgb": {"mae": float(np.mean(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta * delta))),
                "signed_mean": float(delta.mean()), "max_abs": float(np.abs(delta).max()),
                "changed_value_count": int(np.count_nonzero(delta)), "value_count": int(delta.size)}}
    continuation = audit_continuation(e, rows, segment, plan)
    return {"schema": "mdb-flux-step-boundary-audit-v1", "mechanics_status": "valid",
        "instrument_status": "applicable; source-bound independent full-state recurrences and actual CUDA scalar fixture",
        "trend_status": "observed; all signed numerical discrepancies and native endpoint effects retained", "terminal_status": "unassessed",
        "job_id": job["job_id"], "request_sha256": request["request_sha256"], "forecast_seal": summary["seal"],
        "counts": COUNTS, "mathematical_forecast": math_report, "futures": futures, "endpoint_comparisons": endpoint,
        "continuation": continuation,
        "control_observations": control, "audit_seconds": time.monotonic() - started,
        "auditor_sha256": sha(Path(__file__)), "independent_math_sources": request["independent_math_sources"],
        "collection_sha256": sha(root / "collection.json"),
        "scope": "One seed89/prompt with observed step2/pre6. Independent14-block/readout/Euler prediction; native next-step projection/transformer/VAE. No independent next-step or RGB prediction, universal concept dictionary, automatic family induction or speedup claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "job_id", "control_observations", "audit_seconds")}))
