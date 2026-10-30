"""Independent, framework-free verification of a sealed prospective route assay.

The explicit maintainer helper source supplies standard-library tensor/receipt
decoders. Scientific forecasts are read from the retained plan, never generated
from an instruction oracle or inferred from the measured outcome.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import runpy
import struct
from pathlib import Path

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_helpers(path):
    path = Path(path).resolve()
    namespace = runpy.run_path(str(path), run_name="predictive_route_verification_helpers")
    names = ("canonical", "digest", "sha", "read", "geometry", "flat", "raw_heads",
             "difference", "head_effects", "decode", "decoded_agreement", "hard_changes",
             "float32_descriptor", "continuation_roots", "receipt_seals", "references",
             "native_trace_heads", "descendants")
    require(all(callable(namespace.get(name)) for name in names), "maintainer helper API differs")
    return {name: namespace[name] for name in names}


def source_bindings(root, data, helpers):
    read, sha, digest = (helpers[name] for name in ("read", "sha", "digest"))
    request, report, job = (read(root / name) for name in ("request.json", "report.json", "job.json"))
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    request_id = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    require(request.get("request_sha256") == request_id == report.get("request_sha256"),
            "sealed request/report identity differs")
    require(report.get("job_id") == job.get("job_id"), "job/report identity differs")
    verified_sources = {}
    for name, expected in request["source_hashes"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        path = root / "payload" / relative
        if path.is_file():
            require(sha(path) == expected, "staged source differs: " + name)
            verified_sources[name] = expected
    worker_name = "predictive_route_campaign.py"
    require(worker_name in verified_sources, "retained source-bound worker is missing")
    manifest = report.get("file_manifest")
    require(isinstance(manifest, dict) and manifest, "artifact manifest missing")
    for name, entry in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
        path = data / relative
        require(not path.is_symlink() and path.is_file(), "missing artifact: " + name)
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "artifact differs: " + name)
    inventory = {str(path.relative_to(data)) for path in data.rglob("*")
                 if path.is_file() and path.name != "report.json"}
    require(inventory == set(manifest), "retained artifact inventory differs")
    inner = read(data / "report.json")
    require(all(inner.get(key) == report.get(key) for key in ("job_id", "request_sha256")),
            "inner report binding differs")
    forecast_path = data / "forecast.json"
    forecast, binding = read(forecast_path), read(data / "forecast-binding.json")
    forecast_sha = sha(forecast_path)
    require(request.get("forecast_sha256") == request["config"].get("forecast_sha256")
            == request["source_hashes"].get("forecast.json") == report.get("forecast_sha256")
            == binding.get("forecast_sha256") == forecast_sha,
            "forecast report/binding seal differs")
    plan_body = {key: value for key, value in forecast.items() if key != "plan_sha256"}
    require(hashlib.sha256(json.dumps(plan_body, sort_keys=True, allow_nan=False).encode()).hexdigest()
            == forecast.get("plan_sha256"), "forecast content seal differs")
    require(forecast.get("model_identity") == request.get("model_identity") == CHECKPOINT,
            "forecast/request checkpoint differs")
    require(binding.get("request_sha256") == request_id, "forecast/request join differs")
    require(binding.get("physical_forwards_at_binding") == 0,
            "forecast was not verified before model execution")
    require(binding.get("verification") == "sealed-before-first-model-forward",
            "worker forecast admission proof differs")
    require(binding.get("forecast_used_as_model_input") is False,
            "forecast/model input separation missing")
    require(binding.get("model_identity") == CHECKPOINT, "forecast checkpoint differs")
    return {"job_id": report["job_id"], "job_state": job.get("state"),
            "request_sha256": request_id, "request_file_sha256": sha(root / "request.json"),
            "report_file_sha256": sha(root / "report.json"), "forecast_file_sha256": forecast_sha,
            "forecast_content_id": digest(forecast), "forecast_binding": binding,
            "verified_staged_sources": verified_sources, "verified_artifact_files": len(manifest)}, request, report, forecast


def heads_equal(left, right, helpers):
    effects = helpers["head_effects"](left, right)
    byte_equal = {
        name: helpers["float32_descriptor"](helpers["geometry"](left[name]), helpers["flat"](left[name]))
              == helpers["float32_descriptor"](helpers["geometry"](right[name]), helpers["flat"](right[name]))
        for name in effects}
    return {"heads": effects, "fp32_bytes_equal": byte_equal,
            "all_eleven_heads_exact": len(effects) == 11 and all(byte_equal.values())}


def reader_value(heads, reader):
    value = heads[reader.get("head", reader.get("field", "regs"))]
    for index in reader.get("coordinate", [0, reader.get("register", 7), reader.get("bit")]):
        value = value[index]
    require(type(value) in {float, int}, "reader does not select one scalar")
    return float(value)


def signed(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def fp32_sum(left, right, helpers):
    require(helpers["geometry"](left) == helpers["geometry"](right), "carrier operand geometry differs")
    return [struct.unpack("<f", struct.pack("<f", a + b))[0]
            for a, b in zip(helpers["flat"](left), helpers["flat"](right), strict=True)]


def register_words(value):
    return [int(word, 16) if isinstance(word, str) else int(word) for word in value]


def register_comparison(observed, expected):
    observed, expected = register_words(observed), register_words(expected)
    require(len(observed) == len(expected) == 8, "register bank must have eight words")
    differences = {str(index): [bit for bit in range(64) if (left ^ right) & (1 << bit)]
                   for index, (left, right) in enumerate(zip(observed, expected, strict=True))
                   if left != right}
    return {"exact": not differences, "observed": [f"{word:016x}" for word in observed],
            "expected": [f"{word:016x}" for word in expected], "wrong_bits": differences,
            "wrong_bit_count": sum(map(len, differences.values()))}


def own_state_fixture(op, incoming):
    """Supplied fixture grammar; diagnostic oracle, never an execution input."""
    result = register_words(incoming)
    if op == "sub_r64_32":
        result[7] = (result[7] - 1) % (1 << 64)
    elif op == "xor_r64_r64":
        result[7] ^= result[6]
    elif op == "mov_r64_r64":
        result[7] = result[6]
    else:
        raise ValueError("outside the sealed fixture grammar: " + op)
    return result


def continuation_observations(row, case, helpers):
    continuation = row["continuation"]
    first = helpers["decode"](row["heads"])
    second = helpers["decode"](continuation["second_heads"]) if "second_heads" in continuation else None
    commit = continuation["first_commit"]
    before = continuation.get("before", commit.get("before"))
    after = commit.get("after", {})
    if commit.get("committed") is not True:
        return {"status": "not-applicable", "first_decode": first, "second_decode": None,
                "stop_causes": commit.get("stop_causes"), "reason": continuation.get("reason")}
    consistency = {key: after.get(key) == first[key]
                   for key in ("register_words", "flags_CF_ZF_SF_OF")}
    consistency["byte_offset"] = after["byte_offset"] == before["byte_offset"] + first["length"]
    consistency["instruction_index"] = after["instruction_index"] == before["instruction_index"] + 1
    require(all(consistency.values()),
            "first architectural commit differs from retained model prediction")
    if second is None:
        return {"status": "not-applicable", "first_decode": first, "second_decode": None,
                "first_commit_consistency": consistency, "before": before, "after": after,
                "reason": continuation.get("reason")}
    roots = helpers["continuation_roots"](after, continuation["input_roots"])
    require(roots["status"] == "all-three-roots-verified", "successor roots differ from committed predictions")
    if "input_values" in continuation:
        descriptors = [helpers["float32_descriptor"](helpers["geometry"](item), helpers["flat"](item))
                       for item in continuation["input_values"]]
        require(descriptors == roots["expected"], "retained successor root values differ")
    return {"status": "observed", "first_decode": first, "second_decode": second,
            "first_commit_consistency": consistency, "before": before, "after": after,
            "successor_input_roots": roots,
            "recorded_second_decode": helpers["decoded_agreement"](continuation.get("second_decode"), second),
            "own_state_local_fixture": register_comparison(second["register_words"],
                own_state_fixture(case["op_label"], after["register_words"])),
            "fixture_authority": "supplied SUB/XOR/MOV grammar applied to actual first model prediction; never fed to the model",
            "own_state_scope": "second instruction at model-predicted offset; fresh model call resets hidden state"}


def score_case(case, prediction, arms, helpers):
    decoded = {arm: helpers["decode"](row["heads"]) for arm, row in arms.items()}
    native = decoded["identity"]["register_words"]
    expected_native = [0] * 8
    expected_native[6], expected_native[7] = case["register_words"][6], case["expected_first_rdi"]
    outcomes = []
    arm_results = {}
    family = case["op_label"]
    for arm, forecast in prediction["arms"].items():
        row = arms[arm]
        continuation = continuation_observations(row, case, helpers)
        first, second = decoded[arm], continuation["second_decode"]
        first_score = None if forecast["first_register_words"] is None else register_comparison(
            first["register_words"], forecast["first_register_words"])
        second_score = None if forecast["second_register_words"] is None or second is None else register_comparison(
            second["register_words"], forecast["second_register_words"])
        relative = register_comparison(first["register_words"], native)
        prior = register_comparison(first["register_words"], case["register_words"])
        arm_results[arm] = {"first_register_forecast": first_score,
            "second_register_forecast": second_score, "native_relative_registers": relative,
            "incoming_registers_comparison": prior, "continuation": continuation,
            "length_forecast": {"expected": forecast["length"], "observed": first["length"],
                                "exact": forecast["length"] == first["length"]},
            "abstentions": {key: value for key, value in forecast.items() if value == "abstain"},
            "native_fixture_first": register_comparison(first["register_words"], expected_native)}
        for horizon, score in (("first", first_score), ("second", second_score)):
            pattern = "pattern_0001" if arm in {"post0_zero", "projection_sham"} else (
                "pattern_0003" if arm in {"post1_zero", "post2_zero"} else (
                "route_0002" if arm == "projection_native" else "control"))
            outcomes.append({"case_id": case["case_id"], "family": family, "kind": "full-register-" + horizon,
                "arm": arm, "pattern_id": pattern, "status": ("unobserved" if horizon == "second"
                    and second is None and forecast["second_register_words"] is not None else "abstain") if score is None else (
                    "supported" if score["exact"] else "counterexample")})
    raw = {arm: reader_value(row["heads"], case["reader"]) for arm, row in arms.items()}
    differences = {"post0_zero_minus_identity": raw["post0_zero"] - raw["identity"],
                   "projection_native_minus_post0_zero": raw["projection_native"] - raw["post0_zero"]}
    signs = {}
    for name, expected in prediction["signed_effects"].items():
        value = differences[name]
        observed = 1 if value > 0 else -1 if value < 0 else 0
        signs[name] = {"signed_delta": value, "observed_sign": observed,
                       "expected_sign": expected, "matches": observed == expected}
        outcomes.append({"case_id": case["case_id"], "family": family, "kind": "reader-sign",
            "name": name, "pattern_id": "pattern_0001" if name.startswith("post0") else "route_0002",
            "status": "supported" if observed == expected else "counterexample"})
    distances = {arm: abs(raw[arm] - raw["identity"]) for arm in ("projection_native", "skip_native")}
    ranking = distances["projection_native"] < distances["skip_native"]
    route = {"reader": case["reader"], "raw_logits": raw, "distances_from_native": distances,
             "tie": distances["projection_native"] == distances["skip_native"],
             "expected": prediction["route_ranking"]["expected"], "observed": ranking,
             "matches": ranking == prediction["route_ranking"]["expected"]}
    outcomes.append({"case_id": case["case_id"], "family": family, "kind": "reader-route-ranking",
                     "pattern_id": "route_0002", "status": "supported" if route["matches"] else "counterexample"})
    baselines = {}
    for name in ("no_effect_at_any_post", "prior_state_at_every_post"):
        baselines[name] = {arm: register_comparison(decoded[arm]["register_words"], expected)
                           for arm, expected in prediction["baselines"][name].items()}
    for name in ("constant_positive_effect", "constant_negative_effect"):
        baselines[name] = {key: {"expected_sign": value, "observed_sign": signs[key]["observed_sign"],
                                "matches": value == signs[key]["observed_sign"]}
                           for key, value in prediction["baselines"][name].items()}
    return {"case_id": case["case_id"], "family": family, "reader": case["reader"],
            "arms": arm_results, "signed_effects": signs, "route_ranking": route,
            "comparison_baselines": baselines, "abstentions": prediction["abstentions"],
            "native_correctness": {"first_registers": register_comparison(native, expected_native),
                "second_registers": None if arm_results["identity"]["continuation"]["second_decode"] is None
                    else register_comparison(arm_results["identity"]["continuation"]["second_decode"]["register_words"],
                        own_state_fixture(case["op_label"], expected_native)),
                "authority": "supplied fixture grammar; independent of forecast support"}}, outcomes


def descriptor_check(values, descriptor, helpers, label):
    actual = helpers["float32_descriptor"](helpers["geometry"](values), helpers["flat"](values))
    require(all(descriptor.get(key) == value for key, value in actual.items()),
            "retained tensor descriptor differs: " + label)
    return actual


def numeric_audit(observed, native, declared, helpers):
    proof = heads_equal(observed, native, helpers)
    require(set(declared) == set(proof["heads"]) and len(declared) == 11,
            "native audit must include all eleven heads")
    for head, effects in proof["heads"].items():
        native_fp32_max = struct.unpack("<f", struct.pack("<f", effects["max_abs"]))[0]
        require(declared[head]["exact"] == proof["fp32_bytes_equal"][head]
                and declared[head]["max_abs"] == native_fp32_max,
                "declared native audit differs from raw independent heads: " + head)
        effects["native_fp32_subtraction_max_abs"] = native_fp32_max
    return proof


def verify_case_record(record, case, trace, forecast_sha, helpers):
    require(record["case_id"] == case["case_id"] and record["forecast_sha256"] == forecast_sha,
            "case/forecast binding differs")
    condition = record["condition"]
    require(condition["program_hex"] == case["program_hex"]
            and register_words(condition["register_words"]) == case["register_words"]
            and condition["flags_CF_ZF_SF_OF"] == case["flags"]
            and condition["byte_offset"] == condition["instruction_index"] == 0,
            "initial case inputs differ from sealed envelope")
    roots = helpers["continuation_roots"](condition, record["input_roots"])
    require(roots["status"] == "all-three-roots-verified", "initial roots differ from sealed raw inputs")
    for index, values in enumerate(record["input_values"]):
        descriptor_check(values, record["input_roots"][index], helpers, "initial root")
    expected_digest = hashlib.sha256(json.dumps(record["input_roots"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(record["input_digest"] == expected_digest, "case input descriptor digest differs")
    require(trace.get("model_identity") == CHECKPOINT and trace.get("input_identity") == expected_digest
            and trace.get("capture_status") == "complete"
            and trace.get("execution") == {"backend": "cpu", "dtype": "float32", "threads": 1},
            "native trace input/model/execution profile differs")
    require(helpers["digest"]({key: value for key, value in trace.items() if key != "fingerprint"})
            == trace["fingerprint"], "native trace content seal differs")
    trace_roots = [{key: item[key] for key in ("shape", "dtype", "sha256")}
                   for item in trace["values"].values() if item.get("origin") == "input"]
    require(sorted(map(helpers["digest"], trace_roots)) == sorted(map(helpers["digest"], roots["expected"])),
            "actual traced input roots differ from the sealed serving inputs")
    require(record["trace_fingerprint"] == trace["fingerprint"], "case trace binding differs")
    trace_heads = helpers["native_trace_heads"](trace, record["baseline_heads"])
    require(all(row["exact_recorded_bytes"] for row in trace_heads.values()),
            "independent native heads differ from captured trace outputs")
    for name in ("native_capture_control", "native_replay_control"):
        require(set(record[name]) == set(record["baseline_heads"]), "incomplete native control head inventory")
        require(all(row["exact"] is True and row["max_abs"] == 0 for row in record[name].values()),
                "unchanged capture/replay control differs")
    ports = record["ports"]
    producer = trace["events"][trace["values"][ports["carrier"]]["producer"]]
    require(producer == record["carrier_producer"] and producer["operator"] == "aten.add.Tensor"
            and set(producer["inputs"]) == {ports["skip"], ports["projection"]},
            "carrier is not the source-bound skip/projection residual")
    require(producer["arguments"].get("kwargs", {}).get("alpha", 1) == 1,
            "carrier residual coefficient differs")
    require(record["projection_pc"] == trace["values"][ports["projection"]]["producer"] + 1
            and record["carrier_pc"] == trace["values"][ports["carrier"]]["producer"] + 1,
            "carrier safe point binding differs")
    donors = record["donor_values"]
    for name, values in donors.items():
        descriptor_check(values, record["donor_descriptors"][name], helpers, "native donor " + name)
    require(fp32_sum(donors["skip"], donors["projection"], helpers) == helpers["flat"](donors["carrier"]),
            "native carrier does not reconstruct from its retained native operands")
    require(set(record["donor_exact"]) == {"skip", "projection", "carrier"}
            and all(value is True for value in record["donor_exact"].values()),
            "independent native donor and live replay donor differ")
    for site, law in record["laws"].items():
        require(law["module"] == site and law["site_id"].startswith(record["program_id"] + "/site/"),
                "law/program identity differs")
        require(law["outputs"][0] in trace["values"]
                and trace["values"][law["outputs"][0]]["producer"] + 1 <= law["exit"],
                "law output producer differs")
    return {"initial_roots": roots, "trace_heads": trace_heads,
            "native_controls": {name: record[name] for name in ("native_capture_control", "native_replay_control")},
            "donor_add_reconstruction_exact": True, "program_id": record["program_id"],
            "trace_fingerprint": trace["fingerprint"], "law_sites": record["laws"], "ports": ports}


def verify_receipts(data, records, traces, rows, report, helpers):
    acts = helpers["read"](data / "act-receipts.json")
    transitions = [json.loads(line) for line in (data / "transition-receipts.jsonl").read_text().splitlines()
                   if line.strip()]
    require(len(acts) == report["act_receipts"] and len(transitions) == report["replay_operations"],
            "public receipt inventory differs from report")
    seals = helpers["receipt_seals"](acts) + helpers["receipt_seals"](transitions)
    require(len(seals) == len(acts) + len(transitions), "public receipt seal missing")
    by_id = {(item["branch"], item["receipt"]["fingerprint"]): item["receipt"] for item in acts}
    require(len(by_id) == len(acts), "duplicate Act receipt identity")
    per_branch = collections.defaultdict(list)
    for item in transitions:
        branch, receipt = item["branch"], item["receipt"]
        case_id = branch.split("/")[0]
        trace, record = traces[case_id], records[case_id]
        pc = receipt["before_pc"]
        require(type(pc) is int and 0 <= pc < len(trace["events"])
                and receipt["after_pc"] == pc + 1 and receipt["operator"] == trace["events"][pc]["operator"]
                and receipt["trace_fingerprint"] == trace["fingerprint"]
                and receipt["program_id"] == record["program_id"], "transition source binding differs")
        per_branch[branch].append(pc)
    joined = []
    for row in rows:
        case_id, arm = row["case_id"], row["arm"]
        record, trace = records[case_id], traces[case_id]
        restoration = row["restoration"]
        branch = case_id + "/" + arm
        law = record["laws"][row["source_site"]]
        source = by_id[(branch, restoration["source_act"])]
        require(source["operation"] == "mdb.discovery.ssa-act"
                and source["trace_fingerprint"] == trace["fingerprint"]
                and source["value_id"] == law["outputs"][0] == restoration["source_value_id"]
                and source["pc"] == law["exit"], "source Act/trace/site binding differs")
        parent = record["parent_cuts"][row["source_site"]]
        require(restoration["source_parent"] == row["parent_fingerprint"] == parent["fingerprint"],
                "source common-parent cut binding differs")
        for name, receipt_name, descriptor_name in (("output", "before", "source_before"),
                                                   ("payload", "after", "source_payload")):
            actual = descriptor_check(row["source_values"][name], restoration[descriptor_name], helpers, "source " + name)
            descriptor_check(row["source_values"][name], source[receipt_name], helpers, "source Act " + name)
            require(actual["shape"] == trace["values"][source["value_id"]]["shape"], "source tensor geometry differs")
        descriptor_check(row["source_values"]["output"], trace["values"][source["value_id"]], helpers,
                         "native source output before intervention")
        require(len(law["inputs"]) == 1, "source law has ambiguous activation input")
        descriptor_check(row["source_values"]["input"], trace["values"][law["inputs"][0]], helpers,
                         "native source receiving state")
        original, payload = map(helpers["flat"], (row["source_values"]["output"], row["source_values"]["payload"]))
        require(payload == (original if arm == "identity" else [0.] * len(original)), "source Act is not the declared dose")
        require(restoration["donor_case_id"] == case_id and restoration["donor_live_cut"] == record["donor_cut"],
                "restoration donor case/cut binding differs")
        secondary = None
        if "act" in restoration:
            secondary = by_id[(branch, restoration["act"])]
            require(secondary["value_id"] == restoration["site"]
                    and secondary["trace_fingerprint"] == trace["fingerprint"], "secondary Act port/trace differs")
            values = row["restoration_values"]
            for name in ("before", "payload"):
                descriptor_check(values[name], restoration[name], helpers, "secondary " + name)
                descriptor_check(values[name], secondary["after" if name == "payload" else "before"], helpers,
                                 "secondary Act " + name)
            expected_port = record["ports"]["projection" if arm.startswith("projection") else "carrier"]
            expected_pc = record["projection_pc"] if arm.startswith("projection") else record["carrier_pc"]
            require(secondary["value_id"] == expected_port and secondary["pc"] == expected_pc,
                    "secondary Act is not at the declared live frontier")
            actual_payload = helpers["flat"](values["payload"])
            if arm == "projection_native":
                expected = helpers["flat"](record["donor_values"]["projection"])
            elif arm == "projection_sham":
                expected = helpers["flat"](values["before"])
            elif arm == "carrier_native":
                expected = helpers["flat"](record["donor_values"]["carrier"])
            elif arm == "skip_native":
                descriptor_check(values["edited_projection"], restoration["edited_projection"], helpers, "edited projection")
                expected = fp32_sum(record["donor_values"]["skip"], values["edited_projection"], helpers)
            else:
                raise ValueError("unexpected secondary Act arm")
            require(actual_payload == expected, "secondary Act payload does not match declared construction")
        pcs = per_branch[branch]
        require(pcs == list(range(law["exit"], len(trace["events"]))),
                "branch transition receipts do not cover exactly its replay suffix")
        joined.append({"case_id": case_id, "arm": arm, "source_act": source["fingerprint"],
                       "secondary_act": None if secondary is None else secondary["fingerprint"],
                       "parent_fingerprint": row["parent_fingerprint"], "suffix_operations": len(pcs)})
    return {"verified_transition_receipts": len(transitions), "verified_act_receipts": len(acts),
            "row_act_joins": joined, "authority": "public typed seals, trace/PC/SSA, full tensor payload and source-parent joins"}


def count_outcomes(rows, key):
    result = {}
    for row in rows:
        group = row[key]
        result.setdefault(group, collections.Counter())[row["status"]] += 1
    return {group: {status: counts.get(status, 0) for status in (
        "supported", "counterexample", "abstain", "unobserved")} for group, counts in sorted(result.items())}


def analyze(root, data, helper_source):
    helpers = load_helpers(helper_source)
    read, sha = helpers["read"], helpers["sha"]
    source, request, report, forecast = source_bindings(root, data, helpers)
    source["helper_source"] = str(Path(helper_source).resolve())
    source["helper_sha256"] = sha(Path(helper_source))
    source["analyzer_sha256"] = sha(Path(__file__))
    cases = {case["case_id"]: case for case in forecast["cases"]}
    predictions = {prediction["case_id"]: prediction for prediction in forecast["predictions"]}
    require(len(cases) == len(forecast["cases"]) == 6 and set(cases) == set(predictions),
            "sealed six-case forecast inventory differs")
    require(source["forecast_binding"]["case_ids"] == list(cases), "worker bound different forecast cases")
    require(source["forecast_binding"].get("plan_sha256") == forecast["plan_sha256"],
            "worker bound different forecast plan content")
    records_list = read(data / "cases.json")
    records = {record["case_id"]: record for record in records_list}
    require(len(records) == len(records_list) == report["case_count"] and set(records) == set(cases),
            "retained native case inventory differs")
    rows = read(data / "followups.json")["rows"]
    require(len(rows) == report["followup_rows"], "retained branch count differs from report")
    errors = read(data / "errors.json")
    require(errors == report["errors"], "retained mechanics errors differ from report")
    traces, case_proofs = {}, {}
    for case_id, record in records.items():
        path = data / "cases" / case_id
        require(read(path / "case.json") == record, "duplicated case record differs")
        trace = read(path / "trace.json")
        traces[case_id] = trace
        case_proofs[case_id] = verify_case_record(record, cases[case_id], trace,
                                                 source["forecast_file_sha256"], helpers)
    receipt_proof = verify_receipts(data, records, traces, rows, report, helpers)
    by_case = collections.defaultdict(dict)
    row_proofs, index = [], []
    for row in rows:
        case_id, arm = row["case_id"], row["arm"]
        record, case = records[case_id], cases[case_id]
        require(arm in predictions[case_id]["arms"] and arm not in by_case[case_id], "unknown or duplicate branch")
        require(read(data / "cases" / case_id / "branches" / (arm + ".json")) == row,
                "duplicated branch record differs")
        row_body = {key: value for key, value in row.items() if key != "id"}
        row_id = "variant:" + hashlib.sha256(json.dumps(row_body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(row["id"] == row_id, "worker observation content identity differs")
        require(row["forecast_sha256"] == source["forecast_file_sha256"]
                and row["input_digest"] == record["input_digest"]
                and row["program_id"] == record["program_id"]
                and row["trace_fingerprint"] == record["trace_fingerprint"], "row source binding differs")
        by_case[case_id][arm] = row
        native_audit = numeric_audit(row["heads"], row["native_audit_heads"], row["native_audit"], helpers)
        require(native_audit["all_eleven_heads_exact"], "typed replay differs from independent native audit")
        native_comparison = numeric_audit(row["heads"], record["baseline_heads"], row["baseline_comparison"], helpers)
        decoded = helpers["decode"](row["heads"])
        recorded_decode = helpers["decoded_agreement"](row["decode"], decoded)
        recorded_native_decode = helpers["decoded_agreement"](row["native_audit_decode"], helpers["decode"](row["native_audit_heads"]))
        require(recorded_decode["status"] == recorded_native_decode["status"] == "verified",
                "retained decoder projection differs from raw heads")
        reachable = helpers["descendants"](traces[case_id], record["laws"][row["source_site"]]["outputs"])
        changed_heads = sorted(head for head, effect in native_comparison["heads"].items() if effect["changed_elements"])
        row_proofs.append({"case_id": case_id, "arm": arm, "row_id": row_id,
            "independent_native_audit": native_audit, "baseline_comparison": native_comparison,
            "structurally_reachable_heads": reachable["heads"], "observed_changed_heads": changed_heads,
            "changed_outside_structural_descendants": sorted(set(changed_heads) - set(reachable["heads"]))})
        identity = {"model_identity": CHECKPOINT, "input_digest": record["input_digest"],
            "program_id": record["program_id"], "trace_fingerprint": record["trace_fingerprint"],
            "site_id": record["laws"][row["source_site"]]["site_id"], "arm": arm,
            "parent_fingerprint": row["parent_fingerprint"], "heads_digest": helpers["digest"](row["heads"]),
            "forecast_sha256": source["forecast_file_sha256"], "source_row_id": row_id}
        index.append({"id": "computation:" + helpers["digest"](identity), "identity": identity,
            "identity_kind": "traced-scalar-intervention-observation", "semantic_operation": None,
            "source": {"path": "cases/" + case_id + "/branches/" + arm + ".json",
                       "sha256": sha(data / "cases" / case_id / "branches" / (arm + ".json"))}})
    results, outcomes, controls = [], [], []
    incomplete = []
    for case_id, case in cases.items():
        arms = by_case[case_id]
        missing = sorted(set(predictions[case_id]["arms"]) - set(arms))
        if missing:
            incomplete.append({"case_id": case_id, "missing_arms": missing,
                               "reason": "retained mechanics failure; no scientific null assigned"})
            continue
        record = records[case_id]
        checks = {
            "identity_equals_native": heads_equal(arms["identity"]["heads"], record["baseline_heads"], helpers),
            "projection_sham_equals_post0_zero": heads_equal(arms["projection_sham"]["heads"], arms["post0_zero"]["heads"], helpers),
            "complete_carrier_equals_native": heads_equal(arms["carrier_native"]["heads"], record["baseline_heads"], helpers)}
        require(all(check["all_eleven_heads_exact"] for check in checks.values()),
                "identity/no-op/complete-carrier mechanical closure differs")
        controls.append({"case_id": case_id, **checks})
        result, case_outcomes = score_case(case, predictions[case_id], arms, helpers)
        results.append(result)
        outcomes.extend(case_outcomes)
    weights = read(data / "weight-identity.json")
    require(weights["before"] == weights["after"], "frozen model weights changed")
    geometry = report["physical_geometry"]
    require(len(geometry) == report["physical_model_forwards"]
            and [item["attempt"] for item in geometry] == list(range(1, len(geometry) + 1))
            and all(item["batch"] == 1 for item in geometry), "physical forward accounting differs")
    baseline_counts = {}
    for name in ("no_effect_at_any_post", "prior_state_at_every_post", "constant_positive_effect", "constant_negative_effect"):
        entries = [{"case_id": result["case_id"], "family": result["family"], "arm_or_sign": arm,
                    "status": "supported" if score.get("exact", score.get("matches")) else "counterexample"}
                   for result in results for arm, score in result["comparison_baselines"][name].items()]
        baseline_counts[name] = {"per_case": count_outcomes(entries, "case_id"),
                                "per_family": count_outcomes(entries, "family"), "outcomes": entries}
    return {"schema": "mdb-predictive-route-analysis-v1", "source": source,
        "mechanics_status": "valid" if not errors and not incomplete else "partial-mechanics-errors",
        "instrument_status": "independent-all-eleven-head-native-parity-verified",
        "trend_status": "prospective-per-row-forecast-results", "terminal_status": "unestablished",
        "empirical_old_evidence_scope": forecast["empirical_scope"],
        "proposed_applicability": forecast["candidate_applicability"],
        "supplied": forecast["supplied"], "learned_from_old_evidence": forecast["learned_from_old_evidence"],
        "case_verification": case_proofs, "receipt_verification": receipt_proof,
        "row_verification": row_proofs, "controls": controls,
        "results": results, "forecast_outcomes": outcomes,
        "counts": {"per_case": count_outcomes(outcomes, "case_id"),
                   "per_family": count_outcomes(outcomes, "family"),
                   "per_pattern": count_outcomes(outcomes, "pattern_id"),
                   "per_observable": count_outcomes(outcomes, "kind")},
        "comparison_baseline_counts": baseline_counts,
        "counterexamples": [row for row in outcomes if row["status"] == "counterexample"],
        "incomplete_cases": incomplete, "mechanics_errors": errors,
        "generic_record_index": {"schema": "mdb-predictive-route-record-index-v1", "records": index,
            "scope": "generic immutable content identities; no automatic semantic naming"},
        "execution": {key: report.get(key) for key in ("physical_model_forwards", "physical_forward_limit",
            "replay_operations", "act_receipts", "cpu_threads", "backend", "worker_seconds", "physical_batching")},
        "limits": ["Six exact new-operation envelopes expand a previous ADD-only empirical scope.",
            "Operation grammar, semantic targets, reader coordinates and forecast signs were supplied.",
            "Static descendants express architectural capacity; actual signed effects and restorations measure use.",
            "Native fixture correctness is reported independently from prospective forecast agreement.",
            "Second-state frozen forecasts and local grammar applied to actual first predictions are separate.",
            "Every counterexample, tie, zero sign and stated abstention is retained; no scientific stopping threshold.",
            "Carrier restoration establishes broad conditional closure, not a minimal causal cut.",
            "No training-unseen operation, persistent hidden memory or universal semantic route is established."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--helper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    data = (args.data_dir or root / "extracted" / "predictive-route-result").resolve()
    output = args.output.resolve()
    require(not output.exists(), "analysis output already exists; preserve immutable earlier analysis")
    snapshot = output.with_suffix(".source.py")
    helper_snapshot = output.with_suffix(".helper.source.py")
    require(not snapshot.exists(), "analyzer source snapshot already exists")
    require(not helper_snapshot.exists(), "verification helper snapshot already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    with helper_snapshot.open("xb") as stream:
        stream.write(args.helper_source.read_bytes())
    result = analyze(root, data, args.helper_source)
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "mechanics_status": result["mechanics_status"],
                      "counterexamples": len(result["counterexamples"]), "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
