"""Independent retained-evidence analysis of explicit suffix computation forecasts.

No model or component equations are executed. NumPy reads sealed arrays and
computes signed comparisons/readout decompositions. Public typed receipt and
ABI helpers come from an explicitly supplied, source-snapshotted maintainer file.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def native_seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_helpers(path):
    namespace = runpy.run_path(str(path), run_name="explicit_computation_verification_helpers")
    names = ("receipt_seals", "continuation_roots", "decode", "decoded_agreement", "references")
    require(all(callable(namespace.get(name)) for name in names), "verification helper API differs")
    return {name: namespace[name] for name in names}


class ArrayStore:
    """Lossless NPY identity checks; arbitrary pickle arrays are forbidden."""

    def __init__(self, root):
        self.root = Path(root)
        self.checked = {}
        self.cache = {}

    def load(self, reference):
        require(isinstance(reference, dict) and {"path", "sha256", "shape", "dtype"} <= set(reference),
                "numeric artifact is not a complete array reference")
        relative = Path(reference["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe array artifact path")
        path = self.root / relative
        require(path.is_file() and not path.is_symlink(), "array artifact missing")
        key = str(relative)
        if key not in self.cache:
            require(sha(path) == reference["sha256"], "array artifact byte seal differs: " + key)
            array = np.load(path, allow_pickle=False)
            require(isinstance(array, np.ndarray) and array.dtype.kind in "fiu", "array must contain real numbers")
            require(np.isfinite(array).all(), "array contains nonfinite evidence")
            array.flags.writeable = False
            self.cache[key] = array
        array = self.cache[key]
        require(list(array.shape) == reference["shape"] and array.dtype == np.dtype(reference["dtype"]),
                "array descriptor geometry/dtype differs: " + key)
        prior = self.checked.get(key)
        require(prior is None or prior == reference, "one numeric artifact has inconsistent reference descriptors")
        self.checked[key] = reference
        return array

    def mapping(self, value):
        return {name: self.load(reference) for name, reference in value.items()}

    def verify_tree(self, value):
        if isinstance(value, dict):
            if {"path", "sha256", "shape", "dtype"} <= set(value):
                self.load(value)
            else:
                for nested in value.values():
                    self.verify_tree(nested)
        elif isinstance(value, list):
            for nested in value:
                self.verify_tree(nested)


def difference(observed, forecast):
    observed, forecast = np.asarray(observed), np.asarray(forecast)
    require(observed.shape == forecast.shape, "forecast/observed tensor geometry differs")
    delta = observed.astype(np.float64) - forecast.astype(np.float64)
    maximum = int(np.argmax(np.abs(delta))) if delta.size else None
    return {"shape": list(delta.shape), "elements": int(delta.size),
        "exact_numeric": bool(np.array_equal(observed, forecast)),
        "changed_elements": int(np.count_nonzero(delta)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.,
        "l2": float(np.linalg.norm(delta.reshape(-1))), "signed_sum": float(delta.sum()),
        "signed_mean": float(delta.mean()) if delta.size else 0.,
        "minimum_signed": float(delta.min()) if delta.size else 0.,
        "maximum_signed": float(delta.max()) if delta.size else 0.,
        "largest_residual": None if maximum is None else {
            "coordinate": [int(item) for item in np.unravel_index(maximum, delta.shape)],
            "signed": float(delta.reshape(-1)[maximum])}}


def compare_mappings(observed, forecast):
    require(set(observed) == set(forecast), "forecast/observed field inventory differs")
    return {name: difference(observed[name], forecast[name]) for name in sorted(observed)}


def decode(heads):
    require(heads["regs"].shape == (1, 8, 64) and heads["flags"].shape == (1, 4)
            and heads["length"].shape == (1, 15), "R3 forecast consumer ABI differs")
    return {"register_words": [f"{sum(1 << bit for bit, value in enumerate(word) if value > 0):016x}"
                               for word in heads["regs"][0]],
            "flags_CF_ZF_SF_OF": [int(value > 0) for value in heads["flags"][0]],
            "length": int(heads["length"][0].argmax()) + 1,
            "store": bool(heads["store"][0] > 0), "syscall": bool(heads["syscall"][0] > 0)}


def decoded_comparison(observed, expected):
    wrong = {}
    for register, (left, right) in enumerate(zip(observed["register_words"], expected["register_words"], strict=True)):
        mask = int(left, 16) ^ int(right, 16)
        if mask:
            wrong[str(register)] = [bit for bit in range(64) if mask & (1 << bit)]
    fields = {key: observed[key] == expected[key] for key in observed}
    return {"fields": fields, "full_register_bank_equal": not wrong,
            "wrong_register_bits": wrong, "wrong_register_bit_count": sum(map(len, wrong.values())),
            "wrong_flag_indices": [index for index, (a, b) in enumerate(zip(
                observed["flags_CF_ZF_SF_OF"], expected["flags_CF_ZF_SF_OF"], strict=True)) if a != b],
            "observed": observed, "forecast": expected}


def register_terms(stages, reg_bits, weight, bias):
    weight, bias = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    require(weight.shape == (8, 48) and bias.shape == (8,), "register readout weight geometry differs")
    arrays = {name: np.asarray(stages[name], dtype=np.float64) for name in ("S", "U", "M", "P2")}
    require(all(value.shape == (1, 64, 48) for value in arrays.values()), "readout stage geometry differs")
    project = lambda value: (value @ weight.T).transpose(0, 2, 1)
    return {"prior": np.asarray(reg_bits, dtype=np.float64) * 8. - 4.,
            "skip": project(arrays["S"]), "projection": project(arrays["U"]),
            "carrier_override": project(arrays["M"] - (arrays["S"] + arrays["U"])),
            "post": project(arrays["P2"]),
            "bias": np.broadcast_to(bias[None, :, None], (1, 8, 64)).copy()}


def term_closure(terms, heads):
    total = sum(terms.values())
    return {"reglogit_minus_component_sum": difference(heads["regs"], total),
            "per_register": {str(index): difference(heads["regs"][:, index], total[:, index]) for index in range(8)},
            "components": sorted(terms),
            "authority": "full Wreg dot products of retained S/U/M/P2, copy-prior and bias; FP64 analysis, native rounding retained"}


def head_decisions(observed, forecast):
    result = {}
    for name in sorted(observed):
        if name in {"dest", "src", "width", "length"}:
            left, right = observed[name].argmax(axis=-1), forecast[name].argmax(axis=-1)
            result[name] = {"rule": "argmax; length index+1", "equal": bool(np.array_equal(left, right)),
                            "observed_indices": left.tolist(), "forecast_indices": right.tolist()}
        else:
            left, right = observed[name] > 0, forecast[name] > 0
            differences = np.argwhere(left != right)
            result[name] = {"rule": "strict logit>0", "equal": not len(differences),
                "decisions": int(left.size), "different": int(len(differences)),
                "different_coordinates": differences.tolist()}
    return result


def verify_sources(root, data):
    request, report, job = (read(root / name) for name in ("request.json", "report.json", "job.json"))
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    request_id = native_seal(body)
    require(request_id == request.get("request_sha256") == report.get("request_sha256"),
            "request/report seal differs")
    require(request.get("model_identity") == CHECKPOINT, "request checkpoint differs")
    require(report.get("job_id") == job.get("job_id"), "job/report identity differs")
    source_hashes = request.get("source_hashes")
    require(isinstance(source_hashes, dict) and source_hashes, "request source bindings missing")
    checked = {}
    for name, expected in source_hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        path = root / "payload" / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "retained source/checkpoint differs: " + name)
        checked[name] = expected
    for required in ("explicit_suffix_equations.py", "explicit_computation_campaign.py"):
        require(required in checked, "reference/worker source is not request-bound: " + required)
    manifest = report.get("file_manifest")
    require(isinstance(manifest, dict) and manifest, "retained artifact manifest missing")
    for name, entry in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
        path = data / relative
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "raw artifact differs from report: " + name)
    inventory = {str(path.relative_to(data)) for path in data.rglob("*")
                 if path.is_file() and path.name != "report.json"}
    require(inventory == set(manifest), "raw artifact inventory differs from report")
    inner = read(data / "report.json")
    require(all(inner.get(key) == report.get(key) for key in ("job_id", "request_sha256")),
            "inner raw report identity differs")
    return {"job_id": report["job_id"], "job_state": job.get("state"),
            "request_sha256": request_id, "request_file_sha256": sha(root / "request.json"),
            "report_file_sha256": sha(root / "report.json"), "job_file_sha256": sha(root / "job.json"),
            "verified_source_hashes": checked, "verified_artifact_files": len(manifest)}, request, report


def verify_public_transitions(transitions, trace, helpers):
    require(len(helpers["receipt_seals"](transitions)) == len(transitions), "public transition seal missing")
    pcs = []
    for item in transitions:
        receipt = item.get("receipt", item)
        pc = receipt["before_pc"]
        require(type(pc) is int and 0 <= pc < len(trace["events"])
                and receipt["after_pc"] == pc + 1
                and receipt["trace_fingerprint"] == trace["fingerprint"]
                and receipt["operator"] == trace["events"][pc]["operator"],
                "public transition trace/PC/operator binding differs")
        pcs.append(pc)
    return pcs


def verify_trace(trace):
    require(trace.get("model_identity") == CHECKPOINT, "trace model identity differs")
    require(digest({key: value for key, value in trace.items() if key != "fingerprint"}) == trace["fingerprint"],
            "trace content seal differs")


def verify_stage_arrays(stage_refs, store):
    arrays = store.mapping(stage_refs)
    require({"S", "U", "M", "P2", "h"} <= set(arrays), "suffix intermediate inventory incomplete")
    require(all(arrays[name].shape == (1, 64, 48) for name in ("S", "U", "M", "P2", "h")),
            "retained suffix intermediate geometry differs")
    return arrays


def numeric_native_audit(observed, independent, declared):
    require(len(observed) == len(independent) == 11 and set(observed) == set(independent),
            "independent native audit must include all eleven heads")
    checks = compare_mappings(observed, independent)
    if isinstance(declared, dict):
        require(set(declared) == set(checks), "declared audit head inventory differs")
        for name, row in checks.items():
            entry = declared[name]
            exact = entry["exact"] if isinstance(entry, dict) else entry
            require(exact == row["exact_numeric"], "declared native audit exactness differs from raw heads")
    return checks


def continuation_observations(value, heads, helpers, store):
    if not isinstance(value, dict):
        return {"status": "not-retained"}
    first = decode(heads)
    commit = value.get("first_commit", value.get("commit", {}))
    if commit.get("committed") is not True:
        return {"status": "not-applicable", "reason": value.get("reason"), "commit": commit}
    before = value.get("before", commit.get("before", {}))
    after = commit["after"]
    consistency = {key: after.get(key) == first[key] for key in ("register_words", "flags_CF_ZF_SF_OF")}
    consistency["byte_offset"] = after["byte_offset"] == before["byte_offset"] + first["length"]
    consistency["instruction_index"] = after["instruction_index"] == before["instruction_index"] + 1
    require(all(consistency.values()), "continuation commit differs from first model prediction")
    if "heads" not in value:
        return {"status": "not-applicable", "first_commit_consistency": consistency, "after": after,
                "reason": value.get("reason")}
    second = store.mapping(value["heads"])
    inputs = store.mapping(value["inputs"])
    require(set(inputs) == {"window_bits", "reg_bits", "flags"}, "successor root inventory differs")
    descriptors = [{"shape": list(inputs[name].shape), "dtype": "torch.float32",
                    "sha256": hashlib.sha256(np.asarray(inputs[name], dtype="<f4").tobytes()).hexdigest()}
                   for name in ("window_bits", "reg_bits", "flags")]
    root_proof = helpers["continuation_roots"](after, descriptors)
    require(root_proof["status"] == "all-three-roots-verified", "successor inputs differ from committed predictions")
    return {"status": "observed-own-state-continuation", "first_commit_consistency": consistency,
            "after": after, "successor_input_roots": root_proof, "second_decode": decode(second),
            "forecast_status": "not-forecast; no full-next-forward numerical prediction was sealed"}


def verify_forecast_release(data, request, report, plan):
    seal_path = data / "forecast-seal.json"
    seal, ledger = read(seal_path), read(data / "execution-ledger.json")
    require(report["forecast_seal_sha256"] == sha(seal_path), "forecast seal/report byte binding differs")
    require(seal.get("schema") == "mdb-explicit-forecast-seal-v1"
            and seal["plan_sha256"] == plan["plan_sha256"] == report["plan_sha256"]
            and seal["source_sha256"] == request["source_hashes"], "forecast protocol/source seal differs")
    counters = seal["pre_release_counters"]
    require(counters == report["pre_release_counters"] == ledger["pre_release_counters"],
            "pre-release counter joins differ")
    require(counters["completed_full_forwards"] == counters["mixers.2"] == counters["post.2"] == 0
            and all(value == 0 for value in counters["suffix_heads"].values()),
            "native suffix was executed before the global forecast seal")
    require(counters["physical_attempts"] == len(plan["cases"])
            and counters["prefix_auxiliary_heads"] == {"dest": len(plan["cases"]), "src": len(plan["cases"])},
            "prefix attempts or already-computed auxiliary-head inventory differs")
    for name, expected in seal["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe forecast sealed path")
        require(sha(data / relative) == expected, "globally sealed forecast artifact changed: " + name)
    for name in ("prefixes.json", "forecasts.json", "parameter-arrays.json", "weights.json", "formula.json", "plan.json"):
        require(name in seal["files"], "required evidence was not globally sealed before release: " + name)
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines() if line.strip()]
    require([row["sequence"] for row in timeline] == list(range(len(timeline))), "timeline sequence is not contiguous")
    require(all(timeline[index]["elapsed_seconds"] >= timeline[index - 1]["elapsed_seconds"]
                for index in range(1, len(timeline))), "timeline monotonic clock regressed")
    sealed = [row for row in timeline if row["kind"] == "forecast_sealed"]
    released = [row for row in timeline if row["kind"] == "native_suffix_released"]
    require(len(sealed) == len(released) == 1, "timeline has no unique global forecast/release event")
    sealed, released = sealed[0], released[0]
    require(sealed["sequence"] + 1 == released["sequence"] == seal["first_release_sequence"]
            and sealed["forecast_seal_sha256"] == released["forecast_seal_sha256"] == sha(seal_path)
            and sealed["pre_release_counters"] == counters and released == report["release_event"] == ledger["release_event"],
            "global forecast seal/native release order or identity differs")
    forbidden_names = {"mixers.2", "post.2", *counters["suffix_heads"]}
    for row in timeline[:released["sequence"]]:
        require(not (row["kind"] == "native_module_enter" and row["module"] in forbidden_names)
                and row["kind"] != "native_full_forward_completed", "timeline contains a pre-seal native future")
    text = (data / "timeline.txt").read_text().splitlines()
    expected_text = []
    for row in timeline:
        fields = {key: value for key, value in row.items() if key not in ("sequence", "kind", "phase", "elapsed_seconds")}
        expected_text.append(f"{row['sequence']:04d} {row['phase']} {row['kind']} {json.dumps(fields, sort_keys=True)}")
    require(text == expected_text, "plaintext timeline differs from structured execution events")
    attempts = sum(row["kind"] == "native_forward_attempt" for row in timeline)
    completed = sum(row["kind"] == "native_full_forward_completed" for row in timeline)
    require(attempts == ledger["counters"]["physical_attempts"] == report["physical_model_forwards"]
            and completed == ledger["counters"]["completed_full_forwards"] == report["completed_full_forwards"],
            "native attempt/completion accounting differs")
    require(all(ledger[key] == value for key, value in report["work"].items()), "execution ledger/report work differs")
    return {"seal_sha256": sha(seal_path), "pre_release_counters": counters,
            "release_event": released, "sealed_file_count": len(seal["files"]),
            "timeline_events": len(timeline), "timeline_sha256": sha(data / "timeline.jsonl"),
            "plaintext_timeline_sha256": sha(data / "timeline.txt"), "execution_ledger": ledger,
            "interpretation": "all global suffix predictions sealed before the first native final mixer/post2/readout; dest/src already executed in prefixes"}, seal


def tensor_descriptor(array):
    array = np.asarray(array)
    require(array.dtype == np.float32, "native retained tensor must be FP32")
    return {"shape": list(array.shape), "dtype": "torch.float32",
            "sha256": hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()}


def descriptor_join(array, descriptor, message):
    actual = tensor_descriptor(array)
    require(all(descriptor.get(key) == value for key, value in actual.items()), message)
    return actual


def verify_prefix(row, case, data, store, helpers):
    recipe = Path(row["root_recipe"])
    require(not recipe.is_absolute() and ".." not in recipe.parts, "unsafe prefix recipe path")
    capture = read(data / recipe / "capture.json")
    for name, expected in capture["files"].items():
        require(sha(data / recipe / name) == expected, "prefix recipe component seal differs")
    trace = read(data / recipe / "trace.json")
    exclusion = prefix_exclusion(trace)
    require(row["trace_fingerprint"] == trace["fingerprint"]
            and row["prefix_contract"] == exclusion["contract"], "prefix row/trace contract binding differs")
    roots = store.mapping(row["roots"])
    require(set(roots) == {"window_bits", "reg_bits", "flags"}, "prefix root inventory differs")
    descriptors = [tensor_descriptor(roots[name]) for name in ("window_bits", "reg_bits", "flags")]
    condition = {"program_hex": case["program_hex"], "byte_offset": 0,
                 "register_words": [f"{word:016x}" for word in case["register_words"]],
                 "flags_CF_ZF_SF_OF": case["flags"]}
    abi = helpers["continuation_roots"](condition, descriptors)
    require(abi["status"] == "all-three-roots-verified", "prefix serving inputs differ from declared raw case")
    require(trace["input_identity"] == native_seal(row["roots"]), "prefix trace input identity differs")
    actual_roots = [{key: descriptor[key] for key in ("shape", "dtype", "sha256")}
                    for descriptor in trace["values"].values() if descriptor["origin"] == "input"]
    actual_root_set = collections.Counter(map(digest, actual_roots))
    declared_root_set = collections.Counter(map(digest, descriptors))
    require(actual_root_set <= declared_root_set
            and collections.Counter(map(digest, descriptors[:2])) <= actual_root_set,
            "actual traced prefix roots differ from the declared inputs it consumes")
    sn, se, z = (store.load(row[name]) for name in ("Sn", "Se", "z"))
    require(sn.shape == se.shape == (1, 64, 48) and z.shape == (1, 256), "prefix observed frontier geometry differs")
    boundary = helpers["references"](trace["outputs"]["boundary"]["args"])
    decoder = helpers["references"](trace["outputs"]["observed"]["decoder"])
    require(len(boundary) == len(decoder) == 1, "prefix boundary/decoder output binding is ambiguous")
    descriptor_join(sn, trace["values"][next(iter(boundary))], "native prefix S bytes differ from trace")
    descriptor_join(z, trace["values"][next(iter(decoder))], "prefix decoder bytes differ from trace")
    receipt_file = read(data / row["receipt_file"])
    law = row["source_law"]
    cut = row["source_cut_manifest"]
    require(law["module"] == "native.post.0" and law["exit"] == cut["pc"], "prefix source law/cut PC differs")
    source_pcs = verify_public_transitions(receipt_file["source_prefix"], trace, helpers)
    native_pcs = verify_public_transitions(receipt_file["native_suffix_of_prefix"], trace, helpers)
    edited_pcs = verify_public_transitions(receipt_file["edited_suffix_of_prefix"], trace, helpers)
    require(source_pcs == list(range(law["exit"]))
            and native_pcs == edited_pcs == list(range(law["exit"], len(trace["events"]))),
            "prefix replay proof does not cover the declared common parent and both joins")
    act = receipt_file["act"]
    require(act == row["source_act"] and len(helpers["receipt_seals"](act)) == 1
            and act["trace_fingerprint"] == trace["fingerprint"] and act["pc"] == law["exit"]
            and act["value_id"] == row["source_port"] == law["outputs"][0], "prefix source Act binding differs")
    zero = store.load(row["source_zero"])
    require(not np.count_nonzero(zero), "prefix source-removal payload is not zero")
    descriptor_join(zero, act["after"], "prefix source Act zero payload differs")
    require(all(act["before"].get(key) == trace["values"][row["source_port"]].get(key)
                for key in ("shape", "dtype", "sha256")), "prefix source Act is not bound to the native source write")
    full = read(data / "native-traces" / (case["case_id"] + ".json"))
    full_join = prefix_full_join(trace, full)
    full_roots = [{key: descriptor[key] for key in ("shape", "dtype", "sha256")}
                  for descriptor in full["values"].values() if descriptor["origin"] == "input"]
    require(collections.Counter(map(digest, full_roots)) == declared_root_set,
            "released native full trace input roots differ from all three declared serving inputs")
    return {"trace": trace, "full_trace": full, "roots": roots, "Sn": sn, "Se": se, "z": z,
            "proof": {"native_prefix_exclusion": exclusion, "initial_raw_input_abi": abi,
                "prefix_full_join": full_join, "traced_prefix_input_roots": len(actual_roots),
                "deferred_roots": "flags are first consumed by the released suffix; all three roots verified in full trace",
                "source_act": act["fingerprint"],
                "source_cut_fingerprint": cut["fingerprint"], "source_prefix_operations": len(source_pcs),
                "native_join_operations": len(native_pcs), "edited_join_operations": len(edited_pcs)}}


def validate_payloads(forecasts, prefix, store):
    native, edited = forecasts["native"], forecasts["zero"]
    payload_refs = native["raw"]["payloads"]
    require(all(row["raw"]["payloads"] == payload_refs for row in forecasts.values()),
            "same-case branches do not share one frozen payload dictionary")
    payloads = store.mapping(payload_refs)
    expected = {"U_native": np.asarray(native["stages"]["U"], dtype=np.float32),
                "U_edited": np.asarray(edited["stages"]["U"], dtype=np.float32),
                "P2_native": np.asarray(native["stages"]["P2"], dtype=np.float32)}
    expected["M_skip_native"] = np.asarray(prefix["Sn"] + expected["U_edited"], dtype=np.float32)
    expected["M_both_native"] = np.asarray(prefix["Sn"] + expected["U_native"], dtype=np.float32)
    require(set(payloads) == set(expected), "frozen intervention payload inventory differs")
    checks = {}
    for name, actual in payloads.items():
        require(actual.dtype == np.float32 and actual.shape == (1, 64, 48)
                and actual.tobytes() == expected[name].tobytes(), "frozen FP32 payload construction differs: " + name)
        checks[name] = tensor_descriptor(actual)
    for arm, row in forecasts.items():
        stages = row["stages"]
        expected_s = prefix["Sn"] if arm == "native" else prefix["Se"]
        require(np.array_equal(stages["S"], expected_s), "forecast used the wrong observed prefix state")
        if arm in {"U_swap", "U_swap_P2native"}:
            require(np.array_equal(stages["U"], payloads["U_native"]), "forecast U override differs from frozen payload")
        if arm in {"S_swap", "S_swap_P2native", "both"}:
            name = "M_both_native" if arm == "both" else "M_skip_native"
            require(np.array_equal(stages["M"], payloads[name]), "forecast M override differs from frozen payload")
        if arm.endswith("_P2native"):
            require(np.array_equal(stages["P2"], payloads["P2_native"]), "forecast common P2 override differs from frozen payload")
    return checks


def trace_ports(trace):
    def module_events(path):
        result = [event for event in trace["events"]
                  if event["module_path"] == path or event["module_path"].startswith(path + ".")]
        require(result, "native module has no retained events: " + path)
        return result
    def activation(event):
        values = [value for value in event["inputs"] if trace["values"][value]["shape"] == [1, 64, 48]]
        require(len(values) == 1, "native module activation is ambiguous")
        return values[0]
    source, projection, post, mixer, reg = (module_events(path) for path in (
        "post.0", "mixers.2.out_proj", "post.2", "mixers.2", "reg_out"))
    return {"source": source[-1]["outputs"][0], "source_pc": source[-1]["sequence"] + 1,
            "U": projection[-1]["outputs"][0], "U_pc": projection[-1]["sequence"] + 1,
            "M": activation(post[0]), "M_pc": post[0]["sequence"],
            "P2": post[-1]["outputs"][0], "P2_pc": post[-1]["sequence"] + 1,
            "S": activation(mixer[0]), "S_pc": mixer[0]["sequence"],
            "h": activation(reg[0])}


def verify_branch_replay(row, prefix, forecast, observed, all_observed, data, store, helpers):
    trace = prefix["full_trace"]
    ports = trace_ports(trace)
    replay = row["replay"]
    proof = read(data / replay["receipt_file"])
    require(proof["trace_fingerprint"] == replay["trace_fingerprint"] == trace["fingerprint"],
            "branch replay/trace binding differs")
    pcs = verify_public_transitions(proof["receipts"], trace, helpers)
    require(pcs == list(range(ports["source_pc"], len(trace["events"])))
            and len(pcs) == replay["native_operations"], "branch replay primitive suffix coverage differs")
    require(proof["source_cut"]["pc"] == ports["source_pc"]
            and proof["final_cut"]["pc"] == len(trace["events"]), "branch source/final cut PC differs")
    arm = row["arm"]
    acts = proof["acts"]
    expected = []
    if arm != "native":
        expected.append(("source", ports["source_pc"], store.load(prefix["raw"]["source_zero"])))
    if arm in {"U_swap", "U_swap_P2native", "sham"}:
        payload = observed["stages"]["U"] if arm == "sham" else store.load(forecast["raw"]["payloads"]["U_native"])
        expected.append(("U", ports["U_pc"], payload))
    if arm in {"S_swap", "S_swap_P2native", "both"}:
        name = "M_both_native" if arm == "both" else "M_skip_native"
        expected.append(("M", ports["M_pc"], store.load(forecast["raw"]["payloads"][name])))
    if arm.endswith("_P2native"):
        expected.append(("P2", ports["P2_pc"], store.load(forecast["raw"]["payloads"]["P2_native"])))
    require(len(acts) == replay["act_count"] == len(expected)
            and len(helpers["receipt_seals"](acts)) == len(acts), "branch public Act inventory/seals differ")
    joins = []
    for act, (name, pc, payload) in zip(acts, expected, strict=True):
        require(act["operation"] == "mdb.discovery.ssa-act" and act["trace_fingerprint"] == trace["fingerprint"]
                and act["value_id"] == ports[name] and act["pc"] == pc,
                "branch Act trace/producer/consumer frontier binding differs")
        descriptor_join(payload, act["after"], "branch Act payload differs from the sealed bytes")
        if name == "source":
            require(all(act["before"].get(key) == trace["values"][ports[name]].get(key)
                        for key in ("shape", "dtype", "sha256")), "source Act original write differs from native trace")
        else:
            descriptor_join(observed["stages"][name], act["after"], "observed stage differs from Act payload")
            if name == "U":
                before = all_observed["zero"]["stages"]["U"]
            elif name == "M":
                before = np.asarray(observed["stages"]["S"] + observed["stages"]["U"], dtype=np.float32)
            else:
                before = all_observed["U_swap" if arm.startswith("U_") else "S_swap"]["stages"]["P2"]
            descriptor_join(before, act["before"], "Act before-value differs from its independently retained common-parent branch")
        if arm == "sham" and name == "U":
            require(all(act["before"].get(key) == act["after"].get(key) for key in ("shape", "dtype", "sha256")),
                    "same-value sham changed its selected SSA value")
        joins.append({"site": name, "value_id": ports[name], "pc": pc, "fingerprint": act["fingerprint"]})
    exact_flags = collections.Counter(item["receipt"]["exact_outputs"] for item in proof["receipts"])
    return {"trace_fingerprint": trace["fingerprint"], "source_cut": proof["source_cut"]["fingerprint"],
            "suffix_operations": len(pcs), "act_joins": joins,
            "recorded_output_comparisons": {"exact": exact_flags.get(True, 0), "changed": exact_flags.get(False, 0)},
            "ports": ports, "authority": "public typed receipt seals, full primitive suffix, live SSA PCs, payload bytes and independent before-value joins"}


def prefix_exclusion(trace):
    verify_trace(trace)
    contract = trace.get("metadata", {}).get("prefix_contract", {})
    require(contract.get("schema") == "mdb-native-prefix-boundary-v1"
            and contract.get("execution_extent") == "prefix-wrapper"
            and contract.get("stop_before") == "mixers.2"
            and contract.get("wrapper_stop_before") == "native.mixers.2"
            and contract.get("stop_edge") == "before-local-prehooks-and-module-body"
            and contract.get("full_model_outputs_computed") is False,
            "prefix trace has no explicit before-mixer2 execution contract")
    forbidden = ("mixers.2", "post.2", "reg_out", "flag_query", "flags", "imm_out", "width_head",
                 "length", "syscall", "store_flag", "store_out")
    violations = []
    auxiliary = collections.Counter()
    for event in trace["events"]:
        path = event["module_path"]
        path = path[7:] if path.startswith("native.") else path
        if any(path == name or path.startswith(name + ".") for name in forbidden):
            violations.append({"pc": event["sequence"], "module_path": event["module_path"]})
        if path in {"dest", "src"}:
            auxiliary[path] += 1
    require(not violations, "prefix trace contains a forbidden suffix body/readout operation")
    require(contract.get("observations", {}).get("decoder", {}).get("completed_invocations") == 1,
            "decoder prefix observation inventory differs")
    return {"contract": contract, "forbidden_paths": list(forbidden), "violations": violations,
            "prefix_operator_count": len(trace["events"]), "already_executed_auxiliary_head_operators": dict(auxiliary),
            "authority": "source-bound native prefix contract plus retained operation inventory; dest/src are prefix computations"}


def prefix_full_join(prefix, full):
    verify_trace(full)
    require(len(prefix["events"]) < len(full["events"]), "prefix is not a proper full-trace prefix")
    def normalized(event):
        value = {key: event[key] for key in ("operator", "module_path", "inputs", "outputs", "arguments", "result")}
        path = value["module_path"]
        value["module_path"] = "<root>" if path == "native" else path[7:] if path.startswith("native.") else path
        return value
    for before, after in zip(prefix["events"], full["events"][:len(prefix["events"])], strict=True):
        require(normalized(before) == normalized(after), "prefix/full ordered primitive join differs")
    for key, descriptor in prefix["values"].items():
        require(key in full["values"], "prefix SSA value missing from full trace")
        require(all(descriptor.get(field) == full["values"][key].get(field)
                    for field in ("shape", "dtype", "origin", "producer", "sha256")),
                "prefix/full tensor identity differs")
    return {"ordered_operator_joins": len(prefix["events"]), "tensor_joins": len(prefix["values"]),
            "full_suffix_first_pc": len(prefix["events"]),
            "scope": "native prefix versus native full capture; edited prefix has a separate public replay proof"}


def reader_union(case, observed, predicted):
    readers = {(int(item["register"]), int(item["bit"])) for item in case["explanatory_readers"]}
    native = observed["native"]["heads"]["regs"] > 0
    for arm in observed:
        actual = observed[arm]["heads"]["regs"] > 0
        forecast = predicted[arm]["heads"]["regs"] > 0
        for _, register, bit in np.argwhere((actual != native) | (actual != forecast)):
            readers.add((int(register), int(bit)))
    return sorted(readers)


def hidden_unit_account(register, bit, predicted, observed, weights):
    """All 96 algebraic terms, explicitly distinct from unit interventions."""
    wreg = np.asarray(weights["reg_out.weight"], dtype=np.float64)[register]
    wpost = np.asarray(weights["post.2.3.weight"], dtype=np.float64)
    bpost = np.asarray(weights["post.2.3.bias"], dtype=np.float64)
    coefficients = wreg @ wpost
    require(coefficients.shape == (96,), "post2 hidden unit accounting requires the frozen 96-unit layer")
    bias = float(wreg @ bpost)
    arms = {}
    for arm, values in predicted.items():
        common = arm in {"U_swap_P2native", "S_swap_P2native"}
        origin = predicted["native"] if common else values
        activation = origin["stages"]["post2_gelu"][0, bit].astype(np.float64)
        require(activation.shape == (96,), "predicted hidden activation geometry differs")
        contributions = activation * coefficients
        predicted_projected = float(np.asarray(values["stages"]["P2"], dtype=np.float64)[0, bit] @ wreg)
        actual_projected = float(np.asarray(observed[arm]["stages"]["P2"], dtype=np.float64)[0, bit] @ wreg)
        arms[arm] = {"activation_origin": "frozen native P2 payload" if common else "same-arm explicit equation",
            "contributions": contributions.tolist(), "sum_plus_post_bias": float(contributions.sum()) + bias,
            "effective_P2_projection": predicted_projected,
            "payload_rounding_or_substitution_residual": predicted_projected - (float(contributions.sum()) + bias),
            "native_measured_P2_projection": actual_projected,
            "native_minus_forecast_P2_projection": actual_projected - predicted_projected}
    units = []
    for index in range(96):
        row = {"unit": index, "reader_coefficient": float(coefficients[index]),
               "contributions": {arm: value["contributions"][index] for arm, value in arms.items()}}
        row["U_swap_minus_zero"] = row["contributions"]["U_swap"] - row["contributions"]["zero"]
        row["U_swap_minus_native"] = row["contributions"]["U_swap"] - row["contributions"]["native"]
        row["S_swap_minus_zero"] = row["contributions"]["S_swap"] - row["contributions"]["zero"]
        units.append(row)
    ranked = sorted(units, key=lambda item: abs(item["U_swap_minus_zero"]), reverse=True)
    return {"post_bias_projection": bias, "all_96_units": units,
            "largest_U_swap_minus_zero_terms": [row["unit"] for row in ranked[:12]],
            "arm_closures": {arm: {key: value for key, value in row.items() if key != "contributions"}
                             for arm, row in arms.items()},
            "authority": "algebraic full-weight FP64 forecast accounting; actual native P2 projection checked separately",
            "limits": "No neuron necessity, semantic naming or observed native hidden-activation claim. Common-P2 payload casting residual retained."}


def explanatory_readers(case, observed, predicted, actual_terms, predicted_terms, weights):
    rows = []
    for register, bit in reader_union(case, observed, predicted):
        coordinate = (0, register, bit)
        table = {}
        for arm in observed:
            table[arm] = {"actual_logit": float(observed[arm]["heads"]["regs"][coordinate]),
                "forecast_logit": float(predicted[arm]["heads"]["regs"][coordinate]),
                "actual_components": {name: float(value[coordinate]) for name, value in actual_terms[arm].items()},
                "forecast_components": {name: float(value[coordinate]) for name, value in predicted_terms[arm].items()}}
        def contrast(left, right):
            return {"actual": table[left]["actual_logit"] - table[right]["actual_logit"],
                    "forecast": table[left]["forecast_logit"] - table[right]["forecast_logit"],
                    "actual_components": {name: table[left]["actual_components"][name] - table[right]["actual_components"][name]
                                          for name in table[left]["actual_components"]}}
        distances = {}
        for mode, u, s in (("natural_post", "U_swap", "S_swap"),
                            ("common_frozen_post", "U_swap_P2native", "S_swap_P2native")):
            distances[mode] = {}
            for source, key in (("actual", "actual_logit"), ("forecast", "forecast_logit")):
                ud = abs(table[u][key] - table["native"][key])
                sd = abs(table[s][key] - table["native"][key])
                distances[mode][source] = {"U_distance": ud, "S_distance": sd,
                    "U_closer": ud < sd, "S_closer": sd < ud, "tie": ud == sd}
        rows.append({"reader": {"head": "regs", "register": register, "bit": bit}, "table": table,
            "contrasts": {"U_swap_minus_zero": contrast("U_swap", "zero"),
                "S_swap_minus_zero": contrast("S_swap", "zero"),
                "U_commonP2_minus_naturalP2": contrast("U_swap_P2native", "U_swap"),
                "S_commonP2_minus_naturalP2": contrast("S_swap_P2native", "S_swap")},
            "route_distances": distances,
            "hidden_unit_account": hidden_unit_account(register, bit, predicted, observed, weights)})
    return rows


def counts_by(rows, field):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    result = {}
    for key, group in sorted(grouped.items()):
        result[key] = {"rows": len(group),
            "full_register_bank_equal": sum(row["decoded"]["full_register_bank_equal"] for row in group),
            "all_flag_decisions_equal": sum(not row["decoded"]["wrong_flag_indices"] for row in group),
            "length_equal": sum(row["decoded"]["fields"]["length"] for row in group),
            "store_decision_equal": sum(row["decoded"]["fields"]["store"] for row in group),
            "syscall_decision_equal": sum(row["decoded"]["fields"]["syscall"] for row in group),
            "wrong_register_bits": sum(row["decoded"]["wrong_register_bit_count"] for row in group),
            "native_all_eleven_head_parity": sum(row["native_audit_all_heads_exact"] for row in group)}
    return result


def analyze(root, data, helper_source):
    helpers = load_helpers(helper_source)
    store = ArrayStore(data)
    source, request, report = verify_sources(root, data)
    source.update(helper_source=str(Path(helper_source).resolve()), helper_sha256=sha(Path(helper_source)),
                  analyzer_sha256=sha(Path(__file__)))
    plan = read(data / "plan.json")
    require(plan["schema"] == "mdb-explicit-computation-plan-v1"
            and native_seal({key: value for key, value in plan.items() if key != "plan_sha256"}) == plan["plan_sha256"]
            and plan["model_identity"] == CHECKPOINT, "explicit computation protocol self seal differs")
    require(sha(root / "payload/plan.json") == request["config"]["plan_file_sha256"]
            and read(root / "payload/plan.json") == plan
            and request["config"]["plan_sha256"] == plan["plan_sha256"], "request/staged protocol binding differs")
    release_proof, seal = verify_forecast_release(data, request, report, plan)
    cases = {case["case_id"]: case for case in plan["cases"]}
    require(len(cases) == 6 and len(plan["arms"]) == 8, "bounded six-case/eight-arm protocol differs")
    metadata = read(data / "weights.json")
    parameter_refs = read(data / "parameter-arrays.json")
    store.verify_tree(parameter_refs)
    weights = store.mapping(parameter_refs)
    require(metadata["checkpoint_sha256"] == CHECKPOINT and metadata["complete_parameter_dictionary"] is True
            and set(metadata["parameter_arrays"]) == set(weights), "complete full-weight dictionary binding differs")
    for name, value in weights.items():
        descriptor = metadata["parameter_arrays"][name]
        require(list(value.shape) == descriptor["shape"] and value.dtype == np.dtype(descriptor["dtype"])
                and hashlib.sha256(value.tobytes()).hexdigest() == descriptor["sha256"],
                "full-weight original tensor descriptor differs: " + name)
    formula = read(data / "formula.json")
    require(set(formula["parameters"]) <= set(weights), "reference suffix parameter support incomplete")
    require(all(request["source_hashes"].get(name) == value for name, value in formula["source_formulas"].items()),
            "frozen formula source bindings differ from installed-worker request")
    prefix_rows = read(data / "prefixes.json")["rows"]
    prefix_map = {row["case_id"]: row for row in prefix_rows}
    require(len(prefix_map) == len(prefix_rows) == len(cases) == seal["prefix_rows"] == report["case_count"],
            "prefix case inventory differs")
    require(set(prefix_map) == set(cases), "prefix cases differ from sealed protocol")
    prefixes = {}
    for case_id, row in prefix_map.items():
        store.verify_tree(row)
        require(row["scope"] == cases[case_id]["scope"], "prefix anchor/fresh scope changed")
        prefixes[case_id] = {**verify_prefix(row, cases[case_id], data, store, helpers), "raw": row}
    forecasts_raw = read(data / "forecasts.json")["rows"]
    require(len(forecasts_raw) == seal["forecast_rows"] == report["forecast_rows"] == 48,
            "globally sealed forecast inventory differs")
    predicted = collections.defaultdict(dict)
    for index, row in enumerate(forecasts_raw):
        store.verify_tree(row)
        case_id, arm = row["case_id"], row["arm"]
        require(case_id in cases and arm in plan["arms"] and arm not in predicted[case_id]
                and row["scope"] == cases[case_id]["scope"] and row["future_native_values_used"] is False,
                "forecast row case/arm/scope/future-input binding differs")
        require(row["intervention_recipe"] == plan["recipes"][arm], "forecast recipe differs from protocol")
        heads = store.mapping(row["heads"])
        require(len(heads) == 11 and set(heads) == set(formula["heads"]), "forecast all-head inventory differs")
        stages = verify_stage_arrays(row["stages"], store)
        terms = store.mapping(row["register_terms"])
        require(all(row["decoded"].get(key) == value for key, value in decode(heads).items()),
                "recorded forecast decoder differs from retained continuous heads")
        predicted[case_id][arm] = {"heads": heads, "stages": stages, "terms": terms,
                                  "raw": row, "index": index}
    for case_id in cases:
        require(set(predicted[case_id]) == set(plan["arms"]), "one case lacks a pre-sealed intervention forecast")
    payload_checks = {case_id: validate_payloads(predicted[case_id], prefixes[case_id], store) for case_id in cases}
    # These identities must all have existed in the global seal, including each
    # numeric stage/weight/payload; later observations may add new CAS arrays.
    pre_release_numeric = dict(store.checked)
    require(all(seal["files"].get(name) == ref["sha256"] for name, ref in pre_release_numeric.items()),
            "forecast/prefix/weight numeric evidence was not sealed before native release")
    rows = read(data / "observations.json")["rows"]
    primary = read(data / "primary-observations.json")["rows"]
    require(len(rows) == len(primary) == report["primary_rows"] == 48, "primary observation inventory differs")
    require([{key: value for key, value in row.items() if key != "continuation"} for row in rows] == primary,
            "optional future changed an already-published primary observation")
    observed = collections.defaultdict(dict)
    incomplete = []
    for row in rows:
        case_id, arm = row["case_id"], row["arm"]
        require(case_id in cases and arm in plan["arms"] and arm not in observed[case_id]
                and row["scope"] == cases[case_id]["scope"], "observation case/arm/scope differs")
        prediction = predicted[case_id][arm]
        require(row["forecast_row_index"] == prediction["index"]
                and row["forecast_seal_sha256"] == release_proof["seal_sha256"]
                and row["intervention_recipe"] == prediction["raw"]["intervention_recipe"],
                "observed intervention is not joined to its pre-sealed forecast")
        if row.get("status") != "observed" or "audit" not in row:
            incomplete.append({"case_id": case_id, "arm": arm, "status": row.get("status"), "error": row.get("error")})
            continue
        store.verify_tree(row)
        observed[case_id][arm] = {"heads": store.mapping(row["heads"]),
            "stages": verify_stage_arrays(row["stages"], store), "raw": row}
    joins_raw = read(data / "prefix-join-checks.json")["rows"]
    require(joins_raw == report["prefix_join_checks"], "declared prefix join observations differ from report")
    joins = {(row["case_id"], row["arm"]): row for row in joins_raw}
    results, case_results, actual_term_sets, predicted_term_sets = [], [], {}, {}
    receipt_counts = {"prefix_transitions": sum(prefix["proof"][key] for prefix in prefixes.values()
        for key in ("source_prefix_operations", "native_join_operations", "edited_join_operations")),
        "prefix_acts": len(prefixes), "branch_transitions": 0, "branch_acts": 0,
        "native_source_transitions": 0}
    for case_id, case in cases.items():
        if set(observed[case_id]) != set(plan["arms"]):
            continue
        prefix = prefixes[case_id]
        trace = prefix["full_trace"]
        original_receipts = read(data / "native-prefix-receipts" / (case_id + ".json"))
        pcs = verify_public_transitions(original_receipts, trace, helpers)
        require(pcs == list(range(trace_ports(trace)["source_pc"])), "full-capture common-parent prefix coverage differs")
        receipt_counts["native_source_transitions"] += len(pcs)
        actual_terms, predicted_terms = {}, {}
        for arm in plan["arms"]:
            value, prediction = observed[case_id][arm], predicted[case_id][arm]
            row = value["raw"]
            heads, stages = value["heads"], value["stages"]
            independent_heads, independent_stages = store.mapping(row["audit"]["heads"]), store.mapping(row["audit"]["stages"])
            native_audit = numeric_native_audit(heads, independent_heads, row["audit"]["exact_heads"])
            stage_audit = compare_mappings(stages, independent_stages)
            require(set(stage_audit) == set(row["audit"]["exact_stages"])
                    and all(row["audit"]["exact_stages"][name] == proof["exact_numeric"]
                            for name, proof in stage_audit.items()), "declared stage native audit differs from raw arrays")
            require(all(row["decoded"].get(key) == val for key, val in decode(heads).items()),
                    "recorded observed decoder differs from raw heads")
            expected_s = prefix["Sn"] if arm == "native" else prefix["Se"]
            join = joins[(case_id, arm)]
            actual_s, actual_z = bool(np.array_equal(stages["S"], expected_s)), bool(np.array_equal(stages["z"], prefix["z"]))
            require(join["exact_S"] == actual_s and join["exact_z"] == actual_z,
                    "declared prefix/full branch join differs from retained tensors")
            replay = verify_branch_replay(row, prefix, prediction, value, observed[case_id], data, store, helpers)
            receipt_counts["branch_transitions"] += replay["suffix_operations"]
            receipt_counts["branch_acts"] += len(replay["act_joins"])
            actual_terms[arm] = register_terms(stages, prefix["roots"]["reg_bits"], weights["reg_out.weight"], weights["reg_out.bias"])
            predicted_terms[arm] = register_terms(prediction["stages"], prefix["roots"]["reg_bits"], weights["reg_out.weight"], weights["reg_out.bias"])
            term_comparison = compare_mappings(predicted_terms[arm], prediction["terms"])
            common_stages = sorted(set(stages) & set(prediction["stages"]))
            result = {"case_id": case_id, "arm": arm, "scope": case["scope"], "family": case["supplied_family"],
                "head_residuals": compare_mappings(heads, prediction["heads"]),
                "stage_residuals": {name: difference(stages[name], prediction["stages"][name]) for name in common_stages},
                "unobserved_forecast_stages": sorted(set(prediction["stages"]) - set(stages)),
                "prefix_z_join": actual_z, "prefix_state_join": actual_s,
                "decoded": decoded_comparison(decode(heads), decode(prediction["heads"])),
                "all_head_decisions": head_decisions(heads, prediction["heads"]),
                "native_audit": {"heads": native_audit, "stages": stage_audit},
                "native_audit_all_heads_exact": all(entry["exact_numeric"] for entry in native_audit.values()),
                "native_audit_all_stages_exact": all(entry["exact_numeric"] for entry in stage_audit.values()),
                "replay_proof": replay, "actual_readout_closure": term_closure(actual_terms[arm], heads),
                "actual_register_component_arrays": {name: array.tolist() for name, array in actual_terms[arm].items()},
                "forecast_register_component_refs": prediction["raw"]["register_terms"],
                "forecast_readout_closure": term_closure(predicted_terms[arm], prediction["heads"]),
                "frozen_forecast_term_reconstruction": term_comparison,
                "continuation": continuation_observations(row.get("continuation"), heads, helpers, store)}
            if arm == "native":
                require(row["capture_control_status"] == "verified", "native full-capture control is not qualified")
                native_matches = {}
                for name, encoded in trace["outputs"].items():
                    ids = helpers["references"](encoded)
                    require(len(ids) == 1, "native output trace binding is ambiguous")
                    descriptor_join(heads[name], trace["values"][next(iter(ids))], "unchanged native replay differs from full capture")
                    native_matches[name] = True
                result["capture_control"] = native_matches
            results.append(result)
        sham_heads = compare_mappings(observed[case_id]["sham"]["heads"], observed[case_id]["zero"]["heads"])
        sham_stages = compare_mappings(observed[case_id]["sham"]["stages"], observed[case_id]["zero"]["stages"])
        require(all(row["exact_numeric"] for row in sham_heads.values())
                and all(row["exact_numeric"] for row in sham_stages.values()), "same-value sham does not preserve its common-parent future")
        readers = explanatory_readers(case, observed[case_id], predicted[case_id], actual_terms, predicted_terms, weights)
        case_results.append({"case_id": case_id, "scope": case["scope"], "family": case["supplied_family"],
            "readers": readers, "same_value_sham": {"heads": sham_heads, "stages": sham_stages},
            "payloads": payload_checks[case_id], "prefix_proof": prefix["proof"]})
        actual_term_sets[case_id], predicted_term_sets[case_id] = actual_terms, predicted_terms
    require(receipt_counts["prefix_transitions"] + receipt_counts["branch_transitions"]
            + receipt_counts["native_source_transitions"] == report["work"]["replay_native_operations"],
            "total public native replay transition count differs from ledger")
    require(receipt_counts["prefix_acts"] + receipt_counts["branch_acts"] == report["work"]["acts"],
            "total public Act count differs from ledger")
    continuations = [row["continuation"] for row in results]
    return {"schema": "mdb-explicit-computation-analysis-v1", "source": source,
        "mechanics_status": report["mechanics_status"],
        "instrument_status": "all-native-audits-exact" if all(row["native_audit_all_heads_exact"] and row["native_audit_all_stages_exact"]
            for row in results) and not incomplete else "heterogeneous-or-incomplete-native-audits",
        "trend_status": "signed-full-equation-forecast-residuals-and-conditional-restoration-account",
        "terminal_status": "unestablished", "protocol": plan, "formula": formula,
        "global_forecast_exclusion": release_proof,
        "weights": {"complete_dictionary": True, "parameter_arrays": len(weights),
            "scalar_count": sum(int(value.size) for value in weights.values()),
            "manifest_sha256": sha(data / "weights.json"), "array_reference_manifest_sha256": sha(data / "parameter-arrays.json"),
            "compression": "none; full supplied parameter arrays retained"},
        "receipt_counts": receipt_counts, "numeric_artifacts_verified": len(store.checked),
        "case_results": case_results, "rows": results, "incomplete_primary_rows": incomplete,
        "counts": {"by_scope": counts_by(results, "scope"), "by_case": counts_by(results, "case_id"),
                   "by_family": counts_by(results, "family"), "by_arm": counts_by(results, "arm")},
        "own_state_continuations": {"observed": sum(row["status"] == "observed-own-state-continuation" for row in continuations),
            "other_statuses": dict(collections.Counter(row["status"] for row in continuations if row["status"] != "observed-own-state-continuation")),
            "scope": "literal predicted register/flag/offset root reconstruction; no sealed next-forward numerical forecast"},
        "counterexamples": [{"case_id": row["case_id"], "arm": row["arm"], "scope": row["scope"], "decoded": row["decoded"]}
            for row in results if not all(row["decoded"]["fields"].values())],
        "limits": ["The account is conditional on measured native/edited prefix states and decoder z.",
            "dest/src were already computed in each native prefix; their suffix equations are re-expressions, not fresh head forecasts.",
            "NumPy FP64 sequential recurrence differs from native FP32 parallel reduction; every signed residual is retained without a scientific pass threshold.",
            "Per-unit GELU contributions are full-weight forecast algebra; measured native P2 projections are checked separately.",
            "Common analytically predicted P2 payloads are interventions, not the old native future supplied as an oracle.",
            "Restoration effects and mixed responses are conditional; they establish no universal or minimum causal route.",
            "Two retrospective anchors and four fixed fresh states remain separate; no compression or ISA semantic discovery is claimed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--helper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source_path, helper_path = output.with_suffix(".source.py"), output.with_suffix(".helper.source.py")
    require(not output.exists() and not source_path.exists() and not helper_path.exists(),
            "analysis/source output exists; preserve immutable previous versions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    with helper_path.open("xb") as stream:
        stream.write(args.helper_source.read_bytes())
    root = args.run_dir.resolve()
    data = (args.data_dir or root / "extracted/explicit-result").resolve()
    result = analyze(root, data, args.helper_source)
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "mechanics_status": result["mechanics_status"],
        "instrument_status": result["instrument_status"], "counts_by_scope": result["counts"]["by_scope"],
        "counterexamples": len(result["counterexamples"])}, sort_keys=True))


if __name__ == "__main__":
    main()
