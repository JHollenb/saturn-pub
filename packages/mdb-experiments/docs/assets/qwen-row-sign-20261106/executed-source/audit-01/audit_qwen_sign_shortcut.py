"""Saved-only, source-bound observations of the finite Qwen sign shortcut.

No producer/helper/model/tokenizer imports, learned recurrence or projection.
Numerical disagreements remain observations; the real algebra is annotation,
not a claim that BF16 arithmetic can be regrouped without a residual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "83f9d459938ccb96b913def78862cffc869cb6f0b4404a7f116d0a46fed184af"
PROTOCOL_SHA256 = "5ce41dda16fdfd687aba584f9aaedc389d0ec8bf088b2ea63de8b4456ddf9cf2"
PINS = {
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
HEADS = ("product", "M", "Hnext")
CASE_IDS = ("short-objects", "short-procedure", "long-description")
RULE_ID = "law.qwen.sign-shortcut.bf16-primitive-order.v1"
ROUTE_ID = "route.qwen3.block18.own-functional-positive-RGU-to-negative-PMY.v1"
PRODUCER_HELPERS = {
    "qwen_sign_shortcut.py": "8e59051a34bc8e89c9376fd29aec966df9cbe84560c81f86910504172caaa457",
    "qwen_h18_state_history.py": "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
}
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
PATH_KEYS = ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "retrieval_script")
CONTRASTS = {
    "shortcut-minus-full-functional-negative": ("shortcut", "functional"),
    "full-functional-minus-native-negative": ("functional", "native"),
    "shortcut-minus-native-negative": ("shortcut", "native"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def helpers():
    for name, value in PINS.items():
        require(sha(HERE / name) == value, "frozen saved account helper differs: " + name)
    return (runpy.run_path(str(HERE / "audit_flux_raw_to_image.py")),
            runpy.run_path(str(HERE / "audit_flux_functional_qwen_prefix.py")),
            runpy.run_path(str(HERE / "audit_flux_functional_pass.py")))


def signed_descriptor(negative, positive):
    import numpy as np

    delta = np.asarray(negative, np.float64) - np.asarray(positive, np.float64)
    return dict(shape=list(delta.shape), elements=int(delta.size), dtype="float64", orientation="negative-minus-own-positive",
                sha256=hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes(order="C")).hexdigest())


def measurement(left, right):
    import numpy as np

    report = helpers()[1]["compare"](left, right)
    changed = np.flatnonzero((left.astype(np.float64) - right.astype(np.float64)).reshape(-1))
    report["first_numerical_difference"] = None
    if changed.size:
        index = int(changed[0])
        report["first_numerical_difference"] = dict(flat_index=index,
            coordinate=[int(x) for x in np.unravel_index(index, left.shape)], left=float(left.reshape(-1)[index]),
            right=float(right.reshape(-1)[index]), signed_error=float(left.reshape(-1)[index]) - float(right.reshape(-1)[index]))
    byte_indices = np.flatnonzero(np.frombuffer(left.tobytes(), np.uint8) != np.frombuffer(right.tobytes(), np.uint8))
    report["first_differing_byte"] = None if not byte_indices.size else int(byte_indices[0])
    return report


def oddness(positive, negative):
    import numpy as np

    require(positive.shape == negative.shape, "premise shapes differ")
    delta = negative.astype(np.float64) + positive.astype(np.float64)
    expected = -positive
    p = (np.asarray(expected, dtype="<f4").view("<u4") >> 16).astype("<u2")
    n = (np.asarray(negative, dtype="<f4").view("<u4") >> 16).astype("<u2")
    return dict(elements=int(positive.size), numerical_changed=int(np.count_nonzero(delta)),
        max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))),
        bit_changed_elements=int(np.count_nonzero(p != n)), differing_bytes=int(np.count_nonzero(p.view(np.uint8) != n.view(np.uint8))),
        shared_zeros=int(np.count_nonzero((positive == 0) & (negative == 0))),
        zero_signbit_defects=int(np.count_nonzero((positive == 0) & (negative == 0) & (np.signbit(negative) != np.signbit(expected)))),
        status="observation-only", expectation="Finite upstream oddness is measured, never used to reject or repair a shortcut row.")


def compare_values(negative, positive, sealed, *, compare_core=None):
    """Three complete endpoint/response errors; no numerical gate or projection."""
    import numpy as np

    if compare_core is None:
        from mdb.equation_comparison import compare_equations
        compare_core = compare_equations
    require(set(negative) == {"shortcut", "functional", "native"} and set(positive) == {"functional", "native"}, "three negative/two own positive banks required")
    require(all(set(bank) == set(HEADS) for bank in list(negative.values()) + list(positive.values())), "complete P/M/H head set differs")
    response = {lane: {name: np.asarray(value, np.float64) - np.asarray(positive["native" if lane == "native" else "functional"][name], np.float64)
                       for name, value in bank.items()} for lane, bank in negative.items()}
    require(set(sealed) == {"functional", "shortcut"}, "pretarget signed response bank differs")
    for lane in sealed:
        require(set(sealed[lane]) == set(HEADS) and all(sealed[lane][name] == signed_descriptor(negative[lane][name], positive["functional"][name]) for name in HEADS),
                "pretarget full signed response hash differs")
    reports, equations = {}, {}
    elements = sum(int(negative["shortcut"][name].size) for name in HEADS)
    for contrast, (left, right) in CONTRASTS.items():
        fields = {}
        for name in HEADS:
            endpoint = measurement(negative[left][name], negative[right][name])
            endpoint["orientation"] = contrast
            effect = measurement(response[left][name], response[right][name])
            effect["orientation"] = f"({left} negative-minus-own-positive) minus ({right} negative-minus-own-positive)"
            own_responses = {}
            for side, lane in (("left", left), ("right", right)):
                own_responses[side] = measurement(negative[lane][name], positive["native" if lane == "native" else "functional"][name])
                own_responses[side]["orientation"] = lane + " negative-minus-own-positive"
                own_responses[side]["denominator_scope"] = "This lane's own positive operand RMS; no cross-installed baseline."
            fields[name] = dict(endpoint=endpoint, signed_edit_response_error=effect,
                left_response=own_responses["left"], right_response=own_responses["right"])
        # MDB's report residual is observed-minus-predicted, explicitly opposite
        # the left-minus-right scalar account above; do not silently relabel it.
        equations[contrast] = compare_core(negative[left], negative[right],
            predicted_baseline=positive["native" if left == "native" else "functional"],
            baseline=positive["native" if right == "native" else "functional"], max_elements=elements)
        reports[contrast] = dict(fields=fields,
            first_endpoint_bit_discrepancy=next((name for name in HEADS if not fields[name]["endpoint"]["bitwise_exact"]), None),
            first_response_bit_discrepancy=next((name for name in HEADS if not fields[name]["signed_edit_response_error"]["bitwise_exact"]), None))
    return dict(comparisons=reports, comparison_max_elements=elements, prospective_signed_hashes_verified=True,
        residual_directions=dict(scalar_accounts="named left-minus-right contrast", MDB_equations="observed-minus-predicted; right minus left"),
        scope="Complete saved coordinate arrays; three errors and their own-positive signed responses are distinct dependent accounts."), equations


def member(root, name):
    root, name = Path(root).resolve(), Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "unsafe evidence member")
    path = root / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "missing/nonregular evidence member")
    return path


def load_array(root, record):
    """Uniform actual writer arrays; NumPy decodes BF16 payloads without Torch."""
    import numpy as np

    path = member(root, record["path"])
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "array file checksum differs")
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    require(list(raw.shape) == record["shape"] and raw.flags.c_contiguous and str(raw.dtype) == record["stored_dtype"], "array serialized storage differs")
    require(hashlib.sha256(raw.tobytes(order="C")).hexdigest() == record["content_sha256"], "array raw payload checksum differs")
    if record["logical_dtype"] == "torch.bfloat16":
        require(raw.dtype == np.dtype("uint16"), "BF16 payload dtype differs")
        value = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        require(record["logical_dtype"] == "torch." + str(raw.dtype), "native array dtype differs")
        value = np.asarray(raw)
    require(np.isfinite(value).all(), "nonfinite retained operand")
    value.flags.writeable = False
    return value


def tensor_tree(root, value):
    if isinstance(value, dict):
        if set(value) == {"tensor"}:
            return load_array(root, value["tensor"])
        return {key: tensor_tree(root, child) for key, child in value.items()}
    if isinstance(value, list):
        return [tensor_tree(root, child) for child in value]
    return value


def same_tree(left, right):
    if hasattr(left, "shape"):
        return hasattr(right, "shape") and left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def operand_identity(record):
    return dict(shape=record["shape"], stride=record["stride"], dtype=record["logical_dtype"],
                device=record["device_at_capture"], bf16_le_sha256=record["content_sha256"])


def finite_addition_account(ports):
    """Diagnostic reduction of saved fields, not a learned recurrence/projection."""
    import numpy as np

    value = ports["Hnext"].astype(np.float64) - ports["R"].astype(np.float64) - ports["M"].astype(np.float64)
    return dict(max_abs=float(np.abs(value).max()), rms=float(np.sqrt(np.mean(value**2))),
                scope="FP64 saved Hnext-minus-R-minus-M remainder; does not execute the model's addition or any learned operation.")


def execution_accounts(shortcut_execution, runtime_flags):
    """The frozen primitive helper binds four of the campaign's eight flags."""
    keys = {"float32_matmul_precision", "deterministic_algorithms", "cuda_allow_tf32", "cuda_allow_bf16_reduced_precision_reduction"}
    require(set(shortcut_execution) == keys and all(runtime_flags.get(key) == shortcut_execution[key] for key in keys),
        "actual shortcut/runtime primitive execution policy differs")
    return dict(shortcut_bound=shortcut_execution, campaign_additional={key: value for key, value in runtime_flags.items() if key not in keys},
        scope="Compare exactly the frozen shortcut helper's four keys. Additional campaign runtime flags remain authenticated metadata, not missing shortcut fields.")


def authenticate(root, expected_manifest, expected_request):
    require(sha(root / "files.json") == expected_manifest, "raw wrapper checksum differs")
    inventory = read(root / "files.json")
    require(inventory["schema"] == "mdb-image-tail-files-v1" and inventory["request_sha256"] == expected_request, "raw wrapper schema/request differs")
    require(not any(path.is_symlink() for path in root.rglob("*")), "raw symlink forbidden")
    require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != root / "files.json"} == set(inventory["files"]),
        "complete raw file inventory differs")
    for name, record in inventory["files"].items():
        path = member(root, name)
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "retained evidence checksum differs: " + name)
    require(inventory["expanded_bytes"] == sum(row["bytes"] for row in inventory["files"].values()), "complete raw byte census differs")
    plan, request, declared, source, summary = (read(root / name) for name in ("plan.json", "request.json", "declared-config.json", "source.json", "summary.json"))
    require(identity(request) == request["request_sha256"] == expected_request and declared == request["config"], "standard executed request/config identity differs")
    require(plan == dict(declared, output=str(Path(declared["output_scratch"]) / inventory["job_id"] / declared["output_name"]),
        request_source_hashes=request["source_hashes"], request_sha256=expected_request, job_id=inventory["job_id"]), "effective producer plan differs")
    require(source["job_id"] == inventory["job_id"] and source["request_sha256"] == expected_request
        and source["source_hashes"] == request["source_hashes"], "actual source job/request/source joins differ")
    require(summary["job_id"] == inventory["job_id"] and summary["request_sha256"] == expected_request, "actual summary job/request joins differ")
    require(request["source_hashes"]["qwen_sign_shortcut_campaign.py"] == WORKER_SHA256
        and source["helper_source_sha256"] == plan["helper_source_sha256"] == PRODUCER_HELPERS
        and all(request["source_hashes"][name] == value for name, value in PRODUCER_HELPERS.items()), "executed original producer/helper source differs")
    require(plan["schema"] == summary["schema"] == "mdb-qwen-sign-shortcut-campaign-v1" and plan["selection_protocol_sha256"] == PROTOCOL_SHA256
        and sha(root / "protocol.json") == PROTOCOL_SHA256 and plan["edit_index"] == 18 and plan["horizon"] == 19 and plan["selected"] == [9, 18, 19],
        "fixed raw protocol/history differs")
    require(plan["qwen_rope_buffer_policy"] == dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="original native CPU initializer; device-only placement")
        and plan["attention_dispatch"] == dict(backend="sdpa", gqa_policy="repeat_kv"), "original dtype/dispatch source policy differs")
    for key in PATH_KEYS:
        require(sha(member(root, Path(plan[key]).name)) == plan[key + "_sha256"], "retained original source metadata differs: " + key)
    require(plan["retrieval_script_sha256"] == RETRIEVER_SHA256 and source["parameter_manifest_sha256"] == plan["qwen_parameter_manifest_sha256"], "standalone original recipe/retriever identity differs")
    return plan, source, summary, inventory


def native_member(name):
    return name.startswith("rope-after/") or name == "parameter-identity-after.json" or (
        name.startswith("cases/") and ("/native/" in name or name.endswith("/native.json")))


def transaction(root, config, source, summary, inventory, *, case_ids=CASE_IDS, source_identity=None):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    old = helpers()[0]
    tx = root / "transaction"
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"], expected_event_head=summary["transaction"]["event_head"],
        max_bytes=config["forecast_transaction_bytes"])
    events = [read(path) for path in sorted((tx / "events").glob("*.json"))]
    require(verified["phase"] == "closed" and [row["kind"] for row in events] ==
        ["opened", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and not read(tx / "seal.json")["violations"], "transaction journal contains missing/failed/violated occurrence")
    prediction = old["plain"](load_forecast_tree(tx, read(tx / "forecasts/cohort.json")["predictions"]))
    observed = old["plain"](load_forecast_tree(tx, read(tx / "observations/native-verification.json")["value"]))
    cases = list(case_ids)
    require(prediction["cases"] == observed["cases"] == summary["cases"] == cases and prediction["native_targets_at_seal"] == 0
        and observed["counts"] == summary["counts"], "actual globally released cohort/counts differs")
    require(prediction["counts"]["forecast_module_attempts"] == prediction["counts"]["native_module_entries"] == 0
        and all(prediction["counts"][key] == 0 for key in ("native_authorizations", "native_blocks", "native_SDPA", "native_full_model_calls", "native_suffix_calls")),
        "native learned work preceded global forecast seal")
    declared = prediction["external_files"]
    post = {"timeline.jsonl", "comparisons.json", "summary.json", "request.json", "declared-config.json"}
    expected = {name for name in inventory["files"] if not name.startswith("transaction/") and name not in post and not native_member(name)}
    require(set(declared) == expected and all(not native_member(name) for name in declared), "complete pretarget file frontier includes future or omits source")
    require(all(row == inventory["files"][name] for name, row in declared.items()), "pretarget full file checksum commitment differs")
    responses = read(root / "forecast-responses.json")
    require(responses == dict(orientation="negative-minus-own-positive", cases=prediction["response_hashes"]), "full signed-response commitments differ from global seal")
    declarations = read(tx / "seal.json")["declarations"]
    sources = source_identity if source_identity is not None else {**source["helper_source_sha256"], "qwen_native": read(root / "runtime.json")["sources"]["qwen_native_source_sha256"]["sha256"]}
    metadata = {case: read(root / "cases" / case / "inputs.json")["metadata"] for case in cases}
    require(declarations["model_identity"] == "original-qwen3-bias-free" and declarations["input_identity"] == digest(metadata)
        and declarations["source_identity"] == digest(sources) and declarations["frontier"] == dict(kind="raw-input", conditional_frontier="own-functional-positive-R-G-U", cases=cases,
            edit_index=config["edit_index"], horizon=config["horizon"], external_files=declared), "sealed model/input/source/frontier identity differs")
    journal = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    expected_kinds = [kind for _case in cases for kind in ("functional_case_started", "positive_only_shortcut_completed", "functional_negative_reference_started", "functional_case_completed")]
    expected_kinds += ["cohort_globally_sealed", "native_targets_released"]
    expected_kinds += [kind for _case in cases for kind in ("native_component_authorized", "native_component_authorized", "native_case_completed")]
    require([row["kind"] for row in journal] == expected_kinds and [row["number"] for row in journal] == list(range(len(journal)))
        and all(row["native_module_entries"] == 0 for row in journal[:14]), "durable shortcut/reference/seal/native-entry chronology differs")
    require(journal[12]["seal_sha256"] == journal[13]["seal_sha256"] == summary["forecast_seal"], "native release names wrong global seal")
    for index, case in enumerate(cases):
        require(all(row["case"] == case for row in journal[index * 4:index * 4 + 4])
            and journal[index * 4 + 1]["shortcut_sha256"] == sha(root / "cases" / case / "shortcut.json")
            and journal[index * 4 + 1]["no_negative_ports_accessed"] is True, "positive-only shortcut timing/source differs")
    return dict(verified=verified, chronology=journal, external_files=len(declared), native_targets_at_seal=0), prediction


def case_accounts(root, case, *, edit_index, horizon, native_layers, sealed):
    import numpy as np

    _, q, _ = helpers()
    inputs, forecast, native, shortcut = (read(root / "cases" / case / name) for name in
                                         ("inputs.json", "forecast.json", "native.json", "shortcut.json"))
    ids, mask = (load_array(root, inputs[name]) for name in ("input_ids", "attention_mask"))
    require(ids.dtype == np.dtype("int64") and mask.dtype in (np.dtype("int64"), np.dtype("int32")) and ids.shape == mask.shape and ids.shape[0] == 1
        and np.isin(mask, (0, 1)).all() and bool(mask.sum()), "own source IDs/mask geometry differs")
    padded = bool((mask == 0).any())
    profile = forecast["profile"]
    require(profile["edit_index"] == edit_index and profile["horizon"] == horizon
        and profile["attention_dispatch"]["backend"] == "sdpa"
        and profile["attention_dispatch"]["gqa_policy"] == ("repeat_kv" if padded else "enable_gqa")
        and isinstance(profile["attention_dispatch"]["source"], str) and bool(profile["attention_dispatch"]["source"])
        and profile["genuine_qwen_act"] is False, "source functional attention/edit profile differs")
    positions = load_array(root, forecast["position_ids"])
    require(positions.dtype == np.dtype("int64") and np.array_equal(positions, np.arange(ids.shape[1])[None]), "actual functional positions differ")
    require(set(forecast["common"]) == set(native["common"]) == {str(i) for i in range(horizon + 1)}
        and set(forecast["minus"]) == set(native["minus"]) == {str(edit_index), str(horizon)}, "complete own prefix/one-block exit coverage differs")
    for row in (forecast, native):
        records = list(row["common"].values()) + list(row["minus"].values()) + [record for bank in row["ports"].values() for record in bank.values()]
        require(all(record["logical_dtype"] == "torch.bfloat16" for record in records), "source BF16 state/port contract differs")
    require(all(record["logical_dtype"] == "torch.bfloat16" for record in shortcut["ports"].values()), "shortcut BF16 output contract differs")
    prefix = {key: q["row_accounts"](load_array(root, forecast["common"][key]), load_array(root, native["common"][key]), mask)
              for key in forecast["common"]}
    ports = {lane: {sign: {name: load_array(root, record) for name, record in row["ports"][sign].items()}
                    for sign in ("plus", "minus")} for lane, row in (("functional", forecast), ("native", native))}
    for lane, row in (("functional", forecast), ("native", native)):
        for sign in ("plus", "minus"):
            state = row["common"] if sign == "plus" else row["minus"]
            require(same_tree(ports[lane][sign]["h"], load_array(root, state[str(edit_index)]))
                and same_tree(ports[lane][sign]["Hnext"], load_array(root, state[str(horizon)])), "actual block operand/return state join differs")
        require(same_tree(-load_array(root, row["common"][str(edit_index)]), load_array(root, row["minus"][str(edit_index)])), "whole own-state negative edit differs")
    require(native["genuine_qwen_act"] is False and set(native["reference_only"]) == {str(i) for i in range(horizon + 1, native_layers + 1)}
        and all(row["retained"] is False for row in native["reference_only"].values()), "reference-only full native horizon metadata differs")
    consumed, context = tensor_tree(root, native["consumed"]), tensor_tree(root, native["source_context"])
    require(len(consumed) == 1 and consumed[0]["index"] == edit_index
        and same_tree(consumed[0]["hidden"], ports["native"]["minus"]["h"])
        and same_tree(consumed[0]["kwargs"], context), "actual native suffix consumed wrong own hidden/context")
    require(context.get("past_key_values") is None and not context.get("use_cache", False), "native context unexpectedly cached")
    if context.get("position_ids") is not None:
        require(same_tree(context["position_ids"], positions), "native consumed positions differ from own input positions")
    expected_entries = [dict(lane="baseline", index=i) for i in range(native_layers)] + [dict(lane="suffix", index=edit_index)]
    require(native["block_entries"] == expected_entries and len(native["sdpa_calls"]) == native_layers + 1
        and native["counts"]["SDPA"] == native_layers + 1 and len(native["module_entries"]) == native["counts"]["native_module_entries"], "actual native component/SDPA/module entry census differs")
    for receipt, expected in zip(native["sdpa_calls"], expected_entries, strict=True):
        require(receipt["lane"] == expected["lane"] and receipt["index"] == expected["index"] and receipt["dropout_p"] == 0., "actual native attention occurrence differs")
        if receipt["index"] == edit_index:
            sign = "plus" if receipt["lane"] == "baseline" else "minus"
            actual_qkv = [native["ports"][sign][name] for name in ("q_consumed", "k_consumed", "v_consumed")]
            require(receipt["QKV"] == [dict(shape=row["shape"], stride=row["stride"], dtype=row["logical_dtype"], device=row["device_at_capture"]) for row in actual_qkv],
                "actual edited-block SDPA operand receipt differs from saved consumed Q/K/V")
    require(native["counts"]["baseline_blocks"] == native_layers and native["counts"]["suffix_blocks"] == native["counts"]["full_model_calls"] == 1
        and any(row["name"] == "lm_head" and row["lane"] == "baseline" for row in native["module_entries"]), "actual full-model/LM/suffix census differs")
    blocks = horizon + 1
    require(forecast["counts"]["blocks"] == forecast["counts"]["SDPA"] == blocks and forecast["counts"]["native_module_forwards"] == 0,
        "actual functional primitive census differs")
    frontier = shortcut["frontier"]
    require(frontier["only_positive_basis"] == ["R", "gate", "up"] and frontier["before_first_minus_callback"] is True
        and frontier["basis"] == {name: forecast["ports"]["plus"][name] for name in ("R", "gate", "up")}
        and frontier["down_parameter"] == f"model.layers.{edit_index}.mlp.down_proj.weight", "shortcut frontier includes wrong/nonpositive operands")
    for name in ("R", "gate", "up"):
        require(shortcut["input_descriptors"][name] == operand_identity(frontier["basis"][name]), "actual shortcut positive operand identity differs")
    require(shortcut["profile"]["frontier_kind"] == "own-functional-positive-ports" and shortcut["profile"]["sign_scope"] == "all-query/all-feature"
        and shortcut["profile"]["operands"] == shortcut["input_descriptors"], "actual positive-only finite profile differs from input operands")
    shortcut_values = {name: load_array(root, row) for name, row in shortcut["ports"].items()}
    require(shortcut["output_descriptors"] == {name: operand_identity(row) for name, row in shortcut["ports"].items()}, "shortcut full output descriptors differ")
    require(same_tree(shortcut_values["negative_gate"], -ports["functional"]["plus"]["gate"])
        and same_tree(shortcut_values["negative_up"], -ports["functional"]["plus"]["up"])
        and same_tree(shortcut_values["negative_R"], -ports["functional"]["plus"]["R"]), "shortcut actual unary negative operands differ")
    require(shortcut["counts"] == dict(negations=3, SiLU=1, elementwise_products=1, linear=1, residual_additions=1,
        attention=0, RMSNorm=0, gate_projection=0, up_projection=0, module_forwards=0), "shortcut actual work profile differs")
    attention = shortcut["profile"]["attention_profile"]
    require(attention["mask_identity"] == inputs["attention_mask"]["content_sha256"] and attention["positions_identity"] == forecast["position_ids"]["content_sha256"]
        and attention["query_rows"] == ("partially-padded" if padded else "all-attended"), "shortcut source context identities differ")
    negative = {"shortcut": {name: shortcut_values[name] for name in HEADS}, **{lane: {name: ports[lane]["minus"][name] for name in HEADS} for lane in ports}}
    positive = {lane: {name: ports[lane]["plus"][name] for name in HEADS} for lane in ports}
    tail, equations = compare_values(negative, positive, {"functional": sealed["full_functional"], "shortcut": sealed["shortcut"]})
    aligned = {sign: {name: q["row_accounts"](ports["functional"][sign][name], ports["native"][sign][name], mask)
                       for name in ports["functional"][sign].keys() & ports["native"][sign].keys()}
               for sign in ("plus", "minus")}
    return dict(prefix_source_errors=prefix, aligned_ports=aligned, tail=tail,
        oddness={lane: {name: oddness(row["plus"][name], row["minus"][name]) for name in ("R", "gate", "up")} for lane, row in ports.items()},
        finite_addition={lane: {sign: finite_addition_account(value) for sign, value in row.items()} for lane, row in ports.items()},
        source=dict(input_identity=digest(inputs), actual_attended_rows=int(mask.sum()), actual_padded_rows=int((mask == 0).sum()), metadata=inputs["metadata"], functional_profile=profile),
        native=dict(actual_consumed_occurrences=len(consumed), block_entries=len(native["block_entries"]), module_entries=len(native["module_entries"]), SDPA=len(native["sdpa_calls"]),
            baseline_logits_shape=native["baseline_logits_shape"], genuine_qwen_act=False),
        coverage=dict(primary_prefix_fields=len(prefix), overlapping_prefix_regions=2 * len(prefix), aligned_port_fields=sum(map(len, aligned.values())),
            primary_endpoint_error_fields=9, primary_response_error_fields=9),
        first_prefix_bit_discrepancy=next((key for key in sorted(prefix, key=int) if not prefix[key]["full"]["bitwise_exact"]), None),
        scope="Full own-input arrays; regional and port checks overlap prefix/tail accounts. No selected-kernel, symbolic identity or universal rule claim."), equations


def catalog_accounts(output, audit, equations, contexts):
    """One supplied generic rule, three conditional occurrences; no proof/induction."""
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog

    catalog = EquationCatalog(Path(output) / "catalog/equations")
    records, effects, queries = [], {}, []
    for case, context in contexts.items():
        report = equations[case]["shortcut-minus-native-negative"]
        record = catalog.record(law_id=RULE_ID, route_id=ROUTE_ID, alias="cmp.qwen.sign-shortcut." + case,
            binding={key: context[key] for key in ("model_identity", "execution", "input_identity", "frontier", "horizon", "source_identity")},
            conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
            readers=["qwen.block18." + name for name in HEADS], report=report,
            evidence=dict(audit_sha256=sha(Path(output) / "audit.json"), mathematical_identity="real arithmetic annotation only",
                finite_formula="Gm=neg(Gp); Um=neg(Up); Pm=SiLU(Gm)*Um; Mm=linear(Pm,Wd,None); Ym=neg(Rp)+Mm",
                regrouped_real_annotation="M(-x)+M(x)=Wd(G*U) in real arithmetic; no RHS learned projection executed",
                BF16_regrouping_identity=False, independent_kernel=False, actual_negative_targets_not_shortcut_inputs=True))
        context_wrong = {**context, "input_identity": "0" * 64}
        exact, wrong = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to=context_wrong))
        require(exact == 1 and wrong == 0, "conditional catalog applicability differs")
        queries.append(dict(case=case, fingerprint=record["fingerprint"], context=context, exact=exact, wrong_input=wrong))
        records.append(record["fingerprint"])
        fields = audit["cases"][case]["tail"]["comparisons"]["shortcut-minus-native-negative"]["fields"]
        effects[record["fingerprint"]] = [dict(consumer_id="qwen.block18." + name, contrast_id=contrast,
            value=fields[name][metric]["signed_mean"], units="represented coordinate units",
            provenance=[dict(path="audit.json", sha256=sha(Path(output) / "audit.json"))])
            for name in HEADS for contrast, metric in (("shortcut-minus-native", "endpoint"), ("shortcut-response-minus-native-response", "signed_edit_response_error"))]
    atlas = ComputationAtlas(Path(output) / "atlas")
    receipts = atlas.ingest(Path(output) / "catalog/equations", records,
        provenance=dict(run_id=audit["job_id"] + ":saved-sign-shortcut",
                        artifacts=[dict(path="audit.json", sha256=sha(Path(output) / "audit.json"))]), signed_effects=effects)
    require(all(atlas.show(row["fingerprint"])["rule_status"] == "observation; no validated rule" for row in receipts), "atlas observation overclaims rule status")
    save(Path(output) / "catalog/provenance.json", dict(records=len(records), generic_rule_id=RULE_ID, queries=queries,
        automatic_family_induction=False, symbolic_proof=False, BF16_regrouping_identity=False,
        scope="Three dependent source-bound occurrences, not universal validation. Saved lookups are not CLI executions."))
    save(Path(output) / "atlas/provenance.json", dict(observations=[row["fingerprint"] for row in receipts],
        generic_rule_id=RULE_ID, status="observation; no validated rule", family_induction=False,
        scope="Exact-source conditional observations; no new model call or mathematical proof."))
    return dict(records=len(records), atlas_observations=len(receipts), generic_rule_id=RULE_ID)


def source_accounts(root, config, source):
    """Original bytes and runtime identities; no tokenizer or model execution."""
    _, q, streamer = helpers()
    qconfig = read(root / Path(config["qwen_config"]).name)
    manifest = read(root / Path(config["qwen_parameter_manifest"]).name)
    require((qconfig["hidden_size"], qconfig["intermediate_size"], qconfig["num_hidden_layers"], qconfig["num_attention_heads"], qconfig["num_key_value_heads"], qconfig["head_dim"])
        == (2560, 9728, 36, 32, 8, 128) and len(manifest["artifacts"]) == 398, "original Qwen geometry/recipe coverage differs")
    require(not qconfig.get("attention_bias", False) and qconfig.get("attention_dropout", 0.) == 0. and qconfig.get("tie_word_embeddings", True), "original source sign assumptions differ")
    roles = q["parameter_roles"](manifest, qconfig, config["horizon"])
    expected = {row["source"]["tensor"]: row["source"]["payload_sha256"] for row in manifest["artifacts"]}
    before = source["original_parameter_payloads"]
    require(set(before) == set(expected) and all(row["sha256"] == expected[name] and row["dtype"] == "torch.bfloat16" for name, row in before.items()),
        "resident source learned payloads differ from original recipes")
    require(all(before[row["source"]["tensor"]]["shape"] == row["source"]["shape"] for row in manifest["artifacts"]), "resident parameter shape recipe differs")
    restored = streamer["restore_parameters"](manifest, config["parameter_root"], expected, expected_module_sha256=RETRIEVER_SHA256)
    io = source["parameter_recipe_io"]
    require(io["payload_reads"] == io["load_requests"] == len(expected) and io["cache_hits"] == 0
        and io["parameter_payload_bytes"] == restored["payload_bytes"] and io["checkpoint_files_verified_once"] == len(restored["checkpoint_files"]),
        "observed one-hydration recipe access count differs")
    after = read(root / "parameter-identity-after.json")
    require(after["payloads"] == before and source["qwen_tied_alias_before"] is True and after["qwen_tied_alias_after"] is True,
        "final resident learned payload/tied alias check differs")
    require(set(source["rope_buffers"]) == set(after["rope_buffers"]) and source["rope_buffers"], "complete original RoPE custody differs")
    for name, row in after["rope_buffers"].items():
        value = load_array(root, row)
        require(row["logical_dtype"] == "torch.float32" and value.dtype.name == "float32", "final RoPE buffer FP32 policy differs")
        require(same_tree(load_array(root, source["rope_buffers"][name]), value), "original FP32 RoPE buffer changed")
    binding = read(root / Path(config["qwen_source_binding"]).name)
    expected_parser = {name for name in binding["source_files"] if name.startswith("tokenizer/") or
                       name in {"text_encoder/config.json", "text_encoder/generation_config.json"}}
    require(set(source["tokenizer_files"]) == expected_parser and expected_parser, "complete parser source custody coverage differs")
    for name, row in source["tokenizer_files"].items():
        original = binding["source_files"][name]
        require(row["sha256"] == original["sha256"] and row["bytes"] == original["bytes"]
            and sha(member(root, row["retained"])) == row["sha256"], "retained original parser source differs")
    runtime = read(root / "runtime.json")
    require(set(runtime["sources"]) == {"qwen_native_source_sha256", "tokenizers_wrapper_source_sha256", "tokenizers_native_source_sha256",
        "jinja_environment_source_sha256", "jinja_sandbox_source_sha256"}, "actual original runtime source coverage differs")
    for name, row in runtime["sources"].items():
        require(config[name] == row["expected_sha256"] and (row["expected_sha256"] == row["sha256"] or
            name == "tokenizers_native_source_sha256" and row["expected_sha256"] == "runtime-bound"), "source/runtime configured policy differs")
        path = Path(row["path"])
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "installed source runtime bytes differ")
    return dict(parameter_restoration=restored, parameter_recipe_io=io, roles=roles, final_resident_payload_and_tied_alias=True,
        runtime=runtime, parser_source_files=source["tokenizer_files"],
        tokenizer_scope="Original parser-source bytes and retained raw/formatted/ID/mask metadata authenticated. This saved reader does not execute a tokenizer or claim an independent tokenization replay.")


def raw_protocol_accounts(root, cases):
    protocol = read(root / "protocol.json")
    require([row["case_id"] for row in protocol["cases"]] == list(CASE_IDS), "fixed context cohort/order differs")
    rows = {}
    for row in protocol["cases"]:
        raw = member(root, row["raw_file"])
        require(raw.stat().st_size == row["raw_bytes"] and sha(raw) == row["raw_sha256"] and raw.read_bytes() == row["raw_text"].encode("utf-8"), "frozen literal raw context differs")
        metadata = cases[row["case_id"]]["source"]["metadata"]
        require(metadata["raw_text"] == row["raw_text"] and metadata["raw_sha256"] == row["raw_sha256"]
            and metadata["formatted_sha256"] == hashlib.sha256(metadata["formatted_prompt"].encode()).hexdigest()
            and metadata["intended_regime"] == row["intended_mask_regime"], "actual raw/formatted source metadata differs")
        require(metadata["actual_attended_rows"] == cases[row["case_id"]]["source"]["actual_attended_rows"]
            and metadata["actual_padded_rows"] == cases[row["case_id"]]["source"]["actual_padded_rows"], "actual retained mask count differs")
        rows[row["case_id"]] = dict(raw_sha256=row["raw_sha256"], raw_bytes=row["raw_bytes"], intended_regime=row["intended_mask_regime"],
            actual_regime="partially-padded" if metadata["actual_padded_rows"] else "all-attended", mask_regime_match_diagnostic=
                (metadata["actual_padded_rows"] > 0) == (row["intended_mask_regime"] == "padded"), untruncated_token_count=metadata["untruncated_token_count"])
    return dict(protocol_sha256=sha(root / "protocol.json"), contexts=rows, status="selection/source custody; intended mask regimes are diagnostics, not outcome gates")


def audit(root, output, *, expected_manifest, expected_request):
    root, output = Path(root), Path(output)
    config, source, summary, inventory = authenticate(root, expected_manifest, expected_request)
    tx, prediction = transaction(root, config, source, summary, inventory)
    sources = source_accounts(root, config, source)
    qconfig = read(root / Path(config["qwen_config"]).name)
    edit, horizon, layers = config["edit_index"], config["horizon"], qconfig["num_hidden_layers"]
    expected_counts = dict(functional_reference_blocks=3 * (horizon + 1), functional_reference_SDPA=3 * (horizon + 1),
        shortcut_down_projections=3, native_blocks=3 * (layers + 1), native_SDPA=3 * (layers + 1),
        native_full_model_calls=3, native_suffix_calls=3, native_authorizations=6, forecast_module_attempts=0,
        functional_observed_SDPA=3 * (horizon + 1), native_observed_SDPA=3 * (layers + 1), shortcut_observed_SDPA=0, shortcut_observed_linear=3)
    require(all(summary["counts"][key] == value for key, value in expected_counts.items()), "actual bounded campaign census differs")
    require(summary["contract"]["edit_index"] == edit and summary["contract"]["horizon"] == horizon
        and summary["contract"]["native_blocks"] == 3 * (layers + 1) and summary["contract"]["functional_reference_blocks"] == 3 * (horizon + 1)
        and summary["contract"]["genuine_qwen_acts"] == 0 and summary["contract"]["no_flux_vae_noise"] is True, "declared physical geometry/source lane differs")
    cases, equations = {}, {}
    for case in CASE_IDS:
        cases[case], equations[case] = case_accounts(root, case, edit_index=edit, horizon=horizon, native_layers=layers,
            sealed=prediction["response_hashes"][case])
        require(cases[case]["source"]["actual_attended_rows"] + cases[case]["source"]["actual_padded_rows"] == 512, "original complete query-row geometry differs")
        require(all(row["full"]["shape"] == [1, 512, qconfig["hidden_size"]] for row in cases[case]["prefix_source_errors"].values()), "original whole-state hidden geometry differs")
        expected_shapes = dict(product=[1, 512, qconfig["intermediate_size"]], M=[1, 512, qconfig["hidden_size"]], Hnext=[1, 512, qconfig["hidden_size"]])
        require(all(row["endpoint"]["shape"] == expected_shapes[name] for name, row in cases[case]["tail"]["comparisons"]["shortcut-minus-native-negative"]["fields"].items()), "original P/M/H geometry differs")
        require(cases[case]["source"]["functional_profile"]["attention_dispatch"]["source"] == sources["runtime"]["sources"]["qwen_native_source_sha256"]["sha256"],
            "functional attention dispatch lacks original native source binding")
        shortcut = read(root / "cases" / case / "shortcut.json")
        cases[case]["shortcut_execution_binding"] = execution_accounts(shortcut["profile"]["execution"], sources["runtime"]["flags"])
        require(shortcut["profile"]["source_identity"] == {**PRODUCER_HELPERS,
            "qwen_native": sources["runtime"]["sources"]["qwen_native_source_sha256"]["sha256"]}, "actual positive-only source profile identity differs")
        name = f"model.layers.{edit}.mlp.down_proj.weight"
        require(shortcut["input_descriptors"]["down_weight"]["bf16_le_sha256"] == source["original_parameter_payloads"][name]["sha256"], "actual shortcut learned down operand differs")
    protocol = raw_protocol_accounts(root, cases)
    require(sum(row["native"]["module_entries"] for row in cases.values()) == summary["counts"]["native_module_entries"], "global native module entry count differs")
    report = dict(schema="mdb-qwen-sign-shortcut-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, worker_sha256=WORKER_SHA256, audit_sha256=sha(__file__),
        source=sources, protocol=protocol, transaction=tx, counts=summary["counts"], contract=summary["contract"], cases=cases,
        mechanics_status="authenticated", instrument_status="complete saved-coordinate reductions", trend_status="observations; retain misses", terminal_status="not certified",
        limitations=["Conditional shortcut from own functional positive R/G/U; finite oddness is measured, not assumed or repaired.",
            "Three dependent contexts and one block horizon; no universal rule, symbolic proof or BF16 regrouping identity.",
            "Native full36/LM beyond the retained horizon is reference-only; no FLUX, VAE, image or learned continuation in this campaign.",
            "SDPA/module counters and operand receipts do not identify selected CUDA kernels or establish an independent kernel.",
            "Native consumed Q/K/V and head use native layout; only declared common-basis ports are compared directly.",
            "Saved parser-source metadata is authenticated without a new tokenizer execution."])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", report)
    contexts = {case: dict(model_identity=config["qwen_parameter_manifest_sha256"], source_identity=source["source_hashes"],
        execution=dict(backend="sdpa", learned_dtype="bfloat16", runtime_identity=digest(sources["runtime"]), primitive_policy=sources["runtime"]["flags"],
            attention_dispatch=cases[case]["source"]["functional_profile"]["attention_dispatch"]),
        input_identity=cases[case]["source"]["input_identity"], frontier="own-functional-positive-R-G-U", horizon=horizon, case=case)
        for case in CASE_IDS}
    catalog = catalog_accounts(output, report, equations, contexts)
    save(output / "coverage.json", dict(cases={case: row["coverage"] for case, row in cases.items()}, catalog=catalog,
        primary_and_regions_separate=True, dependent_accounts=True, model_calls=0, learned_projections=0, tokenizer_calls=0))
    return report, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(identity(request) == request["request_sha256"] and request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request/config differs")
    for name, value in request["source_hashes"].items():
        require(sha(name) == value, "staged saved-reader source differs")
    started = time.monotonic()
    root = Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    report, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime.json", "files.json", "protocol.json", "forecast-responses.json", "comparisons.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    plan = read(root / "plan.json")
    for key in PATH_KEYS:
        target = output / "source-metadata" / (key + "-" + Path(plan[key]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / Path(plan[key]).name, target)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots=dict(native=str(root)), manifests=dict(native={name: row["sha256"] for name, row in inventory["files"].items()}),
        scope="Complete arrays and original source checkpoint remain near data; compact scalar accounts/catalog/atlas and standalone recipes/retrievers only, no raw activation or weight bank."))
    save(output / "request.json", request)
    save(output / "summary.json", dict(schema="mdb-qwen-sign-shortcut-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=report["job_id"], cases=3, catalog_records=3, atlas_observations=3, model_calls=0, seconds=time.monotonic() - started))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("root", "output", "expected-manifest", "expected-request"):
            parser.add_argument("--" + name, required=True)
        args = parser.parse_args()
        audit(args.root, args.output, expected_manifest=args.expected_manifest, expected_request=args.expected_request)
