"""Saved-only inherited positioned-QKV component observations.

Uses frozen array/metric/custody utilities; no generative helper, model,
tokenizer, attention or learned projection is imported or executed.
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
WORKER_SHA256 = "d2c9e472ac06c667fea0726ac62f629e63ad776173a2e323821bf66f71b0d8f5"
COMPONENT_SHA256 = "81131b179f1a8a5ea52011be3e450e5901255f350d89ff11c053902c16a28580"
PREDICTOR_SHA256 = "95d356185b8bdd64479bc1e7937fa034aa24dc5f5b011b7075f0430e31c1bb21"
PROTOCOL_SHA256 = "8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff"
PINS = {
    "audit_qwen_row_sign.py": "d995c3c5ca3c194a3ce87bfca58e9352d251246d798e21c6e04a7938bd9b5838",
    "audit_qwen_sign_shortcut.py": "d33613bbb1d71ee3b2007fb007e5266acaf141994091264bed50d27342237e2c",
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
PATH_KEYS = ("qwen_config", "qwen_parameter_manifest", "parent_parameter_manifest", "qwen_source_binding", "retrieval_script",
    "selection_protocol", "parent_files", "parent_request", "parent_source", "parent_runtime", "parent_seal", "parent_forecast")
PORTS, QKV = ("head", "R", "M", "Hnext"), ("q", "k", "v")
CHANNELS = {"cached-qkv-minus-full-functional": ("cached", "functional"),
    "full-functional-minus-native": ("functional", "native"), "cached-qkv-minus-native": ("cached", "native")}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


@lru_cache(maxsize=1)
def row():
    for name, expected in PINS.items():
        require(sha(HERE / name) == expected, "frozen saved helper differs: " + name)
    return runpy.run_path(str(HERE / "audit_qwen_row_sign.py"))


def base():
    return row()["base"]()


def metric(left, right):
    import numpy as np

    report = base()["measurement"](left, right)
    shared = (left == 0) & (right == 0)
    report.update(orientation="supplied left-minus-right", denominator="RMS of supplied right operand",
        signed_zero=dict(shared=int(shared.sum()), defects=int(np.count_nonzero(shared & (np.signbit(left) != np.signbit(right))))),
        bit_scope="Complete decoded finite logical values preserve BF16 bits/sign zero; capture layout is a separate account.")
    return report


def windowed(left, right, windows, *, query_axis=1):
    require(left.shape == right.shape and left.shape[0] == 1 and query_axis in (1, 2), "common logical operand basis differs")
    length = left.shape[query_axis]
    require(windows and windows[0][0] == 0 and windows[-1][1] == length and all(0 <= a < b <= length for a, b in windows)
        and all(a[1] == b[0] for a, b in zip(windows, windows[1:])), "protocol windows do not partition query rows")
    result = []
    for start, stop in windows:
        index = [slice(None)] * left.ndim; index[query_axis] = slice(start, stop)
        result.append(dict(bounds=[start, stop], comparison=metric(left[tuple(index)], right[tuple(index)])))
    return dict(whole=metric(left, right), windows=result, query_axis=query_axis,
        scope="Whole field is primary; protocol windows partition it and are dependent views, never accuracy selectors.")


def support(signs, permitted, retained, arm):
    import numpy as np

    require(same(signs, np.asarray(arm["row_signs"], np.float32)) and permitted.dtype == retained.dtype == np.dtype("bool")
        and same(permitted, np.tri(signs.size, dtype=bool)), "declared row signs/actual causal support differ")
    cross = permitted & (signs[:, None] != signs[None, :])
    closed = permitted.any(-1) & ~cross.any(-1)
    require(same(retained, closed) and int(cross.sum()) == arm["permitted_cross_sign_edges"]
        and int(closed.sum()) == arm["structural_support_closed_rows"], "structural support instrument differs")
    return dict(rows=int(signs.size), closed_rows=int(closed.sum()), cross_sign_edges=int(cross.sum()), windows=[dict(bounds=[a, b],
        rows=b-a, closed_rows=int(closed[a:b].sum()), cross_sign_edges=int(cross[a:b].sum())) for a, b in arm["windows"]],
        scope="Nonempty permitted keys with the query sign; structural diagnostic only, all rows retained.")


def same(left, right):
    return row()["same"](left, right)


def dense_layout(shape, stride):
    require(len(shape) == len(stride) > 0 and all(type(n) is int and n > 0 for n in shape)
        and all(type(n) is int and n >= 0 for n in stride), "invalid dense layout schema")
    extent = 1
    for step, size in sorted((step, size) for size, step in zip(shape, stride) if size > 1):
        require(step == extent, "overlapping or nondense capture layout")
        extent *= size
    return extent


def live_join(record, live):
    expected = dict(shape=record["shape"], stride=record["stride"], dtype=record["logical_dtype"],
        device=record["device_at_capture"], content_sha256=record["content_sha256"])
    require({key: live[key] for key in expected} == expected, "actual consumed operand/full saved record differs")


def layout_accounts(root, references):
    value = read(root / "layout-restoration.json")["arrays"]
    require(set(value) == set(references), "complete inherited layout restoration coverage differs")
    for name, record in references.items():
        item, original = value[name], dict(record)
        dense_layout(original["shape"], original["stride"])
        live_join(original, item["restored"])
        staged = item["loaded_logical"]
        require(staged["shape"] == original["shape"] and staged["dtype"] == original["logical_dtype"] and staged["content_sha256"] == original["content_sha256"]
            and staged["device"] == "cpu" and item["value_bit_exact"] is True, "logical archive/restored value or staging policy differs")
        dense_layout(staged["shape"], staged["stride"])
    return dict(arrays=value, scope="Authenticated original logical archive bytes joined to captured staging/restored descriptors; restored capture stride is explicit, not inferred from NPY storage.")


def load_ports(root, records, *, native=False):
    result = {name: base()["load_array"](root, record) for name, record in records.items()}
    if native:
        result["head"] = result["head"].transpose(0, 2, 1, 3)
    return result


def numerical_accounts(root, inherited, protocol):
    import numpy as np

    from mdb.equation_comparison import compare_equations

    forecast, native, responses = (read(root / name) for name in ("forecast.json", "native.json", "forecast-responses.json"))
    arm_ids = [arm["id"] for arm in protocol["arms"]]
    require(set(forecast["arms"]) == set(native["arms"]) == set(responses["arms"]) == set(arm_ids)
        and native["head_to_canonical"] == [0, 2, 1, 3] and responses["orientation"] == "edited-minus-own-positive", "complete named-arm/response/native-head schema differs")
    references = read(root / "source-frontier.json")["source_arrays"]
    require(forecast["positive"] == {name: references[name] for name in ("h", *QKV, *PORTS)}, "archived positive is not the exact inherited baseline refs")
    context = native["context"]
    require(same(base()["load_array"](root, context["position_ids"]), inherited["position_ids"]) and context["attention_mask"] is None and context["past_key_values"] is None
        and context["is_causal"] is True and context["enable_gqa"] is True and context["use_cache"] is False, "actual context is not own all-attended positions/cache policy")
    require(all(base()["load_array"](root, context[name]).shape == (1, protocol["sequence_rows"], inherited["q"].shape[-1]) and context[name]["logical_dtype"] == "torch.bfloat16"
        for name in ("cos", "sin")), "own regenerated RoPE geometry/dtype differs")
    positive = dict(functional=inherited, native=load_ports(root, native["baseline"]["ports"], native=True))
    results, equations = {}, {}
    for arm in protocol["arms"]:
        name, windows = arm["id"], arm["windows"]
        saved = forecast["arms"][name]
        require(saved["windows"] == windows and set(saved["lanes"]) == {"cached", "functional"}, "protocol windows/forecast lanes differ")
        signs, permitted, closed = (base()["load_array"](root, saved[key]) for key in ("row_signs", "permitted", "support_closed"))
        structural = support(signs, permitted, closed, arm)
        lanes = {lane: load_ports(root, saved["lanes"][lane]["ports"]) for lane in ("cached", "functional")}
        lanes["native"] = load_ports(root, native["arms"][name]["ports"], native=True)
        edited = inherited["h"] * signs.reshape(1, -1, 1)
        require(all(same(bank["h"], edited) for bank in lanes.values()) and same(base()["load_array"](root, native["inputs"][name]), edited), "a component lane consumed wrong own edited H18")
        for lane in ("cached", "functional"):
            require(set(responses["arms"][name][lane]) == set(PORTS) and all(responses["arms"][name][lane][port] ==
                row()["descriptor"](lanes[lane][port], inherited[port]) for port in PORTS), "full pretarget signed response commitment differs")
        fields, qkv, equations[name] = {}, {}, {}
        for channel, (left, right) in CHANNELS.items():
            lbase, rbase = positive["native" if left == "native" else "functional"], positive["native" if right == "native" else "functional"]
            fields[channel] = {port: dict(endpoint=windowed(lanes[left][port], lanes[right][port], windows),
                response_error=windowed(lanes[left][port].astype(np.float64)-lbase[port], lanes[right][port].astype(np.float64)-rbase[port], windows),
                left_effect=windowed(lanes[left][port], lbase[port], windows), right_effect=windowed(lanes[right][port], rbase[port], windows)) for port in PORTS}
            qkv[channel] = {port: windowed(lanes[left][port], lanes[right][port], windows, query_axis=2) for port in QKV}
            equations[name][channel] = compare_equations({port: lanes[left][port] for port in PORTS}, {port: lanes[right][port] for port in PORTS},
                predicted_baseline={port: lbase[port] for port in PORTS}, baseline={port: rbase[port] for port in PORTS}, max_elements=sum(lanes[left][port].size for port in PORTS))
        results[name] = dict(channels=fields, positioned_QKV=qkv, support=structural,
            cached_source_sign_relation={port: windowed(lanes["cached"][port], inherited[port]*signs.reshape(1, 1, -1, 1), windows, query_axis=2) for port in QKV},
            first_discrepancies={channel: {kind: next((port for port in (*QKV, *PORTS) if
                ((qkv[channel][port] if port in QKV else fields[channel][port]["endpoint"])["whole"]["changed_values"] > 0 if kind == "numeric" else
                 not (qkv[channel][port] if port in QKV else fields[channel][port]["endpoint"])["whole"]["bitwise_exact"])), None) for kind in ("numeric", "bits")} for channel in CHANNELS},
            coverage=dict(primary_endpoint_fields=len(PORTS)*len(CHANNELS), primary_response_fields=len(PORTS)*len(CHANNELS),
                overlapping_endpoint_windows=len(PORTS)*len(CHANNELS)*len(windows), overlapping_response_windows=len(PORTS)*len(CHANNELS)*len(windows),
                QKV_channel_fields=3*len(CHANNELS), QKV_source_sign_fields=3))
    require(same(positive["native"]["h"], inherited["h"]) and same(base()["load_array"](root, native["inputs"]["baseline"]), inherited["h"]), "fresh native baseline consumed wrong own H18")
    return dict(arms=results, positive_baseline={port: metric(inherited[port], positive["native"][port]) for port in (*PORTS, *QKV)},
        scope="Three endpoint and own-positive signed-response channels remain distinct. QKV/raw bits/sign zero/layout and windows are overlapping observations; every miss and continuation retained."), equations


def operand_accounts(root, protocol, summary, qconfig):
    import numpy as np

    forecast, native, calls, contract = (read(root / name) for name in ("forecast.json", "native.json", "sdpa.json", "native-contract.json"))
    arms = [arm["id"] for arm in protocol["arms"]]
    order = [(lane, arm) for arm in arms for lane in ("cached", "functional")] + [("native", arm) for arm in ("baseline", *arms)]
    require([(call["lane"], call["arm"]) for call in calls["calls"]] == order and read(root / "native-sdpa.json")["calls"] == calls["calls"][-len(arms)-1:], "actual seven SDPA occurrences/order differ")
    native_context, identities, qkv_layouts = native["context"], [], []
    for (lane, arm), call in zip(order, calls["calls"], strict=True):
        ports = (native["baseline"] if arm == "baseline" else native["arms"][arm])["ports"] if lane == "native" else forecast["arms"][arm]["lanes"][lane]["ports"]
        require(call["QKV"] == [ports[name] for name in QKV] and call["attention_mask"] is None and call["is_causal"] is True
            and call["enable_gqa"] is True and call["dropout_p"] == 0. and call["scale"] == qconfig["head_dim"] ** -.5, "actual SDPA ports/delegated profile differs")
        for name, record, live, heads in zip(QKV, call["QKV"], call["operands"], (qconfig["num_attention_heads"], qconfig["num_key_value_heads"], qconfig["num_key_value_heads"]), strict=True):
            base()["load_array"](root, record); live_join(record, live)
            require(record["shape"] == [1, heads, protocol["sequence_rows"], qconfig["head_dim"]] and record["logical_dtype"] == "torch.bfloat16", "actual unequal QKV geometry differs")
            qkv_layouts.append(dict(lane=lane, arm=arm, port=name, capture=live))
        head = base()["load_array"](root, ports["head"])
        if lane != "native":
            head = head.transpose(0, 2, 1, 3)
        raw = (head.view(np.uint32) >> 16).astype("<u2").tobytes(order="C")
        require(list(head.shape) == call["head_return"]["shape"] and hashlib.sha256(raw).hexdigest() == call["head_return"]["content_sha256"]
            and call["head_return"]["dtype"] == "torch.bfloat16" and call["head_return"]["device"] == ports["head"]["device_at_capture"]
            and all(operand["device"] == native_context["cos"]["device_at_capture"] for operand in call["operands"]), "actual SDPA return/full head port differs after declared transpose")
    entries = contract["block_entries"]
    require(len(entries) == 1+len(arms), "actual native component entries differ")
    for arm, entry in zip(("baseline", *arms), entries, strict=True):
        require(entry["arm"] == arm and entry["input"] == native["inputs"][arm] and entry["context"] == native_context, "actual native input/context differs")
        consumed = entry["consumed_context"]
        for name in ("position_ids", "cos", "sin"):
            base()["load_array"](root, native_context[name]); live_join(native_context[name], consumed[name])
        require(consumed["attention_mask"] is None and consumed["past_key_values"] is None and consumed["use_cache"] is False, "actual native cache/mask differs")
        identities.append({name: {key: consumed[name][key] for key in ("version", "object_id", "data_ptr")} for name in ("position_ids", "cos", "sin")})
    require(all(value == identities[0] for value in identities) and native_context == read(root / "source-frontier.json")["context"], "shared actual context identity changed")
    expected = dict(protocol["physical_plan"], native_authorizations=1+len(arms), forecast_module_attempts=0)
    require(contract["expected"] == summary["contract"] == expected and contract["observed"] == summary["counts"]
        and all(summary["counts"][key] == value for key, value in expected.items()) and summary["counts"]["native_module_entries"] == 15*(1+len(arms))
        and summary["counts"]["cached_evaluations"] == summary["counts"]["functional_evaluations"] == len(arms), "actual finite panel census differs")
    return dict(observed=summary["counts"], declared=summary["contract"], actual_SDPA_calls=calls["calls"], QKV_layouts=qkv_layouts, native_entries=entries,
        scope="Seven actual SDPA full-input/head joins and native kwargs. Selected CUDA kernel unprofiled. Excluded-model zero fields are program declarations, not independent profiling.")


def transaction(root, protocol, summary, inventory, source_identity, max_bytes):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx, plain = root / "transaction", base()["helpers"]()[0]["plain"]
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"], expected_event_head=summary["transaction"]["event_head"], max_bytes=max_bytes)
    require(verified["phase"] == "closed" and [read(path)["kind"] for path in sorted((tx / "events").glob("*.json"))] ==
        ["opened", "frontier_published", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and not read(tx / "seal.json")["violations"], "global panel journal differs")
    prediction = plain(load_forecast_tree(tx, read(tx / "forecasts/cohort.json")["predictions"]))
    observation = plain(load_forecast_tree(tx, read(tx / "observations/native-verification.json")["value"]))
    frontier = read(root / "source-frontier.json")
    captured = plain(load_forecast_tree(tx, read(tx / "frontiers/inherited-positive.json")["value"]))
    require(captured == dict(source_arrays=frontier["source_arrays"], context=frontier["context"], scope=row()["FRONTIER"])
        and frontier["frontier"] == summary["frontier"] == row()["FRONTIER"] and summary["genuine_qwen_act"] is False, "architectural inherited frontier differs")
    post = {"timeline.jsonl", "native.json", "native-contract.json", "native-sdpa.json", "sdpa.json", "parameter-identity-after.json", "summary.json", "request.json", "declared-config.json"}
    expected = {name: value for name, value in inventory["files"].items() if not name.startswith(("transaction/", "native/")) and name not in post}
    require(prediction["external_files"] == expected and prediction["inherited_sources"] == frontier["source_arrays"]
        and prediction["response_hashes"] == read(root / "forecast-responses.json")["arms"], "complete pretarget forecast/QKV/source inventory differs")
    arms = [arm["id"] for arm in protocol["arms"]]
    require(prediction["native_targets_at_seal"] == 0 and observation == dict(counts=summary["counts"], arms=["baseline", *arms])
        and all(value == (0 if key.startswith("native_") or key == "forecast_module_attempts" else summary["counts"][key]) for key, value in prediction["counts"].items())
        and set(prediction["counts"]) == set(summary["counts"]), "native work preceded global seal or saved counts differ")
    declaration = read(tx / "seal.json")["declarations"]
    require(declaration["model_identity"] == "original-qwen3-decoder-layer" and declaration["input_identity"] == frontier["source_arrays"]["h"]["content_sha256"]
        and declaration["source_identity"] == base()["digest"](source_identity) and declaration["frontier"] == dict(row()["FRONTIER"], external_files=expected, inherited_sources=frontier["source_arrays"]), "sealed source/model/input/frontier differs")
    journal = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    forecast_pairs = [(arm, lane) for arm in arms for lane in ("cached", "functional")]
    kinds = ["forecast_lane_completed"]*len(forecast_pairs) + ["cohort_globally_sealed", "native_targets_released"] + ["native_component_authorized", "native_component_completed"]*(1+len(arms))
    require([value["kind"] for value in journal] == kinds and [value["number"] for value in journal] == list(range(len(kinds)))
        and [(value["arm"], value["lane"]) for value in journal[:len(forecast_pairs)]] == forecast_pairs
        and all(value["native_module_entries"] == 0 for value in journal[:len(forecast_pairs)+2]), "durable forecast/seal/entry chronology differs")
    tail = journal[len(forecast_pairs)+2:]
    require(all(value["arm"] == arm and value["native_module_entries"] == 15*(index//2+index%2) for index, (value, arm) in enumerate(zip(tail, [arm for arm in ("baseline", *arms) for _ in range(2)], strict=True)))
        and all(value["seal_sha256"] == summary["forecast_seal"] for value in journal[len(forecast_pairs):len(forecast_pairs)+2]), "actual native occurrence/source seal differs")
    return dict(verified=verified, chronology=journal, external_files=len(expected), native_targets_at_seal=0,
        SDPA_receipt_scope="All forecast QKV/head bytes are presealed. Aggregate seven-call metadata sidecar is persisted after targets and checked against those bytes; it is not a presealed call-sidecar claim.")


def producer_helpers():
    return dict(base()["PRODUCER_HELPERS"], **{"qwen_sign_shortcut_campaign.py": base()["WORKER_SHA256"],
        "qwen_row_sign_shortcut.py": row()["ROW_HELPER_SHA256"], "qwen_row_sign_campaign.py": row()["WORKER_SHA256"],
        "qwen_component_experiment.py": COMPONENT_SHA256, "qwen_cached_qkv.py": PREDICTOR_SHA256})


def authenticate(root, expected_manifest, expected_request):
    inventory = row()["wrapper"](root, expected_manifest, expected_request)
    config, request, source, summary = row()["request_envelope"](root, inventory, expected_request)
    expected = producer_helpers()
    require(source["helper_source_sha256"] == config["helper_source_sha256"] == expected and all(request["source_hashes"][name] == digest for name, digest in expected.items())
        and request["source_hashes"]["qwen_cached_qkv_campaign.py"] == WORKER_SHA256, "executed component/predictor/adapter source differs")
    require(config["schema"] == summary["schema"] == "mdb-qwen-cached-qkv-campaign-v1" and source["frontier"] == row()["FRONTIER"]
        and (config["edit_index"], config["horizon"], config["sequence_rows"]) == (18, 19, 512)
        and config["attention_dispatch"] == dict(backend="sdpa", gqa_policy="enable_gqa") and config["qwen_rope_buffer_policy"] ==
        dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="original native CPU initializer; device-only placement"), "source schema/component/profile differs")
    metadata = {}
    require(set(source["metadata"]) == set(PATH_KEYS), "complete source metadata coverage differs")
    for key in PATH_KEYS:
        record = source["metadata"][key]
        require(record == dict(path="metadata/"+key+Path(config[key]).suffix, sha256=config[key+"_sha256"]), "metadata alias/config binding differs")
        path = base()["member"](root, record["path"])
        require(sha(path) == record["sha256"], "retained metadata checksum differs")
        metadata[key] = path
    require(sha(metadata["selection_protocol"]) == PROTOCOL_SHA256 and sha(metadata["qwen_parameter_manifest"]) ==
        "425502121b4091b10124c548c98127785420233b12e95c8f1d972a186026dccf" and sha(metadata["retrieval_script"]) == base()["RETRIEVER_SHA256"], "fixed selection/original recipes/retriever differ")
    protocol = read(metadata["selection_protocol"])
    require(protocol["sequence_rows"] == config["sequence_rows"] and protocol["edit_index"] == config["edit_index"] and protocol["horizon"] == config["horizon"]
        and protocol["comparisons"] == list(CHANNELS), "protocol execution/three-channel basis differs")
    return config, source, summary, inventory, metadata, protocol


def inherited_accounts(root, config, metadata, protocol):
    import numpy as np

    files, request, seal, forecast = (read(metadata[key]) for key in ("parent_files", "parent_request", "parent_seal", "parent_forecast"))
    require(config["input_root"] == protocol["parent_raw_root"] and sha(metadata["parent_files"]) == protocol["parent_raw_manifest_sha256"]
        and sha(metadata["parent_seal"]) == protocol["parent_global_seal_sha256"] and base()["identity"](request) == request["request_sha256"] == protocol["parent_request_sha256"]
        and files["job_id"] == protocol["parent_job"] and files["request_sha256"] == request["request_sha256"], "accepted parent/source closure differs")
    require(seal["schema"] == "mdb-global-forecast-seal-v1" and not seal["violations"] and seal["pre_release_work"]["native_calls"] == 0, "accepted parent global source seal differs")
    external = seal["declarations"]["frontier"]["external_files"]
    names = dict(parent_source="source.json", parent_runtime="runtime.json", parent_parameter_manifest="qwen-parameter-artifacts.json",
        qwen_config="qwen-config.json", qwen_source_binding="qwen-source-binding.json", retrieval_script="retrieve_qwen_weights.py", parent_forecast=f"cases/{protocol['source_case']}/forecast.json")
    require(all(sha(metadata[key]) == files["files"][name]["sha256"] == external[name]["sha256"] for key, name in names.items()), "original parent metadata lineage differs")
    records = protocol["source_arrays"]
    require(set(records) == {"h", *QKV, *PORTS, "input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}, "exact13 positive source whitelist differs")
    for name, record in records.items():
        require(files["files"][record["path"]] == external[record["path"]] == dict(bytes=record["bytes"], sha256=record["sha256"])
            and not any(part in {"native", "minus", "shortcut"} for part in Path(record["path"]).parts), "inherited array is not its authenticated positive/source member")
        if name in forecast["ports"]["plus"] or name == "position_ids":
            require(record == (forecast["position_ids"] if name == "position_ids" else forecast["ports"]["plus"][name]), "inherited actual parent layout descriptor differs")
    values = {name: base()["load_array"](Path(config["input_root"]), record) for name, record in records.items()}
    require(np.all(values["attention_mask"] == 1) and same(values["position_ids"], np.arange(protocol["sequence_rows"], dtype=np.int64)[None])
        and same(values["rope_buffer:model.rotary_emb.inv_freq"], values["rope_buffer:model.rotary_emb.original_inv_freq"])
        and all(records[name]["logical_dtype"] == "torch.float32" for name in records if name.startswith("rope_buffer:")), "own all-attended positions/FP32 RoPE source differs")
    frontier = read(root / "source-frontier.json")
    require(frontier["source_arrays"] == {name: dict(record, inherited=True, root="input_root") for name, record in records.items()}
        and frontier["input_root"] == config["input_root"] and all(frontier[key] == protocol[key] for key in
            ("parent_job", "parent_request_sha256", "parent_raw_manifest_sha256", "parent_global_seal_sha256", "source_case")), "actual inherited source refs/lineage differ")
    return values, dict(job_id=protocol["parent_job"], request_sha256=request["request_sha256"], manifest_sha256=sha(metadata["parent_files"]),
        forecast_seal=sha(metadata["parent_seal"]), arrays=len(records), bytes=sum(record["bytes"] for record in records.values()),
        scope="Only protocol source13 arrays read; accepted parent raw manifest/global seal/source descriptor bytes authenticated. No parent target arrays or recurrence read.")


def source_accounts(root, config, source, metadata):
    before, after = (read(root / name) for name in ("parameter-identity-before.json", "parameter-identity-after.json"))
    require(before == after and source["selected_original_names"] == sorted(before["payloads"]), "final original resident payload/version checks differ")
    qconfig = read(metadata["qwen_config"])
    require(tuple(qconfig[key] for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")) ==
        (2560, 9728, 36, 32, 8, 128) and qconfig.get("attention_bias", False) is False and qconfig.get("attention_dropout", 0.) == 0., "original component assumptions/geometry differ")
    parameters = row()["parameter_accounts"](read(metadata["parent_parameter_manifest"]), read(metadata["qwen_parameter_manifest"]), qconfig,
        before["payloads"], after["payloads"], config["parameter_root"], edit_index=config["edit_index"])
    io, restored = source["parameter_recipe_io"], parameters["selected"]
    require(io["payload_reads"] == io["load_requests"] == 11 and io["cache_hits"] == 0 and io["parameter_payload_bytes"] == restored["payload_bytes"] == 201861632
        and io["checkpoint_files_verified_once"] == len(restored["checkpoint_files"]), "observed one-hydration recipe access differs")
    runtime = read(root / "runtime-contract.json")
    require(runtime["torch"].split("+")[0] == "2.13.0" and runtime["transformers"] == "5.14.1" and set(runtime["sources"]) == {"qwen_native_source_sha256", "sdpa_source_sha256"}, "original runtime identity differs")
    for key, expected in (("qwen_native_source_sha256", row()["NATIVE_SHA256"]), ("sdpa_source_sha256", row()["SDPA_SHA256"])):
        record = runtime["sources"][key]
        require(config[key] == record["sha256"] == expected and Path(record["path"]).stat().st_size == record["bytes"] and sha(record["path"]) == expected, "actual installed native runtime source differs")
    return dict(parameters=parameters, parameter_recipe_io=io, final_resident_parameter_identity=True, runtime=runtime), qconfig


def catalog_contexts(config, source, sources, protocol, report):
    operand = report["census"]["actual_SDPA_calls"][0]["operands"][0]
    return {arm["id"]: dict(model_identity=config["qwen_parameter_manifest_sha256"], source_identity=source["source_hashes"],
        execution=dict(backend=operand["device"].split(":")[0], operand_device=operand["device"], learned_dtype=operand["dtype"],
            runtime_identity=base()["digest"](sources["runtime"]), primitive_policy=sources["runtime"]["flags"], attention_dispatch=config["attention_dispatch"]),
        input_identity=protocol["source_arrays"]["h"]["content_sha256"], frontier="inherited-positioned-QKV-and-H18",
        horizon=protocol["horizon"], arm=arm["id"], windows=arm["windows"]) for arm in protocol["arms"]}


def catalog_accounts(output, report, equations, contexts):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog

    catalog, records, effects = EquationCatalog(output / "catalog/equations"), [], {}
    for arm, context in contexts.items():
        record = catalog.record(law_id="law.qwen.positioned-QKV-row-sign.bf16.v1", route_id="route.qwen.block18.cached-QKV-to-original-attention-tail.v1",
            alias="cmp.qwen.cached-qkv."+arm, binding={key: context[key] for key in ("model_identity", "execution", "input_identity", "frontier", "horizon", "source_identity")},
            conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
            readers=["qwen.block18."+name for name in PORTS], report=equations[arm]["cached-qkv-minus-native"],
            evidence=dict(audit_sha256=sha(output / "audit.json"), status="conditional observation; no validated rule, BF16 algebra proof or independent kernel"))
        require(len(catalog.find(applicable_to=context)) == 1 and not catalog.find(applicable_to={**context, "input_identity": "0"*64}), "conditional catalog lookup differs")
        records.append(dict(arm=arm, fingerprint=record["fingerprint"], context=context))
        fields = report["arms"][arm]["channels"]["cached-qkv-minus-native"]
        effects[record["fingerprint"]] = [dict(consumer_id="qwen.block18."+name, contrast_id=contrast, value=fields[name][term]["whole"]["signed_mean"],
            units="represented coordinate units", provenance=[dict(path="audit.json", sha256=sha(output / "audit.json"))]) for name in PORTS for contrast, term in
            (("native-edited-minus-native-baseline", "right_effect"), ("cached-minus-native-prediction-error", "endpoint"), ("cached-response-minus-native-response-error", "response_error"))]
    observations = ComputationAtlas(output / "atlas").ingest(output / "catalog/equations", [record["fingerprint"] for record in records],
        provenance=dict(run_id=report["job_id"], artifacts=[dict(path="audit.json", sha256=sha(output / "audit.json"))]), signed_effects=effects)
    save(output / "catalog/provenance.json", dict(records=records, actual_CLI_calls=0, independent_occurrences=False, rule_status="observation; no validated rule"))
    save(output / "atlas/provenance.json", dict(observations=[record["fingerprint"] for record in observations],
        scope="Native edit effects and prediction-error effects have separate contrast IDs; no family induction or rule validation."))


def audit(root, output, *, expected_manifest, expected_request):
    root, output = Path(root), Path(output)
    config, source, summary, inventory, metadata, protocol = authenticate(root, expected_manifest, expected_request)
    inherited, parent = inherited_accounts(root, config, metadata, protocol)
    tx = transaction(root, protocol, summary, inventory, dict(source["helper_source_sha256"], qwen_native=row()["NATIVE_SHA256"], sdpa_native=row()["SDPA_SHA256"]), config["forecast_transaction_bytes"])
    sources, qconfig = source_accounts(root, config, source, metadata)
    report, equations = numerical_accounts(root, inherited, protocol)
    report.update(schema="mdb-qwen-cached-qkv-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request, native_manifest_sha256=expected_manifest,
        worker_sha256=WORKER_SHA256, audit_sha256=sha(__file__), source=sources, parent=parent, transaction=tx,
        census=operand_accounts(root, protocol, summary, qconfig), layout_restoration=layout_accounts(root, read(root / "source-frontier.json")["source_arrays"]),
        mechanics_status="authenticated", instrument_status="complete saved-coordinate reductions and operand joins", trend_status="observations; retain all misses", terminal_status="not certified",
        limitations=["Inherited own-functional positioned QKV/H18 conditional frontier, not a new raw prefix or genuine Qwen Act.",
            "Transformed/direct/native QKV and canonical heads compare declared logical bases; capture layout and sign-zero differences remain separate.",
            "Protocol windows, three channels, positive replay, QKV and source-sign accounts overlap; no independent occurrence or whole-arm accuracy claim.",
            "Seven actual SDPA calls and original native operands do not identify a selected CUDA kernel or establish an independent implementation."])
    output.mkdir(parents=True, exist_ok=False); save(output / "audit.json", report)
    catalog_accounts(output, report, equations, catalog_contexts(config, source, sources, protocol, report))
    save(output / "coverage.json", dict(arms={arm: value["coverage"] for arm, value in report["arms"].items()},
        model_calls=0, learned_projections=0, tokenizer_calls=0, primary_and_windows_separate=True, dependent_accounts=True))
    return report, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(base()["identity"](request) == request["request_sha256"] and request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request/config differs")
    for name, expected in request["source_hashes"].items():
        require(sha(name) == expected, "staged saved-reader source differs")
    started, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    report, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime-contract.json", "files.json", "source-frontier.json", "forecast-responses.json",
        "forecast.json", "native.json", "layout-restoration.json", "parameter-identity-before.json", "parameter-identity-after.json", "timeline.jsonl", "native-contract.json", "native-sdpa.json", "sdpa.json"):
        shutil.copyfile(root / name, output / ("source-"+name))
    shutil.copytree(root / "metadata", output / "source-metadata")
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", row()["external_evidence"](root, inventory))
    save(output / "request.json", request)
    save(output / "summary.json", dict(schema="mdb-qwen-cached-qkv-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=report["job_id"], catalog_records=len(report["arms"]), atlas_observations=len(report["arms"]), native_component_observations=1+len(report["arms"]), model_calls=0, seconds=time.monotonic()-started))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY "+json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("root", "output", "expected-manifest", "expected-request"):
            parser.add_argument("--"+name, required=True)
        args = parser.parse_args()
        audit(args.root, args.output, expected_manifest=args.expected_manifest, expected_request=args.expected_request)
