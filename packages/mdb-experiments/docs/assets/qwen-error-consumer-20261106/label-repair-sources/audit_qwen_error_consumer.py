"""Saved-only two-world block19 consumer accounts.

Only authenticated saved arrays, public transaction verification and scalar
reductions run here. No producer, Torch, tokenizer or neural operation imports.
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
WORKER_SHA256 = "ad82b86dd837a1f2bc9134f2c5bda3dcb097e8f8583e7849fcfbe8d5c49b229b"
COMPONENT_SHA256 = "c93324ee76156f2fba16f70d7cef78b7accbf6020f0c899cb04d856c348b5762"
PROTOCOL_SHA256 = "4e62bc3fa8adcd059ac12f775effa0010752e3c5e0cca2a2ebc4702ecef7f847"
PINS = {
    "audit_qwen_cached_qkv.py": "b473d178fd9825009c0c039dc23ab0f2208309f2d34ccc2d784ee5b5710385eb",
    "audit_qwen_row_sign.py": "d995c3c5ca3c194a3ce87bfca58e9352d251246d798e21c6e04a7938bd9b5838",
    "audit_qwen_sign_shortcut.py": "d33613bbb1d71ee3b2007fb007e5266acaf141994091264bed50d27342237e2c",
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
PORTS = ("head", "projection", "R", "M", "Hnext")
WORLDS = ("full-input", "cheap-input")
CONTEXT = ("input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq")
PATH_KEYS = ("selection_protocol", "qwen_parameter_manifest", "parent_parameter_manifest", "qwen_config", "qwen_source_binding", "retrieval_script",
    "world_files", "world_seal", "world_request", "world_forecast", "world_shortcut", "context_files", "context_seal", "context_request")
FRONTIER = dict(kind="architectural", origin="inherited-functional-state", scope="conditional block19 hidden-state interface",
    raw_input_prediction=False, genuine_record_prefix_capture=False, genuine_qwen_act=False)
SUPPORT = dict(status="not-applicable", reason="explicit saved input worlds; no row-sign support claim")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


@lru_cache(maxsize=1)
def saved():
    for name, expected in PINS.items():
        require(sha(HERE / name) == expected, "frozen saved-only helper differs: " + name)
    return runpy.run_path(str(HERE / "audit_qwen_cached_qkv.py"))


def base():
    return saved()["base"]()


def row():
    return saved()["row"]()


def metric(left, right):
    return saved()["metric"](left, right)


def windowed(left, right, windows):
    return saved()["windowed"](left, right, windows)


def same(left, right):
    return saved()["same"](left, right)


def signed_descriptor(left, right, orientation):
    import numpy as np

    delta = left.astype(np.float64) - right.astype(np.float64)
    return dict(shape=list(delta.shape), elements=delta.size, dtype="float64", orientation=orientation,
        sha256=hashlib.sha256(delta.astype("<f8").tobytes(order="C")).hexdigest())


def scalar(value):
    import numpy as np

    value = value.astype(np.float64)
    absolute = np.abs(value)
    return dict(elements=value.size, squared_norm=float(np.sum(value * value)), rms=float(np.sqrt(np.mean(value * value))),
        signed_mean=float(np.mean(value)), max_abs=float(absolute.max()), p95_abs=float(np.percentile(absolute, 95)),
        p99_abs=float(np.percentile(absolute, 99)), changed_values=int(np.count_nonzero(value)))


def vector_pair(left, right):
    import numpy as np

    left, right = left.astype(np.float64), right.astype(np.float64)
    require(left.shape == right.shape, "shared residual coordinate geometry differs")
    dot = float(np.sum(left * right)); le = float(np.sum(left * left)); re = float(np.sum(right * right))
    return dict(dot=dot, signed_cross_term=2 * dot, cosine=None if le * re == 0 else dot / (le * re)**.5,
        left=scalar(left), right=scalar(right), rms_ratio_left_to_right=None if re == 0 else (le / re)**.5,
        difference=scalar(left-right), sum=scalar(left+right), denominator="supplied right vector norm; zero is undefined")


def energy_account(total, terms):
    import numpy as np

    names = list(terms)
    vectors = {name: value.astype(np.float64) for name, value in terms.items()}
    cross = [{"left": a, "right": b, **vector_pair(vectors[a], vectors[b])} for index, a in enumerate(names) for b in names[index+1:]]
    total_energy = float(np.sum(total.astype(np.float64)**2))
    expansion = sum(float(np.sum(value**2)) for value in vectors.values()) + sum(value["signed_cross_term"] for value in cross)
    return dict(total=scalar(total), terms={name: scalar(value) for name, value in vectors.items()}, pairwise_cross_terms=cross,
        expanded_squared_norm=expansion, energy_identity_residual=total_energy-expansion,
        vector_identity_residual=scalar(total-sum(vectors.values())),
        scope="FP64 saved-vector addition identity; signed cross terms are descriptive, not causal shares.")


def residual_accounts(input_delta, ports, windows):
    import numpy as np

    result = []
    for bounds in ([0, input_delta.shape[1]], *windows):
        a, b = bounds; h = input_delta[:, a:b].astype(np.float64)
        d = {name: value[:, a:b].astype(np.float64) for name, value in ports.items() if name != "head"}
        er = d["R"] - h - d["projection"]; ey = d["Hnext"] - d["R"] - d["M"]
        result.append(dict(bounds=list(bounds), input_H19_to_output_H20=vector_pair(d["Hnext"], h),
            residual=energy_account(d["R"], dict(input_H19=h, projection=d["projection"], finite_addition_defect=er)),
            output=energy_account(d["Hnext"], dict(R=d["R"], M=d["M"], finite_addition_defect=ey)),
            R_M=vector_pair(d["R"], d["M"])))
    return dict(whole=result[0], windows=result[1:],
        orientation="cheap-input-minus-full-input", coordinate_space="Common residual feature basis only; head width is excluded from gain accounts.")


def numerical_accounts(root, inherited, protocol):
    import numpy as np

    from mdb.equation_comparison import compare_equations

    forecast, native, responses = (read(root / name) for name in ("forecast.json", "native.json", "forecast-responses.json"))
    require(list(protocol["world_order"]) == list(WORLDS) and set(forecast["worlds"]) == set(native["worlds"]) == set(WORLDS)
        and protocol["baseline_world"] == "full-input" and native["head_to_canonical"] == [0, 2, 1, 3], "explicit two-world/native-head basis differs")
    response = protocol["response"]
    require(response == dict(left_world="cheap-input", right_world="full-input", orientation="cheap-input-minus-full-input", ports=list(PORTS))
        and {key: responses[key] for key in (*response, "baseline_world")} == dict(response, baseline_world="full-input"), "signed response orientation differs")
    windows = protocol["windows"]
    lanes = {lane: {world: saved()["load_ports"](root, value["worlds"][world]["ports"], native=lane == "native")
        for world in WORLDS} for lane, value in (("functional", forecast), ("native", native))}
    references = read(root / "source-frontier.json")["worlds"]
    endpoints, equations = {}, {}
    for world in WORLDS:
        require(forecast["worlds"][world]["input_reference"] == native["worlds"][world]["input_reference"] == references[world]["input_reference"], "world reference differs")
        require(same(lanes["functional"][world]["h"], inherited[world]), "own input world bytes changed at functional consumer")
        saved()["live_join"](references[world]["input_reference"], forecast["worlds"][world]["consumed_input_descriptor"])
        saved()["live_join"](references[world]["input_reference"], native["worlds"][world]["actual_input_descriptor"])
        f, n = lanes["functional"][world], lanes["native"][world]
        count = sum(f[name].size for name in PORTS)
        require(count <= protocol.get("max_comparison_elements", count), "declared complete comparison budget too small")
        equations[world] = compare_equations({name: f[name] for name in PORTS}, {name: n[name] for name in PORTS}, max_elements=count)
        endpoints[world] = dict(fields={name: windowed(f[name], n[name], windows) for name in PORTS},
            first_numeric_difference=next((name for name in PORTS if not np.array_equal(f[name], n[name])), None),
            first_bit_difference=next((name for name in PORTS if not same(f[name], n[name])), None), comparison_elements=count)
    deltas = {lane: {name: values["cheap-input"][name].astype(np.float64)-values["full-input"][name] for name in PORTS}
        for lane, values in lanes.items()}
    require(set(responses["fields"]) == set(PORTS) and all(responses["fields"][name] == signed_descriptor(
        lanes["functional"]["cheap-input"][name], lanes["functional"]["full-input"][name], response["orientation"]) for name in PORTS), "complete pretarget signed response hash differs")
    input_delta = inherited["cheap-input"].astype(np.float64)-inherited["full-input"]
    field_response = {name: dict(error=windowed(deltas["functional"][name], deltas["native"][name], windows),
        functional=windowed(lanes["functional"]["cheap-input"][name], lanes["functional"]["full-input"][name], windows),
        native=windowed(lanes["native"]["cheap-input"][name], lanes["native"]["full-input"][name], windows)) for name in PORTS}
    return dict(worlds=endpoints, responses=field_response, input_difference=windowed(inherited["cheap-input"], inherited["full-input"], windows),
        residual_accounts={lane: residual_accounts(input_delta, fields, windows) for lane, fields in deltas.items()},
        response_commitments=responses, coverage=dict(primary_endpoint_fields=2*len(PORTS), primary_response_fields=len(PORTS),
            dependent_endpoint_windows=2*len(PORTS)*len(windows), dependent_response_windows=len(PORTS)*len(windows)),
        scope="Complete saved-coordinate accounts; prefix equality, magnitude, numeric/raw-bit/signed-zero misses and finite defects are observations, never scientific gates."), equations


def producer_helpers():
    return dict(base()["PRODUCER_HELPERS"], **{
        "qwen_sign_shortcut_campaign.py": base()["WORKER_SHA256"], "qwen_row_sign_shortcut.py": row()["ROW_HELPER_SHA256"],
        "qwen_row_sign_campaign.py": row()["WORKER_SHA256"],
        "qwen_component_experiment.py": "81131b179f1a8a5ea52011be3e450e5901255f350d89ff11c053902c16a28580",
        "qwen_component_experiment_v2.py": COMPONENT_SHA256})


def authenticate(root, expected_manifest, expected_request):
    inventory = row()["wrapper"](root, expected_manifest, expected_request)
    config, request, source, summary = row()["request_envelope"](root, inventory, expected_request)
    helpers = producer_helpers()
    require(source["helper_source_sha256"] == config["helper_source_sha256"] == helpers and
        all(request["source_hashes"][name] == value for name, value in helpers.items()) and
        request["source_hashes"]["qwen_error_consumer_campaign.py"] == WORKER_SHA256, "executed producer/source cohort differs")
    require(config["schema"] == "mdb-qwen-error-consumer-campaign-v1" and summary["schema"] == "mdb-qwen-explicit-worlds-v2"
        and (config["component_index"], config["horizon"], config["sequence_rows"]) == (19, 20, 512)
        and source["frontier"] == FRONTIER and config["attention_dispatch"] == dict(backend="sdpa", gqa_policy="enable_gqa")
        and config["qwen_rope_buffer_policy"] == dict(init_device="cpu", init_dtype="float32", placement_dtype="float32",
            source="original native CPU initializer; device-only placement"), "fixed world/component/source profile differs")
    require(set(source["metadata"]) == set(PATH_KEYS), "complete retained metadata coverage differs")
    metadata = {}
    for key in PATH_KEYS:
        value = source["metadata"][key]
        require(value == dict(path="metadata/"+key+Path(config[key]).suffix, sha256=config[key+"_sha256"]), "metadata alias differs: " + key)
        metadata[key] = base()["member"](root, value["path"])
        require(sha(metadata[key]) == value["sha256"], "metadata source bytes differ: " + key)
    require(sha(metadata["selection_protocol"]) == PROTOCOL_SHA256 and sha(metadata["qwen_parameter_manifest"]) ==
        "69a9189b00413ccb01943718a68eba2865181110aa30b04f17668a8e2e1ffb8c" and sha(metadata["retrieval_script"]) == base()["RETRIEVER_SHA256"], "fixed protocol/recipes/retriever differs")
    protocol = read(metadata["selection_protocol"])
    protocol_frontier = {**{key: value for key, value in FRONTIER.items() if key != "genuine_qwen_act"}, "genuine_Qwen_Act": False}
    require(protocol["component_index"] == config["component_index"] and protocol["horizon"] == config["horizon"]
        and protocol["sequence_rows"] == config["sequence_rows"] and protocol["frontier"] == protocol_frontier
        and protocol["max_comparison_elements"] == 7340032, "fixed protocol/program/element budget differs")
    return config, source, summary, inventory, metadata, protocol


def inherited_accounts(root, config, metadata, protocol):
    import numpy as np

    manifests, externals = {}, {}
    for alias in ("world", "context"):
        declared = protocol["roots"][alias]
        files, seal, request = (read(metadata[alias+"_"+name]) for name in ("files", "seal", "request"))
        require(config[alias+"_root"] == declared["root"] and sha(metadata[alias+"_files"]) == declared["raw_manifest_sha256"]
            and sha(metadata[alias+"_seal"]) == declared["global_seal_sha256"] and
            base()["identity"](request) == request["request_sha256"] == files["request_sha256"] == declared["request_sha256"]
            and files["job_id"] == declared["job_id"], "original inherited root identity differs: " + alias)
        require(seal["schema"] == "mdb-global-forecast-seal-v1" and not seal["violations"] and seal["pre_release_work"]["native_calls"] == 0,
            "inherited source was not in original pretarget seal")
        manifests[alias], externals[alias] = files["files"], seal["declarations"]["frontier"]["external_files"]
    context_metadata = dict(parent_parameter_manifest="qwen-parameter-artifacts.json", qwen_config="qwen-config.json",
        qwen_source_binding="qwen-source-binding.json", retrieval_script="retrieve_qwen_weights.py")
    for key, relative in context_metadata.items():
        require(sha(metadata[key]) == manifests["context"][relative]["sha256"] == externals["context"][relative]["sha256"], "original metadata lineage differs: " + key)
    references = {}
    for world in WORLDS:
        record = protocol["world_sources"][world]
        key, relative = ("world_forecast", "forecast.json") if world == "full-input" else ("world_shortcut", "shortcut.json")
        sidecar = read(metadata[key])
        require(sidecar["arms"]["prefix-negative"]["ports"]["Hnext"] == record and
            manifests["world"][relative] == externals["world"][relative] == dict(bytes=metadata[key].stat().st_size, sha256=sha(metadata[key])), "world H19 is not the sealed original forecast member")
        references[world] = dict(record, root="world", inherited=True)
    require(set(protocol["shared_context_sources"]) == set(CONTEXT), "exact five shared context members differ")
    references.update({name: dict(record, root="context", inherited=True) for name, record in protocol["shared_context_sources"].items()})
    values = {}
    for name, record in references.items():
        alias = record["root"]; binding = dict(bytes=record["bytes"], sha256=record["sha256"])
        require(manifests[alias].get(record["path"]) == externals[alias].get(record["path"]) == binding,
            "inherited input member absent from parent seal/inventory")
        values[name] = base()["load_array"](Path(config[alias+"_root"]), record)
    require(np.all(values["attention_mask"] == 1) and same(values["position_ids"], np.arange(protocol["sequence_rows"], dtype=np.int64)[None])
        and same(values[CONTEXT[-2]], values[CONTEXT[-1]]) and all(references[name]["logical_dtype"] == "torch.float32" for name in CONTEXT[-2:]), "all-attended positions/FP32 RoPE contract differs")
    frontier = read(root / "source-frontier.json")
    require(frontier["roots"] == protocol["roots"] and frontier["worlds"] == {name: dict(input_reference=references[name]) for name in WORLDS}
        and frontier["shared_context"] == {name: references[name] for name in CONTEXT} and frontier["frontier"] == FRONTIER and
        frontier["support"] == SUPPORT, "two explicit source roots/world/context frontier differs")
    saved()["layout_accounts"](root, references)
    return values, dict(roots=protocol["roots"], arrays=7, bytes=sum(value["bytes"] for value in references.values()), references=references,
        scope="Exactly two globally sealed forecast H19 members plus five shared context members. No native target, old QKV or borrowed intermediate is read.")


def source_accounts(root, config, source, metadata):
    before, after = (read(root / name) for name in ("parameter-identity-before.json", "parameter-identity-after.json"))
    require(before == after and source["selected_original_names"] == sorted(before["payloads"]), "resident parameter payload/version identity differs")
    qconfig = read(metadata["qwen_config"])
    require(tuple(qconfig[key] for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")) ==
        (2560, 9728, 36, 32, 8, 128) and not qconfig.get("attention_bias", False) and qconfig.get("attention_dropout", 0.) == 0., "original geometry/source domain differs")
    parameters = row()["parameter_accounts"](read(metadata["parent_parameter_manifest"]), read(metadata["qwen_parameter_manifest"]), qconfig,
        before["payloads"], after["payloads"], config["parameter_root"], edit_index=19)
    io = source["parameter_recipe_io"]
    require(io["payload_reads"] == io["load_requests"] == 11 and io["cache_hits"] == 0 and io["parameter_payload_bytes"] == parameters["selected"]["payload_bytes"] == 201861632
        and io["checkpoint_files_verified_once"] == len(parameters["selected"]["checkpoint_files"]), "single source hydration census differs")
    runtime = read(root / "runtime-contract.json")
    require(runtime["torch"].split("+")[0] == "2.13.0" and runtime["transformers"] == "5.14.1"
        and set(runtime["sources"]) == {"qwen_native_source_sha256", "sdpa_source_sha256"}, "source runtime identity differs")
    for key, expected in (("qwen_native_source_sha256", row()["NATIVE_SHA256"]), ("sdpa_source_sha256", row()["SDPA_SHA256"])):
        record = runtime["sources"][key]
        require(config[key] == record["sha256"] == expected and Path(record["path"]).stat().st_size == record["bytes"] and sha(record["path"]) == expected,
            "installed native source differs")
    runtime_profile(runtime)
    return dict(parameters=parameters, parameter_recipe_io=io, runtime=runtime, final_resident_parameter_identity=True), qconfig


def runtime_profile(runtime):
    expected = dict(cuda_allow_bf16_reduced_precision_reduction=True, cuda_allow_tf32=False, cudnn_allow_tf32=False,
        cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True, float32_matmul_precision="highest", num_threads=4)
    require(runtime["flags"] == expected, "actual runtime primitive policy differs")
    return dict(flags=expected, authority="recorded original source/profile; selected kernels and hardware not independently profiled")


def operand_accounts(root, inherited, protocol, summary, qconfig):
    import numpy as np

    forecast, native, calls, contract = (read(root / name) for name in ("forecast.json", "native.json", "sdpa.json", "native-contract.json"))
    context = native["context"]
    require(context == read(root / "source-frontier.json")["context"] and context["attention_mask"] is None and context["past_key_values"] is None
        and context["use_cache"] is False and context["is_causal"] is True and context["enable_gqa"] is True, "actual shared native context differs")
    require(same(base()["load_array"](root, context["position_ids"]), inherited["position_ids"]), "own inherited positions changed")
    for name in ("cos", "sin"):
        value = base()["load_array"](root, context[name])
        require(value.shape == (1, protocol["sequence_rows"], qconfig["head_dim"]) and context[name]["logical_dtype"] == "torch.bfloat16", "regenerated rotary operand geometry differs")
    order = [(lane, world) for lane in ("functional", "native") for world in WORLDS]
    require([(call["lane"], call["arm"]) for call in calls["calls"]] == order, "exactly four ordered SDPA calls required")
    for (lane, world), call in zip(order, calls["calls"], strict=True):
        ports = (forecast if lane == "functional" else native)["worlds"][world]["ports"]
        require(call["QKV"] == [ports[name] for name in ("q", "k", "v")] and call["attention_mask"] is None and call["is_causal"] is True
            and call["enable_gqa"] is True and call["dropout_p"] == 0. and call["scale"] == qconfig["head_dim"]**-.5, "actual SDPA source/profile differs")
        for name, live, heads in zip(("q", "k", "v"), call["operands"], (qconfig["num_attention_heads"], qconfig["num_key_value_heads"], qconfig["num_key_value_heads"]), strict=True):
            record = ports[name]; base()["load_array"](root, record); saved()["live_join"](record, live)
            require(record["shape"] == [1, heads, protocol["sequence_rows"], qconfig["head_dim"]] and record["logical_dtype"] == "torch.bfloat16", "unequal QKV head geometry differs")
        head = base()["load_array"](root, ports["head"])
        if lane == "functional":
            head = head.transpose(0, 2, 1, 3)
        bits = (head.view(np.uint32) >> 16).astype("<u2")
        live = call["head_return"]
        stride = ports["head"]["stride"]
        expected_stride = [stride[0], stride[2], stride[1], stride[3]] if lane == "functional" else stride
        require(live["shape"] == list(head.shape) and live["content_sha256"] == hashlib.sha256(bits.tobytes(order="C")).hexdigest()
            and live["dtype"] == "torch.bfloat16" and live["device"] == ports["head"]["device_at_capture"]
            and live["stride"] == expected_stride, "actual SDPA head/declared transpose differs")
    require(len(contract["block_entries"]) == 2, "exactly two native worlds and no baseline required")
    identities = []
    for world, entry in zip(WORLDS, contract["block_entries"], strict=True):
        reference = read(root / "source-frontier.json")["worlds"][world]["input_reference"]
        require(entry["world"] == world and entry["input_reference"] == reference and entry["input"] == native["worlds"][world]["actual_input_descriptor"], "actual native world/input receipt differs")
        saved()["live_join"](reference, entry["input"])
        require(entry["attention_mask"] is None and entry["past_key_values"] is None and entry["use_cache"] is False, "native entry cache/mask differs")
        for name in ("position_ids", "cos", "sin"):
            saved()["live_join"](context[name], entry["consumed_context"][name])
        identities.append({name: {key: entry["consumed_context"][name][key] for key in ("version", "object_id", "data_ptr")} for name in ("position_ids", "cos", "sin")})
    require(identities[0] == identities[1], "actual shared context objects changed between worlds")
    expected = protocol["physical_plan"]
    require(contract["expected"] == summary["contract"] == expected and contract["observed"] == summary["counts"] == expected
        and expected["native_module_entries"] == 30 and expected["native_block_calls"] == expected["native_authorizations"] == 2
        and expected["forecast_module_attempts"] == 0 and contract["projection_provenance"] == "actual passive o_proj output hook"
        and contract["R_provenance"] == "actual post_attention_layernorm input hook", "physical census/actual projection provenance differs")
    return dict(observed=summary["counts"], declared=expected, actual_SDPA_calls=calls["calls"], native_entries=contract["block_entries"],
        scope="Two functional and two native SDPA occurrences. Head logical bases align only by declared transpose; native projections are captured, not R-minus-input reconstructions. Selected kernels unprofiled.")


def transaction(root, protocol, summary, inventory, source_identity, max_bytes):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx = root / "transaction"; plain = base()["helpers"]()[0]["plain"]
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"], expected_event_head=summary["transaction"]["event_head"], max_bytes=max_bytes)
    require(verified["phase"] == "closed" and [read(path)["kind"] for path in sorted((tx / "events").glob("*.json"))] ==
        ["opened", "frontier_published", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and not read(tx / "seal.json")["violations"], "closed prospective transaction event chain differs")
    prediction = plain(load_forecast_tree(tx, read(tx / "forecasts/cohort.json")["predictions"]))
    observation = plain(load_forecast_tree(tx, read(tx / "observations/native-verification.json")["value"]))
    frontier = read(root / "source-frontier.json")
    captured = plain(load_forecast_tree(tx, read(tx / "frontiers/inherited-worlds.json")["value"]))
    refs = {**{name: value["input_reference"] for name, value in frontier["worlds"].items()}, **frontier["shared_context"]}
    require(captured == dict(worlds={name: refs[name] for name in WORLDS}, context=frontier["context"], scope=FRONTIER)
        and frontier["frontier"] == summary["frontier"] == FRONTIER and frontier["support"] == summary["support"] == SUPPORT, "sealed inherited two-world frontier differs")
    post = {"timeline.jsonl", "native.json", "native-contract.json", "sdpa.json", "addition-diagnostics.json", "parameter-identity-after.json", "summary.json", "request.json", "declared-config.json"}
    expected = {name: value for name, value in inventory["files"].items() if not name.startswith(("transaction/", "native/")) and name not in post}
    require(prediction["external_files"] == expected and prediction["inherited_sources"] == refs and prediction["response_hashes"] ==
        read(root / "forecast-responses.json")["fields"] and prediction["native_targets_at_seal"] == 0, "complete pretarget own-future/source/response inventory differs")
    require(observation == dict(counts=summary["counts"], worlds=list(WORLDS)) and set(prediction["counts"]) == set(summary["counts"])
        and all(value == (0 if key.startswith("native_") or key == "forecast_module_attempts" else summary["counts"][key]) for key, value in prediction["counts"].items()), "native target work preceded the cohort seal")
    declaration = read(tx / "seal.json")["declarations"]
    input_identity = base()["digest"]({name: refs[name]["content_sha256"] for name in WORLDS})
    require(declaration["model_identity"] == "original-qwen3-decoder-layer-19" and declaration["input_identity"] == input_identity
        and declaration["source_identity"] == base()["digest"](source_identity) and declaration["frontier"] == dict(FRONTIER, external_files=expected, inherited_sources=refs), "sealed model/source/input identity differs")
    journal = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    kinds = ["functional_world_completed"]*2 + ["cohort_globally_sealed", "native_targets_released"] + ["native_component_authorized", "native_component_completed"]*2
    require([value["kind"] for value in journal] == kinds and [value["number"] for value in journal] == list(range(8))
        and [value["native_module_entries"] for value in journal] == [0, 0, 0, 0, 0, 15, 15, 30]
        and [value["world"] for value in journal[:2]] == list(WORLDS) and [value["world"] for value in journal[4:]] == [name for name in WORLDS for _ in range(2)]
        and journal[2]["seal_sha256"] == journal[3]["seal_sha256"] == summary["forecast_seal"] and journal[2]["native_targets_at_seal"] == 0, "actual global seal/native occurrence chronology differs")
    return dict(verified=verified, chronology=journal, external_files=len(expected), inherited_arrays=7, native_targets_at_seal=0,
        scope="All complete functional outputs and FP64 response hashes are presealed; aggregate SDPA receipts are posttarget sidecars joined to the presealed ports.")


def catalog_contexts(config, source, sources, protocol, report):
    operand = report["census"]["actual_SDPA_calls"][0]["operands"][0]
    return {world: dict(model_identity=config["qwen_parameter_manifest_sha256"], source_identity=source["source_hashes"],
        execution=dict(backend=operand["device"].split(":")[0], operand_device=operand["device"], learned_dtype=operand["dtype"],
            runtime_identity=base()["digest"](sources["runtime"]), primitive_policy=sources["runtime"]["flags"], attention_dispatch=config["attention_dispatch"]),
        input_identity=protocol["world_sources"][world]["content_sha256"], context_identity=base()["digest"](protocol["shared_context_sources"]),
        frontier="two-inherited-sealed-forecast-H19-worlds", horizon=20, world=world) for world in WORLDS}


def catalog_accounts(output, report, equations, contexts):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog

    catalog = EquationCatalog(output / "catalog/equations"); records, effects = [], {}
    for world, context in contexts.items():
        record = catalog.record(law_id="law.qwen.complete-component-primitive-graph.bf16.v1", route_id="route.qwen.block19.own-H19-to-five-ports.v1",
            alias="cmp.qwen.error-consumer."+world, binding={key: context[key] for key in ("model_identity", "execution", "input_identity", "frontier", "horizon", "source_identity")},
            conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
            readers=["qwen.block19."+name for name in PORTS], report=equations[world],
            evidence=dict(audit_sha256=sha(output / "audit.json"), status="conditional dependent observation; no validated rule or independent kernel claim"))
        require(len(catalog.find(applicable_to=context)) == 1 and not catalog.find(applicable_to={**context, "horizon": 21}), "catalog exact-context/wrong-clock lookup differs")
        records.append(dict(world=world, fingerprint=record["fingerprint"], context=context))
        effects[record["fingerprint"]] = [dict(consumer_id="qwen.block19."+name, contrast_id="functional-minus-native-endpoint-error",
            value=report["worlds"][world]["fields"][name]["whole"]["signed_mean"], units="represented coordinate units",
            provenance=[dict(path="audit.json", sha256=sha(output / "audit.json"))]) for name in PORTS]
    observations = ComputationAtlas(output / "atlas").ingest(output / "catalog/equations", [value["fingerprint"] for value in records],
        provenance=dict(run_id=report["job_id"], artifacts=[dict(path="audit.json", sha256=sha(output / "audit.json"))]), signed_effects=effects)
    save(output / "catalog/provenance.json", dict(records=records, independent_occurrences=False, rule_status="observation; no family induction"))
    save(output / "atlas/provenance.json", dict(observations=[value["fingerprint"] for value in observations]))


def audit(root, output, *, expected_manifest, expected_request):
    root, output = Path(root), Path(output)
    config, source, summary, inventory, metadata, protocol = authenticate(root, expected_manifest, expected_request)
    inherited, parents = inherited_accounts(root, config, metadata, protocol)
    tx = transaction(root, protocol, summary, inventory, dict(source["helper_source_sha256"], qwen_native=row()["NATIVE_SHA256"], sdpa_native=row()["SDPA_SHA256"]), config["forecast_transaction_bytes"])
    sources, qconfig = source_accounts(root, config, source, metadata)
    report, equations = numerical_accounts(root, inherited, protocol)
    report.update(schema="mdb-qwen-error-consumer-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request, native_manifest_sha256=expected_manifest,
        worker_sha256=WORKER_SHA256, audit_sha256=sha(__file__), source=sources, parents=parents, transaction=tx,
        census=operand_accounts(root, inherited, protocol, summary, qconfig), mechanics_status="authenticated", instrument_status="complete saved-coordinate accounts",
        trend_status="retain all observations and misses", terminal_status="not certified", limitations=[
            "Two inherited own-functional H19 inputs, not a fresh prompt, Qwen Act, image or end-to-end prefix forecast.",
            "Five ports and their prefix/suffix windows overlap; numeric, raw-bit, signed-zero and layout accounts are distinct.",
            "R and projection are actual hooks; residual-space finite-addition defects/cross terms are descriptive, not causal shares or intrinsic gain.",
            "Source kernels/primitives are shared. SDPA receipts bind declared arguments; selected CUDA kernels are unprofiled."])
    output.mkdir(parents=True, exist_ok=False); save(output / "audit.json", report)
    catalog_accounts(output, report, equations, catalog_contexts(config, source, sources, protocol, report))
    save(output / "coverage.json", dict(**report["coverage"], model_calls=0, tokenizer_calls=0, learned_projections=0,
        primary_and_windows_separate=True, independent_occurrences=False))
    return report, inventory


def external_evidence(root, inventory, alias):
    frontier = read(root / "source-frontier.json")
    references = ({value["input_reference"]["path"]: value["input_reference"]["sha256"] for value in frontier["worlds"].values()} if alias == "world" else
        {value["path"]: value["sha256"] for value in frontier["shared_context"].values()})
    return dict(host="beast", roots=dict(native=str(root), reader=frontier["roots"][alias]["root"]),
        manifests=dict(native={name: value["sha256"] for name, value in inventory["files"].items()}, reader=references),
        alias_roles=dict(native="Current component raw output", reader="Authenticated inherited "+alias+" source root; use --manifest for this root"))


COPY_FILES = ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime-contract.json", "files.json", "source-frontier.json",
    "forecast-responses.json", "forecast.json", "native.json", "layout-restoration.json", "parameter-identity-before.json", "parameter-identity-after.json", "timeline.jsonl",
    "native-contract.json", "sdpa.json", "addition-diagnostics.json")


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(base()["identity"](request) == request["request_sha256"] and request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request/config differs")
    for name, expected in request["source_hashes"].items():
        require(sha(name) == expected, "staged saved-only reader source differs")
    started, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    report, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in COPY_FILES:
        shutil.copyfile(root / name, output / ("source-"+name))
    shutil.copytree(root / "metadata", output / "source-metadata")
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", external_evidence(root, inventory, "world"))
    save(output / "external-evidence-context.json", external_evidence(root, inventory, "context"))
    save(output / "request.json", request)
    save(output / "summary.json", dict(schema="mdb-qwen-error-consumer-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=report["job_id"], worlds=2, catalog_records=2, atlas_observations=2, native_component_observations=2, model_calls=0, seconds=time.monotonic()-started))
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
