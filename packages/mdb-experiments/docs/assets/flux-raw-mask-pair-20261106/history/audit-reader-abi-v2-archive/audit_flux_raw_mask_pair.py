"""Saved-only accounts for a fixed two-prompt mask/dispatch coverage panel.

No learned recurrence or CUDA kernel is executed. The frozen single-case
auditor supplies pure array, lineage, recipe and consumer accounts only; its
single-prompt authentication and identities are deliberately not reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "3ee62e9174186935b32ff6e9e33e18e46aa978c4442b19f02982919e63bf3912"
PINS = {
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_remaining.py": "5027875c03f9ceb3157b4b9d9632366ff5c9b1c4beb429b8b73d79aad0db6b08",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
SHORT = ("A cobalt blue cube to the right of a matte orange ring on a gray tabletop. "
    "Two separate objects, simple studio product photograph.")
CASES = {"short": SHORT, "long": SHORT + 128 * " The tabletop is gray and the lighting is even."}
RECIPE = {"orientation": "long-minus-short", "arithmetic": "float64(long)-float64(short)",
    "hash_encoding": "C-order little-endian float64", "duplicate_delta_bank": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def native_member(name):
    return any(part in name for part in ("/native-state-store/", "/qwen/native/", "/frontend/native/", "/decoder/own_forecast/", "/decoder/native_reference/", "/decoder/native-own/", "/decoder/native-reference/")) or name.endswith("/frontend/native.json") or Path(name).name in {
        "native-state-rows.json", "native-qwen.json", "frontend-native.json", "decoder-native-own.json", "decoder-native-reference.json"}


def check_frontier(inventory_names, external_files):
    post = {"source.json", "summary.json", "native-contract.json", "sdpa-native.json", "timeline.jsonl"}
    expected = {name for name in inventory_names if name not in post
        and not name.startswith("forecast-transaction/") and not native_member(name)}
    require(set(external_files) == expected, "complete pretarget source/forecast frontier coverage differs")


def helpers():
    for name, digest in PINS.items():
        require(sha(HERE / name) == digest, "qualified saved-reader source differs: " + name)
    a = runpy.run_path(str(HERE / "audit_flux_raw_to_image.py"))
    h = a["helpers"]()
    return a, h["audit_flux_functional_qwen_prefix.py"], h["audit_flux_functional_pass.py"], h["audit_flux_functional_remaining.py"]


def authenticate(root, *, expected_manifest, expected_request, account):
    require(sha(root / "files.json") == expected_manifest, "raw wrapper checksum differs")
    inventory = read(root / "files.json")
    require(inventory["schema"] == "mdb-image-tail-files-v1" and inventory["request_sha256"] == expected_request,
        "raw wrapper schema/request differs")
    members = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "raw symlink forbidden")
        if path.is_file() and path != root / "files.json":
            members[path.relative_to(root).as_posix()] = path
    require(set(members) == set(inventory["files"]), "complete raw wrapper member set differs")
    for name, row in inventory["files"].items():
        path = account["member"](root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "raw retained member differs: " + name)
    require(inventory["expanded_bytes"] == sum(row["bytes"] for row in inventory["files"].values()), "raw expanded bytes differ")
    config, request, source, summary, contract = (read(root / name) for name in ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json"))
    require(identity(request) == request["request_sha256"] == expected_request, "executed pair request differs")
    require(config["request_sha256"] == expected_request and all(value["request_sha256"] == expected_request
        and value["job_id"] == inventory["job_id"] for value in (source, summary)), "pair actual job/request joins differ")
    require("job_id" not in config or config["job_id"] == inventory["job_id"], "optional effective-plan job identity differs")
    require(all(config[key] == value for key, value in request["config"].items()), "effective pair plan differs")
    require(config["schema"] == "mdb-flux-raw-mask-pair-plan-v1" and config["cases"] == CASES
        and config["case_order"] == list(CASES) and config["seed"] == 277, "fixed pair specimen differs")
    require(request["source_hashes"]["flux_raw_mask_pair.py"] == source["worker_sha256"] == WORKER_SHA256,
        "executed paired worker source differs")
    require(source["helper_source_sha256"] == config["helper_source_sha256"]
        and all(request["source_hashes"][name] == digest for name, digest in source["helper_source_sha256"].items()), "actual helper/source pins differ")
    require(all(source["helper_source_sha256"].get(name) == digest for name, digest in account["NUMERICAL_PINS"].items())
        and source["helper_source_sha256"].get("flux_raw_to_image.py") == account["WORKER_SHA256"], "frozen original arithmetic graph differs")
    require(summary["counts"] == contract["counts"] and contract["forecast_module_entries"] == 0,
        "actual count/guard sidecars differ")
    expected = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=4,
        functional_decode_calls=2, native_vae_calls=4, native_qwen_calls=2, guard_refusals=0)
    transitions = 2 * config["steps"] * (3 + read(root / source["source_files"]["flux_config"]["path"])["num_layers"]
        + read(root / source["source_files"]["flux_config"]["path"])["num_single_layers"])
    expected.update(functional_transitions=transitions, native_transitions=transitions)
    require(all(summary["counts"][key] == value for key, value in expected.items()), "actual finite pair geometry differs")
    require(source["forecast_seal"] == summary["forecast_seal"], "paired source/summary seal differs")
    require(sha(root / "runtime-contract.json") == source["runtime_contract_sha256"], "sealed runtime checksum differs")
    return config, source, summary, contract, inventory


def parameters(root, config, source, contract, metadata, streamer):
    """Stream each original recipe once; never construct a neural model."""
    result = {}
    require(contract["qwen_tied_alias_before"] is contract["qwen_tied_alias_after"] is True, "actual tied LM/embedding alias differs")
    for component in ("qwen", "flux", "vae"):
        before, after = (contract["component_identity_" + when][component] for when in ("before", "after"))
        require(before and before == after, "actual final resident parameter/buffer identity differs")
        manifest = read(metadata[component + "_manifest"])
        expected = {row["source"]["tensor"]: row["source"]["payload_sha256"] for row in manifest["artifacts"]}
        require(len(expected) == len(manifest["artifacts"]), "duplicate parameter recipes")
        for row in manifest["artifacts"]:
            name = row["source"]["tensor"]
            require(name in before and before[name]["sha256"] == row["source"]["payload_sha256"]
                and before[name]["shape"] == row["source"]["shape"] and before[name]["dtype"] == "torch.bfloat16"
                and before[name]["device"] == "cpu", "actual learned tensor/recipe identity differs: " + name)
            require(not (root / row["path"]).exists(), "persistent parameter bank forbidden")
        require(all(value["device"] == "cpu" for value in before.values()), "completed components not CPU offloaded")
        result[component] = streamer["restore_parameters"](manifest, config[component + "_parameter_root"], expected,
            expected_module_sha256=config[component + "_retrieval_script_sha256"])
        result[component]["runtime_buffers"] = sorted(set(before) - set(expected))
    if config.get("profile") != "tiny-cpu":
        require([result[key]["tensors"] for key in ("qwen", "flux", "vae")] == [398, 169, 142], "original709recipe coverage differs")
        rotary = [row for name, row in contract["component_identity_before"]["qwen"].items() if name.endswith("inv_freq")]
        require(rotary and all(row["dtype"] == "torch.float32" and row["shape"] == [64] for row in rotary), "original FP32 RoPE buffer identity differs")
    expected_order = [(component, direction) for component in ("qwen", "flux", "vae") * 2 for direction in ("cuda", "cpu")]
    placements = contract["placement_records"]
    require([(row["component"], row["direction"]) for row in placements] == expected_order, "phase-batched placement order differs")
    for row in placements:
        component = row["component"]
        cpu = contract["component_identity_before"][component]
        gpu = {name: {**value, "device": "cpu" if component == "vae" and name.startswith("bn.") else "cuda:0"} for name, value in cpu.items()}
        first, last = (cpu, gpu) if row["direction"] == "cuda" else (gpu, cpu)
        require(row["before_signature"] == first and row["after_signature"] == last, "actual placement tensor/buffer signatures differ")
    return result


def parser(root, case_root, config, source, forecast, native, input_record, account):
    import numpy as np
    from transformers import AutoTokenizer

    binding = read(root / source["source_files"]["qwen_source_binding"]["path"])
    expected = {name for name in binding["source_files"] if name.startswith("tokenizer/")
        or name in {"text_encoder/config.json", "text_encoder/generation_config.json"}}
    require(set(source["tokenizer_files"]) == expected, "complete retained parser set differs")
    for name, row in source["tokenizer_files"].items():
        path = account["member"](root, row["retained"])
        require(path.stat().st_size == row["bytes"] == binding["source_files"][name]["bytes"]
            and sha(path) == row["sha256"] == binding["source_files"][name]["sha256"], "retained parser source differs")
    tokenizer = AutoTokenizer.from_pretrained(root / "source/tokenizer", local_files_only=True, padding_side="right")
    require(tokenizer.is_fast and type(tokenizer).__name__ == read(root / "source/tokenizer/tokenizer_config.json")["tokenizer_class"], "actual retained parser class differs")
    prompt = input_record["prompt"]
    messages, options = [dict(role="user", content=prompt)], dict(add_generation_prompt=True, enable_thinking=False)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    untruncated = tokenizer.apply_chat_template(messages, tokenize=True, padding=False, truncation=False, return_dict=True, **options)
    actual_uncropped_count = int(np.asarray(untruncated["input_ids"], np.int64).size)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, padding="max_length", truncation=True,
        max_length=config["max_length"], return_tensors="np", return_dict=True, **options)
    ids, mask = (np.asarray(tokens[key], np.int64).reshape(1, -1) for key in ("input_ids", "attention_mask"))
    require(ids.shape == mask.shape == (1, config["max_length"]) and np.isin(mask, (0, 1)).all()
        and mask.sum() > 0 and np.all(mask[0, 1:] <= mask[0, :-1]), "retained row/mask contract differs")
    require(input_record["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        and input_record["formatted_chat"] == forecast["formatted_chat"] == native["formatted_chat"] == formatted, "raw/template/truncation parser join differs")
    for record in (forecast, native):
        require(record["prompt"] == prompt and account["bytes_equal"](account["load_array"](case_root, record["input_ids"]), ids)
            and account["bytes_equal"](account["load_array"](case_root, record["attention_mask"]), mask), "actual retained Qwen operands differ")
    require(account["bytes_equal"](account["load_array"](case_root, forecast["position_ids"]), np.arange(mask.shape[1], dtype=np.int64)[None]), "actual prefix position IDs differ")
    all_attended = bool(mask.all())
    regime = "all-attended" if all_attended else "partially-padded"
    declared = input_record["mask"]
    require(declared["attended_rows"] == int(mask.sum()) and declared["padded_rows"] == int(mask.size - mask.sum())
        and declared["regime"] == regime and declared["expected_regime_observed"] == (regime == declared["expected_regime"]), "actual mask applicability report differs")
    expected_dispatch = {"backend": "sdpa", "gqa_policy": "enable_gqa" if all_attended else "repeat_kv", "source": config["attention_dispatch"]["source"]}
    require(input_record["dispatch"] == forecast["dispatch"] == native["dispatch"] == expected_dispatch
        and forecast["mask_observation"] == native["mask_observation"] == declared, "mask-bound per-case dispatch differs")
    return ids, mask, dict(prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        formatted_chat_sha256=hashlib.sha256(formatted.encode()).hexdigest(), input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), uncropped_token_count=actual_uncropped_count, retained_rows=mask.size,
        producer_uncropped_token_count_recorded=input_record["uncropped_token_count"],
        producer_uncropped_count_matches_actual=input_record["uncropped_token_count"] == actual_uncropped_count,
        uncropped_count_correction="Frozen producer used len(default BatchEncoding), counting mapping fields. Actual token count independently extracted from explicit input_ids; original field preserved, no numerical failure inferred.",
        mask=declared, dispatch=expected_dispatch, scope="Retained tokenizer/template reapplied offline, no original asset lookup or neural calls. Content and truncated suffix also change; not an isolated padding intervention.")


def sdpa_account(rows, *, cases, lane, geometry):
    require([row["ordinal"] for row in rows] == list(range(0 if lane == "forecast" else 2 * geometry["prefix"],
        2 * geometry["prefix"] if lane == "forecast" else 2 * (geometry["prefix"] + geometry["full"]))), "SDPA global ordinal coverage differs")
    result = {}
    for case in CASES:
        selected = [row for row in rows if row["case"] == case and row["lane"] == lane]
        require(len(selected) == geometry["prefix" if lane == "forecast" else "full"], "actual per-case SDPA occurrences differ")
        all_attended = cases[case]["mask"]["regime"] == "all-attended"
        matches = []
        for row in selected:
            require(row["configured_backend"] == "sdpa" and row["selected_kernel"] == "unprofiled", "unwitnessed selected-kernel claim")
            q, k, v = (row[key] for key in ("query", "key", "value"))
            expected_kv = geometry["kv_heads"] if all_attended else geometry["heads"]
            require(all(len(value["shape"]) == len(value["stride"]) == 4
                and isinstance(value["dtype"], str) and isinstance(value["device"], str) for value in (q, k, v)), "SDPA operand descriptor incomplete")
            mask = row["mask"]
            matches.append((mask is None) == all_attended and row["is_causal"] == all_attended and row["enable_gqa"] == all_attended
                and q["shape"] == [1, geometry["heads"], geometry["rows"], geometry["head_dim"]]
                and k["shape"] == v["shape"] == [1, expected_kv, geometry["rows"], geometry["head_dim"]]
                and all(value["dtype"] == "torch.bfloat16" for value in (q, k, v))
                and (mask is None or mask["shape"] == [1, 1, geometry["rows"], geometry["rows"]] and mask["dtype"] == "torch.bool"))
        result[case] = dict(calls=len(selected), all_argument_profiles_match_declared_regime=all(matches), per_call_regime_matches=matches,
            argument_receipts=selected, selected_kernel="unprofiled", scope="Observed SDPA argument descriptors, flags, head geometry and strides. Mask content hashes and selected CUDA kernel were not retained.")
    return result


def qwen_accounts(root, forecast, native, mask, config, q, a, *, hidden_width):
    import numpy as np
    prefix, selected = config["prefix_layers"], config["selected_hidden_states"]
    require(len(forecast["hidden_states"]) == len(native["hidden_states"]) == prefix + 1
        and forecast["forecast_prefix_layers"] == prefix and native["executed_full_layers"] == config["full_layers"], "interior prefix/reference depth differs")
    fields, selected_values = {}, {"forecast": [], "native": []}
    for index, (p, n) in enumerate(zip(forecast["hidden_states"], native["hidden_states"], strict=True)):
        left, right = a["load_array"](root, p), a["load_array"](root, n)
        require(left.shape == right.shape == (*mask.shape, hidden_width)
            and p["logical_dtype"] == n["logical_dtype"] == "torch.bfloat16", "complete original hidden query-row geometry differs")
        fields["hidden." + str(index)] = q["row_accounts"](left, right, mask)
        if index in selected:
            selected_values["forecast"].append(left)
            selected_values["native"].append(right)
    for lane, record in (("forecast", forecast), ("native", native)):
        require(a["bytes_equal"](np.concatenate(selected_values[lane], axis=-1), a["load_array"](root, record["conditioning"])), "conditioning is not own selected interior states")
    fields["conditioning"] = q["row_accounts"](a["load_array"](root, forecast["conditioning"]), a["load_array"](root, native["conditioning"]), mask)
    return dict(fields=fields, first_bit_discrepancy=next((name for name, row in fields.items() if not row["full"]["bitwise_exact"]), None),
        first_value_discrepancy=next((name for name, row in fields.items() if row["full"]["changed_values"]), None),
        scope="All512query rows primary, with attended/padded diagnostics. Selected intermediate states precede full-reference final normalization.")


def tensor_leaves(value, path=""):
    """Independently enumerate all retained tensor leaves, including context."""
    import numpy as np
    if hasattr(value, "detach"):
        item = value.detach().cpu()
        yield path, item.float().numpy() if str(item.dtype) == "torch.bfloat16" else item.numpy()
    elif isinstance(value, np.ndarray):
        yield path, value
    elif hasattr(value, "items"):
        for key, item in sorted(value.items()):
            yield from tensor_leaves(item, path + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, path + "/" + str(index))


def response_descriptor(short, long):
    import numpy as np
    require(short.shape == long.shape and short.size > 0, "signed response geometry differs")
    digest = hashlib.sha256()
    left, right = short.reshape(-1), long.reshape(-1)
    for offset in range(0, short.size, 1 << 18):
        delta = right[offset:offset + (1 << 18)].astype(np.float64) - left[offset:offset + (1 << 18)].astype(np.float64)
        digest.update(np.asarray(delta, dtype="<f8").tobytes())
    return dict(shape=list(short.shape), elements=int(short.size), dtype="float64", orientation="long-minus-short", sha256=digest.hexdigest())


def signed_field(values, sealed, q, selectors=None):
    """Check pretarget hash before reducing the independently archived targets."""
    p0, p1, n0, n1 = (values[key] for key in ("forecast-short", "forecast-long", "native-short", "native-long"))
    require(response_descriptor(p0, p1) == sealed, "sealed prospective signed response differs")
    result = dict(full=q["compare"](p1, n1, predicted_baseline=p0, native_baseline=n0),
        forecast_response=sealed, response_hash_verified=True)
    if selectors:
        for name, selector in selectors.items():
            result[name] = dict(rows=int(selector.sum()), comparison=None if not selector.any() else
                q["compare"](p1[:, selector], n1[:, selector], predicted_baseline=p0[:, selector], native_baseline=n0[:, selector]))
    return result


def chronology(root, summary):
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    require([event["kind"] for event in events].count("forecast_globally_sealed") == 1, "unique paired global seal missing")
    seal_index = next(index for index, event in enumerate(events) if event["kind"] == "forecast_globally_sealed")
    seal = events[seal_index]
    require(seal["seal_sha256"] == summary["forecast_seal"]["seal_sha256"] and seal["native_module_entries"] == 0
        and seal["case_count"] == 2, "paired target-free seal differs")
    entries = [(index, event) for index, event in enumerate(events) if event["kind"] == "phase_entry"]
    expected = [(lane, component, case) for lane in ("forecast", "native") for component in ("qwen", "flux", "vae") for case in CASES]
    require([(row["lane"], row["component"], row["case"]) for _, row in entries] == expected, "case/component phase batching differs")
    require(all(index < seal_index and row["native_module_entries"] == 0 for index, row in entries[:6])
        and all(index > seal_index for index, _ in entries[6:]), "native phase entered before both complete forecasts sealed")
    releases = [(index, event) for index, event in enumerate(events) if event["kind"] == "native_targets_released"]
    require(len(releases) == 1 and seal_index < releases[0][0] < entries[6][0]
        and releases[0][1]["forecast_seal"] == seal["seal_sha256"], "cohort native release chronology differs")
    placements = [(index, row) for index, row in enumerate(events) if row["kind"] == "component_placement"]
    expected_places = [(component, direction) for component in ("qwen", "flux", "vae") * 2 for direction in ("cuda", "cpu")]
    require([(row["component"], row["direction"]) for _, row in placements] == expected_places
        and placements[5][0] < seal_index < placements[6][0], "six placements/offloads around global seal differ")
    return dict(seal=seal, entries=[row for _, row in entries], placements=[row for _, row in placements], release=releases[0][1])


def audit(root, output, *, expected_manifest, expected_request):
    from mdb.equation_comparison import EquationCatalog
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(root), Path(output)
    a, q, streamer, remaining = helpers()
    config, source, summary, contract, inventory = authenticate(root, expected_manifest=expected_manifest,
        expected_request=expected_request, account=a)
    metadata = a["source_metadata"](root, config, source)
    restoration = parameters(root, config, source, contract, metadata, streamer)
    runtime = a["runtime_audit"](config, read(root / "runtime-contract.json"))
    tx = root / "forecast-transaction"
    verification = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"],
        expected_event_head=summary["transaction"]["event_head"], max_bytes=config["forecast_transaction_bytes"])
    require(verification["phase"] == "closed", "paired transaction not closed")
    kinds = [read(path)["kind"] for path in sorted((tx / "events").glob("*.json"))]
    require(kinds == ["opened", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and not read(tx / "seal.json")["violations"], "persisted paired journal failure/violation differs")
    prediction = a["plain"](load_forecast_tree(tx, read(tx / "forecasts/paired-raw-composition.json")["predictions"]))
    observation = a["plain"](load_forecast_tree(tx, read(tx / "observations/both_native_qwen_flux_vae_chains.json")["value"]))
    require(prediction["native_targets_at_seal"] == 0 and set(prediction["cases"]) == set(observation) == set(CASES), "cohort sealed/observed case coverage differs")
    check_frontier(inventory["files"], prediction["external_files"])
    declarations = read(tx / "seal.json")["declarations"]
    compact_json = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(declarations["model_identity"] == "flux2-klein-4b-raw-mask-pair-original"
        and declarations["input_identity"] == hashlib.sha256(compact_json(dict(cases=CASES, seed=277))).hexdigest()
        and declarations["source_identity"] == hashlib.sha256(compact_json({key: config[key + "_sha256"] for key in metadata})).hexdigest(),
        "sealed model/input/source declaration differs")
    require(declarations["frontier"] == dict(kind="raw-input", cases=CASES, seed=277, source_inventory=prediction["external_files"]), "sealed raw input/source frontier differs")
    for name, row in prediction["external_files"].items():
        path = a["member"](root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "globally sealed forecast dependency differs")
        require(not native_member(name), "future native evidence in pretarget frontier")
    for name in ("plan.json", "request.json", "runtime-contract.json", "forecast-responses.json", "sdpa-forecast.json", "common-frontend.json"):
        require(name in prediction["external_files"], "global forecast input/response dependency missing: " + name)
    require(prediction["responses"]["path"] == "forecast-responses.json"
        and prediction["responses"]["sha256"] == sha(root / "forecast-responses.json")
        and prediction["responses"]["recipe"] == config["response_recipe"] == RECIPE, "prospective response recipe/file seal differs")
    sealed_responses = read(root / "forecast-responses.json")
    require(sealed_responses["schema"] == "mdb-raw-mask-pair-responses-v1" and sealed_responses["recipe"] == RECIPE
        and set(sealed_responses["qwen"]) == {str(index) for index in range(config["prefix_layers"] + 1)} | {"conditioning"}
        and set(sealed_responses["decoder"]) == {"decoded", "pixels"}, "complete signed response arithmetic/field contract differs")
    qconfig, fconfig = read(metadata["qwen_config"]), read(metadata["flux_config"])
    roles = q["parameter_roles"](read(metadata["qwen_manifest"]), qconfig, config["prefix_layers"])
    geometry = dict(prefix=config["prefix_layers"], full=config["full_layers"], rows=config["max_length"],
        heads=qconfig["num_attention_heads"], kv_heads=qconfig["num_key_value_heads"], head_dim=qconfig["head_dim"])
    case_reports, saved, masks, ids, loaders, final, endpoints = {}, {}, {}, {}, {}, {}, {}
    for case, prompt in CASES.items():
        case_root = root / "cases" / case
        inputs = read(case_root / "input.json")
        require(inputs["prompt"] == prompt, "case raw prompt identity differs")
        fq, nq = (read(case_root / (lane + "-qwen.json")) for lane in ("forecast", "native"))
        fronts = {lane: read(case_root / ("frontend-" + lane + ".json")) for lane in ("forecast", "native")}
        decoders = {lane: read(case_root / ("decoder-" + lane + ".json")) for lane in ("forecast", "native-own", "native-reference")}
        require(prediction["cases"][case]["qwen"] == fq and prediction["cases"][case]["vae"] == decoders["forecast"]
            and observation[case]["qwen"] == nq and observation[case]["vae"]["own"] == decoders["native-own"]
            and observation[case]["vae"]["reference"] == decoders["native-reference"], "sealed own forecast/native sidecars differ")
        require(summary["cases"][case]["forecast"] == prediction["cases"][case]
            and summary["cases"][case]["native"] == observation[case], "cohort summary is not persisted forecast/native observation")
        ids[case], masks[case], parsed = parser(root, case_root, config, source, fq, nq, inputs, a)
        require(parsed["mask"] == contract["case_dispatch"][case] == summary["cases"][case]["mask"], "actual dispatch sidecars differ")
        load, _ = a["state_loader"](case_root, streamer, config)
        rows = {lane: read(case_root / (lane + "-state-rows.json")) for lane in ("forecast", "native")}
        for lane, cohort_record in (("forecast", prediction["cases"][case]), ("native", observation[case])):
            require(cohort_record["flux"] == dict(rows=lane + "-state-rows.json", transitions=len(rows[lane]) - 1), "sealed/observed complete trajectory row binding differs")
        origins = {lane: load(records[0], lane) for lane, records in rows.items()}
        for lane, record in (("forecast", fq), ("native", nq)):
            require(a["bytes_equal"](a["field_to_numpy"](origins[lane]["conditioning"]), a["load_array"](case_root, record["conditioning"])), "initial own state refreshed conditioning")
        trajectory = a["trajectory_accounts"](rows["forecast"][1:], rows["native"][1:],
            {lane: records[0] for lane, records in rows.items()}, load, joints=fconfig["num_layers"], singles=fconfig["num_single_layers"], steps=config["steps"])
        finals = {lane: load(records[-1], lane) for lane, records in rows.items()}
        decoder, endpoint = a["decoder_accounts"](case_root, config, decoders, finals, streamer, remaining, metadata)
        case_reports[case] = dict(parser=parsed, qwen=qwen_accounts(case_root, fq, nq, masks[case], config, q, a, hidden_width=qconfig["hidden_size"]),
            frontend=a["frontend_audit"](case_root, config, fronts, origins), denoiser=trajectory, decoder=decoder)
        saved[case] = dict(root=case_root, qwen={"forecast": fq, "native": nq}, rows=rows, decoder=decoders, frontend=fronts)
        loaders[case], final[case], endpoints[case] = load, finals, endpoint
    sdpa = {lane: sdpa_account(read(root / ("sdpa-" + lane + ".json")), cases={case: case_reports[case]["parser"] for case in CASES}, lane=lane, geometry=geometry)
        for lane in ("forecast", "native")}
    for lane in ("forecast", "native"):
        for case in CASES:
            declared = saved[case]["qwen"][lane]["actual_sdpa_argument_observations"]
            rows = sdpa[lane][case]["argument_receipts"]
            regime = case_reports[case]["parser"]["mask"]["regime"]
            flags_match = all((row["mask"] is None) == (regime == "all-attended")
                and row["is_causal"] == (regime == "all-attended") and row["enable_gqa"] == (regime == "all-attended") for row in rows)
            require(declared == dict(calls=len(rows), expected_regime=regime,
                all_argument_profiles_match_declared_regime=flags_match, selected_kernel="unprofiled"), "actual argument report/receipt join differs")
    common = {}
    common_declared = read(root / "common-frontend.json")
    for lane in ("forecast", "native"):
        for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas", "generator_before", "generator_after"):
            values = {case: a["load_array"](saved[case]["root"], saved[case]["frontend"][lane][key]) for case in CASES}
            require(a["bytes_equal"](values["short"], values["long"]), "paired same-seed common frontend actual bytes differ: " + key)
            common[lane + "." + key] = a["metric"](values["long"], values["short"])
            if lane == "forecast":
                record = saved["short"]["frontend"][lane][key]
                expected = dict(shape=record["shape"], stride=record["stride"], dtype=record["logical_dtype"],
                    device="cpu", sha256=record["content_sha256"])
                require(common_declared[key] == dict(exact_equal=True, descriptor=expected), "pretarget common frontend descriptor differs from actual saved bytes")
    selectors = dict(both_attended=(masks["short"][0] == 1) & (masks["long"][0] == 1),
        short_padded_long_attended=(masks["short"][0] == 0) & (masks["long"][0] == 1))
    responses = dict(qwen={}, denoiser=[], decoder={})
    for name in [str(index) for index in range(geometry["prefix"] + 1)] + ["conditioning"]:
        values = {}
        for case in CASES:
            for lane in ("forecast", "native"):
                record = saved[case]["qwen"][lane]
                row = record["conditioning"] if name == "conditioning" else record["hidden_states"][int(name)]
                values[lane + "-" + case] = a["load_array"](saved[case]["root"], row)
        responses["qwen"][name] = signed_field(values, sealed_responses["qwen"][name], q, selectors)
    count = len(saved["short"]["rows"]["forecast"])
    require(len(sealed_responses["denoiser"]) == count, "complete prospective response trajectory missing")
    for index in range(count):
        leaves = {}
        for case in CASES:
            for lane in ("forecast", "native"):
                row = saved[case]["rows"][lane][index]
                leaves[lane + "-" + case] = dict(tensor_leaves(loaders[case](row, lane)))
        require(all(set(values) == set(leaves["forecast-short"]) for values in leaves.values()), "paired recursive tensor liveness differs")
        sealed = sealed_responses["denoiser"][index]
        require(sealed["number"] == index - 1 and sealed["clock"] == saved["short"]["rows"]["forecast"][index]["clock"]
            and set(sealed["fields"]) == set(leaves["forecast-short"]), "paired sealed response clock/recursive coverage differs")
        fields = {}
        for name in leaves["forecast-short"]:
            values = {key: value[name] for key, value in leaves.items()}
            fields[name] = signed_field(values, sealed["fields"][name], q)
            fields[name]["per_case_full_leaf_errors"] = {case: q["compare"](values["forecast-" + case], values["native-" + case]) for case in CASES}
        responses["denoiser"].append(dict(number=index - 1, clock=sealed["clock"], fields=fields))
        del leaves
    for name in ("decoded", "pixels"):
        values = {lane + "-" + case: endpoints[case]["forecast" if lane == "forecast" else "native-reference"][name]
            for case in CASES for lane in ("forecast", "native")}
        responses["decoder"][name] = signed_field(values, sealed_responses["decoder"][name], q)
        responses["decoder"][name]["same_latent_conditional_response_error"] = q["compare"](
            endpoints["long"]["forecast"][name], endpoints["long"]["native-own"][name],
            predicted_baseline=endpoints["short"]["forecast"][name], native_baseline=endpoints["short"]["native-own"][name])
        responses["decoder"][name]["inherited_native_consumer_response_error"] = q["compare"](
            endpoints["long"]["native-own"][name], endpoints["long"]["native-reference"][name],
            predicted_baseline=endpoints["short"]["native-own"][name], native_baseline=endpoints["short"]["native-reference"][name])
    equal = ids["short"][0] == ids["long"][0]
    common_prefix = next((index for index, same in enumerate(equal) if not same), int(equal.size))
    prefix_diagnostics = {lane: {str(index): None if common_prefix == 0 else
        a["metric"](a["load_array"](saved["long"]["root"], saved["long"]["qwen"][lane]["hidden_states"][index])[:, :common_prefix],
            a["load_array"](saved["short"]["root"], saved["short"]["qwen"][lane]["hidden_states"][index])[:, :common_prefix])
        for index in range(geometry["prefix"] + 1)} for lane in ("forecast", "native")}
    result = dict(schema="mdb-raw-mask-pair-saved-audit-v1", job_id=source["job_id"], request_sha256=expected_request,
        worker_sha256=WORKER_SHA256, source=source, config=config, native_contract=contract, counts=summary["counts"],
        source_metadata={key: str(path.relative_to(root)) for key, path in metadata.items()}, parameter_restoration=restoration,
        qwen_parameter_roles=roles,
        runtime=runtime, transaction=verification, chronology=chronology(root, summary), cases=case_reports,
        sdpa=sdpa, common_frontend=common, signed_responses=responses,
        common_token_prefix=dict(rows=common_prefix, diagnostics=prefix_diagnostics, authority="Observation only; equality is not an acceptance/quality gate."),
        coverage=dict(cases=2, independent_seeds=1, complete_passes_per_case=config["steps"], retained_rows=config["max_length"],
            producer_plan_job_id="Optional by frozen producer schema; wrapper/source/summary job identities joined explicitly.",
            metadata_correction="Producer uncropped_token_count mislabeled len(BatchEncoding); retained-field value preserved and explicit input_ids token length reported independently.",
            response_context_leaves="Every recursive saved tensor leaf, including carried inputs and all context leaves; coordinate counts overlap across clocks.",
            recursive_per_case_error_scope="Every response leaf also retains both per-case full-field errors; these repeat primary case errors and carried coordinates, not new independent observations or physical transitions.",
            actual_sdpa_arguments="Q/K/V/mask descriptors and flags; no selected CUDA kernel or mask-content hash claim.", original_pngs=6),
        new_neural_forward_calls=0, new_learned_recurrence_calls=0, mechanics_status="authenticated-saved-paired-program", trend_status="observed", terminal_status="unassessed",
        scope="Fixed raw content/truncation plus padding/dispatch contrast. Shared original primitives, dependent same-latent consumers, no pure padding causal effect, universal semantic/kernel claim or automatic family induction.")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "array-index.json", {case: {key: value for key, value in row.items() if key != "root"} for case, row in saved.items()})
    catalog = EquationCatalog(output / "catalog/equations")
    contexts = []
    for case in CASES:
        for target, title in (("native-reference", "whole"), ("native-own", "same-latent-decoder")):
            context = dict(case=case, occurrence=title, prompt_utf8_sha256=case_reports[case]["parser"]["prompt_utf8_sha256"],
                seed=277, mask_sha256=case_reports[case]["parser"]["mask_sha256"], clock=dict(step=0, phase="project", cursor=0),
                request_sha256=expected_request, forecast_seal=summary["forecast_seal"]["seal_sha256"],
                numeric_profile_sha256=source["runtime_contract_sha256"], attention_dispatch=case_reports[case]["parser"]["dispatch"],
                parameter_manifests={key: config[key + "_manifest_sha256"] for key in ("qwen", "flux", "vae")})
            record = catalog.record(law_id="law.flux.raw-mask-pair.fixed-program.v1", route_id="route.flux.shared-original-primitives.raw-mask-pair.v1",
                alias="cmp.flux.raw-mask-pair." + case + "." + title,
                binding=dict(model_identity="original709checkpoint-recipes", execution=dict(id="mdb-flux-raw-mask-pair-v1"),
                    input_identity=case_reports[case]["parser"]["input_ids_sha256"], source_identity=expected_request,
                    frontier="raw prompt and seed before global cohort targets", horizon=dict(steps=config["steps"], decoder=title)),
                report=a["catalog_comparison"](endpoints[case]["forecast"], endpoints[case][target]),
                conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
                evidence=dict(audit_sha256=sha(output / "audit.json"), dependent_paths=True, independent_kernel_arithmetic=False))
            exact, wrong_clock = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to={**context, "clock": dict(step=1, phase="project", cursor=0)}))
            wrong_seed = len(catalog.find(applicable_to={**context, "seed": 278}))
            require(exact == 1 and wrong_clock == wrong_seed == 0, "paired catalog context abstention differs")
            contexts.append(dict(case=case, occurrence=title, fingerprint=record["fingerprint"], context=context, exact=exact, wrong_clock=wrong_clock, wrong_seed=wrong_seed))
    save(output / "catalog/provenance.json", dict(records=4, queries=contexts, automatic_family_induction=False,
        signed_response_scope="Complete prospective response hashes and all signed reduction rows are audit evidence, not additional independent histories."))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(identity(request) == request["request_sha256"] and request["config"] == config
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only pair CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged saved-reader source differs")
    root, started = Path(config["native_root"]), time.monotonic()
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json", "files.json", "runtime-contract.json",
        "timeline.jsonl", "sdpa-forecast.json", "sdpa-native.json", "forecast-responses.json", "common-frontend.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key, row in result["source"]["source_files"].items():
        target = output / "source-metadata" / (key + "-" + Path(row["path"]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["path"], target)
    for case in CASES:
        for name in ("input.json", "forecast-qwen.json", "native-qwen.json", "frontend-forecast.json", "frontend-native.json", "forecast-state-rows.json",
            "native-state-rows.json", "decoder-forecast.json", "decoder-native-own.json", "decoder-native-reference.json"):
            target = output / "cases" / case / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "cases" / case / name, target)
        for lane in ("forecast", "native-own", "native-reference"):
            image = read(root / "cases" / case / ("decoder-" + lane + ".json"))["png"]["path"]
            shutil.copyfile(root / "cases" / case / image, output / (case + "-" + lane + ".png"))
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots=dict(native=str(root)),
        manifests=dict(native={name: row["sha256"] for name, row in inventory["files"].items()}), scope="Full raw arrays remain near data. Compact metadata, standalone recipes/retriever and original6PNGs only; no weight bank."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-raw-mask-pair-compact-v1", job_id=os.environ["MRUN_JOB_ID"],
        request_sha256=request["request_sha256"], source_job=result["job_id"], records=4, cases=2, independent_seeds=1,
        model_calls=0, seconds=time.monotonic() - started, mechanics_status="verified-saved-pair"))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        output_root=str(output), files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        launch = runpy.run_path(str(HERE / "stage_saved_flux_mask_pair_audit.py"))
        print(json.dumps(launch["submit"](args.run, worker=Path(__file__)) if args.submit else launch["stage"](args.run, args.study,
            worker=Path(__file__), helpers=HELPERS, experiment="mdb-flux-raw-mask-pair-audit", operations=2,
            scope="Complete two-case saved array, signed response, argument and709recipe audit; no learned recurrence.")))
