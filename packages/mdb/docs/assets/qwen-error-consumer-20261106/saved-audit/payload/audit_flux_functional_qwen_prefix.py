"""Saved-only audit of a prospective two-prompt Qwen prefix/Flux consumer.

The audit reads committed arrays and checkpoint recipes. It does not tokenize,
construct a neural model, invoke a module forward, or reconstruct a recurrence.
Full hidden-state error accounts are separate from primitive hash/witness data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0"
QWEN_HELPER_SHA256 = "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a"
PARAMETER_AUDITOR_SHA256 = "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80"
HELPERS = ("audit_flux_functional_pass.py", "audit_flux_image_tail_microscope.py", "retrieve_reader_evidence.py")
PROMPTS = (
    "A yellow cone to the left of a green cylinder on a plain white background. Two separate objects, simple studio product photograph.",
    "A small violet torus above a large copper pyramid on a plain ivory background. Two separate objects, simple studio product photograph.",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            digest.update(value)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def compare(predicted, native, *, predicted_baseline=None, native_baseline=None):
    """Complete saved-field reductions with explicit signs/denominator.

    A prompt response is second prompt minus first prompt, in each lane. All
    coordinates are compared in their native positions; text rows are numerical
    tokenizer positions and are not assumed to be lexical correspondences.
    """
    import numpy as np

    require(predicted.shape == native.shape and predicted.size > 0, "saved field pair geometry differs")
    require((predicted_baseline is None) == (native_baseline is None), "both prompt baselines required")
    shape = predicted.shape
    bitwise = predicted.dtype == native.dtype and predicted.tobytes() == native.tobytes()
    p, n = predicted.astype(np.float64), native.astype(np.float64)
    if predicted_baseline is not None:
        require(predicted_baseline.shape == native_baseline.shape == shape, "prompt response geometry differs")
        p -= predicted_baseline
        n -= native_baseline
        bitwise = None
    require(np.isfinite(p).all() and np.isfinite(n).all(), "nonfinite saved field")
    difference = p - n
    p2, n2, e2 = [float(np.dot(v.reshape(-1), v.reshape(-1))) for v in (p, n, difference)]
    dot = float(np.dot(p.reshape(-1), n.reshape(-1)))
    absolute = abs(difference)
    flat = int(np.argmax(absolute))
    coordinate = list(np.unravel_index(flat, shape))
    return dict(shape=list(shape), sample_count=int(predicted.size),
        orientation="functional-minus-native" if predicted_baseline is None else "functional prompt1-minus-prompt0 minus native prompt1-minus-prompt0",
        denominator="RMS of same native field" if predicted_baseline is None else "RMS of native prompt1-minus-prompt0 response",
        rms=math.sqrt(e2 / predicted.size), relative_rms=None if n2 == 0 else math.sqrt(e2 / n2),
        native_rms=math.sqrt(n2 / predicted.size), predicted_rms=math.sqrt(p2 / predicted.size),
        signed_mean=float(difference.mean()), native_signed_mean=float(n.mean()), predicted_signed_mean=float(p.mean()),
        dot=dot, cosine=None if n2 * p2 == 0 else dot / math.sqrt(n2 * p2),
        changed_values=int(np.count_nonzero(difference)), bitwise_exact=bitwise,
        maxabs=float(absolute.reshape(-1)[flat]), p95=float(np.quantile(absolute, .95)), p99=float(np.quantile(absolute, .99)),
        worst=dict(coordinate=[int(x) for x in coordinate], predicted=float(p[tuple(coordinate)]), native=float(n[tuple(coordinate)]), signed_error=float(difference[tuple(coordinate)])),
        scope="Complete saved arrays; native token coordinates, no alignment, quality threshold or semantic correspondence claim.")


def row_accounts(predicted, native, attention_mask):
    """All query rows remain primary; attended/padded rows are diagnostics."""
    import numpy as np

    require(predicted.ndim == 3 and predicted.shape[0] == 1
            and attention_mask.shape == predicted.shape[:2]
            and np.isin(attention_mask, (0, 1)).all(), "query row mask differs")
    result = dict(full=compare(predicted, native), row_count=predicted.shape[1],
        scope="All rows retained and consumed. Padding mask labels query-row diagnostics, not a license to drop outputs.")
    for name, selector in (("attended_query_rows", attention_mask[0] == 1), ("padded_query_rows", attention_mask[0] == 0)):
        result[name] = dict(rows=int(selector.sum()), comparison=None if not selector.any() else compare(predicted[:, selector], native[:, selector]))
    return result


def parameter_roles(manifest, config, prefix_layers):
    """Independently enumerate unequal-width Qwen parameters and visibility."""
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config.get("head_dim", hidden // heads)
    layers = config["num_hidden_layers"]
    require(0 < prefix_layers < layers and heads % kv_heads == 0 and head_dim % 2 == 0, "interior Qwen prefix geometry differs")
    shapes = {"model.embed_tokens.weight": [config["vocab_size"], hidden], "model.norm.weight": [hidden]}
    visible = {"model.embed_tokens.weight"}
    suffixes = {"input_layernorm.weight": [hidden], "post_attention_layernorm.weight": [hidden],
        "self_attn.q_norm.weight": [head_dim], "self_attn.k_norm.weight": [head_dim],
        "self_attn.q_proj.weight": [heads * head_dim, hidden], "self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
        "self_attn.v_proj.weight": [kv_heads * head_dim, hidden], "self_attn.o_proj.weight": [hidden, heads * head_dim],
        "mlp.gate_proj.weight": [intermediate, hidden], "mlp.up_proj.weight": [intermediate, hidden], "mlp.down_proj.weight": [hidden, intermediate]}
    for layer in range(layers):
        for suffix, shape in suffixes.items():
            name = f"model.layers.{layer}." + suffix
            shapes[name] = shape
            if layer < prefix_layers:
                visible.add(name)
    rows = {row["source"]["tensor"]: row["source"] for row in manifest["artifacts"]}
    require(len(rows) == len(manifest["artifacts"]) and set(rows) == set(shapes), "complete Qwen recipe name coverage differs")
    for name, row in rows.items():
        require(row["dtype"] == "BF16" and row["shape"] == shapes[name], "original Qwen recipe shape/dtype differs: " + name)
    return dict(forecast_parameters=sorted(visible), reference_only_parameters=sorted(set(shapes) - visible),
        all_parameters=sorted(shapes), counts=dict(all=len(shapes), forecast=len(visible), reference_only=len(shapes) - len(visible)),
        hidden_width=hidden, query_projection_width=heads * head_dim, key_value_projection_width=kv_heads * head_dim,
        selected_are_interior=True, final_model_norm="Reference-only, not applied to interior prefix target states.")


def audit_arrays(cases, masks, *, case_order, prefix_layers, selected_hidden_states):
    """Pure reduction core; callers supply only authenticated saved arrays.

    Per-case lanes contain hidden_states, conditioning, and projected arrays.
    Native consumer captures additionally bind the actual two E operands and
    corresponding outputs. No matrix multiplication or recurrence is executed.
    """
    import numpy as np

    require(set(cases) == set(masks) == set(case_order) and len(case_order) == 2, "two prompt identities required")
    selected = list(selected_hidden_states)
    require(selected == sorted(set(selected)) and 0 < min(selected) <= max(selected) <= prefix_layers,
            "selected hidden-state indices differ")
    rows, first_bits, first_values = {}, {}, {}
    for case in case_order:
        item, mask = cases[case], masks[case]
        p, n = item["forecast"], item["native"]
        expected = set(range(prefix_layers + 1))
        require(set(p["hidden_states"]) == set(n["hidden_states"]) == expected, "complete prefix hidden-state coverage differs")
        for lane in (p, n):
            geometry = lane["hidden_states"][0].shape
            require(len(geometry) == 3 and geometry[0] == 1 and all(value.shape == geometry for value in lane["hidden_states"].values()),
                    "prefix hidden geometry differs")
            expected_e = np.concatenate([lane["hidden_states"][index] for index in selected], axis=-1)
            require(lane["conditioning"].shape == expected_e.shape and lane["conditioning"].tobytes() == expected_e.tobytes(),
                    "conditioning is not exact own interior-state concatenation")
        require(mask.shape == p["hidden_states"][0].shape[:2], "prompt mask/hidden row geometry differs")
        consumers = item["consumers"]
        for name, operand in (("predicted", p["conditioning"]), ("reference", n["conditioning"])):
            value = consumers[name]["input"]
            require(value.shape == operand.shape and value.dtype == operand.dtype and value.tobytes() == operand.tobytes(),
                    "actual context embedder operand differs from declared complete E")
        require(consumers["reference"]["output"].shape == consumers["predicted"]["output"].shape == p["projected"].shape,
                "context embedder output geometry differs")
        fields = {"hidden."+str(i): row_accounts(p["hidden_states"][i], n["hidden_states"][i], mask) for i in range(prefix_layers + 1)}
        fields["conditioning"] = row_accounts(p["conditioning"], n["conditioning"], mask)
        fields["projected"] = row_accounts(p["projected"], consumers["reference"]["output"], mask)
        first_bits[case] = next((name for name, row in fields.items() if not row["full"]["bitwise_exact"]), None)
        first_values[case] = next((name for name, row in fields.items() if row["full"]["changed_values"]), None)
        rows[case] = dict(fields=fields, same_forecast_E_primitive_closure=row_accounts(p["projected"], consumers["predicted"]["output"], mask),
            actual_predicted_E_minus_reference_E_consumer=compare(consumers["predicted"]["output"], consumers["reference"]["output"]))
        rows[case]["largest_absolute_coordinate_error"] = max(
            (dict(field=name,maxabs=row["full"]["maxabs"],worst=row["full"]["worst"]) for name,row in fields.items()), key=lambda row:row["maxabs"])
        rows[case]["actual_predicted_E_minus_reference_E_consumer"]["orientation"] = "native context_embedder(predicted E) minus native context_embedder(reference E)"
    lo, hi = case_order
    responses = {}
    for index in range(prefix_layers + 1):
        responses["hidden."+str(index)] = compare(cases[hi]["forecast"]["hidden_states"][index], cases[hi]["native"]["hidden_states"][index],
            predicted_baseline=cases[lo]["forecast"]["hidden_states"][index], native_baseline=cases[lo]["native"]["hidden_states"][index])
    responses["conditioning"] = compare(cases[hi]["forecast"]["conditioning"], cases[hi]["native"]["conditioning"],
        predicted_baseline=cases[lo]["forecast"]["conditioning"], native_baseline=cases[lo]["native"]["conditioning"])
    responses["projected"] = compare(cases[hi]["forecast"]["projected"], cases[hi]["consumers"]["reference"]["output"],
        predicted_baseline=cases[lo]["forecast"]["projected"], native_baseline=cases[lo]["consumers"]["reference"]["output"])
    return dict(cases=rows, first_bit_discrepancy=first_bits, first_value_discrepancy=first_values, signed_prompt_response=responses,
        prompt_response_order=dict(positive=hi, negative=lo), retained_prefix_hidden_states=prefix_layers+1,
        consumer_scope="Observed context_embedder on complete predicted and reference E. No downstream Flux, denoiser, image or independent attention-kernel claim.")


def saved_array(root, record, *, case, kind, shape, index=None):
    """Decode the producer's immutable original BF16 bit archives only."""
    import numpy as np

    path = root / record["path"]
    require(not Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()) and path.stat().st_size == record["bytes"]
            and sha(path) == record["sha256"], "full saved array custody differs")
    bits = np.load(path, allow_pickle=False, mmap_mode="r")
    require(bits.dtype == np.dtype("<u2") and bits.flags.c_contiguous and list(bits.shape) == record["shape"] == list(shape)
            and record["logical_dtype"] == "torch.bfloat16" and record["stored_dtype"] == "uint16"
            and record["case"] == case and record["kind"] == kind and record["index"] == index
            and record["all_rows_retained"] is True, "saved array occurrence/complete BF16 geometry differs")
    require(hashlib.sha256(bits.tobytes()).hexdigest() == record["content_sha256"], "BF16 content hash differs")
    value = (bits.astype(np.uint32) << 16).view(np.float32)
    require(np.isfinite(value).all(), "nonfinite Qwen evidence")
    value.flags.writeable = False
    return value


def catalog_comparison(predicted, reference, *, compare_core=None):
    """Budget the complete, already-validated E and projected field pair."""
    if compare_core is None:
        from mdb.equation_comparison import compare_equations
        compare_core = compare_equations
    elements = sum(int(value.size) for value in predicted.values())
    return compare_core(predicted,reference,max_elements=elements)


def metadata_path(root, config, source, key):
    row = source["retained_metadata"][key]
    name = Path(row["path"])
    path = root/name
    require(not name.is_absolute() and ".." not in name.parts and name.name == Path(config[key]).name and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()) and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"] == config[key+"_sha256"],
            "copied source/recipe bytes differ")
    return path


def prompt_inputs(root, config, source, forecast):
    """Reapply the same pinned parser, never a learned encoder."""
    import numpy as np
    from transformers import AutoTokenizer

    inventory = read(root / "source-inventory.json")
    binding = read(metadata_path(root,config,source,"qwen_source_binding"))
    expected = {}
    for name, row in binding["source_files"].items():
        if name.startswith("tokenizer/") or name in {"text_encoder/config.json", "text_encoder/generation_config.json"}:
            expected["source/"+name] = dict(sha256=row["sha256"], bytes=row["bytes"])
    expected.update({row["path"]:dict(sha256=row["sha256"],bytes=row["bytes"]) for row in source["retained_metadata"].values()})
    expected["runtime-contract.json"] = dict(sha256=source["runtime_contract_sha256"],bytes=(root/"runtime-contract.json").stat().st_size)
    require(inventory == expected and expected, "complete pinned tokenizer/config inventory differs")
    for name, row in inventory.items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) and sha(path) == row["sha256"]
                and path.stat().st_size == row["bytes"], "pinned parser bytes differ")
    tokenizer = AutoTokenizer.from_pretrained(root / "source/tokenizer", local_files_only=True, padding_side="right")
    tokenizer_metadata = read(root/"source/tokenizer/tokenizer_config.json")
    require(type(tokenizer).__name__ == tokenizer_metadata["tokenizer_class"] and tokenizer.is_fast is True,
            "pinned tokenizer class/backend differs")
    masks, tokens = {}, {}
    for number, case in enumerate(sorted(forecast)):
        row, prompt = forecast[case], config["prompts"][number]
        require(row["case"] == case and row["prompt"] == prompt and row["prompt_utf8_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
                and row["formatted_chat_sha256"] == hashlib.sha256(row["formatted_chat"].encode()).hexdigest(), "raw/formatted prompt bytes differ")
        messages = [dict(role="user", content=prompt)]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            padding="max_length", truncation=True, max_length=config["max_length"], return_dict=True, return_tensors="np")
        ids = np.asarray(encoded["input_ids"], dtype=np.int64).reshape(1, -1)
        mask = np.asarray(encoded["attention_mask"], dtype=np.int64).reshape(1, -1)
        require(formatted == row["formatted_chat"] and ids.tolist() == row["input_ids"] and mask.tolist() == row["attention_mask"]
                and ids.shape == mask.shape == (1, config["max_length"]) and np.isin(mask, (0, 1)).all()
                and np.all(mask[0, 1:] <= mask[0, :-1]) and mask.sum() > 0, "same-pinned-parser IDs/mask/template differs")
        require(hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() == row["input_ids_sha256"]
                and hashlib.sha256(mask.astype("<i8").tobytes()).hexdigest() == row["attention_mask_sha256"]
                and int(mask.sum()) == row["active_rows"] and int(mask.size-mask.sum()) == row["padded_rows"]
                and row["position_ids"] == [list(range(config["max_length"]))], "integer token/position identity differs")
        masks[case] = mask
        tokens[case] = dict(input_ids_sha256=row["input_ids_sha256"], attention_mask_sha256=row["attention_mask_sha256"],
            prompt_utf8_sha256=row["prompt_utf8_sha256"], formatted_chat_sha256=row["formatted_chat_sha256"], active_rows=row["active_rows"], padded_rows=row["padded_rows"])
    return masks, dict(parser_scope="Same pinned tokenizer/template reapplication, no neural call or independent parser claim.", source_inventory=inventory, tokens=tokens)


def audit(root, output, *, expected_manifest, expected_request):
    from mdb.equation_comparison import EquationCatalog
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(root), Path(output)
    require(sha(HERE/"audit_flux_functional_pass.py") == PARAMETER_AUDITOR_SHA256, "qualified streamed recipe verifier differs")
    parameter_helper = runpy.run_path(str(HERE/"audit_flux_functional_pass.py"))
    inventory = runpy.run_path(str(HERE/"audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == set(inventory["files"]) | {"files.json"}, "complete raw inventory differs")
    config, source, summary, contract = [read(root/name) for name in ("plan.json", "source.json", "summary.json", "native-contract.json")]
    require(all(row["request_sha256"] == expected_request for row in (config, source, summary, inventory))
            and len({source["job_id"], summary["job_id"], inventory["job_id"]}) == 1, "actual Qwen job/request differs")
    require(source["worker_sha256"] == WORKER_SHA256 and source["helper_source_sha256"]["flux_functional_qwen_prefix.py"] == QWEN_HELPER_SHA256,
            "qualified executed Qwen producer differs")
    request = read(root/"request.json")
    request_identity = hashlib.sha256(json.dumps({key:value for key,value in request.items() if key!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(request_identity == request["request_sha256"] == expected_request
            and sha(root/"request.json") == source["request_record_sha256"] and sha(root/"plan.json") == source["effective_plan_sha256"]
            and all(config[key] == value for key,value in request["config"].items())
            and request["source_hashes"]["flux_functional_qwen_prefix_worker.py"] == WORKER_SHA256
            and request["source_hashes"]["flux_functional_qwen_prefix.py"] == QWEN_HELPER_SHA256,
            "executed request/effective plan/source identities differ")
    tiny = config.get("profile") == "tiny-cpu"
    require(config["cohort"] == "prospective-new" and tuple(config["prompts"]) == PROMPTS
            and config["batch_size"] == 1 and config["padding"] == "max_length" and config["padding_side"] == "right"
            and config["truncation"] is True and config["all_512_rows_retained"] is True, "prospective prompt/row contract differs")
    prefix, full, selected, rows = [config[k] for k in ("prefix_layers", "full_layers", "selected_hidden_states", "max_length")]
    require((prefix, full, selected, rows) == (27, 36, [9, 18, 27], 512) or tiny, "production interior prefix geometry differs")
    require(0 < min(selected) <= max(selected) == prefix < full, "reference final norm contaminates an interior target")
    metadata = {key:metadata_path(root,config,source,key) for key in ("qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script", "flux_config", "flux_manifest", "flux_retrieval_script")}
    qconfig,qmanifest,fmanifest = [read(metadata[key]) for key in ("qwen_config","qwen_manifest","flux_manifest")]
    roles = parameter_roles(qmanifest, qconfig, prefix)
    require(qconfig["num_hidden_layers"] == full and (tiny or roles["counts"] == dict(all=398, forecast=298, reference_only=100)), "full/prefix/reference recipe census differs")
    expected_qwen_identity = {entry["source"]["tensor"]:entry["source"]["payload_sha256"] for entry in qmanifest["artifacts"]}
    require(source["qwen_parameter_identity_before"] == source["qwen_parameter_identity_after"] == expected_qwen_identity,
            "original Qwen parameter identities changed or differ from checkpoint")
    restoration = {}
    for label, manifest, key in (("qwen", qmanifest, "qwen"), ("context", fmanifest, "flux")):
        selected_manifest = manifest if label == "qwen" else {**manifest,"artifacts":[entry for entry in manifest["artifacts"] if entry["source"]["tensor"].startswith("context_embedder.")]}
        restoration[label] = parameter_helper["restore_parameters"](selected_manifest, config[key+"_parameter_root"],
            {entry["source"]["tensor"]: entry["source"]["payload_sha256"] for entry in selected_manifest["artifacts"]},
            expected_module_sha256=config[key+"_retrieval_script_sha256"])
        restoration[label]["source_manifest_tensors"] = len(manifest["artifacts"])
        require(not any((root/entry["path"]).exists() for entry in manifest["artifacts"]), "persistent per-run pretrained bank forbidden")
    context_rows = [entry for entry in fmanifest["artifacts"] if entry["source"]["tensor"] == "context_embedder.weight"]
    require(len(context_rows) == 1 and not any(entry["source"]["tensor"] == "context_embedder.bias" for entry in fmanifest["artifacts"]), "original bias-free context projection recipe differs")
    context_shape = context_rows[0]["source"]["shape"]
    require(context_shape[1] == len(selected)*qconfig["hidden_size"] and (tiny or context_shape == [3072, 7680]), "native context projection shape differs")
    require(source["context_embedder_weight_identity_before"] == source["context_embedder_weight_identity_after"] == context_rows[0]["source"]["payload_sha256"],
            "actual context component weight differs or changed")
    forecast, native, consumers = [read(root/name) for name in ("forecast-cases.json", "native-cases.json", "context-consumers.json")]
    order = [f"prompt-{i:02d}" for i in range(2)]
    require(set(forecast) == set(native) == set(consumers) == set(order) and summary["cases"] == order, "two prompt occurrence coverage differs")
    tx = root/"forecast-transaction"
    verification = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"], expected_event_head=summary["transaction"]["event_head"], max_bytes=512 << 20)
    require(verification["phase"] == "closed" and not summary["transaction"]["violations"] and summary["transaction"]["failure"] is None, "closed global forecast transaction differs")
    decode = lambda value: parameter_helper["plain"](load_forecast_tree(tx, value))
    forecast_record = read(tx/"forecasts/qwen-prefix-forecast.json")
    prediction = decode(forecast_record["predictions"])
    require(prediction["cases"] == forecast, "sealed prefix/conditioning/consumer prediction sidecars differ")
    frontier = decode(read(tx/"frontiers/tokenized-inputs.json")["value"])
    require(frontier == {case:dict(input_ids=row["input_ids"],attention_mask=row["attention_mask"],prompt_utf8_sha256=row["prompt_utf8_sha256"]) for case,row in forecast.items()},
            "declared raw-token frontier includes mismatched/future evidence")
    observation = decode(read(tx/"observations/native-targets-and-context-consumers.json")["value"])
    require(observation["native"] == native and observation["consumers"] == consumers, "closed actual native/consumer observations differ")
    runtime = read(root/"runtime-contract.json")
    require(sha(root/"runtime-contract.json") == source["runtime_contract_sha256"] == prediction["runtime_contract_sha256"]
            and runtime["attention_dispatch_source"] == config["attention_dispatch"] and runtime["rope_buffer_policy"] == config["rope_buffer_policy"], "pre-target sealed runtime profile differs")
    for name,row in runtime["implementation_sources"].items():
        if config[name] == "runtime-bound":
            require(name == "tokenizers_native_source_sha256" and row["binding"] == "runtime-observed-on-CUDA-host"
                    and row["expected_sha256"] == "runtime-bound" and len(row["sha256"]) == 64
                    and all(value in "0123456789abcdef" for value in row["sha256"]), "native tokenizer extension runtime binding differs")
        else:
            require(row["sha256"] == config[name] and row["binding"] == "staged-source-hash" and row["expected_sha256"] == config[name], "bound actual implementation source differs")
    if not tiny:
        require(set(runtime["implementation_sources"]) == {"qwen_native_source_sha256","flux_native_source_sha256","tokenizers_wrapper_source_sha256", "tokenizers_native_source_sha256","jinja_environment_source_sha256","jinja_sandbox_source_sha256"},
                "complete encoder/parser/component source coverage differs")
        require(config["rope_buffer_policy"] == dict(init_device="cpu",init_dtype="float32",placement_dtype="float32",source="transformers-qwen3-rotary-embedding-v1-from-pretrained-device-only"),
                "actual original CPU-init/device-only RoPE policy differs")
    flags = runtime["numeric_flags_before_forecast"]
    require(flags["float32_matmul_precision"] == "highest" and flags["cuda_tf32"] is False and flags["cuda_bf16_reduced_precision_reduction"] is True
            and flags["cudnn_tf32"] is False and flags["cudnn_benchmark"] is False and flags["cudnn_deterministic"] is True and flags["deterministic_algorithms"] is True,
            "declared original numerical policy differs")
    masks, parser = prompt_inputs(root, config, source, forecast)
    arrays = {}
    require(set(forecast_record["payloads"]) == {case+".conditioning" for case in order}, "exact conditioning payload occurrence set differs")
    actual_operands = read(root/"context-consumer-operands.json")
    require(len(actual_operands) == 4, "four actual context operand captures required")
    for case_no, case in enumerate(order):
        f, n, c = forecast[case], native[case], consumers[case]
        hidden_shape = [1, rows, qconfig["hidden_size"]]
        e_shape, c_shape = [1, rows, context_shape[1]], [1, rows, context_shape[0]]
        require(len(f["forecast_hidden"]) == len(n["retained_hidden"]) == prefix+1
                and f["native_module_forwards_during_forecast"] == 0 and n["full_native_hidden_state_count"] == full+1
                and n["tail_layers_executed"] == full-prefix and n["output_hidden_states"] is True and n["use_cache"] is False,
                "own complete prefix/native suffix capture differs")
        profile = f["profile"]
        require(profile["native_module_forwards"] == 0 and profile["prefix_layers"] == prefix and profile["selected_hidden_states"] == selected
                and profile["hidden_size"] == qconfig["hidden_size"] and profile["conditioning_width"] == context_shape[1]
                and profile["attention_width"] == qconfig["num_attention_heads"]*qconfig["head_dim"] and profile["rope_buffer_policy"] == config["rope_buffer_policy"]
                and profile["attention_dispatch"] == config["attention_dispatch"] and n["attention_backend"] == config["attention_dispatch"], "native/functional attention or width profile differs")
        arrays[case] = dict(forecast=dict(hidden_states={i: saved_array(root, record, case=case, kind="forecast-hidden-state", shape=hidden_shape, index=i) for i, record in enumerate(f["forecast_hidden"])}),
            native=dict(hidden_states={i: saved_array(root, record, case=case, kind="native-hidden-state", shape=hidden_shape, index=i) for i, record in enumerate(n["retained_hidden"])}))
        arrays[case]["forecast"]["conditioning"] = saved_array(root, f["forecast_conditioning"], case=case, kind="forecast-conditioning", shape=e_shape)
        arrays[case]["native"]["conditioning"] = saved_array(root, n["native_conditioning"], case=case, kind="native-conditioning", shape=e_shape)
        for name in ("value","original"):
            payload = decode(forecast_record["payloads"][case+".conditioning"][name])
            require(payload.dtype.name == "uint16" and list(payload.shape) == e_shape
                    and hashlib.sha256(payload.tobytes()).hexdigest() == f["forecast_conditioning"]["content_sha256"], "sealed native conditioning payload bytes differ")
        arrays[case]["forecast"]["projected"] = saved_array(root, f["forecast_context_embedder"], case=case, kind="forecast-context-embedder", shape=c_shape)
        require(c["predicted_input"] == f["forecast_conditioning"] and c["reference_input"] == n["native_conditioning"]
                and c["full_flux_transformer_calls"] == 0, "component declared source operands differ")
        arrays[case]["consumers"] = {}
        for number, name in enumerate(("predicted", "reference")):
            operand = arrays[case]["forecast" if number == 0 else "native"]["conditioning"]
            capture = actual_operands[2*case_no+number]
            require(capture["ordinal"] == 2*case_no+number and capture["shape"] == e_shape and capture["dtype"] == "torch.bfloat16"
                    and capture["sha256"] == c[name+"_input"]["content_sha256"], "actual context_embedder prehook full operand differs")
            arrays[case]["consumers"][name] = dict(input=operand, output=saved_array(root, c[name+"_output"], case=case, kind="native-context-from-"+name, shape=c_shape))
    accounts = audit_arrays(arrays, masks, case_order=order, prefix_layers=prefix, selected_hidden_states=selected)
    events = [json.loads(line) for line in (root/"timeline.jsonl").read_text().splitlines()]
    seals = [(i,row) for i,row in enumerate(events) if row["kind"] == "forecast_globally_sealed"]
    forecasts = [(i,row) for i,row in enumerate(events) if row["kind"] == "forecast_case_complete"]
    targets = [(i,row) for i,row in enumerate(events) if row["kind"] == "native_case_complete"]
    require(len(seals) == 1 and [row["case"] for _,row in forecasts] == [row["case"] for _,row in targets] == order
            and all(i < seals[0][0] and row["native_forwards"] == 0 for i,row in forecasts)
            and all(i > seals[0][0] and row["retained_hidden_states"] == prefix+1 and row["full_native_hidden_state_count"] == full+1 and row["context_calls"] == 2 for i,row in targets)
            and seals[0][1]["seal_sha256"] == summary["forecast_seal"]["seal_sha256"] and seals[0][1]["native_encoder_calls"] == 0, "global forecast/native chronology differs")
    require(contract["full_native_layers"] == full and contract["retained_states_per_case"] == prefix+1
            and contract["native_encoder_calls"] == 2 and contract["context_embedder_native_calls"] == 4
            and contract["forecast_native_module_forwards"] == 0 and contract["context_embedder_shape"] == context_shape
            and contract["context_embedder_weight_identity_before"] == contract["context_embedder_weight_identity_after"] == source["context_embedder_weight_identity_before"]
            and contract["context_embedder_bias"] is False and contract["no_rgb_forecast"] is True
            and contract["rope_buffer_policy"] == config["rope_buffer_policy"] and contract["attention_dispatch"] == config["attention_dispatch"], "actual prefix/consumer contract differs")
    if not tiny:
        counts = summary["counts"]
        require(counts["model_loads"] == 1 and counts["encodes"] == 2 and counts["noise_draws"] == counts["vae_calls"] == 0,
                "top-level physical census differs")
        expected = dict(forecast=dict(tokenizer_calls=2,functional_prefix_calls=2,functional_module_forwards=0,embedding_states=2,prefix_hidden_states=54,
                                     selected_hidden_states=6,conditioning_ports=2,context_embedder_forecast_linear_calls=2,native_encoder_calls=0),
            native=dict(native_encoder_calls=2,full_native_layers_per_call=36,retained_hidden_states=56,retained_embedding_ports=2,tail_layers_executed_but_not_retained=18,native_module_forwards=2),
            consumers=dict(context_embedder_native_calls=4,predicted_embedding_calls=2,reference_embedding_calls=2,full_flux_transformer_calls=0,vae_calls=0,rgb_forecasts=0))
        for lane,values in expected.items():
            require(all(counts[lane][key] == value for key,value in values.items()), "declared encoder/component census differs")
    result = dict(schema="mdb-functional-qwen-prefix-audit-v1", job_id=source["job_id"], native_request_sha256=expected_request, native_manifest_sha256=expected_manifest,
        worker_sha256=WORKER_SHA256, auditor_sha256=sha(__file__), helper_sha256=QWEN_HELPER_SHA256, source=source, parser=parser,runtime_contract=runtime,
        parameter_roles=roles, parameter_restoration=restoration, transaction_verification=verification, native_contract=contract, counts=summary["counts"],
        census_scope=dict(declared_contract=summary["counts"],actual_artifact_occurrences=dict(native_complete_case_records=2,native_hidden_state_count_each=full+1,instrumented_context_operands=4),
            native_full_submodule_census="Not instrumented/exported. Two complete encoder root calls are source-bound and joined to full native output records; do not read native_module_forwards=2 as all internal module forwards."),
        accounts=accounts, full_array_records=dict(forecast=forecast,native=native,consumers=consumers,actual_consumer_operands=actual_operands),
        coverage=dict(full_hidden_states_per_lane_per_case=prefix+1, all_query_rows=rows, full_conditioning=True, actual_context_consumer_inputs="Full raw content hashes from native prehooks", primitive_arrays_retained=False, original_native_strides_retained=False,
            primitive_checkpoint_scope="This producer exports complete hidden/E/consumer fields, not an all-primitive hash/witness trace; no unrecorded primitive norm or dispatch-kernel identity is inferred."),
        scope="Two prospective raw prompts, own-state Qwen prefix and observed first Flux context projection; shared Torch primitives, no independent attention-kernel, final language norm, downstream image, full Flux session or semantic generalization claim.",
        mechanics_status="authenticated-saved-Qwen-prefix-audit", trend_status="observed", terminal_status="unassessed", new_neural_forward_calls=0)
    output.mkdir(parents=True, exist_ok=False)
    save(output/"audit.json", result)
    save(output/"array-index.json", result["full_array_records"])
    catalog, queries = EquationCatalog(output/"catalog/equations"), []
    for case in order:
        context = dict(case=case, input_identity=parser["tokens"][case]["input_ids_sha256"], mask_identity=parser["tokens"][case]["attention_mask_sha256"],
            raw_prompt_identity=parser["tokens"][case]["prompt_utf8_sha256"], source_request=expected_request,
            qwen_manifest=config["qwen_manifest_sha256"], flux_manifest=config["flux_manifest_sha256"], helper=QWEN_HELPER_SHA256,
            frontier="tokenized prospective prompt before native encoder", horizon=dict(prefix_layers=prefix, selected=selected, consumer="context_embedder"))
        predicted = dict(conditioning=arrays[case]["forecast"]["conditioning"], projected=arrays[case]["forecast"]["projected"])
        reference = dict(conditioning=arrays[case]["native"]["conditioning"], projected=arrays[case]["consumers"]["reference"]["output"])
        record = catalog.record(law_id="law.qwen3.own-state-prefix-and-context-projection.v1", route_id="route.qwen3.shared-torch-primitives.prefix.v1", alias="cmp.qwen-prefix."+case,
            binding=dict(model_identity="qwen3-flux2-klein-4b-original", execution=dict(id="mdb-flux-functional-qwen-prefix-v1", dtype="torch.bfloat16"),
                input_identity=context["input_identity"], source_identity=expected_request, frontier=context["frontier"], horizon=context["horizon"]),
            report=catalog_comparison(predicted,reference), conditions=[dict(field=key,op="eq",value=value) for key,value in context.items()], tested_context=context,
            readers=["native Qwen full encoder reference", "actual context_embedder component"], evidence=dict(audit_sha256=sha(output/"audit.json"), dependent_prompts=2, independent_kernel_arithmetic=False))
        exact = len(catalog.find(applicable_to=context))
        wrong = len(catalog.find(applicable_to={**context,"input_identity":"changed-token-input"}))
        missing = len(catalog.find(applicable_to={key:value for key,value in context.items() if key!="mask_identity"}))
        require(exact == 1 and wrong == missing == 0, "exact-context catalog applicability differs")
        queries.append(dict(case=case,fingerprint=record["fingerprint"],exact=exact,changed_input=wrong,missing_mask=missing))
    save(output/"catalog/provenance.json",dict(records=2,queries=queries,dependent_prompts=2,family_induction=False,
        scope="Two conditioning-plus-component forecast occurrences; complete per-hidden errors remain in audit, not independent replications."))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request,config = read(os.environ["SATURN_DEBUG_REQUEST"]),read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name,digest in request["source_hashes"].items():
        require(sha(name) == digest,"staged audit source differs")
    root,started = Path(config["native_root"]),time.monotonic()
    output = Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"]/"binding-minimal-result"
    result,inventory = audit(root,output,expected_manifest=config["native_manifest_sha256"],expected_request=config["native_request_sha256"])
    original = read(root/"plan.json")
    for name in ("plan.json","source.json","summary.json","native-contract.json","files.json","timeline.jsonl","forecast-cases.json","native-cases.json","context-consumers.json","context-consumer-operands.json","source-inventory.json"):
        shutil.copyfile(root/name,output/("source-"+name))
    shutil.copyfile(root/"request.json",output/"source-request.json")
    for key in ("qwen_config","qwen_manifest","qwen_source_binding","qwen_retrieval_script","flux_config","flux_manifest","flux_retrieval_script"):
        shutil.copyfile(metadata_path(root,original,result["source"],key),output/Path(original[key]).name)
    shutil.copyfile(root/"runtime-contract.json",output/"source-runtime-contract.json")
    shutil.copyfile(HERE/"retrieve_reader_evidence.py",output/"retrieve_evidence.py")
    save(output/"external-evidence.json",dict(host="beast",roots=dict(native=str(root)),manifests=dict(native={name:row["sha256"] for name,row in inventory["files"].items()}),
        scope="All full hidden, E and consumer arrays remain near data; individually retrievable by original checksums."))
    save(output/"request.json",request)
    save(output/"plan.json",config)
    save(output/"summary.json",dict(schema="mdb-functional-qwen-prefix-compact-v1",job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],source_job=result["job_id"],
        records=2,model_calls=0,seconds=time.monotonic()-started,mechanics_status="verified-saved-evidence"))
    save(output/"files.json",{path.relative_to(output).as_posix():sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY "+json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],output_root=str(output),
        files_manifest_sha256=sha(output/"files.json"),summary_sha256=sha(output/"summary.json")),sort_keys=True),flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run",type=Path)
        parser.add_argument("--study",type=Path)
        parser.add_argument("--submit",action="store_true")
        args = parser.parse_args()
        launch = runpy.run_path(str(HERE/"stage_saved_flux_audit.py"))
        print(json.dumps(launch["submit"](args.run,worker=Path(__file__)) if args.submit else launch["stage"](args.run,args.study,worker=Path(__file__),helpers=HELPERS,
            experiment="mdb-functional-qwen-prefix-audit",operations=2,scope="Two source-bound own-state Qwen prefix/conditioning forecasts and native component consumers; full saved arrays, exact recipes, no neural calls.")))
