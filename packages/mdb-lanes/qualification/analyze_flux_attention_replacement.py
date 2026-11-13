"""Saved-only custody, full-field reconstruction and native-consumer audit.

The attention arithmetic below is implemented independently of the staged
forecast helper. Numerical differences are observations, never quality gates.
Only source/ABI/payload/lineage corruption rejects an audit.
"""
from __future__ import annotations

import argparse
import json
import runpy
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GENERIC = runpy.run_path(str(HERE / "analyze_flux_mlp_replacement.py"))
ENTRY, SHARED, COMMON = (GENERIC[key] for key in ("ENTRY", "SHARED", "COMMON"))
require, exact_bits, sha_file = (GENERIC[key] for key in ("require", "exact_bits", "sha_file"))
SavedEvidence, field_error = GENERIC["SavedEvidence"], GENERIC["field_error"]
GLOBAL_COUNTS, PANEL_COUNTS = dict(GENERIC["GLOBAL_COUNTS"]), dict(GENERIC["PANEL_COUNTS"])
QUERIES = [0, 31, 32, 255, 511, 512, 767, 1023]
NEURONS = list(range(0, 9216, 288))
CONTEXT = {"id": "p1-s73", "prompt_id": "p1", "seed": 73,
           "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BRANCHES = [dict(zip(("id", "base", "feature", "source", "selection"), row)) for row in (
    ("zero", "zero", "native", "zero", "native"), ("text", "text", "native", "text", "native"),
    ("sham", "zero", "measured", "zero", "AM"),
    ("low.predicted.AM", "zero", "predicted", "zero", "AM"),
    ("high.predicted.AM", "text", "predicted", "text", "AM"),
    ("AM.measured", "zero", "measured", "text", "AM"),
    ("AM.predicted", "zero", "predicted", "text", "AM"),
    ("A.predicted", "zero", "predicted", "text", "A"),
    ("M.predicted", "zero", "predicted", "text", "M"))]
ROW_IDS = tuple(row["id"] for row in BRANCHES)
EXECUTION_ORDER = ("zero", "text", "AM.measured", "sham", "low.predicted.AM",
                   "high.predicted.AM", "AM.predicted", "A.predicted", "M.predicted")
REFERENCES = ("zero", "text", "AM.measured")
SLOTS = ("image", "text", "latent", "noise_prediction")


def nested_slot_array(e, cut, slot, path):
    """Read a retained BF16/F32 leaf through typed mappings and sequences."""
    root = e.data / "typed-store"
    record = e.read(root / "cuts" / (cut + ".json"))
    page = e.read(root / "pages" / (record["slots"][slot] + ".json"))
    encoded, described = page["encoded"], page["value"]
    for key in path:
        require(encoded["kind"] == described["kind"], "nested slot container descriptors differ")
        if encoded["kind"] == "mapping":
            matches = [item["value"] for item in described["items"] if item["key"]["value"] == key]
            require(key in encoded["items"] and len(matches) == 1, "nested mapping key absent or ambiguous")
            encoded, described = encoded["items"][key], matches[0]
        else:
            require(encoded["kind"] in {"tuple", "list"} and type(key) is int
                    and 0 <= key < len(encoded["items"]) == len(described["items"]), "nested sequence index invalid")
            encoded, described = encoded["items"][key], described["items"][key]
    require(encoded["kind"] == described["kind"] == "tensor", "nested slot is not a tensor")
    blob = root / "blobs" / (page["tensor_blob"] + ".safetensors")
    with blob.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        require(size <= blob.stat().st_size - 8, "nested tensor header exceeds blob")
        tensor = json.loads(stream.read(size))[encoded["name"]]
        shape, dtype = tensor["shape"], tensor["dtype"]
        require(dtype in {"BF16", "F32"} and shape and all(type(v) is int and v > 0 for v in shape),
                "nested tensor dtype/shape differs")
        start, stop = tensor["data_offsets"]
        width = 2 if dtype == "BF16" else 4
        require(0 <= start <= stop <= blob.stat().st_size - 8 - size
                and stop - start == int(np.prod(shape)) * width, "nested tensor byte geometry differs")
        stream.seek(8 + size + start)
        raw = stream.read(stop - start)
    values = ((np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
              if dtype == "BF16" else np.frombuffer(raw, dtype="<f4"))
    values = values.reshape(shape)
    require(np.isfinite(values).all() and e.typed["matches"](values, described, bf16=dtype == "BF16"),
            "nested slot tensor descriptor differs")
    return values


def quantize(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    return ((bits + np.uint32(32767) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)


def rebuild_attention(residual, modulation, weights, bias=None, *, text_rows, heads,
                      norm_q_weight, norm_k_weight, eps, qk_eps, rope_cos, rope_sin,
                      queries, key_indices=None, query_chunk=32, projection_chunk=512):
    """Independent explicit BF16 stream; no native or staged predictor calls."""
    x, modulation, weights = map(np.asarray, (residual, modulation, weights))
    require(x.ndim == 3 and x.shape[0] == 1 and type(heads) is int and heads > 0
            and type(text_rows) is int and 0 < text_rows < x.shape[1], "attention source geometry differs")
    length, width = x.shape[1:]
    require(width % heads == 0 and (width // heads) % 2 == 0, "attention head/RoPE geometry differs")
    dim = width // heads
    key_indices = list(dict.fromkeys((0, text_rows - 1, text_rows, length - 1))) if key_indices is None else key_indices
    source = (x, modulation, weights, np.asarray(norm_q_weight), np.asarray(norm_k_weight),
              np.asarray(rope_cos), np.asarray(rope_sin))
    require(modulation.shape in {(1, 3 * width), (1, 1, 3 * width)} and weights.shape == (3 * width, width)
            and source[3].shape == source[4].shape == (dim,)
            and source[5].shape == source[6].shape == (length, dim)
            and all(v.dtype == np.float32 and np.isfinite(v).all() for v in source)
            and type(query_chunk) is int and query_chunk > 0 and type(projection_chunk) is int and projection_chunk > 0
            and all(type(v) in {int, float} and np.isfinite(v) and v > 0 for v in (eps, qk_eps))
            and queries and len(set(queries)) == len(queries)
            and all(type(i) is int and 0 <= i < length - text_rows for i in queries)
            and key_indices and len(set(key_indices)) == len(key_indices)
            and all(type(i) is int and 0 <= i < length for i in key_indices), "attention source ABI differs")
    for v in source[:5]:
        require(not np.any(np.ascontiguousarray(v).view(np.uint32) & np.uint32(65535)), "native BF16 source bits differ")
    if bias is not None:
        bias = np.asarray(bias)
        require(bias.dtype == np.float32 and bias.shape == (3 * width,) and np.isfinite(bias).all()
                and not np.any(bias.view(np.uint32) & np.uint32(65535)), "attention bias ABI differs")
    wide = x.astype(np.float64)
    centered = wide - np.mean(wide, axis=-1, keepdims=True)
    norm = quantize(centered / np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + eps))
    shift, scale, _ = np.split(modulation.reshape(1, 1, 3 * width).astype(np.float64), 3, axis=-1)
    affine = quantize(quantize(quantize(1 + scale) * norm).astype(np.float64) + shift)
    projected = np.empty((1, length, 3 * width), np.float32)
    for first in range(0, 3 * width, projection_chunk):
        last = min(first + projection_chunk, 3 * width)
        value = affine.astype(np.float64) @ weights[first:last].astype(np.float64).T
        if bias is not None:
            value += bias[first:last].astype(np.float64)
        projected[:, :, first:last] = quantize(value)
    raw_q, raw_k, values = [v.reshape(1, length, heads, dim) for v in np.split(projected, 3, axis=-1)]
    normalized = []
    for raw, weight in ((raw_q, norm_q_weight), (raw_k, norm_k_weight)):
        raw64 = raw.astype(np.float64)
        normalized.append(quantize(raw64 / np.sqrt(np.mean(raw64 ** 2, axis=-1, keepdims=True) + qk_eps)
                                   * np.asarray(weight, np.float64)))
    rotated = []
    for value in normalized:
        paired = np.stack((-value[..., 1::2], value[..., 0::2]), axis=-1).reshape(value.shape)
        left = np.multiply(value, rope_cos[None, :, None, :], dtype=np.float32)
        right = np.multiply(paired, rope_sin[None, :, None, :], dtype=np.float32)
        rotated.append(quantize(np.add(left, right, dtype=np.float32)))
    result = np.empty((1, length - text_rows, width), np.float32)
    mass = np.empty((1, len(queries), heads), np.float64)
    analytic = np.empty((1, len(queries), width), np.float64)
    keys = rotated[1].astype(np.float64).transpose(0, 2, 3, 1)
    vals = values.astype(np.float64).transpose(0, 2, 1, 3)
    for first in range(0, length - text_rows, query_chunk):
        last = min(first + query_chunk, length - text_rows)
        qry = rotated[0][:, text_rows + first:text_rows + last].astype(np.float64).transpose(0, 2, 1, 3)
        score = (qry @ keys) * (1 / np.sqrt(float(dim)))
        score -= np.max(score, axis=-1, keepdims=True)
        probability = np.exp(score)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        out = (probability @ vals).transpose(0, 2, 1, 3).reshape(1, last - first, width)
        result[:, first:last] = quantize(out)
        for target, position in enumerate(queries):
            if first <= position < last:
                mass[:, target] = probability[:, :, position - first, :text_rows].sum(axis=-1)
                analytic[:, target] = out[:, position - first]
    stages = {"input": x, "layernorm": norm, "modulated": affine, "q_raw": raw_q, "k_raw": raw_k,
              "v": values, "q_norm": normalized[0], "k_norm": normalized[1],
              "q_rope": rotated[0], "k_rope": rotated[1]}
    positions = [text_rows + i for i in queries]
    samples = {name: v[:, positions].copy() for name, v in stages.items()}
    samples.update(query_indices=list(queries), key_indices=list(key_indices),
                   key_samples={name: v[:, key_indices].copy() for name, v in stages.items()},
                   attention=result[:, queries].copy(), attention_fp64=analytic,
                   text_mass=mass, image_mass=1 - mass)
    require(np.isfinite(result).all(), "nonfinite rebuilt attention")
    return result, samples


def compare_arrays(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    require(a.shape == b.shape and a.dtype == b.dtype == np.float32 and a.ndim >= 2,
            "numerical witness ABI differs")
    result = field_error(a.reshape(1, a.shape[1], -1), b.reshape(1, b.shape[1], -1))
    result.pop("per_position_error_rms", None)
    result["original_shape"] = list(a.shape)
    return result


def consumer_recipe(original, used, payloads, *, selection, text_rows, width):
    require(selection in {"A", "M", "AM"} and set(payloads) == set(selection)
            and original.shape == used.shape and original.ndim == 3 and 0 < text_rows < original.shape[1]
            and 0 < width < original.shape[2], "consumer partition ABI differs")
    checks = {"text_concat_exact_recipient": exact_bits(original[:, :text_rows], used[:, :text_rows])}
    for name, sl in (("A", slice(None, width)), ("M", slice(width, None))):
        expected = payloads[name] if name in selection else original[:, text_rows:, sl]
        checks[name + "_image_exact_declared"] = exact_bits(used[:, text_rows:, sl], expected)
    require(all(checks.values()), "consumer changed an undeclared operand or used another field")
    return checks


def current_parent(root, *, custody_verify=None):
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request, job, envelope = [h["read"](root / name) for name in ("request.json", "job.json", "report.json")]
    verify = custody_verify or runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    custody = verify(root, limits=request["config"].get("transport_limits"))
    require(custody["mechanics_status"] == "verified" and custody["job_id"] == job["job_id"] == envelope["job_id"]
            and custody["request_sha256"] == request["request_sha256"] == envelope["request_sha256"]
            and envelope["archive_root"] == "flux-attention-replacement-result", "attention custody/job/archive differs")
    for name, digest in request["source_hashes"].items():
        require(sha_file(h["file_path"](root / "payload", name)) == digest, "staged source digest differs")
    data = root / "extracted/flux-attention-replacement-result"
    source, summary, plan, frozen, binding = [h["read"](data / name) for name in
        ("source.json", "summary.json", "plan.json", "plan-frozen.json", "binding.json")]
    require(summary == envelope["report"] and summary["counts"] == GLOBAL_COUNTS
            and summary["panel_counts"] == {k: PANEL_COUNTS[k] for k in
                ("adapter_steps", "acts", "receipt_emissions", "candidate_calls", "images", "vae_calls")}
            and plan == request["config"] and plan["branches"] == BRANCHES
            and plan["context"] == source["context"] == CONTEXT
            and (plan["height"], plan["width"], plan["steps"], plan["single_index"]) == (512, 512, 4, 6)
            and plan["selector"] == {"logical_step": 3, "phase": "single", "index": 6}
            and plan["query_indices"] == QUERIES and plan["neuron_indices"] == NEURONS
            and tuple(plan["branch_execution_order"]) == EXECUTION_ORDER
            and summary["branch_execution_order"] == plan["branch_execution_order"]
            and summary["observation_plan"] == plan["observation_plan"]
            and source["job_id"] == summary["job_id"] == job["job_id"]
            and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
            and source["source_hashes"] == request["source_hashes"]
            and source["model_identity"] == summary["model_identity"]
            and source["execution"]["device"]["type"] == request["device"] == "cuda"
            and source["execution"]["dtype"] == "torch.bfloat16" and request["dtype"] == "bfloat16",
            "attention source/plan/census differs")
    require(frozen["plan_sha256"] == sha_file(data / "plan.json") and frozen["request_sha256"] == request["request_sha256"]
            and frozen["job_id"] == job["job_id"] and all(x == 0 for x in frozen["work"].values())
            and frozen["source_identity"] == h["sha"](h["canonical"](request["source_hashes"]))
            and h["sha"](h["canonical"](source["checkpoint_files"])) == source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"],
            "preload plan/checkpoint differs")
    transaction = h["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    require(transaction["phase"] == "closed"
            and SHARED["transaction_verifications_match"](transaction, summary["transaction_verification"]), "transaction differs")
    return {"root": root, "data": data, "result_root": data, "helpers": h, "request": request,
            "job": job, "source": source, "summary": summary, "plan": plan, "frozen": frozen,
            "binding": binding, "custody": custody, "transaction": transaction}


def audit_forecast(parent, e):
    from mdb_lanes.tensor_archive import load_tensor_archive

    h, data, b = parent["helpers"], parent["data"], parent["binding"]
    tree = h["NumericalTree"](data / "transaction")
    decode = lambda name, field: tree.decode(h["read"](data / "transaction" / name)[field])
    declared = decode("frontiers/source_binding.json", "value")
    predictions = decode("forecasts/attention_mlp_fields.json", "predictions")
    hypotheses = decode("forecasts/replacement_hypotheses.json", "predictions")
    before = {**GLOBAL_COUNTS, "adapter_steps": 108, "acts": 4, "receipt_emissions": 112,
              "candidate_calls": 0, "images": 0, "vae_calls": 0}
    require(declared == b and hypotheses == parent["plan"]["hypotheses"]
            and set(predictions["cases"]) == set(b["inputs"]) == {"zero", "text"}
            and (b["hidden_width"], b["mlp_width"], b["heads"], b["head_dim"], b["eps"]) == (3072, 9216, 24, 128, 1e-6)
            and type(b["qk_eps"]) in {float, int} and b["qk_eps"] > 0 and b["elementwise_affine"] is False
            and b["qkv_weight_row_range"] == [0, 9216] and b["mlp_weight_row_range"] == [9216, 27648]
            and b["counts_before_seal"] == before and b["target_entries_before_seal"] == 0
            and b["query_indices"] == QUERIES and b["neuron_indices"] == NEURONS
            and b["branch_execution_order"] == parent["plan"]["branch_execution_order"]
            and b["observation_plan"] == parent["plan"]["observation_plan"], "attention forecast binding differs")
    profile = b["attention_profile"]
    require(profile == parent["source"]["attention_dispatch_diagnostics"]
            and profile["processor_type"] == "diffusers.models.transformers.transformer_flux2.Flux2ParallelSelfAttnProcessor"
            and all(profile[key] is None for key in ("processor_backend", "parallel_config", "attention_mask", "joint_attention_kwargs"))
            and profile["is_causal"] is False and profile["kv_cache"] is False
            and set(profile["sdpa_enabled_flags"]) == {"flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"}
            and all(type(v) is bool for v in profile["sdpa_enabled_flags"].values()), "declared attention dispatcher profile differs")
    load = lambda key: None if b[key] is None else load_tensor_archive(data, b[key])
    params = {key: load(key) for key in ("qkv_weight_rows", "qkv_bias_rows", "mlp_weight_rows", "mlp_bias_rows",
                                        "norm_q_weight", "norm_k_weight")}
    require(params["qkv_weight_rows"].shape == (9216, 3072)
            and params["mlp_weight_rows"].shape == (18432, 3072)
            and params["norm_q_weight"].shape == params["norm_k_weight"].shape == (128,)
            and b["native_parameters"]["fused"]["weight"]["shape"] == [27648, 3072]
            and b["native_parameters"]["fused"]["weight"]["dtype"] == "torch.bfloat16"
            and (params["qkv_bias_rows"] is None) == (params["mlp_bias_rows"] is None)
            == ("bias" not in b["native_parameters"]["fused"]), "full native parameter profile differs")
    fields, reconstruction = {}, {}
    for role in ("zero", "text"):
        inp, forecast = b["inputs"][role], predictions["cases"][role]
        image, text, mod, cos, sin = [load_tensor_archive(data, inp[key]) for key in
                                      ("image", "text", "single_mod", "rope_cos", "rope_sin")]
        require(exact_bits(image, e.slot(inp["cut"], "image")) and exact_bits(text, e.slot(inp["cut"], "text"))
                and exact_bits(mod, nested_slot_array(e, inp["cut"], "context", ("single_mod",)))
                and exact_bits(cos, nested_slot_array(e, inp["cut"], "context", ("rope", 0)))
                and exact_bits(sin, nested_slot_array(e, inp["cut"], "context", ("rope", 1)))
                and e.point(inp["cut"])["logical_step"] == 3 and e.point(inp["cut"])["index"] == 6,
                "forecast input borrowed another cut/position context")
        a, m = np.asarray(forecast["attention"]["attention"]), np.asarray(forecast["mlp"]["activation"])
        require(a.shape == (1, 1024, 3072) and m.shape == (1, 1024, 9216)
                and a.dtype == m.dtype == np.float32 and np.isfinite(a).all() and np.isfinite(m).all()
                and not np.any(a.view(np.uint32) & 65535) and not np.any(m.view(np.uint32) & 65535), "sealed field ABI differs")
        rebuilt_a, sampled_a = rebuild_attention(np.concatenate((text, image), axis=1), mod,
            params["qkv_weight_rows"], params["qkv_bias_rows"], text_rows=text.shape[1], heads=24,
            norm_q_weight=params["norm_q_weight"], norm_k_weight=params["norm_k_weight"], eps=b["eps"],
            qk_eps=b["qk_eps"], rope_cos=cos, rope_sin=sin, queries=QUERIES)
        rebuilt_m = GENERIC["rebuild_full_mlp"](image, mod, params["mlp_weight_rows"], params["mlp_bias_rows"],
            eps=b["eps"], mlp_width=9216, quantize=quantize)
        # Rebuild errors remain observations: reduction tiling is intentionally different.
        sample_errors = {name: compare_arrays(np.asarray(forecast["attention"]["sampled"][name]), sampled_a[name])
                         for name in ("input", "layernorm", "modulated", "q_raw", "k_raw", "v", "q_norm", "k_norm", "q_rope", "k_rope", "attention")}
        require(exact_bits(forecast["attention"]["sampled"]["attention"], a[:, QUERIES])
                and exact_bits(forecast["mlp"]["sampled"]["activation"], m[:, QUERIES][:, :, NEURONS]),
                "sealed samples do not join complete sealed fields")
        fields[role] = {"A": a, "M": m}
        reconstruction[role] = {"A_sealed_minus_independent": field_error(a, rebuilt_a),
                                "M_sealed_minus_independent": field_error(m, rebuilt_m),
                                "attention_sample_rebuild": sample_errors,
                                "scope": "Independent source-only NumPy attention and standalone full-M reconstruction; no staged predictor call. All discrepancies retained without quality gate."}
        del rebuilt_a, rebuilt_m, sampled_a
    return fields, predictions, hypotheses, reconstruction


def audit_candidates(parent, e, fields, candidates):
    require(set(candidates) == set(ROW_IDS), "candidate inventory differs")
    native, images, texts, checks, controls = {}, {}, {}, {}, {}
    b = parent["binding"]
    for spec in BRANCHES:
        name, base, role = spec["id"], spec["base"], spec["source"]
        row = candidates[name]
        values = {key: e.array(record) for key, record in row["ports"].items()}
        require(set(values) == {"original_concat", "used_concat", "projection", "residual", "gate", "updated", "image", "text"}
                and row["input_parent"] == b["inputs"][base]["cut"] and row["text_rows"] == 512
                and row["attention_width"] == 3072
                and values["original_concat"].shape == values["used_concat"].shape == (1, 1536, 12288)
                and values["projection"].shape == values["residual"].shape == values["updated"].shape == (1, 1536, 3072)
                and values["gate"].shape == (1, 1, 3072),
                "candidate source/port ABI differs")
        residual = np.concatenate((e.slot(row["input_parent"], "text"), e.slot(row["input_parent"], "image")), axis=1)
        require(exact_bits(values["residual"], residual)
                and exact_bits(values["gate"], nested_slot_array(e, row["input_parent"], "context", ("single_mod",)).reshape(1, 1, 9216)[..., 6144:])
                and exact_bits(quantize(values["residual"] + quantize(values["gate"] * values["projection"])), values["updated"])
                and row["equation_exact"] is True and exact_bits(values["updated"][:, 512:], values["image"])
                and exact_bits(values["updated"][:, :512], values["text"]), "native candidate writer equation/closure differs")
        require(row["partition_controls"] == {key: True for key in
                ("text_concat_exact_recipient", "A_image_exact_selected", "M_image_exact_selected")},
                "declared native consumer partition checks differ")
        if spec["feature"] == "native":
            require(row["replacement"] is None and row["source_parent"] is None
                    and row["selection"] == "native" and exact_bits(values["used_concat"], values["original_concat"]), "native capture edited")
            native[base] = values
            checks[name] = {"native_unchanged": True}
        else:
            replacement = row["replacement"]
            require(row["selection"] == spec["selection"] and row["source_parent"] == b["inputs"][role]["cut"]
                    and all(replacement[key] == spec[key] for key in ("feature", "source", "selection")), "replacement source/recipe differs")
            payloads = {}
            for field in spec["selection"]:
                sl = slice(None, 3072) if field == "A" else slice(3072, None)
                expected = fields[role][field] if spec["feature"] == "predicted" else native[role]["original_concat"][:, 512:, sl]
                payload = e.array(replacement["payloads"][field])
                GENERIC["payload_archive_witness"](replacement["payloads"][field], payload, expected)
                payloads[field] = payload
            footprint = consumer_recipe(values["original_concat"], values["used_concat"], payloads,
                selection=spec["selection"], text_rows=512, width=3072)
            require(footprint == replacement["controls"], "worker consumer-footprint observations differ")
            original_exact = exact_bits(values["original_concat"], native[base]["original_concat"])
            text_exact = exact_bits(values["text"], native[base]["text"])
            require(replacement["diagnostics"] == {"native_original_concat_exact_recipient": original_exact,
                    "post_text_exact_recipient_native": text_exact}, "worker native consistency observations differ")
            controls[name + "_native_original_exact_recipient"] = original_exact
            controls[name + "_post_text_exact_recipient"] = text_exact
            checks[name] = footprint
        images[name], texts[name] = values["image"], values["text"]
    full_errors = {role: {field: field_error(native[role]["original_concat"][:, 512:, sl], fields[role][field])
                   for field, sl in (("A", slice(None, 3072)), ("M", slice(3072, None)))} for role in ("zero", "text")}
    for by_field in full_errors.values():
        for error in by_field.values():
            error["native_relative_rms"] = error["rms"] / error["actual_rms"] if error["actual_rms"] else None
            error["relative_rms_denominator"] = "relative_rms uses sealed reference; native_relative_rms uses observed native RMS"
    return native, images, texts, checks, controls, full_errors


def slot_meaning(point, name):
    if name == "latent":
        committed = point["logical_step"] == 4 and point["phase"] == "project"
        return {"phase": "committed-Z4" if committed else "stored-Z3-before-commit",
                "meaningful_propagation_observable": committed, "constant_before_commit": not committed, "stale": not committed}
    if name == "noise_prediction":
        meaningful = point["logical_step"] == 3 and point["phase"] == "scheduler"
        return {"phase": "native-readout-output" if meaningful else "stored-or-absent-noise",
                "meaningful_propagation_observable": meaningful, "stale": not meaningful}
    require(name in {"image", "text"}, "unknown trajectory slot")
    stale = point["phase"] not in {"single", "readout"}
    return {"phase": "last-single-output-after-readout" if stale else "working-single-stream",
            "meaningful_propagation_observable": not stale, "stale": stale}


def witness_values(record, reference=None):
    descriptor, witness = record["descriptor"], record["witness"]
    shape = descriptor["shape"]
    require(descriptor["dtype"] == "torch.bfloat16" and shape
            and all(type(v) is int and v > 0 for v in shape)
            and isinstance(descriptor.get("sha256"), str) and len(descriptor["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in descriptor["sha256"]), "trajectory descriptor ABI differs")
    count = int(np.prod(shape))
    coordinates = np.unique(np.linspace(0, count - 1, min(64, count)).astype(np.int64))
    values = np.asarray(witness["values_fp32"], dtype=np.float32)
    require(witness["flat_indices"] == coordinates.tolist() and witness["value_order"] == "C"
            and values.shape == coordinates.shape and np.isfinite(values).all()
            and not np.any(values.view(np.uint32) & 65535), "fixed-coordinate trajectory witness differs")
    if reference is not None:
        require(list(reference.shape) == shape and exact_bits(values, reference.reshape(-1)[coordinates]),
                "trajectory witness differs from retained full values")
    return values


def audit_observer(parent, e, rows, paths, prefixes):
    record = e.read(e.data / "continuation-observations.json")
    plan = parent["plan"]["observation_plan"]
    require(record["schema"] == "mdb-flux-attention-continuation-observations-v1"
            and record["plan"] == plan and tuple(record["branch_execution_order"]) == EXECUTION_ORDER
            and tuple(record["reference_rows"]) == tuple(plan["reference_rows"]) == REFERENCES
            and plan["boundaries_per_branch"] == 16 and plan["witness_count"] == 64
            and tuple(plan["slots"]) == SLOTS and record["model_calls_added"] == 0
            and set(record["rows"]) == set(ROW_IDS), "continuation observer contract differs")
    full_references, reports = {}, {}
    z3 = e.slot(parent["binding"]["inputs"]["zero"]["cut"], "latent")
    for name in EXECUTION_ORDER:
        trajectory, row = record["rows"][name], rows[name]
        require(len(trajectory) == 16 and row["continuation_observation"] == {"path": "continuation-observations.json", "row": name},
                "sixteen-boundary trajectory missing or retargeted")
        native_ids = [rid for rid in paths[name]["receipts"] if e.receipts[rid]["operation"] == "continue"]
        initial = row["acts"][-1]["post_act"] if row["acts"] else row["parent"]
        incoming = row["acts"][-1]["receipt_id"] if row["acts"] else prefixes[row["base"]]["receipt"]["fingerprint"]
        references = [ref for ref in REFERENCES if EXECUTION_ORDER.index(ref) <= EXECUTION_ORDER.index(name)]
        boundaries = []
        for index, item in enumerate(trajectory):
            expected_cut = initial if index == 0 else e.receipts[native_ids[index - 1]]["result"]
            expected_receipt = incoming if index == 0 else native_ids[index - 1]
            require(item["index"] == index and item["cut"] == expected_cut and item["point"] == e.point(expected_cut)
                    and item["incoming_receipt"] == expected_receipt
                    and item["complete_cut_retained"] == (index in {0, 15})
                    and set(item["slots"]) == set(SLOTS) and set(item["comparisons"]) == set(references),
                    "trajectory clock/receipt/full-retention metadata differs")
            slots, samples = {}, {}
            for slot in SLOTS:
                s = item["slots"][slot]
                require(s["meaning"] == slot_meaning(item["point"], slot) and type(s["present"]) is bool,
                        "stale or precommit slot mislabeled as active propagation")
                retained = None
                if expected_cut in e.store["cuts"]:
                    present = e.fields(expected_cut).get(slot, {}).get("kind") == "tensor"
                    require(s["present"] == present, "trajectory slot presence differs from retained cut")
                    if present:
                        retained = e.slot(expected_cut, slot)
                        require(e.typed["matches"](retained, s["descriptor"], bf16=True), "trajectory descriptor/cut differs")
                if not s["present"]:
                    require("array" not in s and "witness" not in s, "absent trajectory slot carries invented values")
                    continue
                if name in REFERENCES:
                    require("array" in s, "reference trajectory lacks its declared full array")
                    value = e.array(s["array"])
                    require(e.typed["matches"](value, s["descriptor"], bf16=True), "reference array/native value descriptor differs")
                    if retained is not None:
                        require(exact_bits(value, retained), "reference array differs from complete cut")
                    full_references[(name, index, slot)] = value
                    retained = value
                else:
                    require("array" not in s, "candidate trajectory retention contract differs")
                samples[slot] = witness_values(s, retained)
                if slot == "latent" and index < 15:
                    require(e.typed["matches"](z3, s["descriptor"], bf16=True), "stored precommit Z3 changed")
                    witness_values(s, z3)
                slots[slot] = {"descriptor": s["descriptor"], "meaning": s["meaning"],
                               "retained_full_array": name in REFERENCES, "array": s.get("array")}
            comparisons = {}
            for ref in references:
                comparison = item["comparisons"][ref]
                require(comparison["reference_cut"] == record["rows"][ref][index]["cut"]
                        and set(comparison["slots"]) == set(SLOTS), "trajectory reference retargeted")
                compared = {}
                for slot in SLOTS:
                    key = (ref, index, slot)
                    reported = comparison["slots"][slot]
                    if slot not in samples or key not in full_references:
                        require(reported == {"status": "not-present-in-both-current-states"}, "absent slot comparison differs")
                        continue
                    reference = full_references[key]
                    require(item["slots"][slot]["descriptor"]["shape"] == list(reference.shape)
                            and reported["sample_count"] == reference.size
                            and type(reported["changed_value_count"]) is int and 0 <= reported["changed_value_count"] <= reference.size
                            and reported["exact_bits"] == (reported["changed_value_count"] == 0)
                            and all(np.isfinite(reported[k]) for k in ("l2", "rms", "mae", "signed_mean", "max_abs")),
                            "worker full-norm metadata ABI differs")
                    ref_samples = witness_values(record["rows"][ref][index]["slots"][slot], reference)
                    independent = compare_arrays(samples[slot].reshape(1, -1, 1), ref_samples.reshape(1, -1, 1))
                    result = {"fixed_coordinate_native_minus_reference": independent,
                              "worker_observed_full_metrics": reported,
                              "full_metrics_independently_recomputed": name in REFERENCES}
                    if name in REFERENCES:
                        actual = full_references[(name, index, slot)]
                        recomputed = compare_arrays(actual, reference)
                        recomputed["mae"] = SHARED["BASE"]["distance"](actual, reference)["base_mae"]
                        result["independent_full_metrics"] = recomputed
                        result["reported_minus_recomputed"] = {k: reported[k] - recomputed[k] for k in ("l2", "rms", "mae", "signed_mean", "max_abs")}
                    compared[slot] = result
                comparisons[ref] = compared
            boundaries.append({"index": index, "cut": expected_cut, "point": item["point"], "slots": slots,
                               "comparisons": comparisons, "incoming_receipt": expected_receipt})
        reports[name] = boundaries
    return {"rows": reports, "reference_rows": list(REFERENCES), "boundaries_per_row": 16,
            "source_record_sha256": sha_file(e.data / "continuation-observations.json"),
            "scope": "All boundaries/declared phases/source hashes and fixed64 witnesses checked. Full reference arrays permit independent norms; intermediate candidate full norms are source-bound worker observations only. Stored Z3 is not evolving latent; I/T after readout are stale."}


def audit_stage_samples(parent, e, predictions, native):
    attention = e.read(e.data / "attention-stage-observations.json")
    mlp = e.read(e.data / "entry-observations.json")
    require(attention["calls"] == mlp["calls"] == 11 and attention["target_entries_before_seal"] == mlp["target_entries_before_seal"] == 0
            and set(attention["rows"]) == set(mlp["rows"]) == {"zero", "text"}, "native stage call census differs")
    result = {}
    for index, role in enumerate(("zero", "text")):
        actual_a, actual_m = attention["rows"][role], mlp["rows"][role]
        sealed_a, sealed_m = predictions["cases"][role]["attention"]["sampled"], predictions["cases"][role]["mlp"]["sampled"]
        require(actual_a["call_index"] == actual_m["call_index"] == index
                and actual_a["query_indices"] == sealed_a["query_indices"] == QUERIES
                and actual_a["key_indices"] == sealed_a["key_indices"] == [0, 511, 512, 1535], "native stage role/indices differ")
        stages = {}
        for group, forecast in (("query", sealed_a), ("keys", sealed_a["key_samples"])):
            require(set(actual_a[group]) == {"q_raw", "k_raw", "v", "q_norm", "k_norm"}, "native QKV/norm witness inventory differs")
            stages[group] = {key: {"actual": value, "native_minus_prediction": compare_arrays(e.array(value), np.asarray(forecast[key]))}
                             for key, value in actual_a[group].items()}
        stages["mlp"] = {}
        for key, predicted_key in (("layernorm", "layernorm"), ("modulated_input", "modulated"), ("u", "u"), ("v", "v"), ("activation", "activation")):
            value = e.array(actual_m[key])
            stages["mlp"][key] = {"actual": actual_m[key], "native_minus_prediction": compare_arrays(value, np.asarray(sealed_m[predicted_key]))}
        require(exact_bits(e.array(actual_m["activation"]), native[role]["original_concat"][:, np.asarray(QUERIES) + 512][:, :, np.asarray(NEURONS) + 3072]),
                "native sampled M does not join full consumer field")
        result[role] = stages
    return {"cases": result, "calls": 11,
            "scope": "Native raw Q/K/V and learned Q/K RMSNorm sampled after seal. No rotated native Q/K or probability matrix captured; forecast masses are unmeasured. M witnesses cover first two native source calls, not later edited calls."}


def audit_timing(parent, observed):
    events = SHARED["read_jsonl"](parent["data"] / "timeline.jsonl")
    require([v["sequence"] for v in events] == list(range(len(events))), "timeline sequence differs")
    seals = [v for v in events if v["kind"] == "full_field_forecast_sealed"]
    starts = [v for v in events if v["kind"] == "candidate_start"]
    ends = [v for v in events if v["kind"] == "candidate_complete"]
    branches = [v for v in events if v["kind"] == "branch_complete"]
    require(len(seals) == 1 and seals[0]["seal"] == parent["summary"]["seal"]
            and seals[0]["counts"] == parent["binding"]["counts_before_seal"] and seals[0]["target_entries_before_seal"] == 0
            and [v["candidate"] for v in starts] == [v["candidate"] for v in ends] == list(ROW_IDS)
            and [v["row"] for v in branches] == list(EXECUTION_ORDER)
            and all(seals[0]["sequence"] < a["sequence"] < b["sequence"] for a, b in zip(starts, ends))
            and all(v["counts"]["candidate_calls"] == i for i, v in enumerate(starts))
            and all(v["counts"]["candidate_calls"] == i + 1 for i, v in enumerate(ends))
            and observed["target_calls"] == parent["summary"]["target_calls"] == 11,
            "seal/source/target/physical call ordering differs")
    return {"seal_event": seals[0], "candidate_order": list(ROW_IDS), "branch_execution_order": list(EXECUTION_ORDER),
            "target_entries_before_seal": 0, "direct_calls": 9, "native_prefix_calls": 2,
            "scope": "Byte-bound trusted worker counters/guards and public sealed events, not a kernel trace or OS sandbox."}


def endpoint_analysis(images, pixels, latents):
    distance = SHARED["BASE"]["distance"]
    metrics = lambda name, base: {key: distance(values[name], values[base]) for key, values in
                                  (("post_image", images), ("latent", latents), ("rgb", pixels))}
    comparisons = {name: {base: metrics(name, base) for base in ("zero", "text")} for name in ROW_IDS}
    pairs = {label: {"actual": a, "reference": b, **metrics(a, b)} for label, a, b in (
        ("low_drop_in", "low.predicted.AM", "zero"), ("high_drop_in", "high.predicted.AM", "text"),
        ("high_AM_transfer", "AM.predicted", "AM.measured"))}
    ratio = GENERIC["error_effect_ratio"]
    ratios = {key: ratio(distance(values["AM.predicted"], values["AM.measured"])[metric],
                        distance(values["AM.measured"], values["zero"])[metric])
              for key, values, metric in (("post_image_L2", images, "base_l2"),
                                         ("latent_L2", latents, "base_l2"), ("RGB_MAE", pixels, "base_mae"))}
    return comparisons, pairs, {"AM_transfer": ratios,
        "component_scope": "A.predicted and M.predicted are exploratory component-only arms without separate matched measured component controls."}


def analyze(root, *, custody_verify=None):
    import mdb_lanes.tensor_archive as tensor_archive

    reader = Path(tensor_archive.__file__)
    reader_sha = sha_file(reader)
    parent = current_parent(root, custody_verify=custody_verify)
    h, data = parent["helpers"], parent["data"]
    run = {"root": parent["root"], "data": data, "plan": parent["plan"], "request": parent["request"],
           "source": {**parent["custody"], "transaction": parent["transaction"]},
           "transaction": data / "transaction", "tree": h["NumericalTree"](data / "transaction")}
    e = SavedEvidence(data, run=run)
    ENTRY["verify_store_identity"](e, parent["source"])
    lineage = audit_fresh_builder(parent, e)
    anchor = e.fields(parent["binding"]["inputs"]["zero"]["cut"])
    for cut in e.store["cuts"]:
        fields = e.fields(cut)
        require(fields["conditioning"] == anchor["conditioning"] and all(fields[key] == value for key, value in anchor.items()
                if key not in {"image", "text", "context", "latent", "noise_prediction", "step", "cursor", "phase"}),
                "retained schedule/ID/conditioning closure differs")
    points = SHARED["BASE"]["receipt_points"](e.receipts.values())
    require(all(cut not in e.store["cuts"] or e.point(cut) == point for cut, point in points.items()), "receipt/cut point differs")
    fields, predictions, hypotheses, reconstruction = audit_forecast(parent, e)
    observed = run["tree"].decode(h["read"](data / "transaction/observations/native.json")["value"])
    require(observed["counts"] == GLOBAL_COUNTS and observed["control_observations"] == parent["summary"]["control_observations"]
            and observed["branch_execution_order"] == parent["plan"]["branch_execution_order"]
            and observed["observation_plan"] == parent["plan"]["observation_plan"]
            and all(observed[key + "_sha256"] == sha_file(data / name) for key, name in (
                ("source", "source.json"), ("candidates", "candidates.json"), ("entry_observations", "entry-observations.json"),
                ("attention_stage_observations", "attention-stage-observations.json"),
                ("continuation_observations", "continuation-observations.json"))), "native observation/source hashes differ")
    timing = audit_timing(parent, observed)
    native, images, texts, checks, candidate_controls, full_errors = audit_candidates(parent, e, fields,
        e.read(data / "candidates.json")["candidates"])
    stages = audit_stage_samples(parent, e, predictions, native)
    prefixes = e.read(data / "prefixes.json")["prefixes"]
    rows, paths, prefix_paths, pixels, latents, controls = audit_paths(parent, e, images, texts, observed["rows"], prefixes)
    declared = {k: v for k, v in controls.items() if k not in {"sham_latent_exact_zero", "sham_png_exact_zero"}}
    declared.update({key + "_source_exact": lineage["pair_observations"][key + "_exact"] for key in ("image", "text")})
    declared["sham_png_equals_zero"] = controls["sham_png_exact_zero"]
    require(declared == parent["summary"]["control_observations"], "named worker replay observations differ")
    trajectory = audit_observer(parent, e, rows, paths, prefixes)
    controls.update(candidate_controls)
    comparisons, pairs, ratios = endpoint_analysis(images, pixels, latents)
    ratios["full_fields"] = {}
    for field, sl in (("A", slice(None, 3072)), ("M", slice(3072, None))):
        effect = field_error(native["text"]["original_concat"][:, 512:, sl], native["zero"]["original_concat"][:, 512:, sl])
        ratios["full_fields"][field] = {"measured_high_minus_low": effect,
            **GENERIC["error_effect_ratio"](full_errors["text"][field]["l2"], effect["l2"])}
    relative = data.relative_to(parent["root"]).as_posix()
    exported = []
    for name in ROW_IDS:
        row = rows[name]
        exported.append({**{key: row[key] for key in ("id", "base", "feature", "source", "selection", "parent", "final_cut")},
            "image": {**row["image"], "path": relative + "/" + row["image"]["path"]},
            "final_latent": {**row["final_latent"], "path": relative + "/" + row["final_latent"]["path"]},
            "row_record": {"path": relative + "/rows/" + name + ".json", "sha256": sha_file(data / "rows" / (name + ".json"))},
            "comparisons": comparisons[name], "native_path": paths[name], "acts": row["acts"],
            "continuation_observation": row["continuation_observation"]})
    require(sha_file(reader) == reader_sha, "installed tensor reader changed during audit")
    replay_checks = {key: controls[key] for key in ("zero_candidate_image_exact_native", "zero_candidate_text_exact_native",
        "text_candidate_image_exact_native", "text_candidate_text_exact_native", "sham_latent_exact_zero", "sham_png_exact_zero")}
    replay = not [key for key, value in replay_checks.items() if value is not True]
    source = parent["source"]
    return {"schema": "mdb-flux-attention-replacement-audit-v1", "mechanics_status": "valid",
        "instrument_status": "native-replay-qualified" if replay else "replay-unresolved", "terminal_status": "unassessed",
        "trend_status": "One fresh seed73 history; all full-field/rebuild/sample/continuation errors retained observationally.",
        "source": {**parent["custody"], "source_record_sha256": sha_file(data / "source.json"),
                   "staged_source_hashes": parent["request"]["source_hashes"], "global_plan_freeze": parent["frozen"], "custody_verifications": 1},
        "model_identity": source["model_identity"], "checkpoint_sha256": source["checkpoint_sha256"],
        "execution_status": {"job_id": parent["job"]["job_id"], "job_state": parent["job"]["state"], "model_loads": 1},
        "counts": GLOBAL_COUNTS, "panel_counts": PANEL_COUNTS, "fresh_builder": lineage, "typed_store_counts": e.store["counts"],
        "physical_census": {"builder_steps": 108, "prefix_steps": 2, "suffix_steps": 135, "image_acts": 7,
            "global_acts": 11, "global_receipts": 256, "direct_calls": 9, "vae_decodes": 9,
            "scope": "Ordered physical receipt occurrences verified before identity deduplication; branch and direct-call orders are separate."},
        "seal": parent["summary"]["seal"], "transaction_verification": parent["transaction"], "timing": timing,
        "saved_worker_transaction_verification": parent["summary"]["transaction_verification"], "sealed_hypotheses": hypotheses,
        "full_attention_forecast": {"source_binding": parent["binding"], "independent_reconstruction": {r: reconstruction[r]["A_sealed_minus_independent"] for r in reconstruction},
            "sample_reconstruction": {r: reconstruction[r]["attention_sample_rebuild"] for r in reconstruction},
            "native_minus_sealed": {r: full_errors[r]["A"] for r in full_errors},
            "dispatcher_diagnostics": parent["binding"]["attention_profile"], "native_probability_masses_measured": False},
        "full_mlp_forecast": {"independent_reconstruction": {r: reconstruction[r]["M_sealed_minus_independent"] for r in reconstruction},
                              "native_minus_sealed": {r: full_errors[r]["M"] for r in full_errors}},
        "entry_samples": stages, "continuation_trajectory": trajectory,
        "candidate_checks": checks, "control_observations": controls,
        "replay_qualification": {"status": "qualified" if replay else "unresolved", "checks": replay_checks,
            "scope": "Named native-prefix and unchanged-sham replay instrument only. Forecast errors/rebuild discrepancies/geometry have no authority in this qualification."},
        "native_prefixes": prefix_paths, "native_paths": paths, "comparisons": comparisons,
        "matched_pairs": pairs, "error_effect_diagnostics": ratios, "rows": exported,
        "nested_audit_tensor_reference_base": relative,
        "reference_scopes": {"rows": "row PNG/latent/record paths relative to run directory",
            "nested_arrays": "binding, sample and trajectory reference arrays relative to nested_audit_tensor_reference_base"},
        "auditor_source_sha256": sha_file(Path(__file__)), "tensor_archive_source_sha256": reader_sha,
        "audit_helpers": {name: sha_file(HERE / name) for name in ("analyze_flux_mlp_replacement.py", "analyze_flux_single_mlp.py",
            "analyze_flux_single_contraction.py", "analyze_flux_single_contexts.py", "analyze_flux_single_kv.py",
            "analyze_flux_rescue_routes.py", "summarize_context_window.py", "analyze_flux_mlp_forecast.py",
            "analyze_flux_concept_routes.py", "flux_family_custody.py")}, "models_executed": False,
        "limits": ["One fresh seed73 observed history and nine dependent native futures; no semantic or universal inference.",
            "Prediction uses pre-entry I/T, time modulation, actual weights and upstream RoPE. Current native QKV/MLP outputs are not forecast inputs. Trusted callbacks, not OS isolation.",
            "Independent attention uses32-query/512-row tiles; independent M uses64-query/512-neuron tiles. Any reconstruction discrepancies remain signed observations, not integrity thresholds.",
            "Native attention and MLP execute before selected full image fields are overwritten. No compute skipping, kernel independence or speedup claim.",
            "Only the three declared reference trajectories retain full intermediate arrays. Intermediate candidate full norms are worker observations; fixed64 witnesses do not independently prove those norms.",
            "Stored Z3 is constant until scheduler commit. I/T after readout and leftover noise at other phases are stale, not active propagation measurements.",
            "Native rotated Q/K and native probability matrices were not captured. Forecast bank masses and backend-enabled flags are not native probability measurements or a kernel trace.",
            "Array deduplication is by dtype/shape/value hash; archive native_stride describes the first representative, not later occurrence layout.",
            "A.predicted/M.predicted component-only arms lack separate matched measured controls. Error/effect ratios may be undefined at zero effect and never gate collection.",
            "Weight extraction, seed provenance and receipt footprints are source-bound; no parameter Merkle inclusion or independent RNG/kernel-read trace is claimed."]}


def audit_fresh_builder(parent, e):
    source = parent["source"]
    path, record = ENTRY["referenced_json"](parent, source["builder"])
    b = SavedEvidence(path.parent)
    ENTRY["verify_store_identity"](b, source)
    require(record["schema"] == "mdb-flux-single-context-builder-v1" and record["context"] == source["context"]
            and record["job_id"] == parent["job"]["job_id"] and source["context"]["seed"] == 73
            and source["context"]["id"] == "p1-s73", "replacement fresh builder context/job differs")
    own = b.array(record["arrays"]["own_conditioning"])
    require(exact_bits(own, ENTRY["raw_npy"](parent, source["encoding"]["conditioning"]))
            and exact_bits(own, b.slot(record["initial"], "conditioning"))
            and ENTRY["encoding_ids_exact"](ENTRY["raw_npy"](parent, source["encoding"]["txt_ids"]), b.slot(record["initial"], "txt_ids")),
            "replacement own encoding/text IDs differ from fresh native input")
    points = {"initial": (0,"project",None), "Z2": (2,"project",None), "zeroZ2": (2,"project",None),
              "Z3": (3,"project",None), "zero_single0": (3,"single",0), "own_single0": (3,"single",0),
              "matched_text": (3,"single",0)}
    require(all(tuple(b.point(record[name])[key] for key in ("logical_step","phase","index")) == expected
                for name,expected in points.items()), "fresh builder frontier differs")
    for name, slot, cut in (("initial_latent", "latent", record["initial"]), ("Z2_latent", "latent", record["Z2"]),
                            ("Z3_latent", "latent", record["Z3"]), ("own_conditioning", "conditioning", record["Z2"])):
        require(exact_bits(b.array(record["arrays"][name]), b.slot(cut, slot)), "fresh builder operand/cut differs")
    walk, apply = COMMON["builder_path"], COMMON["builder_act"]
    p, acts = record["paths"], record["acts"]
    ids = walk(b, p["to_Z2"], record["initial"], record["Z2"], expected_steps=56)
    zero = apply(b, acts["zero_conditioning"], record["Z2"], np.zeros_like(own), slot="conditioning")
    require(zero == record["zeroZ2"], "fresh zero conditioning result differs")
    ids.append(acts["zero_conditioning"]["receipt_id"])
    ids += walk(b, p["to_Z3"], zero, record["Z3"], expected_steps=28)
    restored = apply(b, acts["restore_conditioning"], record["Z3"], own, slot="conditioning")
    ids.append(acts["restore_conditioning"]["receipt_id"])
    ids += walk(b, p["zero_to_single0"], record["Z3"], record["zero_single0"], expected_steps=6)
    ids += walk(b, p["own_to_single0"], restored, record["own_single0"], expected_steps=6)
    matched = apply(b, acts["transplant_text"], record["zero_single0"], b.slot(record["own_single0"], "text"), slot="text")
    ids.append(acts["transplant_text"]["receipt_id"])
    require(matched == record["matched_text"], "fresh matched text source differs")
    emissions = b.read(path.parent / "receipt-emissions.json")
    require([r["sequence"] for r in emissions] == list(range(99)) and [r["receipt"] for r in emissions] == ids
            and set(ids) == set(b.receipts) and record["counts"] == {"adapter_steps": 96, "acts": 3, "receipt_emissions": 99},
            "fresh original builder physical census differs")
    prep_path, prep = ENTRY["referenced_json"](parent, source["preparation"])
    s = SavedEvidence(prep_path.parent)
    ENTRY["verify_store_identity"](s, source)
    expected = {"zero": record["zero_single0"], "text": matched}
    require(prep["builder"] == source["builder"] and prep["input_cuts"] == expected
            and prep["schema"] == "mdb-flux-single-mlp-preparation-v1", "fresh pre0 preparation source differs")
    ids = []
    for role in ("zero", "text"):
        first = prep["first_single0"][role]
        require(first["input_cut"] == expected[role] and s.store["cuts"][expected[role]] == b.store["cuts"][expected[role]],
                "fresh preparation borrowed another initial pair")
        ids += walk(s, [first["receipt"]], expected[role], first["post_cut"], expected_steps=1)
    high, low = (prep["first_single0"][role]["post_cut"] for role in ("text", "zero"))
    aligned = apply(s, prep["image_act"], low, s.slot(high, "image"), slot="image")
    ids.append(prep["image_act"]["receipt_id"])
    require(prep["matched_post0"] == {"zero": aligned, "text": high}, "post0 native image alignment differs")
    for role in ("zero", "text"):
        cut = prep["pre6_cuts"][role]
        ids += walk(s, prep["paths"][role], prep["matched_post0"][role], cut, expected_steps=5)
        require(s.store["cuts"][cut] == e.store["cuts"][cut] and cut == parent["binding"]["inputs"][role]["cut"]
                and e.point(cut)["logical_step"] == 3 and e.point(cut)["index"] == e.point(cut)["local_clock"] == 6,
                "fresh replacement pre6 input/clock differs")
        require(not np.count_nonzero(e.slot(cut, "conditioning")), "fresh replacement changed stored zero conditioning")
    emitted = s.read(prep_path.parent / "receipt-emissions.json")
    require([r["sequence"] for r in emitted] == list(range(13)) and [r["receipt"] for r in emitted] == ids
            and set(ids) == set(s.receipts) and prep["counts"] == {"adapter_steps": 12, "acts": 1, "receipt_emissions": 13},
            "fresh replacement preparation census differs")
    return {"kind": "fresh-native-builder", "context": source["context"], "job_id": parent["job"]["job_id"],
            "builder": source["builder"], "preparation": source["preparation"],
            "counts": {"adapter_steps": 108, "acts": 4, "receipt_emissions": 112},
            "pair_observations": ENTRY["fresh_pair"](e, prep["pre6_cuts"]["zero"], prep["pre6_cuts"]["text"]),
            "scope": "Actual fresh seed73 own encoding/noise, two typed conditioning Acts, text transplant, native0 image alignment, native1..5. No imported historical states or known prior endpoints; seed provenance source-bound, not an offline RNG replay."}


def audit_paths(parent, e, images, texts, observed_rows, prefixes):
    inputs = {role:item["cut"] for role,item in parent["binding"]["inputs"].items()}
    retained = {cut:e.fields(cut) for cut in e.store["cuts"]}
    conditioning = e.fields(inputs["zero"])["conditioning"]
    require(set(prefixes) == {"zero","text"}, "native prefix inventory differs")
    prefix_paths, controls = {}, {}
    for role,prefix in prefixes.items():
        start,post = inputs[role],prefix["post_single6"]
        path = e.walk(start,post,limit=3)
        require(path["native_steps"] == 1 and path["acts"] == 0 and prefix["input_cut"] == start
                and prefix["receipt"] == e.receipts[path["receipts"][0]] and e.point(post)["index"] == 7,
                "native reference prefix lineage differs")
        prefix_paths[role] = path
        controls[role+"_candidate_image_exact_native"] = exact_bits(images[role],e.slot(post,"image"))
        controls[role+"_candidate_text_exact_native"] = exact_bits(texts[role],e.slot(post,"text"))
    rows = {row["id"]:row for row in observed_rows}
    require(len(rows) == len(observed_rows) == 9 and set(rows) == set(ROW_IDS), "retained row inventory differs")
    paths,pixels,latents = {},{},{}
    for spec in BRANCHES:
        name = spec["id"];row = rows[name]
        require(e.read(e.data/"rows"/(name+".json")) == row and all(row[k] == v for k,v in spec.items())
                and row["parent"] == prefixes[spec["base"]]["post_single6"]
                and row["start_point"] == e.point(row["parent"])
                and row["continuation_policy"] == "recipient-stored-C-no-further-project", "row spec/native parent differs")
        parent_cut = row["parent"]
        require([act["slot"] for act in row["acts"]] == ([] if name in {"zero","text"} else ["image"]), "image Act inventory differs")
        current = parent_cut
        for action in row["acts"]:
            current = e.act(action,current,images[name])
        path = e.walk(parent_cut,row["final_cut"],row["acts"])
        require(path["native_steps"] == 15 and row["continuation_receipts"] == [e.receipts[rid] for rid in path["receipts"] if e.receipts[rid]["operation"] == "continue"]
                and e.point(row["final_cut"])["logical_step"] == 4 and e.point(row["final_cut"])["phase"] == "project",
                "15-transition native future differs")
        path["conditioning_coverage"] = SHARED["BASE"]["conditioning_coverage"](path,e.receipts,retained,conditioning,e.point)
        require(e.typed["matches"](e.slot(inputs["zero"],"conditioning"),row["final_conditioning"],bf16=True), "final conditioning witness differs")
        latents[name] = e.array(row["final_latent"])
        require(exact_bits(latents[name],e.slot(row["final_cut"],"latent")), "final native latent/cut differs")
        pixels[name],paths[name] = e.png(row["image"]),path
    emitted = SHARED["read_jsonl"](e.data/"receipt-emissions.jsonl")
    expected = [("prefix."+role,rid) for role in ("zero","text") for rid in prefix_paths[role]["receipts"]]
    expected += [(name,rid) for name in EXECUTION_ORDER for rid in paths[name]["receipts"]]
    require([x["sequence"] for x in emitted] == list(range(144)) and [(x["branch"],x["receipt"]) for x in emitted] == expected
            and {rid for _,rid in expected} == set(e.receipts)
            and sum(path["native_steps"] for path in [*prefix_paths.values(),*paths.values()]) == 137
            and sum(path["acts"] for path in paths.values()) == 7, "replacement physical emission census differs")
    controls["sham_latent_exact_zero"] = exact_bits(latents["sham"],latents["zero"])
    controls["sham_png_exact_zero"] = rows["sham"]["image"]["sha256"] == rows["zero"]["image"]["sha256"]
    return rows,paths,prefix_paths,pixels,latents,controls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("mechanics_status", "instrument_status", "counts")}))
