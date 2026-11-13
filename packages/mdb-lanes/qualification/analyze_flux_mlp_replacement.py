"""Offline audit of sealed full MLP replacement fields and native futures.

All native arrays are retained evidence. Independent NumPy reconstruction uses
only pre-entry image/modulation and actual full paired weight rows, in bounded
chunks. No native module, staged predictor or model is run by this auditor.
"""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENTRY = runpy.run_path(str(HERE / "analyze_flux_single_mlp.py"))
SHARED, COMMON = ENTRY["SHARED"], ENTRY["COMMON"]
require, exact_bits = SHARED["require"], SHARED["exact_bits"]
sha_file = SHARED["sha_file"]
SavedEvidence = ENTRY["SavedEvidence"]
PANEL_COUNTS = {"model_loads": 0, "prompt_encodings": 0, "noise_draws": 0, "adapter_steps": 137,
                "acts": 7, "candidate_calls": 9, "receipt_emissions": 144, "images": 9, "vae_calls": 9, "physical_batch": 1}
GLOBAL_COUNTS = {**PANEL_COUNTS, "model_loads": 1, "prompt_encodings": 1, "noise_draws": 1,
                 "adapter_steps": 245, "acts": 11, "receipt_emissions": 256}
ROW_IDS = ("zero", "text", "sham", "low.predicted", "M.measured", "M.predicted",
           "M.measured.roll", "M.predicted.roll", "high.predicted")
BRANCHES = [dict(zip(("id", "base", "feature", "source", "roll"), values)) for values in (
    ("zero", "zero", "native", "zero", 0), ("text", "text", "native", "text", 0),
    ("sham", "zero", "measured", "zero", 0), ("low.predicted", "zero", "predicted", "zero", 0),
    ("M.measured", "zero", "measured", "text", 0), ("M.predicted", "zero", "predicted", "text", 0),
    ("M.measured.roll", "zero", "measured", "text", 512), ("M.predicted.roll", "zero", "predicted", "text", 512),
    ("high.predicted", "text", "predicted", "text", 0))]


def rebuild_full_mlp(image, modulation, weights, bias=None, *, eps, mlp_width,
                     query_chunk=64, neuron_chunk=512, quantize, progress=None):
    """Explicit BF16 reference with FP64 LayerNorm/linear and bounded temporaries."""
    image, modulation, weights = map(np.asarray, (image, modulation, weights))
    require(image.ndim == 3 and image.shape[0] == 1 and image.dtype == weights.dtype == modulation.dtype == np.float32
            and modulation.shape in {(1, 3 * image.shape[-1]), (1, 1, 3 * image.shape[-1])}
            and type(mlp_width) is int and mlp_width > 0 and weights.shape == (2 * mlp_width, image.shape[-1])
            and type(query_chunk) is int and query_chunk > 0 and type(neuron_chunk) is int and neuron_chunk > 0
            and type(eps) in {int, float} and np.isfinite(eps) and eps > 0
            and all(np.isfinite(x).all() for x in (image, modulation, weights)), "full MLP operand/chunk geometry differs")
    if bias is not None:
        bias = np.asarray(bias)
        require(bias.shape == (2 * mlp_width,) and bias.dtype == np.float32 and np.isfinite(bias).all(), "full MLP bias differs")
    width = image.shape[-1]
    shift, scale, _ = np.split(modulation.reshape(1, 1, 3 * width).astype(np.float64), 3, axis=-1)
    q = lambda x: quantize(np.asarray(x, dtype=np.float32))
    one_plus = q(1 + scale)
    result = np.empty((1, image.shape[1], mlp_width), np.float32)
    for start in range(0, image.shape[1], query_chunk):
        stop = min(start + query_chunk, image.shape[1])
        x = image[:, start:stop].astype(np.float64)
        centered = x - x.mean(axis=-1, keepdims=True)
        norm = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
        affine = q(q(one_plus * q(norm)).astype(np.float64) + shift).astype(np.float64)
        for first in range(0, mlp_width, neuron_chunk):
            last = min(first + neuron_chunk, mlp_width)
            u = affine @ weights[first:last].astype(np.float64).T
            v = affine @ weights[mlp_width + first:mlp_width + last].astype(np.float64).T
            if bias is not None:
                u += bias[first:last].astype(np.float64)
                v += bias[mlp_width + first:mlp_width + last].astype(np.float64)
            u16, v16 = q(u), q(v)
            result[:, start:stop, first:last] = q(q(ENTRY["stable_silu"](u16)) * v16)
        if progress is not None:
            progress(stop, image.shape[1])
    require(np.isfinite(result).all(), "nonfinite independently rebuilt MLP field")
    return result


def field_error(actual, reference, *, chunk=64):
    """Full-field signed errors and bit census without a giant FP64 delta."""
    actual, reference = map(np.asarray, (actual, reference))
    require(actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape and actual.ndim == 3
            and actual.size and type(chunk) is int and chunk > 0, "full field comparison ABI differs")
    square = signed = actual_energy = reference_energy = dot = 0.
    maximum, mismatches, per_position = 0., 0, []
    for start in range(0, actual.shape[1], chunk):
        a, b = actual[:, start:start + chunk], reference[:, start:start + chunk]
        require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite full field witness")
        mismatches += int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint32) != np.ascontiguousarray(b).view(np.uint32)))
        a, b = a.astype(np.float64), b.astype(np.float64)
        delta = a - b
        square += float(np.sum(delta * delta)); signed += float(delta.sum())
        actual_energy += float(np.sum(a * a)); reference_energy += float(np.sum(b * b)); dot += float(np.sum(a * b))
        maximum = max(maximum, float(np.abs(delta).max()))
        per_position.extend(np.sqrt(np.mean(delta * delta, axis=(0, 2))).tolist())
    count = actual.size
    return {"sample_count": int(count), "bitwise_mismatch_count": mismatches, "bitwise_exact": mismatches == 0,
            "l2": float(np.sqrt(square)), "rms": float(np.sqrt(square / count)), "max_abs": maximum,
            "signed_mean": signed / count, "actual_rms": float(np.sqrt(actual_energy / count)),
            "reference_rms": float(np.sqrt(reference_energy / count)),
            "relative_rms": float(np.sqrt(square / reference_energy)) if reference_energy else None,
            "cosine": dot / np.sqrt(actual_energy * reference_energy) if actual_energy and reference_energy else None,
            "per_position_error_rms": per_position,
            "scope": "Actual minus reference full-field observations; zero reference energy makes relative error undefined, not failed."}


def replacement_recipe(original, used, replacement, *, text_rows, attention_width):
    original, used, replacement = map(np.asarray, (original, used, replacement))
    require(original.ndim == 3 and original.shape == used.shape
            and replacement.shape == (original.shape[0], original.shape[1] - text_rows, original.shape[2] - attention_width)
            and type(text_rows) is int and 0 < text_rows < original.shape[1]
            and type(attention_width) is int and 0 < attention_width < original.shape[2], "replacement consumer partition ABI differs")
    checks = {"text_concat_exact_recipient": exact_bits(original[:, :text_rows], used[:, :text_rows]),
              "image_attention_exact_recipient": exact_bits(original[:, text_rows:, :attention_width], used[:, text_rows:, :attention_width]),
              "image_M_exact_declared_field": exact_bits(used[:, text_rows:, attention_width:], replacement)}
    require(all(checks.values()), "replacement changed an undeclared consumer operand or borrowed another M field")
    return checks


def roll_field(value, *, shift=512):
    value = np.asarray(value)
    require(value.ndim == 3 and value.shape[1] > 0 and type(shift) is int, "invalid wrong-address roll")
    return np.roll(value, shift, axis=1)


def payload_archive_witness(record, payload, expected):
    """Value custody survives producer deduplication across different layouts.

    The producer keys its array archive by native dtype, shape and value hash.
    Its saved stride describes the first archived representative, not every
    later occurrence. The retained used consumer operand proves replacement.
    """
    require(exact_bits(payload,expected) and record["descriptor"]["dtype"] == "torch.bfloat16",
            "consumer payload is not the lossless declared sealed/measured field")
    return {"declared_values_exact":True,"stored_representative_native_stride":record.get("native_stride"),
            "scope":"Stored stride is archive representative metadata only; value-based deduplication may reuse a different occurrence layout. Exact used-M values and unchanged consumer footprint are authoritative."}


def error_effect_ratio(error, effect):
    require(np.isfinite(error) and np.isfinite(effect) and error >= 0 and effect >= 0, "invalid error/effect norm")
    return {"error": float(error), "effect": float(effect), "ratio": float(error / effect) if effect else None,
            "status": "defined" if effect else "undefined-zero-effect",
            "scope": "Diagnostic ratio of declared numerical norms, not a quality gate or semantic score."}


def audit_fresh_builder(parent, e):
    source = parent["source"]
    path, record = ENTRY["referenced_json"](parent, source["builder"])
    b = SavedEvidence(path.parent)
    ENTRY["verify_store_identity"](b, source)
    require(record["schema"] == "mdb-flux-single-context-builder-v1" and record["context"] == source["context"]
            and record["job_id"] == parent["job"]["job_id"] and source["context"]["seed"] == 71
            and source["context"]["id"] == "p1-s71", "replacement fresh builder context/job differs")
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
            "scope": "Actual fresh seed71 own encoding/noise, two typed conditioning Acts, text transplant, native0 image alignment, native1..5. No imported historical states or known prior endpoints; seed provenance source-bound, not an offline RNG replay."}


def current_parent(root, *, custody_verify=None):
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request, job, envelope = (h["read"](root / name) for name in ("request.json", "job.json", "report.json"))
    if custody_verify is None:
        custody_verify = runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    custody = custody_verify(root, limits=request["config"].get("transport_limits"))
    require(custody["mechanics_status"] == "verified" and custody["job_id"] == job["job_id"] == envelope["job_id"]
            and custody["request_sha256"] == envelope["request_sha256"] == request["request_sha256"]
            and envelope["archive_root"] == "flux-mlp-replacement-result", "replacement custody/job/archive differs")
    data = root / "extracted/flux-mlp-replacement-result"
    source, summary, plan, frozen, binding = (h["read"](data / name) for name in
        ("source.json", "summary.json", "plan.json", "plan-frozen.json", "binding.json"))
    require(summary == envelope["report"] and summary["counts"] == GLOBAL_COUNTS
            and summary["panel_counts"] == {key:PANEL_COUNTS[key] for key in ("adapter_steps","acts","receipt_emissions","candidate_calls","images","vae_calls")}
            and plan == request["config"] and plan["branches"] == BRANCHES
            and source["context"] == plan["context"] and source["context"]["seed"] == 71
            and source["job_id"] == summary["job_id"] == job["job_id"]
            and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
            and source["source_hashes"] == request["source_hashes"] and source["model_identity"] == summary["model_identity"],
            "replacement source/plan/global census differs")
    require(frozen["plan_sha256"] == sha_file(data / "plan.json") and frozen["request_sha256"] == request["request_sha256"]
            and frozen["job_id"] == job["job_id"]
            and frozen["source_identity"] == h["sha"](h["canonical"](request["source_hashes"]))
            and all(value == 0 for value in frozen["work"].values())
            and h["sha"](h["canonical"](source["checkpoint_files"])) == source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"],
            "replacement preload plan/checkpoint binding differs")
    verified = h["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    require(SHARED["transaction_verifications_match"](verified, summary["transaction_verification"])
            and verified["phase"] == "closed", "replacement sealed transaction differs")
    return {"root": root, "data": data, "result_root": data, "helpers": h, "request": request, "job": job,
            "custody": custody, "source": source, "summary": summary, "plan": plan, "frozen": frozen,
            "binding": binding, "transaction": verified}


def audit_forecast(parent, e):
    from mdb_lanes.tensor_archive import load_tensor_archive

    h, data, binding = parent["helpers"], parent["data"], parent["binding"]
    tree = h["NumericalTree"](data / "transaction")
    declared = tree.decode(h["read"](data / "transaction/frontiers/source_binding.json")["value"])
    predicted = tree.decode(h["read"](data / "transaction/forecasts/mlp_field.json")["predictions"])
    hypotheses = tree.decode(h["read"](data / "transaction/forecasts/replacement_hypotheses.json")["predictions"])
    preseal = {**GLOBAL_COUNTS, "adapter_steps": 108, "acts": 4, "receipt_emissions": 112,
               "candidate_calls": 0, "images": 0, "vae_calls": 0}
    require(declared == binding and hypotheses == parent["plan"]["hypotheses"]
            and set(predicted["cases"]) == set(binding["inputs"]) == {"zero", "text"}
            and binding["hidden_width"] == 3072 and binding["mlp_width"] == 9216 and binding["eps"] == 1e-6
            and binding["elementwise_affine"] is False and binding["weight_row_range"] == [9216,27648]
            and binding["counts_before_seal"] == preseal and binding["target_entries_before_seal"] == 0
            and binding["query_indices"] == [0,31,32,255,511,512,767,1023]
            and binding["neuron_indices"] == list(range(0,9216,288)), "full-field forecast source/profile/timing differs")
    weights = load_tensor_archive(data, binding["weight_rows"])
    bias = None if binding["bias_rows"] is None else load_tensor_archive(data, binding["bias_rows"])
    require(weights.shape == (18432,3072) and weights.dtype == np.float32
            and binding["linear_path"] == "single_transformer_blocks.6.attn.to_qkv_mlp_proj"
            and binding["weight_rows"]["descriptor"]["dtype"] == "torch.bfloat16"
            and ((bias is None) == ("bias" not in binding["native_parameters"]))
            and binding["native_parameters"]["weight"]["shape"] == [27648,3072]
            and binding["native_parameters"]["weight"]["dtype"] == "torch.bfloat16", "complete paired native weight binding differs")
    fields, reconstruction = {}, {}
    if bias is not None:
        require(bias.shape == (18432,) and binding["bias_rows"]["descriptor"]["dtype"] == "torch.bfloat16"
                and binding["native_parameters"]["bias"]["shape"] == [27648], "complete paired native bias binding differs")
    for role in ("zero", "text"):
        inp, prediction = binding["inputs"][role], predicted["cases"][role]
        image, mod = load_tensor_archive(data, inp["image"]), load_tensor_archive(data, inp["single_mod"])
        require(exact_bits(image, e.slot(inp["cut"], "image"))
                and exact_bits(mod, SHARED["nested_slot_array"](e, inp["cut"], "context", ("single_mod",)))
                and e.point(inp["cut"])["logical_step"] == 3 and e.point(inp["cut"])["index"] == 6,
                "full-field forecast borrowed a different source frontier")
        field = np.asarray(prediction["activation"])
        require(field.dtype == np.float32 and field.shape == (1,1024,9216) and np.isfinite(field).all()
                and not np.any(field.view(np.uint32) & np.uint32(0xffff))
                and prediction["hidden_width"] == 3072 and prediction["mlp_width"] == 9216
                and prediction["neuron_chunk"] == binding["neuron_chunk"], "sealed complete BF16-reference field ABI differs")
        rebuilt = rebuild_full_mlp(image, mod, weights, bias, eps=1e-6, mlp_width=9216,
            neuron_chunk=binding["neuron_chunk"], query_chunk=64, quantize=e.typed["quantize_bf16"])
        reconstruction[role] = {"saved_minus_independent_rebuild": field_error(field, rebuilt),
            "independent_chunks": {"queries":64,"neurons":binding["neuron_chunk"]},
            "arithmetic": prediction["arithmetic"], "scope": prediction["scope"]}
        indices = np.asarray(binding["neuron_indices"])
        selected = np.concatenate((indices, 9216 + indices))
        sample_reference = ENTRY["entry_equations"](image, mod, weights[selected], None if bias is None else bias[selected],
            eps=1e-6, queries=binding["query_indices"], neurons=binding["neuron_indices"], hidden_width=3072,
            mlp_width=9216, quantize=e.typed["quantize_bf16"])["bf16_reference"]
        reconstruction[role]["sample_source_recomputation"] = {name:field_error(np.asarray(prediction["sampled"][name]),
            sample_reference["norm" if name == "layernorm" else name]) for name in ("layernorm","modulated","u","v","activation")}
        fields[role] = field
        del rebuilt
    return fields, predicted, hypotheses, reconstruction


def audit_candidates(parent, e, fields, candidates):
    require(tuple(candidates) == tuple(sorted(ROW_IDS)) or set(candidates) == set(ROW_IDS), "nine direct candidate inventory differs")
    inputs = {role: item["cut"] for role,item in parent["binding"]["inputs"].items()}
    native, images, texts, checks, controls = {}, {}, {}, {}, {}
    for spec in BRANCHES:
        name, base = spec["id"], spec["base"]
        candidate = candidates[name]
        values = {key:e.array(ref) for key,ref in candidate["ports"].items()}
        require(set(values) == {"residual","gate","original_concat","used_concat","projection","updated","image","text"}
                and candidate["input_parent"] == inputs[base] and candidate["text_rows"] == 512 and candidate["attention_width"] == 3072
                and values["original_concat"].shape == values["used_concat"].shape == (1,1536,12288)
                and values["residual"].shape == values["projection"].shape == values["updated"].shape == (1,1536,3072)
                and values["gate"].shape == (1,1,3072)
                and exact_bits(values["residual"][:,:512],e.slot(inputs[base],"text"))
                and exact_bits(values["residual"][:,512:],e.slot(inputs[base],"image"))
                and exact_bits(values["image"],values["updated"][:,512:])
                and exact_bits(values["text"],values["updated"][:,:512]), "candidate native parent/partition ABI differs")
        modulation = SHARED["nested_slot_array"](e,inputs[base],"context",("single_mod",)).reshape(1,1,9216)
        require(exact_bits(values["gate"],modulation[...,6144:]), "candidate gate not actual retained modulation")
        checks[name] = ENTRY["PANEL"]["parallel_residual"](values["residual"],values["gate"],values["projection"],values["updated"],e.typed["quantize_bf16"])
        require(candidate["equation_exact"] is True, "declared candidate residual equation differs")
        require(candidate["partition_controls"] == {key:True for key in
            ("text_concat_exact_recipient","A_image_exact_selected","M_image_exact_selected")}, "declared consumer partition checks differ")
        if name in {"zero","text"}:
            require(candidate["replacement"] is None and candidate["selection"] == "native" and candidate["source_parent"] is None
                    and exact_bits(values["original_concat"],values["used_concat"]), "native source candidate replaced an operand")
            native[name] = values
        else:
            replacement = candidate["replacement"]
            require(all(replacement[key] == spec[key] for key in ("feature","source","roll"))
                    and candidate["selection"] == "M" and candidate["source_parent"] == inputs[spec["source"]], "replacement source/recipe differs")
            expected = fields[spec["source"]] if spec["feature"] == "predicted" else native[spec["source"]]["original_concat"][:,512:,3072:]
            if spec["roll"]:
                expected = roll_field(expected,shift=spec["roll"])
            payload = e.array(replacement["payload"])
            checks[name]["payload_archive"] = payload_archive_witness(replacement["payload"],payload,expected)
            checks[name]["footprint"] = replacement_recipe(values["original_concat"],values["used_concat"],payload,text_rows=512,attention_width=3072)
            controls[name+"_original_concat_exact_recipient"] = exact_bits(values["original_concat"],native[base]["original_concat"])
            controls[name+"_post_text_exact_recipient"] = exact_bits(values["text"],native[base]["text"])
            require(replacement["controls"] == {
                "text_concat_exact_recipient": checks[name]["footprint"]["text_concat_exact_recipient"],
                "image_attention_exact_recipient": checks[name]["footprint"]["image_attention_exact_recipient"],
                "image_mlp_exact_payload": checks[name]["footprint"]["image_M_exact_declared_field"],
                "native_original_concat_exact_recipient": controls[name+"_original_concat_exact_recipient"],
                "post_text_exact_recipient_native": controls[name+"_post_text_exact_recipient"]}, "worker replacement controls differ from retained operands")
        images[name], texts[name] = values["image"], values["text"]
    full_errors = {role:field_error(fields[role],native[role]["original_concat"][:,512:,3072:]) for role in ("zero","text")}
    return native, images, texts, checks, controls, full_errors


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
    expected += [(name,rid) for name in ROW_IDS for rid in paths[name]["receipts"]]
    require([x["sequence"] for x in emitted] == list(range(144)) and [(x["branch"],x["receipt"]) for x in emitted] == expected
            and {rid for _,rid in expected} == set(e.receipts)
            and sum(path["native_steps"] for path in [*prefix_paths.values(),*paths.values()]) == 137
            and sum(path["acts"] for path in paths.values()) == 7, "replacement physical emission census differs")
    controls["sham_latent_exact_zero"] = exact_bits(latents["sham"],latents["zero"])
    controls["sham_png_exact_zero"] = rows["sham"]["image"]["sha256"] == rows["zero"]["image"]["sha256"]
    return rows,paths,prefix_paths,pixels,latents,controls


def audit_timing(parent, observed):
    timeline = SHARED["read_jsonl"](parent["data"] / "timeline.jsonl")
    require([item["sequence"] for item in timeline] == list(range(len(timeline))), "timeline sequence differs")
    sealed = [item for item in timeline if item["kind"] == "full_field_forecast_sealed"]
    starts = [item for item in timeline if item["kind"] == "candidate_start"]
    ends = [item for item in timeline if item["kind"] == "candidate_complete"]
    branches = [item for item in timeline if item["kind"] == "branch_complete"]
    require(len(sealed) == 1 and sealed[0]["seal"] == parent["summary"]["seal"]
            and sealed[0]["counts"] == parent["binding"]["counts_before_seal"]
            and sealed[0]["target_entries_before_seal"] == 0
            and [item["candidate"] for item in starts] == list(ROW_IDS)
            and [item["candidate"] for item in ends] == list(ROW_IDS)
            and [item["row"] for item in branches] == list(ROW_IDS)
            and all(sealed[0]["sequence"] < a["sequence"] < b["sequence"] for a,b in zip(starts,ends))
            and all(starts[i]["counts"]["candidate_calls"] == i for i in range(9))
            and all(ends[i]["counts"]["candidate_calls"] == i+1 for i in range(9))
            and observed["target_calls"] == parent["summary"]["target_calls"] == 11,
            "forecast seal/candidate ordering/target entry census differs")
    return {"seal_event": sealed[0], "candidate_order": list(ROW_IDS), "target_entries_before_seal": 0,
            "direct_calls_after_seal": 9, "independent_prefix_calls_after_seal": 2,
            "scope": "Public sealed transaction plus source-bound guards and physical timeline; not hermetic process isolation."}


def audit_samples(parent, e, predicted, native, candidates):
    record = e.read(e.data / "entry-observations.json")
    binding, cases = parent["binding"], {}
    require(set(record["rows"]) == {"zero","text"} and record["calls"] == 11
            and record["target_entries_before_seal"] == 0, "entry sample target census differs")
    queries, neurons = np.asarray(binding["query_indices"]), np.asarray(binding["neuron_indices"])
    for index,role in enumerate(("zero","text")):
        row, sample = record["rows"][role], predicted["cases"][role]["sampled"]
        require(row["call_index"] == index and sample["query_indices"] == binding["query_indices"]
                and sample["neuron_indices"] == binding["neuron_indices"], "entry sample role/selector differs")
        stages = {}
        for key,source in (("layernorm","layernorm"),("modulated_input","modulated"),("u","u"),("v","v"),("activation","activation")):
            actual, forecast = e.array(row[key]), np.asarray(sample[source])
            shape = (1,8,3072 if key in {"layernorm","modulated_input"} else 32)
            require(actual.shape == forecast.shape == shape and actual.dtype == forecast.dtype == np.float32
                    and row[key]["descriptor"]["dtype"] == "torch.bfloat16" and np.isfinite(forecast).all(), "entry sample arithmetic ABI differs")
            stages[key] = {"native_minus_sealed_reference": field_error(actual,forecast), "actual":row[key]}
        observed_m = native[role]["original_concat"][:,queries+512][:,:,neurons+3072]
        sealed_m = predicted["cases"][role]["activation"][:,queries][:,:,neurons]
        require(exact_bits(e.array(row["activation"]), observed_m) and exact_bits(sample["activation"],sealed_m),
                "sampled activation does not join full native/sealed M field")
        cases[role] = {"call_index":index,"input_cut":candidates[role]["input_parent"],"stages":stages}
    return {"cases":cases,"calls":11,
            "scope":"Native stages from first two source calls only. SiLU-only intermediate is predicted, not separately captured."}


def endpoint_analysis(images, pixels, latents):
    distance = SHARED["BASE"]["distance"]
    comparisons = {name:{base:{"post_image":distance(images[name],images[base]),
        "latent":distance(latents[name],latents[base]),"rgb":distance(pixels[name],pixels[base])}
        for base in ("zero","text")} for name in ROW_IDS}
    pairs = {}
    for label,actual,reference in (("low_drop_in","low.predicted","zero"),("high_drop_in","high.predicted","text"),
            ("high_M_transfer","M.predicted","M.measured"),("rolled_high_M_transfer","M.predicted.roll","M.measured.roll"),
            ("measured_placement","M.measured.roll","M.measured"),("predicted_placement","M.predicted.roll","M.predicted")):
        pairs[label] = {"actual":actual,"reference":reference,"post_image":distance(images[actual],images[reference]),
            "latent":distance(latents[actual],latents[reference]),"rgb":distance(pixels[actual],pixels[reference])}
    ratios = {}
    for label,pred,measured in (("high_M","M.predicted","M.measured"),("rolled_high_M","M.predicted.roll","M.measured.roll")):
        ratios[label] = {key:error_effect_ratio(distance(values[pred],values[measured])[metric],distance(values[measured],values["zero"])[metric])
            for key,values,metric in (("post_image_L2",images,"base_l2"),("latent_L2",latents,"base_l2"),("RGB_MAE",pixels,"base_mae"))}
    return comparisons,pairs,ratios


def analyze(root, *, custody_verify=None):
    import mdb_lanes.tensor_archive as tensor_archive

    reader_path = Path(tensor_archive.__file__)
    reader_sha = sha_file(reader_path)
    parent = current_parent(root,custody_verify=custody_verify)
    h,data = parent["helpers"],parent["data"]
    run = {"root":parent["root"],"data":data,"plan":parent["plan"],"request":parent["request"],
        "source":{**parent["custody"],"transaction":parent["transaction"]},"transaction":data/"transaction",
        "tree":h["NumericalTree"](data/"transaction")}
    e = SavedEvidence(data,run=run)
    ENTRY["verify_store_identity"](e,parent["source"])
    lineage = audit_fresh_builder(parent,e)
    anchor = e.fields(parent["binding"]["inputs"]["zero"]["cut"])
    for cut in e.store["cuts"]:
        fields = e.fields(cut)
        require(fields["conditioning"] == anchor["conditioning"] and all(fields[key] == value for key,value in anchor.items()
            if key not in {"image","text","context","latent","noise_prediction","step","cursor","phase"}),
            "typed constant schedule/conditioning closure differs")
    points = SHARED["BASE"]["receipt_points"](e.receipts.values())
    require(all(cut not in e.store["cuts"] or e.point(cut) == point for cut,point in points.items()), "receipt/cut point disagreement")
    fields,predicted,hypotheses,reconstruction = audit_forecast(parent,e)
    observed = run["tree"].decode(h["read"](data/"transaction/observations/native.json")["value"])
    require(observed["counts"] == GLOBAL_COUNTS and observed["control_observations"] == parent["summary"]["control_observations"]
            and all(observed[key+"_sha256"] == sha_file(data/file) for key,file in
                (("source","source.json"),("candidates","candidates.json"),("entry_observations","entry-observations.json"))),
            "native transaction saved evidence/hash binding differs")
    timing = audit_timing(parent,observed)
    candidates = e.read(data/"candidates.json")["candidates"]
    native,images,texts,checks,replacement_controls,full_errors = audit_candidates(parent,e,fields,candidates)
    samples = audit_samples(parent,e,predicted,native,candidates)
    prefixes = e.read(data/"prefixes.json")["prefixes"]
    rows,paths,prefix_paths,pixels,latents,controls = audit_paths(parent,e,images,texts,observed["rows"],prefixes)
    declared = {key:value for key,value in controls.items() if key not in {"sham_latent_exact_zero","sham_png_exact_zero"}}
    declared.update({key+"_source_exact":lineage["pair_observations"][key+"_exact"] for key in ("image","text")})
    declared["sham_png_equals_zero"] = controls["sham_png_exact_zero"]
    require(declared == parent["summary"]["control_observations"], "named worker replay observations differ from saved evidence")
    controls.update(replacement_controls)
    comparisons,pairs,ratios = endpoint_analysis(images,pixels,latents)
    measured_effect = field_error(native["text"]["original_concat"][:,512:,3072:],native["zero"]["original_concat"][:,512:,3072:])
    ratios["full_M_field_high_vs_low"] = error_effect_ratio(full_errors["text"]["l2"],measured_effect["l2"])
    relative = data.relative_to(parent["root"]).as_posix()
    exported = []
    for name in ROW_IDS:
        row = rows[name]
        exported.append({**{key:row[key] for key in ("id","base","feature","source","roll","parent","final_cut")},
            "image":{**row["image"],"path":relative+"/"+row["image"]["path"]},
            "final_latent":{**row["final_latent"],"path":relative+"/"+row["final_latent"]["path"]},
            "row_record":{"path":relative+"/rows/"+name+".json","sha256":sha_file(data/"rows"/(name+".json"))},
            "comparisons":comparisons[name],"native_path":paths[name],"acts":row["acts"]})
    require(sha_file(reader_path) == reader_sha, "installed tensor reader changed during audit")
    replay_checks = {key:controls[key] for key in (
        "zero_candidate_image_exact_native","zero_candidate_text_exact_native",
        "text_candidate_image_exact_native","text_candidate_text_exact_native",
        "sham_latent_exact_zero","sham_png_exact_zero")}
    replay = not [key for key,value in replay_checks.items() if value is not True]
    return {"schema":"mdb-flux-mlp-replacement-audit-v1","mechanics_status":"valid",
        "instrument_status":"native-replay-qualified" if replay else "replay-unresolved","terminal_status":"unassessed",
        "trend_status":"One fresh seed71 full-field replacement assay; all numerical discrepancies, native futures and signed countertrends retained.",
        "source":{**parent["custody"],"source_record_sha256":sha_file(data/"source.json"),"staged_source_hashes":parent["request"]["source_hashes"],
            "global_plan_freeze":parent["frozen"],"custody_verifications":1},
        "execution_status":{"job_id":parent["job"]["job_id"],"job_state":parent["job"]["state"],"model_loads":1},
        "counts":GLOBAL_COUNTS,"panel_counts":PANEL_COUNTS,"fresh_builder":lineage,"typed_store_counts":e.store["counts"],
        "physical_census":{"builder_steps":108,"prefix_steps":2,"suffix_steps":135,"image_acts":7,"global_acts":11,"global_receipts":256,
            "scope":"Physical ordered emissions checked before receipt identity deduplication; nine direct calls and nine VAE decodes."},
        "seal":parent["summary"]["seal"],"transaction_verification":parent["transaction"],
        "saved_worker_transaction_verification":parent["summary"]["transaction_verification"],"timing":timing,"sealed_hypotheses":hypotheses,
        "full_mlp_forecast":{"source_binding":parent["binding"],"independent_reconstruction":reconstruction,
            "sealed_minus_native_M":full_errors,"measured_high_minus_low_M":measured_effect,
            "scope":"Complete BF16-reference image M field sealed before current-step target entries. Native outputs are audited observations; numerical mismatch never discards futures."},
        "entry_samples":samples,"candidate_checks":checks,"control_observations":controls,
        "replay_qualification":{"status":"qualified" if replay else "unresolved","checks":replay_checks,
            "scope":"Named native direct-capture versus independent prefix and unchanged-sham replay mechanics only. Numerical forecast errors, source contrasts, placement effects and endpoint rankings have no authority in this bundle."},
        "native_prefixes":prefix_paths,"native_paths":paths,"comparisons":comparisons,"matched_pairs":pairs,
        "error_effect_diagnostics":ratios,"rows":exported,"nested_audit_tensor_reference_base":relative,
        "reference_scopes":{"rows":"image/final_latent/row_record paths relative to run directory",
            "nested_arrays":"all binding, native stage and candidate array paths relative to nested_audit_tensor_reference_base"},
        "auditor_source_sha256":sha_file(Path(__file__)),"tensor_archive_source_sha256":reader_sha,
        "audit_helpers":{name:sha_file(HERE/name) for name in ("analyze_flux_single_mlp.py","analyze_flux_single_contraction.py",
            "analyze_flux_single_contexts.py","analyze_flux_single_kv.py","analyze_flux_rescue_routes.py","summarize_context_window.py",
            "analyze_flux_mlp_forecast.py","analyze_flux_concept_routes.py","flux_family_custody.py")},"models_executed":False,
        "limits":["One fresh seed71 prompt history and nine dependent futures; no universal, semantic or isolated text-bank causal claim.",
            "Native MLP and attention still execute. Predicted M replaces only image M at the actual consumer input; no skipped-compute or speed claim.",
            "Both source image and text may differ. Forecast inputs are observed incoming state, time modulation and all actual paired weights; no native current-target M enters prediction.",
            "Explicit BF16 rounding reference with FP64 linear sums is not an exact CUDA kernel emulator. Error/effect ratios are diagnostics; zero effect is undefined.",
            "Native futures/readout/RGB are observations, not independent full-model forecasts. Roll512 preserves values and changes token placement; no destructive-null or semantic spatial guarantee.",
            "Weight row extraction, random seed, execution guards and public receipt footprints are source-bound authority, not independent kernel tracing or parameter Merkle proof.",
            "Array archives deduplicate by dtype, shape and value hash. Stored native_stride describes the first archived representative and cannot prove a later occurrence layout; full used-M values and consumer footprints are verified independently."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    value = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:value[key] for key in ("mechanics_status","instrument_status","counts")}))
