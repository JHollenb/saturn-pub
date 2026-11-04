"""Independent saved-only audit of an autonomous two-block FLUX prediction.

Source, typed closure and chain provenance are integrity checks. Numerical
discrepancies are observations with no precision threshold or error floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = runpy.run_path(str(HERE / "analyze_flux_attention_replacement.py"))
require, exact_bits, quantize = (PREVIOUS[k] for k in ("require", "exact_bits", "quantize"))
field_error, SavedEvidence = PREVIOUS["field_error"], PREVIOUS["SavedEvidence"]
SHARED, COMMON, ENTRY = (PREVIOUS[k] for k in ("SHARED", "COMMON", "ENTRY"))
sha_file = PREVIOUS["sha_file"]
CONTEXT = {"id": "p1-s79", "prompt_id": "p1", "seed": 79,
    "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
SAMPLE_INDICES = [0, 31, 511, 512, 543, 767, 1023, 1535]
BRANCHES = [{"id": name, "base": base, "mode": mode} for name, base, mode in (
    ("low.native", "zero", "native"), ("high.native", "text", "native"), ("low.sham", "zero", "sham"),
    ("low.pred1", "zero", "pred1"), ("low.pred2", "zero", "pred2"),
    ("high.pred1", "text", "pred1"), ("high.pred2", "text", "pred2"))]
ORDER = [row["id"] for row in BRANCHES]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 212,
    "acts": 14, "candidate_calls": 0, "receipt_emissions": 226, "images": 7, "vae_calls": 7, "physical_batch": 1}
PANEL_COUNTS = {"adapter_steps": 104, "acts": 10, "receipt_emissions": 114, "candidate_calls": 0, "images": 7, "vae_calls": 7}


def tensor_identity(value):
    value = np.asarray(value)
    require(value.dtype == np.float32 and value.size and np.isfinite(value).all()
            and not np.any(np.ascontiguousarray(value).view(np.uint32) & 65535), "BF16 field ABI differs")
    return {"shape": list(value.shape), "dtype": "float32-exact-bf16-widening",
            "bf16_le_sha256": hashlib.sha256((np.ascontiguousarray(value).view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest(),
            "fp32_le_sha256": hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()}


def verify_predicted_chain(chain, initial, *, text_rows, sample_indices):
    """Require the saved full prediction to carry its own image AND text."""
    current, outputs = np.asarray(initial), []
    require(chain["chain_length"] == 2 and len(chain["steps"]) == 2
            and chain["input_descriptor"] == tensor_identity(current), "chain initial binding differs")
    for index, step in enumerate(chain["steps"]):
        prediction = step["prediction"]
        descriptor = tensor_identity(current)
        require(step["step"] == index and step["block_index"] == 6 + index
                and step["input_source"] == ("caller pre-entry residual" if index == 0 else "previous predicted full output")
                and step["input_descriptor"] == prediction["input_descriptor"] == descriptor,
                "chain borrowed or misdeclared an intermediate input")
        sampled = prediction["sampled"]
        require(sampled["indices"] == sample_indices and exact_bits(sampled["input"], current[:, sample_indices]),
                "chain sampled input does not join previous complete prediction")
        output = np.asarray(prediction["output"])
        require(output.shape == initial.shape and prediction["output_descriptor"] == tensor_identity(output)
                and exact_bits(sampled["output"], output[:, sample_indices])
                and prediction["text_rows"] == text_rows
                and exact_bits(prediction["text"], output[:, :text_rows])
                and exact_bits(prediction["image"], output[:, text_rows:]), "chain complete output/splits differ")
        outputs.append(output)
        current = output
    require(chain["output_descriptor"] == tensor_identity(current) and exact_bits(chain["output"], current)
            and exact_bits(chain["text"], current[:, :text_rows]) and exact_bits(chain["image"], current[:, text_rows:]),
            "chain final output alias differs")
    return outputs


def independent_transition(residual, modulation, block, *, heads, layernorm_eps,
                           qk_eps, rope_cos, rope_sin, query_chunk=32, row_chunk=512):
    """Own NumPy full-block recurrence, independent of the campaign predictor."""
    r, mod = np.asarray(residual), np.asarray(modulation)
    require(r.ndim == 3 and r.shape[0] == 1 and type(heads) is int and heads > 0
            and r.shape[-1] % heads == 0 and (r.shape[-1] // heads) % 2 == 0
            and type(query_chunk) is int and query_chunk > 0 and type(row_chunk) is int and row_chunk > 0,
            "transition geometry differs")
    length, width = r.shape[1:]
    dim = width // heads
    names = ("qkv_mlp_weight", "to_out_weight", "norm_q_weight", "norm_k_weight")
    w, out_w, gamma_q, gamma_k = [np.asarray(block[k]) for k in names]
    require(w.ndim == 2 and w.shape[1] == width and (w.shape[0] - 3 * width) > 0
            and (w.shape[0] - 3 * width) % 2 == 0, "fused transition weight geometry differs")
    middle = (w.shape[0] - 3 * width) // 2
    require(out_w.shape == (width, width + middle) and gamma_q.shape == gamma_k.shape == (dim,)
            and mod.shape in {(1, 3 * width), (1, 1, 3 * width)}
            and all(type(v) in {int, float} and np.isfinite(v) and v > 0 for v in (layernorm_eps, qk_eps)),
            "transition parameters differ")
    for x in (r, mod, w, out_w, gamma_q, gamma_k):
        tensor_identity(x)
    cos, sin = np.asarray(rope_cos), np.asarray(rope_sin)
    require(cos.shape == sin.shape == (length, dim) and cos.dtype == sin.dtype == np.float32
            and np.isfinite(cos).all() and np.isfinite(sin).all(), "transition position context differs")
    biases = []
    for key, size in (("qkv_mlp_bias", w.shape[0]), ("to_out_bias", width)):
        value = block.get(key)
        if value is not None:
            value = np.asarray(value)
            require(value.shape == (size,), "transition bias geometry differs")
            tensor_identity(value)
        biases.append(value)
    shift, scale, gate = np.split(mod.reshape(1, 1, 3 * width), 3, axis=-1)
    wide = r.astype(np.float64)
    centered = wide - wide.mean(axis=-1, keepdims=True)
    norm = quantize(centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + layernorm_eps))
    affine = quantize(quantize(quantize(1 + scale) * norm).astype(np.float64) + shift)
    affine64 = affine.astype(np.float64)
    projected = np.empty((1, length, w.shape[0]), np.float32)
    for first in range(0, w.shape[0], row_chunk):
        last = min(first + row_chunk, w.shape[0])
        value = affine64 @ w[first:last].astype(np.float64).T
        if biases[0] is not None:
            value += biases[0][first:last].astype(np.float64)
        projected[..., first:last] = quantize(value)
    q, k, v = [x.reshape(1, length, heads, dim) for x in np.split(projected[..., :3 * width], 3, axis=-1)]
    rotated = []
    for x, gamma in ((q, gamma_q), (k, gamma_k)):
        value = quantize(x.astype(np.float64) / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + qk_eps)
                         * gamma.astype(np.float64))
        partner = np.stack((-value[..., 1::2], value[..., 0::2]), axis=-1).reshape(value.shape)
        rotated.append(quantize(np.add(np.multiply(value, cos[None, :, None], dtype=np.float32),
                                      np.multiply(partner, sin[None, :, None], dtype=np.float32), dtype=np.float32)))
    attention = np.empty((1, length, width), np.float32)
    keys = rotated[1].astype(np.float64).transpose(0, 2, 3, 1)
    values = v.astype(np.float64).transpose(0, 2, 1, 3)
    for first in range(0, length, query_chunk):
        last = min(first + query_chunk, length)
        query = rotated[0][:, first:last].astype(np.float64).transpose(0, 2, 1, 3)
        score = (query @ keys) / np.sqrt(float(dim))
        score -= score.max(axis=-1, keepdims=True)
        probability = np.exp(score)
        probability /= probability.sum(axis=-1, keepdims=True)
        attention[:, first:last] = quantize((probability @ values).transpose(0, 2, 1, 3).reshape(1, last - first, width))
    u, v_mlp = np.split(projected[..., 3 * width:], 2, axis=-1)
    positive = u >= 0
    sigmoid = np.empty(u.shape, np.float64)
    sigmoid[positive] = 1 / (1 + np.exp(-u[positive].astype(np.float64)))
    exp_u = np.exp(u[~positive].astype(np.float64))
    sigmoid[~positive] = exp_u / (1 + exp_u)
    mlp = quantize(quantize(u.astype(np.float64) * sigmoid) * v_mlp)
    concat = np.concatenate((attention, mlp), axis=-1)
    concat64 = concat.astype(np.float64)
    projection = np.empty(r.shape, np.float32)
    for first in range(0, width, row_chunk):
        last = min(first + row_chunk, width)
        value = concat64 @ out_w[first:last].astype(np.float64).T
        if biases[1] is not None:
            value += biases[1][first:last].astype(np.float64)
        projection[..., first:last] = quantize(value)
    write = quantize(gate * projection)
    output = quantize(r + write)
    tensor_identity(output)
    return output


def vector_decomposition(predicted, propagated, native):
    """Exact saved-vector decomposition; no attribution from scalar distances."""
    p, c, n = [np.asarray(x) for x in (predicted, propagated, native)]
    require(p.dtype == c.dtype == n.dtype == np.float32 and p.shape == c.shape == n.shape
            and p.size and all(np.isfinite(x).all() for x in (p, c, n)), "decomposition field ABI differs")
    local, carried = p.astype(np.float64) - c, c.astype(np.float64) - n
    total = p.astype(np.float64) - n
    stats = lambda x: {"l2": float(np.linalg.norm(x.ravel())), "rms": float(np.sqrt(np.mean(x * x))),
                       "max_abs": float(np.abs(x).max()), "signed_mean": float(x.mean())}
    return {"formula": "predicted7-native7 = (predicted7-native7(predicted6)) + (native7(predicted6)-native7)",
            "local": stats(local), "carried": stats(carried), "total": stats(total),
            "local_dot_carried": float(np.sum(local * carried)),
            "closure_max_abs": float(np.abs(total - local - carried).max()),
            "scope": "Saved full-vector identity, not a unique causal or semantic attribution."}


def current_parent(root, *, custody_verify=None):
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request, job, envelope = [h["read"](root / name) for name in ("request.json", "job.json", "report.json")]
    verify = custody_verify or runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    custody = verify(root, limits=request["config"].get("transport_limits"))
    require(custody["mechanics_status"] == "verified" and custody["job_id"] == job["job_id"] == envelope["job_id"]
            and custody["request_sha256"] == request["request_sha256"] == envelope["request_sha256"]
            and envelope["archive_root"] == "flux-transition-chain-result", "chain custody/job/archive differs")
    for name, digest in request["source_hashes"].items():
        require(sha_file(h["file_path"](root / "payload", name)) == digest, "staged source digest differs")
    data = root / "extracted/flux-transition-chain-result"
    source, summary, plan, frozen, binding = [h["read"](data / name) for name in
        ("source.json", "summary.json", "plan.json", "plan-frozen.json", "binding.json")]
    require(summary == envelope["report"] and summary["counts"] == COUNTS and summary["panel_counts"] == PANEL_COUNTS
            and plan == request["config"] and plan["branches"] == BRANCHES and plan["branch_execution_order"] == ORDER
            and source["context"] == plan["context"] == CONTEXT and plan["block_indices"] == [6, 7]
            and plan["selector"] == {"logical_step": 3, "phase": "single", "index": 6}
            and (plan["height"], plan["width"], plan["steps"]) == (512, 512, 4)
            and plan["sample_indices"] == SAMPLE_INDICES and plan["retain_fields"] is False
            and source["job_id"] == summary["job_id"] == job["job_id"]
            and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
            and source["source_hashes"] == request["source_hashes"]
            and source["model_identity"] == summary["model_identity"]
            and source["execution"]["device"]["type"] == request["device"] == "cuda"
            and source["execution"]["dtype"] == "torch.bfloat16" and request["dtype"] == "bfloat16",
            "chain source/plan/census differs")
    require(frozen["plan_sha256"] == sha_file(data / "plan.json") and frozen["request_sha256"] == request["request_sha256"]
            and frozen["job_id"] == job["job_id"] and all(value == 0 for value in frozen["work"].values())
            and frozen["source_identity"] == h["sha"](h["canonical"](request["source_hashes"]))
            and h["sha"](h["canonical"](source["checkpoint_files"])) == source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"],
            "chain preloading/checkpoint declaration differs")
    transaction = h["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    require(transaction["phase"] == "closed"
            and SHARED["transaction_verifications_match"](transaction, summary["transaction_verification"]), "chain transaction differs")
    return {"root": root, "data": data, "result_root": data, "helpers": h, "request": request, "job": job,
            "custody": custody, "source": source, "summary": summary, "plan": plan,
            "frozen": frozen, "binding": binding, "transaction": transaction}


def audit_forecasts(parent, e):
    from mdb.tensor_archive import load_tensor_archive

    h, data, binding = parent["helpers"], parent["data"], parent["binding"]
    tree = h["NumericalTree"](data / "transaction")
    decoded = lambda file, field: tree.decode(h["read"](data / "transaction" / file)[field])
    declared = decoded("frontiers/source_binding.json", "value")
    forecast = decoded("forecasts/transition_chain.json", "predictions")
    hypotheses = decoded("forecasts/chain_hypotheses.json", "predictions")
    expected = {**COUNTS, "adapter_steps": 108, "acts": 4, "receipt_emissions": 112, "images": 0, "vae_calls": 0}
    require(declared == binding and hypotheses == parent["plan"]["hypotheses"]
            and binding["counts_before_seal"] == expected and binding["target_entries_before_seal"] == {"6": 0, "7": 0}
            and binding["block_indices"] == [6, 7] and set(binding["blocks"]) == {"6", "7"}
            and set(binding["inputs"]) == set(forecast["cases"]) == {"zero", "text"}
            and binding["sample_indices"] == SAMPLE_INDICES, "chain sealed declaration differs")
    parameter_names = {"qkv_mlp_weight": "attn.to_qkv_mlp_proj.weight", "qkv_mlp_bias": "attn.to_qkv_mlp_proj.bias",
        "to_out_weight": "attn.to_out.weight", "to_out_bias": "attn.to_out.bias",
        "norm_q_weight": "attn.norm_q.weight", "norm_k_weight": "attn.norm_k.weight"}
    blocks = {}
    for index in (6, 7):
        b = binding["blocks"][str(index)]
        require((b["hidden_width"], b["mlp_width"], b["heads"], b["head_dim"], b["layernorm_eps"], b["qk_eps"])
                == (3072, 9216, 24, 128, 1e-6, 1e-6) and set(b["parameters"]) == set(parameter_names),
                "chain full block geometry differs")
        profile = b["attention_profile"]
        require(profile["processor_type"] == "diffusers.models.transformers.transformer_flux2.Flux2ParallelSelfAttnProcessor"
                and all(profile[k] is None for k in ("processor_backend", "parallel_config", "attention_mask", "joint_attention_kwargs"))
                and profile["is_causal"] is False and profile["kv_cache"] is False, "chain attention ABI differs")
        params = {}
        for key, native_name in parameter_names.items():
            record = b["parameters"][key]
            if record is None:
                require(key.endswith("bias") and native_name not in b["native_parameters"], "native bias declaration differs")
                params[key] = None
                continue
            require(native_name in b["native_parameters"] and all(record["descriptor"][k] == b["native_parameters"][native_name][k]
                    for k in ("shape", "dtype", "sha256")), "full block parameter bytes differ from native binding")
            params[key] = load_tensor_archive(data, record)
            tensor_identity(params[key])
        blocks[index] = params
    cases, report = {}, {}
    for role in ("zero", "text"):
        inp, chain = binding["inputs"][role], forecast["cases"][role]
        verify_clock_and_data(e, inp["cut"], inp)
        load = lambda name: load_tensor_archive(data, inp[name])
        residual, mod, cos, sin = [load(name) for name in ("residual", "single_mod", "rope_cos", "rope_sin")]
        require(residual.shape == (1, 1536, 3072) and inp["text_rows"] == 512 and inp["point"] == e.point(inp["cut"])
                and e.point(inp["cut"])["logical_step"] == 3 and e.point(inp["cut"])["index"] == e.point(inp["cut"])["local_clock"] == 6
                and exact_bits(residual, np.concatenate((e.slot(inp["cut"], "text"), e.slot(inp["cut"], "image")), axis=1))
                and exact_bits(mod, PREVIOUS["nested_slot_array"](e, inp["cut"], "context", ("single_mod",)))
                and exact_bits(cos, PREVIOUS["nested_slot_array"](e, inp["cut"], "context", ("rope", 0)))
                and exact_bits(sin, PREVIOUS["nested_slot_array"](e, inp["cut"], "context", ("rope", 1))),
                "chain initial full state/context borrowed another source")
        outputs = verify_predicted_chain(chain, residual, text_rows=512, sample_indices=SAMPLE_INDICES)
        rebuilt, errors = residual, {}
        for stage, index in enumerate((6, 7)):
            rebuilt = independent_transition(rebuilt, mod, blocks[index], heads=24, layernorm_eps=1e-6, qk_eps=1e-6,
                                             rope_cos=cos, rope_sin=sin)
            errors[str(index)] = field_error(outputs[stage], rebuilt)
        cases[role] = outputs
        report[role] = {"sealed_minus_independent_free_chain": errors,
            "chain_input_identity": chain["input_descriptor"], "sealed_step_input_sources": [s["input_source"] for s in chain["steps"]],
            "scope": "Independent source-only all-token whole-block recurrence; block7 consumes independent predicted block6. Rebuild differences are observations, not thresholds."}
    return cases, report, hypotheses


def stable_fields(e, cut):
    return {key: value for key, value in e.fields(cut).items() if key not in {"image", "text", "phase", "cursor"}}


def verify_clock_and_data(e, cut, record):
    fields = e.fields(cut)
    point = e.point(cut)
    require(record["point"] == point, "boundary point differs")
    require(record["clock_state"] == {key: e.typed["unpack"](fields[key]) for key in ("step", "phase", "cursor")},
            "boundary clock-state witness differs")
    for key, expected in (("phase", point["phase"]), ("cursor", point["index"])):
        if key in fields:
            require(e.typed["unpack"](fields[key]) == expected, "routing slot differs from execution point")
    described = record["non_stream_state"]
    payload = e.store["cuts"][cut]["payload"]
    require(described["kind"] == payload["kind"] == "mapping", "non-stream state is not a mapping")
    declared_keys = {item["key"]["value"] for item in described["items"]}
    required_keys = set(fields) - {"image", "text", "phase", "cursor"}
    require(required_keys <= declared_keys <= set(fields) - {"image", "text"}
            and described == {**payload, "items": [item for item in payload["items"] if item["key"]["value"] in declared_keys]},
            "boundary non-stream description differs from complete retained cut")


def boundary_arrays(e, record, *, source_cut):
    cut = record["cut"]
    verify_clock_and_data(e, cut, record)
    require(stable_fields(e, cut) == stable_fields(e, source_cut), "native boundary changed non-stream data")
    image, text = (e.array(record[key]) for key in ("image", "text"))
    require(image.shape == (1, 1024, 3072) and text.shape == (1, 512, 3072)
            and exact_bits(image, e.slot(cut, "image")) and exact_bits(text, e.slot(cut, "text")),
            "boundary full image/text arrays differ from typed cut")
    return np.concatenate((text, image), axis=1)


def audit_native_paths(parent, e, predictions, observed):
    data = e.data
    native_record = e.read(data / "native-boundaries.json")
    prefixes = native_record["prefixes"]
    require(set(prefixes) == {"zero", "text"}, "native reference prefix inventory differs")
    native, prefix_paths = {}, {}
    for role in ("zero", "text"):
        prefix, source_cut = prefixes[role], parent["binding"]["inputs"][role]["cut"]
        require(prefix["input_cut"] == source_cut, "native prefix borrowed a different source")
        path = e.walk(source_cut, prefix["post7"]["cut"], limit=3)
        require(path["native_steps"] == 2 and path["acts"] == 0
                and prefix["receipts"] == [e.receipts[rid] for rid in path["receipts"]]
                and prefix["post6"]["cut"] == prefix["receipts"][0]["result"], "native block6/7 lineage differs")
        native[role] = {}
        for index in (6, 7):
            record = prefix["post" + str(index)]
            point = e.point(record["cut"])
            require(all(point[key] == value for key, value in {"logical_step": 3, "phase": "single", "index": index + 1, "local_clock": index + 1}.items()),
                    "native reference boundary clock differs")
            native[role][index] = boundary_arrays(e, record, source_cut=source_cut)
        prefix_paths[role] = path
    rows = {row["id"]: row for row in observed["rows"]}
    require(len(rows) == len(observed["rows"]) == 7 and set(rows) == set(ORDER), "chain row inventory differs")
    common_record = e.read(data / "common-boundaries.json")["rows"]
    require(set(common_record) == set(ORDER), "common boundary inventory differs")
    paths, common, pixels, latents = {}, {}, {}, {}
    retained = {cut: e.fields(cut) for cut in e.store["cuts"]}
    for spec in BRANCHES:
        name, role, mode = (spec[key] for key in ("id", "base", "mode"))
        row = rows[name]
        source_cut = parent["binding"]["inputs"][role]["cut"]
        boundary = 6 if mode == "pred1" else 7
        expected_parent = prefixes[role]["post" + str(boundary)]["cut"]
        require(row == e.read(data / "rows" / (name + ".json")) and all(row[k] == v for k, v in spec.items())
                and row["parent"] == expected_parent and row["start_point"] == e.point(expected_parent)
                and row["installation_boundary"] == boundary
                and row["continuation_policy"] == "recipient-stored-C-no-further-project"
                and [a["slot"] for a in row["acts"]] == ([] if mode == "native" else ["image", "text"]),
                "chain row source/installation contract differs")
        current = expected_parent
        for action in row["acts"]:
            slot = action["slot"]
            expected = (e.slot(expected_parent, slot) if mode == "sham" else
                        predictions[role][boundary - 6][:, :512] if slot == "text" else predictions[role][boundary - 6][:, 512:])
            current = e.act(action, current, expected)
        if mode != "native":
            require(row["installed"]["cut"] == current, "installed boundary does not join both sequential Acts")
            installed = boundary_arrays(e, row["installed"], source_cut=source_cut)
            require(exact_bits(installed, native[role][7] if mode == "sham" else predictions[role][boundary - 6]),
                    "full image/text installation differs from sealed prediction")
        bridge = row["bridge_receipts"]
        require(len(bridge) == (1 if mode == "pred1" else 0), "native block7 bridge census differs")
        if bridge:
            require(bridge[0] == e.receipts[bridge[0]["fingerprint"]] and bridge[0]["parent"] == current
                    and bridge[0]["operation"] == "continue" and bridge[0]["steps"] == 1,
                    "predicted block6 did not feed actual native block7")
            current = bridge[0]["result"]
        require(row["common_post7"] == common_record[name] and row["common_post7"]["cut"] == current
                and all(e.point(current)[key] == value for key, value in {"logical_step": 3, "phase": "single", "index": 8, "local_clock": 8}.items()),
                "row did not meet at common post7 clock")
        common[name] = boundary_arrays(e, row["common_post7"], source_cut=source_cut)
        path = e.walk(expected_parent, row["final_cut"], row["acts"], limit=20)
        native_receipts = [e.receipts[rid] for rid in path["receipts"] if e.receipts[rid]["operation"] == "continue"]
        require(path["native_steps"] == 14 + len(bridge) and path["acts"] == len(row["acts"])
                and native_receipts == bridge + row["continuation_receipts"] and len(row["continuation_receipts"]) == 14
                and e.point(row["final_cut"])["logical_step"] == 4 and e.point(row["final_cut"])["phase"] == "project",
                "common14-transition continuation/bridge differs")
        path["conditioning_coverage"] = SHARED["BASE"]["conditioning_coverage"](
            path, e.receipts, retained, e.fields(source_cut)["conditioning"], e.point)
        latents[name] = e.array(row["final_latent"])
        require(exact_bits(latents[name], e.slot(row["final_cut"], "latent")), "final latent differs from native cut")
        pixels[name], paths[name] = e.png(row["image"]), path
    emitted = SHARED["read_jsonl"](data / "receipt-emissions.jsonl")
    expected = [("prefix." + role, rid) for role in ("zero", "text") for rid in prefix_paths[role]["receipts"]]
    expected += [(name, rid) for name in ORDER for rid in paths[name]["receipts"]]
    require([item["sequence"] for item in emitted] == list(range(114))
            and [(item["branch"], item["receipt"]) for item in emitted] == expected
            and {rid for _, rid in expected} == set(e.receipts)
            and sum(p["native_steps"] for p in [*prefix_paths.values(), *paths.values()]) == 104
            and sum(p["acts"] for p in paths.values()) == 10, "physical panel receipt occurrence census differs")
    target_calls = {str(i): sum(1 for _, rid in expected if e.receipts[rid]["operation"] == "continue"
        and all(e.point(e.receipts[rid]["parent"])[k] == v for k, v in {"logical_step": 3, "phase": "single", "index": i}.items())) for i in (6, 7)}
    require(target_calls == parent["summary"]["target_calls"] == {"6": 2, "7": 4}, "physical target-entry census differs")
    controls = {"sham_png_exact_low": rows["low.sham"]["image"]["sha256"] == rows["low.native"]["image"]["sha256"],
        "sham_final_latent_exact_low": exact_bits(latents["low.sham"], latents["low.native"])}
    require(controls == observed["control_observations"] == parent["summary"]["control_observations"], "saved replay control observations differ")
    controls["sham_common_post7_exact_low"] = exact_bits(common["low.sham"], common["low.native"])
    return rows, paths, prefix_paths, native, common, pixels, latents, controls


def audit_observer(parent, e, rows, prefixes):
    record = e.read(e.data / "continuation-observations.json")
    plan = parent["plan"]["observation_plan"]
    references = ["low.native", "high.native"]
    slots = ("image", "text", "latent", "noise_prediction")
    require(record["schema"] == "mdb-flux-chain-continuation-observations-v1" and record["plan"] == plan
            and record["branch_execution_order"] == ORDER and record["reference_rows"] == references == plan["reference_rows"]
            and plan["boundaries_per_branch"] == 15 and plan["witness_count"] == 64
            and plan["slots"] == list(slots) and record["model_calls_added"] == 0 and set(record["rows"]) == set(ORDER),
            "chain continuation observation ABI differs")
    full, reports = {}, {}
    for name in ORDER:
        row, trajectory = rows[name], record["rows"][name]
        require(len(trajectory) == 15, "chain observer missing a continuation boundary")
        suffix = row["continuation_receipts"]
        initial_receipt = (row["bridge_receipts"] or row["acts"] or prefixes[row["base"]]["receipts"])[-1]
        incoming = initial_receipt.get("receipt_id", initial_receipt.get("fingerprint"))
        available_refs = [r for r in references if ORDER.index(r) <= ORDER.index(name)]
        z3 = e.slot(parent["binding"]["inputs"][row["base"]]["cut"], "latent")
        reports[name] = []
        for index, item in enumerate(trajectory):
            cut = row["common_post7"]["cut"] if index == 0 else suffix[index - 1]["result"]
            receipt = incoming if index == 0 else suffix[index - 1]["fingerprint"]
            require(item["index"] == index and item["cut"] == cut and item["point"] == e.point(cut)
                    and item["incoming_receipt"] == receipt and item["complete_cut_retained"] == (index in {0, 14})
                    and set(item["slots"]) == set(slots) and set(item["comparisons"]) == set(available_refs),
                    "chain observer receipt/clock/retention differs")
            samples, checked_slots = {}, {}
            for slot in slots:
                value = item["slots"][slot]
                require(type(value["present"]) is bool and value["meaning"] == PREVIOUS["slot_meaning"](item["point"], slot),
                        "stale trajectory slot mislabeled as meaningful propagation")
                retained = None
                if cut in e.store["cuts"]:
                    require(value["present"] == (e.fields(cut).get(slot, {}).get("kind") == "tensor"), "retained slot presence differs")
                    if value["present"]:
                        retained = e.slot(cut, slot)
                        require(e.typed["matches"](retained, value["descriptor"], bf16=True), "retained slot descriptor differs")
                if not value["present"]:
                    require("array" not in value and "witness" not in value, "absent trajectory slot has invented values")
                    continue
                if name in references:
                    require("array" in value, "reference intermediate lacks full retained array")
                    actual = e.array(value["array"])
                    require(e.typed["matches"](actual, value["descriptor"], bf16=True)
                            and (retained is None or exact_bits(actual, retained)), "reference trajectory full array differs")
                    full[(name, index, slot)] = retained = actual
                else:
                    require("array" not in value, "candidate intermediate storage exceeds declared contract")
                samples[slot] = PREVIOUS["witness_values"](value, retained)
                if slot == "latent" and index < 14:
                    require(e.typed["matches"](z3, value["descriptor"], bf16=True), "stored Z3 changed before scheduler commit")
                    PREVIOUS["witness_values"](value, z3)
                checked_slots[slot] = {"descriptor": value["descriptor"], "meaning": value["meaning"], "array": value.get("array"),
                    "retained_full_array": name in references, "complete_cut_retained": cut in e.store["cuts"]}
            comparisons = {}
            for ref in available_refs:
                comparison = item["comparisons"][ref]
                require(comparison["reference_cut"] == record["rows"][ref][index]["cut"] and set(comparison["slots"]) == set(slots),
                        "trajectory reference comparison retargeted")
                comparisons[ref] = {}
                for slot in slots:
                    key, reported = (ref, index, slot), comparison["slots"][slot]
                    if slot not in samples or key not in full:
                        require(reported == {"status": "not-present-in-both-current-states"}, "absent trajectory comparison differs")
                        continue
                    reference = full[key]
                    require(item["slots"][slot]["descriptor"]["shape"] == list(reference.shape)
                            and reported["sample_count"] == reference.size
                            and type(reported["changed_value_count"]) is int and 0 <= reported["changed_value_count"] <= reference.size
                            and reported["exact_bits"] == (reported["changed_value_count"] == 0)
                            and all(np.isfinite(reported[k]) for k in ("l2", "rms", "mae", "signed_mean", "max_abs")),
                            "candidate full-norm metadata ABI differs")
                    ref_values = PREVIOUS["witness_values"](record["rows"][ref][index]["slots"][slot], reference)
                    diagnostic = {"worker_observed_full_metrics": reported,
                        "independent_fixed64_metrics": PREVIOUS["compare_arrays"](samples[slot].reshape(1, -1, 1), ref_values.reshape(1, -1, 1)),
                        "independently_verified_full_norm": name in references}
                    if name in references:
                        metrics = field_error(full[(name, index, slot)], reference)
                        diagnostic["independent_full_metrics"] = metrics
                        diagnostic["worker_minus_independent_reduction"] = {k: reported[k] - metrics[k] for k in ("l2", "rms", "mae", "signed_mean", "max_abs")}
                    comparisons[ref][slot] = diagnostic
            reports[name].append({"index": index, "cut": cut, "point": item["point"], "incoming_receipt": receipt,
                "slots": checked_slots, "comparisons": comparisons})
    return {"rows": reports, "references": references, "boundaries_per_branch": 15,
        "scope": "Full arrays independently recomputed only for native reference trajectories. Candidate intermediate full norms are retained worker observations; fixed64 witnesses and complete endpoint cuts do not prove every intermediate full norm."}


def audit_timing(parent, observed):
    timeline = SHARED["read_jsonl"](parent["data"] / "timeline.jsonl")
    require([v["sequence"] for v in timeline] == list(range(len(timeline)))
            and all(a["seconds"] <= b["seconds"] for a, b in zip(timeline, timeline[1:])), "timeline ordering differs")
    sealed = [v for v in timeline if v["kind"] == "transition_chain_sealed"]
    starts = [v for v in timeline if v["kind"] == "prediction_start"]
    completed = [v for v in timeline if v["kind"] == "branch_complete"]
    require(len(sealed) == len(starts) == 1 and sealed[0]["seal"] == parent["summary"]["seal"]
            and sealed[0]["counts"] == parent["binding"]["counts_before_seal"]
            and sealed[0]["target_entries_before_seal"] == {"6": 0, "7": 0} and sealed[0]["blocked_entries"] == 0
            and starts[0]["sequence"] < sealed[0]["sequence"]
            and [v["row"] for v in completed] == ORDER and all(sealed[0]["sequence"] < v["sequence"] for v in completed)
            and observed["counts"] == parent["summary"]["counts"] == COUNTS,
            "chain seal does not precede target consumption or census differs")
    return {"seal_event": sealed[0], "prediction_start": starts[0], "target_entries_before_seal": {"6": 0, "7": 0},
        "target_entries_after_seal": parent["summary"]["target_calls"],
        "scope": "Public transaction seal, source-bound guards and physical receipt census. Trusted callback/source inspection, not an OS sandbox or CUDA kernel trace."}


def analyze(root):
    import inspect

    from mdb.tensor_archive import load_tensor_archive

    parent = current_parent(root)
    data, h = parent["data"], parent["helpers"]
    e = SavedEvidence(data, run=parent)
    ENTRY["verify_store_identity"](e, parent["source"])
    lineage = audit_fresh_builder(parent, e)
    predictions, reconstruction, hypotheses = audit_forecasts(parent, e)
    tree = h["NumericalTree"](data / "transaction")
    observed = tree.decode(h["read"](data / "transaction/observations/native.json")["value"])
    for field, filename in (("native_boundaries_sha256", "native-boundaries.json"), ("common_boundaries_sha256", "common-boundaries.json"),
                            ("continuation_observations_sha256", "continuation-observations.json")):
        require(observed[field] == sha_file(data / filename), "native observation file binding differs")
    timing = audit_timing(parent, observed)
    rows, paths, prefixes, native, common, pixels, latents, controls = audit_native_paths(parent, e, predictions, observed)
    observer = audit_observer(parent, e, rows, e.read(data / "native-boundaries.json")["prefixes"])
    full_errors, decomposition = {}, {}
    for role, label in (("zero", "low"), ("text", "high")):
        full_errors[role] = {}
        for stage, index in enumerate((6, 7)):
            full_errors[role][str(index)] = {slot: field_error(native[role][index][:, sl], predictions[role][stage][:, sl])
                for slot, sl in (("full", slice(None)), ("text", slice(None, 512)), ("image", slice(512, None)))}
        decomposition[role] = {slot: vector_decomposition(predictions[role][1][:, sl], common[label + ".pred1"][:, sl], native[role][7][:, sl])
            for slot, sl in (("full", slice(None)), ("text", slice(None, 512)), ("image", slice(512, None)))}
    distances, exported = {}, []
    relative = data.relative_to(parent["root"]).as_posix()
    for name in ORDER:
        base = ("low" if rows[name]["base"] == "zero" else "high") + ".native"
        distances[name] = {"matched_reference": base, "latent": PREVIOUS["GENERIC"]["COMMON"]["distance"](latents[name], latents[base]),
            "RGB": PREVIOUS["GENERIC"]["COMMON"]["distance"](pixels[name], pixels[base]),
            "common_post7": field_error(common[name], common[base])}
        row = rows[name]
        exported.append({"id": name, "base": row["base"], "mode": row["mode"], "parent": row["parent"], "acts": row["acts"],
            "final_cut": row["final_cut"], "installation_boundary": row["installation_boundary"],
            "image": {**row["image"], "path": relative + "/" + row["image"]["path"]},
            "final_latent": {**row["final_latent"], "path": relative + "/" + row["final_latent"]["path"]},
            "row_record": {"path": relative + "/rows/" + name + ".json", "sha256": sha_file(data / "rows" / (name + ".json"))},
            "common_post7": row["common_post7"], "comparisons": distances[name], "native_path": paths[name]})
    replay = all(v is True for v in controls.values())
    return {"schema": "mdb-flux-transition-chain-audit-v1", "mechanics_status": "valid",
        "instrument_status": "native-replay-qualified" if replay else "replay-unresolved", "terminal_status": "unassessed",
        "source": {**parent["custody"], "source_record_sha256": sha_file(data / "source.json"),
            "staged_source_hashes": parent["request"]["source_hashes"], "global_plan_freeze": parent["frozen"], "custody_verifications": 1},
        "model_identity": parent["source"]["model_identity"], "checkpoint_sha256": parent["source"]["checkpoint_sha256"],
        "execution_status": {"job_id": parent["job"]["job_id"], "job_state": parent["job"]["state"]},
        "counts": COUNTS, "panel_counts": PANEL_COUNTS, "seal": parent["summary"]["seal"], "timing": timing,
        "transaction_verification": parent["transaction"], "sealed_hypotheses": hypotheses, "fresh_builder": lineage,
        "typed_store_counts": e.store["counts"], "full_chain_forecast": {"source_binding": parent["binding"],
            "independent_reconstruction": reconstruction, "native_minus_sealed": full_errors,
            "propagation_local_decomposition": decomposition},
        "control_observations": controls, "replay_qualification": {"checks": controls,
            "scope": "Named unchanged-sham complete-stream/common-boundary/final latent/PNG controls only; precision errors never gate."},
        "native_prefixes": prefixes, "native_paths": paths, "continuation_trajectory": observer, "comparisons": distances, "rows": exported,
        "physical_census": {"builder_steps": 108, "native_reference_steps": 4, "native_pred1_bridge_steps": 2,
            "common_suffix_steps": 98, "global_steps": 212, "global_acts": 14, "physical_receipt_occurrences": 226,
            "target_entries": {"6": 2, "7": 4}, "direct_calls": 0, "VAE_decodes": 7},
        "nested_audit_tensor_reference_base": relative, "auditor_source_sha256": sha_file(Path(__file__)),
        "tensor_archive_source_sha256": sha_file(Path(inspect.getfile(load_tensor_archive))),
        "audit_helpers": {name: sha_file(HERE / name) for name in ("analyze_flux_attention_replacement.py", "analyze_flux_mlp_replacement.py",
            "analyze_flux_single_mlp.py", "analyze_flux_single_contraction.py", "analyze_flux_single_contexts.py",
            "summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_rescue_routes.py", "flux_family_custody.py")},
        "models_executed": False,
        "limits": ["One fresh seed79 history and seven dependent native futures; no semantic/generalization claim.",
            "Prediction begins from observed native pre6 full image/text, time modulation, upstream RoPE and actual complete block6/7 weights; not raw prompt-to-RGB prediction.",
            "Block7 free prediction consumes predicted block6 image AND text; no native intermediate or correction is an allowed forecast input.",
            "Independent full NumPy recurrence uses32-query/512-row tiles. All rebuild/native discrepancies remain observations, without precision thresholds or error floors.",
            "pred1 installs complete predicted6 I/T then runs actual native7. pred2 installs recursively predicted7 I/T. Both join the same14-transition suffix clock.",
            "Only two native reference trajectories retain full intermediate arrays. Candidate full intermediate norms are worker observations supported by fixed64 witnesses and retained endpoint cuts.",
            "Stored Z3 and post-readout I/T are stale outside their declared meaningful phase. No full intermediate candidate StateCut claim.",
            "Declared receipt footprints and source inspection are not a dynamic CUDA memory-read trace. Shared low-level kernels/framework arithmetic remain relevant to interpretation.",
            "Vector decomposition is an exact saved-array identity, not unique causal attribution or a semantic mediation fraction."]}


audit = analyze


def audit_fresh_builder(parent, e):
    source = parent["source"]
    path, record = ENTRY["referenced_json"](parent, source["builder"])
    b = SavedEvidence(path.parent)
    ENTRY["verify_store_identity"](b, source)
    require(record["schema"] == "mdb-flux-single-context-builder-v1" and record["context"] == source["context"]
            and record["job_id"] == parent["job"]["job_id"] and source["context"]["seed"] == 79
            and source["context"]["id"] == "p1-s79", "replacement fresh builder context/job differs")
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
            "scope": "Actual fresh seed79 own encoding/noise, two typed conditioning Acts, text transplant, native0 image alignment, native1..5. No imported historical states or known prior endpoints; seed provenance source-bound, not an offline RNG replay."}


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
