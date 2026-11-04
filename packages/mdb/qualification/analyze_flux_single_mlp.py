"""Framework-free audit of a fresh-source pre-entry MLP forecast.

No native layer/model or staged forecasting helper executes here. The early
forecast is conditional on authenticated input states and actual weight rows,
before any native logical-step3 single6 target. Later projection forecasts and
all native futures keep separate transactions and evidence scopes.
"""
from __future__ import annotations

import argparse
import json
import runpy
import weakref
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHARED = runpy.run_path(str(HERE / "analyze_flux_single_contraction.py"))
COMMON, PANEL = SHARED["COMMON"], SHARED["PANEL"]
require, exact_bits = SHARED["require"], SHARED["exact_bits"]
sha_file = SHARED["sha_file"]
PANEL_COUNTS = {**SHARED["COUNTS"], "model_loads": 0, "adapter_steps": 137, "receipt_emissions": 147}
GLOBAL_COUNTS = {**SHARED["COUNTS"], "adapter_steps": 245, "acts": 14, "receipt_emissions": 259,
                 "prompt_encodings": 1, "noise_draws": 1}


def stable_silu(value):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = value[positive] / (1 + np.exp(-value[positive]))
    exp = np.exp(value[~positive])
    result[~positive] = value[~positive] * exp / (1 + exp)
    return result


def entry_equations(residual, modulation, weights, bias, *, eps, queries, neurons, hidden_width, mlp_width, quantize):
    """Independent LayerNorm/time-affine/fused-linear/SwiGLU references."""
    residual, modulation, weights = map(np.asarray, (residual, modulation, weights))
    require(residual.ndim == 3 and residual.shape[0] == 1 and residual.shape[-1] == hidden_width
            and modulation.shape in {(1, 3 * hidden_width), (1, 1, 3 * hidden_width)}
            and weights.shape == (2 * len(neurons), hidden_width)
            and len(set(queries)) == len(queries) and len(set(neurons)) == len(neurons)
            and all(type(q) is int and 0 <= q < residual.shape[1] for q in queries)
            and all(type(j) is int and 0 <= j < mlp_width for j in neurons)
            and type(eps) in {int, float} and eps > 0
            and all(np.isfinite(x).all() for x in (residual, modulation, weights)), "entry operands/geometry differ")
    x = residual[:, queries].astype(np.float64)
    mod = modulation.reshape(1, 1, 3 * hidden_width).astype(np.float64)
    shift, scale, _ = np.split(mod, 3, axis=-1)
    norm = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    affine = (1 + scale) * norm + shift
    q = lambda value: quantize(np.asarray(value, dtype=np.float32))
    norm16 = q(norm)
    affine16 = q(q(q(1 + scale) * norm16) + shift)
    linear = affine @ weights.astype(np.float64).T
    linear16 = affine16.astype(np.float64) @ weights.astype(np.float64).T
    if bias is not None:
        require(np.asarray(bias).shape == (2 * len(neurons),) and np.isfinite(bias).all(), "entry bias geometry differs")
        linear += np.asarray(bias, dtype=np.float64)
        linear16 += np.asarray(bias, dtype=np.float64)
    u, v = np.split(linear, 2, axis=-1)
    u16, v16 = np.split(q(linear16), 2, axis=-1)
    return {"fp64": {"norm": norm, "modulated": affine, "u": u, "v": v, "activation": stable_silu(u) * v},
            "bf16_reference": {"norm": norm16, "modulated": affine16, "u": u16, "v": v16,
                               "activation": q(q(stable_silu(u16)) * v16)}}


def paired_row_indices(hidden_width, mlp_width, neurons):
    require(type(hidden_width) is int and type(mlp_width) is int and hidden_width > 0 and mlp_width > 0
            and len(set(neurons)) == len(neurons) and all(type(j) is int and 0 <= j < mlp_width for j in neurons),
            "invalid paired fused row selector")
    return [3 * hidden_width + j for j in neurons] + [3 * hidden_width + mlp_width + j for j in neurons]


class SavedEvidence(COMMON["SavedEvidence"]):
    """Use public lossless tensor custody; legacy FP32 witnesses stay supported."""
    def array(self, record):
        from mdb.tensor_archive import load_tensor_archive

        if not hasattr(self, "_tensor_archive_cache"):
            self._tensor_archive_cache = weakref.WeakValueDictionary()
        identity = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if identity in self._tensor_archive_cache:
            return self._tensor_archive_cache[identity]
        value = load_tensor_archive(self.data, record)
        require(value.dtype == np.float32 and value.size and np.isfinite(value).all(), "native tensor archive ABI differs")
        self._tensor_archive_cache[identity] = value
        return value


def referenced_json(parent, reference):
    path = parent["helpers"]["file_path"](parent["result_root"], reference["path"])
    require(sha_file(path) == reference["sha256"], "fresh source record digest differs")
    return path, parent["helpers"]["read"](path)


def raw_npy(parent, record):
    if "descriptor" in record:
        from mdb.tensor_archive import load_tensor_archive
        return load_tensor_archive(parent["result_root"], record)
    path = parent["helpers"]["file_path"](parent["result_root"], record["path"])
    require(sha_file(path) == record["sha256"], "fresh encoding array hash differs")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(value.size and np.isfinite(value).all(), "fresh encoding array invalid")
    return value


def encoding_ids_exact(encoded, stored):
    return np.asarray(encoded).dtype == np.asarray(stored).dtype == np.dtype("int64") and exact_bits(encoded, stored)


def bitwise_mismatch_counts(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    require(actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape,
            "bitwise sample comparison ABI differs")
    return {"sample_count": int(actual.size), "bf16_reference_mismatch_count": int(np.count_nonzero(
        np.ascontiguousarray(actual).view(np.uint32) != np.ascontiguousarray(reference).view(np.uint32)))}


def fresh_local_ordering(contrasts):
    """Do not inherit the predecessor's predeclared hypothesis into a new plan."""
    prior_label = contrasts.pop("H_M_over_A")
    signed = prior_label["signed_A_minus_M_L2_to_high"]
    contrasts["A_vs_M_observation"] = {"signed_A_minus_M_L2_to_high": signed,
        "observed_ordering": "A-closer" if signed < 0 else "M-closer" if signed > 0 else "tie",
        "scope": "Unpredicted exploratory ordering; earlier seed53 miss remains separate."}


def verify_store_identity(e, source):
    require(all(cut["model_identity"] == source["model_identity"] and cut["execution"] == source["execution"]
                for cut in e.store["cuts"].values())
            and all(r["model_identity"] == source["model_identity"] and r["execution"] == source["execution"]
                    for r in e.receipts.values()), "fresh source typed identity differs")


def fresh_pair(e, low, high):
    a, b = e.fields(low), e.fields(high)
    require(set(a) == set(b) and all(a[k] == b[k] for k in a if k not in {"image", "text"})
            and e.point(low) == e.point(high), "fresh paired state changed outside I/T")
    return {"same_all_other_slots": True, "image_exact": exact_bits(e.slot(low, "image"), e.slot(high, "image")),
            "text_exact": exact_bits(e.slot(low, "text"), e.slot(high, "text")),
            "scope": "Exact contrast/collapse observations, not an effect-presence or collection gate."}


def audit_initial_builder(parent):
    """Verify the actual 96-step, three-Act fresh producer; no imported history."""
    source = parent["root_source"]
    path, record = referenced_json(parent, source["builder"])
    e = SavedEvidence(path.parent)
    verify_store_identity(e, source)
    context = source["context"]
    require(record["schema"] == "mdb-flux-single-context-builder-v1"
            and record["context"] == context and record["job_id"] == parent["job"]["job_id"]
            and context["seed"] == 67 and context["id"] == "p1-s67", "fresh builder context/real-job identity differs")
    own = e.array(record["arrays"]["own_conditioning"])
    require(exact_bits(own, raw_npy(parent, source["encoding"]["conditioning"]))
            and exact_bits(e.slot(record["initial"], "conditioning"), own)
            and encoding_ids_exact(raw_npy(parent, source["encoding"]["txt_ids"]), e.slot(record["initial"], "txt_ids")),
            "fresh native prompt encoding does not match builder own-C/text IDs")
    initial, z2, z3 = (record[key] for key in ("initial", "Z2", "Z3"))
    for name, cut, slot in (("initial_latent", initial, "latent"), ("Z2_latent", z2, "latent"),
                            ("Z3_latent", z3, "latent"), ("own_conditioning", z2, "conditioning")):
        require(exact_bits(e.array(record["arrays"][name]), e.slot(cut, slot)), "fresh builder saved operand/cut differs")
    paths, acts = record["paths"], record["acts"]
    walk, apply = COMMON["builder_path"], COMMON["builder_act"]
    ids = walk(e, paths["to_Z2"], initial, z2, expected_steps=56)
    zero_z2 = apply(e, acts["zero_conditioning"], z2, np.zeros_like(own), slot="conditioning")
    require(zero_z2 == record["zeroZ2"], "fresh zero-C history differs")
    ids.append(acts["zero_conditioning"]["receipt_id"])
    ids += walk(e, paths["to_Z3"], zero_z2, z3, expected_steps=28)
    restored = apply(e, acts["restore_conditioning"], z3, own, slot="conditioning")
    ids.append(acts["restore_conditioning"]["receipt_id"])
    ids += walk(e, paths["zero_to_single0"], z3, record["zero_single0"], expected_steps=6)
    ids += walk(e, paths["own_to_single0"], restored, record["own_single0"], expected_steps=6)
    matched = apply(e, acts["transplant_text"], record["zero_single0"], e.slot(record["own_single0"], "text"), slot="text")
    ids.append(acts["transplant_text"]["receipt_id"])
    require(matched == record["matched_text"] and e.fields(matched) == {
        **e.fields(record["zero_single0"]), "text": e.fields(record["own_single0"])["text"]}, "fresh pre0 text-only transplant differs")
    emissions = e.read(path.parent / "receipt-emissions.json")
    require([r["sequence"] for r in emissions] == list(range(99)) and [r["receipt"] for r in emissions] == ids
            and set(ids) == set(e.receipts) and record["counts"] == {"adapter_steps": 96, "acts": 3, "receipt_emissions": 99},
            "fresh initial builder physical census differs")
    for name in ("Z3", "zero_single0", "matched_text"):
        require(not np.count_nonzero(e.slot(record[name], "conditioning")), "fresh zero-C history not retained")
    return e, record, {"record": source["builder"], "counts": record["counts"],
        "own_prompt_encoding_exact": True, "pre0_only_text_differs": True,
        "seed_scope": "Declared seed67 native noise creation is source-bound; no offline random-generator rerun."}


def audit_preparation(parent, panel_e):
    original_e, original, initial_audit = audit_initial_builder(parent)
    path, record = referenced_json(parent, parent["root_source"]["preparation"])
    e = SavedEvidence(path.parent)
    verify_store_identity(e, parent["root_source"])
    expected_inputs = {"zero": original["zero_single0"], "text": original["matched_text"]}
    require(record["schema"] == "mdb-flux-single-mlp-preparation-v1" and record["input_cuts"] == expected_inputs
            and record["builder"] == parent["root_source"]["builder"], "preparation uses another fresh pre0 pair")
    ids = []
    for role in ("zero", "text"):
        start = expected_inputs[role]
        first = record["first_single0"][role]
        require(e.store["cuts"][start] == original_e.store["cuts"][start]
                and first["input_cut"] == start, "preparation did not start from actual builder cut")
        ids += COMMON["builder_path"](e, [first["receipt"]], start, first["post_cut"], expected_steps=1)
        require(e.point(first["post_cut"])["index"] == 1, "native0 exits at another frontier")
    high = record["first_single0"]["text"]["post_cut"]
    low = record["first_single0"]["zero"]["post_cut"]
    matched = COMMON["builder_act"](e, record["image_act"], low, e.slot(high, "image"), slot="image")
    ids.append(record["image_act"]["receipt_id"])
    require(record["matched_post0"] == {"zero": matched, "text": high}
            and exact_bits(e.slot(matched, "image"), e.slot(high, "image")), "post0 matched-I construction differs")
    for role in ("zero", "text"):
        end = record["pre6_cuts"][role]
        ids += COMMON["builder_path"](e, record["paths"][role], record["matched_post0"][role], end, expected_steps=5)
        require(e.point(end)["index"] == e.point(end)["local_clock"] == 6
                and e.store["cuts"][end] == panel_e.store["cuts"][end]
                and parent["source"]["input_cuts"][role] == end, "prepared pre6 input is not the actual fresh continuation")
    emissions = e.read(path.parent / "receipt-emissions.json")
    require([r["sequence"] for r in emissions] == list(range(13)) and [r["receipt"] for r in emissions] == ids
            and set(ids) == set(e.receipts) and record["counts"] == {"adapter_steps": 12, "acts": 1, "receipt_emissions": 13},
            "source preparation physical census differs")
    pair = fresh_pair(e, record["pre6_cuts"]["zero"], record["pre6_cuts"]["text"])
    return {"kind": "fresh-native-builder", "job_id": parent["job"]["job_id"], "context": parent["root_source"]["context"],
        "initial_builder": initial_audit, "preparation": parent["root_source"]["preparation"],
        "counts": {"adapter_steps": 108, "acts": 4, "receipt_emissions": 112}, "input_cuts": record["pre6_cuts"], "pair_observations": pair,
        "scope": "Fresh own prompt/noise history, direct post0 image alignment followed by native1..5. No imported states, historical PNG controls or extra K/V equivalence assay. Pre6 I/T contrast or collapse is observed separately; no isolated text-bank cause."}


class PreparedSource:
    native_replay_steps = 0
    preseal_counts = {**PANEL_COUNTS, "adapter_steps": 0, "acts": 0, "candidate_calls": 0,
                     "receipt_emissions": 0, "images": 0, "vae_calls": 0}
    outer_preseal_counts = {**GLOBAL_COUNTS, "adapter_steps": 108, "acts": 4, "candidate_calls": 0,
                           "receipt_emissions": 112, "images": 0, "vae_calls": 0}

    def verify_source(self, e, parent):
        source, top = parent["source"], parent["root_source"]
        require(source["prepared_inputs"] is True and source["array_storage"] == "bf16-bits"
                and source["prepared_input_cuts"] == source["input_cuts"]
                and source["source_selection"] == {"schema": "mdb-flux-single-mlp-selection-v1", "kind": "fresh-native-builder",
                    "context_id": "p1-s67", "builder": top["builder"], "preparation": top["preparation"], "input_cuts": source["input_cuts"]}
                and source["checkpoint_sha256"] == top["checkpoint_sha256"] == parent["request"]["config"]["expected_checkpoint_sha256"]
                and source["implementation_sources"] == top["implementation_sources"] and source["versions"] == top["versions"]
                and source["model_identity"] == top["model_identity"] and source["execution"] == top["execution"]
                and source["early_seal_sha256"] == parent["outer"]["entry_seal"]["seal_sha256"], "prepared source/model/early-seal authority differs")
        self.lineage = audit_preparation(parent, e)
        return source["source_selection"], self.lineage

    def audit_prefixes(self, e, source, prefixes):
        require(set(prefixes) == {"zero", "text"}, "fresh native prefix inventory differs")
        paths = {}
        retained = {cut: e.fields(cut) for cut in e.store["cuts"]}
        c = e.fields(source["input_cuts"]["zero"])["conditioning"]
        for role in ("zero", "text"):
            start, post = source["input_cuts"][role], prefixes[role]["post_single6"]
            p = e.walk(start, post, limit=3)
            require(p["native_steps"] == 1 and p["acts"] == 0 and prefixes[role]["input_cut"] == start
                    and prefixes[role]["receipt"] == e.receipts[p["receipts"][0]]
                    and e.point(post)["index"] == 7, "fresh single6 native prefix lineage differs")
            p["conditioning_coverage"] = SHARED["BASE"]["conditioning_coverage"](p, e.receipts, retained, c, e.point)
            paths[role] = p
            for slot in ("image", "text", "latent", "conditioning"):
                ref = source["input_arrays"][role][slot]
                require(exact_bits(e.array(ref), e.slot(start, slot)), "prepared source port differs from complete cut")
        pair = fresh_pair(e, source["input_cuts"]["zero"], source["input_cuts"]["text"])
        require(source["source_contrast_observations"] == {key: pair[key] for key in ("image_exact", "text_exact")}, "declared fresh contrast differs")
        return {}, paths, {}, pair

    def emission_census(self, e, prefix_paths, paths, summary, plan):
        emitted = SHARED["read_jsonl"](e.data / "receipt-emissions.jsonl")
        expected = [("prefix." + role, rid) for role in ("zero", "text") for rid in prefix_paths[role]["receipts"]]
        expected += [(spec["id"], rid) for spec in plan["branches"] for rid in paths[spec["id"]]["receipts"]]
        require([r["sequence"] for r in emitted] == list(range(147))
                and [(r["branch"], r["receipt"]) for r in emitted] == expected and summary["counts"] == PANEL_COUNTS
                and sum(p["native_steps"] for p in [*prefix_paths.values(), *paths.values()]) == 137
                and sum(p["acts"] for p in paths.values()) == 10, "fresh panel physical emission census differs")
        return {"panel_adapter_steps": 137, "panel_acts": 10, "panel_receipt_emissions": 147,
                "builder_adapter_steps": 108, "builder_acts": 4, "builder_receipt_emissions": 112,
                "global_adapter_steps": 245, "global_acts": 14, "global_receipt_emissions": 259,
                "scope": "Physical path/emission accounting before receipt deduplication; one actual job, not child jobs."}


def current_parent(root, *, custody_verify=None):
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request, job, envelope = (h["read"](root / name) for name in ("request.json", "job.json", "report.json"))
    if custody_verify is None:
        custody_verify = runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    custody = custody_verify(root, limits=request["config"].get("transport_limits"))
    require(custody["mechanics_status"] == "verified" and custody["job_id"] == job["job_id"] == envelope["job_id"]
            and custody["request_sha256"] == envelope["request_sha256"] == request["request_sha256"]
            and envelope["archive_root"] == "flux-single-mlp-result", "fresh parent custody/job/archive differs")
    result = root / "extracted/flux-single-mlp-result"
    data = result / "panel"
    top, outer, frozen, root_plan = (h["read"](result / name) for name in ("source.json", "outer-summary.json", "plan-frozen.json", "plan.json"))
    source, summary, plan = (h["read"](data / name) for name in ("source.json", "summary.json", "plan.json"))
    require(outer == envelope["report"] and outer["counts"] == GLOBAL_COUNTS and summary["counts"] == PANEL_COUNTS
            and outer["panel_summary_sha256"] == sha_file(data / "summary.json")
            and root_plan == request["config"] and plan == {**root_plan, "max_adapter_steps": 137, "max_acts": 10,
                "max_wall_seconds": plan["max_wall_seconds"]}
            and 0 < plan["max_wall_seconds"] <= root_plan["max_wall_seconds"]
            and top["context"] == root_plan["context"] and top["context"]["seed"] == 67
            and all(value["job_id"] == job["job_id"] and value["request_sha256"] == request["request_sha256"]
                    and value["model_identity"] == top["model_identity"] for value in (source, summary, top, outer))
            and source["source_hashes"] == top["source_hashes"] == request["source_hashes"], "fresh outer/panel/config identity differs")
    require(frozen["plan_sha256"] == sha_file(result / "plan.json") and frozen["request_sha256"] == request["request_sha256"]
            and frozen["job_id"] == job["job_id"] and all(value == 0 for value in frozen["work"].values())
            and h["sha"](h["canonical"](top["checkpoint_files"])) == top["checkpoint_sha256"] == root_plan["expected_checkpoint_sha256"],
            "fresh preload plan/checkpoint freeze differs")
    verified = h["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    require(SHARED["transaction_verifications_match"](verified, summary["transaction_verification"])
            and verified["phase"] == "closed", "fresh panel transaction differs")
    return {"root": root, "data": data, "result_root": result, "helpers": h, "request": request, "job": job,
            "custody": custody, "root_source": top, "outer": outer, "source": source, "summary": summary,
            "plan": plan, "frozen": frozen, "transaction": verified, "saved_worker_verification": summary["transaction_verification"]}


def audit_entry(parent, e):
    from mdb.tensor_archive import load_tensor_archive

    h, root = parent["helpers"], parent["result_root"]
    load = lambda ref: load_tensor_archive(root, ref)
    binding = h["read"](root / "entry-binding.json")
    observations = h["read"](root / "entry-observations.json")
    tx = root / "mlp-entry-transaction"
    seal = parent["outer"]["entry_seal"]
    verified = h["verify_transaction"](tx, expected_seal_sha256=seal["seal_sha256"])
    require(SHARED["transaction_verifications_match"](verified, parent["outer"]["entry_transaction_verification"])
            and verified["phase"] == "closed" and observations["seal"] == seal, "early transaction verification differs")
    tree = h["NumericalTree"](tx)
    declared = tree.decode(h["read"](tx / "frontiers/source_binding.json")["value"])
    predictions = tree.decode(h["read"](tx / "forecasts/entry_mlp.json")["predictions"])
    native = tree.decode(h["read"](tx / "observations/native_entry.json")["value"])
    require(declared == binding and set(binding["inputs"]) == set(predictions["cases"]) == set(observations["rows"]) == {"zero", "text"}
            and binding["hidden_width"] == 3072 and binding["mlp_width"] == 9216 and binding["eps"] == 1e-6
            and binding["elementwise_affine"] is False
            and binding["query_indices"] == parent["plan"]["query_indices"] == [0, 31, 32, 255, 511, 512, 767, 1023]
            and binding["neuron_indices"] == parent["plan"]["neuron_indices"] == list(range(0, 9216, 288))
            and binding["weight_row_indices"] == paired_row_indices(3072, 9216, binding["neuron_indices"])
            and binding["counts_before_seal"] == PreparedSource.outer_preseal_counts
            and binding["target_entries_before_seal"] == observations["target_entries_before_seal"] == 0
            and binding["builder"] == parent["root_source"]["builder"] and binding["preparation"] == parent["root_source"]["preparation"],
            "early input/site/profile/sample binding differs")
    require(native["entry_observations_sha256"] == sha_file(root / "entry-observations.json")
            and native["panel_summary_sha256"] == parent["outer"]["panel_summary_sha256"]
            and native["counts"] == GLOBAL_COUNTS and native["calls"] == observations["calls"] == 7,
            "early target samples/physical census differs")
    weights = load(binding["weight_rows"])
    bias = None if binding["bias_rows"] is None else load(binding["bias_rows"])
    require(binding["linear_path"] == "single_transformer_blocks.6.attn.to_qkv_mlp_proj"
            and binding["native_parameters"]["weight"]["shape"] == [27648, 3072]
            and binding["native_parameters"]["weight"]["dtype"] == "torch.bfloat16"
            and weights.shape == (64, 3072), "early selected native parameter profile differs")
    error = PANEL["sample_error"]
    cases = {}
    for index, role in enumerate(("zero", "text")):
        entry, pred, witness = binding["inputs"][role], predictions["cases"][role], observations["rows"][role]
        cut = parent["source"]["input_cuts"][role]
        image, mod = load(entry["image"]), load(entry["single_mod"])
        require(entry["cut"] == cut and exact_bits(image, e.slot(cut, "image"))
                and exact_bits(mod, SHARED["nested_slot_array"](e, cut, "context", ("single_mod",)))
                and witness["call_index"] == native["rows"][role]["call_index"] == index
                and pred["query_indices"] == binding["query_indices"] and pred["neuron_indices"] == binding["neuron_indices"]
                and pred["weight_row_indices"] == binding["weight_row_indices"], "early forecaster input/role differs")
        rebuilt = entry_equations(image, mod, weights, bias, eps=1e-6, queries=binding["query_indices"],
            neurons=binding["neuron_indices"], hidden_width=3072, mlp_width=9216, quantize=e.typed["quantize_bf16"])
        stages = {}
        for name, target_name in (("norm", "layernorm"), ("modulated", "modulated_input"), ("u", "u"), ("v", "v"), ("activation", "activation")):
            actual = load(witness[target_name])
            expected_shape = (1, 8, 3072 if name in {"norm", "modulated"} else 32)
            require(actual.shape == expected_shape and witness[target_name]["descriptor"]["dtype"] == "torch.bfloat16"
                    and exact_bits(actual, native["rows"][role][target_name]), "native early sample array/source differs")
            key = "layernorm" if name == "norm" else name
            analytic, reference = pred["analytic"][key], pred["bf16_reference"][key]
            require(np.asarray(analytic).dtype == np.float64 and np.asarray(reference).dtype == np.float32
                    and np.asarray(analytic).shape == np.asarray(reference).shape == expected_shape
                    and np.isfinite(analytic).all() and np.isfinite(reference).all(), "early forecast numeric ABI differs")
            rms = float(np.sqrt(np.mean(actual.astype(np.float64) ** 2)))
            stages[target_name] = {"analytic_error": error(analytic, actual), "bf16_reference_error": error(reference, actual),
                "reference_exact_native": exact_bits(reference, actual), "native_rms": rms,
                **bitwise_mismatch_counts(actual, reference),
                "source_recomputation": {"analytic": error(rebuilt["fp64"][name], analytic),
                                         "bf16_reference": error(rebuilt["bf16_reference"][name], reference)},
                "actual": witness[target_name]}
        candidate = h["read"](e.data / "candidates.json")["candidates"][role]
        concat = e.array(candidate["ports"]["original_concat"])
        sampled_activation = concat[:, np.asarray(binding["query_indices"]) + candidate["text_rows"]][:, :, 3072 + np.asarray(binding["neuron_indices"])]
        require(exact_bits(load(witness["activation"]), sampled_activation), "early activation is not the actual source-candidate MLP feature")
        cases[role] = {"call_index": index, "input_cut": cut, "stages": stages,
            "arithmetic": pred["arithmetic"], "scope": pred["scope"]}
    timeline = SHARED["read_jsonl"](root / "timeline.jsonl")
    seals = [row for row in timeline if row["kind"] == "entry_forecast_sealed"]
    require(len(seals) == 1 and seals[0]["seal"] == seal and seals[0]["target_entries_before_seal"] == 0
            and seals[0]["counts"] == PreparedSource.outer_preseal_counts, "early forecast timing event differs")
    return {"verification": verified, "saved_worker_verification": parent["outer"]["entry_transaction_verification"],
            "seal": seal, "source_binding": binding, "cases": cases, "instrument_status": "source-bound-entry-prediction",
            "signed_error_convention": "actual native sample minus sealed forecast",
            "timing": "Early seal before any current logical3 single6 entry; builder module6 calls at earlier logical steps are allowed. Main panel and sampled projection seals are separate.",
            "limits": ["Errors and misses are observations, not thresholds for discarding futures.",
                "Actual partial native weight rows are source-authorized index selections, not independent partial-row Merkle proofs.",
                "Trusted forecaster receives only incoming image/modulation and selected weights; this is not hermetic process isolation or a raw-prompt/RGB forecast.",
                "SiLU-only intermediate is forecast but not separately measured; native combined activation is measured."]}


def analyze(root, *, custody_verify=None):
    import mdb.tensor_archive as tensor_archive

    reader_path = Path(tensor_archive.__file__)
    reader_sha256 = sha_file(reader_path)
    parent = current_parent(root, custody_verify=custody_verify)
    h, data = parent["helpers"], parent["data"]
    run = {"root": parent["root"], "data": data, "plan": parent["plan"], "request": parent["request"],
        "source": {**parent["custody"], "transaction": parent["transaction"]}, "transaction": data / "transaction",
        "tree": h["NumericalTree"](data / "transaction")}
    e, adapter = SavedEvidence(data, run=run), PreparedSource()
    audit = SHARED["analyze_evidence"](parent, e, source_adapter=adapter)
    fresh_local_ordering(audit["local_feature_contrasts"])
    early = audit_entry(parent, e)
    lineage = audit.pop("imported_source")
    audit["counts"] = PANEL_COUNTS
    audit["execution_status"]["model_loads"] = 0
    audit["execution_status"]["scope"] = "Panel in one real parent job; global model load is one. No child job."
    audit["schema"] = "mdb-flux-single-mlp-feature-panel-audit-v1"
    audit["trend_status"] = "Fresh history feature factorial; local/endpoint errors, exploratory orderings and collapse retained."
    audit["fresh_source"] = lineage
    audit["reference_scopes"]["staged_inputs"] = "Fresh builder/preparation references are relative to the parent extracted result root; no imported-state or prior producer paths."
    audit["limits"][0] = "One fresh seed67 prompt history and nine dependent futures; no imported prior-state or known historical endpoint controls. Native raw-prompt output and MLP weights are not predicted."
    audit["limits"][1] = "Both working image and text differ at this observed pre6 pair. High A/M features encode the whole fresh history, not an isolated text-bank cause."
    audit["limits"][4] = "The later sampled projection/write forecast conditions on two observed A/M sources; it does not itself predict their features. The separate earlier entry transaction predicts sampled MLP features before native6. Neither forecasts attention, readout, the full trajectory or RGB."
    audit["limits"] += early["limits"]
    require(sha_file(reader_path) == reader_sha256, "installed tensor reader changed during audit")
    return {"schema": "mdb-flux-single-mlp-audit-v1", "source": {**parent["custody"],
                "root_source_sha256": sha_file(parent["result_root"] / "source.json"),
                "global_plan_freeze": parent["frozen"], "staged_source_hashes": parent["request"]["source_hashes"], "custody_verifications": 1},
            "counts": GLOBAL_COUNTS, "builder_counts": lineage["counts"], "panel_counts": PANEL_COUNTS,
            "mechanics_status": "valid", "instrument_status": audit["instrument_status"], "terminal_status": "unassessed",
            "execution_status": {"job_id": parent["job"]["job_id"], "job_state": parent["job"]["state"], "model_loads": 1},
            "trend_status": "One fresh history; early entry and later projection forecasts have separate conditional inputs. All native futures and signed disagreements retained.",
            "fresh_builder": lineage, "entry_forecast": early, "panel": audit,
            "entry_seal": early["seal"], "architectural_seal": audit["architectural_seal"], "numeric_seal": audit["numeric_seal"],
            "early_mlp_array_reference_base": parent["result_root"].relative_to(parent["root"]).as_posix(),
            "auditor_source_sha256": sha_file(Path(__file__)),
            "tensor_archive_source_sha256": reader_sha256,
            "tensor_archive_source_scope": "Actual installed public offline reader source; distinct from immutable staged producer codec/wheel. Source digest verified unchanged for this audit.",
            "audit_helpers": {**audit["audit_helpers"], "analyze_flux_single_contraction.py": sha_file(HERE / "analyze_flux_single_contraction.py")},
            "models_executed": False, "limits": audit["limits"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in ("mechanics_status", "instrument_status", "counts")}))
