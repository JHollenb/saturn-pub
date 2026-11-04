"""Saved-array audit of last-step single-block text K/V interventions.

The single block has parallel attention/MLP paths and one fused output
projection. No model or attention kernel is executed by this audit.
"""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = runpy.run_path(str(HERE / "analyze_flux_rescue_routes.py"))
require, exact_bits, distance = (BASE[key] for key in ("require", "exact_bits", "distance"))


def source_run(helper, root, *, failed=False):
    """Explicit failed-publication role; never relabel a failed job succeeded."""
    root = Path(root).resolve()
    recovery = None
    extras = {}
    if failed:
        data = root / "extracted/flux-single-kv-result"
        report, failure = (helper["read"](path) for path in (root / "report.json", data / "failure.json"))
        require(failure == {key: value for key, value in report.items() if key != "archive_sha256"},
                "retained publication failure differs from original transport REPORT")
        summary, attention = (helper["read"](data / name) for name in ("summary.json", "attention-forecast.json"))
        expected = {"adapter_steps": 191, "candidate_calls": 5, "acts": 10, "images": 9, "vae_calls": 9,
                    "receipt_emissions": 201, "prompt_encodings": 0, "noise_draws": 0, "model_loads": 1, "physical_batch": 1}
        require(summary["counts"] == expected and summary["job_id"] == failure["job_id"]
                and summary["request_sha256"] == failure["request_sha256"],
                "failed-publication recovery requires complete declared scientific census")
        checked = {}
        for name, seal in (("transaction", summary["seal"]), ("attention-transaction", attention["seal"])):
            checked[name] = helper["verify_transaction"](data / name, expected_seal_sha256=seal["seal_sha256"])
            events = [helper["read"](path) for path in sorted((data / name / "events").glob("*.json"))]
            require(checked[name]["phase"] == "closed" and any(event["kind"] == "observation_persisted" for event in events)
                    and not any(event["kind"] in {"failed", "violation"} for event in events),
                    "failed-publication recovery requires both complete unviolated transactions")
        tree = helper["NumericalTree"](data / "transaction")
        observed = tree.decode(helper["read"](data / "transaction/observations/native.json")["value"])
        require(observed["counts"] == expected and len(observed["rows"]) == 9
                and set(observed["candidates"]) == {"zero", "text", "K", "V", "KV"},
                "failed-publication science inventory is incomplete")
        timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
        extras = {"failure.json": helper["sha"]((data / "failure.json").read_bytes())}
        recovery = {"scope": "Original failed producer REPORT remains transport authority; unchanged original archive and producer manifest retained. Complete science assessed separately.",
            "failure_metadata": {"path": "extracted/flux-single-kv-result/failure.json", "sha256": extras["failure.json"]},
            "original_failure_report_sha256": helper["sha"]((root / "report.json").read_bytes()),
            "scientific_counts": expected, "transactions": checked, "secondary_timing": numeric_timing(timeline),
            "model_rerun": False}
    run = helper["source_run"](root, "flux-single-kv-result", failed=failed, recover_metadata=failed,
        failed_manifest_extras=extras, max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    job = helper["read"](Path(root) / "job.json")
    require(job["state"] == ("failed" if failed else "succeeded"), "external job state differs from declared audit role")
    run["execution_status"] = {"job_state": job["state"], "failed_publication_role": bool(failed),
        "scope": "Scientific mechanics evaluated separately from external worker/transport status; no status rewrite."}
    if recovery is not None:
        run["source"]["failed_publication_recovery"] = recovery
    return run


class Evidence(BASE["Evidence"]):
    def __init__(self, root, *, failed=False):
        self.h = runpy.run_path(str(HERE / "summarize_context_window.py"))
        self.typed = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))
        self.routes = runpy.run_path(str(HERE / "analyze_flux_concept_routes.py"))
        self.run = source_run(self.h, root, failed=failed)
        self.data = self.run["data"]
        self.store = self.typed["audit_store"](self.data / "typed-store")
        self.receipts, self.edges = {}, {}
        for path in (self.data / "typed-store/receipts").glob("*.json"):
            receipt = self.read(path)
            self.h["public_receipt"](receipt)
            require(receipt["fingerprint"] == path.stem, "receipt filename differs")
            self.receipts[path.stem] = receipt
            if receipt["operation"] == "continue":
                require(receipt["parent"] not in self.edges or self.edges[receipt["parent"]] == receipt,
                        "ambiguous native continuation edge")
                self.edges[receipt["parent"]] = receipt


def operand_selection(observed, used, donor, *, text_rows, select_k, select_v):
    require(set(observed) == set(used) == set(donor) == {"q", "k", "v"}
            and type(text_rows) is int and 0 < text_rows < observed["q"].shape[1]
            and all(value.ndim == 4 and value.shape == observed["q"].shape
                    for value in [*observed.values(), *used.values(), *donor.values()]), "QKV partition geometry differs")
    expected = {key: value.copy() for key, value in observed.items()}
    for key, selected in (("k", select_k), ("v", select_v)):
        require(type(selected) is bool, "nonboolean operand selector")
        if selected:
            expected[key][:, :text_rows] = donor[key][:, :text_rows]
    checks = {"all_query_exact_recipient": exact_bits(used["q"], expected["q"]),
              "key_exact_declared_selection": exact_bits(used["k"], expected["k"]),
              "value_exact_declared_selection": exact_bits(used["v"], expected["v"])}
    require(all(checks.values()), "Q or image bank changed, or wrong text K/V selected")
    return checks


def kv_interaction(cells):
    result = BASE["factorial_interaction"](*(cells[key] for key in ("zero", "K.image", "V.image", "KV.image")))
    result["formula"] = "Y_KV.image - Y_K.image - Y_V.image + Y_zero; FP64 arithmetic on full saved native arrays"
    result["scope"] = "Exploratory final image K/V interaction vector, not a semantic conjunction score or mediated fraction."
    return result


def parallel_residual(residual, gate, projection, output, quantize):
    residual, gate, projection, output = (np.asarray(value) for value in (residual, gate, projection, output))
    require(residual.ndim == 3 and residual.shape[0] == 1 and gate.shape == (1, 1, residual.shape[-1])
            and projection.shape == output.shape == residual.shape
            and all(np.isfinite(value).all() for value in (residual, gate, projection, output)),
            "parallel-block finite/geometry contract differs")
    write = quantize(np.asarray(gate) * np.asarray(projection))
    require(exact_bits(quantize(np.asarray(residual) + write), output), "saved rounded parallel-block residual equation differs")
    return {"rounded_whole_projection_residual_exact": True,
            "scope": "Whole fused attention/MLP projection retained; no exact sum of separately rounded projection slices assumed."}


def sample_error(predicted, actual):
    p, a = np.asarray(predicted), np.asarray(actual)
    require(p.shape == a.shape and p.size and np.isfinite(p).all() and np.isfinite(a).all(),
            "sampled attention finite/shape contract differs")
    delta = a.astype(np.float64) - p.astype(np.float64)
    return {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
            "signed_mean": float(delta.mean()), "l2": float(np.linalg.norm(delta.ravel()))}


def inspect_attention_samples(prediction, candidate_records, load_array, quantize):
    require(prediction["instrument_status"] in {"applicable", "not-applicable"}, "unknown numerical forecast applicability")
    indices = prediction["query_indices"]
    require(indices == [0, 31, 32, 255, 511, 512, 767, 1023], "attention query selection differs from fixed pre-target sample")
    if prediction["instrument_status"] == "not-applicable":
        return {"instrument_status": "not-applicable", "query_indices": indices, "cases": {},
                "scope": "No numerical attention claim; main native interventions remain available."}
    require(set(prediction["cases"]) == {"K", "V", "KV"}, "attention forecast case inventory differs")
    results = {}
    for name, pred in prediction["cases"].items():
        record = candidate_records[name]
        full = load_array(record["attention_dispatch_output"])
        actual = full[:, np.asarray(indices, dtype=np.int64) + record["text_rows"]]
        fp64, cast, mass = (np.asarray(pred[key]) for key in ("attention", "attention_bf16", "text_mass"))
        require(fp64.dtype == np.float64 and cast.dtype == np.float32 and fp64.shape == cast.shape == actual.shape
                and mass.dtype == np.float64 and mass.shape == (1, full.shape[2], len(indices))
                and np.isfinite(fp64).all() and np.isfinite(cast).all() and np.isfinite(mass).all()
                and exact_bits(cast, quantize(fp64.astype(np.float32))), "attention forecast/cast/mass ABI differs")
        role = "native-text represented consistency control" if name == "KV" else "novel conditional forecast"
        require(pred["role"] == role, "numerical forecast novelty/control role differs")
        results[name] = {"role": role, "fp64_error": sample_error(fp64, actual),
            "cast_bf16_error": sample_error(cast, actual), "predicted_cast_bit_exact": exact_bits(cast, actual),
            "actual_rms": float(np.sqrt(np.mean(actual.astype(np.float64) ** 2))),
            "shape": list(actual.shape), "native_dispatch_array": record["attention_dispatch_output"],
            "text_mass": {"min": float(mass.min()), "mean": float(mass.mean()), "max": float(mass.max()),
                "outside_probability_range": bool(np.any((mass < 0) | (mass > 1))),
                "authority": "Forecast only; dispatcher did not expose native attention probabilities, so unmeasured."}}
    return {"instrument_status": "applicable", "query_indices": indices, "cases": results,
        "scope": "Saved sealed forecasts versus saved edited native dispatch samples; no attention/model reevaluation, numerical error is not a gate."}


def binding_array_refs(value):
    found = []
    if isinstance(value, dict):
        if {"path", "sha256", "descriptor"} <= value.keys():
            found.append(value)
        for child in value.values():
            found.extend(binding_array_refs(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(binding_array_refs(child))
    return found


def numeric_timing(timeline):
    """Prospective policy qualification; failures do not erase main branches."""
    seals = [event for event in timeline if event["kind"] == "attention_forecast_sealed"]
    starts = {event["candidate"]: event for event in timeline if event["kind"] == "candidate_start"}
    completes = {event["candidate"]: event for event in timeline if event["kind"] == "candidate_complete"}
    checks = {"one_secondary_seal": len(seals) == 1, "five_candidate_events": set(starts) == set(completes) == {"zero", "text", "K", "V", "KV"}}
    if len(seals) == 1:
        seal = seals[0]
        checks["two_observed_source_calls_at_seal"] = seal["counts"]["candidate_calls"] == 2
        checks["no_prefix_Act_or_image_at_seal"] = all(seal["counts"][key] == 0 for key in ("adapter_steps", "acts", "images", "vae_calls"))
        checks["sources_complete_before_secondary_seal"] = all(name in completes and completes[name]["sequence"] < seal["sequence"] for name in ("zero", "text"))
        checks["edited_targets_start_after_secondary_seal"] = all(name in starts and starts[name]["sequence"] > seal["sequence"] for name in ("K", "V", "KV"))
    return {"checks": checks, "qualified": all(checks.values()),
            "status": "prospective-timing-qualified" if all(checks.values()) else "prospective-timing-unresolved",
            "scope": "Public event/counter policy from byte-bound trusted worker, not an OS sandbox or independent low-level call trace."}


def audit_attention_transaction(e, source, candidates, timeline):
    root, read, h = e.data / "attention-transaction", e.read, e.h
    record = read(e.data / "attention-forecast.json")
    verification = h["verify_transaction"](root, expected_seal_sha256=record["seal"]["seal_sha256"])
    require(verification == record["verification"] and verification["phase"] == "closed", "secondary attention custody differs")
    tree = h["NumericalTree"](root)
    binding = tree.decode(read(root / "frontiers/source_binding.json")["value"])
    prediction = tree.decode(read(root / "forecasts/sampled_attention.json")["predictions"])
    observation = tree.decode(read(root / "observations/edited_candidates.json")["value"])
    require(binding == record["source_binding"] and set(binding["native_candidates"]) == {"zero", "text"}
            and binding["query_indices"] == source["attention_forecast_plan"]["query_indices"] == prediction["query_indices"]
            and binding["main_seal_sha256"] == e.run["source"]["transaction"]["seal_sha256"],
            "secondary source/outer seal/query binding differs")
    for name in ("zero", "text"):
        declared = binding["native_candidates"][name]
        require(declared == {"input_parent": source["input_cuts"][name], "qkv": candidates[name]["observed_qkv"],
                            "dispatch_profile": candidates[name]["dispatch_profile"]}, "secondary forecast used non-source operands")
    expected_refs = {h["canonical"](rec) for name in ("zero", "text") for rec in candidates[name]["observed_qkv"].values()}
    require({h["canonical"](rec) for rec in binding_array_refs(binding)} == expected_refs,
            "secondary frontier contains extra target arrays or lacks declared source QKV")
    for rec in binding_array_refs(binding):
        e.array(rec)
    expected_profile = {"extra_positional_arguments": 0, "attn_mask_present": False, "dropout_p": 0.,
        "is_causal": False, "scale": None, "enable_gqa": False, "attention_kwargs_present": False,
        "backend": None, "parallel_config_present": False}
    supported = all(candidate["dispatch_profile"] == expected_profile for name, candidate in candidates.items() if name in {"zero", "text"})
    require(prediction["instrument_status"] == ("applicable" if supported else "not-applicable")
            and observation["instrument_status"] == record["instrument_status"] == prediction["instrument_status"]
            and observation["counts"]["candidate_calls"] == 5 and observation["counts"]["adapter_steps"] == 0
            and observation["counts"]["acts"] == 0 and record["text_mass_measured"] is False,
            "secondary applicability/target-call census differs")
    result = inspect_attention_samples(prediction, candidates, e.array, e.typed["quantize_bf16"])
    require(set(observation["samples"]) == set(result["cases"]), "saved numerical target sample inventory differs")
    for name, sample in observation["samples"].items():
        candidate = candidates[name]
        expected = e.array(candidate["attention_dispatch_output"])[:, np.asarray(prediction["query_indices"]) + candidate["text_rows"]]
        require(exact_bits(sample["attention"], expected) and sample["source_dispatch_array"] == candidate["attention_dispatch_output"]
                and sample["actual_native_dtype"] == candidate["attention_dispatch_output"]["descriptor"]["dtype"]
                and sample["text_mass_measured"] is False, "secondary observation borrowed another target dispatch sample")
        reported = record["comparisons"][name]
        require(reported == {"comparisons": sample["comparisons"], "actual_rms": sample["actual_rms"]},
                "secondary saved comparison record differs")
        result["cases"][name]["reported_comparisons"] = reported
        result["cases"][name]["comparison_metadata_deltas"] = {
            label: {key: result["cases"][name][result_key][key] - sample["comparisons"][label][key]
                    for key in ("max_abs", "rms", "signed_mean", "l2")}
            for label, result_key in (("fp64", "fp64_error"), ("cast_bf16", "cast_bf16_error"))}
    timing = numeric_timing(timeline)
    seals = [event for event in timeline if event["kind"] == "attention_forecast_sealed"]
    if len(seals) == 1:
        require(seals[0]["seal"] == record["seal"] and seals[0]["counts"] == binding["preseal_counts"]
                and seals[0]["source_binding_sha256"] == h["sha"](h["canonical"](binding)),
                "secondary seal/source-binding event differs")
    result.update(verification=verification, seal=record["seal"], timing=timing, source_binding=binding,
        signed_error_convention="actual native sample minus sealed forecast",
        prospective_claim="qualified" if timing["qualified"] and supported else "not established",
        novelty={"K": "novel conditional forecast", "V": "novel conditional forecast", "KV": "represented native-text consistency control"},
        dataflow_limit="Source-reviewed trusted callback closes over only supplied QKV arrays, while native-source outputs remain resident; not hermetic isolation or an independent MLP/RGB forecast.")
    return result


def analyze(root, *, failed=False):
    return analyze_evidence(Evidence(root, failed=failed))


def analyze_evidence(e, *, source_adapter=None):
    """Audit one real panel through an explicit source-custody adapter.

    The default retains the historical imported-source contract. Fresh-context
    callers supply a separate validator and a real parent-job Evidence view;
    they must not manufacture historical reports or child scheduler jobs.
    """
    run, data, h = e.run, e.data, e.h
    read, sha = e.read, h["sha"]
    plan, request = run["plan"], run["request"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    prefixes = read(data / "prefixes.json")["prefixes"]
    candidates = read(data / "candidates.json")["candidates"]
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = run["tree"].decode(read(run["transaction"] / "forecasts/single_kv_hypotheses.json")["predictions"])
    require((plan == request["config"] if source_adapter is None else
             source_adapter.plan_matches(e, plan, request)) and hypotheses == plan["hypotheses"]
            and source["request_sha256"] == request["request_sha256"] == summary["request_sha256"]
            and source["source_hashes"] == request["source_hashes"]
            and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"]
            and source["model_identity"] == summary["model_identity"] and source["branches"] == plan["branches"],
            "sealed plan/source/model/count binding differs")
    require(observed["source_sha256"] == sha((data / "source.json").read_bytes())
            and observed["prefixes"] == prefixes and observed["candidates"] == candidates
            and observed["counts"] == summary["counts"] and observed["RGB_forecast"] is False,
            "durable observation/source/prefix/candidate binding differs")
    source_lineage = (historical_source(e, source, plan, request) if source_adapter is None
                      else source_adapter.verify(e, source, summary))
    zero, text = (source["input_cuts"][key] for key in ("zero", "text"))
    require(set(e.fields(zero)) == set(e.fields(text))
            and all(e.fields(zero)[key] == e.fields(text)[key] for key in e.fields(zero) if key != "text")
            and e.point(zero) == e.point(text) and e.point(zero)["logical_step"] == 3
            and e.point(zero)["phase"] == "single" and e.point(zero)["index"] == 0,
            "paired single0 inputs differ outside working text")
    return analyze_bound_panel(e, source, summary, prefixes, candidates, observed, hypotheses,
                               source_lineage, source_adapter=source_adapter)


def historical_source(e, source, plan, request):
    """Original historical-source checks, unchanged in scientific scope."""
    run, h, read = e.run, e.h, e.read
    sha = h["sha"]
    payload = run["root"] / "payload"
    selection = read(payload / "source-selection.json")
    prior, prior_audit = (read(payload / name) for name in ("prior-source.json", "prior-audit.json"))
    prior_prefixes = read(payload / selection["prefix_record"]["path"])["prefixes"]
    require(source["source_selection"] == selection and source["input_cuts"] == selection["input_cuts"] == plan["input_cuts"]
            and selection["files"] == {k: v for k, v in request["source_hashes"].items() if k.startswith("input-store/")}
            and sha((payload / selection["source_record"]["path"]).read_bytes()) == selection["source_record"]["sha256"]
            and sha((payload / "prior-audit.json").read_bytes()) == selection["audit_sha256"]
            and sha((payload / selection["prefix_record"]["path"]).read_bytes()) == selection["prefix_record"]["sha256"]
            and prior_audit["mechanics_status"] == "valid"
            and prior_audit["source"]["job_id"] == selection["prior_job_id"] == "job-b0da807f5d60"
            and prior_audit["source"]["archive_sha256"] == selection["archive_sha256"]
            == "fc47bc6de4709de88812d71e301209f5ddaf20ac7af1a174019af7385a265208"
            and prior_audit["source"]["request_sha256"] == selection["request_sha256"]
            and prior["checkpoint_sha256"] == source["checkpoint_sha256"]
            and prior["model_identity"] == source["model_identity"] and prior["execution"] == source["execution"],
            "prior audited occurrence custody differs")
    implementation = lambda mapping: {Path(name).name: digest for name, digest in mapping.items()}
    require(implementation(source["implementation_sources"]) == implementation(prior["implementation_sources"])
            and source["prior_implementation_sources"] == prior["implementation_sources"], "native implementation byte identity changed")
    oldrows = {}
    for name in ("zero", "text"):
        binding = selection["prior_rows"][name]
        require(sha((payload / binding["path"]).read_bytes()) == binding["sha256"], "prior selected row bytes differ")
        oldrows[name] = read(payload / binding["path"])
        require(oldrows[name]["image"]["sha256"] == plan["expected_pngs"][name], "prior reference PNG binding differs")
    zero, text = (source["input_cuts"][key] for key in ("zero", "text"))
    text_actions = oldrows["text"]["acts"]
    require(oldrows["zero"]["id"] == "zero" and oldrows["text"]["id"] == "late.text"
            and zero == prior_prefixes["zero"]["single0"] == oldrows["zero"]["parent"] == oldrows["text"]["parent"]
            and len(text_actions) == 1 and text_actions[0]["slot"] == "text"
            and text_actions[0]["parent"] == zero and text_actions[0]["post_act"] == text,
            "selected single0 pair is not the actual prior text-only Act")
    return {"kind": "historical-import", "source_selection": selection}


def analyze_bound_panel(e, source, summary, prefixes, candidates, observed, hypotheses,
                        source_lineage, *, source_adapter=None):
    """Shared numerical/receipt panel audit after source-specific admission."""
    run, data, h = e.run, e.data, e.h
    read, sha, typed, plan = e.read, h["sha"], e.typed, run["plan"]
    zero, text = (source["input_cuts"][key] for key in ("zero", "text"))
    own_c = e.fields(zero)["conditioning"]
    for name, cut in (("zero", zero), ("text", text)):
        for key in ("latent", "conditioning", "image", "text"):
            rec = source["input_arrays"][name][key]
            require(exact_bits(e.array(rec), e.slot(cut, key)) and rec["native_stride"] == e.fields(cut)[key]["stride"],
                    "saved source port/cut/stride differs")
    retained_fields = {cut: e.fields(cut) for cut in e.store["cuts"]}
    for cut, manifest in e.store["cuts"].items():
        require(manifest["model_identity"] == source["model_identity"] and manifest["execution"] == source["execution"]
                and e.fields(cut)["conditioning"] == own_c
                and all(e.fields(cut)[key] == e.fields(zero)[key] for key in e.fields(zero)
                        if key not in {"text", "image", "context", "latent", "noise_prediction", "step", "cursor", "phase"}),
                "retained model/conditioning/schedule/ID closure differs")
    for receipt in e.receipts.values():
        require(receipt["model_identity"] == source["model_identity"] and receipt["execution"] == source["execution"],
                "public receipt identity differs")
    points = BASE["receipt_points"](e.receipts.values())
    require(all(cut not in e.store["cuts"] or e.point(cut) == point for cut, point in points.items()),
            "retained cut/receipt point disagreement")
    require(set(prefixes) == {"zero", "text"} and set(candidates) == {"zero", "text", "K", "V", "KV"},
            "paired prefixes/five candidate inventory differs")
    prefix_paths = {}
    for name, prefix in prefixes.items():
        start = source["input_cuts"][name]
        path = e.walk(start, prefix["post_single0"], limit=3)
        require(prefix["input_cut"] == start and path["native_steps"] == 1
                and prefix["receipt"] == e.receipts[path["receipts"][0]]
                and e.point(prefix["post_single0"])["phase"] == "single"
                and e.point(prefix["post_single0"])["index"] == 1, "native single0 prefix lineage differs")
        path["conditioning_coverage"] = BASE["conditioning_coverage"](path, e.receipts, retained_fields, own_c, e.point)
        prefix_paths[name] = path
    ports, qkvs, checks = {}, {}, {}
    for name, candidate in candidates.items():
        parent = text if name == "text" else zero
        expected_selection = name if name in {"K", "V", "KV"} else "native"
        require(candidate["input_parent"] == parent and candidate["selection"] == expected_selection
                and candidate["source_parent"] == (text if expected_selection != "native" else None)
                and candidate["dispatch_calls"] == 1, "candidate source/selector/dispatch count differs")
        values = {key: e.array(rec) for key, rec in candidate["ports"].items()}
        n = candidate["text_rows"]
        require(exact_bits(values["residual"][:, :n], e.slot(parent, "text"))
                and exact_bits(values["residual"][:, n:], e.slot(parent, "image"))
                and exact_bits(values["updated"][:, :n], values["text"])
                and exact_bits(values["updated"][:, n:], values["image"]), "candidate input/output partition differs")
        checks[name] = parallel_residual(values["residual"], values["gate"], values["fused_projection"],
            values["updated"], typed["quantize_bf16"])
        pre, act = values["image_mlp_preactivation"], values["image_mlp_activation"]
        require(pre.ndim == act.ndim == 3 and pre.shape[:2] == act.shape[:2] == values["image"].shape[:2]
                and pre.shape[-1] == 2 * act.shape[-1], "parallel image MLP port geometry differs")
        ports[name] = values
        qkvs[name] = ({k: e.array(v) for k, v in candidate["observed_qkv"].items()},
                     {k: e.array(v) for k, v in candidate["used_qkv"].items()})
        require(e.array(candidate["attention_dispatch_output"]).shape == qkvs[name][0]["q"].shape,
                "attention dispatch retained output ABI differs")
    controls = {}
    for name, candidate in candidates.items():
        choose = candidate["selection"]
        checks[name]["operand_selection"] = operand_selection(*qkvs[name], qkvs["text"][0], text_rows=candidate["text_rows"],
            select_k=choose in {"K", "KV"}, select_v=choose in {"V", "KV"})
        require(candidate["partition_controls"] == {key: True for key in
            ("q_exact_recipient", "k_image_exact_recipient", "v_image_exact_recipient", "k_text_exact_selected", "v_text_exact_selected")},
            "recorded partition control disagrees")
        if name in {"zero", "text"}:
            for key in ("image", "text"):
                controls[name + "_candidate_" + key + "_exact_native"] = exact_bits(ports[name][key], e.slot(prefixes[name]["post_single0"], key))
        else:
            controls[name + "_observed_recipient_QKV_exact"] = all(exact_bits(qkvs[name][0][key], qkvs["zero"][0][key]) for key in ("q", "k", "v"))
    n = candidates["zero"]["text_rows"]
    controls["paired_image_QKV_exact"] = all(exact_bits(qkvs["zero"][0][key][:, n:], qkvs["text"][0][key][:, n:]) for key in ("q", "k", "v"))
    for key in ("image_mlp_preactivation", "image_mlp_activation"):
        for name in candidates:
            controls[name + "_" + key + "_exact_zero"] = exact_bits(ports[name][key], ports["zero"][key])
    for key in ("attention_dispatch_output",):
        controls["KV_image_attention_exact_text"] = exact_bits(e.array(candidates["KV"][key])[:, n:], e.array(candidates["text"][key])[:, n:])
    controls["KV_image_fused_projection_exact_text"] = exact_bits(ports["KV"]["fused_projection"][:, n:], ports["text"]["fused_projection"][:, n:])
    controls["KV_image_exact_text"] = exact_bits(ports["KV"]["image"], ports["text"]["image"])
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    require([item["sequence"] for item in timeline] == list(range(len(timeline))), "timeline sequence differs")
    seals = [item for item in timeline if item["kind"] == "hypotheses_sealed"]
    require(len(seals) == 1 and source["preseal_counts"] == seals[0]["counts"]
            and source["preseal_native_entries"] == seals[0]["native_entries_before_release"] == 0
            and all(seals[0]["counts"][key] == 0 for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "prompt_encodings", "noise_draws"))
            and all(item["sequence"] > seals[0]["sequence"] for item in timeline if item["kind"] in {"candidate_start", "candidate_complete", "branch_start", "branch_complete"}),
            "native/candidate work preceded seal")
    specs = {spec["id"]: spec for spec in plan["branches"]}
    rows = {row["id"]: row for row in observed["rows"]}
    starts = {item["row"]: item for item in timeline if item["kind"] == "branch_start"}
    completes = {item["row"]: item for item in timeline if item["kind"] == "branch_complete"}
    require(len(rows) == len(observed["rows"]) == len(specs) == 9 and set(rows) == set(specs) == set(starts) == set(completes), "nine row inventory differs")
    paths, pixels, latents = {}, {}, {}
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row and all(row[key] == value for key, value in specs[name].items())
                and seals[0]["sequence"] < starts[name]["sequence"] < completes[name]["sequence"], "row record/spec/timing differs")
        parent = prefixes["text" if name == "text" else "zero"]["post_single0"]
        require(row["parent"] == parent and row["start_point"] == e.point(parent), "row borrowed a different native prefix")
        if name in {"zero", "text"}:
            expected = {}
        elif name == "sham":
            expected = {key: e.slot(parent, key) for key in ("image", "text")}
        elif name in {"K.image", "V.image", "KV.image", "KV.both"}:
            selected = name.split(".")[0]
            expected = {"image": ports[selected]["image"]}
            if name == "KV.both":
                expected["text"] = ports[selected]["text"]
        elif name == "text.text":
            expected = {"text": e.slot(prefixes["text"]["post_single0"], "text")}
        else:
            require(name == "coherent", "unknown branch recipe")
            expected = {key: e.slot(prefixes["text"]["post_single0"], key) for key in ("image", "text")}
        require([action["slot"] for action in row["acts"]] == list(expected), "Act slot inventory differs")
        current = parent
        for action in row["acts"]:
            current = e.act(action, current, expected[action["slot"]])
        path = e.walk(parent, row["final_cut"], row["acts"])
        require(path["native_steps"] == 21 and e.point(row["final_cut"])["logical_step"] == 4
                and e.point(row["final_cut"])["phase"] == "project"
                and row["continuation_receipts"] == [e.receipts[rid] for rid in path["receipts"] if e.receipts[rid]["operation"] == "continue"],
                "last-step suffix horizon/receipt chain differs")
        path["conditioning_coverage"] = BASE["conditioning_coverage"](path, e.receipts, retained_fields, own_c, e.point)
        require(typed["matches"](e.slot(zero, "conditioning"), row["final_conditioning"], bf16=True), "final C proof differs")
        latents[name] = e.array(row["final_latent"])
        require(exact_bits(latents[name], e.slot(row["final_cut"], "latent")), "final latent array/cut differs")
        pixels[name], paths[name] = e.png(row["image"]), path
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    expected_emissions = [("prefix." + name, rid) for name in ("zero", "text") for rid in prefix_paths[name]["receipts"]]
    expected_emissions += [(spec["id"], rid) for spec in plan["branches"] for rid in paths[spec["id"]]["receipts"]]
    require([item["sequence"] for item in emissions] == list(range(len(emissions)))
            and [(item["branch"], item["receipt"]) for item in emissions] == expected_emissions, "physical receipt emission census differs")
    counts = summary["counts"]
    require(sum(path["native_steps"] for path in paths.values()) + 2 == counts["adapter_steps"] == 191
            and sum(path["acts"] for path in paths.values()) == counts["acts"] == 10
            and len(emissions) == counts["receipt_emissions"] == 201
            and counts["candidate_calls"] == 5 and counts["images"] == counts["vae_calls"] == 9
            and counts["prompt_encodings"] == counts["noise_draws"] == 0
            and counts["model_loads"] == (1 if source_adapter is None else source_adapter.panel_model_loads)
            and counts["physical_batch"] == 1,
            "finite physical work census differs")
    comparisons = {name: {base: {"latent": distance(latents[name], latents[base]), "rgb": distance(pixels[name], pixels[base])}
                          for base in ("zero", "text")} for name in rows}
    if source_adapter is None:
        for name in ("zero", "text"):
            controls[name + "_png_equals_prior"] = rows[name]["image"]["sha256"] == plan["expected_pngs"][name]
    for name, reference in (("sham", "zero"), ("coherent", "text")):
        controls[name + "_PNG_exact_" + reference] = rows[name]["image"]["sha256"] == rows[reference]["image"]["sha256"]
        controls[name + "_latent_exact_" + reference] = comparisons[name][reference]["latent"]["exact_base"]
    declared = {name + "_candidate_" + key + "_exact_native": controls[name + "_candidate_" + key + "_exact_native"]
                for name in ("zero", "text") for key in ("image", "text")}
    declared.update({name + "_" + key + "_exact_zero": controls[name + "_" + key + "_exact_zero"]
                     for name in candidates for key in ("image_mlp_preactivation", "image_mlp_activation")})
    declared.update(KV_image_exact_text_candidate=controls["KV_image_exact_text"],
        KV_image_exact_text_native=exact_bits(ports["KV"]["image"], e.slot(prefixes["text"]["post_single0"], "image")),
        sham_png_equals_zero=controls["sham_PNG_exact_zero"], coherent_png_equals_text=controls["coherent_PNG_exact_text"])
    if source_adapter is None:
        declared.update(zero_png_equals_prior=controls["zero_png_equals_prior"],
                        text_png_equals_prior=controls["text_png_equals_prior"])
    require(observed["control_observations"] == summary["control_observations"] == declared,
            "worker named control observations differ from saved arrays")
    numerical_attention = audit_attention_transaction(e, source, candidates, timeline)
    require(observed["numerical_attention_forecast"] == summary["numerical_attention_forecast"]
            == read(data / "attention-forecast.json"), "main/secondary numerical observation binding differs")
    qualified = all(controls.values())
    differentials = {name: {"from": a, "to": b, "latent": distance(latents[b], latents[a]), "rgb": distance(pixels[b], pixels[a])}
        for name, a, b in (("KV_text_return", "KV.image", "KV.both"), ("native_text_return", "zero", "text.text"),
                           ("image_given_native_text", "text.text", "coherent"))}
    return {"schema": "mdb-flux-single-kv-audit-v1", "source": run["source"], "mechanics_status": "valid",
        "execution_status": run["execution_status"],
        "instrument_status": "native-replay-qualified" if qualified else "replay-unresolved", "terminal_status": "unassessed",
        "counts": counts, "typed_store_counts": e.store["counts"], "control_observations": controls,
        "replay_qualification": {"controls": controls, "status": "qualified" if qualified else "unresolved",
            "scope": "Named native/local-KV/replay controls, separate from scientific trends; negative controls retain all rows."},
        "paired_source_checks": {"only_working_text_differs": True, "input_cuts": source["input_cuts"],
                                 "prefixes": prefix_paths,
                                 **({"lineage": source_lineage} if source_adapter is not None else {})},
        "candidate_checks": checks, "native_paths": paths, "comparisons": comparisons,
        "image_KV_interactions": {"Z4_latent": kv_interaction(latents), "final_RGB": kv_interaction(pixels)},
        "text_propagation_differentials": differentials, "sealed_hypotheses": hypotheses,
        "numerical_attention_forecast": numerical_attention,
        "preseal_counts": seals[0]["counts"], "preseal_seconds": seals[0]["seconds"], "models_executed": False,
        "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha((HERE / name).read_bytes()) for name in
            ("analyze_flux_rescue_routes.py", "summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "limits": (["One altered observed seed47 history; nine dependent futures, not independent prompt/seed replication."]
                   if source_adapter is None else source_adapter.limits) + [
            "Only working text differs at selected single0 source. Stored C identical; no further projection or declared denoiser C read.",
            "Parallel attention/MLP paths with one fused projection; attention module return includes MLP. No sequential joint-block equation assumed.",
            "Retained operands and native reference arrays observed after seal; structural equality prediction, not an independent numeric or RGB forecast.",
            "Only selected full cuts retained; intermediate receipt points/declared footprints checked against unchanged source, not independently read-traced.",
            "All distance ranks, K/V semantic effects and text-only appearance predictions exploratory; no semantic-quality gate."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--failed", action="store_true", help="Audit retained science from a genuinely failed publication job")
    args = parser.parse_args()
    result = analyze(args.run_dir, failed=args.failed)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "counts")}))
