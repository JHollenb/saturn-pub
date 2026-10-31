"""Audit Qwen MLP forecasts from saved arrays and typed receipts only.

This is an observed-prefix-conditioned, selected-vocabulary suffix audit.
No model, independent equation, replay or tokenizer execution is performed.
"""
from __future__ import annotations

import argparse
import json
import runpy
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from mdb.discovery_catalog import StructuralCatalog
from mdb.forward_ports import NativePortCatalog
from mdb.forward_symbols import verify_trace

ARMS = {"native": 1., "sham": 1., "zero": 0., "half": .5}
STAGES = {"postnorm", "delta", "residual", "final_norm", "logits"}


@lru_cache(maxsize=1)
def helpers():
    return runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))


def descriptor(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": "torch." + str(value.dtype),
            "sha256": helpers()["sha"](value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())}


def exact(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def check_descriptor(value, binding):
    return all(binding.get(key) == expected for key, expected in descriptor(value).items())


def candidate_comparison(actual, predicted, ids):
    h = helpers()
    if actual.shape != (1, 1, len(ids)) or predicted.shape != actual.shape:
        raise ValueError("candidate logit geometry differs from fixed selected rows")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("candidate logits are nonfinite")

    def choice(value):
        row = value.reshape(-1)
        ranking = np.argsort(-row, kind="stable")
        top = int(ranking[0])
        return {"candidate_index": top, "token_id": ids[top], "logit": float(row[top]),
                "top_two_margin": None if len(ids) == 1 else float(row[top] - row[ranking[1]])}

    observed, forecast = choice(actual), choice(predicted)
    return {"residual": h["difference"](actual, predicted), "observed": observed, "forecast": forecast,
            "candidate_argmax_matches": observed["token_id"] == forecast["token_id"],
            "per_candidate_signed_residual": (actual.astype(np.float64) - predicted).reshape(-1).tolist(),
            "scope": "argmax over predeclared selected rows; not a full-vocabulary token forecast"}


def continuation_check(row, baseline, expected_steps):
    """Check native own-token consumption and same-KV/same-first closure."""
    errors, previous = [], row["full_vocab_greedy_token"]
    future = row["continuation"]
    if len(future) != expected_steps:
        errors.append("native continuation horizon differs")
    for index, step in enumerate(future, 1):
        if step["step"] != index or step["consumed_token"] != previous:
            errors.append("native continuation does not consume its own preceding token")
        previous = step["greedy_token"]
    if row.get("generated_token_ids") != [row["full_vocab_greedy_token"], *[step["greedy_token"] for step in future]]:
        errors.append("retained branch token IDs differ from native greedy chain")
    same_kv = row["prefill_kv"]["sha256"] == baseline["prefill_kv"]["sha256"]
    same_first = row["full_vocab_greedy_token"] == baseline["full_vocab_greedy_token"]
    same_future = [step["greedy_token"] for step in future] == [step["greedy_token"] for step in baseline["continuation"]]
    diagnostics = []
    if same_kv and same_first and not same_future:
        diagnostics.append("same prefill KV and first token produced different native future tokens")
    return {"errors": errors, "diagnostics": diagnostics, "same_prefill_KV": same_kv,
            "same_first_token": same_first, "same_future_tokens": same_future,
            "closure_applicable": same_kv and same_first,
            "scope": "observed native continuation only; no future-token forecast or restored-cache claim"}


def audit_prefix(run, name, prefix, source, timeline):
    h, errors = helpers(), []
    trace = h["read"](h["file_path"](run["data"], "prefixes/" + name + ".json"))
    verify_trace(trace)
    if (trace["fingerprint"] != prefix["trace_fingerprint"] or trace["input_identity"] != prefix["input_identity"]
            or trace["model_identity"] != source["model_identity"] or trace["execution"] != source["execution"]):
        errors.append("prefix trace/model/input/execution identity differs")
    contract = trace["metadata"]["prefix_contract"]
    if (contract != prefix["prefix_contract"] or contract["full_model_outputs_computed"]
            or contract["stop_edge"] != "before-local-prehooks-and-module-body"
            or not contract["stop_before"].endswith(".post_attention_layernorm")):
        errors.append("genuine prefix boundary contract differs")
    forbidden = contract["wrapper_stop_before"]
    if any(event["module_path"] == forbidden or event["module_path"].startswith(forbidden + ".")
           for event in trace["events"]):
        errors.append("stopped suffix module body occurs in prefix trace")
    if not check_descriptor(prefix["cut"], prefix["cut_descriptor"]):
        errors.append("prefix last-token cut descriptor differs")
    if "boundary_hidden" in prefix:
        encoded = trace["outputs"]["boundary"]["args"]["$tuple"][0]
        if not check_descriptor(prefix["boundary_hidden"], trace["values"][encoded["$value"]]):
            errors.append("full saved prefix boundary differs from trace output")
        if not exact(prefix["boundary_hidden"][:, -1:, :], prefix["cut"]):
            errors.append("last-token cut differs from actual boundary slice")
    else:
        errors.append("prefix full boundary witness missing; last-token slice cannot be independently joined")
    expected_identity = h["sha"](h["canonical"]({"ids": prefix["token_ids"].tolist(), "prompt": prefix["prompt"]}))
    if expected_identity != prefix["input_identity"]:
        errors.append("prompt/token input identity differs")
    if not any(check_descriptor(prefix["token_ids"], value) and value["origin"] == "input" for value in trace["values"].values()):
        errors.append("prompt tokens are not a prefix input root")
    completes = [event for event in timeline if event["kind"] == "prefix_complete" and event["prompt"] == name]
    if len(completes) != 1 or completes[0]["trace_fingerprint"] != trace["fingerprint"]:
        errors.append("prefix completion timeline binding differs")
    return {"errors": errors, "trace_fingerprint": trace["fingerprint"], "input_identity": prefix["input_identity"],
            "prompt": prefix["prompt"], "token_ids": prefix["token_ids"].tolist(), "cut": descriptor(prefix["cut"]),
            "prefix_contract": contract}


def audit_row(run, row_id, row, prefix, forecast, baseline, receipts, weight_descriptors, ids):
    h, errors = helpers(), []
    trace = h["read"](h["file_path"](run["data"], "suffix-traces/" + row_id + ".json"))
    source_trace = h["read"](h["file_path"](run["data"], "suffix-traces/" + row["prompt_id"] + ".native.json"))
    verify_trace(trace)
    verify_trace(source_trace)
    ports, source_ports = NativePortCatalog(trace), NativePortCatalog(source_trace)
    if (trace["fingerprint"] != row["trace_fingerprint"] or trace["input_identity"] != prefix["input_identity"]
            or trace["model_identity"] != run["seal"]["declarations"]["model_identity"]
            or trace["execution"] != run["native_execution"]):
        errors.append("suffix trace binding differs")
    dose = run["arms"][row["arm"]]
    if row["dose"] != dose:
        errors.append("declared arm dose differs")
    stages = row["suffix"]
    if set(stages) != STAGES or set(trace["outputs"]) != STAGES:
        errors.append("suffix stage inventory differs")
    for name, value in stages.items():
        if value.dtype != np.float32 or not np.isfinite(value).all() or not check_descriptor(value, trace["values"][trace["outputs"][name]["$value"]]):
            errors.append("actual suffix stage/trace differs: " + name)
    expected_ports = {"cut": ("module:postnorm:args", [0]), "postnorm": ("module:postnorm:return", None),
                      "delta": ("module:mlp:return", None), "residual": ("module:finalnorm:args", [0]),
                      "final_norm": ("module:finalnorm:return", None), "logits": ("module:selected_lm_head:return", None)}
    if set(row["actual_ports"]) != set(expected_ports):
        errors.append("actual suffix module port inventory differs")
    for name, (selector, path) in expected_ports.items():
        binding = ports.resolve(selector, path=path)
        value = prefix["cut"] if name == "cut" else stages[name]
        if binding != row["actual_ports"].get(name) or not check_descriptor(value, trace["values"][binding["value_id"]]):
            errors.append("authoritative suffix module port differs: " + name)
    inputs = [value for value in trace["values"].values() if value["origin"] == "input"]
    if len(inputs) != 1 or not check_descriptor(prefix["cut"], inputs[0]):
        errors.append("suffix root is not the genuine prefix last-token cut")
    for name, value in weight_descriptors.items():
        port = ports.resolve("parameter:" + name)
        if not all(trace["values"][port["value_id"]].get(key) == expected for key, expected in value.items()):
            errors.append("actual suffix weight differs from sealed forecast weights: " + name)
    same_cut = exact(row["actual_full_model_cut"], prefix["cut"])
    if not same_cut or same_cut != row["prefix_exact"]:
        errors.append("fresh native model cut differs from prefix-conditioned frontier")
    if set(row["typed_replay_exact"]) != STAGES or not all(row["typed_replay_exact"].values()):
        errors.append("typed replay/native suffix parity differs")
    mlp_port = source_ports.resolve("module:mlp:return")
    if row["native_ports"] != ports.resolve("module:mlp:return"):
        errors.append("native MLP return port differs")
    if len(receipts) != len(source_trace["events"]):
        errors.append("typed primitive count differs from complete common-parent replay")
    program = StructuralCatalog(source_trace).program_id
    for pc, receipt in enumerate(receipts):
        h["public_receipt"](receipt)
        if (receipt["operation"] != "mdb.discovery.native-transition" or receipt["trace_fingerprint"] != source_trace["fingerprint"]
                or receipt["program_id"] != program or receipt["before_pc"] != pc or receipt["after_pc"] != pc + 1
                or receipt["operator"] != source_trace["events"][pc]["operator"] or receipt["mechanics_status"] != "valid"):
            errors.append("typed primitive common-parent/source binding differs at " + str(pc))
    expected_delta = baseline["suffix"]["delta"] * np.float32(dose)
    if row["arm"] == "native":
        if row["acts"]:
            errors.append("native row has unexpected intervention")
    else:
        if len(row["acts"]) != 1:
            errors.append("edited row needs exactly one MLP Act")
        for act in row["acts"]:
            receipt = act["receipt"]
            h["public_receipt"](receipt)
            if (act["port"] != mlp_port or receipt["operation"] != "mdb.discovery.ssa-act"
                    or receipt["trace_fingerprint"] != source_trace["fingerprint"] or receipt["pc"] != mlp_port["available_pc"]
                    or receipt["value_id"] != mlp_port["value_id"] or receipt["mechanics_status"] != "valid"
                    or receipt["act"]["operation"]["id"] != "saturn.builtin.replace"
                    or receipt["act"]["reads"] != [mlp_port["value_id"]] or receipt["act"]["writes"] != [mlp_port["value_id"]]
                    or receipt["act"]["numerical_contract"] != "behavioral-intervention"):
                errors.append("typed MLP Act source/port binding differs")
            if not check_descriptor(expected_delta, receipt["after"]) or not check_descriptor(expected_delta, act["payload_descriptor"]):
                errors.append("Act payload is not the actual baseline native delta scaled after release")
            parameters = receipt["act"]["parameters"]["items"]
            if len(parameters) != 1 or parameters[0]["key"]["value"] != "value" or not check_descriptor(expected_delta, parameters[0]["value"]):
                errors.append("typed Act parameter bytes differ")
    if not exact(stages["delta"], expected_delta):
        errors.append("native suffix dose does not scale the baseline actual MLP output")
    if not exact(row["actual_full_model_delta"], row["actual_full_model_unscaled_delta"] * np.float32(dose)):
        errors.append("full native MLP dose does not scale its own actual unedited output")
    if row["arm"] == "sham" and any(not exact(value, baseline["suffix"][name]) for name, value in stages.items()):
        errors.append("sham clone changes an actual suffix output")
    prediction = forecast["doses"][format(dose, ".17g")]
    predicted_stages = {"postnorm": forecast["mlp"]["postnorm"], "delta": forecast["mlp"]["delta"] * dose,
                        "residual": prediction["residual"], "final_norm": prediction["final_norm"], "logits": prediction["logits"]}
    comparison = {name: h["difference"](value, predicted_stages[name]) for name, value in stages.items()}
    full_residual = h["difference"](row["selected_logits"], stages["logits"])
    native_reported_max = float(np.abs(row["selected_logits"] - stages["logits"]).max())
    if bool(full_residual["changed_elements"] == 0) != row["full_model_suffix_selected_exact"] or native_reported_max != row["full_model_suffix_selected_max_abs"]:
        errors.append("reported native full-model/wrapper selected-logit residual differs")
    future_check = continuation_check(row, baseline, run["plan"].get("continuation_tokens", 3))
    errors.extend(future_check["errors"])
    if future_check["same_prefill_KV"] != row["kv_exact_vs_native"]:
        errors.append("reported prefill KV equality differs")
    if h["sha"](h["canonical"](row["prefill_kv"]["layers"])) != row["prefill_kv"]["sha256"]:
        errors.append("prefill KV descriptor inventory fingerprint differs")
    return {"mechanics_errors": errors, "computation_id": "qwen2.final-MLP.SwiGLU.residual.RMSNorm.selected-readout",
            "prompt_id": row["prompt_id"], "arm": row["arm"], "dose": dose, "stage_comparison": comparison,
            "candidate_comparison": candidate_comparison(stages["logits"], prediction["logits"], ids),
            "full_native_selected_vs_wrapper": full_residual,
            "full_native_delta_vs_wrapper": h["difference"](row["actual_full_model_delta"], stages["delta"]),
            "actual_selected_logits": row["selected_logits"].reshape(-1).tolist(),
            "forecast_selected_logits": prediction["logits"].reshape(-1).tolist(),
            "dose_effect_vs_native": {
                "actual_selected_logit_effect": (row["selected_logits"].astype(np.float64) - baseline["selected_logits"]).reshape(-1).tolist(),
                "forecast_selected_logit_effect": (prediction["logits"] - forecast["doses"]["1"]["logits"]).reshape(-1).tolist(),
                "scope": "same observed prefix cut; full native logits versus independent selected suffix dose equations"},
            "full_vocab_greedy_token": row["full_vocab_greedy_token"], "greedy_text": row["greedy_text"],
            "generated_text": row.get("generated_text"), "generated_token_ids": row.get("generated_token_ids"),
            "continuation": [{key: value for key, value in step.items() if key != "selected_logits"} for step in row["continuation"]],
            "continuation_check": future_check,
            "trace_fingerprint": trace["fingerprint"], "prefix_binding": prefix["trace_fingerprint"],
            "typed_transitions": len(receipts), "typed_acts": len(row["acts"]), "payload_policy": row["payload_policy"]}


def audit(run_dir):
    h = helpers()
    run = h["source_run"](run_dir, "qwen-mlp-result")
    data, tx = run["data"], run["transaction"]
    source, summary = h["read"](data / "source.json"), h["read"](data / "summary.json")
    run["native_execution"] = source["execution"]
    run["arms"] = dict(run["plan"].get("arms", ARMS))
    if source["request"] != run["request"] or run["plan"] != run["request"]["config"]:
        raise ValueError("raw plan/source differs from external request configuration")
    if (source["model_identity"] != run["seal"]["declarations"]["model_identity"]
            or source["model_identity"] != summary["model_identity"] or source["execution"] != summary["execution"]
            or source["execution"]["backend"] != "cpu" or source["execution"]["dtype"] != "float32"):
        raise ValueError("model/native execution identity differs across prefix, transaction and summary")
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    seals = [event for event in timeline if event["kind"] == "global_forecast_sealed"]
    releases = [event for event in timeline if event["kind"] == "native_suffix_released"]
    if (len(seals) != 1 or len(releases) != 1 or seals[0]["sequence"] >= releases[0]["sequence"]
            or seals[0]["seal_sha256"] != run["transaction_report"]["seal_sha256"]
            or [event["sequence"] for event in timeline] != list(range(len(timeline)))):
        raise ValueError("prefix/global seal/native suffix release timeline differs")
    pre_release = seals[0]["pre_release_counters"]
    guarded = ("full_model_calls", "final_postnorm_entries", "final_mlp_entries", "final_norm_entries", "denied_calls")
    if any(pre_release[key] for key in guarded) or summary["pre_release_counters"] != pre_release:
        raise ValueError("native full model/suffix entered before the global conditional forecast seal")
    if any(event["sequence"] >= seals[0]["sequence"] for event in timeline if event["kind"] == "prefix_complete"):
        raise ValueError("prefix collection did not complete before the global forecast seal")
    prefixes = run["tree"].decode(h["read"](tx / "frontiers/prefixes.json")["value"])
    combined_input = h["sha"](h["canonical"]({name: prefix["input_identity"] for name, prefix in prefixes.items()}))
    if combined_input != run["seal"]["declarations"]["input_identity"]:
        raise ValueError("transaction input identity differs from declared prompt-prefix identities")
    exclusion = run["tree"].decode(h["read"](tx / "frontiers/exclusion.json")["value"])
    if exclusion["pre_release_counters"] != pre_release or exclusion["native_suffix_entries"] != 0:
        raise ValueError("sealed suffix-exclusion frontier differs from actual pre-release counters")
    if pre_release["prefix_calls"] != len(prefixes):
        raise ValueError("external genuine prefix count differs from sealed frontiers")
    doc = h["read"](tx / "forecasts/campaign.json")
    forecasts = run["tree"].decode(doc["predictions"])
    observations = run["tree"].decode(h["read"](tx / "observations/native.json")["value"])["rows"]
    if forecasts["full_vocab_greedy_forecast"] != run["plan"].get("full_vocabulary_readout", False) or forecasts["continuations_forecast"]:
        raise ValueError("campaign exceeds its declared conditional forecast scope")
    ids = source["candidate_ids"]
    declared_ids = list(run["plan"].get("candidate_token_ids", []))
    for text in run["plan"].get("candidate_texts", []):
        encoded = source["candidate_text_tokenization"][text]
        if len(encoded) == 1:
            declared_ids.append(encoded[0])
    if ids != list(dict.fromkeys(declared_ids)) or len(ids) != len(set(ids)):
        raise ValueError("selected candidate row IDs differ from predeclared candidate policy")
    if h["sha"](h["canonical"](source["checkpoint_files"])) != source["model_identity"]["checkpoint_files_sha256"]:
        raise ValueError("checkpoint inventory identity differs")
    weight_files = {name: row["sha256"] for name, row in source["checkpoint_files"].items() if name.endswith(".safetensors")}
    if not weight_files or h["sha"](h["canonical"](weight_files)) != run["plan"]["expected_weights_sha256"] or source["model_identity"]["weights_sha256"] != run["plan"]["expected_weights_sha256"]:
        raise ValueError("actual checkpoint weight bytes differ from external model identity pin")
    model_sources = source["installed_model_sources"]
    if not model_sources or any(len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in model_sources.values()):
        raise ValueError("installed HF implementation/config/tokenizer source identity is missing")
    expected_files = run["plan"].get("expected_model_files")
    if expected_files and expected_files != source["checkpoint_files"]:
        raise ValueError("actual checkpoint inventory differs from supplied checkpoint pins")
    weights = run["seal"]["declarations"]["actual_weight_manifest"]
    weight_descriptors = {name: descriptor(run["tree"].select(weights, [name])) for name in weights["items"]}
    expected_weights = {"postnorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                        "finalnorm.weight", "selected_lm_head.weight"}
    if set(weight_descriptors) != expected_weights or weight_descriptors["selected_lm_head.weight"]["shape"][0] != len(ids):
        raise ValueError("actual sealed suffix weight inventory/selected-row geometry differs")
    binding = h["read"](data / "weight-binding.json")
    if set(binding["shared_parameters"]) != expected_weights - {"selected_lm_head.weight"}:
        raise ValueError("native-to-suffix shared weight binding inventory differs")
    for name, row in binding["shared_parameters"].items():
        if not row["exact_native"] or any(row["descriptor"].get(key) != value for key, value in weight_descriptors[name].items()):
            raise ValueError("native-to-suffix shared parameter bytes differ: " + name)
    head = binding["selected_head"]
    if (head["candidate_ids"] != ids or head["native_name"] != "lm_head.weight" or not head["exact_native_selected_rows"]
            or any(head["descriptor"].get(key) != value for key, value in weight_descriptors["selected_lm_head.weight"].items())):
        raise ValueError("selected native vocabulary-row extraction binding differs")
    receipts = defaultdict(list)
    for line in (data / "transition-receipts.jsonl").read_text().splitlines():
        entry = json.loads(line)
        receipts[entry["row"]].append(entry["receipt"])
    errors, rows = [], {}
    expected_rows = {name + "." + arm for name in prefixes for arm in run["arms"]}
    if set(observations) != expected_rows or set(receipts) != expected_rows:
        errors.append("native/suffix/typed replay row inventory differs from declared prompt arms")
    prefix_reports = {name: audit_prefix(run, name, prefix, source, timeline) for name, prefix in prefixes.items()}
    expected_prompts = {prompt["id"]: prompt for prompt in run["plan"]["prompts"]}
    if set(expected_prompts) != set(prefixes) or any(prefix["prompt"] != expected_prompts[name] for name, prefix in prefixes.items()):
        errors.append("prefix prompts differ from externally declared prompt inputs")
    payload_reports = {}
    for row_id, row in observations.items():
        try:
            name = row["prompt_id"]
            forecast = forecasts["rows"][name]
            if forecast["candidate_ids"] != ids or not exact(forecast["cut"], prefixes[name]["cut"].astype(np.float64)):
                raise ValueError("forecast candidate IDs or cut differ from declared observed frontier")
            payload = doc["payloads"][row_id]
            original, staged = (run["tree"].decode(payload[key]) for key in ("original", "value"))
            expected = np.asarray(forecast["mlp"]["delta"] * run["arms"][row["arm"]], dtype=np.float32)
            if not exact(expected, staged) or not exact(original, staged):
                raise ValueError("retained prediction-only FP32 payload/cast differs from sealed analytical dose")
            payload_reports[row_id] = {"descriptor": descriptor(staged), "provenance": payload["provenance"],
                                       "usage": "forecast artifact only; actual Acts use native baseline return"}
            report = audit_row(run, row_id, row, prefixes[name], forecast, observations[name + ".native"],
                               receipts[row_id], weight_descriptors, ids)
            rows[row_id] = report
        except (KeyError, ValueError, TypeError, IndexError) as error:
            rows[row_id] = {"mechanics_errors": [f"row instrument: {type(error).__name__}: {error}"]}
    if summary["counts"]["suffix_records"] != len(rows) or summary["counts"]["replay_primitives"] != sum(map(len, receipts.values())):
        errors.append("suffix/replay summary work differs from retained evidence")
    counts = summary["counts"]
    expected_full_calls = len(prefixes) * len(run["arms"]) * (1 + run["plan"].get("continuation_tokens", 3))
    if counts["prefix_calls"] != len(prefixes) or counts["full_model_calls"] != expected_full_calls or counts["physical_attempts"] != counts["prefix_calls"] + counts["full_model_calls"] + counts["suffix_records"]:
        errors.append("physical prefix/suffix/full-model attempt accounting differs")
    if counts["physical_attempts"] > run["plan"].get("max_native_calls", 64):
        errors.append("declared native physical attempt budget exceeded")
    if counts["acts"] != sum(len(row["acts"]) for row in observations.values()):
        errors.append("typed Act summary count differs")
    if counts["replay_primitives"] > run["plan"].get("max_replay_operations", 8192):
        errors.append("declared replay primitive budget exceeded")
    if counts["denied_calls"]:
        errors.append("native-call guard recorded a denied attempt")
    valid = not errors and not any(row["mechanics_errors"] for row in rows.values()) and not any(row["errors"] for row in prefix_reports.values())
    return {"schema": "mdb-qwen-mlp-forecast-audit-v1", "source": run["source"], "model_identity": source["model_identity"],
            "execution": source["execution"], "mechanics_status": "valid" if valid else "instrument-errors-retained",
            "mechanics_errors": errors, "candidate_ids": ids, "candidate_text_tokenization": source["candidate_text_tokenization"],
            "candidate_literal_labels": {str(token): [text for text, encoded in source["candidate_text_tokenization"].items()
                                                       if encoded == [token]] for token in ids},
            "prefixes": prefix_reports, "pre_release_counters": pre_release, "counts": summary["counts"],
            "suffix_weight_descriptors": weight_descriptors, "checkpoint_files": source["checkpoint_files"],
            "native_weight_binding": binding, "installed_model_sources": model_sources,
            "prediction_only_payloads": payload_reports,
            "scientific_summary": {"rows": len(rows),
                "selected_candidate_argmax_matches": sum(row.get("candidate_comparison", {}).get("candidate_argmax_matches", False) for row in rows.values()),
                "max_selected_logit_abs_residual": max((row["candidate_comparison"]["residual"]["max_abs"] for row in rows.values() if "candidate_comparison" in row), default=None),
                "continuation_closure_applicable": sum(row.get("continuation_check", {}).get("closure_applicable", False) for row in rows.values()),
                "continuation_closure_diagnostics": {name: row["continuation_check"]["diagnostics"] for name, row in rows.items()
                                                     if row.get("continuation_check", {}).get("diagnostics")}},
            "rows": rows, "trend_status": "per-row continuous residuals, margins and text retained", "terminal_status": "unassessed",
            "limits": [*summary["limits"], "Genuine native prefixes supply the cut; this is not a raw-token/full-LM forecast.",
                "Only selected fixed vocabulary rows are forecast; full-vocabulary greedy tokens and subsequent text are native observations.",
                "No model, source equations, native replay or tokenizer executed during this saved-array audit.",
                "Checkpoint-to-selected-weight extraction is source-bound worker provenance; full checkpoint numerical tensors are not reloaded by this audit.",
                "Installed HF source SHA inventories are archive-bound worker observations; external installed files are not refetched by this audit.",
                "Same-KV/same-first-token continuation closure is a mechanics diagnostic, not a forecast or independent scientific certification."],
            "references": {name: {"path": str(path.relative_to(Path(run_dir).resolve())), "sha256": h["sha"](path.read_bytes())}
                for name, path in {"source": data / "source.json", "summary": data / "summary.json",
                    "timeline": data / "timeline.jsonl", "weight_binding": data / "weight-binding.json",
                    "transaction_report": data / "transaction-report.json"}.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result["auditor_source_sha256"] = helpers()["sha"](snapshot.read_bytes())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "rows": len(result["rows"]), "model_calls": 0}))


if __name__ == "__main__":
    main()
