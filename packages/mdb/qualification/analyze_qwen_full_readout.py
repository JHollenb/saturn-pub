"""Saved-array audit of a full-vocabulary, first-token Qwen readout forecast.

Reuses the conditional-prefix/typed-suffix auditor, then checks full head custody,
all-vocabulary logits and the declared affine envelope. Does not load a model,
execute equation modules, rerun replay, or fetch the external head matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np


def audit(run_dir):
    base_module = runpy.run_path(str(Path(__file__).with_name("analyze_qwen_mlp_forecast.py")))
    base = base_module["audit"](run_dir)
    h = base_module["helpers"]()
    run = h["source_run"](run_dir, "qwen-mlp-result")
    data, tx, tree = run["data"], run["transaction"], run["tree"]
    if not run["plan"].get("full_vocabulary_readout"):
        raise ValueError("full-readout audit requires an explicitly declared full-vocabulary campaign")
    forecast = tree.decode(h["read"](tx / "forecasts/campaign.json")["predictions"])
    observed = tree.decode(h["read"](tx / "observations/native.json")["value"])["rows"]
    binding = h["read"](data / "full-head-binding.json")
    sealed_binding = tree.decode(h["read"](tx / "frontiers/full_head.json")["value"])
    after = h["read"](data / "full-head-post-execution.json")
    head_checks = {stage: h["read"](data / ("full-head-" + stage + ".json"))
                   for stage in ("pre-prediction", "at-release", "post-execution")}
    if (binding != sealed_binding or binding["native_parameter"] != "lm_head.weight"
            or binding["matrix_serialized"] or not binding["copied_before_prefixes"]
            or binding["readonly_copy_sha256"] != binding["descriptor"]["sha256"]
            or binding["descriptor"]["dtype"] != "torch.float32"
            or binding["checkpoint_weights_sha256"] != base["model_identity"]["weights_sha256"]
            or binding["checkpoint_files_sha256"] != base["model_identity"]["checkpoint_files_sha256"]
            or after["descriptor"] != binding["descriptor"] or not after["exact_preseal_binding"]):
        raise ValueError("full immutable head provenance/preseal/post-execution binding differs")
    vocab, hidden = binding["descriptor"]["shape"]
    for stage, proof in head_checks.items():
        if (proof["descriptor"] != binding["descriptor"] or not proof["exact_preseal_binding"]
                or proof["readonly_copy_sha256"] != binding["readonly_copy_sha256"]
                or proof["checkpoint_weights_sha256"] != binding["checkpoint_weights_sha256"]):
            raise ValueError("full actual/copy head checksum differs at " + stage)
    extraction, norm = binding["extraction_recipe"], binding["final_rmsnorm_contract"]
    if (extraction["callable_source_sha256"] != run["request"]["source_hashes"]["qwen_mlp_forecast_campaign.py"]
            or extraction["parameter_name"] != "lm_head.weight" or extraction["shape"] != [vocab, hidden]
            or extraction["token_id_row_order"] != {"kind": "identity", "start": 0, "stop": vocab, "step": 1}
            or norm["module"] != "model.norm" or not norm["type"].endswith(".Qwen2RMSNorm")
            or norm["bias"] is not None or norm["axis"] != -1 or norm["mean_square_divisor"] != hidden
            or norm["weight_shape"] != [hidden] or norm["weight_dtype"] != "torch.float32"
            or not np.isfinite(norm["epsilon"]) or norm["epsilon"] <= 0):
        raise ValueError("source-bound full-head extraction/token-order/final RMSNorm contract differs")
    envelope_reports, row_reports = {}, {}
    for prompt, full in forecast["full_vocabulary"].items():
        a, b, envelope = full["a"], full["b"], full["envelope"]
        errors = []
        if (a.shape != (vocab,) or b.shape != a.shape or full["vocab_size"] != vocab
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or full["max_chunk_rows"] > 4096 or forecast["rows"][prompt]["cut"].shape != (1, 1, hidden)
                or full["rms_epsilon"] != norm["epsilon"] or full["rms_epsilon"] != forecast["rows"][prompt]["rms_epsilon"]):
            errors.append("affine coefficients/vocabulary/cut geometry or bounded chunk declaration differs")
        segments = envelope["segments"]
        previous = 0.
        for segment in segments:
            left, right = segment["left"], segment["right"]
            if left != previous or not 0 <= left < right <= 1:
                errors.append("winner-envelope interval coverage differs")
            previous = right
            midpoint = (left + right) / 2
            if int(np.argmax(a + midpoint * b)) != segment["token_id"]:
                errors.append("retained affine-envelope midpoint winner differs")
        if previous != 1. or envelope["domain"] != [0., 1.]:
            errors.append("winner envelope does not cover the declared unit-dose domain")
        for point in envelope["breakpoints"]:
            if not point["tied_token_ids"] or point["token_id"] != min(point["tied_token_ids"]):
                errors.append("envelope tie winner is not lowest declared token ID")
        envelope_reports[prompt] = {"mechanics_errors": errors, "vocab_size": vocab,
            "envelope": envelope, "coefficient_descriptors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": h["sha"](value.astype("<f8", copy=False).tobytes())} for name, value in (("a", a), ("b", b))},
            "scope": "saved affine-coefficient self-consistency; external matrix products are not recomputed"}
    for row_id, row in observed.items():
        errors = []
        full = forecast["full_vocabulary"][row["prompt_id"]]
        predicted = full["doses"][format(float(row["dose"]), ".17g")]
        actual, expected = row["full_vocabulary_logits"], predicted["logits"]
        if (actual.shape != (1, 1, vocab) or expected.shape != actual.shape
                or actual.dtype != np.float32 or expected.dtype != np.float64
                or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
            errors.append("full-vocabulary logit ABI/finite differs")
        if not base_module["exact"](actual[:, :, base["candidate_ids"]], row["selected_logits"]):
            errors.append("full native logits do not reproduce the retained selected rows")
        ranking = np.argsort(-expected.reshape(-1), kind="stable")
        first, second = int(ranking[0]), int(ranking[1]) if vocab > 1 else None
        if (first != predicted["token_id"] or second != predicted["runner_up_token_id"]
                or predicted["normalizer"] <= 0 or not np.isfinite(predicted["normalizer"])):
            errors.append("declared full forecast winner/runner-up/normalizer differs")
        if second is not None and float(expected.reshape(-1)[first] - expected.reshape(-1)[second]) != predicted["top2gap"]:
            errors.append("declared full forecast top-two gap differs")
        affine_logits = (full["a"] + row["dose"] * full["b"]) / predicted["normalizer"]
        if not np.array_equal(affine_logits.reshape(expected.shape), expected):
            errors.append("published full forecast logits differ from sealed affine coefficients/normalizer")
        actual_first = int(actual.reshape(-1).argmax())
        if actual_first != row["full_vocab_greedy_token"]:
            errors.append("full native greedy ID differs from actual full logits")
        runner = row["full_vocab_runner_up_token"]
        ordered_values = np.partition(actual.reshape(-1), -2)[-2:]
        if actual.reshape(-1)[runner] != ordered_values.min() or float(ordered_values.max() - ordered_values.min()) != row["full_vocab_top_two_margin"]:
            errors.append("native full runner-up/top-two margin differs")
        comparison = h["difference"](actual, expected)
        row_reports[row_id] = {"mechanics_errors": errors, "computation_id": "qwen2.full-vocabulary.affine-RMS-readout",
            "prompt_id": row["prompt_id"], "arm": row["arm"], "dose": row["dose"], "vocab_size": vocab,
            "full_logit_comparison": comparison, "forecast_token_id": first, "actual_token_id": actual_first,
            "full_vocab_argmax_matches": first == actual_first, "forecast_top2gap": predicted["top2gap"],
            "actual_top2gap": row["full_vocab_top_two_margin"], "normalizer": predicted["normalizer"],
            "full_vs_selected_forecast_residual": h["difference"](expected[:, :, base["candidate_ids"]],
                forecast["rows"][row["prompt_id"]]["doses"][format(float(row["dose"]), ".17g")]["logits"]),
            "observed_first_token_text": row["greedy_text"], "observed_generated_text": row["generated_text"],
            "observed_generated_token_ids": row["generated_token_ids"], "continuation_forecast": False,
            "continuation_check": base["rows"][row_id].get("continuation_check"),
            "selected_suffix_audit": base["rows"][row_id]}
    valid = (base["mechanics_status"] == "valid" and not any(row["mechanics_errors"] for row in row_reports.values())
             and not any(row["mechanics_errors"] for row in envelope_reports.values()))
    return {"schema": "mdb-qwen-full-readout-audit-v1", "mechanics_status": "valid" if valid else "instrument-errors-retained",
        "source": base["source"], "model_identity": base["model_identity"], "execution": base["execution"],
        "full_head_binding": binding, "full_head_checks": head_checks, "counts": base["counts"],
        "base_mechanics_errors": base["mechanics_errors"], "checkpoint_files": base["checkpoint_files"],
        "installed_model_sources": base["installed_model_sources"], "native_weight_binding": base["native_weight_binding"],
        "prefixes": base["prefixes"], "pre_release_counters": base["pre_release_counters"],
        "envelopes": envelope_reports, "rows": row_reports, "base_mechanics_status": base["mechanics_status"],
        "scientific_summary": {"rows": len(row_reports), "full_vocab_argmax_matches": sum(row["full_vocab_argmax_matches"] for row in row_reports.values()),
            "max_full_logit_abs_residual": max((row["full_logit_comparison"]["max_abs"] for row in row_reports.values()), default=None),
            "row_error_counts": dict(Counter(error for row in row_reports.values() for error in row["mechanics_errors"]))},
        "audit_sources": {name: h["sha"](Path(__file__).with_name(name).read_bytes()) for name in
                          ("analyze_qwen_mlp_forecast.py", "summarize_context_window.py")},
        "references": {name: {"path": str(path.relative_to(Path(run_dir).resolve())), "sha256": h["sha"](path.read_bytes())}
            for name, path in {"full_head_binding": data / "full-head-binding.json", "full_head_after": data / "full-head-post-execution.json",
                               "full_head_pre_prediction": data / "full-head-pre-prediction.json", "full_head_at_release": data / "full-head-at-release.json"}.items()},
        "terminal_status": "unassessed", "trend_status": "full-vocabulary first-token errors and counterexamples retained",
        "limits": [item for item in base["limits"] if not item.startswith("Only selected fixed vocabulary")]
                  + ["All vocabulary rows are forecast for the first token from the observed native prefix cut; subsequent text remains native observation.",
                     "The complete head is externally supplied and checkpoint-bound by worker descriptors, not archived or independently reloaded here.",
                     "Audit checks saved coefficients/logits and envelope midpoints; it does not reexecute source equations or native model arithmetic."]}


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
    result["auditor_source_sha256"] = helpers_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "rows": len(result["rows"]), "source_sha256": helpers_sha, "model_calls": 0}))


if __name__ == "__main__":
    main()
