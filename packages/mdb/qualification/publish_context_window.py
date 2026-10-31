"""Publish compact saved-evidence views and public immutable equation records.

No model, forecast equation, intervention, or continuation is executed. The
explicit maintainer audit source supplies only its immutable numerical decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from mdb.equation_comparison import EquationCatalog, compare_equations
from mdb.forecast_transaction import verify_transaction


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def portable(path):
    parts = path.resolve().parts
    return "/".join(parts[parts.index("artifacts"):]) if "artifacts" in parts else path.name


def ref(path, pointer=None):
    result = {"path": portable(path), "sha256": sha(path.read_bytes())}
    if pointer is not None:
        result["json_pointer"] = pointer
    return result


def small_residual(value):
    return {key: value[key] for key in ("shape", "max_abs", "l2", "signed_mean", "minimum_signed", "maximum_signed")}


def small_effect(value):
    return {"rows": value["observed_minus_reference"], "evidence_roles": value["evidence_roles"],
            "RSI_bit1": value["RSI_bit1"], "actual_invariance": value["actual_invariance"],
            "full_bank_flags_and_effects": value["full_bank_flags_and_effects"],
            "head_effects": {name: {"max_abs": item["max_abs"], "signed_mean": item["signed_mean"]}
                             for name, item in value["all_head_effects"].items()}}


def publish(run_dir, output, audit_source):
    run_dir, output, audit_source = run_dir.resolve(), output.resolve(), audit_source.resolve()
    require(not output.exists(), "portable index already exists; choose a new version")
    publisher_raw = Path(__file__).read_bytes()
    snapshot = run_dir / ("publication-source-" + sha(publisher_raw)[:16] + ".py")
    if snapshot.exists():
        require(snapshot.read_bytes() == publisher_raw, "publication source snapshot already binds another version")
    else:
        with snapshot.open("xb") as stream:
            stream.write(publisher_raw)
    analysis_path = run_dir / "analysis-v1.json"
    low_path = run_dir / "low-prefix-comparison-v1.json"
    package_path = run_dir / "package-proof.json"
    analysis, low, package = (read(path) for path in (analysis_path, low_path, package_path))
    require(sha(audit_source.read_bytes()) == analysis["analysis_source"]["sha256"],
            "explicit audit helper bytes differ from the frozen source snapshot")
    helper = runpy.run_path(str(audit_source))
    data = run_dir / "extracted/context-window-result"
    tx = data / "transaction"
    request = read(run_dir / "request.json")
    source = analysis["source"]
    require(sha((run_dir / "request.json").read_bytes()) == source["request_file_sha256"]
            and sha((run_dir / "report.json").read_bytes()) == source["report_file_sha256"]
            and sha((data / "files.json").read_bytes()) == source["manifest_sha256"],
            "audit no longer binds the external run custody files")
    manifest = read(data / "files.json")
    for name, digest in manifest.items():
        require(sha(helper["file_path"](data, name).read_bytes()) == digest, "raw evidence changed after the audit")
    verification = verify_transaction(tx, expected_seal_sha256=source["transaction"]["seal_sha256"],
                                      expected_event_head=source["transaction"]["event_head"])
    require(low["comparison"]["current_seal"] == verification["seal_sha256"]
            and low["comparison"]["current_event_head"] == verification["event_head"],
            "low-prefix comparison addresses another run")
    historical_root = Path(low["index"]["historical_root"])
    for name, digest in low["index"]["metadata_files"].items():
        require(sha(helper["file_path"](historical_root, name).read_bytes()) == digest,
                "historical prefix index metadata changed")
    require(package["worker_sha256"] == request["source_hashes"]["context_window_campaign.py"]
            and package["all_source_installed_payload_modules_equal"] is True,
            "package qualification proof does not bind the staged worker")
    tree = helper["NumericalTree"](tx)
    forecasts = tree.decode(read(tx / "forecasts/campaign.json")["predictions"])["rows"]
    observations = tree.decode(read(tx / "observations/native.json")["value"])
    require(set(forecasts) == set(observations) == set(analysis["rows"]), "publication row inventory differs")
    records_root = run_dir / "equations-source-qualified"
    records = EquationCatalog(records_root)
    rules = {name: "argmax" if name in {"dest", "src", "width", "length"} else "positive"
             for name in helper["HEADS"]}
    targets = {"regs": [[0, 7, bit] for bit in range(64)] + [[0, 6, 1]]}
    immutable = {value["row_id"]: value for value in analysis["generic_records"]}
    compact_rows, record_ids = {}, {}
    for row_id, observation in sorted(observations.items()):
        context, arm = row_id.split(".", 1)
        prediction = forecasts[row_id]
        parent = context + ".native"
        observed_terms = {name: np.asarray(value, dtype=np.float64)
                          for name, value in analysis["rows"][row_id]["register_accounting"]["all_reader_terms"].items()}
        report = compare_equations(prediction["heads"], observation["heads"], decision_rules=rules,
            baseline=observations[parent]["heads"], predicted_baseline=forecasts[parent]["heads"],
            targets=targets, components={"regs": prediction["register_terms"]},
            observed_components={"regs": observed_terms})
        audited = analysis["rows"][row_id]
        for name, field in report["fields"].items():
            require(field["residual"]["max_abs"] == audited["head_comparison"][name]["residual"]["max_abs"]
                    and field["decisions"]["different"] == audited["head_comparison"][name]["decision_differences"],
                    "public comparison differs from audited saved arrays")
        window_hex = bytes.fromhex(analysis["plan"]["contexts"][context]["program_hex"])[:15].ljust(15, b"\0").hex()
        tested_context = {"model_identity": analysis["plan"]["model_identity"], "execution": analysis["plan"]["execution"],
            "program_hex": analysis["plan"]["contexts"][context]["program_hex"], "window_hex": window_hex,
            "before": observation["before"], "recipe": arm, "context": context,
            "source_identity": request["source_hashes"],
            "horizon": "first instruction only; no successor or architectural commit in this assay",
            "interventions": immutable[row_id]["record"]["interventions"]}
        conditions = [{"field": key, "op": "eq", "value": tested_context[key]}
                      for key in ("model_identity", "execution", "source_identity", "horizon",
                                  "program_hex", "window_hex", "before", "recipe", "interventions")]
        binding = {"model_identity": tested_context["model_identity"], "execution": tested_context["execution"],
            "input_identity": observation["input_identity"],
            "frontier": {"kind": "raw-input", "sealed_forecast": verification["seal_sha256"]},
            "horizon": tested_context["horizon"],
            "source_identity": request["source_hashes"],
            "traced_program_id": immutable[row_id]["record"]["traced_program_id"],
            "actual_trace_fingerprint": observation["trace_fingerprint"]}
        record = records.record(law_id="register-readout/additive-write-response",
            route_id="post.2.return→regs[6,1]", binding=binding, report=report, conditions=conditions,
            tested_context=tested_context, readers=["regs[7,0:64]", "regs[6,1]"],
            evidence={"row_id": row_id, "evidence_role": observation["evidence_role"],
                "audit": ref(analysis_path, "/rows/" + row_id),
                "forecast": ref(tx / "forecasts/campaign.json", "/predictions/items/rows/items/" + row_id),
                "observation": ref(tx / "observations/native.json", "/value/items/" + row_id),
                "raw_manifest": ref(data / "files.json"), "native_baseline_row": parent,
                "immutable_raw_record": immutable[row_id]["record_id"],
                "intervention_scope": "Actual native module returns, independent typed replay equality, frozen predicted FP32 replacements.",
                "authority": "Generic supplied equation/route name; no learned semantic or necessity claim."})
        require(records.show(record["fingerprint"]) == record, "public equation record did not round-trip")
        record_ids[row_id] = record["fingerprint"]
        selected_stages = ("z", "input", "layer0.P2", "layer2.S", "layer2.U", "layer2.M", "layer2.P2", "layer2.h")
        compact_rows[row_id] = {"evidence_role": audited["evidence_role"],
            "exposure": analysis["previous_instrument_attempt"]["repaired_exposure_classification"].get(row_id,
                "reused prior control; no fresh model call"),
            "forecast_decode": audited["forecast_decode"], "actual_decode": audited["actual_decode"],
            "RSI_bit1": audited["RSI_bit1"], "vs_same_context_native": audited["vs_same_context_native"],
            "native_replay_heads": audited["native_replay_heads"], "actual_ports": audited["actual_ports"],
            "all_head_comparisons": {name: {"residual": small_residual(value["residual"]),
                "decision_rule": value["decision_rule"], "decision_differences": value["decision_differences"],
                "different_coordinates": value["different_coordinates"]} for name, value in audited["head_comparison"].items()},
            "stage_count": len(audited["stage_comparison"]),
            "maximum_stage_abs": max(value["max_abs"] for value in audited["stage_comparison"].values()),
            "selected_stage_residuals": {name: small_residual(audited["stage_comparison"][name]) for name in selected_stages},
            "all36_stage_comparisons": ref(analysis_path, "/rows/" + row_id + "/stage_comparison"),
            "RSI_bit1_register_terms": audited["register_accounting"]["RSI_bit1"],
            "all512_register_terms": ref(analysis_path, "/rows/" + row_id + "/register_accounting"),
            "equation_record_id": record["fingerprint"], "equation_record": ref(records_root / (record["fingerprint"] + ".json"))}
    compact_transfers = []
    applicability_checks = {}
    for row_id, identifier in record_ids.items():
        record = records.show(identifier)
        context = record["tested_context"]
        exact = records.find(applicable_to=context)
        require([match["fingerprint"] for match in exact] == [identifier], "source-qualified exact applicability is not unique")
        changed_source = json.loads(json.dumps(context))
        original_worker = changed_source["source_identity"]["context_window_campaign.py"]
        changed_source["source_identity"]["context_window_campaign.py"] = ("0" if original_worker[0] != "0" else "1") + original_worker[1:]
        changed_horizon = dict(context, horizon="two instructions; outside the measured horizon")
        require(not records.find(applicable_to=changed_source), "changed implementation remained applicable")
        require(not records.find(applicable_to=changed_horizon), "changed horizon remained applicable")
        applicability_checks[row_id] = {"exact_context_matches": 1, "changed_worker_source_matches": 0,
                                       "changed_horizon_matches": 0}
    for index, transfer in enumerate(analysis["matched_recipient_P2_transfers"]):
        compact_transfers.append({key: value for key, value in transfer.items() if key not in {"reglogit_component_rows", "all_head_effects"}} | {
            "all_head_effects": {name: small_residual(value) for name, value in transfer["all_head_effects"].items()},
            "all512_components": ref(analysis_path, f"/matched_recipient_P2_transfers/{index}/reglogit_component_rows")})
    prefix_rows = {name: {key: row[key] for key in ("evidence_role", "low_rows", "reader_RSI_bit1", "stages", "conditional_vs_full_raw_payloads")}
                   | {"full_heads_and_terms": ref(low_path, "/comparison/rows/" + name)}
                   for name, row in low["comparison"]["rows"].items()}
    previous = analysis["previous_instrument_attempt"]
    previous_small = {key: previous[key] for key in ("failure", "completed_native_rows", "partial_native_rows",
        "completed_old_native_byte_exact_on_repeat", "all_forecast_and_payload_descriptors_and_addressed_bytes_unchanged",
        "original_seal_sha256", "repaired_seal_sha256", "exposure_scope", "changed_staged_sources")}
    result = {"schema": "mdb-context-window-portable-index-v2", "checkpoint": analysis["plan"]["model_identity"],
        "source": {"job_id": source["job_id"], "request_sha256": source["request_sha256"],
            "archive_sha256": source["archive_sha256"], "transaction": verification,
            "audit": ref(analysis_path), "audit_source": ref(audit_source), "raw_manifest": ref(data / "files.json"),
            "low_prefix_comparison": ref(low_path), "package_proof": ref(package_path),
            "publisher_source_sha256": sha(Path(__file__).read_bytes())},
        "package_qualification": package, "aggregate": analysis["aggregate"],
        "execution": analysis["plan"]["execution"], "counts": analysis["raw_summary"]["counts"],
        "first_exposure_qualification": previous_small,
        "initial_seal_scope": "The initial failed-attempt seal predates all NEW native calls in this two-attempt assay. Three source-bound prior controls were observed before that seal.",
        "window_contrast": analysis["window_contrast"],
        "architectural_before": {"register_words": analysis["plan"]["register_words"], "flags": analysis["plan"]["flags"], "offset": 0},
        "scope": analysis["scope"], "rows": compact_rows,
        "contrasts": {name: [small_effect(value) for value in rows] for name, rows in analysis["contrasts"].items()},
        "matched_recipient_P2_transfers": compact_transfers,
        "structural_prefix_hypothesis": analysis["sealed_structural_prefix_hypothesis"],
        "historical_low_prefix": {"rows": prefix_rows, "limits": low["comparison"]["limits"],
            "historical_root": portable(historical_root), "metadata_files": low["index"]["metadata_files"],
            "status": low["comparison"]["status"], "model_calls": 0, "equation_evaluations": 0},
        "derived_publication_correction": {
            "kind": "source-qualified applicability metadata correction; no scientific outcome changed",
            "previous_portable_index": ref(run_dir / "portable-index-v1.json"),
            "preserved_legacy_catalog": portable(run_dir / "equations"),
            "change": "Source identity was already bound in immutable records; v2 additionally requires exact source_identity and horizon in supplied applicability predicates.",
            "raw_evidence_changed": False},
        "equation_catalog": {"path": portable(records_root), "records": record_ids,
            "applicability_query_checks": applicability_checks,
            "law_id": "register-readout/additive-write-response", "route_id": "post.2.return→regs[6,1]",
            "alias_revision": 2, "aliases": {"context-window/" + name: identifier for name, identifier in record_ids.items()},
            "alias_authority": "Revisable human labels in this versioned index; immutable records have no semantic aliases.",
            "applicability_authority": "Exact supplied predicates and measured rows; no automatically inferred generalization."},
        "mechanics_status": analysis["mechanics_status"], "instrument_status": analysis["instrument_status"],
        "trend_status": analysis["trend_status"], "terminal_status": "unassessed",
        "new_model_calls": 0, "equation_evaluations": 0, "limits": analysis["limits"]}
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(len(raw) < 150 * 1024, "portable index exceeds the declared 150KiB bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return {"output": str(output), "sha256": sha(raw), "bytes": len(raw), "equation_records": len(record_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit-source", type=Path, required=True,
                        help="Explicit frozen maintainer audit source; only its immutable array decoder is used.")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.output, args.audit_source), sort_keys=True))


if __name__ == "__main__":
    main()
