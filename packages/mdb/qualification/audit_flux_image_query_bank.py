"""Saved-only query/bank causal square, exact operand joins and compact records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("q0-b0", "q0-b1", "q1-b0", "q1-b1")
HELPERS = ("audit_flux_image_tail_microscope.py", "analyze_flux_reader_arithmetic.py",
           "flux_source_receiver.py", "retrieve_reader_evidence.py", "flux_family_custody.py")
PAIRS = {"query_at_bank0": ("q1-b0", "q0-b0"), "query_at_bank1": ("q1-b1", "q0-b1"),
         "bank_at_query0": ("q0-b1", "q0-b0"), "bank_at_query1": ("q1-b1", "q1-b0"),
         "total": ("q1-b1", "q0-b0")}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def difference(fields, name):
    import numpy as np
    if name == "interaction":
        return (np.asarray(fields["q1-b1"], np.float64) - np.asarray(fields["q1-b0"], np.float64)
                - np.asarray(fields["q0-b1"], np.float64) + np.asarray(fields["q0-b0"], np.float64))
    high, low = PAIRS[name]
    return np.asarray(fields[high], np.float64) - np.asarray(fields[low], np.float64)


def response_account(actual, predicted, measure):
    """Keep shared-total normalization, own response errors and signed cancellation."""
    import numpy as np
    total = difference(actual, "total")
    total_energy = float(np.vdot(total, total))
    rms = lambda value: float(np.sqrt(np.mean(value * value)))
    denominator, rows = rms(total), {}
    for name in (*PAIRS, "interaction"):
        a, p = difference(actual, name), difference(predicted, name)
        rows[name] = {"comparison": measure(a, p), "native_rms": rms(a), "predicted_rms": rms(p),
            "native_signed_mean": float(a.mean()), "native_maxabs": float(abs(a).max()),
            "native_relative_to_total_rms": None if denominator == 0 else rms(a) / denominator,
            "error_relative_to_total_rms": None if denominator == 0 else rms(a - p) / denominator,
            "signed_projection_onto_total": None if total_energy == 0 else float(np.vdot(a, total) / total_energy)}
    terms = {name: difference(actual, name) for name in ("query_at_bank0", "bank_at_query0", "interaction")}
    energies = {name: float(np.vdot(value, value)) for name, value in terms.items()}
    keys = tuple(terms)
    cross = {a + "__" + b: float(2 * np.vdot(terms[a], terms[b])) for a, b in
             ((keys[0], keys[1]), (keys[0], keys[2]), (keys[1], keys[2]))}
    closure = total - terms[keys[0]] - terms[keys[1]] - terms[keys[2]]
    return {"native_total_rms": denominator, "denominator": "Y11-Y00 at fixed midpoint recipient",
        "contrasts": rows, "square": {"total_squared_l2": total_energy, "term_squared_l2": energies,
            "signed_cross_terms": cross, "energy_sum": sum(energies.values()) + sum(cross.values()),
            "closure_rms": rms(closure)}, "scope": "Signed finite contrasts; magnitudes and projections are not independent causal shares."}


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np
    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(root), Path(output)
    inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    load = runpy.run_path(str(HERE / "flux_source_receiver.py"))["load_array"]
    measure = runpy.run_path(str(HERE / "analyze_flux_reader_arithmetic.py"))["measure"]
    summary, config, source = (read(root / name) for name in ("summary.json", "plan.json", "source.json"))
    if (any(row["request_sha256"] != expected_request for row in (summary, config, source, inventory))
            or len({summary["job_id"], source["job_id"], inventory["job_id"]}) != 1
            or source["recipient_cut"] != summary["recipient_cut"]
            or any(source[key] != config[key] for key in ("source_job_id", "source_request_sha256", "source_manifest_sha256", "source_forecast_seal_sha256", "source_input_seal_sha256"))):
        raise ValueError("query/bank source or native lineage differs")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    if [row["kind"] for row in timeline] != ["source_bound", "cohort_sealed", *["target_completed"] * 4]:
        raise ValueError("query/bank seal chronology differs")
    if any(row["target_entries"] != 0 for row in timeline[:2]):
        raise ValueError("target exposed before forecast seal")
    expected_forecast = dict(native_attention=4, linear=12, layernorm=4, swiglu=4)
    expected_target = dict(native_blocks=4, native_attention=4, linear=48, layernorm=16, rmsnorm=16, swiglu=8)
    if (summary["forecast_primitive_counts"] != expected_forecast or summary["target_counts"] != expected_target
            or summary["complete_native_targets"] != 4 or summary["new_prefix_entries"] != 0
            or summary["denied_target_attempts"] != 0):
        raise ValueError("native/forecast census differs")
    if timeline[1]["seal_sha256"] != summary["forecast_seal_sha256"] or timeline[1]["forecast_counts"] != expected_forecast:
        raise ValueError("timeline forecast seal or census differs")
    for index, (case, row) in enumerate(zip(CASES, timeline[2:]), 1):
        if row["branch"] != case or row["target_entries"] != index:
            raise ValueError("native target order or entries differ")
        unit = dict(native_blocks=1, native_attention=1, linear=12, layernorm=4, rmsnorm=4, swiglu=2)
        if row["target_counts"] != {key: value * index for key, value in unit.items()}:
            raise ValueError("native target timeline census differs")
    transaction = root / "transaction"
    verified = verify_transaction(transaction, expected_seal_sha256=summary["forecast_seal_sha256"], max_bytes=config["transaction_max_bytes"])
    encoded = read(transaction / "forecasts/image_query_bank.json")
    predictions = load_forecast_tree(transaction, encoded["predictions"])
    encoded_observation = read(transaction / "observations/native_targets.json")
    observed = plain(load_forecast_tree(transaction, encoded_observation["value"]))
    if (predictions["forecast_counts"] != expected_forecast or observed["forecast_counts"] != expected_forecast
            or observed["target_counts"] != expected_target or observed["recipient_cut"] != source["recipient_cut"]):
        raise ValueError("closed transaction census or recipient differs")
    encoded = read(transaction / "frontiers/saved_qkv.json")
    frontier = load_forecast_tree(transaction, encoded["value"])
    if predictions["target_entries_before_seal"] != 0 or set(predictions["cases"]) != set(CASES):
        raise ValueError("forecast cases or target exposure differs")
    if frontier["recipient"]["cut"] != source["recipient_cut"]:
        raise ValueError("sealed recipient differs")
    targets = {case: read(root / "targets" / (case + ".json")) for case in CASES}
    errors, ports, operands, receiving_replay = {}, {}, {}, {}

    def compare(a, p):
        report = measure(a, p)
        bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) != np.ascontiguousarray(p, dtype=np.float32).view(np.uint32)
        return {**report, "bitwise_fp32_widening_exact": not bool(np.any(bits)), "changed_widened_value_bits": int(np.count_nonzero(bits))}

    for case, row in targets.items():
        if row != observed["targets"][case]:
            raise ValueError("target sidecar differs from closed observation")
        q, b = int(case[1]), int(case[4])
        predicted = predictions["cases"][case]
        if (row["recipient_cut"] != source["recipient_cut"] or predicted["recipient_cut"] != source["recipient_cut"]
                or row["sourceQindex"] != q or row["sourceBankindex"] != b
                or predicted["sourceQindex"] != q or predicted["sourceBankindex"] != b
                or row["operation"]["original_sdpa_calls"] != 1 or row["operation"]["return_replaced"]):
            raise ValueError("declared recipient/operand substitution differs")
        operands[case], receiving_replay[case] = {}, {}
        for name in ("q", "k", "v"):
            chosen = str(q if name == "q" else b)
            actual = load(root, row["consumed_qkv"][name])
            expected = frontier[chosen]["values"][name]
            operands[case][name] = compare(actual, expected)
            if not operands[case][name]["bitwise_fp32_widening_exact"]:
                raise ValueError("native consumer did not receive sealed QKV")
            if (any(row["consumed_descriptors"][name][key] != predicted["consumed_qkv_descriptors"][name][key] for key in ("sha256", "dtype"))
                    or any(tuple(row["consumed_descriptors"][name][key]) != tuple(predicted["consumed_qkv_descriptors"][name][key]) for key in ("shape", "stride"))):
                raise ValueError("consumed layout/descriptor differs from forecast")
            receiving_replay[case][name] = compare(load(root, row["receiving_qkv"][name]), frontier["1"]["values"][name])
        errors[case] = compare(load(root, row["actual"]["image"]), predicted["image"])
        ports[case] = {name: compare(load(root, row["actual"][name]), load(root, value)) for name, value in predicted["primitive_ports"].items()}
    shared_ports = set.intersection(*(set(predictions["cases"][case]["primitive_ports"]) for case in CASES))
    responses = {}
    for port in sorted(shared_ports):
        actual = {case: load(root, targets[case]["actual"][port]) for case in CASES}
        predicted = {case: load(root, predictions["cases"][case]["primitive_ports"][port]) for case in CASES}
        responses[port] = response_account(actual, predicted, measure)
    text_checks = {"bank" + str(b): compare(load(root, targets[f"q0-b{b}"]["actual"]["text"]),
                                           load(root, targets[f"q1-b{b}"]["actual"]["text"])) for b in (0, 1)}
    result = {"schema": "mdb-image-query-bank-audit-v1", "job": summary["job_id"], "native_request_sha256": expected_request,
        "native_manifest_sha256": expected_manifest, "forecast_seal_sha256": summary["forecast_seal_sha256"],
        "source_job_id": source["source_job_id"], "source_manifest_sha256": source["source_manifest_sha256"],
        "recipient_cut": source["recipient_cut"], "errors": errors, "port_errors": ports, "responses": responses,
        "consumed_operand_fidelity": operands, "receiving_qkv_replay": receiving_replay,
        "query_only_text_exits": text_checks, "source_qkv_changes": read(root / "source-qkv-changes.json"),
        "retrospective_controls": {case: row["retrospective_control"] for case, row in targets.items() if "retrospective_control" in row},
        "forecast_counts": summary["forecast_primitive_counts"], "target_counts": summary["target_counts"],
        "transaction_verification": verified, "native_contract": read(root / "native-contract.json"),
        "selected_aten_operations": {case: {"forecast": [dict(item) for item in predictions["cases"][case]["selected_aten_operations"]],
            "target": row["selected_aten_operations"]} for case, row in targets.items()},
        "predeclared_expectations": config.get("predeclared_expectations", {}), "auditor_sha256": sha(__file__),
        "metric_orientation": "native minus predicted; finite contrasts follow named case ordering",
        "new_model_calls": 0, "new_reference_forward_calls": 0,
        "mechanics_status": "authenticated-full-array-audit", "terminal_status": "unassessed",
        "limitations": ["One fixed synthetic midpoint recipient and two known positioned-QKV source banks.",
            "Conditional native primitives, not independent Flash/GEMM implementation.",
            "No raw-prompt, suffix, VAE or semantic claim; effects are not causal shares."]}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    catalog, queries = EquationCatalog(output / "catalog/equations"), []

    def record(case, predicted, actual, kind):
        context = {"model_identity": config["expected_model_identity"], "recipient_cut": source["recipient_cut"],
            "case": case, "kind": kind, "clock": {"step": 1, "phase": "joint", "index": 0},
            "source_manifest_sha256": source["source_manifest_sha256"],
            "forecast_seal_sha256": summary["forecast_seal_sha256"], "numerical": result["native_contract"], "horizon": 1}
        item = catalog.record(law_id="law.flux.image-query-bank.fixed-recipient.v1",
            route_id="route.flux.joint0.image-attention-residual-mlp.v1",
            alias="cmp.flux.image-query-bank." + case + "." + kind,
            binding={"model_identity": config["expected_model_identity"], "input_identity": source["recipient_cut"],
                "source_identity": expected_request, "frontier": context["clock"], "execution": result["native_contract"], "horizon": 1},
            report=compare_equations({"image": predicted}, {"image": actual}, max_elements=25_000_000),
            conditions=[{"field": name, "op": "eq", "value": value} for name, value in context.items()], tested_context=context,
            readers=["joint0.image-reader"], evidence={"audit_sha256": sha(output / "audit.json"),
                "role": "sealed conditional query/bank intervention", "independent_kernel_arithmetic": False,
                "new_target_combinations": ["q0-b1", "q1-b0"], "quality_gate": False})
        wrong = {**context, "clock": {"step": 2, "phase": "joint", "index": 0}}
        exact, changed = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to=wrong))
        if exact != 1 or changed != 0:
            raise ValueError("catalog query mechanics differ")
        queries.append({"fingerprint": item["fingerprint"], "case": case, "kind": kind, "exact": exact, "wrong_clock": changed})
        if case == "q1-b0":
            save(output / "catalog/context-exact.json", context)
            save(output / "catalog/context-wrong-clock.json", wrong)

    actual = {case: load(root, targets[case]["actual"]["image"]) for case in CASES}
    predicted = {case: predictions["cases"][case]["image"] for case in CASES}
    for case in CASES:
        record(case, predicted[case], actual[case], "endpoint")
    for name in (*PAIRS, "interaction"):
        record(name, difference(predicted, name), difference(actual, name), "effect")
    save(output / "catalog/provenance.json", {"records": len(queries), "queries": queries,
        "source_native_manifest_sha256": expected_manifest, "family_induction": False,
        "applicability": "supplied exact predicates", "new_model_calls": 0})
    return result, len(queries), inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    if request["config"] != config or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("saved-only CPU audit binding differs")
    for name, digest in request["source_hashes"].items():
        if sha(name) != digest:
            raise ValueError("audit source differs")
    start, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, count, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("source.json", "summary.json", "files.json", "source-joins.json", "source-qkv-changes.json", "contrasts.json",
                 "native-contract.json", "timeline.jsonl", "transaction-report.json", "transaction-verification.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for name in ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", {"host": "beast", "roots": {"native": str(root)},
        "manifests": {"native": {name: row["sha256"] for name, row in inventory["files"].items()}},
        "scope": "Original full native/forecast arrays retained near data; individual checksum retrieval."})
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", {"schema": "mdb-image-query-bank-compact-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "source_job": result["job"], "records": count,
        "output_root": str(output), "seconds": time.monotonic() - start, "model_calls": 0,
        "reference_forward_calls": 0, "mechanics_status": "verified-saved-array-custody", "terminal_status": "unassessed"})
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps({"job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "output_root": str(output), "files_manifest_sha256": sha(output / "files.json"), "summary_sha256": sha(output / "summary.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        launcher = runpy.run_path(str(HERE / "stage_saved_flux_audit.py"))
        result = (launcher["submit"](args.run, worker=Path(__file__)) if args.submit else
            launcher["stage"](args.run, args.study, worker=Path(__file__), helpers=HELPERS,
                experiment="mdb-image-query-bank-audit", operations=10,
                scope="Four sealed query/bank forecasts, fixed recipient, exact consumed-QKV joins, signed six contrasts and ten equation occurrences; zero model/reference recurrence."))
        print(json.dumps(result))
