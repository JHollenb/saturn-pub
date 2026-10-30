"""Derive signed observations and a mechanics receipt from immutable native evidence."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    root = args.result.resolve()

    def read(name):
        return json.loads((root / name).read_text())

    report, audit, job, custody = (read(name) for name in (
        "report.json", "offline-audit.json", "job.json", "custody.json"))
    report_sha = sha256((root / "report.json").read_bytes()).hexdigest()
    if audit["source_report_sha256"] != report_sha:
        raise ValueError("offline audit refers to another report")
    request = read("request.json")
    worker_sha = sha256((root / "executed-worker.py").read_bytes()).hexdigest()
    if worker_sha != request["source_hashes"]["worker.py"]:
        raise ValueError("retained worker source differs from the executed request")
    rows = []
    for arm in report["interventions"]:
        probe = arm["probe"]
        baseline, candidate = probe["baseline_margin"], probe["candidate_margin"]
        deltas = {key: candidate[key] - baseline[key] for key in (
            "margin_paris_minus_rome", "margin_berlin_minus_paris", "margin_berlin_minus_rome")}
        rows.append({
            "schema": "symbol-lifecycle-signed-observation-v1", "arm": arm["name"],
            "specimen": report["model"], "execution": report["execution"],
            "source_cut": report["source_context"]["source_parent_cut"],
            "candidate_cut": arm["candidate_cut"], "consumer": report["native_consumer"],
            "source_report_sha256": report_sha,
            "interpretation": "no-op-calibration" if arm["name"] == "parent_copy" else "supplied-row-intervention",
            "metrics": {"margin": candidate["margin_paris_minus_rome"], **deltas},
            "signed_delta": deltas, "baseline": baseline, "candidate": candidate,
            "tokens_equal": probe["tokens_equal"], "restore_exact": arm["restore_exact"],
            "mechanics_status": report["mechanics_status"], "instrument_status": report["instrument_status"],
            "trend_status": "observations-only", "terminal_status": "not-assessed",
        })
    with (root / "observations.jsonl").open("x") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    vm_files = report["vm_descriptor_custody"]["files"]
    for project in ("project", "export"):
        for descriptor in vm_files:
            relative = Path(descriptor["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("VM descriptor path escapes the collected project")
            raw = (root / "extracted" / project / relative).read_bytes()
            if len(raw) != descriptor["bytes"] or sha256(raw).hexdigest() != descriptor["sha256"]:
                raise ValueError("VM descriptor custody mismatch")
    control = next(row for row in rows if row["arm"] == "parent_copy")
    payload = custody["payload_custody"]
    seals = {payload[key]["sha256"] for key in ("declared", "sealed", "executed")}
    gates = {
        "native_job_completed": job["state"] == "succeeded",
        "guarded_payload_identity": len(seals) == 1,
        "native_parent_restoration": all(row["restore_exact"] for row in rows),
        "recorded_noop_readout_exact": control["baseline"] == control["candidate"] and control["tokens_equal"],
        "native_operation_bound": report["native_budget"]["bounded_operations"] <= 256,
        "offline_native_reference_closure": all(
            item["payload_reference_closure"] for item in audit["projects"].values()),
        "same_native_lifecycle_export": report["exported_project"]["exact_lifecycle"],
    }
    artifacts = [{"path": str(root / name), "sha256": sha256((root / name).read_bytes()).hexdigest()}
                 for name in ("report.json", "offline-audit.json", "native-project-archive.tar.gz",
                              "job.json", "custody.json", "request.json", "config.json", "observations.jsonl",
                              "executed-worker.py")]
    receipt = {
        "schema": "symbol-lifecycle-mechanics-receipt-v1", "job_id": job["job_id"],
        "gates": gates, "gate_policy": {
            "stage": "mechanics", "purpose": "verify the declared native evidence instrument and custody",
            "authoritative_consumer": "public Session/Project plus exact offline byte verification",
            "observable": "explicit identity/restore/export/budget checks listed individually",
            "direction": "exact equality or the declared finite operation ceiling",
            "threshold": "exact identity; bounded_operations <= 256",
            "calibration_source": "native sealed contracts and declared finite lease",
            "split": "single supplied development specimen; no held-out certification",
            "aggregation": "individual mechanics checks only",
            "controls": "same-parent copy and exact restoration",
            "action_on_failure": "retain observations; report mechanics defect; make no terminal claim",
        },
        "mechanics_status": "valid" if all(gates.values()) else "invalid",
        "instrument_status": report["instrument_status"], "trend_status": "observations-only",
        "terminal_status": "not-assessed", "vm_descriptor_files_verified_per_project": len(vm_files),
        "producer_diagnostics": {
            "export_summary_reported": report["exported_project"]["lifecycle"],
            "export_summary_verified": {key: audit["projects"]["export"][key]
                                        for key in ("definitions", "uses")},
            "note": "Historical nested export counts used the wrapper instead of lifecycle; sealed export identity and verified counts are authoritative. Current producer corrects these display counts.",
        },
        "artifacts": artifacts,
        "claim_boundary": "Native lifecycle mechanics on one supplied coalition; no semantic CoT, necessary-symbol, portability, or automatic-expiry claim.",
    }
    (root / "mechanics-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"observations": len(rows), "mechanics": gates,
                      "vm_descriptors_per_project": len(vm_files), "terminal_status": "not-assessed"}))


if __name__ == "__main__":
    main()
