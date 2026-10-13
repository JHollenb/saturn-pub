"""Take custody and verify the post-job sweep for one collected paper assay."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

from mdb.artifact_custody import take_custody, verify_custody
from mdb.evidence_pipeline import run_pipeline, verify_pipeline


def immutable_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("refusing to rewrite existing evidence: " + str(path))
    path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args()
    job_dir = args.job_dir.resolve()
    report = json.loads((job_dir / "report.json").read_text())
    request = json.loads((job_dir / "request.json").read_text())
    job = json.loads((job_dir / "job.json").read_text())
    audit = json.loads((job_dir / "state-audit.json").read_text())
    if job["state"] != "succeeded" or audit["mechanics_status"] != "valid":
        raise ValueError("successful execution and completed archive audit required")
    if audit["request_sha256"] != request["request_sha256"] or report["mdb"]["request_sha256"] != request["request_sha256"]:
        raise ValueError("job, archive and report request identities differ")
    if audit["source_audit"]["mechanics_status"] != "valid":
        raise ValueError("source packet audit required")
    if request["experiment"].endswith("e20"):
        native = all(s["native_rows_exact"] for s in report["mdb"]["sessions"])
        replay = all(s["baseline_replay_exact"] for s in report["mdb"]["sessions"])
        rows = report["rows"]
        scope = "E20 observable rows equal same-run native reference; full-logit baseline replay exact"
    else:
        if not request["payload"]["deterministic_algorithms"]:
            raise ValueError("repeat-variability diagnostics cannot certify bit-exact execution")
        native = all(c["native_logits_exact"] and c["intervention_roundtrip_exact"] for c in report["mdb"]["calls"])
        replay = all(r["exact"] for s in report["mdb"]["sessions"] for r in s["replays"])
        rows = report["items"]
        scope = "E6 full-logit equality to same-run native calls; first and last input cuts replay exactly"
        geometry = request.get("work_geometry")
        if geometry and (geometry["mdb_consumer_calls"] != len(report["mdb"]["calls"])
                         or geometry["prompt_panels"] != len(report["mdb"]["sessions"])
                         or geometry["local_cut_replays"] != sum(len(s["replays"]) for s in report["mdb"]["sessions"])):
            raise ValueError("observed local work differs from the fixed panel geometry")
    if not native or not replay:
        raise ValueError("bounded native/replay mechanics did not pass")
    observations = job_dir / "observations.jsonl"
    raw = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()
    if observations.exists() and observations.read_bytes() != raw:
        raise ValueError("existing observation rows differ")
    observations.write_bytes(raw)
    files = [job_dir / name for name in ("report.json", "request.json", "config.json",
             "execution-environment.json", "worker.log", "state-audit.json", "observations.jsonl")]
    files += sorted(job_dir.glob("*-state.tar.gz")) + sorted(job_dir.glob("*-source.tar.gz"))
    if len(files) != 9 or any(not path.is_file() for path in files):
        raise ValueError("complete source/state archives and execution records required")
    receipt = {"schema": "mdb-paper-mechanics-receipt-v1", "job_id": job["job_id"],
        "request_sha256": request["request_sha256"], "model_id": report.get("hf_id", report.get("model_name")),
        "gates": {"source_byte_identity": True, "stored_input_integrity": True,
                  "same_run_native_comparison": native, "selected_input_replay": replay},
        "mechanics_status": "valid", "instrument_status": "qualified-for-declared-panel",
        "trend_status": "signed effects and controls retained in raw observations",
        "terminal_status": "bounded-runtime-mechanics-only", "authority": scope,
        "gate_policy": "mechanics certification after finite execution; no scientific effects filtered",
        "limitations": report["mdb"]["limitations"],
        "evidence": [{"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()} for path in files]}
    receipt_path = job_dir / "mechanics-receipt.json"
    immutable_json(receipt_path, receipt)
    for path in [*files, receipt_path]:
        take_custody(path, job_dir)
    custody = verify_custody(job_dir)
    (job_dir / "custody-verification.json").write_text(json.dumps(custody, indent=2) + "\n")
    if not custody["ok"]:
        raise ValueError("custody verification failed")
    results_root = job_dir.parent
    index = results_root.parents[1] / "index/time-circuits"
    settings = {"watch_root": index / "watch", "registry_path": index / "claims-auto.jsonl",
                "results_root": results_root}
    sweep = run_pipeline(job_dir, xref_db=index / "xref.sqlite", xref_root=job_dir, **settings)
    (job_dir / "evidence-sweep.json").write_text(json.dumps(sweep.to_dict(), indent=2) + "\n")
    verification = verify_pipeline(job_dir, **settings)
    (job_dir / "evidence-verification.json").write_text(json.dumps(verification.to_dict(), indent=2) + "\n")
    print(json.dumps({"job": job["job_id"], "custody_ok": custody["ok"],
        "custody_files": custody["files_ok"], "sweep_exit": sweep.exit_code,
        "verification_exit": verification.exit_code, "instrument_status": verification.instrument_status}))
    return 2 if sweep.exit_code == 2 or verification.exit_code == 2 else verification.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
