"""Bind a launcher-only correction to Astra's unchanged scientific review."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_run, run = ROOT / "run-01", ROOT / "run-02"
    old, new = (read(path / "request.json") for path in (old_run, run))
    prior_path = old_run / "finite-run-review-v1.json"
    prior = read(prior_path)
    assert sha(prior_path) == "03d6f98f1bf3a5f6ef39652883f59a1b2ff03b2009c8da987c83e8202416eded"
    assert prior["review_status"] == "ready-for-finite-run" and prior["blocking_issues"] == []
    assert prior["request_sha256"] == old["request_sha256"]
    assert new["request_sha256"] == "1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a"
    changed = [key for key in new if old.get(key) != new[key]]
    assert sorted(changed) == ["launcher_sha256", "request_sha256", "source_hashes"]
    source_changes = [name for name in new["source_hashes"] if old["source_hashes"].get(name) != new["source_hashes"][name]]
    assert source_changes == ["stage_qwen_sign_shortcut.py"]
    before = read(old_run / "staged-command.json")["command"]
    after = read(run / "staged-command.json")["command"]
    assert after == before + ["--config", "saturn_run_config.json"]
    for name, expected in {**new["source_hashes"], **new["runtime_wheels"]}.items():
        assert sha(run / "payload" / name) == expected, name
    failure = read(old_run / "job-first-observed.json")
    assert failure["job_id"] == "job-30027b477a9e" and failure["state"] == "failed"
    assert failure["result"]["returncode"] == 2 and failure["result"]["peak_vram_mb"] == 0
    qual_path = ROOT / "launcher-config-argument-check-root-v1/receipt.json"
    qualification = read(qual_path)
    assert sha(qual_path) == "1c3583127763f0ad0e2b38cf6348cc0f52934b8efa4ddbb0d823ff019c183eb1"
    assert new["launcher_sha256"] == "6f083cc6a039d50f8f206731d4a0fc87e76f9bf464c0d8cedba4caaecf04887b"
    assert qualification["bootstrap_config_argument_forwarded"] is True
    report = dict(
        schema="mdb-finite-review-launcher-correction-v1",
        reviewer="root, narrow delta review following Astra's complete source/resource review",
        request_sha256=new["request_sha256"], review_status="ready-for-finite-run", blocking_issues=[],
        previous_astra_review_sha256=sha(prior_path), previous_request_sha256=old["request_sha256"],
        qualification_sha256=sha(qual_path), changed_request_fields=changed,
        launcher_regression_checks="Root ran all five exact stager tests after the correction: 5 passed in0.06s; exact files F/I Ruff passed.",
        launcher_test_sha256="c09afe28bce37f7d74cb9054bddee23447bd95e3daa63f0ab85ecae4206a9f1f",
        changed_sources=source_changes, command_delta=["--config", "saturn_run_config.json"],
        scientific_code_config_selection_wheels_resources_unchanged=True,
        exact_source_and_wheel_files_rehashed=len(new["source_hashes"]) + len(new["runtime_wheels"]),
        prior_attempt=dict(job_id=failure["job_id"], result="argument parsing stopped before model load",
                           scientific_evidence="none", job_record_sha256=sha(old_run / "job-first-observed.json")),
        timing=dict(estimated_s=300, assay_timer_s=600, whole_worker_timeout_s=900),
        limitations="This is a launcher admission review, not numerical evidence or a second holistic Astra review.",
    )
    with (run / "finite-run-review-v1.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("request_sha256", "review_status", "changed_sources")}, sort_keys=True))


if __name__ == "__main__":
    main()
