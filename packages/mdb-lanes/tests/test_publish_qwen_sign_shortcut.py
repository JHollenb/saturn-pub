"""Synthetic scalar/envelope plot checks; no original-model execution claims."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

P = runpy.run_path(str(Path(__file__).parents[1] / "qualification/publish_qwen_sign_shortcut.py"))


def fixture(tmp_path, monkeypatch):
    root, native = tmp_path / "pack", tmp_path / "native-metadata"
    root.mkdir()
    native.mkdir()
    native_job, audit_job = "job-synthetic-native", "job-synthetic-audit"
    config = dict(output_scratch="/synthetic", output_name="qwen-sign-shortcut-result")
    for key in ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "retrieval_script"):
        config[key] = key + ".json"
        P["save"](native / config[key], dict(scope="Synthetic metadata-only custody fixture", kind=key))
        config[key + "_sha256"] = P["sha"](native / config[key])
    source_hashes = {"qwen_sign_shortcut_campaign.py": P["WORKER_SHA256"]}
    request = dict(config=config, source_hashes=source_hashes, scope="synthetic request, not an actual model job")
    request["request_sha256"] = P["identity"](request)
    native_request = request["request_sha256"]
    monkeypatch.setitem(P["authenticate"].__globals__, "NATIVE_REQUEST_SHA256", native_request)
    P["save"](native / "request.json", request)
    P["save"](native / "declared-config.json", config)
    plan = dict(config, output="/synthetic/" + native_job + "/qwen-sign-shortcut-result", request_source_hashes=source_hashes,
                request_sha256=native_request, job_id=native_job)
    P["save"](native / "plan.json", plan)
    counts = dict(functional_reference_blocks=60, native_blocks=111, shortcut_down_projections=3, forecast_module_attempts=0)
    P["save"](native / "source.json", dict(job_id=native_job, request_sha256=native_request, source_hashes=source_hashes))
    P["save"](native / "summary.json", dict(job_id=native_job, request_sha256=native_request, counts=counts))
    runtime = dict(flags=dict(scope="Synthetic metadata only; not an observed CUDA profile"))
    P["save"](native / "runtime.json", runtime)
    for name in ("protocol.json", "forecast-responses.json", "comparisons.json"):
        P["save"](native / name, dict(scope="Synthetic scalar fixture"))
    (native / "timeline.jsonl").write_text(json.dumps(dict(scope="synthetic event")) + "\n")
    native_files = {path.name: dict(sha256=P["sha"](path), bytes=path.stat().st_size) for path in native.iterdir()}
    P["save"](root / "source-files.json", dict(schema="mdb-image-tail-files-v1", files=native_files, job_id=native_job,
        request_sha256=native_request, expanded_bytes=sum(row["bytes"] for row in native_files.values())))
    for name in ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime.json", "protocol.json", "forecast-responses.json", "comparisons.json", "timeline.jsonl"):
        (root / ("source-" + name)).write_bytes((native / name).read_bytes())
    (root / "source-metadata").mkdir()
    for key in ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "retrieval_script"):
        (root / "source-metadata" / (key + "-" + config[key])).write_bytes((native / config[key]).read_bytes())
    retriever = Path(__file__).parents[1] / "qualification/retrieve_reader_evidence.py"
    (root / "retrieve_evidence.py").write_bytes(retriever.read_bytes())
    P["save"](root / "external-evidence.json", dict(manifests=dict(native={name: row["sha256"] for name, row in native_files.items()})))

    def metric(error=0., relative=0.):
        return dict(rms=error, relative_rms=relative, signed_mean=-error / 2, maxabs=2 * error,
                    changed_values=int(error != 0), bitwise_exact=error == 0, sample_count=12,
                    first_numerical_difference=None if error == 0 else dict(coordinate=[0, 1, 2], signed_error=-error))

    cases = {}
    for index, case in enumerate(P["CASES"]):
        fields = {name: dict(endpoint=metric(.2 if index == 1 and name == "Hnext" else 0),
            signed_edit_response_error=metric(.3 if index == 1 and name == "Hnext" else 0, None if index == 2 else .1),
            left_response=metric(), right_response=metric()) for name in P["HEADS"]}
        oddness = {lane: {name: dict(rms=.05 if index == 1 and name == "R" else 0, bit_changed_elements=2 if name == "R" else 0,
            zero_signbit_defects=2 if name == "R" else 0, numerical_changed=int(index == 1 and name == "R"))
            for name in ("R", "gate", "up")} for lane in ("functional", "native")}
        cases[case] = dict(source=dict(actual_attended_rows=39 if index < 2 else 512, actual_padded_rows=473 if index < 2 else 0,
            functional_profile=dict(attention_dispatch=dict(backend="sdpa", gqa_policy="repeat_kv" if index < 2 else "enable_gqa"))),
            tail=dict(comparisons={contrast: dict(fields=copy.deepcopy(fields), first_endpoint_bit_discrepancy="Hnext" if index == 1 else None)
                for contrast in P["CONTRASTS"]}), oddness=oddness, first_prefix_bit_discrepancy=None, coverage=dict(primary_prefix_fields=20, primary_endpoint_error_fields=9))
    audit = dict(schema="mdb-qwen-sign-shortcut-audit-v1", audit_sha256=P["AUDITOR_SHA256"], worker_sha256=P["WORKER_SHA256"],
        mechanics_status="authenticated", job_id=native_job, request_sha256=native_request,
        native_manifest_sha256=P["sha"](root / "source-files.json"), source=dict(runtime=runtime), cases=cases, counts=counts,
        limitations=["Synthetic scalar fixture; not original-model outcomes."])
    P["save"](root / "audit.json", audit)
    request = dict(config=dict(native_request_sha256=native_request, source_job=native_job, native_manifest_sha256=audit["native_manifest_sha256"]),
        source_hashes={"audit_qwen_sign_shortcut.py": P["AUDITOR_SHA256"], **P["READER_HELPERS"]})
    request["request_sha256"] = P["identity"](request)
    P["save"](root / "request.json", request)
    P["save"](root / "summary.json", dict(schema="mdb-qwen-sign-shortcut-compact-v1", job_id=audit_job,
        request_sha256=request["request_sha256"], source_job=native_job, model_calls=0, cases=3, catalog_records=3, atlas_observations=3))
    P["save"](root / "coverage.json", dict(model_calls=0, learned_projections=0, tokenizer_calls=0, dependent_accounts=True))
    (root / "catalog/equations").mkdir(parents=True)
    fingerprints = [str(index + 1) * 64 for index in range(3)]
    for fingerprint in fingerprints:
        P["save"](root / "catalog/equations" / (fingerprint + ".json"),
                  dict(fingerprint=fingerprint, evidence=dict(audit_sha256=P["sha"](root / "audit.json"))))
    P["save"](root / "catalog/provenance.json", dict(records=3, automatic_family_induction=False, symbolic_proof=False,
        BF16_regrouping_identity=False, queries=[dict(case=case, fingerprint=fingerprint, exact=1, wrong_input=0)
            for case, fingerprint in zip(P["CASES"], fingerprints, strict=True)]))
    (root / "atlas/receipts").mkdir(parents=True)
    observations = [letter * 64 for letter in ("a", "b", "c")]
    for observation, equation in zip(observations, fingerprints, strict=True):
        path = root / "catalog/equations" / (equation + ".json")
        P["save"](root / "atlas/receipts" / (observation + ".json"), dict(fingerprint=observation,
            source_record=dict(fingerprint=equation, file_sha256=P["sha"](path)), equation=P["read"](path),
            rule_status="observation; no validated rule"))
    P["save"](root / "atlas/provenance.json", dict(family_induction=False, observations=observations))
    manifest = {path.relative_to(root).as_posix(): P["sha"](path) for path in root.rglob("*") if path.is_file()}
    P["save"](root / "files.json", manifest)
    collection = tmp_path / "collection.json"
    P["save"](collection, dict(files_manifest_sha256=P["sha"](root / "files.json"), job_id=audit_job,
        request_sha256=request["request_sha256"], raw_tensor_arrays_collected=False, file_count=len(manifest) + 1,
        expanded_bytes=sum(path.stat().st_size for path in root.rglob("*") if path.is_file())))
    qualification = tmp_path / "qualification.json"
    P["save"](qualification, dict(sources={"audit_qwen_sign_shortcut.py": P["AUDITOR_SHA256"]}, worker_sha256=P["WORKER_SHA256"],
        scope="Synthetic publication envelope; not the actual qualification receipt"))
    monkeypatch.setitem(P["authenticate"].__globals__, "QUALIFICATION_SHA256", P["sha"](qualification))
    return root, audit, dict(expected_manifest=P["sha"](root / "files.json"), expected_request=request["request_sha256"],
                            expected_job=audit_job, collection_receipt=collection, qualification=qualification)


def test_complete_saved_envelope_publish_renders_misses_nulls_and_copies_evidence_once(tmp_path, monkeypatch):
    root, audit, kwargs = fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication"
    result = P["publish"](root, output, **kwargs)
    portable = P["read"](output / "publication.json")
    assert portable["endpoint_and_response_accounts"]["short-procedure"] == audit["cases"]["short-procedure"]["tail"]
    assert portable["oddness"]["short-procedure"]["native"]["R"]["rms"] == .05
    assert portable["source"]["prior_attempt"]["status"].startswith("failed before model")
    assert (output / "compact-evidence/audit.json").read_bytes() == (root / "audit.json").read_bytes()
    assert result["raw_arrays_loaded"] is result["weights_loaded"] is False
    assert set(name for name in result["outputs"] if name.endswith(".png")) == {
        "shortcut-errors.png", "oddness-and-zero-signs.png", "conditional-computation-path.png"}
    assert all(row["within_figure"] for rows in P["read"](output / "figure-layout.json").values() for row in rows)
    with pytest.raises(ValueError, match="fresh publication"):
        P["publish"](root, output, **kwargs)


def test_wrong_external_custody_refused_before_output(tmp_path, monkeypatch):
    root, _audit, kwargs = fixture(tmp_path, monkeypatch)
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="compact manifest"):
        P["publish"](root, output, **{**kwargs, "expected_manifest": "0" * 64})
    with pytest.raises(ValueError, match="collection identity"):
        P["publish"](root, output, **{**kwargs, "expected_job": "job-wrong"})
    assert not output.exists()


def test_full_zero_and_undefined_rows_render_without_invented_log_floor(tmp_path):
    metric = dict(rms=0., relative_rms=None)
    audit = dict(cases={case: dict(source=dict(actual_attended_rows=512, actual_padded_rows=0),
        tail=dict(comparisons={contrast: dict(fields={head: dict(endpoint=metric, signed_edit_response_error=metric)
            for head in P["HEADS"]}) for contrast in P["CONTRASTS"]}),
        oddness={lane: {name: dict(rms=0., bit_changed_elements=0, zero_signbit_defects=0) for name in ("R", "gate", "up")}
            for lane in ("functional", "native")}) for case in P["CASES"]})
    qa = P["figures"](audit, tmp_path)
    assert all(row["within_figure"] for rows in qa.values() for row in rows)
    assert metric == dict(rms=0., relative_rms=None)
