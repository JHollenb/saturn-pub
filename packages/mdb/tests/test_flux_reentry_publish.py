"""Public equation/atlas curation mechanics on tiny synthetic saved fields."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog

P = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/publish_flux_reentry.py"))


def fixture():
    binding = {"model_identity": {"checkpoint": "1" * 64},
        "execution": {"backend": "synthetic-cpu"}, "input_identity": "synthetic-input",
        "frontier": {"cut": "2" * 64, "step": 2}, "horizon": "synthetic project+joint",
        "source_identity": {"fixture.py": "3" * 64}}
    context = {key: binding[key] for key in ("model_identity", "execution", "source_identity")}
    context.update(input_cut="2" * 64, seed=97, prompt="tiny synthetic fixture")
    provenance = {"run_id": "synthetic-no-model", "artifacts": [{"path": "fixture.json", "sha256": "4" * 64}]}
    fields = {}
    for stage, names in (("project", P["PROJECT_FIELDS"]), ("joint0", ("image", "text"))):
        for name in names:
            predicted = np.array([[[1., -2.], [3., -4.]]], np.float32)
            observed = predicted.copy()
            if name == "image":
                observed[0, 0, 0] += .25
            fields[stage + "." + name] = {"predicted": predicted, "observed": observed,
                "provenance": {"fixture": True, "array_sha256": "5" * 64}}
    endpoints = {branch: {"latent": {"signed_mean": -.125}, "rgb": {"signed_mean": .5, "mae": 1.25}}
        for branch in ("project.predicted", "joint.predicted")}
    return dict(fields=fields, binding=binding, tested_context=context, provenance=provenance, endpoints=endpoints)


def test_public_curation_retains_exact_and_missed_fields_and_dependent_views(tmp_path):
    output = tmp_path / "publication"
    result = P["curate"](output, **fixture())
    assert result["curated_records"] == 10 and result["distinct_named_field_comparisons"] == 8
    assert result["native_histories"] == 1 and result["family_candidates_created"] == 0
    catalog, atlas = EquationCatalog(output / "equations"), ComputationAtlas(output / "atlas")
    equations = [catalog.show(item["equation_id"]) for item in result["records"]]
    assert len(catalog.find(law_id="flux.project.state-and-time")) == 6
    assert len(catalog.find(law_id="flux.joint.attention-then-mlp")) == 2
    assert len(catalog.find(law_id="flux.reentry.own-state-chain")) == 2
    assert all(value["report"]["decision_fields"] == 0 for value in equations)
    assert all("signed_values" not in field["residual"]
        for value in equations for field in value["report"]["fields"].values())
    assert sum(bool(atlas.show(identifier)["counterexamples"]) for identifier in result["atlas_observation_ids"]) == 3
    # Two composed records reuse joint results; they are labelled rather than
    # counted as additional independent numerical tests.
    dependent = [value for value in equations if value["evidence"]["dependent_reuse_of"]]
    assert len(dependent) == 2
    for value in equations:
        report = value["report"]["fields"][value["evidence"]["field"]]
        if value["evidence"]["field"].endswith("image"):
            assert report["residual"]["signed_mean"] == .0625
            assert report["residual"]["max_abs"] == .25


def test_actual_exact_predicates_do_not_expand_to_changed_or_unknown_context(tmp_path):
    result = P["curate"](tmp_path / "publication", **fixture())
    assert result["queries"] == {"exact_context_matches": 10, "changed_seed_matches": 0,
        "missing_input_cut_matches": 0, "missing_input_cut_applicability": "unknown",
        "authority": "Supplied exact predicates only; query matching neither proves generalization nor induces a family."}


def test_native_continuation_effects_are_signed_and_explicitly_multifield(tmp_path):
    result = P["curate"](tmp_path / "publication", **fixture())
    atlas = ComputationAtlas(tmp_path / "publication/atlas")
    record = atlas.show(result["atlas_observation_ids"][0])
    assert [item["value"] for item in record["signed_effects"]] == [-.125, .5]
    assert "not the isolated causal effect" in record["signed_effects"][0]["scope"]
    assert record["continuation"]["status"] == "observed-native-continuation"
    assert record["rule_status"] == "observation; no validated rule"


def test_fresh_output_and_complete_field_inventory_are_required(tmp_path):
    args = fixture()
    P["curate"](tmp_path / "publication", **args)
    with pytest.raises(FileExistsError):
        P["curate"](tmp_path / "publication", **args)
    del args["fields"]["project.temb"]
    with pytest.raises(ValueError, match="inventory differs"):
        P["curate"](tmp_path / "incomplete", **args)
    assert not (tmp_path / "incomplete").exists()


def test_reshape_tampering_is_mechanics_failure_not_numerical_filter(tmp_path):
    args = fixture()
    args["fields"]["project.image"]["observed"] = np.ones((1, 1, 4), np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        P["curate"](tmp_path / "publication", **args)
    assert P["MAX_ELEMENTS"] >= 1024 * 3072


def test_publisher_refuses_failed_or_unaccepted_job_before_creating_output(tmp_path):
    request = {"config": {}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    audit = {"request_sha256": request["request_sha256"], "job_id": "job-synthetic",
        "mechanics_status": "valid", "models_executed": False}
    for name, value in (("request.json", request), ("job.json", {"job_id": "job-synthetic", "state": "failed"}),
        ("collection.json", {"request_sha256": request["request_sha256"], "job_id": "job-synthetic"}), ("audit.json", audit)):
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="accepted audit/job/request differs"):
        P["publish"](tmp_path, tmp_path / "audit.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def correction_fixture(tmp_path):
    frozen = tmp_path / "run/independent-math-freeze/analyze_flux_reentry.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("original reader; immutable numerical functions")
    current = tmp_path / "corrected.py"
    current.write_text("corrected typed mapping reader; immutable numerical functions")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    original, corrected = digest(frozen), digest(current)
    request = {"request_sha256": "6" * 64,
        "independent_math_sources": {"analyze_flux_reentry.py": original, "flux_reentry_audit_math.py": "7" * 64}}
    record = {"schema": "mdb-flux-reentry-audit-correction-v1", "request_sha256": request["request_sha256"],
        "job_id": "job-synthetic", "original_auditor_sha256": original, "corrected_auditor_sha256": corrected,
        "model_calls": 0, "reason": "Preserve map/tuple type tags while comparing typed Act parameter overlays."}
    receipt = tmp_path / "correction.json"
    receipt.write_text(json.dumps(record))
    audit = {"request_sha256": request["request_sha256"], "job_id": "job-synthetic",
        "auditor_source_sha256": corrected, "preoutcome_auditor_sha256": original,
        "independent_math_sources": dict(request["independent_math_sources"]),
        "audit_correction": {"path": str(receipt), "sha256": digest(receipt), "record": record}}
    return frozen.parents[1], current, request, audit


def test_explicit_offline_reader_correction_retains_original_and_corrected_pins(tmp_path):
    run, current, request, audit = correction_fixture(tmp_path)
    authority = P["audit_authority"](run, audit, request, current_auditor=current)
    assert authority["status"] == "explicit-offline-instrument-correction"
    assert authority["preoutcome_auditor_sha256"] == request["independent_math_sources"]["analyze_flux_reentry.py"]
    assert authority["accepted_auditor_sha256"] == audit["auditor_source_sha256"]
    assert authority["correction"] == audit["audit_correction"]
    assert "no model rerun" in authority["scope"]


def test_plain_changed_auditor_without_receipt_still_refuses(tmp_path):
    run, current, request, audit = correction_fixture(tmp_path)
    del audit["audit_correction"]
    with pytest.raises(ValueError, match="explicit offline correction"):
        P["audit_authority"](run, audit, request, current_auditor=current)


@pytest.mark.parametrize("field,bad", [("request_sha256", "0" * 64), ("job_id", "job-other"),
    ("original_auditor_sha256", "0" * 64), ("corrected_auditor_sha256", "0" * 64),
    ("model_calls", 1), ("model_calls", False), ("reason", "")])
def test_correction_receipt_cannot_rebind_job_request_source_or_model_ledger(tmp_path, field, bad):
    run, current, request, audit = correction_fixture(tmp_path)
    correction = audit["audit_correction"]
    correction["record"][field] = bad
    path = Path(correction["path"])
    path.write_text(json.dumps(correction["record"]))
    correction["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="actual request/job/source repair"):
        P["audit_authority"](run, audit, request, current_auditor=current)


def test_correction_receipt_hash_and_embedded_copy_are_both_required(tmp_path):
    run, current, request, audit = correction_fixture(tmp_path)
    receipt = Path(audit["audit_correction"]["path"])
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="file/hash"):
        P["audit_authority"](run, audit, request, current_auditor=current)
    audit["audit_correction"]["sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    audit["audit_correction"]["record"]["reason"] = "unbound alternate reason"
    with pytest.raises(ValueError, match="actual request/job/source repair"):
        P["audit_authority"](run, audit, request, current_auditor=current)


def test_reader_correction_cannot_change_frozen_numerical_source_map(tmp_path):
    run, current, request, audit = correction_fixture(tmp_path)
    audit["independent_math_sources"]["flux_reentry_audit_math.py"] = "0" * 64
    with pytest.raises(ValueError, match="frozen mathematical source"):
        P["audit_authority"](run, audit, request, current_auditor=current)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b"])
def test_custody_members_cannot_escape_result_root(tmp_path, name):
    with pytest.raises(ValueError, match="evidence path"):
        P["member"](tmp_path, name)
