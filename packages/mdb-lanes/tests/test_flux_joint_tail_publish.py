"""Public-only curation of twelve dependent joint-tail field comparisons."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from mdb_lanes.computation_atlas import ComputationAtlas
from mdb_lanes.equation_comparison import EquationCatalog

P = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/publish_flux_joint_tail.py"))


def fixture():
    binding = {"model_identity": {"checkpoint": "1" * 64},
        "execution": {"backend": "synthetic-cpu"}, "input_identity": "synthetic-input",
        "frontier": {"cut": "2" * 64, "step": 2}, "horizon": "synthetic twojoint recurrence",
        "source_identity": {"fixture.py": "3" * 64}}
    context = {key: binding[key] for key in ("model_identity", "execution", "source_identity")}
    context.update(input_cut="2" * 64, seed=103, prompt="tiny synthetic fixture")
    provenance = {"run_id": "synthetic-no-model", "artifacts": [{"path": "fixture.json", "sha256": "4" * 64}]}
    fields = {}
    for name in P["FIELD_NAMES"]:
        predicted = np.array([[[1., -2.], [3., -4.]]], np.float32)
        observed = predicted.copy()
        if name.endswith("image"):
            observed[0, 0, 0] += .25
        fields[name] = {"predicted": predicted, "observed": observed,
            "provenance": {"fixture": True, "array_sha256": "5" * 64,
                "native_cut": hashlib.sha256(name.encode()).hexdigest()}}
    endpoints = {f"{index}.{contrast}": {"candidate_branch": P["CANDIDATES"][index],
        "reference_branch": "native" if contrast == "native" else "bridge.j"+str(index),
        "latent": {"signed_mean": -.125}, "rgb": {"signed_mean": .5, "mae": 1.25}}
        for index in P["JOINTS"] for contrast in ("native", "matched_bridge")}
    return dict(fields=fields, binding=binding, tested_context=context, provenance=provenance, endpoints=endpoints)


def test_public_catalog_retains_both_references_and_misses_without_decisions(tmp_path):
    output = tmp_path / "publication"
    result = P["curate"](output, **fixture())
    assert result["curated_records"] == result["distinct_named_field_comparisons"] == 12
    assert result["distinct_sealed_output_fields"] == 6
    assert result["native_histories"] == 1 and result["family_candidates_created"] == 0
    catalog, atlas = EquationCatalog(output / "equations"), ComputationAtlas(output / "atlas")
    equations = [catalog.show(item["equation_id"]) for item in result["records"]]
    assert len(catalog.find(law_id="flux.joint.attention-then-mlp")) == 12
    assert {value["evidence"]["contrast"] for value in equations} == {"native", "matched_bridge"}
    assert all(value["report"]["decision_fields"] == 0 for value in equations)
    assert all("dependent" in value["evidence"]["independence"] for value in equations)
    assert sum(bool(atlas.show(identifier)["counterexamples"]) for identifier in result["atlas_observation_ids"]) == 6
    for value in equations:
        field = value["report"]["fields"]["joint"+str(value["evidence"]["joint_index"])+"."+value["evidence"]["contrast"] + "." + value["evidence"]["field"]]
        if value["evidence"]["field"] == "image":
            assert field["residual"]["signed_mean"] == .0625 and field["residual"]["max_abs"] == .25


def test_exact_query_does_not_expand_seed_or_missing_cut(tmp_path):
    result = P["curate"](tmp_path / "publication", **fixture())
    assert result["queries"]["exact_context_matches"] == 12
    assert result["queries"]["changed_seed_matches"] == result["queries"]["missing_input_cut_matches"] == 0
    assert result["queries"]["missing_input_cut_applicability"] == "unknown"
    assert len(result["query_contexts"]) == 12
    assert all(row["exact_context_matches"] == 1 for row in result["queries"]["occurrences"])
    for item in result["query_contexts"]:
        assert {predicate["field"] for predicate in item["conditions"]} >= {
            "block_index","stream","reference_kind","reference_cut","input_cut","source_identity"}


def test_futures_distinguish_matched_bridge_and_native_reference(tmp_path):
    result = P["curate"](tmp_path / "publication", **fixture())
    atlas = ComputationAtlas(tmp_path / "publication/atlas")
    values = [atlas.show(identifier) for identifier in result["atlas_observation_ids"]]
    assert {value["continuation"]["reference_branch"] for value in values} == {"native", "bridge.j2", "bridge.j3", "bridge.j4"}
    assert all([item["value"] for item in value["signed_effects"]] == [-.125, .5] for value in values)
    assert all("no attribution" in value["signed_effects"][0]["scope"] for value in values)
    assert all(value["rule_status"] == "observation; no validated rule" for value in values)
    for value in values:
        future = value["continuation"]
        index = int(future["reference_branch"][-1]) if future["reference_branch"] != "native" else None
        if index is not None:
            assert future["branch"] == P["CANDIDATES"][index]


@pytest.mark.parametrize("index",[2,3,4])
def test_wrong_endpoint_arm_cannot_be_attached_to_local_joint(tmp_path,index):
    args = fixture()
    args["endpoints"][f"{index}.matched_bridge"]["candidate_branch"] = "bridge.j2"
    with pytest.raises(ValueError,match="future mapping differs"):
        P["curate"](tmp_path/"publication",**args)


def test_inventory_freshness_and_shape_are_integrity_requirements(tmp_path):
    args = fixture()
    P["curate"](tmp_path / "publication", **args)
    with pytest.raises(FileExistsError):
        P["curate"](tmp_path / "publication", **args)
    del args["fields"]["joint2.native.image"]
    with pytest.raises(ValueError, match="inventory differs"):
        P["curate"](tmp_path / "incomplete", **args)
    assert not (tmp_path / "incomplete").exists()
    args = fixture()
    args["fields"]["joint2.native.image"]["observed"] = np.ones((1, 1, 4), np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        P["curate"](tmp_path / "shape", **args)


def test_failed_job_refuses_before_output_creation(tmp_path):
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


def test_preoutcome_auditor_must_match_actual_and_frozen_sources(tmp_path):
    path = tmp_path / "independent-math-freeze/analyze_flux_joint_tail.py"
    path.parent.mkdir()
    path.write_text("preoutcome independent equations")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    request = {"independent_math_sources": {"analyze_flux_joint_tail.py": digest}}
    audit = {"auditor_source_sha256": digest, "independent_math_sources": request["independent_math_sources"]}
    result = P["audit_authority"](tmp_path, audit, request, current_auditor=path)
    assert result["status"] == "pre-outcome-auditor" and result["accepted_auditor_sha256"] == digest
    other = tmp_path / "other.py"
    other.write_text("unaccepted alternative equations")
    with pytest.raises(ValueError, match="frozen mathematical source"):
        P["audit_authority"](tmp_path, audit, request, current_auditor=other)


def test_signed_endpoint_diagnostic_has_no_quality_gate():
    actual = np.array([2., 0.], np.float32)
    baseline = np.array([1., 1.], np.float32)
    result = P["difference"](actual, baseline)
    assert result == {"signed_mean": 0., "mae": 1., "rms": 1., "l2": np.sqrt(2.), "max_abs": 1.,
        "changed_value_count": 2, "sample_count": 2}
    assert P["MAX_ELEMENTS"] >= 1024 * 3072


@pytest.mark.parametrize("name", ["../absolute", "/absolute", "a/../b"])
def test_member_paths_are_canonical_and_contained(tmp_path, name):
    with pytest.raises(ValueError, match="evidence path"):
        P["member"](tmp_path, name)
