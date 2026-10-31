from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import mdb.computation_atlas as atlas_module
from mdb.computation_atlas import ComputationAtlas, ComputationAtlasError
from mdb.equation_comparison import EquationCatalog, compare_equations


def reference():
    return {"path": "artifacts/saved-audit.json", "sha256": "e" * 64, "json_pointer": "/rows/a"}


def provenance(run="run-a"):
    return {"run_id": run, "artifacts": [reference()]}


def equation(catalog, case="a", *, omit=None, model_field="model_identity", digest_source=False,
             source="c" * 64, backend="cpu", horizon="one instruction"):
    binding = {"model_identity": "a" * 64, "execution": {"backend": backend, "dtype": "float32"},
               "input_identity": "input-" + case, "frontier": "raw-input", "horizon": horizon,
               "source_identity": {"equation.py": source}}
    context = {model_field: binding["model_identity"], "execution": binding["execution"],
               "horizon": horizon, "source_identity": binding["source_identity"], "case": case}
    if digest_source:
        separators = (",", ":") if digest_source == "compact" else None
        context["source_identity"] = hashlib.sha256(json.dumps(binding["source_identity"], sort_keys=True,
                                                               separators=separators).encode()).hexdigest()
    conditions = [{"field": key, "op": "eq", "value": value} for key, value in context.items() if key != omit]
    report = compare_equations({"head": [1., -1.]}, {"head": [1.1, -.8]},
        baseline={"head": [0., 0.]}, decision_rules={"head": "positive"}, targets={"head": [[0]]})
    return catalog.record(law_id="law_001", route_id="route_001", readers=["head[0]"], binding=binding,
                          report=report, conditions=conditions, tested_context=context)


def effect(value, *, units="logit", consumer="head[0]"):
    return {"consumer_id": consumer, "contrast_id": "restore-minus-zero", "value": value,
            "units": units, "provenance": [reference()]}


def imported(atlas, catalog, row, value=None, **kwargs):
    effects = {} if value is None else {row["fingerprint"]: [effect(value)]}
    return atlas.ingest(catalog.root, [row["fingerprint"]], provenance=provenance(), signed_effects=effects, **kwargs)[0]


def test_import_is_idempotent_self_contained_and_never_changes_source_catalog(tmp_path):
    catalog = EquationCatalog(tmp_path / "source")
    original = equation(catalog)
    source_path = catalog.root / (original["fingerprint"] + ".json")
    before = source_path.read_bytes()
    atlas = ComputationAtlas(tmp_path / "atlas")
    first = imported(atlas, catalog, original, -2.)
    assert imported(atlas, catalog, original, -2.) == first
    assert source_path.read_bytes() == before
    assert first["source_record"]["file_sha256"] == hashlib.sha256(before).hexdigest()
    source_path.unlink()
    reopened = ComputationAtlas(atlas.root)
    assert reopened.show(first["fingerprint"]) == first
    changed = reopened.show(first["fingerprint"])
    changed["signed_effects"][0]["value"] = 999
    assert reopened.show(first["fingerprint"])["signed_effects"][0]["value"] == -2.
    assert len(reopened.find(law_id="law_001", route_id="route_001", consumer_id="head[0]")) == 1
    assert reopened.find(consumer_id="other") == []


@pytest.mark.parametrize("omitted", ["model_identity", "source_identity", "execution", "horizon"])
def test_unqualified_source_records_are_rejected_before_any_batch_receipt_is_written(tmp_path, omitted):
    catalog = EquationCatalog(tmp_path / "source")
    good, bad = equation(catalog, "good"), equation(catalog, "bad", omit=omitted)
    atlas = ComputationAtlas(tmp_path / "atlas")
    with pytest.raises(ComputationAtlasError, match="exact"):
        atlas.ingest(catalog.root, [good["fingerprint"], bad["fingerprint"]], provenance=provenance())
    assert not list((atlas.root / "receipts").glob("*.json"))


@pytest.mark.parametrize("digest_source", [False, "compact", "standard"])
def test_historical_predicate_spelling_and_source_digest_are_preserved_and_match_exactly(tmp_path, digest_source):
    catalog = EquationCatalog(tmp_path / "source")
    original = equation(catalog, model_field="checkpoint", digest_source=digest_source)
    atlas = ComputationAtlas(tmp_path / "atlas")
    row = imported(atlas, catalog, original)
    assert row["equation"]["applicability_conditions"] == original["applicability_conditions"]
    assert len(atlas.find(applicable_to=original["tested_context"])) == 1
    assert atlas.find(applicable_to={"model_identity": "a" * 64}) == []
    for field in ("checkpoint", "source_identity", "execution", "horizon"):
        changed = {**original["tested_context"], field: "other"}
        assert atlas.find(applicable_to=changed) == []


def test_comparison_keeps_countertrends_missing_rows_and_heterogeneous_futures(tmp_path):
    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    positive, negative, missing = (equation(catalog, case) for case in ("positive", "negative", "missing"))
    a = imported(atlas, catalog, positive, 2.)
    exceptions = [{"claim_id": "supplied-positive-response", "status": "counterexample-observed", "provenance": [reference()],
                   "detail": "Measured signed response was negative."}]
    future = {"status": "instrument-error", "provenance": [reference()], "completed_steps": 1}
    b = imported(atlas, catalog, negative, -3., counterexamples={negative["fingerprint"]: exceptions},
                 continuation={negative["fingerprint"]: future})
    c = imported(atlas, catalog, missing)
    result = atlas.compare([row["fingerprint"] for row in (a, b, c)], consumer_id="head[0]", contrast_id="restore-minus-zero")
    assert result["hypothesis_status"] == "family-candidate" and result["rule_status"] == "not-validated"
    assert result["selected_rows"] == 3 and result["measured_rows"] == 2
    cohort = next(iter(result["cohorts"].values()))
    assert cohort["signed_values"] == [2., -3.]
    assert cohort["positive"] == cohort["negative"] == 1 and cohort["mean"] == -.5
    assert result["rows"][1]["counterexamples"] == exceptions and result["rows"][1]["continuation"] == future
    assert result["rows"][2]["measurement_status"] == "not-measured"
    assert atlas.show(result["fingerprint"]) == result


def test_source_backend_horizon_cohorts_are_not_pooled_or_automatically_generalized(tmp_path):
    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    rows = [equation(catalog, "a"), equation(catalog, "b", source="d" * 64),
            equation(catalog, "c", backend="other"), equation(catalog, "d", horizon="two instructions")]
    selected = [imported(atlas, catalog, row, index - 1.) for index, row in enumerate(rows)]
    result = atlas.compare([row["fingerprint"] for row in selected], consumer_id="head[0]", contrast_id="restore-minus-zero")
    assert len(result["cohorts"]) == 4
    assert all(cohort["measured"] == 1 for cohort in result["cohorts"].values())
    assert "not performed" in result["cross_cohort_pooling"]
    assert result["rule_status"] == "not-validated"


def test_duplicate_views_of_one_equation_cannot_masquerade_as_two_observations(tmp_path):
    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    row = equation(catalog)
    a = imported(atlas, catalog, row, 1.)
    b = atlas.ingest(catalog.root, [row["fingerprint"]], provenance=provenance("another-label"))[0]
    with pytest.raises(ComputationAtlasError, match="count twice"):
        atlas.compare([a["fingerprint"], b["fingerprint"]], consumer_id="head[0]", contrast_id="restore-minus-zero")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1 << 1024])
def test_nonfinite_or_boolean_signed_effects_are_integrity_errors(tmp_path, value):
    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    row = equation(catalog)
    with pytest.raises(ComputationAtlasError):
        imported(atlas, catalog, row, value)


def test_unknown_consumer_mixed_units_and_unmeasured_effects_are_explicit(tmp_path):
    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    a, b = equation(catalog, "a"), equation(catalog, "b")
    with pytest.raises(ComputationAtlasError, match="consumer"):
        atlas.ingest(catalog.root, [a["fingerprint"]], provenance=provenance(), signed_effects={a["fingerprint"]: [effect(1., consumer="wrong")]})
    imported_a = imported(atlas, catalog, a, 1.)
    imported_b = atlas.ingest(catalog.root, [b["fingerprint"]], provenance=provenance(), signed_effects={b["fingerprint"]: [effect(2., units="different")]})[0]
    with pytest.raises(ComputationAtlasError, match="units"):
        atlas.compare([imported_a["fingerprint"], imported_b["fingerprint"]], consumer_id="head[0]", contrast_id="restore-minus-zero")
    result = atlas.compare([imported_a["fingerprint"]], consumer_id="head[0]", contrast_id="not-measured")
    assert result["hypothesis_status"] == "evidence-incomplete" and result["measured_rows"] == 0


def test_tampered_atlas_and_inner_report_are_detected_even_after_outer_resealing(tmp_path):
    from saturn_pub import Receipt
    from saturn_pub.store import LocalStore

    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    row = imported(atlas, catalog, equation(catalog), 1.)
    changed = copy.deepcopy(row)
    changed.pop("fingerprint")
    changed["equation"]["report"]["max_abs"] += 10
    resealed = Receipt.make(**changed)
    LocalStore(atlas.root).receipt(resealed)
    with pytest.raises(ComputationAtlasError, match="fingerprint"):
        atlas.show(resealed.fingerprint)
    path = atlas.root / "receipts" / (row["fingerprint"] + ".json")
    changed = json.loads(path.read_text())
    changed["signed_effects"][0]["value"] = 99
    path.write_text(json.dumps(changed))
    with pytest.raises(ComputationAtlasError, match="fingerprint"):
        atlas.show(row["fingerprint"])


def test_explicit_audit_pointer_ingester_binds_bytes_and_preserves_partial_futures(tmp_path):
    configured = os.environ.get("MDB_ATLAS_INGEST_SOURCE")
    source = Path(configured) if configured else Path(__file__).resolve().parents[1] / "qualification/ingest_computation_atlas.py"
    assert source.is_file(), "Set MDB_ATLAS_INGEST_SOURCE to the explicit maintainer helper when copying tests."
    helper = runpy.run_path(str(source))
    catalog = EquationCatalog(tmp_path / "source")
    equation_row = equation(catalog)
    audit_path, selection_path = tmp_path / "audit.json", tmp_path / "selection.json"
    audit = {"rows": {"a/b": {"effect": -1.25, "future": {"completed": False, "steps": 1},
                               "exception": {"countertrend": "negative"}}}}
    audit_path.write_text(json.dumps(audit))
    selection = {"run_id": "job-explicit", "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "records": [{"equation_id": equation_row["fingerprint"],
            "effects": [{"consumer_id": "head[0]", "contrast_id": "restore-minus-zero", "units": "logit",
                         "value_pointer": "/rows/a~1b/effect"}],
            "continuation": {"status": "partial", "pointer": "/rows/a~1b/future"},
            "counterexamples": [{"claim_id": "positive", "status": "observed", "pointer": "/rows/a~1b/exception"}]}]}
    selection_path.write_text(json.dumps(selection))
    root = tmp_path / "atlas"
    imported_row = helper["ingest"](root, catalog.root, audit_path, selection_path)[0]
    assert imported_row["signed_effects"][0]["value"] == -1.25
    assert imported_row["continuation"]["value"] == {"completed": False, "steps": 1}
    assert imported_row["counterexamples"][0]["value"] == {"countertrend": "negative"}
    assert imported_row["provenance"]["artifacts"][1]["sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps({"effect": 99}))
    with pytest.raises(ValueError, match="SHA256"):
        helper["ingest"](tmp_path / "not-created", catalog.root, audit_path, selection_path)
    assert not (tmp_path / "not-created").exists()


def test_installed_metadata_import_find_and_compare_need_no_numpy_or_frameworks(tmp_path):
    catalog, atlas = EquationCatalog(tmp_path / "source"), ComputationAtlas(tmp_path / "atlas")
    row = imported(atlas, catalog, equation(catalog), -1.)
    code = """
import builtins, importlib.util, pathlib, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy','torch','jax','tensorflow','isa_lab'}:
        raise RuntimeError('metadata atlas imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
source = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('mdb', source.with_name('__init__.py'), submodule_search_locations=[str(source.parent)])
package = importlib.util.module_from_spec(spec)
sys.modules['mdb'] = package
spec.loader.exec_module(package)
from mdb.computation_atlas import ComputationAtlas, main
atlas = ComputationAtlas(sys.argv[2])
assert len(atlas.find(consumer_id='head[0]')) == 1
assert atlas.compare([sys.argv[3]], consumer_id='head[0]', contrast_id='restore-minus-zero')['hypothesis_status'] == 'observation'
assert main(['--atlas',sys.argv[2],'show',sys.argv[3]]) == 0
assert not set(sys.modules) & {'numpy','torch','jax','tensorflow','isa_lab'}
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(Path(atlas_module.__file__).resolve()),
        str(atlas.root), row["fingerprint"]], text=True, capture_output=True, timeout=20, env=dict(os.environ))
    assert result.returncode == 0, result.stderr
