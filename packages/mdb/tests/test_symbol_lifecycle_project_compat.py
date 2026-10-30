import json

import pytest
from test_symbol_lifecycle_project import recorded

from mdb.symbol_trace import trace_symbol


def _stored_joins(project):
    with project._db() as db:
        row = db.execute("SELECT id, joins FROM symbol_lifecycles").fetchone()
    return row[0], json.loads(row[1])


def _replace_stored_joins(project, identifier, joins):
    with project._db() as db:
        db.execute(
            "UPDATE symbol_lifecycles SET joins=? WHERE id=?",
            (json.dumps(joins, sort_keys=True), identifier),
        )


def test_v1_join_without_slice_basis_loads_and_traces_unchanged(tmp_path):
    project, _, owner, lifecycle, _, _ = recorded(tmp_path)
    try:
        saved = project.save_symbol_lifecycle("experiment", lifecycle)
        before = trace_symbol(project, "experiment", "counter.value")
        identifier, joins = _stored_joins(project)
        joins["definitions"][0].pop("slice_basis")
        _replace_stored_joins(project, identifier, joins)

        loaded = project.symbol_lifecycle("experiment", saved["lifecycle_id"])
        after = trace_symbol(project, "experiment", "counter.value")
        assert "slice_basis" not in loaded["joins"]["definitions"][0]
        assert loaded["joins"]["uses"] == saved["joins"]["uses"]
        assert after == before
    finally:
        owner.close()


def test_present_slice_basis_change_still_rejects_historical_join(tmp_path):
    project, _, owner, lifecycle, _, _ = recorded(tmp_path)
    try:
        saved = project.save_symbol_lifecycle("experiment", lifecycle)
        identifier, joins = _stored_joins(project)
        joins["definitions"][0]["slice_basis"] = "changed"
        _replace_stored_joins(project, identifier, joins)

        with pytest.raises(ValueError, match="definition join"):
            project.symbol_lifecycle("experiment", saved["lifecycle_id"])
    finally:
        owner.close()


def test_v1_join_without_receipt_closure_loads_and_traces_unchanged(tmp_path):
    project, _, owner, lifecycle, _, _ = recorded(tmp_path)
    try:
        saved = project.save_symbol_lifecycle("experiment", lifecycle)
        before = trace_symbol(project, "experiment", "counter.value")
        identifier, joins = _stored_joins(project)
        joins["uses"][0].pop("receipt_closure")
        _replace_stored_joins(project, identifier, joins)

        loaded = project.symbol_lifecycle("experiment", saved["lifecycle_id"])
        after = trace_symbol(project, "experiment", "counter.value")
        assert "receipt_closure" not in loaded["joins"]["uses"][0]
        assert after == before
    finally:
        owner.close()


def test_present_receipt_closure_change_still_rejects_historical_join(tmp_path):
    project, _, owner, lifecycle, _, _ = recorded(tmp_path)
    try:
        saved = project.save_symbol_lifecycle("experiment", lifecycle)
        identifier, joins = _stored_joins(project)
        joins["uses"][0]["receipt_closure"] = "changed"
        _replace_stored_joins(project, identifier, joins)

        with pytest.raises(ValueError, match="use join"):
            project.symbol_lifecycle("experiment", saved["lifecycle_id"])
    finally:
        owner.close()


@pytest.mark.parametrize("delta", [-1, 1])
def test_use_join_row_count_change_rejects_truncated_or_extra_rows(tmp_path, delta):
    project, _, owner, lifecycle, _, _ = recorded(tmp_path)
    try:
        saved = project.save_symbol_lifecycle("experiment", lifecycle)
        identifier, joins = _stored_joins(project)
        if delta < 0:
            joins["uses"].pop()
        else:
            joins["uses"].append(dict(joins["uses"][0]))
        _replace_stored_joins(project, identifier, joins)

        with pytest.raises(ValueError, match="use join"):
            project.symbol_lifecycle("experiment", saved["lifecycle_id"])
    finally:
        owner.close()


@pytest.mark.parametrize("delta", [-1, 1])
def test_definition_join_row_count_change_rejects_truncated_or_extra_rows(tmp_path, delta):
    project, _, owner, lifecycle, _, _ = recorded(tmp_path)
    try:
        saved = project.save_symbol_lifecycle("experiment", lifecycle)
        identifier, joins = _stored_joins(project)
        if delta < 0:
            joins["definitions"].pop()
        else:
            joins["definitions"].append(dict(joins["definitions"][0]))
        _replace_stored_joins(project, identifier, joins)

        with pytest.raises(ValueError, match="definition join"):
            project.symbol_lifecycle("experiment", saved["lifecycle_id"])
    finally:
        owner.close()
