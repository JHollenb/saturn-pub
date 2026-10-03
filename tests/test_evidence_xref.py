from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

import saturn_pub.evidence.xref as xref_index
from saturn_pub.evidence.citations import make_citation
from saturn_pub.evidence.xref import (
    MAX_FILE_BYTES,
    XREF_USER_VERSION,
    SchemaVersionError,
    build_index,
    classify_path,
    dedup_refs,
    extract_addresses,
    extract_references,
    index_stats,
    main,
    query_refs,
)

DENSE = "ar://residual/cursor/1/position/0/site/q_proj/layer/0"
KV1 = "ar://kv/layer/003/position/00005/key"
KV2 = "ar://kv/layer/004/position/00006/value"
OLD_CAP = 5 * 1024 * 1024  # a plausible smaller cap, well under MAX_FILE_BYTES


@pytest.fixture()
def results_tree(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    (root / "expA" / "job-1a2b").mkdir(parents=True)
    (root / "expB").mkdir()
    (root / "expA" / "report.json").write_text(
        json.dumps(
            {
                "experiment": "expA",
                "config_sha256": "a" * 64,
                "content_sha256": "b" * 64,
                "claim_boundary": "Mechanics only; " + "x" * 300,
                "gates": {"parity": True, "steps": 8, "loss": 0.5},
                "addresses": [DENSE, KV1],
                "template": "ar://kv/layer/{layer}/position/{position}/key",
            }
        ),
        encoding="utf-8",
    )
    (root / "expA" / "job-1a2b" / "receipt.json").write_text(
        json.dumps({"experiment": "expA", "note": f"cursor read at {DENSE}.", "other": KV2}),
        encoding="utf-8",
    )
    (root / "expB" / "bad.json").write_text('{"broken": "ar://loop/site/head", ', encoding="utf-8")
    (root / "expA" / "big.json").write_text(
        json.dumps({"pad": "x" * (OLD_CAP + 1), "addresses": ["ar://big/site/pad"]}),
        encoding="utf-8",
    )
    return root


def test_extract_addresses_strips_punctuation_and_skips_templates() -> None:
    rows = extract_addresses(f"see {DENSE}. and ar://kv/layer/{{layer}}/key plus {KV1},")
    addresses = [address for address, _ in rows]
    assert addresses == [DENSE, KV1]
    assert all(len(snippet) <= 160 for _, snippet in rows)


def test_build_report_counts(results_tree: Path, tmp_path: Path) -> None:
    db = tmp_path / "xref.sqlite"
    report = build_index(results_tree, db)
    assert report.files_scanned == 4
    assert report.files_skipped_unchanged == 0
    assert report.files_unparseable == 1
    assert report.files_too_large == 0
    assert report.refs_added == 6
    assert report.distinct_addresses == 5


def test_large_files_are_indexed(results_tree: Path, tmp_path: Path) -> None:
    assert OLD_CAP < MAX_FILE_BYTES
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)
    rows = query_refs(db, "ar://big/site/pad")
    assert [row["path"] for row in rows] == ["expA/big.json"]
    assert index_stats(db)["files_skipped_too_large"] == 0


def test_receipt_identity_rows(results_tree: Path, tmp_path: Path) -> None:
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)
    connection = sqlite3.connect(db)
    try:
        rows = dict(connection.execute("SELECT path, claim_boundary FROM receipts").fetchall())
        gates_json = connection.execute(
            "SELECT gates_json FROM receipts WHERE path = ?", ("expA/report.json",)
        ).fetchone()[0]
    finally:
        connection.close()
    assert set(rows) == {"expA/report.json", "expA/job-1a2b/receipt.json"}
    assert len(rows["expA/report.json"]) == 200
    assert json.loads(gates_json) == {"parity": True, "steps": 8, "loss": 0.5}


def test_prefix_query_and_grouping(results_tree: Path, tmp_path: Path, capsys) -> None:
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)

    kv_rows = query_refs(db, "ar://kv")
    assert [row["address"] for row in kv_rows] == [KV1, KV2]
    dense_rows = query_refs(db, DENSE)
    assert {row["path"] for row in dense_rows} == {
        "expA/report.json",
        "expA/job-1a2b/receipt.json",
    }
    assert {row["job_id"] for row in dense_rows} == {None, "job-1a2b"}
    assert all(row["snippet"] for row in dense_rows)

    assert main(["query", "ar://", "--db", str(db)]) == 0
    output = capsys.readouterr().out
    assert "experiment expA" in output
    assert "experiment expB" in output
    assert "[job-1a2b]" in output
    assert "ar://loop/site/head" in output
    assert "model_id breakdown:" in output


def test_incremental_rebuild_skips_unchanged_and_sees_touch(
    results_tree: Path, tmp_path: Path
) -> None:
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)

    second = build_index(results_tree, db)
    assert second.files_scanned == 0
    assert second.files_skipped_unchanged == 4
    assert second.files_too_large == 0
    assert second.refs_added == 0
    assert second.distinct_addresses == 5

    touched = results_tree / "expA" / "job-1a2b" / "receipt.json"
    stat = touched.stat()
    os.utime(touched, (stat.st_atime, stat.st_mtime + 10))
    third = build_index(results_tree, db)
    assert third.files_scanned == 1
    assert third.files_skipped_unchanged == 3
    assert third.refs_added == 2
    assert third.distinct_addresses == 5
    assert len(query_refs(db, DENSE)) == 2


def test_stats_output(results_tree: Path, tmp_path: Path, capsys) -> None:
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)
    stats = index_stats(db)
    assert stats["files_indexed"] == 4
    assert stats["refs"] == 6
    assert stats["distinct_addresses"] == 5
    assert stats["top_addresses"][0] == {"address": DENSE, "refs": 2}
    assert stats["files_skipped_too_large"] == 0

    assert main(["stats", "--db", str(db)]) == 0
    output = capsys.readouterr().out
    assert "distinct addresses: 5" in output
    assert "skipped too large:  0" in output
    assert DENSE in output


def test_build_cli_reports_counts(results_tree: Path, tmp_path: Path, capsys) -> None:
    db = tmp_path / "xref.sqlite"
    assert main(["build", "--root", str(results_tree), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["files_scanned"] == 4
    assert payload["files_too_large"] == 0
    assert "warning" not in captured.err


def test_build_limit_caps_scanned_files(results_tree: Path, tmp_path: Path) -> None:
    db = tmp_path / "xref.sqlite"
    report = build_index(results_tree, db, limit=1)
    assert report.files_scanned == 1


def test_full_build_purges_deleted_files_limited_build_does_not(
    results_tree: Path, tmp_path: Path
) -> None:
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)
    assert index_stats(db)["files_indexed"] == 4
    (results_tree / "expA" / "job-1a2b" / "receipt.json").unlink()

    limited = build_index(results_tree, db, limit=1)
    assert limited.files_scanned == 0
    assert query_refs(db, "ar://kv/layer/004") != []
    assert index_stats(db)["files_indexed"] == 4

    full = build_index(results_tree, db)
    assert full.files_scanned == 0
    assert query_refs(db, "ar://kv/layer/004") == []
    assert len(query_refs(db, DENSE)) == 1
    stats = index_stats(db)
    assert stats["files_indexed"] == 3
    assert stats["receipts"] == 1
    assert stats["distinct_addresses"] == 4


def test_extract_addresses_skips_degenerate_scheme_only() -> None:
    assert extract_addresses("see ar://: and ar:///") == []
    assert [address for address, _ in extract_addresses("ok ar://x.")] == ["ar://x"]


def test_missing_db_raises_and_creates_nothing(tmp_path: Path, capsys) -> None:
    absent = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        query_refs(absent, "ar://")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        index_stats(absent)
    assert not absent.exists()

    assert main(["query", "ar://", "--db", str(absent)]) == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
    assert main(["stats", "--db", str(absent)]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert not absent.exists()


def test_classify_path() -> None:
    assert classify_path("report.json") == "report"
    assert classify_path("run-receipt.json") == "run-receipt"
    assert classify_path("port-observation.json") == "observation"
    assert classify_path("scored-report.json") == "other"
    assert classify_path("variant-run-receipt.json") == "other"
    assert classify_path("summary.json") == "other"


def test_oversized_files_recorded_in_skipped_table(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "results"
    (root / "expS").mkdir(parents=True)
    (root / "expS" / "small.json").write_text(
        json.dumps({"a": "ar://tiny/site/x"}), encoding="utf-8"
    )
    big = root / "expS" / "big.json"
    big.write_text(
        json.dumps({"pad": "y" * 500, "addresses": ["ar://big/site/pad"]}), encoding="utf-8"
    )
    db = tmp_path / "xref.sqlite"

    monkeypatch.setattr(xref_index, "MAX_FILE_BYTES", 64)
    assert main(["build", "--root", str(root), "--db", str(db)]) == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err and "skipped" in captured.err
    assert json.loads(captured.out)["files_too_large"] == 1

    connection = sqlite3.connect(db)
    try:
        rows = connection.execute("SELECT path, reason, size FROM skipped").fetchall()
    finally:
        connection.close()
    assert len(rows) == 1
    path, reason, size = rows[0]
    assert path == "expS/big.json"
    assert "MAX_FILE_BYTES" in reason
    assert size == big.stat().st_size
    assert query_refs(db, "ar://big") == []
    assert index_stats(db)["files_skipped_too_large"] == 1

    monkeypatch.setattr(xref_index, "MAX_FILE_BYTES", 256 * 1024 * 1024)
    report = build_index(root, db)
    assert report.files_too_large == 0
    assert [row["path"] for row in query_refs(db, "ar://big")] == ["expS/big.json"]
    assert index_stats(db)["files_skipped_too_large"] == 0


def test_wrong_version_database_fails_reads_and_migrates_on_build(
    results_tree: Path, tmp_path: Path, capsys
) -> None:
    db = tmp_path / "xref.sqlite"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER,
                           sha256 TEXT, experiment TEXT, job_id TEXT);
        CREATE TABLE refs(address TEXT, path TEXT, experiment TEXT,
                          job_id TEXT, snippet TEXT);
        """
    )
    connection.execute(
        "INSERT INTO refs VALUES ('ar://stale/site/x', 'gone.json', NULL, NULL, 'stale')"
    )
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()

    with pytest.raises(SchemaVersionError, match="run 'build' to migrate"):
        index_stats(db)
    with pytest.raises(SchemaVersionError):
        query_refs(db, "ar://")
    assert main(["stats", "--db", str(db)]) == 2
    assert "user_version" in capsys.readouterr().err

    report = build_index(results_tree, db)
    assert report.files_scanned == 4
    connection = sqlite3.connect(db)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(refs)")}
    finally:
        connection.close()
    assert version == XREF_USER_VERSION
    assert {"path_class", "model_id", "revision", "input_sha256"} <= columns
    assert query_refs(db, "ar://stale") == []
    assert len(query_refs(db, DENSE)) == 2


def test_model_attribution_from_json_walk(tmp_path: Path) -> None:
    shared = "ar://residual/cursor/0/phase/prefill/site/residual/layer/12"
    root = tmp_path / "results"
    (root / "expM" / "job-9f").mkdir(parents=True)
    (root / "expM" / "job-9f" / "report.json").write_text(
        json.dumps(
            {
                "schema": "context-trace-v1",
                "models": [
                    {
                        "model": {
                            "name": "model-a",
                            "model_id": "Example/Model-0.5B",
                            "revision": "060db6499f",
                        },
                        "layer_count": 24,
                        "probes": [{"trace": [{"address": shared}]}],
                    },
                    {
                        "model": {"name": "model-b-0.6b"},
                        "layer_count": 28,
                        "probes": [{"trace": [{"address": shared}]}],
                    },
                ],
                "note": f"pooled mention of {shared} outside any model",
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "xref.sqlite"
    report = build_index(root, db)
    assert report.refs_added == 3
    assert report.refs_with_model_id == 2

    rows = query_refs(db, shared)
    attribution = sorted(
        (row["model_id"], row["revision"]) for row in rows if row["model_id"] is not None
    )
    assert attribution == [("Example/Model-0.5B", "060db6499f"), ("model-b-0.6b", None)]
    assert sum(1 for row in rows if row["model_id"] is None) == 1
    assert all(row["path_class"] == "report" for row in rows)


def test_bare_name_key_is_never_a_model_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "expN").mkdir(parents=True)
    (root / "expN" / "notes.json").write_text(
        json.dumps({"name": "phase-3-sweep", "address": "ar://residual/site/embed"}),
        encoding="utf-8",
    )
    db = tmp_path / "xref.sqlite"
    build_index(root, db)
    rows = query_refs(db, "ar://residual/site/embed")
    assert [row["model_id"] for row in rows] == [None]


def test_extract_references_walk_and_fallback() -> None:
    references, unparseable = extract_references(
        json.dumps({"model_id": "m1", "trace": [{"address": "ar://residual/site/embed"}]})
    )
    assert not unparseable
    assert [(r.address, r.model_id) for r in references] == [("ar://residual/site/embed", "m1")]

    references, unparseable = extract_references('{"broken": "ar://loop/site/head", ')
    assert unparseable
    assert [(r.address, r.model_id) for r in references] == [("ar://loop/site/head", None)]


def test_citation_dicts_populate_columns_directly(tmp_path: Path) -> None:
    cited = KV1
    citation = make_citation(
        cited,
        model_id="Example/Model-0.5B",
        revision="deadbeef",
        input_sha256="c" * 64,
        quantity="rms",
        note="see also ar://residual/site/embed",
    )
    inherited = make_citation("ar://kv/layer/007/position/00001/value", quantity="logit-delta")
    root = tmp_path / "results"
    (root / "expC").mkdir(parents=True)
    (root / "expC" / "report.json").write_text(
        json.dumps(
            {
                "experiment": "expC",
                "model": {"name": "ambient-model"},
                "citations": [citation, inherited],
                "bad": {"saturn_pub_citation": "v1", "address": "ar://broken/{layer}"},
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "xref.sqlite"
    report = build_index(root, db)
    assert report.refs_from_citations == 2

    [row] = query_refs(db, cited)
    assert row["model_id"] == "Example/Model-0.5B"
    assert row["revision"] == "deadbeef"
    assert row["input_sha256"] == "c" * 64
    assert "citation" in row["snippet"]

    [row] = query_refs(db, "ar://kv/layer/007/position/00001/value")
    assert row["model_id"] == "ambient-model"
    assert row["input_sha256"] is None

    [row] = query_refs(db, "ar://residual/site/embed")
    assert row["input_sha256"] is None

    assert query_refs(db, "ar://broken") == []


def test_dedup_collapses_report_runreceipt_mirrors(tmp_path: Path, capsys) -> None:
    mirrored = {"addresses": ["ar://a/site/x", "ar://a/site/y"]}
    root = tmp_path / "results"
    (root / "expD" / "job-7c").mkdir(parents=True)
    (root / "expD" / "job-7c" / "report.json").write_text(json.dumps(mirrored), encoding="utf-8")
    (root / "expD" / "job-7c" / "run-receipt.json").write_text(
        json.dumps({**mirrored, "extra": "ar://a/site/only-in-receipt"}), encoding="utf-8"
    )
    (root / "expD" / "job-7c" / "port-observation.json").write_text(
        json.dumps({"addresses": ["ar://a/site/x"]}), encoding="utf-8"
    )
    db = tmp_path / "xref.sqlite"
    build_index(root, db)

    plain = query_refs(db, "ar://a")
    assert len(plain) == 6
    deduped = query_refs(db, "ar://a", dedup=True)
    assert len(deduped) == 4
    assert {row["path_class"] for row in deduped} == {"report", "observation", "run-receipt"}
    survivors = {(row["address"], row["path_class"]) for row in deduped}
    assert ("ar://a/site/only-in-receipt", "run-receipt") in survivors
    assert ("ar://a/site/x", "run-receipt") not in survivors
    assert dedup_refs(plain) == deduped

    stats = index_stats(db)
    assert stats["refs"] == stats["refs_total"] == 6
    assert stats["refs_deduped"] == 4
    dedup_stats = index_stats(db, dedup=True)
    assert dedup_stats["refs"] == 4

    assert main(["query", "ar://a", "--db", str(db), "--dedup", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4
    assert main(["stats", "--db", str(db), "--dedup", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["refs"] == 4


def test_citation_revision_without_model_id_is_not_overwritten(tmp_path: Path) -> None:
    revision_only = make_citation(
        "ar://kv/layer/3/key", revision="rev-cited-abc", input_sha256="a" * 64
    )
    model_only = make_citation("ar://kv/layer/4/key", model_id="Cited/Model")
    root = tmp_path / "results"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "report.json").write_text(
        json.dumps(
            {
                "model": {"model_id": "Example/StructuralModel", "revision": "rev-structural"},
                "evidence": [revision_only, model_only],
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "xref.sqlite"
    build_index(root, db)

    [row] = query_refs(db, "ar://kv/layer/3/key")
    assert row["model_id"] is None
    assert row["revision"] == "rev-cited-abc"

    [row] = query_refs(db, "ar://kv/layer/4/key")
    assert row["model_id"] == "Cited/Model"
    assert row["revision"] is None


def test_build_on_missing_root_raises_and_preserves_index(
    results_tree: Path, tmp_path: Path, capsys
) -> None:
    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)
    before = index_stats(db)
    assert before["files_indexed"] == 4

    missing = tmp_path / "no_such_root"
    with pytest.raises(FileNotFoundError, match="not a directory"):
        build_index(missing, db)
    assert index_stats(db)["files_indexed"] == before["files_indexed"]
    assert index_stats(db)["refs"] == before["refs"]

    assert main(["build", "--root", str(missing), "--db", str(db)]) == 2
    assert "error:" in capsys.readouterr().err
    assert index_stats(db)["files_indexed"] == before["files_indexed"]


def test_full_build_refuses_purge_when_sweep_sees_nothing(
    results_tree: Path, tmp_path: Path
) -> None:
    from saturn_pub.evidence.xref import EmptySweepError

    db = tmp_path / "xref.sqlite"
    build_index(results_tree, db)
    assert index_stats(db)["files_indexed"] == 4

    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    with pytest.raises(EmptySweepError, match="refusing to purge"):
        build_index(empty_root, db)
    assert index_stats(db)["files_indexed"] == 4

    fresh_db = tmp_path / "fresh.sqlite"
    report = build_index(empty_root, fresh_db)
    assert report.files_scanned == 0

    report = build_index(results_tree, db)
    assert report.files_scanned == 0
    assert report.files_skipped_unchanged == 4
    assert index_stats(db)["files_indexed"] == 4


def test_stats_dedup_counts_model_id_over_the_deduped_pool(tmp_path: Path) -> None:
    address = "ar://kv/layer/001/position/00001/key"
    payload = json.dumps({"model": "Some/Model", "ref": address})
    root = tmp_path / "results"
    (root / "exp" / "job-1a2b").mkdir(parents=True)
    (root / "exp" / "job-1a2b" / "report.json").write_text(payload, encoding="utf-8")
    (root / "exp" / "job-1a2b" / "run-receipt.json").write_text(payload, encoding="utf-8")
    db = tmp_path / "xref.sqlite"
    build_index(root, db)

    plain = index_stats(db)
    assert plain["refs"] == 2
    assert plain["refs_with_model_id"] == 2
    assert plain["refs_with_model_id_total"] == 2

    deduped = index_stats(db, dedup=True)
    assert deduped["refs"] == 1
    assert deduped["refs_with_model_id"] == 1
    assert deduped["refs_with_model_id_total"] == 2
