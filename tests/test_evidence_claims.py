from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from saturn_pub.evidence.claims import (
    ClaimRow,
    ClaimsRegistry,
    ClaimsRegistryError,
    DependencyFingerprint,
    ReplayRecipe,
    fingerprint_paths,
    gates_match,
    harvest_declared_refs,
    ingest_receipt,
    main,
    verify,
)

FIXTURE_GATES = {"replication": True, "route_score": 0.912, "panels": 3}


def _write_receipt(path: Path, gates: dict | None = None) -> dict:
    payload = {
        "experiment": "fixture-experiment",
        "config_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "claim_boundary": "fixture receipt exercising registry mechanics only",
        "gates": dict(gates if gates is not None else FIXTURE_GATES),
        "environment": {"backend": "cpu"},
        "models": ["fixture-model"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


@pytest.fixture()
def registry(tmp_path: Path) -> ClaimsRegistry:
    return ClaimsRegistry(tmp_path / "claims-registry.jsonl")


def test_ingest_captures_receipt_identity_gates_and_recipe(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    payload = _write_receipt(receipt)

    row = ingest_receipt(
        registry,
        receipt,
        "fixture-claim",
        "route score replicates on the fixture panel",
        worker="fixture-worker",
        submit_command="saturn-pub evidence claims ingest fixture",
    )

    assert row.status == "measured"
    assert row.gates == payload["gates"]
    assert row.receipt_content_sha256 == "b" * 64
    assert row.recipe is not None
    assert row.recipe.worker == "fixture-worker"
    assert row.recipe.config_sha256 == "a" * 64
    assert row.recipe.expected_gates == payload["gates"]
    receipt_entry = row.fingerprint.entries[0]
    assert receipt_entry["kind"] == "file"
    assert receipt_entry["name"] == "receipt"
    assert len(receipt_entry["sha256"]) == 64
    assert row.prev_sha256 is None
    assert len(row.sha256) == 64
    assert registry.verify_chain() is True
    assert registry.current()["fixture-claim"].sha256 == row.sha256


def test_ingest_rejects_bad_slug_and_status(tmp_path: Path, registry: ClaimsRegistry) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)

    with pytest.raises(ClaimsRegistryError):
        ingest_receipt(registry, receipt, "Bad_Slug", "text")
    with pytest.raises(ClaimsRegistryError):
        ingest_receipt(registry, receipt, "ok-slug", "text", status="believed")


def test_fingerprint_paths_records_missing_file_as_none(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("bytes", encoding="utf-8")

    fingerprint = fingerprint_paths([present, tmp_path / "absent.txt"])

    shas = {entry["name"]: entry["sha256"] for entry in fingerprint.entries}
    assert len(shas["present.txt"]) == 64
    assert shas["absent.txt"] is None
    assert any("missing" in reason for reason in fingerprint.drift())


def test_chain_links_rows_and_tamper_breaks_verification(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    for claim_id in ("claim-a", "claim-b", "claim-c"):
        ingest_receipt(registry, receipt, claim_id, f"text for {claim_id}")
    assert registry.verify_chain() is True

    lines = registry.path.read_text(encoding="utf-8").splitlines()
    middle = json.loads(lines[1])
    assert middle["prev_sha256"] == json.loads(lines[0])["sha256"]
    middle["text"] = "tampered mid-file"
    lines[1] = json.dumps(middle, sort_keys=True, separators=(",", ":"))
    registry.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert registry.verify_chain() is False


def test_dependency_drift_marks_stale_exactly_once(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    dependency = tmp_path / "dep.txt"
    dependency.write_text("original", encoding="utf-8")
    ingest_receipt(
        registry, receipt, "drift-claim", "depends on dep.txt", dependency_paths=[dependency]
    )

    clean = verify(registry)
    assert (clean.checked, clean.still_valid, clean.newly_stale) == (1, 1, 0)
    assert clean.chain_ok is True

    dependency.write_text("modified", encoding="utf-8")
    drifted = verify(registry)
    assert drifted.newly_stale == 1
    assert drifted.newly_stale_claims == ["drift-claim"]
    current = registry.current()["drift-claim"]
    assert current.status == "stale"
    assert "changed" in current.stale_reason
    assert "dep.txt" in current.stale_reason

    again = verify(registry)
    assert again.newly_stale == 0
    assert again.checked == 1
    assert len(registry.history("drift-claim")) == 2
    assert registry.verify_chain() is True


def test_missing_dependency_marks_stale_with_missing(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    dependency = tmp_path / "dep.txt"
    dependency.write_text("original", encoding="utf-8")
    ingest_receipt(
        registry, receipt, "missing-claim", "depends on dep.txt", dependency_paths=[dependency]
    )

    dependency.unlink()
    report = verify(registry)
    assert report.newly_stale == 1
    current = registry.current()["missing-claim"]
    assert current.status == "stale"
    assert "missing" in current.stale_reason

    assert verify(registry).newly_stale == 0


def test_gates_match_exact_isclose_and_mismatch(tmp_path: Path, registry: ClaimsRegistry) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    row = ingest_receipt(registry, receipt, "gate-claim", "gate text", worker="fixture-worker")

    fresh = {"gates": {"replication": True, "route_score": 0.912 + 1e-13, "panels": 3}}
    matched, mismatches = gates_match(row, fresh)
    assert matched is True
    assert mismatches == []

    bad = {"gates": {"replication": False, "route_score": 0.5}}
    matched, mismatches = gates_match(row, bad)
    assert matched is False
    reasons = {item["gate"]: item["reason"] for item in mismatches}
    assert reasons == {"replication": "value", "route_score": "value", "panels": "missing"}


def test_gates_match_falls_back_to_snapshot_without_recipe(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    row = ingest_receipt(registry, receipt, "snapshot-claim", "no recipe")

    assert row.recipe is None
    matched, mismatches = gates_match(row, {"gates": dict(FIXTURE_GATES)})
    assert matched is True
    assert mismatches == []


def test_history_preserves_ledger_order_and_chain(tmp_path: Path, registry: ClaimsRegistry) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    dependency = tmp_path / "dep.txt"
    dependency.write_text("original", encoding="utf-8")
    ingest_receipt(
        registry, receipt, "history-claim", "history text", dependency_paths=[dependency]
    )
    dependency.write_text("modified", encoding="utf-8")
    verify(registry)

    rows = registry.history("history-claim")
    assert [row.status for row in rows] == ["measured", "stale"]
    assert rows[1].prev_sha256 == rows[0].sha256
    assert rows[1].created_at == rows[0].created_at
    assert registry.history("never-ingested") == []


def test_row_roundtrips_through_json_dict() -> None:
    row = ClaimRow(
        claim_id="round-trip",
        text="roundtrip text",
        status="asserted",
        receipt_path="report.json",
        receipt_content_sha256="c" * 64,
        gates={"ok": True, "score": 0.5},
        recipe=ReplayRecipe(worker="w", expected_gates={"ok": True}),
        fingerprint=DependencyFingerprint(
            entries=[{"kind": "literal", "name": "pin", "sha256": "d" * 64}]
        ),
        created_at="2026-08-17T00:00:00+00:00",
        updated_at="2026-08-17T00:00:00+00:00",
    )

    rebuilt = ClaimRow.from_dict(row.to_dict())
    assert rebuilt.to_dict() == row.to_dict()


def test_cli_ingest_list_verify_history_smoke(tmp_path: Path, capsys) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    ledger = tmp_path / "claims-registry.jsonl"

    assert (
        main(
            [
                "ingest",
                str(receipt),
                "--claim-id",
                "cli-claim",
                "--text",
                "cli text",
                "--registry",
                str(ledger),
            ]
        )
        == 0
    )
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["claim_id"] == "cli-claim"
    assert len(ingested["sha256"]) == 64

    assert main(["list", "--registry", str(ledger), "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["cli-claim"]["status"] == "measured"

    assert main(["list", "--registry", str(ledger), "--status", "stale", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {}

    assert main(["verify", "--registry", str(ledger), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["chain_ok"] is True
    assert (report["checked"], report["still_valid"], report["newly_stale"]) == (1, 1, 0)

    assert main(["history", "cli-claim", "--registry", str(ledger), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["status"] == "measured"


def test_gates_match_huge_ints_short_circuit_before_float() -> None:
    row = ClaimRow(
        claim_id="huge-gate",
        text="huge int gate",
        status="measured",
        receipt_path="report.json",
        gates={"count": 10**400},
    )

    matched, mismatches = gates_match(row, {"gates": {"count": 10**400}})
    assert matched is True
    assert mismatches == []

    matched, mismatches = gates_match(row, {"gates": {"count": 10**400 + 1}})
    assert matched is False
    assert mismatches == [
        {"gate": "count", "expected": 10**400, "observed": 10**400 + 1, "reason": "value"}
    ]


def test_ingest_rejects_non_finite_gate_value(tmp_path: Path, registry: ClaimsRegistry) -> None:
    receipt = tmp_path / "report.json"
    receipt.write_text(
        '{"gates": {"score": NaN}, "content_sha256": "' + "b" * 64 + '"}',
        encoding="utf-8",
    )

    with pytest.raises(ClaimsRegistryError, match="non-finite"):
        ingest_receipt(registry, receipt, "nan-claim", "nan gate text")
    assert registry.rows() == []


def test_head_pointer_detects_tail_truncation_and_legacy_verifies(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    for claim_id in ("head-a", "head-b", "head-c"):
        ingest_receipt(registry, receipt, claim_id, f"text for {claim_id}")
    assert registry.head_path.is_file()
    head = json.loads(registry.head_path.read_text(encoding="utf-8"))
    assert head["rows"] == 3
    assert head["tail_sha256"] == registry.rows()[-1].sha256
    assert registry.verify_chain() is True

    lines = registry.path.read_text(encoding="utf-8").splitlines()
    registry.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert registry.verify_chain() is False

    registry.head_path.unlink()
    assert registry.verify_chain() is True


def test_concurrent_appends_preserve_chain(tmp_path: Path, registry: ClaimsRegistry) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    errors: list[Exception] = []

    def append_rows(tag: str) -> None:
        try:
            for index in range(20):
                ingest_receipt(registry, receipt, f"race-{tag}-{index}", "race text")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=append_rows, args=(tag,)) for tag in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(registry.rows()) == 40
    assert registry.verify_chain() is True


def test_claim_id_with_trailing_newline_rejected(tmp_path: Path, registry: ClaimsRegistry) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)

    with pytest.raises(ClaimsRegistryError):
        ingest_receipt(registry, receipt, "valid-slug\n", "text")
    assert registry.rows() == []


def test_verify_counts_restored_claims_without_unstaling(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "report.json"
    _write_receipt(receipt)
    dependency = tmp_path / "dep.txt"
    dependency.write_text("original", encoding="utf-8")
    ingest_receipt(
        registry, receipt, "restore-claim", "depends on dep.txt", dependency_paths=[dependency]
    )

    dependency.write_text("modified", encoding="utf-8")
    stale_report = verify(registry)
    assert stale_report.newly_stale == 1
    assert stale_report.restored == 0

    dependency.write_text("original", encoding="utf-8")
    restored_report = verify(registry)
    assert restored_report.restored == 1
    assert restored_report.newly_stale == 0
    assert restored_report.still_valid == 0
    assert restored_report.checked == 1
    assert restored_report.to_dict()["restored"] == 1
    assert registry.current()["restore-claim"].status == "stale"
    assert len(registry.history("restore-claim")) == 2


# ---------------------------------------------------------------------------
# declared-refs harvest + read-only verify --check
# ---------------------------------------------------------------------------


def _canonical_sha256(payload) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _raw_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_declared_receipt(path: Path, refs: list[dict]) -> None:
    payload = {
        "experiment": "declared-fixture",
        "gates": {"ok": True},
        "request": {"source_closure": {"files": refs}},
        # mirrored copy, as real receipts duplicate the closure verbatim
        "mirror_receipt": {"request": {"source_closure": {"files": list(refs)}}},
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def test_harvest_declared_semantics_resolution_and_dedup(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    dep = tmp_path / "dep.txt"
    dep.write_text("dependable bytes", encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal_payload = {"b": [1, 2], "a": {"x": 1.5}}
    seal.write_text(json.dumps(seal_payload, indent=2), encoding="utf-8")

    refs = [
        {"path": "dep.txt", "sha256": _raw_sha256(dep)},
        {"file": str(seal), "content_sha256": _canonical_sha256(seal_payload)},
        {"path": "gone/forever.bin", "sha256": "c" * 64},
        {"path": "dep.txt", "other_sha256": "d" * 64},
        {"path": "dep.txt", "sha256": "not-a-sha"},
        {"path": "", "sha256": "e" * 64},
    ]
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(receipt, refs)

    row = ingest_receipt(
        registry, receipt, "declared-claim", "declared refs", harvest_declared=True
    )

    declared = [e for e in row.fingerprint.entries if e["kind"] == "declared"]
    by_key = {(e["path"], e["semantics"]): e for e in declared}
    # mirrored closure deduplicates: 4 entries, not 8
    assert len(declared) == 4
    raw = by_key[("dep.txt", "raw-bytes")]
    assert raw["resolution"] == "job-dir"
    assert raw["resolved_path"] == str(tmp_path / "dep.txt")
    sealed = by_key[(str(seal), "canonical-json-seal")]
    assert sealed["resolution"] == "absolute"
    missing = by_key[("gone/forever.bin", "raw-bytes")]
    assert missing["resolution"] == "missing"
    assert missing["resolved_path"] is None
    unknown = by_key[("dep.txt", "unknown")]
    assert unknown["sha256"] == "d" * 64
    assert unknown["resolution"] == "missing"
    assert row.fingerprint.unresolvable() == 2
    assert row.fingerprint.drift() == []


def test_harvest_declared_jobdir_and_base_root_resolution(tmp_path: Path) -> None:
    """job-dir and base-root bind only by digest; mismatches fail closed.

    A relative path digest-matches under the receipt directory (job-dir) or the
    caller-supplied base_root; a digest-mismatched relative path never binds
    (that is the bare-name false-bind vector), and an existing absolute path
    binds by path identity so its drift stays detectable.
    """

    base = tmp_path / "base"
    (base / "configs").mkdir(parents=True)
    (base / "configs" / "run.yaml").write_text("a: 1\n", encoding="utf-8")
    (base / "configs" / "other.yaml").write_text("z: 9\n", encoding="utf-8")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "local.txt").write_text("local bytes\n", encoding="utf-8")
    present_abs = tmp_path / "present.bin"
    present_abs.write_bytes(b"absolute bytes")
    receipt = job_dir / "run-receipt.json"

    refs = [
        {"path": "local.txt", "sha256": _raw_sha256(job_dir / "local.txt")},
        {"path": "configs/run.yaml", "sha256": _raw_sha256(base / "configs" / "run.yaml")},
        {"path": "configs/other.yaml", "sha256": "a" * 64},
        {"path": str(present_abs), "sha256": "b" * 64},
    ]
    _write_declared_receipt(receipt, refs)

    entries = harvest_declared_refs(
        json.loads(receipt.read_text(encoding="utf-8")), receipt, base_root=base
    )

    by_path = {e["path"]: e for e in entries}
    assert by_path["local.txt"]["resolution"] == "job-dir"
    assert by_path["local.txt"]["match_basis"] == "digest"
    assert by_path["configs/run.yaml"]["resolution"] == "base-root"
    assert by_path["configs/run.yaml"]["resolved_path"] == str(base / "configs" / "run.yaml")
    assert by_path["configs/run.yaml"]["match_basis"] == "digest"
    assert by_path["configs/other.yaml"]["resolution"] == "missing"
    assert by_path["configs/other.yaml"]["resolved_path"] is None
    assert by_path[str(present_abs)]["resolution"] == "absolute"
    assert by_path[str(present_abs)]["match_basis"] == "path"


def test_harvest_declared_bare_name_resolves_unique_nested_artifact(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    scene_dir = job_dir / "scene"
    scene_dir.mkdir(parents=True)
    artifact = scene_dir / "depth.png"
    artifact.write_bytes(b"scene depth bytes")
    receipt = job_dir / "report.json"
    _write_declared_receipt(receipt, [{"path": "depth.png", "sha256": _raw_sha256(artifact)}])

    entries = harvest_declared_refs(json.loads(receipt.read_text(encoding="utf-8")), receipt)

    [entry] = entries
    assert entry["resolution"] == "job-dir"
    assert entry["match_basis"] == "digest"
    assert entry["resolved_path"] == str(artifact)


def test_harvest_declared_nested_basename_stays_missing_when_ambiguous(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    for name in ("scene-a", "scene-b"):
        directory = job_dir / name
        directory.mkdir(parents=True)
        (directory / "depth.png").write_bytes(b"same scene depth bytes")
    receipt = job_dir / "report.json"
    _write_declared_receipt(
        receipt, [{"path": "depth.png", "sha256": _raw_sha256(job_dir / "scene-a" / "depth.png")}]
    )

    entries = harvest_declared_refs(json.loads(receipt.read_text(encoding="utf-8")), receipt)

    [entry] = entries
    assert entry["resolution"] == "missing"
    assert entry["resolved_path"] is None
    assert entry["match_basis"] is None


def test_declared_raw_bytes_drift_marks_stale(tmp_path: Path, registry: ClaimsRegistry) -> None:
    dep = tmp_path / "source.py"
    dep.write_text("original = True\n", encoding="utf-8")
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(receipt, [{"path": "source.py", "sha256": _raw_sha256(dep)}])
    ingest_receipt(registry, receipt, "raw-drift", "raw bytes ref", harvest_declared=True)

    assert verify(registry).still_valid == 1

    dep.write_text("original = False\n", encoding="utf-8")
    report = verify(registry)
    assert report.newly_stale == 1
    current = registry.current()["raw-drift"]
    assert current.status == "stale"
    assert "declared source.py" in current.stale_reason
    assert "changed" in current.stale_reason
    assert verify(registry).newly_stale == 0


def test_declared_unresolvable_counts_but_never_stales(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(
        receipt,
        [
            {"path": "/no/such/dir/dead/batch.json", "sha256": "a" * 64},
            {"path": "/absent/gone.safetensors", "sha256": "b" * 64},
        ],
    )
    ingest_receipt(registry, receipt, "rot-claim", "dead scratch refs", harvest_declared=True)

    for _ in range(2):  # idempotent: repeated sweeps append nothing
        report = verify(registry)
        assert (report.still_valid, report.newly_stale, report.unresolvable) == (1, 0, 2)
    assert registry.current()["rot-claim"].status == "measured"
    assert len(registry.rows()) == 1


def test_declared_canonical_json_seal_semantics(tmp_path: Path, registry: ClaimsRegistry) -> None:
    seal = tmp_path / "config.json"
    payload = {"beta": 2, "alpha": [1, {"k": True}]}
    seal.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(
        receipt, [{"file": "config.json", "config_sha256": _canonical_sha256(payload)}]
    )
    ingest_receipt(registry, receipt, "seal-claim", "sealed config", harvest_declared=True)

    seal.write_text(json.dumps({"alpha": [1, {"k": True}], "beta": 2}), encoding="utf-8")
    assert verify(registry).still_valid == 1

    seal.write_text(json.dumps({"alpha": [1, {"k": True}], "beta": 3}), encoding="utf-8")
    report = verify(registry)
    assert report.newly_stale == 1
    assert "changed" in registry.current()["seal-claim"].stale_reason

    seal.write_text("not json {", encoding="utf-8")
    verify(registry)
    assert "unparseable" in registry.current()["seal-claim"].stale_reason


def test_declared_ref_disappearing_after_ingest_is_drift(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    dep = tmp_path / "artifact.bin"
    dep.write_bytes(b"artifact bytes")
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(receipt, [{"path": "artifact.bin", "sha256": _raw_sha256(dep)}])
    ingest_receipt(registry, receipt, "vanish-claim", "vanishing ref", harvest_declared=True)

    dep.unlink()
    report = verify(registry)
    assert (report.newly_stale, report.unresolvable) == (1, 0)
    assert "declared artifact.bin" in registry.current()["vanish-claim"].stale_reason
    assert "missing" in registry.current()["vanish-claim"].stale_reason


def test_declared_unknown_semantics_never_verified(
    tmp_path: Path, registry: ClaimsRegistry
) -> None:
    dep = tmp_path / "tree.txt"
    dep.write_text("tree bytes", encoding="utf-8")
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(receipt, [{"path": "tree.txt", "tree_sha256": "f" * 64}])
    row = ingest_receipt(registry, receipt, "unknown-claim", "unknown", harvest_declared=True)

    declared = [e for e in row.fingerprint.entries if e["kind"] == "declared"]
    assert [e["semantics"] for e in declared] == ["unknown"]
    assert declared[0]["resolution"] == "missing"
    report = verify(registry)
    assert (report.still_valid, report.newly_stale, report.unresolvable) == (1, 0, 1)


def test_declared_entries_roundtrip_and_fail_closed(tmp_path: Path) -> None:
    fingerprint = DependencyFingerprint()
    fingerprint.add_declared(
        path="pkg/mod.py",
        sha256="a" * 64,
        semantics="raw-bytes",
        resolution="base-root",
        resolved_path="/base/pkg/mod.py",
    )
    fingerprint.add_declared(
        path="/no/such/gone.json",
        sha256="b" * 64,
        semantics="canonical-json-seal",
        resolution="missing",
    )
    rebuilt = DependencyFingerprint.from_dict(fingerprint.to_dict())
    assert rebuilt.to_dict() == fingerprint.to_dict()

    with pytest.raises(ClaimsRegistryError):
        fingerprint.add_declared(
            path="x", sha256="a" * 64, semantics="guessy", resolution="missing"
        )
    with pytest.raises(ClaimsRegistryError):
        fingerprint.add_declared(
            path="x", sha256="short", semantics="raw-bytes", resolution="missing"
        )
    bad = {
        "entries": [
            {
                "kind": "declared",
                "path": "x",
                "sha256": "a" * 64,
                "semantics": "raw-bytes",
                "resolution": "teleport",
                "resolved_path": "/x",
            }
        ]
    }
    with pytest.raises(ClaimsRegistryError):
        DependencyFingerprint.from_dict(bad)


def test_cli_harvest_declared_and_check_mode_exit_codes(tmp_path: Path, capsys) -> None:
    dep = tmp_path / "dep.txt"
    dep.write_text("cli bytes", encoding="utf-8")
    receipt = tmp_path / "run-receipt.json"
    _write_declared_receipt(
        receipt,
        [
            {"path": "dep.txt", "sha256": _raw_sha256(dep)},
            {"path": "/no/such/dir/dead/x.json", "sha256": "a" * 64},
        ],
    )
    ledger = tmp_path / "ledger.jsonl"

    assert (
        main(
            [
                "ingest",
                str(receipt),
                "--claim-id",
                "cli-declared",
                "--text",
                "cli declared",
                "--harvest-declared",
                "--registry",
                str(ledger),
            ]
        )
        == 0
    )
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["declared_refs"] == 2
    assert ingested["declared_unresolvable"] == 1
    ledger_bytes = ledger.read_bytes()

    assert main(["verify", "--check", "--json", "--registry", str(ledger)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["newly_stale"], report["unresolvable"]) == (0, 1)
    assert ledger.read_bytes() == ledger_bytes

    dep.write_text("mutated cli bytes", encoding="utf-8")
    assert main(["verify", "--check", "--registry", str(ledger)]) == 3
    capsys.readouterr()
    assert ledger.read_bytes() == ledger_bytes

    assert main(["verify", "--registry", str(ledger)]) == 0
    capsys.readouterr()
    assert len(ledger.read_bytes()) > len(ledger_bytes)
    assert main(["verify", "--check", "--registry", str(ledger)]) == 0
    capsys.readouterr()

    lines = ledger.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["text"] = "tampered"
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["verify", "--check", "--registry", str(ledger)]) == 2
    capsys.readouterr()
