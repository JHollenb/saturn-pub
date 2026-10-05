"""Mechanics and evidence tests for the native AR circuit measurement panel."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy

import pytest
import torch
from transformers import MambaConfig, MambaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchError, BenchSession
from mdb.bench_mamba import MemoryInterchangeError


@pytest.fixture(params=["mamba", "qwen2"])
def family(request):
    return request.param


@pytest.fixture
def model(family):
    torch.manual_seed(113)
    if family == "mamba":
        return MambaForCausalLM(
            MambaConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=2,
                state_size=4,
                expand=2,
                conv_kernel=4,
            )
        ).eval()
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            attn_implementation="eager",
        )
    ).eval().to(torch.bfloat16)


def _bench(model, tmp_path):
    return BenchSession.from_resident(
        model,
        root=tmp_path / "session",
        continuation_capacity=8,
    )


def _spec(family, **changes):
    value = {
        "name": "state-path",
        "sites": [0, 1],
        "contrasts": [{"name": "weaken", "mode": "scale", "value": 0.9}],
        "observable": {
            "kind": "token_margin",
            "token_id": 7,
            "against_token_id": 9,
        },
        "query_token_ids": [4],
        "continuation_tokens": 1,
        "memory_mediation": family == "mamba",
        "max_native_operations": 1000,
        "max_capture_bytes": 67108864,
        "seed": 0,
    }
    value.update(changes)
    return value


def _role(value):
    """Normalize the stable role vocabulary while tolerating dash/underscore spelling."""
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if text.startswith("matched-random"):
        return "matched-random"
    if text.startswith("residual-restore"):
        return "residual-restore"
    if text.startswith("memory-transfer"):
        return "memory-transfer"
    if text.startswith("memory-restore"):
        return "memory-restore"
    return text


def _site(value):
    if isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _assert_l2(value, label):
    """Check that a residual/memory metric exposes a finite nonnegative L2."""
    if value is None:
        return
    found = []

    def visit(node):
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_text = str(key).lower().replace("-", "_")
                if "l2" in key_text or key_text in {"norm", "l2_norm"}:
                    assert _number(child), f"{label} L2 must be finite"
                    assert float(child) >= 0.0, f"{label} L2 must be nonnegative"
                    found.append(float(child))
                else:
                    visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(value)
    assert found, f"{label} must retain an L2 metric"


def _assert_signed_effect(value, label):
    """Check that an effect keeps its signed value instead of only a magnitude."""
    found = []

    def visit(node):
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_text = str(key).lower().replace("-", "_")
                if "signed" in key_text or key_text in {"delta", "effect"}:
                    assert _number(child), f"{label} signed effect must be finite"
                    found.append(float(child))
                else:
                    visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(value)
    assert found, f"{label} must retain a signed effect"


def _find_true_closure_flag(value):
    """Find the explicit exact/full-closure flag on a no-op observation."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower().replace("-", "_")
            if isinstance(child, bool) and child and (
                "closure" in key_text and "exact" in key_text
                or key_text in {"noop_exact", "full_closure_exact", "state_exact"}
            ):
                return True
            if key_text in {"closure", "full_closure", "fullclosure"}:
                if _find_true_closure_flag(child):
                    return True
            elif isinstance(child, (Mapping, Sequence)):
                if _find_true_closure_flag(child):
                    return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_find_true_closure_flag(child) for child in value)
    return False


def _contains(value, target):
    if value == target:
        return True
    if isinstance(value, Mapping):
        return any(
            _contains(key, target) or _contains(child, target)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains(child, target) for child in value)
    return False


def _measurement_rows(result):
    report = result["report"]
    rows = report["rows"]
    assert rows
    assert isinstance(rows, list)
    return report, rows


def _same_state(bench, left, right):
    comparison = bench.compare(left, right)
    assert all(comparison[key] for key in comparison if key.startswith("same_"))


def _selected_pointer(bench):
    return json.loads((bench.root / "selected.json").read_text())


def _memory_row(report):
    return next(row for row in report["rows"] if _role(row["role"]) == "memory-transfer")


def test_measure_circuit_emits_cas_rows_and_exact_replay(model, family, tmp_path):
    b = _bench(model, tmp_path)
    try:
        parent = b.prefill([1, 2, 3])
        result = b.measure_circuit(_spec(family))
        report, rows = _measurement_rows(result)
        assert report["symbol_status"] == "measured-causal-response-binding"
        assert report["symbol_ref"] is not None

        assert result["report_ref"].startswith("cas://sha256/")
        assert b.artifacts.get(result["report_ref"]) == report
        assert report["semantic_binding"] is False
        assert {_site(row["site"]) for row in rows} == {0, 1}

        expected_roles = {"reference", "candidate", "matched-random", "noop", "residual-restore"}
        for site in (0, 1):
            site_rows = [row for row in rows if _site(row["site"]) == site]
            assert expected_roles <= {_role(row["role"]) for row in site_rows}

        native_rows = []
        noop_rows = []
        for row in rows:
            assert row["name"]
            assert row["evidence_ref"].startswith("cas://sha256/")
            evidence = b.artifacts.get(row["evidence_ref"])
            assert evidence["schema"] == "saturn-bench-circuit-observation-v1"
            assert evidence["session_id"] == b.session_id

            metrics = row["metrics"]
            for key in (
                "residual",
                "memory",
                "first_effect",
                "future_effect",
            ):
                assert key in metrics
            _assert_l2(metrics["residual"], f"{row['name']} residual")
            _assert_l2(metrics["memory"], f"{row['name']} memory")
            if _role(row["role"]) in {"memory-transfer", "memory-restore"}:
                assert row["first"]["kind"] == "recipient-boundary-snapshot"
                assert metrics["first_effect"]["signed_effect"] is None
                assert metrics["first_effect"]["native_output_measured"] is False
            else:
                _assert_signed_effect(metrics["first_effect"], f"{row['name']} first")
            _assert_signed_effect(metrics["future_effect"], f"{row['name']} future")
            assert isinstance(row["first"], Mapping)
            assert isinstance(row["future"], Mapping)
            assert isinstance(row["first"]["closure"], Mapping)
            assert isinstance(row["future"]["closure"], Mapping)

            replay = b.replay_circuit(row["evidence_ref"])
            assert replay["exact"] is True
            assert replay["session_id"] == b.session_id
            if _role(row["role"]) == "reference":
                native_rows.append(row)
            if _role(row["role"]) == "noop":
                noop_rows.append(row)

        assert native_rows
        assert noop_rows
        assert all(
            row.get("semantic_claim", False) is False
            for row in rows
            if _role(row["role"]) != "candidate"
        )
        assert all(_find_true_closure_flag(row["metrics"]) for row in noop_rows)

        measurements = b.inspect("measurements")
        assert _contains(measurements, result["report_ref"])
        comparison = b.compare(parent, b.branch.cut)
        assert all(comparison[key] for key in comparison if key.startswith("same_"))
    finally:
        b.close()


def test_zero_dose_single_site_report_is_valid_without_symbol(model, family, tmp_path):
    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        result = b.measure_circuit(
            _spec(
                family,
                sites=[0],
                contrasts=[{"name": "zero-dose", "mode": "scale", "value": 1.0}],
            )
        )
        report, rows = _measurement_rows(result)
        assert report["mechanics_status"] == "valid"
        assert report["instrument_status"] == "native-consumer-executed"
        assert report["symbol_status"] == "not-applicable-insufficient-exposure-coordinates"
        assert report["symbol_ref"] is None
        assert {_site(row["site"]) for row in rows} == {0}

        for row in rows:
            future_effect = row["metrics"]["future_effect"]
            assert future_effect["signed_effect"] == pytest.approx(0.0)
            assert future_effect["closure_exact"] is True
            if _role(row["role"]) in {"noop", "residual-restore"}:
                assert row["metrics"]["first_effect"]["closure_exact"] is True
                assert _find_true_closure_flag(row["metrics"])
            replay = b.replay_circuit(row["evidence_ref"])
            assert replay["exact"] is True

        measurements = b.inspect("measurements")
        frozen = next(item for item in measurements if item["report_ref"] == result["report_ref"])
        assert frozen["symbol_ref"] is None
        assert frozen["context_compatible"] is True
    finally:
        b.close()


def test_sparse_physical_sites_publish_without_hyperion_fit(model, family, tmp_path):
    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        parent_cursor = dict(b.cursor())
        result = b.measure_circuit(_spec(family, sites=[0, 2]))
        report, rows = _measurement_rows(result)
        assert report["mechanics_status"] == "valid"
        assert report["symbol_ref"] is None
        assert report["symbol_status"] == "not-applicable-sparse-exposure-lattice"
        assert {_site(row["site"]) for row in rows} == {0, 2}
        assert b.cursor() == parent_cursor
        frozen = b.inspect("measurements")[-1]
        assert frozen["report_ref"] == result["report_ref"]
        assert frozen["context_compatible"] is True
        candidate = next(row for row in rows if _role(row["role"]) == "candidate")
        assert b.replay_circuit(candidate["evidence_ref"])["exact"] is True
    finally:
        b.close()


def test_mamba_measurement_records_memory_transfer_and_restore(model, family, tmp_path):
    if family != "mamba":
        pytest.skip("memory mediation is a Mamba-specific closure")

    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        spec = _spec(family)
        assert spec["query_token_ids"] == [4]
        result = b.measure_circuit(spec)
        report, rows = _measurement_rows(result)
        assert report.get("memory_mediation") is True
        assert report["spec"]["query_token_ids"] == [4]

        normalized = {_role(row["role"]) for row in rows}
        assert {"memory-transfer", "memory-restore"} <= normalized
        transfer_rows = [row for row in rows if _role(row["role"]) == "memory-transfer"]
        restore_rows = [row for row in rows if _role(row["role"]) == "memory-restore"]
        assert transfer_rows and restore_rows

        paired_future_exact = []
        aligned_l2 = []
        cache_flag_exact = []
        for row in [*transfer_rows, *restore_rows]:
            memory = row["metrics"]["memory"]
            for key, value in memory.items():
                key_text = str(key).lower().replace("-", "_")
                if "l2" in key_text and _number(value):
                    aligned_l2.append(float(value))
                if "cache" in key_text and "flag" in key_text and value is True:
                    cache_flag_exact.append(True)
            paired_future = row["metrics"]["paired_future"]
            paired_future_exact.append(paired_future["closure_exact"] is True)

        assert aligned_l2 and all(value >= 0.0 for value in aligned_l2)
        assert paired_future_exact and all(paired_future_exact)
        assert cache_flag_exact
        assert "query_token_ids" in report["spec"]
    finally:
        b.close()


def test_measure_circuit_rejects_invalid_requests_before_forward(
    model, family, tmp_path, monkeypatch
):
    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        calls = []
        original_forward = model.forward

        def counted_forward(*args, **kwargs):
            calls.append(1)
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(model, "forward", counted_forward)
        parent = b.branch.cut
        invalid = [
            (_spec(family, max_native_operations=1), "budget"),
            (_spec(family, sites=[1, 1]), "distinct"),
            (_spec(family, max_capture_bytes=1), "capture"),
        ]
        if family == "mamba":
            invalid.append((_spec(family, query_token_ids=[]), "query"))

        for bad_spec, message in invalid:
            with pytest.raises(BenchError, match=message):
                b.measure_circuit(bad_spec)
            assert calls == []
            assert b.branch.cut == parent

        b.step("layer")
        edit = b.preview_edit("residual", 0.9, mode="scale")
        pending_parent = b.branch.cut
        pending_cursor = deepcopy(b.cursor())
        pending_residual = b.resolve_value("residual").clone()
        with pytest.raises(BenchError, match="pending"):
            b.measure_circuit(_spec(family))
        assert calls == []
        assert b.branch.cut == pending_parent
        assert b.cursor() == pending_cursor
        assert torch.equal(b.resolve_value("residual"), pending_residual)
        committed = b.commit_edit(edit["act_id"])
        assert committed["act_id"] == edit["act_id"]
    finally:
        b.close()


def test_measure_circuit_restores_parent_keeps_sibling_and_restarts_from_export(
    model, family, tmp_path
):
    b = _bench(model, tmp_path)
    export_root = tmp_path / "export"
    try:
        b.prefill([1, 2, 3])
        parent = b.branch.cut
        b.fork("sibling", cut=parent)
        sibling_before_checkout = b.branches["sibling"].cut
        b.checkout("main")
        sibling_cut = b.branches["sibling"].cut
        sibling_checkout_comparison = b.compare(sibling_before_checkout, sibling_cut)
        assert all(
            sibling_checkout_comparison[key]
            for key in sibling_checkout_comparison
            if key.startswith("same_")
        )
        main_cut = b.branch.cut
        assert main_cut == parent

        result = b.measure_circuit(_spec(family))
        assert b.active == "main"
        comparison = b.compare(parent, b.branch.cut)
        assert all(comparison[key] for key in comparison if key.startswith("same_"))
        assert b.branches["sibling"].cut == sibling_cut

        b.checkout("sibling")
        assert b.branch.cut == sibling_cut
        b.checkout("main")
        assert all(
            b.compare(parent, b.branch.cut)[key]
            for key in b.compare(parent, b.branch.cut)
            if key.startswith("same_")
        )

        b.export(export_root)
        from mdb.bench_verify import verify

        checked = verify(export_root)
        assert checked["mechanics_status"] == "valid"
        b.close()

        fresh = BenchSession.resume(model, root=export_root)
        try:
            replay = fresh.replay_circuit(result["report"]["rows"][0]["evidence_ref"])
            assert replay["exact"] is True
            assert replay["session_id"] == fresh.session_id
            assert _contains(fresh.inspect("measurements"), result["report_ref"])
        finally:
            fresh.close()
    finally:
        if not b.closed:
            b.close()


def test_measure_circuit_publication_failure_restores_selected_parent_and_sibling(
    model, family, tmp_path, monkeypatch
):
    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        parent = b.branch.cut
        b.fork("sibling", cut=parent)
        sibling_before_checkout = b.branches["sibling"].cut
        b.checkout("main")
        sibling = b.branches["sibling"].cut
        _same_state(b, sibling_before_checkout, sibling)
        selected_before = _selected_pointer(b)
        manifest_before = b.artifacts.get(selected_before["manifest"])
        events_before = deepcopy(b.events)
        cuts_before = set(b.cuts)

        original_checkpoint = b.checkpoint

        def fail_final_publication(label=None, **kwargs):
            if label == "circuit-measurement-published":
                raise OSError("injected publication failure")
            return original_checkpoint(label, **kwargs)

        # Fault the measurement's final publication marker only. Earlier
        # checkpoints remain usable so this exercises recovery of the selected
        # parent rather than making every publication operation fail.
        monkeypatch.setattr(b, "checkpoint", fail_final_publication)
        with pytest.raises(OSError, match="publication"):
            b.measure_circuit(_spec(family))

        selected_after = _selected_pointer(b)
        assert selected_after == selected_before
        assert b.artifacts.get(selected_after["manifest"]) == manifest_before
        assert b.active == "main"
        _same_state(b, parent, b.branch.cut)
        assert b.branches["sibling"].cut == sibling
        _same_state(b, sibling, b.branches["sibling"].cut)
        assert b.events == events_before
        assert set(b.cuts) == cuts_before
    finally:
        b.close()


def test_measure_circuit_final_seal_failure_keeps_custody_and_fresh_restart(
    model, family, tmp_path, monkeypatch
):
    b = _bench(model, tmp_path)
    root = b.root
    try:
        b.prefill([1, 2, 3])
        parent = b.branch.cut
        selected_before = _selected_pointer(b)
        manifest_before = b.artifacts.get(selected_before["manifest"])
        cuts_before = deepcopy(b.cuts)
        events_before = deepcopy(b.events)

        original_seal = b.publication._seal
        seal_calls = []

        def fail_once(manifest):
            seal_calls.append(manifest)
            if len(seal_calls) == 1:
                raise OSError("injected final journal seal failure")
            return original_seal(manifest)

        monkeypatch.setattr(b.publication, "_seal", fail_once)
        with pytest.raises(OSError, match="journal seal"):
            b.measure_circuit(_spec(family))

        assert len(seal_calls) >= 2
        assert _selected_pointer(b) == selected_before
        assert b.artifacts.get(selected_before["manifest"]) == manifest_before
        assert b.branch.cut == parent
        assert b.cuts == cuts_before
        assert b.events == events_before

        candidate_index_path = root / "journal" / "candidates.json"
        candidate_index = json.loads(candidate_index_path.read_text())
        candidate_packs = candidate_index["packs"]
        assert candidate_packs
        assert all((root / "journal" / f"{digest}.pack").exists() for digest in candidate_packs)

        # Finish the writer cleanly, then remove every materialized store. The
        # journal and selected pointer are the only durable recovery authority.
        b.close()
        restart_selected = _selected_pointer(b)
        restart_manifest = b.artifacts.get(restart_selected["manifest"])
        restart_catalog = deepcopy(restart_manifest["cuts"])
        restart_events = deepcopy(restart_manifest["events"])
        restart_packs = list(candidate_packs)
        for child in root.iterdir():
            if child.name in {"selected.json", "journal", ".writer.lock"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        fresh = BenchSession.resume(model, root=root, family=family)
        try:
            fresh_selected = _selected_pointer(fresh)
            assert fresh_selected["manifest"] == restart_selected["manifest"]
            assert fresh_selected["publication_schema"] == restart_selected["publication_schema"]
            assert fresh_selected["nonce"] == restart_selected["nonce"]
            assert fresh.session_id == b.session_id
            assert fresh.active == restart_manifest["active"] == "main"
            assert fresh.branch.cut == restart_manifest["branches"]["main"]["cut"]
            assert fresh.cuts == restart_catalog
            assert fresh.events == restart_events
            assert fresh.inspect("measurements") == []
            recovered_packs = json.loads(candidate_index_path.read_text())["packs"]
            assert set(restart_packs) <= set(recovered_packs)
            assert all((root / "journal" / f"{digest}.pack").exists() for digest in recovered_packs)
            assert (root / "artifacts").exists()
            assert (root / "programs.json").exists()
        finally:
            fresh.close()
    finally:
        if not b.closed:
            b.close()


def test_mamba_memory_rejects_dtype_flags_and_history_without_recipient_writes(
    model, family, tmp_path
):
    if family != "mamba":
        pytest.skip("complete successor memory is a Mamba-specific closure")

    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        report, _ = _measurement_rows(b.measure_circuit(_spec(family)))
        memory = _memory_row(report)
        recipient_cut = memory["recipient_cut"]
        donor_cut = memory["donor_cut"]

        def invalid_donor(kind):
            cut = deepcopy(b.artifacts.get(donor_cut))
            page = cut["state"]["pages"][0]
            if kind == "dtype":
                page["conv"]["$geometry"]["dtype"] = "torch.float64"
            elif kind == "flags":
                flags = page["flags"]
                bool_key = next(
                    (key for key, value in flags.items() if isinstance(value, bool)),
                    None,
                )
                if bool_key is None:
                    flags["circuit_test_flag"] = True
                else:
                    flags[bool_key] = not flags[bool_key]
            elif kind == "history":
                token_ids = cut["state"]["token_ids"]
                token_ids[-1] = (token_ids[-1] + 1) % model.config.vocab_size
            else:
                raise AssertionError(kind)
            return b.artifacts.put(cut)

        for kind in ("dtype", "flags", "history"):
            bad_donor = invalid_donor(kind)
            b.restore(recipient_cut)
            before_cut = b.branch.cut
            before_selected = _selected_pointer(b)
            before_events = deepcopy(b.events)
            before_cuts = set(b.cuts)
            before_artifacts = {
                path.relative_to(b.artifacts.root).as_posix()
                for path in b.artifacts.root.rglob("*.json")
            }
            with pytest.raises(Exception, match="memory|flags|history|tensor|incompatible"):
                b._circuit_memory_act(bad_donor)
            assert b.branch.cut == before_cut
            assert _selected_pointer(b) == before_selected
            assert b.events == before_events
            assert set(b.cuts) == before_cuts
            after_artifacts = {
                path.relative_to(b.artifacts.root).as_posix()
                for path in b.artifacts.root.rglob("*.json")
            }
            assert after_artifacts == before_artifacts
    finally:
        b.close()


def test_mamba_memory_interchange_error_is_a_not_applicable_row(
    model, family, tmp_path, monkeypatch
):
    if family != "mamba":
        pytest.skip("complete successor memory is a Mamba-specific closure")

    b = _bench(model, tmp_path)
    try:
        parent = b.prefill([1, 2, 3])

        def unavailable(_donor_cut):
            raise MemoryInterchangeError("typed successor memory is not applicable")

        monkeypatch.setattr(b, "_circuit_memory_act", unavailable)
        result = b.measure_circuit(_spec(family))
        report, rows = _measurement_rows(result)
        memory_rows = [
            row for row in rows if _role(row["role"]) in {"memory-transfer", "memory-restore"}
        ]
        assert memory_rows
        assert report["inapplicable_rows"] == len(memory_rows)
        assert all(row["instrument_status"] == "not-applicable" for row in memory_rows)
        assert all(row["replay_stage"] == "not-applicable" for row in memory_rows)
        assert all(row["first"] is None and row["future"] is None for row in memory_rows)
        _same_state(b, parent, b.branch.cut)
    finally:
        b.close()


def test_mamba_generic_memory_error_aborts_and_restores_catalog(
    model, family, tmp_path, monkeypatch
):
    if family != "mamba":
        pytest.skip("complete successor memory is a Mamba-specific closure")

    b = _bench(model, tmp_path)
    try:
        parent = b.prefill([1, 2, 3])
        selected_before = _selected_pointer(b)
        manifest_before = b.artifacts.get(selected_before["manifest"])
        cuts_before = deepcopy(b.cuts)
        events_before = deepcopy(b.events)

        def fatal(_donor_cut):
            raise BenchError("fatal memory integrity failure")

        monkeypatch.setattr(b, "_circuit_memory_act", fatal)
        with pytest.raises(BenchError, match="fatal memory integrity"):
            b.measure_circuit(_spec(family))

        assert _selected_pointer(b) == selected_before
        assert b.artifacts.get(selected_before["manifest"]) == manifest_before
        assert b.cuts == cuts_before
        assert b.events == events_before
        _same_state(b, parent, b.branch.cut)
        assert b.inspect("measurements") == []
    finally:
        b.close()


def test_offline_verify_rejects_missing_nested_discarded_act_receipt(model, family, tmp_path):
    b = _bench(model, tmp_path)
    export_root = tmp_path / "export"
    try:
        b.prefill([1, 2, 3])
        report, rows = _measurement_rows(b.measure_circuit(_spec(family)))
        restore_row = next(row for row in rows if _role(row["role"]) == "residual-restore")
        evidence = deepcopy(b.artifacts.get(restore_row["evidence_ref"]))
        assert evidence["discarded_act"]["receipt_ref"]
        evidence["discarded_act"]["receipt_ref"] = "cas://sha256/" + "f" * 64
        b.artifacts.put(evidence)
        b.export(export_root)
    finally:
        b.close()

    from mdb.bench_verify import verify

    with pytest.raises(Exception, match="(?i)missing|receipt|artifact"):
        verify(export_root)


def test_mamba_replay_reexecutes_memory_transplant_and_rejects_fake_donor(
    model, family, tmp_path, monkeypatch
):
    if family != "mamba":
        pytest.skip("successor memory replay is a Mamba-specific closure")

    b = _bench(model, tmp_path)
    try:
        b.prefill([1, 2, 3])
        report, _ = _measurement_rows(b.measure_circuit(_spec(family)))
        memory = _memory_row(report)
        evidence_ref = memory["evidence_ref"]
        calls = []
        original_memory_act = b._circuit_memory_act

        def counted_memory_act(donor_cut):
            calls.append(donor_cut)
            return original_memory_act(donor_cut)

        monkeypatch.setattr(b, "_circuit_memory_act", counted_memory_act)
        before = b.branch.cut
        replay = b.replay_circuit(evidence_ref)
        assert replay["exact"] is True
        assert calls == [memory["donor_cut"]]
        _same_state(b, before, b.branch.cut)

        fake = deepcopy(b.artifacts.get(evidence_ref))
        fake_donor = "cas://sha256/" + "0" * 64
        fake["donor_cut"] = fake_donor
        fake_ref = b.artifacts.put(fake)
        before_fake = b.branch.cut
        with pytest.raises(Exception, match="missing|incompatible|donor|artifact"):
            b.replay_circuit(fake_ref)
        _same_state(b, before_fake, b.branch.cut)
    finally:
        b.close()
