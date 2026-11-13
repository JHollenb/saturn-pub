"""Native joins, stale contexts and durable live observations."""
import pytest
from saturn_pub import StateAddress

from mdb_lanes.live_symbols import LiveSymbols
from mdb_lanes.project import Project
from test_saturn_pub_adapter import opened


def test_live_records_are_registered_and_reopen_without_payload_reads(tmp_path, monkeypatch):
    project, adapter, owner, _ = opened(tmp_path)
    # The fixture counter has no tokens; use a native Qwen to exercise real context slots.
    owner.close()
    from saturn_pub.adapters.qwen import QwenAdapter
    from mdb_lanes.saturn_pub_adapter import SaturnPubDebuggerAdapter
    from mdb_lanes.prompt_qwen import PromptQwenAdapter
    native = PromptQwenAdapter(QwenAdapter.tiny().model).session([5, 7, 11])
    adapter = SaturnPubDebuggerAdapter(native, project=project, session_id="qwen")
    recorder = LiveSymbols(adapter)
    monkeypatch.setattr(project, "slice", lambda *a, **kw: pytest.fail("eager historical slice"))
    source = recorder.observe("input.tokens", (StateAddress("tokens"),), context="prompt",
        claim="Supplied token sequence at this exact native boundary.")
    assert recorder.observe("input.tokens", (StateAddress("tokens"),), context="prompt",
        claim=source.binding.claim).fingerprint == source.fingerprint
    before = native.read("tokens")
    receipt = recorder.advance(1, symbols=(source,))
    assert native.read("tokens").equal(before)
    assert receipt.to_dict()["parent"] == source.definitions[0].source_cut
    with pytest.raises(ValueError, match="clock"):
        recorder.advance(1, symbols=(source,))
    with pytest.raises(ValueError, match="context"):
        source.binding.resolve(native, context="other")
    saved = recorder.records[source.binding_fingerprint]
    assert len(saved.uses) == 1
    import hashlib
    raw = (project.root / "project.sqlite").read_bytes()
    reopened = Project(project.root, read_only=True)
    monkeypatch.setattr(reopened.store, "load", lambda *a, **kw: pytest.fail("hydration"))
    row = reopened.symbol_lifecycle("qwen", saved.fingerprint)
    assert row["joins"]["uses"][0]["receipt_closure"] == "native-parent-result-matched"
    assert reopened.symbols("qwen")[0]["symbol"] == "input.tokens"
    assert reopened.symbols_at("qwen", "tokens", cut_id=source.definitions[0].source_cut)["matches"]
    assert not reopened.symbols_at("qwen", "tokens", cut_id=receipt.to_dict()["result"])["matches"]
    assert hashlib.sha256((project.root / "project.sqlite").read_bytes()).digest() == hashlib.sha256(raw).digest()
    from mdb_lanes.symbol_lifecycle_project import main
    assert main([str(project.root), "list", "--session-id", "qwen", "--read-only"]) == 0
    assert (project.root / "project.sqlite").read_bytes() == raw
    import sqlite3
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with reopened._db() as db:
            db.execute("DELETE FROM symbols")


def test_observation_is_passive_and_restore_rebinds_exactly(tmp_path):
    from saturn_pub.adapters.qwen import QwenAdapter
    from mdb_lanes.saturn_pub_adapter import SaturnPubDebuggerAdapter
    from mdb_lanes.prompt_qwen import PromptQwenAdapter
    native = PromptQwenAdapter(QwenAdapter.tiny().model).session([5, 7, 11])
    adapter = SaturnPubDebuggerAdapter(native, project=Project(tmp_path), session_id="qwen")
    recorder = LiveSymbols(adapter)
    before = native.capture(retain=False)
    source = recorder.observe("source.value", (StateAddress("values", (0, 0, 0, 0)),),
        context="prompt", claim="Observed cache carrier; no inferred semantic function.")
    assert native.capture(retain=False).fingerprint == before.fingerprint
    recorder.advance(1, symbols=(source,))
    adapter.restore(before.fingerprint)
    source.binding.resolve(native, context="prompt")
    assert native.capture(retain=False).fingerprint == before.fingerprint


def test_prompt_staging_preserves_history_and_native_parity():
    import torch
    from saturn_pub.adapters.qwen import QwenAdapter
    from saturn_pub.core import Act
    from mdb_lanes.prompt_qwen import PromptQwenAdapter
    base = QwenAdapter.tiny()
    extended = PromptQwenAdapter(base.model)
    native = extended.session([5, 7])
    native.continue_(extended.layers + 5)
    extended.stage_prompt_token(native, 11)
    assert native.read("tokens").tolist() == [[5, 7, 11]]
    native.continue_(extended.layers + 3)  # before greedy sample
    torch.testing.assert_close(native.read("logits"), base.native_logits([5, 7, 11]), atol=1e-6, rtol=1e-5)
    assert native.read("logits").argmax() == base.native_logits([5, 7, 11]).argmax()
    bad = native.read("tokens")
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="read-only|prompt ingestion"):
        native.apply(Act.replace("tokens", bad))
    assert base.execution != extended.execution
